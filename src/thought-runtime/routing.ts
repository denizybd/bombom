import { seededThoughtUnit, thoughtDigest } from './deterministic'
import { clamp, composeNeuronRepresentation, entropyBranchCount, positiveCompatibility, shannonEntropy } from './math'
import { allThoughtNeurons } from './model'
import {
  createThoughtAdaptiveCandidateTelemetry,
  evaluateThoughtAdaptiveCompatibility,
  MAXIMUM_ADAPTIVE_CANDIDATE_TELEMETRY,
  validateThoughtAdaptiveRegionContext,
} from './adaptive-regions'
import type {
  DerivedFamilyGraph,
  ThoughtBranch,
  ThoughtFamilyRouteCandidate,
  ThoughtHead,
  ThoughtModel,
  ThoughtNeuron,
  ThoughtNeuronRouteCandidate,
  ThoughtRouteResult,
  ThoughtRoutingHints,
  ThoughtRuntimeConfig,
  ThoughtStepOptions,
  ThoughtAdaptiveCompatibilityTelemetryV1,
} from './types'

const ROUTE_EPSILON = 1e-12

export interface CompiledThoughtEdge {
  fromNeuronId: string
  toNeuronId: string
  baseH0: number
  persistentExcitation: number
  persistentInhibition: number
  trustMass: number
  updateCount: number
  baseLogit: number
  baseProbability: number
}

export interface ThoughtRuntimeIndex {
  modelFingerprint: string
  headFingerprint: string
  seed: number
  neurons: readonly ThoughtNeuron[]
  neuronById: ReadonlyMap<string, ThoughtNeuron>
  relationPatchByNeuron: ReadonlyMap<string, Float32Array>
  edgeRows: ReadonlyMap<string, readonly CompiledThoughtEdge[]>
  familyMembers: ReadonlyMap<string, ReadonlyArray<{ neuronId: string; weight: number }>>
  membershipWeightByFamily: ReadonlyMap<string, ReadonlyMap<string, number>>
  familyRows: ReadonlyMap<string, readonly DerivedFamilyGraph['transitions'][number][]>
  familyRowTailEvidence: ReadonlyMap<string, number>
  explorationNeuronIds: readonly string[]
  temporaryNeuronIds: readonly string[]
  familyGraph: DerivedFamilyGraph
  compiledNeuronCount: number
  compiledEdgeCount: number
  signalIndex: ReadonlyArray<{
    coordinates: readonly number[]
    buckets: ReadonlyMap<number, readonly string[]>
  }>
}

function signalBucket(vector: ArrayLike<number>, coordinates: readonly number[]): number {
  let bucket = 0
  for (let bit = 0; bit < coordinates.length; bit += 1) if (vector[coordinates[bit]] >= 0) bucket |= 1 << bit
  return bucket
}

function neuronSignalBucket(
  neuron: ThoughtNeuron,
  relationPatch: Float32Array | undefined,
  layout: ThoughtModel['layout'],
  coordinates: readonly number[],
): number {
  let bucket = 0
  for (let bit = 0; bit < coordinates.length; bit += 1) {
    const coordinate = coordinates[bit]
    const value = coordinate < layout.relationStart
      ? neuron.identity[coordinate]
      : clamp(neuron.relation[coordinate - layout.relationStart] +
        (relationPatch?.[coordinate - layout.relationStart] ?? 0), -1, 1)
    if (value >= 0) bucket |= 1 << bit
  }
  return bucket
}

function softmax<T>(values: readonly T[], logit: (value: T) => number): Array<T & { probability: number }> {
  if (!values.length) return []
  const maximum = values.reduce((current, value) => Math.max(current, logit(value)), Number.NEGATIVE_INFINITY)
  const weights = values.map((value) => Math.exp(logit(value) - maximum))
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  return values.map((value, index) => ({ ...value, probability: weights[index] / total }))
}

function compiledRows(
  model: ThoughtModel,
  head: ThoughtHead,
  config: ThoughtRuntimeConfig,
): Map<string, CompiledThoughtEdge[]> {
  const byKey = new Map<string, Omit<CompiledThoughtEdge, 'baseLogit' | 'baseProbability'>>()
  const validNeuronIds = new Set([
    ...model.neurons.map((neuron) => neuron.id),
    ...head.births.map((birth) => birth.id),
  ])
  const addPersistentEvidence = (
    fromNeuronId: string,
    toNeuronId: string,
    excitation: number,
    trustMass: number,
    updateCount: number,
  ) => {
    if (!validNeuronIds.has(fromNeuronId) || !validNeuronIds.has(toNeuronId)) return
    const key = `${fromNeuronId}\u0000${toNeuronId}`
    const current = byKey.get(key) ?? {
      fromNeuronId,
      toNeuronId,
      baseH0: 0,
      persistentExcitation: 0,
      persistentInhibition: 0,
      trustMass: 0,
      updateCount: 0,
    }
    current.persistentExcitation = clamp(
      current.persistentExcitation + excitation,
      0,
      config.evidenceCeiling,
    )
    current.trustMass = Math.max(0, current.trustMass + trustMass)
    current.updateCount = Math.max(0, current.updateCount + updateCount)
    byKey.set(key, current)
  }
  for (const row of model.habitRows) {
    for (const edge of row.edges) {
      byKey.set(`${row.fromNeuronId}\u0000${edge.toNeuronId}`, {
        fromNeuronId: row.fromNeuronId,
        toNeuronId: edge.toNeuronId,
        baseH0: Math.max(0, edge.priorEvidence),
        persistentExcitation: Math.max(0, edge.excitationEvidence),
        persistentInhibition: Math.max(0, edge.inhibitionEvidence),
        trustMass: Math.max(0, edge.trustMass),
        updateCount: Math.max(0, edge.updateCount),
      })
    }
  }
  for (const edge of head.neuronEdges) {
    addPersistentEvidence(
      edge.fromMemberKey,
      edge.toMemberKey,
      Math.max(0, edge.excitation),
      Math.max(0, edge.trustMass),
      Math.max(0, edge.updateCount),
    )
    const key = `${edge.fromMemberKey}\u0000${edge.toMemberKey}`
    const current = byKey.get(key)
    if (!current) continue
    current.persistentInhibition = clamp(current.persistentInhibition + edge.inhibition, 0, config.evidenceCeiling)
    byKey.set(key, current)
  }
  // Promotion must not sever the temporary neuron's learned entry/exit path.
  // These are outcome-gated observations from the nursery, so they join E+
  // rather than the immutable H0 prior.
  for (const birth of head.births) {
    const observationCount = Math.max(1, birth.contextEvidence?.observationCount ?? birth.sourceGroupIds.length)
    const trustPerObservation = Math.max(0, birth.trustMass) / observationCount
    for (const entry of birth.entryEvidence ?? []) {
      if (!(entry.weight > 0) || entry.memberKey === birth.id) continue
      addPersistentEvidence(entry.memberKey, birth.id, entry.weight, trustPerObservation, observationCount)
    }
    for (const entry of birth.outgoingEvidence ?? []) {
      if (!(entry.weight > 0) || entry.memberKey === birth.id) continue
      addPersistentEvidence(birth.id, entry.memberKey, entry.weight, trustPerObservation, observationCount)
    }
  }
  const grouped = new Map<string, Array<Omit<CompiledThoughtEdge, 'baseLogit' | 'baseProbability'>>>()
  for (const edge of byKey.values()) {
    const row = grouped.get(edge.fromNeuronId) ?? []
    row.push(edge)
    grouped.set(edge.fromNeuronId, row)
  }
  const result = new Map<string, CompiledThoughtEdge[]>()
  for (const [origin, row] of grouped) {
    const withLogit = row.map((edge) => ({
      ...edge,
      baseLogit: Math.log(edge.baseH0 + edge.persistentExcitation + ROUTE_EPSILON) -
        Math.log(1 + edge.persistentInhibition),
    })).sort((left, right) => right.baseLogit - left.baseLogit || left.toNeuronId.localeCompare(right.toNeuronId))
      .slice(0, config.maximumHabitEdgesPerNeuron)
    result.set(origin, softmax(withLogit, (edge) => edge.baseLogit).map((edge) => ({ ...edge, baseProbability: edge.probability })))
  }
  return result
}

/** Exact derived view G=RowNorm(MᵀHM), without a separately learned family graph. */
function deriveFamilyGraphFromIndex(
  neurons: readonly ThoughtNeuron[],
  rows: ReadonlyMap<string, readonly CompiledThoughtEdge[]>,
  modelFingerprint: string,
  headFingerprint: string,
): DerivedFamilyGraph {
  const membershipByNeuron = new Map(neurons.map((neuron) => [neuron.id, neuron.familyMemberships]))
  const sums = new Map<string, number>()
  for (const [origin, row] of rows) {
    const fromMemberships = membershipByNeuron.get(origin) ?? []
    for (const edge of row) {
      const toMemberships = membershipByNeuron.get(edge.toNeuronId) ?? []
      for (const from of fromMemberships) {
        for (const to of toMemberships) {
          const key = `${from.familyId}\u0000${to.familyId}`
          sums.set(key, (sums.get(key) ?? 0) + from.weight * edge.baseProbability * to.weight)
        }
      }
    }
  }
  const grouped = new Map<string, Array<{ toFamilyId: string; evidence: number }>>()
  for (const [key, evidence] of sums) {
    const [fromFamilyId, toFamilyId] = key.split('\u0000')
    const row = grouped.get(fromFamilyId) ?? []
    row.push({ toFamilyId, evidence })
    grouped.set(fromFamilyId, row)
  }
  const transitions = [...grouped].sort(([left], [right]) => left.localeCompare(right)).flatMap(([fromFamilyId, row]) => {
    const total = row.reduce((sum, entry) => sum + entry.evidence, 0)
    return row.sort((left, right) => right.evidence - left.evidence || left.toFamilyId.localeCompare(right.toFamilyId))
      .map((entry) => ({
        fromFamilyId,
        toFamilyId: entry.toFamilyId,
        evidence: entry.evidence,
        probability: total > ROUTE_EPSILON ? entry.evidence / total : 0,
      }))
  })
  return { transitions, fingerprint: thoughtDigest('derived-family-graph', modelFingerprint, headFingerprint, transitions) }
}

/** O(N+E) once per episode/head revision; tick routing never scans all neurons or global edges. */
export function createThoughtRuntimeIndex(
  model: ThoughtModel,
  head: ThoughtHead,
  config: ThoughtRuntimeConfig,
  seed: number,
): ThoughtRuntimeIndex {
  const neurons = allThoughtNeurons(model, head)
  const neuronById = new Map(neurons.map((neuron) => [neuron.id, neuron]))
  const relationPatchByNeuron = new Map(head.contextPatches.map((patch) => [patch.neuronId, new Float32Array(patch.delta)]))
  let edgeRows = compiledRows(model, head, config)
  const allCompiled = [...edgeRows.values()].flat()
  if (allCompiled.length > config.maximumTotalHabitEdges) {
    const retained = new Set(allCompiled
      .sort((left, right) => right.baseLogit - left.baseLogit || left.fromNeuronId.localeCompare(right.fromNeuronId) || left.toNeuronId.localeCompare(right.toNeuronId))
      .slice(0, config.maximumTotalHabitEdges)
      .map((edge) => `${edge.fromNeuronId}\u0000${edge.toNeuronId}`))
    edgeRows = new Map([...edgeRows].flatMap(([origin, row]) => {
      const filtered = row.filter((edge) => retained.has(`${origin}\u0000${edge.toNeuronId}`))
      return filtered.length
        ? [[origin, softmax(filtered, (edge) => edge.baseLogit).map((edge) => ({ ...edge, baseProbability: edge.probability }))] as const]
        : []
    }))
  }
  const familyMembers = new Map<string, Array<{ neuronId: string; weight: number }>>()
  const membershipWeightByFamily = new Map<string, Map<string, number>>()
  for (const neuron of neurons) {
    for (const membership of neuron.familyMemberships) {
      const members = familyMembers.get(membership.familyId) ?? []
      members.push({ neuronId: neuron.id, weight: membership.weight })
      familyMembers.set(membership.familyId, members)
      const weights = membershipWeightByFamily.get(membership.familyId) ?? new Map<string, number>()
      weights.set(neuron.id, membership.weight)
      membershipWeightByFamily.set(membership.familyId, weights)
    }
  }
  for (const members of familyMembers.values()) {
    members.sort((left, right) => right.weight - left.weight || left.neuronId.localeCompare(right.neuronId))
  }
  const explorationCount = Math.max(1, Math.ceil(config.maximumCandidatePool * config.explorationReserve))
  const explorationNeuronIds = neurons.map((neuron) => neuron.id)
    .sort((left, right) => seededThoughtUnit(seed, 'explore', left) - seededThoughtUnit(seed, 'explore', right) || left.localeCompare(right))
    .slice(0, explorationCount)
  const familyGraph = deriveFamilyGraphFromIndex(neurons, edgeRows, model.fingerprint, head.fingerprint)
  const familyRows = new Map<string, DerivedFamilyGraph['transitions']>()
  const familyRowTailEvidence = new Map<string, number>()
  for (const transition of familyGraph.transitions) {
    const row = familyRows.get(transition.fromFamilyId) ?? []
    row.push(transition)
    familyRows.set(transition.fromFamilyId, row)
  }
  for (const [familyId, row] of familyRows) {
    const sorted = row.sort((left, right) => right.probability - left.probability || left.toFamilyId.localeCompare(right.toFamilyId))
    const retained = sorted.slice(0, config.maximumCandidatePool)
    familyRows.set(familyId, retained)
    familyRowTailEvidence.set(familyId, sorted.slice(config.maximumCandidatePool)
      .reduce((sum, transition) => sum + transition.evidence, 0))
  }
  const compiledEdgeCount = [...edgeRows.values()].reduce((sum, row) => sum + row.length, 0)
  const signalIndex = Array.from({ length: 4 }, (_, table) => {
    const coordinates = Array.from({ length: 10 }, (_, bit) =>
      Math.floor(seededThoughtUnit(seed, 'signal-lsh', table, bit) * model.layout.cellCount))
    const buckets = new Map<number, string[]>()
    for (const neuron of neurons) {
      const bucket = neuronSignalBucket(neuron, relationPatchByNeuron.get(neuron.id), model.layout, coordinates)
      const ids = buckets.get(bucket) ?? []
      if (ids.length < config.maximumCandidatePool) ids.push(neuron.id)
      buckets.set(bucket, ids)
    }
    return { coordinates, buckets }
  })
  return {
    modelFingerprint: model.fingerprint,
    headFingerprint: head.fingerprint,
    seed,
    neurons,
    neuronById,
    relationPatchByNeuron,
    edgeRows,
    familyMembers,
    membershipWeightByFamily,
    familyRows,
    familyRowTailEvidence,
    explorationNeuronIds,
    temporaryNeuronIds: [],
    familyGraph,
    compiledNeuronCount: neurons.length,
    compiledEdgeCount,
    signalIndex,
  }
}

/** Bounded multi-table sign-LSH query; no query-time N scan. */
export function queryThoughtSignalIndex(
  index: ThoughtRuntimeIndex,
  signal: ArrayLike<number>,
  maximum: number,
): string[] {
  const ids: string[] = []
  const seen = new Set<string>()
  for (const table of index.signalIndex) {
    for (const id of table.buckets.get(signalBucket(signal, table.coordinates)) ?? []) {
      if (seen.has(id)) continue
      seen.add(id); ids.push(id)
      if (ids.length >= maximum) return ids
    }
  }
  return ids
}

class OverlayReadonlyMap<K, V> implements ReadonlyMap<K, V> {
  constructor(private readonly base: ReadonlyMap<K, V>, private readonly overlay: ReadonlyMap<K, V>) {}
  get size(): number { return this.base.size + [...this.overlay.keys()].filter((key) => !this.base.has(key)).length }
  get(key: K): V | undefined { return this.overlay.get(key) ?? this.base.get(key) }
  has(key: K): boolean { return this.overlay.has(key) || this.base.has(key) }
  *entries(): MapIterator<[K, V]> {
    const yielded = new Set<K>()
    for (const entry of this.overlay) { yielded.add(entry[0]); yield entry }
    for (const entry of this.base) if (!yielded.has(entry[0])) yield entry
  }
  *keys(): MapIterator<K> { for (const [key] of this.entries()) yield key }
  *values(): MapIterator<V> { for (const [, value] of this.entries()) yield value }
  [Symbol.iterator](): MapIterator<[K, V]> { return this.entries() }
  forEach(callbackfn: (value: V, key: K, map: ReadonlyMap<K, V>) => void, thisArg?: unknown): void {
    for (const [key, value] of this.entries()) callbackfn.call(thisArg, value, key, this)
  }
}

/** Adds a tiny lookup overlay; it never copies or iterates the persistent N-sized maps. */
export function withTemporaryThoughtNeurons(
  base: ThoughtRuntimeIndex,
  temporary: readonly ThoughtNeuron[],
): ThoughtRuntimeIndex {
  if (!temporary.length) return base
  const overlay = new Map<string, ThoughtNeuron>()
  const temporaryNeuronIds: string[] = []
  for (const neuron of temporary) {
    if (base.neuronById.has(neuron.id) || overlay.has(neuron.id)) continue
    overlay.set(neuron.id, neuron)
    temporaryNeuronIds.push(neuron.id)
  }
  return {
    ...base,
    neuronById: new OverlayReadonlyMap(base.neuronById, overlay),
    temporaryNeuronIds,
  }
}

export function deriveFamilyGraph(index: ThoughtRuntimeIndex): DerivedFamilyGraph { return index.familyGraph }

function representation(index: ThoughtRuntimeIndex, model: ThoughtModel, branch: ThoughtBranch, neuron: ThoughtNeuron): Float32Array {
  return composeNeuronRepresentation(
    neuron.identity,
    neuron.relation,
    model.layout,
    index.relationPatchByNeuron.get(neuron.id),
    branch.fastRelationPatches.find((patch) => patch.neuronId === neuron.id)?.delta,
  )
}

function rowWithFastEvidence(
  index: ThoughtRuntimeIndex,
  branch: ThoughtBranch,
  config: ThoughtRuntimeConfig,
): CompiledThoughtEdge[] {
  const byTarget = new Map((index.edgeRows.get(branch.currentNeuronId) ?? []).map((edge) => [edge.toNeuronId, { ...edge }]))
  for (const trace of branch.fastTraces) {
    if (trace.fromNeuronId !== branch.currentNeuronId || byTarget.has(trace.toNeuronId)) continue
    byTarget.set(trace.toNeuronId, {
      fromNeuronId: branch.currentNeuronId,
      toNeuronId: trace.toNeuronId,
      baseH0: 0,
      persistentExcitation: 0,
      persistentInhibition: 0,
      trustMass: 0,
      updateCount: 0,
      baseLogit: Math.log(ROUTE_EPSILON),
      baseProbability: 0,
    })
  }
  const logits = [...byTarget.values()].map((edge) => {
    const fast = branch.fastTraces.find((trace) => trace.fromNeuronId === branch.currentNeuronId && trace.toNeuronId === edge.toNeuronId)
    return {
      ...edge,
      baseLogit: Math.log(edge.baseH0 + edge.persistentExcitation + (fast?.excitation ?? 0) + ROUTE_EPSILON) -
        Math.log(1 + edge.persistentInhibition + (fast?.inhibition ?? 0)),
    }
  }).sort((left, right) => right.baseLogit - left.baseLogit || left.toNeuronId.localeCompare(right.toNeuronId))
    .slice(0, config.maximumHabitEdgesPerNeuron)
  return softmax(logits, (edge) => edge.baseLogit).map((edge) => ({ ...edge, baseProbability: edge.probability }))
}

function familyCandidates(
  index: ThoughtRuntimeIndex,
  model: ThoughtModel,
  branch: ThoughtBranch,
  current: ThoughtNeuron,
  currentRow: readonly CompiledThoughtEdge[],
  hints: ThoughtRoutingHints,
  config: ThoughtRuntimeConfig,
  options: ThoughtStepOptions,
  compatibilityForNeuron: (neuron: ThoughtNeuron) => number,
): { candidates: ThoughtFamilyRouteCandidate[]; inspected: number; unmatchedFamilyProbability: number } {
  const graphScores = new Map<string, number>()
  let graphTailScore = 0
  const baseRow = index.edgeRows.get(current.id) ?? []
  for (const membership of current.familyMemberships) {
    const localEvidence = new Map(
      (index.familyRows.get(membership.familyId) ?? []).map((transition) => [transition.toFamilyId, transition.evidence]),
    )
    let localTailEvidence = index.familyRowTailEvidence.get(membership.familyId) ?? 0
    const projectRow = (row: readonly CompiledThoughtEdge[], sign: number) => {
      for (const edge of row) {
        const target = index.neuronById.get(edge.toNeuronId)
        if (!target) continue
        for (const targetMembership of target.familyMemberships) {
          const delta = sign * membership.weight * edge.baseProbability * targetMembership.weight
          if (localEvidence.has(targetMembership.familyId)) {
            localEvidence.set(targetMembership.familyId, Math.max(0,
              (localEvidence.get(targetMembership.familyId) ?? 0) + delta,
            ))
          } else if (sign < 0) {
            localTailEvidence = Math.max(0, localTailEvidence + delta)
          } else if (delta > 0) {
            localEvidence.set(targetMembership.familyId, delta)
          }
        }
      }
    }
    projectRow(baseRow, -1)
    projectRow(currentRow, 1)
    const total = [...localEvidence.values()].reduce((sum, value) => sum + value, localTailEvidence)
    for (const [toFamilyId, evidence] of localEvidence) {
      if (evidence <= 0 || total <= ROUTE_EPSILON) continue
      graphScores.set(toFamilyId, (graphScores.get(toFamilyId) ?? 0) + membership.weight * evidence / total)
    }
    if (localTailEvidence > 0 && total > ROUTE_EPSILON) {
      graphTailScore += membership.weight * localTailEvidence / total
    }
  }
  for (const membership of current.familyMemberships) if (!graphScores.has(membership.familyId)) graphScores.set(membership.familyId, 0)
  for (const familyId of Object.keys(hints.coalitionFamilyWeights ?? {})) if (index.familyMembers.has(familyId)) graphScores.set(familyId, graphScores.get(familyId) ?? 0)
  for (const familyId of Object.keys(hints.goalFamilyWeights ?? {})) if (index.familyMembers.has(familyId)) graphScores.set(familyId, graphScores.get(familyId) ?? 0)
  const proposedMembersByFamily = new Map<string, string[]>()
  const signalFamilyAffinity = new Map<string, number>()
  const signalCompatibilityByNeuron = new Map<string, number>()
  let inspected = 0
  const exploratorySources = [
    [...index.temporaryNeuronIds],
    queryThoughtSignalIndex(index, branch.signal, config.maximumCandidatePool),
    [...(options.teacherEdgeNeuronIds ?? [])].slice(0, config.maximumCandidatePool),
    [...(options.candidateNeuronIds ?? [])].slice(0, config.maximumCandidatePool),
    [...index.explorationNeuronIds],
  ]
  const exploratoryIds: string[] = []
  const exploratorySeen = new Set<string>()
  const maximumSourceLength = exploratorySources.reduce((maximum, source) => Math.max(maximum, source.length), 0)
  for (let rank = 0; rank < maximumSourceLength && exploratoryIds.length < config.maximumEvaluatedCandidates; rank += 1) {
    for (const source of exploratorySources) {
      const neuronId = source[rank]
      if (!neuronId || exploratorySeen.has(neuronId)) continue
      exploratorySeen.add(neuronId)
      exploratoryIds.push(neuronId)
      if (exploratoryIds.length >= config.maximumEvaluatedCandidates) break
    }
  }
  for (const neuronId of exploratoryIds) {
    const neuron = index.neuronById.get(neuronId)
    if (!neuron) continue
    const compatibility = compatibilityForNeuron(neuron)
    signalCompatibilityByNeuron.set(neuron.id, compatibility)
    inspected += 1
    for (const membership of neuron.familyMemberships) {
      if (!graphScores.has(membership.familyId)) graphScores.set(membership.familyId, 0)
      signalFamilyAffinity.set(membership.familyId, Math.max(
        signalFamilyAffinity.get(membership.familyId) ?? 0,
        compatibility * membership.weight,
      ))
      const ids = proposedMembersByFamily.get(membership.familyId) ?? []
      if (ids.length < 2 && !ids.includes(neuron.id)) ids.push(neuron.id)
      proposedMembersByFamily.set(membership.familyId, ids)
    }
  }
  const proposedFamilies = new Set<string>()
  const addTop = (entries: Array<[string, number]>) => {
    for (const [familyId] of entries.slice(0, config.maximumCandidateFamilies)) proposedFamilies.add(familyId)
  }
  addTop([...graphScores].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])))
  addTop(Object.entries(hints.coalitionFamilyWeights ?? {})
    .filter(([familyId]) => graphScores.has(familyId))
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])))
  addTop(Object.entries(hints.goalFamilyWeights ?? {})
    .filter(([familyId]) => graphScores.has(familyId))
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])))
  addTop([...signalFamilyAffinity].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])))
  const prelim = [...proposedFamilies].sort().map((familyId) => [familyId, graphScores.get(familyId) ?? 0] as [string, number])
  for (const [familyId, probability] of graphScores) if (!proposedFamilies.has(familyId)) graphTailScore += probability
  const recentFamilies = new Map<string, number>()
  for (const neuronId of branch.path.slice(-config.repetitionWindow)) {
    const neuron = index.neuronById.get(neuronId)
    if (!neuron) continue
    for (const membership of neuron.familyMemberships) recentFamilies.set(membership.familyId, (recentFamilies.get(membership.familyId) ?? 0) + membership.weight)
  }
  const raw = prelim.map(([familyId, graphProbability]) => {
    let signalCompatibility = 0
    const signalMemberIds = [...new Set([
      ...(proposedMembersByFamily.get(familyId) ?? []),
      ...(index.familyMembers.get(familyId) ?? []).slice(0, 2).map((member) => member.neuronId),
    ])].sort((left, right) => (signalCompatibilityByNeuron.get(right) ?? -1) -
      (signalCompatibilityByNeuron.get(left) ?? -1) || left.localeCompare(right)).slice(0, 2)
    for (const neuronId of signalMemberIds) {
      const neuron = index.neuronById.get(neuronId)
      if (!neuron) continue
      let compatibility = signalCompatibilityByNeuron.get(neuron.id)
      if (compatibility === undefined) {
        compatibility = compatibilityForNeuron(neuron)
        signalCompatibilityByNeuron.set(neuron.id, compatibility)
        inspected += 1
      }
      signalCompatibility = Math.max(signalCompatibility, compatibility)
    }
    const coalitionSupport = clamp(hints.coalitionFamilyWeights?.[familyId] ?? 0)
    const goalSupport = clamp(hints.goalFamilyWeights?.[familyId] ?? 0)
    const repetitionPenalty = clamp((recentFamilies.get(familyId) ?? 0) / Math.max(1, config.repetitionWindow) * config.repetitionPenalty)
    const uniform = prelim.length ? 1 / prelim.length : 0
    const exploredGraph = (1 - config.explorationReserve) * graphProbability + config.explorationReserve * uniform
    const signalEvidence = -Math.log(Math.max(ROUTE_EPSILON, 1 - 0.99 * signalCompatibility))
    const logit = Math.log(exploredGraph + ROUTE_EPSILON) + signalEvidence + coalitionSupport + goalSupport - repetitionPenalty
    return { familyId, evidence: graphProbability, graphProbability, signalCompatibility, coalitionSupport, goalSupport, repetitionPenalty, logit }
  })
  const ranked = raw.sort((left, right) => right.logit - left.logit || left.familyId.localeCompare(right.familyId))
  const finalists = ranked.slice(0, config.maximumCandidateFamilies)
  const omitted = ranked.slice(config.maximumCandidateFamilies)
  const tailLogit = Math.log(graphTailScore + ROUTE_EPSILON)
  const maximumLogit = Math.max(tailLogit, ...ranked.map((candidate) => candidate.logit))
  const finalWeights = finalists.map((candidate) => Math.exp(candidate.logit - maximumLogit))
  const omittedWeight = omitted.reduce((sum, candidate) => sum + Math.exp(candidate.logit - maximumLogit), 0)
  const tailWeight = Math.exp(tailLogit - maximumLogit) + omittedWeight
  const totalWeight = finalWeights.reduce((sum, weight) => sum + weight, tailWeight)
  return {
    candidates: finalists.map((candidate, candidateIndex) => ({ ...candidate, probability: finalWeights[candidateIndex] / totalWeight })),
    inspected,
    unmatchedFamilyProbability: tailWeight / totalWeight,
  }
}

function boundedPool(
  index: ThoughtRuntimeIndex,
  row: readonly CompiledThoughtEdge[],
  families: readonly ThoughtFamilyRouteCandidate[],
  options: ThoughtStepOptions,
  currentNeuronId: string,
  maximum: number,
  signal: ArrayLike<number>,
): string[] {
  const ids: string[] = []
  const seen = new Set<string>([currentNeuronId])
  const explorationCount = Math.max(1, Math.ceil(maximum * 0.02))
  const ordinaryLimit = Math.max(1, maximum - explorationCount)
  const add = (id: string, limit = ordinaryLimit) => {
    if (ids.length >= limit || seen.has(id) || !index.neuronById.has(id)) return
    seen.add(id)
    ids.push(id)
  }
  for (const edge of row) add(edge.toNeuronId)
  for (const id of index.temporaryNeuronIds) add(id)
  const signalQuota = Math.max(1, Math.ceil(maximum / 16))
  for (const id of queryThoughtSignalIndex(index, signal, maximum).slice(0, signalQuota)) add(id)
  for (const id of options.teacherEdgeNeuronIds ?? []) add(id)
  for (const id of options.candidateNeuronIds ?? []) add(id)
  let memberRank = 0
  while (ids.length < maximum) {
    let added = false
    for (const family of families) {
      const member = index.familyMembers.get(family.familyId)?.[memberRank]
      if (member) { const before = ids.length; add(member.neuronId); added ||= ids.length > before }
    }
    if (!added) break
    memberRank += 1
  }
  for (const id of index.explorationNeuronIds) add(id, maximum)
  return ids
}

export function routeThoughtBranch(
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  branch: ThoughtBranch,
  config: ThoughtRuntimeConfig,
  options: ThoughtStepOptions = {},
  hints: ThoughtRoutingHints = {},
): ThoughtRouteResult {
  if (index.modelFingerprint !== model.fingerprint) throw new TypeError('Routing index modelle eşleşmiyor.')
  if (options.adaptiveRegionContext) {
    validateThoughtAdaptiveRegionContext(options.adaptiveRegionContext, model.layout)
  }
  const current = index.neuronById.get(branch.currentNeuronId)
  if (!current) throw new RangeError(`Geçerli nöron bulunamadı: ${branch.currentNeuronId}`)
  const adaptiveByNeuron = new Map<string, ThoughtAdaptiveCompatibilityTelemetryV1>()
  const compatibilityByNeuron = new Map<string, number>()
  const compatibilityForNeuron = (neuron: ThoughtNeuron): number => {
    const cached = compatibilityByNeuron.get(neuron.id)
    if (cached !== undefined) return cached
    const candidateRepresentation = representation(index, model, branch, neuron)
    if (!options.adaptiveRegionContext) {
      const compatibility = positiveCompatibility(branch.signal, candidateRepresentation)
      compatibilityByNeuron.set(neuron.id, compatibility)
      return compatibility
    }
    const adaptive = evaluateThoughtAdaptiveCompatibility(
      branch.signal,
      candidateRepresentation,
      model.layout,
      options.adaptiveRegionContext,
    )
    adaptiveByNeuron.set(neuron.id, adaptive)
    compatibilityByNeuron.set(neuron.id, adaptive.effectiveCompatibility)
    return adaptive.effectiveCompatibility
  }
  const row = rowWithFastEvidence(index, branch, config)
  const edgeByTarget = new Map(row.map((edge) => [edge.toNeuronId, edge]))
  const fastByTarget = new Map(branch.fastTraces.filter((trace) => trace.fromNeuronId === branch.currentNeuronId).map((trace) => [trace.toNeuronId, trace]))
  const signalCandidates = queryThoughtSignalIndex(index, branch.signal, config.maximumCandidatePool)
  const boundedOptions: ThoughtStepOptions = {
    ...options,
    candidateNeuronIds: [...new Set([
      ...(options.candidateNeuronIds ?? []).slice(0, config.maximumCandidatePool),
      ...signalCandidates.slice(0, Math.max(1, Math.ceil(config.maximumCandidatePool / 16))),
    ])],
  }
  const familyStage = familyCandidates(
    index,
    model,
    branch,
    current,
    row,
    hints,
    config,
    boundedOptions,
    compatibilityForNeuron,
  )
  const poolIds = boundedPool(index, row, familyStage.candidates, boundedOptions, current.id, config.maximumCandidatePool, branch.signal)
  const poolSet = new Set(poolIds)
  const familyWork = familyStage.candidates.map((family) => {
    const entries = poolIds.map((neuronId) => {
      const membershipWeight = index.membershipWeightByFamily.get(family.familyId)?.get(neuronId) ??
        index.neuronById.get(neuronId)?.familyMemberships.find((membership) => membership.familyId === family.familyId)?.weight ?? 0
      const edge = edgeByTarget.get(neuronId)
      return { neuronId, membershipWeight, edge, routeEvidence: (edge?.baseProbability ?? 0) * membershipWeight }
    }).filter((entry) => entry.membershipWeight > 0)
      .sort((left, right) => right.routeEvidence - left.routeEvidence || left.neuronId.localeCompare(right.neuronId))
    return { family, entries }
  })
  const neuronCandidates: ThoughtNeuronRouteCandidate[] = []
  let unmatchedProbability = familyStage.unmatchedFamilyProbability
  let inspected = familyStage.inspected
  let remainingBudget = config.maximumEvaluatedCandidates
  for (let familyIndex = 0; familyIndex < familyWork.length; familyIndex += 1) {
    if (remainingBudget <= 0) {
      for (const skipped of familyWork.slice(familyIndex)) unmatchedProbability += skipped.family.probability
      break
    }
    const work = familyWork[familyIndex]
    const familiesLeft = familyWork.length - familyIndex
    const quota = Math.max(1, Math.floor(remainingBudget / Math.max(1, familiesLeft)))
    const entries = work.entries.slice(0, quota)
    const maximumR = entries.reduce((maximum, entry) => Math.max(maximum, entry.routeEvidence), 0)
    let reach = 1
    for (let rank = 0; rank < entries.length; rank += 1) {
      const entry = entries[rank]
      const neuron = index.neuronById.get(entry.neuronId)
      if (!neuron || !poolSet.has(neuron.id)) continue
      inspected += 1
      const edge = entry.edge
      const fast = fastByTarget.get(neuron.id)
      const compatibility = compatibilityForNeuron(neuron)
      const repeats = branch.path.slice(-config.repetitionWindow).filter((id) => id === neuron.id).length
      const repetitionPenalty = clamp(repeats * config.repetitionPenalty)
      const q = clamp(compatibility)
      const normalizedHabit = maximumR > ROUTE_EPSILON ? entry.routeEvidence / maximumR : 0
      const acceptanceProbability = clamp(1 - (1 - q) ** (1 + normalizedHabit))
      const selectionProbability = work.family.probability * reach * acceptanceProbability
      neuronCandidates.push({
        neuronId: neuron.id,
        familyId: work.family.familyId,
        rank,
        baseH0: edge?.baseH0 ?? 0,
        persistentExcitation: edge?.persistentExcitation ?? 0,
        fastExcitation: fast?.excitation ?? 0,
        persistentInhibition: edge?.persistentInhibition ?? 0,
        fastInhibition: fast?.inhibition ?? 0,
        habitLogit: edge?.baseLogit ?? Math.log(ROUTE_EPSILON),
        habitProbability: edge?.baseProbability ?? 0,
        membershipWeight: entry.membershipWeight,
        routeEvidence: entry.routeEvidence,
        normalizedHabit,
        compatibility,
        repetitionPenalty,
        q,
        acceptanceProbability,
        reachProbability: reach,
        selectionProbability,
      })
      reach *= 1 - acceptanceProbability
    }
    remainingBudget -= entries.length
    unmatchedProbability += work.family.probability * reach
  }
  if (!familyWork.length) unmatchedProbability = 1
  unmatchedProbability = clamp(unmatchedProbability)
  const distribution = [...neuronCandidates.map((candidate) => candidate.selectionProbability), unmatchedProbability]
  const entropy = shannonEntropy(distribution)
  const adaptiveRegionCandidates = options.adaptiveRegionContext
    ? neuronCandidates.slice(0, MAXIMUM_ADAPTIVE_CANDIDATE_TELEMETRY)
      .map((candidate) => {
        const telemetry = adaptiveByNeuron.get(candidate.neuronId)
        return telemetry ? createThoughtAdaptiveCandidateTelemetry(candidate.neuronId, telemetry) : null
      })
      .filter((candidate): candidate is NonNullable<typeof candidate> => candidate !== null)
    : null
  return {
    fromNeuronId: current.id,
    familyCandidates: familyStage.candidates,
    neuronCandidates,
    unmatchedProbability,
    maximumCompatibility: neuronCandidates.reduce((maximum, candidate) => Math.max(maximum, candidate.compatibility), 0),
    totalProbability: neuronCandidates.reduce((sum, candidate) => sum + candidate.selectionProbability, unmatchedProbability),
    entropy,
    branchCount: entropyBranchCount(distribution, config),
    candidatePoolCount: poolIds.length,
    evaluatedCandidateCount: neuronCandidates.length,
    inspectedNeuronCount: inspected,
    ...(adaptiveRegionCandidates ? { adaptiveRegionCandidates } : {}),
  }
}
