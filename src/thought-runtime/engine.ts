import { DEFAULT_THOUGHT_RUNTIME_CONFIG } from './config'
import {
  deterministicWeightedSample,
  replayFingerprint,
  thoughtDigest,
  thoughtPartitionDigest,
  thoughtVectorDigest,
} from './deterministic'
import {
  clamp,
  composeNeuronRepresentation,
  entropyBranchCount,
  cosineSimilarity,
  positiveCompatibility,
  transformThoughtSignal,
  unexplainedResidual,
  updateFastRelationPatch,
  vectorNorm,
  projectThoughtVector,
} from './math'
import { assertNeuronIdentityIntegrity, assertThoughtIdentityIntegrity, createThoughtNeuron } from './model'
import { queryThoughtSignalIndex, routeThoughtBranch, withTemporaryThoughtNeurons, type ThoughtRuntimeIndex } from './routing'
import {
  adaptiveRegionRelationWriteWeights,
  assertThoughtAdaptiveRegionEpisodeBinding,
  cloneThoughtAdaptiveRegionBinding,
  createThoughtAdaptiveBranchTelemetry,
  evaluateThoughtAdaptiveCompatibility,
} from './adaptive-regions'
import {
  applyThoughtMacroEffect,
  rankThoughtMacroCandidates,
  type ThoughtMacroApplicationV1,
  type ThoughtMacroFallbackV1,
  type ThoughtMacroRankingV1,
  type ThoughtRankedMacroCandidateV1,
} from './concept-macro-runtime'
import {
  THOUGHT_MATRIX_CELLS,
  type StartThoughtEpisodeOptions,
  type ThoughtBirthEvent,
  type ThoughtAdaptiveBranchTelemetryV1,
  type ThoughtBranch,
  type ThoughtEpisodeState,
  type ThoughtFastRelationPatch,
  type ThoughtFastTrace,
  type ThoughtHead,
  type ThoughtModel,
  type ThoughtNeuron,
  type ThoughtNeuronRouteCandidate,
  type ThoughtRouteResult,
  type ThoughtRoutingHints,
  type ThoughtRuntimeConfig,
  type ThoughtStepOptions,
  type ThoughtTick,
  type ThoughtTickBranchTrace,
  type ThoughtMacroExecutionTraceV1,
  type ThoughtVector,
} from './types'

function cloneTrace(trace: ThoughtFastTrace): ThoughtFastTrace { return { ...trace } }
function clonePatch(patch: ThoughtFastRelationPatch): ThoughtFastRelationPatch {
  return { ...patch, delta: new Float32Array(patch.delta) }
}

export function cloneThoughtBranch(branch: ThoughtBranch): ThoughtBranch {
  return {
    ...branch,
    signal: new Float32Array(branch.signal),
    path: [...branch.path],
    transitions: branch.transitions.map((transition) => ({ ...transition })),
    fastTraces: branch.fastTraces.map(cloneTrace),
    fastRelationPatches: branch.fastRelationPatches.map(clonePatch),
    pressure: { ...branch.pressure },
    residual: {
      ...branch.residual,
      sum: new Float32Array(branch.residual.sum),
      cohesionUnitSum: new Float32Array(branch.residual.cohesionUnitSum),
      originEvidence: branch.residual.originEvidence.map((entry) => ({ ...entry })),
      familyEvidence: branch.residual.familyEvidence.map((entry) => ({ ...entry })),
    },
  }
}

function branchFingerprint(branch: ThoughtBranch): string {
  return thoughtDigest('branch', branch.id, branch.parentId, branch.currentNeuronId,
    thoughtVectorDigest(branch.signal), branch.mass, branch.cumulativeLogProbability,
    branch.path, branch.transitions, branch.fastTraces,
    branch.fastRelationPatches.map((patch) => [patch.neuronId, patch.eligibility, patch.visits, thoughtDigest('fast-r', patch.delta)]),
    branch.pressure, branch.residual.weight, branch.residual.observationCount,
    thoughtDigest('residual', branch.residual.sum),
    thoughtDigest('residual-unit-sum', branch.residual.cohesionUnitSum),
    branch.residual.originEvidence, branch.residual.familyEvidence, branch.status)
}

function episodeFingerprint(episode: Omit<ThoughtEpisodeState, 'fingerprint'>): string {
  return thoughtDigest('episode', episode.runKey, episode.seed, episode.modelFingerprint,
    episode.headRevision, episode.headFingerprint, episode.initialSignalDigest,
    episode.tick, episode.maximumTicks,
    episode.branches.map(branchFingerprint), replayFingerprint(episode.replay),
    episode.temporaryBirths.map((neuron) => [neuron.id, neuron.identityDigest, thoughtDigest('temporary-r', neuron.relation)]),
    ...(episode.adaptiveRegion ? [episode.adaptiveRegion] : []))
}

function emptyResidual() {
  return {
    sum: new Float32Array(THOUGHT_MATRIX_CELLS),
    weight: 0,
    observationCount: 0,
    pressureTotal: 0,
    cohesionTotal: 0,
    cohesionComparisons: 0,
    cohesionUnitSum: new Float32Array(THOUGHT_MATRIX_CELLS),
    originEvidence: [] as Array<{ key: string; weight: number }>,
    familyEvidence: [] as Array<{ key: string; weight: number }>,
  }
}

function representation(index: ThoughtRuntimeIndex, model: ThoughtModel, neuron: ThoughtNeuron, branch?: ThoughtBranch): ThoughtVector {
  return composeNeuronRepresentation(
    neuron.identity,
    neuron.relation,
    model.layout,
    index.relationPatchByNeuron.get(neuron.id),
    branch?.fastRelationPatches.find((patch) => patch.neuronId === neuron.id)?.delta,
  )
}

function initialCandidates(
  signal: ThoughtVector,
  index: ThoughtRuntimeIndex,
  model: ThoughtModel,
  options: StartThoughtEpisodeOptions,
  config: ThoughtRuntimeConfig,
): Array<{ neuron: ThoughtNeuron; compatibility: number }> {
  const ids: string[] = []
  const seen = new Set<string>()
  const add = (id: string) => {
    if (ids.length >= config.maximumCandidatePool || seen.has(id) || !index.neuronById.has(id)) return
    seen.add(id); ids.push(id)
  }
  for (const id of options.initialNeuronIds ?? []) add(id)
  // Preview/explicit candidates are a hint, never an exclusive gate. The
  // signal index and a bounded exploration reserve must remain reachable.
  for (const id of queryThoughtSignalIndex(index, signal, config.maximumCandidatePool)) add(id)
  for (const id of index.explorationNeuronIds) add(id)
  for (const members of index.familyMembers.values()) {
    if (members[0]) add(members[0].neuronId)
    if (ids.length >= config.maximumCandidatePool) break
  }
  return ids.map((id) => {
    const neuron = index.neuronById.get(id)!
    const candidateRepresentation = representation(index, model, neuron)
    const compatibility = options.adaptiveRegionContext
      ? evaluateThoughtAdaptiveCompatibility(
        signal,
        candidateRepresentation,
        model.layout,
        options.adaptiveRegionContext,
      ).effectiveCompatibility
      : positiveCompatibility(signal, candidateRepresentation)
    return { neuron, compatibility }
  }).sort((left, right) => right.compatibility - left.compatibility || left.neuron.id.localeCompare(right.neuron.id))
    .slice(0, config.maximumEvaluatedCandidates)
}

export function startThoughtEpisode(
  model: ThoughtModel,
  head: ThoughtHead,
  index: ThoughtRuntimeIndex,
  initialSignal: ThoughtVector,
  options: StartThoughtEpisodeOptions = {},
  config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG,
): ThoughtEpisodeState {
  assertThoughtIdentityIntegrity(model)
  if (index.headFingerprint !== head.fingerprint || index.modelFingerprint !== model.fingerprint) {
    throw new TypeError('Episode indeksi model/head tabanıyla eşleşmiyor.')
  }
  assertThoughtAdaptiveRegionEpisodeBinding(
    { adaptiveRegion: options.adaptiveRegionContext?.binding },
    options.adaptiveRegionContext,
    model.layout,
  )
  const candidates = initialCandidates(initialSignal, index, model, options, config)
  if (!candidates.length) throw new RangeError('Başlangıç sinyali için bounded aday nöron yok.')
  const raw = candidates.map((candidate) => Math.max(config.minimumBranchProbability, candidate.compatibility ** 2))
  const total = raw.reduce((sum, value) => sum + value, 0)
  const probabilities = raw.map((value) => value / total)
  const count = entropyBranchCount(probabilities, config)
  const episodeId = options.episodeId ?? `episode-${thoughtDigest('episode-id', thoughtVectorDigest(initialSignal), options.seed ?? config.hashSeed).slice(0, 16)}`
  const runKey = options.runKey ?? thoughtDigest(
    'semantic-run',
    thoughtVectorDigest(initialSignal),
    model.fingerprint,
    head.fingerprint,
    options.seed ?? config.hashSeed,
    ...(options.adaptiveRegionContext?.binding.mode === 'active'
      ? [options.adaptiveRegionContext.binding.fingerprint]
      : []),
  )
  const branches = candidates.slice(0, count).map((candidate, rank): ThoughtBranch => ({
    id: `${runKey.slice(0, 24)}:b0:${rank}:${candidate.neuron.id}`,
    parentId: null,
    currentNeuronId: candidate.neuron.id,
    signal: new Float32Array(initialSignal),
    mass: probabilities[rank],
    cumulativeLogProbability: Math.log(Math.max(config.minimumBranchProbability, probabilities[rank])),
    path: [candidate.neuron.id],
    transitions: [],
    fastTraces: [],
    fastRelationPatches: [],
    pressure: { habitDistance: 0, representationError: 1 - candidate.compatibility, edge: 0, birth: 0, smoothedBirth: 0, consecutiveBirthTicks: 0 },
    residual: emptyResidual(),
    status: 'active',
  }))
  const retainedMass = branches.reduce((sum, branch) => sum + branch.mass, 0)
  for (const branch of branches) branch.mass /= retainedMass
  const withoutFingerprint: Omit<ThoughtEpisodeState, 'fingerprint'> = {
    format: 'bombom-thought-episode', version: 1, episodeId, runKey,
    seed: options.seed ?? config.hashSeed,
    modelFingerprint: model.fingerprint,
    headRevision: head.revision,
    headFingerprint: head.fingerprint,
    initialSignalDigest: thoughtVectorDigest(initialSignal),
    tick: 0,
    maximumTicks: Math.min(config.maximumTicks, Math.max(1, Math.trunc(options.maximumTicks ?? config.defaultTicks))),
    branches,
    status: 'active',
    replay: [],
    temporaryBirths: [],
    ...(options.adaptiveRegionContext
      ? { adaptiveRegion: cloneThoughtAdaptiveRegionBinding(options.adaptiveRegionContext.binding) }
      : {}),
  }
  return { ...withoutFingerprint, fingerprint: episodeFingerprint(withoutFingerprint) }
}

function decayFastState(branch: ThoughtBranch, config: ThoughtRuntimeConfig): void {
  branch.fastTraces = branch.fastTraces.map((trace) => ({
    ...trace,
    excitation: trace.excitation * config.fastEligibilityDecay,
    inhibition: trace.inhibition * config.fastEligibilityDecay,
    eligibility: trace.eligibility * config.fastEligibilityDecay,
  })).filter((trace) => trace.eligibility > config.minimumBranchProbability)
  branch.fastRelationPatches = branch.fastRelationPatches.map((patch) => ({
    ...patch,
    eligibility: patch.eligibility * config.fastEligibilityDecay,
  })).filter((patch) => patch.eligibility > config.minimumBranchProbability)
}

function addFastEdge(
  branch: ThoughtBranch,
  from: string,
  to: string,
  kind: 'excitation' | 'inhibition',
  config: ThoughtRuntimeConfig,
  scale = 1,
): void {
  let trace = branch.fastTraces.find((entry) => entry.fromNeuronId === from && entry.toNeuronId === to)
  if (!trace) {
    trace = { fromNeuronId: from, toNeuronId: to, excitation: 0, inhibition: 0, eligibility: 1, visits: 0 }
    branch.fastTraces.push(trace)
  }
  const rate = kind === 'excitation' ? config.fastExcitationRate : config.fastInhibitionRate
  trace[kind] = clamp(trace[kind] + rate * clamp(scale), 0, config.evidenceCeiling)
  trace.eligibility = 1
  trace.visits += 1
}

/** Leaves a fast patch only on the neuron actually traversed this tick. */
function leaveCurrentRelationTrace(
  branch: ThoughtBranch,
  current: ThoughtNeuron,
  incomingSignal: ThoughtVector,
  currentRepresentation: ThoughtVector,
  model: ThoughtModel,
  config: ThoughtRuntimeConfig,
  relationWriteWeights?: Float32Array,
  captureAppliedDelta = false,
): Float32Array | undefined {
  let patch = branch.fastRelationPatches.find((entry) => entry.neuronId === current.id)
  const previousDelta = patch?.delta
  const delta = updateFastRelationPatch(
    incomingSignal,
    currentRepresentation,
    patch?.delta,
    model.layout,
    config.fastExcitationRate,
    patch?.eligibility ?? 1,
    relationWriteWeights,
  )
  if (!patch) branch.fastRelationPatches.push({ neuronId: current.id, delta, eligibility: 1, visits: 1 })
  else { patch.delta = delta; patch.eligibility = 1; patch.visits += 1 }
  if (!captureAppliedDelta) return undefined
  const applied = new Float32Array(model.layout.relationLength)
  for (let index = 0; index < applied.length; index += 1) {
    applied[index] = Math.fround(delta[index] - (previousDelta?.[index] ?? 0))
  }
  return applied
}

function addEvidence(entries: Array<{ key: string; weight: number }>, key: string, weight: number): void {
  const existing = entries.find((entry) => entry.key === key)
  if (existing) existing.weight += weight
  else entries.push({ key, weight })
  entries.sort((left, right) => right.weight - left.weight || left.key.localeCompare(right.key))
}

interface EpisodeNewbornResult {
  neuron: ThoughtNeuron
  contributorBranchIds: string[]
}

function createEpisodeNewborn(
  episodeId: string,
  tick: number,
  branches: readonly ThoughtBranch[],
  existing: readonly ThoughtNeuron[],
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  config: ThoughtRuntimeConfig,
): EpisodeNewbornResult | null {
  if (existing.length >= config.maximumBirthsPerEpisode) return null
  const contributing = branches.filter((branch) => branch.residual.observationCount >= config.minimumResidualObservations && branch.residual.weight > 0)
  const observations = contributing.reduce((sum, branch) => sum + branch.residual.observationCount, 0)
  if (observations < config.minimumResidualObservations) return null
  const residualCohesion = calculateResidualCohesion(contributing)
  if (residualCohesion < config.nurseryCohesionThreshold) return null
  const prototype = averageResidualPrototype(contributing)
  const originEvidence = new Map<string, number>()
  const familyEvidence = new Map<string, number>()
  for (const branch of contributing) {
    for (const evidence of branch.residual.originEvidence) {
      originEvidence.set(evidence.key, (originEvidence.get(evidence.key) ?? 0) +
        evidence.weight / Math.max(config.minimumBranchProbability, branch.residual.weight) / contributing.length)
    }
    for (const evidence of branch.residual.familyEvidence) {
      familyEvidence.set(evidence.key, (familyEvidence.get(evidence.key) ?? 0) +
        evidence.weight / Math.max(config.minimumBranchProbability, branch.residual.weight) / contributing.length)
    }
  }
  if (vectorNorm(prototype.slice(0, model.layout.identityLength)) <= config.minimumBranchProbability) return null
  for (const neuron of existing) {
    const existingZ = composeNeuronRepresentation(neuron.identity, neuron.relation, model.layout)
    if (cosineSimilarity(prototype, existingZ) >= 0.98) return null
  }
  const boundedBaseIds = new Set<string>(index.explorationNeuronIds)
  for (const branch of branches) {
    for (const neuronId of branch.path.slice(-config.repetitionWindow)) boundedBaseIds.add(neuronId)
    for (const edge of index.edgeRows.get(branch.currentNeuronId) ?? []) boundedBaseIds.add(edge.toNeuronId)
  }
  for (const familyId of familyEvidence.keys()) {
    for (const member of (index.familyMembers.get(familyId) ?? []).slice(0, 2)) boundedBaseIds.add(member.neuronId)
  }
  for (const neuronId of queryThoughtSignalIndex(index, prototype, config.maximumCandidatePool)) boundedBaseIds.add(neuronId)
  const familySimilarity = new Map<string, number>()
  for (const neuronId of [...boundedBaseIds].slice(0, config.maximumCandidatePool)) {
    const neuron = index.neuronById.get(neuronId)
    if (!neuron) continue
    const existingZ = composeNeuronRepresentation(
      neuron.identity,
      neuron.relation,
      model.layout,
      index.relationPatchByNeuron.get(neuron.id),
    )
    const similarity = Math.max(0, cosineSimilarity(prototype, existingZ))
    if (similarity >= 0.98) return null
    for (const membership of neuron.familyMemberships) {
      familySimilarity.set(membership.familyId, Math.max(
        familySimilarity.get(membership.familyId) ?? 0,
        similarity * membership.weight,
      ))
    }
  }
  const retainedFamilies = [...familySimilarity].filter(([, score]) => score > config.minimumBranchProbability)
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, config.maximumFamilyMembershipsPerNeuron)
  const familyTotal = retainedFamilies.reduce((sum, [, weight]) => sum + weight, 0)
  const id = `temp-${thoughtDigest('episode-newborn', episodeId, tick, prototype).slice(0, 24)}`
  const newbornIdentity = projectThoughtVector(prototype.slice(0, model.layout.identityLength), 1)
  const newbornRelation = projectThoughtVector(prototype.slice(model.layout.relationStart), 1)
  const provisionalFamilyId = `nursery-provisional:${thoughtDigest(
    'provisional-family', thoughtPartitionDigest('identity', newbornIdentity),
  ).slice(0, 16)}`
  const familyMemberships = retainedFamilies.length
    ? retainedFamilies.map(([familyId, weight]) => ({ familyId, weight: weight / familyTotal }))
    : [{ familyId: provisionalFamilyId, weight: 1 }]
  const neuron = createThoughtNeuron({
    id,
    identity: newbornIdentity,
    relation: newbornRelation,
    familyMemberships,
    provenance: {
      kind: 'episode-newborn',
      sourceId: episodeId,
      residualCohesion,
      residualObservationCount: observations,
      birthBranchIds: contributing.map((branch) => branch.id).sort(),
      originEvidence: [...originEvidence].map(([key, weight]) => ({ key, weight }))
        .sort((left, right) => right.weight - left.weight || left.key.localeCompare(right.key)),
      familyEvidence: [...familyEvidence].map(([key, weight]) => ({ key, weight }))
        .sort((left, right) => right.weight - left.weight || left.key.localeCompare(right.key)),
    },
  }, model.layout, config)
  return { neuron, contributorBranchIds: contributing.map((branch) => branch.id).sort() }
}

export function calculateResidualCohesion(branches: readonly ThoughtBranch[]): number {
  const cohesionTotal = branches.reduce((sum, branch) => sum + branch.residual.cohesionTotal, 0)
  let crossCohesion = 0
  let crossComparisons = 0
  for (let left = 0; left < branches.length; left += 1) {
    for (let right = left + 1; right < branches.length; right += 1) {
      const leftResidual = branches[left].residual
      const rightResidual = branches[right].residual
      for (let coordinate = 0; coordinate < THOUGHT_MATRIX_CELLS; coordinate += 1) {
        crossCohesion += leftResidual.cohesionUnitSum[coordinate] * rightResidual.cohesionUnitSum[coordinate]
      }
      crossComparisons += leftResidual.observationCount * rightResidual.observationCount
    }
  }
  const cohesionComparisons = branches.reduce((sum, branch) => sum + branch.residual.cohesionComparisons, 0) + crossComparisons
  return cohesionComparisons ? (cohesionTotal + crossCohesion) / cohesionComparisons : 0
}

/** Equal mean of branch-local residual means; branch probability cannot seize newborn identity. */
export function averageResidualPrototype(branches: readonly ThoughtBranch[]): ThoughtVector {
  const valid = branches.filter((branch) => branch.residual.weight > 0)
  if (!valid.length) return new Float32Array(THOUGHT_MATRIX_CELLS)
  const combined = new Float32Array(THOUGHT_MATRIX_CELLS)
  for (const branch of valid) {
    for (let index = 0; index < THOUGHT_MATRIX_CELLS; index += 1) {
      combined[index] += branch.residual.sum[index] / branch.residual.weight / valid.length
    }
  }
  return projectThoughtVector(combined, 1)
}

function updatePressureAndResidual(
  branch: ThoughtBranch,
  route: ThoughtRouteResult,
  signal: ThoughtVector,
  bestRepresentation: ThoughtVector | undefined,
  origin: ThoughtNeuron,
  selected: ThoughtNeuronRouteCandidate | null,
  config: ThoughtRuntimeConfig,
): ThoughtBirthEvent | null {
  const maximumQ = route.neuronCandidates.reduce((maximum, candidate) => Math.max(maximum, candidate.q), 0)
  branch.pressure = calculateThoughtPressure({
    selectedHabitProbability: selected?.habitProbability ?? null,
    candidatePoolCount: route.candidatePoolCount,
    maximumQ,
    unmatchedProbability: route.unmatchedProbability,
    previousSmoothedBirth: branch.pressure.smoothedBirth,
    previousConsecutiveBirthTicks: branch.pressure.consecutiveBirthTicks,
  }, config)
  const { birth, smoothedBirth, consecutiveBirthTicks } = branch.pressure
  if (smoothedBirth < config.birthPressureThreshold) {
    branch.residual = emptyResidual()
    return null
  }
  const residual = unexplainedResidual(signal, bestRepresentation)
  const weight = smoothedBirth
  const residualNorm = vectorNorm(residual)
  if (residualNorm > config.minimumBranchProbability) {
    let dotWithPreviousSum = 0
    for (let index = 0; index < residual.length; index += 1) {
      const unit = residual[index] / residualNorm
      dotWithPreviousSum += unit * branch.residual.cohesionUnitSum[index]
      branch.residual.cohesionUnitSum[index] = Math.fround(branch.residual.cohesionUnitSum[index] + unit)
    }
    branch.residual.cohesionTotal += dotWithPreviousSum
    branch.residual.cohesionComparisons += branch.residual.observationCount
  }
  for (let index = 0; index < THOUGHT_MATRIX_CELLS; index += 1) branch.residual.sum[index] += residual[index] * weight
  branch.residual.weight += weight
  branch.residual.observationCount += 1
  branch.residual.pressureTotal += birth
  addEvidence(branch.residual.originEvidence, origin.id, weight)
  for (const membership of origin.familyMemberships) addEvidence(branch.residual.familyEvidence, membership.familyId, membership.weight * weight)
  if (consecutiveBirthTicks < config.birthPressureConsecutiveTicks) return null
  return { branchId: branch.id, kind: 'pressure-threshold', pressure: smoothedBirth, consecutiveTicks: consecutiveBirthTicks }
}

export function calculateThoughtPressure(input: {
  selectedHabitProbability: number | null
  candidatePoolCount: number
  maximumQ: number
  unmatchedProbability: number
  previousSmoothedBirth: number
  previousConsecutiveBirthTicks: number
}, config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG): ThoughtBranch['pressure'] {
  const habitDistance = input.selectedHabitProbability === null
    ? 1
    : clamp(-Math.log(input.selectedHabitProbability + config.minimumBranchProbability) /
      Math.log(Math.max(2, input.candidatePoolCount)))
  const representationError = clamp(1 - input.maximumQ)
  const edge = habitDistance * (1 - representationError)
  const birth = habitDistance * representationError * clamp(input.unmatchedProbability)
  const smoothedBirth = config.birthPressureMemory * input.previousSmoothedBirth +
    (1 - config.birthPressureMemory) * birth
  const consecutiveBirthTicks = smoothedBirth >= config.birthPressureThreshold
    ? input.previousConsecutiveBirthTicks + 1
    : 0
  return { habitDistance, representationError, edge, birth, smoothedBirth, consecutiveBirthTicks }
}

interface BranchProposal {
  probability: number
  stableKey: string
  materialize: () => MaterializedBranchProposal
}

interface MaterializedBranchProposal {
  branch: ThoughtBranch
  probability: number
  stableKey: string
  trace: ThoughtTickBranchTrace
}

interface ThoughtMacroTraceAudit {
  ranking: ThoughtMacroRankingV1 | null
  execution?: ThoughtMacroExecutionTraceV1 | null
  extraFallbacks?: readonly ThoughtMacroFallbackV1[]
}

function cloneAdaptiveBranchTelemetry(
  telemetry: ThoughtAdaptiveBranchTelemetryV1,
): ThoughtAdaptiveBranchTelemetryV1 {
  const cloneCompatibility = <T extends {
    pageCompatibilities: number[]
    pageWeights: number[]
    gates: ThoughtAdaptiveBranchTelemetryV1['gates']
  }>(value: T): T => ({
    ...value,
    pageCompatibilities: [...value.pageCompatibilities],
    pageWeights: [...value.pageWeights],
    gates: {
      ...value.gates,
      gates: value.gates.gates.map((gate) => ({ ...gate })),
    },
  })
  return cloneCompatibility({
    ...telemetry,
    pageWriteMask: [...telemetry.pageWriteMask],
    pageObservations: telemetry.pageObservations.map((observation) => ({
      ...observation,
      behavior: [...observation.behavior],
    })),
    candidateCompatibilities: telemetry.candidateCompatibilities.map((candidate) =>
      cloneCompatibility({ ...candidate })),
  })
}

function baseTrace(
  branch: ThoughtBranch,
  child: ThoughtBranch,
  beforeSignal: ThoughtVector,
  afterSignal: ThoughtVector,
  signalChangeMagnitude: number,
  route: ThoughtRouteResult,
  selected: ThoughtNeuronRouteCandidate | null,
  hints: ThoughtRoutingHints,
  birthEvent: ThoughtBirthEvent | null,
  macroAudit: ThoughtMacroTraceAudit = { ranking: null },
  adaptiveRegion?: ThoughtAdaptiveBranchTelemetryV1,
): ThoughtTickBranchTrace {
  const ranking = macroAudit.ranking
  return {
    branchId: child.id,
    parentBranchId: branch.id,
    mass: child.mass,
    beforeSignal: Array.from(beforeSignal),
    afterSignal: Array.from(afterSignal),
    beforeSignalDigest: thoughtVectorDigest(beforeSignal),
    afterSignalDigest: thoughtVectorDigest(afterSignal),
    beforeSignalNorm: vectorNorm(beforeSignal),
    afterSignalNorm: vectorNorm(afterSignal),
    signalChangeMagnitude,
    selectedFamilyId: selected?.familyId ?? null,
    selectedNeuronId: selected?.neuronId ?? null,
    familyCandidates: route.familyCandidates.map((candidate) => ({ ...candidate })),
    neuronCandidates: route.neuronCandidates.map((candidate) => ({ ...candidate })),
    unmatchedProbability: route.unmatchedProbability,
    edgePressure: child.pressure.edge,
    birthPressure: child.pressure.birth,
    smoothedBirthPressure: child.pressure.smoothedBirth,
    coalitionFamilyWeights: { ...(hints.coalitionFamilyWeights ?? {}) },
    roleHypotheses: (hints.roleHypotheses ?? []).map((hypothesis) => ({ ...hypothesis })),
    birthEvents: birthEvent ? [birthEvent] : [],
    macroRuntimeFingerprint: ranking?.runtimeFingerprint ?? null,
    rawProbabilityScale: ranking?.rawProbabilityScale ?? 1,
    macroProbabilityMass: ranking?.macroProbabilityMass ?? 0,
    macroCandidates: (ranking?.candidates ?? []).map((candidate) => ({
      macroId: candidate.macroId,
      entryCompatibility: candidate.entryCompatibility,
      activationWeight: candidate.activationWeight,
      selectionProbability: candidate.selectionProbability,
      duration: candidate.duration,
      compressedTransitionCount: candidate.compressedTransitionCount,
      fidelity: candidate.fidelity,
      expandedConstituentIds: [...candidate.expandedConstituentIds],
    })),
    macroExecution: macroAudit.execution ?? null,
    macroFallbacks: [...(ranking?.fallbacks ?? []), ...(macroAudit.extraFallbacks ?? [])]
      .map((entry) => ({ ...entry }))
      .sort((left, right) => (left.macroId ?? '').localeCompare(right.macroId ?? '') || left.reason.localeCompare(right.reason))
      .slice(0, 8),
    ...(adaptiveRegion ? { adaptiveRegion: cloneAdaptiveBranchTelemetry(adaptiveRegion) } : {}),
  }
}

function candidateProposal(
  traversed: ThoughtBranch,
  original: ThoughtBranch,
  beforeSignal: ThoughtVector,
  signalChangeMagnitude: number,
  route: ThoughtRouteResult,
  candidate: ThoughtNeuronRouteCandidate,
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  config: ThoughtRuntimeConfig,
  tick: number,
  runKey: string,
  hints: ThoughtRoutingHints,
  probabilityScale = 1,
  macroRanking: ThoughtMacroRankingV1 | null = null,
  adaptiveRegion?: ThoughtAdaptiveBranchTelemetryV1,
): BranchProposal {
  const effectiveSelectionProbability = candidate.selectionProbability * probabilityScale
  const probability = original.mass * effectiveSelectionProbability
  const stableKey = `${runKey.slice(0, 20)}:b${tick}:${thoughtDigest(
    'branch-child', original.id, candidate.familyId, candidate.rank, candidate.neuronId,
  ).slice(0, 20)}`
  return {
    probability,
    stableKey,
    materialize: () => {
      const child = cloneThoughtBranch(traversed)
      child.parentId = original.id
      child.id = stableKey
      child.mass = probability
      child.cumulativeLogProbability += Math.log(Math.max(config.minimumBranchProbability, effectiveSelectionProbability))
      const sameFamilyEarlier = route.neuronCandidates.filter((entry) => entry.familyId === candidate.familyId && entry.rank < candidate.rank)
      for (const rejected of sameFamilyEarlier) addFastEdge(child, original.currentNeuronId, rejected.neuronId, 'inhibition', config)
      addFastEdge(child, original.currentNeuronId, candidate.neuronId, 'excitation', config)
      const target = index.neuronById.get(candidate.neuronId)
      const origin = index.neuronById.get(original.currentNeuronId)
      if (!target || !origin) throw new RangeError('Yönlendirme nöronu bounded indekste bulunamadı.')
      assertNeuronIdentityIntegrity(target, model.layout)
      assertNeuronIdentityIntegrity(origin, model.layout)
      const bestQ = [...route.neuronCandidates].sort((left, right) => right.q - left.q || left.neuronId.localeCompare(right.neuronId))[0]
      const bestQNeuron = bestQ ? index.neuronById.get(bestQ.neuronId) : undefined
      if (bestQNeuron) assertNeuronIdentityIntegrity(bestQNeuron, model.layout)
      const bestQRepresentation = bestQNeuron ? representation(index, model, bestQNeuron, child) : undefined
      const event = updatePressureAndResidual(child, route, traversed.signal, bestQRepresentation, origin, candidate, config)
      if (child.pressure.edge > 0) {
        addFastEdge(child, original.currentNeuronId, candidate.neuronId, 'excitation', config, child.pressure.edge)
      }
      child.transitions.push({
        tick,
        fromNeuronId: original.currentNeuronId,
        toNeuronId: candidate.neuronId,
        familyId: candidate.familyId,
        compatibility: candidate.compatibility,
        baseH0: candidate.baseH0,
        persistentExcitation: candidate.persistentExcitation,
        fastExcitation: candidate.fastExcitation,
        persistentInhibition: candidate.persistentInhibition,
        fastInhibition: candidate.fastInhibition,
        habitLogit: candidate.habitLogit,
        habitProbability: candidate.habitProbability,
        membershipWeight: candidate.membershipWeight,
        routeEvidence: candidate.routeEvidence,
        normalizedHabit: candidate.normalizedHabit,
        q: candidate.q,
        acceptanceProbability: candidate.acceptanceProbability,
        selectionProbability: candidate.selectionProbability,
        signalChangeMagnitude,
      })
      child.currentNeuronId = candidate.neuronId
      child.path.push(candidate.neuronId)
      child.status = 'active'
      return {
        branch: child,
        probability,
        stableKey,
        trace: baseTrace(
          original, child, beforeSignal, traversed.signal, signalChangeMagnitude,
          route, candidate, hints, event, { ranking: macroRanking }, adaptiveRegion,
        ),
      }
    },
  }
}

function unmatchedProposal(
  traversed: ThoughtBranch,
  original: ThoughtBranch,
  beforeSignal: ThoughtVector,
  signalChangeMagnitude: number,
  route: ThoughtRouteResult,
  origin: ThoughtNeuron,
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  config: ThoughtRuntimeConfig,
  tick: number,
  runKey: string,
  hints: ThoughtRoutingHints,
  probabilityScale = 1,
  macroRanking: ThoughtMacroRankingV1 | null = null,
  adaptiveRegion?: ThoughtAdaptiveBranchTelemetryV1,
): BranchProposal {
  const effectiveSelectionProbability = route.unmatchedProbability * probabilityScale
  const probability = original.mass * effectiveSelectionProbability
  const stableKey = `${runKey.slice(0, 20)}:b${tick}:${thoughtDigest('branch-unmatched', original.id).slice(0, 20)}`
  return {
    probability,
    stableKey,
    materialize: () => {
      const child = cloneThoughtBranch(traversed)
      child.parentId = original.id
      child.id = stableKey
      child.mass = probability
      child.cumulativeLogProbability += Math.log(Math.max(config.minimumBranchProbability, effectiveSelectionProbability))
      const bestCandidate = [...route.neuronCandidates]
        .sort((left, right) => right.q - left.q || left.neuronId.localeCompare(right.neuronId))[0]
      const bestNeuron = bestCandidate ? index.neuronById.get(bestCandidate.neuronId) : undefined
      if (bestNeuron) assertNeuronIdentityIntegrity(bestNeuron, model.layout)
      const bestRepresentation = bestNeuron ? representation(index, model, bestNeuron, child) : undefined
      const event = updatePressureAndResidual(child, route, traversed.signal, bestRepresentation, origin, null, config)
      if (bestCandidate && child.pressure.edge > 0) {
        addFastEdge(child, original.currentNeuronId, bestCandidate.neuronId, 'excitation', config, child.pressure.edge)
      }
      // ∅ is a live self-hold, so consecutive pressure can reach the birth gate.
      child.status = 'active'
      return {
        branch: child,
        probability,
        stableKey,
        trace: baseTrace(
          original, child, beforeSignal, traversed.signal, signalChangeMagnitude,
          route, null, hints, event, { ranking: macroRanking }, adaptiveRegion,
        ),
      }
    },
  }
}

function macroProposal(
  original: ThoughtBranch,
  beforeSignal: ThoughtVector,
  route: ThoughtRouteResult,
  candidate: ThoughtRankedMacroCandidateV1,
  application: ThoughtMacroApplicationV1 & { applied: true },
  ranking: ThoughtMacroRankingV1,
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  config: ThoughtRuntimeConfig,
  tick: number,
  runKey: string,
  hints: ThoughtRoutingHints,
  adaptiveRegion?: ThoughtAdaptiveBranchTelemetryV1,
): BranchProposal {
  const probability = original.mass * candidate.selectionProbability
  const stableKey = `${runKey.slice(0, 20)}:b${tick}:${thoughtDigest(
    'branch-macro-child', original.id, candidate.macroId, candidate.fingerprint,
  ).slice(0, 20)}`
  return {
    probability,
    stableKey,
    materialize: () => {
      const child = cloneThoughtBranch(original)
      decayFastState(child, config)
      child.parentId = original.id
      child.id = stableKey
      child.mass = probability
      child.cumulativeLogProbability += Math.log(Math.max(
        config.minimumBranchProbability,
        candidate.selectionProbability,
      ))
      child.signal = new Float32Array(application.signal)
      child.pressure = {
        habitDistance: 0,
        representationError: 1 - candidate.entryCompatibility,
        edge: 0,
        birth: 0,
        smoothedBirth: config.birthPressureMemory * child.pressure.smoothedBirth,
        consecutiveBirthTicks: 0,
      }
      child.residual = emptyResidual()
      const ids = candidate.expandedConstituentIds
      if (ids[0] !== original.currentNeuronId) {
        throw new RangeError('Makro kökeni aktif branch nöronuyla uyuşmuyor.')
      }
      const edgeCount = Math.max(1, ids.length - 1)
      for (let indexInMacro = 1; indexInMacro < ids.length; indexInMacro += 1) {
        const fromNeuronId = ids[indexInMacro - 1]
        const toNeuronId = ids[indexInMacro]
        const from = index.neuronById.get(fromNeuronId)
        const to = index.neuronById.get(toNeuronId)
        if (!from || !to) throw new RangeError('Makro ham üyesi bounded indekste bulunamadı.')
        assertNeuronIdentityIntegrity(from, model.layout)
        assertNeuronIdentityIntegrity(to, model.layout)
        addFastEdge(child, fromNeuronId, toNeuronId, 'excitation', config)
        const membership = [...to.familyMemberships]
          .sort((left, right) => right.weight - left.weight || left.familyId.localeCompare(right.familyId))[0]
        child.transitions.push({
          tick,
          fromNeuronId,
          toNeuronId,
          familyId: membership?.familyId ?? '',
          compatibility: candidate.entryCompatibility,
          baseH0: 0,
          persistentExcitation: 0,
          fastExcitation: config.fastExcitationRate,
          persistentInhibition: 0,
          fastInhibition: 0,
          habitLogit: 0,
          habitProbability: candidate.selectionProbability,
          membershipWeight: membership?.weight ?? 0,
          routeEvidence: candidate.activationWeight,
          normalizedHabit: candidate.activationWeight,
          q: candidate.entryCompatibility,
          acceptanceProbability: candidate.entryCompatibility,
          selectionProbability: indexInMacro === 1 ? candidate.selectionProbability : 1,
          signalChangeMagnitude: application.signalChangeMagnitude / edgeCount,
          macroId: candidate.macroId,
          macroLogicalDuration: candidate.duration,
        })
      }
      child.path.push(...ids.slice(1))
      child.currentNeuronId = candidate.terminalNeuronId
      child.status = 'active'
      const terminal = index.neuronById.get(candidate.terminalNeuronId)
      const terminalMembership = [...(terminal?.familyMemberships ?? [])]
        .sort((left, right) => right.weight - left.weight || left.familyId.localeCompare(right.familyId))[0]
      const execution: ThoughtMacroExecutionTraceV1 = {
        macroId: candidate.macroId,
        macroHeadFingerprint: candidate.macroHeadFingerprint,
        runtimeFingerprint: ranking.runtimeFingerprint,
        compiled: true,
        expandedConstituentIds: [...candidate.expandedConstituentIds],
        duration: candidate.duration,
        compressedTransitionCount: candidate.compressedTransitionCount,
        fidelity: candidate.fidelity,
        entryCompatibility: candidate.entryCompatibility,
        selectionProbability: candidate.selectionProbability,
        creditMode: 'constituent-edges',
        fallbackReason: null,
      }
      const trace = baseTrace(
        original,
        child,
        beforeSignal,
        application.signal,
        application.signalChangeMagnitude,
        route,
        null,
        hints,
        null,
        { ranking, execution },
        adaptiveRegion,
      )
      trace.selectedFamilyId = terminalMembership?.familyId ?? null
      trace.selectedNeuronId = candidate.terminalNeuronId
      return { branch: child, probability, stableKey, trace }
    },
  }
}

function retainProposals(
  proposals: readonly BranchProposal[],
  episode: ThoughtEpisodeState,
  config: ThoughtRuntimeConfig,
  selectionMode: ThoughtStepOptions['selectionMode'],
): MaterializedBranchProposal[] {
  const viable = proposals.filter((proposal) => proposal.probability >= config.minimumBranchProbability)
  if (!viable.length) return []
  const keys = selectThoughtProposalKeys(viable, episode.seed, episode.tick, config, selectionMode)
  const byKey = new Map(viable.map((proposal) => [proposal.stableKey, proposal]))
  const selected = keys.map((key) => byKey.get(key)).filter((proposal): proposal is BranchProposal => Boolean(proposal))
  const retained = selected.map((proposal) => proposal.materialize())
  const total = retained.reduce((sum, proposal) => sum + proposal.branch.mass, 0)
  for (const proposal of retained) {
    proposal.branch.mass = total > 0 ? proposal.branch.mass / total : 1 / retained.length
    proposal.trace.mass = proposal.branch.mass
  }
  return retained.sort((left, right) => right.branch.mass - left.branch.mass || left.stableKey.localeCompare(right.stableKey))
}

/** Keeps K-1 strongest anchors and reserves at most one seeded residual branch. */
export function selectThoughtProposalKeys(
  proposals: ReadonlyArray<{ probability: number; stableKey: string }>,
  seed: number,
  tick: number,
  config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG,
  selectionMode: ThoughtStepOptions['selectionMode'] = config.selectionMode,
): string[] {
  const viable = proposals.filter((proposal) => proposal.probability >= config.minimumBranchProbability)
  if (!viable.length) return []
  const count = entropyBranchCount(viable.map((proposal) => proposal.probability), config)
  const strongest = [...viable].sort((left, right) => right.probability - left.probability || left.stableKey.localeCompare(right.stableKey))
  if (selectionMode !== 'seeded-stochastic' || count <= 1) {
    return strongest.slice(0, count).map((proposal) => proposal.stableKey)
  }
  const anchors = strongest.slice(0, count - 1)
  const remainder = strongest.slice(count - 1)
  const residualMass = remainder.reduce((sum, proposal) => sum + proposal.probability, 0)
  const exploration = residualMass >= config.minimumBranchProbability
    ? deterministicWeightedSample(
      remainder, 1, seed,
      (proposal) => proposal.stableKey,
      (proposal) => proposal.probability,
      [tick, 'residual-branch'],
    )
    : []
  return [...anchors, ...exploration].map((proposal) => proposal.stableKey)
}

export function stepThoughtEpisode(
  episode: ThoughtEpisodeState,
  model: ThoughtModel,
  head: ThoughtHead,
  index: ThoughtRuntimeIndex,
  config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG,
  options: ThoughtStepOptions = {},
  hints: ThoughtRoutingHints = {},
): { episode: ThoughtEpisodeState; tick: ThoughtTick; routes: ThoughtRouteResult[] } {
  if (episode.modelFingerprint !== model.fingerprint || episode.headFingerprint !== head.fingerprint || episode.headRevision !== head.revision ||
      index.modelFingerprint !== model.fingerprint || index.headFingerprint !== head.fingerprint) {
    throw new TypeError('Episode, bounded index, model ve başlık tabanı aynı değil; replay sürdürülemez.')
  }
  assertThoughtAdaptiveRegionEpisodeBinding(episode, options.adaptiveRegionContext, model.layout)
  if (episode.status !== 'active') throw new TypeError('Yalnız aktif episode ilerletilebilir.')
  const runtimeIndex = withTemporaryThoughtNeurons(index, episode.temporaryBirths)
  for (const newborn of episode.temporaryBirths) assertNeuronIdentityIntegrity(newborn, model.layout)
  const beforeFingerprint = episode.fingerprint
  const nextTick = episode.tick + 1
  const proposals: BranchProposal[] = []
  const routes: ThoughtRouteResult[] = []
  for (const original of episode.branches) {
    if (original.status !== 'active') continue
    const current = runtimeIndex.neuronById.get(original.currentNeuronId)
    if (!current) throw new RangeError(`Episode nöronu bounded indekste yok: ${original.currentNeuronId}`)
    assertNeuronIdentityIntegrity(current, model.layout)
    const beforeSignal = new Float32Array(original.signal)
    let macroRanking = options.macroRuntime
      ? rankThoughtMacroCandidates(options.macroRuntime, current.id, beforeSignal)
      : null
    const macroApplications: Array<{
      candidate: ThoughtRankedMacroCandidateV1
      application: ThoughtMacroApplicationV1 & { applied: true }
    }> = []
    const applicationFallbacks: ThoughtMacroFallbackV1[] = []
    for (const candidate of macroRanking?.candidates ?? []) {
      const application = applyThoughtMacroEffect(options.macroRuntime!, candidate, beforeSignal)
      if (application.applied) {
        macroApplications.push({
          candidate,
          application: application as ThoughtMacroApplicationV1 & { applied: true },
        })
      } else {
        applicationFallbacks.push({
          macroId: candidate.macroId,
          reason: application.fallbackReason ?? 'compiled-effect-invalid',
          detail: 'Compiled etki uygulanamadı; ham yönlendirme tam olasılık kütlesiyle korundu.',
        })
      }
    }
    if (macroRanking && applicationFallbacks.length) {
      if (!macroApplications.length) {
        macroRanking = {
          ...macroRanking,
          candidates: [],
          rawProbabilityScale: 1,
          macroProbabilityMass: 0,
          fallbacks: [...macroRanking.fallbacks, ...applicationFallbacks],
        }
      } else {
        const remainingWeight = macroApplications.reduce(
          (sum, entry) => sum + entry.candidate.activationWeight,
          0,
        )
        const adjusted = macroApplications.map((entry) => ({
          ...entry,
          candidate: {
            ...entry.candidate,
            selectionProbability: macroRanking!.macroProbabilityMass *
              entry.candidate.activationWeight / remainingWeight,
          },
        }))
        macroApplications.length = 0
        macroApplications.push(...adjusted)
        macroRanking = {
          ...macroRanking,
          candidates: adjusted.map((entry) => entry.candidate),
          fallbacks: [...macroRanking.fallbacks, ...applicationFallbacks],
        }
      }
    }
    const traversed = cloneThoughtBranch(original)
    decayFastState(traversed, config)
    const currentZ = representation(runtimeIndex, model, current, traversed)
    const adaptiveCompatibility = options.adaptiveRegionContext
      ? evaluateThoughtAdaptiveCompatibility(
        beforeSignal,
        currentZ,
        model.layout,
        options.adaptiveRegionContext,
      )
      : null
    const activeAdaptive = options.adaptiveRegionContext?.binding.mode === 'active'
    const transformed = transformThoughtSignal(
      beforeSignal,
      currentZ,
      config,
      activeAdaptive ? adaptiveCompatibility!.effectiveCompatibility : undefined,
    )
    traversed.signal = transformed.signal
    const relationWriteWeights = activeAdaptive
      ? adaptiveRegionRelationWriteWeights(options.adaptiveRegionContext!, adaptiveCompatibility!)
      : undefined
    const relationDelta = leaveCurrentRelationTrace(
      traversed,
      current,
      beforeSignal,
      currentZ,
      model,
      config,
      relationWriteWeights,
      Boolean(options.adaptiveRegionContext),
    )
    const branchOptions: ThoughtStepOptions = {
      ...options,
      teacherEdgeNeuronIds: [...(options.teacherEdgeNeuronIdsByOrigin?.[current.id] ?? options.teacherEdgeNeuronIds ?? [])],
    }
    const route = routeThoughtBranch(model, runtimeIndex, traversed, config, branchOptions, hints)
    routes.push(route)
    const adaptiveTelemetry = options.adaptiveRegionContext && adaptiveCompatibility
      ? createThoughtAdaptiveBranchTelemetry({
        episodeId: episode.runKey,
        tick: nextTick,
        branchId: original.id,
        signal: beforeSignal,
        representation: currentZ,
        relationDelta: relationDelta!,
        context: options.adaptiveRegionContext,
        compatibility: adaptiveCompatibility,
        candidateCompatibilities: route.adaptiveRegionCandidates,
      })
      : undefined
    const macroAdaptiveTelemetry = adaptiveTelemetry && macroApplications.length
      ? {
        ...createThoughtAdaptiveBranchTelemetry({
          episodeId: episode.runKey,
          tick: nextTick,
          branchId: original.id,
          signal: beforeSignal,
          representation: currentZ,
          relationDelta: new Float32Array(model.layout.relationLength),
          context: options.adaptiveRegionContext!,
          compatibility: adaptiveCompatibility!,
          candidateCompatibilities: route.adaptiveRegionCandidates,
        }),
        activeApplied: false,
      }
      : adaptiveTelemetry
    if (macroRanking) {
      for (const entry of macroApplications) {
        if (entry.candidate.selectionProbability < config.minimumBranchProbability) continue
        proposals.push(macroProposal(
          original, beforeSignal, route, entry.candidate, entry.application, macroRanking,
          model, runtimeIndex, config, nextTick, episode.runKey, hints, macroAdaptiveTelemetry,
        ))
      }
    }
    const rawProbabilityScale = macroRanking?.rawProbabilityScale ?? 1
    for (const candidate of route.neuronCandidates) {
      if (candidate.selectionProbability * rawProbabilityScale >= config.minimumBranchProbability) {
        proposals.push(candidateProposal(
          traversed, original, beforeSignal, transformed.changeMagnitude, route, candidate,
          model, runtimeIndex, config, nextTick, episode.runKey, hints,
          rawProbabilityScale, macroRanking, adaptiveTelemetry,
        ))
      }
    }
    if (route.unmatchedProbability * rawProbabilityScale >= config.minimumBranchProbability) {
      proposals.push(unmatchedProposal(
        traversed, original, beforeSignal, transformed.changeMagnitude, route, current,
        model, runtimeIndex, config, nextTick, episode.runKey, hints,
        rawProbabilityScale, macroRanking, adaptiveTelemetry,
      ))
    }
  }
  const retained = retainProposals(proposals, episode, config, options.selectionMode ?? config.selectionMode)
  const branches = retained.map((proposal) => proposal.branch)
  const traces = retained.map((proposal) => proposal.trace)
  const temporaryBirths = [...episode.temporaryBirths]
  const birth = createEpisodeNewborn(episode.runKey, nextTick, branches, temporaryBirths, model, runtimeIndex, config)
  if (birth) {
    temporaryBirths.push(birth.neuron)
    const contributorIds = new Set(birth.contributorBranchIds)
    for (const branch of branches) {
      if (!contributorIds.has(branch.id)) continue
      const event: ThoughtBirthEvent = {
        branchId: branch.id,
        kind: 'pressure-threshold',
        pressure: branch.pressure.smoothedBirth,
        consecutiveTicks: branch.pressure.consecutiveBirthTicks,
        neuronId: birth.neuron.id,
      }
      const trace = traces.find((entry) => entry.branchId === branch.id)
      const pendingEvent = trace?.birthEvents.find((entry) => (
        entry.kind === 'pressure-threshold' && entry.neuronId === undefined
      ))
      if (pendingEvent) pendingEvent.neuronId = birth.neuron.id
      else trace?.birthEvents.push(event)
      // A birth consumes only the residual evidence that actually formed it.
      // Unrelated parallel branches keep accumulating their own observations.
      branch.residual = emptyResidual()
    }
  }
  const stopped = nextTick >= episode.maximumTicks || !branches.some((branch) => branch.status === 'active')
  if (nextTick >= episode.maximumTicks) for (const branch of branches) if (branch.status === 'active') branch.status = 'max-ticks'
  const traceFingerprint = thoughtDigest('tick-trace', nextTick, traces)
  const routingInputFingerprint = thoughtDigest('routing-input',
    [...(options.candidateNeuronIds ?? [])], [...(options.teacherEdgeNeuronIds ?? [])],
    options.teacherEdgeNeuronIdsByOrigin ?? {}, options.selectionMode ?? config.selectionMode,
    options.macroRuntime?.fingerprint ?? null, hints,
    ...(episode.adaptiveRegion
      ? [episode.adaptiveRegion.mode, episode.adaptiveRegion.atlasFingerprint, episode.adaptiveRegion.fingerprint]
      : []))
  const provisional: Omit<ThoughtEpisodeState, 'fingerprint'> = {
    ...episode,
    tick: nextTick,
    branches,
    status: stopped ? 'stopped' : 'active',
    replay: [...episode.replay],
    temporaryBirths,
  }
  const compactBranchState = branches.map((branch) => ({
    id: branch.id,
    currentNeuronId: branch.currentNeuronId,
    signalDigest: thoughtVectorDigest(branch.signal),
    mass: branch.mass,
    cumulativeLogProbability: branch.cumulativeLogProbability,
    recentPath: branch.path.slice(-config.repetitionWindow),
    latestTransition: branch.transitions.at(-1) ?? null,
    fastTraces: branch.fastTraces,
    fastRelationPatches: branch.fastRelationPatches.map((patch) => ({
      neuronId: patch.neuronId,
      eligibility: patch.eligibility,
      visits: patch.visits,
      digest: thoughtDigest('fast-r', patch.delta),
    })),
    pressure: branch.pressure,
    residualDigest: thoughtDigest('residual-state', branch.residual.sum, branch.residual.cohesionUnitSum,
      branch.residual.weight, branch.residual.observationCount, branch.residual.cohesionTotal,
      branch.residual.cohesionComparisons, branch.residual.originEvidence, branch.residual.familyEvidence),
    status: branch.status,
  }))
  const afterCore = thoughtDigest(
    'episode-core-step', beforeFingerprint, provisional.runKey, provisional.tick,
    compactBranchState, head.revision, head.fingerprint, traceFingerprint,
    temporaryBirths.map((neuron) => [neuron.id, neuron.identityDigest, thoughtDigest('temporary-r', neuron.relation)]),
    ...(episode.adaptiveRegion
      ? [episode.adaptiveRegion.mode, episode.adaptiveRegion.atlasFingerprint, episode.adaptiveRegion.fingerprint]
      : []),
  )
  const frameBase = {
    tick: nextTick,
    beforeFingerprint,
    afterFingerprint: afterCore,
    retainedBranchIds: branches.map((branch) => branch.id),
    retainedMasses: branches.map((branch) => branch.mass),
    baseHeadRevision: head.revision,
    baseHeadFingerprint: head.fingerprint,
    traceFingerprint,
    routingInputFingerprint,
    ...(episode.adaptiveRegion ? {
      adaptiveRegionMode: episode.adaptiveRegion.mode,
      adaptiveRegionAtlasFingerprint: episode.adaptiveRegion.atlasFingerprint,
      adaptiveRegionBindingFingerprint: episode.adaptiveRegion.fingerprint,
    } : {}),
  }
  provisional.replay.push({
    ...frameBase,
    chainFingerprint: thoughtDigest('replay-chain', replayFingerprint(episode.replay), {
      ...frameBase,
      retainedMasses: frameBase.retainedMasses.map(Math.fround),
    }),
  })
  const nextEpisode = {
    ...provisional,
    fingerprint: thoughtDigest('episode-incremental', beforeFingerprint, afterCore, replayFingerprint(provisional.replay)),
  }
  const postIndex = withTemporaryThoughtNeurons(index, temporaryBirths)
  for (const branch of branches) {
    const current = postIndex.neuronById.get(branch.currentNeuronId)
    if (current) assertNeuronIdentityIntegrity(current, model.layout)
  }
  const dominant = branches[0] ?? null
  const birthEvents = traces.flatMap((trace) => trace.birthEvents)
  const tick: ThoughtTick = {
    tick: nextTick,
    branchCount: branches.length,
    activeBranchCount: branches.filter((branch) => branch.status === 'active').length,
    dominantBranchId: dominant?.id ?? null,
    dominantNeuronId: dominant?.currentNeuronId ?? null,
    dominantMass: dominant?.mass ?? 0,
    maximumBirthPressure: branches.reduce((maximum, branch) => Math.max(maximum, branch.pressure.smoothedBirth), 0),
    replayFingerprint: replayFingerprint(nextEpisode.replay),
    branches: traces,
    birthEvents,
    ...(episode.adaptiveRegion ? {
      adaptiveRegion: {
        format: 'thought-adaptive-region-tick-telemetry' as const,
        version: 1 as const,
        mode: episode.adaptiveRegion.mode,
        atlasFingerprint: episode.adaptiveRegion.atlasFingerprint,
        bindingFingerprint: episode.adaptiveRegion.fingerprint,
        observedBranchCount: traces.filter((trace) => Boolean(trace.adaptiveRegion)).length,
        activeApplied: traces.some((trace) => trace.adaptiveRegion?.activeApplied),
      },
    } : {}),
  }
  return { episode: nextEpisode, tick, routes }
}

export function withEpisodeStatus(episode: ThoughtEpisodeState, status: 'paused' | 'cancelled'): ThoughtEpisodeState {
  const cloned = cloneThoughtEpisodeState(episode)
  return { ...cloned, status }
}

export function cloneThoughtEpisodeState(episode: ThoughtEpisodeState): ThoughtEpisodeState {
  return {
    ...episode,
    ...(episode.adaptiveRegion
      ? { adaptiveRegion: cloneThoughtAdaptiveRegionBinding(episode.adaptiveRegion) }
      : {}),
    branches: episode.branches.map(cloneThoughtBranch),
    replay: episode.replay.map((frame) => ({ ...frame, retainedBranchIds: [...frame.retainedBranchIds], retainedMasses: [...frame.retainedMasses] })),
    temporaryBirths: episode.temporaryBirths.map((neuron) => ({
      ...neuron,
      identity: new Float32Array(neuron.identity),
      relation: new Float32Array(neuron.relation),
      familyMemberships: neuron.familyMemberships.map((membership) => ({ ...membership })),
      provenance: {
        ...neuron.provenance,
        ...(neuron.provenance.birthBranchIds ? { birthBranchIds: [...neuron.provenance.birthBranchIds] } : {}),
        ...(neuron.provenance.originEvidence
          ? { originEvidence: neuron.provenance.originEvidence.map((entry) => ({ ...entry })) }
          : {}),
        ...(neuron.provenance.familyEvidence
          ? { familyEvidence: neuron.provenance.familyEvidence.map((entry) => ({ ...entry })) }
          : {}),
      },
    })),
  }
}

export function resumeThoughtEpisode(episode: ThoughtEpisodeState): ThoughtEpisodeState {
  if (episode.status !== 'paused') throw new TypeError('Yalnız güvenli sınırda paused episode sürdürülebilir.')
  const cloned = cloneThoughtEpisodeState(episode)
  return { ...cloned, status: 'active' }
}
