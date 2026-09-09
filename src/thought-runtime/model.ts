import { DEFAULT_THOUGHT_RUNTIME_CONFIG } from './config'
import { thoughtDigest, thoughtPartitionDigest } from './deterministic'
import { assertThoughtPartition, clamp, projectThoughtVector } from './math'
import {
  THOUGHT_MATRIX_CELLS,
  THOUGHT_MATRIX_SIDE,
  type BornNeuronV1,
  type SparseHabitEdge,
  type SparseHabitRow,
  type ThoughtFamily,
  type ThoughtFamilyMembership,
  type ThoughtHead,
  type ThoughtMatrixLayout,
  type ThoughtModel,
  type ThoughtNeuron,
  type ThoughtRelationPatch,
  type ThoughtRuntimeConfig,
} from './types'

export function createThoughtLayout(identityLength = THOUGHT_MATRIX_SIDE * 16): ThoughtMatrixLayout {
  if (!Number.isInteger(identityLength) || identityLength < 1 || identityLength >= THOUGHT_MATRIX_CELLS) {
    throw new RangeError('I bölümü 4096 hücrenin içinde boş olmayan bir önek olmalıdır.')
  }
  return {
    side: THOUGHT_MATRIX_SIDE,
    cellCount: THOUGHT_MATRIX_CELLS,
    identityLength,
    relationStart: identityLength,
    relationLength: THOUGHT_MATRIX_CELLS - identityLength,
  }
}

function normalizedMemberships(
  memberships: readonly ThoughtFamilyMembership[] = [],
  maximum = DEFAULT_THOUGHT_RUNTIME_CONFIG.maximumFamilyMembershipsPerNeuron,
): ThoughtFamilyMembership[] {
  const byFamily = new Map<string, number>()
  for (const membership of memberships) {
    const familyId = String(membership.familyId ?? '').trim()
    if (!familyId || !Number.isFinite(membership.weight) || membership.weight <= 0) continue
    byFamily.set(familyId, Math.max(byFamily.get(familyId) ?? 0, membership.weight))
  }
  const retained = [...byFamily].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])).slice(0, maximum)
  if (!retained.length) return [{ familyId: '@thought/unassigned', weight: 1 }]
  const total = retained.reduce((sum, [, weight]) => sum + weight, 0)
  return retained.map(([familyId, weight]) => ({ familyId, weight: weight / total }))
}

export function createThoughtNeuron(
  input: Omit<ThoughtNeuron, 'identityDigest' | 'familyMemberships'> & {
    identityDigest?: string
    familyMemberships?: readonly ThoughtFamilyMembership[]
  },
  layout: ThoughtMatrixLayout = createThoughtLayout(),
  config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG,
): ThoughtNeuron {
  const id = String(input.id ?? '').trim()
  if (!id) throw new RangeError('Nöron kimliği boş olamaz.')
  assertThoughtPartition(input.identity, layout.identityLength, `${id} I`)
  assertThoughtPartition(input.relation, layout.relationLength, `${id} R`)
  const identity = new Float32Array(input.identity)
  const identityDigest = thoughtPartitionDigest('identity', identity)
  if (input.identityDigest && input.identityDigest !== identityDigest) {
    throw new TypeError(`${id} sabit I özeti verilen değerle uyuşmuyor.`)
  }
  return {
    id,
    identity,
    relation: new Float32Array(input.relation),
    identityDigest,
    ...(input.sourceIdentityDigest ? { sourceIdentityDigest: input.sourceIdentityDigest } : {}),
    familyMemberships: normalizedMemberships(input.familyMemberships, config.maximumFamilyMembershipsPerNeuron),
    provenance: {
      ...input.provenance,
      ...(input.provenance.birthBranchIds ? { birthBranchIds: [...input.provenance.birthBranchIds] } : {}),
      ...(input.provenance.originEvidence
        ? { originEvidence: input.provenance.originEvidence.map((entry) => ({ ...entry })) }
        : {}),
      ...(input.provenance.familyEvidence
        ? { familyEvidence: input.provenance.familyEvidence.map((entry) => ({ ...entry })) }
        : {}),
    },
  }
}

function cloneEdge(edge: SparseHabitEdge): SparseHabitEdge {
  return {
    toNeuronId: edge.toNeuronId,
    priorEvidence: clamp(edge.priorEvidence, 0, DEFAULT_THOUGHT_RUNTIME_CONFIG.evidenceCeiling),
    excitationEvidence: clamp(edge.excitationEvidence, 0, DEFAULT_THOUGHT_RUNTIME_CONFIG.evidenceCeiling),
    inhibitionEvidence: clamp(edge.inhibitionEvidence, 0, DEFAULT_THOUGHT_RUNTIME_CONFIG.evidenceCeiling),
    trustMass: Math.max(0, Number.isFinite(edge.trustMass) ? edge.trustMass : 0),
    updateCount: Math.max(0, Math.trunc(edge.updateCount)),
    independentSourceIds: [...new Set(edge.independentSourceIds)].sort(),
  }
}

export function createThoughtModel(input: {
  revision?: number
  layout?: ThoughtMatrixLayout
  neurons: readonly ThoughtNeuron[]
  families?: readonly ThoughtFamily[]
  habitRows?: readonly SparseHabitRow[]
}, config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG): ThoughtModel {
  const layout = input.layout ?? createThoughtLayout()
  if (layout.side !== THOUGHT_MATRIX_SIDE || layout.cellCount !== THOUGHT_MATRIX_CELLS ||
      layout.relationStart !== layout.identityLength ||
      layout.identityLength + layout.relationLength !== THOUGHT_MATRIX_CELLS) {
    throw new RangeError('Model I/R yerleşimi 64x64 matrisi eksiksiz bölmelidir.')
  }
  const neuronIds = new Set<string>()
  const neurons = input.neurons.map((neuron) => {
    if (neuronIds.has(neuron.id)) throw new TypeError(`Yinelenen nöron kimliği: ${neuron.id}`)
    neuronIds.add(neuron.id)
    return createThoughtNeuron(neuron, layout, config)
  }).sort((left, right) => left.id.localeCompare(right.id))
  if (!neurons.length) throw new RangeError('Düşünce modeli en az bir vektörlü nöron gerektirir.')

  const familyMap = new Map<string, ThoughtFamily>()
  for (const family of input.families ?? []) {
    const id = String(family.id ?? '').trim()
    if (id) familyMap.set(id, { ...family, id })
  }
  for (const neuron of neurons) {
    for (const membership of neuron.familyMemberships) {
      if (!familyMap.has(membership.familyId)) familyMap.set(membership.familyId, { id: membership.familyId })
    }
  }
  const families = [...familyMap.values()].sort((left, right) => left.id.localeCompare(right.id))

  let totalEdges = 0
  const habitRows = (input.habitRows ?? [])
    .filter((row) => neuronIds.has(row.fromNeuronId))
    .map((row): SparseHabitRow => {
      const byTarget = new Map<string, SparseHabitEdge>()
      for (const edge of row.edges) {
        if (!neuronIds.has(edge.toNeuronId) || edge.toNeuronId === row.fromNeuronId) continue
        const cloned = cloneEdge(edge)
        const previous = byTarget.get(cloned.toNeuronId)
        if (!previous || edgeStrength(cloned) > edgeStrength(previous)) byTarget.set(cloned.toNeuronId, cloned)
      }
      const edges = [...byTarget.values()]
        .sort((left, right) => edgeStrength(right) - edgeStrength(left) || left.toNeuronId.localeCompare(right.toNeuronId))
        .slice(0, config.maximumHabitEdgesPerNeuron)
      totalEdges += edges.length
      return { fromNeuronId: row.fromNeuronId, edges }
    })
    .sort((left, right) => left.fromNeuronId.localeCompare(right.fromNeuronId))
  if (totalEdges > config.maximumTotalHabitEdges) {
    throw new RangeError(`Seyrek H ${config.maximumTotalHabitEdges} küresel kenar sınırını aşıyor.`)
  }
  const revision = Math.max(0, Math.trunc(input.revision ?? 0))
  const fingerprint = thoughtDigest('model', revision, layout, neurons.map((neuron) => ({
    id: neuron.id,
    identityDigest: neuron.identityDigest,
    relationDigest: thoughtPartitionDigest('relation', neuron.relation),
    memberships: neuron.familyMemberships,
  })), families, habitRows)
  return { format: 'bombom-thought-model', version: 1, revision, layout: { ...layout }, neurons, families, habitRows, fingerprint }
}

export function assertThoughtIdentityIntegrity(model: ThoughtModel): void {
  for (const neuron of model.neurons) {
    assertNeuronIdentityIntegrity(neuron, model.layout)
  }
}

export function assertNeuronIdentityIntegrity(neuron: ThoughtNeuron, layout: ThoughtMatrixLayout): void {
  assertThoughtPartition(neuron.identity, layout.identityLength, `${neuron.id} I`)
  if (thoughtPartitionDigest('identity', neuron.identity) !== neuron.identityDigest) {
    throw new TypeError(`${neuron.id} sabit I alanı çalışma sırasında değişmiş.`)
  }
}

export function edgeStrength(edge: Pick<SparseHabitEdge, 'priorEvidence' | 'excitationEvidence' | 'inhibitionEvidence'>): number {
  return Math.max(0, edge.priorEvidence + edge.excitationEvidence) / (1 + Math.max(0, edge.inhibitionEvidence))
}

function headFingerprintValue(head: Omit<ThoughtHead, 'fingerprint'>): string {
  return thoughtDigest('head', head.version, head.revision, head.baseModelFingerprint, head.totalTicks,
    head.neuronEdges, head.contextPatches.map((patch) => ({ ...patch, delta: thoughtPartitionDigest('relation', patch.delta) })),
    head.nursery.map((candidate) => ({ ...candidate,
      identity: thoughtPartitionDigest('identity', candidate.identity),
      relationPrototype: thoughtPartitionDigest('relation', candidate.relationPrototype),
    })), head.births)
}

export function createThoughtHead(
  model: ThoughtModel,
  input?: ThoughtHead | null,
  expectedBaseFingerprint = model.fingerprint,
): ThoughtHead {
  if (input && input.baseModelFingerprint !== expectedBaseFingerprint) {
    throw new TypeError('Düşünce başlığı farklı bir snapshot temel başlığına ait.')
  }
  const relationLength = model.layout.relationLength
  const base = input ?? {
    format: 'bombom-thought-head' as const,
    version: 1 as const,
    revision: 0,
    baseModelFingerprint: expectedBaseFingerprint,
    totalTicks: 0,
    neuronEdges: [],
    contextPatches: [],
    nursery: [],
    births: [],
    fingerprint: '',
  }
  const withoutFingerprint: Omit<ThoughtHead, 'fingerprint'> = {
    format: 'bombom-thought-head',
    version: 1,
    revision: Math.max(0, Math.trunc(base.revision)),
    baseModelFingerprint: expectedBaseFingerprint,
    totalTicks: Math.max(0, Math.trunc(base.totalTicks)),
    neuronEdges: base.neuronEdges.map((edge) => ({
      ...edge,
      excitation: clamp(edge.excitation, 0, DEFAULT_THOUGHT_RUNTIME_CONFIG.evidenceCeiling),
      inhibition: clamp(edge.inhibition, 0, DEFAULT_THOUGHT_RUNTIME_CONFIG.evidenceCeiling),
      trustMass: Math.max(0, edge.trustMass),
      updateCount: Math.max(0, Math.trunc(edge.updateCount)),
    })).sort((left, right) => left.fromMemberKey.localeCompare(right.fromMemberKey) || left.toMemberKey.localeCompare(right.toMemberKey)),
    contextPatches: base.contextPatches.map((patch): ThoughtRelationPatch => {
      assertThoughtPartition(patch.delta, relationLength, `${patch.neuronId} kalıcı R yaması`)
      return { ...patch, delta: new Float32Array(patch.delta) }
    }).sort((left, right) => left.neuronId.localeCompare(right.neuronId)),
    nursery: base.nursery.map((candidate) => {
      assertThoughtPartition(candidate.identity, model.layout.identityLength, `${candidate.id} nursery I`)
      assertThoughtPartition(candidate.relationPrototype, model.layout.relationLength, `${candidate.id} nursery R`)
      if (thoughtPartitionDigest('identity', candidate.identity) !== candidate.identityDigest) {
        throw new TypeError(`${candidate.id} nursery sabit I digest'i uyuşmuyor.`)
      }
      return {
        ...candidate,
        identity: new Float32Array(candidate.identity),
        relationPrototype: new Float32Array(candidate.relationPrototype),
        birthEpisodeIds: [...candidate.birthEpisodeIds],
        independentSourceIds: [...candidate.independentSourceIds],
        originEvidence: candidate.originEvidence.map((entry) => ({ ...entry })),
        ...(candidate.outgoingEvidence ? {
          outgoingEvidence: candidate.outgoingEvidence.map((entry) => ({ ...entry })),
        } : {}),
        familyEvidence: candidate.familyEvidence.map((entry) => ({ ...entry })),
        residualCohesion: candidate.residualCohesion ?? 0,
        sourceRelationPrototypes: (candidate.sourceRelationPrototypes ?? []).map((prototype) => ({
          ...prototype,
          relation: new Float32Array(prototype.relation),
        })),
      }
    }),
    births: base.births.map((birth) => ({
      ...birth,
      identity: [...birth.identity],
      relation: [...birth.relation],
      familyMemberships: birth.familyMemberships.map((item) => ({ ...item })),
      sourceGroupIds: [...birth.sourceGroupIds],
      ...(birth.entryEvidence ? { entryEvidence: birth.entryEvidence.map((entry) => ({ ...entry })) } : {}),
      ...(birth.outgoingEvidence ? { outgoingEvidence: birth.outgoingEvidence.map((entry) => ({ ...entry })) } : {}),
      ...(birth.contextEvidence ? {
        contextEvidence: {
          ...birth.contextEvidence,
          sourceGroupIds: [...birth.contextEvidence.sourceGroupIds],
        },
      } : {}),
    })),
  }
  return { ...withoutFingerprint, fingerprint: headFingerprintValue(withoutFingerprint) }
}

export function neuronFromBirth(birth: BornNeuronV1, layout: ThoughtMatrixLayout): ThoughtNeuron {
  return createThoughtNeuron({
    id: birth.id,
    identity: Float32Array.from(birth.identity),
    relation: Float32Array.from(birth.relation),
    identityDigest: birth.identityDigest,
    familyMemberships: birth.familyMemberships,
    provenance: { kind: 'nursery-promotion', nurseryId: birth.nurseryId },
  }, layout)
}

export function allThoughtNeurons(model: ThoughtModel, head: ThoughtHead): ThoughtNeuron[] {
  const byId = new Map(model.neurons.map((neuron) => [neuron.id, neuron]))
  for (const birth of head.births) if (!byId.has(birth.id)) byId.set(birth.id, neuronFromBirth(birth, model.layout))
  return [...byId.values()].sort((left, right) => left.id.localeCompare(right.id))
}

export function persistentRelationPatch(head: ThoughtHead, neuronId: string, relationLength: number): Float32Array {
  const patch = head.contextPatches.find((entry) => entry.neuronId === neuronId)
  return patch ? projectThoughtVector(patch.delta, 1) : new Float32Array(relationLength)
}
