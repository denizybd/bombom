import { createHash } from 'node:crypto'
import { getHeapStatistics } from 'node:v8'
import {
  bootstrapSourceGroupInterval,
  createStructuralBenchmark,
  scoreEventGraph,
} from '../src/thought-runtime/benchmark.ts'
import { createThoughtCatalogFromResolved } from '../src/thought-runtime/catalog.ts'
import { cosineSimilarity, projectThoughtVector } from '../src/thought-runtime/math.ts'
import { createThoughtHead } from '../src/thought-runtime/model.ts'
import { encodeThoughtPrompt } from '../src/thought-runtime/prompt.ts'
import { resolveThoughtEngineBase } from '../server/thought-service.mjs'
import { evaluateResourceState, MEBIBYTE, sampleHostResources } from '../server/resource-policy.mjs'
import {
  evaluateIndependentStructuralBenchmark,
  evaluateIndependentStructuralBenchmarkEarlyFailure,
  evaluateIndependentStructuralBenchmarkScreen,
  INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND,
  INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD,
  INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_ARTIFACT_KIND,
  INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_FORMAT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD,
  INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
  INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_ARTIFACT_KIND,
  INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_FORMAT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_POLICY_FINGERPRINT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION,
} from '../server/structural-stop-gate.mjs'

export const STRUCTURAL_BENCHMARK_RUNNER_VERSION = 1
export const STRUCTURAL_BENCHMARK_TICKS = 64
export const STRUCTURAL_BENCHMARK_SEEDS = Object.freeze([
  0x51a7,
  0x9e37,
  0xc0de,
  0xd1ce,
  0xf17e,
])
const BOOTSTRAP_SAMPLES = 2_000
const EDGE_SEPARATOR = '\u0000'
const SIGNAL_PERTURBATION_COORDINATES = 8
const SIGNAL_PERTURBATION_AMPLITUDE = 0.001
const MINIMUM_SMALL_SIGNAL_COSINE = 0.999
const CORRECT_GRAPH_F1 = 0.75
const OBSERVATION_CHUNK_KIND = 'independent-structural-benchmark-observations-v1'
const OBSERVATION_CHUNK_FORMAT = 'bombom-independent-structural-observations'
const EARLY_FAILURE_SKIPPED_VARIANTS = Object.freeze([
  'h0-only',
  'no-recurrence',
  'shuffled-edges',
  'target-family-ablation',
  'target-edge-ablation',
  'sham-family-ablation',
  'sham-edge-ablation',
  'perturbed-signal',
  'persistent-birth-classification',
])
const MAXIMUM_DENSE_BENCHMARK_NEURONS = 16_384
const MAXIMUM_DENSE_BENCHMARK_PEAK_BYTES = 768 * MEBIBYTE
const DENSE_BENCHMARK_PEAK_MULTIPLIER = 3
const BENCHMARK_BASE_ARTIFACT_KINDS = new Set([
  'family-run', 'runtime-model-state', 'text-lab-analysis', 'synthesis-run', 'thought-runtime-head',
])

const NOVELTY_CASES = Object.freeze([
  Object.freeze({
    id: 'novel-opal-sandik',
    prompt: 'Zerin opal sandığı prizma anahtarıyla döndürdü.',
    expected: Object.freeze({ format: 'event-graph-v1', actor: 'Zerin', action: 'döndürdü', patient: 'opal sandık', instrument: 'prizma anahtarı', operators: Object.freeze([]), negated: false, goals: Object.freeze([]) }),
  }),
  Object.freeze({
    id: 'novel-kehribar-kutu',
    prompt: 'Lalin kehribar kutuyu sarmal kıskaçla mühürledi.',
    expected: Object.freeze({ format: 'event-graph-v1', actor: 'Lalin', action: 'mühürledi', patient: 'kehribar kutu', instrument: 'sarmal kıskaç', operators: Object.freeze([]), negated: false, goals: Object.freeze([]) }),
  }),
  Object.freeze({
    id: 'novel-kobalt-kure',
    prompt: 'Tuna kobalt küreyi optik maşayla hizaladı.',
    expected: Object.freeze({ format: 'event-graph-v1', actor: 'Tuna', action: 'hizaladı', patient: 'kobalt küre', instrument: 'optik maşa', operators: Object.freeze([]), negated: false, goals: Object.freeze([]) }),
  }),
  Object.freeze({
    id: 'novel-mercan-kapsul',
    prompt: 'Ekin mercan kapsülü sessiz mengeneyle sabitledi.',
    expected: Object.freeze({ format: 'event-graph-v1', actor: 'Ekin', action: 'sabitledi', patient: 'mercan kapsül', instrument: 'sessiz mengene', operators: Object.freeze([]), negated: false, goals: Object.freeze([]) }),
  }),
  Object.freeze({
    id: 'novel-indiga-prizma',
    prompt: 'Nisan indigo prizmayı seramik kancayla yönlendirdi.',
    expected: Object.freeze({ format: 'event-graph-v1', actor: 'Nisan', action: 'yönlendirdi', patient: 'indigo prizma', instrument: 'seramik kanca', operators: Object.freeze([]), negated: false, goals: Object.freeze([]) }),
  }),
])

const RUNNER_SPEC = Object.freeze({
  version: STRUCTURAL_BENCHMARK_RUNNER_VERSION,
  benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
  seeds: STRUCTURAL_BENCHMARK_SEEDS,
  ticks: STRUCTURAL_BENCHMARK_TICKS,
  bootstrapSamples: BOOTSTRAP_SAMPLES,
  noRecurrence: 'full-tick-zero-persistent-transition-evidence-v2',
  h0Only: 'family-h0-without-head-or-language-v1',
  shuffledEdges: 'deterministic-all-edge-target-remap-v1',
  ablationIntervals: 'paired-signed-source-group-ci-v2',
  targetSelection: 'snapshot-model-only-no-evaluation-data-v2',
  shamSelection: 'snapshot-digest-random-nontarget-edge-v1',
  familySham: 'deterministic-model-only-exact-reference-footprint-mask-v1',
  durableObservationCache: 'exact-snapshot-variant-seed-chunks-v1',
  denseCatalogSafety: Object.freeze({
    mode: 'bounded-dense-bootstrap-until-residency-benchmark-v1',
    maximumNeurons: MAXIMUM_DENSE_BENCHMARK_NEURONS,
    maximumProjectedPeakBytes: MAXIMUM_DENSE_BENCHMARK_PEAK_BYTES,
    peakMultiplier: DENSE_BENCHMARK_PEAK_MULTIPLIER,
    processHeapCeilingBytes: 4 * 1024 * MEBIBYTE,
  }),
  perturbation: Object.freeze({
    coordinates: SIGNAL_PERTURBATION_COORDINATES,
    amplitude: SIGNAL_PERTURBATION_AMPLITUDE,
    minimumCosine: MINIMUM_SMALL_SIGNAL_COSINE,
  }),
  birth: Object.freeze({
    positives: 'four-deterministic-observations-two-independent-sources-v2',
    negatives: 'frozen-train-calibration-four-deterministic-observations-v2',
    precision: 'TP/(TP+FP)',
  }),
})

function stableValue(value) {
  if (ArrayBuffer.isView(value)) return Array.from(value)
  if (Array.isArray(value)) return value.map(stableValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]))
  }
  return value
}

function stableHash(...values) {
  const hash = createHash('sha256')
  for (const value of values) hash.update(JSON.stringify(stableValue(value)))
  return hash.digest('hex')
}

export const STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT = stableHash(RUNNER_SPEC)

function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
}

function clampSigned(value) {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0))
}

function edgeKey(from, to) {
  return `${from}${EDGE_SEPARATOR}${to}`
}

function abortIfRequested(signal) {
  if (!signal?.aborted) return
  throw signal.reason ?? new DOMException('Aborted', 'AbortError')
}

function benchmarkMetadataPreflight(store, snapshot) {
  if (typeof store?.getArtifactMetadata !== 'function' || typeof store?.getConfig !== 'function') {
    return { available: false, reason: 'metadata-api-unavailable' }
  }
  const metadata = (snapshot.artifactIds ?? [])
    .map((id) => store.getArtifactMetadata(id))
    .filter((entry) => entry && BENCHMARK_BASE_ARTIFACT_KINDS.has(entry.kind))
  const aggregatePayloadBytes = metadata.reduce((sum, entry) => {
    const bytes = Number(entry.payloadBytes)
    return sum + (Number.isSafeInteger(bytes) && bytes >= 0 ? bytes : 0)
  }, 0)
  const estimatedTransientBytes = Math.ceil(aggregatePayloadBytes * 2.5)
  const config = store.getConfig()
  const sample = sampleHostResources({ dataDirectory: config.dataDirectory ?? store.dataDirectory })
  const projected = {
    ...sample,
    rssBytes: sample.rssBytes + estimatedTransientBytes,
    heapUsedBytes: sample.heapUsedBytes + estimatedTransientBytes,
    hostFreeMemoryBytes: Math.max(0, sample.hostFreeMemoryBytes - estimatedTransientBytes),
  }
  const state = config.resourcePolicy?.enabled === false
    ? { allowed: true, reasonCodes: [], memoryTier: 'disabled', diskTier: 'disabled' }
    : evaluateResourceState(config, projected, {
      logicalStorageBytes: store.approximateStorageBytes(),
      admission: true,
    })
  const heapCeilingBytes = Math.min(getHeapStatistics().heap_size_limit, 4 * 1024 * MEBIBYTE)
  const heapSafe = projected.heapUsedBytes < heapCeilingBytes
  if (!state.allowed || !heapSafe) {
    const error = new Error('Yapısal benchmark tabanı güvenli aggregate RAM rezervi sağlanana kadar decode edilmedi.')
    // Teacher-feed propagates JOB_PAUSED across the isolated worker instead of
    // misclassifying a resource wait as benchmark/model failure.
    error.code = 'JOB_PAUSED'
    error.retryable = true
    error.reasonCode = 'STRUCTURAL_BENCHMARK_RESOURCE_WAIT'
    error.details = {
      phase: 'metadata-hydration-preflight',
      artifactCount: metadata.length,
      aggregatePayloadBytes,
      estimatedTransientBytes,
      rssBytes: sample.rssBytes,
      projectedRssBytes: projected.rssBytes,
      heapCeilingBytes,
      reasonCodes: [...(state.reasonCodes ?? []), ...(heapSafe ? [] : ['PROCESS_HEAP_4G_FAILSAFE'])],
    }
    throw error
  }
  return {
    available: true,
    artifactCount: metadata.length,
    aggregatePayloadBytes,
    estimatedTransientBytes,
    projectedMemoryTier: state.memoryTier,
    heapCeilingBytes,
  }
}

function denseBenchmarkDiagnostics(base) {
  const neuronCount = base.neurons.length
  const denseVectorBytes = neuronCount * base.layout.cellCount * Float32Array.BYTES_PER_ELEMENT
  const projectedPeakBytes = denseVectorBytes * DENSE_BENCHMARK_PEAK_MULTIPLIER
  const supported = neuronCount <= MAXIMUM_DENSE_BENCHMARK_NEURONS &&
    projectedPeakBytes <= MAXIMUM_DENSE_BENCHMARK_PEAK_BYTES
  if (!supported) {
    const error = new Error('Bu snapshot benchmarkı henüz SSD aday kapısına bağlı değil; full-dense katalog güvenli sınırı aşıyor.')
    error.code = 'JOB_PAUSED'
    error.retryable = true
    error.reasonCode = 'STRUCTURAL_BENCHMARK_DENSE_BASE_UNSUPPORTED'
    error.details = {
      neuronCount,
      denseVectorBytes,
      projectedPeakBytes,
      maximumNeurons: MAXIMUM_DENSE_BENCHMARK_NEURONS,
      maximumProjectedPeakBytes: MAXIMUM_DENSE_BENCHMARK_PEAK_BYTES,
      retryableAfter: 'structural-benchmark-residency-integration',
    }
    throw error
  }
  return {
    mode: 'bounded-dense-benchmark-v1',
    neuronCount,
    denseVectorBytes,
    projectedPeakBytes,
    maximumNeurons: MAXIMUM_DENSE_BENCHMARK_NEURONS,
    maximumProjectedPeakBytes: MAXIMUM_DENSE_BENCHMARK_PEAK_BYTES,
    exactSsdPaging: false,
    limitation: 'benchmark variants retain one dense validated catalog; 100K is intentionally rejected',
    isolatedWorker4GiBFailSafe: process.env.BOMBOM_JOB_WORKER_CHILD === '1',
    processHeapCeilingBytes: Math.min(getHeapStatistics().heap_size_limit, 4 * 1024 * MEBIBYTE),
  }
}

function requestBase(base, bundle) {
  return {
    snapshotId: base.reference.snapshotId,
    snapshotStatus: base.reference.snapshotStatus,
    selectedBy: 'independent-structural-benchmark',
    activeAtStart: base.reference.activeAtStart,
    commitEligible: false,
    artifactManifestHash: base.reference.artifactManifestHash,
    baseModelFingerprint: base.reference.baseModelFingerprint,
    familyRunFingerprint: bundle.familyRun.fingerprint,
    languageStateFingerprint: bundle.languageState?.fingerprint ?? null,
    thoughtHeadFingerprint: bundle.thoughtHead.fingerprint,
    thoughtHeadRevision: bundle.thoughtHead.revision,
    mode: 'independent-deterministic-benchmark',
  }
}

function modelFor(base, familyRun) {
  return createThoughtCatalogFromResolved(base.neurons, familyRun).model
}

function canonicalHead(base, familyRun, head) {
  return createThoughtHead(modelFor(base, familyRun), head, base.reference.baseModelFingerprint)
}

function bundleFor(base, variant, {
  familyRun = base.familyRun,
  languageState = base.languageState,
  thoughtHead = base.thoughtHead,
} = {}) {
  return {
    variant,
    familyRun,
    languageState,
    thoughtHead: canonicalHead(base, familyRun, thoughtHead),
  }
}

function emptyHeadBundle(base) {
  const familyRun = structuredClone(base.familyRun)
  const model = modelFor(base, familyRun)
  return {
    variant: 'h0-only',
    familyRun,
    languageState: null,
    thoughtHead: createThoughtHead(model, null, base.reference.baseModelFingerprint),
  }
}

function recurrenceOffBundle(base) {
  const familyRun = structuredClone(base.familyRun)
  familyRun.neuronTransitions = []
  familyRun.familyTransitions = []
  const languageState = base.languageState ? structuredClone(base.languageState) : null
  if (languageState) languageState.neuronTransitions = []
  const thoughtHead = { ...structuredClone(base.thoughtHead), neuronEdges: [] }
  return bundleFor(base, 'recurrence-off', { familyRun, languageState, thoughtHead })
}

function persistentEdgeKeys(base) {
  const keys = new Set()
  for (const edge of base.familyRun.neuronTransitions ?? []) keys.add(edgeKey(edge.fromMemberKey, edge.toMemberKey))
  for (const edge of base.thoughtHead.neuronEdges ?? []) keys.add(edgeKey(edge.fromMemberKey, edge.toMemberKey))
  for (const edge of base.languageState?.neuronTransitions ?? []) keys.add(edgeKey(edge.fromKey, edge.toKey))
  return [...keys].sort()
}

function persistentEdgeEvidence(base) {
  const evidence = new Map()
  const add = (from, to, value) => {
    const key = edgeKey(from, to)
    evidence.set(key, (evidence.get(key) ?? 0) + Math.max(0, Number(value) || 0))
  }
  for (const edge of base.familyRun.neuronTransitions ?? []) add(edge.fromMemberKey, edge.toMemberKey, edge.probability)
  for (const edge of base.thoughtHead.neuronEdges ?? []) {
    add(edge.fromMemberKey, edge.toMemberKey, (edge.excitation ?? 0) + (edge.trustMass ?? 0) - (edge.inhibition ?? 0))
  }
  for (const edge of base.languageState?.neuronTransitions ?? []) {
    add(edge.fromKey, edge.toKey, (edge.excitatoryEvidence ?? 0) + (edge.score ?? 0) - (edge.inhibitoryEvidence ?? 0))
  }
  return evidence
}

function chooseTargetEdge(base) {
  const evidence = persistentEdgeEvidence(base)
  return persistentEdgeKeys(base)
    .sort((left, right) => (evidence.get(right) ?? 0) - (evidence.get(left) ?? 0) || left.localeCompare(right))[0] ?? null
}

function chooseTargetFamily(base) {
  const evidence = new Map((base.familyRun.families ?? []).map((family) => [family.id, 0]))
  for (const membership of base.familyRun.memberships ?? []) {
    evidence.set(membership.familyId, (evidence.get(membership.familyId) ?? 0) + Math.max(0, Number(membership.weight) || 0))
  }
  for (const edge of base.familyRun.familyTransitions ?? []) {
    const weight = Math.max(0, Number(edge.probability) || 0)
    evidence.set(edge.fromFamilyId, (evidence.get(edge.fromFamilyId) ?? 0) + weight)
    evidence.set(edge.toFamilyId, (evidence.get(edge.toFamilyId) ?? 0) + weight)
  }
  return [...evidence.keys()]
    .sort((left, right) => (evidence.get(right) ?? 0) - (evidence.get(left) ?? 0) || left.localeCompare(right))[0] ?? null
}

function chooseShamEdge(base, targetEdge) {
  const candidates = persistentEdgeKeys(base).filter((key) => key !== targetEdge)
  return candidates.sort((left, right) => stableHash('sham-edge', base.reference.snapshotId, left)
    .localeCompare(stableHash('sham-edge', base.reference.snapshotId, right)) || left.localeCompare(right))[0] ?? null
}

function withoutEdge(base, key, variant) {
  if (!key) return bundleFor(base, variant)
  const [from, to] = key.split(EDGE_SEPARATOR)
  const familyRun = structuredClone(base.familyRun)
  familyRun.neuronTransitions = (familyRun.neuronTransitions ?? []).filter((edge) => edge.fromMemberKey !== from || edge.toMemberKey !== to)
  const languageState = base.languageState ? structuredClone(base.languageState) : null
  if (languageState) languageState.neuronTransitions = (languageState.neuronTransitions ?? []).filter((edge) => edge.fromKey !== from || edge.toKey !== to)
  const thoughtHead = {
    ...structuredClone(base.thoughtHead),
    neuronEdges: (base.thoughtHead.neuronEdges ?? []).filter((edge) => edge.fromMemberKey !== from || edge.toMemberKey !== to),
  }
  return bundleFor(base, variant, { familyRun, languageState, thoughtHead })
}

const FAMILY_FOOTPRINT_KINDS = Object.freeze([
  'memberships',
  'familyTransitions',
  'utilityReports',
  'coalitions',
  'coalitionActivations',
  'coalitionBindings',
  'birthMemberships',
])

function completeFamilyFootprint(counts = {}) {
  const footprint = Object.fromEntries(FAMILY_FOOTPRINT_KINDS.map((kind) => [kind, counts[kind] ?? 0]))
  return {
    ...footprint,
    total: FAMILY_FOOTPRINT_KINDS.reduce((sum, kind) => sum + footprint[kind], 0),
  }
}

function familyReferenceFootprint(base, familyId) {
  if (!familyId) return completeFamilyFootprint()
  const targetCoalitions = new Set()
  let coalitionActivations = 0
  let coalitionBindings = 0
  for (const [coalitionIndex, coalition] of (base.familyRun.coalition?.coalitions ?? []).entries()) {
    if (coalition.eventFamilyId === familyId) {
      targetCoalitions.add(coalitionIndex)
      continue
    }
    coalitionActivations += (coalition.familyActivations ?? []).filter((entry) => entry.familyId === familyId).length
    coalitionBindings += (coalition.bindings ?? []).filter((entry) => entry.roleFamilyId === familyId).length
  }
  return completeFamilyFootprint({
    memberships: (base.familyRun.memberships ?? []).filter((entry) => entry.familyId === familyId).length,
    familyTransitions: (base.familyRun.familyTransitions ?? [])
      .filter((entry) => entry.fromFamilyId === familyId || entry.toFamilyId === familyId).length,
    utilityReports: (base.familyRun.utilityReports ?? []).filter((entry) => entry.familyId === familyId).length,
    coalitions: targetCoalitions.size,
    coalitionActivations,
    coalitionBindings,
    birthMemberships: (base.thoughtHead.births ?? []).reduce((sum, birth) => sum +
      (birth.familyMemberships ?? []).filter((entry) => entry.familyId === familyId).length, 0),
  })
}

function targetFamilyMask(base, familyId) {
  const footprint = familyReferenceFootprint(base, familyId)
  if (!familyId) return { bundle: bundleFor(base, 'target-family-ablation'), footprint }
  const familyRun = structuredClone(base.familyRun)
  // The family definition remains as an inert coordinate. Only references
  // capable of routing, activating, evaluating, or assigning it are removed.
  // This lets the sham remove exactly the same reference kinds and counts.
  familyRun.memberships = (familyRun.memberships ?? []).filter((membership) => membership.familyId !== familyId)
  familyRun.familyTransitions = (familyRun.familyTransitions ?? []).filter((edge) => edge.fromFamilyId !== familyId && edge.toFamilyId !== familyId)
  familyRun.utilityReports = (familyRun.utilityReports ?? []).filter((report) => report.familyId !== familyId)
  if (familyRun.coalition) {
    familyRun.coalition.coalitions = (familyRun.coalition.coalitions ?? [])
      .filter((coalition) => coalition.eventFamilyId !== familyId)
      .map((coalition) => ({
        ...coalition,
        familyActivations: (coalition.familyActivations ?? []).filter((entry) => entry.familyId !== familyId),
        bindings: (coalition.bindings ?? []).filter((entry) => entry.roleFamilyId !== familyId),
      }))
  }
  const thoughtHead = structuredClone(base.thoughtHead)
  thoughtHead.births = (thoughtHead.births ?? []).map((birth) => ({
    ...birth,
    familyMemberships: (birth.familyMemberships ?? []).filter((membership) => membership.familyId !== familyId),
  }))
  return { bundle: bundleFor(base, 'target-family-ablation', { familyRun, thoughtHead }), footprint }
}

function deterministicReferenceSelection(entries, count, label, snapshotId) {
  const ranked = entries.map((entry) => ({
    ...entry,
    rank: stableHash('family-sham-reference', snapshotId, label, entry.key, entry.descriptor),
  })).sort((left, right) => left.rank.localeCompare(right.rank) || left.key.localeCompare(right.key))
  const selectedEntries = ranked.slice(0, Math.min(count, ranked.length))
  return {
    selected: new Set(selectedEntries.map((entry) => entry.key)),
    selectedEvidence: selectedEntries.map((entry) => ({ key: entry.key, descriptor: entry.descriptor })),
    sufficient: ranked.length >= count,
  }
}

function familyShamMask(base, targetFamilyId, targetFootprint) {
  const snapshotId = base.reference.snapshotId
  const memberships = deterministicReferenceSelection(
    (base.familyRun.memberships ?? []).flatMap((entry, index) => entry.familyId === targetFamilyId ? [] : [{
      key: String(index), descriptor: { memberKey: entry.memberKey, familyId: entry.familyId },
    }]),
    targetFootprint.memberships,
    'memberships',
    snapshotId,
  )
  const familyTransitions = deterministicReferenceSelection(
    (base.familyRun.familyTransitions ?? []).flatMap((entry, index) => (
      entry.fromFamilyId === targetFamilyId || entry.toFamilyId === targetFamilyId
        ? []
        : [{ key: String(index), descriptor: { id: entry.id, from: entry.fromFamilyId, to: entry.toFamilyId } }]
    )),
    targetFootprint.familyTransitions,
    'family-transitions',
    snapshotId,
  )
  const utilityReports = deterministicReferenceSelection(
    (base.familyRun.utilityReports ?? []).flatMap((entry, index) => entry.familyId === targetFamilyId ? [] : [{
      key: String(index), descriptor: { familyId: entry.familyId, mode: entry.mode, status: entry.status },
    }]),
    targetFootprint.utilityReports,
    'utility-reports',
    snapshotId,
  )
  const sourceCoalitions = base.familyRun.coalition?.coalitions ?? []
  const coalitions = deterministicReferenceSelection(
    sourceCoalitions.flatMap((entry, index) => (
      entry.eventFamilyId === targetFamilyId ||
      (entry.familyActivations ?? []).some((activation) => activation.familyId === targetFamilyId) ||
      (entry.bindings ?? []).some((binding) => binding.roleFamilyId === targetFamilyId)
        ? []
        : [{ key: String(index), descriptor: { id: entry.id, eventFamilyId: entry.eventFamilyId, eventMemberKey: entry.eventMemberKey } }]
    )),
    targetFootprint.coalitions,
    'coalitions',
    snapshotId,
  )
  const activationCandidates = []
  const bindingCandidates = []
  sourceCoalitions.forEach((coalition, coalitionIndex) => {
    if (coalitions.selected.has(String(coalitionIndex))) return
    const familyActivations = coalition.familyActivations ?? []
    familyActivations.forEach((entry, entryIndex) => {
      if (entry.familyId !== targetFamilyId) activationCandidates.push({
        key: `${coalitionIndex}:${entryIndex}`,
        descriptor: { coalitionId: coalition.id, familyId: entry.familyId },
      })
    })
    const bindings = coalition.bindings ?? []
    bindings.forEach((entry, entryIndex) => {
      if (entry.roleFamilyId !== targetFamilyId) bindingCandidates.push({
        key: `${coalitionIndex}:${entryIndex}`,
        descriptor: { coalitionId: coalition.id, role: entry.role, roleFamilyId: entry.roleFamilyId, memberKey: entry.memberKey },
      })
    })
  })
  const coalitionActivations = deterministicReferenceSelection(
    activationCandidates,
    targetFootprint.coalitionActivations,
    'coalition-activations',
    snapshotId,
  )
  const coalitionBindings = deterministicReferenceSelection(
    bindingCandidates,
    targetFootprint.coalitionBindings,
    'coalition-bindings',
    snapshotId,
  )
  const birthMemberships = deterministicReferenceSelection(
    (base.thoughtHead.births ?? []).flatMap((birth, birthIndex) => (birth.familyMemberships ?? [])
      .flatMap((entry, entryIndex) => entry.familyId === targetFamilyId ? [] : [{
        key: `${birthIndex}:${entryIndex}`,
        descriptor: { birthId: birth.id, familyId: entry.familyId },
      }])),
    targetFootprint.birthMemberships,
    'birth-memberships',
    snapshotId,
  )
  const selections = {
    memberships,
    familyTransitions,
    utilityReports,
    coalitions,
    coalitionActivations,
    coalitionBindings,
    birthMemberships,
  }
  const familyRun = structuredClone(base.familyRun)
  familyRun.memberships = (familyRun.memberships ?? []).filter((_entry, index) => !memberships.selected.has(String(index)))
  familyRun.familyTransitions = (familyRun.familyTransitions ?? []).filter((_entry, index) => !familyTransitions.selected.has(String(index)))
  familyRun.utilityReports = (familyRun.utilityReports ?? []).filter((_entry, index) => !utilityReports.selected.has(String(index)))
  if (familyRun.coalition) {
    familyRun.coalition.coalitions = (familyRun.coalition.coalitions ?? []).flatMap((coalition, coalitionIndex) => {
      if (coalitions.selected.has(String(coalitionIndex))) return []
      return [{
        ...coalition,
        familyActivations: (coalition.familyActivations ?? [])
          .filter((_entry, entryIndex) => !coalitionActivations.selected.has(`${coalitionIndex}:${entryIndex}`)),
        bindings: (coalition.bindings ?? [])
          .filter((_entry, entryIndex) => !coalitionBindings.selected.has(`${coalitionIndex}:${entryIndex}`)),
      }]
    })
  }
  const thoughtHead = structuredClone(base.thoughtHead)
  thoughtHead.births = (thoughtHead.births ?? []).map((birth, birthIndex) => ({
    ...birth,
    familyMemberships: (birth.familyMemberships ?? [])
      .filter((_entry, entryIndex) => !birthMemberships.selected.has(`${birthIndex}:${entryIndex}`)),
  }))
  const footprint = completeFamilyFootprint(Object.fromEntries(
    FAMILY_FOOTPRINT_KINDS.map((kind) => [kind, selections[kind].selected.size]),
  ))
  const matched = FAMILY_FOOTPRINT_KINDS.every((kind) => selections[kind].sufficient && footprint[kind] === targetFootprint[kind])
  const maskEvidence = Object.fromEntries(FAMILY_FOOTPRINT_KINDS.map((kind) => [kind, selections[kind].selectedEvidence]))
  return {
    bundle: bundleFor(base, 'sham-family-ablation', { familyRun, thoughtHead }),
    footprint,
    matched,
    maskDigest: stableHash('family-sham-mask', snapshotId, targetFamilyId, targetFootprint, maskEvidence),
  }
}

function remappedTarget(keys, from, to, label, index) {
  if (keys.length < 2) return to
  const offset = Number.parseInt(stableHash(label, from, to, index).slice(0, 8), 16) % keys.length
  for (let step = 0; step < keys.length; step += 1) {
    const candidate = keys[(offset + step) % keys.length]
    if (candidate !== from && candidate !== to) return candidate
  }
  return keys.find((candidate) => candidate !== from) ?? to
}

function remapEdges(edges, fromField, toField, keys, label) {
  const seen = new Set()
  const result = []
  edges.forEach((edge, index) => {
    const from = edge[fromField]
    const to = remappedTarget(keys, from, edge[toField], label, index)
    const key = edgeKey(from, to)
    if (seen.has(key)) return
    seen.add(key)
    result.push({ ...edge, [toField]: to })
  })
  return result
}

function shuffledBundle(base) {
  const neuronKeys = [...new Set([
    ...base.neurons.map((neuron) => neuron.memberKey),
    ...(base.thoughtHead.births ?? []).map((birth) => birth.id),
  ])].sort()
  const familyRun = structuredClone(base.familyRun)
  familyRun.neuronTransitions = remapEdges(
    familyRun.neuronTransitions ?? [], 'fromMemberKey', 'toMemberKey', neuronKeys, 'family-h0-shuffle',
  )
  const thoughtHead = structuredClone(base.thoughtHead)
  thoughtHead.neuronEdges = remapEdges(
    thoughtHead.neuronEdges ?? [], 'fromMemberKey', 'toMemberKey', neuronKeys, 'head-edge-shuffle',
  )
  const languageState = base.languageState ? structuredClone(base.languageState) : null
  if (languageState) {
    languageState.neuronTransitions = remapEdges(
      languageState.neuronTransitions ?? [], 'fromKey', 'toKey', neuronKeys, 'language-edge-shuffle',
    )
  }
  return bundleFor(base, 'shuffled-edge', { familyRun, thoughtHead, languageState })
}

function createTraceCollector() {
  return {
    familyCounts: new Map(),
    edgeCounts: new Map(),
    birthEpisode: false,
    onTick(tick) {
      if ((tick.birthEvents ?? []).length) this.birthEpisode = true
      for (const branch of tick.branches ?? []) {
        if (branch.selectedFamilyId) this.familyCounts.set(branch.selectedFamilyId, (this.familyCounts.get(branch.selectedFamilyId) ?? 0) + 1)
      }
    },
    observeEpisode(episode) {
      if ((episode?.temporaryBirths ?? []).length) this.birthEpisode = true
      for (const branch of episode?.branches ?? []) {
        for (const transition of branch.transitions ?? []) {
          const key = edgeKey(transition.fromNeuronId, transition.toNeuronId)
          this.edgeCounts.set(key, (this.edgeCounts.get(key) ?? 0) + 1)
          if (transition.familyId) this.familyCounts.set(transition.familyId, (this.familyCounts.get(transition.familyId) ?? 0) + 1)
        }
      }
    },
  }
}

function mergeCountMap(target, source) {
  for (const [key, count] of source) target.set(key, (target.get(key) ?? 0) + count)
}

async function runMeasuredEpisode({
  thoughtEngine,
  base,
  bundle,
  benchmarkCase,
  seed,
  maxTicks,
  signal,
  benchmarkInitialSignal,
  learningMode = 'off',
  outcomeEvidence = null,
}) {
  abortIfRequested(signal)
  const collector = createTraceCollector()
  const result = await thoughtEngine.runEpisode({
    format: 'thought-engine-request',
    version: 1,
    episodeId: `benchmark-${stableHash(base.reference.snapshotId, bundle.variant, benchmarkCase.id, seed, bundle.thoughtHead.fingerprint).slice(0, 24)}`,
    inputText: benchmarkCase.prompt,
    seed,
    maxTicks,
    strategy: 'seeded-stochastic',
    learningMode,
    teacherMode: 'off',
    sourceGroupId: benchmarkCase.sourceGroupId,
    base: requestBase(base, bundle),
    expectedOutcome: benchmarkCase.expected,
    ...(outcomeEvidence ? { outcomeEvidence } : {}),
  }, {
    signal,
    neurons: base.neurons,
    familyRun: bundle.familyRun,
    languageState: bundle.languageState,
    thoughtHead: bundle.thoughtHead,
    baseDiagnostics: {
      benchmarkVariant: bundle.variant,
      snapshotId: base.reference.snapshotId,
      runnerFingerprint: STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT,
    },
    ...(benchmarkInitialSignal ? { benchmarkInitialSignal } : {}),
    onTick: (tick) => collector.onTick(tick),
  })
  abortIfRequested(signal)
  if (result?.episode?.status === 'cancelled') {
    const error = new Error('Yapısal benchmark düşünce episode’u iptal edildi.')
    error.code = 'JOB_CANCELLED'
    throw error
  }
  collector.observeEpisode(result?.episode)
  const graphScore = scoreEventGraph(result.graph.eventGraph, benchmarkCase.expected)
  return { result, collector, graphF1: graphScore.f1 }
}

function observation(seed, benchmarkCase, score) {
  return { seed, sourceGroupId: benchmarkCase.sourceGroupId, score: clamp01(score) }
}

function keyedObservations(rows) {
  return new Map(rows.map((row) => [`${row.seed}${EDGE_SEPARATOR}${row.caseId}`, row]))
}

export function pairedSignedAblationDrops(fullRows, ablatedRows) {
  const ablated = keyedObservations(ablatedRows)
  return fullRows.map((row) => ({
    seed: row.seed,
    sourceGroupId: row.sourceGroupId,
    score: clampSigned(row.score - (ablated.get(`${row.seed}${EDGE_SEPARATOR}${row.caseId}`)?.score ?? row.score)),
  }))
}

function pairedBaselineAdvantageObservations(fullRows, ...baselineRows) {
  const baselines = baselineRows.map(keyedObservations)
  return fullRows.map((row) => {
    const key = `${row.seed}${EDGE_SEPARATOR}${row.caseId}`
    const strongest = Math.max(...baselines.map((entries) => entries.get(key)?.score ?? row.score))
    return {
      seed: row.seed,
      sourceGroupId: row.sourceGroupId,
      score: clampSigned(row.score - strongest),
    }
  })
}

function interval(rows, samples = BOOTSTRAP_SAMPLES) {
  return bootstrapSourceGroupInterval(rows.map((row) => ({
    seed: row.seed,
    sourceGroupId: row.sourceGroupId,
    score: clamp01(row.score),
  })), samples)
}

export function signedSourceGroupInterval(rows, samples = BOOTSTRAP_SAMPLES) {
  // The shared bootstrap accepts ratios only. This affine mapping is exact and
  // preserves negative paired effects instead of clipping them to zero.
  const mapped = bootstrapSourceGroupInterval(rows.map((row) => ({
    seed: row.seed,
    sourceGroupId: row.sourceGroupId,
    score: (clampSigned(row.score) + 1) / 2,
  })), samples)
  return {
    ...mapped,
    mean: mapped.mean * 2 - 1,
    low: mapped.low * 2 - 1,
    high: mapped.high * 2 - 1,
  }
}

function percentile(values, probability) {
  if (!values.length) return 0
  const ordered = [...values].sort((left, right) => left - right)
  const position = (ordered.length - 1) * probability
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return ordered[lower]
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
}

function deterministicUnit(...parts) {
  return Number.parseInt(stableHash('structural-bootstrap', ...parts).slice(0, 8), 16) / 0x1_0000_0000
}

function birthPrecisionInterval(rows, samples = BOOTSTRAP_SAMPLES) {
  const seedValues = [...new Set(rows.map((row) => row.seed))].sort((left, right) => left - right)
  if (seedValues.length !== 5) throw new RangeError('Doğum precision raporu tam beş seed gerektirir.')
  const positiveGroups = [...new Set(rows.filter((row) => row.truth).map((row) => row.sourceGroupId))].sort()
  const negativeGroups = [...new Set(rows.filter((row) => !row.truth).map((row) => row.sourceGroupId))].sort()
  if (!positiveGroups.length || !negativeGroups.length) throw new RangeError('Doğum precision raporu hem pozitif hem bilinen negatif kaynak grubu gerektirir.')
  const indexed = new Map(rows.map((row) => [`${row.seed}${EDGE_SEPARATOR}${row.sourceGroupId}`, row]))
  const confusion = rows.reduce((counts, row) => {
    if (row.truth && row.predicted) counts.truePositive += 1
    else if (!row.truth && row.predicted) counts.falsePositive += 1
    else if (row.truth) counts.falseNegative += 1
    else counts.trueNegative += 1
    return counts
  }, { truePositive: 0, falsePositive: 0, falseNegative: 0, trueNegative: 0 })
  const pointDenominator = confusion.truePositive + confusion.falsePositive
  const point = pointDenominator ? confusion.truePositive / pointDenominator : 0
  const estimates = []
  for (let sample = 0; sample < samples; sample += 1) {
    let truePositive = 0
    let falsePositive = 0
    for (const seed of seedValues) {
      for (let draw = 0; draw < positiveGroups.length; draw += 1) {
        const group = positiveGroups[Math.min(positiveGroups.length - 1, Math.floor(deterministicUnit(sample, seed, 'positive', draw) * positiveGroups.length))]
        if (indexed.get(`${seed}${EDGE_SEPARATOR}${group}`)?.predicted) truePositive += 1
      }
      for (let draw = 0; draw < negativeGroups.length; draw += 1) {
        const group = negativeGroups[Math.min(negativeGroups.length - 1, Math.floor(deterministicUnit(sample, seed, 'negative', draw) * negativeGroups.length))]
        if (indexed.get(`${seed}${EDGE_SEPARATOR}${group}`)?.predicted) falsePositive += 1
      }
    }
    estimates.push(truePositive + falsePositive ? truePositive / (truePositive + falsePositive) : 0)
  }
  return {
    interval: {
      mean: point,
      low: percentile(estimates, 0.025),
      high: percentile(estimates, 0.975),
      confidence: 0.95,
      seedCount: seedValues.length,
      sourceGroupCount: positiveGroups.length + negativeGroups.length,
      samples,
    },
    confusion,
  }
}

function caseDigest(cases) {
  return stableHash(cases.map((item) => ({
    id: item.id,
    sourceGroupId: item.sourceGroupId,
    prompt: item.prompt,
    expected: item.expected,
  })))
}

function initializeObservationCache(store) {
  const byKey = new Map()
  for (const metadata of store?.listArtifacts?.() ?? []) {
    if (metadata?.kind !== OBSERVATION_CHUNK_KIND) continue
    const artifact = store.getArtifact?.(metadata.id)
    if (typeof artifact?.payload?.chunkKey === 'string') byKey.set(artifact.payload.chunkKey, artifact)
  }
  return { byKey }
}

function observationChunkKey(cache, { chunkType, variant, seed, cases, maxTicks }) {
  return stableHash({
    format: OBSERVATION_CHUNK_FORMAT,
    version: 1,
    runnerFingerprint: STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT,
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    snapshotId: cache.snapshotId,
    inputManifestHash: cache.inputManifestHash,
    chunkType,
    variant,
    seed,
    maxTicks,
    caseDigest: caseDigest(cases),
  })
}

function caseObservationArtifacts(cache, { variant, seeds, cases, maxTicks }) {
  return seeds.map((seed) => cache.index.byKey.get(observationChunkKey(cache, {
    chunkType: 'case-seed', variant, seed, cases, maxTicks,
  })) ?? null)
}

function completeCaseObservationCache(cache, options) {
  return caseObservationArtifacts(cache, options).every(Boolean)
}

function caseObservationChunkDigests(cache, options) {
  const artifacts = caseObservationArtifacts(cache, options)
  if (artifacts.some((artifact) => !artifact || typeof artifact.contentHash !== 'string')) {
    const error = new Error('Yapısal benchmark karar kanıtı eksik observation chunk içeriyor.')
    error.code = 'INVALID_STRUCTURAL_BENCHMARK_CHUNK'
    throw error
  }
  return artifacts.map((artifact) => artifact.contentHash)
}

function validateCaseObservationChunk(artifact, expected) {
  const value = artifact?.payload
  const expectedCases = new Map(expected.cases.map((item) => [item.id, item.sourceGroupId]))
  const countsValid = (entries) => Array.isArray(entries) && entries.every((entry) => (
    Array.isArray(entry) && entry.length === 2 && typeof entry[0] === 'string' && Number.isSafeInteger(entry[1]) && entry[1] >= 0
  ))
  const valid = value?.format === OBSERVATION_CHUNK_FORMAT && value.version === 1 &&
    value.runnerFingerprint === STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT &&
    value.benchmarkFingerprint === INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT &&
    value.snapshotId === expected.cache.snapshotId && value.inputManifestHash === expected.cache.inputManifestHash &&
    value.chunkType === 'case-seed' && value.variant === expected.variant && value.seed === expected.seed &&
    value.maxTicks === expected.maxTicks && value.caseDigest === caseDigest(expected.cases) &&
    value.chunkKey === expected.chunkKey &&
    Array.isArray(value.rows) && value.rows.length === expected.cases.length &&
    value.rows.every((row) => (
      row?.seed === expected.seed && expectedCases.get(row.caseId) === row.sourceGroupId &&
      Number.isFinite(row.score) && row.score >= 0 && row.score <= 1 &&
      Number.isFinite(row.graphF1) && row.graphF1 >= 0 && row.graphF1 <= 1 && typeof row.birth === 'boolean'
    )) && new Set(value.rows.map((row) => row.caseId)).size === expected.cases.length &&
    countsValid(value.familyCounts) && countsValid(value.edgeCounts) &&
    Number.isSafeInteger(value.birthEpisodes) && value.birthEpisodes >= 0 &&
    Array.isArray(value.signalCosines) && value.signalCosines.every((item) => Number.isFinite(item) && item >= -1 && item <= 1) &&
    artifact.contentHash === stableHash(OBSERVATION_CHUNK_KIND, value)
  if (!valid) {
    const error = new Error('Snapshot-bound yapısal benchmark observation chunk doğrulanamadı.')
    error.code = 'INVALID_STRUCTURAL_BENCHMARK_CHUNK'
    throw error
  }
  return value
}

function persistObservationChunk(cache, payload) {
  const artifact = cache.store.createClaimedArtifact(cache.jobId, cache.jobClaim.owner, cache.jobClaim.attempt, {
    kind: OBSERVATION_CHUNK_KIND,
    contentHash: stableHash(OBSERVATION_CHUNK_KIND, payload),
    payload,
  })
  cache.index.byKey.set(payload.chunkKey, artifact)
  return artifact
}

async function evaluateCases({
  thoughtEngine,
  base,
  bundle,
  cases,
  seeds,
  maxTicks,
  signal,
  checkpoint,
  progress,
  perturbSignal = null,
  counter,
  cache,
}) {
  const rows = []
  const familyCounts = new Map()
  const edgeCounts = new Map()
  let birthEpisodes = 0
  const signalCosines = []
  for (const seed of seeds) {
    const chunkKey = observationChunkKey(cache, {
      chunkType: 'case-seed', variant: bundle.variant, seed, cases, maxTicks,
    })
    const cachedArtifact = cache.index.byKey.get(chunkKey)
    if (cachedArtifact) {
      const cached = validateCaseObservationChunk(cachedArtifact, {
        cache, variant: bundle.variant, seed, cases, maxTicks, chunkKey,
      })
      rows.push(...structuredClone(cached.rows))
      mergeCountMap(familyCounts, new Map(cached.familyCounts))
      mergeCountMap(edgeCounts, new Map(cached.edgeCounts))
      birthEpisodes += cached.birthEpisodes
      signalCosines.push(...cached.signalCosines)
      counter.completed += cases.length
      await progress?.(`teacher-feed:benchmark:${bundle.variant}`, Math.min(0.999, counter.completed / counter.total))
      continue
    }
    const seedRows = []
    const seedFamilyCounts = new Map()
    const seedEdgeCounts = new Map()
    let seedBirthEpisodes = 0
    const seedSignalCosines = []
    for (const benchmarkCase of cases) {
      abortIfRequested(signal)
      const perturbation = perturbSignal ? await perturbSignal(benchmarkCase, seed) : null
      const measured = await runMeasuredEpisode({
        thoughtEngine, base, bundle, benchmarkCase, seed, maxTicks, signal,
        ...(perturbation ? { benchmarkInitialSignal: perturbation.signal } : {}),
      })
      const score = perturbation
        ? (perturbation.cosine >= MINIMUM_SMALL_SIGNAL_COSINE && measured.graphF1 >= CORRECT_GRAPH_F1 ? 1 : 0)
        : measured.graphF1
      seedRows.push({
        ...observation(seed, benchmarkCase, score),
        caseId: benchmarkCase.id,
        graphF1: measured.graphF1,
        birth: measured.collector.birthEpisode,
      })
      mergeCountMap(seedFamilyCounts, measured.collector.familyCounts)
      mergeCountMap(seedEdgeCounts, measured.collector.edgeCounts)
      if (measured.collector.birthEpisode) seedBirthEpisodes += 1
      if (perturbation) seedSignalCosines.push(perturbation.cosine)
      counter.completed += 1
      if (counter.completed % 8 === 0) {
        await checkpoint?.(`teacher-feed:benchmark:${bundle.variant}:${counter.completed}`)
        await progress?.(`teacher-feed:benchmark:${bundle.variant}`, Math.min(0.999, counter.completed / counter.total))
      }
    }
    const payload = {
      format: OBSERVATION_CHUNK_FORMAT,
      version: 1,
      runnerFingerprint: STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT,
      benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
      snapshotId: cache.snapshotId,
      inputManifestHash: cache.inputManifestHash,
      chunkType: 'case-seed',
      variant: bundle.variant,
      seed,
      maxTicks,
      caseDigest: caseDigest(cases),
      chunkKey,
      rows: seedRows,
      familyCounts: [...seedFamilyCounts].sort(([left], [right]) => left.localeCompare(right)),
      edgeCounts: [...seedEdgeCounts].sort(([left], [right]) => left.localeCompare(right)),
      birthEpisodes: seedBirthEpisodes,
      signalCosines: seedSignalCosines,
    }
    persistObservationChunk(cache, payload)
    rows.push(...seedRows)
    mergeCountMap(familyCounts, seedFamilyCounts)
    mergeCountMap(edgeCounts, seedEdgeCounts)
    birthEpisodes += seedBirthEpisodes
    signalCosines.push(...seedSignalCosines)
    await checkpoint?.(`teacher-feed:benchmark:${bundle.variant}:seed:${seed}:persisted`)
  }
  return { rows, familyCounts, edgeCounts, birthEpisodes, signalCosines }
}

function mergeCaseEvaluations(...evaluations) {
  const merged = {
    rows: [],
    familyCounts: new Map(),
    edgeCounts: new Map(),
    birthEpisodes: 0,
    signalCosines: [],
  }
  for (const evaluation of evaluations) {
    merged.rows.push(...evaluation.rows)
    mergeCountMap(merged.familyCounts, evaluation.familyCounts)
    mergeCountMap(merged.edgeCounts, evaluation.edgeCounts)
    merged.birthEpisodes += evaluation.birthEpisodes
    merged.signalCosines.push(...evaluation.signalCosines)
  }
  return merged
}

function meanGraphF1(rows) {
  return rows.length ? rows.reduce((sum, row) => sum + row.graphF1, 0) / rows.length : 0
}

async function perturbedSignal(base, benchmarkCase, seed) {
  const encoded = await encodeThoughtPrompt(benchmarkCase.prompt, {
    allowFlatFallback: true,
    identityRatio: base.layout.identityLength / base.layout.cellCount,
  })
  if (encoded.representationSpaceId !== base.familyRun.representationSpaceId || encoded.signal.length !== base.layout.cellCount) {
    throw new TypeError('Benchmark sinyal bozuntusu snapshot temsil uzayıyla uyuşmuyor.')
  }
  const perturbed = new Float32Array(encoded.signal)
  for (let index = 0; index < SIGNAL_PERTURBATION_COORDINATES; index += 1) {
    const digest = stableHash('signal-perturbation', benchmarkCase.id, seed, index)
    const offset = Number.parseInt(digest.slice(0, 8), 16) % base.layout.relationLength
    const coordinate = base.layout.relationStart + offset
    const direction = Number.parseInt(digest.slice(8, 10), 16) % 2 === 0 ? -1 : 1
    perturbed[coordinate] = Math.fround(Math.max(-1, Math.min(1, perturbed[coordinate] + direction * SIGNAL_PERTURBATION_AMPLITUDE)))
  }
  const projected = projectThoughtVector(perturbed, 1)
  return { signal: projected, cosine: cosineSimilarity(encoded.signal, projected) }
}

function applyRawHeadDelta(base, familyRun, head, delta) {
  if (!delta) return head
  const edgeMap = new Map((head.neuronEdges ?? []).map((edge) => [edgeKey(edge.fromMemberKey, edge.toMemberKey), { ...edge }]))
  for (const edge of delta.neuronEdges ?? []) {
    const key = edgeKey(edge.fromMemberKey, edge.toMemberKey)
    const previous = edgeMap.get(key) ?? {
      fromMemberKey: edge.fromMemberKey,
      toMemberKey: edge.toMemberKey,
      excitation: 0,
      inhibition: 0,
      trustMass: 0,
      updateCount: 0,
    }
    edgeMap.set(key, {
      ...previous,
      excitation: Math.min(50, previous.excitation + edge.excitation),
      inhibition: Math.min(50, previous.inhibition + edge.inhibition),
      trustMass: previous.trustMass + edge.trustMass,
      updateCount: previous.updateCount + edge.updateCount,
    })
  }
  const patchMap = new Map((head.contextPatches ?? []).map((patch) => [patch.neuronId, {
    ...patch,
    delta: new Float32Array(patch.delta),
  }]))
  for (const patch of delta.contextPatches ?? []) {
    const previous = patchMap.get(patch.memberKey) ?? {
      neuronId: patch.memberKey,
      delta: new Float32Array(base.layout.relationLength),
      weight: 0,
      updateCount: 0,
    }
    const merged = new Float32Array(base.layout.relationLength)
    for (let index = 0; index < merged.length; index += 1) {
      merged[index] = Math.fround(Math.max(-1, Math.min(1, previous.delta[index] + (patch.relationDelta[index] ?? 0))))
    }
    patchMap.set(patch.memberKey, {
      neuronId: patch.memberKey,
      delta: merged,
      weight: previous.weight + patch.trustMass,
      updateCount: previous.updateCount + patch.observationCount,
    })
  }
  const nursery = new Map((head.nursery ?? []).map((candidate) => [candidate.id, structuredClone(candidate)]))
  for (const candidate of delta.nurseryUpserts ?? []) nursery.set(candidate.id, structuredClone(candidate))
  for (const id of delta.nurseryRemoveIds ?? []) nursery.delete(id)
  const births = new Map((head.births ?? []).map((birth) => [birth.id, structuredClone(birth)]))
  for (const birth of delta.births ?? []) births.set(birth.id, structuredClone(birth))
  return createThoughtHead(modelFor(base, familyRun), {
    format: 'bombom-thought-head',
    version: 1,
    revision: head.revision + 1,
    baseModelFingerprint: head.baseModelFingerprint,
    totalTicks: head.totalTicks + Math.max(0, Math.trunc(delta.stats?.totalTicks ?? 0)),
    neuronEdges: [...edgeMap.values()],
    contextPatches: [...patchMap.values()],
    nursery: [...nursery.values()],
    births: [...births.values()],
    fingerprint: '',
  }, base.reference.baseModelFingerprint)
}

function noveltyCasesAbsentFromSnapshot(base) {
  const surfaces = new Set(base.neurons.map((neuron) => String(neuron.text ?? '').normalize('NFC').toLocaleLowerCase('tr-TR').trim()).filter(Boolean))
  return NOVELTY_CASES.filter((item) => !surfaces.has(item.prompt.toLocaleLowerCase('tr-TR')))
}

function frozenKnownNegativeCases(benchmark) {
  const seen = new Set()
  return benchmark
    .filter((item) => item.split === 'train' && item.surfaceIndex === 0)
    .filter((item) => {
      if (seen.has(item.combinationId)) return false
      seen.add(item.combinationId)
      return true
    })
    .sort((left, right) => left.id.localeCompare(right.id))
    .slice(0, NOVELTY_CASES.length)
}

function validateBirthObservationChunk(artifact, expected) {
  const value = artifact?.payload
  const valid = value?.format === OBSERVATION_CHUNK_FORMAT && value.version === 1 &&
    value.runnerFingerprint === STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT &&
    value.benchmarkFingerprint === INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT &&
    value.snapshotId === expected.cache.snapshotId && value.inputManifestHash === expected.cache.inputManifestHash &&
    value.chunkType === 'birth-classification' && value.variant === expected.variant && value.seed === expected.seed &&
    value.maxTicks === expected.maxTicks && value.caseDigest === caseDigest([expected.benchmarkCase]) &&
    value.chunkKey === expected.chunkKey && value.caseId === expected.benchmarkCase.id &&
    value.sourceGroupId === expected.sourceGroupId && value.truth === expected.truth &&
    typeof value.predicted === 'boolean' && value.repetitions === 4 &&
    artifact.contentHash === stableHash(OBSERVATION_CHUNK_KIND, value)
  if (!valid) {
    const error = new Error('Snapshot-bound kalıcı doğum benchmark chunk doğrulanamadı.')
    error.code = 'INVALID_STRUCTURAL_BENCHMARK_CHUNK'
    throw error
  }
  return value
}

async function evaluatePersistentBirthClassification({
  thoughtEngine,
  base,
  benchmark,
  seeds,
  maxTicks,
  signal,
  checkpoint,
  progress,
  counter,
  cache,
}) {
  const eligibleNovelty = new Set(noveltyCasesAbsentFromSnapshot(base).map((item) => item.id))
  const knownNegatives = frozenKnownNegativeCases(benchmark)
  if (knownNegatives.length !== NOVELTY_CASES.length) {
    const error = new Error('Kalıcı doğum precision ölçümü beş frozen train/calibration negatif örnek gerektirir.')
    error.code = 'INVALID_STRUCTURAL_BENCHMARK_CORPUS'
    throw error
  }
  const rows = []
  const classifiedCases = [
    ...NOVELTY_CASES.map((benchmarkCase) => ({ benchmarkCase, truth: true, variant: 'repeated-novelty-birth', sourceGroupId: `novelty:${benchmarkCase.id}` })),
    ...knownNegatives.map((benchmarkCase) => ({ benchmarkCase, truth: false, variant: 'known-repeated-negative', sourceGroupId: `known:${benchmarkCase.sourceGroupId}` })),
  ]
  for (const seed of seeds) {
    for (const classified of classifiedCases) {
      const { benchmarkCase, truth, variant, sourceGroupId } = classified
      if (truth && !eligibleNovelty.has(benchmarkCase.id)) {
        rows.push({ seed, sourceGroupId, truth, predicted: false })
        counter.completed += 4
        continue
      }
      const chunkKey = observationChunkKey(cache, {
        chunkType: 'birth-classification', variant, seed, cases: [benchmarkCase], maxTicks,
      })
      const cachedArtifact = cache.index.byKey.get(chunkKey)
      if (cachedArtifact) {
        const cached = validateBirthObservationChunk(cachedArtifact, {
          cache, variant, seed, benchmarkCase, maxTicks, chunkKey, truth, sourceGroupId,
        })
        rows.push({ seed, sourceGroupId, truth, predicted: cached.predicted })
        counter.completed += 4
        await progress?.(`teacher-feed:benchmark:${variant}`, Math.min(0.999, counter.completed / counter.total))
        continue
      }
      let head = canonicalHead(base, base.familyRun, base.thoughtHead)
      let promoted = false
      for (let repetition = 0; repetition < 4; repetition += 1) {
        const bundle = bundleFor(base, variant, { thoughtHead: head })
        const measuredCase = {
          ...benchmarkCase,
          sourceGroupId: `${sourceGroupId}:observation:${repetition % 2}`,
        }
        const measured = await runMeasuredEpisode({
          thoughtEngine,
          base,
          bundle,
          benchmarkCase: measuredCase,
          seed: seed + repetition * 0x10001,
          maxTicks,
          signal,
          learningMode: 'proposal',
          outcomeEvidence: {
            kind: 'deterministic',
            confidence: 1,
            sourceGroupId: measuredCase.sourceGroupId,
          },
        })
        if ((measured.result.headDelta?.births ?? []).length) promoted = true
        head = applyRawHeadDelta(base, base.familyRun, head, measured.result.headDelta)
        counter.completed += 1
        if (counter.completed % 8 === 0) {
          await checkpoint?.(`teacher-feed:benchmark:${variant}:${counter.completed}`)
          await progress?.(`teacher-feed:benchmark:${variant}`, Math.min(0.999, counter.completed / counter.total))
        }
      }
      const payload = {
        format: OBSERVATION_CHUNK_FORMAT,
        version: 1,
        runnerFingerprint: STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT,
        benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
        snapshotId: cache.snapshotId,
        inputManifestHash: cache.inputManifestHash,
        chunkType: 'birth-classification',
        variant,
        seed,
        maxTicks,
        caseDigest: caseDigest([benchmarkCase]),
        chunkKey,
        caseId: benchmarkCase.id,
        sourceGroupId,
        truth,
        predicted: promoted,
        repetitions: 4,
      }
      persistObservationChunk(cache, payload)
      rows.push({ seed, sourceGroupId, truth, predicted: promoted })
      await checkpoint?.(`teacher-feed:benchmark:${variant}:seed:${seed}:case:${benchmarkCase.id}:persisted`)
    }
  }
  return {
    rows,
    eligibleNoveltyCaseCount: eligibleNovelty.size,
    knownNegativeCaseCount: knownNegatives.length,
  }
}

function existingRunnerReport(store, snapshot) {
  const terminalKinds = new Set([
    INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND,
    INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_ARTIFACT_KIND,
    INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_ARTIFACT_KIND,
  ])
  const ids = new Set(snapshot.artifactIds ?? [])
  for (const artifact of store.listArtifacts?.() ?? []) {
    if (terminalKinds.has(artifact?.kind)) ids.add(artifact.id)
  }
  const candidates = [...ids]
    .map((id) => store.getArtifact?.(id))
    .filter((artifact) => terminalKinds.has(artifact?.kind))
    .filter((artifact) => (
      artifact?.payload?.snapshotId === snapshot.id &&
      artifact.payload.inputManifestHash === snapshot.inputManifestHash &&
      artifact.payload.runnerFingerprint === STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT
    ))
    .sort((left, right) => String(right.createdAt ?? '').localeCompare(String(left.createdAt ?? '')))
  for (const artifact of candidates) {
    const options = {
      expectedSnapshotId: snapshot.id,
      expectedInputManifestHash: snapshot.inputManifestHash,
    }
    const evaluated = artifact.kind === INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_ARTIFACT_KIND
      ? evaluateIndependentStructuralBenchmarkScreen(artifact.payload, options)
      : artifact.kind === INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_ARTIFACT_KIND
        ? evaluateIndependentStructuralBenchmarkEarlyFailure(artifact.payload, options)
        : evaluateIndependentStructuralBenchmark(artifact.payload, options)
    if (evaluated.valid === true) return { artifactId: artifact.id, report: artifact.payload }
  }
  return null
}

function persistReport({ store, jobId, jobClaim, report, kind = INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND }) {
  if (
    typeof store?.createClaimedArtifact !== 'function' || typeof jobId !== 'string' || !jobId ||
    typeof jobClaim?.owner !== 'string' || !Number.isInteger(jobClaim?.attempt)
  ) {
    const error = new Error('Yapısal benchmark raporu için claim-korumalı immutable artifact sözleşmesi eksik.')
    error.code = 'STORE_CLAIM_CONTRACT_MISSING'
    throw error
  }
  return store.createClaimedArtifact(jobId, jobClaim.owner, jobClaim.attempt, {
    kind,
    contentHash: stableHash(kind, report),
    payload: report,
  })
}

function releaseDenseBase(base) {
  base.neurons.length = 0
  base.neuronByKey?.clear?.()
}

function earlyEvaluationPolicy() {
  return {
    sourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
    validationOllamaExampleCount: 0,
    testOllamaExampleCount: 0,
    splitFrozenBeforeTraining: true,
    automaticActivation: false,
    proposalOnly: true,
  }
}

function earlyReportBase({ format, snapshotId, inputManifestHash }) {
  return {
    format,
    version: INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION,
    benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    runnerVersion: STRUCTURAL_BENCHMARK_RUNNER_VERSION,
    runnerFingerprint: STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT,
    origin: 'independent-deterministic',
    deterministic: true,
    usesOllama: false,
    snapshotId,
    inputManifestHash,
  }
}

function earlyEvidenceBase({
  combinationCount,
  holdoutCombinationCount,
  seedCount,
  episodeCount,
  maxTicks,
  observationDigest,
  observationChunkDigests,
  hydrationPreflight,
  denseRuntime,
}) {
  return {
    combinationCount,
    surfacesPerCombination: 2,
    splitPolicy: 'combination-before-surface-v1',
    holdoutRatio: holdoutCombinationCount / combinationCount,
    holdoutCombinationCount,
    seedCount,
    sourceGroupCount: holdoutCombinationCount,
    episodeCount,
    maximumTicks: maxTicks,
    observationDigest,
    observationChunkDigests,
    resourceSafety: {
      metadataPreflight: hydrationPreflight,
      denseRuntime,
    },
  }
}

function fullObservationDigest(rows) {
  return stableHash(rows.map(({ seed, sourceGroupId, caseId, score, graphF1, birth }) => (
    [seed, sourceGroupId, caseId, score, graphF1, birth]
  )))
}

/**
 * Production adapter for teacher-feed's closed stop gate. It never calls
 * Ollama and never writes to a model/snapshot; only the final, replayable
 * snapshot-bound measurement report is persisted under the current job claim.
 */
export function createIndependentStructuralBenchmarkRunner({
  thoughtEngine,
  resolveBase = resolveThoughtEngineBase,
  benchmarkFactory = createStructuralBenchmark,
  seeds = STRUCTURAL_BENCHMARK_SEEDS,
  maxTicks = STRUCTURAL_BENCHMARK_TICKS,
  bootstrapSamples = BOOTSTRAP_SAMPLES,
} = {}) {
  if (typeof thoughtEngine?.runEpisode !== 'function') throw new TypeError('Gerçek Thought engine benchmark runner için gerekli.')
  if (!Array.isArray(seeds) || seeds.length !== 5 || new Set(seeds).size !== 5 || seeds.some((seed) => !Number.isInteger(seed))) {
    throw new TypeError('Yapısal benchmark tam beş farklı tam sayı seed gerektirir.')
  }
  if (!Number.isInteger(maxTicks) || maxTicks < 1 || maxTicks > 512) throw new TypeError('Benchmark tick sayısı 1..512 olmalı.')
  if (!Number.isInteger(bootstrapSamples) || bootstrapSamples < 1_000) throw new TypeError('Benchmark bootstrap en az 1000 örnek gerektirir.')

  return async function structuralBenchmarkReport({
    store,
    snapshot,
    snapshotId,
    signal,
    jobId,
    jobClaim,
    checkpoint,
    progress,
    emit,
  }) {
    abortIfRequested(signal)
    if (!snapshot || snapshot.id !== snapshotId || typeof snapshot.inputManifestHash !== 'string' || !snapshot.inputManifestHash) {
      const error = new Error('Yapısal benchmark exact snapshot/input manifest bağı olmadan çalışmaz.')
      error.code = 'INVALID_STRUCTURAL_BENCHMARK_BASE'
      throw error
    }
    if (
      typeof store?.createClaimedArtifact !== 'function' || typeof store?.listArtifacts !== 'function' || typeof store?.getArtifact !== 'function' ||
      typeof jobId !== 'string' || !jobId || typeof jobClaim?.owner !== 'string' || !Number.isInteger(jobClaim?.attempt)
    ) {
      const error = new Error('Yapısal benchmark durable observation cache için claim-korumalı artifact sözleşmesi eksik.')
      error.code = 'STORE_CLAIM_CONTRACT_MISSING'
      throw error
    }
    const cached = existingRunnerReport(store, snapshot)
    if (cached) return cached
    await checkpoint?.('teacher-feed:benchmark:resolve-base')
    const hydrationPreflight = benchmarkMetadataPreflight(store, snapshot)
    const base = resolveBase(store, snapshotId)
    if (base.reference.snapshotId !== snapshotId) {
      const error = new Error('Benchmark motoru exact 64x64 snapshot tabanını çözemedi.')
      error.code = 'INVALID_STRUCTURAL_BENCHMARK_BASE'
      throw error
    }
    if (base.layout.side !== 64 || base.layout.cellCount !== 4_096 || !base.neurons.length) {
      const error = new Error('Benchmark motoru tam 64x64 fiziksel nöron kataloğu gerektirir.')
      error.code = 'INVALID_STRUCTURAL_BENCHMARK_BASE'
      throw error
    }
    const denseRuntime = denseBenchmarkDiagnostics(base)
    await emit?.({
      type: 'teacher-feed-structural-benchmark-started',
      stage: 'teacher-feed:benchmark',
      data: {
        snapshotId,
        benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
        usesOllama: false,
        denseRuntime,
      },
    })

    const benchmark = benchmarkFactory(0x5eed, 0.30)
    const holdout = benchmark.filter((item) => item.split === 'validation' || item.split === 'test')
    const combinationCount = new Set(benchmark.map((item) => item.combinationId)).size
    const holdoutCombinationCount = new Set(holdout.map((item) => item.combinationId)).size
    if (combinationCount !== 144 || benchmark.length !== 288 || holdoutCombinationCount < 44) {
      const error = new Error('Derlenmiş kontrollü benchmark 144 kombinasyon/iki yüzey/frozen holdout sözleşmesini sağlamıyor.')
      error.code = 'INVALID_STRUCTURAL_BENCHMARK_CORPUS'
      throw error
    }
    const cache = {
      store,
      jobId,
      jobClaim,
      snapshotId,
      inputManifestHash: snapshot.inputManifestHash,
      index: initializeObservationCache(store),
    }
    const counter = {
      completed: 0,
      total: holdout.length * seeds.length * 9 + NOVELTY_CASES.length * seeds.length * 4 * 2,
    }
    const fullBundle = bundleFor(base, 'full')
    const fullCacheOptions = { variant: fullBundle.variant, seeds, cases: holdout, maxTicks }
    const allFullChunksAlreadyDurable = completeCaseObservationCache(cache, fullCacheOptions)
    let full
    if (allFullChunksAlreadyDurable) {
      full = await evaluateCases({ thoughtEngine, base, bundle: fullBundle, cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache })
    } else {
      const screenSeed = seeds[0]
      const screen = await evaluateCases({
        thoughtEngine,
        base,
        bundle: fullBundle,
        cases: holdout,
        seeds: [screenSeed],
        maxTicks,
        signal,
        checkpoint,
        progress,
        counter,
        cache,
      })
      const diagnosticMeanGraphF1 = meanGraphF1(screen.rows)
      if (diagnosticMeanGraphF1 < INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD) {
        const observationChunkDigests = caseObservationChunkDigests(cache, {
          variant: fullBundle.variant,
          seeds: [screenSeed],
          cases: holdout,
          maxTicks,
        })
        const report = {
          ...earlyReportBase({
            format: INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_FORMAT,
            snapshotId,
            inputManifestHash: snapshot.inputManifestHash,
          }),
          screenPolicyFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_POLICY_FINGERPRINT,
          completion: 'diagnostic-screened-out',
          diagnostic: {
            variant: 'full',
            seed: screenSeed,
            threshold: INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD,
            meanGraphF1: diagnosticMeanGraphF1,
          },
          evidence: {
            ...earlyEvidenceBase({
              combinationCount,
              holdoutCombinationCount,
              seedCount: 1,
              episodeCount: screen.rows.length,
              maxTicks,
              observationDigest: fullObservationDigest(screen.rows),
              observationChunkDigests,
              hydrationPreflight,
              denseRuntime,
            }),
            skippedVariants: ['full-seeds-2-5', ...EARLY_FAILURE_SKIPPED_VARIANTS],
            skippedEpisodeCount: counter.total - screen.rows.length,
          },
          evaluationPolicy: earlyEvaluationPolicy(),
        }
        const verification = evaluateIndependentStructuralBenchmarkScreen(report, {
          expectedSnapshotId: snapshotId,
          expectedInputManifestHash: snapshot.inputManifestHash,
        })
        if (!verification.valid) {
          const error = new Error('Yapısal ön ölçüm artifact sözleşmesi doğrulanamadı.')
          error.code = 'INVALID_STRUCTURAL_BENCHMARK_SCREEN'
          error.checks = verification.checks
          throw error
        }
        await checkpoint?.('teacher-feed:benchmark:screen:persist')
        const artifact = persistReport({
          store,
          jobId,
          jobClaim,
          report,
          kind: INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_ARTIFACT_KIND,
        })
        releaseDenseBase(base)
        await emit?.({
          type: 'teacher-feed-structural-benchmark-screened-out',
          stage: 'teacher-feed:benchmark',
          data: {
            snapshotId,
            artifactId: artifact.id,
            diagnosticMeanGraphF1,
            threshold: INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD,
          },
        })
        await progress?.('teacher-feed:benchmark:screened-out', 0.999)
        return { artifactId: artifact.id, report }
      }
      const remaining = await evaluateCases({
        thoughtEngine,
        base,
        bundle: fullBundle,
        cases: holdout,
        seeds: seeds.slice(1),
        maxTicks,
        signal,
        checkpoint,
        progress,
        counter,
        cache,
      })
      full = mergeCaseEvaluations(screen, remaining)
    }
    abortIfRequested(signal)
    const fullInterval = interval(full.rows, bootstrapSamples)
    if (fullInterval.low < INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD) {
      const observationChunkDigests = caseObservationChunkDigests(cache, fullCacheOptions)
      const report = {
        ...earlyReportBase({
          format: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT,
          snapshotId,
          inputManifestHash: snapshot.inputManifestHash,
        }),
        earlyFailurePolicyFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT,
        completion: 'decisive-early-failure',
        decisiveCheck: {
          id: 'holdout-graph-f1',
          threshold: {
            direction: 'minimum',
            boundary: 'ci.low',
            value: INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD,
          },
          interval: fullInterval,
          passed: false,
        },
        evidence: {
          ...earlyEvidenceBase({
            combinationCount,
            holdoutCombinationCount,
            seedCount: seeds.length,
            episodeCount: full.rows.length,
            maxTicks,
            observationDigest: fullObservationDigest(full.rows),
            observationChunkDigests,
            hydrationPreflight,
            denseRuntime,
          }),
          completedVariant: 'full',
          bootstrapConfidence: 0.95,
          bootstrapSamples,
          skippedVariants: [...EARLY_FAILURE_SKIPPED_VARIANTS],
          skippedEpisodeCount: counter.total - full.rows.length,
        },
        evaluationPolicy: earlyEvaluationPolicy(),
      }
      const verification = evaluateIndependentStructuralBenchmarkEarlyFailure(report, {
        expectedSnapshotId: snapshotId,
        expectedInputManifestHash: snapshot.inputManifestHash,
      })
      if (!verification.valid) {
        const error = new Error('Kesin erken benchmark başarısızlık artifact sözleşmesi doğrulanamadı.')
        error.code = 'INVALID_STRUCTURAL_BENCHMARK_EARLY_FAILURE'
        error.checks = verification.checks
        throw error
      }
      await checkpoint?.('teacher-feed:benchmark:early-failure:persist')
      const artifact = persistReport({
        store,
        jobId,
        jobClaim,
        report,
        kind: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_ARTIFACT_KIND,
      })
      releaseDenseBase(base)
      await emit?.({
        type: 'teacher-feed-structural-benchmark-early-failed',
        stage: 'teacher-feed:benchmark',
        data: {
          snapshotId,
          artifactId: artifact.id,
          holdoutGraphF1: fullInterval.mean,
          holdoutGraphF1Low: fullInterval.low,
          holdoutGraphF1High: fullInterval.high,
          threshold: INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD,
          skippedEpisodeCount: report.evidence.skippedEpisodeCount,
        },
      })
      await progress?.('teacher-feed:benchmark:failed-early', 0.999)
      return { artifactId: artifact.id, report }
    }

    const targetFamilyId = chooseTargetFamily(base)
    const targetEdge = chooseTargetEdge(base)
    const shamEdge = chooseShamEdge(base, targetEdge)
    const targetFamily = targetFamilyMask(base, targetFamilyId)
    const shamFamily = familyShamMask(base, targetFamilyId, targetFamily.footprint)
    const variants = {
      h0Only: await evaluateCases({ thoughtEngine, base, bundle: emptyHeadBundle(base), cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache }),
      noRecurrence: await evaluateCases({ thoughtEngine, base, bundle: recurrenceOffBundle(base), cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache }),
      shuffled: await evaluateCases({ thoughtEngine, base, bundle: shuffledBundle(base), cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache }),
      targetFamily: await evaluateCases({ thoughtEngine, base, bundle: targetFamily.bundle, cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache }),
      targetEdge: await evaluateCases({ thoughtEngine, base, bundle: withoutEdge(base, targetEdge, 'target-edge-ablation'), cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache }),
      shamFamily: await evaluateCases({ thoughtEngine, base, bundle: shamFamily.bundle, cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache }),
      shamEdge: await evaluateCases({ thoughtEngine, base, bundle: withoutEdge(base, shamEdge, 'sham-edge-ablation'), cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache }),
      perturbed: await evaluateCases({
        thoughtEngine, base, bundle: bundleFor(base, 'perturbed'), cases: holdout, seeds, maxTicks, signal, checkpoint, progress, counter, cache,
        perturbSignal: (benchmarkCase, seed) => perturbedSignal(base, benchmarkCase, seed),
      }),
    }
    const births = await evaluatePersistentBirthClassification({
      thoughtEngine, base, benchmark, seeds, maxTicks, signal, checkpoint, progress, counter, cache,
    })
    abortIfRequested(signal)

    const h0Interval = interval(variants.h0Only.rows, bootstrapSamples)
    const noRecurrenceInterval = interval(variants.noRecurrence.rows, bootstrapSamples)
    const shuffledInterval = interval(variants.shuffled.rows, bootstrapSamples)
    const baselineAdvantageInterval = signedSourceGroupInterval(pairedBaselineAdvantageObservations(
      full.rows,
      variants.h0Only.rows,
      variants.noRecurrence.rows,
      variants.shuffled.rows,
    ), bootstrapSamples)
    const familyDropInterval = signedSourceGroupInterval(pairedSignedAblationDrops(full.rows, variants.targetFamily.rows), bootstrapSamples)
    const edgeDropInterval = signedSourceGroupInterval(pairedSignedAblationDrops(full.rows, variants.targetEdge.rows), bootstrapSamples)
    const shamFamilyDropInterval = signedSourceGroupInterval(pairedSignedAblationDrops(full.rows, variants.shamFamily.rows), bootstrapSamples)
    const shamEdgeDropInterval = signedSourceGroupInterval(pairedSignedAblationDrops(full.rows, variants.shamEdge.rows), bootstrapSamples)
    const perturbInterval = interval(variants.perturbed.rows, bootstrapSamples)
    const knownBirthRows = full.rows.map((row) => ({
      seed: row.seed,
      sourceGroupId: row.sourceGroupId,
      score: row.birth ? 1 : 0,
    }))
    const falseBirthInterval = interval(knownBirthRows, bootstrapSamples)
    const noveltySensitivityRows = births.rows.filter((row) => row.truth).map((row) => ({
      seed: row.seed,
      sourceGroupId: row.sourceGroupId,
      score: row.predicted ? 1 : 0,
    }))
    const noveltySensitivityInterval = interval(noveltySensitivityRows, bootstrapSamples)
    const birthPrecision = birthPrecisionInterval(births.rows, bootstrapSamples)
    const minimumPerturbationCosine = variants.perturbed.signalCosines.length
      ? Math.min(...variants.perturbed.signalCosines)
      : 0
    const metrics = {
      holdoutGraphF1: clamp01(fullInterval.mean),
      h0OnlyGraphF1: clamp01(h0Interval.mean),
      noRecurrenceGraphF1: clamp01(noRecurrenceInterval.mean),
      shuffledEdgeGraphF1: clamp01(shuffledInterval.mean),
      targetFamilyAblationDrop: clampSigned(familyDropInterval.mean),
      targetEdgeAblationDrop: clampSigned(edgeDropInterval.mean),
      shamFamilyAblationDrop: clampSigned(shamFamilyDropInterval.mean),
      shamEdgeAblationDrop: clampSigned(shamEdgeDropInterval.mean),
      perturbationRecoveryRate: minimumPerturbationCosine >= MINIMUM_SMALL_SIGNAL_COSINE ? clamp01(perturbInterval.mean) : 0,
      knownStructureFalseBirthRate: clamp01(falseBirthInterval.mean),
      repeatedNoveltyBirthPrecision: clamp01(birthPrecision.interval.mean),
      repeatedNoveltyBirthSensitivity: clamp01(noveltySensitivityInterval.mean),
    }
    const observationEvidence = {
      full: full.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      h0Only: variants.h0Only.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      noRecurrence: variants.noRecurrence.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      shuffled: variants.shuffled.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      targetFamily: variants.targetFamily.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      targetEdge: variants.targetEdge.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      shamFamily: variants.shamFamily.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      shamEdge: variants.shamEdge.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      perturbed: variants.perturbed.rows.map(({ seed, sourceGroupId, caseId, score }) => [seed, sourceGroupId, caseId, score]),
      births: births.rows.map(({ seed, sourceGroupId, truth, predicted }) => [seed, sourceGroupId, truth, predicted]),
    }
    const report = {
      format: INDEPENDENT_STRUCTURAL_BENCHMARK_FORMAT,
      version: INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION,
      benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
      benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
      runnerVersion: STRUCTURAL_BENCHMARK_RUNNER_VERSION,
      runnerFingerprint: STRUCTURAL_BENCHMARK_RUNNER_FINGERPRINT,
      origin: 'independent-deterministic',
      deterministic: true,
      usesOllama: false,
      snapshotId,
      inputManifestHash: snapshot.inputManifestHash,
      evidence: {
        combinationCount,
        surfacesPerCombination: 2,
        splitPolicy: 'combination-before-surface-v1',
        holdoutRatio: holdoutCombinationCount / combinationCount,
        holdoutCombinationCount,
        seedCount: seeds.length,
        sourceGroupCount: holdoutCombinationCount,
        bootstrapConfidence: 0.95,
        bootstrapSamples,
        episodeCount: counter.completed,
        maximumTicks: maxTicks,
        ablationTargetSelection: 'snapshot-model-only-no-evaluation-data-v2',
        edgeShamProtocol: 'snapshot-digest-random-nontarget-edge-v1',
        noRecurrenceProtocol: 'full-tick-zero-persistent-transition-evidence-v2',
        targetFamilyId,
        targetEdge,
        shamEdge,
        familyShamProtocol: 'deterministic-model-only-exact-reference-footprint-mask-v1',
        targetFamilyFootprint: targetFamily.footprint,
        shamFamilyFootprint: shamFamily.footprint,
        shamFamilyFootprintMatched: shamFamily.matched,
        shamFamilyMaskDigest: shamFamily.maskDigest,
        minimumPerturbationCosine,
        eligibleNoveltyCaseCount: births.eligibleNoveltyCaseCount,
        knownBirthPrecisionNegativeCaseCount: births.knownNegativeCaseCount,
        birthPrecisionDefinition: 'TP/(TP+FP)',
        birthNegativeSource: 'frozen-train-calibration-repeated-four-observations-v2',
        birthConfusion: birthPrecision.confusion,
        observationDigest: stableHash(observationEvidence),
        resourceSafety: {
          metadataPreflight: hydrationPreflight,
          denseRuntime,
        },
        intervals: {
          holdoutGraphF1: fullInterval,
          h0OnlyGraphF1: h0Interval,
          noRecurrenceGraphF1: noRecurrenceInterval,
          shuffledEdgeGraphF1: shuffledInterval,
          baselineAdvantage: baselineAdvantageInterval,
          targetFamilyAblationDrop: familyDropInterval,
          targetEdgeAblationDrop: edgeDropInterval,
          shamFamilyAblationDrop: shamFamilyDropInterval,
          shamEdgeAblationDrop: shamEdgeDropInterval,
          perturbationRecoveryRate: perturbInterval,
          knownStructureFalseBirthRate: falseBirthInterval,
          repeatedNoveltyBirthPrecision: birthPrecision.interval,
          repeatedNoveltyBirthSensitivity: noveltySensitivityInterval,
        },
      },
      evaluationPolicy: {
        sourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
        validationOllamaExampleCount: 0,
        testOllamaExampleCount: 0,
        splitFrozenBeforeTraining: true,
        automaticActivation: false,
        proposalOnly: true,
      },
      metrics,
    }
    await checkpoint?.('teacher-feed:benchmark:persist')
    const artifact = persistReport({ store, jobId, jobClaim, report })
    // No later stage in this invocation consumes physical I/R. Drop the sole
    // dense catalog reference before notifications/checkpoint handoff.
    releaseDenseBase(base)
    await emit?.({
      type: 'teacher-feed-structural-benchmark-completed',
      stage: 'teacher-feed:benchmark',
      data: { snapshotId, artifactId: artifact.id, observationDigest: report.evidence.observationDigest },
    })
    await progress?.('teacher-feed:benchmark:completed', 0.999)
    return { artifactId: artifact.id, report }
  }
}
