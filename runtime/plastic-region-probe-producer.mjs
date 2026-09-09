import {
  ADAPTIVE_REGION_GEOMETRY,
  PAGE_BEHAVIOR_DIMENSION,
  validateAdaptiveRegionAtlas,
} from '../src/adaptive-regions/index.ts'
import { createStructuralBenchmark } from '../src/thought-runtime/benchmark.ts'
import { persistArtifact, stableRuntimeHash } from './artifacts.mjs'
import {
  SEMANTIC_PROBE_LAYER,
  createSemanticProbeCalibration,
  createSemanticProbeEvaluationEvidence,
  createSemanticProbeObservationSet,
  createSemanticProbePublicSummary,
  semanticProbeArtifactWrite,
} from './plastic-region-probes.mjs'
import { readPlasticRegionSnapshot } from './plastic-region-cache.mjs'

export const STRUCTURAL_SEMANTIC_PROBE_SPEC_ARTIFACT_KIND = 'plastic-region-semantic-probe-spec'
export const STRUCTURAL_SEMANTIC_PROBE_PRODUCER_VERSION = 1

const STRUCTURAL_HYPOTHESIS_SPECS = Object.freeze([
  {
    namespace: 'event-role.actor',
    label: 'yapan=Ada',
    rule: { kind: 'event-field-equals', field: 'actor', value: 'Ada' },
  },
  {
    namespace: 'event-role.action',
    label: 'eylem=taşıdı',
    rule: { kind: 'event-field-equals', field: 'action', value: 'taşıdı' },
  },
  {
    namespace: 'event-role.patient',
    label: 'etkilenen=kırmızı kutu',
    rule: { kind: 'event-field-equals', field: 'patient', value: 'kırmızı kutu' },
  },
  {
    namespace: 'event-role.instrument',
    label: 'araç=kanca',
    rule: { kind: 'event-field-equals', field: 'instrument', value: 'kanca' },
  },
  {
    namespace: 'event-operator.negation',
    label: 'olumsuzluk=var',
    rule: { kind: 'event-field-equals', field: 'negated', value: true },
  },
  {
    namespace: 'morphology.word-order',
    label: 'araç-önde-yüzey',
    rule: { kind: 'surface-index-equals', value: 1 },
  },
])

const PRODUCER_SPEC = Object.freeze({
  format: 'plastic-region-structural-semantic-producer',
  version: STRUCTURAL_SEMANTIC_PROBE_PRODUCER_VERSION,
  layer: SEMANTIC_PROBE_LAYER,
  interpretationOnly: true,
  trainingWritesAllowed: false,
  routingMutationAllowed: false,
  learningMode: 'off',
  teacherMode: 'off',
  adaptiveRegionMode: 'shadow',
  responseReducer: 'tick-mean-mass-weighted-region-gate-v1',
  benchmarkSeed: 0x5eed,
  holdoutRatio: 0.30,
  hypotheses: STRUCTURAL_HYPOTHESIS_SPECS,
})

const MAXIMUM_SEMANTIC_PROBE_ARTIFACT_BYTES = 8 * 1024 * 1024
const DEFAULT_SEMANTIC_PROBE_ARTIFACT_BYTES = 4 * 1024 * 1024

export const STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT = stableRuntimeHash(
  'plastic-region-structural-semantic-producer-v1',
  PRODUCER_SPEC,
)

function producerError(code, message, details) {
  const error = new Error(message)
  error.code = code
  if (details !== undefined) error.details = details
  return error
}

function compareText(left, right) {
  return String(left).localeCompare(String(right))
}

function freezeDeep(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freezeDeep(child)
    Object.freeze(value)
  }
  return value
}

function configuredArtifactLimit(store) {
  const configured = Number(
    store?.getConfig?.().plasticRegions?.maximumArtifactBytes ?? DEFAULT_SEMANTIC_PROBE_ARTIFACT_BYTES,
  )
  if (!Number.isSafeInteger(configured) || configured < 512) {
    throw producerError('SEMANTIC_PROBE_ARTIFACT_LIMIT_INVALID', 'Probe store artifact byte sınırı geçersiz.')
  }
  return Math.min(configured, MAXIMUM_SEMANTIC_PROBE_ARTIFACT_BYTES)
}

function assertProbeSpecBound(store, value) {
  const maximumArtifactBytes = configuredArtifactLimit(store)
  const bytes = Buffer.byteLength(JSON.stringify(value), 'utf8')
  if (bytes > maximumArtifactBytes) {
    throw producerError(
      'SEMANTIC_PROBE_ARTIFACT_LIMIT_EXCEEDED',
      'Deterministik probe spec artifacti store byte sınırını aşıyor.',
      { bytes, maximumArtifactBytes },
    )
  }
  return bytes
}

function clamp01(value, label) {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw producerError('INVALID_SEMANTIC_PROBE_TELEMETRY', `${label} [0,1] aralığında olmalı.`)
  }
  const result = Math.fround(value)
  return Object.is(result, -0) ? 0 : result
}

function hypothesisFromSpec(spec) {
  return {
    namespace: spec.namespace,
    label: spec.label,
    schemaVersion: 1,
    definitionFingerprint: stableRuntimeHash('plastic-region-structural-hypothesis-v1', spec),
  }
}

function labelForCase(spec, benchmarkCase) {
  if (spec.rule.kind === 'surface-index-equals') return Number(benchmarkCase.surfaceIndex === spec.rule.value)
  if (spec.rule.kind === 'event-field-equals') return Number(benchmarkCase.expected?.[spec.rule.field] === spec.rule.value)
  throw producerError('INVALID_SEMANTIC_PROBE_SPEC', 'Bilinmeyen deterministik probe etiket kuralı.')
}

function canonicalCaseManifest(cases) {
  return [...cases].sort((left, right) => compareText(left.id, right.id)).map((item) => ({
    caseId: item.id,
    combinationId: item.combinationId,
    sourceGroupId: item.sourceGroupId,
    surfaceIndex: item.surfaceIndex,
    split: item.split,
    promptDigest: stableRuntimeHash('plastic-region-semantic-prompt-v1', item.prompt),
    expectedDigest: stableRuntimeHash('plastic-region-semantic-expected-v1', item.expected),
  }))
}

export function createStructuralSemanticProbeSpec() {
  const cases = createStructuralBenchmark(PRODUCER_SPEC.benchmarkSeed, PRODUCER_SPEC.holdoutRatio)
  const caseManifest = canonicalCaseManifest(cases)
  const partitionManifestFingerprint = stableRuntimeHash(
    'plastic-region-structural-probe-partition-v1',
    caseManifest.map(({ caseId, combinationId, sourceGroupId, split }) => ({ caseId, combinationId, sourceGroupId, split })),
  )
  const payload = {
    format: STRUCTURAL_SEMANTIC_PROBE_SPEC_ARTIFACT_KIND,
    version: 1,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    usesOllama: false,
    producerFingerprint: STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT,
    datasetId: 'controlled-event-graph-144-semantic-probes-v1',
    partitionManifestFingerprint,
    caseCount: caseManifest.length,
    combinationCount: new Set(caseManifest.map((item) => item.combinationId)).size,
    hypotheses: STRUCTURAL_HYPOTHESIS_SPECS.map((spec) => ({
      ...hypothesisFromSpec(spec),
      ruleFingerprint: stableRuntimeHash('plastic-region-structural-label-rule-v1', spec.rule),
    })),
    caseManifest,
  }
  if (payload.caseCount !== 288 || payload.combinationCount !== 144 ||
      new Set(caseManifest.filter((item) => item.split !== 'train').map((item) => item.combinationId)).size < 44) {
    throw producerError('INVALID_SEMANTIC_PROBE_SPEC', 'Frozen yapısal probe paketi 144 kombinasyon/iki yüzey/holdout sözleşmesini sağlamıyor.')
  }
  return freezeDeep({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-structural-semantic-spec-v1', payload),
  })
}

export function validateStructuralSemanticProbeSpec(value) {
  const expected = createStructuralSemanticProbeSpec()
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw producerError('INVALID_SEMANTIC_PROBE_SPEC', 'Yapısal probe paketi frozen üretici tanımıyla uyuşmuyor.')
  }
  return value
}

function structuralFeatures(atlas, maximumFeatures) {
  const candidates = atlas.regions
    .filter((region) => region.lifecycle === 'active')
    .map((region) => {
      const footprint = atlas.pages.flatMap((page) => page.memberships
        .filter((membership) => membership.regionId === region.id)
        .map((membership) => ({ pageIndex: page.pageIndex, weight: membership.weight })))
      return {
        featureId: region.id,
        responseKind: 'region-gate',
        footprint,
        capacity: footprint.reduce((sum, entry) => sum + entry.weight, 0),
      }
    })
    .filter((feature) => feature.footprint.length > 0)
    .sort((left, right) => right.capacity - left.capacity || compareText(left.featureId, right.featureId))
  return {
    features: candidates.slice(0, maximumFeatures).map(({ capacity: _capacity, ...feature }) => feature),
    omittedFeatureCount: Math.max(0, candidates.length - maximumFeatures),
  }
}

/**
 * Reads only shadow telemetry. `activeApplied:true` is rejected so running a
 * semantic probe cannot itself influence the route it is measuring.
 */
export function extractStructuralSemanticFeatureScores(atlas, result, features) {
  validateAdaptiveRegionAtlas(atlas)
  const selectedFeatures = features ?? structuralFeatures(atlas, 56).features
  const index = new Map(selectedFeatures.map((feature, featureIndex) => [feature.featureId, featureIndex]))
  const totals = new Array(selectedFeatures.length).fill(0)
  let measuredTicks = 0
  for (const tick of result?.ticks ?? []) {
    const tickTotals = new Array(selectedFeatures.length).fill(0)
    let branchMass = 0
    let measuredBranch = false
    for (const branch of tick?.branches ?? []) {
      const telemetry = branch?.adaptiveRegion
      if (!telemetry) continue
      if (telemetry.format !== 'thought-adaptive-region-branch-telemetry' || telemetry.version !== 1 ||
          telemetry.mode !== 'shadow' || telemetry.activeApplied !== false ||
          telemetry.atlasFingerprint !== atlas.fingerprint) {
        throw producerError(
          'SEMANTIC_PROBE_ROUTING_ISOLATION_FAILED',
          'Probe yalnız exact atlas bağlı ve uygulanmamış shadow bölge telemetrisi okuyabilir.',
        )
      }
      const mass = clamp01(Number(branch.mass ?? 0), 'Probe dal kütlesi')
      branchMass += mass
      measuredBranch = true
      const seen = new Set()
      let gateProbabilityMass = 0
      for (const gate of telemetry.gates?.gates ?? []) {
        const probability = clamp01(Number(gate.probability), 'Probe bölge kapısı')
        gateProbabilityMass += probability
        const featureIndex = index.get(gate.regionId)
        if (featureIndex === undefined) continue
        if (seen.has(gate.regionId)) throw producerError('INVALID_SEMANTIC_PROBE_TELEMETRY', 'Aynı dalda bölge kapısı yinelenemez.')
        seen.add(gate.regionId)
        tickTotals[featureIndex] += mass * probability
      }
      const unmatched = clamp01(Number(telemetry.gates?.unmatchedProbability ?? 0), 'Probe eşleşmeyen kütlesi')
      const omitted = clamp01(Number(telemetry.gates?.omittedRegionProbability ?? 0), 'Probe atlanan bölge kütlesi')
      if (gateProbabilityMass + unmatched + omitted > 1 + 1e-5) {
        throw producerError('INVALID_SEMANTIC_PROBE_TELEMETRY', 'Probe kapı dağılımı olasılık kütlesini aşıyor.')
      }
    }
    if (!measuredBranch || branchMass <= 0) continue
    for (let featureIndex = 0; featureIndex < totals.length; featureIndex += 1) {
      totals[featureIndex] += tickTotals[featureIndex] / branchMass
    }
    measuredTicks += 1
  }
  if (!measuredTicks) {
    throw producerError('SEMANTIC_PROBE_TELEMETRY_REQUIRED', 'Deterministik probe vakası ölçülebilir shadow bölge telemetrisi üretmedi.')
  }
  return freezeDeep(selectedFeatures.map((feature, featureIndex) => ({
    featureId: feature.featureId,
    score: clamp01(totals[featureIndex] / measuredTicks, 'Probe özellik skoru'),
  })))
}

export function resolveStructuralSemanticProbeBase(store, snapshot) {
  if (!snapshot || typeof snapshot.id !== 'string' || !snapshot.id ||
      typeof snapshot.inputManifestHash !== 'string' || !snapshot.inputManifestHash) {
    throw producerError('SEMANTIC_PROBE_EXPLICIT_SNAPSHOT_REQUIRED', 'Probe üreticisi açık snapshot ve input manifest bağı ister.')
  }
  const record = readPlasticRegionSnapshot(store, snapshot)
  if (!record.compatible || !record.head || !record.artifact || !record.metadata) {
    return freezeDeep({
      status: 'unknown',
      reason: record.reason ?? 'atlas-unavailable',
      snapshotId: snapshot.id,
      inputManifestHash: snapshot.inputManifestHash,
      atlasRecord: null,
    })
  }
  validateAdaptiveRegionAtlas(record.head)
  return freezeDeep({
    status: 'ready',
    reason: null,
    snapshotId: snapshot.id,
    inputManifestHash: snapshot.inputManifestHash,
    atlasRecord: record,
  })
}

function persistProbeValue(context, value, commonMetadata = {}) {
  const write = semanticProbeArtifactWrite(value)
  return persistArtifact(context, {
    kind: write.kind,
    contentHash: write.contentHash,
    value: write.value,
    metadata: {
      ...write.metadata,
      ...commonMetadata,
      representationSpaceId: value.atlasBinding.representationSpaceId,
      geometryFingerprint: value.atlasBinding.geometryFingerprint,
      baseAtlasFingerprint: value.atlasBinding.atlasFingerprint,
      inputManifestHash: value.producerBinding.inputManifestHash,
      producerFingerprint: STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT,
      probeSpecFingerprint: value.producerBinding.probeSpecFingerprint,
      trainingWritesAllowed: false,
      routingMutationAllowed: false,
      releaseAuthority: false,
    },
  }).artifact
}

function persistProbeSpec(context, value, commonMetadata) {
  validateStructuralSemanticProbeSpec(value)
  assertProbeSpecBound(context.store, value)
  return persistArtifact(context, {
    kind: STRUCTURAL_SEMANTIC_PROBE_SPEC_ARTIFACT_KIND,
    contentHash: value.fingerprint,
    value,
    metadata: {
      ...commonMetadata,
      layer: SEMANTIC_PROBE_LAYER,
      fingerprint: value.fingerprint,
      producerFingerprint: STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT,
      trainingWritesAllowed: false,
      routingMutationAllowed: false,
      releaseAuthority: false,
      usesOllama: false,
    },
  }).artifact
}

function sourceForCase(spec, benchmarkCase) {
  const locator = `deterministic-probe:${spec.fingerprint}:${benchmarkCase.combinationId}`
  return {
    id: benchmarkCase.id,
    sourceType: 'deterministic-diagnostic',
    locator,
    contentHash: stableRuntimeHash(
      'plastic-region-semantic-combination-v1',
      benchmarkCase.combinationId,
      benchmarkCase.expected,
    ),
    provenance: { kind: 'deterministic', canonicalPath: locator },
  }
}

function observationArtifactReference(artifact, value) {
  if (!artifact || artifact.kind !== value.format || artifact.contentHash !== value.fingerprint) {
    throw producerError('SEMANTIC_PROBE_ARTIFACT_BINDING_MISMATCH', 'Kalıcı probe artifact bağı beklenen immutable içerikle uyuşmuyor.')
  }
  return { artifactId: artifact.id, contentHash: artifact.contentHash, fingerprint: value.fingerprint }
}

/**
 * Real producer: executes the frozen deterministic event-graph pack through a
 * caller-supplied live thought runner, reads only shadow gate telemetry, then
 * persists train calibration and isolated validation/test evidence. It writes
 * no empty skeleton and never attaches artifacts to a snapshot automatically.
 */
export async function produceAndPersistStructuralSemanticProbes(input) {
  const { context, snapshot, runCase } = input
  if (!context?.store || typeof runCase !== 'function') {
    throw producerError('SEMANTIC_PROBE_PRODUCER_CONTRACT_MISSING', 'Probe üreticisi store bağlamı ve canlı vaka çalıştırıcısı ister.')
  }
  const base = resolveStructuralSemanticProbeBase(context.store, snapshot)
  if (base.status !== 'ready') {
    return freezeDeep({
      format: 'plastic-region-semantic-probe-production-result',
      version: 1,
      status: 'unknown',
      reason: base.reason,
      artifactIds: [],
      artifactsWritten: 0,
      trainingWritesAllowed: false,
      routingMutationAllowed: false,
    })
  }
  const atlas = base.atlasRecord.head
  const maximumFeatures = Number(input.maximumFeatures ?? 56)
  if (!Number.isSafeInteger(maximumFeatures) || maximumFeatures < 2 || maximumFeatures > 128) {
    throw producerError('SEMANTIC_PROBE_FEATURE_LIMIT_EXCEEDED', 'Producer özellik sınırı 2..128 arasında tam sayı olmalı.')
  }
  const selected = structuralFeatures(atlas, maximumFeatures)
  if (selected.features.length < 2) {
    return freezeDeep({
      format: 'plastic-region-semantic-probe-production-result',
      version: 1,
      status: 'unknown',
      reason: 'insufficient-active-region-features',
      artifactIds: [],
      artifactsWritten: 0,
      activeFeatureCount: selected.features.length,
      trainingWritesAllowed: false,
      routingMutationAllowed: false,
    })
  }
  const spec = createStructuralSemanticProbeSpec()
  assertProbeSpecBound(context.store, spec)
  const storeArtifactLimit = configuredArtifactLimit(context.store)
  const effectiveLimits = {
    ...input.limits,
    maximumArtifactBytes: Math.min(
      Number(input.limits?.maximumArtifactBytes ?? storeArtifactLimit),
      storeArtifactLimit,
    ),
  }
  const cases = createStructuralBenchmark(PRODUCER_SPEC.benchmarkSeed, PRODUCER_SPEC.holdoutRatio)
  const measured = []
  for (let index = 0; index < cases.length; index += 1) {
    await context.checkpoint?.(`plastic-region-semantic-probe:${index + 1}`)
    const benchmarkCase = cases[index]
    const result = await runCase({
      benchmarkCase,
      request: {
        strategy: 'deterministic',
        learningMode: 'off',
        teacherMode: 'off',
        adaptiveRegion: { mode: 'shadow', atlasFingerprint: atlas.fingerprint },
        trainingWritesAllowed: false,
        routingMutationAllowed: false,
      },
      atlas,
    })
    measured.push({
      benchmarkCase,
      scores: extractStructuralSemanticFeatureScores(atlas, result, selected.features),
    })
    await context.progress?.('plasticRegionSemanticProbes', (index + 1) / cases.length)
  }

  const atlasBinding = {
    representationProfile: 'adaptive-eighth-id-v1',
    representationSpaceId: atlas.representationSpaceId,
    geometryFingerprint: atlas.geometryFingerprint,
    atlasArtifactId: base.atlasRecord.artifact.id,
    atlasContentHash: base.atlasRecord.artifact.contentHash,
    atlasFingerprint: atlas.fingerprint,
    atlasRevision: atlas.revision,
  }
  const datasetBinding = {
    datasetId: spec.datasetId,
    partitionManifestFingerprint: spec.partitionManifestFingerprint,
  }
  const producerBinding = {
    mode: 'trusted-deterministic-pack',
    producerFingerprint: STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT,
    inputManifestHash: snapshot.inputManifestHash,
    probeSpecFingerprint: spec.fingerprint,
    usesOllama: false,
  }
  const commonMetadata = {
    baseSnapshotId: snapshot.id,
    atlasArtifactId: base.atlasRecord.artifact.id,
    representationSpaceId: atlas.representationSpaceId,
    geometryFingerprint: ADAPTIVE_REGION_GEOMETRY.fingerprint,
    baseAtlasFingerprint: atlas.fingerprint,
    inputManifestHash: snapshot.inputManifestHash,
  }
  const specArtifact = persistProbeSpec(context, spec, commonMetadata)
  const artifacts = [specArtifact]
  const observationValues = []
  const calibrations = []
  const evaluations = []
  const diagnostics = []

  for (const hypothesisSpec of STRUCTURAL_HYPOTHESIS_SPECS) {
    const hypothesis = hypothesisFromSpec(hypothesisSpec)
    const bySplit = new Map(['train', 'validation', 'test'].map((split) => [split, []]))
    for (const item of measured) {
      bySplit.get(item.benchmarkCase.split).push({
        sampleContentHash: stableRuntimeHash('plastic-region-semantic-sample-v1', item.benchmarkCase.prompt),
        source: sourceForCase(spec, item.benchmarkCase),
        label: labelForCase(hypothesisSpec, item.benchmarkCase),
        labelBinding: {
          authority: 'deterministic',
          artifactId: specArtifact.id,
          contentHash: specArtifact.contentHash,
        },
        scores: item.scores,
      })
    }
    const values = {}
    const valueArtifacts = {}
    for (const split of ['train', 'validation', 'test']) {
      if (!bySplit.get(split).length) continue
      values[split] = createSemanticProbeObservationSet({
        hypothesis,
        atlasBinding,
        datasetBinding,
        producerBinding,
        split,
        features: selected.features,
        records: bySplit.get(split),
        limits: effectiveLimits,
      })
      valueArtifacts[split] = persistProbeValue(context, values[split], {
        ...commonMetadata,
        split,
        hypothesisId: values[split].hypothesis.hypothesisId,
      })
      artifacts.push(valueArtifacts[split])
      observationValues.push(values[split])
    }
    if (!values.train) continue
    let calibration
    try {
      calibration = createSemanticProbeCalibration({
        observations: values.train,
        observationArtifact: observationArtifactReference(valueArtifacts.train, values.train),
        policy: input.calibrationPolicy,
      })
    } catch (error) {
      if (error?.code !== 'SEMANTIC_PROBE_SHAM_MATCH_UNAVAILABLE') throw error
      diagnostics.push({
        hypothesisId: values.train.hypothesis.hypothesisId,
        status: 'unknown',
        reason: 'footprint-matched-sham-unavailable',
      })
      continue
    }
    const calibrationArtifact = persistProbeValue(context, calibration, {
      ...commonMetadata,
      split: 'train',
      status: calibration.status,
      hypothesisId: calibration.hypothesis.hypothesisId,
    })
    artifacts.push(calibrationArtifact)
    calibrations.push(calibration)
    for (const split of ['validation', 'test']) {
      if (!values[split]) continue
      const evidence = createSemanticProbeEvaluationEvidence({
        trainingObservations: values.train,
        evaluationObservations: values[split],
        calibration,
        evaluationArtifact: observationArtifactReference(valueArtifacts[split], values[split]),
        calibrationArtifact: observationArtifactReference(calibrationArtifact, calibration),
        policy: input.evaluationPolicy,
      })
      const evidenceArtifact = persistProbeValue(context, evidence, {
        ...commonMetadata,
        split,
        status: evidence.status,
        hypothesisId: evidence.hypothesis.hypothesisId,
      })
      artifacts.push(evidenceArtifact)
      evaluations.push(evidence)
    }
  }
  const summary = createSemanticProbePublicSummary({
    observations: observationValues,
    calibrations,
    evaluations,
    maximumItems: 64,
  })
  return freezeDeep({
    format: 'plastic-region-semantic-probe-production-result',
    version: 1,
    status: evaluations.some((value) => value.status === 'calibrated')
      ? 'calibrated'
      : calibrations.some((value) => value.status === 'provisional') ? 'provisional' : 'unknown',
    reason: null,
    snapshotId: snapshot.id,
    inputManifestHash: snapshot.inputManifestHash,
    atlasFingerprint: atlas.fingerprint,
    probeSpecFingerprint: spec.fingerprint,
    producerFingerprint: STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT,
    activeFeatureCount: selected.features.length,
    omittedFeatureCount: selected.omittedFeatureCount,
    measuredCaseCount: measured.length,
    artifactsWritten: artifacts.length,
    artifactIds: artifacts.map((artifact) => artifact.id),
    summary,
    diagnostics,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    releaseAuthority: false,
  })
}

/** Historical empty caches are readable for audit only and are never promoted. */
export function inspectLegacyPlasticRegionProbeSkeleton(value) {
  const recognized = value?.format === 'plastic-region-probe-cache' && value?.version === 1 &&
    value?.trainingWritesAllowed === false && Array.isArray(value?.semanticAssignments) &&
    value.semanticAssignments.length === 0 && Array.isArray(value?.packs) &&
    value.packs.every((pack) => pack?.status === 'skeleton' && Array.isArray(pack.probes) &&
      pack.probes.length === 0 && Array.isArray(pack.learnedBindings) && pack.learnedBindings.length === 0)
  return freezeDeep({
    recognized,
    status: recognized ? 'unknown' : 'unsupported',
    auditOnly: true,
    migrationAllowed: false,
    persistAgain: false,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
  })
}
