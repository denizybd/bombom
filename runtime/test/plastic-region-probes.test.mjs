import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer as createViteServer } from 'vite'

async function loadModules(t) {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  return vite.ssrLoadModule('/runtime/plastic-region-probes.mjs')
}

function hypothesis(overrides = {}) {
  return {
    namespace: 'morphology',
    label: 'yönelme hâli',
    schemaVersion: 1,
    definitionFingerprint: 'definition:dative:v1',
    ...overrides,
  }
}

function atlas(regions, overrides = {}) {
  return {
    representationProfile: 'adaptive-eighth-id-v1',
    representationSpaceId: 'RS-ADAPTIVE-PROBES',
    geometryFingerprint: regions.ADAPTIVE_REGION_GEOMETRY.fingerprint,
    atlasArtifactId: 'artifact-atlas-1',
    atlasContentHash: 'content-atlas-1',
    atlasFingerprint: 'atlas-fingerprint-1',
    atlasRevision: 3,
    ...overrides,
  }
}

const dataset = {
  datasetId: 'frozen-diagnostic-pack-v1',
  partitionManifestFingerprint: 'partition-manifest-fingerprint-v1',
}

const features = [
  {
    featureId: 'opaque-region-a',
    responseKind: 'region-gate',
    footprint: [{ pageIndex: 1, weight: 0.5 }, { pageIndex: 2, weight: 0.5 }],
  },
  {
    featureId: 'opaque-region-b',
    responseKind: 'region-gate',
    footprint: [{ pageIndex: 2, weight: 0.5 }, { pageIndex: 1, weight: 0.5 }],
  },
]

function record(index, split, options = {}) {
  const label = options.label ?? index % 2
  const targetScore = label ? 0.95 : 0.05
  return {
    sampleContentHash: options.sampleContentHash ?? `${split}-sample-${index}`,
    source: {
      id: options.sourceId ?? `${split}-source-${index}`,
      sourceType: options.sourceType ?? 'manual-text',
      locator: options.locator ?? `/trusted/${split}/book-${index}.txt`,
      contentHash: options.contentHash ?? `${split}-book-content-${index}`,
      provenance: {
        kind: options.provenanceKind ?? 'human',
        canonicalPath: options.locator ?? `/trusted/${split}/book-${index}.txt`,
        independenceGroupId: options.forgedIndependence ?? `FORGED-${index}`,
        model: options.model,
      },
    },
    label,
    labelBinding: {
      authority: options.labelAuthority ?? 'deterministic',
      artifactId: options.labelArtifactId ?? `label-artifact-${split}-${index}`,
      contentHash: options.labelContentHash ?? 'probe-spec-v1',
    },
    scores: [
      { featureId: 'opaque-region-b', score: options.shamScore ?? 0.5 },
      { featureId: 'opaque-region-a', score: options.targetScore ?? targetScore },
    ],
  }
}

function observations(probes, regions, split, count = 16, overrides = {}) {
  return probes.createSemanticProbeObservationSet({
    hypothesis: overrides.hypothesis ?? hypothesis(),
    atlasBinding: overrides.atlasBinding ?? atlas(regions),
    datasetBinding: overrides.datasetBinding ?? dataset,
    producerBinding: overrides.producerBinding ?? {
      mode: 'trusted-deterministic-pack',
      producerFingerprint: 'trusted-producer-v1',
      inputManifestHash: 'input-manifest-v1',
      probeSpecFingerprint: 'probe-spec-v1',
      usesOllama: false,
    },
    split,
    features: overrides.features ?? [...features].reverse(),
    records: overrides.records ?? Array.from({ length: count }, (_, index) => record(index, split)),
    limits: overrides.limits,
  })
}

function reference(value, prefix) {
  return { artifactId: `${prefix}-artifact`, contentHash: value.fingerprint, fingerprint: value.fingerprint }
}

test('NEW probe observations are immutable, deterministic, bounded, and ignore forged independence ids', async (t) => {
  const probes = await loadModules(t)
  // The probe module exports the exact geometry through its atlas validation;
  // load the adaptive constants only to create a valid external binding.
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  const firstRecord = record(0, 'train', { forgedIndependence: 'fake-a' })
  const duplicate = record(0, 'train', { forgedIndependence: 'fake-b' })
  const copiedAtAnotherPath = record(0, 'train', {
    forgedIndependence: 'fake-copy',
    locator: '/trusted/copy/of-book-0.txt',
    sourceId: 'copied-source-id',
  })
  const forward = observations(probes, adaptive, 'train', 0, {
    records: [firstRecord, duplicate, copiedAtAnotherPath, record(1, 'train')],
  })
  const reverse = observations(probes, adaptive, 'train', 0, {
    records: [record(1, 'train'), copiedAtAnotherPath, duplicate, firstRecord],
  })

  assert.equal(forward.fingerprint, reverse.fingerprint)
  assert.equal(forward.observationCount, 2)
  assert.equal(forward.trainingWritesAllowed, false)
  assert.equal(forward.routingMutationAllowed, false)
  assert.equal(forward.atlasMutationAllowed, false)
  assert.equal(forward.layer, 'new-opt-in-interpretation')
  assert.ok(Object.isFrozen(forward))
  assert.ok(Object.isFrozen(forward.records[0].scores))
  assert.doesNotThrow(() => probes.validateSemanticProbeObservationSet(forward))
  assert.equal(forward.lineages.some((lineage) => JSON.stringify(lineage).includes('fake-a')), false)
  assert.equal(forward.lineages.some((lineage) => JSON.stringify(lineage).includes('fake-b')), false)
  assert.equal(forward.lineages.some((lineage) => JSON.stringify(lineage).includes('fake-copy')), false)

  const otherAtlas = observations(probes, adaptive, 'train', 2, {
    atlasBinding: atlas(adaptive, {
      atlasArtifactId: 'artifact-atlas-2',
      atlasContentHash: 'content-atlas-2',
      atlasFingerprint: 'atlas-fingerprint-2',
      atlasRevision: 4,
    }),
  })
  const multiAtlasSummary = probes.createSemanticProbePublicSummary({ observations: [forward, otherAtlas] })
  assert.equal(multiAtlasSummary.itemCount, 2, 'the same external hypothesis must not merge across atlas lineages')

  const idOnlyLineage = record(2, 'train')
  delete idOnlyLineage.source.locator
  delete idOnlyLineage.source.contentHash
  delete idOnlyLineage.source.provenance.canonicalPath
  assert.throws(
    () => observations(probes, adaptive, 'train', 0, { records: [idOnlyLineage] }),
    (error) => error?.code === 'SEMANTIC_PROBE_PHYSICAL_LINEAGE_REQUIRED',
  )

  const conflict = record(0, 'train', { forgedIndependence: 'fake-c', targetScore: 0.8 })
  assert.throws(
    () => observations(probes, adaptive, 'train', 0, { records: [firstRecord, conflict] }),
    (error) => error?.code === 'SEMANTIC_PROBE_DUPLICATE_CONFLICT',
  )
})

test('external hypothesis namespaces cannot impersonate atlas, region, neuron, or family internals', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  for (const namespace of ['region', 'region.foo', 'atlas-label', 'neuron_private', 'family/test']) {
    assert.throws(
      () => observations(probes, adaptive, 'train', 2, { hypothesis: hypothesis({ namespace }) }),
      (error) => error?.code === 'INVALID_SEMANTIC_PROBE',
    )
  }
  assert.doesNotThrow(() => observations(probes, adaptive, 'train', 2, {
    hypothesis: hypothesis({ namespace: 'mathematics.arithmetic' }),
  }))
})

test('train calibration selects a label-correlated target and a label-blind footprint-matched sham', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  const train = observations(probes, adaptive, 'train', 16)
  const calibration = probes.createSemanticProbeCalibration({
    observations: train,
    observationArtifact: reference(train, 'train'),
  })

  assert.equal(calibration.status, 'provisional')
  assert.equal(calibration.target.feature.featureId, 'opaque-region-a')
  assert.equal(calibration.sham.feature.featureId, 'opaque-region-b')
  assert.equal(calibration.sham.footprintDistance, 0)
  assert.ok(calibration.target.trainingBrier < calibration.sham.trainingBrier)
  assert.equal(calibration.selection.shamSelectedWithoutLabels, true)
  assert.equal(calibration.trainingWritesAllowed, false)
  assert.doesNotThrow(() => probes.validateSemanticProbeCalibration(calibration, train))

  const write = probes.semanticProbeArtifactWrite(calibration)
  assert.equal(write.kind, probes.SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND)
  assert.equal(write.contentHash, calibration.fingerprint)
  assert.equal(write.metadata.releaseAuthority, false)
  assert.equal(write.metadata.routingMutationAllowed, false)
})

test('frozen validation evidence reports calibrated Brier/ECE and target-vs-sham bootstrap confidence', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  const train = observations(probes, adaptive, 'train', 16)
  const validation = observations(probes, adaptive, 'validation', 12)
  const calibration = probes.createSemanticProbeCalibration({
    observations: train,
    observationArtifact: reference(train, 'train'),
  })
  const evidence = probes.createSemanticProbeEvaluationEvidence({
    trainingObservations: train,
    evaluationObservations: validation,
    calibration,
    evaluationArtifact: reference(validation, 'validation'),
    calibrationArtifact: reference(calibration, 'calibration'),
  })

  assert.equal(evidence.status, 'calibrated')
  assert.equal(evidence.releaseAuthority, false)
  assert.ok(evidence.metrics.target.brier < evidence.metrics.baseline.brier)
  assert.ok(evidence.metrics.target.brier < evidence.metrics.sham.brier)
  assert.ok(evidence.metrics.target.ece <= 0.15)
  assert.ok(evidence.metrics.gainVsBaseline.low >= 0.01)
  assert.ok(evidence.metrics.gainVsSham.low >= 0.01)
  assert.equal(evidence.metrics.gainVsSham.bootstrapSamples, 1_000)
  assert.deepEqual(probes.replaySemanticProbeEvaluation(evidence, {
    trainingObservations: train,
    evaluationObservations: validation,
    calibration,
  }), {
    matched: true,
    fingerprint: evidence.fingerprint,
    replayBinding: evidence.replayBinding,
  })
  assert.doesNotThrow(() => probes.validateSemanticProbeEvaluationEvidence(evidence, {
    trainingObservations: train,
    evaluationObservations: validation,
    calibration,
  }))

  const summary = probes.createSemanticProbePublicSummary({
    observations: [train], calibrations: [calibration], evaluations: [evidence], maximumItems: 1,
  })
  const summaryReplay = probes.createSemanticProbePublicSummary({
    evaluations: [evidence], calibrations: [calibration], observations: [train], maximumItems: 1,
  })
  assert.equal(summaryReplay.fingerprint, summary.fingerprint)
  assert.equal(summary.items[0].status, 'calibrated')
  assert.equal(summary.trainingWritesAllowed, false)
  assert.equal(summary.routingMutationAllowed, false)
  assert.equal(summary.omittedCount, 0)
  assert.equal('records' in summary.items[0], false)
  assert.equal('lineages' in summary.items[0], false)
})

test('split leakage and Ollama-generated or Ollama-labeled validation/test records fail closed', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  const train = observations(probes, adaptive, 'train', 8)
  const calibration = probes.createSemanticProbeCalibration({
    observations: train,
    observationArtifact: reference(train, 'train'),
  })
  const leaked = observations(probes, adaptive, 'validation', 0, {
    records: [record(0, 'validation', {
      locator: '/trusted/train/book-0.txt',
      contentHash: 'train-book-content-0',
      sampleContentHash: 'train-sample-0',
    }), ...Array.from({ length: 7 }, (_, index) => record(index + 1, 'validation'))],
  })
  assert.throws(
    () => probes.createSemanticProbeEvaluationEvidence({
      trainingObservations: train,
      evaluationObservations: leaked,
      calibration,
      evaluationArtifact: reference(leaked, 'leaked'),
      calibrationArtifact: reference(calibration, 'calibration'),
    }),
    (error) => error?.code === 'SEMANTIC_PROBE_SPLIT_LEAKAGE',
  )

  assert.throws(
    () => observations(probes, adaptive, 'validation', 0, { records: [record(0, 'validation', {
      sourceType: 'ollama-generated', provenanceKind: 'ollama', model: 'qwen14b',
    })] }),
    (error) => error?.code === 'OLLAMA_SEMANTIC_EVALUATION_FORBIDDEN',
  )
  assert.throws(
    () => observations(probes, adaptive, 'test', 0, { records: [record(0, 'test', {
      labelAuthority: 'ollama',
    })] }),
    (error) => error?.code === 'OLLAMA_SEMANTIC_EVALUATION_FORBIDDEN',
  )
  const validation = observations(probes, adaptive, 'validation', 8)
  assert.throws(
    () => probes.createSemanticProbeCalibration({
      observations: validation,
      observationArtifact: reference(validation, 'validation'),
    }),
    (error) => error?.code === 'SEMANTIC_PROBE_TRAIN_SPLIT_REQUIRED',
  )
})

test('partition, atlas, feature, and replay bindings are exact and tampering is rejected', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  const train = observations(probes, adaptive, 'train', 12)
  const calibration = probes.createSemanticProbeCalibration({ observations: train, observationArtifact: reference(train, 'train') })
  const validation = observations(probes, adaptive, 'validation', 12)
  const evidence = probes.createSemanticProbeEvaluationEvidence({
    trainingObservations: train,
    evaluationObservations: validation,
    calibration,
    evaluationArtifact: reference(validation, 'validation'),
    calibrationArtifact: reference(calibration, 'calibration'),
  })
  const wrongPartition = observations(probes, adaptive, 'validation', 12, {
    datasetBinding: { ...dataset, partitionManifestFingerprint: 'other-partition' },
  })
  assert.throws(
    () => probes.createSemanticProbeEvaluationEvidence({
      trainingObservations: train,
      evaluationObservations: wrongPartition,
      calibration,
      evaluationArtifact: reference(wrongPartition, 'validation'),
      calibrationArtifact: reference(calibration, 'calibration'),
    }),
    (error) => error?.code === 'SEMANTIC_PROBE_BINDING_MISMATCH',
  )

  const tampered = structuredClone(train)
  tampered.records[0].scores[0] = 0.123
  assert.throws(
    () => probes.validateSemanticProbeObservationSet(tampered),
    (error) => error?.code === 'INVALID_SEMANTIC_PROBE',
  )
  const forgedCalibrationStatus = structuredClone(calibration)
  forgedCalibrationStatus.status = 'unknown'
  assert.throws(
    () => probes.validateSemanticProbeCalibration(forgedCalibrationStatus),
    (error) => error?.code === 'INVALID_SEMANTIC_PROBE',
  )
  const forgedEvaluationClaim = structuredClone(evidence)
  forgedEvaluationClaim.calibrationStatus = 'unknown'
  assert.throws(
    () => probes.validateSemanticProbeEvaluationEvidence(forgedEvaluationClaim),
    (error) => error?.code === 'INVALID_SEMANTIC_PROBE',
  )
  const forgedReplayBinding = structuredClone(evidence)
  forgedReplayBinding.replayBinding.inputManifestHash = 'forged-manifest'
  assert.throws(
    () => probes.validateSemanticProbeEvaluationEvidence(forgedReplayBinding),
    (error) => error?.code === 'INVALID_SEMANTIC_PROBE',
  )
  const invalidProfile = { ...atlas(adaptive), representationProfile: 'legacy-quarter-id-v2' }
  assert.throws(
    () => observations(probes, adaptive, 'train', 2, { atlasBinding: invalidProfile }),
    (error) => error?.code === 'SEMANTIC_PROBE_REPRESENTATION_MISMATCH',
  )
})

test('record, feature, footprint, and encoded byte bounds fail before persistence', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  assert.throws(
    () => observations(probes, adaptive, 'train', 3, { limits: { maximumRecords: 2 } }),
    (error) => error?.code === 'SEMANTIC_PROBE_RECORD_LIMIT_EXCEEDED',
  )
  assert.throws(
    () => observations(probes, adaptive, 'train', 2, { limits: { maximumFeatures: 2 }, features: [
      ...features,
      { featureId: 'extra', responseKind: 'region-gate', footprint: [{ pageIndex: 3, weight: 1 }] },
    ] }),
    (error) => error?.code === 'SEMANTIC_PROBE_FEATURE_LIMIT_EXCEEDED',
  )
  assert.throws(
    () => observations(probes, adaptive, 'train', 2, { features: [
      features[0],
      { ...features[1], footprint: Array.from({ length: 57 }, (_, pageIndex) => ({ pageIndex, weight: 1 })) },
    ] }),
    (error) => error?.code === 'INVALID_SEMANTIC_PROBE',
  )
  assert.throws(
    () => observations(probes, adaptive, 'train', 8, { limits: { maximumArtifactBytes: 512 } }),
    (error) => error?.code === 'SEMANTIC_PROBE_ARTIFACT_LIMIT_EXCEEDED',
  )
})

test('insufficient support remains honestly unknown and never emits a calibrated claim', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  const train = observations(probes, adaptive, 'train', 2)
  const calibration = probes.createSemanticProbeCalibration({ observations: train, observationArtifact: reference(train, 'train') })
  assert.equal(calibration.status, 'unknown')
  const summary = probes.createSemanticProbePublicSummary({ observations: [train], calibrations: [calibration] })
  assert.equal(summary.items[0].status, 'unknown')
})

test('producer Ollama declaration must match server-derived source and label lineage', async (t) => {
  const probes = await loadModules(t)
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const adaptive = await vite.ssrLoadModule('/src/adaptive-regions/index.ts')
  const ollamaRecord = record(0, 'train', {
    sourceType: 'ollama-generated',
    provenanceKind: 'ollama',
    model: 'qwen14b',
  })
  assert.throws(
    () => observations(probes, adaptive, 'train', 0, { records: [ollamaRecord] }),
    (error) => error?.code === 'SEMANTIC_PROBE_PRODUCER_BINDING_MISMATCH',
  )
  assert.doesNotThrow(() => observations(probes, adaptive, 'train', 0, {
    records: [ollamaRecord],
    producerBinding: {
      mode: 'audited-external-labels',
      producerFingerprint: 'audited-ollama-train-producer',
      inputManifestHash: 'ollama-train-manifest',
      probeSpecFingerprint: 'audited-label-spec',
      usesOllama: true,
    },
  }))
  const missingModelLineage = structuredClone(ollamaRecord)
  delete missingModelLineage.source.provenance.model
  assert.throws(
    () => observations(probes, adaptive, 'train', 0, {
      records: [missingModelLineage],
      producerBinding: {
        mode: 'audited-external-labels',
        producerFingerprint: 'audited-ollama-train-producer',
        inputManifestHash: 'ollama-train-manifest',
        probeSpecFingerprint: 'audited-label-spec',
        usesOllama: true,
      },
    }),
    (error) => error?.code === 'SEMANTIC_PROBE_OLLAMA_LINEAGE_REQUIRED',
  )
})
