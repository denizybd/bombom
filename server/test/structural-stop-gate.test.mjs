import test from 'node:test'
import assert from 'node:assert/strict'
import {
  evaluateContinuousResourceBudget,
  evaluateIndependentStructuralBenchmark,
  evaluateIndependentStructuralBenchmarkEarlyFailure,
  evaluateIndependentStructuralBenchmarkScreen,
  evaluateIndependentStructuralEvidence,
  INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND,
  INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD,
  INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_FORMAT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_POLICY_FINGERPRINT,
  readIndependentStructuralBenchmark,
} from '../structural-stop-gate.mjs'

function passingReport(overrides = {}) {
  const ci = (mean, low, high, sourceGroupCount = 45) => ({
    mean, low, high, confidence: 0.95, seedCount: 5, sourceGroupCount, samples: 2_000,
  })
  return {
    format: 'bombom-independent-structural-benchmark',
    version: 1,
    benchmarkId: 'controlled-event-graph-144-v1',
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    origin: 'independent-deterministic',
    deterministic: true,
    usesOllama: false,
    snapshotId: 'snapshot-1',
    inputManifestHash: 'manifest-1',
    evidence: {
      combinationCount: 144,
      surfacesPerCombination: 2,
      splitPolicy: 'combination-before-surface-v1',
      holdoutRatio: 0.31,
      holdoutCombinationCount: 45,
      seedCount: 5,
      sourceGroupCount: 45,
      bootstrapConfidence: 0.95,
      bootstrapSamples: 2_000,
      intervals: {
        holdoutGraphF1: ci(0.81, 0.80, 0.82),
        h0OnlyGraphF1: ci(0.62, 0.61, 0.63),
        noRecurrenceGraphF1: ci(0.61, 0.60, 0.62),
        shuffledEdgeGraphF1: ci(0.60, 0.59, 0.61),
        baselineAdvantage: ci(0.19, 0.17, 0.21),
        targetFamilyAblationDrop: ci(0.12, 0.11, 0.13),
        targetEdgeAblationDrop: ci(0.11, 0.105, 0.12),
        shamFamilyAblationDrop: ci(0.02, 0.015, 0.025),
        shamEdgeAblationDrop: ci(0.02, 0.015, 0.025),
        perturbationRecoveryRate: ci(0.74, 0.72, 0.76),
        knownStructureFalseBirthRate: ci(0.04, 0.03, 0.05),
        repeatedNoveltyBirthPrecision: ci(0.84, 0.81, 0.87, 10),
        repeatedNoveltyBirthSensitivity: ci(0.84, 0.81, 0.87, 5),
      },
      ablationTargetSelection: 'snapshot-model-only-no-evaluation-data-v2',
      edgeShamProtocol: 'snapshot-digest-random-nontarget-edge-v1',
      targetEdge: 'target-a\u0000target-b',
      shamEdge: 'sham-a\u0000sham-b',
      familyShamProtocol: 'deterministic-model-only-exact-reference-footprint-mask-v1',
      targetFamilyFootprint: { memberships: 2, familyTransitions: 1, utilityReports: 0, coalitions: 0, coalitionActivations: 0, coalitionBindings: 0, birthMemberships: 0, total: 3 },
      shamFamilyFootprint: { memberships: 2, familyTransitions: 1, utilityReports: 0, coalitions: 0, coalitionActivations: 0, coalitionBindings: 0, birthMemberships: 0, total: 3 },
      shamFamilyFootprintMatched: true,
      shamFamilyMaskDigest: 'a'.repeat(64),
      noRecurrenceProtocol: 'full-tick-zero-persistent-transition-evidence-v2',
      birthPrecisionDefinition: 'TP/(TP+FP)',
      birthNegativeSource: 'frozen-train-calibration-repeated-four-observations-v2',
      eligibleNoveltyCaseCount: 5,
      knownBirthPrecisionNegativeCaseCount: 5,
      birthConfusion: { truePositive: 21, falsePositive: 4, falseNegative: 4, trueNegative: 21 },
    },
    evaluationPolicy: {
      sourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
      validationOllamaExampleCount: 0,
      testOllamaExampleCount: 0,
      splitFrozenBeforeTraining: true,
    },
    metrics: {
      holdoutGraphF1: 0.81,
      h0OnlyGraphF1: 0.62,
      noRecurrenceGraphF1: 0.61,
      shuffledEdgeGraphF1: 0.60,
      targetFamilyAblationDrop: 0.12,
      targetEdgeAblationDrop: 0.11,
      shamFamilyAblationDrop: 0.02,
      shamEdgeAblationDrop: 0.02,
      perturbationRecoveryRate: 0.74,
      knownStructureFalseBirthRate: 0.04,
      repeatedNoveltyBirthPrecision: 0.84,
      repeatedNoveltyBirthSensitivity: 0.84,
    },
    ...overrides,
  }
}

function earlyBase(format) {
  return {
    format,
    version: 1,
    benchmarkId: 'controlled-event-graph-144-v1',
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    runnerVersion: 1,
    runnerFingerprint: 'b'.repeat(64),
    origin: 'independent-deterministic',
    deterministic: true,
    usesOllama: false,
    snapshotId: 'snapshot-1',
    inputManifestHash: 'manifest-1',
    evaluationPolicy: {
      sourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
      validationOllamaExampleCount: 0,
      testOllamaExampleCount: 0,
      splitFrozenBeforeTraining: true,
    },
  }
}

function earlyEvidence(seedCount) {
  return {
    combinationCount: 144,
    surfacesPerCombination: 2,
    splitPolicy: 'combination-before-surface-v1',
    holdoutRatio: 44 / 144,
    holdoutCombinationCount: 44,
    seedCount,
    sourceGroupCount: 44,
    episodeCount: 44 * 2 * seedCount,
    maximumTicks: 64,
    observationDigest: 'c'.repeat(64),
    observationChunkDigests: Array.from({ length: seedCount }, (_, index) => index.toString(16).padStart(64, '0')),
  }
}

function screenedReport(meanGraphF1 = 0.59) {
  return {
    ...earlyBase(INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_FORMAT),
    screenPolicyFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_POLICY_FINGERPRINT,
    completion: 'diagnostic-screened-out',
    diagnostic: {
      variant: 'full',
      seed: 123,
      threshold: INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD,
      meanGraphF1,
    },
    evidence: {
      ...earlyEvidence(1),
      skippedVariants: ['full-seeds-2-5'],
      skippedEpisodeCount: 4_072,
    },
  }
}

function decisiveEarlyFailure(low = 0.70) {
  return {
    ...earlyBase(INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT),
    earlyFailurePolicyFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT,
    completion: 'decisive-early-failure',
    decisiveCheck: {
      id: 'holdout-graph-f1',
      threshold: { direction: 'minimum', boundary: 'ci.low', value: 0.75 },
      interval: { mean: 0.80, low, high: 0.86, confidence: 0.95, seedCount: 5, sourceGroupCount: 44, samples: 2_000 },
      passed: false,
    },
    evidence: {
      ...earlyEvidence(5),
      completedVariant: 'full',
      bootstrapConfidence: 0.95,
      bootstrapSamples: 2_000,
      skippedVariants: Array.from({ length: 9 }, (_, index) => `variant-${index}`),
      skippedEpisodeCount: 3_720,
    },
  }
}

test('one-seed admission screen is valid evidence but can never pass the stop gate', () => {
  const report = screenedReport()
  const result = evaluateIndependentStructuralEvidence(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.status, 'screened-out')
  assert.equal(result.valid, true)
  assert.equal(result.conclusive, false)
  assert.equal(result.passed, false)

  const aboveThreshold = evaluateIndependentStructuralBenchmarkScreen(screenedReport(0.61), {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(aboveThreshold.status, 'invalid')
  assert.equal(aboveThreshold.passed, false)
})

test('complete full-variant CI below the gate is a conclusive failure and never a partial pass', () => {
  const report = decisiveEarlyFailure(0.74)
  const result = evaluateIndependentStructuralBenchmarkEarlyFailure(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.status, 'failed-early')
  assert.equal(result.valid, true)
  assert.equal(result.conclusive, true)
  assert.equal(result.passed, false)
  assert.equal(result.holdoutGraphF1, 0.8)

  const boundaryNotFailed = evaluateIndependentStructuralBenchmarkEarlyFailure(decisiveEarlyFailure(0.75), {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(boundaryNotFailed.status, 'invalid')
  assert.equal(boundaryNotFailed.conclusive, false)
  assert.equal(boundaryNotFailed.passed, false)
})

test('wrong snapshot, manifest, or decision policy invalidates terminal shortcut evidence', () => {
  const wrongManifest = decisiveEarlyFailure()
  wrongManifest.inputManifestHash = 'another-manifest'
  const wrongPolicy = decisiveEarlyFailure()
  wrongPolicy.earlyFailurePolicyFingerprint = 'd'.repeat(64)
  for (const report of [wrongManifest, wrongPolicy]) {
    const result = evaluateIndependentStructuralEvidence(report, {
      expectedSnapshotId: 'snapshot-1',
      expectedInputManifestHash: 'manifest-1',
    })
    assert.equal(result.status, 'invalid')
    assert.equal(result.valid, false)
    assert.equal(result.passed, false)
  }
})

test('independent structural report recomputes every frozen gate and ignores a supplied verdict', () => {
  const report = passingReport({ passed: false })
  const first = evaluateIndependentStructuralBenchmark(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  const replay = evaluateIndependentStructuralBenchmark(structuredClone(report), {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.deepEqual(first, replay)
  assert.equal(first.passed, true)
  assert.equal(first.status, 'passed')
  assert.ok(Math.abs(first.baselineAdvantage - 0.19) < 1e-12)
})

test('Ollama evaluation, synthetic holdout, weak ablation, or wrong snapshot can never stop training', () => {
  const cases = [
    passingReport({ usesOllama: true, passed: true }),
    passingReport({ evaluationPolicy: { ...passingReport().evaluationPolicy, validationOllamaExampleCount: 1 }, passed: true }),
    passingReport({ metrics: { ...passingReport().metrics, targetEdgeAblationDrop: 0.09 }, passed: true }),
    passingReport({ snapshotId: 'another-snapshot', passed: true }),
  ]
  for (const report of cases) {
    const result = evaluateIndependentStructuralBenchmark(report, {
      expectedSnapshotId: 'snapshot-1',
      expectedInputManifestHash: 'manifest-1',
    })
    assert.equal(result.passed, false)
    assert.equal(result.status, 'failed')
  }
})

test('point metrics without paired source-group confidence intervals can never pass', () => {
  const report = passingReport()
  delete report.evidence.intervals
  const result = evaluateIndependentStructuralBenchmark(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.passed, false)
  assert.equal(result.checks.find((check) => check.id === 'baseline-advantage').passed, false)
})

test('novelty recall cannot masquerade as birth precision without known negative confusion evidence', () => {
  const report = passingReport()
  report.evidence.birthConfusion = { truePositive: 21, falsePositive: 10, falseNegative: 4, trueNegative: 15 }
  const result = evaluateIndependentStructuralBenchmark(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.passed, false)
  assert.equal(result.checks.find((check) => check.id === 'birth-precision-recomputed').passed, false)
})

test('perfect birth precision cannot pass when repeated-novelty sensitivity CI is below 0.80', () => {
  const report = passingReport()
  report.evidence.birthConfusion = { truePositive: 5, falsePositive: 0, falseNegative: 20, trueNegative: 25 }
  report.metrics.repeatedNoveltyBirthPrecision = 1
  report.metrics.repeatedNoveltyBirthSensitivity = 0.2
  report.evidence.intervals.repeatedNoveltyBirthPrecision = {
    mean: 1, low: 1, high: 1,
    confidence: 0.95, seedCount: 5, sourceGroupCount: 10, samples: 2_000,
  }
  report.evidence.intervals.repeatedNoveltyBirthSensitivity = {
    mean: 0.2, low: 0.1, high: 0.3,
    confidence: 0.95, seedCount: 5, sourceGroupCount: 5, samples: 2_000,
  }
  const result = evaluateIndependentStructuralBenchmark(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.checks.find((check) => check.id === 'birth-precision-recomputed').passed, true)
  assert.equal(result.checks.find((check) => check.id === 'birth-sensitivity-evidence').passed, true)
  assert.equal(result.checks.find((check) => check.id === 'novelty-birth-precision').passed, true)
  assert.equal(result.checks.find((check) => check.id === 'novelty-birth-sensitivity').passed, false)
  assert.equal(result.passed, false)
})

test('signed negative target ablation is preserved and cannot satisfy the target gate', () => {
  const report = passingReport()
  report.metrics.targetEdgeAblationDrop = -0.2
  report.evidence.intervals.targetEdgeAblationDrop = {
    mean: -0.2, low: -0.22, high: -0.18,
    confidence: 0.95, seedCount: 5, sourceGroupCount: 45, samples: 2_000,
  }
  const result = evaluateIndependentStructuralBenchmark(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.checks.find((check) => check.id === 'metric-interval-means').passed, true)
  assert.equal(result.checks.find((check) => check.id === 'target-edge-ablation').passed, false)
  assert.equal(result.checks.find((check) => check.id === 'target-edge-ablation').actual, -0.22)
  assert.equal(result.passed, false)
})

test('signed negative sham effect remains negative and satisfies only the sham upper bound', () => {
  const report = passingReport()
  report.metrics.shamEdgeAblationDrop = -0.05
  report.evidence.intervals.shamEdgeAblationDrop = {
    mean: -0.05, low: -0.07, high: -0.03,
    confidence: 0.95, seedCount: 5, sourceGroupCount: 45, samples: 2_000,
  }
  const result = evaluateIndependentStructuralBenchmark(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.checks.find((check) => check.id === 'metric-interval-means').passed, true)
  assert.equal(result.checks.find((check) => check.id === 'sham-edge-ablation').passed, true)
  assert.equal(result.checks.find((check) => check.id === 'sham-edge-ablation').actual, -0.03)
  assert.equal(result.passed, true)
})

test('family sham must prove an exact reference-kind footprint match', () => {
  const report = passingReport()
  report.evidence.shamFamilyFootprint.memberships = 1
  report.evidence.shamFamilyFootprint.total = 2
  const result = evaluateIndependentStructuralBenchmark(report, {
    expectedSnapshotId: 'snapshot-1',
    expectedInputManifestHash: 'manifest-1',
  })
  assert.equal(result.checks.find((check) => check.id === 'family-sham-footprint').passed, false)
  assert.equal(result.passed, false)
})

test('default adapter reads only the exact snapshot attached immutable benchmark artifact', () => {
  const artifacts = new Map([
    ['ordinary', { id: 'ordinary', kind: 'runtime-model-state', payload: passingReport(), createdAt: '2026-01-01' }],
    ['benchmark', { id: 'benchmark', kind: INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND, payload: passingReport(), createdAt: '2026-01-02' }],
  ])
  const store = {
    getSnapshot(id) {
      return id === 'snapshot-1'
        ? { id, inputManifestHash: 'manifest-1', artifactIds: ['ordinary', 'benchmark'] }
        : undefined
    },
    getArtifact(id) { return structuredClone(artifacts.get(id)) },
  }
  const result = readIndependentStructuralBenchmark(store, 'snapshot-1')
  assert.equal(result.passed, true)
  assert.equal(result.artifactId, 'benchmark')

  artifacts.delete('benchmark')
  const missing = readIndependentStructuralBenchmark(store, 'snapshot-1')
  assert.equal(missing.status, 'missing')
  assert.equal(missing.passed, false)
})

test('continuous resource guard pauses before low logical disk headroom or source overflow', () => {
  const plan = { minimumDiskHeadroomBytes: 100, maximumSyntheticSources: 3, blockAccepted: 2 }
  const config = { budgets: { diskMb: 1 } }
  const store = {
    approximateStorageBytes: () => 800,
    getEffectiveDiskBudgetBytes: () => 1_000,
    listSources: () => [{ sourceType: 'ollama-generated' }, { sourceType: 'local-upload' }],
  }
  assert.equal(evaluateContinuousResourceBudget({ store, config, plan }).passed, true)

  store.approximateStorageBytes = () => 950
  assert.equal(evaluateContinuousResourceBudget({ store, config, plan }).code, 'CONTINUOUS_DISK_HEADROOM_LOW')
  store.approximateStorageBytes = () => 800
  store.listSources = () => Array.from({ length: 2 }, () => ({ sourceType: 'ollama-generated' }))
  assert.equal(evaluateContinuousResourceBudget({ store, config, plan }).code, 'CONTINUOUS_SOURCE_LIMIT_REACHED')
})
