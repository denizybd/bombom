import { createHash } from 'node:crypto'

export const INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND = 'independent-structural-benchmark-v1'
export const INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_ARTIFACT_KIND = 'independent-structural-benchmark-screen-v1'
export const INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_ARTIFACT_KIND = 'independent-structural-benchmark-early-fail-v1'
export const INDEPENDENT_STRUCTURAL_BENCHMARK_FORMAT = 'bombom-independent-structural-benchmark'
export const INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_FORMAT = 'bombom-independent-structural-benchmark-screen'
export const INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT = 'bombom-independent-structural-benchmark-early-failure'
export const INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION = 1
export const INDEPENDENT_STRUCTURAL_BENCHMARK_ID = 'controlled-event-graph-144-v1'
export const INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD = 0.60
export const INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD = 0.75

const GATE_THRESHOLDS = Object.freeze({
  holdoutGraphF1: Object.freeze({ direction: 'minimum', value: 0.75 }),
  baselineAdvantage: Object.freeze({ direction: 'minimum', value: 0.15 }),
  targetFamilyAblationDrop: Object.freeze({ direction: 'minimum', value: 0.10 }),
  targetEdgeAblationDrop: Object.freeze({ direction: 'minimum', value: 0.10 }),
  shamFamilyAblationDrop: Object.freeze({ direction: 'maximum', value: 0.03 }),
  shamEdgeAblationDrop: Object.freeze({ direction: 'maximum', value: 0.03 }),
  perturbationRecoveryRate: Object.freeze({ direction: 'minimum', value: 0.70 }),
  knownStructureFalseBirthRate: Object.freeze({ direction: 'maximum', value: 0.05 }),
  repeatedNoveltyBirthPrecision: Object.freeze({ direction: 'minimum', value: 0.80 }),
  repeatedNoveltyBirthSensitivity: Object.freeze({ direction: 'minimum', value: 0.80 }),
})

const BENCHMARK_SPEC = Object.freeze({
  version: INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION,
  benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
  combinations: Object.freeze({ actions: 3, actors: 4, patients: 4, instruments: 3, total: 144 }),
  surfacesPerCombination: 2,
  splitPolicy: 'combination-before-surface-v1',
  minimumHoldoutRatio: 0.30,
  independentSeeds: 5,
  bootstrapConfidence: 0.95,
  minimumBootstrapSamples: 1_000,
  confidenceDecision: 'paired-source-group-ci-boundary-v1',
  ablationIntervals: 'paired-signed-source-group-ci-v2',
  ablationTargetSelection: 'snapshot-model-only-no-evaluation-data-v2',
  recurrenceOff: 'full-tick-zero-persistent-transition-evidence-v2',
  birthPrecision: Object.freeze({
    definition: 'TP/(TP+FP)',
    positives: 'repeated-novelty-four-observations-v2',
    negatives: 'frozen-train-calibration-repeated-four-observations-v2',
  }),
  edgeSham: 'snapshot-digest-random-nontarget-edge-v1',
  familySham: 'deterministic-model-only-exact-reference-footprint-mask-v1',
  sourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
  thresholds: GATE_THRESHOLDS,
})

const stableValue = (value) => Array.isArray(value)
  ? value.map(stableValue)
  : value && typeof value === 'object'
    ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]))
    : value

const stableHash = (value) => createHash('sha256').update(JSON.stringify(stableValue(value))).digest('hex')

export const INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT = stableHash(BENCHMARK_SPEC)
export const INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_POLICY_FINGERPRINT = stableHash({
  version: 1,
  benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
  purpose: 'non-gate-admission-screen',
  variant: 'full',
  coverage: 'frozen-holdout-one-seed',
  metric: 'mean-graph-f1',
  operator: '<',
  threshold: INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD,
  outcome: 'never-passes-structural-gate',
})
export const INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT = stableHash({
  version: 1,
  benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
  purpose: 'decisive-structural-gate-short-circuit',
  requiredVariant: 'full-complete',
  metric: 'holdout-graph-f1',
  boundary: 'ci.low',
  operator: '<',
  threshold: INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD,
  outcome: 'failure-only',
})

function finiteRatio(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
}

function finiteSignedRatio(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= -1 && value <= 1
}

function contractCheck(id, passed, actual, expected) {
  return { id, passed: Boolean(passed), actual: actual ?? null, expected }
}

function metricCheck(id, actual, threshold) {
  const passed = finiteRatio(actual) && (threshold.direction === 'minimum'
    ? actual >= threshold.value
    : actual <= threshold.value)
  return contractCheck(
    id,
    passed,
    finiteRatio(actual) ? actual : null,
    `${threshold.direction === 'minimum' ? '>=' : '<='} ${threshold.value.toFixed(2)}`,
  )
}

function finiteInterval(value, { signed = false } = {}) {
  const minimum = signed ? -1 : 0
  return value && typeof value === 'object' && !Array.isArray(value) &&
    [value.mean, value.low, value.high].every((item) => typeof item === 'number' && Number.isFinite(item) && item >= minimum && item <= 1) &&
    value.low <= value.mean + 1e-12 && value.mean <= value.high + 1e-12 &&
    value.confidence === 0.95 && value.seedCount === 5 &&
    Number.isInteger(value.sourceGroupCount) && value.sourceGroupCount >= 1 &&
    Number.isInteger(value.samples) && value.samples >= 1_000
}

function confidenceMetricCheck(id, value, threshold, options = {}) {
  const valid = finiteInterval(value, options)
  const boundary = threshold.direction === 'minimum' ? value?.low : value?.high
  const passed = valid && (threshold.direction === 'minimum'
    ? boundary >= threshold.value
    : boundary <= threshold.value)
  return contractCheck(
    id,
    passed,
    typeof boundary === 'number' && Number.isFinite(boundary) ? boundary : null,
    `${threshold.direction === 'minimum' ? 'CI.low >=' : 'CI.high <='} ${threshold.value.toFixed(2)}`,
  )
}

function earlyEvidenceBaseChecks(value, evidence, policy, {
  expectedFormat,
  expectedSnapshotId,
  expectedInputManifestHash,
  expectedPolicyFingerprint,
  policyField,
  expectedSeedCount,
}) {
  const expectedObservationCount = Number.isInteger(evidence.holdoutCombinationCount)
    ? evidence.holdoutCombinationCount * 2 * expectedSeedCount
    : null
  return [
    contractCheck('format', value.format === expectedFormat, value.format, expectedFormat),
    contractCheck('version', value.version === INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION, value.version, INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION),
    contractCheck('benchmark-id', value.benchmarkId === INDEPENDENT_STRUCTURAL_BENCHMARK_ID, value.benchmarkId, INDEPENDENT_STRUCTURAL_BENCHMARK_ID),
    contractCheck('benchmark-fingerprint', value.benchmarkFingerprint === INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT, value.benchmarkFingerprint, INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT),
    contractCheck('decision-policy', value[policyField] === expectedPolicyFingerprint, value[policyField], expectedPolicyFingerprint),
    contractCheck('runner-version', value.runnerVersion === 1, value.runnerVersion, 1),
    contractCheck('runner-fingerprint', typeof value.runnerFingerprint === 'string' && /^[a-f0-9]{64}$/u.test(value.runnerFingerprint), value.runnerFingerprint, 'sha256'),
    contractCheck('origin', value.origin === 'independent-deterministic', value.origin, 'independent-deterministic'),
    contractCheck('deterministic', value.deterministic === true, value.deterministic, true),
    contractCheck('no-ollama-evaluator', value.usesOllama === false, value.usesOllama, false),
    contractCheck('snapshot-binding', typeof expectedSnapshotId === 'string' && expectedSnapshotId.length > 0 && value.snapshotId === expectedSnapshotId, value.snapshotId, expectedSnapshotId ?? 'existing proposal snapshot'),
    contractCheck(
      'manifest-binding',
      typeof expectedInputManifestHash === 'string' && expectedInputManifestHash.length > 0 && value.inputManifestHash === expectedInputManifestHash,
      value.inputManifestHash,
      expectedInputManifestHash ?? 'existing proposal input manifest',
    ),
    contractCheck('combination-count', evidence.combinationCount === 144, evidence.combinationCount, 144),
    contractCheck('surface-count', evidence.surfacesPerCombination === 2, evidence.surfacesPerCombination, 2),
    contractCheck('split-before-surface', evidence.splitPolicy === 'combination-before-surface-v1', evidence.splitPolicy, 'combination-before-surface-v1'),
    contractCheck('holdout-ratio', finiteRatio(evidence.holdoutRatio) && evidence.holdoutRatio >= 0.30, evidence.holdoutRatio, '>= 0.30'),
    contractCheck('holdout-combinations', Number.isInteger(evidence.holdoutCombinationCount) && evidence.holdoutCombinationCount >= 44, evidence.holdoutCombinationCount, '>= 44'),
    contractCheck('independent-seeds', evidence.seedCount === expectedSeedCount, evidence.seedCount, expectedSeedCount),
    contractCheck('source-groups', evidence.sourceGroupCount === evidence.holdoutCombinationCount, evidence.sourceGroupCount, 'holdoutCombinationCount'),
    contractCheck('observation-count', Number.isInteger(evidence.episodeCount) && evidence.episodeCount === expectedObservationCount, evidence.episodeCount, expectedObservationCount ?? 'exact frozen coverage'),
    contractCheck('maximum-ticks', Number.isInteger(evidence.maximumTicks) && evidence.maximumTicks >= 1 && evidence.maximumTicks <= 512, evidence.maximumTicks, '1..512'),
    contractCheck('observation-digest', typeof evidence.observationDigest === 'string' && /^[a-f0-9]{64}$/u.test(evidence.observationDigest), evidence.observationDigest, 'sha256'),
    contractCheck(
      'observation-chunks',
      Array.isArray(evidence.observationChunkDigests) && evidence.observationChunkDigests.length === expectedSeedCount &&
        evidence.observationChunkDigests.every((digest) => typeof digest === 'string' && /^[a-f0-9]{64}$/u.test(digest)) &&
        new Set(evidence.observationChunkDigests).size === expectedSeedCount,
      Array.isArray(evidence.observationChunkDigests) ? evidence.observationChunkDigests.length : null,
      expectedSeedCount,
    ),
    contractCheck('evaluation-source-policy', policy.sourcePolicy === 'controlled-benchmark-only-no-ollama-v1', policy.sourcePolicy, 'controlled-benchmark-only-no-ollama-v1'),
    contractCheck('validation-no-ollama', policy.validationOllamaExampleCount === 0, policy.validationOllamaExampleCount, 0),
    contractCheck('test-no-ollama', policy.testOllamaExampleCount === 0, policy.testOllamaExampleCount, 0),
    contractCheck('split-frozen', policy.splitFrozenBeforeTraining === true, policy.splitFrozenBeforeTraining, true),
  ]
}

/**
 * Validates a cheap, one-seed admission screen. This evidence is deliberately
 * non-conclusive and can never satisfy the structural stop gate.
 */
export function evaluateIndependentStructuralBenchmarkScreen(report, {
  expectedSnapshotId = null,
  expectedInputManifestHash = null,
} = {}) {
  const value = report && typeof report === 'object' && !Array.isArray(report) ? report : {}
  const evidence = value.evidence && typeof value.evidence === 'object' && !Array.isArray(value.evidence) ? value.evidence : {}
  const policy = value.evaluationPolicy && typeof value.evaluationPolicy === 'object' && !Array.isArray(value.evaluationPolicy) ? value.evaluationPolicy : {}
  const diagnostic = value.diagnostic && typeof value.diagnostic === 'object' && !Array.isArray(value.diagnostic) ? value.diagnostic : {}
  const checks = [
    ...earlyEvidenceBaseChecks(value, evidence, policy, {
      expectedFormat: INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_FORMAT,
      expectedSnapshotId,
      expectedInputManifestHash,
      expectedPolicyFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_POLICY_FINGERPRINT,
      policyField: 'screenPolicyFingerprint',
      expectedSeedCount: 1,
    }),
    contractCheck('completion', value.completion === 'diagnostic-screened-out', value.completion, 'diagnostic-screened-out'),
    contractCheck('diagnostic-variant', diagnostic.variant === 'full', diagnostic.variant, 'full'),
    contractCheck('diagnostic-seed', Number.isInteger(diagnostic.seed), diagnostic.seed, 'integer'),
    contractCheck('diagnostic-threshold', diagnostic.threshold === INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD, diagnostic.threshold, INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD),
    contractCheck('diagnostic-mean', finiteRatio(diagnostic.meanGraphF1), diagnostic.meanGraphF1, '0..1'),
    contractCheck('screened-out', finiteRatio(diagnostic.meanGraphF1) && diagnostic.meanGraphF1 < INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD, diagnostic.meanGraphF1, `< ${INDEPENDENT_STRUCTURAL_BENCHMARK_ADMISSION_THRESHOLD.toFixed(2)}`),
  ]
  const valid = checks.every((check) => check.passed)
  return {
    status: valid ? 'screened-out' : 'invalid',
    passed: false,
    valid,
    conclusive: false,
    completion: valid ? 'diagnostic-screened-out' : null,
    evaluationKind: 'screen',
    benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    snapshotId: typeof value.snapshotId === 'string' ? value.snapshotId : null,
    inputManifestHash: typeof value.inputManifestHash === 'string' ? value.inputManifestHash : null,
    diagnosticMeanGraphF1: finiteRatio(diagnostic.meanGraphF1) ? diagnostic.meanGraphF1 : null,
    checks,
  }
}

/**
 * Validates a mathematically decisive early failure after the full variant's
 * entire frozen holdout x five-seed coverage is complete. It never passes.
 */
export function evaluateIndependentStructuralBenchmarkEarlyFailure(report, {
  expectedSnapshotId = null,
  expectedInputManifestHash = null,
} = {}) {
  const value = report && typeof report === 'object' && !Array.isArray(report) ? report : {}
  const evidence = value.evidence && typeof value.evidence === 'object' && !Array.isArray(value.evidence) ? value.evidence : {}
  const policy = value.evaluationPolicy && typeof value.evaluationPolicy === 'object' && !Array.isArray(value.evaluationPolicy) ? value.evaluationPolicy : {}
  const decisive = value.decisiveCheck && typeof value.decisiveCheck === 'object' && !Array.isArray(value.decisiveCheck) ? value.decisiveCheck : {}
  const threshold = decisive.threshold && typeof decisive.threshold === 'object' && !Array.isArray(decisive.threshold) ? decisive.threshold : {}
  const intervalValue = decisive.interval
  const skipped = evidence.skippedVariants
  const checks = [
    ...earlyEvidenceBaseChecks(value, evidence, policy, {
      expectedFormat: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT,
      expectedSnapshotId,
      expectedInputManifestHash,
      expectedPolicyFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT,
      policyField: 'earlyFailurePolicyFingerprint',
      expectedSeedCount: 5,
    }),
    contractCheck('completion', value.completion === 'decisive-early-failure', value.completion, 'decisive-early-failure'),
    contractCheck('completed-variant', evidence.completedVariant === 'full', evidence.completedVariant, 'full'),
    contractCheck('bootstrap-confidence', evidence.bootstrapConfidence === 0.95, evidence.bootstrapConfidence, 0.95),
    contractCheck('bootstrap-samples', Number.isInteger(evidence.bootstrapSamples) && evidence.bootstrapSamples >= 1_000, evidence.bootstrapSamples, '>= 1000'),
    contractCheck('skipped-variants', Array.isArray(skipped) && skipped.length === 9 && new Set(skipped).size === 9, Array.isArray(skipped) ? skipped.length : null, 9),
    contractCheck('skipped-episodes', Number.isInteger(evidence.skippedEpisodeCount) && evidence.skippedEpisodeCount > 0, evidence.skippedEpisodeCount, '> 0'),
    contractCheck('decisive-check-id', decisive.id === 'holdout-graph-f1', decisive.id, 'holdout-graph-f1'),
    contractCheck('threshold-direction', threshold.direction === 'minimum', threshold.direction, 'minimum'),
    contractCheck('threshold-boundary', threshold.boundary === 'ci.low', threshold.boundary, 'ci.low'),
    contractCheck('threshold-value', threshold.value === INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD, threshold.value, INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD),
    contractCheck('full-interval', finiteInterval(intervalValue) && intervalValue.sourceGroupCount === evidence.sourceGroupCount && intervalValue.samples === evidence.bootstrapSamples, finiteInterval(intervalValue) ? intervalValue.low : null, 'valid paired-source-group 95% interval'),
    contractCheck('decisive-failure', decisive.passed === false && finiteInterval(intervalValue) && intervalValue.low < INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD, finiteInterval(intervalValue) ? intervalValue.low : null, `< ${INDEPENDENT_STRUCTURAL_BENCHMARK_HOLDOUT_THRESHOLD.toFixed(2)}`),
  ]
  const valid = checks.every((check) => check.passed)
  return {
    status: valid ? 'failed-early' : 'invalid',
    passed: false,
    valid,
    conclusive: valid,
    completion: valid ? 'decisive-early-failure' : null,
    evaluationKind: 'full-early-failure',
    decisiveCheckId: valid ? 'holdout-graph-f1' : null,
    benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    snapshotId: typeof value.snapshotId === 'string' ? value.snapshotId : null,
    inputManifestHash: typeof value.inputManifestHash === 'string' ? value.inputManifestHash : null,
    holdoutGraphF1: finiteInterval(intervalValue) ? intervalValue.mean : null,
    holdoutGraphF1Low: finiteInterval(intervalValue) ? intervalValue.low : null,
    holdoutGraphF1High: finiteInterval(intervalValue) ? intervalValue.high : null,
    checks,
  }
}

export function evaluateIndependentStructuralEvidence(report, options = {}) {
  if (report?.format === INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_FORMAT) {
    return evaluateIndependentStructuralBenchmarkScreen(report, options)
  }
  if (report?.format === INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT) {
    return evaluateIndependentStructuralBenchmarkEarlyFailure(report, options)
  }
  return evaluateIndependentStructuralBenchmark(report, options)
}

/**
 * Validates an internal, snapshot-bound benchmark report. A caller-provided
 * `passed` flag is deliberately ignored: every gate is recomputed here from
 * the frozen protocol and numeric evidence.
 */
export function evaluateIndependentStructuralBenchmark(report, {
  expectedSnapshotId = null,
  expectedInputManifestHash = null,
} = {}) {
  const value = report && typeof report === 'object' && !Array.isArray(report) ? report : {}
  const evidence = value.evidence && typeof value.evidence === 'object' && !Array.isArray(value.evidence)
    ? value.evidence
    : {}
  const policy = value.evaluationPolicy && typeof value.evaluationPolicy === 'object' && !Array.isArray(value.evaluationPolicy)
    ? value.evaluationPolicy
    : {}
  const metrics = value.metrics && typeof value.metrics === 'object' && !Array.isArray(value.metrics)
    ? value.metrics
    : {}
  const intervals = evidence.intervals && typeof evidence.intervals === 'object' && !Array.isArray(evidence.intervals)
    ? evidence.intervals
    : {}
  const birthConfusion = evidence.birthConfusion && typeof evidence.birthConfusion === 'object' && !Array.isArray(evidence.birthConfusion)
    ? evidence.birthConfusion
    : {}
  const birthCountsValid = ['truePositive', 'falsePositive', 'falseNegative', 'trueNegative']
    .every((key) => Number.isInteger(birthConfusion[key]) && birthConfusion[key] >= 0) &&
    birthConfusion.truePositive + birthConfusion.falseNegative > 0 &&
    birthConfusion.falsePositive + birthConfusion.trueNegative > 0 &&
    birthConfusion.truePositive + birthConfusion.falsePositive > 0 &&
    birthConfusion.truePositive + birthConfusion.falseNegative === 25 &&
    birthConfusion.falsePositive + birthConfusion.trueNegative === 25
  const measuredBirthPrecision = birthCountsValid
    ? birthConfusion.truePositive / (birthConfusion.truePositive + birthConfusion.falsePositive)
    : Number.NaN
  const measuredBirthSensitivity = birthCountsValid
    ? birthConfusion.truePositive / (birthConfusion.truePositive + birthConfusion.falseNegative)
    : Number.NaN
  const footprintKeys = [
    'memberships', 'familyTransitions', 'utilityReports', 'coalitions',
    'coalitionActivations', 'coalitionBindings', 'birthMemberships', 'total',
  ]
  const targetFamilyFootprint = evidence.targetFamilyFootprint && typeof evidence.targetFamilyFootprint === 'object'
    ? evidence.targetFamilyFootprint
    : {}
  const shamFamilyFootprint = evidence.shamFamilyFootprint && typeof evidence.shamFamilyFootprint === 'object'
    ? evidence.shamFamilyFootprint
    : {}
  const validFamilyFootprint = (value) => footprintKeys.every((key) => Number.isInteger(value[key]) && value[key] >= 0) &&
    value.total === footprintKeys.slice(0, -1).reduce((sum, key) => sum + value[key], 0)
  const familyFootprintsMatch = validFamilyFootprint(targetFamilyFootprint) && validFamilyFootprint(shamFamilyFootprint) &&
    targetFamilyFootprint.total > 0 && footprintKeys.every((key) => targetFamilyFootprint[key] === shamFamilyFootprint[key])

  const contractChecks = [
    contractCheck('format', value.format === INDEPENDENT_STRUCTURAL_BENCHMARK_FORMAT, value.format, INDEPENDENT_STRUCTURAL_BENCHMARK_FORMAT),
    contractCheck('version', value.version === INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION, value.version, INDEPENDENT_STRUCTURAL_BENCHMARK_VERSION),
    contractCheck('benchmark-id', value.benchmarkId === INDEPENDENT_STRUCTURAL_BENCHMARK_ID, value.benchmarkId, INDEPENDENT_STRUCTURAL_BENCHMARK_ID),
    contractCheck('benchmark-fingerprint', value.benchmarkFingerprint === INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT, value.benchmarkFingerprint, INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT),
    contractCheck('origin', value.origin === 'independent-deterministic', value.origin, 'independent-deterministic'),
    contractCheck('deterministic', value.deterministic === true, value.deterministic, true),
    contractCheck('no-ollama-evaluator', value.usesOllama === false, value.usesOllama, false),
    contractCheck('snapshot-binding', typeof expectedSnapshotId === 'string' && expectedSnapshotId.length > 0 && value.snapshotId === expectedSnapshotId, value.snapshotId, expectedSnapshotId ?? 'existing proposal snapshot'),
    contractCheck(
      'manifest-binding',
      typeof expectedInputManifestHash === 'string' && expectedInputManifestHash.length > 0 && value.inputManifestHash === expectedInputManifestHash,
      value.inputManifestHash,
      expectedInputManifestHash ?? 'existing proposal input manifest',
    ),
    contractCheck('combination-count', evidence.combinationCount === 144, evidence.combinationCount, 144),
    contractCheck('surface-count', evidence.surfacesPerCombination === 2, evidence.surfacesPerCombination, 2),
    contractCheck('split-before-surface', evidence.splitPolicy === 'combination-before-surface-v1', evidence.splitPolicy, 'combination-before-surface-v1'),
    contractCheck('holdout-ratio', finiteRatio(evidence.holdoutRatio) && evidence.holdoutRatio >= 0.30, evidence.holdoutRatio, '>= 0.30'),
    contractCheck('holdout-combinations', Number.isInteger(evidence.holdoutCombinationCount) && evidence.holdoutCombinationCount >= 44, evidence.holdoutCombinationCount, '>= 44'),
    contractCheck('independent-seeds', evidence.seedCount === 5, evidence.seedCount, 5),
    contractCheck('source-groups', Number.isInteger(evidence.sourceGroupCount) && evidence.sourceGroupCount >= 44, evidence.sourceGroupCount, '>= 44'),
    contractCheck('bootstrap-confidence', evidence.bootstrapConfidence === 0.95, evidence.bootstrapConfidence, 0.95),
    contractCheck('bootstrap-samples', Number.isInteger(evidence.bootstrapSamples) && evidence.bootstrapSamples >= 1_000, evidence.bootstrapSamples, '>= 1000'),
    contractCheck('ablation-target-selection', evidence.ablationTargetSelection === 'snapshot-model-only-no-evaluation-data-v2', evidence.ablationTargetSelection, 'snapshot-model-only-no-evaluation-data-v2'),
    contractCheck('edge-sham-protocol', evidence.edgeShamProtocol === 'snapshot-digest-random-nontarget-edge-v1', evidence.edgeShamProtocol, 'snapshot-digest-random-nontarget-edge-v1'),
    contractCheck(
      'edge-sham-selection',
      typeof evidence.targetEdge === 'string' && evidence.targetEdge.length > 0 &&
        typeof evidence.shamEdge === 'string' && evidence.shamEdge.length > 0 && evidence.shamEdge !== evidence.targetEdge,
      typeof evidence.shamEdge === 'string' ? evidence.shamEdge : null,
      'deterministic non-target persistent edge',
    ),
    contractCheck('family-sham-protocol', evidence.familyShamProtocol === 'deterministic-model-only-exact-reference-footprint-mask-v1', evidence.familyShamProtocol, 'deterministic-model-only-exact-reference-footprint-mask-v1'),
    contractCheck('family-sham-footprint', evidence.shamFamilyFootprintMatched === true && familyFootprintsMatch, evidence.shamFamilyFootprintMatched === true ? shamFamilyFootprint : null, 'exact target/sham family reference counts and kinds'),
    contractCheck('family-sham-mask-digest', typeof evidence.shamFamilyMaskDigest === 'string' && /^[a-f0-9]{64}$/u.test(evidence.shamFamilyMaskDigest), evidence.shamFamilyMaskDigest, 'sha256'),
    contractCheck('recurrence-off-protocol', evidence.noRecurrenceProtocol === 'full-tick-zero-persistent-transition-evidence-v2', evidence.noRecurrenceProtocol, 'full-tick-zero-persistent-transition-evidence-v2'),
    contractCheck('birth-precision-definition', evidence.birthPrecisionDefinition === 'TP/(TP+FP)', evidence.birthPrecisionDefinition, 'TP/(TP+FP)'),
    contractCheck('birth-negative-source', evidence.birthNegativeSource === 'frozen-train-calibration-repeated-four-observations-v2', evidence.birthNegativeSource, 'frozen-train-calibration-repeated-four-observations-v2'),
    contractCheck('birth-positive-cases', evidence.eligibleNoveltyCaseCount === 5, evidence.eligibleNoveltyCaseCount, 5),
    contractCheck('birth-negative-cases', evidence.knownBirthPrecisionNegativeCaseCount === 5, evidence.knownBirthPrecisionNegativeCaseCount, 5),
    contractCheck('birth-confusion-matrix', birthCountsValid, birthCountsValid ? birthConfusion : null, 'non-empty positive and negative deterministic trials'),
    contractCheck(
      'birth-precision-recomputed',
      birthCountsValid && finiteRatio(metrics.repeatedNoveltyBirthPrecision) && Math.abs(metrics.repeatedNoveltyBirthPrecision - measuredBirthPrecision) <= 1e-9,
      Number.isFinite(measuredBirthPrecision) ? measuredBirthPrecision : null,
      'repeatedNoveltyBirthPrecision = TP/(TP+FP)',
    ),
    contractCheck(
      'birth-sensitivity-evidence',
      finiteInterval(intervals.repeatedNoveltyBirthSensitivity) &&
        intervals.repeatedNoveltyBirthSensitivity.sourceGroupCount === 5 &&
        Math.abs(intervals.repeatedNoveltyBirthSensitivity.mean - measuredBirthSensitivity) <= 1e-9 &&
        finiteInterval(intervals.repeatedNoveltyBirthPrecision) &&
        intervals.repeatedNoveltyBirthPrecision.sourceGroupCount === 10,
      finiteInterval(intervals.repeatedNoveltyBirthSensitivity) ? intervals.repeatedNoveltyBirthSensitivity.mean : null,
      'separate 5-positive-group sensitivity and 10-group precision 95% intervals',
    ),
    contractCheck(
      'metric-interval-means',
      [
        'holdoutGraphF1', 'h0OnlyGraphF1', 'noRecurrenceGraphF1', 'shuffledEdgeGraphF1',
        'perturbationRecoveryRate', 'knownStructureFalseBirthRate',
        'repeatedNoveltyBirthPrecision', 'repeatedNoveltyBirthSensitivity',
      ].every((key) => finiteRatio(metrics[key]) && finiteInterval(intervals[key]) && Math.abs(metrics[key] - intervals[key].mean) <= 1e-9) &&
        ['targetFamilyAblationDrop', 'targetEdgeAblationDrop', 'shamFamilyAblationDrop', 'shamEdgeAblationDrop']
          .every((key) => finiteSignedRatio(metrics[key]) && finiteInterval(intervals[key], { signed: true }) && Math.abs(metrics[key] - intervals[key].mean) <= 1e-9),
      'paired source-group intervals',
      'every scalar metric equals its 95% source-group bootstrap mean',
    ),
    contractCheck('evaluation-source-policy', policy.sourcePolicy === 'controlled-benchmark-only-no-ollama-v1', policy.sourcePolicy, 'controlled-benchmark-only-no-ollama-v1'),
    contractCheck('validation-no-ollama', policy.validationOllamaExampleCount === 0, policy.validationOllamaExampleCount, 0),
    contractCheck('test-no-ollama', policy.testOllamaExampleCount === 0, policy.testOllamaExampleCount, 0),
    contractCheck('split-frozen', policy.splitFrozenBeforeTraining === true, policy.splitFrozenBeforeTraining, true),
  ]

  const strongestBaseline = ['h0OnlyGraphF1', 'noRecurrenceGraphF1', 'shuffledEdgeGraphF1']
    .map((key) => metrics[key])
    .filter(finiteRatio)
    .reduce((maximum, value) => Math.max(maximum, value), 0)
  const baselineAdvantage = finiteRatio(metrics.holdoutGraphF1) &&
    ['h0OnlyGraphF1', 'noRecurrenceGraphF1', 'shuffledEdgeGraphF1'].every((key) => finiteRatio(metrics[key]))
    ? metrics.holdoutGraphF1 - strongestBaseline
    : Number.NaN
  const metricChecks = [
    confidenceMetricCheck('holdout-graph-f1', intervals.holdoutGraphF1, GATE_THRESHOLDS.holdoutGraphF1),
    confidenceMetricCheck('baseline-advantage', intervals.baselineAdvantage, GATE_THRESHOLDS.baselineAdvantage, { signed: true }),
    confidenceMetricCheck('target-family-ablation', intervals.targetFamilyAblationDrop, GATE_THRESHOLDS.targetFamilyAblationDrop, { signed: true }),
    confidenceMetricCheck('target-edge-ablation', intervals.targetEdgeAblationDrop, GATE_THRESHOLDS.targetEdgeAblationDrop, { signed: true }),
    confidenceMetricCheck('sham-family-ablation', intervals.shamFamilyAblationDrop, GATE_THRESHOLDS.shamFamilyAblationDrop, { signed: true }),
    confidenceMetricCheck('sham-edge-ablation', intervals.shamEdgeAblationDrop, GATE_THRESHOLDS.shamEdgeAblationDrop, { signed: true }),
    confidenceMetricCheck('perturbation-recovery', intervals.perturbationRecoveryRate, GATE_THRESHOLDS.perturbationRecoveryRate),
    confidenceMetricCheck('known-structure-false-birth', intervals.knownStructureFalseBirthRate, GATE_THRESHOLDS.knownStructureFalseBirthRate),
    confidenceMetricCheck('novelty-birth-precision', intervals.repeatedNoveltyBirthPrecision, GATE_THRESHOLDS.repeatedNoveltyBirthPrecision),
    confidenceMetricCheck('novelty-birth-sensitivity', intervals.repeatedNoveltyBirthSensitivity, GATE_THRESHOLDS.repeatedNoveltyBirthSensitivity),
  ]
  const checks = [...contractChecks, ...metricChecks]
  const contractValid = contractChecks.every((check) => check.passed)
  return {
    status: checks.every((check) => check.passed) ? 'passed' : 'failed',
    passed: checks.every((check) => check.passed),
    valid: contractValid,
    contractValid,
    conclusive: contractValid,
    completion: contractValid ? 'complete' : null,
    evaluationKind: 'full',
    benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    snapshotId: typeof value.snapshotId === 'string' ? value.snapshotId : null,
    inputManifestHash: typeof value.inputManifestHash === 'string' ? value.inputManifestHash : null,
    strongestBaseline,
    baselineAdvantage: Number.isFinite(baselineAdvantage) ? baselineAdvantage : null,
    checks,
  }
}

/**
 * Default production adapter. It accepts only immutable artifacts attached to
 * the exact proposal snapshot; ordinary pipeline metrics can never satisfy it.
 */
export function readIndependentStructuralBenchmark(store, snapshotId) {
  const snapshot = typeof snapshotId === 'string' ? store?.getSnapshot?.(snapshotId) : null
  if (!snapshot) {
    return {
      status: 'missing', passed: false, snapshotId: snapshotId ?? null,
      benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
      benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
      checks: [contractCheck('snapshot', false, null, snapshotId ?? 'existing snapshot')],
    }
  }
  const artifacts = (snapshot.artifactIds ?? [])
    .map((artifactId) => store.getArtifactMetadata?.(artifactId) ?? store.getArtifact?.(artifactId))
    .filter((artifact) => artifact?.kind === INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND)
    .map((artifact) => store.getArtifact?.(artifact.id))
    .filter(Boolean)
    .sort((left, right) => String(right.createdAt ?? '').localeCompare(String(left.createdAt ?? '')))
  if (!artifacts.length) {
    return {
      status: 'missing', passed: false, snapshotId: snapshot.id,
      benchmarkId: INDEPENDENT_STRUCTURAL_BENCHMARK_ID,
      benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
      checks: [contractCheck('benchmark-artifact', false, null, INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND)],
    }
  }
  const evaluated = artifacts.map((artifact) => ({
    artifactId: artifact.id,
    ...evaluateIndependentStructuralBenchmark(artifact.payload, {
      expectedSnapshotId: snapshot.id,
      expectedInputManifestHash: snapshot.inputManifestHash ?? null,
    }),
  }))
  return evaluated.find((item) => item.passed) ?? evaluated[0]
}

/** Returns raw immutable evidence for an injected runner/adapter contract. */
export function findIndependentStructuralBenchmarkReport(store, snapshotId) {
  const snapshot = typeof snapshotId === 'string' ? store?.getSnapshot?.(snapshotId) : null
  if (!snapshot) return null
  const artifact = (snapshot.artifactIds ?? [])
    .map((artifactId) => store.getArtifactMetadata?.(artifactId) ?? store.getArtifact?.(artifactId))
    .filter((item) => item?.kind === INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND)
    .map((item) => store.getArtifact?.(item.id))
    .filter(Boolean)
    .sort((left, right) => String(right.createdAt ?? '').localeCompare(String(left.createdAt ?? '')))[0]
  return artifact ? { artifactId: artifact.id, report: artifact.payload, snapshot } : null
}

/** Pure logical resource guard used before purchasing another training block. */
export function evaluateContinuousResourceBudget({ store, config, plan }) {
  const storageBytes = Number(store?.approximateStorageBytes?.() ?? 0)
  const diskBudgetBytes = Number(store?.getEffectiveDiskBudgetBytes?.(config) ?? (Number(config?.budgets?.diskMb ?? 0) * 1024 * 1024))
  const remainingDiskBytes = Math.max(0, diskBudgetBytes - storageBytes)
  const sources = typeof store?.listSources === 'function' ? store.listSources() : []
  const syntheticSourceCount = sources.filter((source) => source?.sourceType === 'ollama-generated').length
  const projectedSyntheticSourceCount = syntheticSourceCount + Math.max(0, Number(plan.blockAccepted ?? 0) || 0)
  const checks = [
    contractCheck('disk-headroom', remainingDiskBytes >= plan.minimumDiskHeadroomBytes, remainingDiskBytes, `>= ${plan.minimumDiskHeadroomBytes}`),
    contractCheck('synthetic-source-limit', projectedSyntheticSourceCount <= plan.maximumSyntheticSources, projectedSyntheticSourceCount, `<= ${plan.maximumSyntheticSources}`),
  ]
  const failed = checks.find((check) => !check.passed)
  return {
    passed: !failed,
    code: failed?.id === 'disk-headroom' ? 'CONTINUOUS_DISK_HEADROOM_LOW' : failed ? 'CONTINUOUS_SOURCE_LIMIT_REACHED' : null,
    storageBytes,
    diskBudgetBytes,
    remainingDiskBytes,
    syntheticSourceCount,
    projectedSyntheticSourceCount,
    checks,
  }
}
