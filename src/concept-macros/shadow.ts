import { createMacroConfig } from './config.ts'
import {
  canonicalFloat,
  canonicalVector,
  compareMacroText,
  macroDigest,
  squaredError,
  vectorFidelity,
} from './deterministic.ts'
import type {
  CompositeNeuronV1,
  MacroConfigV1,
  MacroShadowCaseV1,
  MacroShadowEvaluationV1,
} from './types.ts'

/** Excludes lifecycle/evidence ledgers but changes for every executable definition change. */
export function computeMacroExecutionFingerprint(composite: CompositeNeuronV1): string {
  return macroDigest('CMACROEXEC', {
    kind: composite.kind,
    executionSemantics: composite.executionSemantics,
    exactSignature: composite.exactSignature,
    familySignature: composite.familySignature,
    orderedExpansion: composite.orderedExpansion,
    expansionDigest: composite.expansionDigest,
    depth: composite.depth,
    entryPrototype: composite.entryPrototype,
    exitPrototype: composite.exitPrototype,
    trajectoryPrototype: composite.trajectoryPrototype,
    effectPrototype: composite.effectPrototype,
    effectCohesion: composite.effectCohesion,
  })
}

function validateShadowCase(value: MacroShadowCaseV1): void {
  if (!value.id.trim()) throw new TypeError('Shadow vaka kimliği boş olamaz.')
  if (!value.independenceGroupId.trim()) {
    throw new TypeError('Shadow vaka bağımsız kaynak grubu taşımalıdır.')
  }
  if (value.split !== 'validation' && value.split !== 'test') {
    throw new TypeError('Shadow değerlendirmesi yalnız validation/test vakalarını kabul eder.')
  }
  if (!value.expandedOutput.length || value.expandedOutput.length !== value.macroOutput.length) {
    throw new RangeError('Expanded ve makro shadow çıktıları aynı pozitif boyutta olmalıdır.')
  }
  if (value.expectedOutput && value.expectedOutput.length !== value.expandedOutput.length) {
    throw new RangeError('Beklenen shadow çıktısı karşılaştırılan çıktılarla aynı boyutta olmalıdır.')
  }
  const vectors = [value.expandedOutput, value.macroOutput, ...(value.expectedOutput ? [value.expectedOutput] : [])]
  if (vectors.some((vector) => vector.some((entry) => !Number.isFinite(entry)))) {
    throw new TypeError('Shadow çıktıları yalnız sonlu değerler içerebilir.')
  }
  if (!Number.isFinite(value.expandedCost) || !Number.isFinite(value.macroCost) ||
    value.expandedCost < 0 || value.macroCost < 0) {
    throw new RangeError('Shadow yürütme maliyetleri sonlu ve negatif olmayan değerler olmalıdır.')
  }
}

function shadowFailures(
  metrics: Pick<
    MacroShadowEvaluationV1,
    | 'caseCount'
    | 'independentGroupCount'
    | 'positiveCaseCount'
    | 'negativeCaseCount'
    | 'fidelity'
    | 'falseActivationRate'
    | 'activationRecall'
    | 'holdoutUtility'
    | 'observedCompressionGain'
  >,
  config: MacroConfigV1,
): string[] {
  const failures: string[] = []
  if (metrics.caseCount < config.minimumShadowCases) failures.push('insufficient-shadow-cases')
  if (metrics.independentGroupCount < config.minimumShadowIndependentGroups) {
    failures.push('insufficient-shadow-source-groups')
  }
  if (metrics.positiveCaseCount < config.minimumShadowPositiveCases) {
    failures.push('insufficient-positive-cases')
  }
  if (metrics.negativeCaseCount < config.minimumShadowNegativeCases) {
    failures.push('insufficient-negative-cases')
  }
  if (metrics.fidelity < config.minimumShadowFidelity) failures.push('fidelity-below-threshold')
  if (metrics.falseActivationRate > config.maximumFalseActivationRate) {
    failures.push('false-activation-above-threshold')
  }
  if (metrics.activationRecall < config.minimumActivationRecall) {
    failures.push('activation-recall-below-threshold')
  }
  if (metrics.holdoutUtility < config.minimumHoldoutUtility) {
    failures.push('holdout-utility-below-threshold')
  }
  if (metrics.observedCompressionGain < config.minimumShadowCompressionGain) {
    failures.push('shadow-compression-below-threshold')
  }
  return failures
}

export function validateMacroShadowEvaluation(
  report: MacroShadowEvaluationV1,
  configInput: MacroConfigV1 = createMacroConfig(),
): void {
  const config = createMacroConfig(configInput)
  if (report.format !== 'concept-macro-shadow-report' || report.version !== 1 ||
    !report.id || !report.compositeId || !report.compositeExecutionFingerprint) {
    throw new TypeError('Geçersiz concept-macro-shadow-report kaydı.')
  }
  if (!Number.isInteger(report.caseCount) || report.caseCount < 0 ||
    report.caseIds.length !== report.caseCount ||
    report.caseEvidence.length !== report.caseCount ||
    new Set(report.caseIds).size !== report.caseIds.length ||
    [...report.caseIds].sort(compareMacroText).some((value, index) => value !== report.caseIds[index])) {
    throw new Error('Shadow raporu vaka ledgerı bozuk.')
  }
  const counts = [
    report.falseActivationCount,
    report.negativeCaseCount,
    report.positiveCaseCount,
    report.missedActivationCount,
    report.utilityCaseCount,
  ]
  const canonicalEvidence = [...report.caseEvidence].sort((left, right) =>
    compareMacroText(left.caseId, right.caseId))
  if (canonicalEvidence.some((value, index) => (
    value.caseId !== report.caseIds[index] ||
    !value.independenceGroupId?.trim() ||
    (value.split !== 'validation' && value.split !== 'test')
  ))) {
    throw new Error('Shadow raporu vaka kaynak ledgerı bozuk.')
  }
  const independentGroupCount = new Set(canonicalEvidence.map((value) => value.independenceGroupId)).size
  const validationCaseCount = canonicalEvidence.filter((value) => value.split === 'validation').length
  const testCaseCount = canonicalEvidence.filter((value) => value.split === 'test').length
  if (report.independentGroupCount !== independentGroupCount ||
    report.validationCaseCount !== validationCaseCount ||
    report.testCaseCount !== testCaseCount ||
    validationCaseCount + testCaseCount !== report.caseCount) {
    throw new Error('Shadow raporu kaynak/split sayaçları ledger ile uyuşmuyor.')
  }
  if (counts.some((value) => !Number.isInteger(value) || value < 0) ||
    report.negativeCaseCount + report.positiveCaseCount !== report.caseCount ||
    report.falseActivationCount > report.negativeCaseCount ||
    report.missedActivationCount > report.positiveCaseCount ||
    report.utilityCaseCount > report.caseCount) {
    throw new Error('Shadow raporu sonuç sayaçları bozuk.')
  }
  const expectedFalseRate = report.negativeCaseCount
    ? report.falseActivationCount / report.negativeCaseCount
    : 0
  const expectedRecall = report.positiveCaseCount
    ? (report.positiveCaseCount - report.missedActivationCount) / report.positiveCaseCount
    : 0
  const finite = [
    report.fidelity,
    report.falseActivationRate,
    report.activationRecall,
    report.holdoutUtility,
    report.observedCompressionGain,
  ]
  if (finite.some((value) => !Number.isFinite(value)) ||
    report.fidelity < 0 || report.fidelity > 1 ||
    Math.abs(report.falseActivationRate - expectedFalseRate) > 1e-6 ||
    Math.abs(report.activationRecall - expectedRecall) > 1e-6) {
    throw new Error('Shadow raporu oranları sayaçlarla uyuşmuyor.')
  }
  const expectedFailures = shadowFailures(report, config)
  if (expectedFailures.length !== report.failures.length ||
    expectedFailures.some((value, index) => value !== report.failures[index]) ||
    report.passed !== (expectedFailures.length === 0)) {
    throw new Error('Shadow raporu kapı sonucu metriklerle uyuşmuyor.')
  }
  if (report.id !== macroDigest(
    'MSHADOW', report.compositeId, report.compositeExecutionFingerprint, report.caseIds,
  )) {
    throw new Error('Shadow raporu kimliği vaka ledgerıyla uyuşmuyor.')
  }
  const { fingerprint, ...base } = report
  if (fingerprint !== macroDigest('MSHADOWFP', base)) {
    throw new Error('Shadow raporu fingerprint doğrulamasını geçmedi.')
  }
}

/** Compares the unchanged expanded path with its virtual macro on held-out cases. */
export function evaluateMacroShadow(
  composite: CompositeNeuronV1,
  cases: readonly MacroShadowCaseV1[],
  configInput: MacroConfigV1 = createMacroConfig(),
): MacroShadowEvaluationV1 {
  const config = createMacroConfig(configInput)
  const unique = new Map<string, MacroShadowCaseV1>()
  for (const value of cases) {
    validateShadowCase(value)
    const canonical: MacroShadowCaseV1 = {
      ...value,
      expandedOutput: canonicalVector(value.expandedOutput),
      macroOutput: canonicalVector(value.macroOutput),
      expectedOutput: value.expectedOutput ? canonicalVector(value.expectedOutput) : undefined,
      expandedCost: canonicalFloat(value.expandedCost),
      macroCost: canonicalFloat(value.macroCost),
    }
    const previous = unique.get(value.id)
    if (previous && macroDigest('MSHADOWCASE', previous) !== macroDigest('MSHADOWCASE', canonical)) {
      throw new Error(`Aynı shadow vaka kimliği farklı içerikle kullanıldı: ${value.id}`)
    }
    unique.set(value.id, canonical)
  }
  const canonicalCases = [...unique.values()].sort((left, right) => compareMacroText(left.id, right.id))
  const caseCount = canonicalCases.length
  const caseEvidence = canonicalCases.map((value) => ({
    caseId: value.id,
    split: value.split,
    independenceGroupId: value.independenceGroupId,
  }))
  const independentGroupCount = new Set(caseEvidence.map((value) => value.independenceGroupId)).size
  const validationCaseCount = caseEvidence.filter((value) => value.split === 'validation').length
  const testCaseCount = caseEvidence.filter((value) => value.split === 'test').length
  const fidelity = caseCount
    ? canonicalCases.reduce(
      (sum, value) => sum + vectorFidelity(value.expandedOutput, value.macroOutput),
      0,
    ) / caseCount
    : 0
  const negatives = canonicalCases.filter((value) => !value.shouldActivate)
  const falseActivationCount = negatives.filter((value) => value.macroActivated).length
  const falseActivationRate = negatives.length ? falseActivationCount / negatives.length : 0
  const positives = canonicalCases.filter((value) => value.shouldActivate)
  const missedActivationCount = positives.filter((value) => !value.macroActivated).length
  const activationRecall = positives.length
    ? (positives.length - missedActivationCount) / positives.length
    : 0
  const utilityCases = canonicalCases.filter(
    (value): value is MacroShadowCaseV1 & { expectedOutput: number[] } => Boolean(value.expectedOutput),
  )
  const holdoutUtility = utilityCases.length
    ? utilityCases.reduce((sum, value) =>
      sum + squaredError(value.expandedOutput, value.expectedOutput) -
      squaredError(value.macroOutput, value.expectedOutput), 0) / utilityCases.length
    : 0
  const observedCompressionGain = canonicalCases.reduce(
    (sum, value) => sum + value.expandedCost - value.macroCost,
    0,
  )
  const failures = shadowFailures({
    caseCount,
    independentGroupCount,
    positiveCaseCount: positives.length,
    negativeCaseCount: negatives.length,
    fidelity,
    falseActivationRate,
    activationRecall,
    holdoutUtility,
    observedCompressionGain,
  }, config)
  const compositeExecutionFingerprint = computeMacroExecutionFingerprint(composite)
  const id = macroDigest(
    'MSHADOW', composite.id, compositeExecutionFingerprint, canonicalCases.map((value) => value.id),
  )
  const base: Omit<MacroShadowEvaluationV1, 'fingerprint'> = {
    format: 'concept-macro-shadow-report',
    version: 1,
    id,
    compositeId: composite.id,
    compositeExecutionFingerprint,
    caseIds: canonicalCases.map((value) => value.id),
    caseEvidence,
    caseCount,
    independentGroupCount,
    validationCaseCount,
    testCaseCount,
    fidelity: canonicalFloat(fidelity),
    falseActivationRate: canonicalFloat(falseActivationRate),
    falseActivationCount,
    negativeCaseCount: negatives.length,
    activationRecall: canonicalFloat(activationRecall),
    positiveCaseCount: positives.length,
    missedActivationCount,
    holdoutUtility: canonicalFloat(holdoutUtility),
    utilityCaseCount: utilityCases.length,
    observedCompressionGain: canonicalFloat(observedCompressionGain),
    passed: failures.length === 0,
    failures,
  }
  const report = { ...base, fingerprint: macroDigest('MSHADOWFP', base) }
  validateMacroShadowEvaluation(report, config)
  return report
}
