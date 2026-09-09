import {
  computeMacroConfigFingerprint,
  createMacroConfig,
  validateMacroConfig,
} from './config.ts'
import { computeCompositeDepth, validateCompositeDag } from './dag.ts'
import {
  canonicalExpansion,
  canonicalFamilyIds,
  canonicalFloat,
  compareMacroText,
  computeExactSignature,
  computeExpansionDigest,
  computeFamilySignature,
  computeMdlMetrics,
  createSignalSignature,
  macroDigest,
} from './deterministic.ts'
import { meetsMacroProposalThresholds, mineMacroMotifs } from './miner.ts'
import {
  aggregateMacroPrototypeGroups,
  mergeMacroPrototypeGroups,
  validateMacroPrototypeGroup,
} from './packed.ts'
import {
  boundedMacroReplayLedger,
  createMacroReplayBloom,
  validateMacroReplayBloom,
} from './replay.ts'
import {
  computeMacroExecutionFingerprint,
  validateMacroShadowEvaluation,
} from './shadow.ts'
import type {
  CompositeNeuronV1,
  MacroConfigV1,
  MacroEvidenceReferenceV1,
  MacroExpansionVariantV1,
  MacroHeadRebaseResultV1,
  MacroHeadV1,
  MacroLifecycle,
  MacroMiningIngestResultV1,
  MacroMotifCandidateV1,
  MacroReferenceCatalogV1,
  MacroShadowEvaluationV1,
  MacroStructuralEquivalenceAuditV1,
  MacroTraceV1,
} from './types.ts'
import { CONCEPT_MACRO_EXECUTION_SEMANTICS } from './types.ts'

export interface MacroHeadReferenceV1 {
  representationSpaceId: string
  familyRunFingerprint: string | null
}

function lifecycleRank(lifecycle: MacroLifecycle): number {
  return {
    observing: 0,
    'eligible-proposal': 1,
    'virtual-provisional': 2,
    'shadow-validated': 3,
    active: 4,
    deprecated: -1,
  }[lifecycle]
}

function canonicalEvidence(
  values: readonly MacroEvidenceReferenceV1[],
  maximum: number,
): MacroEvidenceReferenceV1[] {
  const unique = new Map(values.map((value) => [value.observationId, { ...value }]))
  const sorted = [...unique.values()].sort((left, right) =>
    compareMacroText(left.independenceGroupId, right.independenceGroupId) ||
    compareMacroText(left.observationId, right.observationId),
  )
  const firstByGroup: MacroEvidenceReferenceV1[] = []
  const remainder: MacroEvidenceReferenceV1[] = []
  const groups = new Set<string>()
  for (const value of sorted) {
    if (!groups.has(value.independenceGroupId)) {
      groups.add(value.independenceGroupId)
      firstByGroup.push(value)
    } else remainder.push(value)
  }
  const retained = firstByGroup.slice(0, maximum)
  if (retained.length < maximum) {
    remainder.sort((left, right) => compareMacroText(left.observationId, right.observationId))
    retained.push(...remainder.slice(0, maximum - retained.length))
  }
  return retained.sort((left, right) => compareMacroText(left.observationId, right.observationId))
}

function uniqueCount(values: readonly string[]): number {
  return new Set(values).size
}

function canonicalVariants(
  variants: readonly MacroExpansionVariantV1[],
  maximum: number,
): MacroExpansionVariantV1[] {
  const merged = new Map<string, MacroExpansionVariantV1>()
  for (const value of variants) {
    const expansion = canonicalExpansion(value.orderedExpansion)
    const digest = computeExpansionDigest(expansion)
    if (digest !== value.expansionDigest) throw new Error('Makro varyant expansion digest uyuşmazlığı.')
    if (computeExactSignature(expansion) !== value.exactSignature) {
      throw new Error('Makro varyant exact signature uyuşmazlığı.')
    }
    const previous = merged.get(value.exactSignature)
    if (previous && previous.expansionDigest !== digest) {
      throw new Error('Aynı exact signature farklı bir makro genişletmesine bağlandı.')
    }
    merged.set(value.exactSignature, {
      exactSignature: value.exactSignature,
      orderedExpansion: expansion,
      expansionDigest: digest,
      observationCount: (previous?.observationCount ?? 0) + value.observationCount,
    })
  }
  return [...merged.values()]
    .sort((left, right) =>
      right.observationCount - left.observationCount ||
      compareMacroText(left.exactSignature, right.exactSignature),
    )
    .slice(0, maximum)
}

function compositeId(candidate: MacroMotifCandidateV1): string {
  return macroDigest(
    'CMACRO',
    candidate.kind,
    candidate.kind === 'sequence-macro' ? candidate.exactSignature : candidate.familySignature,
  )
}

function compositePayload(composite: Omit<CompositeNeuronV1, 'fingerprint'>): unknown {
  return composite
}

function emptyShadowMetrics(compressionGain: number): CompositeNeuronV1['metrics'] {
  return {
    compressionGain,
    shadowCompressionGain: 0,
    holdoutUtility: 0,
    holdoutUtilityCaseCount: 0,
    falseActivationRate: 0,
    falseActivationCount: 0,
    negativeCaseCount: 0,
    activationRecall: 0,
    positiveCaseCount: 0,
    missedActivationCount: 0,
    fidelity: 0,
    shadowCaseCount: 0,
  }
}

export function computeCompositeNeuronFingerprint(
  composite: Omit<CompositeNeuronV1, 'fingerprint'>,
): string {
  return macroDigest('CMACROFP', compositePayload(composite))
}

function headPayload(head: Omit<MacroHeadV1, 'fingerprint'>): unknown {
  return head
}

export function computeMacroHeadFingerprint(head: Omit<MacroHeadV1, 'fingerprint'>): string {
  return macroDigest('CMHEADFP', headPayload(head))
}

export function computeMacroStructuralEquivalenceAuditFingerprint(
  audit: Omit<MacroStructuralEquivalenceAuditV1, 'fingerprint'>,
): string {
  return macroDigest('CMSTRUCTAUDITFP', audit)
}

export function createMacroStructuralEquivalenceAudit(input: {
  composite: CompositeNeuronV1
  evaluator: string
  evidenceDigest: string
  caseCount: number
  passed: boolean
}): MacroStructuralEquivalenceAuditV1 {
  if (!input.evaluator.trim() || !input.evidenceDigest.trim() ||
    !Number.isInteger(input.caseCount) || input.caseCount < 1) {
    throw new TypeError('Yapısal eşdeğerlik denetimi evaluator, kanıt digest ve vaka sayısı ister.')
  }
  const compositeExecutionFingerprint = computeMacroExecutionFingerprint(input.composite)
  const id = macroDigest(
    'CMSTRUCTAUDIT',
    input.composite.id,
    compositeExecutionFingerprint,
    input.evaluator,
    input.evidenceDigest,
    input.caseCount,
    input.passed,
  )
  const base: Omit<MacroStructuralEquivalenceAuditV1, 'fingerprint'> = {
    format: 'macro-structural-equivalence-audit-v1',
    version: 1,
    id,
    compositeId: input.composite.id,
    compositeExecutionFingerprint,
    evaluator: input.evaluator.trim(),
    evidenceDigest: input.evidenceDigest.trim(),
    caseCount: input.caseCount,
    passed: input.passed,
  }
  return { ...base, fingerprint: computeMacroStructuralEquivalenceAuditFingerprint(base) }
}

export function validateMacroStructuralEquivalenceAudit(
  audit: MacroStructuralEquivalenceAuditV1,
): void {
  if (audit.format !== 'macro-structural-equivalence-audit-v1' || audit.version !== 1 ||
    !audit.id || !audit.compositeId || !audit.compositeExecutionFingerprint ||
    !audit.evaluator || !audit.evidenceDigest ||
    !Number.isInteger(audit.caseCount) || audit.caseCount < 1) {
    throw new TypeError('Geçersiz yapısal eşdeğerlik denetimi.')
  }
  const { fingerprint, ...base } = audit
  if (fingerprint !== computeMacroStructuralEquivalenceAuditFingerprint(base)) {
    throw new Error('Yapısal eşdeğerlik denetimi fingerprint uyuşmazlığı.')
  }
}

function finalizeComposite(
  value: Omit<CompositeNeuronV1, 'fingerprint'> | CompositeNeuronV1,
): CompositeNeuronV1 {
  const { fingerprint: _fingerprint, ...base } = value as CompositeNeuronV1
  return { ...base, fingerprint: computeCompositeNeuronFingerprint(base) }
}

function finalizeHead(value: Omit<MacroHeadV1, 'fingerprint'> | MacroHeadV1): MacroHeadV1 {
  const { fingerprint: _fingerprint, ...base } = value as MacroHeadV1
  return { ...base, fingerprint: computeMacroHeadFingerprint(base) }
}

export function createMacroHead(
  reference: MacroHeadReferenceV1,
  configOverrides: Partial<MacroConfigV1> = {},
): MacroHeadV1 {
  if (!reference.representationSpaceId.trim()) {
    throw new TypeError('Makro head için temsil uzayı kimliği gerekir.')
  }
  const config = createMacroConfig(configOverrides)
  const configFingerprint = computeMacroConfigFingerprint(config)
  const base: Omit<MacroHeadV1, 'fingerprint'> = {
    format: 'concept-macro-head',
    version: 1,
    id: macroDigest(
      'CMHEAD',
      reference.representationSpaceId,
      reference.familyRunFingerprint,
      configFingerprint,
    ),
    revision: 0,
    parentFingerprint: null,
    representationSpaceId: reference.representationSpaceId,
    familyRunFingerprint: reference.familyRunFingerprint,
    config,
    configFingerprint,
    macros: [],
    processedTraceKeys: [],
    processedSemanticTraceKeys: [],
    traceReplayBloom: createMacroReplayBloom(
      config.replayBloomBitCount,
      config.replayBloomHashCount,
    ),
    semanticReplayBloom: createMacroReplayBloom(
      config.replayBloomBitCount,
      config.replayBloomHashCount,
    ),
    rawConstituentsRetained: true,
    stats: {
      processedTraceCount: 0,
      acceptedObservationCount: 0,
      duplicateTraceCount: 0,
      heavyHitterReplacementCount: 0,
      lifecycleTransitionCount: 0,
    },
  }
  return finalizeHead(base)
}

function candidateToComposite(
  candidate: MacroMotifCandidateV1,
  knownMacros: readonly CompositeNeuronV1[],
): CompositeNeuronV1 {
  const id = compositeId(candidate)
  const variants = canonicalVariants(candidate.variants, Number.POSITIVE_INFINITY)
  const representative = variants[0]
  if (!representative) throw new Error('Makro adayı en az bir concrete genişletme taşımalıdır.')
  const nestedDepths = representative.orderedExpansion
    .filter((member) => member.kind === 'composite-neuron')
    .map((member) => {
      const child = knownMacros.find((macro) => macro.id === member.id)
      if (!child) throw new Error(`Makro adayı bilinmeyen nested üyeye başvuruyor: ${member.id}`)
      return child.depth
    })
  const depth = 1 + (nestedDepths.length ? Math.max(...nestedDepths) : 0)
  const base: Omit<CompositeNeuronV1, 'fingerprint'> = {
    format: 'composite-neuron-v1',
    version: 1,
    id,
    kind: candidate.kind,
    executionSemantics: candidate.executionSemantics,
    lifecycle: candidate.lifecycle,
    exactSignature: candidate.exactSignature,
    familySignature: candidate.familySignature,
    orderedExpansion: representative.orderedExpansion,
    expansionDigest: representative.expansionDigest,
    variants,
    depth,
    entryPrototype: candidate.entryPrototype,
    exitPrototype: candidate.exitPrototype,
    trajectoryPrototype: [...candidate.trajectoryPrototype],
    effectPrototype: [...candidate.effectPrototype],
    effectCohesion: candidate.effectCohesion,
    prototypeGroups: candidate.prototypeGroups,
    prototypeGroupCount: candidate.prototypeGroupCount,
    prototypeObservationCount: candidate.prototypeObservationCount,
    evidence: canonicalEvidence(candidate.evidence, candidate.evidence.length),
    observationCount: candidate.observationCount,
    retainedEvidenceCount: candidate.retainedEvidenceCount,
    independentGroupCount: candidate.independentGroupCount,
    sourceCount: candidate.sourceCount,
    episodeCount: candidate.episodeCount,
    runKeyCount: candidate.runKeyCount,
    mdl: candidate.mdl,
    metrics: emptyShadowMetrics(candidate.mdl.compressionGain),
    shadowReportIds: [],
    shadowCaseIds: [],
    shadowCaseEvidence: [],
    structuralEquivalencePassed: false,
    structuralEquivalenceAudit: null,
    physicalIdentityAllocated: false,
    rawConstituentsRetained: true,
    deprecatedReason: null,
  }
  return finalizeComposite(base)
}

function mergeCompositeCandidate(
  current: CompositeNeuronV1,
  candidate: MacroMotifCandidateV1,
  config: MacroConfigV1,
  knownMacros: readonly CompositeNeuronV1[],
): CompositeNeuronV1 {
  if (current.kind !== candidate.kind || current.id !== compositeId(candidate)) {
    throw new Error('Uyumsuz makro adayları aynı bileşik nöronda birleştirilemez.')
  }
  if (current.executionSemantics !== CONCEPT_MACRO_EXECUTION_SEMANTICS ||
    candidate.executionSemantics !== CONCEPT_MACRO_EXECUTION_SEMANTICS) {
    throw new Error('Uyumsuz makro yürütme semantiği birleştirilemez.')
  }
  const observationCount = current.observationCount + candidate.observationCount
  const evidence = canonicalEvidence(
    [...current.evidence, ...candidate.evidence],
    config.maximumEvidencePerCandidate,
  )
  const variants = canonicalVariants(
    [...current.variants, ...candidate.variants],
    config.maximumVariantsPerCandidate,
  )
  const representative = variants[0]
  if (!representative) throw new Error('Birleştirilen makro varyantsız kalamaz.')
  const prototypeGroups = mergeMacroPrototypeGroups(
    [...current.prototypeGroups, ...candidate.prototypeGroups],
    config.maximumPrototypeGroupsPerCandidate,
  )
  const prototype = aggregateMacroPrototypeGroups(prototypeGroups)
  const mdl = computeMdlMetrics(observationCount, representative.orderedExpansion.length, config.macroDefinitionOverhead)
  const groupCount = Math.max(
    current.independentGroupCount,
    uniqueCount(evidence.map((value) => value.independenceGroupId)),
  )
  const sourceCount = Math.max(current.sourceCount, uniqueCount(evidence.map((value) => value.sourceId)))
  const episodeCount = Math.max(current.episodeCount, uniqueCount(evidence.map((value) => value.episodeId)))
  const runKeyCount = Math.max(current.runKeyCount, uniqueCount(evidence.map((value) => value.runKey)))
  const effectCohesion = prototype.effectCohesion
  const thresholdView: MacroMotifCandidateV1 = {
    ...candidate,
    observationCount,
    independentGroupCount: groupCount,
    variants,
    mdl,
    effectCohesion,
  }
  let lifecycle = current.lifecycle
  if (lifecycle === 'observing' && meetsMacroProposalThresholds(thresholdView, config)) {
    lifecycle = 'eligible-proposal'
  }
  const nestedDepths = representative.orderedExpansion
    .filter((member) => member.kind === 'composite-neuron')
    .map((member) => {
      const child = knownMacros.find((macro) => macro.id === member.id)
      if (!child) throw new Error(`Makro adayı bilinmeyen nested üyeye başvuruyor: ${member.id}`)
      return child.depth
    })
  const depth = 1 + (nestedDepths.length ? Math.max(...nestedDepths) : 0)
  const nextDefinition = {
    ...current,
    familySignature: candidate.kind === 'sequence-macro'
      ? candidate.familySignature
      : current.familySignature,
    orderedExpansion: representative.orderedExpansion,
    expansionDigest: representative.expansionDigest,
    variants,
    depth,
    entryPrototype: createSignalSignature(prototype.entry),
    exitPrototype: createSignalSignature(prototype.exit),
    trajectoryPrototype: prototype.trajectory,
    effectPrototype: prototype.effect,
    effectCohesion,
    prototypeGroups,
    prototypeGroupCount: prototypeGroups.length,
    prototypeObservationCount: prototype.observationCount,
  }
  const executionChanged = computeMacroExecutionFingerprint(nextDefinition) !==
    computeMacroExecutionFingerprint(current)
  const requiresFreshShadow = executionChanged && (
    current.lifecycle === 'virtual-provisional' ||
    current.lifecycle === 'shadow-validated' ||
    current.lifecycle === 'active'
  )
  if (requiresFreshShadow) lifecycle = 'virtual-provisional'
  return finalizeComposite({
    ...current,
    ...nextDefinition,
    lifecycle,
    evidence,
    observationCount,
    retainedEvidenceCount: evidence.length,
    independentGroupCount: groupCount,
    sourceCount,
    episodeCount,
    runKeyCount,
    mdl,
    metrics: requiresFreshShadow
      ? emptyShadowMetrics(mdl.compressionGain)
      : { ...current.metrics, compressionGain: mdl.compressionGain },
    shadowReportIds: requiresFreshShadow ? [] : current.shadowReportIds,
    shadowCaseIds: requiresFreshShadow ? [] : current.shadowCaseIds,
    shadowCaseEvidence: requiresFreshShadow ? [] : current.shadowCaseEvidence,
    structuralEquivalencePassed: requiresFreshShadow
      ? false
      : current.structuralEquivalencePassed,
    structuralEquivalenceAudit: requiresFreshShadow
      ? null
      : current.structuralEquivalenceAudit,
  })
}

function commitHead(
  current: MacroHeadV1,
  changes: Pick<MacroHeadV1, 'macros' | 'processedTraceKeys' | 'stats'> &
    Partial<Pick<MacroHeadV1, 'processedSemanticTraceKeys'>> &
    Partial<Pick<MacroHeadV1, 'familyRunFingerprint'>>,
): MacroHeadV1 {
  const traceLedger = boundedMacroReplayLedger(
    current.traceReplayBloom,
    current.processedTraceKeys,
    changes.processedTraceKeys,
    current.config.maximumExactReplayKeys,
  )
  const semanticLedger = boundedMacroReplayLedger(
    current.semanticReplayBloom,
    current.processedSemanticTraceKeys,
    changes.processedSemanticTraceKeys ?? current.processedSemanticTraceKeys,
    current.config.maximumExactReplayKeys,
  )
  const base: Omit<MacroHeadV1, 'fingerprint'> = {
    ...current,
    revision: current.revision + 1,
    parentFingerprint: current.fingerprint,
    familyRunFingerprint: changes.familyRunFingerprint ?? current.familyRunFingerprint,
    macros: [...changes.macros].sort((left, right) => compareMacroText(left.id, right.id)),
    processedTraceKeys: traceLedger.exactKeys,
    processedSemanticTraceKeys: semanticLedger.exactKeys,
    traceReplayBloom: traceLedger.bloom,
    semanticReplayBloom: semanticLedger.bloom,
    stats: changes.stats,
  }
  const next = finalizeHead(base)
  validateMacroHead(next)
  return next
}

export function validateCompositeNeuron(composite: CompositeNeuronV1, config: MacroConfigV1): void {
  const { fingerprint, ...base } = composite
  if (composite.format !== 'composite-neuron-v1' || composite.version !== 1 || !composite.id) {
    throw new TypeError('Geçersiz CompositeNeuronV1 kaydı.')
  }
  if (composite.executionSemantics !== CONCEPT_MACRO_EXECUTION_SEMANTICS) {
    throw new TypeError('Bileşik nöron terminal-anchor-v1 yürütme kanıtı taşımıyor.')
  }
  if (computeCompositeNeuronFingerprint(base) !== fingerprint) {
    throw new Error(`Bileşik nöron fingerprint uyuşmazlığı: ${composite.id}`)
  }
  if (composite.physicalIdentityAllocated !== false || composite.rawConstituentsRetained !== true) {
    throw new Error('V1 makroları sanal olmalı ve ham üyelerini korumalıdır.')
  }
  if (!Number.isInteger(composite.observationCount) || composite.observationCount < 1 ||
    !Number.isInteger(composite.retainedEvidenceCount) ||
    composite.retainedEvidenceCount !== composite.evidence.length ||
    composite.retainedEvidenceCount > composite.observationCount ||
    composite.evidence.length > config.maximumEvidencePerCandidate) {
    throw new RangeError('Bileşik nöron gözlem sayaçları bozuk.')
  }
  if (new Set(composite.evidence.map((value) => value.observationId)).size !== composite.evidence.length) {
    throw new Error('Bileşik nöron kanıtları benzersiz olmalıdır.')
  }
  if (composite.evidence.some((value) => value.split !== 'train')) {
    throw new Error('Bileşik nöron lifecycle kanıtı yalnız train bölümünden gelebilir.')
  }
  if (new Set(composite.shadowReportIds).size !== composite.shadowReportIds.length) {
    throw new Error('Shadow rapor kimlikleri benzersiz olmalıdır.')
  }
  if (new Set(composite.shadowCaseIds).size !== composite.shadowCaseIds.length) {
    throw new Error('Shadow vaka kimlikleri benzersiz olmalıdır.')
  }
  if (composite.shadowCaseEvidence.length !== composite.shadowCaseIds.length ||
    composite.shadowCaseEvidence.some((value, index) => value.caseId !== composite.shadowCaseIds[index])) {
    throw new Error('Bileşik nöron shadow vaka kaynak ledgerı bozuk.')
  }
  if (composite.entryPrototype.digest !== createSignalSignature(composite.entryPrototype.vector).digest ||
    composite.exitPrototype.digest !== createSignalSignature(composite.exitPrototype.vector).digest) {
    throw new Error('Makro giriş/çıkış prototip digest uyuşmazlığı.')
  }
  if (composite.trajectoryPrototype.length !== composite.entryPrototype.dimension ||
    composite.effectPrototype.length !== composite.entryPrototype.dimension ||
    composite.exitPrototype.dimension !== composite.entryPrototype.dimension) {
    throw new RangeError('Makro prototip boyutları uyuşmuyor.')
  }
  if (!Number.isInteger(composite.prototypeGroupCount) ||
    composite.prototypeGroupCount !== composite.prototypeGroups.length ||
    composite.prototypeGroupCount < 1 ||
    composite.prototypeGroupCount > config.maximumPrototypeGroupsPerCandidate ||
    !Number.isInteger(composite.prototypeObservationCount) ||
    composite.prototypeObservationCount < composite.prototypeGroupCount ||
    new Set(composite.prototypeGroups.map((group) => group.independenceGroupId)).size !==
      composite.prototypeGroups.length) {
    throw new Error('Makro kaynak-dengeli prototip grup ledgerı bozuk.')
  }
  for (const group of composite.prototypeGroups) validateMacroPrototypeGroup(group)
  const aggregate = aggregateMacroPrototypeGroups(composite.prototypeGroups)
  if (aggregate.observationCount !== composite.prototypeObservationCount ||
    createSignalSignature(aggregate.entry).digest !== composite.entryPrototype.digest ||
    createSignalSignature(aggregate.exit).digest !== composite.exitPrototype.digest ||
    macroDigest('MPROTO', aggregate.trajectory) !==
      macroDigest('MPROTO', composite.trajectoryPrototype) ||
    macroDigest('MPROTO', aggregate.effect) !== macroDigest('MPROTO', composite.effectPrototype) ||
    Math.abs(aggregate.effectCohesion - composite.effectCohesion) > 1e-6) {
    throw new Error('Makro üst prototipi kaynak grup ortalamalarıyla uyuşmuyor.')
  }
  if (typeof composite.structuralEquivalencePassed !== 'boolean') {
    throw new TypeError('Makro yapısal eşdeğerlik kapısı boolean olmalıdır.')
  }
  if (composite.structuralEquivalenceAudit) {
    validateMacroStructuralEquivalenceAudit(composite.structuralEquivalenceAudit)
    if (composite.structuralEquivalenceAudit.compositeId !== composite.id ||
      composite.structuralEquivalenceAudit.compositeExecutionFingerprint !==
        computeMacroExecutionFingerprint(composite) ||
      composite.structuralEquivalencePassed !== composite.structuralEquivalenceAudit.passed) {
      throw new Error('Makro yapısal eşdeğerlik denetimi yürütme tanımıyla uyuşmuyor.')
    }
  } else if (composite.structuralEquivalencePassed) {
    throw new Error('Audit olmadan yapısal eşdeğerlik kapısı geçilemez.')
  }
  if (composite.lifecycle === 'active' && !composite.structuralEquivalencePassed) {
    throw new Error('Aktif makro yapısal eşdeğerlik kanıtı taşımıyor.')
  }
  const unitMetrics = [
    composite.effectCohesion,
    composite.metrics.falseActivationRate,
    composite.metrics.activationRecall,
    composite.metrics.fidelity,
  ]
  if (unitMetrics.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new RangeError('Makro cohesion/fidelity/hata metrikleri [0,1] aralığında olmalıdır.')
  }
  if (composite.metrics.compressionGain !== composite.mdl.compressionGain) {
    throw new Error('Makro compressionGain değeri MDL kaydı ile uyuşmuyor.')
  }
  const representative = composite.orderedExpansion
  const expectedExact = computeExactSignature(representative)
  const expectedFamily = computeFamilySignature(representative)
  if ((composite.kind === 'sequence-macro' && composite.exactSignature !== expectedExact) ||
    (composite.kind === 'equivalence-concept' && composite.exactSignature !== null) ||
    composite.familySignature !== expectedFamily) {
    throw new Error('Makro exact/family imzası sıralı genişletmesiyle uyuşmuyor.')
  }
  const finiteMetrics = [
    composite.metrics.shadowCompressionGain,
    composite.metrics.holdoutUtility,
  ]
  if (finiteMetrics.some((value) => !Number.isFinite(value))) {
    throw new Error('Makro kümülatif shadow metrikleri sonlu olmalıdır.')
  }
  const countMetrics = [
    composite.metrics.shadowCaseCount,
    composite.metrics.holdoutUtilityCaseCount,
    composite.metrics.falseActivationCount,
    composite.metrics.negativeCaseCount,
    composite.metrics.positiveCaseCount,
    composite.metrics.missedActivationCount,
  ]
  if (countMetrics.some((value) => !Number.isInteger(value) || value < 0) ||
    composite.metrics.falseActivationCount > composite.metrics.negativeCaseCount ||
    composite.metrics.missedActivationCount > composite.metrics.positiveCaseCount ||
    composite.shadowCaseIds.length !== composite.metrics.shadowCaseCount) {
    throw new Error('Makro kümülatif shadow sayaçları bozuk.')
  }
  canonicalVariants(composite.variants, config.maximumVariantsPerCandidate)
}

export function validateMacroHead(head: MacroHeadV1): void {
  if (head.format !== 'concept-macro-head' || head.version !== 1 || !head.id ||
    !head.representationSpaceId || head.rawConstituentsRetained !== true) {
    throw new TypeError('Geçersiz MacroHeadV1 kaydı.')
  }
  validateMacroConfig(head.config)
  if (computeMacroConfigFingerprint(head.config) !== head.configFingerprint) {
    throw new Error('Makro head config fingerprint uyuşmazlığı.')
  }
  if (head.macros.length > head.config.maximumComposites) {
    throw new RangeError('Makro head yapılandırılmış bileşik kapasitesini aşıyor.')
  }
  if (head.processedTraceKeys.length > head.config.maximumExactReplayKeys ||
    head.processedSemanticTraceKeys.length > head.config.maximumExactReplayKeys) {
    throw new RangeError('Makro exact replay ledger kapasitesini aşıyor.')
  }
  validateMacroReplayBloom(head.traceReplayBloom)
  validateMacroReplayBloom(head.semanticReplayBloom)
  if (head.traceReplayBloom.bitCount !== head.config.replayBloomBitCount ||
    head.semanticReplayBloom.bitCount !== head.config.replayBloomBitCount ||
    head.traceReplayBloom.hashCount !== head.config.replayBloomHashCount ||
    head.semanticReplayBloom.hashCount !== head.config.replayBloomHashCount) {
    throw new Error('Makro replay Bloom özeti config ile uyuşmuyor.')
  }
  if (new Set(head.processedTraceKeys).size !== head.processedTraceKeys.length ||
    new Set(head.processedSemanticTraceKeys).size !== head.processedSemanticTraceKeys.length ||
    new Set(head.macros.map((macro) => macro.id)).size !== head.macros.length) {
    throw new Error('Makro head kimlik listeleri benzersiz olmalıdır.')
  }
  for (const macro of head.macros) validateCompositeNeuron(macro, head.config)
  validateCompositeDag(head.macros, head.config.maximumNestingDepth)
  const { fingerprint, ...base } = head
  if (computeMacroHeadFingerprint(base) !== fingerprint) throw new Error('Makro head fingerprint uyuşmazlığı.')
}

export function ingestMacroTraces(
  head: MacroHeadV1,
  traces: readonly MacroTraceV1[],
): MacroMiningIngestResultV1 {
  validateMacroHead(head)
  for (const trace of traces) {
    if (trace.split !== 'train') {
      throw new Error('Validation/test izi makro lifecycle eğitim kanıtına alınamaz.')
    }
    if (trace.representationSpaceId !== head.representationSpaceId) {
      throw new Error('Farklı temsil uzayındaki iz MacroHead içine alınamaz.')
    }
    if (trace.familyRunFingerprint !== head.familyRunFingerprint) {
      throw new Error('FamilyRun değişti; iz ingest edilmeden önce rebaseMacroHeadReference çağrılmalıdır.')
    }
  }
  const mining = mineMacroMotifs(
    traces,
    head.config,
    head.processedTraceKeys,
    head.processedSemanticTraceKeys,
    {
      traceBloom: head.traceReplayBloom,
      semanticBloom: head.semanticReplayBloom,
    },
  )
  if (!mining.processedTraceKeys.length) return { head, mining }

  const macros = [...head.macros]
  const byId = new Map(macros.map((macro) => [macro.id, macro]))
  let ingestLifecycleTransitionCount = 0
  for (const candidate of mining.candidates) {
    const id = compositeId(candidate)
    const existing = byId.get(id)
    if (existing) {
      const merged = mergeCompositeCandidate(existing, candidate, head.config, [...byId.values()])
      if (merged.lifecycle !== existing.lifecycle) ingestLifecycleTransitionCount += 1
      byId.set(id, merged)
      continue
    }
    if (byId.size >= head.config.maximumComposites) continue
    const created = candidateToComposite(candidate, [...byId.values()])
    if (created.depth > head.config.maximumNestingDepth) continue
    byId.set(created.id, created)
  }
  // Parent macros bind a nested child by its exact expansion digest. If a
  // child's representative changes, the old parent definition is retained as
  // audit evidence but cannot silently expand to a new raw path.
  let nestedInvalidationChanged = true
  while (nestedInvalidationChanged) {
    nestedInvalidationChanged = false
    for (const macro of [...byId.values()].sort((left, right) => compareMacroText(left.id, right.id))) {
      if (macro.lifecycle === 'deprecated') continue
      const expansions = [macro.orderedExpansion, ...macro.variants.map((variant) => variant.orderedExpansion)]
      const invalid = expansions.some((expansion) => expansion.some((reference) => {
        if (reference.kind !== 'composite-neuron') return false
        const child = byId.get(reference.id)
        return !child || child.lifecycle === 'deprecated' || reference.identityDigest !== child.expansionDigest
      }))
      if (!invalid) continue
      byId.set(macro.id, finalizeComposite({
        ...macro,
        lifecycle: 'deprecated',
        deprecatedReason: 'nested-child-expansion-changed',
      }))
      ingestLifecycleTransitionCount += 1
      nestedInvalidationChanged = true
    }
  }
  const next = commitHead(head, {
    macros: [...byId.values()],
    processedTraceKeys: [...head.processedTraceKeys, ...mining.processedTraceKeys],
    processedSemanticTraceKeys: [
      ...head.processedSemanticTraceKeys,
      ...mining.processedSemanticTraceKeys,
    ],
    stats: {
      ...head.stats,
      processedTraceCount: head.stats.processedTraceCount + mining.diagnostics.processedTraceCount,
      acceptedObservationCount: head.stats.acceptedObservationCount +
        mining.diagnostics.acceptedObservationCount,
      duplicateTraceCount: head.stats.duplicateTraceCount + mining.diagnostics.duplicateTraceCount,
      heavyHitterReplacementCount: head.stats.heavyHitterReplacementCount +
        mining.diagnostics.heavyHitterReplacementCount,
      lifecycleTransitionCount: head.stats.lifecycleTransitionCount + ingestLifecycleTransitionCount,
    },
  })
  return { head: next, mining }
}

const NEXT_LIFECYCLES: Readonly<Record<MacroLifecycle, readonly MacroLifecycle[]>> = Object.freeze({
  observing: ['eligible-proposal', 'deprecated'],
  'eligible-proposal': ['virtual-provisional', 'deprecated'],
  'virtual-provisional': ['shadow-validated', 'deprecated'],
  'shadow-validated': ['active', 'deprecated'],
  active: ['deprecated'],
  deprecated: [],
})

function passesShadowLifecycleGate(
  composite: CompositeNeuronV1,
  config: MacroConfigV1,
): boolean {
  return composite.effectCohesion >= config.minimumEffectCohesion &&
    composite.metrics.shadowCaseCount >= config.minimumShadowCases &&
    new Set(composite.shadowCaseEvidence.map((value) => value.independenceGroupId)).size >=
      config.minimumShadowIndependentGroups &&
    composite.metrics.positiveCaseCount >= config.minimumShadowPositiveCases &&
    composite.metrics.negativeCaseCount >= config.minimumShadowNegativeCases &&
    composite.metrics.fidelity >= config.minimumShadowFidelity &&
    composite.metrics.falseActivationRate <= config.maximumFalseActivationRate &&
    composite.metrics.activationRecall >= config.minimumActivationRecall &&
    composite.metrics.holdoutUtility >= config.minimumHoldoutUtility &&
    composite.metrics.shadowCompressionGain >= config.minimumShadowCompressionGain
}

export function transitionCompositeLifecycle(
  head: MacroHeadV1,
  compositeIdValue: string,
  nextLifecycle: MacroLifecycle,
  reason: string | null = null,
): MacroHeadV1 {
  validateMacroHead(head)
  const current = head.macros.find((macro) => macro.id === compositeIdValue)
  if (!current) throw new Error(`Bilinmeyen bileşik nöron: ${compositeIdValue}`)
  if (current.lifecycle === nextLifecycle) return head
  if (!NEXT_LIFECYCLES[current.lifecycle].includes(nextLifecycle)) {
    throw new Error(`Geçersiz makro yaşam döngüsü geçişi: ${current.lifecycle} -> ${nextLifecycle}`)
  }
  if (nextLifecycle === 'eligible-proposal') {
    const thresholdView = {
      kind: current.kind,
      observationCount: current.observationCount,
      independentGroupCount: current.independentGroupCount,
      variants: current.variants,
      mdl: current.mdl,
      effectCohesion: current.effectCohesion,
    }
    if (!meetsMacroProposalThresholds(thresholdView, head.config)) {
      throw new Error('Makro henüz öneri eşiklerini karşılamıyor.')
    }
  }
  if ((nextLifecycle === 'shadow-validated' || nextLifecycle === 'active') &&
    !passesShadowLifecycleGate(current, head.config)) {
    throw new Error('Makro shadow doğrulama kapısını geçmedi.')
  }
  if (nextLifecycle === 'active' && !current.structuralEquivalencePassed) {
    throw new Error('Makro yapısal yürütme eşdeğerliği doğrulanmadan active olamaz.')
  }
  const updated = finalizeComposite({
    ...current,
    lifecycle: nextLifecycle,
    deprecatedReason: nextLifecycle === 'deprecated' ? (reason?.trim() || 'unspecified') : null,
  })
  return commitHead(head, {
    macros: head.macros.map((macro) => macro.id === current.id ? updated : macro),
    processedTraceKeys: head.processedTraceKeys,
    stats: {
      ...head.stats,
      lifecycleTransitionCount: head.stats.lifecycleTransitionCount + 1,
    },
  })
}

export function applyMacroStructuralEquivalenceAudit(
  head: MacroHeadV1,
  audit: MacroStructuralEquivalenceAuditV1,
): MacroHeadV1 {
  validateMacroHead(head)
  validateMacroStructuralEquivalenceAudit(audit)
  const current = head.macros.find((macro) => macro.id === audit.compositeId)
  if (!current) throw new Error(`Yapısal denetim bilinmeyen makroya ait: ${audit.compositeId}`)
  if (current.lifecycle !== 'shadow-validated') {
    throw new Error('Yapısal eşdeğerlik denetimi yalnız shadow-validated makroya uygulanabilir.')
  }
  if (audit.compositeExecutionFingerprint !== computeMacroExecutionFingerprint(current)) {
    throw new Error('Yapısal denetim değişmiş bir makro yürütme tanımına ait.')
  }
  if (current.structuralEquivalenceAudit?.id === audit.id) return head
  const updated = finalizeComposite({
    ...current,
    structuralEquivalencePassed: audit.passed,
    structuralEquivalenceAudit: audit,
  })
  return commitHead(head, {
    macros: head.macros.map((macro) => macro.id === current.id ? updated : macro),
    processedTraceKeys: head.processedTraceKeys,
    stats: head.stats,
  })
}

export function applyMacroShadowEvaluation(
  head: MacroHeadV1,
  report: MacroShadowEvaluationV1,
): MacroHeadV1 {
  validateMacroHead(head)
  validateMacroShadowEvaluation(report, head.config)
  const current = head.macros.find((macro) => macro.id === report.compositeId)
  if (!current) throw new Error(`Shadow raporu bilinmeyen makroya ait: ${report.compositeId}`)
  if (report.compositeExecutionFingerprint !== computeMacroExecutionFingerprint(current)) {
    throw new Error('Shadow raporu değişmiş bir makro yürütme tanımına ait; yeniden ölçülmelidir.')
  }
  if (current.shadowReportIds.includes(report.id)) return head
  if (current.lifecycle !== 'virtual-provisional') {
    throw new Error('Shadow raporu yalnız virtual-provisional makroya uygulanabilir.')
  }
  if (report.caseIds.some((caseId) => current.shadowCaseIds.includes(caseId))) {
    throw new Error('Yeni shadow raporu daha önce sayılmış vaka kimlikleri içeriyor.')
  }
  const previous = current.metrics
  const shadowCaseCount = previous.shadowCaseCount + report.caseCount
  const holdoutUtilityCaseCount = previous.holdoutUtilityCaseCount + report.utilityCaseCount
  const falseActivationCount = previous.falseActivationCount + report.falseActivationCount
  const negativeCaseCount = previous.negativeCaseCount + report.negativeCaseCount
  const positiveCaseCount = previous.positiveCaseCount + report.positiveCaseCount
  const missedActivationCount = previous.missedActivationCount + report.missedActivationCount
  const metrics = {
    compressionGain: current.mdl.compressionGain,
    shadowCompressionGain: canonicalFloat(
      previous.shadowCompressionGain + report.observedCompressionGain,
    ),
    holdoutUtility: canonicalFloat(holdoutUtilityCaseCount
      ? (previous.holdoutUtility * previous.holdoutUtilityCaseCount +
        report.holdoutUtility * report.utilityCaseCount) / holdoutUtilityCaseCount
      : 0),
    holdoutUtilityCaseCount,
    falseActivationRate: canonicalFloat(negativeCaseCount
      ? falseActivationCount / negativeCaseCount
      : 0),
    falseActivationCount,
    negativeCaseCount,
    activationRecall: canonicalFloat(positiveCaseCount
      ? (positiveCaseCount - missedActivationCount) / positiveCaseCount
      : 0),
    positiveCaseCount,
    missedActivationCount,
    fidelity: canonicalFloat(shadowCaseCount
      ? (previous.fidelity * previous.shadowCaseCount + report.fidelity * report.caseCount) /
        shadowCaseCount
      : 0),
    shadowCaseCount,
  }
  const shadowCaseEvidence = [...current.shadowCaseEvidence, ...report.caseEvidence]
    .sort((left, right) => compareMacroText(left.caseId, right.caseId))
  const aggregatePassed = passesShadowLifecycleGate({
    ...current,
    metrics,
    shadowCaseEvidence,
  }, head.config)
  const updated = finalizeComposite({
    ...current,
    lifecycle: aggregatePassed ? 'shadow-validated' : current.lifecycle,
    metrics,
    shadowReportIds: [...current.shadowReportIds, report.id].sort(compareMacroText),
    shadowCaseIds: [...current.shadowCaseIds, ...report.caseIds].sort(compareMacroText),
    shadowCaseEvidence,
  })
  return commitHead(head, {
    macros: head.macros.map((macro) => macro.id === current.id ? updated : macro),
    processedTraceKeys: head.processedTraceKeys,
    stats: {
      ...head.stats,
      lifecycleTransitionCount: head.stats.lifecycleTransitionCount + Number(aggregatePassed),
    },
  })
}

/**
 * Updates a FamilyRun reference without rewriting raw identity. Exact digest
 * mismatches are retained as deprecated records, never silently migrated.
 */
export function rebaseMacroHeadReference(
  head: MacroHeadV1,
  catalog: MacroReferenceCatalogV1,
): MacroHeadRebaseResultV1 {
  validateMacroHead(head)
  if (catalog.representationSpaceId !== head.representationSpaceId) {
    throw new Error('Temsil uzayı değişti; otomatik makro migration yasaktır.')
  }
  const members = new Map<string, { identityDigest: string; familyIds: string[] }>()
  for (const member of catalog.members) {
    if (!member.id || !member.identityDigest || members.has(member.id)) {
      throw new Error('Rebase üye kataloğu benzersiz kimlik ve identity digest taşımalıdır.')
    }
    members.set(member.id, {
      identityDigest: member.identityDigest,
      familyIds: canonicalFamilyIds(member.familyIds),
    })
  }
  const preservedMacroIds: string[] = []
  const deprecatedMacroIds: string[] = []
  const familyChangedMacroIds: string[] = []
  let revalidationTransitionCount = 0
  const exactValidity = new Map<string, boolean>()
  const macroById = new Map(head.macros.map((macro) => [macro.id, macro]))
  const visiting = new Set<string>()
  const exactValid = (macro: CompositeNeuronV1): boolean => {
    const memo = exactValidity.get(macro.id)
    if (memo !== undefined) return memo
    if (visiting.has(macro.id)) return false
    visiting.add(macro.id)
    const expansionValid = (expansion: CompositeNeuronV1['orderedExpansion']) => expansion.every((reference) => {
      if (reference.kind === 'raw-neuron') {
        const next = members.get(reference.id)
        return Boolean(next && reference.identityDigest && next.identityDigest === reference.identityDigest)
      }
      const nested = macroById.get(reference.id)
      return Boolean(nested && exactValid(nested))
    })
    const valid = expansionValid(macro.orderedExpansion) &&
      macro.variants.every((variant) => expansionValid(variant.orderedExpansion))
    visiting.delete(macro.id)
    exactValidity.set(macro.id, valid)
    return valid
  }
  const updateExpansion = (expansion: CompositeNeuronV1['orderedExpansion']) =>
    expansion.map((reference) => {
      if (reference.kind !== 'raw-neuron') return { ...reference, familyIds: [...reference.familyIds] }
      const next = members.get(reference.id)
      return next && reference.identityDigest === next.identityDigest
        ? { ...reference, familyIds: [...next.familyIds] }
        : { ...reference, familyIds: [...reference.familyIds] }
    })

  const updated = head.macros.map((macro) => {
    const valid = exactValid(macro)
    const variants = macro.variants.map((variant) => {
      const orderedExpansion = updateExpansion(variant.orderedExpansion)
      return { ...variant, orderedExpansion, expansionDigest: computeExpansionDigest(orderedExpansion) }
    })
    const orderedExpansion = updateExpansion(macro.orderedExpansion)
    const variantFamilySignatures = variants.map((variant) =>
      computeFamilySignature(variant.orderedExpansion))
    const nextFamilySignature = computeFamilySignature(orderedExpansion)
    const familyChanged = nextFamilySignature !== macro.familySignature ||
      new Set(variantFamilySignatures).size !== 1 ||
      variantFamilySignatures.some((signature) => signature !== nextFamilySignature)
    if (familyChanged) familyChangedMacroIds.push(macro.id)
    const executionDefinitionChanged = computeMacroExecutionFingerprint({
      ...macro,
      familySignature: nextFamilySignature,
      orderedExpansion,
      expansionDigest: computeExpansionDigest(orderedExpansion),
      variants,
    }) !== computeMacroExecutionFingerprint(macro)
    let lifecycle = macro.lifecycle
    let deprecatedReason = macro.deprecatedReason
    const requiresFreshShadow = lifecycle === 'virtual-provisional' ||
      lifecycle === 'shadow-validated' || lifecycle === 'active'
    if (!valid) {
      lifecycle = 'deprecated'
      deprecatedReason = 'identity-digest-mismatch-after-family-rebase'
    } else if (macro.kind === 'equivalence-concept' && familyChanged) {
      lifecycle = 'deprecated'
      deprecatedReason = 'equivalence-family-signature-changed'
    } else if (requiresFreshShadow) {
      // Even unchanged identities can acquire different H0/family routing in
      // a new FamilyRun. Old shadow evidence must not authorize that runtime.
      lifecycle = 'virtual-provisional'
      deprecatedReason = null
      if (macro.lifecycle !== 'virtual-provisional') revalidationTransitionCount += 1
    }
    if (lifecycle === 'deprecated' && macro.lifecycle !== 'deprecated') deprecatedMacroIds.push(macro.id)
    else preservedMacroIds.push(macro.id)
    return finalizeComposite({
      ...macro,
      lifecycle,
      familySignature: nextFamilySignature,
      orderedExpansion,
      expansionDigest: computeExpansionDigest(orderedExpansion),
      variants,
      metrics: lifecycle === 'virtual-provisional' && requiresFreshShadow
        ? emptyShadowMetrics(macro.mdl.compressionGain)
        : macro.metrics,
      shadowReportIds: lifecycle === 'virtual-provisional' && requiresFreshShadow
        ? []
        : macro.shadowReportIds,
      shadowCaseIds: lifecycle === 'virtual-provisional' && requiresFreshShadow
        ? []
        : macro.shadowCaseIds,
      shadowCaseEvidence: lifecycle === 'virtual-provisional' && requiresFreshShadow
        ? []
        : macro.shadowCaseEvidence,
      structuralEquivalencePassed: executionDefinitionChanged ||
        (lifecycle === 'virtual-provisional' && requiresFreshShadow)
        ? false
        : macro.structuralEquivalencePassed,
      structuralEquivalenceAudit: executionDefinitionChanged ||
        (lifecycle === 'virtual-provisional' && requiresFreshShadow)
        ? null
        : macro.structuralEquivalenceAudit,
      deprecatedReason,
    })
  })
  const next = commitHead(head, {
    familyRunFingerprint: catalog.familyRunFingerprint,
    macros: updated,
    processedTraceKeys: head.processedTraceKeys,
    stats: {
      ...head.stats,
      lifecycleTransitionCount: head.stats.lifecycleTransitionCount +
        deprecatedMacroIds.length + revalidationTransitionCount,
    },
  })
  return {
    head: next,
    preservedMacroIds: preservedMacroIds.sort(compareMacroText),
    deprecatedMacroIds: deprecatedMacroIds.sort(compareMacroText),
    familyChangedMacroIds: familyChangedMacroIds.sort(compareMacroText),
  }
}

export function macroLifecycleProgress(lifecycle: MacroLifecycle): number {
  return lifecycleRank(lifecycle)
}

export function macroDepth(head: MacroHeadV1, macroId: string): number {
  validateMacroHead(head)
  return computeCompositeDepth(macroId, head.macros)
}
