export const CONCEPT_MACRO_FORMAT = 'concept-macro-head' as const
export const CONCEPT_MACRO_VERSION = 1 as const
export const CONCEPT_MACRO_EXECUTION_SEMANTICS = 'terminal-anchor-v1' as const

export type MacroExecutionSemantics = typeof CONCEPT_MACRO_EXECUTION_SEMANTICS

export type CompositeNeuronKind = 'sequence-macro' | 'equivalence-concept'

export type MacroLifecycle =
  | 'observing'
  | 'eligible-proposal'
  | 'virtual-provisional'
  | 'shadow-validated'
  | 'active'
  | 'deprecated'

export type MacroDatasetSplit = 'train' | 'validation' | 'test'
export type MacroMemberKind = 'raw-neuron' | 'composite-neuron'

/** An ordered reference. Family ids are descriptive evidence, never an identity replacement. */
export interface MacroMemberReferenceV1 {
  kind: MacroMemberKind
  id: string
  identityDigest: string | null
  familyIds: string[]
}

export interface MacroSignalSignatureV1 {
  dimension: number
  vector: number[]
  digest: string
}

/** JSON-safe little-endian Float32 payload used only for bounded aggregate state. */
export interface MacroPackedVectorV1 {
  format: 'macro-packed-f32-v1'
  version: 1
  dimension: number
  dataBase64: string
  digest: string
}

export interface MacroPrototypeGroupV1 {
  independenceGroupId: string
  observationCount: number
  nonZeroEffectObservationCount: number
  entryMean: MacroPackedVectorV1
  exitMean: MacroPackedVectorV1
  trajectoryMean: MacroPackedVectorV1
  effectMean: MacroPackedVectorV1
  effectDirectionMean: MacroPackedVectorV1
  fingerprint: string
}

export interface MacroReplayBloomV1 {
  format: 'macro-replay-bloom-v1'
  version: 1
  bitCount: number
  hashCount: number
  insertCount: number
  bitsBase64: string
  fingerprint: string
}

export interface MacroReplayGuardV1 {
  traceBloom: MacroReplayBloomV1
  semanticBloom: MacroReplayBloomV1
}

export interface MacroStructuralEquivalenceAuditV1 {
  format: 'macro-structural-equivalence-audit-v1'
  version: 1
  id: string
  compositeId: string
  compositeExecutionFingerprint: string
  evaluator: string
  evidenceDigest: string
  caseCount: number
  passed: boolean
  fingerprint: string
}

export interface MacroEvidenceReferenceV1 {
  observationId: string
  sourceId: string
  sourceDigest: string | null
  independenceGroupId: string
  episodeId: string
  runKey: string
  semanticInputDigest: string
  semanticOccurrenceKey: string
  split: MacroDatasetSplit
  tickStart: number
  tickEnd: number
}

/** One non-overlapping occurrence extracted from a deterministic trace. */
export interface MacroObservationV1 {
  format: 'macro-observation-v1'
  version: 1
  id: string
  kind: CompositeNeuronKind
  executionSemantics: MacroExecutionSemantics
  exactSignature: string
  familySignature: string
  orderedExpansion: MacroMemberReferenceV1[]
  expansionDigest: string
  entrySignature: MacroSignalSignatureV1
  exitSignature: MacroSignalSignatureV1
  trajectory: number[]
  effect: number[]
  evidence: MacroEvidenceReferenceV1
  fingerprint: string
}

export interface MacroExpansionVariantV1 {
  exactSignature: string
  orderedExpansion: MacroMemberReferenceV1[]
  expansionDigest: string
  observationCount: number
}

export interface MacroMdlMetricsV1 {
  rawDescriptionCost: number
  macroDescriptionCost: number
  definitionCost: number
  compressionGain: number
  compressionRatio: number
}

export interface CompositeNeuronMetricsV1 {
  compressionGain: number
  shadowCompressionGain: number
  holdoutUtility: number
  holdoutUtilityCaseCount: number
  falseActivationRate: number
  falseActivationCount: number
  negativeCaseCount: number
  activationRecall: number
  positiveCaseCount: number
  missedActivationCount: number
  fidelity: number
  shadowCaseCount: number
}

/** Bounded mining output. It is a proposal aggregate, not yet a neuron. */
export interface MacroMotifCandidateV1 {
  format: 'macro-motif-candidate-v1'
  version: 1
  id: string
  kind: CompositeNeuronKind
  executionSemantics: MacroExecutionSemantics
  lifecycle: 'observing' | 'eligible-proposal'
  exactSignature: string | null
  familySignature: string
  length: number
  representativeExpansion: MacroMemberReferenceV1[]
  expansionDigest: string
  variants: MacroExpansionVariantV1[]
  entryPrototype: MacroSignalSignatureV1
  exitPrototype: MacroSignalSignatureV1
  trajectoryPrototype: number[]
  effectPrototype: number[]
  effectCohesion: number
  prototypeGroups: MacroPrototypeGroupV1[]
  prototypeGroupCount: number
  prototypeObservationCount: number
  evidence: MacroEvidenceReferenceV1[]
  observationCount: number
  retainedEvidenceCount: number
  estimatedOccurrences: number
  maximumEstimationError: number
  independentGroupCount: number
  sourceCount: number
  episodeCount: number
  runKeyCount: number
  mdl: MacroMdlMetricsV1
  fingerprint: string
}

export interface CompositeNeuronV1 {
  format: 'composite-neuron-v1'
  version: 1
  id: string
  kind: CompositeNeuronKind
  executionSemantics: MacroExecutionSemantics
  lifecycle: MacroLifecycle
  exactSignature: string | null
  familySignature: string
  orderedExpansion: MacroMemberReferenceV1[]
  expansionDigest: string
  variants: MacroExpansionVariantV1[]
  depth: number
  entryPrototype: MacroSignalSignatureV1
  exitPrototype: MacroSignalSignatureV1
  trajectoryPrototype: number[]
  effectPrototype: number[]
  effectCohesion: number
  prototypeGroups: MacroPrototypeGroupV1[]
  prototypeGroupCount: number
  prototypeObservationCount: number
  evidence: MacroEvidenceReferenceV1[]
  observationCount: number
  retainedEvidenceCount: number
  independentGroupCount: number
  sourceCount: number
  episodeCount: number
  runKeyCount: number
  mdl: MacroMdlMetricsV1
  metrics: CompositeNeuronMetricsV1
  shadowReportIds: string[]
  shadowCaseIds: string[]
  shadowCaseEvidence: MacroShadowEvaluationV1['caseEvidence']
  structuralEquivalencePassed: boolean
  structuralEquivalenceAudit: MacroStructuralEquivalenceAuditV1 | null
  /** V1 is an overlay: no synthetic 64x64 identity is allocated. */
  physicalIdentityAllocated: false
  /** Constituents remain addressable and can always be replayed. */
  rawConstituentsRetained: true
  deprecatedReason: string | null
  fingerprint: string
}

export interface MacroConfigV1 {
  executionSemantics: MacroExecutionSemantics
  minimumMotifLength: number
  maximumMotifLength: number
  maximumHeavyHitters: number
  maximumEvidencePerCandidate: number
  maximumVariantsPerCandidate: number
  maximumPrototypeGroupsPerCandidate: number
  maximumComposites: number
  maximumVectorDimension: number
  maximumExpandedConstituents: number
  minimumProposalObservations: number
  minimumIndependentGroups: number
  minimumEquivalenceVariants: number
  minimumCompressionGain: number
  macroDefinitionOverhead: number
  minimumEffectCohesion: number
  minimumEntryCompatibility: number
  minimumShadowCases: number
  minimumShadowIndependentGroups: number
  minimumShadowPositiveCases: number
  minimumShadowNegativeCases: number
  minimumShadowFidelity: number
  maximumFalseActivationRate: number
  minimumActivationRecall: number
  minimumHoldoutUtility: number
  minimumShadowCompressionGain: number
  maximumNestingDepth: number
  maximumExactReplayKeys: number
  replayBloomBitCount: number
  replayBloomHashCount: number
}

export interface MacroHeadStatsV1 {
  processedTraceCount: number
  acceptedObservationCount: number
  duplicateTraceCount: number
  heavyHitterReplacementCount: number
  lifecycleTransitionCount: number
}

export interface MacroHeadV1 {
  format: typeof CONCEPT_MACRO_FORMAT
  version: typeof CONCEPT_MACRO_VERSION
  id: string
  revision: number
  parentFingerprint: string | null
  representationSpaceId: string
  familyRunFingerprint: string | null
  config: MacroConfigV1
  configFingerprint: string
  macros: CompositeNeuronV1[]
  /** Exact episode/run pairs make ingest replay-safe. */
  processedTraceKeys: string[]
  /** Content-level replay guard; independent of cycle/episode naming. */
  processedSemanticTraceKeys: string[]
  traceReplayBloom: MacroReplayBloomV1
  semanticReplayBloom: MacroReplayBloomV1
  rawConstituentsRetained: true
  stats: MacroHeadStatsV1
  fingerprint: string
}

export interface MacroTraceStepV1 {
  member: MacroMemberReferenceV1
  signalBefore: number[]
  signalAfter: number[]
}

export interface MacroTraceV1 {
  format: 'macro-trace-v1'
  version: 1
  id: string
  representationSpaceId: string
  familyRunFingerprint: string | null
  sourceId: string
  sourceDigest?: string | null
  independenceGroupId: string
  episodeId: string
  runKey: string
  /** Stable digest of source occurrence + offsets/path, excluding cycle/run ids. */
  semanticInputDigest: string
  split: MacroDatasetSplit
  steps: MacroTraceStepV1[]
}

export interface MacroMiningDiagnosticsV1 {
  inputTraceCount: number
  processedTraceCount: number
  duplicateTraceCount: number
  generatedWindowCount: number
  acceptedObservationCount: number
  overlappingObservationCount: number
  heavyHitterReplacementCount: number
  retainedCandidateCount: number
}

export interface MacroMiningResultV1 {
  format: 'concept-macro-observation-batch'
  version: 1
  representationSpaceId: string | null
  familyRunFingerprint: string | null
  configFingerprint: string
  processedTraceKeys: string[]
  processedSemanticTraceKeys: string[]
  candidates: MacroMotifCandidateV1[]
  diagnostics: MacroMiningDiagnosticsV1
  fingerprint: string
}

export interface MacroShadowCaseV1 {
  id: string
  split: 'validation' | 'test'
  independenceGroupId: string
  shouldActivate: boolean
  macroActivated: boolean
  expandedOutput: number[]
  macroOutput: number[]
  expectedOutput?: number[]
  expandedCost: number
  macroCost: number
}

export interface MacroShadowEvaluationV1 {
  format: 'concept-macro-shadow-report'
  version: 1
  id: string
  compositeId: string
  /** Binds evidence to the exact path/prototypes it evaluated. */
  compositeExecutionFingerprint: string
  caseIds: string[]
  caseEvidence: Array<{
    caseId: string
    split: 'validation' | 'test'
    independenceGroupId: string
  }>
  caseCount: number
  independentGroupCount: number
  validationCaseCount: number
  testCaseCount: number
  fidelity: number
  falseActivationRate: number
  falseActivationCount: number
  negativeCaseCount: number
  activationRecall: number
  positiveCaseCount: number
  missedActivationCount: number
  holdoutUtility: number
  utilityCaseCount: number
  observedCompressionGain: number
  passed: boolean
  failures: string[]
  fingerprint: string
}

export interface MacroMiningIngestResultV1 {
  head: MacroHeadV1
  mining: MacroMiningResultV1
}

export interface MacroReferenceCatalogMemberV1 {
  id: string
  identityDigest: string
  familyIds: string[]
}

export interface MacroReferenceCatalogV1 {
  representationSpaceId: string
  familyRunFingerprint: string | null
  members: MacroReferenceCatalogMemberV1[]
}

export interface MacroHeadRebaseResultV1 {
  head: MacroHeadV1
  preservedMacroIds: string[]
  deprecatedMacroIds: string[]
  familyChangedMacroIds: string[]
}

export type ConceptMacroObservationBatchV1 = MacroMiningResultV1
export type ConceptMacroShadowReportV1 = MacroShadowEvaluationV1
