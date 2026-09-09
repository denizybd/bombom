import type {
  CoalitionState,
  FamilyRun,
  FunctionalRegionId,
} from '../families/types'

export type DatasetSplit = 'train' | 'validation' | 'test'

export interface SourceGroupSplitAssignment {
  independenceGroupId: string
  split: DatasetSplit
}

export type SurfacePieceLevel =
  | 'document'
  | 'paragraph'
  | 'sentence'
  | 'clause'
  | 'token'
  | 'subword'
  | 'grapheme'

export type LanguageOriginKey = string
export type LanguageTargetKey = string

export const LANGUAGE_BOS_MEMBER = '@language/BOS/member'
export const LANGUAGE_EOS_MEMBER = '@language/EOS/member'
export const LANGUAGE_BOS_FAMILY = '@language/BOS/family'
export const LANGUAGE_EOS_FAMILY = '@language/EOS/family'

export interface LanguageCorpusSource {
  sourceId: string
  sourceDigest: string
  /**
   * Chunks of the same physical book/document share this id and are assigned
   * atomically to one split. Defaults to sourceId for ordinary sources.
   */
  independenceGroupId?: string
  title: string
  text: string
  /** Optional provenance; it is never interpreted as an instruction. */
  provenance?: {
    kind: 'human' | 'file' | 'ollama' | 'generated' | 'unknown'
    locator?: string
    generatorModel?: string
    license?: string
  }
}

export interface LanguageExperienceConfig {
  seed: number
  trainRatio: number
  validationRatio: number
  testRatio: number
  maximumPrefixPieces: number
  maximumExamplesPerSource: number
  maximumPieceBindings: number
  minimumAlignmentScore: number
  minimumSubwordGraphemes: number
  includeGraphemeExperiences: boolean
  includeHierarchyLevels: SurfacePieceLevel[]
  maximumOutgoingFamilyTransitions: number
  maximumOutgoingNeuronTransitions: number
  maximumRouteFamilies: number
  maximumRouteNeurons: number
  learningRate: number
  inhibitionRate: number
  evidenceDecay: number
  evidenceCeiling: number
  learnedFamilyWeight: number
  learnedNeuronWeight: number
  frozenFamilyPriorWeight: number
  coalitionWeight: number
  membershipWeight: number
  historyWeight: number
  repetitionPenalty: number
  routeTemperature: number
  noveltyThreshold: number
  surpriseThreshold: number
  birthMinimumObservations: number
  /** Minimum independent physical-source groups, not chunk count. */
  birthMinimumSources: number
  birthMergeSurfaceSimilarity: number
  maximumBirthProposals: number
  maximumBirthObservationsPerProposal: number
  emergentFamilySimilarityThreshold: number
  emergentFamilyMinimumMembers: number
  emergentFamilyMinimumSourceGroups: number
  emergentFamilyUtilityThreshold: number
  maximumEmergentLanguageFamilies: number
  metricTopK: number[]
}

export interface SurfaceBinding {
  memberKey: string
  score: number
  familyWeights: Array<{ familyId: string; weight: number }>
  identityDigest: string
}

export interface SurfacePiece {
  id: string
  sourceId: string
  sourceDigest: string
  level: SurfacePieceLevel
  start: number
  end: number
  text: string
  normalizedText: string
  parentId: string | null
  childIds: string[]
  bindings: SurfaceBinding[]
}

export interface LanguageExample {
  id: string
  sourceId: string
  sourceDigest: string
  split: DatasetSplit
  granularity: SurfacePieceLevel
  parentPieceId: string
  prefixPieceIds: string[]
  targetPieceId: string | null
  targetText: string
  targetNormalizedText: string
  targetBindings: SurfaceBinding[]
  eos: boolean
  /** Deterministic diagnostic negative, never used as a positive target. */
  corruptionBindings: SurfaceBinding[]
}

export interface DuplicateSourceRecord {
  canonicalSourceDigest: string
  keptSourceId: string
  rejectedSourceId: string
  reason: 'exact-canonical-duplicate' | 'normalized-surface-duplicate' | 'duplicate-source-id' | 'digest-content-conflict'
}

export interface DatasetSplitStats {
  /** Backward-friendly alias for chunkCount. */
  sourceCount: number
  chunkCount: number
  independenceGroupCount: number
  exampleCount: number
  characterCount: number
}

export interface LanguageDataset {
  format: 'language-experience-dataset'
  version: 1
  id: string
  fingerprint: string
  representationSpaceId: string
  behaviorSpaceId: string
  familyRunFingerprint: string
  configFingerprint: string
  sources: Array<LanguageCorpusSource & {
    independenceGroupId: string
    canonicalFingerprint: string
    split: DatasetSplit
  }>
  pieces: SurfacePiece[]
  examples: LanguageExample[]
  duplicates: DuplicateSourceRecord[]
  stats: Record<DatasetSplit, DatasetSplitStats>
  warnings: string[]
}

export interface LanguageTransitionEdge {
  fromKey: LanguageOriginKey
  toKey: LanguageTargetKey
  excitatoryEvidence: number
  inhibitoryEvidence: number
  updateCount: number
  sourceCount: number
  sourceDigests: string[]
  score: number
}

export interface BirthObservation {
  sourceDigest: string
  exampleId: string
  surface: string
  normalizedSurface: string
  granularity: SurfacePieceLevel
  surprise: number
  novelty: number
}

export type BirthProposalStatus = 'observing' | 'eligible'

export interface NeuronBirthProposal {
  id: string
  fingerprint: string
  status: BirthProposalStatus
  normalizedSurface: string
  representativeSurface: string
  observationCount: number
  sourceCount: number
  sourceDigests: string[]
  sourceGroupIds: string[]
  meanSurprise: number
  meanNovelty: number
  observations: BirthObservation[]
  /** A proposal only. The language engine never allocates or writes I/R. */
  requiresExternalApproval: true
}

export interface LanguageTrainingStats {
  revision: number
  epochs: number
  processedExamples: number
  correctFamilyStrengthenings: number
  correctNeuronStrengthenings: number
  inhibitedFamilyAlternatives: number
  inhibitedNeuronAlternatives: number
  unboundTargets: number
  lastDatasetFingerprint: string | null
}

export type EmergentLanguageFamilyStatus = 'observing' | 'eligible'

/**
 * A measured functional hypothesis derived from next-piece behavior. It does
 * not relabel or mutate any frozen family/member in the upstream model.
 */
export interface EmergentLanguageFamilyEvidence {
  id: string
  fingerprint: string
  status: EmergentLanguageFamilyStatus
  memberKeys: string[]
  sourceGroupCount: number
  sourceGroupIds: string[]
  transitionEvidenceCount: number
  heldOutCaseCount: number
  fullHeldOutLoss: number
  ablatedHeldOutLoss: number
  predictiveUtility: number
  outgoingSignature: Array<{ targetFamilyId: string; weight: number }>
  explanation: string
}

export interface LanguageExperienceState {
  format: 'language-experience-state'
  version: 1
  id: string
  fingerprint: string
  parentFingerprint: string | null
  revision: number
  representationSpaceId: string
  behaviorSpaceId: string
  familyRunFingerprint: string
  config: LanguageExperienceConfig
  configFingerprint: string
  familyTransitions: LanguageTransitionEdge[]
  neuronTransitions: LanguageTransitionEdge[]
  birthProposals: NeuronBirthProposal[]
  emergentLanguageFamilies: EmergentLanguageFamilyEvidence[]
  sourceGroupSplits: SourceGroupSplitAssignment[]
  trainedSourceFingerprints: string[]
  training: LanguageTrainingStats
}

export interface FamilyRouteCandidate {
  familyId: string
  region: FunctionalRegionId | 'language-control'
  probability: number
  score: number
  learnedEvidence: number
  frozenPrior: number
  coalitionActivation: number
  historyEvidence: number
}

export interface NeuronRouteCandidate {
  memberKey: string
  text: string
  identityDigest: string
  probability: number
  score: number
  familyProbability: number
  learnedEvidence: number
  membershipEvidence: number
  historyEvidence: number
  repetitionPenalty: number
}

export interface LanguageRouteContext {
  coalition?: CoalitionState | null
  recentMemberKeys: string[]
  candidateMemberKeys?: string[]
}

export interface LanguageRouteResult {
  familyCandidates: FamilyRouteCandidate[]
  neuronCandidates: NeuronRouteCandidate[]
  eosProbability: number
  unmatchedProbability: number
  selectedFamilyId: string | null
  selectedMemberKey: string | null
  selectedText: string | null
  explanation: string
}

export interface LanguageGenerationStep {
  index: number
  route: LanguageRouteResult
  emittedMemberKey: string | null
  emittedText: string | null
  eos: boolean
}

export interface LanguageGenerationResult {
  text: string
  memberKeys: string[]
  steps: LanguageGenerationStep[]
  stoppedBy: 'eos' | 'maximum-steps' | 'unmatched'
}

export interface MetricAtK {
  k: number
  accuracy: number
}

export interface LanguageMetrics {
  split: DatasetSplit
  metricScope: 'language-head-source-holdout-over-frozen-family-space'
  exampleCount: number
  boundTargetCount: number
  topK: MetricAtK[]
  meanReciprocalRank: number
  negativeLogLikelihood: number
  perplexityLike: number
  grammarCorruptionAccuracy: number
  grammarCorruptionMargin: number
  grammarCorruptionCaseCount: number
  roleSwapSensitivity: number
  roleSwapCaseCount: number
  eosAccuracy: number
  warnings: string[]
}

export interface TrainLanguageOptions {
  epochs?: number
  /** Explicitly repeat already committed source groups; false prevents accidental overweighting. */
  allowSourceReplay?: boolean
  evaluateSplits?: DatasetSplit[]
  shouldCancel?: () => boolean
  onProgress?: (event: {
    epoch: number
    processed: number
    total: number
    message: string
  }) => void
}

export interface LanguageTrainingResult {
  state: LanguageExperienceState
  metrics: Partial<Record<DatasetSplit, LanguageMetrics>>
  birthProposals: NeuronBirthProposal[]
  emergentLanguageFamilies: EmergentLanguageFamilyEvidence[]
}

export interface LanguageCheckpoint {
  format: 'language-experience-checkpoint'
  version: 1
  id: string
  stateFingerprint: string
  state: LanguageExperienceState
}

export interface LanguageAdapterInput {
  familyRun: FamilyRun
  sources: LanguageCorpusSource[]
}
