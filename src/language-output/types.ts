import type { FamilyRun } from '../families/types'
import type {
  LanguageDataset,
  LanguageExperienceState,
  SourceGroupSplitAssignment,
  SurfacePieceLevel,
} from '../language-experience/types'

export type LanguageOutputFamilyId =
  | '@language-output/family/word-order'
  | '@language-output/family/verb-tense'
  | '@language-output/family/case-marking'
  | '@language-output/family/connective-negation'
  | '@language-output/family/punctuation'
  | '@language/EOS/family'

export type LanguageOutputRole =
  | 'actor'
  | 'action'
  | 'patient'
  | 'instrument'
  | 'operator'
  | 'goal'

export type LanguageOutputFamilyKind = 'control' | 'morphology' | 'surface' | 'terminal'
export type LanguageOutputNeuronKind = 'control' | 'lexeme' | 'morpheme' | 'punctuation' | 'terminal'

export interface LanguageOutputAtlasMetadata {
  layer: 'language-output'
  placement: 'peripheral-island'
  islandId: 'manual-language-scaffold'
  ordinal: number
  /** Suggested visual radius relative to the main learned-family cloud. */
  radialScale: number
}

export interface LanguageOutputFamilyDefinition {
  id: LanguageOutputFamilyId
  label: string
  description: string
  kind: LanguageOutputFamilyKind
  order: number
  emitsSurface: boolean
  frozen: true
  acceptedSpanLevels: SurfacePieceLevel[]
  atlas: LanguageOutputAtlasMetadata
}

export interface LanguageMorphFeatures {
  lemma?: string
  partOfSpeech?: 'noun' | 'verb' | 'connector' | 'particle' | 'punctuation' | 'control'
  tense?: 'none' | 'past' | 'evidential-past' | 'present-continuous' | 'future' | 'aorist'
  person?: 1 | 2 | 3
  number?: 'singular' | 'plural'
  case?: 'none' | 'accusative' | 'dative' | 'locative' | 'ablative' | 'instrumental'
  polarity?: 'positive' | 'negative'
  properNoun?: boolean
  /**
   * Vowel-initial noun suffixes can soften a final p/ç/t/k. `auto` uses the
   * conservative native-word heuristic; learned lexical units may explicitly
   * force or suppress the mutation for exceptions.
   */
  consonantMutation?: 'auto' | 'soften' | 'none'
}

export interface LanguageOutputSeedNeuron {
  id: string
  familyId: LanguageOutputFamilyId
  kind: LanguageOutputNeuronKind
  surface: string | null
  normalizedSurface: string
  features: LanguageMorphFeatures
  allowedRoles: LanguageOutputRole[]
  priorWeight: number
  frozen: true
}

export interface LanguageOutputManualTransition {
  fromKey: string
  toKey: string
  priorEvidence: number
  explanation: string
}

export interface LanguageRoleGatewayV1 {
  id: '@language-output/gateway/role-to-word'
  virtual: true
  acceptedRoles: LanguageOutputRole[]
  explanation: string
}

export interface LanguageOutputScaffoldV1 {
  format: 'language-output-scaffold'
  version: 1
  id: string
  fingerprint: string
  seedRevision: 1
  roleGateway: LanguageRoleGatewayV1
  families: LanguageOutputFamilyDefinition[]
  seedNeurons: LanguageOutputSeedNeuron[]
  manualTransitions: LanguageOutputManualTransition[]
}

export interface LanguageOutputLearningConfig {
  learningRate: number
  inhibitionRate: number
  evidenceDecay: number
  evidenceCeiling: number
  maximumOutgoingEdges: number
  maximumMembershipsPerMember: number
  maximumNurseryProposals: number
  maximumProvisionalNeurons: number
  classifierMinimumConfidence: number
  nurseryMinimumObservations: number
  nurseryMinimumSourceGroups: number
}

export type LanguageOutputEvidenceKind = 'family-transition' | 'member-transition'

export interface LanguageOutputEvidenceEdge {
  kind: LanguageOutputEvidenceKind
  fromKey: string
  toKey: string
  excitatoryEvidence: number
  inhibitoryEvidence: number
  score: number
  updateCount: number
  sourceDigests: string[]
  sourceGroupIds: string[]
}

export interface LanguageOutputMembershipEvidence {
  memberKey: string
  familyId: LanguageOutputFamilyId
  excitatoryEvidence: number
  inhibitoryEvidence: number
  score: number
  updateCount: number
  sourceDigests: string[]
  sourceGroupIds: string[]
  classifierConfidence: number
}

export interface LanguageOutputFamilyCandidateEvidence {
  familyId: LanguageOutputFamilyId
  evidence: number
  observationCount: number
}

export type LanguageOutputNurseryStatus = 'observing' | 'eligible'

export interface LanguageOutputNurseryProposal {
  id: string
  fingerprint: string
  status: LanguageOutputNurseryStatus
  normalizedSurface: string
  representativeSurface: string
  observationCount: number
  sourceDigests: string[]
  sourceGroupIds: string[]
  exampleIds: string[]
  meanSurprise: number
  meanNovelty: number
  familyCandidates: LanguageOutputFamilyCandidateEvidence[]
  sourceProposalIds: string[]
  requiresValidation: true
}

/**
 * Reversible output-layer unit promoted from repeated surface evidence. It is
 * deliberately not a physical 64x64 I/R neuron and never mutates FamilyRun.
 */
export interface LanguageOutputProvisionalNeuronV1 {
  id: string
  fingerprint: string
  status: 'provisional'
  kind: 'surface-controller'
  normalizedSurface: string
  representativeSurface: string
  familyId: LanguageOutputFamilyId
  familyEvidence: number
  meanFamilyConfidence: number
  observationCount: number
  sourceDigests: string[]
  sourceGroupIds: string[]
  exampleIds: string[]
  sourceProposalIds: string[]
  promotedAtRevision: number
  reversible: true
  physicalIdentityAllocated: false
}

export interface LanguageOutputTrainingStats {
  revision: number
  processedExamples: number
  positiveEdgeUpdates: number
  negativeEdgeUpdates: number
  membershipUpdates: number
  nurseryUpdates: number
  provisionalNeuronPromotions: number
  skippedUnclassified: number
  lastDatasetFingerprint: string | null
}

export interface LanguageOutputHeadV1 {
  format: 'language-output-head'
  version: 1
  id: string
  fingerprint: string
  parentFingerprint: string | null
  revision: number
  scaffoldFingerprint: string
  representationSpaceId: string | null
  familyRunFingerprint: string | null
  config: LanguageOutputLearningConfig
  configFingerprint: string
  transitionEvidence: LanguageOutputEvidenceEdge[]
  membershipEvidence: LanguageOutputMembershipEvidence[]
  nursery: LanguageOutputNurseryProposal[]
  outputNeurons: LanguageOutputProvisionalNeuronV1[]
  sourceGroupSplits: SourceGroupSplitAssignment[]
  trainedDatasetFingerprints: string[]
  training: LanguageOutputTrainingStats
}

export interface LanguageOutputSeedClassification {
  familyId: LanguageOutputFamilyId
  confidence: number
  membershipEligible: boolean
  reason: string
}

export interface LanguageOutputSeedClassifierInput {
  surface: string
  normalizedSurface?: string
  granularity: SurfacePieceLevel
  eos?: boolean
}

export interface LanguageIntentSlotCandidate {
  memberKey: string
  surface: string
  score: number
}

export interface LanguageIntentSlot {
  role: LanguageOutputRole
  candidates: LanguageIntentSlotCandidate[]
}

export interface LanguageIntentV1 {
  format: 'language-output-intent'
  version: 1
  id: string
  fingerprint: string
  slots: LanguageIntentSlot[]
  tense: NonNullable<LanguageMorphFeatures['tense']>
  polarity: NonNullable<LanguageMorphFeatures['polarity']>
  sentenceKind: 'statement' | 'question' | 'exclamation'
}

export interface LanguageOutputToken {
  id: string
  kind: LanguageOutputNeuronKind
  familyId: LanguageOutputFamilyId
  surface: string
  attached: boolean
  features: LanguageMorphFeatures
  sourceMemberKey: string | null
}

export interface SentenceAssemblyState {
  intentFingerprint: string
  emittedRoles: LanguageOutputRole[]
  tokens: LanguageOutputToken[]
  nextFamilyIds: LanguageOutputFamilyId[]
  punctuationEmitted: boolean
  complete: boolean
}

export interface LanguageOutputGenerationStep {
  index: number
  familyId: LanguageOutputFamilyId
  emittedToken: LanguageOutputToken | null
  state: SentenceAssemblyState
  explanation: string
}

export interface LanguageOutputGenerationResult {
  text: string
  tokens: LanguageOutputToken[]
  steps: LanguageOutputGenerationStep[]
  stoppedBy: 'eos' | 'unmatched' | 'maximum-steps'
  intentFingerprint: string
  scaffoldFingerprint: string
  headFingerprint: string
}

export interface LanguageOutputTrainingOptions {
  allowDatasetReplay?: boolean
}

export interface LanguageOutputTrainingResult {
  head: LanguageOutputHeadV1
  processedExampleCount: number
  nurseryProposalCount: number
  provisionalNeuronCount: number
  promotedNeuronCount: number
  replaySkipped: boolean
}

export type LanguageOutputFamilyRunReference = Pick<
  FamilyRun,
  'format' | 'version' | 'fingerprint' | 'representationSpaceId'
>

export type LanguageOutputTrainingInput = {
  head: LanguageOutputHeadV1
  scaffold: LanguageOutputScaffoldV1
  dataset: LanguageDataset
  languageState: LanguageExperienceState
  familyRun?: LanguageOutputFamilyRunReference
}
