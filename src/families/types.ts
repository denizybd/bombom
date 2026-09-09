import type {
  CanonicalSource,
  ContextTrace,
  MatrixLayout,
  NodeBranch,
  SplitBoundary,
  TextLabConfig,
} from '../text-lab/domain/types'
import type { SynthesisCandidate, SynthesisConfig } from '../synthesis/types'

export type FunctionalRegionId =
  | 'content'
  | 'event'
  | 'role'
  | 'operator'
  | 'unknown'
  | 'goal'
  | 'language'
  | 'control'

export type StaticRegionVerification = 'static-hypothesis' | 'runtime-required'

export interface FunctionalRegionDefinition {
  id: FunctionalRegionId
  label: string
  description: string
  verification: StaticRegionVerification
  dormantInStaticCorpus: boolean
}

export const FUNCTIONAL_REGIONS: readonly FunctionalRegionDefinition[] = Object.freeze([
  {
    id: 'content',
    label: 'İçerik',
    description: 'Kişi, nesne, yer, özellik veya kavramsal taşıyıcı olma hipotezi.',
    verification: 'static-hypothesis',
    dormantInStaticCorpus: false,
  },
  {
    id: 'event',
    label: 'Olay / eylem',
    description: 'Katılımcıları geçici bir olay çerçevesinde bağlayan merkez olma hipotezi.',
    verification: 'static-hypothesis',
    dormantInStaticCorpus: false,
  },
  {
    id: 'role',
    label: 'Rol / ilişki',
    description: 'Bir içeriğin olay içindeki geçici görevini bağlayan ilişki hipotezi.',
    verification: 'static-hypothesis',
    dormantInStaticCorpus: false,
  },
  {
    id: 'operator',
    label: 'Operatör',
    description: 'Olayın kapsamını, zamanını, olumsuzluğunu veya kesinliğini değiştirme hipotezi.',
    verification: 'static-hypothesis',
    dormantInStaticCorpus: false,
  },
  {
    id: 'unknown',
    label: 'Belirsiz',
    description: 'Statik yapısal kanıtı tek bir işlev bölgesini güvenle seçmeye yetmeyen aile.',
    verification: 'static-hypothesis',
    dormantInStaticCorpus: false,
  },
  {
    id: 'goal',
    label: 'Amaç / değer',
    description: 'Beklenen sonuca ve tercihlere göre düşünceyi yönlendiren çalışma bölgesi.',
    verification: 'runtime-required',
    dormantInStaticCorpus: true,
  },
  {
    id: 'language',
    label: 'Dil',
    description: 'Baskın düşünceyi sıralı bir ifadeye dönüştüren çalışma bölgesi.',
    verification: 'runtime-required',
    dormantInStaticCorpus: true,
  },
  {
    id: 'control',
    label: 'Kontrol',
    description: 'Devam, durma, çelişki, tekrar ve düzeltme sinyallerini düzenleyen bölge.',
    verification: 'runtime-required',
    dormantInStaticCorpus: true,
  },
])

export interface SparseCoordinate {
  index: number
  value: number
}

/** Sorted, unique and bounded sparse vector. No dense Q or M table is materialized. */
export interface SparseVector {
  dimension: number
  entries: SparseCoordinate[]
  norm: number
}

export interface FamilyNeuronInput {
  id: string
  sourceId: string
  paragraphIndex: number
  start: number
  end: number
  text: string
  graphemeCount: number
  depth: number
  branch: NodeBranch
  parentId: string | null
  childIds: string[]
  splitBoundary: SplitBoundary
  identity: Float32Array
  relation: Float32Array
  identityDigest: string
  identityNorm: number
  relationNorm: number
  contextSupport: number
  contextRevision: number
  trace: ContextTrace
}

/** Thin TextLab-v2 input. Dense K is intentionally omitted; H⁰ is consumed read-only. */
export interface FamilyAnalysisInput {
  format: 'metin-laboratuvari-analysis'
  version: 2
  runId: string
  createdAt: string
  independenceGroupId?: string
  sourceDigest: string
  identitySpaceId: string
  contextSpaceId: string
  representationSpaceId: string
  source: CanonicalSource
  config: TextLabConfig
  layout: MatrixLayout
  rootIds: string[]
  nodes: FamilyNeuronInput[]
  habitProbabilities: Float32Array
}

/** Structural shape accepted from the independent synthesis decision archive. */
export interface FamilySynthesisDecisionInput {
  id: string
  format: 'noron-sentez-karari'
  version: 1
  candidateId: string
  candidateFingerprint: string
  decision: 'approved' | 'rejected'
  reviewedAt: string
  runId: string
  runFingerprint: string
  representationSpaceId: string
  analysisRunIds: string[]
  sourceDigests: string[]
  config: SynthesisConfig
  candidate: SynthesisCandidate
}

export interface FamilyConfig {
  profileDimension: number
  maximumProfileFeatures: number
  contextRadius: number
  maximumTransitionNeighbors: number
  maximumIndexBucketSize: number
  maximumCandidateNeighbors: number
  maximumFamilyMembers: number
  maximumFamilies: number
  maximumMembershipsPerNeuron: number
  minimumFormationSimilarity: number
  minimumMembershipScore: number
  minimumCoreMembership: number
  minimumSupport: number
  minimumSources: number
  maximumOutgoingFamilyTransitions: number
  runtimeActivationThreshold: number
  maximumActiveNeurons: number
  maximumActiveFamilies: number
  maximumCoalitions: number
  maximumRoleCandidates: number
  coalitionBeamWidth: number
  routingCandidateLimit: number
  routingTemperature: number
  habitRouteWeight: number
  familyRouteWeight: number
  signalRouteWeight: number
  repetitionPenaltyWeight: number
  utilityMinimumCases: number
  utilityAlternativeCount: number
  utilityEpsilon: number
  lifecycleStabilityThreshold: number
  lifecycleUtilityThreshold: number
  yieldEveryOperations: number
  hashSeed: number
}

export type FamilyStage =
  | 'validating'
  | 'profiling'
  | 'indexing'
  | 'clustering'
  | 'membership'
  | 'transitions'
  | 'evaluating'
  | 'coalition'
  | 'complete'

export interface FamilyProgress {
  stage: FamilyStage
  completed: number
  total: number
  ratio: number
  message: string
  profileCount: number
  familyCount: number
  membershipCount: number
  transitionCount: number
}

export interface IndependentEvidenceRef {
  memberKey: string
  analysisRunId: string
  /** Stable provenance lineage; legacy evidence may omit it. */
  independenceGroupId?: string
  sourceId: string
  sourceDigest: string
  neuronId: string
  paragraphIndex: number
  start: number
  end: number
  text: string
}

export interface RegionAffinity {
  region: FunctionalRegionId
  score: number
  verified: boolean
  explanation: string
}

export interface BehaviorDiagnostics {
  spanScale: number
  relativePosition: number
  depthScale: number
  relationStrength: number
  transitionEntropy: number
  transitionConcentration: number
  localContinuity: number
  bridgeImpact: number
  leftRightAsymmetry: number
  shortScopePotential: number
}

export interface BehaviorProfile {
  memberKey: string
  kind: 'raw' | 'synthesis'
  representationSpaceId: string
  behaviorSpaceId: string
  vector: SparseVector
  diagnostics: BehaviorDiagnostics
  regionAffinities: RegionAffinity[]
  evidence: IndependentEvidenceRef[]
  identityDigest: string
  text: string
  derivedFromMemberKeys: string[]
}

export interface MembershipScoreBreakdown {
  behaviorSimilarity: number
  regionCompatibility: number
  evidenceConfidence: number
  combined: number
}

export interface FamilyMembership {
  memberKey: string
  familyId: string
  rank: number
  score: number
  weight: number
  core: boolean
  breakdown: MembershipScoreBreakdown
}

export type FamilyLifecycle =
  | 'provisional'
  | 'strengthening'
  | 'stable'
  | 'split-proposed'
  | 'merge-proposed'
  | 'retired'

export interface LearnedFamily {
  id: string
  fingerprint: string
  behaviorSpaceId: string
  region: FunctionalRegionId
  regionVerified: boolean
  lifecycle: FamilyLifecycle
  lifecycleReason: string
  prototype: SparseVector
  prototypeDigest: string
  memberKeys: string[]
  coreMemberKeys: string[]
  independentEvidence: IndependentEvidenceRef[]
  supportCount: number
  sourceCount: number
  cohesion: number
  separation: number
  stability: number
  utility: number | null
  predecessorFamilyIds: string[]
}

export interface FamilyTransition {
  id: string
  fromFamilyId: string
  toFamilyId: string
  priorEvidence: number
  experienceEvidence: number
  inhibitionEvidence: number
  sourceCount: number
  supportCount: number
  probability: number
}

export interface SparseNeuronTransition {
  fromMemberKey: string
  toMemberKey: string
  probability: number
}

export interface FamilyInputManifestEntry {
  kind: 'text-lab-analysis' | 'synthesis-decision'
  id: string
  sourceDigest?: string
  analysisRunId?: string
  representationSpaceId: string
  revision: string
  fingerprint: string
}

export interface FamilyInputManifest {
  entries: FamilyInputManifestEntry[]
  corpusFingerprint: string
  /**
   * Optional split contract used when family discovery is isolated from its
   * evaluation corpus. Older v1 archives omit this field and retain the
   * original all-source cross-validation behavior.
   */
  dataSplit?: FamilyDataSplitManifest
}

export interface FamilyDataSplitManifest {
  mode: 'declared-independent-groups'
  /** Content fingerprint of inputs allowed to change learned structure. */
  discoveryCorpusFingerprint: string
  discoveryIndependenceGroupIds: string[]
  evaluationIndependenceGroupIds: string[]
  discoveryAnalysisRunIds: string[]
  evaluationAnalysisRunIds: string[]
  projectionOnlyAnalysisRunIds: string[]
  utilityFallback: 'none' | 'discovery-source-holdout'
}

export interface HoldoutFoldMetric {
  /** Stable physical-source identity; all chunks of one book are held out together. Older archives may omit it. */
  heldOutSourceId?: string
  /** Representative content digest retained for traceability and v1 UI compatibility. */
  heldOutSourceDigest: string
  caseCount: number
  fullLoss: number
  ablatedLoss: number
  deltaUtility: number
}

export interface FamilyUtilityReport {
  familyId: string
  mode: 'source-holdout-diagnostic' | 'declared-evaluation-diagnostic'
  status: 'validated' | 'insufficient-evidence' | 'neutral-or-harmful'
  caseCount: number
  sourceFoldCount: number
  fullLoss: number
  ablatedLoss: number
  deltaUtility: number
  confidenceLow: number
  confidenceHigh: number
  folds: HoldoutFoldMetric[]
  explanation: string
}

export interface NeuronActivation {
  memberKey: string
  activation: number
  goalRelevance?: number
  conflict?: number
  repetition?: number
}

export type EventRoleKind =
  | 'agent-hypothesis'
  | 'patient-hypothesis'
  | 'instrument-hypothesis'
  | 'participant'
  | 'operator'

export interface EventRoleBinding {
  role: EventRoleKind
  memberKey: string
  roleFamilyId: string | null
  score: number
  temporary: true
  explanation: string
}

export interface CoalitionScoreBreakdown {
  activation: number
  coherence: number
  persistence: number
  goalRelevance: number
  conflict: number
  repetition: number
  cooldown: number
  total: number
}

export interface EventCoalition {
  id: string
  eventMemberKey: string
  eventFamilyId: string
  familyActivations: Array<{ familyId: string; activation: number }>
  bindings: EventRoleBinding[]
  memberKeys: string[]
  score: CoalitionScoreBreakdown
  dominance: number
  confidence: number
  temporary: true
}

export interface CoalitionState {
  tick: number
  activeFamilyCount: number
  coalitions: EventCoalition[]
  unmatchedReason: string | null
}

export interface RouteScoreBreakdown {
  habit: number
  family: number
  signal: number
  repetitionPenalty: number
  combinedLogit: number
}

export interface RouteEntry {
  memberKey: string
  probability: number
  targetFamilyProbability: number
  breakdown: RouteScoreBreakdown
}

export interface RouteDistribution {
  currentMemberKey: string
  entries: RouteEntry[]
  unmatchedProbability: number
  totalProbability: number
  explanation: string
}

export interface FamilyRunStats {
  inputAnalysisCount: number
  /** Present for split-aware runs; omitted by legacy archives. */
  discoveryAnalysisCount?: number
  /** Present for split-aware runs; omitted by legacy archives. */
  evaluationAnalysisCount?: number
  /** Profiles are still built for these analyses, but they cannot teach. */
  projectionOnlyAnalysisCount?: number
  approvedSynthesisCount: number
  distinctSourceCount: number
  rawNeuronCount: number
  profileCount: number
  profileFeatureCount: number
  candidateComparisonCount: number
  retainedBehaviorEdgeCount: number
  familyCount: number
  membershipCount: number
  peakMembershipsPerNeuron: number
  retainedFamilyTransitionCount: number
  peakOutgoingFamilyTransitions: number
  utilityCaseCount: number
  coalitionExpansionCount: number
  peakCoalitionBeamWidth: number
}

export interface FamilyRun {
  format: 'noron-aileleri-run'
  version: 1
  id: string
  fingerprint: string
  createdAt: string
  representationSpaceId: string
  behaviorSpaceId: string
  config: FamilyConfig
  inputManifest: FamilyInputManifest
  profiles: BehaviorProfile[]
  families: LearnedFamily[]
  memberships: FamilyMembership[]
  neuronTransitions: SparseNeuronTransition[]
  familyTransitions: FamilyTransition[]
  utilityReports: FamilyUtilityReport[]
  coalition: CoalitionState
  routingPreview: RouteDistribution | null
  warnings: string[]
  stats: FamilyRunStats
}

export interface FamilyRunInput {
  analyses: FamilyAnalysisInput[]
  /**
   * Provenance lineages allowed to form prototypes and teach H⁰/G. When
   * omitted together with evaluationIndependenceGroupIds, legacy all-source
   * behavior is preserved. If only evaluation groups are declared, every
   * non-evaluation analysis becomes discovery data.
   */
  discoveryIndependenceGroupIds?: string[]
  /**
   * Provenance lineages reserved for out-of-sample membership projection and
   * utility measurement. They never form prototypes or transition evidence.
   */
  evaluationIndependenceGroupIds?: string[]
  synthesisDecisions?: FamilySynthesisDecisionInput[]
  previousRun?: FamilyRun | null
  activations?: NeuronActivation[]
}

export interface FamilyRunOptions {
  onProgress?: (progress: FamilyProgress) => void
  shouldCancel?: () => boolean
  yieldControl?: () => Promise<void>
}
