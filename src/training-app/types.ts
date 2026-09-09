export type RunState =
  | 'offline'
  | 'idle'
  | 'starting'
  | 'running'
  | 'pausing'
  | 'paused'
  | 'cancelling'
  | 'failed'

export type AutomationLevel = 'proposal' | 'controlled'

export type PipelineStageId =
  | 'collect'
  | 'quarantine'
  | 'validate'
  | 'encode'
  | 'synthesize'
  | 'families'
  | 'language_experience'
  | 'plastic_regions'
  | 'concept_macros'
  | 'evaluate'
  | 'commit'
  | 'rollback'

export type PipelineStageState =
  | 'pending'
  | 'active'
  | 'complete'
  | 'failed'
  | 'skipped'
  | 'rolled_back'

export interface ResourceBudgets {
  cpuPercent: number
  ramMb: number
  maxRunMinutes: number
  diskMb: number
}

export interface TrainingConfig {
  ollamaBaseUrl: string
  model: string
  automationLevel: AutomationLevel
  /** Queue a bounded cycle when enough approved, unprocessed sources accumulate. */
  autoRun: boolean
  batchSize: number
  budgets: ResourceBudgets
  conceptMacros?: {
    enabled: boolean
    mode: 'off' | 'observe' | 'proposal-shadow' | 'controlled'
    minimumLength: number
    maximumLength: number
    maximumObservationsPerCycle: number
    maximumCandidates: number
    maximumActiveMacros: number
    proposalMinimumOccurrences: number
    proposalMinimumSourceGroups: number
    activationMinimumOccurrences: number
    activationMinimumSourceGroups: number
    minimumCompletionProbability: number
    minimumCohesion: number
    minimumShadowFidelity: number
    maximumFalseActivationRate: number
    maximumNestedDepth: number
    thoughtSamplesPerCycle: number
    thoughtTicksPerSample: number
  }
  /** Read-only companion mechanism status. This screen never enables it. */
  plasticRegions?: Readonly<{
    enabled: boolean
    mode: 'off' | 'shadow' | 'proposal' | 'active'
    representationProfile: 'legacy' | 'adaptive-eighth-id-v1' | string
    thoughtSamplesPerCycle: number
    thoughtTicksPerSample: number
    maximumObservationsPerBatch: number
    maximumObservationBatchesPerCycle: number
    maximumCumulativeObservations: number
    maximumEvidenceHeadsPerCycle: number
    maximumRegions: number
    maximumMembershipsPerPage: number
    maximumActiveGates: number
    compatibilityBlend: number
    minimumTrace: number
    releaseMinimumCases: number
    releaseMinimumSourceGroups: number
    releaseMinimumHoldoutGain: number
    releaseMaximumShamGain: number
    releaseMaximumRegressionLoss: number
    maximumArtifactBytes: number
  }>
}

export interface OllamaStatus {
  connected: boolean
  version?: string
  endpoint: string
  checkedAt: string
  models?: OllamaModel[]
  message?: string
}

export interface OllamaModel {
  name: string
  parameterSize?: string
  quantization?: string
  sizeBytes?: number
  modifiedAt?: string
}

export interface PipelineStage {
  id: PipelineStageId
  state: PipelineStageState
  progress: number
  processed?: number
  total?: number
  startedAt?: string
  finishedAt?: string
  message?: string
}

export interface TrainingStatus {
  serviceVersion: string
  state: RunState
  runId: string | null
  startedAt: string | null
  updatedAt: string
  activeStage: PipelineStageId | null
  progress: number
  queuedItems: number
  processedItems: number
  acceptedItems: number
  rejectedItems: number
  activeSnapshotId: string | null
  lastError?: string
  ollama: OllamaStatus
  config: TrainingConfig
  resources?: {
    memoryTier: 'normal' | 'soft' | 'spill' | 'hard' | 'disabled'
    diskTier: 'normal' | 'warning' | 'admission-blocked' | 'hard' | 'disabled'
    processRssMb: number
    ramLimitMb: number
    hostFreeMb: number
    diskFreeMb: number
    usableDiskFreeMb: number
    logicalStorageMb: number
    logicalDiskLimitMb: number
    canStart: boolean
    shouldSpill: boolean
    reasonCodes: string[]
  } | null
  stages: PipelineStage[]
  warnings: ServiceWarning[]
}

export type WarningSeverity = 'info' | 'warning' | 'critical'

export interface ServiceWarning {
  id: string
  severity: WarningSeverity
  title: string
  message: string
  createdAt: string
  code?: string
}

export type SourceKind = 'pasted_text' | 'file' | 'ollama_generated'
export type SourceState = 'ready' | 'quarantined' | 'processing' | 'disabled' | 'failed'

export interface TrainingSource {
  id: string
  name: string
  kind: SourceKind
  state: SourceState
  createdAt: string
  updatedAt: string
  bytes: number
  itemCount: number
  selected: boolean
  provenance: string
  contentFingerprint?: string
  error?: string
}

export type QualityCheckState = 'pass' | 'warning' | 'fail'

export interface QualityCheck {
  id: string
  label: string
  state: QualityCheckState
  message?: string
}

export interface QuarantineItem {
  id: string
  sourceId: string
  sourceName: string
  createdAt: string
  preview: string
  language?: string
  characterCount: number
  duplicateScore?: number
  state: 'pending' | 'approved' | 'rejected'
  checks: QualityCheck[]
  provenance: string
}

export type QueueItemState = 'waiting' | 'active' | 'complete' | 'failed' | 'cancelled'

export interface QueueItem {
  id: string
  runId: string | null
  stage: PipelineStageId
  state: QueueItemState
  label: string
  sourceName?: string
  createdAt: string
  updatedAt: string
  attempts: number
  progress: number
  message?: string
}

export type TrainingEventLevel = 'debug' | 'info' | 'success' | 'warning' | 'error'

export interface TrainingEvent {
  id: string
  /** Stable machine-readable event name; older services may omit it. */
  type?: string
  at: string
  level: TrainingEventLevel
  stage?: PipelineStageId
  message: string
  details?: Record<string, unknown>
}

export interface MetricValue {
  current: number | null
  previous: number | null
  unit: 'ratio' | 'count' | 'milliseconds'
}

export interface TrainingMetrics {
  measuredAt: string
  holdoutTop1: MetricValue
  holdoutTop5: MetricValue
  unseenCombinationAccuracy: MetricValue
  grammaticalPreference: MetricValue
  familyUtility: MetricValue
  neuronCount: MetricValue
  familyCount: MetricValue
  birthProposalCount: MetricValue
  eligibleBirthProposalCount: MetricValue
  emergentLanguageFamilyCount: MetricValue
  eligibleEmergentLanguageFamilyCount: MetricValue
  conceptMacroCount: MetricValue
  conceptMacroProposalCount: MetricValue
  conceptMacroVirtualCount: MetricValue
  conceptMacroShadowValidatedCount: MetricValue
  conceptMacroActiveCount: MetricValue
  committedBatches: MetricValue
  rollbackRate: MetricValue
  evaluationSampleCount: number
  baselineSnapshotId: string | null
}

export type SnapshotState = 'candidate' | 'committed' | 'rejected' | 'rolled_back'

export interface ModelSnapshot {
  id: string
  name: string
  createdAt: string
  state: SnapshotState
  runId: string | null
  parentId: string | null
  bytes: number
  corpusFingerprint: string
  metrics: Pick<
    TrainingMetrics,
    'holdoutTop1' | 'unseenCombinationAccuracy' | 'grammaticalPreference' | 'familyUtility'
  >
  note?: string
  rollbackToken?: string
}

export type FamilyLifecycle =
  | 'provisional'
  | 'strengthening'
  | 'stable'
  | 'split-proposed'
  | 'merge-proposed'
  | 'retired'

export type FamilyRegion = 'content' | 'event' | 'role' | 'operator'

export interface FamilyInspectionSelection {
  selectedBy: 'requested-snapshot' | 'latest-snapshot' | 'latest-artifact' | 'none'
  snapshot: null | {
    id: string
    name: string
    status: string
    createdAt: string
    active: boolean
  }
  artifact: null | {
    id: string
    kind: 'family-run' | 'runtime-model-state'
    createdAt: string
    contentHash: string | null
    jobId: string | null
  }
}

export interface FamilyInspectionMember {
  memberKey: string
  kind: string
  text: string
  textTruncated: boolean
  softScore: number | null
  weight: number | null
  rank: number
  core: boolean
  sourceCount: number
  sources: Array<{ id: string; name: string }>
}

export interface FamilyInspectionTransition {
  id: string
  toFamilyId: string
  toRegion: FamilyRegion | string
  toLifecycle: FamilyLifecycle | string
  probability: number | null
  sourceCount: number
  supportCount: number
  priorEvidence: number | null
  experienceEvidence: number | null
  inhibitionEvidence: number | null
}

export interface FamilyUtilityEvidence {
  status: 'validated' | 'insufficient-evidence' | 'neutral-or-harmful'
  caseCount: number
  sourceFoldCount: number
  fullLoss: number | null
  ablatedLoss: number | null
  deltaUtility: number | null
  confidenceLow: number | null
  confidenceHigh: number | null
  explanation: string
}

export interface InspectedFamily {
  id: string
  region: FamilyRegion | string
  regionVerified: boolean
  lifecycle: FamilyLifecycle | string
  lifecycleReason: string
  supportCount: number
  sourceCount: number
  memberCount: number
  coreMemberCount: number
  cohesion: number | null
  separation: number | null
  stability: number | null
  utility: number | null
  predecessorFamilyIds: string[]
  utilityEvidence: FamilyUtilityEvidence | null
  topMembers: FamilyInspectionMember[]
  outgoingTransitions: FamilyInspectionTransition[]
}

export interface FamilyInspection {
  available: boolean
  reason?: string
  selection: FamilyInspectionSelection
  run: null | {
    id: string
    fingerprint: string
    createdAt: string
    representationSpaceId: string
    behaviorSpaceId: string
  }
  summary: {
    familyCount: number
    membershipCount: number
    profileCount: number
    familyTransitionCount: number
    distinctSourceCount: number
    multiSourceFamilyCount: number
    utilityMeasuredFamilyCount: number
    validatedUtilityFamilyCount: number
    byLifecycle: Array<{ key: string; count: number }>
    byRegion: Array<{ key: string; count: number }>
  }
  pagination: {
    offset: number
    limit: number
    total: number
    returned: number
    hasMore: boolean
    nextOffset: number | null
  }
  limits: {
    memberLimit: number
    transitionLimit: number
    maxFamilyLimit: number
    maxMemberLimit: number
    maxTransitionLimit: number
  }
  families: InspectedFamily[]
}

export interface DashboardData {
  status: TrainingStatus
  models: OllamaModel[]
  sources: TrainingSource[]
  quarantine: QuarantineItem[]
  queue: QueueItem[]
  metrics: TrainingMetrics | null
  snapshots: ModelSnapshot[]
}

export interface StartRunRequest {
  automationLevel: AutomationLevel
  model: string
  sourceIds: string[]
  config: TrainingConfig
}

export interface ActionResponse {
  accepted: boolean
  status: TrainingStatus
  message?: string
}

export interface TextLabSourceManifest {
  format: 'text-lab-recompute-proof-v1'
  originalSourceId: string
  sourceDigest: string
  representationSpaceId: string
  config: {
    representationProfile?: 'legacy-quarter-id-v2' | 'adaptive-eighth-id-v1'
    matrixSide: number
    identityRatio: number
    minimumLeafGraphemes: number
    hashSeed: number
    contextRetention: number
    parentTraceWeight: number
    siblingTraceWeight: number
    minimumTrace: number
    identitySimilarityWeight: number
    habitExponent: number
  }
  identityProofs: Array<{ start: number; end: number; identityDigest: string }>
}

export interface SourceTextRequest {
  name: string
  text: string
  provenance: string
  analysisManifest?: TextLabSourceManifest
}

export interface TeacherGenerateRequest {
  topic: string
  objective: string
  count: number
  language: 'tr'
  diversity: 'focused' | 'balanced' | 'broad'
  model: string
}

export interface TeacherGenerateResponse {
  accepted: boolean
  quarantineItemIds: string[]
  generatedCount: number
  rejectedCount: number
  message?: string
}

export type TeacherFeedCampaignStatus =
  | 'queued'
  | 'running'
  | 'pausing'
  | 'paused'
  | 'completed'
  | 'cancelling'
  | 'cancelled'
  | 'failed'

export type StructuralStopGateStatus =
  | 'waiting'
  | 'missing'
  | 'failed'
  | 'passed'
  | 'screened-out'
  | 'failed-early'
  | 'decisive-early-failure'
  // Keep the dashboard readable when a newer local service introduces a state
  // before this client bundle is refreshed.
  | (string & {})

export interface StructuralStopGate {
  required: true
  passed: boolean
  status: StructuralStopGateStatus
  valid?: boolean
  conclusive?: boolean
  completion?: string | null
  evaluationKind?: string | null
  diagnosticMeanGraphF1?: number | null
  holdoutGraphF1?: number | null
  holdoutGraphF1Low?: number | null
  holdoutGraphF1High?: number | null
  snapshotId: string | null
  benchmarkFingerprint: string | null
  failedChecks: string[]
}

export interface TeacherFeedCampaignRequest {
  model: string
  seed: number
  targetAccepted: number
  generationBatchSize: number
  trainEveryAccepted: number
  curriculum: 'turkish-foundations-v1'
  mode: 'research-only'
  generationMode?: 'controlled' | 'hybrid'
  continuousPlan?: {
    blockAccepted: number
    maxBlocks: number
    minimumDiskHeadroomMb: number
    maximumSyntheticSources: number
  }
}

/**
 * A bounded, restart-safe teacher feed. Ollama output remains synthetic research
 * evidence and this campaign can never activate a model by itself.
 */
export interface TeacherFeedCampaign extends TeacherFeedCampaignRequest {
  id: string
  status: TeacherFeedCampaignStatus
  generated: number
  accepted: number
  duplicates: number
  rejectedQuality: number
  teacherRequests: number
  teacherTokens: number
  completedTrainingCycles: number
  nextBatchIndex: number
  lastSnapshotId: string | null
  stopReason: string | null
  lanes?: null | {
    controlled: TeacherFeedLaneCounters
    discovery: TeacherFeedLaneCounters
  }
  continuous: null | {
    blockAccepted: number
    maximumBlocks: number
    completedBlocks: number
    nextBlockAt: number
    minimumDiskHeadroomMb: number
    maximumSyntheticSources: number
    bootstrapSourceCount: number
    bootstrapSourceFingerprint: string | null
    stopGate: StructuralStopGate
  }
  createdAt: string
  updatedAt: string
}

export interface TeacherFeedLaneCounters {
  generated: number
  accepted: number
  duplicates: number
  rejectedQuality: number
  teacherRequests: number
  teacherTokens: number
}

export interface ReviewRequest {
  reason?: string
}

export interface RollbackRequest {
  confirmToken: string
  reason: string
}

export interface ApiErrorBody {
  error?: string
  message?: string
  code?: string
  details?: unknown
}
