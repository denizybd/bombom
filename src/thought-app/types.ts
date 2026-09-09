export type ThoughtEpisodeState =
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'unknown'

export type TeacherMode = 'off' | 'ollama-cross-check'
export type ThoughtLearningMode = 'off' | 'proposal'
export type ThoughtRegionMode = 'off' | 'shadow' | 'proposal' | 'active'

export interface ThoughtRegionRuntime {
  behaviorApplied: boolean
  observationsEnabled: boolean
  reason: string
  atlasArtifactId: string | null
  atlasFingerprint: string | null
  atlasRevision: number | null
  geometryFingerprint: string | null
  representationSpaceId: string | null
  ephemeralAtlas: boolean
  compatibilityBlend: number
  minimumTrace: number
  bindingFingerprint: string | null
  releaseManifestArtifactId: string | null
  releaseFingerprint: string | null
}

export interface ThoughtAdaptiveGate {
  regionId: string
  compatibility: number | null
  prototypeCompatibility: number | null
  contextScore: number | null
  confidence: number | null
  probability: number | null
}

export interface ThoughtAdaptiveGateDistribution {
  gates: ThoughtAdaptiveGate[]
  unmatchedProbability: number | null
  omittedRegionProbability: number | null
}

export interface ThoughtAdaptiveBranchTelemetry {
  mode: Exclude<ThoughtRegionMode, 'off'>
  atlasFingerprint: string
  bindingFingerprint: string
  activeApplied: boolean
  legacyCompatibility: number | null
  regionCompatibility: number | null
  blendedCompatibility: number | null
  effectiveCompatibility: number | null
  pageCompatibilities: number[]
  pageWeights: number[]
  pageWriteMask: number[]
  gates: ThoughtAdaptiveGateDistribution
}

export interface ThoughtAdaptiveTickTelemetry {
  mode: Exclude<ThoughtRegionMode, 'off'>
  atlasFingerprint: string
  bindingFingerprint: string
  observedBranchCount: number
  activeApplied: boolean
}

/**
 * Optional outside evidence for an episode. It is an evaluation target only;
 * the runtime must not turn these labels directly into neuron-family identity.
 */
export interface ExpectedStructuralOutcome {
  format: 'event-graph-v1'
  actor: string | null
  action: string | null
  patient: string | null
  instrument: string | null
  operators: string[]
  negated: boolean
  goals: Array<{ relation: string; target: string }>
}

export interface ThoughtMatrix {
  side: 64
  values: number[]
  min: number
  max: number
  mean: number
}

export interface ThoughtCandidate {
  id: string
  label: string
  neuronId?: string
  familyId?: string
  rank: number
  probability: number | null
  total: number | null
  h0Prior: number | null
  persistentExperience: number | null
  branchTrace: number | null
  family: number | null
  signalQ: number | null
  repetitionPenalty: number | null
  habit: number | null
  signal: number | null
  identity: number | null
  context: number | null
  coalition: number | null
  edge: number | null
  novelty: number | null
  selected: boolean
}

export interface ThoughtCoalition {
  id: string
  label: string
  score: number | null
  agreement: number | null
  dominantFamilyId?: string
  memberIds: string[]
  memberLabels: string[]
  roleHypotheses: ThoughtRoleHypothesis[]
}

export interface ThoughtRoleHypothesis {
  role: string
  target: string
  score: number | null
}

export type NewbornPhase =
  | 'none'
  | 'watching'
  | 'candidate'
  | 'eligible'
  | 'born'
  | 'quarantined'
  | 'rejected'
  | 'unknown'

export interface ThoughtNewborn {
  phase: NewbornPhase
  id?: string
  label?: string
  reason?: string
  confidence: number | null
}

export interface ThoughtBranch {
  id: string
  rank: number
  label: string
  targetNeuronId?: string
  targetFamilyId?: string
  probability: number | null
  selected: boolean
  pruned: boolean
  candidates: ThoughtCandidate[]
  coalition: ThoughtCoalition | null
  unmatched: number | null
  edge: number | null
  birthPressure: number | null
  newborn: ThoughtNewborn | null
  signalBefore: ThoughtMatrix | null
  signalAfter: ThoughtMatrix | null
  adaptiveRegion: ThoughtAdaptiveBranchTelemetry | null
}

export interface ThoughtTick {
  index: number
  at?: string
  label: string
  activeNeuronId?: string
  activeNeuronLabel?: string
  signalBefore: ThoughtMatrix
  signalAfter: ThoughtMatrix
  branches: ThoughtBranch[]
  selectedBranchId?: string
  candidates: ThoughtCandidate[]
  coalition: ThoughtCoalition | null
  unmatched: number | null
  edge: number | null
  birthPressure: number | null
  newborn: ThoughtNewborn | null
  structuralScore: number | null
  trust: number | null
  note?: string
  adaptiveRegion: ThoughtAdaptiveTickTelemetry | null
}

export interface ThoughtEpisodeSummary {
  id: string
  prompt: string
  snapshotId: string | null
  snapshotName?: string
  teacherMode: TeacherMode
  learningMode: ThoughtLearningMode
  regionMode: ThoughtRegionMode
  requestedRegionMode: ThoughtRegionMode
  regionRuntime: ThoughtRegionRuntime | null
  mode: string
  commitEligible: boolean
  commitStatus: string
  review: ThoughtEpisodeReview | null
  candidateSnapshotId: string | null
  seed: number
  maxTicks: number
  tickCount: number
  currentTick: number
  state: ThoughtEpisodeState
  createdAt?: string
  updatedAt?: string
  resultText?: string
  stopReason?: string
  structuralScore: number | null
  trust: number | null
}

export interface ThoughtEpisodeReview {
  decision: 'approve' | 'reject' | string
  reason: string
  reviewedAt?: string
}

export interface ThoughtEpisode extends ThoughtEpisodeSummary {
  ticks: ThoughtTick[]
  warnings: string[]
}

export interface ThoughtServiceStatus {
  available: boolean
  enabled?: boolean
  researchBaseAvailable?: boolean
  activeSnapshotId: string | null
  maximumTicks: number | null
  version?: string
  state?: string
  activeEpisodes: number
  queuedEpisodes: number
  message?: string
}

export interface ThoughtSnapshotOption {
  id: string
  name: string
  active: boolean
  commitEligible: boolean
  state?: string
  createdAt?: string
}

export interface StartThoughtEpisodeRequest {
  prompt: string
  snapshotId?: string
  teacherMode: TeacherMode
  learningMode: ThoughtLearningMode
  seed: number
  maxTicks: number
  expectedOutcome: ExpectedStructuralOutcome | null
}

export interface ThoughtFeedbackResult {
  episode: ThoughtEpisode
  duplicate: boolean
  snapshot: null | { id: string; status?: string; parentId?: string | null }
}

export interface ReplayVerification {
  ok: boolean
  matches: boolean | null
  message: string
  expectedHash?: string
  actualHash?: string
  firstMismatchTick: number | null
}

export type ThoughtGraphLevel = 'neuron' | 'family'

export interface ThoughtGraphNode {
  id: string
  label: string
  kind?: string
  weight: number | null
}

export interface ThoughtGraphEdge {
  id: string
  from: string
  to: string
  label?: string
  weight: number | null
}

export interface ThoughtLanguageOutput {
  text: string
  stoppedBy: string
  scaffoldFingerprint: string | null
  headFingerprint: string | null
  steps: Array<{
    familyId: string
    emittedSurface: string | null
    explanation: string
    complete: boolean
  }>
  error: string | null
}

export interface ThoughtGraph {
  level: ThoughtGraphLevel
  nodes: ThoughtGraphNode[]
  edges: ThoughtGraphEdge[]
  totalNodeCount: number
  totalEdgeCount: number
  truncated: boolean
  episodeId: string | null
  structuralOutcome: ExpectedStructuralOutcome | null
  structuralEvaluation: {
    graphF1: number | null
    constraintAccuracy: number | null
    resultScore: number | null
  } | null
  languageOutput: ThoughtLanguageOutput | null
}
