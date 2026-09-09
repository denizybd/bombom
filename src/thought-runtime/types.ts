import type { FamilyRun, FunctionalRegionId } from '../families/types'
import type {
  TextLabAnalysis,
  TextLabRepresentationProfileId,
} from '../text-lab/domain/types'
import type { MacroHeadV1 } from '../concept-macros/types'
import type {
  PlasticPageObservationV1,
  PlasticRegionAtlasV1,
  RegionGateDistributionV1,
} from '../adaptive-regions/types'
import type {
  ThoughtMacroFallbackV1,
  ThoughtMacroRuntimeV1,
} from './concept-macro-runtime'

export const THOUGHT_MATRIX_SIDE = 64 as const
export const THOUGHT_MATRIX_CELLS = THOUGHT_MATRIX_SIDE * THOUGHT_MATRIX_SIDE

export type ThoughtVector = Float32Array
export type ThoughtSelectionMode = 'deterministic' | 'seeded-stochastic'
export type ThoughtBranchStatus = 'active' | 'unmatched' | 'settled' | 'max-ticks'
export type ThoughtAdaptiveRegionMode = 'off' | 'shadow' | 'proposal' | 'active'
export type ThoughtEnabledAdaptiveRegionMode = Exclude<ThoughtAdaptiveRegionMode, 'off'>

/** Serializable opt-in request. Absence and `off` are deliberately equivalent. */
export interface ThoughtAdaptiveRegionRequestV1 {
  mode: ThoughtAdaptiveRegionMode
  /** Required for every non-off run; the supplied atlas must match it exactly. */
  atlasFingerprint?: string
  /** Used only by active mode. Shadow/proposal are normalized to zero. */
  compatibilityBlend?: number
  /** Minimum page write trace (`m_min`) used only by active mode. */
  minimumTrace?: number
  /** Maximum active region gates used only by active mode. */
  maximumActiveGates?: number
}

/** Immutable atlas identity persisted in episode checkpoints and replay state. */
export interface ThoughtAdaptiveRegionBindingV1 {
  format: 'thought-adaptive-region-binding'
  version: 1
  mode: ThoughtEnabledAdaptiveRegionMode
  atlasFingerprint: string
  atlasRevision: number
  representationSpaceId: string
  geometryFingerprint: string
  sourceGroupId: string
  compatibilityBlend: number
  minimumTrace: number
  /** Present only for active routing; omitted for shadow/proposal compatibility. */
  maximumActiveGates?: number
  fingerprint: string
}

/** Trusted, resolved runtime context. The atlas itself is never accepted from a request. */
export interface ThoughtAdaptiveRegionContextV1 {
  binding: ThoughtAdaptiveRegionBindingV1
  atlas: PlasticRegionAtlasV1
  sourceGroupId: string
}

export interface ThoughtAdaptiveCompatibilityTelemetryV1 {
  legacyCompatibility: number
  regionCompatibility: number
  blendedCompatibility: number
  effectiveCompatibility: number
  pageCompatibilities: number[]
  pageWeights: number[]
  gates: RegionGateDistributionV1
}

export interface ThoughtAdaptiveCandidateTelemetryV1 extends ThoughtAdaptiveCompatibilityTelemetryV1 {
  neuronId: string
}

/** Fixed-size (56 page) observations plus at most eight routed candidates. */
export interface ThoughtAdaptiveBranchTelemetryV1 extends ThoughtAdaptiveCompatibilityTelemetryV1 {
  format: 'thought-adaptive-region-branch-telemetry'
  version: 1
  mode: ThoughtEnabledAdaptiveRegionMode
  atlasFingerprint: string
  bindingFingerprint: string
  activeApplied: boolean
  pageWriteMask: number[]
  pageObservations: PlasticPageObservationV1[]
  candidateCompatibilities: ThoughtAdaptiveCandidateTelemetryV1[]
}

export interface ThoughtAdaptiveTickTelemetryV1 {
  format: 'thought-adaptive-region-tick-telemetry'
  version: 1
  mode: ThoughtEnabledAdaptiveRegionMode
  atlasFingerprint: string
  bindingFingerprint: string
  observedBranchCount: number
  activeApplied: boolean
}

export interface ThoughtMatrixLayout {
  side: typeof THOUGHT_MATRIX_SIDE
  cellCount: typeof THOUGHT_MATRIX_CELLS
  identityLength: number
  relationStart: number
  relationLength: number
}

export interface ThoughtRuntimeConfig {
  matrixSide: typeof THOUGHT_MATRIX_SIDE
  defaultTicks: number
  maximumTicks: number
  signalNormMaximum: number
  repetitionWindow: number
  repetitionPenalty: number
  explorationReserve: number
  maximumHabitEdgesPerNeuron: number
  maximumTotalHabitEdges: number
  maximumCandidatePool: number
  maximumEvaluatedCandidates: number
  maximumCandidateFamilies: number
  minimumBranches: number
  maximumBranches: number
  minimumBranchProbability: number
  fastEligibilityDecay: number
  fastExcitationRate: number
  fastInhibitionRate: number
  evidenceCeiling: number
  birthPressureMemory: number
  birthPressureThreshold: number
  birthPressureConsecutiveTicks: number
  minimumResidualObservations: number
  maximumBirthsPerEpisode: number
  nurseryCohesionThreshold: number
  nurseryMinimumObservations: number
  nurseryMinimumIndependentSources: number
  nurseryMinimumTrustMass: number
  promotionCooldownTicks: number
  minimumOutcomeScore: number
  outcomeGraphWeight: number
  outcomeConstraintWeight: number
  maximumFamilyMembershipsPerNeuron: number
  hashSeed: number
  selectionMode: ThoughtSelectionMode
}

export interface ThoughtFamilyMembership {
  familyId: string
  weight: number
}

export interface ThoughtFamily {
  id: string
  region?: FunctionalRegionId
}

export interface ThoughtNeuron {
  id: string
  /** Frozen surface identity I. Its digest is checked before and after every tick. */
  identity: Float32Array
  /** Plastic base context R. Learning is emitted as a head delta, never written here. */
  relation: Float32Array
  identityDigest: string
  sourceIdentityDigest?: string
  familyMemberships: ThoughtFamilyMembership[]
  provenance: {
    kind: 'seed' | 'textlab-surface' | 'synthesis-surface' | 'episode-newborn' | 'nursery-promotion'
    sourceId?: string
    surface?: string
    nurseryId?: string
    residualCohesion?: number
    residualObservationCount?: number
    /** Exact retained branches that supplied the residual prototype. */
    birthBranchIds?: string[]
    /** Branch-local origins/families captured before the temporary identity exists. */
    originEvidence?: Array<{ key: string; weight: number }>
    familyEvidence?: Array<{ key: string; weight: number }>
  }
}

export interface SparseHabitEdge {
  toNeuronId: string
  priorEvidence: number
  excitationEvidence: number
  inhibitionEvidence: number
  trustMass: number
  updateCount: number
  independentSourceIds: string[]
}

export interface SparseHabitRow {
  fromNeuronId: string
  edges: SparseHabitEdge[]
}

export interface ThoughtRelationPatch {
  neuronId: string
  delta: Float32Array
  weight: number
  updateCount: number
}

export interface ThoughtHeadEdge {
  fromMemberKey: string
  toMemberKey: string
  excitation: number
  inhibition: number
  trustMass: number
  updateCount: number
}

export interface ThoughtNurseryCandidate {
  id: string
  identity: Float32Array
  relationPrototype: Float32Array
  identityDigest: string
  birthEpisodeIds: string[]
  independentSourceIds: string[]
  observationCount: number
  residualObservationCount: number
  supportWeight: number
  trustMass: number
  resultScoreTotal: number
  originEvidence: Array<{ key: string; weight: number }>
  /** Destinations actually traversed after the temporary neuron. */
  outgoingEvidence?: Array<{ key: string; weight: number }>
  familyEvidence: Array<{ key: string; weight: number }>
  nextPromotionTick: number
  residualCohesion: number
  sourceRelationPrototypes: Array<{
    sourceGroupId: string
    relation: Float32Array
    observationCount: number
  }>
}

export interface BornNeuronV1 {
  format: 'born-neuron-v1'
  id: string
  identity: number[]
  relation: number[]
  identityDigest: string
  familyMemberships: ThoughtFamilyMembership[]
  sourceGroupIds: string[]
  trustMass: number
  bornAtTick: number
  nurseryId: string
  /** Durable shortcut evidence retained from temporary-neuron traversals. */
  entryEvidence?: Array<{ memberKey: string; weight: number }>
  outgoingEvidence?: Array<{ memberKey: string; weight: number }>
  /** Audit proof for the source-balanced plastic R prototype. */
  contextEvidence?: {
    observationCount: number
    sourceGroupIds: string[]
  }
}

export interface ThoughtHead {
  format: 'bombom-thought-head'
  version: 1
  revision: number
  baseModelFingerprint: string
  totalTicks: number
  neuronEdges: ThoughtHeadEdge[]
  contextPatches: ThoughtRelationPatch[]
  nursery: ThoughtNurseryCandidate[]
  births: BornNeuronV1[]
  fingerprint: string
}

export interface ThoughtHeadDeltaEdge {
  fromMemberKey: string
  toMemberKey: string
  excitation: number
  inhibition: number
  trustMass: number
  updateCount: number
}

export interface ThoughtHeadDeltaContextPatch {
  memberKey: string
  relationDelta: number[]
  trustMass: number
  observationCount: number
}

/** Server-facing canonical, quantizable learning delta. */
export interface ThoughtHeadDelta {
  neuronEdges: ThoughtHeadDeltaEdge[]
  contextPatches?: ThoughtHeadDeltaContextPatch[]
  nurseryUpserts?: ThoughtNurseryCandidate[]
  nurseryRemoveIds?: string[]
  births?: BornNeuronV1[]
  stats?: {
    baseHeadRevision: number
    baseHeadFingerprint: string
    episodeId: string
    replayFingerprint: string
    outcomeScore: number
    graphF1: number
    constraintAccuracy: number
    confidence: number
    credit: number
    totalTicks: number
    nurseryRetained: number
    nurseryRejected: number
    nurseryPromoted: number
  }
}

export interface ThoughtModel {
  format: 'bombom-thought-model'
  version: 1
  revision: number
  layout: ThoughtMatrixLayout
  neurons: ThoughtNeuron[]
  families: ThoughtFamily[]
  habitRows: SparseHabitRow[]
  fingerprint: string
}

export interface DerivedFamilyTransition {
  fromFamilyId: string
  toFamilyId: string
  evidence: number
  probability: number
}

export interface DerivedFamilyGraph {
  transitions: DerivedFamilyTransition[]
  fingerprint: string
}

export interface ThoughtFamilyRouteCandidate {
  familyId: string
  probability: number
  evidence: number
  graphProbability: number
  signalCompatibility: number
  coalitionSupport: number
  goalSupport: number
  repetitionPenalty: number
  logit: number
}

export interface ThoughtNeuronRouteCandidate {
  neuronId: string
  familyId: string
  rank: number
  baseH0: number
  persistentExcitation: number
  fastExcitation: number
  persistentInhibition: number
  fastInhibition: number
  habitLogit: number
  habitProbability: number
  membershipWeight: number
  routeEvidence: number
  normalizedHabit: number
  compatibility: number
  repetitionPenalty: number
  q: number
  acceptanceProbability: number
  reachProbability: number
  selectionProbability: number
}

export interface ThoughtRouteResult {
  fromNeuronId: string
  familyCandidates: ThoughtFamilyRouteCandidate[]
  neuronCandidates: ThoughtNeuronRouteCandidate[]
  unmatchedProbability: number
  maximumCompatibility: number
  totalProbability: number
  entropy: number
  branchCount: number
  candidatePoolCount: number
  evaluatedCandidateCount: number
  inspectedNeuronCount: number
  /** Bounded diagnostics only; never present in off mode. */
  adaptiveRegionCandidates?: ThoughtAdaptiveCandidateTelemetryV1[]
}

export interface ThoughtFastTrace {
  fromNeuronId: string
  toNeuronId: string
  excitation: number
  inhibition: number
  eligibility: number
  visits: number
}

export interface ThoughtFastRelationPatch {
  neuronId: string
  delta: Float32Array
  eligibility: number
  visits: number
}

export interface ThoughtTransitionRecord {
  tick: number
  fromNeuronId: string
  toNeuronId: string
  familyId: string
  compatibility: number
  baseH0: number
  persistentExcitation: number
  fastExcitation: number
  persistentInhibition: number
  fastInhibition: number
  habitLogit: number
  habitProbability: number
  membershipWeight: number
  routeEvidence: number
  normalizedHabit: number
  q: number
  acceptanceProbability: number
  selectionProbability: number
  signalChangeMagnitude: number
  /** Present when a validated macro logically traversed this retained raw edge. */
  macroId?: string
  macroLogicalDuration?: number
}

export interface ThoughtResidualState {
  sum: Float32Array
  weight: number
  observationCount: number
  pressureTotal: number
  cohesionTotal: number
  cohesionComparisons: number
  /** Sum of L2-normalized residuals; enough for exact incremental pairwise cosine. */
  cohesionUnitSum: Float32Array
  originEvidence: Array<{ key: string; weight: number }>
  familyEvidence: Array<{ key: string; weight: number }>
}

export interface ThoughtBranch {
  id: string
  parentId: string | null
  currentNeuronId: string
  signal: ThoughtVector
  mass: number
  cumulativeLogProbability: number
  path: string[]
  transitions: ThoughtTransitionRecord[]
  fastTraces: ThoughtFastTrace[]
  fastRelationPatches: ThoughtFastRelationPatch[]
  pressure: {
    habitDistance: number
    representationError: number
    edge: number
    birth: number
    smoothedBirth: number
    consecutiveBirthTicks: number
  }
  residual: ThoughtResidualState
  status: ThoughtBranchStatus
}

export interface ThoughtReplayFrame {
  tick: number
  beforeFingerprint: string
  afterFingerprint: string
  retainedBranchIds: string[]
  retainedMasses: number[]
  baseHeadRevision: number
  baseHeadFingerprint: string
  traceFingerprint: string
  routingInputFingerprint: string
  /** Incremental replay-chain digest; avoids re-hashing the full history. */
  chainFingerprint: string
  /** Present only for non-off runs and included in the replay chain. */
  adaptiveRegionMode?: ThoughtEnabledAdaptiveRegionMode
  adaptiveRegionAtlasFingerprint?: string
  adaptiveRegionBindingFingerprint?: string
}

export interface ThoughtEpisodeState {
  format: 'bombom-thought-episode'
  version: 1
  episodeId: string
  runKey: string
  seed: number
  modelFingerprint: string
  headRevision: number
  headFingerprint: string
  initialSignalDigest: string
  tick: number
  maximumTicks: number
  branches: ThoughtBranch[]
  status: 'active' | 'stopped' | 'paused' | 'cancelled'
  replay: ThoughtReplayFrame[]
  temporaryBirths: ThoughtNeuron[]
  /** Fixed for the full episode; absence is the byte-compatible off mode. */
  adaptiveRegion?: ThoughtAdaptiveRegionBindingV1
  fingerprint: string
}

export interface ThoughtTick {
  tick: number
  branchCount: number
  activeBranchCount: number
  dominantBranchId: string | null
  dominantNeuronId: string | null
  dominantMass: number
  maximumBirthPressure: number
  replayFingerprint: string
  branches: ThoughtTickBranchTrace[]
  birthEvents: ThoughtBirthEvent[]
  adaptiveRegion?: ThoughtAdaptiveTickTelemetryV1
}

/** Temporary runtime-only role binding; never stored on a neuron or family. */
export interface ThoughtRoleHypothesis {
  role: 'actor' | 'action' | 'patient' | 'instrument' | 'operator' | 'goal'
  memberKey: string
  score: number
  temporary: true
}

export interface ThoughtRoutingHints {
  coalitionFamilyWeights?: Readonly<Record<string, number>>
  goalFamilyWeights?: Readonly<Record<string, number>>
  roleHypotheses?: readonly ThoughtRoleHypothesis[]
}

export interface ThoughtBirthEvent {
  branchId: string
  kind: 'pressure-threshold'
  pressure: number
  consecutiveTicks: number
  neuronId?: string
}

export interface ThoughtTickBranchTrace {
  branchId: string
  parentBranchId: string | null
  mass: number
  beforeSignal: number[]
  afterSignal: number[]
  beforeSignalDigest: string
  afterSignalDigest: string
  beforeSignalNorm: number
  afterSignalNorm: number
  signalChangeMagnitude: number
  selectedFamilyId: string | null
  selectedNeuronId: string | null
  familyCandidates: ThoughtFamilyRouteCandidate[]
  neuronCandidates: ThoughtNeuronRouteCandidate[]
  unmatchedProbability: number
  edgePressure: number
  birthPressure: number
  smoothedBirthPressure: number
  coalitionFamilyWeights: Record<string, number>
  roleHypotheses: ThoughtRoleHypothesis[]
  birthEvents: ThoughtBirthEvent[]
  macroRuntimeFingerprint: string | null
  rawProbabilityScale: number
  macroProbabilityMass: number
  macroCandidates: ThoughtMacroCandidateTraceV1[]
  macroExecution: ThoughtMacroExecutionTraceV1 | null
  macroFallbacks: ThoughtMacroFallbackV1[]
  adaptiveRegion?: ThoughtAdaptiveBranchTelemetryV1
}

export interface ThoughtMacroCandidateTraceV1 {
  macroId: string
  entryCompatibility: number
  activationWeight: number
  selectionProbability: number
  duration: number
  compressedTransitionCount: number
  fidelity: number
  expandedConstituentIds: string[]
}

export interface ThoughtMacroExecutionTraceV1 {
  macroId: string
  macroHeadFingerprint: string
  runtimeFingerprint: string
  compiled: true
  expandedConstituentIds: string[]
  duration: number
  compressedTransitionCount: number
  fidelity: number
  entryCompatibility: number
  selectionProbability: number
  /** Durable eligibility is distributed over retained constituent edges; compiled R is not fabricated. */
  creditMode: 'constituent-edges'
  fallbackReason: null
}

export interface StartThoughtEpisodeOptions {
  episodeId?: string
  runKey?: string
  seed?: number
  maximumTicks?: number
  initialNeuronIds?: string[]
  adaptiveRegionContext?: ThoughtAdaptiveRegionContextV1
}

export interface ThoughtStepOptions {
  /** Candidates supplied by an external ANN/index; the exact list becomes replay input. */
  candidateNeuronIds?: string[]
  teacherEdgeNeuronIds?: string[]
  /** Precompiled language-experience rows; each branch sees only its origin row. */
  teacherEdgeNeuronIdsByOrigin?: Readonly<Record<string, readonly string[]>>
  selectionMode?: ThoughtSelectionMode
  /** Snapshot-bound, integrity-checked active concept shortcut overlay. */
  macroRuntime?: ThoughtMacroRuntimeV1
  /** Trusted resolved context; must exactly match the checkpoint binding. */
  adaptiveRegionContext?: ThoughtAdaptiveRegionContextV1
}

export interface ThoughtBranchScore {
  branchId: string
  transitionCount: number
  meanCompatibility: number
  meanGraphSupport: number
  meanFamilySupport: number
  uniqueTransitionRatio: number
  loopRatio: number
  score: number
  eligible: boolean
  explanation: string
}

export interface EventGraphGoal { relation: string; target: string }

export interface EventGraphV1 {
  format: 'event-graph-v1'
  actor: string | null
  action: string | null
  patient: string | null
  instrument: string | null
  operators: string[]
  negated: boolean
  goals: EventGraphGoal[]
}

export interface OutcomeEvidence {
  kind: 'human' | 'deterministic' | 'observed' | 'ollama-cross-checked' | 'self'
  confidence: number
  sourceGroupId: string
}

export interface EventGraphEvaluation {
  graphF1: number
  constraintAccuracy: number
  resultScore: number
  truePositive: number
  falsePositive: number
  falseNegative: number
}

export interface ThoughtBranchEvaluation {
  branchId: string
  eventGraph: EventGraphV1
  evaluation: EventGraphEvaluation
}

export interface PromptEncodingOptions {
  identityRatio?: number
  hashSeed?: number
  minimumLeafGraphemes?: number
  allowFlatFallback?: boolean
  /** Explicit only: an absent profile retains the historical encoder contract. */
  representationProfile?: TextLabRepresentationProfileId
}

export interface TemporaryPromptNeuronDiagnostic {
  id: string
  parentId: string | null
  depth: number
  text: string
  weight: number
  identityDigest: string
}

export interface ThoughtPromptEncoding {
  signal: ThoughtVector
  temporaryNeurons: TemporaryPromptNeuronDiagnostic[]
  mode: 'textlab-hierarchy' | 'flat-fallback'
  fallbackReason: string | null
  identityLength: number
  representationSpaceId: string
  analysisFingerprint: string
}

export interface ThoughtCatalogDiagnostic {
  memberKey: string
  reason: string
  kind: 'included' | 'excluded'
}

export interface ThoughtCatalogResult { model: ThoughtModel; diagnostics: ThoughtCatalogDiagnostic[] }

export interface ThoughtEngineBase {
  snapshotId: string | null
  snapshotStatus?: string | null
  selectedBy?: string | null
  activeAtStart?: boolean
  commitEligible?: boolean
  artifactManifestHash: string | null
  baseModelFingerprint: string | null
  familyRunFingerprint: string | null
  languageStateFingerprint: string | null
  thoughtHeadFingerprint: string | null
  thoughtHeadRevision: number
  conceptMacroHeadFingerprint?: string | null
  mode?: string | null
  /** Exact physical encoder proof; absence preserves historical profileless bases. */
  representationProfile?: TextLabRepresentationProfileId | null
}

export interface ThoughtReplayRequest {
  verifyOnly: boolean
  expectedTickCount: number
  /** Optional caller assertion; if supplied it must match the episode binding. */
  adaptiveRegion?: {
    mode: ThoughtEnabledAdaptiveRegionMode
    atlasFingerprint: string
    bindingFingerprint: string
  }
}

export interface ThoughtEngineRequest {
  format: 'thought-engine-request'
  version: 1
  episodeId: string
  inputText: string
  seed: number
  maxTicks: number
  strategy: ThoughtSelectionMode | string
  learningMode: string
  teacherMode?: string
  candidateMemberKeys?: string[]
  sourceGroupId?: string | null
  adaptiveRegion?: ThoughtAdaptiveRegionRequestV1
  base: ThoughtEngineBase
  resume?: ThoughtEpisodeState | null
  replay?: ThoughtReplayRequest | null
  resumeAudit?: { tickCount: number; lastTickDigest: string | null } | null
  expectedOutcome?: EventGraphV1 | null
  outcomeEvidence?: OutcomeEvidence | null
  /** Temporary server migration alias; outcomeEvidence wins when both exist. */
  teacherEvidence?: OutcomeEvidence | null
}

/** Physical I/R row restored by server/thought-service before pure-core adaptation. */
export interface ThoughtResolvedNeuronInput {
  memberKey: string
  kind: 'raw' | 'synthesis'
  text?: string
  identity: Float32Array
  relation: Float32Array
  identityDigest?: string
  sourceDigest?: string
  analysisRunId?: string
  neuronId?: string
  candidateFingerprint?: string
}

export type ThoughtNeuronContextInput = ThoughtModel | ThoughtCatalogResult | ThoughtNeuron[] | TextLabAnalysis[] | ThoughtResolvedNeuronInput[]

export interface ThoughtEngineContext {
  signal?: AbortSignal
  /** Internal deterministic-evaluator hook; HTTP/API callers never control it. */
  benchmarkInitialSignal?: ThoughtVector
  /** Trusted evaluator-only RNG pairing key; never accepted from public requests. */
  benchmarkPairingKey?: string
  onTick?: (tick: ThoughtTick) => void | Promise<void>
  onCheckpoint?: (checkpoint: { tick: number; state: ThoughtEpisodeState; ticks: ThoughtTick[] }) => void | Promise<void>
  shouldPause?: () => boolean | Promise<boolean>
  shouldCancel?: () => boolean | Promise<boolean>
  loadArtifact?: (artifactId: string) => unknown | Promise<unknown>
  neurons: ThoughtNeuronContextInput
  familyRun: FamilyRun
  languageState: unknown
  thoughtHead?: ThoughtHead | null
  /** Only the concept head attached to the selected snapshot is accepted. */
  conceptMacroHead?: MacroHeadV1 | null
  /** Trusted snapshot-bound atlas. Never inferred from the matrix dimensions. */
  adaptiveRegionAtlas?: PlasticRegionAtlasV1 | null
  baseDiagnostics?: unknown
  /** Stable full-catalog fingerprint used by bounded SSD working-set pages. */
  modelFingerprintOverride?: string
  /** Trusted snapshot proof; must exactly match request.base when present. */
  representationProfile?: TextLabRepresentationProfileId | null
}

/**
 * Optional trusted snapshot inputs for SSD pre-routing prompt encoding. The
 * physical profile is explicit snapshot evidence and is never inferred from
 * atlas mode or matrix dimensions.
 */
export interface ThoughtResidencyPromptContextV1 {
  /** Trusted physical encoder proof, independent from adaptive-region mode. */
  representationProfile?: TextLabRepresentationProfileId | null
  adaptiveRegion?: ThoughtAdaptiveRegionRequestV1 | null
  adaptiveRegionAtlas?: PlasticRegionAtlasV1 | null
  sourceGroupId?: string | null
}

export interface ThoughtGraphDiagnostics {
  eventGraph: EventGraphV1
  evaluation: EventGraphEvaluation | null
  branchEvaluations: ThoughtBranchEvaluation[]
  familyGraph: DerivedFamilyGraph
  branchScores: ThoughtBranchScore[]
  catalog: ThoughtCatalogDiagnostic[]
  prompt: Omit<ThoughtPromptEncoding, 'signal'>
  baseDiagnostics: unknown
  nodes: Array<{ id: string; kind: 'neuron' | 'family'; mass?: number }>
  edges: Array<{
    from: string
    to: string
    kind: 'neuron' | 'family'
    probability: number
    excitation?: number
    inhibition?: number
  }>
}

export interface ThoughtEpisodeResult {
  episode: ThoughtEpisodeState
  ticks: ThoughtTick[]
  headDelta?: ThoughtHeadDelta
  graph: ThoughtGraphDiagnostics
}
