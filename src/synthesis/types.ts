import type {
  CanonicalSource,
  MatrixLayout,
  RawTextNeuron,
  TextLabConfig,
} from '../text-lab/domain/types'

export type SynthesisNodeSelection =
  | 'leaves'
  | 'short-spans'
  | 'leaves-or-short-spans'
  | 'all'

export type SynthesisStage =
  | 'validating'
  | 'selecting'
  | 'comparing'
  | 'clustering'
  | 'synthesizing'
  | 'complete'

/**
 * Phase-one surface synthesis configuration.
 *
 * The hard limits make memory use proportional to eligible neurons rather than
 * to the square of that count. Threshold-passing edges are streamed, while
 * only each member's strongest neighbourhood is retained for clustering.
 */
export interface SynthesisConfig {
  nodeSelection: SynthesisNodeSelection
  maximumSpanGraphemes: number
  minimumIdentitySimilarity: number
  minimumRelationSimilarity: number
  minimumCombinedSimilarity: number
  /** A near-identical surface may pass even when its learned contexts disagree. */
  identityOverrideSimilarity: number
  identityWeight: number
  maximumEdgesPerMember: number
  maximumRetainedEdges: number
  maximumClusterMembers: number
  maximumCandidates: number
  minimumSupport: number
  minimumSources: number
  yieldEveryComparisons: number
}

export interface SynthesisProgress {
  stage: SynthesisStage
  completed: number
  total: number
  ratio: number
  message: string
  eligibleMemberCount: number
  retainedEdgeCount: number
  candidateCount: number
}

/** Only the raw-neuron fields used by synthesis; dense K/H matrices stay in IndexedDB. */
export type SynthesisNeuronInput = Pick<
  RawTextNeuron,
  | 'id'
  | 'sourceId'
  | 'paragraphIndex'
  | 'start'
  | 'end'
  | 'text'
  | 'graphemeCount'
  | 'depth'
  | 'parentId'
  | 'childIds'
  | 'splitBoundary'
  | 'identity'
  | 'relation'
  | 'identityDigest'
>

/** Thin, integrity-checkable Worker input without similarities/habitProbabilities. */
export interface SynthesisAnalysisInput {
  format: 'metin-laboratuvari-analysis'
  version: 2
  runId: string
  sourceDigest: string
  identitySpaceId: string
  contextSpaceId: string
  representationSpaceId: string
  source: CanonicalSource
  config: TextLabConfig
  layout: MatrixLayout
  nodes: SynthesisNeuronInput[]
}

/** A compact, immutable reference back to one raw text neuron. */
export interface SynthesisMemberRef {
  key: string
  analysisRunId: string
  sourceId: string
  sourceDigest: string
  neuronId: string
  paragraphIndex: number
  start: number
  end: number
  text: string
  graphemeCount: number
  depth: number
  parentId: string | null
}

/** A retained cross-source relation; no dense N x N table is produced. */
export interface SynthesisEdge {
  id: string
  fingerprint: string
  left: SynthesisMemberRef
  right: SynthesisMemberRef
  identitySimilarity: number
  relationSimilarity: number
  relationConfidence: number
  combinedSimilarity: number
}

export interface SynthesisCandidate {
  id: string
  fingerprint: string
  representationSpaceId: string
  members: SynthesisMemberRef[]
  /** Non-overlapping evidence used for birth support and medoid selection. */
  independentSupports: SynthesisMemberRef[]
  medoid: SynthesisMemberRef
  /** A byte-for-byte copy of the medoid's immutable identity vector. */
  identity: Float32Array
  /** Deterministic integrity digest calculated from the frozen identity bytes. */
  identityDigest: string
  /** Initial plastic context, averaged from independent evidence. */
  relation: Float32Array
  identityFrozen: true
  /** Plastic context starts at revision zero when the candidate is born. */
  contextRevision: 0
  /** Number of independent, non-overlapping observations in the initial context. */
  contextSupport: number
  supportCount: number
  sourceCount: number
  identityCohesion: number
  relationCohesion: number
  birthScore: number
}

export interface SynthesisRunStats {
  inputAnalysisCount: number
  distinctSourceCount: number
  inputNeuronCount: number
  eligibleNeuronCount: number
  comparedPairCount: number
  thresholdEdgeCount: number
  retainedEdgeCount: number
  clusterCount: number
  candidateCount: number
  peakEdgesPerMember: number
}

export interface SynthesisRun {
  format: 'noron-sentez-run'
  version: 1
  id: string
  fingerprint: string
  representationSpaceId: string
  analysisRunIds: string[]
  sourceDigests: string[]
  config: SynthesisConfig
  edges: SynthesisEdge[]
  candidates: SynthesisCandidate[]
  stats: SynthesisRunStats
}

export interface SynthesisRunOptions {
  onProgress?: (progress: SynthesisProgress) => void
  /** Receives every threshold-passing edge before bounded retention. */
  onEdge?: (edge: SynthesisEdge) => void
  shouldCancel?: () => boolean
  /** Test/worker hook; defaults to yielding to the browser event loop. */
  yieldControl?: () => Promise<void>
}

export type SynthesisInput = readonly SynthesisAnalysisInput[]
