export type NodeBranch = 'root' | 'left' | 'right'

export type SplitBoundary = 'sentence' | 'space' | 'grapheme' | 'leaf'

export type PipelineStage =
  | 'preparing'
  | 'segmenting'
  | 'encoding'
  | 'propagating'
  | 'comparing'
  | 'connecting'
  | 'complete'

/**
 * A named physical/encoder contract. Historical projects did not carry this
 * field; an absent value is deliberately resolved as the legacy profile.
 */
export type TextLabRepresentationProfileId =
  | 'legacy-quarter-id-v2'
  | 'adaptive-eighth-id-v1'

export interface TextLabConfig {
  /** Optional on the wire so pre-profile projects remain byte-compatible. */
  representationProfile?: TextLabRepresentationProfileId
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

export interface TextSource {
  id: string
  title: string
  rawText: string
}

export interface CanonicalSource extends TextSource {
  /** NFC-normalized source with CRLF/CR line endings converted to LF. */
  text: string
}

export interface MatrixLayout {
  side: number
  cellCount: number
  identityRows: number
  identityLength: number
  relationStart: number
  relationLength: number
}

export interface ContextTrace {
  inheritedNorm: number
  parentIdentityCompatibility: number
  parentGain: number
  siblingId: string | null
  siblingIdentityCompatibility: number | null
  siblingGain: number
}

export interface RawTextNeuron {
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
  /** Frozen surface identity. Context code must never receive a writable alias to it. */
  identity: Float32Array
  /** Plastic context state. It is physically separate from identity. */
  relation: Float32Array
  identityDigest: string
  surfaceKey: string
  identityNorm: number
  relationNorm: number
  contextSupport: number
  contextRevision: number
  trace: ContextTrace
}

export interface AnalysisStats {
  characterCount: number
  graphemeCount: number
  paragraphCount: number
  neuronCount: number
  uniquePairCount: number
  matrixBytes: number
  similarityBytes: number
  durationMs: number
  medianRelationNorm: number
}

export interface TextLabAnalysis {
  format: 'metin-laboratuvari-analysis'
  version: 2
  runId: string
  createdAt: string
  /**
   * Optional evidence-lineage identity supplied by the ingest/runtime layer.
   * It never participates in I/R encoding; it only prevents correlated source
   * outputs from being counted as independent witnesses downstream.
   */
  independenceGroupId?: string
  sourceDigest: string
  identitySpaceId: string
  contextSpaceId: string
  representationSpaceId: string
  source: CanonicalSource
  config: TextLabConfig
  layout: MatrixLayout
  rootIds: string[]
  nodes: RawTextNeuron[]
  /** Row-major, symmetric K matrix. */
  similarities: Float32Array
  /** Row-major, directed initial H probabilities derived from positive K evidence. */
  habitProbabilities: Float32Array
  stats: AnalysisStats
}

export interface PipelineProgress {
  stage: PipelineStage
  completed: number
  total: number
  ratio: number
  message: string
}

export interface PairSimilarityDetail {
  combined: number
  identity: number
  relation: number
  relationConfidence: number
}

export interface ContextStepExplanation {
  before: Float32Array
  trace: Float32Array
  after: Float32Array
  parentTrace: Float32Array
  siblingTrace: Float32Array
}

export interface PersistedProject {
  format: 'metin-laboratuvari-project'
  version: 2
  /** IndexedDB compare-and-swap token. Zero means the source has not been stored yet. */
  storageRevision: number
  savedAt: string
  source: TextSource
  config: TextLabConfig
  analysis: TextLabAnalysis | null
}
