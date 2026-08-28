import { createMatrixLayout, validateConfig } from '../text-lab/domain/config'
import { canonicalizeSource } from '../text-lab/engine/segment'
import {
  createContextSpaceId,
  createIdentitySpaceId,
  createRepresentationSpaceId,
  digestText,
  digestVector,
} from '../text-lab/engine/representation'
import type {
  SynthesisCandidate,
  SynthesisAnalysisInput,
  SynthesisConfig,
  SynthesisEdge,
  SynthesisMemberRef,
  SynthesisNeuronInput,
  SynthesisProgress,
  SynthesisRun,
  SynthesisRunOptions,
  SynthesisStage,
} from './types'

const EPSILON = 1e-12

export const DEFAULT_SYNTHESIS_CONFIG: Readonly<SynthesisConfig> = Object.freeze({
  nodeSelection: 'leaves-or-short-spans',
  maximumSpanGraphemes: 24,
  minimumIdentitySimilarity: 0.82,
  minimumRelationSimilarity: -1,
  minimumCombinedSimilarity: 0.74,
  identityOverrideSimilarity: 0.97,
  identityWeight: 0.8,
  maximumEdgesPerMember: 12,
  maximumRetainedEdges: 20_000,
  maximumClusterMembers: 64,
  maximumCandidates: 256,
  minimumSupport: 3,
  minimumSources: 2,
  yieldEveryComparisons: 2_048,
})

type ValidationCode =
  | 'EMPTY_INPUT'
  | 'INVALID_ANALYSIS'
  | 'DUPLICATE_RUN'
  | 'DUPLICATE_MEMBER'
  | 'REPRESENTATION_SPACE_MISMATCH'
  | 'VECTOR_LENGTH_MISMATCH'
  | 'INVALID_VECTOR'
  | 'INVALID_CONFIG'

export class SynthesisValidationError extends TypeError {
  constructor(
    public readonly code: ValidationCode,
    message: string,
  ) {
    super(message)
    this.name = 'SynthesisValidationError'
  }
}

export class SynthesisCancelledError extends Error {
  constructor() {
    super('Nöron sentezi kullanıcı tarafından durduruldu.')
    this.name = 'SynthesisCancelledError'
  }
}

interface SynthesisReadyNeuron extends SynthesisNeuronInput {
  identity: Float32Array
  relation: Float32Array
}

interface SynthesisReadyAnalysis extends SynthesisAnalysisInput {
  representationSpaceId: string
  sourceDigest: string
  runId: string
  nodes: SynthesisReadyNeuron[]
}

interface PreparedMember {
  ref: SynthesisMemberRef
  node: SynthesisReadyNeuron
  identityNorm: number
  relationNorm: number
}

interface PreparedInput {
  analyses: SynthesisReadyAnalysis[]
  members: PreparedMember[]
  representationSpaceId: string
  identityLength: number
  relationLength: number
}

const textEncoder = new TextEncoder()

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

/** Small deterministic 128-bit content fingerprint, independent of object order. */
function fingerprintParts(parts: readonly (string | number | boolean | Float32Array)[]): string {
  const states = new Uint32Array([
    0x811c9dc5,
    0x9e3779b9,
    0x85ebca6b,
    0xc2b2ae35,
  ])
  const primes = [0x01000193, 0x27d4eb2d, 0x165667b1, 0x9e3779b1]

  const updateByte = (value: number) => {
    for (let index = 0; index < states.length; index += 1) {
      states[index] = Math.imul(states[index] ^ value, primes[index]) >>> 0
    }
  }
  const updateBytes = (bytes: Uint8Array) => {
    const length = bytes.length >>> 0
    updateByte(length & 0xff)
    updateByte((length >>> 8) & 0xff)
    updateByte((length >>> 16) & 0xff)
    updateByte((length >>> 24) & 0xff)
    for (const byte of bytes) updateByte(byte)
  }

  for (const part of parts) {
    if (part instanceof Float32Array) {
      updateByte(0x66)
      const bytes = new Uint8Array(part.length * Float32Array.BYTES_PER_ELEMENT)
      const view = new DataView(bytes.buffer)
      for (let index = 0; index < part.length; index += 1) {
        view.setFloat32(index * Float32Array.BYTES_PER_ELEMENT, part[index], true)
      }
      updateBytes(bytes)
    } else {
      updateByte(typeof part === 'string' ? 0x73 : typeof part === 'number' ? 0x6e : 0x62)
      updateBytes(textEncoder.encode(String(part)))
    }
  }

  return Array.from(states, (value) => value.toString(16).padStart(8, '0')).join('')
}

function finiteThreshold(value: number): boolean {
  return Number.isFinite(value) && value >= -1 && value <= 1
}

export function createSynthesisConfig(
  overrides: Partial<SynthesisConfig> = {},
): SynthesisConfig {
  const config: SynthesisConfig = { ...DEFAULT_SYNTHESIS_CONFIG, ...overrides }
  const selections = new Set([
    'leaves',
    'short-spans',
    'leaves-or-short-spans',
    'all',
  ])
  const positiveIntegers: Array<keyof SynthesisConfig> = [
    'maximumSpanGraphemes',
    'maximumEdgesPerMember',
    'maximumRetainedEdges',
    'maximumClusterMembers',
    'maximumCandidates',
    'minimumSupport',
    'minimumSources',
    'yieldEveryComparisons',
  ]

  if (!selections.has(config.nodeSelection)) {
    throw new SynthesisValidationError('INVALID_CONFIG', 'Nöron seçim filtresi tanınmıyor.')
  }
  if (
    positiveIntegers.some(
      (key) => !Number.isInteger(config[key]) || (config[key] as number) < 1,
    )
  ) {
    throw new SynthesisValidationError(
      'INVALID_CONFIG',
      'Sentez sınırları pozitif tam sayı olmalıdır.',
    )
  }
  if (
    !finiteThreshold(config.minimumIdentitySimilarity) ||
    !finiteThreshold(config.minimumRelationSimilarity) ||
    !finiteThreshold(config.minimumCombinedSimilarity) ||
    !finiteThreshold(config.identityOverrideSimilarity) ||
    !(config.identityWeight > 0 && config.identityWeight <= 1)
  ) {
    throw new SynthesisValidationError(
      'INVALID_CONFIG',
      'Benzerlik eşikleri -1 ile 1, kimlik ağırlığı 0 ile 1 arasında olmalıdır.',
    )
  }
  if (config.identityOverrideSimilarity < config.minimumIdentitySimilarity) {
    throw new SynthesisValidationError(
      'INVALID_CONFIG',
      'Kimlik geçersiz kılma eşiği asgari kimlik eşiğinden küçük olamaz.',
    )
  }
  if (config.minimumSources < 2) {
    throw new SynthesisValidationError(
      'INVALID_CONFIG',
      'Yeni bir sentez nöronu için en az iki bağımsız kaynak gerekir.',
    )
  }
  if (config.minimumSupport < config.minimumSources) {
    throw new SynthesisValidationError(
      'INVALID_CONFIG',
      'Asgari destek, asgari kaynak sayısından küçük olamaz.',
    )
  }
  if (config.maximumClusterMembers < config.minimumSupport) {
    throw new SynthesisValidationError(
      'INVALID_CONFIG',
      'Küme sınırı asgari destek sayısını karşılamalıdır.',
    )
  }
  return config
}

function vectorNorm(vector: Float32Array): number {
  let sum = 0
  for (let index = 0; index < vector.length; index += 1) sum += vector[index] ** 2
  return Math.sqrt(sum)
}

function cosine(
  left: Float32Array,
  right: Float32Array,
  leftNorm = vectorNorm(left),
  rightNorm = vectorNorm(right),
): number {
  if (leftNorm <= EPSILON || rightNorm <= EPSILON) return 0
  let dot = 0
  for (let index = 0; index < left.length; index += 1) dot += left[index] * right[index]
  return Math.max(-1, Math.min(1, dot / (leftNorm * rightNorm)))
}

function isFiniteVector(value: unknown): value is Float32Array {
  return (
    value instanceof Float32Array &&
    value.length > 0 &&
    value.every((entry) => Number.isFinite(entry))
  )
}

function assertReadyAnalysis(value: SynthesisAnalysisInput): asserts value is SynthesisReadyAnalysis {
  const candidate = value as Partial<SynthesisReadyAnalysis>
  if (
    !candidate ||
    typeof candidate !== 'object' ||
    typeof candidate.representationSpaceId !== 'string' ||
    !candidate.representationSpaceId ||
    typeof candidate.sourceDigest !== 'string' ||
    !candidate.sourceDigest ||
    typeof candidate.runId !== 'string' ||
    !candidate.runId ||
    !candidate.source ||
    typeof candidate.source.id !== 'string' ||
    !candidate.config ||
    !Array.isArray(candidate.nodes)
  ) {
    throw new SynthesisValidationError(
      'INVALID_ANALYSIS',
      'Metin Laboratuvarı kaydında uzay, kaynak veya deney kimliği eksik.',
    )
  }
  try {
    validateConfig(candidate.config)
  } catch (cause) {
    throw new SynthesisValidationError(
      'INVALID_ANALYSIS',
      cause instanceof Error ? cause.message : 'Metin Laboratuvarı ayarları geçersiz.',
    )
  }
  const canonical = canonicalizeSource(candidate.source)
  const expectedLayout = createMatrixLayout(candidate.config)
  const expectedIdentitySpaceId = createIdentitySpaceId(candidate.config, expectedLayout)
  const expectedContextSpaceId = createContextSpaceId(candidate.config, expectedLayout)
  const expectedRepresentationSpaceId = createRepresentationSpaceId(
    expectedIdentitySpaceId,
    expectedContextSpaceId,
  )
  if (
    candidate.source.text !== canonical.text ||
    candidate.sourceDigest !== digestText(canonical.text) ||
    !candidate.layout ||
    candidate.layout.side !== expectedLayout.side ||
    candidate.layout.cellCount !== expectedLayout.cellCount ||
    candidate.layout.identityRows !== expectedLayout.identityRows ||
    candidate.layout.identityLength !== expectedLayout.identityLength ||
    candidate.layout.relationStart !== expectedLayout.relationStart ||
    candidate.layout.relationLength !== expectedLayout.relationLength ||
    candidate.identitySpaceId !== expectedIdentitySpaceId ||
    candidate.contextSpaceId !== expectedContextSpaceId ||
    candidate.representationSpaceId !== expectedRepresentationSpaceId
  ) {
    throw new SynthesisValidationError(
      'INVALID_ANALYSIS',
      'Analizin kaynak özeti veya temsil uzayı parmak izi doğrulanamadı.',
    )
  }
  for (const node of candidate.nodes) {
    if (
      !node ||
      typeof node.id !== 'string' ||
      !node.id ||
      !isFiniteVector(node.identity) ||
      node.identity.length !== expectedLayout.identityLength ||
      !isFiniteVector(node.relation) ||
      node.relation.length !== expectedLayout.relationLength ||
      node.identityDigest !== digestVector(node.identity) ||
      node.sourceId !== candidate.source.id ||
      !Number.isInteger(node.start) ||
      !Number.isInteger(node.end) ||
      node.start < 0 ||
      node.end <= node.start ||
      node.end > canonical.text.length ||
      canonical.text.slice(node.start, node.end) !== node.text
    ) {
      throw new SynthesisValidationError(
        'INVALID_VECTOR',
        'Ham nöronun sabit kimliği, kaynak aralığı veya plastik ilişki vektörü geçersiz.',
      )
    }
  }
}

function eligible(node: SynthesisReadyNeuron, config: SynthesisConfig): boolean {
  const leaf = node.childIds.length === 0 || node.splitBoundary === 'leaf'
  const short = node.graphemeCount <= config.maximumSpanGraphemes
  switch (config.nodeSelection) {
    case 'leaves':
      return leaf
    case 'short-spans':
      return short
    case 'leaves-or-short-spans':
      return leaf || short
    case 'all':
      return true
  }
}

function memberKey(analysis: SynthesisReadyAnalysis, node: SynthesisReadyNeuron): string {
  return `${analysis.sourceDigest}\u0000${analysis.runId}\u0000${node.id}`
}

function prepareInput(
  input: readonly SynthesisAnalysisInput[],
  config: SynthesisConfig,
): PreparedInput {
  if (!input.length) {
    throw new SynthesisValidationError('EMPTY_INPUT', 'Sentez için en az bir analiz kaydı gerekir.')
  }
  for (const analysis of input) assertReadyAnalysis(analysis)

  const analyses = [...input] as SynthesisReadyAnalysis[]
  analyses.sort((left, right) =>
    compareText(left.sourceDigest, right.sourceDigest) ||
    compareText(left.runId, right.runId) ||
    compareText(left.source.id, right.source.id),
  )

  const representationSpaceId = analyses[0].representationSpaceId
  if (analyses.some((analysis) => analysis.representationSpaceId !== representationSpaceId)) {
    throw new SynthesisValidationError(
      'REPRESENTATION_SPACE_MISMATCH',
      'Farklı temsil uzaylarındaki nöronlar karşılaştırılamaz.',
    )
  }
  const runIds = new Set<string>()
  for (const analysis of analyses) {
    if (runIds.has(analysis.runId)) {
      throw new SynthesisValidationError(
        'DUPLICATE_RUN',
        `Aynı analiz deneyi iki kez verildi: ${analysis.runId}`,
      )
    }
    runIds.add(analysis.runId)
  }

  const firstNode = analyses.flatMap((analysis) => analysis.nodes).at(0)
  const identityLength = firstNode?.identity.length ?? 0
  const relationLength = firstNode?.relation.length ?? 0
  for (const analysis of analyses) {
    for (const node of analysis.nodes) {
      if (node.identity.length !== identityLength || node.relation.length !== relationLength) {
        throw new SynthesisValidationError(
          'VECTOR_LENGTH_MISMATCH',
          'Aynı temsil uzayındaki nöron vektörlerinin boyutları eşleşmiyor.',
        )
      }
    }
  }

  const members: PreparedMember[] = []
  const memberKeys = new Set<string>()
  for (const analysis of analyses) {
    for (const node of analysis.nodes) {
      if (!eligible(node, config)) continue
      const key = memberKey(analysis, node)
      if (memberKeys.has(key)) {
        throw new SynthesisValidationError(
          'DUPLICATE_MEMBER',
          `Aynı ham nöron bir analizde iki kez yer alıyor: ${node.id}`,
        )
      }
      memberKeys.add(key)
      const ref: SynthesisMemberRef = {
        key,
        analysisRunId: analysis.runId,
        sourceId: analysis.source.id,
        sourceDigest: analysis.sourceDigest,
        neuronId: node.id,
        paragraphIndex: node.paragraphIndex,
        start: node.start,
        end: node.end,
        text: node.text,
        graphemeCount: node.graphemeCount,
        depth: node.depth,
        parentId: node.parentId,
      }
      members.push({
        ref,
        node,
        identityNorm: vectorNorm(node.identity),
        relationNorm: vectorNorm(node.relation),
      })
    }
  }
  members.sort((left, right) => compareText(left.ref.key, right.ref.key))
  return { analyses, members, representationSpaceId, identityLength, relationLength }
}

function medianPositive(values: readonly number[]): number {
  const positive = values.filter((value) => value > EPSILON).sort((a, b) => a - b)
  if (!positive.length) return 0
  const middle = Math.floor(positive.length / 2)
  return positive.length % 2
    ? positive[middle]
    : (positive[middle - 1] + positive[middle]) / 2
}

function compareEdgeQuality(left: SynthesisEdge, right: SynthesisEdge): number {
  return (
    right.combinedSimilarity - left.combinedSimilarity ||
    right.identitySimilarity - left.identitySimilarity ||
    right.relationSimilarity - left.relationSimilarity ||
    compareText(left.left.key, right.left.key) ||
    compareText(left.right.key, right.right.key)
  )
}

function edgeFingerprint(
  representationSpaceId: string,
  left: SynthesisMemberRef,
  right: SynthesisMemberRef,
  identitySimilarity: number,
  relationSimilarity: number,
  relationConfidence: number,
  combinedSimilarity: number,
): string {
  return fingerprintParts([
    'synthesis-edge-v1',
    representationSpaceId,
    left.key,
    right.key,
    Math.fround(identitySimilarity),
    Math.fround(relationSimilarity),
    Math.fround(relationConfidence),
    Math.fround(combinedSimilarity),
  ])
}

class BoundedEdgeStore {
  private readonly buckets = new Map<string, SynthesisEdge[]>()

  constructor(private readonly maximumPerMember: number) {}

  add(edge: SynthesisEdge): void {
    this.addToBucket(edge.left.key, edge)
    this.addToBucket(edge.right.key, edge)
  }

  private addToBucket(key: string, edge: SynthesisEdge): void {
    const bucket = this.buckets.get(key) ?? []
    bucket.push(edge)
    bucket.sort(compareEdgeQuality)
    if (bucket.length > this.maximumPerMember) bucket.length = this.maximumPerMember
    this.buckets.set(key, bucket)
  }

  get peakBucketSize(): number {
    let peak = 0
    for (const bucket of this.buckets.values()) peak = Math.max(peak, bucket.length)
    return peak
  }

  materialize(maximum: number): SynthesisEdge[] {
    const unique = new Map<string, SynthesisEdge>()
    for (const bucket of this.buckets.values()) {
      for (const edge of bucket) unique.set(edge.fingerprint, edge)
    }
    return [...unique.values()].sort(compareEdgeQuality).slice(0, maximum)
  }
}

const stageRanges: Record<SynthesisStage, readonly [number, number]> = {
  validating: [0, 0.05],
  selecting: [0.05, 0.1],
  comparing: [0.1, 0.74],
  clustering: [0.74, 0.86],
  synthesizing: [0.86, 0.99],
  complete: [1, 1],
}

function report(
  options: SynthesisRunOptions,
  stage: SynthesisStage,
  completed: number,
  total: number,
  message: string,
  eligibleMemberCount: number,
  retainedEdgeCount: number,
  candidateCount: number,
): void {
  const [start, end] = stageRanges[stage]
  const localRatio = total === 0 ? 1 : completed / total
  options.onProgress?.({
    stage,
    completed,
    total,
    ratio: start + (end - start) * localRatio,
    message,
    eligibleMemberCount,
    retainedEdgeCount,
    candidateCount,
  })
}

function throwIfCancelled(options: SynthesisRunOptions): void {
  if (options.shouldCancel?.()) throw new SynthesisCancelledError()
}

async function yieldToHost(options: SynthesisRunOptions): Promise<void> {
  if (options.yieldControl) await options.yieldControl()
  else await new Promise<void>((resolve) => setTimeout(resolve, 0))
  throwIfCancelled(options)
}

function totalCrossSourcePairs(members: readonly PreparedMember[]): number {
  const sourceCounts = new Map<string, number>()
  for (const member of members) {
    sourceCounts.set(member.ref.sourceDigest, (sourceCounts.get(member.ref.sourceDigest) ?? 0) + 1)
  }
  let prior = 0
  let total = 0
  for (const count of sourceCounts.values()) {
    total += prior * count
    prior += count
  }
  return total
}

function groupMembersBySource(
  members: readonly PreparedMember[],
): Array<{ sourceDigest: string; members: PreparedMember[] }> {
  const groups = new Map<string, PreparedMember[]>()
  for (const member of members) {
    const group = groups.get(member.ref.sourceDigest) ?? []
    group.push(member)
    groups.set(member.ref.sourceDigest, group)
  }
  return [...groups.entries()]
    .sort(([left], [right]) => compareText(left, right))
    .map(([sourceDigest, sourceMembers]) => ({ sourceDigest, members: sourceMembers }))
}

function buildEdge(
  left: PreparedMember,
  right: PreparedMember,
  representationSpaceId: string,
  medianRelationNorm: number,
  config: SynthesisConfig,
): SynthesisEdge | null {
  const identitySimilarity = cosine(
    left.node.identity,
    right.node.identity,
    left.identityNorm,
    right.identityNorm,
  )
  if (identitySimilarity < config.minimumIdentitySimilarity) return null

  const relationSimilarity = cosine(
    left.node.relation,
    right.node.relation,
    left.relationNorm,
    right.relationNorm,
  )
  const relationConfidence = medianRelationNorm <= EPSILON
    ? 0
    : Math.min(1, left.relationNorm / medianRelationNorm) *
      Math.min(1, right.relationNorm / medianRelationNorm)
  const relationWeight = (1 - config.identityWeight) * relationConfidence
  const denominator = config.identityWeight + relationWeight
  const combinedSimilarity = denominator <= EPSILON
    ? identitySimilarity
    : (config.identityWeight * identitySimilarity + relationWeight * relationSimilarity) /
      denominator
  const identityOverride = identitySimilarity >= config.identityOverrideSimilarity
  if (
    !identityOverride &&
    (relationSimilarity < config.minimumRelationSimilarity ||
      combinedSimilarity < config.minimumCombinedSimilarity)
  ) {
    return null
  }

  const [first, second] = left.ref.key < right.ref.key ? [left.ref, right.ref] : [right.ref, left.ref]
  const fingerprint = edgeFingerprint(
    representationSpaceId,
    first,
    second,
    identitySimilarity,
    relationSimilarity,
    relationConfidence,
    combinedSimilarity,
  )
  return {
    id: `SE-${fingerprint.slice(0, 16)}`,
    fingerprint,
    left: first,
    right: second,
    identitySimilarity,
    relationSimilarity,
    relationConfidence,
    combinedSimilarity,
  }
}

class BoundedDisjointSet {
  private readonly parent = new Map<string, string>()
  private readonly members = new Map<string, string[]>()

  constructor(
    keys: readonly string[],
    private readonly maximumMembers: number,
    private readonly canMerge: (
      leftMembers: readonly string[],
      rightMembers: readonly string[],
    ) => boolean = () => true,
  ) {
    for (const key of keys) {
      this.parent.set(key, key)
      this.members.set(key, [key])
    }
  }

  private root(key: string): string {
    const parent = this.parent.get(key)
    if (parent === undefined) throw new Error(`Bilinmeyen küme üyesi: ${key}`)
    if (parent === key) return key
    const root = this.root(parent)
    this.parent.set(key, root)
    return root
  }

  union(left: string, right: string): boolean {
    const leftRoot = this.root(left)
    const rightRoot = this.root(right)
    if (leftRoot === rightRoot) return true
    const leftMembers = this.members.get(leftRoot) ?? []
    const rightMembers = this.members.get(rightRoot) ?? []
    if (leftMembers.length + rightMembers.length > this.maximumMembers) return false
    // Single-link clustering can join A and C through B even when A and C fail
    // the similarity contract. Requiring every cross-cluster pair to pass keeps
    // each candidate internally coherent instead of accepting similarity chains.
    if (!this.canMerge(leftMembers, rightMembers)) return false

    // Lexical root choice makes equal-score edge order and input order irrelevant.
    const [root, child] = leftRoot < rightRoot
      ? [leftRoot, rightRoot]
      : [rightRoot, leftRoot]
    const merged = [...(this.members.get(root) ?? []), ...(this.members.get(child) ?? [])]
      .sort(compareText)
    this.parent.set(child, root)
    this.members.set(root, merged)
    this.members.delete(child)
    return true
  }

  groups(): string[][] {
    return [...this.members.values()]
      .filter((members) => members.length > 1)
      .map((members) => [...members])
      .sort((left, right) => compareText(left[0], right[0]))
  }
}

/**
 * Counts observations without treating an ancestor and its overlapping
 * descendants (or duplicate runs of the same source text) as new evidence.
 */
function independentSupports(members: readonly PreparedMember[]): PreparedMember[] {
  const groups = new Map<string, PreparedMember[]>()
  for (const member of members) {
    const key = `${member.ref.sourceDigest}\u0000${member.ref.paragraphIndex}`
    const group = groups.get(key) ?? []
    group.push(member)
    groups.set(key, group)
  }

  const independent: PreparedMember[] = []
  for (const group of [...groups.values()]) {
    group.sort((left, right) =>
      left.ref.end - right.ref.end ||
      left.ref.start - right.ref.start ||
      compareText(left.ref.key, right.ref.key),
    )
    let lastEnd = Number.NEGATIVE_INFINITY
    for (const member of group) {
      if (member.ref.start < lastEnd) continue
      independent.push(member)
      lastEnd = member.ref.end
    }
  }
  return independent.sort((left, right) => compareText(left.ref.key, right.ref.key))
}

function averagePairSimilarity(
  members: readonly PreparedMember[],
  selector: 'identity' | 'relation',
): number {
  if (members.length < 2) return 1
  let sum = 0
  let count = 0
  for (let leftIndex = 0; leftIndex < members.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < members.length; rightIndex += 1) {
      const left = members[leftIndex]
      const right = members[rightIndex]
      sum += selector === 'identity'
        ? cosine(left.node.identity, right.node.identity, left.identityNorm, right.identityNorm)
        : cosine(left.node.relation, right.node.relation, left.relationNorm, right.relationNorm)
      count += 1
    }
  }
  return count ? sum / count : 1
}

function chooseIdentityMedoid(members: readonly PreparedMember[]): PreparedMember {
  let best = members[0]
  let bestScore = Number.NEGATIVE_INFINITY
  for (const candidate of members) {
    let score = 0
    for (const other of members) {
      score += cosine(
        candidate.node.identity,
        other.node.identity,
        candidate.identityNorm,
        other.identityNorm,
      )
    }
    if (
      score > bestScore + EPSILON ||
      (Math.abs(score - bestScore) <= EPSILON && candidate.ref.key < best.ref.key)
    ) {
      best = candidate
      bestScore = score
    }
  }
  return best
}

function averageRelation(members: readonly PreparedMember[], length: number): Float32Array {
  const relation = new Float32Array(length)
  if (!members.length) return relation
  for (let coordinate = 0; coordinate < length; coordinate += 1) {
    let sum = 0
    for (const member of members) sum += member.node.relation[coordinate]
    relation[coordinate] = sum / members.length
  }
  return relation
}

function buildCandidate(
  members: readonly PreparedMember[],
  representationSpaceId: string,
  relationLength: number,
  config: SynthesisConfig,
): SynthesisCandidate | null {
  const supports = independentSupports(members)
  const sourceCount = new Set(supports.map((member) => member.ref.sourceDigest)).size
  if (supports.length < config.minimumSupport || sourceCount < config.minimumSources) return null

  const medoid = chooseIdentityMedoid(supports)
  // Copying freezes the birth identity against later mutation of the raw archive.
  const identity = new Float32Array(medoid.node.identity)
  const identityDigest = digestVector(identity)
  const relation = averageRelation(supports, relationLength)
  const identityCohesion = averagePairSimilarity(supports, 'identity')
  const relationCohesion = averagePairSimilarity(supports, 'relation')
  const contextFactor = Math.max(0, Math.min(1, (relationCohesion + 1) / 2))
  const birthScore =
    Math.max(0, identityCohesion) * contextFactor * Math.log1p(supports.length) * sourceCount
  const sortedMembers = [...members].sort((left, right) => compareText(left.ref.key, right.ref.key))
  const fingerprint = fingerprintParts([
    'synthesis-candidate-v1',
    representationSpaceId,
    ...sortedMembers.map((member) => member.ref.key),
    'supports',
    ...supports.map((member) => member.ref.key),
    medoid.ref.key,
    identity,
    identityDigest,
    relation,
  ])

  return {
    id: `SN-${fingerprint.slice(0, 16)}`,
    fingerprint,
    representationSpaceId,
    members: sortedMembers.map((member) => member.ref),
    independentSupports: supports.map((member) => member.ref),
    medoid: medoid.ref,
    identity,
    identityDigest,
    relation,
    identityFrozen: true,
    contextRevision: 0,
    contextSupport: supports.length,
    supportCount: supports.length,
    sourceCount,
    identityCohesion,
    relationCohesion,
    birthScore,
  }
}

function compareCandidate(left: SynthesisCandidate, right: SynthesisCandidate): number {
  return (
    right.birthScore - left.birthScore ||
    right.sourceCount - left.sourceCount ||
    right.supportCount - left.supportCount ||
    right.identityCohesion - left.identityCohesion ||
    compareText(left.fingerprint, right.fingerprint)
  )
}

function configFingerprintParts(config: SynthesisConfig): Array<string | number> {
  return [
    config.nodeSelection,
    config.maximumSpanGraphemes,
    config.minimumIdentitySimilarity,
    config.minimumRelationSimilarity,
    config.minimumCombinedSimilarity,
    config.identityOverrideSimilarity,
    config.identityWeight,
    config.maximumEdgesPerMember,
    config.maximumRetainedEdges,
    config.maximumClusterMembers,
    config.maximumCandidates,
    config.minimumSupport,
    config.minimumSources,
    config.yieldEveryComparisons,
  ]
}

/**
 * Compares only cross-source nodes, streams qualifying evidence and produces
 * deterministic, bounded surface-pattern candidates without allocating NxN.
 */
export async function runSynthesis(
  input: readonly SynthesisAnalysisInput[],
  overrides: Partial<SynthesisConfig> = {},
  options: SynthesisRunOptions = {},
): Promise<SynthesisRun> {
  throwIfCancelled(options)
  const config = createSynthesisConfig(overrides)
  report(options, 'validating', 0, input.length, 'Analiz kayıtları doğrulanıyor…', 0, 0, 0)
  const prepared = prepareInput(input, config)
  report(
    options,
    'validating',
    input.length,
    input.length,
    'Temsil uzayı ve vektörler uyumlu',
    0,
    0,
    0,
  )
  throwIfCancelled(options)

  const inputNeuronCount = prepared.analyses.reduce(
    (total, analysis) => total + analysis.nodes.length,
    0,
  )
  report(
    options,
    'selecting',
    prepared.members.length,
    inputNeuronCount,
    `${prepared.members.length.toLocaleString('tr-TR')} uygun ham nöron seçildi`,
    prepared.members.length,
    0,
    0,
  )
  await yieldToHost(options)

  const comparisonTotal = totalCrossSourcePairs(prepared.members)
  const medianRelationNorm = medianPositive(
    prepared.members.map((member) => member.relationNorm),
  )
  const edgeStore = new BoundedEdgeStore(config.maximumEdgesPerMember)
  let comparisons = 0
  let thresholdEdges = 0
  report(
    options,
    'comparing',
    0,
    comparisonTotal,
    'Farklı kaynaklardaki uygun nöronlar karşılaştırılıyor…',
    prepared.members.length,
    0,
    0,
  )

  const sourceGroups = groupMembersBySource(prepared.members)
  for (let leftSourceIndex = 0; leftSourceIndex < sourceGroups.length; leftSourceIndex += 1) {
    for (
      let rightSourceIndex = leftSourceIndex + 1;
      rightSourceIndex < sourceGroups.length;
      rightSourceIndex += 1
    ) {
      for (const left of sourceGroups[leftSourceIndex].members) {
        for (const right of sourceGroups[rightSourceIndex].members) {
          const edge = buildEdge(
            left,
            right,
            prepared.representationSpaceId,
            medianRelationNorm,
            config,
          )
          comparisons += 1
          if (edge) {
            thresholdEdges += 1
            options.onEdge?.(edge)
            edgeStore.add(edge)
          }
          if (comparisons % config.yieldEveryComparisons === 0) {
            report(
              options,
              'comparing',
              comparisons,
              comparisonTotal,
              `${comparisons.toLocaleString('tr-TR')} kaynaklar arası çift incelendi`,
              prepared.members.length,
              Math.min(thresholdEdges, config.maximumRetainedEdges),
              0,
            )
            await yieldToHost(options)
          }
        }
      }
    }
  }
  throwIfCancelled(options)

  const edges = edgeStore.materialize(config.maximumRetainedEdges)
  report(
    options,
    'comparing',
    comparisonTotal,
    comparisonTotal,
    `${edges.length.toLocaleString('tr-TR')} seyrek bağ kümeleme için tutuldu`,
    prepared.members.length,
    edges.length,
    0,
  )

  const touchedKeys = new Set<string>()
  for (const edge of edges) {
    touchedKeys.add(edge.left.key)
    touchedKeys.add(edge.right.key)
  }
  const membersByKey = new Map(prepared.members.map((member) => [member.ref.key, member]))
  const clustersAreCompatible = (
    leftKeys: readonly string[],
    rightKeys: readonly string[],
  ): boolean => {
    for (const leftKey of leftKeys) {
      const left = membersByKey.get(leftKey)
      if (!left) return false
      for (const rightKey of rightKeys) {
        const right = membersByKey.get(rightKey)
        if (!right) return false
        if (
          !buildEdge(
            left,
            right,
            prepared.representationSpaceId,
            medianRelationNorm,
            config,
          )
        ) return false
      }
    }
    return true
  }
  const disjointSet = new BoundedDisjointSet(
    [...touchedKeys].sort(compareText),
    config.maximumClusterMembers,
    clustersAreCompatible,
  )
  for (let index = 0; index < edges.length; index += 1) {
    const edge = edges[index]
    disjointSet.union(edge.left.key, edge.right.key)
    if ((index + 1) % 512 === 0) {
      report(
        options,
        'clustering',
        index + 1,
        edges.length,
        'Güçlü bağlar sınırlı kümelere ayrılıyor…',
        prepared.members.length,
        edges.length,
        0,
      )
      await yieldToHost(options)
    }
  }
  const groups = disjointSet.groups()
  report(
    options,
    'clustering',
    edges.length,
    edges.length,
    `${groups.length.toLocaleString('tr-TR')} kanıt kümesi bulundu`,
    prepared.members.length,
    edges.length,
    0,
  )
  throwIfCancelled(options)

  const candidates: SynthesisCandidate[] = []
  for (let index = 0; index < groups.length; index += 1) {
    const members = groups[index]
      .map((key) => membersByKey.get(key))
      .filter((member): member is PreparedMember => member !== undefined)
    const candidate = buildCandidate(
      members,
      prepared.representationSpaceId,
      prepared.relationLength,
      config,
    )
    if (candidate) candidates.push(candidate)
    if ((index + 1) % 32 === 0) {
      report(
        options,
        'synthesizing',
        index + 1,
        groups.length,
        'Bağımsız desteklerden sabit kimlik adayları doğuyor…',
        prepared.members.length,
        edges.length,
        candidates.length,
      )
      await yieldToHost(options)
    }
  }
  candidates.sort(compareCandidate)
  if (candidates.length > config.maximumCandidates) candidates.length = config.maximumCandidates
  throwIfCancelled(options)

  const analysisRunIds = prepared.analyses.map((analysis) => analysis.runId).sort()
  const sourceDigests = [...new Set(prepared.analyses.map((analysis) => analysis.sourceDigest))].sort()
  const runFingerprint = fingerprintParts([
    'synthesis-run-v1',
    prepared.representationSpaceId,
    ...analysisRunIds,
    ...sourceDigests,
    ...configFingerprintParts(config),
    ...edges.map((edge) => edge.fingerprint),
    ...candidates.map((candidate) => candidate.fingerprint),
  ])
  const run: SynthesisRun = {
    format: 'noron-sentez-run',
    version: 1,
    id: `SR-${runFingerprint.slice(0, 16)}`,
    fingerprint: runFingerprint,
    representationSpaceId: prepared.representationSpaceId,
    analysisRunIds,
    sourceDigests,
    config: { ...config },
    edges,
    candidates,
    stats: {
      inputAnalysisCount: prepared.analyses.length,
      distinctSourceCount: sourceDigests.length,
      inputNeuronCount,
      eligibleNeuronCount: prepared.members.length,
      comparedPairCount: comparisons,
      thresholdEdgeCount: thresholdEdges,
      retainedEdgeCount: edges.length,
      clusterCount: groups.length,
      candidateCount: candidates.length,
      peakEdgesPerMember: edgeStore.peakBucketSize,
    },
  }
  report(
    options,
    'complete',
    1,
    1,
    `${candidates.length.toLocaleString('tr-TR')} sentez nöronu adayı hazır`,
    prepared.members.length,
    edges.length,
    candidates.length,
  )
  return run
}

export const synthesizeAnalyses = runSynthesis
