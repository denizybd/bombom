import { digestVector } from '../text-lab/engine/representation'
import type {
  BehaviorDiagnostics,
  BehaviorProfile,
  FamilyAnalysisInput,
  FamilyConfig,
  FamilyNeuronInput,
  FamilySynthesisDecisionInput,
  IndependentEvidenceRef,
  RegionAffinity,
  SparseVector,
} from './types'
import {
  FAMILY_EPSILON,
  clamp,
  compareText,
  createSparseVector,
  hashString,
  sparseCosine,
  stableDigest,
  weightedSparseMean,
} from './sparse'

export const BEHAVIOR_PROFILE_ENCODER_VERSION = 'structural-behavior-v1'

const graphemeSegmenter = new Intl.Segmenter('tr', { granularity: 'grapheme' })

function graphemes(value: string): string[] {
  return Array.from(graphemeSegmenter.segment(value), (entry) => entry.segment)
}

export function rawMemberKey(
  sourceDigest: string,
  analysisRunId: string,
  neuronId: string,
): string {
  return `${sourceDigest}\u0000${analysisRunId}\u0000${neuronId}`
}

export function synthesisMemberKey(
  representationSpaceId: string,
  candidateFingerprint: string,
): string {
  return `${representationSpaceId}\u0000${candidateFingerprint}`
}

export function createBehaviorSpaceId(
  representationSpaceId: string,
  analyses: readonly FamilyAnalysisInput[],
  config: FamilyConfig,
): string {
  const transitionSemantics = [...new Set(analyses.map((analysis) => stableDigest([
    analysis.config.identitySimilarityWeight,
    analysis.config.habitExponent,
    analysis.config.minimumLeafGraphemes,
  ])))].sort(compareText)
  const digest = stableDigest([
    BEHAVIOR_PROFILE_ENCODER_VERSION,
    representationSpaceId,
    config.profileDimension,
    config.maximumProfileFeatures,
    config.contextRadius,
    config.maximumTransitionNeighbors,
    config.hashSeed,
    ...transitionSemantics,
  ])
  return `BS-${digest.toUpperCase()}`
}

function denseNorm(vector: ArrayLike<number>): number {
  let sum = 0
  for (let index = 0; index < vector.length; index += 1) sum += vector[index] * vector[index]
  return Math.sqrt(sum)
}

function denseCosine(left: ArrayLike<number> | undefined, right: ArrayLike<number> | undefined): number {
  if (!left || !right || left.length !== right.length) return 0
  const leftNorm = denseNorm(left)
  const rightNorm = denseNorm(right)
  if (leftNorm <= FAMILY_EPSILON || rightNorm <= FAMILY_EPSILON) return 0
  let dot = 0
  for (let index = 0; index < left.length; index += 1) dot += left[index] * right[index]
  return clamp(dot / (leftNorm * rightNorm), -1, 1)
}

const topCoordinateCache = new WeakMap<Float32Array, Array<{ index: number; value: number }>>()

function topCoordinates(vector: Float32Array, maximum: number): Array<{ index: number; value: number }> {
  const cached = topCoordinateCache.get(vector)
  if (cached) return cached.slice(0, maximum)
  const retained: Array<{ index: number; value: number }> = []
  for (let index = 0; index < vector.length; index += 1) {
    const value = vector[index]
    if (Math.abs(value) <= FAMILY_EPSILON) continue
    retained.push({ index, value })
    retained.sort((left, right) => Math.abs(right.value) - Math.abs(left.value) || left.index - right.index)
    if (retained.length > maximum) retained.length = maximum
  }
  topCoordinateCache.set(vector, retained)
  return retained
}

function addHashed(
  accumulator: Map<number, number>,
  label: string,
  value: number,
  config: FamilyConfig,
): void {
  if (!Number.isFinite(value) || Math.abs(value) <= FAMILY_EPSILON) return
  const index = hashString(label, config.hashSeed) % config.profileDimension
  const sign = (hashString(label, config.hashSeed ^ 0x9e3779b9) & 1) === 0 ? 1 : -1
  accumulator.set(index, (accumulator.get(index) ?? 0) + sign * value)
}

function addVectorSignature(
  accumulator: Map<number, number>,
  prefix: string,
  vector: Float32Array,
  weight: number,
  config: FamilyConfig,
): void {
  for (const coordinate of topCoordinates(vector, 8)) {
    addHashed(
      accumulator,
      `${prefix}:${coordinate.index}:${coordinate.value >= 0 ? '+' : '-'}`,
      weight * Math.abs(coordinate.value),
      config,
    )
  }
}

interface AnalysisIndex {
  byId: Map<string, FamilyNeuronInput>
  nodeIndex: Map<string, number>
  leavesByParagraph: Map<number, FamilyNeuronInput[]>
  paragraphBounds: Map<number, { start: number; end: number }>
}

function createAnalysisIndex(analysis: FamilyAnalysisInput): AnalysisIndex {
  const byId = new Map(analysis.nodes.map((node) => [node.id, node]))
  const nodeIndex = new Map(analysis.nodes.map((node, index) => [node.id, index]))
  const leavesByParagraph = new Map<number, FamilyNeuronInput[]>()
  const paragraphBounds = new Map<number, { start: number; end: number }>()
  for (const node of analysis.nodes) {
    const bounds = paragraphBounds.get(node.paragraphIndex)
    paragraphBounds.set(node.paragraphIndex, {
      start: Math.min(bounds?.start ?? node.start, node.start),
      end: Math.max(bounds?.end ?? node.end, node.end),
    })
    if (node.childIds.length > 0 && node.splitBoundary !== 'leaf') continue
    const leaves = leavesByParagraph.get(node.paragraphIndex) ?? []
    leaves.push(node)
    leavesByParagraph.set(node.paragraphIndex, leaves)
  }
  for (const leaves of leavesByParagraph.values()) {
    leaves.sort((left, right) => left.start - right.start || left.end - right.end || compareText(left.id, right.id))
  }
  return { byId, nodeIndex, leavesByParagraph, paragraphBounds }
}

function contextLeaves(
  node: FamilyNeuronInput,
  index: AnalysisIndex,
  radius: number,
): { left: FamilyNeuronInput[]; right: FamilyNeuronInput[] } {
  const leaves = index.leavesByParagraph.get(node.paragraphIndex) ?? []
  let insertion = 0
  while (insertion < leaves.length && leaves[insertion].start < node.end) insertion += 1
  const left: FamilyNeuronInput[] = []
  for (let cursor = insertion - 1; cursor >= 0 && left.length < radius; cursor -= 1) {
    const candidate = leaves[cursor]
    if (candidate.id !== node.id && candidate.end <= node.start) left.push(candidate)
  }
  const right: FamilyNeuronInput[] = []
  for (let cursor = insertion; cursor < leaves.length && right.length < radius; cursor += 1) {
    const candidate = leaves[cursor]
    if (candidate.id !== node.id && candidate.start >= node.end) right.push(candidate)
  }
  return { left, right }
}

function topHabitNeighbors(
  analysis: FamilyAnalysisInput,
  index: AnalysisIndex,
  node: FamilyNeuronInput,
  maximum: number,
): Array<{ node: FamilyNeuronInput; probability: number }> {
  const row = index.nodeIndex.get(node.id)
  if (row === undefined) return []
  const count = analysis.nodes.length
  const retained: Array<{ node: FamilyNeuronInput; probability: number }> = []
  for (let column = 0; column < count; column += 1) {
    if (column === row) continue
    const probability = analysis.habitProbabilities[row * count + column] ?? 0
    if (!Number.isFinite(probability) || probability <= 0) continue
    retained.push({ node: analysis.nodes[column], probability })
    retained.sort((left, right) => right.probability - left.probability || compareText(left.node.id, right.node.id))
    if (retained.length > maximum) retained.length = maximum
  }
  return retained
}

function normalizedEntropy(probabilities: readonly number[]): number {
  const total = probabilities.reduce((sum, value) => sum + Math.max(0, value), 0)
  if (total <= FAMILY_EPSILON || probabilities.length <= 1) return 0
  let entropy = 0
  for (const value of probabilities) {
    const probability = Math.max(0, value) / total
    if (probability > FAMILY_EPSILON) entropy -= probability * Math.log(probability)
  }
  return clamp(entropy / Math.log(probabilities.length))
}

function regionAffinities(diagnostics: BehaviorDiagnostics): RegionAffinity[] {
  const shortness = 1 - diagnostics.spanScale
  const centrality = 1 - Math.abs(0.5 - diagnostics.relativePosition) * 2
  const content = clamp(
    0.25 * diagnostics.relationStrength +
    0.25 * diagnostics.transitionConcentration +
    0.25 * (1 - diagnostics.bridgeImpact) +
    0.25 * (0.35 + 0.65 * (1 - shortness)),
  )
  const event = clamp(
    0.35 * diagnostics.bridgeImpact +
    0.25 * diagnostics.transitionEntropy +
    0.2 * diagnostics.relationStrength +
    0.2 * centrality,
  )
  const role = clamp(
    0.3 * diagnostics.leftRightAsymmetry +
    0.25 * diagnostics.relationStrength +
    0.25 * shortness +
    0.2 * diagnostics.transitionConcentration,
  )
  const operator = clamp(
    0.4 * diagnostics.shortScopePotential +
    0.25 * shortness +
    0.2 * diagnostics.localContinuity +
    0.15 * diagnostics.leftRightAsymmetry,
  )
  const hypothesis = (
    region: RegionAffinity['region'],
    score: number,
    explanation: string,
  ): RegionAffinity => ({ region, score, verified: false, explanation })
  return [
    hypothesis('content', content, 'Taşıyıcılık, bağlam kararlılığı ve geçiş yoğunlaşmasına dayalı statik hipotez.'),
    hypothesis('event', event, 'Bağlam köprüleme, merkezilik ve geçiş çeşitliliğine dayalı statik hipotez.'),
    hypothesis('role', role, 'Yön asimetrisi, kısa biçim ve ilişki gücüne dayalı statik hipotez.'),
    hypothesis('operator', operator, 'Kısa kapsam, komşu sürekliliği ve yön etkisine dayalı statik hipotez.'),
    hypothesis('goal', 0, 'Amaç üyeliği statik metinden doğrulanamaz; beklenen çıktı deneyimi gerekir.'),
    hypothesis('language', 0, 'Dil üretim üyeliği statik metinden doğrulanamaz; söze dökme deneyimi gerekir.'),
    hypothesis('control', 0, 'Kontrol üyeliği statik metinden doğrulanamaz; çalışma zamanı geri bildirimi gerekir.'),
  ]
}

function evidenceRef(
  analysis: FamilyAnalysisInput,
  node: FamilyNeuronInput,
): IndependentEvidenceRef {
  return {
    memberKey: rawMemberKey(analysis.sourceDigest, analysis.runId, node.id),
    analysisRunId: analysis.runId,
    sourceId: analysis.source.id,
    sourceDigest: analysis.sourceDigest,
    neuronId: node.id,
    paragraphIndex: node.paragraphIndex,
    start: node.start,
    end: node.end,
    text: node.text,
  }
}

function rawBehaviorProfile(
  analysis: FamilyAnalysisInput,
  node: FamilyNeuronInput,
  index: AnalysisIndex,
  behaviorSpaceId: string,
  config: FamilyConfig,
): BehaviorProfile {
  const accumulator = new Map<number, number>()
  const context = contextLeaves(node, index, config.contextRadius)
  const parent = node.parentId ? index.byId.get(node.parentId) : undefined
  const sibling = parent?.childIds
    .map((id) => index.byId.get(id))
    .find((candidate) => candidate && candidate.id !== node.id)
  const habits = topHabitNeighbors(analysis, index, node, config.maximumTransitionNeighbors)
  const bounds = index.paragraphBounds.get(node.paragraphIndex) ?? { start: node.start, end: node.end }
  const paragraphLength = Math.max(1, bounds.end - bounds.start)
  const relativePosition = clamp(((node.start + node.end) / 2 - bounds.start) / paragraphLength)
  const spanScale = clamp(node.graphemeCount / Math.max(1, Math.sqrt(paragraphLength) * 2))
  const depthScale = clamp(node.depth / 12)
  const relationStrength = clamp(node.relationNorm / (1 + node.relationNorm))

  addHashed(accumulator, `structure:branch:${node.branch}`, 0.7, config)
  addHashed(accumulator, `structure:split:${node.splitBoundary}`, 0.7, config)
  addHashed(accumulator, `structure:leaf:${node.childIds.length === 0}`, 0.6, config)
  addHashed(accumulator, 'scalar:span', spanScale, config)
  addHashed(accumulator, 'scalar:depth', depthScale, config)
  addHashed(accumulator, 'scalar:position', relativePosition, config)
  addHashed(accumulator, 'scalar:relation-strength', relationStrength, config)
  addHashed(accumulator, 'trace:parent-gain', clamp(node.trace.parentGain), config)
  addHashed(accumulator, 'trace:sibling-gain', clamp(node.trace.siblingGain), config)

  context.left.forEach((neighbor, neighborIndex) => {
    const attenuation = 1 / (neighborIndex + 1)
    addVectorSignature(accumulator, `left:${neighborIndex}:I`, neighbor.identity, 0.65 * attenuation, config)
    addVectorSignature(accumulator, `left:${neighborIndex}:R`, neighbor.relation, 0.45 * attenuation, config)
    addHashed(accumulator, `left:${neighborIndex}:split:${neighbor.splitBoundary}`, 0.25 * attenuation, config)
  })
  context.right.forEach((neighbor, neighborIndex) => {
    const attenuation = 1 / (neighborIndex + 1)
    addVectorSignature(accumulator, `right:${neighborIndex}:I`, neighbor.identity, 0.65 * attenuation, config)
    addVectorSignature(accumulator, `right:${neighborIndex}:R`, neighbor.relation, 0.45 * attenuation, config)
    addHashed(accumulator, `right:${neighborIndex}:split:${neighbor.splitBoundary}`, 0.25 * attenuation, config)
  })
  if (parent) {
    addVectorSignature(accumulator, 'parent:I', parent.identity, 0.35, config)
    addVectorSignature(accumulator, 'parent:R', parent.relation, 0.25, config)
  }
  if (sibling) {
    addVectorSignature(accumulator, `sibling:${node.branch}:I`, sibling.identity, 0.35, config)
    addVectorSignature(accumulator, `sibling:${node.branch}:R`, sibling.relation, 0.25, config)
  }
  habits.forEach((neighbor, rank) => {
    const weight = Math.sqrt(neighbor.probability) / (rank + 1)
    addVectorSignature(accumulator, `habit:${rank}:I`, neighbor.node.identity, 0.4 * weight, config)
    addVectorSignature(accumulator, `habit:${rank}:R`, neighbor.node.relation, 0.25 * weight, config)
    addHashed(accumulator, `habit:${rank}:branch:${neighbor.node.branch}`, 0.2 * weight, config)
  })

  const units = graphemes(node.text)
  const trimmed = node.text.trim()
  const boundaryUnits = [
    ...units.slice(0, 3).map((unit, offset) => `prefix:${offset}:${unit}`),
    ...units.slice(-3).map((unit, offset) => `suffix:${offset}:${unit}`),
  ]
  for (const boundary of boundaryUnits) addHashed(accumulator, boundary, 0.12, config)
  addHashed(accumulator, 'surface:whitespace-ratio', node.text.length ? (node.text.match(/\s/gu)?.length ?? 0) / node.text.length : 0, config)
  addHashed(accumulator, 'surface:punctuation-ratio', trimmed.length ? (trimmed.match(/[^\p{L}\p{N}\s]/gu)?.length ?? 0) / trimmed.length : 0, config)

  const leftNearest = context.left[0]
  const rightNearest = context.right[0]
  const localContinuity = clamp((denseCosine(leftNearest?.relation, rightNearest?.relation) + 1) / 2)
  const leftFit = clamp((denseCosine(node.relation, leftNearest?.relation) + 1) / 2)
  const rightFit = clamp((denseCosine(node.relation, rightNearest?.relation) + 1) / 2)
  const bridgeImpact = clamp((leftFit + rightFit) / 2 - localContinuity)
  const leftRightAsymmetry = clamp(Math.abs(leftFit - rightFit))
  const transitionProbabilities = habits.map((habit) => habit.probability)
  const transitionEntropy = normalizedEntropy(transitionProbabilities)
  const transitionTotal = transitionProbabilities.reduce((sum, value) => sum + value, 0)
  const transitionConcentration = transitionTotal > FAMILY_EPSILON
    ? clamp((transitionProbabilities[0] ?? 0) / transitionTotal)
    : 0
  const shortScopePotential = clamp((1 - spanScale) * localContinuity * (0.5 + 0.5 * leftRightAsymmetry))
  const diagnostics: BehaviorDiagnostics = {
    spanScale,
    relativePosition,
    depthScale,
    relationStrength,
    transitionEntropy,
    transitionConcentration,
    localContinuity,
    bridgeImpact,
    leftRightAsymmetry,
    shortScopePotential,
  }
  ;([
    ['diagnostic:entropy', transitionEntropy],
    ['diagnostic:concentration', transitionConcentration],
    ['diagnostic:continuity', localContinuity],
    ['diagnostic:bridge', bridgeImpact],
    ['diagnostic:asymmetry', leftRightAsymmetry],
    ['diagnostic:scope', shortScopePotential],
  ] as const).forEach(([label, value]) => addHashed(accumulator, label, value, config))

  return {
    memberKey: rawMemberKey(analysis.sourceDigest, analysis.runId, node.id),
    kind: 'raw',
    representationSpaceId: analysis.representationSpaceId,
    behaviorSpaceId,
    vector: createSparseVector(config.profileDimension, accumulator, config.maximumProfileFeatures),
    diagnostics,
    regionAffinities: regionAffinities(diagnostics),
    evidence: [evidenceRef(analysis, node)],
    identityDigest: node.identityDigest,
    text: node.text,
    derivedFromMemberKeys: [],
  }
}

function averageDiagnostics(profiles: readonly BehaviorProfile[]): BehaviorDiagnostics {
  const keys: Array<keyof BehaviorDiagnostics> = [
    'spanScale',
    'relativePosition',
    'depthScale',
    'relationStrength',
    'transitionEntropy',
    'transitionConcentration',
    'localContinuity',
    'bridgeImpact',
    'leftRightAsymmetry',
    'shortScopePotential',
  ]
  const output = {} as BehaviorDiagnostics
  for (const key of keys) {
    output[key] = profiles.reduce((sum, profile) => sum + profile.diagnostics[key], 0) / Math.max(1, profiles.length)
  }
  return output
}

export function selectIndependentEvidence(
  evidence: readonly IndependentEvidenceRef[],
): IndependentEvidenceRef[] {
  const unique = new Map<string, IndependentEvidenceRef>()
  for (const item of evidence) {
    const key = `${item.sourceDigest}\u0000${item.paragraphIndex}\u0000${item.start}\u0000${item.end}`
    if (!unique.has(key)) unique.set(key, item)
  }
  const groups = new Map<string, IndependentEvidenceRef[]>()
  for (const item of unique.values()) {
    const key = `${item.sourceDigest}\u0000${item.paragraphIndex}`
    const group = groups.get(key) ?? []
    group.push(item)
    groups.set(key, group)
  }
  const selected: IndependentEvidenceRef[] = []
  for (const group of groups.values()) {
    group.sort((left, right) => left.end - right.end || left.start - right.start || compareText(left.memberKey, right.memberKey))
    let lastEnd = Number.NEGATIVE_INFINITY
    for (const item of group) {
      if (item.start < lastEnd) continue
      selected.push(item)
      lastEnd = item.end
    }
  }
  return selected.sort((left, right) =>
    compareText(left.sourceDigest, right.sourceDigest) ||
    left.paragraphIndex - right.paragraphIndex ||
    left.start - right.start ||
    left.end - right.end ||
    compareText(left.memberKey, right.memberKey),
  )
}

function synthesisProfile(
  decision: FamilySynthesisDecisionInput,
  rawProfilesByKey: ReadonlyMap<string, BehaviorProfile>,
  behaviorSpaceId: string,
  config: FamilyConfig,
): BehaviorProfile | null {
  if (decision.decision !== 'approved') return null
  const supports = decision.candidate.independentSupports
    .map((support) => rawProfilesByKey.get(support.key))
    .filter((profile): profile is BehaviorProfile => profile !== undefined)
  if (!supports.length) return null
  const diagnostics = averageDiagnostics(supports)
  const evidence = selectIndependentEvidence(supports.flatMap((profile) => profile.evidence))
  return {
    memberKey: synthesisMemberKey(decision.representationSpaceId, decision.candidateFingerprint),
    kind: 'synthesis',
    representationSpaceId: decision.representationSpaceId,
    behaviorSpaceId,
    vector: weightedSparseMean(
      supports.map((profile) => ({ vector: profile.vector, weight: 1 })),
      config.maximumProfileFeatures,
    ),
    diagnostics,
    regionAffinities: regionAffinities(diagnostics),
    evidence,
    identityDigest: decision.candidate.identityDigest,
    text: decision.candidate.medoid.text,
    derivedFromMemberKeys: supports.map((profile) => profile.memberKey).sort(compareText),
  }
}

export interface BuildProfilesResult {
  profiles: BehaviorProfile[]
  rawProfiles: BehaviorProfile[]
  synthesisProfiles: BehaviorProfile[]
  leafMemberKeys: Set<string>
}

export function buildBehaviorProfiles(
  analyses: readonly FamilyAnalysisInput[],
  decisions: readonly FamilySynthesisDecisionInput[],
  behaviorSpaceId: string,
  config: FamilyConfig,
  onProfile?: (completed: number, total: number) => void,
  shouldCancel?: () => boolean,
): BuildProfilesResult {
  const total = analyses.reduce((sum, analysis) => sum + analysis.nodes.length, 0)
  const rawProfiles: BehaviorProfile[] = []
  const leafMemberKeys = new Set<string>()
  let completed = 0
  for (const analysis of [...analyses].sort((left, right) => compareText(left.sourceDigest, right.sourceDigest))) {
    const index = createAnalysisIndex(analysis)
    for (const node of analysis.nodes) {
      if (shouldCancel?.()) throw new Error('FAMILY_CANCELLED')
      const profile = rawBehaviorProfile(analysis, node, index, behaviorSpaceId, config)
      rawProfiles.push(profile)
      if (node.childIds.length === 0 || node.splitBoundary === 'leaf') leafMemberKeys.add(profile.memberKey)
      completed += 1
      onProfile?.(completed, total)
    }
  }
  rawProfiles.sort((left, right) => compareText(left.memberKey, right.memberKey))
  const rawProfilesByKey = new Map(rawProfiles.map((profile) => [profile.memberKey, profile]))
  const synthesisProfiles = decisions
    .filter((decision) => decision.decision === 'approved')
    .sort((left, right) => compareText(left.candidateFingerprint, right.candidateFingerprint))
    .map((decision) => synthesisProfile(decision, rawProfilesByKey, behaviorSpaceId, config))
    .filter((profile): profile is BehaviorProfile => profile !== null)
  const profiles = [...rawProfiles, ...synthesisProfiles].sort((left, right) => compareText(left.memberKey, right.memberKey))
  return { profiles, rawProfiles, synthesisProfiles, leafMemberKeys }
}

export function profileSimilarity(left: BehaviorProfile, right: BehaviorProfile): number {
  if (left.behaviorSpaceId !== right.behaviorSpaceId) return 0
  return sparseCosine(left.vector, right.vector)
}

export function assertIdentityUnchanged(
  analyses: readonly FamilyAnalysisInput[],
  expected: ReadonlyMap<string, string>,
): void {
  for (const analysis of analyses) {
    for (const node of analysis.nodes) {
      const key = rawMemberKey(analysis.sourceDigest, analysis.runId, node.id)
      if (expected.get(key) !== node.identityDigest || digestVector(node.identity) !== node.identityDigest) {
        throw new TypeError('Aile işlemi sırasında sabit kimlik alanlarından biri değişti veya bozuldu.')
      }
    }
  }
}

export function snapshotIdentityDigests(
  analyses: readonly FamilyAnalysisInput[],
): Map<string, string> {
  return new Map(analyses.flatMap((analysis) => analysis.nodes.map((node) => [
    rawMemberKey(analysis.sourceDigest, analysis.runId, node.id),
    node.identityDigest,
  ] as const)))
}

export function profilesFingerprint(profiles: readonly BehaviorProfile[]): string {
  return stableDigest([
    'behavior-profiles-v1',
    ...[...profiles].sort((left, right) => compareText(left.memberKey, right.memberKey)).flatMap((profile) => [
      profile.memberKey,
      profile.identityDigest,
      ...profile.vector.entries.flatMap((entry) => [entry.index, Math.fround(entry.value)]),
    ]),
  ])
}

export function profileEvidenceConfidence(profile: BehaviorProfile): number {
  const sources = new Set(profile.evidence.map((item) => item.sourceDigest)).size
  return clamp(Math.log1p(profile.evidence.length) / Math.log(9)) * clamp(sources / 3)
}

export function profileRegionScore(profile: BehaviorProfile, region: RegionAffinity['region']): number {
  return profile.regionAffinities.find((item) => item.region === region)?.score ?? 0
}

export function emptySparseProfile(dimension: number): SparseVector {
  return { dimension, entries: [], norm: 0 }
}
