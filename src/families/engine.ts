import { createMatrixLayout, validateConfig as validateTextLabConfig } from '../text-lab/domain/config'
import { canonicalizeSource } from '../text-lab/engine/segment'
import { encodeIdentity } from '../text-lab/engine/encode'
import { vectorNorm } from '../text-lab/engine/matrix'
import {
  createContextSpaceId,
  createIdentitySpaceId,
  createRepresentationSpaceId,
  digestText,
  digestVector,
} from '../text-lab/engine/representation'
import { discoverFamilies } from './clustering'
import {
  assertIdentityUnchanged,
  buildBehaviorProfiles,
  createBehaviorSpaceId,
  profilesFingerprint,
  snapshotIdentityDigests,
} from './profile'
import {
  applyUtilityLifecycle,
  familyUtilityCaseCount,
  measureFamilyAblationUtilities,
  measureFamilyAblationUtilitiesOnEvaluation,
} from './utility'
import {
  buildFamilyTransitions,
  buildSparseNeuronTransitions,
  computeRouteDistribution,
} from './transitions'
import { buildEventCoalitions } from './coalition'
import type {
  FamilyAnalysisInput,
  FamilyConfig,
  FamilyDataSplitManifest,
  FamilyInputManifest,
  FamilyInputManifestEntry,
  FamilyProgress,
  FamilyRun,
  FamilyRunInput,
  FamilyRunOptions,
  FamilyStage,
  FamilySynthesisDecisionInput,
} from './types'
import { compareText, stableDigest } from './sparse'
import { createIndependentSourceIndex } from './independence'

export const DEFAULT_FAMILY_CONFIG: Readonly<FamilyConfig> = Object.freeze({
  profileDimension: 256,
  maximumProfileFeatures: 96,
  contextRadius: 2,
  maximumTransitionNeighbors: 8,
  maximumIndexBucketSize: 128,
  maximumCandidateNeighbors: 24,
  maximumFamilyMembers: 64,
  // Family formation is bounded by independent evidence and the number of
  // seed profiles.  A fixed product-wide ceiling used to truncate this list
  // at 128, making a larger corpus look as if it had stopped developing.
  // Keep the field for explicit experiment budgets, but make the default
  // effectively unbounded by an arbitrary family count.
  maximumFamilies: Number.MAX_SAFE_INTEGER,
  maximumMembershipsPerNeuron: 5,
  minimumFormationSimilarity: 0.48,
  minimumMembershipScore: 0.34,
  minimumCoreMembership: 0.56,
  minimumSupport: 3,
  minimumSources: 2,
  maximumOutgoingFamilyTransitions: 12,
  runtimeActivationThreshold: 0.18,
  maximumActiveNeurons: 128,
  maximumActiveFamilies: 16,
  maximumCoalitions: 8,
  maximumRoleCandidates: 4,
  coalitionBeamWidth: 32,
  routingCandidateLimit: 32,
  routingTemperature: 0.35,
  habitRouteWeight: 0.35,
  familyRouteWeight: 0.45,
  signalRouteWeight: 0.25,
  repetitionPenaltyWeight: 0.2,
  utilityMinimumCases: 8,
  utilityAlternativeCount: 4,
  utilityEpsilon: 1e-6,
  lifecycleStabilityThreshold: 0.75,
  lifecycleUtilityThreshold: 0.01,
  yieldEveryOperations: 512,
  hashSeed: 0x6a09e667,
})

type FamilyValidationCode =
  | 'EMPTY_INPUT'
  | 'INVALID_CONFIG'
  | 'INVALID_ANALYSIS'
  | 'INVALID_DATA_SPLIT'
  | 'REPRESENTATION_SPACE_MISMATCH'
  | 'INVALID_SYNTHESIS_DECISION'
  | 'CANCELLED'

export class FamilyValidationError extends TypeError {
  constructor(
    public readonly code: FamilyValidationCode,
    message: string,
  ) {
    super(message)
    this.name = 'FamilyValidationError'
  }
}

export class FamilyCancelledError extends Error {
  readonly code = 'CANCELLED'

  constructor() {
    super('Nöron aileleri deneyi kullanıcı tarafından durduruldu.')
    this.name = 'FamilyCancelledError'
  }
}

const POSITIVE_INTEGER_KEYS: Array<keyof FamilyConfig> = [
  'profileDimension',
  'maximumProfileFeatures',
  'contextRadius',
  'maximumTransitionNeighbors',
  'maximumIndexBucketSize',
  'maximumCandidateNeighbors',
  'maximumFamilyMembers',
  'maximumFamilies',
  'maximumMembershipsPerNeuron',
  'minimumSupport',
  'minimumSources',
  'maximumOutgoingFamilyTransitions',
  'maximumActiveNeurons',
  'maximumActiveFamilies',
  'maximumCoalitions',
  'maximumRoleCandidates',
  'coalitionBeamWidth',
  'routingCandidateLimit',
  'utilityMinimumCases',
  'utilityAlternativeCount',
  'yieldEveryOperations',
]

export function createFamilyConfig(overrides: Partial<FamilyConfig> = {}): FamilyConfig {
  const config = { ...DEFAULT_FAMILY_CONFIG, ...overrides }
  if (POSITIVE_INTEGER_KEYS.some((key) => !Number.isInteger(config[key]) || config[key] < 1)) {
    throw new FamilyValidationError('INVALID_CONFIG', 'Aile kapasite ve destek sınırları pozitif tam sayı olmalıdır.')
  }
  const unitKeys: Array<keyof FamilyConfig> = [
    'minimumFormationSimilarity',
    'minimumMembershipScore',
    'minimumCoreMembership',
    'runtimeActivationThreshold',
    'habitRouteWeight',
    'familyRouteWeight',
    'signalRouteWeight',
    'repetitionPenaltyWeight',
    'lifecycleStabilityThreshold',
  ]
  if (unitKeys.some((key) => !Number.isFinite(config[key]) || config[key] < 0 || config[key] > 1)) {
    throw new FamilyValidationError('INVALID_CONFIG', 'Aile eşikleri ve yönlendirme ağırlıkları 0 ile 1 arasında olmalıdır.')
  }
  if (
    !Number.isFinite(config.routingTemperature) || config.routingTemperature <= 0 ||
    !Number.isFinite(config.utilityEpsilon) || config.utilityEpsilon <= 0 || config.utilityEpsilon >= 1 ||
    !Number.isFinite(config.lifecycleUtilityThreshold) || config.lifecycleUtilityThreshold < 0 ||
    !Number.isInteger(config.hashSeed)
  ) {
    throw new FamilyValidationError('INVALID_CONFIG', 'Sıcaklık, fayda ve hash ayarları geçersiz.')
  }
  if (config.maximumProfileFeatures > config.profileDimension) {
    throw new FamilyValidationError('INVALID_CONFIG', 'Seyrek profil özelliği sayısı davranış uzayı boyutunu aşamaz.')
  }
  if (config.maximumFamilyMembers < config.minimumSupport) {
    throw new FamilyValidationError('INVALID_CONFIG', 'Aile üye sınırı asgari bağımsız desteği karşılamalıdır.')
  }
  if (config.minimumSources < 2 || config.minimumSupport < config.minimumSources) {
    throw new FamilyValidationError('INVALID_CONFIG', 'Aile doğumu için en az iki kaynak ve kaynak sayısından az olmayan destek gerekir.')
  }
  if (config.minimumCoreMembership < config.minimumMembershipScore) {
    throw new FamilyValidationError('INVALID_CONFIG', 'Çekirdek üyelik eşiği asgari üyelik eşiğinden küçük olamaz.')
  }
  return config
}

function finiteVector(value: unknown, length: number): value is Float32Array {
  return value instanceof Float32Array && value.length === length && value.every(Number.isFinite)
}

function validateAnalysis(analysis: FamilyAnalysisInput): void {
  if (
    !analysis ||
    analysis.format !== 'metin-laboratuvari-analysis' ||
    analysis.version !== 2 ||
    typeof analysis.runId !== 'string' || !analysis.runId ||
    (analysis.independenceGroupId !== undefined && (
      typeof analysis.independenceGroupId !== 'string' ||
      !analysis.independenceGroupId.trim() ||
      analysis.independenceGroupId.includes('\u0000')
    )) ||
    typeof analysis.sourceDigest !== 'string' || !analysis.sourceDigest ||
    typeof analysis.representationSpaceId !== 'string' || !analysis.representationSpaceId ||
    !analysis.source || !analysis.config || !analysis.layout ||
    !Array.isArray(analysis.nodes) || !analysis.nodes.length ||
    !(analysis.habitProbabilities instanceof Float32Array)
  ) {
    throw new FamilyValidationError('INVALID_ANALYSIS', 'Metin Laboratuvarı v2 aile girdisi eksik veya bozuk.')
  }
  try {
    validateTextLabConfig(analysis.config)
  } catch (cause) {
    throw new FamilyValidationError(
      'INVALID_ANALYSIS',
      cause instanceof Error ? cause.message : 'Metin Laboratuvarı ayarları geçersiz.',
    )
  }
  const canonical = canonicalizeSource(analysis.source)
  const expectedLayout = createMatrixLayout(analysis.config)
  const expectedIdentitySpaceId = createIdentitySpaceId(analysis.config, expectedLayout)
  const expectedContextSpaceId = createContextSpaceId(analysis.config, expectedLayout)
  if (
    canonical.text !== analysis.source.text ||
    digestText(canonical.text) !== analysis.sourceDigest ||
    analysis.layout.side !== expectedLayout.side ||
    analysis.layout.identityLength !== expectedLayout.identityLength ||
    analysis.layout.relationLength !== expectedLayout.relationLength ||
    analysis.identitySpaceId !== expectedIdentitySpaceId ||
    analysis.contextSpaceId !== expectedContextSpaceId ||
    analysis.representationSpaceId !== createRepresentationSpaceId(
      expectedIdentitySpaceId,
      expectedContextSpaceId,
    ) ||
    analysis.habitProbabilities.length !== analysis.nodes.length * analysis.nodes.length
  ) {
    throw new FamilyValidationError('INVALID_ANALYSIS', 'Aile girdisinin kaynak özeti, matris düzeni veya H⁰ boyutu doğrulanamadı.')
  }
  const ids = new Set<string>()
  for (const node of analysis.nodes) {
    if (!node || typeof node !== 'object' || typeof node.text !== 'string') {
      throw new FamilyValidationError('INVALID_ANALYSIS', 'Ham nöron kayıtlarından biri eksik veya bozuk.')
    }
    const expectedIdentity = encodeIdentity(node.text, expectedLayout, analysis.config)
    if (
      typeof node.id !== 'string' || !node.id || ids.has(node.id) ||
      node.sourceId !== analysis.source.id ||
      !Number.isInteger(node.start) || !Number.isInteger(node.end) ||
      node.start < 0 || node.end <= node.start || node.end > canonical.text.length ||
      canonical.text.slice(node.start, node.end) !== node.text ||
      !finiteVector(node.identity, expectedLayout.identityLength) ||
      !finiteVector(node.relation, expectedLayout.relationLength) ||
      digestVector(node.identity) !== node.identityDigest ||
      expectedIdentity.some((value, index) => value !== node.identity[index]) ||
      !Number.isFinite(node.identityNorm) || !Number.isFinite(node.relationNorm) ||
      Math.abs(vectorNorm(node.identity) - node.identityNorm) > 1e-5 ||
      Math.abs(vectorNorm(node.relation) - node.relationNorm) > 1e-5 ||
      node.identity.some((value) => value < -1 || value > 1) ||
      node.relation.some((value) => value < -1 || value > 1) ||
      !Number.isFinite(node.contextSupport) || node.contextSupport < 0 ||
      !Number.isInteger(node.contextRevision) || node.contextRevision < 0 ||
      !node.trace ||
      !Number.isFinite(node.trace.inheritedNorm) ||
      !Number.isFinite(node.trace.parentIdentityCompatibility) ||
      !Number.isFinite(node.trace.parentGain) ||
      !Number.isFinite(node.trace.siblingGain) ||
      (node.trace.siblingIdentityCompatibility !== null && !Number.isFinite(node.trace.siblingIdentityCompatibility)) ||
      !Array.isArray(node.childIds)
    ) {
      throw new FamilyValidationError('INVALID_ANALYSIS', 'Ham nöron kimliği, aralığı veya I/R vektörü geçersiz.')
    }
    ids.add(node.id)
  }
  for (const node of analysis.nodes) {
    if ((node.parentId && !ids.has(node.parentId)) || node.childIds.some((id) => !ids.has(id))) {
      throw new FamilyValidationError('INVALID_ANALYSIS', 'Ham nöronun bölünme ağacı eksik düğüm içeriyor.')
    }
  }
  if (!Array.isArray(analysis.rootIds) || analysis.rootIds.length < 1 || analysis.rootIds.some((id) => !ids.has(id))) {
    throw new FamilyValidationError('INVALID_ANALYSIS', 'Analizin paragraf kökleri eksik veya geçersiz.')
  }
  const count = analysis.nodes.length
  for (let row = 0; row < count; row += 1) {
    let sum = 0
    for (let column = 0; column < count; column += 1) {
      const probability = analysis.habitProbabilities[row * count + column]
      if (!Number.isFinite(probability) || probability < 0 || probability > 1.000001 || (row === column && probability !== 0)) {
        throw new FamilyValidationError('INVALID_ANALYSIS', 'Başlangıç H⁰ olasılıklarından biri geçersiz.')
      }
      sum += probability
    }
    if (sum > 1e-6 && Math.abs(sum - 1) > 1e-4) {
      throw new FamilyValidationError('INVALID_ANALYSIS', 'Başlangıç H⁰ satırlarından biri normalize değil.')
    }
  }
}

function validateDecision(
  decision: FamilySynthesisDecisionInput,
  representationSpaceId: string,
  identityLength: number,
  relationLength: number,
): void {
  const candidate = decision?.candidate
  if (
    !decision || decision.format !== 'noron-sentez-karari' || decision.version !== 1 ||
    (decision.decision !== 'approved' && decision.decision !== 'rejected') ||
    typeof decision.candidateFingerprint !== 'string' || !decision.candidateFingerprint ||
    decision.candidateFingerprint !== candidate?.fingerprint ||
    decision.representationSpaceId !== candidate?.representationSpaceId ||
    decision.representationSpaceId !== representationSpaceId ||
    candidate.identityFrozen !== true ||
    !finiteVector(candidate.identity, identityLength) ||
    !finiteVector(candidate.relation, relationLength) ||
    digestVector(candidate.identity) !== candidate.identityDigest ||
    !Array.isArray(candidate.independentSupports)
  ) {
    throw new FamilyValidationError('INVALID_SYNTHESIS_DECISION', 'Sentez kararı veya dondurulmuş aday kimliği doğrulanamadı.')
  }
}

function throwIfCancelled(options: FamilyRunOptions): void {
  if (options.shouldCancel?.()) throw new FamilyCancelledError()
}

function translateCancellation<T>(operation: () => T): T {
  try {
    return operation()
  } catch (cause) {
    if (cause instanceof Error && cause.message === 'FAMILY_CANCELLED') {
      throw new FamilyCancelledError()
    }
    throw cause
  }
}

async function yieldToHost(options: FamilyRunOptions): Promise<void> {
  if (options.yieldControl) await options.yieldControl()
  else await new Promise<void>((resolve) => setTimeout(resolve, 0))
  throwIfCancelled(options)
}

const STAGE_RANGES: Record<FamilyStage, readonly [number, number]> = {
  validating: [0, 0.06],
  profiling: [0.06, 0.34],
  indexing: [0.34, 0.42],
  clustering: [0.42, 0.59],
  membership: [0.59, 0.68],
  transitions: [0.68, 0.78],
  evaluating: [0.78, 0.91],
  coalition: [0.91, 0.99],
  complete: [1, 1],
}

function report(
  options: FamilyRunOptions,
  stage: FamilyStage,
  completed: number,
  total: number,
  message: string,
  counts: { profiles?: number; families?: number; memberships?: number; transitions?: number } = {},
): void {
  const [start, end] = STAGE_RANGES[stage]
  const local = total <= 0 ? 1 : Math.max(0, Math.min(1, completed / total))
  options.onProgress?.({
    stage,
    completed,
    total,
    ratio: start + (end - start) * local,
    message,
    profileCount: counts.profiles ?? 0,
    familyCount: counts.families ?? 0,
    membershipCount: counts.memberships ?? 0,
    transitionCount: counts.transitions ?? 0,
  })
}

export function createFamilyInputManifest(
  analyses: readonly FamilyAnalysisInput[],
  decisions: readonly FamilySynthesisDecisionInput[],
  dataSplit?: FamilyDataSplitManifest,
): FamilyInputManifest {
  const entries: FamilyInputManifestEntry[] = []
  for (const analysis of analyses) {
    const fingerprint = stableDigest([
      'family-analysis-input-v1',
      analysis.runId,
      analysis.independenceGroupId ?? '',
      analysis.sourceDigest,
      analysis.representationSpaceId,
      JSON.stringify(analysis.config),
      analysis.nodes.length,
      ...analysis.nodes.flatMap((node) => [
        node.id,
        node.identityDigest,
        digestVector(node.relation),
        node.contextRevision,
      ]),
      digestVector(analysis.habitProbabilities),
    ])
    entries.push({
      kind: 'text-lab-analysis',
      id: analysis.runId,
      sourceDigest: analysis.sourceDigest,
      analysisRunId: analysis.runId,
      representationSpaceId: analysis.representationSpaceId,
      revision: analysis.createdAt,
      fingerprint,
    })
  }
  for (const decision of decisions) {
    const fingerprint = stableDigest([
      'family-synthesis-input-v1',
      decision.id,
      decision.decision,
      decision.reviewedAt,
      decision.runFingerprint,
      decision.candidateFingerprint,
      decision.candidate.identityDigest,
      digestVector(decision.candidate.relation),
    ])
    entries.push({
      kind: 'synthesis-decision',
      id: decision.id,
      representationSpaceId: decision.representationSpaceId,
      revision: decision.reviewedAt,
      fingerprint,
    })
  }
  entries.sort((left, right) => compareText(left.kind, right.kind) || compareText(left.id, right.id))
  return {
    entries,
    corpusFingerprint: stableDigest([
      'family-corpus-v1',
      ...entries.flatMap((entry) => [entry.kind, entry.id, entry.revision, entry.fingerprint]),
      ...(dataSplit ? ['split-v1', JSON.stringify(dataSplit)] : []),
    ]),
    ...(dataSplit ? { dataSplit } : {}),
  }
}

interface ResolvedFamilyDataSplit {
  explicit: boolean
  discoveryAnalyses: FamilyAnalysisInput[]
  evaluationAnalyses: FamilyAnalysisInput[]
  projectionOnlyAnalyses: FamilyAnalysisInput[]
  manifest?: FamilyDataSplitManifest
  warnings: string[]
}

function normalizedGroupIds(
  value: readonly string[] | undefined,
  label: string,
): string[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) {
    throw new FamilyValidationError('INVALID_DATA_SPLIT', `${label} bağımsız kaynak grupları bir dizi olmalıdır.`)
  }
  const result = new Set<string>()
  for (const item of value) {
    const normalized = typeof item === 'string' ? item.trim() : ''
    if (!normalized || normalized.includes('\u0000')) {
      throw new FamilyValidationError('INVALID_DATA_SPLIT', `${label} bağımsız kaynak gruplarından biri geçersiz.`)
    }
    result.add(normalized)
  }
  return [...result].sort(compareText)
}

function analysisGroupId(analysis: FamilyAnalysisInput): string {
  return analysis.independenceGroupId?.trim() ?? ''
}

function resolveFamilyDataSplit(
  analyses: readonly FamilyAnalysisInput[],
  input: FamilyRunInput,
): ResolvedFamilyDataSplit {
  const requestedDiscovery = normalizedGroupIds(
    input.discoveryIndependenceGroupIds,
    'Keşif/eğitim',
  )
  const requestedEvaluation = normalizedGroupIds(
    input.evaluationIndependenceGroupIds,
    'Değerlendirme',
  )
  const explicit = requestedDiscovery !== undefined || requestedEvaluation !== undefined
  if (!explicit) {
    return {
      explicit: false,
      discoveryAnalyses: [...analyses],
      evaluationAnalyses: [...analyses],
      projectionOnlyAnalyses: [],
      warnings: [],
    }
  }

  const discoveryIds = new Set(requestedDiscovery ?? [])
  const evaluationIds = new Set(requestedEvaluation ?? [])
  for (const groupId of discoveryIds) {
    if (evaluationIds.has(groupId)) {
      throw new FamilyValidationError(
        'INVALID_DATA_SPLIT',
        `Aynı bağımsız kaynak grubu keşif ve değerlendirme bölümlerinde kullanılamaz: ${groupId}`,
      )
    }
  }

  const discoveryAnalyses = requestedDiscovery !== undefined
    ? analyses.filter((analysis) => discoveryIds.has(analysisGroupId(analysis)))
    : analyses.filter((analysis) => !evaluationIds.has(analysisGroupId(analysis)))
  const evaluationAnalyses = requestedEvaluation !== undefined
    ? analyses.filter((analysis) => evaluationIds.has(analysisGroupId(analysis)))
    : []
  if (!discoveryAnalyses.length) {
    throw new FamilyValidationError(
      'INVALID_DATA_SPLIT',
      'Bildirilmiş keşif/eğitim kaynak grupları hiçbir analizle eşleşmedi; sızıntıyı önlemek için tüm-korpus geri dönüşü yapılmadı.',
    )
  }

  // A copied chunk or shared provenance lineage can bridge two differently
  // named declarations. Resolve that transitive equivalence before allowing a
  // split, otherwise the same physical evidence appears on both sides.
  if (evaluationAnalyses.length) {
    const sourceIndex = createIndependentSourceIndex(analyses.map((analysis) => ({
      independenceGroupId: analysis.independenceGroupId,
      sourceId: analysis.source.id,
      sourceDigest: analysis.sourceDigest,
    })))
    const discoveryGroups = new Set(discoveryAnalyses.map((analysis) => sourceIndex.groupKey({
      independenceGroupId: analysis.independenceGroupId,
      sourceId: analysis.source.id,
      sourceDigest: analysis.sourceDigest,
    })))
    const overlap = new Set(evaluationAnalyses.map((analysis) => sourceIndex.groupKey({
      independenceGroupId: analysis.independenceGroupId,
      sourceId: analysis.source.id,
      sourceDigest: analysis.sourceDigest,
    })).filter((group) => discoveryGroups.has(group)))
    if (overlap.size) {
      throw new FamilyValidationError(
        'INVALID_DATA_SPLIT',
        'Keşif ve değerlendirme kümeleri kopya içerik, ortak kaynak kimliği veya ortak üretim soyu üzerinden kesişiyor.',
      )
    }
  }

  const discoveryRunIds = new Set(discoveryAnalyses.map((analysis) => analysis.runId))
  const evaluationRunIds = new Set(evaluationAnalyses.map((analysis) => analysis.runId))
  const projectionOnlyAnalyses = analyses.filter((analysis) => (
    !discoveryRunIds.has(analysis.runId) && !evaluationRunIds.has(analysis.runId)
  ))
  const availableGroupIds = new Set(analyses.map(analysisGroupId).filter(Boolean))
  const missingDiscovery = (requestedDiscovery ?? []).filter((group) => !availableGroupIds.has(group))
  const missingEvaluation = (requestedEvaluation ?? []).filter((group) => !availableGroupIds.has(group))
  const utilityFallback = evaluationAnalyses.length ? 'none' : 'discovery-source-holdout'
  const warnings: string[] = []
  if (missingDiscovery.length) {
    warnings.push(`${missingDiscovery.length} bildirilmiş keşif/eğitim kaynak grubu mevcut analizlerde bulunamadı.`)
  }
  if (missingEvaluation.length) {
    warnings.push(`${missingEvaluation.length} bildirilmiş değerlendirme kaynak grubu mevcut analizlerde bulunamadı.`)
  }
  if (utilityFallback !== 'none') {
    warnings.push('Ayrı değerlendirme analizi bulunamadı; fayda yalnız keşif korpusu içindeki kaynak-holdout tanısı olarak hesaplandı ve dış holdout kanıtı sayılmamalıdır.')
  }
  const manifest: FamilyDataSplitManifest = {
    mode: 'declared-independent-groups',
    discoveryCorpusFingerprint: createFamilyInputManifest(discoveryAnalyses, []).corpusFingerprint,
    discoveryIndependenceGroupIds: requestedDiscovery ?? [...new Set(discoveryAnalyses.map(analysisGroupId).filter(Boolean))].sort(compareText),
    evaluationIndependenceGroupIds: requestedEvaluation ?? [],
    discoveryAnalysisRunIds: [...discoveryRunIds].sort(compareText),
    evaluationAnalysisRunIds: [...evaluationRunIds].sort(compareText),
    projectionOnlyAnalysisRunIds: projectionOnlyAnalyses.map((analysis) => analysis.runId).sort(compareText),
    utilityFallback,
  }
  return {
    explicit: true,
    discoveryAnalyses,
    evaluationAnalyses,
    projectionOnlyAnalyses,
    manifest,
    warnings,
  }
}

function peakCountBy<T>(items: readonly T[], key: (item: T) => string): number {
  const counts = new Map<string, number>()
  let peak = 0
  for (const item of items) {
    const id = key(item)
    const count = (counts.get(id) ?? 0) + 1
    counts.set(id, count)
    peak = Math.max(peak, count)
  }
  return peak
}

/** Full deterministic profile -> family -> G -> ablation -> coalition pipeline. */
export async function runFamilyLearning(
  input: FamilyRunInput,
  overrides: Partial<FamilyConfig> = {},
  options: FamilyRunOptions = {},
): Promise<FamilyRun> {
  const config = createFamilyConfig(overrides)
  throwIfCancelled(options)
  if (!input.analyses?.length) {
    throw new FamilyValidationError('EMPTY_INPUT', 'Nöron aileleri için en az bir tamamlanmış Metin Laboratuvarı analizi gerekir.')
  }
  report(options, 'validating', 0, input.analyses.length, 'Salt okunur I/R/H⁰ girdileri doğrulanıyor…')
  const analyses = [...input.analyses].sort((left, right) =>
    compareText(left.sourceDigest, right.sourceDigest) || compareText(left.runId, right.runId),
  )
  analyses.forEach(validateAnalysis)
  const analysisKeys = new Set<string>()
  for (const analysis of analyses) {
    const key = `${analysis.sourceDigest}\u0000${analysis.runId}`
    if (analysisKeys.has(key)) {
      throw new FamilyValidationError('INVALID_ANALYSIS', `Aynı analiz aile girdisine iki kez verildi: ${analysis.runId}`)
    }
    analysisKeys.add(key)
  }
  const representationSpaceId = analyses[0].representationSpaceId
  if (analyses.some((analysis) => analysis.representationSpaceId !== representationSpaceId)) {
    throw new FamilyValidationError('REPRESENTATION_SPACE_MISMATCH', 'Farklı I/R temsil uzayları aynı aile deneyinde karıştırılamaz.')
  }
  const decisions = (input.synthesisDecisions ?? [])
    .filter((decision) => decision.representationSpaceId === representationSpaceId)
    .sort((left, right) => compareText(left.candidateFingerprint, right.candidateFingerprint))
  const layout = analyses[0].layout
  decisions.forEach((decision) => validateDecision(
    decision,
    representationSpaceId,
    layout.identityLength,
    layout.relationLength,
  ))
  const decisionKeys = new Set<string>()
  for (const decision of decisions) {
    const key = `${decision.id}\u0000${decision.candidateFingerprint}`
    if (decisionKeys.has(key)) {
      throw new FamilyValidationError('INVALID_SYNTHESIS_DECISION', `Aynı sentez kararı aile girdisine iki kez verildi: ${decision.id}`)
    }
    decisionKeys.add(key)
  }
  const approvedDecisions = decisions.filter((decision) => decision.decision === 'approved')
  const dataSplit = resolveFamilyDataSplit(analyses, input)
  const discoveryAnalyses = dataSplit.discoveryAnalyses
  const independentSources = createIndependentSourceIndex(discoveryAnalyses.map((analysis) => ({
    independenceGroupId: analysis.independenceGroupId,
    sourceId: analysis.source.id,
    sourceDigest: analysis.sourceDigest,
  })))
  const identitySnapshot = snapshotIdentityDigests(analyses)
  const inputManifest = createFamilyInputManifest(analyses, decisions, dataSplit.manifest)
  // Evaluation-only configuration must not rename the learned behavior space.
  // All profiles are projected into the discovery-defined space below.
  const behaviorSpaceId = createBehaviorSpaceId(representationSpaceId, discoveryAnalyses, config)
  // Replaying the same corpus (for example to inspect a transient coalition)
  // is not a new developmental observation. Lifecycle promotion requires a
  // different, compatible corpus snapshot.
  const previousLearningRun = input.previousRun &&
    input.previousRun.behaviorSpaceId === behaviorSpaceId &&
    (
      input.previousRun.inputManifest.dataSplit?.discoveryCorpusFingerprint ??
      input.previousRun.inputManifest.corpusFingerprint
    ) !== (
      inputManifest.dataSplit?.discoveryCorpusFingerprint ??
      inputManifest.corpusFingerprint
    )
      ? input.previousRun
      : null
  report(options, 'validating', analyses.length, analyses.length, 'Temsil ve arşiv bütünlüğü doğrulandı')
  await yieldToHost(options)

  const rawNeuronCount = analyses.reduce((sum, analysis) => sum + analysis.nodes.length, 0)
  report(options, 'profiling', 0, rawNeuronCount, 'Q davranış profilleri çıkarılıyor…')
  const profileResult = translateCancellation(() => buildBehaviorProfiles(
    analyses,
    approvedDecisions,
    behaviorSpaceId,
    config,
    (completed, total) => report(options, 'profiling', completed, total, 'Komşuluk, yapı, H⁰ ve çıkarma izleri Q profiline yazılıyor…', { profiles: completed }),
    options.shouldCancel,
  ))
  assertIdentityUnchanged(analyses, identitySnapshot)
  report(options, 'profiling', rawNeuronCount, rawNeuronCount, `${profileResult.profiles.length} bounded seyrek Q profili hazır`, { profiles: profileResult.profiles.length })
  await yieldToHost(options)

  report(options, 'indexing', 0, profileResult.rawProfiles.length, 'Ters özellik indeksi davranış adaylarını sınırlıyor…', { profiles: profileResult.profiles.length })
  const discoveryRunIds = new Set(discoveryAnalyses.map((analysis) => analysis.runId))
  const discoveryRawProfiles = profileResult.rawProfiles.filter((profile) => (
    profile.evidence.some((evidence) => discoveryRunIds.has(evidence.analysisRunId))
  ))
  const discoveryMemberKeys = new Set(discoveryRawProfiles.map((profile) => profile.memberKey))
  const discoveryLeafMemberKeys = new Set(
    [...profileResult.leafMemberKeys].filter((memberKey) => discoveryMemberKeys.has(memberKey)),
  )
  const discovery = translateCancellation(() => discoverFamilies(
    discoveryRawProfiles,
    profileResult.profiles,
    discoveryLeafMemberKeys,
    config,
    previousLearningRun,
    options.shouldCancel,
  ))
  report(options, 'indexing', profileResult.rawProfiles.length, profileResult.rawProfiles.length, `${discovery.edges.length} seyrek davranış bağı tutuldu`, {
    profiles: profileResult.profiles.length,
    families: discovery.families.length,
  })
  report(options, 'clustering', discovery.edges.length, discovery.edges.length, `${discovery.families.length} complete-link aile adayı doğdu`, {
    profiles: profileResult.profiles.length,
    families: discovery.families.length,
  })
  report(options, 'membership', discovery.memberships.length, discovery.memberships.length, `${discovery.memberships.length} top-k yumuşak M üyeliği hazır`, {
    profiles: profileResult.profiles.length,
    families: discovery.families.length,
    memberships: discovery.memberships.length,
  })
  await yieldToHost(options)

  report(options, 'transitions', 0, discoveryAnalyses.length, 'Yalnız keşif/eğitim H⁰ önseli aile düzeyindeki seyrek G kanıtına toplanıyor…', {
    profiles: profileResult.profiles.length,
    families: discovery.families.length,
    memberships: discovery.memberships.length,
  })
  const discoveryMemberships = discovery.memberships.filter((membership) => discoveryMemberKeys.has(membership.memberKey))
  const neuronTransitions = translateCancellation(() => buildSparseNeuronTransitions(
    discoveryAnalyses,
    discoveryMemberKeys,
    config,
    options.shouldCancel,
  ))
  const transitionResult = buildFamilyTransitions(
    neuronTransitions,
    discoveryRawProfiles,
    discovery.families,
    discoveryMemberships,
    config,
  )
  report(options, 'transitions', discoveryAnalyses.length, discoveryAnalyses.length, `${transitionResult.familyTransitions.length} bounded aile geçişi hazır`, {
    profiles: profileResult.profiles.length,
    families: discovery.families.length,
    memberships: discovery.memberships.length,
    transitions: transitionResult.familyTransitions.length,
  })
  await yieldToHost(options)

  report(options, 'evaluating', 0, discovery.families.length, 'Aileler kaynak-holdout karşı-olgusal susturma testine giriyor…', {
    profiles: profileResult.profiles.length,
    families: discovery.families.length,
    memberships: discovery.memberships.length,
    transitions: transitionResult.familyTransitions.length,
  })
  const utilityProgress = (completed: number, total: number) => report(
    options,
    'evaluating',
    completed,
    total,
    dataSplit.explicit && dataSplit.evaluationAnalyses.length
      ? 'Keşifte öğrenilen model ile aile-susturulmuş model ayrı değerlendirme aday evreninde karşılaştırılıyor…'
      : 'Full model ile aile-susturulmuş model aynı aday evreninde karşılaştırılıyor…',
    {
      profiles: profileResult.profiles.length,
      families: discovery.families.length,
      memberships: discovery.memberships.length,
      transitions: transitionResult.familyTransitions.length,
    },
  )
  const measuredUtilityReports = translateCancellation(() => (
    dataSplit.explicit && dataSplit.evaluationAnalyses.length
      ? measureFamilyAblationUtilitiesOnEvaluation(
          discoveryAnalyses,
          dataSplit.evaluationAnalyses,
          discovery.families,
          discovery.memberships,
          config,
          options.shouldCancel,
          utilityProgress,
        )
      : measureFamilyAblationUtilities(
          discoveryAnalyses,
          discovery.families,
          discovery.memberships,
          config,
          options.shouldCancel,
          utilityProgress,
        )
  ))
  // An explicitly requested split without evaluation data may still expose a
  // useful within-discovery diagnostic, but it must never unlock lifecycle
  // promotion as though an external holdout had passed.
  const utilityReports = dataSplit.explicit && !dataSplit.evaluationAnalyses.length
    ? measuredUtilityReports.map((report) => ({
        ...report,
        status: 'insufficient-evidence' as const,
        explanation: `Ayrı değerlendirme korpusu bulunmadı; yalnız tanısal değer hesaplandı. ${report.explanation}`,
      }))
    : measuredUtilityReports
  const families = applyUtilityLifecycle(discovery.families, utilityReports, config)
  await yieldToHost(options)

  report(options, 'coalition', 0, 1, 'Runtime aktivasyonlarından geçici olay koalisyonları kuruluyor…', {
    profiles: profileResult.profiles.length,
    families: families.length,
    memberships: discovery.memberships.length,
    transitions: transitionResult.familyTransitions.length,
  })
  const coalitionResult = buildEventCoalitions(
    input.activations ?? [],
    profileResult.profiles,
    families,
    discovery.memberships,
    transitionResult.familyTransitions,
    config,
    input.previousRun?.coalition,
  )
  const activations = [...(input.activations ?? [])]
    .filter((activation) => Number.isFinite(activation.activation) && activation.activation > 0)
    .sort((left, right) => right.activation - left.activation || compareText(left.memberKey, right.memberKey))
    .slice(0, config.maximumActiveNeurons)
  const current = activations[0]
  const routingPreview = current
    ? computeRouteDistribution(
        current.memberKey,
        profileResult.profiles,
        discovery.memberships,
        transitionResult.neuronTransitions,
        transitionResult.familyTransitions,
        config,
        {
          candidateMemberKeys: new Set(activations.slice(1).map((activation) => activation.memberKey)),
          signalCompatibility: new Map(activations.map((activation) => [activation.memberKey, activation.activation])),
          repetition: new Map(activations.map((activation) => [activation.memberKey, activation.repetition ?? 0])),
        },
      )
    : null
  report(options, 'coalition', 1, 1, `${coalitionResult.state.coalitions.length} geçici olay koalisyonu etkin`, {
    profiles: profileResult.profiles.length,
    families: families.length,
    memberships: discovery.memberships.length,
    transitions: transitionResult.familyTransitions.length,
  })

  assertIdentityUnchanged(analyses, identitySnapshot)
  const warnings: string[] = [
    'İçerik, olay, rol ve operatör bölgeleri statik davranış hipotezidir; insan anlamı veya doğruluk kanıtı değildir.',
    'Amaç, dil ve kontrol bölgeleri runtime/expected-output deneyimi gelene kadar doğrulanmamış ve dormant kalır.',
    'Metin Laboratuvarı H⁰ değerleri yaşanmış kas hafızası değil, benzerlikten türetilmiş başlangıç önselidir.',
    ...dataSplit.warnings,
  ]
  const declaredSourceCount = new Set(discoveryAnalyses.map((analysis) => analysis.source.id)).size
  if (independentSources.groupCount < declaredSourceCount) {
    warnings.push(`${declaredSourceCount - independentSources.groupCount} kaynak kimliği; yeniden adlandırılmış içerik kopyası veya ortak üretim soyu olduğu için bağımsız kaynak desteği sayılmadı.`)
  }
  const transitionSemantics = new Set(discoveryAnalyses.map((analysis) => `${analysis.config.identitySimilarityWeight}:${analysis.config.habitExponent}`))
  if (transitionSemantics.size > 1) warnings.push('Seçili analizlerin H⁰ ayarları farklı; davranış uzayı bu farkı parmak izine dahil etti.')
  if (!families.length) warnings.push('Seçili eşiklerle bağımsız ve complete-link uyumlu aile oluşmadı.')
  const unknownRegionCount = families.filter((family) => family.region === 'unknown').length
  if (unknownRegionCount) {
    warnings.push(`${unknownRegionCount} ailede statik bölge puanı veya ilk iki hipotez farkı yetersiz kaldı; semantik etiket yerine “belirsiz” korundu.`)
  }
  if (approvedDecisions.length > profileResult.synthesisProfiles.length) {
    warnings.push(`${approvedDecisions.length - profileResult.synthesisProfiles.length} onaylı sentez adayı, ham destek profili bulunamadığı için aile deneyine katılamadı.`)
  }
  const fingerprint = stableDigest([
    'family-run-v1',
    inputManifest.corpusFingerprint,
    behaviorSpaceId,
    JSON.stringify(config),
    profilesFingerprint(profileResult.profiles),
    ...families.map((family) => family.fingerprint),
    ...discovery.memberships.flatMap((membership) => [membership.memberKey, membership.familyId, Math.fround(membership.weight)]),
    ...transitionResult.familyTransitions.flatMap((transition) => [transition.fromFamilyId, transition.toFamilyId, Math.fround(transition.probability)]),
  ])
  const run: FamilyRun = {
    format: 'noron-aileleri-run',
    version: 1,
    id: `FR-${fingerprint.slice(0, 16)}`,
    fingerprint,
    createdAt: new Date().toISOString(),
    representationSpaceId,
    behaviorSpaceId,
    config: { ...config },
    inputManifest,
    profiles: profileResult.profiles,
    families,
    memberships: discovery.memberships,
    neuronTransitions: transitionResult.neuronTransitions,
    familyTransitions: transitionResult.familyTransitions,
    utilityReports,
    coalition: coalitionResult.state,
    routingPreview,
    warnings,
    stats: {
      inputAnalysisCount: analyses.length,
      ...(dataSplit.explicit ? {
        discoveryAnalysisCount: discoveryAnalyses.length,
        evaluationAnalysisCount: dataSplit.evaluationAnalyses.length,
        projectionOnlyAnalysisCount: dataSplit.projectionOnlyAnalyses.length,
      } : {}),
      approvedSynthesisCount: profileResult.synthesisProfiles.length,
      distinctSourceCount: independentSources.groupCount,
      rawNeuronCount,
      profileCount: profileResult.profiles.length,
      profileFeatureCount: profileResult.profiles.reduce((sum, profile) => sum + profile.vector.entries.length, 0),
      candidateComparisonCount: discovery.stats.candidateComparisonCount,
      retainedBehaviorEdgeCount: discovery.stats.retainedBehaviorEdgeCount,
      familyCount: families.length,
      membershipCount: discovery.memberships.length,
      peakMembershipsPerNeuron: peakCountBy(discovery.memberships, (membership) => membership.memberKey),
      retainedFamilyTransitionCount: transitionResult.familyTransitions.length,
      peakOutgoingFamilyTransitions: peakCountBy(transitionResult.familyTransitions, (transition) => transition.fromFamilyId),
      utilityCaseCount: familyUtilityCaseCount(utilityReports),
      coalitionExpansionCount: coalitionResult.expansionCount,
      peakCoalitionBeamWidth: coalitionResult.peakBeamWidth,
    },
  }
  report(options, 'complete', 1, 1, `${families.length} aile, ${run.coalition.coalitions.length} geçici koalisyon ve held-out fayda raporu hazır`, {
    profiles: run.profiles.length,
    families: run.families.length,
    memberships: run.memberships.length,
    transitions: run.familyTransitions.length,
  })
  return run
}

export const learnNeuronFamilies = runFamilyLearning
