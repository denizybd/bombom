import { createHash } from 'node:crypto'
import { normalizeTextSource } from './source-policy.mjs'
import { checkTurkishSpelling, TURKISH_SPELLING_POLICY_ID } from './turkish-spelling.mjs'
import {
  teacherAttemptPromptTokenUpperBound,
  teacherPromptTokenUpperBound,
  TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
} from './teacher-usage.mjs'
import {
  evaluateContinuousResourceBudget,
  evaluateIndependentStructuralEvidence,
  findIndependentStructuralBenchmarkReport,
  INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND,
  INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
} from './structural-stop-gate.mjs'
import {
  createTeacherChoiceBoard,
  materializeTeacherChoice,
  teacherChoicePromptBoard,
  teacherChoiceSchema,
  TEACHER_FEED_CATALOG_FINGERPRINT,
  TEACHER_FEED_CATALOG_ID,
  TEACHER_FEED_FINITE_VERB_SURFACE,
  TEACHER_FEED_MORPHOLOGY_ID,
  TEACHER_FEED_RENDERER_ID,
} from './teacher-feed-catalog.mjs'
import { canonicalOllamaIndependenceGroupId } from './ollama-lineage.mjs'
import {
  createTeacherDiscoveryAttempt,
  TEACHER_DISCOVERY_PROTOCOL,
  TEACHER_DISCOVERY_PROTOCOL_FINGERPRINT,
} from './teacher-free-discovery.mjs'

export const TEACHER_FEED_JOB_TYPE = 'teacher-feed-campaign'
export const TEACHER_FEED_CURRICULUM_ID = 'turkish-foundations-v1'
export const TEACHER_FEED_GENERATION_MODES = Object.freeze(['controlled', 'hybrid'])
export const TEACHER_FEED_HYBRID_POLICY = Object.freeze({
  id: 'alternating-controlled-first-v1',
  controlledShare: 0.5,
  discoveryShare: 0.5,
})

const MAX_BATCH_SIZE = 2
const MAX_GENERATION_CALLS_PER_BATCH = 3
const DEFAULT_DEDUP_WINDOW = 2_048
const STRUCTURAL_BENCHMARK_PROPAGATED_SAFETY_CODES = new Set([
  'JOB_PAUSED',
  'JOB_CANCELLED',
  'STALE_JOB_LEASE',
  'LEASE_EXPIRED',
  'SERVICE_STOP',
  'RUN_TIME_BUDGET_EXCEEDED',
  'RAM_BUDGET_EXCEEDED',
  'DISK_RESOURCE_RESERVE_EXCEEDED',
  'DISK_BUDGET_EXCEEDED',
  'ARTIFACT_PAYLOAD_TOO_LARGE',
  'NEURON_RESIDENCY_RESOURCE_WAIT',
])
const TURKISH_NEGATION_SURFACE = /(?:^|[^\p{L}\p{M}])(?:(?:değil|yok)|\p{L}{2,}(?:m(?:ı|i|u|ü)yor|(?:ma|me)(?:dı|di|du|dü|tı|ti|tu|tü|mış|miş|muş|müş|yacak|yecek|z))\p{L}{0,6})(?=$|[^\p{L}\p{M}])/iu
const TURKISH_QUESTION_CUE = /(?:^|[^\p{L}\p{M}])(?:mı|mi|mu|mü|kim|ne|neden|niçin|nasıl|nerede|nereye|nereden|hangi|kaç)(?=$|[^\p{L}\p{M}])/iu
const LEADING_DISCOURSE_CONNECTOR = /^(?:ama|fakat|ancak|oysa|sonra|ardından|böylece|bunun\s+üzerine|çünkü|bu\s+nedenle|o\s+yüzden|ve)(?=$|[^\p{L}\p{M}])/iu

/**
 * A deliberately small, auditable first curriculum. It controls coverage; the
 * seeded model call still supplies lexical and surface diversity. No item here
 * is an evaluation label and every generated source remains training-only.
 */
export const TURKISH_FOUNDATIONS_V1 = Object.freeze([
  { id: 'canonical-transitive', title: 'Temel yapan-eylem-etkilenen', forms: ['statement', 'contrast'], instruction: 'Açık bir yapan, somut bir eylem ve eylemden etkilenen bir varlık kullan.' },
  { id: 'accusative-object', title: 'Belirtili nesne', forms: ['statement', 'contrast'], instruction: 'Belirtili nesne eki alan doğal ve kısa cümleler kur.' },
  { id: 'dative-goal', title: 'Yönelme ve hedef', forms: ['statement', 'continuation'], instruction: 'Bir hareketin veya aktarımın yöneldiği hedefi açıkça belirt.' },
  { id: 'locative-context', title: 'Yer bağlamı', forms: ['statement', 'contrast'], instruction: 'Olay yerini -da, -de, -ta veya -te bulunma ekiyle açıkça belirt.', requiredSurface: /\b\p{L}{3,}(?:da|de|ta|te)\b/iu },
  { id: 'ablative-source', title: 'Çıkış ve kaynak', forms: ['statement', 'continuation'], instruction: 'Çıkış kaynağını -dan, -den, -tan veya -ten ekiyle açıkça belirt.', requiredSurface: /\b\p{L}{3,}(?:dan|den|tan|ten)\b/iu },
  { id: 'instrumental-tool', title: 'Araç ilişkisi', forms: ['statement', 'contrast'], instruction: 'Eylemin aracını ile, kullanarak, yardımıyla veya -la/-le araç ekiyle açıkça belirt.', requiredSurface: /\b(?:ile|kullanarak|yardımıyla|\p{L}{3,}(?:la|le))\b/iu },
  { id: 'negation', title: 'Olumsuzluk', forms: ['negative'], instruction: 'Olumsuzluğu yüzeyde açıkça görülen, tek anlamlı cümleler kur.', requiredSurface: TURKISH_NEGATION_SURFACE },
  { id: 'interrogative', title: 'Soru biçimi', forms: ['question'], instruction: 'Tek bir olayı soran, cevabı olay rollerinden çıkarılabilen soru cümleleri kur.', requiredSurface: /\?$/u },
  { id: 'tense-aspect', title: 'Zaman ve görünüş', forms: ['statement', 'contrast'], instruction: 'Fiilde geçmiş, şimdiki, gelecek veya geniş zaman ekini yüzeyde açıkça kullan.', requiredSurface: TEACHER_FEED_FINITE_VERB_SURFACE },
  { id: 'modifier-possessive', title: 'Niteleme ve iyelik', forms: ['statement', 'contrast'], instruction: 'İyelik veya ad tamlamasını yüzeyde açıkça kullan.', requiredSurface: /(?:['’](?:ın|in|un|ün|nın|nin|nun|nün)|\b\p{L}{3,}(?:ın|in|un|ün|nın|nin|nun|nün|sı|si|su|sü|ları|leri))\b/iu },
  { id: 'coordination', title: 'Bağlama', forms: ['statement', 'continuation'], instruction: 'İki olayı ve, ama, sonra, ardından veya çünkü bağlarından biriyle ilişkilendir.', requiredSurface: /\b(?:ve|ama|sonra|ardından|çünkü)\b/iu },
  { id: 'pronoun-reference', title: 'Gönderim', forms: ['statement', 'continuation'], instruction: 'Gönderimi açık olan o, ona, onu, onları, onun, bu, buna veya bunu zamirlerinden birini kullan.', requiredSurface: /(?:^|[^\p{L}\p{M}])(?:o|ona|onu|onları|onun|bu|buna|bunu)(?=$|[^\p{L}\p{M}])/iu },
])

const CURRICULUM_SURFACE_AXES = Object.freeze({
  domain: ['ev ve gündelik yaşam', 'atölye ve üretim', 'okul ve öğrenme', 'ulaşım ve lojistik', 'doğa ve bahçe', 'alışveriş ve hizmet', 'sanat ve hobi', 'spor ve hareket'],
  tense: ['belirgin geçmiş zaman', 'şimdiki zaman', 'gelecek zaman', 'geniş zaman'],
  length: ['kısa: 4–8 kelime', 'orta: 9–15 kelime', 'uzun: 16–24 kelime'],
  wordOrder: ['doğal özne-nesne-yüklem', 'doğal ama devrik olmayan odaklı sıra', 'zaman veya yer ifadesiyle başlayan sıra'],
})

const FORM_SURFACE_POLICIES = Object.freeze({
  statement: {
    instruction: 'Tek bir bildirme cümlesi kur ve noktayla bitir.',
    generationInstruction: 'clause alanına tek bir bildirme cümlesinin noktalamasız gövdesini yaz.',
    requiredSurface: /\.$/u,
  },
  contrast: {
    instruction: 'Ama, fakat, ancak veya oysa ile açık bir karşıtlık kur.',
    generationInstruction: 'firstClause ve secondClause alanlarına anlamca karşıt iki noktalamasız cümle parçası yaz; bağlaç yazma.',
    requiredSurface: /\b(?:ama|fakat|ancak|oysa)\b/iu,
  },
  continuation: {
    instruction: 'Sonra, ardından, bunun üzerine veya böylece ile açık bir devam ilişkisi kur.',
    generationInstruction: 'firstClause ve secondClause alanlarına birbirini izleyen iki noktalamasız olay parçası yaz; bağlaç yazma.',
    requiredSurface: /\b(?:sonra|ardından|böylece)\b|\bbunun\s+üzerine\b/iu,
  },
  negative: {
    instruction: 'Yüklemde açık olumsuzluk kullan.',
    generationInstruction: 'clause alanına yükleminde açık olumsuzluk bulunan tek bir noktalamasız cümle gövdesi yaz.',
    requiredSurface: TURKISH_NEGATION_SURFACE,
  },
  question: {
    instruction: 'Tek bir soru cümlesi kur ve soru işaretiyle bitir.',
    generationInstruction: 'clause alanına mı/mi/mu/mü parçacığı veya kim, ne, neden, nasıl, nerede, hangi, kaç gibi açık bir soru sözcüğü taşıyan tek bir noktalamasız soru gövdesi yaz.',
    requiredSurface: /\?$/u,
  },
})

const COMPOSITE_FORM_CONNECTORS = Object.freeze({
  contrast: Object.freeze(['ama', 'fakat', 'ancak', 'oysa']),
  continuation: Object.freeze(['sonra', 'ardından', 'böylece']),
})
const COMPOSITE_UNIT_CONNECTORS = Object.freeze({
  coordination: Object.freeze(['ve', 'çünkü']),
})

export const TEACHER_SURFACE_COMPOSITION_POLICY_ID = TEACHER_FEED_RENDERER_ID

const sha256 = (value) => createHash('sha256').update(String(value)).digest('hex')

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]))
  }
  return value
}

const stableHash = (value) => sha256(JSON.stringify(stableValue(value)))

export const TEACHER_FEED_PROTOCOL_VERSION = 9
export const TEACHER_FEED_PROTOCOL = Object.freeze({
  version: TEACHER_FEED_PROTOCOL_VERSION,
  promptPolicyId: 'turkish-foundations-closed-choice-board-v2',
  outputSchemaPolicyId: 'bounded-unique-integer-choice-array-v1',
  compositionPolicyId: TEACHER_SURFACE_COMPOSITION_POLICY_ID,
  catalogId: TEACHER_FEED_CATALOG_ID,
  catalogFingerprint: TEACHER_FEED_CATALOG_FINGERPRINT,
  morphologyPolicyId: TEACHER_FEED_MORPHOLOGY_ID,
  qualityPolicyId: 'teacher-text-structural-plus-surface-faithful-semantics-v7',
  checkpointPolicyId: 'per-attempt-pre-reserved-request-and-context-v3',
  completionBudgetPolicyId: 'closed-choice-fixed128-context4096-batch-max2-v3',
  contextTokenLimit: TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
  spellingPolicyId: TURKISH_SPELLING_POLICY_ID,
  duplicatePolicyId: 'semantic-identity-plus-character-4gram-window-v3',
  samplingPolicyId: 'ollama-seeded-closed-choice-context4096-bounded-v3',
})
export const TEACHER_FEED_PROTOCOL_FINGERPRINT = stableHash({
  ...TEACHER_FEED_PROTOCOL,
  curriculum: TURKISH_FOUNDATIONS_V1.map((unit) => ({
    id: unit.id,
    title: unit.title,
    forms: unit.forms,
    instruction: unit.instruction,
    requiredSurface: unit.requiredSurface ? String(unit.requiredSurface) : null,
  })),
  surfaceAxes: CURRICULUM_SURFACE_AXES,
  formPolicies: Object.fromEntries(Object.entries(FORM_SURFACE_POLICIES).map(([id, policy]) => [id, {
    instruction: policy.instruction,
    generationInstruction: policy.generationInstruction,
    requiredSurface: String(policy.requiredSurface),
  }])),
  compositeConnectors: COMPOSITE_FORM_CONNECTORS,
  compositeUnitConnectors: COMPOSITE_UNIT_CONNECTORS,
  leadingDiscourseConnector: String(LEADING_DISCOURSE_CONNECTOR),
  questionCue: String(TURKISH_QUESTION_CUE),
  maxGenerationBatchSize: MAX_BATCH_SIZE,
  catalogFingerprint: TEACHER_FEED_CATALOG_FINGERPRINT,
  deterministicLogic: [
    mulberry32,
    mixSeed,
    seededShuffle,
    planTeacherFeedBatch,
    requestedWordRange,
    canonicalizeTeacherText,
    canonicalizeForDuplicate,
    gramsFor,
    createTeacherDuplicateSignature,
    teacherNearDuplicateScore,
    evaluateTeacherText,
    filterTeacherLessons,
    buildMessages,
    teacherCompletionTokenLimit,
  ].map(String),
})
export const TEACHER_FEED_HYBRID_PROTOCOL_FINGERPRINT = stableHash({
  baseProtocolFingerprint: TEACHER_FEED_PROTOCOL_FINGERPRINT,
  discoveryProtocolFingerprint: TEACHER_DISCOVERY_PROTOCOL_FINGERPRINT,
  scheduling: TEACHER_FEED_HYBRID_POLICY,
})

export function teacherFeedGenerationLane(generationMode, batchIndex) {
  if (generationMode !== 'hybrid') return 'controlled'
  return batchIndex % 2 === 0 ? 'controlled' : 'discovery'
}

function emptyLaneCounters() {
  return { generated: 0, accepted: 0, duplicates: 0, rejectedQuality: 0, teacherRequests: 0, teacherTokens: 0 }
}

function integer(value, name, minimum, maximum, fallback) {
  const resolved = value === undefined ? fallback : value
  if (!Number.isInteger(resolved) || resolved < minimum || resolved > maximum) {
    const error = new TypeError(`${name}, ${minimum} ile ${maximum} arasında bir tam sayı olmalı.`)
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  return resolved
}

function ratio(value, name, minimum, maximum, fallback) {
  const resolved = value === undefined ? fallback : value
  if (typeof resolved !== 'number' || !Number.isFinite(resolved) || resolved < minimum || resolved > maximum) {
    const error = new TypeError(`${name}, ${minimum} ile ${maximum} arasında olmalı.`)
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  return resolved
}

function stringValue(value, name, { minimum = 1, maximum = 200, fallback } = {}) {
  const resolved = value === undefined ? fallback : value
  if (typeof resolved !== 'string' || resolved.trim().length < minimum || resolved.trim().length > maximum) {
    const error = new TypeError(`${name}, ${minimum}–${maximum} karakter arasında olmalı.`)
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  return resolved.trim()
}

function normalizeModelIdentity(modelRecord, requestedModel) {
  const name = stringValue(modelRecord?.name ?? modelRecord?.model ?? requestedModel, 'model', { maximum: 160 })
  const details = modelRecord?.details && typeof modelRecord.details === 'object' ? modelRecord.details : {}
  const identity = {
    name,
    digest: typeof modelRecord?.digest === 'string' ? modelRecord.digest : null,
    size: Number.isFinite(modelRecord?.size) ? Number(modelRecord.size) : null,
    modifiedAt: typeof modelRecord?.modifiedAt === 'string' ? modelRecord.modifiedAt : (typeof modelRecord?.modified_at === 'string' ? modelRecord.modified_at : null),
    family: typeof details.family === 'string' ? details.family : null,
    parameterSize: typeof details.parameter_size === 'string' ? details.parameter_size : null,
    quantizationLevel: typeof details.quantization_level === 'string' ? details.quantization_level : null,
  }
  return { ...identity, fingerprint: stableHash(identity) }
}

function initialFeedState(campaign) {
  return {
    version: 1,
    campaignId: campaign.id,
    curriculum: campaign.curriculum,
    seed: campaign.seed,
    protocolFingerprint: campaign.protocolFingerprint,
    nextBatchIndex: 0,
    generated: 0,
    accepted: 0,
    duplicates: 0,
    rejectedQuality: 0,
    teacherRequests: 0,
    teacherTokens: 0,
    completedTrainingCycles: 0,
    lastSnapshotId: null,
    stopReason: null,
    consecutiveEmptyBatches: 0,
    lastBatchId: null,
    lastTrainedAccepted: 0,
    nextTrainingAt: campaign.trainEveryAccepted,
    sourceIds: [],
    pendingBatchAccounting: null,
    pendingGenerationBatch: null,
    pendingTrainingMilestone: null,
    dedupInitialized: false,
    dedupWindow: [],
    promptTokens: 0,
    completionTokens: 0,
    ...(campaign.generationMode === 'hybrid' ? {
      lanes: {
        controlled: emptyLaneCounters(),
        discovery: emptyLaneCounters(),
      },
    } : {}),
    ...(campaign.continuousPlan ? {
      bootstrapSourceIds: null,
      bootstrapSourceFingerprint: null,
      continuous: {
        version: 1,
        completedBlocks: 0,
        nextBlockAt: campaign.continuousPlan.blockAccepted,
        gatePassed: false,
        gateSnapshotId: null,
        lastGate: null,
        blockHistory: [],
      },
    } : {}),
  }
}

function normalizeContinuousPlan(value, fallbackBlockAccepted) {
  if (value === undefined || value === false || value === null) return null
  if (value !== true && (!value || typeof value !== 'object' || Array.isArray(value))) {
    const error = new TypeError('continuousPlan true veya bir ayar nesnesi olmalı.')
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  const input = value === true ? {} : value
  const allowed = new Set(['blockAccepted', 'maxBlocks', 'minimumDiskHeadroomMb', 'maximumSyntheticSources'])
  const unknown = Object.keys(input).filter((key) => !allowed.has(key))
  if (unknown.length) {
    const error = new TypeError(`continuousPlan desteklenmeyen alanlar içeriyor: ${unknown.join(', ')}`)
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  const blockAccepted = integer(input.blockAccepted, 'continuousPlan.blockAccepted', 4, 10_000, fallbackBlockAccepted)
  const maxBlocks = integer(input.maxBlocks, 'continuousPlan.maxBlocks', 1, 100, 32)
  const maximumAccepted = blockAccepted * maxBlocks
  if (!Number.isSafeInteger(maximumAccepted) || maximumAccepted > 100_000) {
    const error = new TypeError('continuousPlan toplam kabul sınırı 100.000 örneği aşamaz.')
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  const minimumDiskHeadroomMb = integer(input.minimumDiskHeadroomMb, 'continuousPlan.minimumDiskHeadroomMb', 4, 16_384, 512)
  return {
    version: 1,
    blockAccepted,
    maxBlocks,
    maximumAccepted,
    minimumDiskHeadroomBytes: minimumDiskHeadroomMb * 1024 * 1024,
    maximumSyntheticSources: integer(input.maximumSyntheticSources, 'continuousPlan.maximumSyntheticSources', 16, 1_000_000, 100_000),
    stopGate: {
      required: true,
      reportOrigin: 'independent-deterministic',
      artifactKind: INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND,
      benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    },
  }
}

/**
 * Normalizes the public campaign request and freezes the exact Ollama model
 * identity selected by the HTTP layer. The returned object is suitable for a
 * job payload as-is: `{ campaign, feedState }`.
 */
export function normalizeTeacherFeedCampaignRequest(body = {}, modelRecord, id) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    const error = new TypeError('Öğretmen kampanyası isteği bir nesne olmalı.')
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  const curriculum = body.curriculum ?? body.curriculumId ?? TEACHER_FEED_CURRICULUM_ID
  if (curriculum !== TEACHER_FEED_CURRICULUM_ID) {
    const error = new Error(`Bilinmeyen öğretmen müfredatı: ${curriculum}`)
    error.code = 'UNKNOWN_TEACHER_CURRICULUM'
    throw error
  }
  const campaignId = stringValue(id ?? body.id ?? `teacher-${stableHash(body).slice(0, 20)}`, 'campaignId', { maximum: 160 })
  const requestedTargetAccepted = integer(body.targetAccepted ?? body.batchCount, 'targetAccepted', 1, 100_000, 500)
  const continuousPlan = normalizeContinuousPlan(body.continuousPlan ?? body.continuous, requestedTargetAccepted)
  const targetAccepted = continuousPlan?.maximumAccepted ?? requestedTargetAccepted
  const generationBatchSize = integer(body.generationBatchSize ?? body.batchSize, 'generationBatchSize', 1, MAX_BATCH_SIZE, 2)
  const trainingCeiling = continuousPlan?.blockAccepted ?? targetAccepted
  const trainEveryAccepted = integer(body.trainEveryAccepted, 'trainEveryAccepted', 1, trainingCeiling, Math.min(100, trainingCeiling))
  const seed = integer(body.seed, 'seed', 0, 0xffff_ffff, mixSeed(campaignId, curriculum))
  const mode = body.mode ?? 'research-only'
  if (mode !== 'research-only') {
    const error = new Error('Öğretmen kampanyası yalnız research-only kipinde çalışabilir.')
    error.code = 'INVALID_TEACHER_CAMPAIGN_MODE'
    throw error
  }
  const generationMode = body.generationMode ?? 'controlled'
  if (!TEACHER_FEED_GENERATION_MODES.includes(generationMode)) {
    const error = new Error('generationMode yalnız controlled veya hybrid olabilir.')
    error.code = 'INVALID_TEACHER_CAMPAIGN'
    throw error
  }
  const modelIdentity = normalizeModelIdentity(modelRecord, body.model)
  const campaign = {
    id: campaignId,
    model: modelIdentity.name,
    seed,
    targetAccepted,
    generationBatchSize,
    trainEveryAccepted,
    curriculum,
    mode,
    generationMode,
    maxSafetyBatches: Math.min(continuousPlan ? 100_000 : 10_000, Math.max(10, Math.ceil(targetAccepted / generationBatchSize) * 10)),
    maxGenerationCallsPerBatch: integer(body.maxGenerationCallsPerBatch, 'maxGenerationCallsPerBatch', 1, MAX_GENERATION_CALLS_PER_BATCH, 3),
    qualityThreshold: ratio(body.qualityThreshold, 'qualityThreshold', 0.55, 1, 0.78),
    nearDuplicateThreshold: ratio(body.nearDuplicateThreshold, 'nearDuplicateThreshold', 0.75, 0.99, 0.86),
    dedupWindowSize: integer(body.dedupWindowSize, 'dedupWindowSize', 128, 10_000, DEFAULT_DEDUP_WINDOW),
    maxConsecutiveEmptyBatches: integer(body.maxConsecutiveEmptyBatches, 'maxConsecutiveEmptyBatches', 1, 10, 3),
    language: 'tr',
    proposalOnly: true,
    protocolVersion: TEACHER_FEED_PROTOCOL_VERSION,
    protocolFingerprint: generationMode === 'hybrid' ? TEACHER_FEED_HYBRID_PROTOCOL_FINGERPRINT : TEACHER_FEED_PROTOCOL_FINGERPRINT,
    protocol: generationMode === 'hybrid'
      ? { ...TEACHER_FEED_PROTOCOL, hybrid: TEACHER_FEED_HYBRID_POLICY, discovery: TEACHER_DISCOVERY_PROTOCOL }
      : TEACHER_FEED_PROTOCOL,
    modelIdentity,
    ...(continuousPlan ? { continuousPlan } : {}),
  }
  return { campaign, feedState: initialFeedState(campaign) }
}

export function createTeacherFeedCampaignInput(body, modelRecord, id) {
  const payload = normalizeTeacherFeedCampaignRequest(body, modelRecord, id)
  return {
    type: TEACHER_FEED_JOB_TYPE,
    idempotencyKey: typeof body?.idempotencyKey === 'string' && body.idempotencyKey.trim() ? body.idempotencyKey.trim().slice(0, 240) : `teacher-feed:${payload.campaign.id}`,
    payload,
  }
}

function mulberry32(seed) {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296
  }
}

export function mixSeed(...parts) {
  const digest = createHash('sha256').update(parts.map(String).join('\u0000')).digest()
  return digest.readUInt32BE(0)
}

function seededShuffle(values, seed) {
  const output = [...values]
  const random = mulberry32(seed)
  for (let index = output.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1))
    ;[output[index], output[swap]] = [output[swap], output[index]]
  }
  return output
}

export function planTeacherFeedBatch({ seed, batchIndex, campaignId = '' }) {
  const epoch = Math.floor(batchIndex / TURKISH_FOUNDATIONS_V1.length)
  const order = seededShuffle(TURKISH_FOUNDATIONS_V1, mixSeed(seed, 'curriculum-epoch', epoch))
  const unit = order[batchIndex % order.length]
  // The user seed, curriculum epoch and batch position fully determine the
  // learning sequence. campaignId remains only an identity namespace.
  const batchSeed = mixSeed(seed, 'batch', batchIndex, unit.id)
  const axis = (name) => {
    const values = CURRICULUM_SURFACE_AXES[name]
    return values[mixSeed(batchSeed, name) % values.length]
  }
  return {
    batchIndex,
    batchId: `tfb-${stableHash({ campaignId, seed, batchIndex, unit: unit.id }).slice(0, 24)}`,
    batchSeed,
    unit,
    form: unit.forms[mixSeed(batchSeed, 'form') % unit.forms.length],
    axes: {
      domain: axis('domain'),
      tense: axis('tense'),
      length: axis('length'),
      wordOrder: axis('wordOrder'),
    },
  }
}

function requestedWordRange(axes) {
  const match = typeof axes?.length === 'string' ? axes.length.match(/(\d+)[–-](\d+)\s+kelime/iu) : null
  return match ? { minimum: Number(match[1]), maximum: Number(match[2]) } : { minimum: 3, maximum: 28 }
}

export function createTeacherFeedLessonSchema(_unit, count, { board } = {}) {
  if (!board?.choices?.length) {
    const error = new Error('Kapalı öğretmen seçim şeması için doğrulanmış bir seçim panosu gerekli.')
    error.code = 'TEACHER_CHOICE_BOARD_REQUIRED'
    throw error
  }
  return teacherChoiceSchema(board, count)
}

export function canonicalizeTeacherText(text) {
  return String(text ?? '')
    .normalize('NFC')
    .replace(/[\u0000-\u001f\u007f]/gu, ' ')
    .replace(/[’‘`]/gu, "'")
    .replace(/\s+/gu, ' ')
    .trim()
}

/** Materializes one model-selected integer from a server-owned immutable board. */
export function materializeTeacherLesson(choiceId, { board } = {}) {
  return materializeTeacherChoice(board, choiceId)
}

function canonicalizeForDuplicate(text) {
  return canonicalizeTeacherText(text)
    .toLocaleLowerCase('tr-TR')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/gu, ' ')
    .trim()
}

function gramsFor(text) {
  const compact = ` ${text.replace(/\s+/gu, ' ')} `
  const grams = new Set()
  if (compact.length < 4) grams.add(sha256(compact).slice(0, 12))
  else for (let index = 0; index <= compact.length - 4; index += 1) grams.add(sha256(compact.slice(index, index + 4)).slice(0, 12))
  return [...grams].sort().slice(0, 160)
}

export function createTeacherDuplicateSignature(text, choiceSignature = null) {
  const canonical = canonicalizeForDuplicate(text)
  return {
    exactHash: sha256(canonical),
    choiceHash: typeof choiceSignature === 'string' && choiceSignature ? choiceSignature : null,
    characterCount: canonical.length,
    grams: gramsFor(canonical),
  }
}

export function teacherNearDuplicateScore(left, right) {
  if (!left || !right) return 0
  if (left.exactHash === right.exactHash) return 1
  const maximumLength = Math.max(left.characterCount ?? 0, right.characterCount ?? 0, 1)
  const lengthRatio = Math.min(left.characterCount ?? 0, right.characterCount ?? 0) / maximumLength
  if (lengthRatio < 0.68) return 0
  const a = new Set(left.grams ?? [])
  const b = new Set(right.grams ?? [])
  if (!a.size || !b.size) return 0
  let intersection = 0
  for (const gram of a) if (b.has(gram)) intersection += 1
  const jaccard = intersection / (a.size + b.size - intersection)
  return jaccard * (0.75 + 0.25 * lengthRatio)
}

export function evaluateTeacherText(text, { unit, form, axes, qualityThreshold = 0.78 } = {}) {
  const normalized = canonicalizeTeacherText(text)
  const reasons = []
  if (normalized.length < 12 || normalized.length > 280) reasons.push('length')
  if (!/^[\p{Lu}\d]/u.test(normalized)) reasons.push('initial-case')
  if (!/[.!?]$/u.test(normalized)) reasons.push('terminal-punctuation')
  const terminalGroups = normalized.match(/[.!?]+/gu) ?? []
  if (terminalGroups.length !== 1) reasons.push('not-single-sentence')
  if (/https?:\/\/|www\.|\S+@\S+|```|[#*_]{2,}/iu.test(normalized)) reasons.push('markup-or-link')
  if (/\b(?:örnek cümle|yapay zek[aâ]|dil modeli|istenen çıktı|bu cümle)\b/iu.test(normalized)) reasons.push('teacher-meta-language')
  if (/(.)\1{4,}/u.test(normalized) || /[!?.,;:]{3,}/u.test(normalized)) reasons.push('repeated-symbol')
  const words = normalized.match(/[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)?/gu) ?? []
  if (words.length < 3 || words.length > 28) reasons.push('word-count')
  const foldedWords = words.map((word) => word.toLocaleLowerCase('tr-TR'))
  if (foldedWords.some((word, index) => index > 0 && word === foldedWords[index - 1])) reasons.push('adjacent-repeat')
  const lexicalDiversity = foldedWords.length ? new Set(foldedWords).size / foldedWords.length : 0
  if (lexicalDiversity < 0.58) reasons.push('low-lexical-diversity')
  if (unit?.requiredSurface && !unit.requiredSurface.test(normalized)) reasons.push('missing-surface-feature')
  const formPolicy = form ? FORM_SURFACE_POLICIES[form] : null
  if (form && !formPolicy) reasons.push('unknown-form-policy')
  else if (formPolicy && !formPolicy.requiredSurface.test(normalized)) reasons.push('missing-form-feature')
  if (form && form !== 'question' && !/\.$/u.test(normalized)) reasons.push('wrong-form-terminal')
  if (form === 'question' && !/\?$/u.test(normalized)) reasons.push('wrong-form-terminal')
  if (form === 'question' && !TURKISH_QUESTION_CUE.test(normalized)) reasons.push('missing-question-cue')
  const lengthRange = axes ? requestedWordRange(axes) : null
  if (lengthRange && (words.length < lengthRange.minimum || words.length > lengthRange.maximum)) reasons.push('surface-length')
  const spelling = checkTurkishSpelling(normalized)
  if (!spelling.valid) reasons.push('turkish-spelling')

  let score = 1
  score -= Math.min(0.2, Math.abs(normalized.length - 90) / 900)
  score -= Math.max(0, 0.8 - lexicalDiversity) * 0.35
  score -= reasons.length * 0.24
  score = Math.max(0, Math.min(1, score))
  if (score < qualityThreshold && !reasons.includes('quality-threshold')) reasons.push('quality-threshold')
  return {
    accepted: reasons.length === 0,
    text: normalized,
    score,
    reasons,
    wordCount: words.length,
    lexicalDiversity,
    spellingUnknownCount: spelling.unknownCount,
    spellingDiagnostics: spelling.diagnostics,
  }
}

export function filterTeacherLessons(lessons, {
  unit,
  form,
  axes,
  history = [],
  qualityThreshold = 0.78,
  nearDuplicateThreshold = 0.86,
} = {}) {
  const accepted = []
  const rejected = []
  const signatures = [...history]
  for (const [index, lesson] of (Array.isArray(lessons) ? lessons : []).entries()) {
    const structuralReasons = []
    if (!lesson || typeof lesson !== 'object') structuralReasons.push('invalid-lesson')
    if (Array.isArray(lesson?.surfaceReasons)) structuralReasons.push(...lesson.surfaceReasons)
    if (lesson?.skill !== unit?.id) structuralReasons.push('wrong-skill')
    if (!unit?.forms?.includes(lesson?.form) || (form && lesson?.form !== form)) structuralReasons.push('wrong-form')
    const quality = evaluateTeacherText(lesson?.text, { unit, form, axes, qualityThreshold })
    const signature = createTeacherDuplicateSignature(quality.text, lesson?.choiceSignature)
    const exact = signatures.some((item) => item.exactHash === signature.exactHash)
    const semanticDuplicate = Boolean(signature.choiceHash && signatures.some((item) => item.choiceHash === signature.choiceHash))
    const nearScore = signatures.reduce((maximum, item) => Math.max(maximum, teacherNearDuplicateScore(item, signature)), 0)
    if (exact) structuralReasons.push('exact-duplicate')
    else if (semanticDuplicate) structuralReasons.push('catalog-choice-duplicate')
    else if (nearScore >= nearDuplicateThreshold) structuralReasons.push('near-duplicate')
    const reasons = [...new Set([...structuralReasons, ...quality.reasons])]
    if (reasons.length) {
      rejected.push({
        index,
        text: quality.text,
        reasons,
        score: quality.score,
        nearDuplicateScore: nearScore,
        spellingUnknownCount: quality.spellingUnknownCount,
        spellingDiagnostics: quality.spellingDiagnostics,
        surfaceAssembly: lesson?.surfaceAssembly ?? null,
      })
      continue
    }
    const item = {
      ...lesson,
      text: quality.text,
      qualityScore: quality.score,
      wordCount: quality.wordCount,
      lexicalDiversity: quality.lexicalDiversity,
      signature,
    }
    accepted.push(item)
    signatures.push(signature)
  }
  return { accepted, rejected, signatures }
}

function rejectionSummary(rejected) {
  const counts = {}
  for (const item of rejected) for (const reason of item.reasons) counts[reason] = (counts[reason] ?? 0) + 1
  return counts
}

function buildMessages({ plan, count, attempt, board }) {
  return [
    {
      role: 'system',
      content: [
        `JSON şemasındaki choices alanında tam ${count} farklı tam sayı döndür.`,
        'Yalnız aşağıdaki doğrulanmış olay kartlarının kimliklerini seç; cümle, sözcük, açıklama veya gerekçe yazma.',
        'Aynı kimliği tekrarlama. Birlikte seçilen olayları özne ve eylem bakımından olabildiğince çeşitlendir.',
        'Türkçe yüzey sunucudaki dondurulmuş katalogdan kurulacaktır; sen yalnız kapalı seçenekleri sıralıyorsun.',
      ].join(' '),
    },
    {
      role: 'user',
      content: [
        `Beceri: ${plan.unit.title}. Biçim: ${plan.form}.`,
        `Alan: ${plan.axes.domain}. Zaman: ${plan.axes.tense}. Uzunluk: ${plan.axes.length}.`,
        `Parti=${plan.batchIndex + 1}; deneme=${attempt + 1}.`,
        'Seçim panosu:',
        teacherChoicePromptBoard(board),
      ].filter(Boolean).join('\n'),
    },
  ]
}

function teacherCompletionTokenLimit() {
  return 128
}

function createControlledAttemptMaterial({ campaign, plan, requestedCount, attemptIndex, attemptSeed }) {
  const board = createTeacherChoiceBoard(plan, {
    seed: attemptSeed,
    size: 16,
    evaluate: (text) => evaluateTeacherText(text, {
      unit: plan.unit,
      form: plan.form,
      axes: plan.axes,
      qualityThreshold: campaign.qualityThreshold,
    }),
  })
  return {
    lane: 'controlled',
    id: board.id,
    fingerprint: board.fingerprint,
    board,
    messages: buildMessages({ campaign, plan, count: requestedCount, attempt: attemptIndex, board }),
    schema: createTeacherFeedLessonSchema(plan.unit, requestedCount, { board }),
    maxCompletionTokens: teacherCompletionTokenLimit(requestedCount, plan.axes),
  }
}

function createGenerationAttemptMaterial({ campaign, plan, requestedCount, attemptIndex, attemptSeed, lane }) {
  if (lane === 'controlled') {
    return createControlledAttemptMaterial({ campaign, plan, requestedCount, attemptIndex, attemptSeed })
  }
  const discovery = createTeacherDiscoveryAttempt({
    plan,
    count: requestedCount,
    attempt: attemptIndex,
    seed: attemptSeed,
  })
  return { lane: 'discovery', ...discovery, board: null }
}

function materializeGenerationLessons(result, material, plan) {
  if (material.lane === 'controlled') {
    return (result.value?.choices ?? []).map((choiceId) => ({
      ...materializeTeacherLesson(choiceId, { board: material.board }),
      choiceId,
      skill: plan.unit.id,
      form: plan.form,
    }))
  }
  return (result.value?.sentences ?? []).map((text, index) => ({
    text,
    skill: plan.unit.id,
    form: plan.form,
    surfaceAssembly: {
      policyId: TEACHER_DISCOVERY_PROTOCOL.id,
      discoveryProtocolFingerprint: TEACHER_DISCOVERY_PROTOCOL_FINGERPRINT,
      discoveryAttemptId: material.id,
      discoveryPromptFingerprint: material.fingerprint,
      discoveryIndex: index,
      surfaceOrigin: 'ollama-authored-discovery',
      selectionOrigin: 'ollama-free-text',
      finalTextDigest: sha256(canonicalizeTeacherText(text)),
    },
  }))
}

function validateFeedState(state, campaign) {
  if (
    !state || state.version !== 1 || state.campaignId !== campaign.id ||
    state.curriculum !== campaign.curriculum || state.seed !== campaign.seed ||
    state.protocolFingerprint !== campaign.protocolFingerprint
  ) {
    const error = new Error('Öğretmen kampanyası checkpoint’i istekle uyuşmuyor.')
    error.code = 'TEACHER_FEED_CHECKPOINT_MISMATCH'
    throw error
  }
  if (campaign.continuousPlan) {
    const continuous = state.continuous
    const bootstrapSourceIds = state.bootstrapSourceIds
    const bootstrapSourceFingerprint = state.bootstrapSourceFingerprint
    const hasFrozenBootstrap = Array.isArray(bootstrapSourceIds)
    const normalizedBootstrapSourceIds = hasFrozenBootstrap
      ? [...new Set(bootstrapSourceIds.map(String))].sort()
      : null
    if (
      !continuous || continuous.version !== 1 ||
      !Number.isInteger(continuous.completedBlocks) || continuous.completedBlocks < 0 ||
      continuous.completedBlocks > campaign.continuousPlan.maxBlocks ||
      !Number.isInteger(continuous.nextBlockAt) ||
      continuous.nextBlockAt < campaign.continuousPlan.blockAccepted ||
      continuous.nextBlockAt > campaign.targetAccepted ||
      !Array.isArray(continuous.blockHistory) || continuous.blockHistory.length > campaign.continuousPlan.maxBlocks
    ) {
      const error = new Error('Sürekli öğretmen kampanyası checkpoint’i blok planıyla uyuşmuyor.')
      error.code = 'TEACHER_FEED_CHECKPOINT_MISMATCH'
      throw error
    }
    if (
      (bootstrapSourceIds !== undefined && bootstrapSourceIds !== null && !hasFrozenBootstrap) ||
      (hasFrozenBootstrap && (
        bootstrapSourceIds.some((id) => typeof id !== 'string' || !id) ||
        !exactSourceIdsMatch(bootstrapSourceIds, normalizedBootstrapSourceIds) ||
        bootstrapSourceFingerprint !== stableHash({ sourceIds: normalizedBootstrapSourceIds }) ||
        !Array.isArray(state.sourceIds) ||
        normalizedBootstrapSourceIds.some((id) => !state.sourceIds.includes(id))
      )) ||
      (!hasFrozenBootstrap && bootstrapSourceFingerprint !== undefined && bootstrapSourceFingerprint !== null)
    ) {
      const error = new Error('Sürekli öğretmen kampanyası dondurulmuş korpus sınırıyla uyuşmuyor.')
      error.code = 'TEACHER_FEED_CHECKPOINT_MISMATCH'
      throw error
    }
  } else if (state.continuous !== undefined) {
    const error = new Error('Sınırlı kampanya checkpoint’i beklenmeyen sürekli plan durumu taşıyor.')
    error.code = 'TEACHER_FEED_CHECKPOINT_MISMATCH'
    throw error
  }
  return structuredClone(state)
}

function isBudgetError(error) {
  return /BUDGET|QUOTA|USAGE_LIMIT/u.test(String(error?.code ?? ''))
}

function isOllamaTransportError(error) {
  return ['OLLAMA_TIMEOUT', 'OLLAMA_TEMPORARY_ERROR', 'OLLAMA_INVALID_RESPONSE', 'OLLAMA_INVALID_JSON'].includes(String(error?.code ?? ''))
}

function isTeacherCatalogError(error) {
  return ['NO_REALIZABLE_EVENT_CARD', 'STALE_TEACHER_CATALOG', 'TEACHER_CHOICE_BOARD_REQUIRED'].includes(String(error?.code ?? ''))
}

function reportedUsageComponent(source, tokenKey, reportedKey) {
  const nested = source?.usage
  const directHasValue = Object.prototype.hasOwnProperty.call(source ?? {}, tokenKey)
  const nestedHasValue = Object.prototype.hasOwnProperty.call(nested ?? {}, tokenKey)
  const directFlag = typeof source?.[reportedKey] === 'boolean' ? source[reportedKey] : undefined
  const nestedFlag = typeof nested?.[reportedKey] === 'boolean' ? nested[reportedKey] : undefined
  const explicitFlag = directFlag ?? nestedFlag
  const raw = directHasValue ? source[tokenKey] : nestedHasValue ? nested[tokenKey] : undefined
  const validValue = typeof raw === 'number' && Number.isSafeInteger(raw) && raw >= 0
  const reported = explicitFlag === false
    ? false
    : explicitFlag === true
      ? validValue
      : (directHasValue || nestedHasValue) && validValue
  return { reported, value: reported ? raw : 0 }
}

function conservativeStructuredUsage(source, attempt) {
  const prompt = reportedUsageComponent(source, 'promptTokens', 'promptTokensReported')
  const completion = reportedUsageComponent(source, 'completionTokens', 'completionTokensReported')
  const fallbackPromptTokens = Math.max(0, Number(attempt?.maxPromptTokens ?? 0) || 0)
  const fallbackCompletionTokens = Math.max(0, Number(attempt?.maxCompletionTokens ?? 0) || 0)
  const promptTokens = prompt.reported ? prompt.value : fallbackPromptTokens
  const completionTokens = completion.reported ? completion.value : fallbackCompletionTokens
  return {
    promptTokens,
    completionTokens,
    usageEstimated: source?.usageEstimated === true || source?.usage?.usageEstimated === true || !prompt.reported || !completion.reported,
  }
}

function markTeacherFeedUsagePersisted(error) {
  if (error && typeof error === 'object') {
    error.teacherFeedUsagePersisted = true
  }
  return error
}

function compactGenerationError(error, usageEstimated = error?.usageEstimated === true) {
  return {
    code: String(error?.code ?? 'TEACHER_GENERATION_FAILED').slice(0, 120),
    message: String(error?.message ?? 'Öğretmen üretimi başarısız oldu.').replace(/\s+/gu, ' ').slice(0, 500),
    timeoutMs: Number(error?.timeoutMs ?? 0) || null,
    usageEstimated,
  }
}

function restoreGenerationError(record) {
  const error = new Error(record?.message ?? 'Öğretmen üretimi başarısız oldu.')
  error.code = record?.code ?? 'TEACHER_GENERATION_FAILED'
  if (record?.timeoutMs) error.timeoutMs = record.timeoutMs
  if (record?.usageEstimated === true) error.usageEstimated = true
  return markTeacherFeedUsagePersisted(error)
}

function pauseError(reason, cause) {
  const error = new Error(reason)
  error.code = 'JOB_PAUSED'
  error.cause = cause
  return error
}

function requestedJobState(context) {
  const state = context.store.getJob?.(context.jobId)?.state
  return ['pause_requested', 'cancel_requested'].includes(state) ? state : null
}

function requestedStopError(state, cause) {
  const error = new Error(state === 'cancel_requested'
    ? 'Öğretmen çağrısının kullanımı kaydedildi; iş kullanıcı isteğiyle iptal ediliyor.'
    : 'Öğretmen çağrısının kullanımı kaydedildi; iş kullanıcı isteğiyle duraklatılıyor.')
  error.code = state === 'cancel_requested' ? 'JOB_CANCELLED' : 'JOB_PAUSED'
  error.cause = cause
  return error
}

function requireExecutorContext(context) {
  if (!context?.store || typeof context.store.commitTeacherFeedBatch !== 'function') {
    const error = new Error('Store, atomik commitTeacherFeedBatch(jobId, { inputs, checkpoint, artifact }) metodunu sağlamalı.')
    error.code = 'TEACHER_FEED_STORE_CONTRACT_MISSING'
    throw error
  }
  if (!context?.ollama || typeof context.ollama.generateDiverseStructured !== 'function' || typeof context.ollama.listModels !== 'function') {
    const error = new Error('Ollama adaptörü listModels ve seed destekli generateDiverseStructured metotlarını sağlamalı.')
    error.code = 'TEACHER_FEED_OLLAMA_CONTRACT_MISSING'
    throw error
  }
}

async function persistState(context, state, artifact, {
  usageReservations = [],
  usageSettlements = [],
  allowRequestedUsageFinalization = false,
} = {}) {
  return context.store.commitTeacherFeedBatch(context.jobId, {
    inputs: [],
    checkpoint: structuredClone(state),
    artifact: { kind: 'teacher-feed-checkpoint', ...artifact, contentHash: artifact.contentHash ?? stableHash(artifact.payload ?? artifact) },
    usageReservations,
    usageSettlements,
    ...(allowRequestedUsageFinalization ? { allowRequestedUsageFinalization: true } : {}),
    ...(context.jobClaim ? { claim: context.jobClaim } : {}),
  })
}

function approvedSourceIds(store, ids) {
  if (typeof store.getSource !== 'function') return [...ids]
  return ids.filter((id) => store.getSource(id)?.status === 'approved')
}

function selectedApprovedSourceIds(store, ids) {
  if (typeof store.getSource !== 'function') return [...ids]
  return ids.filter((id) => {
    const source = store.getSource(id)
    return source?.status === 'approved' && source.selected !== false
  })
}

function exactSourceIdsMatch(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
  const orderedLeft = [...left].map(String).sort()
  const orderedRight = [...right].map(String).sort()
  return orderedLeft.every((id, index) => id === orderedRight[index])
}

async function initializeCorpusFrontier(context, campaign, state) {
  if (!campaign.continuousPlan || Array.isArray(state.bootstrapSourceIds)) return state
  const campaignSourceIds = new Set([
    ...(Array.isArray(state.sourceIds) ? state.sourceIds : []),
    ...(Array.isArray(state.pendingBatchAccounting?.sourceIds) ? state.pendingBatchAccounting.sourceIds : []),
  ].map(String))
  const bootstrapSourceIds = typeof context.store.listSources === 'function'
    ? context.store.listSources()
      .filter((source) => (
        typeof source?.id === 'string' && source.id &&
        source.status === 'approved' && source.selected !== false &&
        !campaignSourceIds.has(source.id)
      ))
      .map((source) => source.id)
      .sort()
    : []
  state.bootstrapSourceIds = [...new Set(bootstrapSourceIds)]
  state.bootstrapSourceFingerprint = stableHash({ sourceIds: state.bootstrapSourceIds })
  state.sourceIds = [...new Set([...state.bootstrapSourceIds, ...(state.sourceIds ?? []).map(String)])]
  if (state.pendingTrainingMilestone) {
    state.pendingTrainingMilestone.sourceIds = [...new Set([
      ...state.bootstrapSourceIds,
      ...(state.pendingTrainingMilestone.sourceIds ?? []).map(String),
    ])]
  }
  await persistState(context, state, {
    phase: 'corpus-frontier-frozen',
    payload: {
      campaignId: campaign.id,
      bootstrapSourceIds: [...state.bootstrapSourceIds],
      bootstrapSourceCount: state.bootstrapSourceIds.length,
      bootstrapSourceFingerprint: state.bootstrapSourceFingerprint,
      campaignAccepted: state.accepted,
      proposalOnly: true,
    },
  })
  await context.emit({
    type: 'teacher-feed-corpus-frontier-frozen',
    stage: 'teacher-feed',
    data: {
      campaignId: campaign.id,
      bootstrapSourceCount: state.bootstrapSourceIds.length,
      bootstrapSourceFingerprint: state.bootstrapSourceFingerprint,
    },
  })
  return state
}

function replannedTeacherSnapshotId(campaignId, milestoneIndex, sourceIds) {
  const selectedSourceIds = [...sourceIds].map(String).sort()
  const selectionDigest = stableHash({ selectedSourceIds }).slice(0, 16)
  return `${teacherFeedCandidateSnapshotId(campaignId, milestoneIndex)}-${selectionDigest}`
}

async function accountPendingBatch(context, campaign, state) {
  const pending = state.pendingBatchAccounting
  if (!pending) return state
  const sourceIds = approvedSourceIds(context.store, pending.sourceIds)
  const duplicateCount = Math.max(0, pending.candidateCount - sourceIds.length)
  state.accepted += sourceIds.length
  state.duplicates += duplicateCount
  const lane = pending.generationLane
  if (lane && state.lanes?.[lane]) {
    state.lanes[lane].accepted += sourceIds.length
    state.lanes[lane].duplicates += duplicateCount
  }
  state.sourceIds = [...new Set([...state.sourceIds, ...sourceIds])]
  state.lastBatchId = pending.batchId
  state.consecutiveEmptyBatches = sourceIds.length ? 0 : state.consecutiveEmptyBatches + 1
  state.pendingBatchAccounting = null
  await persistState(context, state, {
    phase: 'batch-accounted',
    payload: { campaignId: campaign.id, batchId: pending.batchId, batchIndex: pending.batchIndex, generationLane: lane ?? 'controlled', inserted: sourceIds.length, duplicates: duplicateCount, campaignAccepted: state.accepted },
  })
  await context.emit({
    type: 'teacher-feed-batch-completed',
    stage: 'teacher-feed',
    data: { campaignId: campaign.id, batchId: pending.batchId, batchIndex: pending.batchIndex, generationLane: lane ?? 'controlled', inserted: sourceIds.length, duplicates: duplicateCount },
  })
  return state
}

function safeIdentifier(value) {
  return String(value).toLocaleLowerCase('en-US').replace(/[^a-z0-9_-]+/gu, '-').replace(/^-+|-+$/gu, '').slice(0, 48) || 'campaign'
}

export function teacherFeedCandidateSnapshotId(campaignId, milestoneIndex) {
  return `teacher-feed-${safeIdentifier(campaignId)}-${String(milestoneIndex).padStart(6, '0')}`
}

async function ensureTrainingMilestone(context, campaign, state, { final = false } = {}) {
  if (state.pendingTrainingMilestone) return state
  const thresholdReached = state.accepted >= state.nextTrainingAt
  const finalRemainder = final && state.accepted > state.lastTrainedAccepted
  if (!thresholdReached && !finalRemainder) return state
  const milestoneIndex = state.completedTrainingCycles + 1
  state.pendingTrainingMilestone = {
    milestoneIndex,
    acceptedAtMilestone: state.accepted,
    sourceIds: [...state.sourceIds],
    candidateSnapshotId: teacherFeedCandidateSnapshotId(campaign.id, milestoneIndex),
  }
  await persistState(context, state, {
    phase: 'training-milestone-ready',
    payload: { campaignId: campaign.id, ...state.pendingTrainingMilestone, proposalOnly: true },
  })
  return state
}

async function completePendingTrainingMilestone({ job, context, campaign, state, pipelineExecutor }) {
  const pending = state.pendingTrainingMilestone
  if (!pending) return state
  const sourceIds = selectedApprovedSourceIds(context.store, pending.sourceIds)
  let pipelineResult = null
  let existingSnapshot = context.store.getSnapshot?.(pending.candidateSnapshotId)
  if (existingSnapshot) {
    if (existingSnapshot.status === 'active') {
      const error = new Error('Research-only öğretmen snapshot’ı etkin modele dönüşmüş; kampanya güvenli biçimde durduruldu.')
      error.code = 'TEACHER_FEED_PROPOSAL_VIOLATION'
      throw error
    }
    if (!exactSourceIdsMatch(existingSnapshot.sourceIds ?? [], sourceIds)) {
      const staleEvaluation = {
        ...(existingSnapshot.evaluation ?? {}),
        passed: false,
        invalidated: true,
        code: 'SOURCE_SELECTION_CHANGED',
        selectionChange: 'DESELECTED',
        reason: 'Öğretmen kilometre taşı snapshot\'ının kaynakları güncel seçili ve onaylı kaynak kümesiyle uyuşmuyor.',
        snapshotSourceIds: [...(existingSnapshot.sourceIds ?? [])],
        selectedApprovedSourceIds: [...sourceIds],
      }
      if (['candidate', 'rejected'].includes(existingSnapshot.status) && typeof context.store.rejectSnapshot === 'function') {
        context.store.rejectSnapshot(existingSnapshot.id, staleEvaluation, context.jobClaim
          ? { jobId: context.jobId, claim: context.jobClaim }
          : undefined)
      }
      const previousCandidateSnapshotId = pending.candidateSnapshotId
      const nextCandidateSnapshotId = replannedTeacherSnapshotId(campaign.id, pending.milestoneIndex, sourceIds)
      if (nextCandidateSnapshotId === previousCandidateSnapshotId) {
        const error = new Error('Aynı deterministik snapshot kimliği farklı bir kaynak kümesine bağlanmış; kampanya güvenli biçimde durduruldu.')
        error.code = 'STALE_TEACHER_MILESTONE_SOURCES'
        error.snapshotSourceIds = staleEvaluation.snapshotSourceIds
        error.selectedApprovedSourceIds = staleEvaluation.selectedApprovedSourceIds
        throw error
      }
      pending.candidateSnapshotId = nextCandidateSnapshotId
      await persistState(context, state, {
        phase: 'training-milestone-replanned',
        payload: {
          campaignId: campaign.id,
          milestoneIndex: pending.milestoneIndex,
          previousCandidateSnapshotId,
          candidateSnapshotId: nextCandidateSnapshotId,
          selectedApprovedSourceIds: [...sourceIds],
          selectionChange: 'DESELECTED',
          proposalOnly: true,
        },
      })
      existingSnapshot = context.store.getSnapshot?.(nextCandidateSnapshotId)
      if (existingSnapshot && !exactSourceIdsMatch(existingSnapshot.sourceIds ?? [], sourceIds)) {
        if (['candidate', 'rejected'].includes(existingSnapshot.status) && typeof context.store.rejectSnapshot === 'function') {
          context.store.rejectSnapshot(existingSnapshot.id, {
            ...(existingSnapshot.evaluation ?? {}),
            passed: false,
            invalidated: true,
            code: 'SOURCE_SELECTION_CHANGED',
            selectionChange: 'DESELECTED',
            reason: 'Seçim digest\'li öğretmen snapshot kimliği farklı bir kaynak kümesiyle kaydedilmiş.',
            snapshotSourceIds: [...(existingSnapshot.sourceIds ?? [])],
            selectedApprovedSourceIds: [...sourceIds],
          }, context.jobClaim ? { jobId: context.jobId, claim: context.jobClaim } : undefined)
        }
        const error = new Error('Seçim digest\'li snapshot kimliği kaynak kümesiyle uyuşmuyor; kampanya güvenli biçimde durduruldu.')
        error.code = 'STALE_TEACHER_MILESTONE_SOURCES'
        throw error
      }
    }
    if (existingSnapshot?.status === 'active') {
      const error = new Error('Research-only öğretmen snapshot’ı etkin modele dönüşmüş; kampanya güvenli biçimde durduruldu.')
      error.code = 'TEACHER_FEED_PROPOSAL_VIOLATION'
      throw error
    }
    if (existingSnapshot && ['candidate', 'rejected'].includes(existingSnapshot.status)) {
      pipelineResult = {
        snapshotId: existingSnapshot.id,
        committed: false,
        awaitingApproval: existingSnapshot.status === 'candidate',
        evaluation: existingSnapshot.evaluation ?? null,
        recovered: true,
      }
    } else if (existingSnapshot) {
      const error = new Error('Önbellekteki öğretmen snapshot\'ı yeniden kullanılamayan bir durumda.')
      error.code = 'TEACHER_FEED_SNAPSHOT_STATE_CONFLICT'
      throw error
    }
  }
  if (!pipelineResult && sourceIds.length) {
    await context.checkpoint(`teacher-feed:training:${pending.milestoneIndex}`)
    const baseProgress = Math.min(0.99, state.accepted / campaign.targetAccepted)
    const pipelineContext = {
      ...context,
      progress: async (stage, progress) => context.progress(`teacher-feed:pipeline:${stage}`, Math.min(0.999, baseProgress + progress * 0.001)),
    }
    if (typeof context.ollama?.unload === 'function') {
      let unloadEvent
      try {
        const release = await context.ollama.unload(campaign.modelIdentity?.name ?? campaign.model, { signal: context.signal })
        unloadEvent = {
          type: 'teacher-feed-model-unloaded',
          stage: 'teacher-feed',
          data: {
            campaignId: campaign.id,
            milestoneIndex: pending.milestoneIndex,
            model: release?.model ?? campaign.modelIdentity?.name ?? campaign.model,
            unloaded: release?.unloaded !== false,
          },
        }
      } catch (error) {
        context.signal?.throwIfAborted?.()
        unloadEvent = {
          type: 'teacher-feed-model-unload-warning',
          level: 'warning',
          stage: 'teacher-feed',
          data: {
            campaignId: campaign.id,
            milestoneIndex: pending.milestoneIndex,
            model: campaign.modelIdentity?.name ?? campaign.model,
            code: String(error?.code ?? 'OLLAMA_UNLOAD_FAILED').slice(0, 120),
            message: String(error?.message ?? 'Ollama modeli bellekten çıkarılamadı.').slice(0, 500),
          },
        }
      }
      await context.emit(unloadEvent)
      // Unload can take long enough for pause/cancel to linearize. Re-check the
      // durable claim before entering the memory-heavy deterministic pipeline.
      await context.checkpoint(`teacher-feed:training:${pending.milestoneIndex}:model-unloaded`)
    }
    pipelineResult = await pipelineExecutor({
      id: `${job.id ?? context.jobId}:proposal:${pending.milestoneIndex}`,
      type: 'full-cycle',
      payload: {
        sourceIds,
        resolvedSourceIds: sourceIds,
        reprocess: true,
        automationLevel: 'proposal',
        candidateSnapshotId: pending.candidateSnapshotId,
        label: `Öğretmen kampanyası ${campaign.id} / kilometre taşı ${pending.milestoneIndex}`,
        teacherCampaignId: campaign.id,
        teacherTrainingMilestone: pending.milestoneIndex,
      },
    }, pipelineContext)
    if (pipelineResult?.committed === true) {
      const error = new Error('Öğretmen kampanyası proposal-only sınırını ihlal eden bir pipeline sonucu aldı.')
      error.code = 'TEACHER_FEED_PROPOSAL_VIOLATION'
      throw error
    }
  }

  state.completedTrainingCycles += sourceIds.length ? 1 : 0
  state.lastSnapshotId = pipelineResult?.snapshotId ?? state.lastSnapshotId
  state.lastTrainedAccepted = pending.acceptedAtMilestone
  state.nextTrainingAt = (Math.floor(state.accepted / campaign.trainEveryAccepted) + 1) * campaign.trainEveryAccepted
  state.pendingTrainingMilestone = null
  await persistState(context, state, {
    phase: 'training-milestone-completed',
    payload: {
      campaignId: campaign.id,
      milestoneIndex: pending.milestoneIndex,
      sourceCount: sourceIds.length,
      proposalOnly: true,
      snapshotId: pipelineResult?.snapshotId ?? null,
      awaitingApproval: pipelineResult?.awaitingApproval === true,
      evaluationPassed: pipelineResult?.evaluation?.passed === true,
    },
  })
  await context.emit({
    type: 'teacher-feed-training-cycle-completed',
    stage: 'teacher-feed',
    data: { campaignId: campaign.id, milestoneIndex: pending.milestoneIndex, accepted: state.accepted, proposalSnapshotId: pipelineResult?.snapshotId ?? null },
  })
  return state
}

function compactStructuralGate(result, blockIndex, snapshotId) {
  return {
    blockIndex,
    snapshotId: snapshotId ?? null,
    status: result?.status ?? 'missing',
    passed: result?.passed === true,
    valid: result?.valid === true,
    conclusive: result?.conclusive === true,
    completion: typeof result?.completion === 'string' ? result.completion : null,
    evaluationKind: typeof result?.evaluationKind === 'string' ? result.evaluationKind : null,
    decisiveCheckId: typeof result?.decisiveCheckId === 'string' ? result.decisiveCheckId : null,
    artifactId: result?.artifactId ?? null,
    benchmarkId: result?.benchmarkId ?? null,
    benchmarkFingerprint: result?.benchmarkFingerprint ?? INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    strongestBaseline: Number.isFinite(result?.strongestBaseline) ? result.strongestBaseline : null,
    baselineAdvantage: Number.isFinite(result?.baselineAdvantage) ? result.baselineAdvantage : null,
    diagnosticMeanGraphF1: Number.isFinite(result?.diagnosticMeanGraphF1) ? result.diagnosticMeanGraphF1 : null,
    holdoutGraphF1: Number.isFinite(result?.holdoutGraphF1) ? result.holdoutGraphF1 : null,
    holdoutGraphF1Low: Number.isFinite(result?.holdoutGraphF1Low) ? result.holdoutGraphF1Low : null,
    holdoutGraphF1High: Number.isFinite(result?.holdoutGraphF1High) ? result.holdoutGraphF1High : null,
    checks: Array.isArray(result?.checks)
      ? result.checks.slice(0, 40).map((check) => ({
          id: String(check?.id ?? '').slice(0, 120),
          passed: check?.passed === true,
          actual: ['string', 'number', 'boolean'].includes(typeof check?.actual) ? check.actual : null,
          expected: ['string', 'number', 'boolean'].includes(typeof check?.expected) ? check.expected : null,
        }))
      : [],
  }
}

async function evaluateContinuousBlock({ context, campaign, state, structuralBenchmarkReport }) {
  const plan = campaign.continuousPlan
  if (!plan || state.accepted < state.continuous.nextBlockAt) return { state, gatePassed: false }
  const blockIndex = state.continuous.completedBlocks + 1
  const snapshotId = state.lastSnapshotId
  const snapshot = snapshotId ? context.store.getSnapshot?.(snapshotId) : null
  let rawEvidence = null
  try {
    rawEvidence = await structuralBenchmarkReport({
      store: context.store,
      campaign: structuredClone(campaign),
      feedState: structuredClone(state),
      blockIndex,
      snapshot: snapshot ? structuredClone(snapshot) : null,
      snapshotId,
      signal: context.signal,
      jobId: context.jobId,
      jobClaim: context.jobClaim ? structuredClone(context.jobClaim) : null,
      checkpoint: context.checkpoint,
      progress: context.progress,
      emit: context.emit,
    })
  } catch (error) {
    const errorCode = String(error?.code ?? '')
    if (
      context.signal?.aborted ||
      STRUCTURAL_BENCHMARK_PROPAGATED_SAFETY_CODES.has(errorCode) ||
      errorCode.startsWith('JOB_WORKER_')
    ) {
      throw error
    }
    // A benchmark runner failure cannot be interpreted as model success. Keep
    // the block durable and continue until another safety boundary is reached.
    rawEvidence = {
      error: {
        code: String(error?.code ?? 'STRUCTURAL_BENCHMARK_FAILED').slice(0, 120),
        message: String(error?.message ?? 'Yapısal benchmark çalıştırılamadı.').slice(0, 500),
      },
    }
  }
  const reportEnvelope = rawEvidence?.report ? rawEvidence : { report: rawEvidence }
  const evaluated = reportEnvelope.report
    ? {
        ...(reportEnvelope.artifactId ? { artifactId: reportEnvelope.artifactId } : {}),
        ...evaluateIndependentStructuralEvidence(reportEnvelope.report, {
          expectedSnapshotId: snapshotId,
          expectedInputManifestHash: snapshot?.inputManifestHash ?? null,
        }),
      }
    : {
        status: 'missing',
        passed: false,
        benchmarkId: null,
        benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
        checks: [{ id: 'benchmark-artifact', passed: false, actual: null, expected: INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND }],
      }
  const gate = compactStructuralGate(evaluated, blockIndex, snapshotId)
  state.continuous.completedBlocks = blockIndex
  state.continuous.gatePassed = gate.passed
  state.continuous.gateSnapshotId = gate.passed ? snapshotId : null
  state.continuous.lastGate = gate
  state.continuous.nextBlockAt = Math.min(
    campaign.targetAccepted,
    (blockIndex + 1) * plan.blockAccepted,
  )
  state.continuous.blockHistory = [...state.continuous.blockHistory, {
    blockIndex,
    accepted: state.accepted,
    snapshotId: snapshotId ?? null,
    gateStatus: gate.status,
    gatePassed: gate.passed,
    evaluationKind: gate.evaluationKind,
    conclusive: gate.conclusive,
  }].slice(-plan.maxBlocks)
  await persistState(context, state, {
    kind: 'teacher-feed-continuous-block',
    phase: 'continuous-block-evaluated',
    payload: {
      campaignId: campaign.id,
      blockIndex,
      accepted: state.accepted,
      maximumBlocks: plan.maxBlocks,
      nextBlockAt: state.continuous.nextBlockAt,
      proposalOnly: true,
      evaluationSourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
      structuralGate: gate,
    },
  })
  await context.emit({
    type: gate.passed ? 'teacher-feed-structural-gate-passed' : 'teacher-feed-structural-gate-not-passed',
    stage: 'teacher-feed',
    data: { campaignId: campaign.id, blockIndex, snapshotId, status: gate.status },
  })
  return { state, gatePassed: gate.passed }
}

function teacherFeedResult(campaign, state) {
  return {
    campaignId: campaign.id,
    curriculum: campaign.curriculum,
    seed: campaign.seed,
    modelIdentity: campaign.modelIdentity,
    protocolVersion: campaign.protocolVersion,
    protocolFingerprint: campaign.protocolFingerprint,
    proposalOnly: true,
    generationMode: campaign.generationMode ?? 'controlled',
    ...(state.lanes ? { lanes: structuredClone(state.lanes) } : {}),
    generated: state.generated,
    accepted: state.accepted,
    duplicates: state.duplicates,
    rejectedQuality: state.rejectedQuality,
    teacherRequests: state.teacherRequests,
    teacherTokens: state.teacherTokens,
    completedTrainingCycles: state.completedTrainingCycles,
    lastSnapshotId: state.lastSnapshotId,
    ...(campaign.continuousPlan ? {
      continuous: structuredClone(state.continuous),
      stopGatePassed: state.continuous.gatePassed === true,
    } : {}),
    feedState: state,
  }
}

async function completeContinuousBoundary({
  job,
  context,
  campaign,
  state,
  pipelineExecutor,
  structuralBenchmarkReport,
}) {
  if (!campaign.continuousPlan || state.accepted < state.continuous.nextBlockAt) {
    return { state, gatePassed: false }
  }
  if (state.continuous.completedBlocks >= campaign.continuousPlan.maxBlocks) {
    await pauseForSafety(
      context,
      state,
      'CONTINUOUS_MAX_BLOCKS_REACHED',
      'Planlanan güvenli eğitim blokları tamamlandı; bağımsız yapısal benchmark henüz bütün kapıları geçmediği için kampanya otomatik başarı sayılmadan duraklatıldı.',
      {
        completedBlocks: state.continuous.completedBlocks,
        maximumBlocks: campaign.continuousPlan.maxBlocks,
        lastGateStatus: state.continuous.lastGate?.status ?? 'missing',
      },
    )
  }
  state = await ensureTrainingMilestone(context, campaign, state, { final: true })
  state = await completePendingTrainingMilestone({ job, context, campaign, state, pipelineExecutor })
  const gate = await evaluateContinuousBlock({ context, campaign, state, structuralBenchmarkReport })
  state = gate.state
  if (gate.gatePassed) return { state, gatePassed: true }
  if (state.continuous.completedBlocks >= campaign.continuousPlan.maxBlocks) {
    await pauseForSafety(
      context,
      state,
      'CONTINUOUS_MAX_BLOCKS_REACHED',
      'Planlanan güvenli eğitim blokları tamamlandı; bağımsız yapısal benchmark henüz bütün kapıları geçmediği için kampanya otomatik başarı sayılmadan duraklatıldı.',
      {
        completedBlocks: state.continuous.completedBlocks,
        maximumBlocks: campaign.continuousPlan.maxBlocks,
        lastGateStatus: state.continuous.lastGate?.status ?? 'missing',
      },
    )
  }
  await assertContinuousBlockAdmission(context, campaign, state, { initial: false })
  return { state, gatePassed: false }
}

async function assertContinuousBlockAdmission(context, campaign, state, { initial }) {
  const resources = evaluateContinuousResourceBudget({
    store: context.store,
    config: context.config ?? context.store.getConfig(),
    plan: campaign.continuousPlan,
  })
  if (!resources.passed) {
    await pauseForSafety(
      context,
      state,
      resources.code,
      resources.code === 'CONTINUOUS_DISK_HEADROOM_LOW'
        ? `${initial ? 'İlk' : 'Bir sonraki'} güvenli eğitim bloğu için ayrılan disk payı ${initial ? 'bulunmadığından kampanya başlatılmadı' : 'kalmadığından kampanya duraklatıldı'}.`
        : `${initial ? 'Sentetik kaynak güvenlik sınırı dolu olduğundan kampanya başlatılmadı' : 'Sentetik kaynak güvenlik sınırına ulaşıldığından kampanya duraklatıldı'}.`,
      {
        remainingDiskBytes: resources.remainingDiskBytes,
        requiredDiskHeadroomBytes: campaign.continuousPlan.minimumDiskHeadroomBytes,
        syntheticSourceCount: resources.syntheticSourceCount,
        projectedSyntheticSourceCount: resources.projectedSyntheticSourceCount,
        maximumSyntheticSources: campaign.continuousPlan.maximumSyntheticSources,
      },
    )
  }
  const physical = typeof context.resourceGovernor?.inspect === 'function'
    ? context.resourceGovernor.inspect({ admission: true })
    : null
  if (physical?.state?.canStart === false) {
    await pauseForSafety(
      context,
      state,
      'CONTINUOUS_PHYSICAL_RESOURCE_RESERVE_LOW',
      `${initial ? 'İlk' : 'Bir sonraki'} güvenli eğitim bloğu fiziksel RAM/SSD rezervi yeniden yeterli olana kadar ${initial ? 'başlatılmadı' : 'duraklatıldı'}.`,
      {
        resourceState: structuredClone(physical.state),
        resourceSample: physical.sample ? structuredClone(physical.sample) : null,
      },
    )
  }
}

function captureGenerationAccounting(pending, state) {
  return {
    pendingPromptTokens: pending.promptTokens,
    pendingCompletionTokens: pending.completionTokens,
    pendingGenerationErrors: pending.generationErrors,
    pendingUsageEstimated: pending.usageEstimated,
    pendingLastError: structuredClone(pending.lastError),
    pendingInFlightAttempt: structuredClone(pending.inFlightAttempt),
    pendingAcceptedLength: pending.accepted.length,
    pendingRejectedLength: pending.rejected.length,
    statePromptTokens: state.promptTokens,
    stateCompletionTokens: state.completionTokens,
    stateTeacherTokens: state.teacherTokens,
  }
}

function restoreGenerationAccounting(pending, state, captured) {
  pending.promptTokens = captured.pendingPromptTokens
  pending.completionTokens = captured.pendingCompletionTokens
  pending.generationErrors = captured.pendingGenerationErrors
  pending.usageEstimated = captured.pendingUsageEstimated
  pending.lastError = structuredClone(captured.pendingLastError)
  pending.inFlightAttempt = structuredClone(captured.pendingInFlightAttempt)
  pending.accepted.length = captured.pendingAcceptedLength
  pending.rejected.length = captured.pendingRejectedLength
  state.promptTokens = captured.statePromptTokens
  state.completionTokens = captured.stateCompletionTokens
  state.teacherTokens = captured.stateTeacherTokens
}

function teacherTokenReservation(amount, day) {
  return amount > 0 ? [{ kind: 'teacherTokens', amount, ...(day ? { day } : {}) }] : []
}

function teacherTokenFinalization(attempt, actualAmount) {
  const reservedAmount = Math.max(0, Number(attempt?.maxPromptTokens ?? 0) || 0) +
    Math.max(0, Number(attempt?.maxCompletionTokens ?? 0) || 0)
  if (attempt?.tokenUsageReserved === true) {
    return {
      usageSettlements: [{
        kind: 'teacherTokens',
        reservedAmount,
        actualAmount,
        ...(attempt.usageDay ? { day: attempt.usageDay } : {}),
      }],
    }
  }
  return { usageReservations: teacherTokenReservation(actualAmount, attempt?.usageDay) }
}

async function generateBatch({ campaign, plan, state, context, desiredCount, lane = 'controlled' }) {
  let pending = state.pendingGenerationBatch
  if (pending && (
    pending.version !== 1 || pending.batchId !== plan.batchId || pending.batchIndex !== plan.batchIndex ||
    pending.desiredCount !== desiredCount || (pending.lane ?? 'controlled') !== lane
  )) {
    const error = new Error('Kalıcı kısmi üretim partisi mevcut müfredat planıyla uyuşmuyor.')
    error.code = 'TEACHER_FEED_GENERATION_CHECKPOINT_MISMATCH'
    throw error
  }
  if (!pending) {
    pending = {
      version: 1,
      batchId: plan.batchId,
      batchIndex: plan.batchIndex,
      desiredCount,
      lane,
      accepted: [],
      rejected: [],
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      generationErrors: 0,
      usageEstimated: false,
      lastError: null,
      inFlightAttempt: null,
    }
    state.pendingGenerationBatch = pending
    await persistState(context, state, {
      phase: 'generation-batch-started',
      payload: { campaignId: campaign.id, batchId: plan.batchId, batchIndex: plan.batchIndex, desiredCount, lane },
    })
  }

  const persistAttempt = async (phase, extra = {}, options = {}) => persistState(context, state, {
    phase,
    payload: {
      campaignId: campaign.id,
      batchId: plan.batchId,
      batchIndex: plan.batchIndex,
      calls: pending.calls,
      accepted: pending.accepted.length,
      rejected: pending.rejected.length,
      promptTokens: pending.promptTokens,
      completionTokens: pending.completionTokens,
      ...extra,
    },
  }, options)

  // A persisted reserved attempt may have reached Ollama before a process
  // stopped, so replaying it would risk a duplicate expensive call. Account a
  // conservative bounded output and continue with the next deterministic seed.
  if (pending.inFlightAttempt?.requestUsageReserved === true) {
    const captured = captureGenerationAccounting(pending, state)
    const interrupted = pending.inFlightAttempt
    const estimatedPromptTokens = teacherAttemptPromptTokenUpperBound(interrupted)
    const estimatedCompletionTokens = interrupted.maxCompletionTokens
    const estimatedTokens = estimatedPromptTokens + estimatedCompletionTokens
    pending.promptTokens += estimatedPromptTokens
    pending.completionTokens += estimatedCompletionTokens
    pending.generationErrors += 1
    pending.usageEstimated = true
    pending.lastError = compactGenerationError(Object.assign(new Error('Önceki 14B çağrısının sonucu servis kesintisi nedeniyle belirsiz kaldı.'), {
      code: 'OLLAMA_TEMPORARY_ERROR',
      usageEstimated: true,
    }))
    pending.inFlightAttempt = null
    state.promptTokens += estimatedPromptTokens
    state.completionTokens += estimatedCompletionTokens
    state.teacherTokens += estimatedTokens
    try {
      await persistAttempt('generation-interrupted-attempt-recovered', {
        usageEstimated: true,
      }, {
        ...teacherTokenFinalization(interrupted, estimatedTokens),
        allowRequestedUsageFinalization: true,
      })
    } catch (error) {
      // The persisted checkpoint still contains the reserved in-flight call.
      // Restore that exact state so a later resume charges it conservatively
      // without ever replaying the same expensive model request.
      restoreGenerationAccounting(pending, state, captured)
      throw markTeacherFeedUsagePersisted(error)
    }
  }

  let history = [...state.dedupWindow, ...pending.accepted.map((item) => item.signature)]
  while (pending.calls < campaign.maxGenerationCallsPerBatch && pending.accepted.length < desiredCount) {
    let attempt = pending.inFlightAttempt
    if (!attempt) {
      await context.checkpoint(`teacher-feed:generate:${plan.batchIndex}:${pending.calls}`)
      const attemptIndex = pending.calls
      const requestedCount = Math.max(1, desiredCount - pending.accepted.length)
      const attemptSeed = mixSeed(plan.batchSeed, 'attempt', attemptIndex)
      // Kontrollü pano ve yeni keşif promptu aynı dayanıklı rezervasyon
      // mekanizmasını kullanır; hangi hat olduğu checkpoint'te ayrıca yazılır.
      const material = createGenerationAttemptMaterial({ campaign, plan, requestedCount, attemptIndex, attemptSeed, lane })
      attempt = {
        attemptIndex,
        attemptSeed,
        requestedCount,
        lane,
        maxCompletionTokens: material.maxCompletionTokens,
        maxPromptTokens: teacherPromptTokenUpperBound(material.messages, material.schema, {
          contextTokenLimit: TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
        }),
        contextTokenLimit: TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
        promptBoundPolicy: 'ollama-num-ctx-v1',
        usageDay: new Date().toISOString().slice(0, 10),
        requestUsageReserved: false,
        tokenUsageReserved: false,
        generationSurfaceId: material.id,
        generationSurfaceFingerprint: material.fingerprint,
        ...(material.board ? { boardId: material.board.id, boardFingerprint: material.board.fingerprint } : {}),
      }
      pending.inFlightAttempt = attempt
      await persistAttempt(lane === 'controlled' ? 'generation-choice-board-ready' : 'generation-discovery-prompt-ready', {
        attemptIndex,
        requestedCount,
        lane,
        requestUsageReserved: false,
        generationSurfaceId: material.id,
        generationSurfaceFingerprint: material.fingerprint,
        ...(material.board ? { boardId: material.board.id, boardFingerprint: material.board.fingerprint, boardSize: material.board.choices.length } : {}),
      })
    }

    if (!attempt.requestUsageReserved || !attempt.tokenUsageReserved) {
      const material = createGenerationAttemptMaterial({
        campaign,
        plan,
        requestedCount: attempt.requestedCount,
        attemptIndex: attempt.attemptIndex,
        attemptSeed: attempt.attemptSeed,
        lane,
      })
      if (attempt.generationSurfaceFingerprint && attempt.generationSurfaceFingerprint !== material.fingerprint) {
        const error = new Error('Aynı seed için üretilen öğretmen yüzeyinin fingerprint’i değişti.')
        error.code = 'STALE_TEACHER_CATALOG'
        throw error
      }
      attempt.generationSurfaceId = material.id
      attempt.generationSurfaceFingerprint = material.fingerprint
      if (material.board) {
        attempt.boardId = material.board.id
        attempt.boardFingerprint = material.board.fingerprint
      }
      const reserveRequest = attempt.requestUsageReserved !== true
      const reserveTokens = attempt.tokenUsageReserved !== true
      if (reserveRequest) {
        pending.calls += 1
        state.teacherRequests += 1
        attempt.requestUsageReserved = true
      }
      if (reserveTokens) attempt.tokenUsageReserved = true
      const maximumTeacherTokens = Math.max(0, Number(attempt.maxPromptTokens ?? 0) || 0) +
        Math.max(0, Number(attempt.maxCompletionTokens ?? 0) || 0)
      try {
        await persistAttempt('generation-attempt-reserved', {
          attemptIndex: attempt.attemptIndex,
          requestUsageReserved: attempt.requestUsageReserved,
          tokenUsageReserved: attempt.tokenUsageReserved,
        }, {
          usageReservations: [
            ...(reserveRequest ? [{ kind: 'teacherRequests', amount: 1, day: attempt.usageDay }] : []),
            ...(reserveTokens ? teacherTokenReservation(maximumTeacherTokens, attempt.usageDay) : []),
          ],
        })
      } catch (error) {
        if (reserveRequest) {
          attempt.requestUsageReserved = false
          pending.calls -= 1
          state.teacherRequests -= 1
        }
        if (reserveTokens) attempt.tokenUsageReserved = false
        await persistAttempt('generation-attempt-reservation-denied', { attemptIndex: attempt.attemptIndex, errorCode: String(error?.code ?? 'BUDGET') })
        throw markTeacherFeedUsagePersisted(error)
      }
    }

    const material = createGenerationAttemptMaterial({
      campaign,
      plan,
      requestedCount: attempt.requestedCount,
      attemptIndex: attempt.attemptIndex,
      attemptSeed: attempt.attemptSeed,
      lane,
    })
    const persistedSurfaceFingerprint = attempt.generationSurfaceFingerprint ?? attempt.boardFingerprint
    const persistedSurfaceId = attempt.generationSurfaceId ?? attempt.boardId
    if (persistedSurfaceFingerprint !== material.fingerprint || persistedSurfaceId !== material.id) {
      const error = new Error('Öğretmen üretim yüzeyi istek rezervasyonundan sonra değişti.')
      error.code = 'STALE_TEACHER_CATALOG'
      throw error
    }
    const { schema, messages } = material
    if (
      Number.isInteger(attempt.maxPromptTokens) &&
      attempt.maxPromptTokens !== teacherPromptTokenUpperBound(messages, schema, {
        contextTokenLimit: TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
      })
    ) {
      const error = new Error('Öğretmen prompt üst sınırı istek rezervasyonundan sonra değişti.')
      error.code = 'STALE_TEACHER_CATALOG'
      throw error
    }
    if (attempt.contextTokenLimit !== TEACHER_FEED_CONTEXT_TOKEN_LIMIT || attempt.promptBoundPolicy !== 'ollama-num-ctx-v1') {
      const error = new Error('Öğretmen prompt sınırı rezervasyondan sonra değişti.')
      error.code = 'STALE_TEACHER_PROTOCOL'
      throw error
    }
    let result
    try {
      result = await context.ollama.generateDiverseStructured({
        model: campaign.modelIdentity.name,
        messages,
        schema,
        signal: context.signal,
        seed: attempt.attemptSeed,
        maxCompletionTokens: attempt.maxCompletionTokens,
      })
    } catch (error) {
      const failedUsage = conservativeStructuredUsage(error, attempt)
      const captured = captureGenerationAccounting(pending, state)
      pending.promptTokens += failedUsage.promptTokens
      pending.completionTokens += failedUsage.completionTokens
      pending.generationErrors += 1
      pending.usageEstimated ||= failedUsage.usageEstimated
      pending.lastError = compactGenerationError(error, failedUsage.usageEstimated)
      pending.inFlightAttempt = null
      state.promptTokens += failedUsage.promptTokens
      state.completionTokens += failedUsage.completionTokens
      const failedTokens = failedUsage.promptTokens + failedUsage.completionTokens
      state.teacherTokens += failedTokens
      try {
        await persistAttempt('generation-attempts-failed', {
          attemptIndex: attempt.attemptIndex,
          errorCode: String(error?.code ?? 'TEACHER_GENERATION_FAILED'),
          usageEstimated: failedUsage.usageEstimated,
        }, {
          ...teacherTokenFinalization(attempt, failedTokens),
          allowRequestedUsageFinalization: true,
        })
      } catch (budgetError) {
        restoreGenerationAccounting(pending, state, captured)
        throw markTeacherFeedUsagePersisted(budgetError)
      }
      const stopState = requestedJobState(context)
      if (stopState) throw markTeacherFeedUsagePersisted(requestedStopError(stopState, error))
      if (isBudgetError(error) || error?.code === 'OLLAMA_TIMEOUT') throw markTeacherFeedUsagePersisted(error)
      continue
    }

    const successfulUsage = conservativeStructuredUsage(result, attempt)
    const successfulPromptTokens = successfulUsage.promptTokens
    const successfulCompletionTokens = successfulUsage.completionTokens
    const successfulTokens = successfulPromptTokens + successfulCompletionTokens
    const resolvedModel = typeof result.model === 'string' ? result.model : ''
    if (resolvedModel !== campaign.modelIdentity.name) {
      const modelError = new Error('Ollama yanıtı dondurulmuş kampanya modelinden farklı bir model kimliği bildirdi.')
      modelError.code = 'STALE_TEACHER_MODEL_RESPONSE'
      modelError.expectedModel = campaign.modelIdentity.name
      modelError.actualModel = resolvedModel || null
      const captured = captureGenerationAccounting(pending, state)
      pending.promptTokens += successfulPromptTokens
      pending.completionTokens += successfulCompletionTokens
      pending.generationErrors += 1
      pending.usageEstimated ||= successfulUsage.usageEstimated
      pending.lastError = compactGenerationError(modelError)
      pending.inFlightAttempt = null
      state.promptTokens += successfulPromptTokens
      state.completionTokens += successfulCompletionTokens
      state.teacherTokens += successfulTokens
      try {
        await persistAttempt('generation-attempt-model-mismatch', {
          attemptIndex: attempt.attemptIndex,
          errorCode: modelError.code,
          expectedModel: modelError.expectedModel,
          actualModel: modelError.actualModel,
        }, {
          ...teacherTokenFinalization(attempt, successfulTokens),
          allowRequestedUsageFinalization: true,
        })
      } catch (error) {
        restoreGenerationAccounting(pending, state, captured)
        throw markTeacherFeedUsagePersisted(error)
      }
      throw markTeacherFeedUsagePersisted(modelError)
    }
    const serverLabeledLessons = materializeGenerationLessons(result, material, plan)
    const filtered = filterTeacherLessons(serverLabeledLessons, {
      unit: plan.unit,
      form: plan.form,
      axes: plan.axes,
      history,
      qualityThreshold: campaign.qualityThreshold,
      nearDuplicateThreshold: campaign.nearDuplicateThreshold,
    })
    const captured = captureGenerationAccounting(pending, state)
    const previousHistory = history
    pending.promptTokens += successfulPromptTokens
    pending.completionTokens += successfulCompletionTokens
    pending.usageEstimated ||= successfulUsage.usageEstimated
    state.promptTokens += successfulPromptTokens
    state.completionTokens += successfulCompletionTokens
    state.teacherTokens += successfulTokens
    for (const lesson of filtered.accepted) pending.accepted.push({ ...lesson, attemptSeed: attempt.attemptSeed, resolvedModel })
    pending.rejected.push(...filtered.rejected)
    pending.inFlightAttempt = null
    history = filtered.signatures
    try {
      await persistAttempt('generation-attempt-completed', {
        attemptIndex: attempt.attemptIndex,
      }, {
        ...teacherTokenFinalization(attempt, successfulTokens),
      })
    } catch (error) {
      restoreGenerationAccounting(pending, state, captured)
      history = previousHistory
      const stopState = requestedJobState(context)
      if (!stopState) throw markTeacherFeedUsagePersisted(error)

      // The model call already consumed tokens, but pause/cancel must prevent
      // its newly generated lessons from entering the training data plane.
      // Persist only accounting + no-replay state under the still-owned claim.
      const stopError = requestedStopError(stopState, error)
      pending.promptTokens += successfulPromptTokens
      pending.completionTokens += successfulCompletionTokens
      pending.generationErrors += 1
      pending.usageEstimated ||= successfulUsage.usageEstimated
      pending.lastError = compactGenerationError(stopError)
      pending.inFlightAttempt = null
      state.promptTokens += successfulPromptTokens
      state.completionTokens += successfulCompletionTokens
      state.teacherTokens += successfulTokens
      try {
        await persistAttempt('generation-attempt-discarded-after-stop', {
          attemptIndex: attempt.attemptIndex,
          stopState,
          discardedChoices: serverLabeledLessons.length,
          lane,
        }, {
          ...teacherTokenFinalization(attempt, successfulTokens),
          allowRequestedUsageFinalization: true,
        })
      } catch (accountingError) {
        restoreGenerationAccounting(pending, state, captured)
        throw markTeacherFeedUsagePersisted(accountingError)
      }
      throw markTeacherFeedUsagePersisted(stopError)
    }
  }
  if (!pending.accepted.length && pending.generationErrors >= campaign.maxGenerationCallsPerBatch && pending.lastError) {
    throw restoreGenerationError(pending.lastError)
  }
  return {
    accepted: pending.accepted.slice(0, desiredCount),
    rejected: pending.rejected,
    history,
    calls: pending.calls,
    promptTokens: pending.promptTokens,
    completionTokens: pending.completionTokens,
    usageEstimated: pending.usageEstimated,
    lane,
  }
}

export function sourceFromLesson({ lesson, campaign, plan, config }) {
  const modelFingerprint = campaign.modelIdentity.fingerprint
  const independenceGroupId = canonicalOllamaIndependenceGroupId({
    resolvedModelName: lesson.resolvedModel ?? campaign.modelIdentity.name,
    weightsDigest: campaign.modelIdentity.digest,
  })
  const assembly = lesson.surfaceAssembly ?? {}
  const boundedSurfaceTrace = {
    policyId: assembly.policyId ?? null,
    catalogFingerprint: assembly.catalogFingerprint ?? null,
    boardFingerprint: assembly.boardFingerprint ?? null,
    choiceId: assembly.choiceId ?? null,
    resolvedSelection: assembly.resolvedSelection ?? null,
    semanticCardDigest: assembly.semanticCardDigest ?? null,
    semanticIdentityDigest: assembly.semanticIdentityDigest ?? null,
    renderPlan: {
      adjunctIds: assembly.renderPlan?.adjunctIds ?? [],
      connector: assembly.renderPlan?.connector ?? null,
      terminal: assembly.renderPlan?.terminal ?? null,
    },
    finalTextDigest: assembly.finalTextDigest ?? null,
    surfaceOrigin: assembly.surfaceOrigin ?? null,
    selectionOrigin: assembly.selectionOrigin ?? null,
    ...(assembly.surfaceOrigin === 'ollama-authored-discovery' ? {
      discoveryProtocolFingerprint: assembly.discoveryProtocolFingerprint ?? null,
      discoveryAttemptId: assembly.discoveryAttemptId ?? null,
      discoveryPromptFingerprint: assembly.discoveryPromptFingerprint ?? null,
      discoveryIndex: Number.isInteger(assembly.discoveryIndex) ? assembly.discoveryIndex : null,
    } : {}),
  }
  const discovery = assembly.surfaceOrigin === 'ollama-authored-discovery'
  const provenanceKind = discovery ? 'ollama-authored-discovery' : 'ollama-selected-server-rendered'
  const normalized = normalizeTextSource({
    text: lesson.text,
    title: `${discovery ? 'Türkçe keşif' : 'Türkçe temelleri'} · ${plan.unit.title} · ${plan.batchIndex + 1}`,
    language: 'tr',
    provenance: {
      kind: provenanceKind,
      managedByTeacherCampaign: true,
      campaignId: campaign.id,
      curriculumId: campaign.curriculum,
      batchIndex: plan.batchIndex,
      batchId: plan.batchId,
      seed: lesson.attemptSeed,
      campaignSeed: campaign.seed,
      modelIdentity: modelFingerprint,
      modelDigest: campaign.modelIdentity.digest,
      generationModel: lesson.resolvedModel,
      independenceGroupId,
      evaluationEligible: false,
      lessonSkill: lesson.skill,
      lessonForm: lesson.form,
      requestedSurfaceDomain: plan.axes.domain,
      requestedSurfaceTense: plan.axes.tense,
      requestedSurfaceLength: plan.axes.length,
      requestedSurfaceWordOrder: plan.axes.wordOrder,
      actualWordCount: lesson.wordCount,
      validatedLessonForm: true,
      validatedLengthRange: true,
      surfaceCompositionPolicyId: assembly.policyId,
      surfaceTrace: JSON.stringify(boundedSurfaceTrace),
      teacherFeedProtocolVersion: campaign.protocolVersion,
      teacherFeedProtocolFingerprint: campaign.protocolFingerprint,
      spellingPolicyId: campaign.protocol?.spellingPolicyId,
      curriculumInstruction: plan.unit.instruction,
      labelOrigin: 'server-curriculum',
      qualityScore: Number(lesson.qualityScore.toFixed(6)),
      proposalOnly: true,
    },
  }, config)
  normalized.id = `teacher-${stableHash({ campaignId: campaign.id, batchId: plan.batchId, contentHash: normalized.contentHash }).slice(0, 32)}`
  normalized.sourceType = 'ollama-generated'
  normalized.provenance.kind = provenanceKind
  return normalized
}

async function pauseForSafety(context, state, code, message, details = {}) {
  state.stopReason = { code, message, ...details }
  await persistState(context, state, { phase: 'safety-stop', payload: { campaignId: state.campaignId, stopReason: state.stopReason } })
  throw pauseError(message)
}

async function verifyFrozenModel(context, campaign) {
  const models = await context.ollama.listModels({ signal: context.signal })
  const current = models.find((model) => model.name === campaign.model)
  if (!current) {
    const error = new Error(`Dondurulmuş Ollama modeli artık bulunamıyor: ${campaign.model}`)
    error.code = 'STALE_TEACHER_MODEL'
    throw error
  }
  const currentIdentity = normalizeModelIdentity(current, campaign.model)
  if (currentIdentity.fingerprint !== campaign.modelIdentity.fingerprint) {
    const error = new Error('Ollama modelinin digest/kimlik bilgisi kampanya başladıktan sonra değişti.')
    error.code = 'STALE_TEACHER_MODEL'
    error.expectedFingerprint = campaign.modelIdentity.fingerprint
    error.actualFingerprint = currentIdentity.fingerprint
    throw error
  }
  return currentIdentity
}

function initializeDedupWindow(context, campaign, state) {
  if (state.dedupInitialized) return state
  const metadata = typeof context.store.listSources === 'function'
    ? context.store.listSources().slice(0, campaign.dedupWindowSize)
    : []
  const signatures = []
  for (const source of metadata) {
    const content = typeof context.store.getSourceContent === 'function'
      ? context.store.getSourceContent(source.id)
      : typeof context.store.getSource === 'function'
        ? context.store.getSource(source.id)?.content
        : undefined
    if (typeof content !== 'string') continue
    let choiceSignature = null
    try {
      const trace = JSON.parse(source.provenance?.surfaceTrace ?? 'null')
      const identityDigest = trace?.semanticIdentityDigest ?? trace?.semanticCardDigest
      if (identityDigest && trace?.catalogFingerprint) {
        choiceSignature = stableHash({ catalog: trace.catalogFingerprint, semanticIdentityDigest: identityDigest })
      }
    } catch { /* legacy or truncated provenance has text-only dedup */ }
    signatures.push(createTeacherDuplicateSignature(content, choiceSignature))
  }
  state.dedupWindow = signatures.slice(-campaign.dedupWindowSize)
  state.dedupInitialized = true
  return state
}

/**
 * Queue-executor decorator. Required store contract:
 *
 *   commitTeacherFeedBatch(jobId, {
 *     inputs, checkpoint, artifact, usageReservations?, claim?,
 *     allowRequestedUsageFinalization?
 *   })
 *     -> { items: Array<{source, duplicate}>, artifact, job }
 *
 * The method must insert/deduplicate `inputs`, approve+select only newly inserted
 * campaign sources, persist `job.payload.feedState = checkpoint`, and create the
 * immutable batch artifact in one transaction. `generateDiverseStructured` must
 * enforce the supplied JSON schema and use the supplied seed.
 */
export function createTeacherFeedExecutor({ pipelineExecutor, structuralBenchmarkReport } = {}) {
  if (typeof pipelineExecutor !== 'function') throw new TypeError('pipelineExecutor bir fonksiyon olmalı.')
  const benchmarkReportReader = structuralBenchmarkReport ?? (({ store, snapshotId }) =>
    findIndependentStructuralBenchmarkReport(store, snapshotId))
  if (typeof benchmarkReportReader !== 'function') throw new TypeError('structuralBenchmarkReport bir fonksiyon olmalı.')
  return async function teacherFeedExecutor(job, context) {
    if (job?.type !== TEACHER_FEED_JOB_TYPE) return pipelineExecutor(job, context)
    requireExecutorContext(context)
    const campaign = job.payload?.campaign
    if (!campaign || campaign.curriculum !== TEACHER_FEED_CURRICULUM_ID || campaign.mode !== 'research-only' || campaign.proposalOnly !== true) {
      const error = new Error('Öğretmen kampanyası payload sözleşmesine uymuyor.')
      error.code = 'INVALID_TEACHER_CAMPAIGN'
      throw error
    }
    const persisted = context.store.getJob?.(context.jobId)
    let state = validateFeedState(persisted?.payload?.feedState ?? job.payload?.feedState, campaign)
    state.stopReason = null
    const structuralGateProtocolMatches = !campaign.continuousPlan || (
      campaign.continuousPlan.stopGate?.required === true &&
      campaign.continuousPlan.stopGate?.reportOrigin === 'independent-deterministic' &&
      campaign.continuousPlan.stopGate?.artifactKind === INDEPENDENT_STRUCTURAL_BENCHMARK_ARTIFACT_KIND &&
      campaign.continuousPlan.stopGate?.benchmarkFingerprint === INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT
    )
    const generationMode = campaign.generationMode ?? 'controlled'
    const expectedProtocolFingerprint = generationMode === 'hybrid'
      ? TEACHER_FEED_HYBRID_PROTOCOL_FINGERPRINT
      : TEACHER_FEED_PROTOCOL_FINGERPRINT
    if (
      campaign.protocolVersion !== TEACHER_FEED_PROTOCOL_VERSION ||
      campaign.protocolFingerprint !== expectedProtocolFingerprint ||
      campaign.protocol?.spellingPolicyId !== TURKISH_SPELLING_POLICY_ID ||
      !structuralGateProtocolMatches
    ) {
      await pauseForSafety(
        context,
        state,
        'STALE_TEACHER_PROTOCOL',
        'Kampanyanın prompt/kalite/yapısal kapı protokolü çalışan servis sürümüyle uyuşmadığı için güvenli biçimde duraklatıldı.',
        {
          expectedProtocolFingerprint,
          actualProtocolFingerprint: campaign.protocolFingerprint ?? null,
          expectedStructuralBenchmarkFingerprint: campaign.continuousPlan ? INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT : null,
          actualStructuralBenchmarkFingerprint: campaign.continuousPlan?.stopGate?.benchmarkFingerprint ?? null,
        },
      )
    }
    state = await initializeCorpusFrontier(context, campaign, state)
    state = initializeDedupWindow(context, campaign, state)

    if (state.pendingBatchAccounting) state = await accountPendingBatch(context, campaign, state)
    state = await ensureTrainingMilestone(context, campaign, state, { final: state.accepted >= campaign.targetAccepted })
    if (state.pendingTrainingMilestone) state = await completePendingTrainingMilestone({ job, context, campaign, state, pipelineExecutor })

    if (campaign.continuousPlan) {
      if (state.continuous.gatePassed) {
        await context.progress('teacher-feed:structural-gate-passed', 1)
        return teacherFeedResult(campaign, state)
      }
      const boundary = await completeContinuousBoundary({
        job,
        context,
        campaign,
        state,
        pipelineExecutor,
        structuralBenchmarkReport: benchmarkReportReader,
      })
      state = boundary.state
      if (boundary.gatePassed) {
        await context.progress('teacher-feed:structural-gate-passed', 1)
        return teacherFeedResult(campaign, state)
      }
      await assertContinuousBlockAdmission(context, campaign, state, { initial: true })
    }

    while (state.accepted < campaign.targetAccepted) {
      if (state.nextBatchIndex >= campaign.maxSafetyBatches) {
        await pauseForSafety(context, state, 'TEACHER_FEED_SAFETY_BATCH_LIMIT', 'Kampanya kabul hedefine güvenli parti sınırı içinde ulaşamadığı için duraklatıldı.')
      }
      const plan = planTeacherFeedBatch({ seed: campaign.seed, batchIndex: state.nextBatchIndex, campaignId: campaign.id })
      const generationLane = teacherFeedGenerationLane(generationMode, plan.batchIndex)
      await context.progress('teacher-feed:generating', state.accepted / campaign.targetAccepted)
      try {
        await verifyFrozenModel(context, campaign)
      } catch (error) {
        if (error?.code === 'STALE_TEACHER_MODEL') {
          await pauseForSafety(context, state, 'STALE_TEACHER_MODEL', error.message, {
            expectedFingerprint: error.expectedFingerprint ?? campaign.modelIdentity.fingerprint,
            actualFingerprint: error.actualFingerprint ?? null,
          })
        }
        throw error
      }
      let generated
      try {
        const generationTarget = campaign.continuousPlan
          ? state.continuous.nextBlockAt
          : campaign.targetAccepted
        generated = await generateBatch({
          campaign,
          plan,
          state,
          context,
          lane: generationLane,
          desiredCount: Math.min(campaign.generationBatchSize, generationTarget - state.accepted),
        })
      } catch (error) {
        if (isBudgetError(error)) {
          await pauseForSafety(context, state, 'TEACHER_FEED_BUDGET_STOP', 'Öğretmen kampanyası kullanım bütçesine ulaştığı için güvenli biçimde duraklatıldı.', { causeCode: String(error.code ?? 'BUDGET') })
        }
        if (isTeacherCatalogError(error)) {
          await pauseForSafety(
            context,
            state,
            'TEACHER_FEED_CATALOG_STOP',
            'Doğrulanmış Türkçe olay kartı panosu kurulamadığı için kampanya Ollama’ya yeni istek göndermeden duraklatıldı.',
            { causeCode: String(error?.code ?? 'CATALOG_ERROR') },
          )
        }
        if (error?.code === 'STALE_TEACHER_MODEL_RESPONSE') {
          await pauseForSafety(
            context,
            state,
            'STALE_TEACHER_MODEL',
            'Ollama yanıtı kampanyada dondurulan model kimliğiyle uyuşmadığı için kampanya duraklatıldı.',
            {
              expectedModel: error.expectedModel ?? campaign.modelIdentity.name,
              actualModel: error.actualModel ?? null,
              causeCode: error.code,
            },
          )
        }
        if (isOllamaTransportError(error)) {
          const timeout = error?.code === 'OLLAMA_TIMEOUT'
          await pauseForSafety(
            context,
            state,
            timeout ? 'TEACHER_FEED_OLLAMA_TIMEOUT' : 'TEACHER_FEED_OLLAMA_UNAVAILABLE',
            timeout
              ? 'Ollama çağrısı süre sınırını aştığı için kampanya aynı pahalı çağrıyı yinelemeden duraklatıldı.'
              : 'Ollama geçici olarak güvenilir bir yapılandırılmış yanıt veremediği için kampanya duraklatıldı.',
            { causeCode: String(error?.code ?? 'OLLAMA_ERROR'), timeoutMs: Number(error?.timeoutMs ?? 0) || null },
          )
        }
        throw error
      }

      const inputs = generated.accepted.map((lesson) => sourceFromLesson({ lesson, campaign, plan, config: context.config ?? context.store.getConfig() }))
      const duplicateRejections = generated.rejected.filter((item) => item.reasons.some((reason) => ['exact-duplicate', 'near-duplicate', 'catalog-choice-duplicate'].includes(reason))).length
      state.nextBatchIndex = plan.batchIndex + 1
      state.generated += generated.accepted.length + generated.rejected.length
      state.duplicates += duplicateRejections
      state.rejectedQuality += generated.rejected.length - duplicateRejections
      state.dedupWindow = generated.history.slice(-campaign.dedupWindowSize)
      state.pendingGenerationBatch = null
      state.pendingBatchAccounting = {
        batchIndex: plan.batchIndex,
        batchId: plan.batchId,
        sourceIds: inputs.map((input) => input.id),
        candidateCount: inputs.length,
        generationLane,
      }
      if (state.lanes?.[generationLane]) {
        const laneCounters = state.lanes[generationLane]
        laneCounters.generated += generated.accepted.length + generated.rejected.length
        laneCounters.duplicates += duplicateRejections
        laneCounters.rejectedQuality += generated.rejected.length - duplicateRejections
        laneCounters.teacherRequests += generated.calls
        laneCounters.teacherTokens += generated.promptTokens + generated.completionTokens
      }

      await context.store.commitTeacherFeedBatch(context.jobId, {
        inputs,
        checkpoint: structuredClone(state),
        ...(context.jobClaim ? { claim: context.jobClaim } : {}),
        artifact: {
          kind: 'teacher-feed-batch',
          contentHash: stableHash({ campaignId: campaign.id, batchId: plan.batchId, contentHashes: inputs.map((input) => input.contentHash) }),
          phase: 'batch-stored',
          payload: {
            campaignId: campaign.id,
            curriculumId: campaign.curriculum,
            batchIndex: plan.batchIndex,
            batchId: plan.batchId,
            batchSeed: plan.batchSeed,
            unitId: plan.unit.id,
            plannedForm: plan.form,
            generationLane,
            requestedSurfaceAxes: plan.axes,
            protocolVersion: campaign.protocolVersion,
            protocolFingerprint: campaign.protocolFingerprint,
            spellingPolicyId: campaign.protocol?.spellingPolicyId,
            generatedCalls: generated.calls,
            accepted: inputs.length,
            rejected: generated.rejected.length,
            rejectionReasons: rejectionSummary(generated.rejected),
            rejectedDiagnostics: generated.rejected.slice(0, 24).map((item) => ({
              text: String(item.text ?? '').slice(0, 280),
              reasons: item.reasons,
              spellingUnknownCount: item.spellingUnknownCount,
              spellingDiagnostics: item.spellingDiagnostics,
              surfaceAssembly: item.surfaceAssembly,
            })),
            acceptedChoiceEvidence: generationLane === 'controlled' ? generated.accepted.map((lesson) => ({
              boardId: lesson.surfaceAssembly?.boardId ?? null,
              boardFingerprint: lesson.surfaceAssembly?.boardFingerprint ?? null,
              choiceId: lesson.choiceId ?? null,
              semanticCardDigest: lesson.semanticCardDigest ?? null,
              semanticIdentityDigest: lesson.semanticIdentityDigest ?? null,
              semanticCard: lesson.surfaceAssembly?.semanticCard ?? null,
              renderPlan: lesson.surfaceAssembly?.renderPlan ?? null,
              finalTextDigest: lesson.surfaceAssembly?.finalTextDigest ?? null,
            })) : [],
            acceptedDiscoveryEvidence: generationLane === 'discovery' ? generated.accepted.map((lesson) => ({
              discoveryAttemptId: lesson.surfaceAssembly?.discoveryAttemptId ?? null,
              discoveryPromptFingerprint: lesson.surfaceAssembly?.discoveryPromptFingerprint ?? null,
              finalTextDigest: lesson.surfaceAssembly?.finalTextDigest ?? null,
              qualityScore: Number(lesson.qualityScore.toFixed(6)),
            })) : [],
            promptTokens: generated.promptTokens,
            completionTokens: generated.completionTokens,
            usageEstimated: generated.usageEstimated,
            evaluationEligible: false,
            proposalOnly: true,
          },
        },
      })

      state = await accountPendingBatch(context, campaign, state)
      state = await ensureTrainingMilestone(context, campaign, state)
      state = await completePendingTrainingMilestone({ job, context, campaign, state, pipelineExecutor })
      if (campaign.continuousPlan) {
        const boundary = await completeContinuousBoundary({
          job,
          context,
          campaign,
          state,
          pipelineExecutor,
          structuralBenchmarkReport: benchmarkReportReader,
        })
        state = boundary.state
        if (boundary.gatePassed) {
          await context.progress('teacher-feed:structural-gate-passed', 1)
          return teacherFeedResult(campaign, state)
        }
      }
      if (state.consecutiveEmptyBatches >= campaign.maxConsecutiveEmptyBatches) {
        await pauseForSafety(context, state, 'TEACHER_FEED_QUALITY_STALL', `${state.consecutiveEmptyBatches} ardışık partide yeni kaliteli kaynak oluşmadığı için kampanya duraklatıldı.`)
      }
    }

    state = await ensureTrainingMilestone(context, campaign, state, { final: true })
    state = await completePendingTrainingMilestone({ job, context, campaign, state, pipelineExecutor })
    state.stopReason = null
    await context.progress('teacher-feed:completed', 1)
    return teacherFeedResult(campaign, state)
  }
}
