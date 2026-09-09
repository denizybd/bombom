import http from 'node:http'
import { randomUUID, timingSafeEqual } from 'node:crypto'
import path from 'node:path'
import { createTrainingStore, sourceRepresentationRevision } from './store.mjs'
import { OllamaTeacher } from './ollama.mjs'
import { createPipelineExecutor, generateTeacherSources } from './pipeline.mjs'
import {
  createTeacherFeedCampaignInput,
  createTeacherFeedExecutor,
  TEACHER_FEED_JOB_TYPE,
} from './teacher-feed.mjs'
import { DurableJobQueue } from './queue.mjs'
import { fetchAllowedUrl, normalizeTextSource, readAllowedFile, sha256 } from './source-policy.mjs'
import { mergeAndValidateConfig, publicConfig } from './config.mjs'
import { scanWatchedSources, SourceWatcher } from './watcher.mjs'
import { inspectFamilies, parseFamilyInspectionQuery } from './family-inspection.mjs'
import { createNeuronAtlas } from './neuron-atlas.mjs'
import {
  inspectConceptMacro,
  inspectConceptMacros,
  parseConceptInspectionQuery,
} from './concept-inspection.mjs'
import {
  inspectCurrentPlasticRegionAtlas,
  inspectPlasticRegionAtlas,
  inspectPlasticRegionStatus,
  listPlasticRegionRecords,
  parsePlasticRegionListQuery,
} from './plastic-region-inspection.mjs'
import { inspectSemanticProbeHypotheses } from './plastic-region-semantic-inspection.mjs'
import {
  plasticRegionSnapshotActivationGate,
  readPlasticRegionRelease,
  snapshotUsesAdaptiveRegionRepresentation,
} from '../runtime/plastic-region-cache.mjs'
import {
  PLASTIC_REGION_RELEASE_JOB_TYPE,
  createPlasticRegionReleaseExecutor,
  preparePlasticRegionReleaseJob,
} from '../runtime/plastic-region-release-runner.mjs'
import {
  PLASTIC_REGION_RENEWAL_JOB_TYPE,
  createPlasticRegionRenewalExecutor,
  preparePlasticRegionRenewalJob,
} from '../runtime/plastic-region-renewal-runner.mjs'
import {
  createThoughtService,
  parseThoughtGraphQuery,
  parseThoughtListQuery,
  parseThoughtTickQuery,
} from './thought-service.mjs'
import {
  fromPublicConfig, toPublicConfig, toPublicEvent, toPublicMetrics, toPublicModel,
  toPublicQuarantine, toPublicQueueItem, toPublicSnapshot, toPublicSource, toPublicStatus,
  toPublicTeacherFeedCampaign,
} from './public-api.mjs'

const LOOPBACK_HOSTS = new Set(['127.0.0.1', '::1', 'localhost'])
const LOCAL_ORIGIN = /^http:\/\/(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?$/

function apiError(message, code = 'BAD_REQUEST', status = 400, details) {
  const error = new Error(message)
  error.code = code
  error.status = status
  if (details !== undefined) error.details = details
  return error
}

function assertReviewableSnapshotProducer(store, snapshot) {
  if (typeof snapshot.producerJobId !== 'string' || !snapshot.producerJobId) {
    throw apiError(
      'Adayın tamamlanmış bir eğitim işiyle üretildiği doğrulanamıyor; aday yeniden oluşturulmalı.',
      'SNAPSHOT_PRODUCER_UNVERIFIED',
      409,
    )
  }
  const producer = store.getJob(snapshot.producerJobId)
  const certifiedSnapshotId = producer?.result?.snapshotId ?? producer?.result?.lastSnapshotId ?? null
  if (producer?.state !== 'succeeded' || certifiedSnapshotId !== snapshot.id) {
    throw apiError(
      'Adayı üreten eğitim denemesi başarıyla tamamlanmadığı için manuel etkinleştirme kapalı.',
      'SNAPSHOT_PRODUCER_INCOMPLETE',
      409,
      {
        producerJobId: snapshot.producerJobId,
        producerState: producer?.state ?? null,
        certifiedSnapshotId,
      },
    )
  }
}

function sendJson(response, status, body, headers = {}) {
  const payload = JSON.stringify(body)
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': Buffer.byteLength(payload), 'Cache-Control': 'no-store', ...headers })
  response.end(payload)
}

async function readJson(request, limit) {
  const contentType = request.headers['content-type'] ?? ''
  if (!contentType.toLowerCase().startsWith('application/json')) throw apiError('Content-Type application/json olmalı.', 'UNSUPPORTED_MEDIA_TYPE', 415)
  const chunks = []
  let total = 0
  for await (const chunk of request) {
    total += chunk.length
    if (total > limit) {
      request.destroy()
      throw apiError('İstek gövdesi boyut sınırını aşıyor.', 'REQUEST_TOO_LARGE', 413)
    }
    chunks.push(chunk)
  }
  if (!chunks.length) return {}
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')) } catch { throw apiError('Geçersiz JSON gövdesi.', 'INVALID_JSON', 400) }
}

async function readRaw(request, limit) {
  const chunks = []
  let total = 0
  for await (const chunk of request) {
    total += chunk.length
    if (total > limit) {
      request.destroy()
      throw apiError('İstek gövdesi boyut sınırını aşıyor.', 'REQUEST_TOO_LARGE', 413)
    }
    chunks.push(chunk)
  }
  return Buffer.concat(chunks)
}

async function readMultipart(request, limit, contentType) {
  const match = /boundary=(?:"([^"]+)"|([^;\s]+))/i.exec(contentType)
  const boundary = match?.[1] ?? match?.[2]
  if (!boundary || boundary.length > 200 || /[\r\n]/.test(boundary)) throw apiError('Multipart sınırı geçersiz.', 'INVALID_MULTIPART')
  const body = await readRaw(request, limit)
  const delimiter = Buffer.from(`--${boundary}`)
  const marker = Buffer.from(`\r\n--${boundary}`)
  const headerSeparator = Buffer.from('\r\n\r\n')
  const files = []
  const fields = {}
  let cursor = body.indexOf(delimiter)
  while (cursor >= 0) {
    cursor += delimiter.length
    if (body.subarray(cursor, cursor + 2).equals(Buffer.from('--'))) break
    if (body.subarray(cursor, cursor + 2).equals(Buffer.from('\r\n'))) cursor += 2
    const headerEnd = body.indexOf(headerSeparator, cursor)
    if (headerEnd < 0 || headerEnd - cursor > 16_384) throw apiError('Multipart başlığı geçersiz.', 'INVALID_MULTIPART')
    const next = body.indexOf(marker, headerEnd + headerSeparator.length)
    if (next < 0) throw apiError('Multipart gövdesi tamamlanmamış.', 'INVALID_MULTIPART')
    const headers = body.subarray(cursor, headerEnd).toString('utf8')
    const disposition = /content-disposition:\s*form-data;([^\r\n]+)/i.exec(headers)?.[1] ?? ''
    const name = /(?:^|;)\s*name="([^"]+)"/i.exec(disposition)?.[1]
    const filename = /(?:^|;)\s*filename="([^"]*)"/i.exec(disposition)?.[1]
    const data = body.subarray(headerEnd + headerSeparator.length, next)
    if (name && filename !== undefined) {
      if (files.length >= 100) throw apiError('Bir istekte en fazla 100 dosya kabul edilir.', 'TOO_MANY_FILES')
      files.push({ name, filename: safeFilename(filename), data, contentType: /content-type:\s*([^\r\n]+)/i.exec(headers)?.[1]?.trim() ?? 'application/octet-stream' })
    } else if (name && data.length <= 10_000) {
      fields[name] = data.toString('utf8')
    }
    cursor = next + 2
  }
  return { files, fields }
}

function safeFilename(value) {
  return path.basename(String(value).replaceAll('\\', '/')).replace(/[\u0000-\u001f\u007f]/g, '').slice(0, 240) || 'adsiz.txt'
}

function normalizeUploadedFile(file, provenance, config) {
  const extension = path.extname(file.filename).toLowerCase()
  if (!config.sourcePolicy.allowedExtensions.includes(extension)) throw apiError(`${extension || 'Uzantısız dosya'} kabul edilmiyor.`, 'EXTENSION_NOT_ALLOWED', 415)
  if (file.data.length > config.budgets.maxSourceBytes) throw apiError('Dosya kaynak boyutu sınırını aşıyor.', 'SOURCE_TOO_LARGE', 413)
  let text
  try { text = new TextDecoder('utf-8', { fatal: true }).decode(file.data) } catch { throw apiError(`${file.filename} geçerli UTF-8 metni değil.`, 'INVALID_FILE_ENCODING', 422) }
  const normalized = normalizeTextSource({ text, title: file.filename, provenance: { kind: 'local-upload', filename: file.filename, label: String(provenance).slice(0, 1_000) } }, config)
  normalized.sourceType = 'local-upload'
  normalized.locator = null
  normalized.provenance.kind = 'local-upload'
  return normalized
}

const TEXT_LAB_CONFIG_KEYS = Object.freeze([
  'matrixSide',
  'identityRatio',
  'minimumLeafGraphemes',
  'hashSeed',
  'contextRetention',
  'parentTraceWeight',
  'siblingTraceWeight',
  'minimumTrace',
  'identitySimilarityWeight',
  'habitExponent',
])
const TEXT_LAB_REPRESENTATION_PROFILES = Object.freeze({
  'legacy-quarter-id-v2': Object.freeze({ matrixSide: 64, identityRatio: 0.25 }),
  'adaptive-eighth-id-v1': Object.freeze({ matrixSide: 64, identityRatio: 0.125 }),
})

function normalizeTextLabManifestConfig(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw apiError('Metin Laboratuvarı temsil ayarları geçersiz.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  }
  const ownKeys = Object.keys(value)
  if (
    (ownKeys.length !== TEXT_LAB_CONFIG_KEYS.length && ownKeys.length !== TEXT_LAB_CONFIG_KEYS.length + 1) ||
    ownKeys.some((key) => !TEXT_LAB_CONFIG_KEYS.includes(key) && key !== 'representationProfile') ||
    TEXT_LAB_CONFIG_KEYS.some((key) => !Object.hasOwn(value, key) || typeof value[key] !== 'number' || !Number.isFinite(value[key]))
  ) {
    throw apiError('Metin Laboratuvarı temsil ayarları eksik, fazla alanlı veya sayısal olarak geçersiz.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  }
  const config = Object.fromEntries(TEXT_LAB_CONFIG_KEYS.map((key) => [key, value[key]]))
  const representationProfile = Object.hasOwn(value, 'representationProfile')
    ? value.representationProfile
    : null
  const profileContract = representationProfile !== null &&
    typeof representationProfile === 'string' &&
    Object.hasOwn(TEXT_LAB_REPRESENTATION_PROFILES, representationProfile)
    ? TEXT_LAB_REPRESENTATION_PROFILES[representationProfile]
    : null
  if (representationProfile !== null && !profileContract) {
    throw apiError('Metin Laboratuvarı temsil profili bilinmiyor.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  }
  const contextWeight = config.contextRetention + config.parentTraceWeight + config.siblingTraceWeight
  if (
    !Number.isInteger(config.matrixSide) || config.matrixSide < 8 || config.matrixSide > 64 ||
    !(config.identityRatio > 0 && config.identityRatio < 1) ||
    !Number.isInteger(config.minimumLeafGraphemes) || config.minimumLeafGraphemes < 1 ||
    !Number.isInteger(config.hashSeed) ||
    config.contextRetention < 0 || config.parentTraceWeight < 0 || config.siblingTraceWeight < 0 ||
    Math.abs(contextWeight - 1) > 1e-8 ||
    config.minimumTrace < 0 || config.minimumTrace > 1 ||
    !(config.identitySimilarityWeight > 0 && config.identitySimilarityWeight <= 1) ||
    !(config.habitExponent > 0)
  ) {
    throw apiError('Metin Laboratuvarı temsil ayarları motorun güvenli sınırlarıyla uyuşmuyor.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  }
  if (profileContract && (
    config.matrixSide !== profileContract.matrixSide ||
    config.identityRatio !== profileContract.identityRatio
  )) {
    throw apiError('Metin Laboratuvarı temsil profili fiziksel matris sözleşmesiyle uyuşmuyor.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  }
  return representationProfile === null ? config : { ...config, representationProfile }
}

function normalizeTextLabManifest(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || value.format !== 'text-lab-recompute-proof-v1') {
    throw apiError('Metin Laboratuvarı yeniden üretim kanıtı geçersiz.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  }
  if (
    typeof value.originalSourceId !== 'string' || !value.originalSourceId || value.originalSourceId.length > 500 ||
    typeof value.sourceDigest !== 'string' || !/^[0-9a-f]{16}$/.test(value.sourceDigest) ||
    typeof value.representationSpaceId !== 'string' || !/^RS-[0-9A-F]{16}$/.test(value.representationSpaceId) ||
    !Array.isArray(value.identityProofs) || value.identityProofs.length < 1 || value.identityProofs.length > 1_600
  ) throw apiError('Metin Laboratuvarı kanıt alanları eksik veya sınır dışında.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  const config = normalizeTextLabManifestConfig(value.config)
  const identityProofs = value.identityProofs.map((proof) => {
    if (
      !proof || typeof proof !== 'object' ||
      !Number.isInteger(proof.start) || !Number.isInteger(proof.end) || proof.start < 0 || proof.end <= proof.start ||
      typeof proof.identityDigest !== 'string' || !/^[0-9a-f]{16}$/.test(proof.identityDigest)
    ) throw apiError('Metin Laboratuvarı kimlik kanıtı bozuk.', 'INVALID_TEXT_LAB_MANIFEST', 422)
    return { start: proof.start, end: proof.end, identityDigest: proof.identityDigest }
  }).sort((left, right) => left.start - right.start || left.end - right.end || left.identityDigest.localeCompare(right.identityDigest))
  const uniqueProofs = new Set(identityProofs.map((proof) => `${proof.start}:${proof.end}:${proof.identityDigest}`))
  if (uniqueProofs.size !== identityProofs.length) {
    throw apiError('Metin Laboratuvarı kimlik kanıtı yinelenen düğümler içeriyor.', 'INVALID_TEXT_LAB_MANIFEST', 422)
  }
  return {
    format: value.format,
    originalSourceId: value.originalSourceId,
    sourceDigest: value.sourceDigest,
    representationSpaceId: value.representationSpaceId,
    config,
    identityProofs,
  }
}

function textItemProvenance(item) {
  const provenance = typeof item.provenance === 'string'
    ? { label: item.provenance }
    : structuredClone(item.provenance ?? {})
  if (item.analysisManifest !== undefined) {
    provenance.kind = 'text-lab-archive'
    provenance.textLabManifest = normalizeTextLabManifest(item.analysisManifest)
  }
  return provenance
}

function normalizeTextItem(item, config) {
  const provenance = textItemProvenance(item)
  const normalized = normalizeTextSource({
    text: item.text ?? item.content,
    title: item.name ?? item.title,
    language: item.language,
    provenance,
    preserveWhitespace: item.analysisManifest !== undefined,
  }, config)
  if (provenance.textLabManifest) {
    if (provenance.textLabManifest.identityProofs.some((proof) => proof.end > normalized.content.length)) {
      throw apiError('Metin Laboratuvarı kimlik kanıtındaki düğüm aralığı kaynak metnin dışına taşıyor.', 'INVALID_TEXT_LAB_MANIFEST', 422)
    }
    normalized.provenance.textLabManifest = provenance.textLabManifest
  }
  return normalized
}

function validateTeacherFeedHttpRequest(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw apiError('14B besleme isteği bir nesne olmalı.', 'INVALID_TEACHER_CAMPAIGN')
  const allowed = new Set(['model', 'seed', 'targetAccepted', 'generationBatchSize', 'trainEveryAccepted', 'curriculum', 'mode', 'generationMode', 'idempotencyKey', 'continuousPlan'])
  const unknown = Object.keys(value).filter((key) => !allowed.has(key))
  if (unknown.length) throw apiError(`14B besleme isteğinde desteklenmeyen alanlar var: ${unknown.join(', ')}`, 'INVALID_TEACHER_CAMPAIGN')
  const integer = (name, minimum, maximum) => {
    if (!Number.isInteger(value[name]) || value[name] < minimum || value[name] > maximum) {
      throw apiError(`${name}, ${minimum}–${maximum} arasında bir tam sayı olmalı.`, 'INVALID_TEACHER_CAMPAIGN')
    }
  }
  if (typeof value.model !== 'string' || !value.model.trim() || value.model.length > 160) throw apiError('Geçerli bir yerel Ollama modeli seçilmeli.', 'INVALID_TEACHER_CAMPAIGN')
  integer('seed', 0, 0xffff_ffff)
  integer('targetAccepted', 4, 10_000)
  integer('generationBatchSize', 1, 2)
  integer('trainEveryAccepted', 4, 512)
  let trainingCeiling = value.targetAccepted
  if (value.continuousPlan !== undefined) {
    if (!value.continuousPlan || typeof value.continuousPlan !== 'object' || Array.isArray(value.continuousPlan)) {
      throw apiError('continuousPlan bir ayar nesnesi olmalı.', 'INVALID_TEACHER_CAMPAIGN')
    }
    const planAllowed = new Set(['blockAccepted', 'maxBlocks', 'minimumDiskHeadroomMb', 'maximumSyntheticSources'])
    const planUnknown = Object.keys(value.continuousPlan).filter((key) => !planAllowed.has(key))
    if (planUnknown.length) throw apiError(`continuousPlan desteklenmeyen alanlar içeriyor: ${planUnknown.join(', ')}`, 'INVALID_TEACHER_CAMPAIGN')
    const planInteger = (name, minimum, maximum, fallback) => {
      const resolved = value.continuousPlan[name] ?? fallback
      if (!Number.isInteger(resolved) || resolved < minimum || resolved > maximum) {
        throw apiError(`continuousPlan.${name}, ${minimum}–${maximum} arasında bir tam sayı olmalı.`, 'INVALID_TEACHER_CAMPAIGN')
      }
      return resolved
    }
    const blockAccepted = planInteger('blockAccepted', 4, 10_000, value.targetAccepted)
    const maxBlocks = planInteger('maxBlocks', 1, 100, 32)
    planInteger('minimumDiskHeadroomMb', 4, 16_384, 512)
    planInteger('maximumSyntheticSources', 16, 1_000_000, 100_000)
    if (blockAccepted * maxBlocks > 100_000) throw apiError('continuousPlan toplam kabul sınırı 100.000 örneği aşamaz.', 'INVALID_TEACHER_CAMPAIGN')
    trainingCeiling = blockAccepted
  }
  if (value.trainEveryAccepted > trainingCeiling) throw apiError('Eğitim çevrimi eşiği güvenli blok kabul hedefini aşamaz.', 'INVALID_TEACHER_CAMPAIGN')
  if (value.curriculum !== 'turkish-foundations-v1' || value.mode !== 'research-only') throw apiError('Kampanya yalnız turkish-foundations-v1 / research-only sözleşmesiyle çalışabilir.', 'INVALID_TEACHER_CAMPAIGN')
  if (value.generationMode !== undefined && !['controlled', 'hybrid'].includes(value.generationMode)) {
    throw apiError('generationMode yalnız controlled veya hybrid olabilir.', 'INVALID_TEACHER_CAMPAIGN')
  }
  if (value.idempotencyKey !== undefined && (typeof value.idempotencyKey !== 'string' || !value.idempotencyKey.trim() || value.idempotencyKey.length > 240)) {
    throw apiError('idempotencyKey 1–240 karakterlik bir metin olmalı.', 'INVALID_TEACHER_CAMPAIGN')
  }
  return value
}

function canonicalTeacherFeedHttpRequest(value) {
  const generationMode = value.generationMode ?? 'controlled'
  const continuous = value.continuousPlan
    ? {
        blockAccepted: value.continuousPlan.blockAccepted ?? value.targetAccepted,
        maxBlocks: value.continuousPlan.maxBlocks ?? 32,
        minimumDiskHeadroomMb: value.continuousPlan.minimumDiskHeadroomMb ?? 512,
        maximumSyntheticSources: value.continuousPlan.maximumSyntheticSources ?? 100_000,
      }
    : null
  // Fixed field order and normalized defaults make this a semantic request
  // identity. Transport metadata and the later random campaign id are absent.
  return {
    format: generationMode === 'hybrid' ? 'teacher-feed-campaign-request-v2-hybrid' : 'teacher-feed-campaign-request-v1',
    model: value.model.trim(),
    seed: value.seed,
    targetAccepted: value.targetAccepted,
    generationBatchSize: value.generationBatchSize,
    trainEveryAccepted: value.trainEveryAccepted,
    curriculum: value.curriculum,
    mode: value.mode,
    ...(generationMode === 'hybrid' ? { generationMode } : {}),
    continuousPlan: continuous,
  }
}

function teacherFeedHttpRequestHash(value) {
  return sha256(JSON.stringify(canonicalTeacherFeedHttpRequest(value)))
}

function assertTeacherFeedIdempotency(existing, requestHash) {
  if (existing.type !== TEACHER_FEED_JOB_TYPE || (existing.requestHash && existing.requestHash !== requestHash)) {
    throw apiError(
      'Idempotency anahtarı farklı bir 14B kampanya isteği için kullanılmış.',
      'TEACHER_CAMPAIGN_IDEMPOTENCY_CONFLICT',
      409,
    )
  }
}

function requestSignal(request, response) {
  const controller = new AbortController()
  const abort = () => {
    if (controller.signal.aborted) return
    controller.abort(Object.assign(new Error('İstemci bağlantıyı kapattı.'), { code: 'CLIENT_DISCONNECTED' }))
  }
  const onResponseClose = () => {
    if (!response?.writableEnded) abort()
  }
  const cleanup = () => {
    request.off('aborted', abort)
    response?.off('close', onResponseClose)
  }
  request.once('aborted', abort)
  response?.once('close', onResponseClose)
  response?.once('finish', cleanup)
  return controller.signal
}

class SseHub {
  constructor() { this.clients = new Set() }
  add(response) {
    this.clients.add(response)
    response.on('close', () => this.clients.delete(response))
  }
  publish(event) {
    const visible = toPublicEvent(event)
    const message = `id: ${visible.id}\ndata: ${JSON.stringify(visible)}\n\n`
    for (const client of this.clients) client.write(message)
  }
  close() {
    for (const client of this.clients) client.end()
    this.clients.clear()
  }
}

export async function createTrainingService(options = {}) {
  const store = options.store ?? await createTrainingStore({ dataDirectory: options.dataDirectory, forceJson: options.forceJson, requireSqlite: options.requireSqlite })
  const ollama = options.ollama ?? new OllamaTeacher(() => store.getConfig(), options.fetchImpl)
  const pipelineExecutor = options.pipelineExecutor ?? createPipelineExecutor({ stageHandlers: options.stageHandlers })
  const ordinaryJobExecutor = options.jobExecutor ?? createTeacherFeedExecutor({
    pipelineExecutor,
    structuralBenchmarkReport: options.structuralBenchmarkReport,
  })
  const plasticRegionReleaseExecutor = options.plasticRegionReleaseExecutor ?? createPlasticRegionReleaseExecutor({
    ...(options.thoughtEngine ? { thoughtEngine: options.thoughtEngine } : {}),
  })
  const plasticRegionRenewalExecutor = options.plasticRegionRenewalExecutor ?? createPlasticRegionRenewalExecutor({
    ...(options.thoughtEngine ? { thoughtEngine: options.thoughtEngine } : {}),
  })
  const jobExecutor = (job, context) => job?.type === PLASTIC_REGION_RELEASE_JOB_TYPE
    ? plasticRegionReleaseExecutor(job, context)
    : job?.type === PLASTIC_REGION_RENEWAL_JOB_TYPE
      ? plasticRegionRenewalExecutor(job, context)
      : ordinaryJobExecutor(job, context)
  const sse = new SseHub()
  const queue = options.queue ?? new DurableJobQueue({ store, ollama, executor: jobExecutor, onEvent: (event) => sse.publish(event) })
  const thought = createThoughtService({
    store,
    ollama,
    thoughtEngine: options.thoughtEngine ?? null,
    onEvent: (event) => queue.emit(event),
  })
  const watcher = new SourceWatcher(store, (result) => { void queue.emit({ type: result.error ? 'source-scan-failed' : 'source-scan-completed', level: result.error ? 'error' : 'info', data: result }) })
  const sockets = new Set()

  const server = http.createServer(async (request, response) => {
    const requestId = randomUUID()
    try {
      applyCors(request, response)
      if (request.method === 'OPTIONS') {
        response.writeHead(204)
        response.end()
        return
      }
      const url = new URL(request.url ?? '/', 'http://127.0.0.1')
      const segments = url.pathname.split('/').filter(Boolean).map((segment) => decodeURIComponent(segment))
      if (segments[0] !== 'api') throw apiError('Uç nokta bulunamadı.', 'NOT_FOUND', 404)
      const route = segments.slice(1)

      if (request.method === 'GET' && route.join('/') === 'health') {
        sendJson(response, 200, { ok: true, service: 'autonomous-training', version: 1, storage: store.backendKind, now: new Date().toISOString() })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'status') {
        const ollamaStatus = await ollama.check({ signal: requestSignal(request, response) })
        sendJson(response, 200, toPublicStatus(store, ollamaStatus))
        return
      }
      if (request.method === 'GET' && route.join('/') === 'ollama/models') {
        const models = await ollama.listModels({ signal: requestSignal(request, response) })
        sendJson(response, 200, models.map(toPublicModel))
        return
      }
      if ((request.method === 'GET' || request.method === 'POST') && route.join('/') === 'ollama/check') {
        const check = await ollama.check({ signal: requestSignal(request, response) })
        sendJson(response, check.available ? 200 : 503, { ok: check.available, ...check })
        return
      }

      if (request.method === 'GET' && route.join('/') === 'sources') {
        sendJson(response, 200, store.listSources({
          status: url.searchParams.get('status') || undefined,
          includeSummary: true,
        }).map(toPublicSource))
        return
      }
      if (request.method === 'GET' && route.join('/') === 'quarantine') {
        sendJson(response, 200, store.listSources({ includeSummary: true }).map(toPublicQuarantine))
        return
      }
      if (request.method === 'POST' && ['sources/text', 'sources/ingest'].includes(route.join('/'))) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const items = Array.isArray(body.items) ? body.items : [body]
        if (!items.length || items.length > 100) throw apiError('Bir istekte 1–100 metin kabul edilir.', 'INVALID_ITEM_COUNT')
        const created = store.createSources(items.map((item) => normalizeTextItem(item, store.getConfig())))
        await queue.emit({ type: 'sources-quarantined', data: { inserted: created.filter((item) => !item.duplicate).length, duplicates: created.filter((item) => item.duplicate).length } })
        if (route.join('/') === 'sources/text' && !Array.isArray(body.items)) sendJson(response, 201, toPublicSource(store.getSource(created[0].source.id)))
        else sendJson(response, 201, created.map((item) => ({ source: toPublicSource(store.getSource(item.source.id)), duplicate: item.duplicate })))
        return
      }
      if (request.method === 'POST' && route.join('/') === 'sources/files') {
        const items = []
        const contentType = request.headers['content-type'] ?? ''
        if (contentType.startsWith('multipart/form-data')) {
          const parts = await readMultipart(request, store.getConfig().budgets.maxBatchBytes, contentType)
          const provenance = parts.fields.provenance || 'Kullanıcının yerel dosya seçicisi'
          items.push(...store.createSources(parts.files.map((file) => normalizeUploadedFile(file, provenance, store.getConfig()))))
        } else {
          const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
          if (!Array.isArray(body.paths) || !body.paths.length || body.paths.length > 100 || body.paths.some((item) => typeof item !== 'string')) throw apiError('paths alanı 1–100 dosya yolu içermeli.', 'INVALID_PATHS')
          for (const filePath of body.paths) items.push(store.createSource(await readAllowedFile(filePath, store.getConfig())))
        }
        if (!items.length) throw apiError('En az bir dosya gerekli.', 'INVALID_FILES')
        await queue.emit({ type: 'sources-quarantined', data: { inserted: items.filter((item) => !item.duplicate).length, duplicates: items.filter((item) => item.duplicate).length } })
        sendJson(response, 201, items.map((item) => toPublicSource(store.getSource(item.source.id))))
        return
      }
      if (request.method === 'POST' && route.join('/') === 'sources/url') {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        if (typeof body.url !== 'string') throw apiError('url alanı gerekli.', 'INVALID_URL')
        const created = store.createSource(await fetchAllowedUrl(body.url, store.getConfig(), { signal: requestSignal(request, response) }))
        await queue.emit({ type: 'source-quarantined', data: { sourceId: created.source.id, duplicate: created.duplicate } })
        sendJson(response, 201, { ok: true, ...sanitizeCreatedSource(created) })
        return
      }
      if (request.method === 'POST' && route.join('/') === 'sources/scan') {
        const result = await scanWatchedSources(store, { signal: requestSignal(request, response) })
        await queue.emit({ type: 'source-scan-completed', data: result })
        sendJson(response, 200, { ok: true, ...result })
        return
      }
      if (request.method === 'POST' && route.join('/') === 'teacher/generate') {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const result = await generateTeacherSources({ topic: body.topic, objective: body.objective, diversity: body.diversity, count: body.count, language: body.language, model: body.model }, { store, ollama, signal: requestSignal(request, response) })
        await queue.emit({ type: 'teacher-sources-quarantined', data: { generated: result.generated, inserted: result.inserted, duplicates: result.duplicates, model: result.model } })
        sendJson(response, 201, { accepted: true, quarantineItemIds: result.items.filter((item) => !item.duplicate).map((item) => item.source.id), generatedCount: result.generated, rejectedCount: result.duplicates, message: `${result.inserted} yeni örnek karantinaya alındı; ${result.duplicates} tekrar ayrıldı.` })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'teacher/feed/campaigns') {
        const campaigns = store.listJobs()
          .filter((job) => job.type === TEACHER_FEED_JOB_TYPE)
          .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)))
          .map(toPublicTeacherFeedCampaign)
        sendJson(response, 200, campaigns)
        return
      }
      if (request.method === 'POST' && route.join('/') === 'teacher/feed/campaigns') {
        const body = validateTeacherFeedHttpRequest(await readJson(request, store.getConfig().budgets.maxRequestBytes))
        const requestHash = teacherFeedHttpRequestHash(body)
        const requestedIdempotencyKey = String(request.headers['idempotency-key'] ?? body.idempotencyKey ?? '').trim().slice(0, 240)
        const repeated = requestedIdempotencyKey ? store.findJobByIdempotencyKey(requestedIdempotencyKey) : null
        if (repeated) {
          assertTeacherFeedIdempotency(repeated, requestHash)
          sendJson(response, 200, toPublicTeacherFeedCampaign(repeated))
          return
        }
        const busy = store.listJobs().find((job) =>
          job.type === TEACHER_FEED_JOB_TYPE &&
          ['queued', 'running', 'pause_requested', 'paused', 'cancel_requested'].includes(job.state),
        )
        if (busy) throw apiError('Önce mevcut 14B besleme kampanyasını tamamlayın, sürdürün veya iptal edin.', 'TEACHER_CAMPAIGN_BUSY', 409)
        const models = await ollama.listModels({ signal: requestSignal(request, response) })
        const modelName = typeof body.model === 'string' ? body.model.trim() : ''
        const modelRecord = models.find((model) => model.name === modelName)
        if (!modelRecord) throw apiError('Seçilen Ollama modeli yerel model listesinde bulunamadı.', 'OLLAMA_MODEL_NOT_FOUND', 409)
        const campaignId = randomUUID()
        const input = createTeacherFeedCampaignInput(body, modelRecord, campaignId)
        const idempotencyKey = requestedIdempotencyKey || input.idempotencyKey
        let created
        try {
          created = queue.createExclusiveTeacherCampaign({ ...input, id: campaignId, idempotencyKey, requestHash })
        } catch (error) {
          if (error?.code === 'JOB_IDEMPOTENCY_CONFLICT') {
            throw apiError(
              'Idempotency anahtarı farklı bir 14B kampanya isteği için kullanılmış.',
              'TEACHER_CAMPAIGN_IDEMPOTENCY_CONFLICT',
              409,
            )
          }
          throw error
        }
        if (!created.duplicate) {
          await queue.emit({
            type: 'teacher-feed-campaign-created',
            jobId: created.job.id,
            data: {
              campaignId: created.job.id,
              model: modelName,
              targetAccepted: input.payload.campaign.targetAccepted,
              continuous: Boolean(input.payload.campaign.continuousPlan),
              researchOnly: true,
            },
          })
        }
        sendJson(response, created.duplicate ? 200 : 202, toPublicTeacherFeedCampaign(created.job))
        return
      }
      if (
        request.method === 'POST' && route.length === 5 && route[0] === 'teacher' && route[1] === 'feed' &&
        route[2] === 'campaigns' && ['pause', 'resume', 'cancel'].includes(route[4])
      ) {
        await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const job = store.getJob(route[3])
        if (!job || job.type !== TEACHER_FEED_JOB_TYPE) throw apiError('14B besleme kampanyası bulunamadı.', 'TEACHER_CAMPAIGN_NOT_FOUND', 404)
        const transition = queue.control(job.id, route[4])
        if (!transition) throw apiError('14B besleme kampanyası kontrol edilemedi.', 'TEACHER_CAMPAIGN_NOT_FOUND', 404)
        if (transition.changed) {
          await queue.emit({
            type: `teacher-feed-campaign-${route[4]}-requested`,
            jobId: job.id,
            data: { campaignId: job.id, previousState: transition.previousState, state: transition.state },
          })
        }
        sendJson(response, 200, toPublicTeacherFeedCampaign(transition.job))
        return
      }
      if (request.method === 'POST' && route.length === 3 && route[0] === 'quarantine' && ['approve', 'reject'].includes(route[2])) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const source = store.decideSource(route[1], route[2] === 'approve' ? 'approved' : 'rejected', body.reason)
        if (!source) throw apiError('Karantina kaydı bulunamadı.', 'NOT_FOUND', 404)
        await queue.emit({ type: `source-${source.status}`, data: { sourceId: source.id } })
        if (source.status === 'approved') maybeAutoSchedule(queue, store)
        sendJson(response, 200, toPublicQuarantine(store.getSource(source.id)))
        return
      }
      if (request.method === 'POST' && route.length === 3 && route[0] === 'sources' && ['approve', 'reject'].includes(route[2])) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const source = store.decideSource(route[1], route[2] === 'approve' ? 'approved' : 'rejected', body.reason)
        if (!source) throw apiError('Kaynak bulunamadı.', 'NOT_FOUND', 404)
        await queue.emit({ type: `source-${source.status}`, data: { sourceId: source.id } })
        if (source.status === 'approved') maybeAutoSchedule(queue, store)
        sendJson(response, 200, toPublicSource(store.getSource(source.id)))
        return
      }

      if (request.method === 'PATCH' && route.length === 2 && route[0] === 'sources') {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        if (typeof body.selected !== 'boolean') throw apiError('selected alanı boolean olmalı.', 'INVALID_SELECTION')
        const source = store.setSourceSelected(route[1], body.selected)
        if (!source) throw apiError('Kaynak bulunamadı.', 'NOT_FOUND', 404)
        if (source.selected) maybeAutoSchedule(queue, store)
        sendJson(response, 200, toPublicSource(store.getSource(source.id)))
        return
      }

      if (request.method === 'GET' && ['queue', 'jobs'].includes(route.join('/'))) {
        if (route.join('/') === 'queue') sendJson(response, 200, store.listJobs().map(toPublicQueueItem))
        else sendJson(response, 200, { ok: true, jobs: store.listJobs() })
        return
      }
      if (request.method === 'GET' && route.length === 2 && route[0] === 'jobs') {
        const job = store.getJob(route[1])
        if (!job) throw apiError('İş bulunamadı.', 'NOT_FOUND', 404)
        // Individual polling is bounded and uses the same redacted projection
        // as the queue surface. Large campaign checkpoints and raw worker
        // payloads never cross the browser boundary.
        sendJson(response, 200, { ok: true, job: toPublicQueueItem(job) })
        return
      }
      if (request.method === 'POST' && route.join('/') === 'jobs') {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        if (body.type === TEACHER_FEED_JOB_TYPE) {
          throw apiError('14B besleme kampanyaları yalnız doğrulanan kampanya uç noktasından başlatılabilir.', 'INVALID_TEACHER_CAMPAIGN', 400)
        }
        if (body.type === PLASTIC_REGION_RELEASE_JOB_TYPE || body.type === PLASTIC_REGION_RENEWAL_JOB_TYPE) {
          throw apiError(
            'Plastik bölge release işleri yalnız öneriyi ve güncel base’i sunucuda donduran değerlendirme uç noktasından başlatılabilir.',
            'INVALID_PLASTIC_REGION_RELEASE_JOB',
            400,
          )
        }
        const created = queue.createJob({ type: body.type, payload: body.payload, idempotencyKey: request.headers['idempotency-key'] ?? body.idempotencyKey })
        sendJson(response, created.duplicate ? 200 : 202, { ok: true, ...created })
        return
      }
      if (request.method === 'POST' && route.join('/') === 'run/start') {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        if (body.config) store.updateConfig(fromPublicConfig(body.config))
        const created = queue.createJob({ type: 'full-cycle', payload: body, idempotencyKey: request.headers['idempotency-key'] ?? body.idempotencyKey })
        const status = toPublicStatus(store, await ollama.check({ signal: requestSignal(request, response) }))
        sendJson(response, created.duplicate ? 200 : 202, { accepted: true, status, message: created.duplicate ? 'Aynı eğitim isteği zaten kuyrukta.' : 'Eğitim partisi kuyruğa alındı.', job: created.job, duplicate: created.duplicate })
        return
      }
      if (request.method === 'POST' && route.length === 2 && route[0] === 'run' && ['pause', 'resume', 'cancel'].includes(route[1])) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const transition = queue.control(body.jobId, route[1])
        if (!transition) throw apiError('Kontrol edilecek iş bulunamadı.', 'NOT_FOUND', 404)
        const status = toPublicStatus(store, await ollama.check({ signal: requestSignal(request, response) }))
        const changedMessage = route[1] === 'pause' ? 'Duraklatma istendi.' : route[1] === 'resume' ? 'Eğitim devam ediyor.' : 'İptal istendi.'
        sendJson(response, 200, {
          accepted: transition.changed,
          status,
          message: transition.changed ? changedMessage : 'İş zaten bu eylemin değiştiremeyeceği bir durumda.',
        })
        return
      }
      if (request.method === 'POST' && route.length === 3 && route[0] === 'jobs' && ['pause', 'resume', 'cancel'].includes(route[2])) {
        const transition = queue.control(route[1], route[2])
        if (!transition) throw apiError('İş bulunamadı.', 'NOT_FOUND', 404)
        sendJson(response, 200, { ok: true, changed: transition.changed, job: transition.job })
        return
      }

      if (request.method === 'GET' && route.join('/') === 'pipeline/state') {
        sendJson(response, 200, { ok: true, pipeline: store.getPipelineState() })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'pipeline/events') {
        const after = Number(url.searchParams.get('after') ?? 0)
        sendJson(response, 200, { ok: true, events: store.listEvents(Number.isFinite(after) ? after : 0) })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'events') {
        response.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache, no-transform', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' })
        response.write(': connected\n\n')
        const lastId = Number(request.headers['last-event-id'] ?? url.searchParams.get('after') ?? 0)
        for (const event of store.listEvents(Number.isFinite(lastId) ? lastId : 0)) {
          const visible = toPublicEvent(event)
          response.write(`id: ${visible.id}\ndata: ${JSON.stringify(visible)}\n\n`)
        }
        sse.add(response)
        return
      }
      if (request.method === 'GET' && route.join('/') === 'metrics') {
        sendJson(response, 200, toPublicMetrics(store))
        return
      }
      if (request.method === 'GET' && route.join('/') === 'thought/status') {
        sendJson(response, 200, { ok: true, ...await thought.status() })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'thought/episodes') {
        sendJson(response, 200, { ok: true, ...thought.list(parseThoughtListQuery(url.searchParams)) })
        return
      }
      if (request.method === 'POST' && route.join('/') === 'thought/episodes') {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const result = await thought.start(body, { idempotencyKey: request.headers['idempotency-key'] })
        sendJson(response, result.duplicate ? 200 : 202, { ok: true, accepted: !result.duplicate, ...result })
        return
      }
      if (request.method === 'GET' && route.length === 3 && route[0] === 'thought' && route[1] === 'episodes') {
        sendJson(response, 200, { ok: true, ...thought.detail(route[2]) })
        return
      }
      if (request.method === 'GET' && route.length === 4 && route[0] === 'thought' && route[1] === 'episodes' && route[3] === 'ticks') {
        sendJson(response, 200, { ok: true, ...thought.ticks(route[2], parseThoughtTickQuery(url.searchParams)) })
        return
      }
      if (request.method === 'GET' && route.length === 4 && route[0] === 'thought' && route[1] === 'episodes' && route[3] === 'graph') {
        const options = parseThoughtGraphQuery(url.searchParams)
        sendJson(response, 200, thought.graph({ ...options, episodeId: route[2] }))
        return
      }
      if (request.method === 'GET' && route.join('/') === 'thought/graph') {
        sendJson(response, 200, thought.graph(parseThoughtGraphQuery(url.searchParams)))
        return
      }
      if (
        request.method === 'POST' && route.length === 4 && route[0] === 'thought' && route[1] === 'episodes' &&
        ['pause', 'resume', 'cancel'].includes(route[3])
      ) {
        sendJson(response, 200, { ok: true, episode: thought[route[3]](route[2]) })
        return
      }
      if (
        request.method === 'POST' && route.length === 4 && route[0] === 'thought' && route[1] === 'episodes' && route[3] === 'feedback'
      ) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        sendJson(response, 200, { ok: true, ...thought.feedback(route[2], body) })
        return
      }
      if (
        request.method === 'POST' && route.length === 4 && route[0] === 'thought' && route[1] === 'episodes' && route[3] === 'replay'
      ) {
        sendJson(response, 200, { ok: true, replay: await thought.replay(route[2], { signal: requestSignal(request, response) }) })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'families/inspection') {
        sendJson(response, 200, inspectFamilies(store, parseFamilyInspectionQuery(url.searchParams)))
        return
      }
      if (request.method === 'GET' && route.join('/') === 'neuron-atlas') {
        const snapshotId = url.searchParams.get('snapshotId') || undefined
        if (snapshotId !== undefined && snapshotId.length > 256) throw apiError('snapshotId uzunluğu geçersiz.', 'INVALID_NEURON_ATLAS_QUERY', 400)
        const atlas = createNeuronAtlas(store, { snapshotId })
        sendJson(response, 200, atlas, { ETag: atlas.fingerprint ? `"${atlas.fingerprint}"` : atlas.run?.fingerprint ? `"${atlas.run.fingerprint}"` : '"empty-neuron-atlas"' })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'concepts/status') {
        sendJson(response, 200, { ok: true, ...inspectConceptMacros(store, { limit: 1 }) })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'concepts/macros') {
        sendJson(response, 200, { ok: true, ...inspectConceptMacros(store, parseConceptInspectionQuery(url.searchParams)) })
        return
      }
      if (request.method === 'GET' && route.length === 3 && route[0] === 'concepts' && route[1] === 'macros') {
        const detail = inspectConceptMacro(store, route[2])
        if (!detail) throw apiError('Kavram/makro adayı bulunamadı.', 'CONCEPT_MACRO_NOT_FOUND', 404)
        sendJson(response, 200, { ok: true, ...detail })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'regions/status') {
        sendJson(response, 200, { ok: true, ...inspectPlasticRegionStatus(store) })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'regions/artifacts') {
        sendJson(response, 200, { ok: true, ...listPlasticRegionRecords(store, parsePlasticRegionListQuery(url.searchParams)) })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'regions/atlas') {
        sendJson(response, 200, { ok: true, ...inspectCurrentPlasticRegionAtlas(store) })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'regions/probes') {
        const atlasFingerprint = url.searchParams.get('atlasFingerprint')?.trim() || null
        if (atlasFingerprint && atlasFingerprint.length > 256) {
          throw apiError('atlasFingerprint uzunluğu geçersiz.', 'INVALID_SEMANTIC_PROBE_INSPECTION', 400)
        }
        const rawLimit = url.searchParams.get('limit')
        if (rawLimit !== null && !/^\d+$/u.test(rawLimit)) {
          throw apiError('Probe limit değeri bir tam sayı olmalı.', 'INVALID_SEMANTIC_PROBE_INSPECTION', 400)
        }
        const maximumItems = rawLimit === null ? 64 : Number(rawLimit)
        if (!Number.isSafeInteger(maximumItems) || maximumItems < 1 || maximumItems > 128) {
          throw apiError('Probe limit değeri 1–128 aralığında olmalı.', 'INVALID_SEMANTIC_PROBE_INSPECTION', 400)
        }
        sendJson(response, 200, {
          ok: true,
          ...inspectSemanticProbeHypotheses(store, { atlasFingerprint, maximumItems }),
        })
        return
      }
      if (request.method === 'GET' && route.length === 3 && route[0] === 'regions' && route[1] === 'atlases') {
        const detail = inspectPlasticRegionAtlas(store, route[2])
        if (!detail) throw apiError('Plastik bölge atlası bulunamadı.', 'PLASTIC_REGION_ATLAS_NOT_FOUND', 404)
        sendJson(response, 200, { ok: true, ...detail })
        return
      }
      if (
        request.method === 'POST' && route.length === 4 && route[0] === 'regions' &&
        route[1] === 'proposals' && route[3] === 'evaluate'
      ) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const allowedKeys = new Set(['idempotencyKey'])
        if (!body || typeof body !== 'object' || Array.isArray(body) ||
            Object.keys(body).some((key) => !allowedKeys.has(key))) {
          throw apiError(
            'Release değerlendirmesi istemciden metrik, holdout veya sham sonucu kabul etmez.',
            'INVALID_PLASTIC_REGION_RELEASE_REQUEST',
            400,
          )
        }
        const prepared = preparePlasticRegionReleaseJob(store, route[2])
        const requestedKey = String(request.headers['idempotency-key'] ?? body.idempotencyKey ?? '').trim()
        if (requestedKey.length > 240) {
          throw apiError('Idempotency anahtarı çok uzun.', 'INVALID_PLASTIC_REGION_RELEASE_REQUEST', 400)
        }
        const created = queue.createJob({
          type: PLASTIC_REGION_RELEASE_JOB_TYPE,
          payload: prepared.payload,
          requestHash: prepared.requestHash,
          idempotencyKey: requestedKey || prepared.defaultIdempotencyKey,
          maxAttempts: 1,
        })
        if (!created.duplicate) {
          await queue.emit({
            type: 'plastic-region-release-queued',
            jobId: created.job.id,
            data: {
              proposalArtifactId: prepared.payload.proposalArtifactId,
              sourceSnapshotId: prepared.payload.sourceSnapshotId,
              baseSnapshotId: prepared.payload.base.snapshotId,
              usesOllama: false,
              automaticActivation: false,
            },
          })
        }
        sendJson(response, created.duplicate ? 200 : 202, {
          ok: true,
          accepted: !created.duplicate,
          duplicate: created.duplicate,
          job: created.job,
          proposalArtifactId: prepared.payload.proposalArtifactId,
          sourceSnapshotId: prepared.payload.sourceSnapshotId,
          baseSnapshotId: prepared.payload.base.snapshotId,
          evaluator: {
            deterministic: true,
            usesOllama: false,
            acceptsClientMetrics: false,
            automaticActivation: false,
          },
        })
        return
      }
      if (
        request.method === 'POST' && route.length === 4 && route[0] === 'regions' &&
        route[1] === 'snapshots' && route[3] === 'renew-release'
      ) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        if (!body || typeof body !== 'object' || Array.isArray(body) ||
            Object.keys(body).some((key) => key !== 'idempotencyKey')) {
          throw apiError(
            'Atlas renewal değerlendirmesi istemciden metrik, benchmark veya model sonucu kabul etmez.',
            'INVALID_PLASTIC_REGION_RENEWAL_REQUEST',
            400,
          )
        }
        const prepared = preparePlasticRegionRenewalJob(store, route[2])
        const requestedKey = String(request.headers['idempotency-key'] ?? body.idempotencyKey ?? '').trim()
        if (requestedKey.length > 240) {
          throw apiError('Idempotency anahtarı çok uzun.', 'INVALID_PLASTIC_REGION_RENEWAL_REQUEST', 400)
        }
        const created = queue.createJob({
          type: PLASTIC_REGION_RENEWAL_JOB_TYPE,
          payload: prepared.payload,
          requestHash: prepared.requestHash,
          idempotencyKey: requestedKey || prepared.defaultIdempotencyKey,
          maxAttempts: 1,
        })
        sendJson(response, created.duplicate ? 200 : 202, {
          ok: true,
          accepted: !created.duplicate,
          duplicate: created.duplicate,
          job: created.job,
          sourceSnapshotId: prepared.payload.sourceSnapshotId,
          baseSnapshotId: prepared.payload.base.snapshotId,
          evaluator: {
            kind: 'unchanged-atlas-model-renewal',
            deterministic: true,
            usesOllama: false,
            acceptsClientMetrics: false,
            automaticActivation: false,
          },
        })
        return
      }

      if (request.method === 'GET' && route.join('/') === 'snapshots') {
        sendJson(response, 200, store.listSnapshots().map((snapshot) => toPublicSnapshot(snapshot, store)))
        return
      }
      if (request.method === 'POST' && route.length === 3 && route[0] === 'snapshots' && route[2] === 'activate') {
        const snapshot = store.getSnapshot(route[1])
        if (!snapshot) throw apiError('Anlık görüntü bulunamadı.', 'NOT_FOUND', 404)
        if (snapshot.evaluation?.passed !== true || snapshot.evaluation?.runtimeVerified !== true) throw apiError('Yalnızca doğrulanmış değerlendirme kapısını geçen aday etkinleştirilebilir.', 'EVALUATION_GATE_REQUIRED', 409)
        assertReviewableSnapshotProducer(store, snapshot)
        const regionActivation = plasticRegionSnapshotActivationGate(store, snapshot)
        if (regionActivation.required && !regionActivation.eligible) {
          throw apiError(
            'Adaptive 8/56 aday, bağımsız holdout/sham kanıtına bağlı plastik bölge release zinciri olmadan etkinleştirilemez.',
            'PLASTIC_REGION_RELEASE_REQUIRED',
            409,
            { reason: regionActivation.reason },
          )
        }
        const activated = store.activateSnapshot(snapshot.id)
        await queue.emit({ type: 'snapshot-activated', data: { snapshotId: snapshot.id } })
        sendJson(response, 200, { ok: true, snapshot: withRollbackToken(activated) })
        return
      }
      if (request.method === 'POST' && ((route.length === 3 && route[0] === 'snapshots' && route[2] === 'rollback') || route.join('/') === 'snapshots/rollback')) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const snapshotId = route.length === 3 ? route[1] : body.snapshotId
        if (!safeTokenEquals(body.confirmToken, rollbackToken(snapshotId))) throw apiError('Geri alma onay anahtarı geçersiz.', 'CONFIRMATION_REQUIRED', 409)
        const rollbackTarget = store.getSnapshot(snapshotId)
        if (rollbackTarget) {
          // rollbackToSnapshot creates a new wrapper snapshot. An adaptive-region
          // release is cryptographically bound to the snapshot that was measured,
          // so copying its artifact ids into that wrapper would make the newly
          // active snapshot release-invalid. Keep rollback fail-closed until a
          // server-owned rebind/release-renewal job has certified that exact child.
          if (snapshotUsesAdaptiveRegionRepresentation(store, rollbackTarget)) {
            throw apiError(
              'Adaptive 8/56 modele geri dönüş, hedef için yeni bir release yenilemesi üretilmeden yapılamaz.',
              'PLASTIC_REGION_ROLLBACK_RENEWAL_REQUIRED',
              409,
              { targetSnapshotId: rollbackTarget.id },
            )
          }
          const regionActivation = plasticRegionSnapshotActivationGate(store, rollbackTarget)
          if (regionActivation.required && !regionActivation.eligible) {
            throw apiError(
              'Adaptive 8/56 geri dönüş hedefi güncel plastik bölge release kapısını geçmiyor.',
              'PLASTIC_REGION_RELEASE_REQUIRED',
              409,
              { reason: regionActivation.reason },
            )
          }
        }
        const restored = store.rollbackToSnapshot(snapshotId)
        if (!restored) throw apiError('Geri alınabilir anlık görüntü bulunamadı.', 'NOT_FOUND', 404)
        await queue.emit({ type: 'snapshot-rolled-back', level: 'warning', data: { targetSnapshotId: snapshotId, activeSnapshotId: restored.id } })
        const status = toPublicStatus(store, await ollama.check({ signal: requestSignal(request, response) }))
        sendJson(response, 200, { accepted: true, status, message: 'Model güvenli sürüme geri alındı.' })
        return
      }
      if (request.method === 'GET' && route.join('/') === 'artifacts') {
        sendJson(response, 200, { ok: true, artifacts: store.listArtifacts() })
        return
      }
      if (request.method === 'GET' && route.length === 2 && route[0] === 'artifacts') {
        const artifact = store.getArtifact(route[1])
        if (!artifact) throw apiError('Çıktı kaydı bulunamadı.', 'NOT_FOUND', 404)
        sendJson(response, 200, { ok: true, artifact })
        return
      }

      if (request.method === 'GET' && ['config', 'settings'].includes(route.join('/'))) {
        sendJson(response, 200, toPublicConfig(store.getConfig()))
        return
      }
      if (request.method === 'GET' && route.join('/') === 'budgets') {
        sendJson(response, 200, { ok: true, budgets: store.getConfig().budgets, usage: store.getUsage() })
        return
      }
      if (request.method === 'PATCH' && ['config', 'settings', 'budgets'].includes(route.join('/'))) {
        const body = await readJson(request, store.getConfig().budgets.maxRequestBytes)
        const patch = route[0] === 'budgets' ? { budgets: body.budgets ?? body } : fromPublicConfig(body.config ?? body)
        const proposedConfig = mergeAndValidateConfig(store.getConfig(), patch)
        if (patch.plasticRegions?.mode === 'active') {
          const activeSnapshot = store.getActiveSnapshot()
          const usesAdaptiveRepresentation = snapshotUsesAdaptiveRegionRepresentation(store, activeSnapshot)
          const release = usesAdaptiveRepresentation && activeSnapshot
            ? readPlasticRegionRelease(store, activeSnapshot, {
              releasePolicy: proposedConfig.plasticRegions,
            })
            : { compatible: false, reason: 'active-snapshot-is-not-adaptive' }
          if (!release.compatible) {
            throw apiError(
              'Active plastik bölge kipi yalnız güncel çalışma ayarlarıyla birebir doğrulanmış etkin bir atlas release’i üzerinde açılabilir.',
              'PLASTIC_REGION_RELEASE_REQUIRED',
              409,
              { reason: release.reason },
            )
          }
        }
        const config = store.updateConfig(patch)
        await queue.emit({ type: 'config-updated', data: { sections: Object.keys(patch) } })
        maybeAutoSchedule(queue, store)
        if (route[0] === 'config') sendJson(response, 200, toPublicStatus(store, await ollama.check({ signal: requestSignal(request, response) })))
        else sendJson(response, 200, { ok: true, config: publicConfig(config) })
        return
      }
      throw apiError('Uç nokta bulunamadı.', 'NOT_FOUND', 404)
    } catch (error) {
      if (response.headersSent) {
        response.end()
        return
      }
      const status = Number(error.status ?? statusForCode(error.code))
      const message = status >= 500 ? 'Yerel eğitim servisi isteği tamamlayamadı.' : String(error.message)
      sendJson(response, status, { ok: false, error: message, message, code: String(error.code ?? 'INTERNAL_ERROR'), details: error.details, requestId })
      if (status >= 500) console.error(`[${requestId}]`, error)
    }
  })
  server.on('connection', (socket) => { sockets.add(socket); socket.on('close', () => sockets.delete(socket)) })

  return {
    store, ollama, queue, watcher, thought, server,
    async listen({ port = 8787, host = '127.0.0.1' } = {}) {
      if (!LOOPBACK_HOSTS.has(host)) throw new Error('Servis güvenlik gereği yalnızca loopback adresine bağlanabilir.')
      await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, host, resolve) })
      queue.start()
      watcher.start()
      return server.address()
    },
    async close() {
      await watcher.stop()
      await thought.close()
      await queue.stop({ abort: true })
      sse.close()
      for (const socket of sockets) socket.destroy()
      if (server.listening) await new Promise((resolve) => server.close(resolve))
      store.close()
    },
  }
}

function applyCors(request, response) {
  const origin = request.headers.origin
  if (origin && !LOCAL_ORIGIN.test(origin)) throw apiError('Bu origin yerel servis API’sine erişemez.', 'ORIGIN_NOT_ALLOWED', 403)
  if (origin) response.setHeader('Access-Control-Allow-Origin', origin)
  response.setHeader('Vary', 'Origin')
  response.setHeader('Access-Control-Allow-Methods', 'GET,POST,PATCH,OPTIONS')
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type,Idempotency-Key,Last-Event-ID')
  response.setHeader('Access-Control-Max-Age', '600')
}

function sanitizeSource(source) { const result = { ...source }; delete result.content; return result }
function sanitizeCreatedSource({ source, duplicate }) { return { source: sanitizeSource(source), duplicate } }
function rollbackToken(id) { return `ROLLBACK:${id}` }
function withRollbackToken(snapshot) { return snapshot ? { ...snapshot, rollbackConfirmToken: rollbackToken(snapshot.id) } : snapshot }
function safeTokenEquals(received, expected) {
  if (typeof received !== 'string') return false
  const left = Buffer.from(received)
  const right = Buffer.from(expected)
  return left.length === right.length && timingSafeEqual(left, right)
}
function statusForCode(code) {
  if (code === 'BUDGET_EXCEEDED') return 429
  if (code === 'DISK_BUDGET_EXCEEDED') return 507
  if (code === 'SOURCE_ALREADY_DECIDED') return 409
  if (code === 'SOURCE_BUSY') return 409
  if (code === 'TEXT_LAB_MANIFEST_CONFLICT') return 409
  if (code === 'SNAPSHOT_NOT_CANDIDATE') return 409
  if (code === 'SNAPSHOT_PARENT_MISMATCH') return 409
  if (code === 'SNAPSHOT_SOURCE_STALE') return 409
  if (code === 'SNAPSHOT_IDEMPOTENCY_CONFLICT') return 409
  if (code === 'ACTIVE_SNAPSHOT_INCONSISTENT') return 409
  if (['THOUGHT_BUSY', 'THOUGHT_STATE_CONFLICT', 'THOUGHT_IDEMPOTENCY_CONFLICT', 'THOUGHT_ALREADY_REVIEWED',
    'THOUGHT_NOT_COMPLETE', 'THOUGHT_COMMIT_NOT_ELIGIBLE', 'STALE_THOUGHT_BASE', 'STALE_THOUGHT_HEAD',
    'THOUGHT_HEAD_PROPOSAL_MISSING', 'THOUGHT_BASE_NOT_FOUND', 'INVALID_THOUGHT_BASE',
    'INVALID_THOUGHT_HEAD', 'INVALID_THOUGHT_HEAD_CHAIN', 'REPRESENTATION_SPACE_MISMATCH'].includes(code)) return 409
  if (code === 'THOUGHT_ENGINE_UNAVAILABLE') return 503
  if (code === 'THOUGHT_ARTIFACT_TOO_LARGE') return 413
  if (code === 'THOUGHT_EPISODE_NOT_FOUND' || code === 'SNAPSHOT_NOT_FOUND') return 404
  if (code === 'NO_APPROVED_SOURCES') return 409
  if (code === 'TEACHER_CAMPAIGN_BUSY' || code === 'STALE_TEACHER_MODEL') return 409
  if (code === 'TEACHER_CAMPAIGN_NOT_FOUND') return 404
  if (code === 'OLLAMA_MODEL_REQUIRED') return 409
  if (code?.startsWith('OLLAMA_')) return 502
  if (['SOURCE_TOO_LARGE', 'REQUEST_TOO_LARGE'].includes(code)) return 413
  if (['PATH_OUTSIDE_ALLOWLIST', 'URL_NOT_ALLOWLISTED', 'PRIVATE_ADDRESS_BLOCKED', 'SOURCE_POLICY_REJECTED'].includes(code)) return 403
  return 400
}
function maybeAutoSchedule(queue, store) {
  const config = store.getConfig()
  if (!config.automation.autoRun) return
  const approved = store.listSources({ status: 'approved' })
  const attemptedRevisions = new Set(
    store.listJobs()
      .filter((job) => job.type === 'full-cycle' && job.state === 'succeeded')
      .flatMap((job) => Object.entries(job.payload?.resolvedSourceRevisions ?? {}))
      .map(([sourceId, revision]) => `${sourceId}\u0000${revision}`),
  )
  const eligibleSourceIds = approved
    .filter((source) =>
      source.selected !== false &&
      source.provenance?.managedByTeacherCampaign !== true &&
      !source.processedSnapshotId &&
      !attemptedRevisions.has(`${source.id}\u0000${sourceRepresentationRevision(source)}`),
    )
    .map((source) => source.id)
    .sort()
  const selectedSourceIds = approved
    .filter((source) => source.selected !== false)
    .map((source) => source.id)
    .sort()
  const busy = store.listJobs().some((job) => ['queued', 'running', 'pause_requested', 'paused'].includes(job.state))
  if (eligibleSourceIds.length >= config.automation.batchSize && !busy) {
    try {
      const corpusKey = sha256(JSON.stringify({ eligibleSourceIds, selectedSourceIds }))
      queue.createJob({
        type: 'full-cycle',
        payload: { trigger: 'approved-source-batch', sourceIds: selectedSourceIds, automationLevel: config.automation.level },
        idempotencyKey: `auto:${corpusKey}`,
      })
    } catch (error) {
      void queue.emit({ type: 'auto-schedule-skipped', level: 'warning', data: { code: error.code ?? 'AUTO_SCHEDULE_FAILED', message: error.message } })
    }
  }
}
