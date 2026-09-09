import { mkdirSync, readFileSync, renameSync, writeFileSync, openSync, fsyncSync, closeSync, existsSync } from 'node:fs'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
import { gzipSync, gunzipSync } from 'node:zlib'
import { defaultConfig, mergeAndValidateConfig } from './config.mjs'
import { teacherAttemptPromptTokenUpperBound } from './teacher-usage.mjs'

const now = () => new Date().toISOString()
const clone = (value) => value === undefined ? undefined : structuredClone(value)
export const ATOMIC_JSON_MAX_LOGICAL_BYTES = 64 * 1024 * 1024
const TEACHER_FEED_JOB_TYPE = 'teacher-feed-campaign'
const ACTIVE_TEACHER_FEED_STATES = new Set(['queued', 'running', 'pause_requested', 'paused', 'cancel_requested'])
const MAX_INDEXED_JOB_QUERY_RESULTS = 4_096

function jobMetadataValue(value) {
  if (!value || typeof value !== 'object' || typeof value.id !== 'string') return null
  return {
    id: value.id,
    state: typeof value.state === 'string' ? value.state : '',
    type: typeof value.type === 'string' ? value.type : '',
    idempotencyKey: typeof value.idempotencyKey === 'string' && value.idempotencyKey
      ? value.idempotencyKey
      : null,
    requestHash: typeof value.requestHash === 'string' && value.requestHash
      ? value.requestHash
      : null,
    createdAt: typeof value.createdAt === 'string' ? value.createdAt : '',
    updatedAt: typeof value.updatedAt === 'string' ? value.updatedAt : '',
  }
}

function normalizeIndexedJobQuery({ states, types, order = 'newest', limit = 1, offset = 0 } = {}) {
  const normalizeValues = (values, label) => {
    if (values === undefined) return []
    if (!Array.isArray(values) || values.length > 32 || values.some((value) => typeof value !== 'string' || !value || value.length > 128)) {
      throw new TypeError(`İş ${label} dizin sorgusu geçersiz.`)
    }
    return [...new Set(values)]
  }
  if (!['oldest', 'newest'].includes(order)) throw new TypeError('İş dizin sıralaması geçersiz.')
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > MAX_INDEXED_JOB_QUERY_RESULTS) {
    throw new TypeError(`İş dizin sorgu sınırı 1..${MAX_INDEXED_JOB_QUERY_RESULTS} olmalı.`)
  }
  if (!Number.isSafeInteger(offset) || offset < 0) throw new TypeError('İş dizin sorgu başlangıcı geçersiz.')
  return {
    states: normalizeValues(states, 'durum'),
    types: normalizeValues(types, 'tür'),
    order,
    limit,
    offset,
  }
}

function canonicalTextLabManifest(value) {
  if (!value || typeof value !== 'object') return null
  const config = Object.fromEntries(Object.entries(value.config ?? {}).sort(([left], [right]) => left.localeCompare(right)))
  const identityProofs = Array.isArray(value.identityProofs)
    ? value.identityProofs
      .map(({ start, end, identityDigest }) => ({ start, end, identityDigest }))
      .sort((left, right) => left.start - right.start || left.end - right.end || String(left.identityDigest).localeCompare(String(right.identityDigest)))
    : []
  // originalSourceId is provenance, not part of the mathematical proof. Two
  // browser projects may legitimately hold the same text and representation.
  return JSON.stringify({
    format: value.format,
    sourceDigest: value.sourceDigest,
    representationSpaceId: value.representationSpaceId,
    config,
    identityProofs,
  })
}

function recordBytes(value) {
  if (value === undefined) return 0
  return Buffer.byteLength(JSON.stringify(value ?? null), 'utf8')
}

function sourceMetadataValue(value) {
  if (!value || typeof value !== 'object') return value
  const metadata = clone(value)
  delete metadata.content
  return metadata
}

function sourceContentSummary(value) {
  const content = String(value ?? '')
  return {
    itemCount: content ? content.split(/(?<=[.!?])\s+/u).filter(Boolean).length : 1,
    preview: content.replace(/\s+/gu, ' ').slice(0, 220),
    characterCount: content.length,
  }
}

function decodeSourceMetadataRow(row, includeSummary = false) {
  if (!row) return undefined
  const metadata = JSON.parse(row.data)
  if (includeSummary) {
    metadata.contentSummary = {
      itemCount: Number(row.item_count),
      preview: row.preview,
      characterCount: Number(row.character_count),
    }
  }
  return metadata
}

const GZIP_JSON_ARTIFACT_PAYLOAD = 'bombom/gzip-json-v1'
// A single gzip decode temporarily needs the compressed buffer, decoded
// buffer, UTF-8 string and parsed object at once. Keep one artifact far below
// the 4 GiB worker envelope; larger logical results use chunk manifests.
export const MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES = 256 * 1024 * 1024

export function assertSingleArtifactPayloadBytes(payloadBytes) {
  if (!Number.isSafeInteger(payloadBytes) || payloadBytes < 0 || payloadBytes > MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES) {
    const error = new Error('Eğitim eseri tek parça güvenli açma sınırını aşıyor; parçalara ayrılarak yazılmalı.')
    error.code = 'ARTIFACT_PAYLOAD_TOO_LARGE'
    error.payloadBytes = payloadBytes
    error.maximumPayloadBytes = MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES
    throw error
  }
  return payloadBytes
}

function artifactPayloadBytes(value) {
  const encodedBytes = Number(value?.payload?.uncompressedBytes)
  if (value?.payload?.$type === GZIP_JSON_ARTIFACT_PAYLOAD && Number.isSafeInteger(encodedBytes) && encodedBytes >= 0) {
    return encodedBytes
  }
  return recordBytes(value?.payload ?? null)
}

function encodeArtifactPayload(payload, minimumBytes) {
  const json = JSON.stringify(payload ?? null)
  const uncompressedBytes = Buffer.byteLength(json, 'utf8')
  // Never persist an artifact that this same store cannot read back. Large
  // logical results must use the existing chunk/checkpoint artifact protocol
  // instead of relying on a highly-compressible single JSON value.
  assertSingleArtifactPayloadBytes(uncompressedBytes)
  if (uncompressedBytes < minimumBytes) return payload
  const compressed = gzipSync(Buffer.from(json, 'utf8'), { level: 6 })
  const data = compressed.toString('base64')
  // Keep already-compressed or high-entropy payloads inline. This avoids both
  // extra CPU and base64 expansion while retaining transparent compatibility.
  if (Buffer.byteLength(data, 'utf8') + 192 >= uncompressedBytes) return payload
  return {
    $type: GZIP_JSON_ARTIFACT_PAYLOAD,
    codec: 'gzip',
    uncompressedBytes,
    compressedBytes: compressed.byteLength,
    sha256: createHash('sha256').update(json).digest('hex'),
    data,
  }
}

function decodeArtifactRecord(value) {
  if (!value || value.payload?.$type !== GZIP_JSON_ARTIFACT_PAYLOAD) return clone(value)
  const envelope = value.payload
  if (
    envelope.codec !== 'gzip' ||
    !Number.isSafeInteger(envelope.uncompressedBytes) ||
    envelope.uncompressedBytes < 0 ||
    envelope.uncompressedBytes > MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES ||
    typeof envelope.sha256 !== 'string' ||
    typeof envelope.data !== 'string'
  ) {
    const error = new Error('Sıkıştırılmış eğitim eseri zarfı geçersiz.')
    error.code = 'ARTIFACT_PAYLOAD_CORRUPT'
    throw error
  }
  let json
  try {
    json = gunzipSync(Buffer.from(envelope.data, 'base64'), {
      maxOutputLength: envelope.uncompressedBytes,
    }).toString('utf8')
  } catch (cause) {
    const error = new Error('Sıkıştırılmış eğitim eseri açılamadı.')
    error.code = 'ARTIFACT_PAYLOAD_CORRUPT'
    error.cause = cause
    throw error
  }
  if (
    Buffer.byteLength(json, 'utf8') !== envelope.uncompressedBytes ||
    createHash('sha256').update(json).digest('hex') !== envelope.sha256
  ) {
    const error = new Error('Sıkıştırılmış eğitim eserinin uzunluk veya SHA-256 kanıtı uyuşmuyor.')
    error.code = 'ARTIFACT_PAYLOAD_CORRUPT'
    throw error
  }
  let payload
  try { payload = JSON.parse(json) } catch (cause) {
    const error = new Error('Sıkıştırılmış eğitim eseri geçerli JSON değil.')
    error.code = 'ARTIFACT_PAYLOAD_CORRUPT'
    error.cause = cause
    throw error
  }
  return { ...clone(value), payload }
}

function boundedUsageInteger(value) {
  const number = Number(value ?? 0)
  return Number.isSafeInteger(number) && number > 0 ? number : 0
}

function finalizeExpiredTeacherAttempt(job) {
  if (job?.type !== 'teacher-feed-campaign') return null
  const state = job.payload?.feedState
  const pending = state?.pendingGenerationBatch
  const attempt = pending?.inFlightAttempt
  if (attempt?.requestUsageReserved !== true) return null
  // Only a persisted v1 num_ctx declaration proves that its smaller bound was
  // actually enforced. Truly legacy attempts remain charged at the historical
  // 32K floor by teacherAttemptPromptTokenUpperBound.
  const persistedPromptTokens = boundedUsageInteger(attempt.maxPromptTokens)
  const promptTokens = teacherAttemptPromptTokenUpperBound(attempt)
  const completionTokens = boundedUsageInteger(attempt.maxCompletionTokens)
  const teacherTokens = promptTokens + completionTokens
  const alreadyReservedTokens = attempt.tokenUsageReserved === true
    ? persistedPromptTokens + completionTokens
    : 0
  pending.promptTokens = boundedUsageInteger(pending.promptTokens) + promptTokens
  pending.completionTokens = boundedUsageInteger(pending.completionTokens) + completionTokens
  pending.generationErrors = boundedUsageInteger(pending.generationErrors) + 1
  pending.usageEstimated = true
  pending.lastError = {
    code: 'OLLAMA_TEMPORARY_ERROR',
    message: 'Süresi dolan öğretmen çağrısı güvenli üst sınırdan muhasebeleştirildi ve tekrar edilmeyecek.',
    timeoutMs: null,
    usageEstimated: true,
  }
  pending.inFlightAttempt = null
  state.promptTokens = boundedUsageInteger(state.promptTokens) + promptTokens
  state.completionTokens = boundedUsageInteger(state.completionTokens) + completionTokens
  state.teacherTokens = boundedUsageInteger(state.teacherTokens) + teacherTokens
  return {
    day: typeof attempt.usageDay === 'string' && /^\d{4}-\d{2}-\d{2}$/u.test(attempt.usageDay)
      ? attempt.usageDay
      : new Date().toISOString().slice(0, 10),
    teacherTokens,
    unreservedTeacherTokens: Math.max(0, teacherTokens - alreadyReservedTokens),
  }
}

const RESUMABLE_TEACHER_RESOURCE_FAILURES = new Set([
  'RAM_BUDGET_EXCEEDED',
  'RUN_TIME_BUDGET_EXCEEDED',
  'LEASE_EXPIRED',
  // A disposable pipeline process can terminate on a transient V8/OS memory
  // failure after the milestone input has already been durably checkpointed.
  // Only explicitly retryable exits pass the guard below.
  'JOB_WORKER_EXITED',
])

function validPendingTeacherMilestone(feedState) {
  const pending = feedState?.pendingTrainingMilestone
  return Boolean(
    pending && typeof pending === 'object' &&
    Number.isInteger(pending.milestoneIndex) && pending.milestoneIndex >= 1 &&
    Array.isArray(pending.sourceIds) && pending.sourceIds.length > 0 &&
    pending.sourceIds.every((id) => typeof id === 'string' && id.length > 0) &&
    typeof pending.candidateSnapshotId === 'string' && pending.candidateSnapshotId.length > 0 &&
    !feedState.pendingGenerationBatch && !feedState.pendingBatchAccounting
  )
}

function canResumeFailedTeacherCampaign(job) {
  const campaign = job?.payload?.campaign
  const feedState = job?.payload?.feedState
  return job?.state === 'failed' &&
    job?.type === 'teacher-feed-campaign' &&
    campaign?.mode === 'research-only' &&
    campaign?.proposalOnly === true &&
    validPendingTeacherMilestone(feedState) &&
    RESUMABLE_TEACHER_RESOURCE_FAILURES.has(job?.error?.code) &&
    (job?.error?.code !== 'JOB_WORKER_EXITED' || job?.error?.retryable === true)
}

function canResumeInterruptedTeacherBenchmark(job, timestamp = Date.now()) {
  const campaign = job?.payload?.campaign
  const feedState = job?.payload?.feedState
  const heartbeatDeadline = Date.parse(job?.leaseUntil ?? '')
  const accepted = Number(feedState?.accepted)
  const lastTrainedAccepted = Number(feedState?.lastTrainedAccepted)
  return job?.state === 'running' &&
    job?.type === 'teacher-feed-campaign' &&
    campaign?.mode === 'research-only' &&
    campaign?.proposalOnly === true &&
    Boolean(campaign?.continuousPlan) &&
    job?.error?.code === 'JOB_WORKER_EXITED' &&
    job?.error?.retryable === true &&
    Number.isFinite(heartbeatDeadline) && heartbeatDeadline <= timestamp &&
    String(job?.stage ?? '').startsWith('teacher-feed:benchmark:') &&
    !feedState?.pendingGenerationBatch &&
    !feedState?.pendingBatchAccounting &&
    !feedState?.pendingTrainingMilestone &&
    Number.isInteger(accepted) && accepted > 0 &&
    Number.isInteger(lastTrainedAccepted) && lastTrainedAccepted === accepted &&
    typeof feedState?.lastSnapshotId === 'string' && feedState.lastSnapshotId.length > 0
}

function canResumeInterruptedTeacherGeneration(job, timestamp = Date.now()) {
  const campaign = job?.payload?.campaign
  const feedState = job?.payload?.feedState
  const pending = feedState?.pendingGenerationBatch
  const heartbeatDeadline = Date.parse(job?.leaseUntil ?? '')
  return job?.state === 'running' &&
    job?.type === 'teacher-feed-campaign' &&
    campaign?.mode === 'research-only' &&
    campaign?.proposalOnly === true &&
    Boolean(campaign?.continuousPlan) &&
    Number.isFinite(heartbeatDeadline) && heartbeatDeadline <= timestamp &&
    job?.stage === 'teacher-feed:generating' &&
    pending?.version === 1 &&
    pending?.inFlightAttempt?.requestUsageReserved === true &&
    !feedState?.pendingBatchAccounting &&
    !feedState?.pendingTrainingMilestone
}

export function sourceRepresentationRevision(source) {
  const manifest = canonicalTextLabManifest(source?.provenance?.textLabManifest) ?? 'raw-representation'
  return createHash('sha256')
    .update(String(source?.contentHash ?? ''))
    .update('\0')
    .update(manifest)
    .digest('hex')
}

class SqliteRecordBackend {
  constructor(DatabaseSync, filename) {
    mkdirSync(path.dirname(filename), { recursive: true, mode: 0o700 })
    this.kind = 'sqlite'
    this.db = new DatabaseSync(filename)
    // Install the lock wait before journal_mode itself may need the database
    // write lock; simultaneous process startup must not fail at the first
    // pragma merely because the later combined block had not reached timeout.
    this.db.exec('PRAGMA busy_timeout=1000;')
    this.db.exec(`
      PRAGMA journal_mode=WAL;
      PRAGMA synchronous=FULL;
      PRAGMA foreign_keys=ON;
      CREATE TABLE IF NOT EXISTS records (
        bucket TEXT NOT NULL,
        id TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (bucket, id)
      );
      CREATE INDEX IF NOT EXISTS records_bucket_updated ON records(bucket, updated_at DESC);
      CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS artifact_metadata (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        content_hash TEXT,
        created_at TEXT NOT NULL,
        job_id TEXT,
        payload_bytes INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS artifact_metadata_created ON artifact_metadata(created_at DESC, id DESC);
      CREATE INDEX IF NOT EXISTS artifact_metadata_kind_created ON artifact_metadata(kind, created_at DESC, id DESC);
      CREATE TABLE IF NOT EXISTS job_metadata (
        id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        type TEXT NOT NULL,
        idempotency_key TEXT,
        request_hash TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS job_metadata_state_created
        ON job_metadata(state, created_at ASC, id ASC);
      CREATE INDEX IF NOT EXISTS job_metadata_type_state_created
        ON job_metadata(type, state, created_at ASC, id ASC);
      CREATE INDEX IF NOT EXISTS job_metadata_idempotency_created
        ON job_metadata(idempotency_key, created_at DESC, id DESC)
        WHERE idempotency_key IS NOT NULL;
      CREATE TABLE IF NOT EXISTS source_metadata (
        id TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        selected INTEGER,
        source_type TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        provenance_kind TEXT,
        provenance_label TEXT,
        managed_by_teacher_campaign INTEGER,
        item_count INTEGER NOT NULL DEFAULT 1,
        preview TEXT NOT NULL DEFAULT '',
        character_count INTEGER NOT NULL DEFAULT 0,
        summary_version INTEGER NOT NULL DEFAULT 0,
        data TEXT NOT NULL
      );
      CREATE UNIQUE INDEX IF NOT EXISTS source_metadata_content_hash ON source_metadata(content_hash);
      CREATE INDEX IF NOT EXISTS source_metadata_status_selected_created
        ON source_metadata(status, selected, created_at DESC);
    `)
    const sourceMetadataColumns = new Set(
      this.db.prepare('PRAGMA table_info(source_metadata)').all().map((column) => column.name),
    )
    for (const [column, declaration] of [
      ['item_count', 'INTEGER NOT NULL DEFAULT 1'],
      ['preview', "TEXT NOT NULL DEFAULT ''"],
      ['character_count', 'INTEGER NOT NULL DEFAULT 0'],
      ['summary_version', 'INTEGER NOT NULL DEFAULT 0'],
    ]) {
      if (!sourceMetadataColumns.has(column)) this.db.exec(`ALTER TABLE source_metadata ADD COLUMN ${column} ${declaration}`)
    }
    this.db.function('bombom_source_item_count', { deterministic: true }, (content) => sourceContentSummary(content).itemCount)
    this.db.function('bombom_source_preview', { deterministic: true }, (content) => sourceContentSummary(content).preview)
    this.db.function('bombom_source_character_count', { deterministic: true }, (content) => sourceContentSummary(content).characterCount)
    this.getStatement = this.db.prepare('SELECT data FROM records WHERE bucket = ? AND id = ?')
    this.listStatement = this.db.prepare('SELECT data FROM records WHERE bucket = ? ORDER BY created_at DESC')
    this.putStatement = this.db.prepare(`
      INSERT INTO records(bucket, id, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(bucket, id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at
    `)
    this.deleteStatement = this.db.prepare('DELETE FROM records WHERE bucket = ? AND id = ?')
    this.metaGetStatement = this.db.prepare('SELECT value FROM metadata WHERE key = ?')
    this.metaSetStatement = this.db.prepare('INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value')
    this.recordBytesStatement = this.db.prepare('SELECT COALESCE(SUM(length(CAST(data AS BLOB))), 0) AS bytes FROM records')
    this.metadataBytesStatement = this.db.prepare("SELECT COALESCE(SUM(length(CAST(value AS BLOB))), 0) AS bytes FROM metadata WHERE key <> 'logicalStorageBytes'")
    this.artifactMetadataGetStatement = this.db.prepare(`
      SELECT id, kind, content_hash, created_at, job_id, payload_bytes
      FROM artifact_metadata WHERE id = ?
    `)
    this.artifactMetadataListStatement = this.db.prepare(`
      SELECT id, kind, content_hash, created_at, job_id, payload_bytes
      FROM artifact_metadata ORDER BY created_at DESC, id DESC
    `)
    this.artifactMetadataListKindStatement = this.db.prepare(`
      SELECT id, kind, content_hash, created_at, job_id, payload_bytes
      FROM artifact_metadata WHERE kind = ? ORDER BY created_at DESC, id DESC LIMIT ?
    `)
    this.artifactMetadataCountKindStatement = this.db.prepare(`
      SELECT COUNT(*) AS count FROM artifact_metadata WHERE kind = ?
    `)
    this.artifactMetadataPutStatement = this.db.prepare(`
      INSERT INTO artifact_metadata(id, kind, content_hash, created_at, job_id, payload_bytes)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        kind=excluded.kind,
        content_hash=excluded.content_hash,
        created_at=excluded.created_at,
        job_id=excluded.job_id,
        payload_bytes=excluded.payload_bytes
    `)
    this.artifactMetadataDeleteStatement = this.db.prepare('DELETE FROM artifact_metadata WHERE id = ?')
    this.jobMetadataFindIdempotencyStatement = this.db.prepare(`
      SELECT id FROM job_metadata
      WHERE idempotency_key = ?
      ORDER BY created_at DESC, id DESC LIMIT 1
    `)
    this.jobMetadataHasQueuedStatement = this.db.prepare("SELECT 1 AS present FROM job_metadata WHERE state = 'queued' LIMIT 1")
    this.jobMetadataNextQueuedStatement = this.db.prepare(`
      SELECT id FROM job_metadata
      WHERE state = 'queued'
      ORDER BY created_at ASC, id ASC LIMIT 1
    `)
    this.jobMetadataPutStatement = this.db.prepare(`
      INSERT INTO job_metadata(id, state, type, idempotency_key, request_hash, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        state=excluded.state,
        type=excluded.type,
        idempotency_key=excluded.idempotency_key,
        request_hash=excluded.request_hash,
        created_at=excluded.created_at,
        updated_at=excluded.updated_at
    `)
    this.jobMetadataDeleteStatement = this.db.prepare('DELETE FROM job_metadata WHERE id = ?')
    this.sourceMetadataGetStatement = this.db.prepare('SELECT data, item_count, preview, character_count FROM source_metadata WHERE id = ?')
    this.sourceMetadataFindHashStatement = this.db.prepare('SELECT data, item_count, preview, character_count FROM source_metadata WHERE content_hash = ? LIMIT 1')
    this.sourceMetadataListStatement = this.db.prepare('SELECT data, item_count, preview, character_count FROM source_metadata ORDER BY created_at DESC')
    this.sourceMetadataListStatusStatement = this.db.prepare('SELECT data, item_count, preview, character_count FROM source_metadata WHERE status = ? ORDER BY created_at DESC')
    this.sourceContentGetStatement = this.db.prepare("SELECT json_extract(data, '$.content') AS content FROM records WHERE bucket = 'sources' AND id = ?")
    this.sourceMetadataPutStatement = this.db.prepare(`
      INSERT INTO source_metadata(
        id, content_hash, status, selected, source_type, bytes, created_at, updated_at,
        provenance_kind, provenance_label, managed_by_teacher_campaign,
        item_count, preview, character_count, summary_version, data
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        content_hash=excluded.content_hash,
        status=excluded.status,
        selected=excluded.selected,
        source_type=excluded.source_type,
        bytes=excluded.bytes,
        created_at=excluded.created_at,
        updated_at=excluded.updated_at,
        provenance_kind=excluded.provenance_kind,
        provenance_label=excluded.provenance_label,
        managed_by_teacher_campaign=excluded.managed_by_teacher_campaign,
        item_count=excluded.item_count,
        preview=excluded.preview,
        character_count=excluded.character_count,
        summary_version=excluded.summary_version,
        data=excluded.data
    `)
    this.sourceMetadataDeleteStatement = this.db.prepare('DELETE FROM source_metadata WHERE id = ?')
    // Older databases stored artifact headers only inside the large JSON row.
    // Backfill once in SQLite itself so JavaScript never has to parse and clone
    // every historical payload merely to build a status page. The count guard
    // makes later startups O(metadata), without re-scanning payload pages.
    const artifactRecordCount = Number(this.db.prepare("SELECT COUNT(*) AS count FROM records WHERE bucket = 'artifacts'").get().count)
    const artifactMetadataCount = Number(this.db.prepare('SELECT COUNT(*) AS count FROM artifact_metadata').get().count)
    if (artifactRecordCount !== artifactMetadataCount) {
      this.db.exec(`
        INSERT OR IGNORE INTO artifact_metadata(id, kind, content_hash, created_at, job_id, payload_bytes)
        SELECT
          id,
          json_extract(data, '$.kind'),
          json_extract(data, '$.contentHash'),
        json_extract(data, '$.createdAt'),
        json_extract(data, '$.jobId'),
        CASE
          WHEN json_extract(data, '$.payload.$type') = 'bombom/gzip-json-v1'
            AND json_type(data, '$.payload.uncompressedBytes') = 'integer'
            THEN json_extract(data, '$.payload.uncompressedBytes')
          WHEN json_type(data, '$.payload') IS NULL OR json_type(data, '$.payload') = 'null' THEN 4
            WHEN json_type(data, '$.payload') = 'true' THEN 4
            WHEN json_type(data, '$.payload') = 'false' THEN 5
            WHEN json_type(data, '$.payload') = 'text' THEN length(CAST(json_quote(json_extract(data, '$.payload')) AS BLOB))
            ELSE length(CAST(json_extract(data, '$.payload') AS BLOB))
          END
        FROM records
        WHERE bucket = 'artifacts' AND json_valid(data) AND json_type(data, '$.kind') = 'text';
      `)
    }
    // Jobs may carry large checkpoints and diagnostics. Keep their scheduling
    // and idempotency fields in a compact side table so polling never parses
    // every historical terminal job. Existing databases are backfilled in SQL.
    const jobRecordCount = Number(this.db.prepare("SELECT COUNT(*) AS count FROM records WHERE bucket = 'jobs'").get().count)
    const jobMetadataCount = Number(this.db.prepare('SELECT COUNT(*) AS count FROM job_metadata').get().count)
    if (jobRecordCount !== jobMetadataCount) {
      this.db.exec(`
        INSERT OR REPLACE INTO job_metadata(
          id, state, type, idempotency_key, request_hash, created_at, updated_at
        )
        SELECT
          id,
          COALESCE(json_extract(data, '$.state'), ''),
          COALESCE(json_extract(data, '$.type'), ''),
          json_extract(data, '$.idempotencyKey'),
          json_extract(data, '$.requestHash'),
          COALESCE(json_extract(data, '$.createdAt'), created_at),
          COALESCE(json_extract(data, '$.updatedAt'), updated_at)
        FROM records
        WHERE bucket = 'jobs' AND json_valid(data) AND json_type(data, '$.id') = 'text';
        DELETE FROM job_metadata
        WHERE id NOT IN (SELECT id FROM records WHERE bucket = 'jobs');
      `)
    }
    // Sources predate the side table and can contain multi-megabyte text. Build
    // the compact projection inside SQLite so migration never parses the full
    // corpus in JavaScript. Normal source writes keep both rows synchronized.
    const sourceRecordCount = Number(this.db.prepare("SELECT COUNT(*) AS count FROM records WHERE bucket = 'sources'").get().count)
    const sourceMetadataCount = Number(this.db.prepare('SELECT COUNT(*) AS count FROM source_metadata').get().count)
    const sourceMetadataIncompleteCount = Number(this.db.prepare('SELECT COUNT(*) AS count FROM source_metadata WHERE summary_version <> 1').get().count)
    if (sourceRecordCount !== sourceMetadataCount || sourceMetadataIncompleteCount > 0) {
      this.db.exec(`
        INSERT OR REPLACE INTO source_metadata(
          id, content_hash, status, selected, source_type, bytes, created_at, updated_at,
          provenance_kind, provenance_label, managed_by_teacher_campaign,
          item_count, preview, character_count, summary_version, data
        )
        SELECT
          id,
          json_extract(data, '$.contentHash'),
          json_extract(data, '$.status'),
          CASE json_type(data, '$.selected')
            WHEN 'true' THEN 1
            WHEN 'false' THEN 0
            ELSE NULL
          END,
          json_extract(data, '$.sourceType'),
          COALESCE(json_extract(data, '$.bytes'), 0),
          json_extract(data, '$.createdAt'),
          json_extract(data, '$.updatedAt'),
          json_extract(data, '$.provenance.kind'),
          json_extract(data, '$.provenance.label'),
          CASE json_type(data, '$.provenance.managedByTeacherCampaign')
            WHEN 'true' THEN 1
            WHEN 'false' THEN 0
            ELSE NULL
          END,
          bombom_source_item_count(json_extract(data, '$.content')),
          bombom_source_preview(json_extract(data, '$.content')),
          bombom_source_character_count(json_extract(data, '$.content')),
          1,
          json_remove(data, '$.content')
        FROM records
        WHERE bucket = 'sources' AND json_valid(data)
          AND json_type(data, '$.contentHash') = 'text'
          AND json_type(data, '$.status') = 'text'
          AND json_type(data, '$.sourceType') = 'text';
        DELETE FROM source_metadata
        WHERE id NOT IN (SELECT id FROM records WHERE bucket = 'sources');
      `)
    }
    this.transactionDepth = 0
  }

  get(bucket, id) {
    const row = this.getStatement.get(bucket, id)
    return row ? JSON.parse(row.data) : undefined
  }

  list(bucket) {
    return this.listStatement.all(bucket).map((row) => JSON.parse(row.data))
  }

  put(bucket, id, value) {
    if (this.transactionDepth === 0 && (bucket === 'artifacts' || bucket === 'sources' || bucket === 'jobs')) {
      return this.transaction(() => this.put(bucket, id, value))
    }
    const timestamp = now()
    this.putStatement.run(bucket, id, JSON.stringify(value), value.createdAt ?? timestamp, timestamp)
    if (bucket === 'artifacts') {
      this.artifactMetadataPutStatement.run(
        id,
        value.kind,
        value.contentHash ?? null,
        value.createdAt ?? timestamp,
        value.jobId ?? null,
        artifactPayloadBytes(value),
      )
    }
    if (bucket === 'sources') {
      const metadata = sourceMetadataValue(value)
      const summary = sourceContentSummary(value.content)
      this.sourceMetadataPutStatement.run(
        metadata.id,
        metadata.contentHash,
        metadata.status,
        typeof metadata.selected === 'boolean' ? Number(metadata.selected) : null,
        metadata.sourceType,
        Number.isSafeInteger(metadata.bytes) && metadata.bytes >= 0 ? metadata.bytes : 0,
        metadata.createdAt ?? timestamp,
        metadata.updatedAt ?? timestamp,
        typeof metadata.provenance?.kind === 'string' ? metadata.provenance.kind : null,
        typeof metadata.provenance?.label === 'string' ? metadata.provenance.label : null,
        typeof metadata.provenance?.managedByTeacherCampaign === 'boolean'
          ? Number(metadata.provenance.managedByTeacherCampaign)
          : null,
        summary.itemCount,
        summary.preview,
        summary.characterCount,
        1,
        JSON.stringify(metadata),
      )
    }
    if (bucket === 'jobs') {
      const metadata = jobMetadataValue(value)
      if (!metadata) throw new TypeError('İş dizin kaydı geçersiz.')
      this.jobMetadataPutStatement.run(
        metadata.id,
        metadata.state,
        metadata.type,
        metadata.idempotencyKey,
        metadata.requestHash,
        metadata.createdAt || timestamp,
        metadata.updatedAt || timestamp,
      )
    }
  }

  delete(bucket, id) {
    if (this.transactionDepth === 0 && (bucket === 'artifacts' || bucket === 'sources' || bucket === 'jobs')) {
      return this.transaction(() => this.delete(bucket, id))
    }
    this.deleteStatement.run(bucket, id)
    if (bucket === 'artifacts') this.artifactMetadataDeleteStatement.run(id)
    if (bucket === 'sources') this.sourceMetadataDeleteStatement.run(id)
    if (bucket === 'jobs') this.jobMetadataDeleteStatement.run(id)
  }
  getArtifactMetadata(id) {
    const row = this.artifactMetadataGetStatement.get(id)
    return row ? {
      id: row.id,
      kind: row.kind,
      contentHash: row.content_hash ?? null,
      createdAt: row.created_at,
      jobId: row.job_id ?? null,
      payloadBytes: Number(row.payload_bytes),
    } : undefined
  }
  listArtifactMetadata() {
    return this.artifactMetadataListStatement.all().map((row) => ({
      id: row.id,
      kind: row.kind,
      contentHash: row.content_hash ?? null,
      createdAt: row.created_at,
      jobId: row.job_id ?? null,
      payloadBytes: Number(row.payload_bytes),
    }))
  }
  listArtifactMetadataByKind(kind, { limit = 512 } = {}) {
    if (typeof kind !== 'string' || !kind || !Number.isSafeInteger(limit) || limit < 1 || limit > 4_096) {
      throw new TypeError('Artifact türü sorgusu geçerli kind ve 1..4096 limit ister.')
    }
    return this.artifactMetadataListKindStatement.all(kind, limit).map((row) => ({
      id: row.id,
      kind: row.kind,
      contentHash: row.content_hash ?? null,
      createdAt: row.created_at,
      jobId: row.job_id ?? null,
      payloadBytes: Number(row.payload_bytes),
    }))
  }
  countArtifactMetadataByKind(kind) {
    if (typeof kind !== 'string' || !kind) throw new TypeError('Artifact türü sayımı geçerli kind ister.')
    return Number(this.artifactMetadataCountKindStatement.get(kind).count)
  }
  findJobByIdempotencyKey(key) {
    const row = this.jobMetadataFindIdempotencyStatement.get(key)
    return row ? this.get('jobs', row.id) : undefined
  }
  hasQueuedJob() {
    return Boolean(this.jobMetadataHasQueuedStatement.get())
  }
  nextQueuedJob() {
    const row = this.jobMetadataNextQueuedStatement.get()
    return row ? this.get('jobs', row.id) : undefined
  }
  queryJobs(query = {}) {
    const { states, types, order, limit, offset } = normalizeIndexedJobQuery(query)
    const clauses = []
    const parameters = []
    if (states.length) {
      clauses.push(`state IN (${states.map(() => '?').join(', ')})`)
      parameters.push(...states)
    }
    if (types.length) {
      clauses.push(`type IN (${types.map(() => '?').join(', ')})`)
      parameters.push(...types)
    }
    const direction = order === 'oldest' ? 'ASC' : 'DESC'
    const rows = this.db.prepare(`
      SELECT id FROM job_metadata
      ${clauses.length ? `WHERE ${clauses.join(' AND ')}` : ''}
      ORDER BY created_at ${direction}, id ${direction}
      LIMIT ? OFFSET ?
    `).all(...parameters, limit, offset)
    return rows.map((row) => this.get('jobs', row.id)).filter(Boolean)
  }
  getSourceMetadata(id, { includeSummary = false } = {}) {
    return decodeSourceMetadataRow(this.sourceMetadataGetStatement.get(id), includeSummary)
  }
  findSourceMetadataByHash(contentHash, { includeSummary = false } = {}) {
    return decodeSourceMetadataRow(this.sourceMetadataFindHashStatement.get(contentHash), includeSummary)
  }
  listSourceMetadata({ status, includeSummary = false } = {}) {
    const rows = status
      ? this.sourceMetadataListStatusStatement.all(status)
      : this.sourceMetadataListStatement.all()
    return rows.map((row) => decodeSourceMetadataRow(row, includeSummary))
  }
  getSourceContent(id) {
    const row = this.sourceContentGetStatement.get(id)
    return typeof row?.content === 'string' ? row.content : undefined
  }
  getMeta(key) {
    const row = this.metaGetStatement.get(key)
    return row ? JSON.parse(row.value) : undefined
  }
  setMeta(key, value) { this.metaSetStatement.run(key, JSON.stringify(value)) }

  calculateLogicalBytes() {
    return Number(this.recordBytesStatement.get().bytes) + Number(this.metadataBytesStatement.get().bytes)
  }

  transaction(fn) {
    if (this.transactionDepth > 0) {
      this.transactionDepth += 1
      try { return fn() } finally { this.transactionDepth -= 1 }
    }
    this.db.exec('BEGIN IMMEDIATE')
    this.transactionDepth = 1
    try {
      const result = fn()
      this.db.exec('COMMIT')
      this.transactionDepth = 0
      return result
    } catch (error) {
      try { this.db.exec('ROLLBACK') } finally { this.transactionDepth = 0 }
      throw error
    }
  }

  close() { this.db.close() }
}

class AtomicJsonRecordBackend {
  constructor(filename) {
    mkdirSync(path.dirname(filename), { recursive: true, mode: 0o700 })
    this.kind = 'atomic-json'
    this.filename = filename
    this.transactionDepth = 0
    this.state = { records: {}, metadata: {} }
    if (existsSync(filename)) {
      const parsed = JSON.parse(readFileSync(filename, 'utf8'))
      if (!parsed || typeof parsed !== 'object' || !parsed.records || !parsed.metadata) throw new Error('Kalıcı servis dosyası bozuk.')
      this.state = parsed
    }
    this.artifactMetadata = new Map()
    this.sourceMetadata = new Map()
    this.sourceHashIndex = new Map()
    this.sourceSummaries = new Map()
    this.jobMetadata = new Map()
    this.jobStateIndex = new Map()
    this.jobTypeIndex = new Map()
    this.jobIdempotencyIndex = new Map()
    this.#rebuildArtifactMetadata()
    this.#rebuildSourceMetadata()
    this.#rebuildJobMetadata()
  }

  #rebuildArtifactMetadata() {
    this.artifactMetadata.clear()
    for (const artifact of Object.values(this.state.records.artifacts ?? {})) {
      this.#rememberArtifactMetadata(artifact)
    }
  }

  #rememberArtifactMetadata(value) {
    if (!value || typeof value.id !== 'string' || typeof value.kind !== 'string') return
    this.artifactMetadata.set(value.id, {
      id: value.id,
      kind: value.kind,
      contentHash: value.contentHash ?? null,
      createdAt: value.createdAt,
      jobId: value.jobId ?? null,
      payloadBytes: artifactPayloadBytes(value),
    })
  }

  #rebuildSourceMetadata() {
    this.sourceMetadata.clear()
    this.sourceHashIndex.clear()
    this.sourceSummaries.clear()
    for (const source of Object.values(this.state.records.sources ?? {})) {
      this.#rememberSourceMetadata(source)
    }
  }

  #rememberSourceMetadata(value) {
    if (!value || typeof value.id !== 'string' || typeof value.contentHash !== 'string') return
    const previous = this.sourceMetadata.get(value.id)
    if (previous && this.sourceHashIndex.get(previous.contentHash) === value.id) {
      this.sourceHashIndex.delete(previous.contentHash)
    }
    const metadata = sourceMetadataValue(value)
    this.sourceMetadata.set(value.id, metadata)
    this.sourceHashIndex.set(metadata.contentHash, metadata.id)
    this.sourceSummaries.set(metadata.id, sourceContentSummary(value.content))
  }

  #rebuildJobMetadata() {
    this.jobMetadata.clear()
    this.jobStateIndex.clear()
    this.jobTypeIndex.clear()
    this.jobIdempotencyIndex.clear()
    const jobs = Object.values(this.state.records.jobs ?? {})
      .sort((left, right) => String(left.createdAt).localeCompare(String(right.createdAt)) || String(left.id).localeCompare(String(right.id)))
    for (const job of jobs) this.#rememberJobMetadata(job)
  }

  #addJobIndex(index, key, id) {
    if (!key) return
    let ids = index.get(key)
    if (!ids) index.set(key, ids = new Set())
    ids.add(id)
  }

  #removeJobIndex(index, key, id) {
    const ids = index.get(key)
    if (!ids) return
    ids.delete(id)
    if (!ids.size) index.delete(key)
  }

  #newestJobIdForIdempotencyKey(key) {
    let newest = null
    for (const metadata of this.jobMetadata.values()) {
      if (metadata.idempotencyKey !== key) continue
      if (!newest || String(metadata.createdAt).localeCompare(String(newest.createdAt)) > 0 || (
        metadata.createdAt === newest.createdAt && String(metadata.id).localeCompare(String(newest.id)) > 0
      )) newest = metadata
    }
    return newest?.id ?? null
  }

  #forgetJobMetadata(id) {
    const previous = this.jobMetadata.get(id)
    if (!previous) return
    this.jobMetadata.delete(id)
    this.#removeJobIndex(this.jobStateIndex, previous.state, id)
    this.#removeJobIndex(this.jobTypeIndex, previous.type, id)
    if (previous.idempotencyKey && this.jobIdempotencyIndex.get(previous.idempotencyKey) === id) {
      this.jobIdempotencyIndex.delete(previous.idempotencyKey)
      const replacement = this.#newestJobIdForIdempotencyKey(previous.idempotencyKey)
      if (replacement) this.jobIdempotencyIndex.set(previous.idempotencyKey, replacement)
    }
  }

  #rememberJobMetadata(value) {
    const metadata = jobMetadataValue(value)
    if (!metadata) return
    this.#forgetJobMetadata(metadata.id)
    this.jobMetadata.set(metadata.id, metadata)
    this.#addJobIndex(this.jobStateIndex, metadata.state, metadata.id)
    this.#addJobIndex(this.jobTypeIndex, metadata.type, metadata.id)
    if (metadata.idempotencyKey) {
      const current = this.jobMetadata.get(this.jobIdempotencyIndex.get(metadata.idempotencyKey))
      if (!current || String(metadata.createdAt).localeCompare(String(current.createdAt)) > 0 || (
        metadata.createdAt === current.createdAt && String(metadata.id).localeCompare(String(current.id)) > 0
      )) this.jobIdempotencyIndex.set(metadata.idempotencyKey, metadata.id)
    }
  }

  bucket(name) { return this.state.records[name] ??= {} }
  get(bucket, id) { return clone(this.bucket(bucket)[id]) }
  list(bucket) { return Object.values(this.bucket(bucket)).map(clone).sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt))) }
  put(bucket, id, value) {
    this.bucket(bucket)[id] = clone(value)
    if (bucket === 'artifacts') this.#rememberArtifactMetadata(value)
    if (bucket === 'sources') this.#rememberSourceMetadata(value)
    if (bucket === 'jobs') this.#rememberJobMetadata(value)
    if (this.transactionDepth === 0) this.flush()
  }
  delete(bucket, id) {
    delete this.bucket(bucket)[id]
    if (bucket === 'artifacts') this.artifactMetadata.delete(id)
    if (bucket === 'sources') {
      const metadata = this.sourceMetadata.get(id)
      this.sourceMetadata.delete(id)
      this.sourceSummaries.delete(id)
      if (metadata && this.sourceHashIndex.get(metadata.contentHash) === id) {
        this.sourceHashIndex.delete(metadata.contentHash)
      }
    }
    if (bucket === 'jobs') this.#forgetJobMetadata(id)
    if (this.transactionDepth === 0) this.flush()
  }
  getArtifactMetadata(id) { return clone(this.artifactMetadata.get(id)) }
  listArtifactMetadata() {
    return [...this.artifactMetadata.values()]
      .map(clone)
      .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))
  }
  listArtifactMetadataByKind(kind, { limit = 512 } = {}) {
    if (typeof kind !== 'string' || !kind || !Number.isSafeInteger(limit) || limit < 1 || limit > 4_096) {
      throw new TypeError('Artifact türü sorgusu geçerli kind ve 1..4096 limit ister.')
    }
    return [...this.artifactMetadata.values()]
      .filter((metadata) => metadata.kind === kind)
      .map(clone)
      .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))
      .slice(0, limit)
  }
  countArtifactMetadataByKind(kind) {
    if (typeof kind !== 'string' || !kind) throw new TypeError('Artifact türü sayımı geçerli kind ister.')
    let count = 0
    for (const metadata of this.artifactMetadata.values()) if (metadata.kind === kind) count += 1
    return count
  }
  findJobByIdempotencyKey(key) {
    const id = this.jobIdempotencyIndex.get(key)
    return id ? this.get('jobs', id) : undefined
  }
  hasQueuedJob() {
    return Boolean(this.jobStateIndex.get('queued')?.size)
  }
  nextQueuedJob() {
    return this.queryJobs({ states: ['queued'], order: 'oldest', limit: 1 })[0]
  }
  queryJobs(query = {}) {
    const { states, types, order, limit, offset } = normalizeIndexedJobQuery(query)
    const union = (index, keys) => {
      const ids = new Set()
      for (const key of keys) for (const id of index.get(key) ?? []) ids.add(id)
      return ids
    }
    const stateIds = states.length ? union(this.jobStateIndex, states) : null
    const typeIds = types.length ? union(this.jobTypeIndex, types) : null
    let candidateIds
    if (stateIds && typeIds) candidateIds = stateIds.size <= typeIds.size ? stateIds : typeIds
    else candidateIds = stateIds ?? typeIds ?? this.jobMetadata.keys()
    const direction = order === 'oldest' ? 1 : -1
    return [...candidateIds]
      .map((id) => this.jobMetadata.get(id))
      .filter((metadata) => metadata && (!states.length || states.includes(metadata.state)) && (!types.length || types.includes(metadata.type)))
      .sort((left, right) => direction * (
        String(left.createdAt).localeCompare(String(right.createdAt)) || String(left.id).localeCompare(String(right.id))
      ))
      .slice(offset, offset + limit)
      .map((metadata) => this.get('jobs', metadata.id))
      .filter(Boolean)
  }
  getSourceMetadata(id, { includeSummary = false } = {}) {
    const metadata = clone(this.sourceMetadata.get(id))
    if (metadata && includeSummary) metadata.contentSummary = clone(this.sourceSummaries.get(id))
    return metadata
  }
  findSourceMetadataByHash(contentHash, { includeSummary = false } = {}) {
    const id = this.sourceHashIndex.get(contentHash)
    return id ? this.getSourceMetadata(id, { includeSummary }) : undefined
  }
  listSourceMetadata({ status, includeSummary = false } = {}) {
    return [...this.sourceMetadata.values()]
      .filter((source) => !status || source.status === status)
      .map((source) => this.getSourceMetadata(source.id, { includeSummary }))
      .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)))
  }
  getSourceContent(id) {
    const content = this.bucket('sources')[id]?.content
    return typeof content === 'string' ? content : undefined
  }
  getMeta(key) { return clone(this.state.metadata[key]) }
  setMeta(key, value) { this.state.metadata[key] = clone(value); if (this.transactionDepth === 0) this.flush() }
  calculateLogicalBytes() {
    let total = 0
    for (const bucket of Object.values(this.state.records)) {
      for (const record of Object.values(bucket)) total += recordBytes(record)
    }
    for (const [key, value] of Object.entries(this.state.metadata)) {
      if (key !== 'logicalStorageBytes') total += recordBytes(value)
    }
    return total
  }
  transaction(fn) {
    if (this.transactionDepth > 0) {
      this.transactionDepth += 1
      try { return fn() } finally { this.transactionDepth -= 1 }
    }
    const before = structuredClone(this.state)
    this.transactionDepth = 1
    try {
      const result = fn()
      this.flush()
      this.transactionDepth = 0
      return result
    } catch (error) {
      this.state = before
      this.#rebuildArtifactMetadata()
      this.#rebuildSourceMetadata()
      this.#rebuildJobMetadata()
      try {
        this.flush()
      } catch (recoveryError) {
        if (error && typeof error === 'object') error.recoveryError = recoveryError
      } finally {
        this.transactionDepth = 0
      }
      throw error
    }
  }
  flush() {
    const temporary = `${this.filename}.${process.pid}.tmp`
    writeFileSync(temporary, JSON.stringify(this.state), { encoding: 'utf8', mode: 0o600 })
    const descriptor = openSync(temporary, 'r')
    try { fsyncSync(descriptor) } finally { closeSync(descriptor) }
    renameSync(temporary, this.filename)
  }
  close() {}
}

async function createBackend(dataDirectory, options = {}) {
  if (!options.forceJson) {
    try {
      const { DatabaseSync } = await import('node:sqlite')
      return new SqliteRecordBackend(DatabaseSync, path.join(dataDirectory, 'autonomous-training.sqlite'))
    } catch (error) {
      if (options.requireSqlite) throw error
    }
  }
  return new AtomicJsonRecordBackend(path.join(dataDirectory, 'autonomous-training.json'))
}

export class TrainingStore {
  constructor(backend, dataDirectory) {
    this.backend = backend
    this.dataDirectory = dataDirectory
    const storedConfig = this.backend.getMeta('config')
    if (!storedConfig) {
      this.backend.setMeta('config', defaultConfig(dataDirectory))
    } else if (
      storedConfig.version !== defaultConfig(dataDirectory).version ||
      !storedConfig.thought ||
      Object.keys(defaultConfig(dataDirectory).thought).some((key) => !Object.hasOwn(storedConfig.thought, key)) ||
      !storedConfig.conceptMacros ||
      Object.keys(defaultConfig(dataDirectory).conceptMacros).some((key) => !Object.hasOwn(storedConfig.conceptMacros, key)) ||
      !storedConfig.plasticRegions ||
      Object.keys(defaultConfig(dataDirectory).plasticRegions).some((key) => !Object.hasOwn(storedConfig.plasticRegions, key)) ||
      !storedConfig.resourcePolicy
    ) {
      const defaults = defaultConfig(dataDirectory)
      const storedThought = { ...(storedConfig.thought ?? {}) }
      if ((storedConfig.version ?? 1) < 3 && storedThought.maximumEpisodeBytes === 4 * 1024 * 1024) {
        storedThought.maximumEpisodeBytes = defaults.thought.maximumEpisodeBytes
      }
      const upgradingToResourcePolicy = (storedConfig.version ?? 1) < 4
      this.backend.setMeta('config', {
        ...defaults,
        ...storedConfig,
        version: defaults.version,
        dataDirectory,
        automation: { ...defaults.automation, ...(storedConfig.automation ?? {}) },
        thought: { ...defaults.thought, ...storedThought },
        conceptMacros: { ...defaults.conceptMacros, ...(storedConfig.conceptMacros ?? {}) },
        plasticRegions: { ...defaults.plasticRegions, ...(storedConfig.plasticRegions ?? {}) },
        budgets: {
          ...defaults.budgets,
          ...(storedConfig.budgets ?? {}),
          // v4 deliberately replaces the old workstation-wide 8–20 GiB
          // allowance. Existing data is preserved; only future admission is
          // capped to a 4 GiB process/store envelope on migration.
          ...(upgradingToResourcePolicy ? {
            ramMb: Math.min(Number(storedConfig.budgets?.ramMb ?? defaults.budgets.ramMb), defaults.budgets.ramMb),
            diskMb: Math.min(Number(storedConfig.budgets?.diskMb ?? defaults.budgets.diskMb), defaults.budgets.diskMb),
          } : {}),
        },
        resourcePolicy: { ...defaults.resourcePolicy, ...(storedConfig.resourcePolicy ?? {}) },
        sourcePolicy: { ...defaults.sourcePolicy, ...(storedConfig.sourcePolicy ?? {}) },
        evaluation: { ...defaults.evaluation, ...(storedConfig.evaluation ?? {}) },
      })
    }
    if (!this.backend.getMeta('pipeline')) {
      this.backend.setMeta('pipeline', { phase: 'idle', activeJobId: null, paused: false, updatedAt: now(), message: 'Hazır' })
    }
    // Reconcile once at startup (SQL computes byte lengths without parsing
    // payloads); normal writes maintain this counter transactionally in O(1).
    this.backend.setMeta('logicalStorageBytes', this.backend.calculateLogicalBytes())
    // A process can stop after an older teacher-feed version reserved a small
    // heuristic prompt bound and released the job. Queued, paused and terminal
    // rows have no runnable owner, so close that ambiguous call before any new
    // reservation can spend the apparent gap. Running/requested rows remain
    // owned and are deliberately left to their claim-fenced worker/recovery
    // path.
    this.reconcileUnownedTeacherUsage()
  }

  get backendKind() { return this.backend.kind }
  close() { this.backend.close() }
  getConfig() { return clone(this.backend.getMeta('config')) }
  updateConfig(patch) {
    const current = this.getConfig()
    const next = mergeAndValidateConfig(current, patch)
    this.#setMetaWithinBudget('config', next, current, next)
    return clone(next)
  }
  getPipelineState() { return clone(this.backend.getMeta('pipeline')) }
  setPipelineState(patch) {
    const current = this.getPipelineState()
    const next = { ...current, ...clone(patch), updatedAt: now() }
    this.#setMetaWithinBudget('pipeline', next, current)
    return clone(next)
  }

  findSourceByHash(contentHash) {
    const metadata = this.backend.findSourceMetadataByHash(contentHash)
    return metadata ? this.backend.get('sources', metadata.id) : undefined
  }
  createSource(input) {
    const existing = this.findSourceByHash(input.contentHash)
    if (existing) {
      // Exact-content deduplication must not throw away a later, stronger
      // reproducibility proof imported from TextLab.
      const incomingManifest = input.provenance?.textLabManifest
      const existingManifest = existing.provenance?.textLabManifest
      if (incomingManifest && existingManifest && canonicalTextLabManifest(incomingManifest) !== canonicalTextLabManifest(existingManifest)) {
        const error = new Error('Aynı metin için farklı bir Metin Laboratuvarı temsil kanıtı zaten kayıtlı.')
        error.code = 'TEXT_LAB_MANIFEST_CONFLICT'
        throw error
      }
      if (incomingManifest && !existingManifest) {
        const busy = this.backend.list('jobs').some((job) => {
          if (!['queued', 'running', 'pause_requested', 'paused', 'cancel_requested'].includes(job.state)) return false
          const requested = Array.isArray(job.payload?.sourceIds)
            ? job.payload.sourceIds
            : Array.isArray(job.payload?.resolvedSourceIds)
              ? job.payload.resolvedSourceIds
              : null
          // Before an implicit job freezes its exact corpus, every approved
          // source may still be collected. Selection changes must not open a
          // representation-upgrade race against that in-flight decision.
          return requested ? requested.includes(existing.id) : existing.status === 'approved'
        })
        if (busy) {
          const error = new Error('Kaynak etkin bir eğitim işindeyken Metin Laboratuvarı kanıtı yükseltilemez.')
          error.code = 'SOURCE_BUSY'
          throw error
        }
        const previousProvenance = existing.provenance ?? {}
        const nextProvenance = {
          ...previousProvenance,
          kind: 'text-lab-archive',
          previousKind: previousProvenance.kind ?? null,
          label: input.provenance?.label ?? previousProvenance.label,
          textLabManifest: clone(incomingManifest),
        }
        this.assertDiskBudget(Math.max(0, recordBytes(nextProvenance) - recordBytes(previousProvenance)) + 256)
        existing.provenance = nextProvenance
        // The content is unchanged, but the representation configuration is
        // now stronger and may differ from the snapshot that processed it. A
        // human must therefore approve this materially changed input again.
        existing.status = 'quarantine'
        existing.selected = false
        existing.decidedAt = null
        existing.decisionReason = null
        existing.processedSnapshotId = null
        existing.updatedAt = now()
        this.backend.transaction(() => {
          this.#putWithinBudget('sources', existing.id, existing)
          this.#invalidateCandidateSnapshotsForSource(existing.id)
        })
      }
      return { source: clone(existing), duplicate: true }
    }
    this.assertDiskBudget(input.bytes + recordBytes(input.provenance) + 2_048)
    const timestamp = now()
    const source = {
      id: input.id ?? randomUUID(),
      title: input.title,
      sourceType: input.sourceType,
      locator: input.locator ?? null,
      content: input.content,
      contentHash: input.contentHash,
      bytes: input.bytes,
      language: input.language ?? 'tr',
      status: 'quarantine',
      provenance: clone(input.provenance ?? {}),
      createdAt: timestamp,
      updatedAt: timestamp,
      decidedAt: null,
      decisionReason: null,
      processedSnapshotId: null,
      selected: false,
    }
    this.#putWithinBudget('sources', source.id, source)
    return { source: clone(source), duplicate: false }
  }
  createSources(inputs) {
    if (!Array.isArray(inputs)) throw new TypeError('Kaynak partisi bir dizi olmalı.')
    return this.backend.transaction(() => inputs.map((input) => this.createSource(input)))
  }
  /**
   * Persist one synthetic-teacher batch and its durable cursor as one unit.
   * A crash can therefore happen either before the batch exists or after both
   * the sources and cursor exist, never in between.
   */
  commitTeacherFeedBatch(jobId, {
    inputs,
    checkpoint,
    artifact,
    usageReservations = [],
    usageSettlements = [],
    claim,
    allowRequestedUsageFinalization = false,
  }) {
    if (!Array.isArray(inputs)) throw new TypeError('Öğretmen besleme girdileri bir dizi olmalı.')
    if (!checkpoint || typeof checkpoint !== 'object' || Array.isArray(checkpoint)) {
      throw new TypeError('Öğretmen besleme checkpoint kaydı geçersiz.')
    }
    if (!Array.isArray(usageReservations)) throw new TypeError('Öğretmen besleme kullanım rezervasyonları bir dizi olmalı.')
    if (!Array.isArray(usageSettlements)) throw new TypeError('Öğretmen besleme kullanım kapatmaları bir dizi olmalı.')
    if (claim !== undefined && (
      !claim || typeof claim !== 'object' || Array.isArray(claim) ||
      typeof claim.owner !== 'string' || !claim.owner ||
      !Number.isInteger(claim.attempt) || claim.attempt < 1
    )) throw new TypeError('Öğretmen besleme worker claim kaydı geçersiz.')
    if (allowRequestedUsageFinalization && (
      !claim || inputs.length !== 0 ||
      artifact?.kind !== 'teacher-feed-checkpoint' ||
      usageReservations.some((reservation) => reservation?.kind !== 'teacherTokens') ||
      usageSettlements.some((settlement) => settlement?.kind !== 'teacherTokens')
    )) throw new TypeError('Durdurulmuş öğretmen çağrısı yalnız token muhasebesini ve checkpoint bilgisini kapatabilir.')
    const reservations = usageReservations.map((reservation) => {
      if (!reservation || typeof reservation !== 'object' || Array.isArray(reservation)) {
        throw new TypeError('Öğretmen besleme kullanım rezervasyonu geçersiz.')
      }
      const kind = String(reservation.kind ?? '')
      const amount = Number(reservation.amount ?? 0)
      const day = reservation.day === undefined ? undefined : String(reservation.day)
      if (!Number.isSafeInteger(amount) || amount <= 0) {
        throw new TypeError('Öğretmen besleme kullanım rezervasyonu pozitif bir tam sayı olmalı.')
      }
      if (day !== undefined && !/^\d{4}-\d{2}-\d{2}$/u.test(day)) {
        throw new TypeError('Öğretmen besleme kullanım rezervasyonu günü geçersiz.')
      }
      return { kind, amount, day }
    })
    const settlements = usageSettlements.map((settlement) => {
      if (!settlement || typeof settlement !== 'object' || Array.isArray(settlement)) {
        throw new TypeError('Öğretmen besleme kullanım kapatması geçersiz.')
      }
      const kind = String(settlement.kind ?? '')
      const reservedAmount = Number(settlement.reservedAmount ?? 0)
      const actualAmount = Number(settlement.actualAmount ?? 0)
      const day = settlement.day === undefined ? new Date().toISOString().slice(0, 10) : String(settlement.day)
      if (!Number.isSafeInteger(reservedAmount) || reservedAmount <= 0 || !Number.isSafeInteger(actualAmount) || actualAmount < 0) {
        throw new TypeError('Öğretmen besleme kullanım kapatması geçerli tam sayı miktarları taşımalı.')
      }
      if (!/^\d{4}-\d{2}-\d{2}$/u.test(day)) {
        throw new TypeError('Öğretmen besleme kullanım kapatma günü geçersiz.')
      }
      return { kind, reservedAmount, actualAmount, day }
    })
    return this.backend.transaction(() => {
      if (claim) {
        if (allowRequestedUsageFinalization) this.assertJobClaim(jobId, claim.owner, claim.attempt)
        else this.assertRunnableJobClaim(jobId, claim.owner, claim.attempt)
      }
      const job = this.backend.get('jobs', jobId)
      if (!job || job.type !== 'teacher-feed-campaign') {
        const error = new Error('Öğretmen besleme kampanyası bulunamadı.')
        error.code = 'TEACHER_CAMPAIGN_NOT_FOUND'
        throw error
      }
      for (const reservation of reservations) {
        this.consumeUsage(reservation.kind, reservation.amount, reservation.day)
      }
      for (const settlement of settlements) {
        const usage = this.backend.get('usage', settlement.day) ?? {
          day: settlement.day,
          jobs: 0,
          teacherRequests: 0,
          teacherTokens: 0,
        }
        if (!(settlement.kind in usage) || boundedUsageInteger(usage[settlement.kind]) < settlement.reservedAmount) {
          const error = new Error('Öğretmen token rezervasyonu bulunamadı veya daha önce kapatıldı.')
          error.code = 'TEACHER_USAGE_RESERVATION_MISSING'
          throw error
        }
        usage[settlement.kind] = boundedUsageInteger(usage[settlement.kind]) - settlement.reservedAmount + settlement.actualAmount
        this.#putWithinBudget('usage', settlement.day, usage)
      }
      const items = []
      for (const input of inputs) {
        if (input?.sourceType !== 'ollama-generated' || input?.provenance?.evaluationEligible !== false) {
          const error = new Error('Besleme kampanyası yalnız değerlendirme dışı Ollama kaynakları yazabilir.')
          error.code = 'INVALID_TEACHER_SOURCE'
          throw error
        }
        const created = this.createSource(input)
        if (!created.duplicate) {
          const source = this.backend.get('sources', created.source.id)
          source.status = 'approved'
          source.selected = true
          source.decisionReason = 'Kullanıcının başlattığı araştırma kampanyasında kaynağa kaydedilen dondurulmuş öğretmen protokolü kapısını geçti.'
          source.decidedAt = now()
          source.updatedAt = source.decidedAt
          this.#putWithinBudget('sources', source.id, source)
          created.source = clone(source)
        }
        items.push(created)
      }
      const batchArtifact = artifact
        ? this.createArtifact({ ...clone(artifact), jobId })
        : null
      job.payload = {
        ...(job.payload ?? {}),
        feedState: clone(checkpoint),
      }
      job.updatedAt = now()
      this.#putWithinBudget('jobs', job.id, job)
      return { items: clone(items), artifact: clone(batchArtifact), job: clone(job) }
    })
  }
  getSource(id) { return clone(this.backend.get('sources', id)) }
  getSourceContent(id) { return this.backend.getSourceContent(id) }
  listSources({ status, includeContent = false, includeSummary = false } = {}) {
    if (!includeContent) return clone(this.backend.listSourceMetadata({ status, includeSummary }))
    return this.backend.list('sources')
      .filter((source) => !status || source.status === status)
  }
  decideSource(id, status, reason = '') {
    if (!['approved', 'rejected'].includes(status)) throw new TypeError('Geçersiz kaynak kararı.')
    const source = this.backend.get('sources', id)
    if (!source) return undefined
    if (source.status !== 'quarantine' && source.status !== status) {
      const error = new Error('Yalnızca karantinadaki kaynaklar yeniden karara bağlanabilir.')
      error.code = 'SOURCE_ALREADY_DECIDED'
      throw error
    }
    Object.assign(source, { status, selected: status === 'approved', decisionReason: String(reason).slice(0, 1_000), decidedAt: now(), updatedAt: now() })
    this.#putWithinBudget('sources', id, source)
    return clone(source)
  }
  setSourceSelected(id, selected) {
    return this.backend.transaction(() => {
      const source = this.backend.get('sources', id)
      if (!source) return undefined
      if (source.status !== 'approved' && selected) {
        const error = new Error('Yalnızca onaylı kaynaklar eğitime seçilebilir.')
        error.code = 'SOURCE_NOT_APPROVED'
        throw error
      }
      source.selected = Boolean(selected)
      source.updatedAt = now()
      this.#putWithinBudget('sources', id, source)
      if (source.selected === false) {
        this.#invalidateCandidateSnapshotsForSource(id, {
          code: 'SOURCE_SELECTION_CHANGED',
          selectionChange: 'DESELECTED',
          reason: 'Kaynak aday model oluşturulduktan sonra eğitim seçiminden çıkarıldı.',
        })
      }
      return clone(source)
    })
  }
  markSourcesProcessed(ids, snapshotId) {
    this.backend.transaction(() => {
      for (const id of ids) {
        const source = this.backend.get('sources', id)
        if (source) {
          source.processedSnapshotId = snapshotId
          source.updatedAt = now()
          this.#putWithinBudget('sources', id, source)
        }
      }
    })
  }
  #invalidateCandidateSnapshotsForSource(sourceId, {
    code = 'SOURCE_REPRESENTATION_CHANGED',
    selectionChange,
    reason = 'Kaynağın Metin Laboratuvarı temsil kanıtı aday değerlendirildikten sonra değişti.',
  } = {}) {
    for (const snapshot of this.backend.list('snapshots')) {
      if (snapshot.status !== 'candidate' || !snapshot.sourceIds?.includes(sourceId)) continue
      snapshot.status = 'rejected'
      snapshot.rejectedAt = now()
      snapshot.evaluation = {
        ...(snapshot.evaluation ?? {}),
        passed: false,
        invalidated: true,
        code,
        ...(selectionChange ? { selectionChange } : {}),
        reason,
      }
      this.#putWithinBudget('snapshots', snapshot.id, snapshot)
    }
  }

  createJob(input) {
    return this.backend.transaction(() => {
      const requestedType = input.type ?? 'full-cycle'
      const requestedHash = input.requestHash ?? null
      if (requestedHash !== null && (typeof requestedHash !== 'string' || !requestedHash || requestedHash.length > 256)) {
        const error = new TypeError('İş isteği özeti geçersiz.')
        error.code = 'INVALID_JOB_REQUEST_HASH'
        error.status = 400
        throw error
      }
      if (input.idempotencyKey) {
        const existing = this.backend.findJobByIdempotencyKey(input.idempotencyKey)
        if (existing) {
          const typeConflict = existing.type !== requestedType
          // Legacy jobs have no requestHash. Keep their historical retry
          // behavior, but never let the same key cross job-type boundaries.
          const hashConflict = Boolean(existing.requestHash && requestedHash && existing.requestHash !== requestedHash)
          if (typeConflict || hashConflict) {
            const error = new Error(typeConflict
              ? 'Aynı idempotency anahtarı başka türde bir iş için kullanılmış.'
              : 'Aynı idempotency anahtarı farklı bir istek için kullanılmış.')
            error.code = 'JOB_IDEMPOTENCY_CONFLICT'
            error.status = 409
            error.details = {
              reason: typeConflict ? 'job-type' : 'request-hash',
              existingJobId: existing.id,
              existingType: existing.type,
              requestedType,
            }
            throw error
          }
          return { job: clone(existing), duplicate: true }
        }
      }
      const timestamp = now()
      const job = {
        id: input.id ?? randomUUID(),
        type: requestedType,
        payload: clone(input.payload ?? {}),
        idempotencyKey: input.idempotencyKey ?? null,
        requestHash: requestedHash,
        state: input.state ?? 'queued',
        stage: 'queued',
        progress: 0,
        attempt: 0,
        maxAttempts: input.maxAttempts ?? this.getConfig().budgets.maxJobAttempts,
        leaseOwner: null,
        leaseUntil: null,
        error: null,
        result: null,
        createdAt: timestamp,
        updatedAt: timestamp,
        startedAt: null,
        finishedAt: null,
      }
      this.#putWithinBudget('jobs', job.id, job)
      return { job: clone(job), duplicate: false }
    })
  }
  createExclusiveTeacherCampaignJob(input) {
    if ((input?.type ?? 'full-cycle') !== TEACHER_FEED_JOB_TYPE) {
      const error = new TypeError('Tekil kampanya primitive’i yalnız 14B besleme işleri için kullanılabilir.')
      error.code = 'INVALID_TEACHER_CAMPAIGN'
      throw error
    }
    return this.backend.transaction(() => {
      // Idempotency wins over the exclusivity check: a lost HTTP response may be
      // retried while its original campaign is still active and must resolve to
      // that exact job instead of being reported as a competing campaign.
      if (input.idempotencyKey) {
        const existing = this.backend.findJobByIdempotencyKey(input.idempotencyKey)
        if (existing) return this.createJob(input)
      }
      const active = this.backend.queryJobs({
        states: [...ACTIVE_TEACHER_FEED_STATES],
        types: [TEACHER_FEED_JOB_TYPE],
        order: 'newest',
        limit: 1,
      })[0]
      if (active) {
        const error = new Error('Önce mevcut 14B besleme kampanyasını tamamlayın, sürdürün veya iptal edin.')
        error.code = 'TEACHER_CAMPAIGN_BUSY'
        error.status = 409
        error.details = { campaignId: active.id, state: active.state }
        throw error
      }
      // Both backends support nested transactions; for SQLite the outer BEGIN
      // IMMEDIATE keeps the no-active-campaign predicate and insert linearizable.
      return this.createJob(input)
    })
  }
  findJobByIdempotencyKey(key) {
    if (!key) return undefined
    return clone(this.backend.findJobByIdempotencyKey(key))
  }
  hasQueuedJob() { return this.backend.hasQueuedJob() }
  findJobsByState(states, { types, order = 'newest', limit = 1, offset = 0 } = {}) {
    return this.backend.queryJobs({ states, types, order, limit, offset }).map(clone)
  }
  findJobByState(states, options = {}) {
    return this.findJobsByState(states, { ...options, limit: 1 })[0]
  }
  getJob(id) { return clone(this.backend.get('jobs', id)) }
  listJobs() { return this.backend.list('jobs') }
  updateJob(id, patch) {
    return this.backend.transaction(() => {
      const job = this.backend.get('jobs', id)
      if (!job) return undefined
      Object.assign(job, clone(patch), { updatedAt: now() })
      this.#putWithinBudget('jobs', id, job)
      return clone(job)
    })
  }
  requestJobControl(id, action) {
    return this.requestJobControlTransition(id, action)?.job
  }
  requestJobControlTransition(id, action) {
    if (!['pause', 'resume', 'cancel'].includes(action)) {
      throw new TypeError('Bilinmeyen iş kontrol eylemi.')
    }
    return this.backend.transaction(() => {
      const job = this.backend.get('jobs', id)
      if (!job) return undefined
      const previousState = job.state
      const interruptedBenchmarkRecovery = action === 'resume' && canResumeInterruptedTeacherBenchmark(job)
      const interruptedGenerationRecovery = action === 'resume' && canResumeInterruptedTeacherGeneration(job)
      let patch = null
      if (action === 'pause') {
        if (job.state === 'queued') {
          this.#finalizeOutstandingTeacherUsage(job)
          patch = { state: 'paused', stage: 'paused' }
        }
        else if (job.state === 'running') patch = { state: 'pause_requested', stage: 'pause-requested' }
      } else if (action === 'cancel') {
        if (['queued', 'paused'].includes(job.state)) {
          this.#finalizeOutstandingTeacherUsage(job)
          patch = {
            state: 'cancelled',
            stage: 'cancelled',
            finishedAt: now(),
            leaseOwner: null,
            leaseUntil: null,
            executionLeaseUntil: null,
          }
        } else if (['running', 'pause_requested'].includes(job.state)) {
          // Keep the current claim alive so the owning worker can atomically
          // finalize an already-issued teacher call's conservative usage.
          patch = { state: 'cancel_requested', stage: 'cancel-requested' }
        }
      } else if (
        job.state === 'paused' ||
        canResumeFailedTeacherCampaign(job) ||
        interruptedBenchmarkRecovery ||
        interruptedGenerationRecovery
      ) {
        if (job.state === 'paused' || interruptedBenchmarkRecovery || interruptedGenerationRecovery) {
          this.#finalizeOutstandingTeacherUsage(job)
        }
        const resourceRecovery = job.state === 'failed'
        patch = {
          state: 'queued',
          stage: interruptedGenerationRecovery
            ? 'interrupted-generation-recovery-queued'
            : interruptedBenchmarkRecovery
              ? 'interrupted-benchmark-recovery-queued'
              : resourceRecovery
                ? 'resource-recovery-queued'
                : 'resumed',
          error: null,
          finishedAt: null,
          leaseOwner: null,
          leaseUntil: null,
          executionLeaseUntil: null,
          ...(resourceRecovery || interruptedBenchmarkRecovery || interruptedGenerationRecovery
            ? { maxAttempts: Math.max(job.maxAttempts, job.attempt + 1) }
            : {}),
        }
      }
      if (!patch) return { job: clone(job), changed: false, previousState, state: job.state, action }
      Object.assign(job, patch, { updatedAt: now() })
      this.#putWithinBudget('jobs', id, job)
      return { job: clone(job), changed: true, previousState, state: job.state, action }
    })
  }
  assertJobClaim(id, owner, attempt) {
    const job = this.backend.get('jobs', id)
    const executionDeadline = Date.parse(job?.executionLeaseUntil ?? '')
    const heartbeatDeadline = Date.parse(job?.leaseUntil ?? '')
    const executionExpired = Number.isFinite(executionDeadline) && executionDeadline <= Date.now()
    const legacyHeartbeatExpired = !Number.isFinite(executionDeadline) && Number.isFinite(heartbeatDeadline) && heartbeatDeadline <= Date.now()
    if (
      !job || job.leaseOwner !== owner || job.attempt !== attempt ||
      !['running', 'pause_requested', 'cancel_requested'].includes(job.state) ||
      executionExpired || legacyHeartbeatExpired
    ) {
      const error = new Error('Worker lease/attempt sahipliği artık güncel değil.')
      error.code = 'STALE_JOB_LEASE'
      error.jobId = id
      error.expectedOwner = owner
      error.expectedAttempt = attempt
      error.actualOwner = job?.leaseOwner ?? null
      error.actualAttempt = job?.attempt ?? null
      error.executionLeaseExpired = executionExpired
      error.legacyHeartbeatExpired = legacyHeartbeatExpired
      throw error
    }
    return clone(job)
  }
  assertRunnableJobClaim(id, owner, attempt) {
    const job = this.assertJobClaim(id, owner, attempt)
    if (job.state !== 'running') {
      const error = new Error('Worker sahipliği geçerli olsa da iş artık yeni veri yazabilecek durumda değil.')
      error.code = 'STALE_JOB_LEASE'
      error.jobId = id
      error.expectedOwner = owner
      error.expectedAttempt = attempt
      error.actualOwner = job.leaseOwner
      error.actualAttempt = job.attempt
      error.actualState = job.state
      throw error
    }
    return job
  }
  updateClaimedJob(id, owner, attempt, patch) {
    return this.backend.transaction(() => {
      this.assertJobClaim(id, owner, attempt)
      const job = this.backend.get('jobs', id)
      Object.assign(job, clone(patch), { updatedAt: now() })
      this.#putWithinBudget('jobs', id, job)
      return clone(job)
    })
  }
  settleClaimedJob(id, owner, attempt, { action, result, error } = {}) {
    if (!['success', 'pause', 'cancel', 'release', 'retry', 'fail'].includes(action)) {
      throw new TypeError('Bilinmeyen claimed iş sonuçlandırma eylemi.')
    }
    return this.backend.transaction(() => {
      this.assertJobClaim(id, owner, attempt)
      const job = this.backend.get('jobs', id)
      let effectiveAction = job.state === 'cancel_requested'
        ? 'cancel'
        : job.state === 'pause_requested'
          ? 'pause'
          : action
      if (effectiveAction === 'retry' && job.attempt >= job.maxAttempts) effectiveAction = 'fail'

      let patch
      if (effectiveAction === 'success') {
        this.#finalizeOutstandingTeacherUsage(job)
        patch = {
          state: 'succeeded', stage: 'completed', progress: 1,
          result: clone(result), error: null, finishedAt: now(),
          leaseOwner: null, leaseUntil: null, executionLeaseUntil: null,
        }
      } else if (effectiveAction === 'pause') {
        // A safety pause may happen before the campaign executor reaches its
        // normal interrupted-attempt recovery (for example after a protocol
        // upgrade). Close any ambiguous paid call before releasing the claim.
        this.#finalizeOutstandingTeacherUsage(job)
        patch = {
          state: 'paused', stage: 'paused',
          leaseOwner: null, leaseUntil: null, executionLeaseUntil: null,
        }
      } else if (effectiveAction === 'cancel') {
        this.#finalizeOutstandingTeacherUsage(job)
        patch = {
          state: 'cancelled', stage: 'cancelled', finishedAt: now(),
          leaseOwner: null, leaseUntil: null, executionLeaseUntil: null,
        }
      } else if (effectiveAction === 'release') {
        this.#finalizeOutstandingTeacherUsage(job)
        patch = {
          state: 'queued', stage: 'recovered',
          leaseOwner: null, leaseUntil: null, executionLeaseUntil: null,
        }
      } else if (effectiveAction === 'retry') {
        this.#finalizeOutstandingTeacherUsage(job)
        patch = {
          state: 'queued', stage: 'retrying', error: clone(error),
          leaseOwner: null, leaseUntil: null, executionLeaseUntil: null,
        }
      } else {
        this.#finalizeOutstandingTeacherUsage(job)
        patch = {
          state: 'failed', stage: 'failed', error: clone(error), finishedAt: now(),
          leaseOwner: null, leaseUntil: null, executionLeaseUntil: null,
        }
      }
      Object.assign(job, patch, { updatedAt: now() })
      this.#putWithinBudget('jobs', id, job)
      return { job: clone(job), action: effectiveAction }
    })
  }
  updateClaimedJobPayload(id, owner, attempt, payloadPatch) {
    if (!payloadPatch || typeof payloadPatch !== 'object' || Array.isArray(payloadPatch)) {
      throw new TypeError('Claim korumalı iş payload yaması bir nesne olmalı.')
    }
    return this.backend.transaction(() => {
      this.assertRunnableJobClaim(id, owner, attempt)
      const job = this.backend.get('jobs', id)
      job.payload = { ...(job.payload ?? {}), ...clone(payloadPatch) }
      job.updatedAt = now()
      this.#putWithinBudget('jobs', id, job)
      return clone(job)
    })
  }
  claimNextJob(owner, leaseMs, executionLeaseMs = leaseMs) {
    return this.backend.transaction(() => {
      const job = this.backend.nextQueuedJob()
      if (!job) return undefined
      this.#finalizeOutstandingTeacherUsage(job)
      const timestamp = now()
      const claimedAt = Date.now()
      Object.assign(job, {
        state: 'running', stage: 'starting', attempt: job.attempt + 1,
        leaseOwner: owner,
        leaseUntil: new Date(claimedAt + leaseMs).toISOString(),
        executionLeaseUntil: new Date(claimedAt + Math.max(leaseMs, executionLeaseMs)).toISOString(),
        attemptStartedAt: timestamp,
        startedAt: job.startedAt ?? timestamp, updatedAt: timestamp,
      })
      this.#putWithinBudget('jobs', job.id, job)
      return clone(job)
    })
  }
  renewLease(id, owner, leaseMs, attempt) {
    return this.backend.transaction(() => {
      const job = this.backend.get('jobs', id)
      if (
        !job || job.state !== 'running' || job.leaseOwner !== owner ||
        job.attempt !== attempt
      ) return false
      const executionDeadline = Date.parse(job.executionLeaseUntil ?? '')
      const currentTime = Date.now()
      const nextHeartbeat = currentTime + leaseMs
      if (Number.isFinite(executionDeadline) && nextHeartbeat >= executionDeadline) {
        if (currentTime >= executionDeadline) return false
        job.leaseUntil = new Date(executionDeadline).toISOString()
      } else {
        job.leaseUntil = new Date(nextHeartbeat).toISOString()
      }
      // The current row was read after BEGIN IMMEDIATE. Persist only the
      // heartbeat fields changed by renewal; all other values are the latest
      // values protected by this transaction rather than a pre-claim copy.
      job.updatedAt = now()
      this.#putWithinBudget('jobs', id, job)
      return true
    })
  }
  recoverExpiredJobs() {
    let count = 0
    const timestamp = Date.now()
    this.backend.transaction(() => {
      const recoverable = this.backend.queryJobs({
        states: ['running', 'pause_requested', 'cancel_requested'],
        order: 'oldest',
        limit: MAX_INDEXED_JOB_QUERY_RESULTS,
      })
      for (const job of recoverable) {
        const heartbeatAlive = job.leaseUntil && Date.parse(job.leaseUntil) > timestamp
        const executionLeaseAlive = job.executionLeaseUntil && Date.parse(job.executionLeaseUntil) > timestamp
        if (heartbeatAlive || executionLeaseAlive) continue
        this.#finalizeOutstandingTeacherUsage(job)
        if (job.state === 'cancel_requested') {
          Object.assign(job, { state: 'cancelled', stage: 'cancelled', finishedAt: now(), leaseOwner: null, leaseUntil: null, executionLeaseUntil: null })
        } else if (job.state === 'pause_requested') {
          Object.assign(job, { state: 'paused', stage: 'paused', leaseOwner: null, leaseUntil: null, executionLeaseUntil: null })
        } else if (job.attempt >= job.maxAttempts) {
          Object.assign(job, { state: 'failed', stage: 'lease-expired', error: { code: 'LEASE_EXPIRED', message: 'İşlem sahibi kayboldu; deneme sınırı doldu.' }, finishedAt: now(), leaseOwner: null, leaseUntil: null, executionLeaseUntil: null })
        } else {
          Object.assign(job, { state: 'queued', stage: 'recovered', leaseOwner: null, leaseUntil: null, executionLeaseUntil: null })
        }
        job.updatedAt = now()
        this.#putWithinBudget('jobs', job.id, job)
        count += 1
      }
    })
    return count
  }

  reconcileUnownedTeacherUsage() {
    let count = 0
    this.backend.transaction(() => {
      const batchSize = 256
      let offset = 0
      while (true) {
        const jobs = this.backend.queryJobs({
          states: ['queued', 'paused', 'succeeded', 'failed', 'cancelled'],
          types: [TEACHER_FEED_JOB_TYPE],
          order: 'oldest',
          limit: batchSize,
          offset,
        })
        for (const job of jobs) {
          if (!this.#finalizeOutstandingTeacherUsage(job)) continue
          job.updatedAt = now()
          this.#putWithinBudget('jobs', job.id, job)
          count += 1
        }
        if (jobs.length < batchSize) break
        offset += jobs.length
      }
    })
    return count
  }

  #finalizeOutstandingTeacherUsage(job) {
    const teacherUsage = finalizeExpiredTeacherAttempt(job)
    if (!(teacherUsage?.teacherTokens > 0)) return null
    if (!(teacherUsage.unreservedTeacherTokens > 0)) return teacherUsage
    const usage = this.backend.get('usage', teacherUsage.day) ?? {
      day: teacherUsage.day,
      jobs: 0,
      teacherRequests: 0,
      teacherTokens: 0,
    }
    usage.teacherTokens = boundedUsageInteger(usage.teacherTokens) + teacherUsage.unreservedTeacherTokens
    this.#putWithinBudget('usage', teacherUsage.day, usage)
    return teacherUsage
  }

  addEvent(event) {
    return this.backend.transaction(() => {
      const max = this.getConfig().budgets.maxEvents
      const existingEvents = this.backend.list('events').sort((a, b) => a.id - b.id)
      for (const stale of existingEvents.slice(0, Math.max(0, existingEvents.length - max + 1))) {
        this.#deleteWithinBudget('events', String(stale.id).padStart(16, '0'))
      }
      const sequence = (this.backend.getMeta('eventSequence') ?? 0) + 1
      const record = { id: sequence, at: now(), level: 'info', ...clone(event) }
      this.#setMetaWithinBudget('eventSequence', sequence)
      this.#putWithinBudget('events', String(sequence).padStart(16, '0'), record)
      return clone(record)
    })
  }
  addClaimedEvent(jobId, owner, attempt, event) {
    return this.backend.transaction(() => {
      this.assertRunnableJobClaim(jobId, owner, attempt)
      return this.addEvent({ ...clone(event), jobId, jobAttempt: attempt })
    })
  }
  listEvents(afterId = 0, limit = 500) {
    return this.backend.list('events').filter((event) => event.id > afterId).sort((a, b) => a.id - b.id).slice(0, limit)
  }

  createArtifact(input) {
    if (input.contentHash) {
      const existing = this.findArtifactByContentHash(input.kind, input.contentHash)
      if (existing) return existing
    }
    const artifact = { id: input.id ?? randomUUID(), kind: input.kind, payload: clone(input.payload), contentHash: input.contentHash ?? null, createdAt: now(), jobId: input.jobId ?? null }
    const minimumBytes = Number(this.getConfig().resourcePolicy?.artifactCompressionMinBytes ?? Number.POSITIVE_INFINITY)
    const storedArtifact = {
      ...artifact,
      payload: encodeArtifactPayload(artifact.payload, minimumBytes),
    }
    this.backend.transaction(() => {
      this.#putWithinBudget('artifacts', artifact.id, storedArtifact)
      if (artifact.contentHash) this.#putWithinBudget('artifact-index', `${artifact.kind}:${artifact.contentHash}`, { artifactId: artifact.id, kind: artifact.kind, contentHash: artifact.contentHash, createdAt: artifact.createdAt })
    })
    return clone(artifact)
  }
  createClaimedArtifact(jobId, owner, attempt, input) {
    if (input?.jobId !== undefined && input.jobId !== jobId) {
      throw new TypeError('Artifact iş kimliği worker claim kimliğiyle uyuşmalı.')
    }
    return this.backend.transaction(() => {
      this.assertRunnableJobClaim(jobId, owner, attempt)
      return this.createArtifact({ ...clone(input), jobId })
    })
  }
  /**
   * Creates an immutable artifact only while an active snapshot and one of its
   * attached immutable heads still match the producer's base. This is a
   * generic compare-and-swap primitive; it does not attach the new artifact or
   * activate a model. Callers must still pass their independent evaluation
   * gate before creating a candidate snapshot.
   */
  createArtifactOnSnapshotHeadBase(input, expectedBase, claim = null) {
    return this.backend.transaction(() => {
      if (claim) this.assertRunnableJobClaim(claim.jobId, claim.owner, claim.attempt)
      if (
        !expectedBase || typeof expectedBase !== 'object' ||
        typeof expectedBase.snapshotId !== 'string' || !expectedBase.snapshotId ||
        typeof expectedBase.headKind !== 'string' || !expectedBase.headKind
      ) {
        throw new TypeError('Artifact CAS tabanı snapshotId ve headKind taşımalıdır.')
      }
      const active = this.getActiveSnapshot()
      const headMetadata = [...(active?.artifactIds ?? [])]
        .reverse()
        .map((artifactId) => this.getArtifactMetadata(artifactId))
        .find((metadata) => metadata?.kind === expectedBase.headKind) ?? null
      const headArtifact = headMetadata ? this.getArtifact(headMetadata.id) : null
      const headFingerprint = headArtifact?.payload?.metadata?.fingerprint ??
        headArtifact?.payload?.value?.fingerprint ?? null
      const checks = [
        [true, active?.id ?? null, expectedBase.snapshotId, 'snapshotId'],
        [Object.hasOwn(expectedBase, 'headArtifactId'), headMetadata?.id ?? null, expectedBase.headArtifactId ?? null, 'headArtifactId'],
        [Object.hasOwn(expectedBase, 'headContentHash'), headMetadata?.contentHash ?? null, expectedBase.headContentHash ?? null, 'headContentHash'],
        [Object.hasOwn(expectedBase, 'headFingerprint'), headFingerprint, expectedBase.headFingerprint ?? null, 'headFingerprint'],
      ]
      const mismatch = checks.find(([enabled, actual, expected]) => enabled && actual !== expected)
      if (mismatch) {
        const error = new Error('Artifactın dayandığı etkin snapshot veya başlık değişti; öneri güncel tabanda yeniden üretilmeli.')
        error.code = expectedBase.staleCode ?? 'STALE_ARTIFACT_BASE'
        error.details = {
          field: mismatch[3],
          expected: mismatch[2],
          actual: mismatch[1],
          activeSnapshotId: active?.id ?? null,
          activeHeadArtifactId: headMetadata?.id ?? null,
          activeHeadFingerprint: headFingerprint,
        }
        throw error
      }
      if (claim && input?.jobId !== undefined && input.jobId !== claim.jobId) {
        throw new TypeError('Artifact iş kimliği CAS worker claim kimliğiyle uyuşmalı.')
      }
      return this.createArtifact({ ...clone(input), ...(claim ? { jobId: claim.jobId } : {}) })
    })
  }
  createClaimedArtifactOnSnapshotHeadBase(jobId, owner, attempt, input, expectedBase) {
    return this.createArtifactOnSnapshotHeadBase(input, expectedBase, { jobId, owner, attempt })
  }
  getArtifact(id) { return decodeArtifactRecord(this.backend.get('artifacts', id)) }
  getArtifactMetadata(id) { return clone(this.backend.getArtifactMetadata(id)) }
  listArtifactMetadata() { return clone(this.backend.listArtifactMetadata()) }
  listArtifactMetadataByKind(kind, options) {
    return clone(this.backend.listArtifactMetadataByKind(kind, options))
  }
  countArtifactMetadataByKind(kind) { return this.backend.countArtifactMetadataByKind(kind) }
  listArtifacts() { return this.listArtifactMetadata().map((artifact) => without(artifact, ['payloadBytes'])) }
  findArtifactByContentHash(kind, contentHash, { repairIndex = true } = {}) {
    if (typeof kind !== 'string' || typeof contentHash !== 'string' || !kind || !contentHash) return undefined
    const key = `${kind}:${contentHash}`
    const index = this.backend.get('artifact-index', key)
    if (index) {
      const artifact = this.backend.get('artifacts', index.artifactId)
      if (artifact?.kind === kind && artifact?.contentHash === contentHash) return decodeArtifactRecord(artifact)
    }
    const artifact = this.backend.list('artifacts').find((item) => item.kind === kind && item.contentHash === contentHash)
    if (artifact && repairIndex) {
      try {
        this.#putWithinBudget('artifact-index', key, { artifactId: artifact.id, kind, contentHash, createdAt: artifact.createdAt })
      } catch (error) {
        // A missing acceleration index must not make an otherwise valid,
        // immutable artifact unreadable when the logical disk budget is full.
        if (error.code !== 'DISK_BUDGET_EXCEEDED') throw error
      }
    }
    return decodeArtifactRecord(artifact)
  }

  findThoughtEpisodeByIdempotencyKey(key) {
    if (typeof key !== 'string' || !key) return undefined
    const indexed = this.backend.get('thought-idempotency', key)
    if (indexed?.episodeId) {
      const episode = this.backend.get('thought-episodes', indexed.episodeId)
      if (episode?.idempotencyKey === key) return clone(episode)
    }
    const episode = this.backend.list('thought-episodes').find((item) => item.idempotencyKey === key)
    if (episode) {
      try {
        this.#putWithinBudget('thought-idempotency', key, {
          episodeId: episode.id,
          requestHash: episode.requestHash,
          createdAt: episode.createdAt,
        })
      } catch (error) {
        if (error.code !== 'DISK_BUDGET_EXCEEDED') throw error
      }
    }
    return clone(episode)
  }

  createThoughtEpisode(input) {
    return this.backend.transaction(() => {
      if (input.idempotencyKey) {
        const existing = this.findThoughtEpisodeByIdempotencyKey(input.idempotencyKey)
        if (existing) {
          if (existing.requestHash !== input.requestHash) {
            const error = new Error('Aynı idempotency anahtarı farklı bir düşünce isteği için kullanılamaz.')
            error.code = 'THOUGHT_IDEMPOTENCY_CONFLICT'
            throw error
          }
          return { episode: existing, duplicate: true }
        }
      }
      const timestamp = now()
      const episode = {
        ...clone(input),
        format: 'thought-episode',
        version: 1,
        id: input.id ?? randomUUID(),
        status: input.status ?? 'queued',
        traceArtifactIds: [],
        graphArtifactId: null,
        checkpointArtifactId: null,
        headProposalArtifactId: null,
        headArtifactId: null,
        candidateSnapshotId: null,
        review: null,
        error: null,
        tickCount: 0,
        lastTickDigest: null,
        createdAt: timestamp,
        updatedAt: timestamp,
        startedAt: null,
        finishedAt: null,
      }
      this.#putWithinBudget('thought-episodes', episode.id, episode)
      if (episode.idempotencyKey) {
        this.#putWithinBudget('thought-idempotency', episode.idempotencyKey, {
          episodeId: episode.id,
          requestHash: episode.requestHash,
          createdAt: timestamp,
        })
      }
      return { episode: clone(episode), duplicate: false }
    })
  }

  getThoughtEpisode(id) { return clone(this.backend.get('thought-episodes', id)) }
  listThoughtEpisodes() { return this.backend.list('thought-episodes') }

  updateThoughtEpisode(id, patch, { expectedStatuses } = {}) {
    return this.backend.transaction(() => {
      const episode = this.backend.get('thought-episodes', id)
      if (!episode) return undefined
      if (expectedStatuses && !expectedStatuses.includes(episode.status)) {
        const error = new Error('Düşünce bölümü istenen durum geçişine uygun değil.')
        error.code = 'THOUGHT_STATE_CONFLICT'
        throw error
      }
      const safePatch = clone(patch ?? {})
      for (const protectedKey of ['id', 'format', 'version', 'requestHash', 'idempotencyKey', 'base']) {
        delete safePatch[protectedKey]
      }
      Object.assign(episode, safePatch, { updatedAt: now() })
      this.#putWithinBudget('thought-episodes', id, episode)
      return clone(episode)
    })
  }

  requestThoughtEpisodeControl(id, action) {
    if (!['pause', 'resume', 'cancel'].includes(action)) {
      throw new TypeError('Bilinmeyen düşünce bölümü kontrol eylemi.')
    }
    return this.backend.transaction(() => {
      const episode = this.backend.get('thought-episodes', id)
      if (!episode) return undefined
      const previousStatus = episode.status
      let patch = null
      if (action === 'pause') {
        if (episode.status === 'queued') patch = { status: 'paused' }
        else if (episode.status === 'running') patch = { status: 'pause_requested' }
      } else if (action === 'resume') {
        if (episode.status === 'paused') patch = { status: 'queued', error: null, finishedAt: null }
      } else if (['queued', 'paused'].includes(episode.status)) {
        patch = { status: 'cancelled', finishedAt: now() }
      } else if (['running', 'pause_requested'].includes(episode.status)) {
        patch = { status: 'cancel_requested' }
      }
      if (!patch) {
        return { episode: clone(episode), changed: false, previousStatus, status: episode.status, action }
      }
      Object.assign(episode, patch, { updatedAt: now() })
      this.#putWithinBudget('thought-episodes', id, episode)
      return { episode: clone(episode), changed: true, previousStatus, status: episode.status, action }
    })
  }

  attachThoughtEpisodeArtifacts(id, entries, patch = {}, { verifyCompletionBase = false, expectedStatuses } = {}) {
    const allowedFields = new Set(['traceArtifactIds', 'graphArtifactId', 'checkpointArtifactId', 'headProposalArtifactId'])
    return this.backend.transaction(() => {
      const episode = this.backend.get('thought-episodes', id)
      if (!episode) return undefined
      if (expectedStatuses && !expectedStatuses.includes(episode.status)) {
        const error = new Error('Düşünce bölümü artifact geçişi güncel durumla çelişiyor.')
        error.code = 'THOUGHT_STATE_CONFLICT'
        error.actualStatus = episode.status
        error.expectedStatuses = [...expectedStatuses]
        throw error
      }
      let staleBase = false
      let effectiveEntries = [...(entries ?? [])]
      const effectivePatch = clone(patch ?? {})
      if (verifyCompletionBase && episode.commitEligible === true) {
        const active = this.getActiveSnapshot()
        const currentHeadArtifact = [...(active?.artifactIds ?? [])]
          .reverse()
          .map((artifactId) => this.getArtifact(artifactId))
          .find((artifact) => artifact?.kind === 'thought-runtime-head') ?? null
        staleBase = !active ||
          active.id !== episode.base?.snapshotId ||
          active.evaluation?.passed !== true ||
          active.evaluation?.runtimeVerified !== true ||
          (currentHeadArtifact?.id ?? null) !== (episode.base?.thoughtHeadArtifactId ?? null)
        if (staleBase) {
          // The immutable trace remains useful for audit/replay, but a delta
          // computed on an obsolete model/head must never become reviewable.
          effectiveEntries = effectiveEntries.filter((entry) => entry.field !== 'headProposalArtifactId')
          effectivePatch.commitEligible = false
          effectivePatch.commitStatus = 'not-eligible'
          effectivePatch.deltaSuppressedReason = 'stale-thought-base'
          effectivePatch.error = {
            code: 'STALE_THOUGHT_BASE',
            message: 'Bölüm çalışırken etkin model veya düşünce başlığı değişti; iz saklandı ancak öğrenme önerisi atıldı.',
          }
        }
      }
      const artifacts = []
      for (const entry of effectiveEntries) {
        if (!allowedFields.has(entry.field)) {
          const error = new TypeError('Düşünce eseri bilinmeyen bir kayıt alanına bağlanamaz.')
          error.code = 'INVALID_THOUGHT_ARTIFACT_FIELD'
          throw error
        }
        const artifact = this.createArtifact({ ...entry.artifact, jobId: null })
        artifacts.push({ field: entry.field, artifact })
        if (entry.field === 'traceArtifactIds') {
          episode.traceArtifactIds = [...new Set([...(episode.traceArtifactIds ?? []), artifact.id])]
        } else {
          episode[entry.field] = artifact.id
        }
      }
      const safePatch = effectivePatch
      for (const protectedKey of [
        'id', 'format', 'version', 'requestHash', 'idempotencyKey', 'base',
        'traceArtifactIds', 'graphArtifactId', 'checkpointArtifactId', 'headProposalArtifactId', 'headArtifactId', 'candidateSnapshotId',
      ]) delete safePatch[protectedKey]
      Object.assign(episode, safePatch, { updatedAt: now() })
      this.#putWithinBudget('thought-episodes', id, episode)
      return { episode: clone(episode), artifacts: clone(artifacts), staleBase }
    })
  }

  reviewThoughtEpisode(id, input) {
    return this.backend.transaction(() => {
      const episode = this.backend.get('thought-episodes', id)
      if (!episode) return undefined
      if (episode.review) {
        if (episode.review.decision !== input.decision) {
          const error = new Error('Düşünce bölümü daha önce ters yönde karara bağlandı.')
          error.code = 'THOUGHT_ALREADY_REVIEWED'
          throw error
        }
        return {
          episode: clone(episode),
          duplicate: true,
          headArtifact: episode.headArtifactId ? this.getArtifact(episode.headArtifactId) : null,
          snapshot: episode.candidateSnapshotId ? this.getSnapshot(episode.candidateSnapshotId) : null,
        }
      }
      if (episode.status !== 'completed') {
        const error = new Error('Yalnızca tamamlanmış bir düşünce bölümü karara bağlanabilir.')
        error.code = 'THOUGHT_NOT_COMPLETE'
        throw error
      }
      const reviewedAt = now()
      if (input.decision === 'reject') {
        episode.review = { decision: 'reject', reason: input.reason ?? '', reviewedAt }
        episode.commitStatus = 'rejected'
        episode.updatedAt = reviewedAt
        this.#putWithinBudget('thought-episodes', id, episode)
        return { episode: clone(episode), duplicate: false, headArtifact: null, snapshot: null }
      }
      if (input.decision !== 'approve') throw new TypeError('Geçersiz düşünce bölümü kararı.')
      if (episode.commitEligible !== true || !episode.headProposalArtifactId) {
        const error = new Error('Bu bölüm salt araştırma izidir veya öğrenme önerisi taşımıyor.')
        error.code = 'THOUGHT_COMMIT_NOT_ELIGIBLE'
        throw error
      }
      const active = this.getActiveSnapshot()
      if (!active || active.id !== episode.base?.snapshotId || active.id !== input.expectedBaseSnapshotId) {
        const error = new Error('Düşüncenin dayandığı etkin model değişti; öneri yeniden oynatılmalı ve değerlendirilmelidir.')
        error.code = 'STALE_THOUGHT_BASE'
        throw error
      }
      const currentHeadArtifact = [...(active.artifactIds ?? [])]
        .reverse()
        .map((artifactId) => this.getArtifact(artifactId))
        .find((artifact) => artifact?.kind === 'thought-runtime-head') ?? null
      if ((currentHeadArtifact?.id ?? null) !== (episode.base?.thoughtHeadArtifactId ?? null)) {
        const error = new Error('Düşünce başlığının revizyonu değişti; öneri güncel başlık üzerinde yeniden oynatılmalıdır.')
        error.code = 'STALE_THOUGHT_HEAD'
        throw error
      }
      if (!this.getArtifact(episode.headProposalArtifactId)) {
        const error = new Error('Düşünce başlığı öneri eseri bulunamadı.')
        error.code = 'THOUGHT_HEAD_PROPOSAL_MISSING'
        throw error
      }
      const headArtifact = this.createArtifact({ ...input.headArtifact, jobId: null })
      this.#putWithinBudget('thought-heads', headArtifact.id, {
        artifactId: headArtifact.id,
        episodeId: episode.id,
        baseSnapshotId: active.id,
        parentHeadArtifactId: episode.base?.thoughtHeadArtifactId ?? null,
        revision: input.headRevision,
        fingerprint: input.headFingerprint,
        createdAt: reviewedAt,
      })
      const snapshot = this.createSnapshot({
        status: 'candidate',
        parentId: active.id,
        label: input.snapshotLabel,
        metrics: clone(input.metrics ?? active.metrics),
        artifactIds: [...new Set([...(active.artifactIds ?? []), headArtifact.id])],
        sourceIds: clone(active.sourceIds ?? []),
        sourceRevisions: clone(active.sourceRevisions ?? {}),
        inputManifestHash: active.inputManifestHash,
        evaluation: clone(input.evaluation),
      })
      episode.review = { decision: 'approve', reason: input.reason ?? '', reviewedAt }
      episode.commitStatus = 'candidate'
      episode.headArtifactId = headArtifact.id
      episode.candidateSnapshotId = snapshot.id
      episode.updatedAt = reviewedAt
      this.#putWithinBudget('thought-episodes', id, episode)
      return { episode: clone(episode), duplicate: false, headArtifact, snapshot }
    })
  }

  recoverThoughtEpisodes() {
    let recovered = 0
    this.backend.transaction(() => {
      for (const episode of this.backend.list('thought-episodes')) {
        if (!['queued', 'running', 'pause_requested', 'cancel_requested'].includes(episode.status)) continue
        episode.status = episode.status === 'cancel_requested' ? 'cancelled' : 'paused'
        episode.error = episode.status === 'paused'
          ? { code: 'THOUGHT_SERVICE_RESTARTED', message: 'Servis yeniden başladığı için bölüm güvenli kontrol noktasında duraklatıldı.' }
          : null
        episode.finishedAt = episode.status === 'cancelled' ? now() : null
        episode.updatedAt = now()
        this.#putWithinBudget('thought-episodes', episode.id, episode)
        recovered += 1
      }
    })
    return recovered
  }

  approximateStorageBytes() {
    const tracked = this.backend.getMeta('logicalStorageBytes')
    return Number.isFinite(tracked) && tracked >= 0 ? tracked : this.backend.calculateLogicalBytes()
  }
  getEffectiveDiskBudgetBytes(config = this.getConfig()) {
    const configured = config.budgets.diskMb * 1024 * 1024
    return this.backendKind === 'atomic-json'
      ? Math.min(configured, ATOMIC_JSON_MAX_LOGICAL_BYTES)
      : configured
  }
  assertDiskBudget(additionalBytes = 0, config = this.getConfig()) {
    const limit = this.getEffectiveDiskBudgetBytes(config)
    if (this.approximateStorageBytes() + additionalBytes > limit) {
      const error = new Error(this.backendKind === 'atomic-json'
        ? 'Atomik JSON yedek depolamasının 64 MiB güvenli kullanım sınırı aşıldı; büyük korpus için SQLite destekli Node.js kullanın.'
        : 'Kalıcı servis disk bütçesi aşıldı.')
      error.code = 'DISK_BUDGET_EXCEEDED'
      throw error
    }
  }
  #putWithinBudget(bucket, id, value) {
    return this.backend.transaction(() => {
      const previous = this.backend.get(bucket, id)
      const delta = recordBytes(value) - recordBytes(previous)
      if (delta > 0) this.assertDiskBudget(delta)
      this.backend.put(bucket, id, value)
      this.backend.setMeta('logicalStorageBytes', Math.max(0, this.approximateStorageBytes() + delta))
    })
  }
  #deleteWithinBudget(bucket, id) {
    return this.backend.transaction(() => {
      const previous = this.backend.get(bucket, id)
      if (previous === undefined) return
      const nextBytes = Math.max(0, this.approximateStorageBytes() - recordBytes(previous))
      this.backend.delete(bucket, id)
      this.backend.setMeta('logicalStorageBytes', nextBytes)
    })
  }
  #setMetaWithinBudget(key, value, previous = this.backend.getMeta(key), config = this.getConfig()) {
    return this.backend.transaction(() => {
      const delta = recordBytes(value) - recordBytes(previous)
      if (delta > 0 || key === 'config') this.assertDiskBudget(delta, config)
      this.backend.setMeta(key, value)
      this.backend.setMeta('logicalStorageBytes', Math.max(0, this.approximateStorageBytes() + delta))
    })
  }

  createSnapshot(input) {
    const active = this.getActiveSnapshot()
    const sourceIds = clone(input.sourceIds ?? [])
    const sourceRevisions = Object.prototype.hasOwnProperty.call(input, 'sourceRevisions')
      ? clone(input.sourceRevisions ?? {})
      : Object.fromEntries(sourceIds.flatMap((id) => {
        const source = this.backend.getSourceMetadata(id)
        return source ? [[id, sourceRepresentationRevision(source)]] : []
      }))
    const snapshot = {
      id: input.id ?? randomUUID(),
      parentId: input.parentId ?? active?.id ?? null,
      status: input.status ?? 'candidate',
      label: input.label ?? `Aday ${new Date().toLocaleString('tr-TR')}`,
      metrics: clone(input.metrics ?? null),
      artifactIds: clone(input.artifactIds ?? []),
      sourceIds,
      sourceRevisions,
      inputManifestHash: input.inputManifestHash ?? null,
      evaluation: clone(input.evaluation ?? null),
      createdAt: now(),
      activatedAt: null,
      rejectedAt: null,
      rollbackOf: input.rollbackOf ?? null,
      producerJobId: input.producerJobId ?? null,
      producerAttempt: Number.isInteger(input.producerAttempt) ? input.producerAttempt : null,
    }
    if (input.id) {
      const existing = this.backend.get('snapshots', input.id)
      if (existing) {
        const stable = (value) => JSON.stringify(value)
        const compatible =
          (existing.parentId ?? null) === (snapshot.parentId ?? null) &&
          (existing.inputManifestHash ?? null) === (snapshot.inputManifestHash ?? null) &&
          (existing.producerJobId ?? null) === (snapshot.producerJobId ?? null) &&
          stable(existing.sourceIds ?? []) === stable(snapshot.sourceIds ?? []) &&
          stable(existing.sourceRevisions ?? {}) === stable(snapshot.sourceRevisions ?? {}) &&
          stable(existing.artifactIds ?? []) === stable(snapshot.artifactIds ?? [])
        if (!compatible) {
          const error = new Error('Aynı snapshot kimliği farklı bir eğitim girdisi için kullanılamaz.')
          error.code = 'SNAPSHOT_IDEMPOTENCY_CONFLICT'
          throw error
        }
        return clone(existing)
      }
    }
    this.#putWithinBudget('snapshots', snapshot.id, snapshot)
    return clone(snapshot)
  }
  createClaimedSnapshot(jobId, owner, attempt, input) {
    return this.backend.transaction(() => {
      this.assertRunnableJobClaim(jobId, owner, attempt)
      return this.createSnapshot({
        ...clone(input),
        producerJobId: jobId,
        producerAttempt: attempt,
      })
    })
  }
  /**
   * Creates a candidate snapshot only while both the worker claim and the
   * exact active snapshot/head observed by an external evaluator still match.
   * Unlike createClaimedSnapshot, this closes the final gap between a
   * release evaluator's last read and its candidate publication. It never
   * activates the new snapshot.
   */
  createClaimedSnapshotOnSnapshotHeadBase(jobId, owner, attempt, input, expectedBase) {
    return this.backend.transaction(() => {
      this.assertRunnableJobClaim(jobId, owner, attempt)
      if (
        !expectedBase || typeof expectedBase !== 'object' ||
        typeof expectedBase.snapshotId !== 'string' || !expectedBase.snapshotId ||
        typeof expectedBase.headKind !== 'string' || !expectedBase.headKind
      ) {
        throw new TypeError('Snapshot CAS tabanı snapshotId ve headKind taşımalıdır.')
      }
      const active = this.getActiveSnapshot()
      const headMetadata = [...(active?.artifactIds ?? [])]
        .reverse()
        .map((artifactId) => this.getArtifactMetadata(artifactId))
        .find((metadata) => metadata?.kind === expectedBase.headKind) ?? null
      const headArtifact = headMetadata ? this.getArtifact(headMetadata.id) : null
      const headFingerprint = headArtifact?.payload?.metadata?.fingerprint ??
        headArtifact?.payload?.value?.fingerprint ?? null
      const checks = [
        [true, active?.id ?? null, expectedBase.snapshotId, 'snapshotId'],
        [Object.hasOwn(expectedBase, 'headArtifactId'), headMetadata?.id ?? null, expectedBase.headArtifactId ?? null, 'headArtifactId'],
        [Object.hasOwn(expectedBase, 'headContentHash'), headMetadata?.contentHash ?? null, expectedBase.headContentHash ?? null, 'headContentHash'],
        [Object.hasOwn(expectedBase, 'headFingerprint'), headFingerprint, expectedBase.headFingerprint ?? null, 'headFingerprint'],
        [true, input?.parentId ?? null, expectedBase.snapshotId, 'candidateParentId'],
      ]
      const mismatch = checks.find(([enabled, actual, expected]) => enabled && actual !== expected)
      if (mismatch) {
        const error = new Error('Adayın dayandığı etkin snapshot veya başlık değişti; release güncel tabanda yeniden ölçülmeli.')
        error.code = expectedBase.staleCode ?? 'STALE_ARTIFACT_BASE'
        error.details = {
          field: mismatch[3],
          expected: mismatch[2],
          actual: mismatch[1],
          activeSnapshotId: active?.id ?? null,
          activeHeadArtifactId: headMetadata?.id ?? null,
          activeHeadFingerprint: headFingerprint,
        }
        throw error
      }
      return this.createSnapshot({
        ...clone(input),
        producerJobId: jobId,
        producerAttempt: attempt,
      })
    })
  }
  /**
   * Publishes a non-active candidate and settles its producer job in one
   * transaction. This is the candidate-only counterpart of
   * completeClaimedJobWithSnapshot: a crash can expose neither half of the
   * publication, and no automatic activation occurs.
   */
  completeClaimedJobWithCandidateOnSnapshotHeadBase(
    jobId,
    owner,
    attempt,
    input,
    expectedBase,
    result,
    candidateGuard,
  ) {
    if (typeof candidateGuard !== 'function') {
      const error = new TypeError('Atomik aday yayını senkron doğrulama kapısı ister.')
      error.code = 'SNAPSHOT_CANDIDATE_GUARD_REQUIRED'
      throw error
    }
    return this.backend.transaction(() => {
      const claimedJob = this.assertRunnableJobClaim(jobId, owner, attempt)
      if (typeof input?.id !== 'string' || !input.id || result?.snapshotId !== input.id) {
        const error = new Error('İş sonucu ile yayınlanacak aday snapshot kimliği uyuşmuyor.')
        error.code = 'SNAPSHOT_PRODUCER_MISMATCH'
        throw error
      }
      const snapshot = this.createClaimedSnapshotOnSnapshotHeadBase(
        jobId,
        owner,
        attempt,
        input,
        expectedBase,
      )
      const guardResult = candidateGuard(clone(snapshot), clone(claimedJob), clone(result))
      if (guardResult && typeof guardResult.then === 'function') {
        const error = new TypeError('Aday snapshot kapısı transaction içinde senkron çalışmalıdır.')
        error.code = 'ASYNC_SNAPSHOT_CANDIDATE_GUARD'
        throw error
      }
      if (guardResult === false || guardResult?.eligible === false) {
        const error = new Error(guardResult?.message ?? 'Aday snapshot immutable yayın politikasını geçemedi.')
        error.code = guardResult?.code ?? 'SNAPSHOT_CANDIDATE_GUARD_REJECTED'
        error.details = clone(guardResult?.details ?? null)
        throw error
      }
      this.assertRunnableJobClaim(jobId, owner, attempt)
      const settlement = this.settleClaimedJob(jobId, owner, attempt, { action: 'success', result })
      if (settlement.action !== 'success') {
        const error = new Error('Atomik aday yayını iş başarısıyla birlikte sonuçlanamadı.')
        error.code = 'SNAPSHOT_CANDIDATE_SETTLEMENT_REJECTED'
        throw error
      }
      return { job: settlement.job, snapshot: clone(snapshot) }
    })
  }
  getSnapshot(id) { return clone(this.backend.get('snapshots', id)) }
  listSnapshots() { return this.backend.list('snapshots') }
  getActiveSnapshot() {
    const id = this.backend.getMeta('activeSnapshotId')
    return id ? this.getSnapshot(id) : undefined
  }
  activateSnapshot(id, { rollbackOf = null, resetAbsentSources = false } = {}) {
    return this.backend.transaction(() => this.#activateSnapshot(id, { rollbackOf, resetAbsentSources, allowSourceDrift: false }))
  }
  activateClaimedSnapshot(jobId, owner, attempt, id, { rollbackOf = null, resetAbsentSources = false } = {}) {
    return this.backend.transaction(() => {
      this.assertRunnableJobClaim(jobId, owner, attempt)
      return this.#activateSnapshot(id, { rollbackOf, resetAbsentSources, allowSourceDrift: false })
    })
  }
  completeClaimedJobWithSnapshot(jobId, owner, attempt, snapshotId, result, activationGuard) {
    if (typeof activationGuard !== 'function') {
      const error = new TypeError('Atomik snapshot tamamlama işlemi senkron etkinleştirme kapısı ister.')
      error.code = 'SNAPSHOT_ACTIVATION_GUARD_REQUIRED'
      throw error
    }
    return this.backend.transaction(() => {
      const claimedJob = this.assertRunnableJobClaim(jobId, owner, attempt)
      const snapshot = this.backend.get('snapshots', snapshotId)
      if (!snapshot || snapshot.producerJobId !== jobId || result?.snapshotId !== snapshotId) {
        const error = new Error('İş sonucu ile etkinleştirilecek snapshot üretim kanıtı uyuşmuyor.')
        error.code = 'SNAPSHOT_PRODUCER_MISMATCH'
        throw error
      }
      const guardResult = activationGuard(clone(snapshot), clone(claimedJob), clone(result))
      if (guardResult && typeof guardResult.then === 'function') {
        const error = new TypeError('Snapshot etkinleştirme kapısı transaction içinde senkron çalışmalıdır.')
        error.code = 'ASYNC_SNAPSHOT_ACTIVATION_GUARD'
        throw error
      }
      if (guardResult === false || guardResult?.eligible === false) {
        const error = new Error(guardResult?.message ?? 'Snapshot canlı etkinleştirme politikasını geçemedi.')
        error.code = guardResult?.code ?? 'SNAPSHOT_ACTIVATION_GUARD_REJECTED'
        error.details = clone(guardResult?.details ?? null)
        throw error
      }
      // The guard is supplied by orchestration code but executes under this
      // backend transaction. Re-check ownership after it so even an accidental
      // stateful callback cannot bypass the claim fence.
      this.assertRunnableJobClaim(jobId, owner, attempt)
      const activated = this.#activateSnapshot(snapshotId, {
        rollbackOf: null,
        resetAbsentSources: false,
        allowSourceDrift: false,
      })
      const job = this.backend.get('jobs', jobId)
      this.#finalizeOutstandingTeacherUsage(job)
      Object.assign(job, {
        state: 'succeeded',
        stage: 'completed',
        progress: 1,
        result: clone(result),
        error: null,
        finishedAt: now(),
        leaseOwner: null,
        leaseUntil: null,
        executionLeaseUntil: null,
        updatedAt: now(),
      })
      this.#putWithinBudget('jobs', jobId, job)
      return { job: clone(job), snapshot: clone(activated) }
    })
  }
  rollbackToSnapshot(id) {
    return this.backend.transaction(() => {
      const target = this.backend.get('snapshots', id)
      if (!target || !['active', 'superseded'].includes(target.status)) return undefined
      const previousId = this.backend.getMeta('activeSnapshotId') ?? null
      const restored = this.createSnapshot({
        parentId: previousId,
        status: 'candidate',
        label: `Geri dönüş: ${target.label}`,
        metrics: target.metrics,
        artifactIds: target.artifactIds,
        sourceIds: target.sourceIds,
        sourceRevisions: target.sourceRevisions ?? {},
        inputManifestHash: target.inputManifestHash,
        evaluation: { passed: true, rollback: true, sourceSnapshotId: target.id },
        rollbackOf: previousId,
      })
      return this.#activateSnapshot(restored.id, {
        rollbackOf: previousId,
        // A rollback restores the target's exact source frontier. Anything
        // learned only after that point must become eligible for a later run.
        resetAbsentSources: true,
        allowSourceDrift: true,
      })
    })
  }
  #activateSnapshot(id, { rollbackOf, resetAbsentSources, allowSourceDrift }) {
    const snapshot = this.backend.get('snapshots', id)
    if (!snapshot) return undefined
    if (snapshot.status !== 'candidate') {
      const error = new Error('Yalnızca bekleyen bir aday anlık görüntü etkinleştirilebilir.')
      error.code = 'SNAPSHOT_NOT_CANDIDATE'
      throw error
    }

    const previousId = this.backend.getMeta('activeSnapshotId') ?? null
    if ((snapshot.parentId ?? null) !== previousId) {
      const error = new Error('Adayın dayandığı etkin model artık güncel değil; aday yeniden üretilmeli ve değerlendirilmelidir.')
      error.code = 'SNAPSHOT_PARENT_MISMATCH'
      throw error
    }
    if (!allowSourceDrift) this.#assertSnapshotSourcesCurrent(snapshot)

    if (previousId) {
      const previous = this.backend.get('snapshots', previousId)
      if (!previous || previous.status !== 'active') {
        const error = new Error('Etkin anlık görüntü kaydı tutarsız.')
        error.code = 'ACTIVE_SNAPSHOT_INCONSISTENT'
        throw error
      }
      previous.status = 'superseded'
      this.#putWithinBudget('snapshots', previous.id, previous)
    }

    snapshot.status = 'active'
    snapshot.activatedAt = now()
    if (rollbackOf) snapshot.rollbackOf = rollbackOf
    this.#putWithinBudget('snapshots', snapshot.id, snapshot)
    this.#reconcileProcessedSources(snapshot, resetAbsentSources)
    this.#setMetaWithinBudget('activeSnapshotId', id)
    return clone(snapshot)
  }
  #assertSnapshotSourcesCurrent(snapshot) {
    for (const sourceId of snapshot.sourceIds ?? []) {
      const source = this.backend.getSourceMetadata(sourceId)
      const expectedRevision = snapshot.sourceRevisions?.[sourceId]
      if (!source || source.status !== 'approved' || source.selected !== true || !expectedRevision || sourceRepresentationRevision(source) !== expectedRevision) {
        const error = new Error('Adayın kullandığı kaynak veya temsil kanıtı artık güncel değil; aday yeniden üretilmelidir.')
        error.code = 'SNAPSHOT_SOURCE_STALE'
        throw error
      }
    }
  }
  #reconcileProcessedSources(snapshot, resetAbsentSources) {
    const included = new Set(snapshot.sourceIds ?? [])
    for (const metadata of this.backend.listSourceMetadata()) {
      const source = metadata
      let next = source.processedSnapshotId
      const revisionMatches = snapshot.sourceRevisions?.[source.id] === sourceRepresentationRevision(source)
      if (included.has(source.id) && source.status === 'approved' && revisionMatches) next = snapshot.id
      else if (included.has(source.id)) next = null
      else if (resetAbsentSources) next = null
      if (source.processedSnapshotId === next) continue
      const fullSource = this.backend.get('sources', source.id)
      if (!fullSource) continue
      fullSource.processedSnapshotId = next
      fullSource.updatedAt = now()
      this.#putWithinBudget('sources', fullSource.id, fullSource)
    }
  }
  rejectSnapshot(id, evaluation, { jobId, claim } = {}) {
    if ((claim !== undefined || jobId !== undefined) && (
      typeof jobId !== 'string' || !jobId ||
      !claim || typeof claim !== 'object' || Array.isArray(claim) ||
      typeof claim.owner !== 'string' || !claim.owner ||
      !Number.isInteger(claim.attempt) || claim.attempt < 1
    )) throw new TypeError('Snapshot reddi worker claim kaydı geçersiz.')
    return this.backend.transaction(() => {
      if (claim) this.assertRunnableJobClaim(jobId, claim.owner, claim.attempt)
      const snapshot = this.backend.get('snapshots', id)
      if (!snapshot) return undefined
      Object.assign(snapshot, { status: 'rejected', rejectedAt: now(), evaluation: clone(evaluation) })
      this.#putWithinBudget('snapshots', id, snapshot)
      return clone(snapshot)
    })
  }

  getUsage(day = new Date().toISOString().slice(0, 10)) {
    return clone(this.backend.get('usage', day) ?? { day, jobs: 0, teacherRequests: 0, teacherTokens: 0 })
  }
  reserveTeacherCallUsage({
    id = randomUUID(),
    usageDay = new Date().toISOString().slice(0, 10),
    model = '',
    maxPromptTokens,
    maxCompletionTokens,
    reservedTokens,
    thoughtEpisodeId,
  }) {
    if (typeof id !== 'string' || !id || id.length > 200) throw new TypeError('Öğretmen kullanım rezervasyonu kimliği geçersiz.')
    if (!/^\d{4}-\d{2}-\d{2}$/u.test(usageDay)) throw new TypeError('Öğretmen kullanım rezervasyonu günü geçersiz.')
    if (typeof model !== 'string' || model.length > 160) throw new TypeError('Öğretmen kullanım rezervasyonu model adı geçersiz.')
    if (!Number.isSafeInteger(maxPromptTokens) || maxPromptTokens <= 0) throw new TypeError('Öğretmen prompt üst sınırı geçersiz.')
    if (!Number.isSafeInteger(maxCompletionTokens) || maxCompletionTokens <= 0) throw new TypeError('Öğretmen çıktı üst sınırı geçersiz.')
    const fencedThoughtEpisodeId = thoughtEpisodeId === undefined || thoughtEpisodeId === null
      ? null
      : String(thoughtEpisodeId)
    if (fencedThoughtEpisodeId !== null && (!fencedThoughtEpisodeId || fencedThoughtEpisodeId.length > 256)) {
      throw new TypeError('Öğretmen kullanım rezervasyonunun düşünce bölümü kimliği geçersiz.')
    }
    const expectedReservedTokens = maxPromptTokens + maxCompletionTokens
    if (!Number.isSafeInteger(expectedReservedTokens) || reservedTokens !== expectedReservedTokens) {
      throw new TypeError('Öğretmen toplam token rezervasyonu bileşenleriyle uyuşmuyor.')
    }
    const request = {
      id,
      usageDay,
      model,
      maxPromptTokens,
      maxCompletionTokens,
      reservedTokens,
      ...(fencedThoughtEpisodeId ? { thoughtEpisodeId: fencedThoughtEpisodeId } : {}),
    }
    return this.backend.transaction(() => {
      if (fencedThoughtEpisodeId) {
        const episode = this.backend.get('thought-episodes', fencedThoughtEpisodeId)
        if (!episode) {
          const error = new Error('Öğretmen kullanım rezervasyonunun düşünce bölümü bulunamadı.')
          error.code = 'THOUGHT_EPISODE_NOT_FOUND'
          throw error
        }
        if (episode.status !== 'running') {
          const error = new Error('Düşünce bölümü artık yeni bir öğretmen çağrısı başlatabilecek durumda değil.')
          error.code = ['cancel_requested', 'cancelled'].includes(episode.status)
            ? 'THOUGHT_CANCELLED'
            : ['pause_requested', 'paused'].includes(episode.status)
              ? 'THOUGHT_PAUSED'
              : 'THOUGHT_STATE_CONFLICT'
          error.actualStatus = episode.status
          throw error
        }
      }
      const existing = this.backend.get('teacherUsageReservations', id)
      if (existing) {
        const same = existing.usageDay === usageDay && existing.model === model &&
          existing.maxPromptTokens === maxPromptTokens && existing.maxCompletionTokens === maxCompletionTokens &&
          existing.reservedTokens === reservedTokens &&
          (existing.thoughtEpisodeId ?? null) === fencedThoughtEpisodeId
        if (!same) {
          const error = new Error('Öğretmen kullanım rezervasyonu kimliği farklı bir istek için kullanılmış.')
          error.code = 'TEACHER_USAGE_RESERVATION_CONFLICT'
          throw error
        }
        return clone(existing)
      }
      this.consumeUsageBatch([
        { kind: 'teacherRequests', amount: 1, day: usageDay },
        { kind: 'teacherTokens', amount: reservedTokens, day: usageDay },
      ])
      const timestamp = now()
      const reservation = {
        ...request,
        status: 'reserved',
        promptTokens: null,
        completionTokens: null,
        actualTokens: null,
        usageEstimated: true,
        createdAt: timestamp,
        settledAt: null,
      }
      this.#putWithinBudget('teacherUsageReservations', id, reservation)
      return clone(reservation)
    })
  }
  settleTeacherCallUsage(id, { promptTokens, completionTokens, usageEstimated = false }) {
    if (typeof id !== 'string' || !id || id.length > 200) throw new TypeError('Öğretmen kullanım rezervasyonu kimliği geçersiz.')
    if (!Number.isSafeInteger(promptTokens) || promptTokens < 0) throw new TypeError('Öğretmen prompt token kullanımı geçersiz.')
    if (!Number.isSafeInteger(completionTokens) || completionTokens < 0) throw new TypeError('Öğretmen çıktı token kullanımı geçersiz.')
    const actualTokens = promptTokens + completionTokens
    if (!Number.isSafeInteger(actualTokens)) throw new TypeError('Öğretmen toplam token kullanımı geçersiz.')
    return this.backend.transaction(() => {
      const reservation = this.backend.get('teacherUsageReservations', id)
      if (!reservation) {
        const error = new Error('Öğretmen kullanım rezervasyonu bulunamadı.')
        error.code = 'TEACHER_USAGE_RESERVATION_MISSING'
        throw error
      }
      if (reservation.status === 'settled') {
        const same = reservation.promptTokens === promptTokens && reservation.completionTokens === completionTokens &&
          reservation.usageEstimated === Boolean(usageEstimated)
        if (!same) {
          const error = new Error('Öğretmen kullanım rezervasyonu farklı bir sonuçla zaten kapatılmış.')
          error.code = 'TEACHER_USAGE_SETTLEMENT_CONFLICT'
          throw error
        }
        return clone(reservation)
      }
      if (reservation.status !== 'reserved') {
        const error = new Error('Öğretmen kullanım rezervasyonu kapatılamaz durumda.')
        error.code = 'TEACHER_USAGE_RESERVATION_INVALID'
        throw error
      }
      const usage = this.backend.get('usage', reservation.usageDay) ?? {
        day: reservation.usageDay,
        jobs: 0,
        teacherRequests: 0,
        teacherTokens: 0,
      }
      const currentTokens = boundedUsageInteger(usage.teacherTokens)
      if (currentTokens < reservation.reservedTokens) {
        const error = new Error('Öğretmen token rezervasyonu kullanım toplamında bulunamadı.')
        error.code = 'TEACHER_USAGE_RESERVATION_MISSING'
        throw error
      }
      const settledTokens = currentTokens - reservation.reservedTokens + actualTokens
      if (!Number.isSafeInteger(settledTokens) || settledTokens < 0) {
        throw new TypeError('Öğretmen token kapatma toplamı güvenli tam sayı sınırını aşıyor.')
      }
      // The call has already happened. If a provider reports more than the
      // pre-reserved bound, record it even when this crosses today's quota;
      // subsequent reservations will then fail closed.
      usage.teacherTokens = settledTokens
      this.#putWithinBudget('usage', reservation.usageDay, usage)
      const settled = {
        ...reservation,
        status: 'settled',
        promptTokens,
        completionTokens,
        actualTokens,
        usageEstimated: Boolean(usageEstimated),
        settledAt: now(),
      }
      this.#putWithinBudget('teacherUsageReservations', id, settled)
      return clone(settled)
    })
  }
  consumeUsage(kind, amount = 1, day = new Date().toISOString().slice(0, 10)) {
    return this.backend.transaction(() => {
      const usage = this.getUsage(day)
      const limits = this.getConfig().budgets
      const mapping = { jobs: 'maxDailyJobs', teacherRequests: 'maxDailyTeacherRequests', teacherTokens: 'maxDailyTeacherTokens' }
      if (!(kind in mapping)) throw new TypeError('Bilinmeyen bütçe türü.')
      if (!Number.isSafeInteger(amount) || amount <= 0) throw new TypeError('Kullanım miktarı pozitif bir tam sayı olmalı.')
      if (usage[kind] + amount > limits[mapping[kind]]) {
        const error = new Error(`${kind} günlük bütçesi aşıldı.`)
        error.code = 'BUDGET_EXCEEDED'
        throw error
      }
      usage[kind] += amount
      this.#putWithinBudget('usage', day, usage)
      return clone(usage)
    })
  }
  consumeUsageBatch(reservations) {
    if (!Array.isArray(reservations) || !reservations.length) {
      throw new TypeError('Kullanım rezervasyonu en az bir kayıt içermeli.')
    }
    return this.backend.transaction(() => reservations.map((reservation) => {
      if (!reservation || typeof reservation !== 'object' || Array.isArray(reservation)) {
        throw new TypeError('Kullanım rezervasyonu geçersiz.')
      }
      return this.consumeUsage(reservation.kind, reservation.amount, reservation.day)
    }))
  }
  refundUsage(kind, amount = 1, day = new Date().toISOString().slice(0, 10)) {
    return this.backend.transaction(() => {
      const usage = this.getUsage(day)
      if (!(kind in usage) || typeof usage[kind] !== 'number') throw new TypeError('Bilinmeyen bütçe türü.')
      if (!Number.isSafeInteger(amount) || amount <= 0) throw new TypeError('İade miktarı pozitif bir tam sayı olmalı.')
      usage[kind] = Math.max(0, usage[kind] - amount)
      this.#putWithinBudget('usage', day, usage)
      return clone(usage)
    })
  }

  metrics() {
    const jobs = this.listJobs()
    const sources = this.listSources()
    const snapshots = this.listSnapshots()
    const count = (items, key, value) => items.filter((item) => item[key] === value).length
    return {
      sources: { total: sources.length, quarantine: count(sources, 'status', 'quarantine'), approved: count(sources, 'status', 'approved'), rejected: count(sources, 'status', 'rejected') },
      jobs: { total: jobs.length, queued: count(jobs, 'state', 'queued'), running: count(jobs, 'state', 'running'), paused: count(jobs, 'state', 'paused') + count(jobs, 'state', 'pause_requested'), failed: count(jobs, 'state', 'failed'), succeeded: count(jobs, 'state', 'succeeded') },
      snapshots: { total: snapshots.length, activeId: this.getActiveSnapshot()?.id ?? null, rejected: count(snapshots, 'status', 'rejected') },
      usage: this.getUsage(),
    }
  }
}

function without(value, keys) {
  const output = clone(value)
  for (const key of keys) delete output[key]
  return output
}

export async function createTrainingStore(options = {}) {
  const dataDirectory = path.resolve(options.dataDirectory ?? path.resolve('server-data'))
  const backend = await createBackend(dataDirectory, options)
  return new TrainingStore(backend, dataDirectory)
}
