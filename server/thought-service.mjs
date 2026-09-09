import { createHash, randomUUID } from 'node:crypto'
import {
  decodeArtifactValue,
  encodeArtifactValue,
  encodedByteLength,
  restoreArtifact,
} from '../runtime/artifacts.mjs'
import {
  createThoughtNeuronRoutingTopology,
  createThoughtNeuronResidencySnapshot,
  measureThoughtNeuronResidencyBytes,
  openThoughtNeuronResidency,
} from '../runtime/thought-neuron-residency.mjs'
import { createLanguageOutputScaffold } from '../src/language-output/scaffold.ts'
import { eventGraphToLanguageIntent } from '../src/language-output/intent.ts'
import { generateTurkishSentence } from '../src/language-output/generation.ts'
import { readLanguageOutputSnapshot } from '../runtime/language-output-cache.mjs'
import { PLASTIC_REGION_MODES } from '../runtime/plastic-region-cache.mjs'
import {
  resolvePlasticRegionRuntime,
} from './plastic-region-runtime.mjs'
import { reserveBoundedTeacherCall, settleBoundedTeacherCall } from './teacher-usage.mjs'
import { evaluateResourceState, MEBIBYTE, sampleHostResources } from './resource-policy.mjs'

export const OLLAMA_CROSS_CHECK_TRUST = 0.25
export const OLLAMA_CROSS_CHECK_MIN_CONFIDENCE = 0.85

const MAX_ID_LENGTH = 256
const MAX_LIST_LIMIT = 50
const MAX_TICK_PAGE_LIMIT = 128
const MAX_STORED_CANDIDATES_PER_TICK = 32
const MAX_PUBLIC_CANDIDATES_PER_TICK = 16
const MAX_BRANCHES_PER_TICK = 8
const MAX_FAMILIES_PER_TICK = 16
const MAX_COALITIONS_PER_TICK = 8
const ADAPTIVE_REGION_PAGE_COUNT = 56
const ADAPTIVE_REGION_BEHAVIOR_DIMENSION = 8
const MAX_ADAPTIVE_REGION_GATES = 8
const MAX_ADAPTIVE_REGION_CANDIDATES = 8
const MAX_GRAPH_NODES = 128
const MAX_GRAPH_EDGES = 96
const MAX_GRAPH_LIMIT = 96
const MAX_TEXT_PREVIEW = 500
const MAX_HEAD_DELTA_EDGES = 4_096
const MAX_MATERIALIZED_HEAD_EDGES = 250_000
const MAX_MATERIALIZED_EDGES_PER_ORIGIN = 96
const MAX_CONTEXT_PATCHES = 2_048
const MAX_BIRTHS = 128
const MAX_NURSERY_UPSERTS = 128
const MAX_MATERIALIZED_NURSERY = 4_096
const MAX_MATERIALIZED_BIRTHS = 4_096
const MAX_Q = 50_000_000
const Q_SCALE = 1_000_000
const TERMINAL_EPISODE_STATES = new Set(['completed', 'cancelled', 'failed'])
const BASE_ARTIFACT_KINDS = new Set([
  'family-run',
  'runtime-model-state',
  'text-lab-analysis',
  'synthesis-run',
  'thought-runtime-head',
  'concept-macro-head',
  'language-output-scaffold',
  'language-output-head',
])

function hashInteger(value, seed) {
  let hash = (Math.trunc(value) ^ seed) >>> 0
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d)
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b)
  return (hash ^ (hash >>> 16)) >>> 0
}

function digestVector(vector) {
  const buffer = new ArrayBuffer(4)
  const view = new DataView(buffer)
  let first = 0x811c9dc5
  let second = 0x9e3779b9
  for (let index = 0; index < vector.length; index += 1) {
    view.setFloat32(0, vector[index], true)
    const bits = view.getUint32(0, true)
    first = hashInteger(bits ^ index, first)
    second = hashInteger(bits ^ Math.imul(index + 1, 0x85ebca6b), second)
  }
  return `${first.toString(16).padStart(8, '0')}${second.toString(16).padStart(8, '0')}`
}

function canonicalThoughtValue(value) {
  if (value instanceof Float32Array) return ['Float32Array', ...Array.from(value, (entry) => Math.fround(entry))]
  if (Array.isArray(value)) return value.map(canonicalThoughtValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonicalThoughtValue(entry)]))
  }
  if (typeof value === 'number') return Number.isFinite(value) ? Math.fround(value) : String(value)
  return value
}

function thoughtDigest(label, ...parts) {
  const primes = [0x01000193, 0x27d4eb2d, 0x165667b1, 0x9e3779b1]
  const states = [0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35]
  const serialized = JSON.stringify(canonicalThoughtValue(['thought-runtime-v1', label, ...parts]))
  for (let index = 0; index < serialized.length; index += 1) {
    const code = serialized.charCodeAt(index)
    for (let stateIndex = 0; stateIndex < states.length; stateIndex += 1) {
      states[stateIndex] = Math.imul(
        (states[stateIndex] ^ code ^ Math.imul(index + 1, stateIndex + 19)) >>> 0,
        primes[stateIndex],
      ) >>> 0
      states[stateIndex] ^= states[stateIndex] >>> 13
    }
  }
  return states.map((state) => (state >>> 0).toString(16).padStart(8, '0')).join('')
}

function thoughtPartitionDigest(label, vector) {
  if (!(vector instanceof Float32Array) || vector.length < 1 || !vector.every(Number.isFinite)) {
    throw thoughtError(`${label} bölümü geçersiz.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  return thoughtDigest(label, vector)
}

function canonicalizeThoughtPrompt(value) {
  return String(value ?? '').replace(/\r\n?/g, '\n').normalize('NFC')
}

function defaultAdaptiveRegionSourceGroupId(inputText) {
  return `prompt:${thoughtDigest(
    'adaptive-default-source-group',
    canonicalizeThoughtPrompt(inputText),
  ).slice(0, 24)}`
}

const EVENT_GRAPH_KEYS = Object.freeze([
  'format', 'actor', 'action', 'patient', 'instrument', 'operators', 'negated', 'goals',
])
const GOAL_KEYS = Object.freeze(['relation', 'target'])

const eventGraphSchema = {
  type: 'object',
  additionalProperties: false,
  required: EVENT_GRAPH_KEYS,
  properties: {
    format: { type: 'string', enum: ['event-graph-v1'] },
    // The local schema helper intentionally leaves nullable fields to the
    // stricter canonical normalizer below.
    actor: {}, action: {}, patient: {}, instrument: {},
    operators: { type: 'array', maxItems: 16, items: { type: 'string', maxLength: 240 } },
    negated: { type: 'boolean' },
    goals: {
      type: 'array',
      maxItems: 16,
      items: {
        type: 'object',
        additionalProperties: false,
        required: GOAL_KEYS,
        properties: {
          relation: { type: 'string', minLength: 1, maxLength: 64 },
          target: { type: 'string', minLength: 1, maxLength: 240 },
        },
      },
    },
  },
}

const crossCheckSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['graph', 'confidence'],
  properties: {
    graph: eventGraphSchema,
    confidence: { type: 'number', minimum: 0, maximum: 1 },
  },
}

function thoughtError(message, code, status = 400, details) {
  const error = new Error(message)
  error.code = code
  error.status = status
  if (details !== undefined) error.details = details
  return error
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize)
  if (value && typeof value === 'object' && !ArrayBuffer.isView(value)) {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]))
  }
  return value
}

function digest(...values) {
  const hash = createHash('sha256')
  for (const value of values) {
    const serialized = typeof value === 'string' ? value : JSON.stringify(canonicalize(value))
    hash.update(String(Buffer.byteLength(serialized)))
    hash.update(':')
    hash.update(serialized)
    hash.update('\n')
  }
  return hash.digest('hex')
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value)
}

function boundedString(value, maximum = MAX_ID_LENGTH) {
  return typeof value === 'string' ? value.slice(0, maximum) : ''
}

function displayText(value, maximum = MAX_TEXT_PREVIEW) {
  const normalized = typeof value === 'string' ? value.replace(/\s+/gu, ' ').trim() : ''
  if (normalized.length <= maximum) return { text: normalized, truncated: false }
  return { text: `${normalized.slice(0, maximum - 1).trimEnd()}…`, truncated: true }
}

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function probability(value) {
  const numeric = finite(value)
  return numeric === null ? null : Math.max(0, Math.min(1, numeric))
}

function nonNegativeInteger(value, fallback = 0) {
  return Number.isSafeInteger(value) && value >= 0 ? value : fallback
}

function parseInteger(value, { name, fallback, minimum, maximum }) {
  if (value === undefined || value === null || value === '') return fallback
  if (!/^-?\d+$/u.test(String(value))) {
    throw thoughtError(`${name} bir tam sayı olmalı.`, 'INVALID_THOUGHT_QUERY', 400)
  }
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw thoughtError(`${name} ${minimum}–${maximum} aralığında olmalı.`, 'INVALID_THOUGHT_QUERY', 400)
  }
  return parsed
}

function exactKeys(value, allowed, label, status = 422) {
  if (!isRecord(value)) throw thoughtError(`${label} bir nesne olmalı.`, 'INVALID_EVENT_GRAPH', status)
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key))
  const missing = allowed.filter((key) => !Object.hasOwn(value, key))
  if (unknown.length || missing.length) {
    throw thoughtError(`${label} bilinmeyen veya eksik alan içeriyor.`, 'INVALID_EVENT_GRAPH', status, {
      unknown: unknown.slice(0, 8), missing: missing.slice(0, 8),
    })
  }
}

function normalizedTerm(value, { nullable = false, maximum = 240, label }) {
  if (nullable && value === null) return null
  if (typeof value !== 'string') throw thoughtError(`${label} metin${nullable ? ' veya null' : ''} olmalı.`, 'INVALID_EVENT_GRAPH', 422)
  const normalized = value.normalize('NFC').replace(/\s+/gu, ' ').trim().toLocaleLowerCase('tr-TR')
  if (!normalized || normalized.length > maximum || /[\u0000-\u001f\u007f]/u.test(normalized)) {
    throw thoughtError(`${label} boş veya sınır dışında.`, 'INVALID_EVENT_GRAPH', 422)
  }
  return normalized
}

export function normalizeEventGraph(value) {
  exactKeys(value, EVENT_GRAPH_KEYS, 'expectedOutcome')
  if (value.format !== 'event-graph-v1') {
    throw thoughtError('expectedOutcome formatı event-graph-v1 olmalı.', 'INVALID_EVENT_GRAPH', 422)
  }
  if (!Array.isArray(value.operators) || value.operators.length > 16) {
    throw thoughtError('expectedOutcome.operators en fazla 16 öğe içermeli.', 'INVALID_EVENT_GRAPH', 422)
  }
  if (!Array.isArray(value.goals) || value.goals.length > 16) {
    throw thoughtError('expectedOutcome.goals en fazla 16 öğe içermeli.', 'INVALID_EVENT_GRAPH', 422)
  }
  if (typeof value.negated !== 'boolean') {
    throw thoughtError('expectedOutcome.negated boolean olmalı.', 'INVALID_EVENT_GRAPH', 422)
  }
  const operators = [...new Set(value.operators.map((item, index) => normalizedTerm(item, {
    label: `expectedOutcome.operators[${index}]`,
  })))].sort()
  const goals = value.goals.map((goal, index) => {
    exactKeys(goal, GOAL_KEYS, `expectedOutcome.goals[${index}]`)
    const relation = normalizedTerm(goal.relation, { label: `expectedOutcome.goals[${index}].relation`, maximum: 64 })
    if (!/^[\p{L}\p{N}][\p{L}\p{N}_-]{0,63}$/u.test(relation)) {
      throw thoughtError('Amaç ilişkisi yalnız harf, sayı, _ veya - içerebilir.', 'INVALID_EVENT_GRAPH', 422)
    }
    return {
      relation,
      target: normalizedTerm(goal.target, { label: `expectedOutcome.goals[${index}].target` }),
    }
  }).sort((left, right) => left.relation.localeCompare(right.relation) || left.target.localeCompare(right.target))
  const graph = {
    format: 'event-graph-v1',
    actor: normalizedTerm(value.actor, { nullable: true, label: 'expectedOutcome.actor' }),
    action: normalizedTerm(value.action, { nullable: true, label: 'expectedOutcome.action' }),
    patient: normalizedTerm(value.patient, { nullable: true, label: 'expectedOutcome.patient' }),
    instrument: normalizedTerm(value.instrument, { nullable: true, label: 'expectedOutcome.instrument' }),
    operators,
    negated: value.negated,
    goals,
  }
  if (
    graph.actor === null && graph.action === null && graph.patient === null && graph.instrument === null &&
    !graph.operators.length && !graph.negated && !graph.goals.length
  ) throw thoughtError('expectedOutcome en az bir yapısal olay alanı taşımalı.', 'EMPTY_EVENT_GRAPH', 422)
  return graph
}

function normalizeOllamaGraph(value) {
  try {
    return normalizeEventGraph(value)
  } catch (error) {
    error.code = 'OLLAMA_EVENT_GRAPH_INVALID'
    error.status = 502
    throw error
  }
}

function normalizeId(value, label, { required = false } = {}) {
  if (value === undefined || value === null || value === '') {
    if (required) throw thoughtError(`${label} gerekli.`, 'INVALID_THOUGHT_REQUEST', 400)
    return null
  }
  if (typeof value !== 'string' || value.length > MAX_ID_LENGTH || /[\u0000-\u001f\u007f]/u.test(value)) {
    throw thoughtError(`${label} geçersiz.`, 'INVALID_THOUGHT_REQUEST', 400)
  }
  return value
}

function normalizeMemberKey(value, label, { required = true } = {}) {
  if (value === undefined || value === null || value === '') {
    if (required) throw thoughtError(`${label} gerekli.`, 'INVALID_THOUGHT_REQUEST', 400)
    return null
  }
  if (typeof value !== 'string' || value.length > 1_024 || /[\u0001-\u001f\u007f]/u.test(value)) {
    throw thoughtError(`${label} geçersiz.`, 'INVALID_THOUGHT_REQUEST', 400)
  }
  return value
}

function normalizeStartRequest(body, config) {
  if (!isRecord(body)) throw thoughtError('Düşünce isteği bir nesne olmalı.', 'INVALID_THOUGHT_REQUEST', 400)
  for (const untrusted of ['teacherEvidence', 'teacherConfidence', 'confidence', 'trust', 'trustWeight']) {
    if (Object.hasOwn(body, untrusted)) {
      throw thoughtError('Teacher güveni istemci tarafından belirlenemez.', 'UNTRUSTED_TEACHER_METADATA', 422)
    }
  }
  const allowed = new Set([
    'text', 'prompt', 'inputText', 'input', 'snapshotId', 'seed', 'maxTicks', 'strategy',
    'learningMode', 'teacherMode', 'crossCheck', 'expectedOutcome', 'sourceGroupId', 'idempotencyKey', 'candidateMemberKeys',
    'regionMode',
  ])
  const unknown = Object.keys(body).filter((key) => !allowed.has(key))
  if (unknown.length) throw thoughtError('Düşünce isteği bilinmeyen alan içeriyor.', 'INVALID_THOUGHT_REQUEST', 400, { unknown })
  if (body.input !== undefined && (!isRecord(body.input) || Object.keys(body.input).some((key) => key !== 'text'))) {
    throw thoughtError('input yalnız text alanını taşıyabilir.', 'INVALID_THOUGHT_REQUEST', 400)
  }
  const text = body.text ?? body.prompt ?? body.inputText ?? body.input?.text
  if (typeof text !== 'string') throw thoughtError('text alanı gerekli.', 'INVALID_THOUGHT_REQUEST', 400)
  const normalizedText = text.normalize('NFC').trim()
  if (!normalizedText || normalizedText.length > config.thought.maximumInputCharacters) {
    throw thoughtError(`text 1–${config.thought.maximumInputCharacters} karakter arasında olmalı.`, 'INVALID_THOUGHT_REQUEST', 400)
  }
  const seed = body.seed ?? 1
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffff_ffff) {
    throw thoughtError('seed 0–4294967295 aralığında bir tam sayı olmalı.', 'INVALID_THOUGHT_REQUEST', 400)
  }
  const maxTicks = body.maxTicks ?? Math.min(64, config.thought.maximumTicks)
  if (!Number.isInteger(maxTicks) || maxTicks < 1 || maxTicks > config.thought.maximumTicks) {
    throw thoughtError(`maxTicks 1–${config.thought.maximumTicks} aralığında olmalı.`, 'INVALID_THOUGHT_REQUEST', 400)
  }
  const strategy = body.strategy ?? 'seeded-sample'
  if (!['greedy', 'seeded-sample'].includes(strategy)) {
    throw thoughtError('strategy greedy veya seeded-sample olmalı.', 'INVALID_THOUGHT_REQUEST', 400)
  }
  const learningMode = body.learningMode ?? 'off'
  if (!['off', 'proposal'].includes(learningMode)) {
    throw thoughtError('learningMode off veya proposal olmalı.', 'INVALID_THOUGHT_REQUEST', 400)
  }
  if (body.crossCheck !== undefined && typeof body.crossCheck !== 'boolean') {
    throw thoughtError('crossCheck boolean olmalı.', 'INVALID_THOUGHT_REQUEST', 400)
  }
  if (body.teacherMode !== undefined && !['off', 'ollama-cross-check'].includes(body.teacherMode)) {
    throw thoughtError('teacherMode off veya ollama-cross-check olmalı.', 'INVALID_THOUGHT_REQUEST', 400)
  }
  if (
    body.teacherMode !== undefined && body.crossCheck !== undefined &&
    (body.teacherMode === 'ollama-cross-check') !== body.crossCheck
  ) throw thoughtError('teacherMode ile crossCheck aliası birbiriyle çelişiyor.', 'INVALID_THOUGHT_REQUEST', 422)
  const teacherMode = body.teacherMode ?? ((body.crossCheck ?? config.thought.ollamaCrossCheck) ? 'ollama-cross-check' : 'off')
  const requestedRegionMode = body.regionMode ?? config.plasticRegions?.mode ?? 'off'
  if (!PLASTIC_REGION_MODES.includes(requestedRegionMode)) {
    throw thoughtError('regionMode off, shadow, proposal veya active olmalı.', 'INVALID_THOUGHT_REQUEST', 400)
  }
  let candidateMemberKeys = []
  if (body.candidateMemberKeys !== undefined) {
    if (!Array.isArray(body.candidateMemberKeys) || body.candidateMemberKeys.length > 128) {
      throw thoughtError('candidateMemberKeys en fazla 128 kimlik içermeli.', 'INVALID_THOUGHT_REQUEST', 400)
    }
    candidateMemberKeys = [...new Set(body.candidateMemberKeys.map((key) => normalizeMemberKey(key, 'candidateMemberKeys öğesi')))].sort()
  }
  return {
    inputText: normalizedText,
    snapshotId: normalizeId(body.snapshotId, 'snapshotId'),
    seed,
    maxTicks,
    strategy,
    learningMode,
    teacherMode,
    requestedRegionMode,
    crossCheckRequested: teacherMode === 'ollama-cross-check',
    expectedOutcome: body.expectedOutcome === undefined || body.expectedOutcome === null
      ? null
      : normalizeEventGraph(body.expectedOutcome),
    sourceGroupId: normalizeId(body.sourceGroupId, 'sourceGroupId'),
    idempotencyKey: normalizeId(body.idempotencyKey, 'idempotencyKey'),
    candidateMemberKeys,
  }
}

function snapshotFamilyArtifact(store, snapshot) {
  const artifacts = (snapshot?.artifactIds ?? []).map((id) => (
    typeof store.getArtifactMetadata === 'function' ? store.getArtifactMetadata(id) : store.getArtifact(id)
  )).filter(Boolean)
  return artifacts.find((artifact) => artifact.kind === 'family-run') ??
    [...artifacts].reverse().find((artifact) => artifact.kind === 'runtime-model-state') ?? null
}

function ancestorHeadCandidate(store, snapshot) {
  const visited = new Set([snapshot.id])
  let parentId = snapshot.parentId ?? null
  for (let depth = 0; parentId && depth < 32; depth += 1) {
    if (visited.has(parentId)) throw thoughtError('Snapshot ebeveyn zinciri döngülü.', 'INVALID_THOUGHT_BASE', 409)
    visited.add(parentId)
    const parent = store.getSnapshot(parentId)
    if (!parent) return null
    const metadata = (parent.artifactIds ?? []).map((id) => (
      typeof store.getArtifactMetadata === 'function' ? store.getArtifactMetadata(id) : store.getArtifact(id)
    )).filter(Boolean)
    const headMetadata = [...metadata].reverse().find((artifact) => artifact.kind === 'thought-runtime-head') ?? null
    if (headMetadata) {
      const familyMetadata = metadata.find((artifact) => artifact.kind === 'family-run') ??
        [...metadata].reverse().find((artifact) => artifact.kind === 'runtime-model-state') ?? null
      const headArtifact = store.getArtifact(headMetadata.id)
      const familyArtifact = familyMetadata ? store.getArtifact(familyMetadata.id) : null
      const familyValue = familyArtifact ? artifactValue(familyArtifact) : null
      const ancestorFamilyRun = familyArtifact?.kind === 'family-run' ? familyValue : familyValue?.familyRun
      return { snapshot: parent, headArtifact, familyRun: ancestorFamilyRun ?? null }
    }
    parentId = parent.parentId ?? null
  }
  return null
}

function selectBaseSnapshot(store, requestedSnapshotId) {
  const active = store.getActiveSnapshot() ?? null
  if (requestedSnapshotId) {
    const snapshot = store.getSnapshot(requestedSnapshotId)
    if (!snapshot) throw thoughtError('İstenen düşünce tabanı bulunamadı.', 'SNAPSHOT_NOT_FOUND', 404)
    if (!snapshotFamilyArtifact(store, snapshot)) {
      throw thoughtError('İstenen model sürümü bir aile çalışması taşımıyor.', 'THOUGHT_BASE_NOT_FOUND', 409)
    }
    return { snapshot, active, selectedBy: 'requested-snapshot' }
  }
  if (active && snapshotFamilyArtifact(store, active)) return { snapshot: active, active, selectedBy: 'active-snapshot' }
  const latest = [...store.listSnapshots()]
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))
    .find((snapshot) => snapshotFamilyArtifact(store, snapshot))
  if (!latest) throw thoughtError('Düşünce motoru için denetlenebilir bir aile modeli bulunamadı.', 'THOUGHT_BASE_NOT_FOUND', 409)
  return { snapshot: latest, active, selectedBy: 'latest-family-snapshot' }
}

function artifactValue(artifact) {
  try {
    return restoreArtifact(artifact, artifact.kind)
  } catch (cause) {
    throw thoughtError('Düşünce tabanındaki model eseri çözülemedi.', 'INVALID_THOUGHT_BASE_ARTIFACT', 409, {
      artifactId: artifact?.id,
      kind: artifact?.kind,
      cause: cause instanceof Error ? cause.message : String(cause),
    })
  }
}

function modelManifest(store, snapshot) {
  return (snapshot.artifactIds ?? [])
    .map((id) => typeof store.getArtifactMetadata === 'function' ? store.getArtifactMetadata(id) : store.getArtifact(id))
    .filter(Boolean)
    .map((artifact) => ({ id: artifact.id, kind: artifact.kind, contentHash: artifact.contentHash ?? null }))
    .sort((left, right) => left.id.localeCompare(right.id))
}

function thoughtHydrationMetadata(store, snapshot) {
  const retained = new Map()
  const addSnapshot = (candidate, kinds = BASE_ARTIFACT_KINDS) => {
    const rows = (candidate?.artifactIds ?? []).map((id) => (
      typeof store.getArtifactMetadata === 'function' ? store.getArtifactMetadata(id) : null
    )).filter((metadata) => metadata && kinds.has(metadata.kind))
    for (const metadata of rows) retained.set(metadata.id, metadata)
    return rows
  }
  const current = addSnapshot(snapshot)
  if (current.some((metadata) => metadata.kind === 'thought-runtime-head')) return [...retained.values()]
  const visited = new Set([snapshot.id])
  let parentId = snapshot.parentId ?? null
  for (let depth = 0; parentId && depth < 32; depth += 1) {
    if (visited.has(parentId)) break
    visited.add(parentId)
    const parent = store.getSnapshot(parentId)
    if (!parent) break
    const rows = addSnapshot(parent, new Set(['thought-runtime-head', 'family-run', 'runtime-model-state']))
    if (rows.some((metadata) => metadata.kind === 'thought-runtime-head')) break
    parentId = parent.parentId ?? null
  }
  return [...retained.values()]
}

function hydrationPayloadBytes(metadata) {
  const bytes = Number(metadata?.payloadBytes)
  return Number.isSafeInteger(bytes) && bytes >= 0 ? bytes : 0
}

function materializedHeadFingerprint(head) {
  return thoughtDigest(
    'head',
    1,
    head.revision,
    head.baseModelFingerprint,
    head.totalTicks,
    head.neuronEdges,
    head.contextPatches.map((patch) => ({ ...patch, delta: thoughtPartitionDigest('relation', patch.delta) })),
    head.nursery.map((candidate) => ({
      ...candidate,
      identity: thoughtPartitionDigest('identity', candidate.identity),
      relationPrototype: thoughtPartitionDigest('relation', candidate.relationPrototype),
    })),
    head.births,
  )
}

function emptyThoughtHead(baseModelFingerprint) {
  const head = {
    format: 'bombom-thought-head',
    version: 1,
    revision: 0,
    baseModelFingerprint,
    totalTicks: 0,
    neuronEdges: [],
    contextPatches: [],
    nursery: [],
    births: [],
  }
  return { ...head, fingerprint: materializedHeadFingerprint(head) }
}

function migrateExactMemberExperience(
  sourceHead,
  baseModelFingerprint,
  exactMemberKeys,
  { layout, currentFamilyIds, newBaseMemberKeys },
) {
  const provisionalFamily = (identityDigest) =>
    `nursery-provisional:${thoughtDigest('nursery-family', identityDigest).slice(0, 16)}`
  const filterMemberships = (memberships, identityDigest) => {
    const retained = (memberships ?? [])
      .filter((membership) => (
        (currentFamilyIds.has(membership.familyId) || String(membership.familyId).startsWith('nursery-provisional:')) &&
        (finite(membership.weight) ?? 0) > 0
      ))
      .map((membership) => ({ familyId: membership.familyId, weight: Math.max(0, finite(membership.weight) ?? 0) }))
      .sort((left, right) => right.weight - left.weight || left.familyId.localeCompare(right.familyId))
      .slice(0, 5)
    if (!retained.length) return [{ familyId: provisionalFamily(identityDigest), weight: 1 }]
    const total = retained.reduce((sum, membership) => sum + membership.weight, 0)
    return retained.map((membership) => ({ ...membership, weight: membership.weight / total }))
  }
  const retainedBirths = []
  let quarantinedBirthCount = 0
  for (const birth of sourceHead.births) {
    const identity = birth.identity instanceof Float32Array ? Float32Array.from(birth.identity) : Float32Array.from(birth.identity ?? [])
    const relation = birth.relation instanceof Float32Array ? Float32Array.from(birth.relation) : Float32Array.from(birth.relation ?? [])
    if (
      newBaseMemberKeys.has(birth.id) || identity.length !== layout.identityLength || relation.length !== layout.relationLength ||
      identity.some((value) => !Number.isFinite(value)) || relation.some((value) => !Number.isFinite(value)) ||
      thoughtPartitionDigest('identity', identity) !== birth.identityDigest
    ) {
      quarantinedBirthCount += 1
      continue
    }
    retainedBirths.push({
      ...birth,
      identity,
      relation,
      familyMemberships: filterMemberships(birth.familyMemberships, birth.identityDigest),
      sourceGroupIds: [...(birth.sourceGroupIds ?? [])],
      ...(birth.contextEvidence ? {
        contextEvidence: {
          observationCount: nonNegativeInteger(birth.contextEvidence.observationCount),
          sourceGroupIds: [...(birth.contextEvidence.sourceGroupIds ?? [])],
        },
      } : {}),
    })
  }
  const retainedBirthIds = new Set(retainedBirths.map((birth) => birth.id))
  const durableIds = new Set([...exactMemberKeys, ...retainedBirthIds])
  const filterMemberEvidence = (entries) => (entries ?? [])
    .filter((entry) => durableIds.has(entry.memberKey) && (finite(entry.weight) ?? 0) > 0)
    .map((entry) => ({ memberKey: entry.memberKey, weight: Math.max(0, finite(entry.weight) ?? 0) }))
    .sort((left, right) => left.memberKey.localeCompare(right.memberKey))
  const births = retainedBirths.map((birth) => ({
    ...birth,
    ...(birth.entryEvidence === undefined ? {} : { entryEvidence: filterMemberEvidence(birth.entryEvidence) }),
    ...(birth.outgoingEvidence === undefined ? {} : { outgoingEvidence: filterMemberEvidence(birth.outgoingEvidence) }),
  })).sort((left, right) => left.id.localeCompare(right.id))

  let quarantinedNurseryCount = 0
  const nursery = []
  for (const candidate of sourceHead.nursery) {
    const identity = candidate.identity instanceof Float32Array
      ? Float32Array.from(candidate.identity) : Float32Array.from(candidate.identity ?? [])
    const relationPrototype = candidate.relationPrototype instanceof Float32Array
      ? Float32Array.from(candidate.relationPrototype) : Float32Array.from(candidate.relationPrototype ?? [])
    if (
      identity.length !== layout.identityLength || relationPrototype.length !== layout.relationLength ||
      identity.some((value) => !Number.isFinite(value)) || relationPrototype.some((value) => !Number.isFinite(value)) ||
      thoughtPartitionDigest('identity', identity) !== candidate.identityDigest
    ) {
      quarantinedNurseryCount += 1
      continue
    }
    const familyEvidence = (candidate.familyEvidence ?? [])
      .filter((entry) => currentFamilyIds.has(entry.key) || String(entry.key).startsWith('nursery-provisional:'))
      .map((entry) => ({ ...entry }))
    if (!familyEvidence.length) familyEvidence.push({ key: provisionalFamily(candidate.identityDigest), weight: 1 })
    nursery.push({
      ...candidate,
      identity,
      relationPrototype,
      birthEpisodeIds: [...candidate.birthEpisodeIds],
      independentSourceIds: [...candidate.independentSourceIds],
      originEvidence: (candidate.originEvidence ?? [])
        .filter((entry) => durableIds.has(entry.key)).map((entry) => ({ ...entry })),
      ...(candidate.outgoingEvidence === undefined ? {} : {
        outgoingEvidence: candidate.outgoingEvidence
          .filter((entry) => durableIds.has(entry.key)).map((entry) => ({ ...entry })),
      }),
      familyEvidence,
      sourceRelationPrototypes: (candidate.sourceRelationPrototypes ?? []).map((entry) => ({
        ...entry,
        relation: Float32Array.from(entry.relation),
      })),
    })
  }
  const neuronEdges = retainMaterializedEdges(sourceHead.neuronEdges
    .filter((edge) => durableIds.has(edge.fromMemberKey) && durableIds.has(edge.toMemberKey))
    .map((edge) => ({ ...edge })))
  const contextPatches = sourceHead.contextPatches
    .filter((patch) => durableIds.has(patch.neuronId))
    .map((patch) => ({ ...patch, delta: Float32Array.from(patch.delta) }))
    .sort((left, right) => left.neuronId.localeCompare(right.neuronId))
  const migrated = {
    format: 'bombom-thought-head',
    version: 1,
    revision: 0,
    baseModelFingerprint,
    // Preserve the global tick clock so promotion cooldowns cannot be reset
    // or artificially delayed by a compatible FamilyRun refresh.
    totalTicks: sourceHead.totalTicks,
    neuronEdges,
    contextPatches,
    nursery: nursery.sort((left, right) => left.id.localeCompare(right.id)),
    births,
  }
  const head = { ...migrated, fingerprint: materializedHeadFingerprint(migrated) }
  return {
    head,
    diagnostics: {
      retainedNurseryCount: nursery.length,
      quarantinedNurseryCount,
      retainedBirthCount: births.length,
      quarantinedBirthCount,
    },
  }
}

function edgeExperienceScore(edge) {
  return Math.max(0, edge.excitation - edge.inhibition)
}

export function retainMaterializedEdges(edges) {
  const byOrigin = new Map()
  for (const edge of edges) {
    if (!byOrigin.has(edge.fromMemberKey)) byOrigin.set(edge.fromMemberKey, [])
    byOrigin.get(edge.fromMemberKey).push(edge)
  }
  const retained = []
  for (const originEdges of byOrigin.values()) {
    retained.push(...originEdges.sort((left, right) => (
      edgeExperienceScore(right) - edgeExperienceScore(left) ||
      right.trustMass - left.trustMass ||
      right.updateCount - left.updateCount ||
      left.toMemberKey.localeCompare(right.toMemberKey)
    )).slice(0, MAX_MATERIALIZED_EDGES_PER_ORIGIN))
  }
  return retained.sort((left, right) => (
    edgeExperienceScore(right) - edgeExperienceScore(left) ||
    right.trustMass - left.trustMass ||
    right.updateCount - left.updateCount ||
    left.fromMemberKey.localeCompare(right.fromMemberKey) || left.toMemberKey.localeCompare(right.toMemberKey)
  )).slice(0, MAX_MATERIALIZED_HEAD_EDGES)
    .sort((left, right) => left.fromMemberKey.localeCompare(right.fromMemberKey) || left.toMemberKey.localeCompare(right.toMemberKey))
}

function validateMaterializedHead(value, baseModelFingerprint, layout) {
  if (
    !isRecord(value) || value.format !== 'bombom-thought-head' || value.version !== 1 ||
    value.baseModelFingerprint !== baseModelFingerprint || !Number.isSafeInteger(value.revision) || value.revision < 0 ||
    !Number.isSafeInteger(value.totalTicks) || value.totalTicks < 0 ||
    !Array.isArray(value.neuronEdges) || value.neuronEdges.length > MAX_MATERIALIZED_HEAD_EDGES ||
    !Array.isArray(value.contextPatches) || value.contextPatches.length > MAX_CONTEXT_PATCHES ||
    !Array.isArray(value.nursery) || value.nursery.length > MAX_MATERIALIZED_NURSERY ||
    !Array.isArray(value.births) || value.births.length > MAX_MATERIALIZED_BIRTHS
  ) throw thoughtError('Kalıcı düşünce başlığı biçim veya kapasite sınırını aşmış.', 'INVALID_THOUGHT_HEAD', 409)
  const head = {
    format: 'bombom-thought-head', version: 1, revision: value.revision,
    baseModelFingerprint, totalTicks: value.totalTicks,
    neuronEdges: value.neuronEdges.map((edge) => ({
      fromMemberKey: normalizeMemberKey(edge?.fromMemberKey, 'thoughtHead edge from'),
      toMemberKey: normalizeMemberKey(edge?.toMemberKey, 'thoughtHead edge to'),
      excitation: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(edge?.excitation) ?? 0)),
      inhibition: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(edge?.inhibition) ?? 0)),
      trustMass: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(edge?.trustMass) ?? 0)),
      updateCount: nonNegativeInteger(edge?.updateCount),
    })).sort((left, right) => left.fromMemberKey.localeCompare(right.fromMemberKey) || left.toMemberKey.localeCompare(right.toMemberKey)),
    contextPatches: value.contextPatches.map((patch) => ({
      neuronId: normalizeMemberKey(patch?.neuronId ?? patch?.memberKey, 'thoughtHead context neuron'),
      delta: vectorFrom(patch?.delta ?? patch?.relationDelta, 'thoughtHead context delta', layout.relationLength),
      weight: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(patch?.weight) ?? 0)),
      updateCount: nonNegativeInteger(patch?.updateCount),
    })).sort((left, right) => left.neuronId.localeCompare(right.neuronId)),
    nursery: value.nursery.map((candidate, index) => materializedNurseryCandidate(candidate, layout, `thoughtHead.nursery[${index}]`)),
    births: value.births.map((birth, index) => materializedBirth(birth, layout, `thoughtHead.births[${index}]`)),
  }
  const edgeCounts = new Map()
  const edgeKeys = new Set()
  for (const edge of head.neuronEdges) {
    const key = `${edge.fromMemberKey}\u0000${edge.toMemberKey}`
    if (edgeKeys.has(key)) throw thoughtError('Kalıcı düşünce başlığı yinelenen kenar içeriyor.', 'INVALID_THOUGHT_HEAD', 409)
    edgeKeys.add(key)
    edgeCounts.set(edge.fromMemberKey, (edgeCounts.get(edge.fromMemberKey) ?? 0) + 1)
  }
  if ([...edgeCounts.values()].some((count) => count > MAX_MATERIALIZED_EDGES_PER_ORIGIN)) {
    throw thoughtError('Kalıcı düşünce başlığı köken başına 96 experience kenarı sınırını aşıyor.', 'INVALID_THOUGHT_HEAD', 409)
  }
  const fingerprint = materializedHeadFingerprint(head)
  if (value.fingerprint !== fingerprint) throw thoughtError('Kalıcı düşünce başlığının fingerprint kanıtı uyuşmuyor.', 'INVALID_THOUGHT_HEAD', 409)
  return { ...head, fingerprint }
}

function applyHeadDelta(head, delta, layout, episode = {}) {
  if (!delta || delta.format !== 'thought-head-delta' || delta.version !== 1) {
    throw thoughtError('Düşünce başlığı delta eseri eksik veya uyumsuz.', 'INVALID_THOUGHT_HEAD', 409)
  }
  if (delta.baseModelFingerprint !== head.baseModelFingerprint || delta.baseRevision !== head.revision || delta.baseHeadFingerprint !== head.fingerprint) {
    throw thoughtError('Düşünce başlığı delta zincirinin tabanı uyuşmuyor.', 'INVALID_THOUGHT_HEAD_CHAIN', 409)
  }
  const edgeMap = new Map(head.neuronEdges.map((edge) => [`${edge.fromMemberKey}\u0000${edge.toMemberKey}`, { ...edge }]))
  for (const edge of delta.neuronEdges ?? []) {
    const key = `${edge.fromMemberKey}\u0000${edge.toMemberKey}`
    const previous = edgeMap.get(key) ?? {
      fromMemberKey: edge.fromMemberKey, toMemberKey: edge.toMemberKey,
      excitation: 0, inhibition: 0, trustMass: 0, updateCount: 0,
    }
    edgeMap.set(key, {
      ...previous,
      excitation: Math.min(MAX_Q / Q_SCALE, previous.excitation + edge.excitationQ / Q_SCALE),
      inhibition: Math.min(MAX_Q / Q_SCALE, previous.inhibition + edge.inhibitionQ / Q_SCALE),
      trustMass: Math.min(MAX_Q / Q_SCALE, previous.trustMass + edge.trustMassQ / Q_SCALE),
      updateCount: previous.updateCount + nonNegativeInteger(edge.updateCount),
    })
  }
  const patchMap = new Map(head.contextPatches.map((patch) => [patch.neuronId, { ...patch, delta: Float32Array.from(patch.delta) }]))
  for (const patch of delta.contextPatches ?? []) {
    const relationDelta = vectorFrom(patch.relationDelta, 'thought head delta context', layout.relationLength, -1, 1)
    const previous = patchMap.get(patch.memberKey) ?? {
      neuronId: patch.memberKey, delta: new Float32Array(layout.relationLength), weight: 0, updateCount: 0,
    }
    const merged = new Float32Array(layout.relationLength)
    for (let index = 0; index < merged.length; index += 1) {
      merged[index] = Math.fround(Math.max(-1, Math.min(1, previous.delta[index] + relationDelta[index])))
    }
    patchMap.set(patch.memberKey, {
      neuronId: patch.memberKey,
      delta: merged,
      weight: Math.min(MAX_Q / Q_SCALE, previous.weight + patch.trustMassQ / Q_SCALE),
      updateCount: previous.updateCount + nonNegativeInteger(patch.observationCount),
    })
  }
  const nurseryMap = new Map(head.nursery.map((candidate) => [candidate.id, materializedNurseryCandidate(candidate, layout, `head.nursery.${candidate.id}`)]))
  for (const candidate of delta.nurseryUpserts ?? []) {
    nurseryMap.set(candidate.id, materializedNurseryCandidate({
      ...candidate,
      supportWeight: candidate.supportWeightQ / Q_SCALE,
      trustMass: candidate.trustMassQ / Q_SCALE,
      resultScoreTotal: candidate.resultScoreTotalQ / Q_SCALE,
      originEvidence: (candidate.originEvidence ?? []).map((entry) => ({ key: entry.key, weight: entry.weightQ / Q_SCALE })),
      ...(candidate.outgoingEvidence ? {
        outgoingEvidence: candidate.outgoingEvidence.map((entry) => ({ key: entry.key, weight: entry.weightQ / Q_SCALE })),
      } : {}),
      familyEvidence: (candidate.familyEvidence ?? []).map((entry) => ({ key: entry.key, weight: entry.weightQ / Q_SCALE })),
      residualCohesion: candidate.residualCohesionQ / Q_SCALE,
      sourceRelationPrototypes: (candidate.sourceRelationPrototypes ?? []).map((entry) => ({
        sourceGroupId: entry.sourceGroupId,
        relation: entry.relation,
        observationCount: entry.observationCount,
      })),
    }, layout, `headDelta.nursery.${candidate.id}`))
  }
  for (const nurseryId of delta.nurseryRemoveIds ?? []) nurseryMap.delete(nurseryId)
  const birthMap = new Map(head.births.map((birth) => [birth.id, birth]))
  for (const birth of delta.births ?? []) {
    if (birthMap.has(birth.id)) throw thoughtError('Aynı nöron doğumu düşünce başlığına iki kez eklenemez.', 'INVALID_THOUGHT_HEAD_CHAIN', 409)
    birthMap.set(birth.id, {
      format: 'born-neuron-v1', id: birth.id,
      identity: Float32Array.from(birth.identity), relation: Float32Array.from(birth.relation),
      identityDigest: birth.identityDigest,
      familyMemberships: (birth.familyMemberships ?? []).map((membership) => ({ familyId: membership.familyId, weight: membership.weightQ / Q_SCALE })),
      sourceGroupIds: [...(birth.sourceGroupIds ?? [])],
      trustMass: birth.trustMassQ / Q_SCALE,
      bornAtTick: nonNegativeInteger(birth.bornAtTick),
      nurseryId: birth.nurseryId,
      ...(birth.entryEvidence ? {
        entryEvidence: birth.entryEvidence.map((entry) => ({ memberKey: entry.memberKey, weight: entry.weightQ / Q_SCALE })),
      } : {}),
      ...(birth.outgoingEvidence ? {
        outgoingEvidence: birth.outgoingEvidence.map((entry) => ({ memberKey: entry.memberKey, weight: entry.weightQ / Q_SCALE })),
      } : {}),
      ...(birth.contextEvidence ? {
        contextEvidence: {
          observationCount: nonNegativeInteger(birth.contextEvidence.observationCount),
          sourceGroupIds: [...(birth.contextEvidence.sourceGroupIds ?? [])],
        },
      } : {}),
    })
  }
  const next = {
    format: 'bombom-thought-head', version: 1, revision: head.revision + 1,
    baseModelFingerprint: head.baseModelFingerprint,
    totalTicks: head.totalTicks + nonNegativeInteger(episode.tickCount ?? delta.stats?.totalTicks),
    neuronEdges: retainMaterializedEdges([...edgeMap.values()]),
    contextPatches: [...patchMap.values()].sort((left, right) => left.neuronId.localeCompare(right.neuronId)),
    nursery: [...nurseryMap.values()]
      .sort((left, right) => right.trustMass - left.trustMass || right.observationCount - left.observationCount || left.id.localeCompare(right.id))
      .slice(0, MAX_MATERIALIZED_NURSERY)
      .sort((left, right) => left.id.localeCompare(right.id)),
    births: [...birthMap.values()].sort((left, right) => left.id.localeCompare(right.id)),
  }
  return { ...next, fingerprint: materializedHeadFingerprint(next) }
}

function restoreThoughtHead(store, artifact, baseModelFingerprint, layout) {
  if (!artifact) return emptyThoughtHead(baseModelFingerprint)
  const visited = new Set()
  const deltas = []
  let cursor = artifact
  for (let depth = 0; depth <= 64; depth += 1) {
    if (!cursor || visited.has(cursor.id)) throw thoughtError('Düşünce başlığı zinciri eksik veya döngülü.', 'INVALID_THOUGHT_HEAD_CHAIN', 409)
    visited.add(cursor.id)
    const value = artifactValue(cursor)
    if (value?.format === 'bombom-thought-head') {
      let head = validateMaterializedHead(value, baseModelFingerprint, layout)
      for (const delta of deltas.reverse()) head = applyHeadDelta(head, delta, layout)
      return head
    }
    if (value?.format !== 'thought-runtime-head' || value.version !== 1 || value.baseModelFingerprint !== baseModelFingerprint) {
      throw thoughtError('Düşünce başlığı zinciri temel modelle uyuşmuyor.', 'INVALID_THOUGHT_HEAD_CHAIN', 409)
    }
    if (value.deltaArtifactId) {
      const deltaArtifact = store.getArtifact(value.deltaArtifactId)
      if (!deltaArtifact || deltaArtifact.kind !== 'thought-head-proposal') {
        throw thoughtError('Düşünce başlığı zincirinde delta eseri bulunamadı.', 'INVALID_THOUGHT_HEAD_CHAIN', 409)
      }
      deltas.push(artifactValue(deltaArtifact))
    }
    if (!value.parentHeadArtifactId) {
      let head = emptyThoughtHead(baseModelFingerprint)
      for (const delta of deltas.reverse()) head = applyHeadDelta(head, delta, layout)
      return head
    }
    cursor = store.getArtifact(value.parentHeadArtifactId)
  }
  throw thoughtError('Düşünce başlığı zinciri 64 revizyon sınırını aşıyor.', 'INVALID_THOUGHT_HEAD_CHAIN', 409)
}

function resolveEngineBase(store, selection) {
  const { snapshot, active, selectedBy } = selection
  // Resolve headers first so unrelated trace/graph artifacts are never
  // hydrated merely because they share a snapshot with the model.
  const artifacts = (snapshot.artifactIds ?? [])
    .map((id) => typeof store.getArtifactMetadata === 'function' ? store.getArtifactMetadata(id) : store.getArtifact(id))
    .filter((metadata) => metadata && BASE_ARTIFACT_KINDS.has(metadata.kind))
    .map((metadata) => store.getArtifact(metadata.id))
    .filter(Boolean)
  const dedicatedFamilyArtifact = artifacts.find((artifact) => artifact.kind === 'family-run') ?? null
  const runtimeArtifact = [...artifacts].reverse().find((artifact) => artifact.kind === 'runtime-model-state') ?? null
  const runtimeModel = runtimeArtifact ? artifactValue(runtimeArtifact) : null
  const familyRun = dedicatedFamilyArtifact ? artifactValue(dedicatedFamilyArtifact) : runtimeModel?.familyRun
  if (
    !familyRun || familyRun.format !== 'noron-aileleri-run' || familyRun.version !== 1 ||
    !Array.isArray(familyRun.profiles) || !Array.isArray(familyRun.memberships) || !Array.isArray(familyRun.neuronTransitions)
  ) throw thoughtError('Düşünce tabanındaki FamilyRun eksik veya uyumsuz.', 'INVALID_THOUGHT_BASE', 409)
  const profileKeys = new Set(familyRun.profiles.map((profile) => profile.memberKey))
  const profileByKey = new Map(familyRun.profiles.map((profile) => [profile.memberKey, profile]))
  const neurons = new Map()
  let layout = null
  const rawRepresentationProfileEvidence = new Set()
  let resolvedRawCount = 0
  let resolvedSynthesisCount = 0
  for (const artifact of artifacts.filter((item) => item.kind === 'text-lab-analysis')) {
    const analysis = artifactValue(artifact)
    if (!analysis || !Array.isArray(analysis.nodes) || !analysis.layout) continue
    const includedNodes = analysis.nodes.filter((node) => profileKeys.has(`${analysis.sourceDigest}\u0000${analysis.runId}\u0000${node.id}`))
    if (!includedNodes.length) continue
    rawRepresentationProfileEvidence.add(
      typeof analysis.config?.representationProfile === 'string'
        ? analysis.config.representationProfile
        : null,
    )
    const candidateLayout = analysis.layout
    if (
      analysis.representationSpaceId !== familyRun.representationSpaceId ||
      candidateLayout.side !== 64 || candidateLayout.side * candidateLayout.side !== 4_096 || candidateLayout.cellCount !== 4_096 ||
      !Number.isInteger(candidateLayout.identityLength) || !Number.isInteger(candidateLayout.relationLength) ||
      candidateLayout.relationStart !== candidateLayout.identityLength ||
      candidateLayout.identityLength + candidateLayout.relationLength !== 4_096 ||
      (layout && (
        layout.side !== candidateLayout.side ||
        layout.identityLength !== candidateLayout.identityLength ||
        layout.relationLength !== candidateLayout.relationLength
      ))
    ) throw thoughtError('FamilyRun ile nöron I/R matrisleri aynı 64x64 temsil uzayında değil.', 'REPRESENTATION_SPACE_MISMATCH', 409, {
      artifactId: artifact.id,
      analysisRunId: analysis.runId,
      expectedRepresentationSpaceId: familyRun.representationSpaceId,
      actualRepresentationSpaceId: analysis.representationSpaceId,
    })
    layout ??= { ...candidateLayout }
    for (const node of analysis.nodes) {
      const memberKey = `${analysis.sourceDigest}\u0000${analysis.runId}\u0000${node.id}`
      if (!profileKeys.has(memberKey) || neurons.has(memberKey)) continue
      if (
        !(node.identity instanceof Float32Array) || !(node.relation instanceof Float32Array) ||
        node.identity.length !== layout.identityLength || node.relation.length !== layout.relationLength ||
        node.identity.some((item) => !Number.isFinite(item)) || node.relation.some((item) => !Number.isFinite(item)) ||
        digestVector(node.identity) !== node.identityDigest ||
        profileByKey.get(memberKey)?.identityDigest !== node.identityDigest
      ) throw thoughtError('FamilyRun içindeki ham nöronun I/R bütünlük kanıtı doğrulanamadı.', 'REPRESENTATION_SPACE_MISMATCH', 409, {
        artifactId: artifact.id, memberKey,
      })
      neurons.set(memberKey, {
        memberKey,
        kind: 'raw',
        text: boundedString(node.text, 2_000),
        identity: node.identity,
        relation: node.relation,
        identityDigest: node.identityDigest,
        contextSupport: nonNegativeInteger(node.contextSupport),
        contextRevision: nonNegativeInteger(node.contextRevision),
        sourceDigest: analysis.sourceDigest,
        analysisRunId: analysis.runId,
        neuronId: node.id,
      })
      resolvedRawCount += 1
    }
  }
  for (const artifact of artifacts.filter((item) => item.kind === 'synthesis-run')) {
    const run = artifactValue(artifact)
    const includedCandidates = (Array.isArray(run?.candidates) ? run.candidates : [])
      .filter((candidate) => profileKeys.has(`${run.representationSpaceId}\u0000${candidate.fingerprint}`))
    if (includedCandidates.length && run.representationSpaceId !== familyRun.representationSpaceId) {
      throw thoughtError('Sentez nöronu FamilyRun temsil uzayıyla uyuşmuyor.', 'REPRESENTATION_SPACE_MISMATCH', 409, { artifactId: artifact.id })
    }
    for (const candidate of Array.isArray(run?.candidates) ? run.candidates : []) {
      const memberKey = `${run.representationSpaceId}\u0000${candidate.fingerprint}`
      if (!profileKeys.has(memberKey) || neurons.has(memberKey)) continue
      if (
        !layout || !(candidate.identity instanceof Float32Array) || !(candidate.relation instanceof Float32Array) ||
        candidate.identity.length !== layout.identityLength || candidate.relation.length !== layout.relationLength ||
        candidate.identity.some((item) => !Number.isFinite(item)) || candidate.relation.some((item) => !Number.isFinite(item)) ||
        digestVector(candidate.identity) !== candidate.identityDigest ||
        profileByKey.get(memberKey)?.identityDigest !== candidate.identityDigest
      ) throw thoughtError('Sentez nöronunun I/R bütünlük kanıtı doğrulanamadı.', 'REPRESENTATION_SPACE_MISMATCH', 409, {
        artifactId: artifact.id, memberKey,
      })
      neurons.set(memberKey, {
        memberKey,
        kind: 'synthesis',
        text: boundedString(candidate.medoid?.text, 2_000),
        identity: candidate.identity,
        relation: candidate.relation,
        identityDigest: candidate.identityDigest,
        contextSupport: nonNegativeInteger(candidate.contextSupport),
        contextRevision: nonNegativeInteger(candidate.contextRevision),
        candidateFingerprint: candidate.fingerprint,
      })
      resolvedSynthesisCount += 1
    }
  }
  const unresolvedSynthesis = familyRun.profiles
    .filter((profile) => profile.kind === 'synthesis' && !neurons.has(profile.memberKey))
    .map((profile) => profile.memberKey)
    .sort()
  const unresolvedRaw = familyRun.profiles
    .filter((profile) => profile.kind === 'raw' && !neurons.has(profile.memberKey))
    .map((profile) => profile.memberKey)
    .sort()
  if (!layout || unresolvedRaw.length) {
    throw thoughtError('FamilyRun ham nöronlarının tam I/R kataloğu snapshot içinde bulunamadı.', 'REPRESENTATION_SPACE_MISMATCH', 409, {
      unresolvedRawMemberKeys: unresolvedRaw.slice(0, 32),
      unresolvedRawTruncated: unresolvedRaw.length > 32,
    })
  }
  const representationProfile = rawRepresentationProfileEvidence.size === 1
    ? [...rawRepresentationProfileEvidence][0]
    : null
  const manifest = modelManifest(store, snapshot)
  const artifactManifestHash = digest('thought-base-artifacts-v1', manifest)
  const languageState = runtimeModel?.languageState?.familyRunFingerprint === familyRun.fingerprint
    ? runtimeModel.languageState
    : null
  const outputSnapshot = readLanguageOutputSnapshot(store, snapshot)
  const languageOutputScaffold = outputSnapshot.scaffold ?? createLanguageOutputScaffold()
  const languageOutputHead = outputSnapshot.head &&
    (outputSnapshot.head.familyRunFingerprint === null || outputSnapshot.head.familyRunFingerprint === familyRun.fingerprint) &&
    (outputSnapshot.head.representationSpaceId === null || outputSnapshot.head.representationSpaceId === familyRun.representationSpaceId)
    ? outputSnapshot.head
    : null
  // Runtime macros are strictly snapshot-bound. Never consult the global
  // newest/research cache here: only an artifact explicitly attached to this
  // immutable snapshot may compete with raw routing.
  const conceptMacroHeadArtifact = [...artifacts].reverse()
    .find((artifact) => artifact.kind === 'concept-macro-head') ?? null
  let conceptMacroHead = null
  let conceptMacroDiagnostics = conceptMacroHeadArtifact
    ? { status: 'attached-unverified', artifactId: conceptMacroHeadArtifact.id, reason: null }
    : { status: 'absent', artifactId: null, reason: 'snapshot-head-absent' }
  if (conceptMacroHeadArtifact) {
    try {
      const value = restoreArtifact(conceptMacroHeadArtifact, 'concept-macro-head')
      if (value?.format === 'concept-macro-head' && value.version === 1) {
        conceptMacroHead = value
        conceptMacroDiagnostics = { status: 'attached', artifactId: conceptMacroHeadArtifact.id, reason: null }
      } else {
        conceptMacroDiagnostics = {
          status: 'quarantined', artifactId: conceptMacroHeadArtifact.id, reason: 'invalid-format-version',
        }
      }
    } catch (error) {
      conceptMacroDiagnostics = {
        status: 'quarantined', artifactId: conceptMacroHeadArtifact.id,
        reason: boundedString(error?.message ?? String(error), 240),
      }
    }
  }
  const baseModelFingerprint = digest(
    'thought-base-model-v1',
    familyRun.fingerprint,
    familyRun.representationSpaceId,
    layout,
  )
  let thoughtHeadArtifact = [...artifacts].reverse().find((artifact) => artifact.kind === 'thought-runtime-head') ?? null
  let headCarry = thoughtHeadArtifact
    ? { status: 'snapshot-owned', fromSnapshotId: snapshot.id, sourceArtifactId: thoughtHeadArtifact.id }
    : { status: 'empty', fromSnapshotId: null, sourceArtifactId: null }
  let restoreHeadArtifact = thoughtHeadArtifact
  let migratedThoughtHead = null
  if (!thoughtHeadArtifact) {
    const ancestor = ancestorHeadCandidate(store, snapshot)
    if (ancestor?.familyRun?.fingerprint === familyRun.fingerprint) {
      restoreHeadArtifact = ancestor.headArtifact
      headCarry = {
        status: 'direct-family-fingerprint-carry',
        fromSnapshotId: ancestor.snapshot.id,
        sourceArtifactId: ancestor.headArtifact.id,
      }
    } else if (ancestor?.familyRun?.representationSpaceId === familyRun.representationSpaceId) {
      const ancestorBaseModelFingerprint = digest(
        'thought-base-model-v1',
        ancestor.familyRun.fingerprint,
        ancestor.familyRun.representationSpaceId,
        layout,
      )
      try {
        const ancestorHead = restoreThoughtHead(store, ancestor.headArtifact, ancestorBaseModelFingerprint, layout)
        const ancestorProfiles = new Map((ancestor.familyRun.profiles ?? []).map((profile) => [profile.memberKey, profile]))
        const exactMemberKeys = new Set([...profileByKey.entries()]
          .filter(([memberKey, profile]) => (
            neurons.has(memberKey) &&
            typeof profile?.identityDigest === 'string' &&
            ancestorProfiles.get(memberKey)?.identityDigest === profile.identityDigest
          ))
          .map(([memberKey]) => memberKey))
        const migration = migrateExactMemberExperience(ancestorHead, baseModelFingerprint, exactMemberKeys, {
          layout,
          currentFamilyIds: new Set((familyRun.families ?? []).map((family) => family.id)),
          newBaseMemberKeys: new Set(neurons.keys()),
        })
        migratedThoughtHead = migration.head
        headCarry = {
          status: 'filtered-exact-member-migration',
          fromSnapshotId: ancestor.snapshot.id,
          sourceArtifactId: ancestor.headArtifact.id,
          exactMemberCount: exactMemberKeys.size,
          retainedEdgeCount: migratedThoughtHead.neuronEdges.length,
          retainedContextPatchCount: migratedThoughtHead.contextPatches.length,
          ...migration.diagnostics,
          sourceRevision: ancestorHead.revision,
        }
      } catch (error) {
        headCarry = {
          status: 'family-changed-migration-reset',
          fromSnapshotId: ancestor.snapshot.id,
          sourceArtifactId: ancestor.headArtifact.id,
          reasonCode: boundedString(error?.code ?? 'INCOMPATIBLE_ANCESTOR_HEAD', 100),
        }
      }
    } else if (ancestor?.headArtifact) {
      headCarry = {
        status: 'representation-mismatch-reset',
        fromSnapshotId: ancestor.snapshot.id,
        sourceArtifactId: ancestor.headArtifact.id,
      }
    }
  }
  const thoughtHead = migratedThoughtHead ?? restoreThoughtHead(store, restoreHeadArtifact, baseModelFingerprint, layout)
  const commitEligible = active?.id === snapshot.id &&
    snapshot.evaluation?.passed === true && snapshot.evaluation?.runtimeVerified === true
  return {
    reference: {
      snapshotId: snapshot.id,
      snapshotStatus: snapshot.status,
      selectedBy,
      activeAtStart: active?.id === snapshot.id,
      commitEligible,
      artifactIds: [...(snapshot.artifactIds ?? [])],
      artifactManifestHash,
      baseModelFingerprint,
      familyRunFingerprint: familyRun.fingerprint,
      languageStateFingerprint: languageState?.fingerprint ?? null,
      thoughtHeadArtifactId: thoughtHeadArtifact?.id ?? null,
      thoughtHeadFingerprint: thoughtHead.fingerprint,
      thoughtHeadRevision: thoughtHead.revision,
      conceptMacroHeadArtifactId: conceptMacroHeadArtifact?.id ?? null,
      conceptMacroHeadFingerprint: conceptMacroHead?.fingerprint ?? null,
      ...(representationProfile ? { representationProfile } : {}),
    },
    snapshot,
    familyRun,
    languageState,
    languageOutputScaffold,
    languageOutputHead,
    thoughtHead,
    conceptMacroHead,
    neurons: [...neurons.values()].sort((left, right) => left.memberKey.localeCompare(right.memberKey)),
    neuronByKey: neurons,
    layout,
    representationProfile,
    diagnostics: {
      profileCount: familyRun.profiles.length,
      resolvedNeuronCount: neurons.size,
      resolvedRawCount,
      resolvedSynthesisCount,
      excludedSynthesisCount: unresolvedSynthesis.length,
      excludedSynthesisMemberKeys: unresolvedSynthesis.slice(0, 128),
      excludedSynthesisTruncated: unresolvedSynthesis.length > 128,
      headCarry,
      conceptMacros: conceptMacroDiagnostics,
      representationProfile: {
        value: representationProfile,
        explicit: representationProfile !== null,
        evidenceCount: rawRepresentationProfileEvidence.size,
      },
      languageOutput: {
        scaffoldFingerprint: languageOutputScaffold.fingerprint,
        headFingerprint: languageOutputHead?.fingerprint ?? null,
        headRevision: nonNegativeInteger(languageOutputHead?.revision),
        persisted: Boolean(outputSnapshot.scaffold),
        compatible: Boolean(languageOutputHead),
        mismatch: outputSnapshot.mismatch,
      },
    },
  }
}

/**
 * Read-only adapter for deterministic, snapshot-bound evaluators. It exposes
 * the same integrity-checked I/R catalog used by ThoughtService without
 * creating an episode or mutating the selected snapshot.
 */
export function resolveThoughtEngineBase(store, snapshotId) {
  if (typeof snapshotId !== 'string' || !snapshotId.trim()) {
    throw thoughtError('Benchmark için açık bir snapshot kimliği gerekli.', 'SNAPSHOT_NOT_FOUND', 404)
  }
  return resolveEngineBase(store, selectBaseSnapshot(store, snapshotId.trim()))
}

function artifactSpec(kind, value, { metadata = {}, maximumBytes, contentParts = [] } = {}) {
  const encoded = encodeArtifactValue(value)
  const bytes = encodedByteLength(encoded)
  if (maximumBytes !== undefined && bytes > maximumBytes) {
    throw thoughtError('Düşünce eseri bölüm boyutu sınırını aşıyor.', 'THOUGHT_ARTIFACT_TOO_LARGE', 413, { kind, bytes, maximumBytes })
  }
  return {
    kind,
    contentHash: digest(kind, ...contentParts, encoded),
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      bytes,
      metadata,
      value: encoded,
    },
  }
}

function strictBoundedValue(value, state = { depth: 0, counter: { nodes: 0 } }) {
  state.counter.nodes += 1
  if (state.counter.nodes > 100_000 || state.depth > 10) throw thoughtError('Motor çıktısı yapısal sınırı aşıyor.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    if (typeof value === 'string' && value.length > 2_000) return value.slice(0, 2_000)
    return value
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw thoughtError('Motor çıktısı sonlu olmayan sayı içeriyor.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
    return value
  }
  if (ArrayBuffer.isView(value)) {
    if (!(value instanceof Float32Array) || value.length > 4_096 || value.some((item) => !Number.isFinite(item))) {
      throw thoughtError('Motor checkpoint vektörü geçersiz veya fazla büyük.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
    }
    return Float32Array.from(value)
  }
  if (Array.isArray(value)) {
    if (value.length > 4_096) throw thoughtError('Motor çıktısı dizi sınırını aşıyor.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
    return value.map((item) => strictBoundedValue(item, { depth: state.depth + 1, counter: state.counter }))
  }
  if (!isRecord(value) || Object.keys(value).length > 128) {
    throw thoughtError('Motor çıktısı geçersiz nesne içeriyor.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
  }
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    boundedString(key, 128), strictBoundedValue(item, { depth: state.depth + 1, counter: state.counter }),
  ]))
}

function qValue(value, q, label) {
  if (q !== undefined) {
    if (!Number.isSafeInteger(q) || q < 0 || q > MAX_Q) {
      throw thoughtError(`${label}Q güvenli tam sayı sınırında değil.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
    }
    return q
  }
  if (value === undefined) return 0
  if (!Number.isFinite(value) || value < 0 || value * Q_SCALE > MAX_Q + 0.5) {
    throw thoughtError(`${label} 0–${MAX_Q / Q_SCALE} aralığında olmalı.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  return Math.round(value * Q_SCALE)
}

function vectorFrom(value, label, expectedLength, minimum = -2, maximum = 2) {
  let decoded = value
  if (isRecord(value) && value.$type) decoded = decodeArtifactValue(value)
  const vector = decoded instanceof Float32Array
    ? Float32Array.from(decoded)
    : Array.isArray(decoded) ? Float32Array.from(decoded) : null
  if (!vector || vector.length !== expectedLength || vector.some((item) => !Number.isFinite(item) || item < minimum || item > maximum)) {
    throw thoughtError(`${label} representation relation uzunluğu veya sayı sınırıyla uyuşmuyor.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  return vector
}

function boundedEvidenceEntries(value, label, { memberKeys = false } = {}) {
  if (!Array.isArray(value) || value.length > 64) {
    throw thoughtError(`${label} en fazla 64 kanıt girdisi taşımalı.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  const entries = value.map((entry, index) => ({
    key: memberKeys
      ? normalizeMemberKey(entry?.key, `${label}[${index}].key`)
      : normalizeId(entry?.key, `${label}[${index}].key`, { required: true }),
    weight: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(entry?.weight) ?? 0)),
  })).sort((left, right) => left.key.localeCompare(right.key))
  if (new Set(entries.map((entry) => entry.key)).size !== entries.length) {
    throw thoughtError(`${label} yinelenen kanıt anahtarı içeriyor.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  return entries
}

function boundedMemberEvidenceEntries(value, label) {
  if (!Array.isArray(value) || value.length > 64) {
    throw thoughtError(`${label} en fazla 64 nöron kanıtı taşımalı.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  const entries = value.map((entry, index) => ({
    memberKey: normalizeMemberKey(entry?.memberKey ?? entry?.key, `${label}[${index}].memberKey`),
    weight: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(entry?.weight) ?? 0)),
  })).sort((left, right) => left.memberKey.localeCompare(right.memberKey))
  if (new Set(entries.map((entry) => entry.memberKey)).size !== entries.length) {
    throw thoughtError(`${label} yinelenen nöron kanıtı içeriyor.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  return entries
}

function materializedNurseryCandidate(candidate, layout, label) {
  if (!isRecord(candidate)) throw thoughtError(`${label} geçersiz.`, 'INVALID_THOUGHT_HEAD', 409)
  const identity = vectorFrom(candidate.identity, `${label}.identity`, layout.identityLength, -1, 1)
  const relationPrototype = vectorFrom(candidate.relationPrototype, `${label}.relationPrototype`, layout.relationLength, -1, 1)
  const identityDigest = normalizeId(candidate.identityDigest, `${label}.identityDigest`, { required: true })
  if (thoughtPartitionDigest('identity', identity) !== identityDigest) {
    throw thoughtError(`${label} sabit I digest'i uyuşmuyor.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  const birthEpisodeIds = [...new Set((Array.isArray(candidate.birthEpisodeIds) ? candidate.birthEpisodeIds : [])
    .map((id) => normalizeId(id, `${label}.birthEpisodeIds`, { required: true })))].sort()
  const independentSourceIds = [...new Set((Array.isArray(candidate.independentSourceIds) ? candidate.independentSourceIds : [])
    .map((id) => normalizeId(id, `${label}.independentSourceIds`, { required: true })))].sort()
  if (birthEpisodeIds.length > 64 || independentSourceIds.length > 64) {
    throw thoughtError(`${label} episode/kaynak kanıtı kapasite sınırını aşıyor.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  const rawSourceRelations = Array.isArray(candidate.sourceRelationPrototypes) ? candidate.sourceRelationPrototypes : []
  if (rawSourceRelations.length > 64) {
    throw thoughtError(`${label} en fazla 64 kaynak-dengeli R prototipi taşımalı.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  const sourceRelationPrototypes = rawSourceRelations.map((entry, index) => ({
    sourceGroupId: normalizeId(entry?.sourceGroupId, `${label}.sourceRelationPrototypes[${index}].sourceGroupId`, { required: true }),
    relation: vectorFrom(entry?.relation, `${label}.sourceRelationPrototypes[${index}].relation`, layout.relationLength, -1, 1),
    observationCount: Math.max(1, nonNegativeInteger(entry?.observationCount, 1)),
  })).sort((left, right) => left.sourceGroupId.localeCompare(right.sourceGroupId))
  if (new Set(sourceRelationPrototypes.map((entry) => entry.sourceGroupId)).size !== sourceRelationPrototypes.length) {
    throw thoughtError(`${label} yinelenen kaynak R prototipi içeriyor.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  if (sourceRelationPrototypes.some((entry) => !independentSourceIds.includes(entry.sourceGroupId))) {
    throw thoughtError(`${label} kaynak R prototipi bağımsız kaynak kanıtında bulunmuyor.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  if (sourceRelationPrototypes.length !== independentSourceIds.length) {
    throw thoughtError(`${label} her bağımsız kaynak için tam bir R prototipi taşımalı.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  if (sourceRelationPrototypes.length) {
    const sourceBalancedRelation = new Float32Array(layout.relationLength)
    for (const source of sourceRelationPrototypes) {
      for (let index = 0; index < sourceBalancedRelation.length; index += 1) {
        sourceBalancedRelation[index] = Math.fround(
          sourceBalancedRelation[index] + source.relation[index] / sourceRelationPrototypes.length,
        )
      }
    }
    for (let index = 0; index < relationPrototype.length; index += 1) {
      if (Math.abs(relationPrototype[index] - sourceBalancedRelation[index]) > 2e-6) {
        throw thoughtError(
          `${label}.relationPrototype kaynak gruplarının eşit ağırlıklı R ortalamasıyla uyuşmuyor.`,
          'INVALID_THOUGHT_HEAD',
          409,
        )
      }
    }
  }
  const residualCohesion = probability(candidate.residualCohesion ?? candidate.cohesion)
  if (residualCohesion === null) throw thoughtError(`${label}.residualCohesion 0–1 arasında olmalı.`, 'INVALID_THOUGHT_HEAD', 409)
  const outgoingEvidence = candidate.outgoingEvidence === undefined
    ? null
    : boundedEvidenceEntries(candidate.outgoingEvidence, `${label}.outgoingEvidence`, { memberKeys: true })
  return {
    id: normalizeId(candidate.id, `${label}.id`, { required: true }),
    identity,
    relationPrototype,
    identityDigest,
    birthEpisodeIds,
    independentSourceIds,
    observationCount: nonNegativeInteger(candidate.observationCount),
    residualObservationCount: nonNegativeInteger(candidate.residualObservationCount),
    supportWeight: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(candidate.supportWeight) ?? 0)),
    trustMass: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(candidate.trustMass) ?? 0)),
    resultScoreTotal: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(candidate.resultScoreTotal) ?? 0)),
    originEvidence: boundedEvidenceEntries(candidate.originEvidence ?? [], `${label}.originEvidence`, { memberKeys: true }),
    ...(outgoingEvidence ? { outgoingEvidence } : {}),
    familyEvidence: boundedEvidenceEntries(candidate.familyEvidence ?? [], `${label}.familyEvidence`),
    nextPromotionTick: nonNegativeInteger(candidate.nextPromotionTick),
    residualCohesion,
    sourceRelationPrototypes,
  }
}

function materializedBirth(birth, layout, label) {
  if (!isRecord(birth)) throw thoughtError(`${label} geçersiz.`, 'INVALID_THOUGHT_HEAD', 409)
  const identity = vectorFrom(birth.identity, `${label}.identity`, layout.identityLength, -1, 1)
  const relation = vectorFrom(birth.relation, `${label}.relation`, layout.relationLength, -1, 1)
  const identityDigest = normalizeId(birth.identityDigest, `${label}.identityDigest`, { required: true })
  if (thoughtPartitionDigest('identity', identity) !== identityDigest) {
    throw thoughtError(`${label} sabit I digest'i uyuşmuyor.`, 'INVALID_THOUGHT_HEAD', 409)
  }
  const memberships = Array.isArray(birth.familyMemberships) ? birth.familyMemberships : []
  if (memberships.length > 5) throw thoughtError(`${label} en fazla 5 aile üyeliği taşımalı.`, 'INVALID_THOUGHT_HEAD', 409)
  const familyMemberships = memberships.map((membership, index) => ({
    familyId: normalizeId(membership?.familyId, `${label}.familyMemberships[${index}].familyId`, { required: true }),
    weight: probability(membership?.weight) ?? 0,
  })).sort((left, right) => left.familyId.localeCompare(right.familyId))
  const sourceGroupIds = [...new Set((Array.isArray(birth.sourceGroupIds) ? birth.sourceGroupIds : [])
    .map((id) => normalizeId(id, `${label}.sourceGroupIds`, { required: true })))].sort()
  if (sourceGroupIds.length > 64) throw thoughtError(`${label} kaynak kanıtı sınırını aşıyor.`, 'INVALID_THOUGHT_HEAD', 409)
  const entryEvidence = birth.entryEvidence === undefined
    ? null
    : boundedMemberEvidenceEntries(birth.entryEvidence, `${label}.entryEvidence`)
  const outgoingEvidence = birth.outgoingEvidence === undefined
    ? null
    : boundedMemberEvidenceEntries(birth.outgoingEvidence, `${label}.outgoingEvidence`)
  const contextEvidence = birth.contextEvidence === undefined ? null : (() => {
    if (!isRecord(birth.contextEvidence)) {
      throw thoughtError(`${label}.contextEvidence geçersiz.`, 'INVALID_THOUGHT_HEAD', 409)
    }
    const contextSourceGroupIds = [...new Set((Array.isArray(birth.contextEvidence.sourceGroupIds)
      ? birth.contextEvidence.sourceGroupIds : [])
      .map((id) => normalizeId(id, `${label}.contextEvidence.sourceGroupIds`, { required: true })))].sort()
    if (
      contextSourceGroupIds.length > 64 ||
      contextSourceGroupIds.some((id) => !sourceGroupIds.includes(id))
    ) throw thoughtError(`${label}.contextEvidence kaynak kanıtıyla uyuşmuyor.`, 'INVALID_THOUGHT_HEAD', 409)
    const observationCount = nonNegativeInteger(birth.contextEvidence.observationCount)
    if (observationCount < contextSourceGroupIds.length) {
      throw thoughtError(`${label}.contextEvidence gözlem sayısı kaynak sayısından küçük.`, 'INVALID_THOUGHT_HEAD', 409)
    }
    return { observationCount, sourceGroupIds: contextSourceGroupIds }
  })()
  return {
    format: 'born-neuron-v1',
    id: normalizeId(birth.id, `${label}.id`, { required: true }),
    identity,
    relation,
    identityDigest,
    familyMemberships,
    sourceGroupIds,
    trustMass: Math.min(MAX_Q / Q_SCALE, Math.max(0, finite(birth.trustMass) ?? 0)),
    bornAtTick: nonNegativeInteger(birth.bornAtTick),
    nurseryId: normalizeId(birth.nurseryId, `${label}.nurseryId`, { required: true }),
    ...(entryEvidence ? { entryEvidence } : {}),
    ...(outgoingEvidence ? { outgoingEvidence } : {}),
    ...(contextEvidence ? { contextEvidence } : {}),
  }
}

function normalizeHeadDelta(delta, base, episode, teacherEvidence, engineRunKey = null) {
  if (!isRecord(delta)) throw thoughtError('Motor headDelta çıktısı bir nesne olmalı.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
  const allowed = new Set(['format', 'version', 'neuronEdges', 'contextPatches', 'nurseryUpserts', 'nurseryRemoveIds', 'births', 'stats'])
  const unknown = Object.keys(delta).filter((key) => !allowed.has(key))
  if (unknown.length) throw thoughtError('headDelta bilinmeyen alan içeriyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422, { unknown })
  const edges = Array.isArray(delta.neuronEdges) ? delta.neuronEdges : []
  const contextPatches = Array.isArray(delta.contextPatches) ? delta.contextPatches : []
  const nurseryUpserts = Array.isArray(delta.nurseryUpserts) ? delta.nurseryUpserts : []
  const nurseryRemoveIds = Array.isArray(delta.nurseryRemoveIds) ? delta.nurseryRemoveIds : []
  const births = Array.isArray(delta.births) ? delta.births : []
  if (
    edges.length > MAX_HEAD_DELTA_EDGES || contextPatches.length > MAX_CONTEXT_PATCHES ||
    nurseryUpserts.length > MAX_NURSERY_UPSERTS || nurseryRemoveIds.length > MAX_NURSERY_UPSERTS || births.length > MAX_BIRTHS
  ) {
    throw thoughtError('headDelta seyrek kapasite sınırını aşıyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  const existingBirthById = new Map(base.thoughtHead.births.map((birth) => [birth.id, birth]))
  const pendingBirthIds = new Set(births.map((birth, index) => normalizeId(birth?.id, `births[${index}].id`, { required: true })))
  const durableMemberKeys = new Set([
    ...base.neuronByKey.keys(),
    ...existingBirthById.keys(),
    ...pendingBirthIds,
  ])
  const maximumTrustQ = Math.round((teacherEvidence?.trustWeight ?? 0) * Q_SCALE)
  const edgeKeys = new Set()
  const neuronEdges = edges.map((edge, index) => {
    if (!isRecord(edge)) throw thoughtError(`neuronEdges[${index}] geçersiz.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const fromMemberKey = normalizeMemberKey(edge.fromMemberKey, `neuronEdges[${index}].fromMemberKey`)
    const toMemberKey = normalizeMemberKey(edge.toMemberKey, `neuronEdges[${index}].toMemberKey`)
    if (!durableMemberKeys.has(fromMemberKey) || !durableMemberKeys.has(toMemberKey)) {
      throw thoughtError('Kalıcı nöron kenarı katalog/nursery terfisi dışında dangling kimlik içeriyor.', 'THOUGHT_DANGLING_MEMBER_KEY', 422)
    }
    const key = `${fromMemberKey}\u0000${toMemberKey}`
    if (edgeKeys.has(key)) throw thoughtError('headDelta yinelenen nöron kenarı içeriyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    edgeKeys.add(key)
    const trustMassQ = qValue(edge.trustMass, edge.trustMassQ, `neuronEdges[${index}].trustMass`)
    if (trustMassQ > maximumTrustQ) {
      throw thoughtError('Nöron kenarı dış öğretmen güveninden fazla kredi taşıyamaz.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    }
    return {
      fromMemberKey,
      toMemberKey,
      excitationQ: qValue(edge.excitation, edge.excitationQ, `neuronEdges[${index}].excitation`),
      inhibitionQ: qValue(edge.inhibition, edge.inhibitionQ, `neuronEdges[${index}].inhibition`),
      trustMassQ,
      updateCount: Math.max(1, nonNegativeInteger(edge.updateCount, 1)),
    }
  }).sort((left, right) => left.fromMemberKey.localeCompare(right.fromMemberKey) || left.toMemberKey.localeCompare(right.toMemberKey))
  const patches = contextPatches.map((patch, index) => {
    if (!isRecord(patch)) throw thoughtError(`contextPatches[${index}] geçersiz.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const memberKey = normalizeMemberKey(patch.memberKey, `contextPatches[${index}].memberKey`)
    const neuron = base.neuronByKey.get(memberKey) ?? existingBirthById.get(memberKey)
    if (!neuron) throw thoughtError('Bağlam patch’i katalogda olmayan bir nörona başvuruyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const relationDelta = vectorFrom(
      patch.relationDelta ?? patch.relationDeltaEncoded,
      `contextPatches[${index}].relationDelta`,
      neuron.relation.length,
      -1,
      1,
    )
    const trustMassQ = qValue(patch.trustMass ?? patch.weight, patch.trustMassQ, `contextPatches[${index}].trustMass`)
    if (trustMassQ > maximumTrustQ) throw thoughtError('Bağlam patch’i dış öğretmen güveninden fazla kredi taşıyamaz.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    return {
      memberKey,
      relationDelta,
      trustMassQ,
      observationCount: Math.max(1, nonNegativeInteger(patch.observationCount ?? patch.updateCount, 1)),
    }
  }).sort((left, right) => left.memberKey.localeCompare(right.memberKey))
  if (new Set(patches.map((patch) => patch.memberKey)).size !== patches.length) {
    throw thoughtError('headDelta yinelenen bağlam patch’i içeriyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  const previousNursery = new Map(base.thoughtHead.nursery.map((candidate) => [candidate.id, candidate]))
  const normalizedNursery = nurseryUpserts.map((candidate, index) => {
    const withFloats = {
      ...candidate,
      supportWeight: candidate?.supportWeight ?? (Number.isSafeInteger(candidate?.supportWeightQ) ? candidate.supportWeightQ / Q_SCALE : undefined),
      trustMass: candidate?.trustMass ?? (Number.isSafeInteger(candidate?.trustMassQ) ? candidate.trustMassQ / Q_SCALE : undefined),
      resultScoreTotal: candidate?.resultScoreTotal ?? (Number.isSafeInteger(candidate?.resultScoreTotalQ) ? candidate.resultScoreTotalQ / Q_SCALE : undefined),
      residualCohesion: candidate?.residualCohesion ?? candidate?.cohesion ?? (
        Number.isSafeInteger(candidate?.residualCohesionQ) ? candidate.residualCohesionQ / Q_SCALE :
          Number.isSafeInteger(candidate?.cohesionQ) ? candidate.cohesionQ / Q_SCALE : undefined
      ),
      originEvidence: (candidate?.originEvidence ?? []).map((entry) => ({
        ...entry, weight: entry?.weight ?? (Number.isSafeInteger(entry?.weightQ) ? entry.weightQ / Q_SCALE : undefined),
      })),
      ...(candidate?.outgoingEvidence === undefined ? {} : {
        outgoingEvidence: candidate.outgoingEvidence.map((entry) => ({
          ...entry, weight: entry?.weight ?? (Number.isSafeInteger(entry?.weightQ) ? entry.weightQ / Q_SCALE : undefined),
        })),
      }),
      familyEvidence: (candidate?.familyEvidence ?? []).map((entry) => ({
        ...entry, weight: entry?.weight ?? (Number.isSafeInteger(entry?.weightQ) ? entry.weightQ / Q_SCALE : undefined),
      })),
    }
    const materialized = materializedNurseryCandidate(withFloats, base.layout, `nurseryUpserts[${index}]`)
    const previous = previousNursery.get(materialized.id)
    if (previous && previous.identityDigest !== materialized.identityDigest) {
      throw thoughtError('Nursery adayının sabit kimliği revizyonlar arasında değişemez.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    }
    if (
      previous && (
        materialized.observationCount < previous.observationCount ||
        materialized.residualObservationCount < previous.residualObservationCount ||
        materialized.trustMass + 1e-9 < previous.trustMass ||
        previous.birthEpisodeIds.some((id) => !materialized.birthEpisodeIds.includes(id)) ||
        previous.independentSourceIds.some((id) => !materialized.independentSourceIds.includes(id))
      )
    ) throw thoughtError('Nursery kanıt sayaçları geriye gidemez.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const evidenceDoesNotRegress = (before, after) => (before ?? []).every((entry) => {
      const next = (after ?? []).find((candidate) => candidate.key === entry.key)
      return next && next.weight + 1e-9 >= entry.weight
    })
    if (previous && (
      !evidenceDoesNotRegress(previous.originEvidence, materialized.originEvidence) ||
      !evidenceDoesNotRegress(previous.outgoingEvidence, materialized.outgoingEvidence) ||
      !evidenceDoesNotRegress(previous.familyEvidence, materialized.familyEvidence)
    )) throw thoughtError('Nursery köken/çıkış/aile kanıtları revizyonlar arasında geriye gidemez.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const previousTrust = previous?.trustMass ?? 0
    if (materialized.trustMass - previousTrust > maximumTrustQ / Q_SCALE + 1e-6) {
      throw thoughtError('Nursery trust artışı dış kanıt güvenini aşamaz.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    }
    const durableRunKey = normalizeId(engineRunKey, 'engine runKey') ?? episode.id
    if (materialized.observationCount > (previous?.observationCount ?? 0) && !materialized.birthEpisodeIds.includes(durableRunKey)) {
      throw thoughtError('Nursery gözlemi mevcut deterministik runKey kimliğini kanıt olarak taşımalı.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    }
    if (materialized.trustMass > previousTrust && teacherEvidence?.sourceGroupId && !materialized.independentSourceIds.includes(teacherEvidence.sourceGroupId)) {
      throw thoughtError('Nursery trust artışı bağımsız kaynak grubuyla ilişkilendirilmeli.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    }
    const originQByKey = new Map((candidate?.originEvidence ?? []).map((entry) => [entry?.key, entry?.weightQ]))
    const outgoingQByKey = new Map((candidate?.outgoingEvidence ?? []).map((entry) => [entry?.key, entry?.weightQ]))
    const familyQByKey = new Map((candidate?.familyEvidence ?? []).map((entry) => [entry?.key, entry?.weightQ]))
    return {
      id: materialized.id,
      identity: materialized.identity,
      relationPrototype: materialized.relationPrototype,
      identityDigest: materialized.identityDigest,
      birthEpisodeIds: materialized.birthEpisodeIds,
      independentSourceIds: materialized.independentSourceIds,
      observationCount: materialized.observationCount,
      residualObservationCount: materialized.residualObservationCount,
      supportWeightQ: qValue(materialized.supportWeight, candidate?.supportWeightQ, 'nursery.supportWeight'),
      trustMassQ: qValue(materialized.trustMass, candidate?.trustMassQ, 'nursery.trustMass'),
      resultScoreTotalQ: qValue(materialized.resultScoreTotal, candidate?.resultScoreTotalQ, 'nursery.resultScoreTotal'),
      originEvidence: materialized.originEvidence.map((entry) => ({
        key: entry.key,
        weightQ: qValue(entry.weight, originQByKey.get(entry.key), 'nursery.originEvidence.weight'),
      })),
      ...(materialized.outgoingEvidence ? {
        outgoingEvidence: materialized.outgoingEvidence.map((entry) => ({
          key: entry.key,
          weightQ: qValue(entry.weight, outgoingQByKey.get(entry.key), 'nursery.outgoingEvidence.weight'),
        })),
      } : {}),
      familyEvidence: materialized.familyEvidence.map((entry) => ({
        key: entry.key,
        weightQ: qValue(entry.weight, familyQByKey.get(entry.key), 'nursery.familyEvidence.weight'),
      })),
      nextPromotionTick: materialized.nextPromotionTick,
      residualCohesionQ: qValue(
        materialized.residualCohesion,
        candidate?.residualCohesionQ ?? candidate?.cohesionQ,
        'nursery.residualCohesion',
      ),
      sourceRelationPrototypes: materialized.sourceRelationPrototypes.map((entry) => ({
        sourceGroupId: entry.sourceGroupId,
        relation: entry.relation,
        observationCount: entry.observationCount,
      })),
    }
  }).sort((left, right) => left.id.localeCompare(right.id))
  if (new Set(normalizedNursery.map((candidate) => candidate.id)).size !== normalizedNursery.length) {
    throw thoughtError('headDelta yinelenen nursery adayı içeriyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  const removeIds = [...new Set(nurseryRemoveIds.map((id) => normalizeId(id, 'nurseryRemoveIds öğesi', { required: true })))].sort()
  if (removeIds.length !== nurseryRemoveIds.length) throw thoughtError('nurseryRemoveIds yinelenen kimlik içeriyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
  const nurseryAfter = new Map(previousNursery)
  for (const candidate of normalizedNursery) nurseryAfter.set(candidate.id, {
    ...candidate,
    trustMass: candidate.trustMassQ / Q_SCALE,
    resultScoreTotal: candidate.resultScoreTotalQ / Q_SCALE,
  })
  const born = births.map((birth, index) => {
    if (!isRecord(birth)) throw thoughtError(`births[${index}] geçersiz.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
    if (!base.layout) throw thoughtError('Nöron doğumu için temsil layout kanıtı bulunamadı.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const identity = vectorFrom(birth.identity, `births[${index}].identity`, base.layout.identityLength, -1, 1)
    const relation = vectorFrom(birth.relation, `births[${index}].relation`, base.layout.relationLength, -1, 1)
    const identityDigest = normalizeId(birth.identityDigest, `births[${index}].identityDigest`, { required: true })
    if (thoughtPartitionDigest('identity', identity) !== identityDigest) {
      throw thoughtError('Doğan nöronun sabit I digest’i doğrulanamadı.', 'THOUGHT_BIRTH_IDENTITY_MISMATCH', 422)
    }
    const sourceGroupIds = Array.isArray(birth.sourceGroupIds)
      ? [...new Set(birth.sourceGroupIds.map((item) => normalizeId(item, 'birth.sourceGroupIds öğesi', { required: true })))].sort()
      : []
    if (sourceGroupIds.length > 64) throw thoughtError('Doğan nöron kaynak grubu sınırını aşıyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const memberships = Array.isArray(birth.familyMemberships) ? birth.familyMemberships : Array.isArray(birth.softMemberships) ? birth.softMemberships : []
    if (memberships.length > 5) throw thoughtError('Doğan nöron en fazla 5 yumuşak üyelik taşıyabilir.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
    const birthId = normalizeId(birth.id, `births[${index}].id`, { required: true })
    const nurseryId = normalizeId(birth.nurseryId, `births[${index}].nurseryId`, { required: true })
    const nursery = nurseryAfter.get(nurseryId)
    const trustMassQ = qValue(birth.trustMass, birth.trustMassQ, `births[${index}].trustMass`)
    const averageScore = nursery?.observationCount > 0 ? (nursery.resultScoreTotal ?? 0) / nursery.observationCount : 0
    const cohesion = nursery?.residualCohesionQ !== undefined
      ? nursery.residualCohesionQ / Q_SCALE
      : nursery?.residualCohesion ?? nursery?.cohesion ?? 0
    const bornAtTick = nonNegativeInteger(birth.bornAtTick)
    const sourceEvidenceMatches = sourceGroupIds.length >= 2 && sourceGroupIds.every((id) => nursery?.independentSourceIds.includes(id))
    if (
      !nursery || nursery.identityDigest !== identityDigest || nursery.observationCount < 3 ||
      nursery.independentSourceIds.length < 2 || (nursery.trustMass ?? 0) < 1 || averageScore < 0.8 ||
      cohesion < 0.82 || bornAtTick < nursery.nextPromotionTick || !sourceEvidenceMatches ||
      !removeIds.includes(nurseryId) || trustMassQ > Math.round((nursery.trustMass ?? 0) * Q_SCALE)
    ) throw thoughtError('Nöron doğumu 3 gözlem/2 kaynak/trust 1/score 0.80/cohesion 0.82/cooldown nursery terfi kapısını geçmedi.', 'THOUGHT_BIRTH_GATE_FAILED', 422)
    if (
      relation.length !== nursery.relationPrototype.length ||
      relation.some((value, coordinate) => value !== nursery.relationPrototype[coordinate]) ||
      trustMassQ !== Math.round((nursery.trustMass ?? 0) * Q_SCALE) ||
      sourceGroupIds.length !== nursery.independentSourceIds.length ||
      sourceGroupIds.some((id, sourceIndex) => id !== [...nursery.independentSourceIds].sort()[sourceIndex])
    ) throw thoughtError('Doğum I/R, trust ve kaynak kanıtını nursery adayından eksiksiz devralmalı.', 'THOUGHT_BIRTH_GATE_FAILED', 422)

    const normalizePromotionEvidence = (raw, label) => {
      if (raw === undefined) return null
      if (!Array.isArray(raw)) {
        throw thoughtError(`${label} bir dizi olmalı.`, 'INVALID_THOUGHT_HEAD_DELTA', 422)
      }
      const withFloats = raw.map((entry) => ({
        ...entry,
        weight: entry?.weight ?? (Number.isSafeInteger(entry?.weightQ) ? entry.weightQ / Q_SCALE : undefined),
      }))
      const materialized = boundedMemberEvidenceEntries(withFloats, label)
      const qByKey = new Map(raw.map((entry) => [entry?.memberKey ?? entry?.key, entry?.weightQ]))
      return materialized.map((entry) => ({
        memberKey: entry.memberKey,
        weightQ: qValue(entry.weight, qByKey.get(entry.memberKey), `${label}.weight`),
      }))
    }
    const entryEvidence = normalizePromotionEvidence(birth.entryEvidence, `births[${index}].entryEvidence`)
    const outgoingEvidence = normalizePromotionEvidence(birth.outgoingEvidence, `births[${index}].outgoingEvidence`)
    for (const entry of [...(entryEvidence ?? []), ...(outgoingEvidence ?? [])]) {
      if (!durableMemberKeys.has(entry.memberKey) || entry.memberKey === birthId) {
        throw thoughtError('Doğum geçiş kanıtı kalıcı katalog dışında dangling kimlik içeriyor.', 'THOUGHT_DANGLING_MEMBER_KEY', 422)
      }
    }
    const nurseryEvidenceQ = (entries, key) => {
      const entry = (entries ?? []).find((candidate) => candidate.key === key)
      if (!entry) return 0
      if (Number.isSafeInteger(entry.weightQ)) return entry.weightQ
      return Math.round(Math.max(0, finite(entry.weight) ?? 0) * Q_SCALE)
    }
    const evidenceExactlyMatches = (nurseryEntries, promotedEntries) => {
      const expected = (nurseryEntries ?? [])
        .map((entry) => ({ memberKey: entry.key, weightQ: nurseryEvidenceQ(nurseryEntries, entry.key) }))
        .filter((entry) => entry.weightQ > 0)
        .sort((left, right) => left.memberKey.localeCompare(right.memberKey))
      const actual = (promotedEntries ?? []).filter((entry) => entry.weightQ > 0)
        .sort((left, right) => left.memberKey.localeCompare(right.memberKey))
      return expected.length === actual.length && expected.every((entry, evidenceIndex) => (
        entry.memberKey === actual[evidenceIndex].memberKey && entry.weightQ === actual[evidenceIndex].weightQ
      ))
    }
    if (
      !evidenceExactlyMatches(nursery.originEvidence, entryEvidence) ||
      !evidenceExactlyMatches(nursery.outgoingEvidence, outgoingEvidence)
    ) {
      throw thoughtError('Doğum geçiş kanıtını nursery adayından eksiksiz devralmalı.', 'THOUGHT_BIRTH_GATE_FAILED', 422)
    }
    const contextEvidence = birth.contextEvidence === undefined ? null : (() => {
      if (!isRecord(birth.contextEvidence)) {
        throw thoughtError('Doğum contextEvidence alanı geçersiz.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
      }
      const contextSourceGroupIds = [...new Set((Array.isArray(birth.contextEvidence.sourceGroupIds)
        ? birth.contextEvidence.sourceGroupIds : [])
        .map((id) => normalizeId(id, 'birth.contextEvidence.sourceGroupIds öğesi', { required: true })))].sort()
      const maximumObservations = (nursery.sourceRelationPrototypes ?? [])
        .reduce((sum, source) => sum + nonNegativeInteger(source.observationCount), 0)
      const expectedContextSources = (nursery.sourceRelationPrototypes ?? [])
        .map((source) => source.sourceGroupId).sort()
      const observationCount = nonNegativeInteger(birth.contextEvidence.observationCount)
      if (
        contextSourceGroupIds.length > 64 ||
        contextSourceGroupIds.some((id) => !sourceGroupIds.includes(id)) ||
        observationCount !== maximumObservations ||
        contextSourceGroupIds.length !== expectedContextSources.length ||
        contextSourceGroupIds.some((id, sourceIndex) => id !== expectedContextSources[sourceIndex])
      ) throw thoughtError('Doğum context kanıtı nursery kaynak prototipleriyle uyuşmuyor.', 'THOUGHT_BIRTH_GATE_FAILED', 422)
      return { observationCount, sourceGroupIds: contextSourceGroupIds }
    })()
    if ((nursery.sourceRelationPrototypes ?? []).length && !contextEvidence) {
      throw thoughtError('Doğum kaynak-dengeli R context kanıtını düşüremez.', 'THOUGHT_BIRTH_GATE_FAILED', 422)
    }
    return {
      format: 'born-neuron-v1',
      id: birthId,
      identity,
      relation,
      identityDigest,
      familyMemberships: memberships.map((membership, membershipIndex) => ({
        familyId: normalizeId(membership?.familyId, `births[${index}].familyMemberships[${membershipIndex}].familyId`, { required: true }),
        weightQ: qValue(membership?.weight, membership?.weightQ, 'birth membership weight'),
      })).sort((left, right) => left.familyId.localeCompare(right.familyId)),
      sourceGroupIds,
      trustMassQ,
      bornAtTick,
      nurseryId,
      ...(entryEvidence ? { entryEvidence } : {}),
      ...(outgoingEvidence ? { outgoingEvidence } : {}),
      ...(contextEvidence ? { contextEvidence } : {}),
    }
  }).sort((left, right) => left.id.localeCompare(right.id))
  if (new Set(born.map((birth) => birth.id)).size !== born.length) {
    throw thoughtError('headDelta yinelenen nöron doğumu içeriyor.', 'INVALID_THOUGHT_HEAD_DELTA', 422)
  }
  if (!neuronEdges.length && !patches.length && !normalizedNursery.length && !removeIds.length && !born.length) return null
  return {
    format: 'thought-head-delta',
    version: 1,
    episodeId: episode.id,
    baseModelFingerprint: base.reference.baseModelFingerprint,
    baseHeadFingerprint: base.thoughtHead.fingerprint,
    baseRevision: base.thoughtHead.revision,
    sourceGroupId: teacherEvidence.sourceGroupId,
    teacherEvidence,
    neuronEdges,
    contextPatches: patches,
    nurseryUpserts: normalizedNursery,
    nurseryRemoveIds: removeIds,
    births: born,
    stats: isRecord(delta.stats) ? strictBoundedValue(delta.stats) : {},
  }
}

function tickMember(value) {
  return boundedString(value, MAX_ID_LENGTH) || null
}

function tickVector(value, label) {
  if (value === undefined || value === null) return null
  let decoded = value
  if (isRecord(value) && value.$type) decoded = decodeArtifactValue(value)
  const vector = decoded instanceof Float32Array
    ? Float32Array.from(decoded)
    : Array.isArray(decoded) ? Float32Array.from(decoded) : null
  if (!vector || vector.length !== 4_096 || vector.some((item) => !Number.isFinite(item))) {
    throw thoughtError(`${label} tam 4096 sonlu Float32 değeri taşımalı.`, 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
  }
  return vector
}

function candidateContributions(candidate) {
  return {
    baseH0: finite(candidate?.baseH0 ?? candidate?.H0),
    persistentExcitation: finite(candidate?.persistentExcitation ?? candidate?.E),
    fastExcitation: finite(candidate?.fastExcitation ?? candidate?.e),
    persistentInhibition: finite(candidate?.persistentInhibition),
    fastInhibition: finite(candidate?.fastInhibition),
    habitLogit: finite(candidate?.habitLogit),
    habitProbability: probability(candidate?.habitProbability),
    membershipWeight: finite(candidate?.membershipWeight),
    routeEvidence: finite(candidate?.routeEvidence ?? candidate?.G),
    normalizedHabit: finite(candidate?.normalizedHabit),
    compatibility: finite(candidate?.signalCompatibility ?? candidate?.compatibility),
    repetitionPenalty: finite(candidate?.repetitionPenalty),
    q: finite(candidate?.q),
    acceptanceProbability: probability(candidate?.acceptanceProbability),
    reachProbability: probability(candidate?.reachProbability),
    selectionProbability: probability(candidate?.selectionProbability ?? candidate?.probability),
  }
}

function roleBinding(value) {
  if (!isRecord(value)) return null
  const role = boundedString(value.role ?? value.relation ?? value.kind, 64) || null
  const memberKey = tickMember(value.memberKey ?? value.neuronId ?? value.targetMemberKey)
  const familyId = tickMember(value.familyId ?? value.targetFamilyId)
  if (!role && !memberKey && !familyId) return null
  return {
    role,
    memberKey,
    familyId,
    confidence: probability(value.confidence ?? value.probability ?? value.weight),
    score: finite(value.score),
    text: displayText(value.text ?? value.target, 160).text || null,
  }
}

function birthEvent(value) {
  if (!isRecord(value)) return null
  return {
    id: tickMember(value.id ?? value.neuronId ?? value.candidateId),
    branchId: tickMember(value.branchId),
    kind: boundedString(value.kind ?? value.status, 48) || 'proposal',
    pressure: probability(value.pressure ?? value.birthPressure),
    consecutiveTicks: nonNegativeInteger(value.consecutiveTicks),
    supportCount: nonNegativeInteger(value.supportCount ?? value.observationCount),
    sourceGroupCount: nonNegativeInteger(value.sourceGroupCount ?? value.independentSourceCount),
    identityDigest: boundedString(value.identityDigest, 128) || null,
  }
}

function macroFallback(value) {
  if (!isRecord(value)) return null
  const reason = boundedString(value.reason, 80) || null
  if (!reason) return null
  return {
    macroId: tickMember(value.macroId),
    reason,
    detail: boundedString(value.detail, 240) || null,
  }
}

function macroCandidate(value) {
  if (!isRecord(value)) return null
  const macroId = tickMember(value.macroId)
  if (!macroId) return null
  return {
    macroId,
    entryCompatibility: probability(value.entryCompatibility),
    activationWeight: finite(value.activationWeight),
    selectionProbability: probability(value.selectionProbability),
    duration: nonNegativeInteger(value.duration),
    compressedTransitionCount: nonNegativeInteger(value.compressedTransitionCount),
    fidelity: probability(value.fidelity),
    expandedConstituentIds: (Array.isArray(value.expandedConstituentIds)
      ? value.expandedConstituentIds : []).slice(0, 128).map(tickMember).filter(Boolean),
  }
}

function macroExecution(value) {
  if (!isRecord(value)) return null
  const macroId = tickMember(value.macroId)
  if (!macroId || value.compiled !== true || value.fallbackReason !== null) return null
  return {
    macroId,
    macroHeadFingerprint: boundedString(value.macroHeadFingerprint, 128) || null,
    runtimeFingerprint: boundedString(value.runtimeFingerprint, 128) || null,
    compiled: true,
    expandedConstituentIds: (Array.isArray(value.expandedConstituentIds)
      ? value.expandedConstituentIds : []).slice(0, 128).map(tickMember).filter(Boolean),
    duration: nonNegativeInteger(value.duration),
    compressedTransitionCount: nonNegativeInteger(value.compressedTransitionCount),
    fidelity: probability(value.fidelity),
    entryCompatibility: probability(value.entryCompatibility),
    selectionProbability: probability(value.selectionProbability),
    creditMode: value.creditMode === 'constituent-edges' ? 'constituent-edges' : null,
    fallbackReason: null,
  }
}

function adaptiveRegionError(message, details = undefined) {
  return thoughtError(message, 'THOUGHT_ENGINE_OUTPUT_INVALID', 502, details)
}

function adaptiveRegionFinite(value, label, minimum = -Infinity, maximum = Infinity) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw adaptiveRegionError(`${label} sonlu ve ${minimum}–${maximum} aralığında olmalı.`)
  }
  return value
}

function adaptiveRegionArray(value, length, label, minimum, maximum) {
  const source = ArrayBuffer.isView(value) ? Array.from(value) : value
  if (!Array.isArray(source) || source.length !== length) {
    throw adaptiveRegionError(`${label} tam ${length} değer taşımalı.`)
  }
  return Float32Array.from(source.map((entry, index) =>
    adaptiveRegionFinite(entry, `${label}[${index}]`, minimum, maximum)))
}

function adaptiveRegionId(value, label) {
  const result = boundedString(value, MAX_ID_LENGTH)
  if (!result || result !== value || /[\u0000-\u001f\u007f]/u.test(result)) {
    throw adaptiveRegionError(`${label} geçersiz.`)
  }
  return result
}

function sanitizeAdaptiveGateDistribution(value, label) {
  if (!isRecord(value) || !Array.isArray(value.gates) || value.gates.length > MAX_ADAPTIVE_REGION_GATES) {
    throw adaptiveRegionError(`${label} geçerli ve sınırlı bir kapı dağılımı olmalı.`)
  }
  const gates = value.gates.map((gate, index) => {
    if (!isRecord(gate)) throw adaptiveRegionError(`${label}.gates[${index}] geçersiz.`)
    return {
      regionId: adaptiveRegionId(gate.regionId, `${label}.gates[${index}].regionId`),
      compatibility: adaptiveRegionFinite(gate.compatibility, `${label}.gates[${index}].compatibility`, -1, 1),
      prototypeCompatibility: adaptiveRegionFinite(
        gate.prototypeCompatibility,
        `${label}.gates[${index}].prototypeCompatibility`,
        -1,
        1,
      ),
      contextScore: adaptiveRegionFinite(gate.contextScore, `${label}.gates[${index}].contextScore`, -1, 1),
      confidence: adaptiveRegionFinite(gate.confidence, `${label}.gates[${index}].confidence`, 0, 1),
      capacityPenalty: adaptiveRegionFinite(gate.capacityPenalty, `${label}.gates[${index}].capacityPenalty`, 0),
      logit: adaptiveRegionFinite(gate.logit, `${label}.gates[${index}].logit`),
      probability: adaptiveRegionFinite(gate.probability, `${label}.gates[${index}].probability`, 0, 1),
    }
  })
  if (new Set(gates.map((gate) => gate.regionId)).size !== gates.length) {
    throw adaptiveRegionError(`${label} yinelenen bölge kapısı içeriyor.`)
  }
  const unmatchedProbability = adaptiveRegionFinite(value.unmatchedProbability, `${label}.unmatchedProbability`, 0, 1)
  const omittedRegionProbability = adaptiveRegionFinite(value.omittedRegionProbability, `${label}.omittedRegionProbability`, 0, 1)
  const total = unmatchedProbability + gates.reduce((sum, gate) => sum + gate.probability, 0)
  if (Math.abs(total - 1) > 1e-5 || omittedRegionProbability > unmatchedProbability + 1e-6) {
    throw adaptiveRegionError(`${label} olasılık kütlesini korumuyor.`, { total })
  }
  return { gates, unmatchedProbability, omittedRegionProbability }
}

function sanitizeAdaptiveCompatibility(value, label) {
  if (!isRecord(value)) throw adaptiveRegionError(`${label} geçersiz.`)
  return {
    legacyCompatibility: adaptiveRegionFinite(value.legacyCompatibility, `${label}.legacyCompatibility`, 0, 1),
    regionCompatibility: adaptiveRegionFinite(value.regionCompatibility, `${label}.regionCompatibility`, 0, 1),
    blendedCompatibility: adaptiveRegionFinite(value.blendedCompatibility, `${label}.blendedCompatibility`, 0, 1),
    effectiveCompatibility: adaptiveRegionFinite(value.effectiveCompatibility, `${label}.effectiveCompatibility`, 0, 1),
    pageCompatibilities: adaptiveRegionArray(
      value.pageCompatibilities,
      ADAPTIVE_REGION_PAGE_COUNT,
      `${label}.pageCompatibilities`,
      -1,
      1,
    ),
    pageWeights: adaptiveRegionArray(
      value.pageWeights,
      ADAPTIVE_REGION_PAGE_COUNT,
      `${label}.pageWeights`,
      0,
      1,
    ),
    gates: sanitizeAdaptiveGateDistribution(value.gates, `${label}.gates`),
  }
}

function sanitizeAdaptiveObservation(value, label) {
  if (!isRecord(value)) throw adaptiveRegionError(`${label} geçersiz.`)
  const pageIndex = nonNegativeInteger(value.pageIndex, -1)
  if (pageIndex < 0 || pageIndex >= ADAPTIVE_REGION_PAGE_COUNT) {
    throw adaptiveRegionError(`${label}.pageIndex geometri dışında.`)
  }
  return {
    observationId: adaptiveRegionId(value.observationId, `${label}.observationId`),
    episodeId: adaptiveRegionId(value.episodeId, `${label}.episodeId`),
    sourceGroupId: adaptiveRegionId(value.sourceGroupId, `${label}.sourceGroupId`),
    pageIndex,
    confidence: adaptiveRegionFinite(value.confidence, `${label}.confidence`, Number.MIN_VALUE, 1),
    behavior: adaptiveRegionArray(
      value.behavior,
      ADAPTIVE_REGION_BEHAVIOR_DIMENSION,
      `${label}.behavior`,
      -1,
      1,
    ),
    outcomeCredit: adaptiveRegionFinite(value.outcomeCredit, `${label}.outcomeCredit`, -1, 1),
    ablationUtility: adaptiveRegionFinite(value.ablationUtility, `${label}.ablationUtility`, -1, 1),
  }
}

function sanitizeAdaptiveBranchTelemetry(value, label) {
  if (value === undefined || value === null) return null
  if (!isRecord(value) || value.format !== 'thought-adaptive-region-branch-telemetry' || value.version !== 1) {
    throw adaptiveRegionError(`${label} sözleşmesi geçersiz.`)
  }
  if (!['shadow', 'proposal', 'active'].includes(value.mode)) {
    throw adaptiveRegionError(`${label}.mode geçersiz.`)
  }
  if (
    typeof value.activeApplied !== 'boolean' ||
    (value.mode !== 'active' && value.activeApplied)
  ) {
    throw adaptiveRegionError(`${label}.activeApplied kip ile uyuşmuyor.`)
  }
  if (!Array.isArray(value.pageObservations) || value.pageObservations.length !== ADAPTIVE_REGION_PAGE_COUNT) {
    throw adaptiveRegionError(`${label}.pageObservations tam ${ADAPTIVE_REGION_PAGE_COUNT} gözlem taşımalı.`)
  }
  const pageObservations = value.pageObservations.map((observation, index) =>
    sanitizeAdaptiveObservation(observation, `${label}.pageObservations[${index}]`))
  if (
    new Set(pageObservations.map((observation) => observation.pageIndex)).size !== ADAPTIVE_REGION_PAGE_COUNT ||
    pageObservations.some((observation, index) => observation.pageIndex !== index)
  ) throw adaptiveRegionError(`${label}.pageObservations kanonik 0–55 sırasında olmalı.`)
  if (!Array.isArray(value.candidateCompatibilities) || value.candidateCompatibilities.length > MAX_ADAPTIVE_REGION_CANDIDATES) {
    throw adaptiveRegionError(`${label}.candidateCompatibilities en fazla ${MAX_ADAPTIVE_REGION_CANDIDATES} öğe taşımalı.`)
  }
  const candidateCompatibilities = value.candidateCompatibilities.map((candidate, index) => ({
    neuronId: adaptiveRegionId(candidate?.neuronId, `${label}.candidateCompatibilities[${index}].neuronId`),
    ...sanitizeAdaptiveCompatibility(candidate, `${label}.candidateCompatibilities[${index}]`),
  }))
  return {
    format: value.format,
    version: value.version,
    mode: value.mode,
    atlasFingerprint: adaptiveRegionId(value.atlasFingerprint, `${label}.atlasFingerprint`),
    bindingFingerprint: adaptiveRegionId(value.bindingFingerprint, `${label}.bindingFingerprint`),
    activeApplied: value.activeApplied,
    ...sanitizeAdaptiveCompatibility(value, label),
    pageWriteMask: adaptiveRegionArray(
      value.pageWriteMask,
      ADAPTIVE_REGION_PAGE_COUNT,
      `${label}.pageWriteMask`,
      0,
      1,
    ),
    pageObservations,
    candidateCompatibilities,
  }
}

function sanitizeAdaptiveTickTelemetry(value, branches) {
  const adaptiveBranches = branches.filter((branch) => branch.adaptiveRegion)
  if (value === undefined || value === null) {
    if (adaptiveBranches.length) throw adaptiveRegionError('Dal telemetrisi üst tick atlas bağını taşımıyor.')
    return null
  }
  if (!isRecord(value) || value.format !== 'thought-adaptive-region-tick-telemetry' || value.version !== 1) {
    throw adaptiveRegionError('Tick uyarlanabilir bölge sözleşmesi geçersiz.')
  }
  if (!['shadow', 'proposal', 'active'].includes(value.mode)) throw adaptiveRegionError('Tick bölge kipi geçersiz.')
  if (typeof value.activeApplied !== 'boolean') throw adaptiveRegionError('Tick activeApplied boolean olmalı.')
  const atlasFingerprint = adaptiveRegionId(value.atlasFingerprint, 'tick.adaptiveRegion.atlasFingerprint')
  const bindingFingerprint = adaptiveRegionId(value.bindingFingerprint, 'tick.adaptiveRegion.bindingFingerprint')
  if (
    Boolean(value.activeApplied) !== adaptiveBranches.some((branch) => branch.adaptiveRegion.activeApplied) ||
    value.observedBranchCount !== adaptiveBranches.length ||
    adaptiveBranches.some((branch) => (
      branch.adaptiveRegion.mode !== value.mode ||
      branch.adaptiveRegion.atlasFingerprint !== atlasFingerprint ||
      branch.adaptiveRegion.bindingFingerprint !== bindingFingerprint
    ))
  ) throw adaptiveRegionError('Tick ve dal uyarlanabilir bölge bağları uyuşmuyor.')
  return {
    format: value.format,
    version: value.version,
    mode: value.mode,
    atlasFingerprint,
    bindingFingerprint,
    observedBranchCount: adaptiveBranches.length,
    activeApplied: Boolean(value.activeApplied),
  }
}

function sanitizeTick(raw, fallbackIndex, candidateLimit = MAX_STORED_CANDIDATES_PER_TICK) {
  const tick = isRecord(raw) ? raw : {}
  const mapCandidate = (candidate, rank) => ({
    rank: nonNegativeInteger(candidate?.rank, rank + 1),
    memberKey: tickMember(candidate?.memberKey ?? candidate?.neuronId),
    familyId: tickMember(candidate?.familyId),
    probability: probability(candidate?.probability ?? candidate?.selectionProbability),
    habitProbability: probability(candidate?.habitProbability),
    signalCompatibility: finite(candidate?.signalCompatibility ?? candidate?.compatibility),
    score: finite(candidate?.score),
    outcome: boundedString(candidate?.outcome, 40) || null,
    contributions: candidateContributions(candidate),
  })
  const mapFamily = (family) => ({
    familyId: tickMember(family?.familyId),
    region: boundedString(family?.region, 80) || null,
    probability: probability(family?.probability),
    score: finite(family?.score),
    activation: probability(family?.activation ?? family?.coalitionActivation),
    evidence: finite(family?.evidence),
  })
  const candidates = (Array.isArray(tick.candidates) ? tick.candidates : Array.isArray(tick.neuronCandidates) ? tick.neuronCandidates : [])
    .slice(0, candidateLimit)
    .map(mapCandidate)
  const families = (Array.isArray(tick.families) ? tick.families : Array.isArray(tick.familyCandidates) ? tick.familyCandidates : [])
    .slice(0, MAX_FAMILIES_PER_TICK)
    .map(mapFamily)
  const coalitions = (Array.isArray(tick.coalitions) ? tick.coalitions : Array.isArray(tick.coalition?.coalitions) ? tick.coalition.coalitions : [])
    .slice(0, MAX_COALITIONS_PER_TICK)
    .map((coalition) => ({
      id: tickMember(coalition?.id),
      eventMemberKey: tickMember(coalition?.eventMemberKey),
      eventFamilyId: tickMember(coalition?.eventFamilyId),
      dominance: probability(coalition?.dominance),
      confidence: probability(coalition?.confidence),
      memberKeys: (Array.isArray(coalition?.memberKeys) ? coalition.memberKeys : []).slice(0, 8).map(tickMember).filter(Boolean),
      roleBindings: (Array.isArray(coalition?.roleBindings) ? coalition.roleBindings : Array.isArray(coalition?.roles) ? coalition.roles : [])
        .slice(0, 8).map(roleBinding).filter(Boolean),
      roleHypotheses: (Array.isArray(coalition?.roleHypotheses) ? coalition.roleHypotheses : [])
        .slice(0, 8).map(roleBinding).filter(Boolean),
    }))
  const rawBranches = Array.isArray(tick.branches) ? tick.branches : Array.isArray(tick.branchStates) ? tick.branchStates : []
  const branches = rawBranches.slice(0, MAX_BRANCHES_PER_TICK).map((branch) => {
    const pressure = isRecord(branch?.pressure) ? branch.pressure : {}
    const adaptiveRegion = sanitizeAdaptiveBranchTelemetry(branch?.adaptiveRegion, 'branch.adaptiveRegion')
    const coalitionWeights = isRecord(branch?.coalitionFamilyWeights)
      ? Object.entries(branch.coalitionFamilyWeights)
        .filter(([key, value]) => key && finite(value) !== null)
        .sort((left, right) => (right[1] - left[1]) || left[0].localeCompare(right[0]))
        .slice(0, MAX_FAMILIES_PER_TICK)
        .map(([familyId, weight]) => ({ familyId: tickMember(familyId), weight: finite(weight) }))
      : []
    return {
      id: tickMember(branch?.id ?? branch?.branchId),
      parentId: tickMember(branch?.parentId ?? branch?.parentBranchId),
      currentMemberKey: tickMember(branch?.currentMemberKey ?? branch?.currentNeuronId ?? branch?.selectedNeuronId),
      selectedFamilyId: tickMember(branch?.selectedFamilyId),
      selectedMemberKey: tickMember(branch?.selectedMemberKey ?? branch?.selectedNeuronId),
      mass: probability(branch?.mass),
      cumulativeLogProbability: finite(branch?.cumulativeLogProbability),
      status: boundedString(branch?.status, 40) || null,
      pathTail: (Array.isArray(branch?.path) ? branch.path : []).slice(-16).map(tickMember).filter(Boolean),
      signal: {
        before: tickVector(branch?.beforeSignal, 'branch.beforeSignal'),
        after: tickVector(branch?.afterSignal, 'branch.afterSignal'),
        beforeDigest: boundedString(branch?.beforeSignalDigest, 128) || null,
        afterDigest: boundedString(branch?.afterSignalDigest, 128) || null,
        beforeNorm: finite(branch?.beforeSignalNorm),
        afterNorm: finite(branch?.afterSignalNorm),
        changeMagnitude: finite(branch?.signalChangeMagnitude),
      },
      candidates: (Array.isArray(branch?.neuronCandidates) ? branch.neuronCandidates : [])
        .slice(0, candidateLimit).map(mapCandidate),
      families: (Array.isArray(branch?.familyCandidates) ? branch.familyCandidates : [])
        .slice(0, MAX_FAMILIES_PER_TICK).map(mapFamily),
      unmatchedProbability: probability(branch?.unmatchedProbability),
      pressure: {
        habitDistance: probability(pressure.habitDistance),
        representationError: probability(pressure.representationError),
        edge: probability(branch?.edgePressure ?? pressure.edge),
        birth: probability(branch?.birthPressure ?? pressure.birth),
        smoothedBirth: probability(branch?.smoothedBirthPressure ?? pressure.smoothedBirth),
        consecutiveBirthTicks: nonNegativeInteger(pressure.consecutiveBirthTicks),
      },
      coalitionFamilyWeights: coalitionWeights,
      roleHypotheses: (Array.isArray(branch?.roleHypotheses) ? branch.roleHypotheses : [])
        .slice(0, 16).map(roleBinding).filter(Boolean),
      birthEvents: (Array.isArray(branch?.birthEvents) ? branch.birthEvents : []).slice(0, 8).map(birthEvent).filter(Boolean),
      macroRuntimeFingerprint: boundedString(branch?.macroRuntimeFingerprint, 128) || null,
      rawProbabilityScale: probability(branch?.rawProbabilityScale),
      macroProbabilityMass: probability(branch?.macroProbabilityMass),
      macroCandidates: (Array.isArray(branch?.macroCandidates) ? branch.macroCandidates : [])
        .slice(0, 8).map(macroCandidate).filter(Boolean),
      macroExecution: macroExecution(branch?.macroExecution),
      macroFallbacks: (Array.isArray(branch?.macroFallbacks) ? branch.macroFallbacks : [])
        .slice(0, 8).map(macroFallback).filter(Boolean),
      ...(adaptiveRegion ? { adaptiveRegion } : {}),
    }
  })
  const signalSource = isRecord(tick.signal) ? tick.signal : {}
  const beforeSignal = signalSource.before ?? signalSource.beforeMatrix ?? tick.beforeSignal ?? tick.incomingSignal
  const afterSignal = signalSource.after ?? signalSource.afterMatrix ?? tick.afterSignal ?? tick.transformedSignal
  const adaptiveRegion = sanitizeAdaptiveTickTelemetry(tick.adaptiveRegion, branches)
  return {
    index: nonNegativeInteger(tick.index ?? tick.tick ?? tick.step, fallbackIndex),
    branchCount: nonNegativeInteger(tick.branchCount, branches.length),
    activeBranchCount: nonNegativeInteger(tick.activeBranchCount, branches.filter((branch) => branch.status === 'active').length),
    dominantBranchId: tickMember(tick.dominantBranchId),
    dominantMemberKey: tickMember(tick.dominantMemberKey ?? tick.dominantNeuronId),
    dominantMass: probability(tick.dominantMass),
    maximumBirthPressure: probability(tick.maximumBirthPressure),
    replayFingerprint: boundedString(tick.replayFingerprint, 128) || null,
    fromMemberKey: tickMember(tick.fromMemberKey ?? tick.fromNeuronId),
    toMemberKey: tickMember(tick.toMemberKey ?? tick.toNeuronId ?? tick.emittedMemberKey),
    selectedFamilyId: tickMember(tick.selectedFamilyId),
    compatibility: finite(tick.compatibility),
    alpha: probability(tick.alpha),
    changeMagnitude: finite(tick.changeMagnitude),
    unmatchedProbability: probability(tick.unmatchedProbability),
    unmatched: Boolean(tick.unmatched),
    rngSample: probability(tick.rngSample),
    stopReason: boundedString(tick.stopReason, 80) || null,
    signal: {
      beforeDigest: boundedString(signalSource.beforeDigest ?? tick.incomingSignalDigest, 128) || null,
      afterDigest: boundedString(signalSource.afterDigest ?? tick.transformedSignalDigest, 128) || null,
      beforeNorm: finite(signalSource.beforeNorm),
      afterNorm: finite(signalSource.afterNorm),
      before: tickVector(beforeSignal, 'signal.before'),
      after: tickVector(afterSignal, 'signal.after'),
    },
    branches,
    branchTruncated: rawBranches.length > branches.length,
    candidates,
    families,
    residualCandidateCount: nonNegativeInteger(tick.residualCandidateCount),
    coalitions,
    roleHypotheses: (Array.isArray(tick.roleHypotheses) ? tick.roleHypotheses : [])
      .slice(0, 16).map(roleBinding).filter(Boolean),
    birthEvents: (Array.isArray(tick.birthEvents) ? tick.birthEvents : Array.isArray(tick.births) ? tick.births : [])
      .slice(0, 16).map(birthEvent).filter(Boolean),
    pressure: isRecord(tick.pressure) ? {
      habitDistance: probability(tick.pressure.habitDistance),
      representationError: probability(tick.pressure.representationError),
      instantaneous: probability(tick.pressure.instantaneous),
      smoothed: probability(tick.pressure.smoothed),
    } : null,
    learningEligible: Boolean(tick.learningEligible),
    emission: isRecord(tick.emission) ? {
      text: displayText(tick.emission.text, 240).text,
      memberKey: tickMember(tick.emission.memberKey),
      eos: Boolean(tick.emission.eos),
    } : null,
    ...(adaptiveRegion ? { adaptiveRegion } : {}),
  }
}

function normalizeTicks(rawTicks, maximum, previousDigest = null) {
  if (!Array.isArray(rawTicks)) return []
  if (rawTicks.length > maximum) throw thoughtError('Motor tick sınırını aştı.', 'THOUGHT_ENGINE_LIMIT_EXCEEDED', 502)
  const indexed = rawTicks.map((tick, index) => sanitizeTick(tick, index))
    .sort((left, right) => left.index - right.index)
  if (new Set(indexed.map((tick) => tick.index)).size !== indexed.length) {
    throw thoughtError('Motor yinelenen tick kimliği döndürdü.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
  }
  let prior = previousDigest
  return indexed.map((tick) => {
    const tickDigest = digest('thought-tick-v1', prior, tick)
    const result = { ...tick, previousTickDigest: prior, tickDigest }
    prior = tickDigest
    return result
  })
}

function sanitizeGraph(raw) {
  if (!isRecord(raw)) return { nodes: [], edges: [] }
  const nestedFamilyTransitions = Array.isArray(raw.familyGraph?.transitions) ? raw.familyGraph.transitions : []
  const rawEdges = [
    ...(Array.isArray(raw.edges) ? raw.edges : Array.isArray(raw.neuronEdges) ? raw.neuronEdges : []),
    ...nestedFamilyTransitions.map((transition) => ({
      from: transition.fromFamilyId, to: transition.toFamilyId, kind: 'family', probability: transition.probability,
    })),
  ]
  const edges = rawEdges.slice(0, MAX_GRAPH_EDGES).map((edge) => ({
    from: tickMember(edge?.from ?? edge?.fromMemberKey ?? edge?.fromFamilyId),
    to: tickMember(edge?.to ?? edge?.toMemberKey ?? edge?.toFamilyId),
    kind: boundedString(edge?.kind, 40) || 'neuron',
    probability: probability(edge?.probability),
    excitation: finite(edge?.excitation ?? (Number.isSafeInteger(edge?.excitationQ) ? edge.excitationQ / Q_SCALE : null)),
    inhibition: finite(edge?.inhibition ?? (Number.isSafeInteger(edge?.inhibitionQ) ? edge.inhibitionQ / Q_SCALE : null)),
    trustMass: finite(edge?.trustMass ?? (Number.isSafeInteger(edge?.trustMassQ) ? edge.trustMassQ / Q_SCALE : null)),
  })).filter((edge) => edge.from && edge.to)
  const rawNodes = Array.isArray(raw.nodes) ? raw.nodes : []
  const nodes = rawNodes.slice(0, MAX_GRAPH_NODES).map((node) => ({
    id: tickMember(node?.id ?? node?.memberKey ?? node?.familyId),
    kind: boundedString(node?.kind, 40) || 'neuron',
    label: displayText(node?.label ?? node?.text ?? node?.id, 160).text,
    activation: probability(node?.activation),
    familyId: tickMember(node?.familyId),
  })).filter((node) => node.id)
  let eventGraph = null
  if (raw.eventGraph) {
    try { eventGraph = normalizeEventGraph(raw.eventGraph) } catch (cause) {
      throw thoughtError('Motor EventGraph çıktısı canonical şemaya uymuyor.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502, {
        cause: cause instanceof Error ? cause.message : String(cause),
      })
    }
  }
  const evaluation = isRecord(raw.evaluation) ? {
    graphF1: probability(raw.evaluation.graphF1),
    constraintAccuracy: probability(raw.evaluation.constraintAccuracy),
    resultScore: probability(raw.evaluation.resultScore),
    truePositive: nonNegativeInteger(raw.evaluation.truePositive),
    falsePositive: nonNegativeInteger(raw.evaluation.falsePositive),
    falseNegative: nonNegativeInteger(raw.evaluation.falseNegative),
  } : null
  const branchScores = (Array.isArray(raw.branchScores) ? raw.branchScores : []).slice(0, MAX_BRANCHES_PER_TICK).map((score) => ({
    branchId: tickMember(score?.branchId),
    transitionCount: nonNegativeInteger(score?.transitionCount),
    meanCompatibility: finite(score?.meanCompatibility),
    meanGraphSupport: finite(score?.meanGraphSupport),
    meanFamilySupport: finite(score?.meanFamilySupport),
    uniqueTransitionRatio: probability(score?.uniqueTransitionRatio),
    loopRatio: probability(score?.loopRatio),
    score: finite(score?.score),
    eligible: Boolean(score?.eligible),
    explanation: boundedString(score?.explanation, 500) || null,
  }))
  return {
    nodes, edges,
    eventGraph,
    evaluation,
    branchScores,
    catalog: (Array.isArray(raw.catalog) ? raw.catalog : []).slice(0, 256).map((entry) => ({
      memberKey: tickMember(entry?.memberKey),
      reason: boundedString(entry?.reason, 240),
      kind: boundedString(entry?.kind, 40),
    })),
    prompt: isRecord(raw.prompt) ? strictBoundedValue(raw.prompt) : null,
    baseDiagnostics: isRecord(raw.baseDiagnostics) ? strictBoundedValue(raw.baseDiagnostics) : null,
    familyGraphFingerprint: boundedString(raw.familyGraph?.fingerprint, 128) || null,
    truncated: rawEdges.length > edges.length || rawNodes.length > nodes.length ||
      (Array.isArray(raw.branchScores) && raw.branchScores.length > branchScores.length),
  }
}

function publicBase(base) {
  return {
    snapshotId: base?.snapshotId ?? null,
    snapshotStatus: boundedString(base?.snapshotStatus, 40) || null,
    selectedBy: boundedString(base?.selectedBy, 80) || null,
    activeAtStart: Boolean(base?.activeAtStart),
    commitEligible: Boolean(base?.commitEligible),
    artifactManifestHash: boundedString(base?.artifactManifestHash, 128) || null,
    baseModelFingerprint: boundedString(base?.baseModelFingerprint, 128) || null,
    familyRunFingerprint: boundedString(base?.familyRunFingerprint, 128) || null,
    languageStateFingerprint: boundedString(base?.languageStateFingerprint, 128) || null,
    thoughtHeadFingerprint: boundedString(base?.thoughtHeadFingerprint, 128) || null,
    thoughtHeadRevision: nonNegativeInteger(base?.thoughtHeadRevision),
    conceptMacroHeadFingerprint: boundedString(base?.conceptMacroHeadFingerprint, 128) || null,
    ...(boundedString(base?.representationProfile, 80)
      ? { representationProfile: boundedString(base.representationProfile, 80) }
      : {}),
    regionAtlasArtifactId: boundedString(base?.regionAtlasArtifactId, MAX_ID_LENGTH) || null,
    regionAtlasFingerprint: boundedString(base?.regionAtlasFingerprint, 256) || null,
    regionAtlasRevision: base?.regionAtlasRevision === null || base?.regionAtlasRevision === undefined
      ? null : nonNegativeInteger(base.regionAtlasRevision),
    regionGeometryFingerprint: boundedString(base?.regionGeometryFingerprint, 256) || null,
    regionReleaseManifestArtifactId: boundedString(base?.regionReleaseManifestArtifactId, MAX_ID_LENGTH) || null,
    regionReleaseFingerprint: boundedString(base?.regionReleaseFingerprint, 256) || null,
    ...(boundedString(base?.residencyManifestFingerprint, 256)
      ? { residencyManifestFingerprint: boundedString(base.residencyManifestFingerprint, 256) }
      : {}),
  }
}

function assertEpisodeResidencyManifest(episode, residency) {
  const expected = boundedString(episode?.base?.residencyManifestFingerprint, 256) || null
  // Episodes created before the SSD-manifest binding rollout remain replayable
  // under their legacy tick contract. New episodes fail closed when physical
  // page layout/policy no longer resolves to the exact recorded manifest.
  if (!expected) return residency
  const actual = boundedString(residency?.manifest?.fingerprint, 256) || null
  if (actual !== expected) {
    throw thoughtError(
      'Düşünce bölümünün SSD nöron yerleşimi artık aynı fiziksel manifesti üretmiyor.',
      'STALE_THOUGHT_RESIDENCY',
      409,
      { expected, actual },
    )
  }
  return residency
}

function assertEpisodeRegionRuntime(episode, runtime) {
  const expected = {
    mode: boundedString(episode.regionMode, 40) || 'off',
    atlasArtifactId: boundedString(episode.regionAtlasArtifactId, MAX_ID_LENGTH) || null,
    atlasFingerprint: boundedString(episode.regionAtlasFingerprint, 256) || null,
    atlasRevision: episode.regionAtlasRevision === null || episode.regionAtlasRevision === undefined
      ? null : nonNegativeInteger(episode.regionAtlasRevision),
    geometryFingerprint: boundedString(episode.regionGeometryFingerprint, 256) || null,
    representationSpaceId: boundedString(episode.regionRepresentationSpaceId, 256) || null,
    releaseManifestArtifactId: boundedString(episode.regionReleaseManifestArtifactId, MAX_ID_LENGTH) || null,
    releaseFingerprint: boundedString(episode.regionReleaseFingerprint, 256) || null,
  }
  const actual = {
    mode: runtime.mode,
    atlasArtifactId: runtime.atlasArtifactId,
    atlasFingerprint: runtime.atlas?.fingerprint ?? null,
    atlasRevision: runtime.atlas?.revision ?? null,
    geometryFingerprint: runtime.atlas?.geometryFingerprint ?? null,
    representationSpaceId: runtime.atlas?.representationSpaceId ?? null,
    releaseManifestArtifactId: runtime.releaseManifestArtifactId ?? null,
    releaseFingerprint: runtime.releaseFingerprint ?? null,
  }
  const mismatch = Object.keys(expected).find((key) => expected[key] !== actual[key])
  if (mismatch) {
    throw thoughtError(
      'Düşünce bölümünün plastik bölge atlası veya çalışma kipi artık aynı değil.',
      'STALE_PLASTIC_REGION_BASE',
      409,
      { field: mismatch, expected: expected[mismatch], actual: actual[mismatch] },
    )
  }
  return runtime
}

function verifiedEngineRegionBindingFingerprint(binding, episode, runtime) {
  if (runtime.mode === 'off') {
    if (binding !== undefined && binding !== null) {
      throw thoughtError('Off düşünce bölümü beklenmedik bölge bağı üretti.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
    }
    return null
  }
  if (!isRecord(binding) || binding.format !== 'thought-adaptive-region-binding' || binding.version !== 1) {
    throw thoughtError('Düşünce motoru doğrulanabilir bölge bağı üretmedi.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
  }
  const expected = {
    mode: runtime.mode,
    atlasFingerprint: runtime.atlas?.fingerprint ?? null,
    atlasRevision: runtime.atlas?.revision ?? null,
    representationSpaceId: runtime.atlas?.representationSpaceId ?? null,
    geometryFingerprint: runtime.atlas?.geometryFingerprint ?? null,
    sourceGroupId: episode.teacherEvidence?.sourceGroupId ?? episode.sourceGroupId ??
      defaultAdaptiveRegionSourceGroupId(episode.inputText),
    compatibilityBlend: runtime.adaptiveRegion?.compatibilityBlend ?? 0,
    minimumTrace: runtime.adaptiveRegion?.minimumTrace ?? 0,
    ...(runtime.mode === 'active' ? {
      maximumActiveGates: runtime.adaptiveRegion?.maximumActiveGates ?? 4,
    } : {}),
  }
  const mismatch = Object.keys(expected).find((key) => binding[key] !== expected[key])
  const fingerprint = boundedString(binding.fingerprint, 256)
  const canonicalBinding = {
    format: 'thought-adaptive-region-binding',
    version: 1,
    ...expected,
  }
  const expectedFingerprint = thoughtDigest('adaptive-region-binding', canonicalBinding)
  if (mismatch || !fingerprint || fingerprint !== binding.fingerprint || fingerprint !== expectedFingerprint) {
    throw thoughtError('Motorun bölge bağı episode/snapshot politikasıyla uyuşmuyor.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502, {
      field: mismatch ?? 'fingerprint', expectedFingerprint,
    })
  }
  const storedFingerprint = boundedString(episode.regionBindingFingerprint, 256) || null
  if (storedFingerprint && storedFingerprint !== fingerprint) {
    throw thoughtError('Checkpoint bölge bağı daha önceki bağla uyuşmuyor.', 'STALE_PLASTIC_REGION_BASE', 409)
  }
  return fingerprint
}

function pinnedRegionReplayPolicy(episode, current) {
  if ((episode.regionMode ?? 'off') === 'off') return current
  return {
    ...current,
    enabled: true,
    mode: episode.regionMode,
    representationProfile: 'adaptive-eighth-id-v1',
    compatibilityBlend: Number(episode.regionCompatibilityBlend ?? 0),
    minimumTrace: Number(episode.regionMinimumTrace ?? 0.02),
    maximumActiveGates: nonNegativeInteger(episode.regionMaximumActiveGates) || 4,
  }
}

export function toPublicThoughtEpisode(episode, { detail = false } = {}) {
  if (!episode) return null
  const preview = displayText(episode.inputText)
  return {
    id: boundedString(episode.id),
    status: boundedString(episode.status, 40),
    mode: boundedString(episode.mode, 40),
    commitEligible: Boolean(episode.commitEligible),
    commitStatus: boundedString(episode.commitStatus, 40) || 'not-eligible',
    learningMode: boundedString(episode.learningMode, 40) || 'off',
    teacherMode: boundedString(episode.teacherMode, 40) || (episode.crossCheckRequested ? 'ollama-cross-check' : 'off'),
    regionMode: boundedString(episode.regionMode, 40) || 'off',
    requestedRegionMode: boundedString(episode.requestedRegionMode, 40) || 'off',
    regionRuntime: {
      behaviorApplied: episode.regionMode === 'active',
      observationsEnabled: ['shadow', 'proposal', 'active'].includes(episode.regionMode),
      reason: boundedString(episode.regionModeReason, 160) || 'request-off',
      atlasArtifactId: boundedString(episode.regionAtlasArtifactId, MAX_ID_LENGTH) || null,
      atlasFingerprint: boundedString(episode.regionAtlasFingerprint, 256) || null,
      atlasRevision: episode.regionAtlasRevision === null || episode.regionAtlasRevision === undefined
        ? null : nonNegativeInteger(episode.regionAtlasRevision),
      geometryFingerprint: boundedString(episode.regionGeometryFingerprint, 256) || null,
      representationSpaceId: boundedString(episode.regionRepresentationSpaceId, 256) || null,
      ephemeralAtlas: Boolean(episode.regionEphemeralAtlas),
      compatibilityBlend: Math.max(0, Math.min(1, finite(episode.regionCompatibilityBlend) ?? 0)),
      minimumTrace: Math.max(0, Math.min(1, finite(episode.regionMinimumTrace) ?? 0)),
      maximumActiveGates: episode.regionMode === 'active'
        ? Math.max(1, Math.min(56, nonNegativeInteger(episode.regionMaximumActiveGates) || 4))
        : null,
      bindingFingerprint: boundedString(episode.regionBindingFingerprint, 256) || null,
      releaseManifestArtifactId: boundedString(episode.regionReleaseManifestArtifactId, MAX_ID_LENGTH) || null,
      releaseFingerprint: boundedString(episode.regionReleaseFingerprint, 256) || null,
    },
    input: { digest: boundedString(episode.inputDigest, 128), preview: preview.text, truncated: preview.truncated },
    seed: nonNegativeInteger(episode.seed),
    maxTicks: nonNegativeInteger(episode.maxTicks),
    strategy: boundedString(episode.strategy, 40),
    tickCount: nonNegativeInteger(episode.tickCount),
    lastTickDigest: boundedString(episode.lastTickDigest, 128) || null,
    stopReason: boundedString(episode.stopReason, 80) || null,
    crossCheck: episode.crossCheck ?? { status: episode.crossCheckRequested ? 'pending' : 'not-requested', trustWeight: OLLAMA_CROSS_CHECK_TRUST },
    outcomeEvidence: episode.teacherEvidence ?? null,
    teacherEvidence: episode.teacherEvidence ?? null,
    expectedOutcome: detail ? episode.expectedOutcome ?? null : undefined,
    base: publicBase(episode.base),
    baseDiagnostics: detail ? episode.baseDiagnostics ?? null : undefined,
    engine: detail ? episode.engineSummary ?? null : undefined,
    artifacts: detail ? {
      traceCount: Array.isArray(episode.traceArtifactIds) ? episode.traceArtifactIds.length : 0,
      graphAvailable: Boolean(episode.graphArtifactId),
      checkpointAvailable: Boolean(episode.checkpointArtifactId),
      headProposalAvailable: Boolean(episode.headProposalArtifactId),
      headAvailable: Boolean(episode.headArtifactId),
      candidateSnapshotId: episode.candidateSnapshotId ?? null,
    } : undefined,
    review: detail ? episode.review ?? null : undefined,
    error: episode.error ? {
      code: boundedString(episode.error.code, 100),
      message: boundedString(episode.error.message, 500),
    } : null,
    createdAt: episode.createdAt,
    updatedAt: episode.updatedAt,
    startedAt: episode.startedAt,
    finishedAt: episode.finishedAt,
  }
}

export function parseThoughtListQuery(searchParams) {
  return {
    offset: parseInteger(searchParams.get('offset'), { name: 'offset', fallback: 0, minimum: 0, maximum: 1_000_000 }),
    limit: parseInteger(searchParams.get('limit'), { name: 'limit', fallback: 20, minimum: 1, maximum: MAX_LIST_LIMIT }),
  }
}

export function parseThoughtTickQuery(searchParams) {
  return {
    after: parseInteger(searchParams.get('after'), { name: 'after', fallback: -1, minimum: -1, maximum: 1_000_000 }),
    limit: parseInteger(searchParams.get('limit'), { name: 'limit', fallback: 64, minimum: 1, maximum: MAX_TICK_PAGE_LIMIT }),
    candidateLimit: parseInteger(searchParams.get('candidateLimit'), { name: 'candidateLimit', fallback: 16, minimum: 0, maximum: MAX_PUBLIC_CANDIDATES_PER_TICK }),
  }
}

export function parseThoughtGraphQuery(searchParams) {
  const allowed = new Set(['episodeId', 'limit', 'level'])
  const unknown = [...searchParams.keys()].filter((key) => !allowed.has(key))
  if (unknown.length) throw thoughtError('Düşünce grafiği sorgusu bilinmeyen alan içeriyor.', 'INVALID_THOUGHT_QUERY', 400, { unknown })
  const level = searchParams.get('level') ?? 'neuron'
  if (!['neuron', 'family'].includes(level)) {
    throw thoughtError('level neuron veya family olmalı.', 'INVALID_THOUGHT_QUERY', 400)
  }
  return {
    episodeId: normalizeId(searchParams.get('episodeId'), 'episodeId'),
    limit: parseInteger(searchParams.get('limit'), { name: 'limit', fallback: 48, minimum: 1, maximum: MAX_GRAPH_LIMIT }),
    level,
  }
}

function graphEnvelope(level, rawNodes, rawEdges, limit, audit = null) {
  const edgeLimit = Math.min(MAX_GRAPH_EDGES, limit)
  const edges = rawEdges.slice(0, edgeLimit)
  const connected = new Set(edges.flatMap((edge) => [edge.from, edge.to]).filter(Boolean))
  const prioritizedNodes = [...rawNodes].sort((left, right) => (
    Number(connected.has(right.id)) - Number(connected.has(left.id)) || String(left.id).localeCompare(String(right.id))
  ))
  const nodes = prioritizedNodes.slice(0, Math.min(MAX_GRAPH_NODES, Math.max(edgeLimit * 2, 1)))
  return {
    ok: true,
    graph: {
      level,
      nodes,
      edges,
      totalNodeCount: rawNodes.length,
      totalEdgeCount: rawEdges.length,
      truncated: rawNodes.length > nodes.length || rawEdges.length > edges.length,
      ...(audit ? { audit } : {}),
    },
  }
}

export function deriveFamilyGraphView(base) {
  const memberships = new Map()
  for (const membership of base.familyRun.memberships ?? []) {
    if (!memberships.has(membership.memberKey)) memberships.set(membership.memberKey, [])
    memberships.get(membership.memberKey).push({ familyId: membership.familyId, weight: Math.max(0, finite(membership.weight) ?? 0) })
  }
  for (const birth of base.thoughtHead.births ?? []) {
    memberships.set(birth.id, (birth.familyMemberships ?? []).map((membership) => ({
      familyId: membership.familyId, weight: Math.max(0, finite(membership.weight) ?? 0),
    })))
  }
  const neuronStrength = new Map()
  for (const edge of base.familyRun.neuronTransitions ?? []) {
    neuronStrength.set(`${edge.fromMemberKey}\u0000${edge.toMemberKey}`, {
      from: edge.fromMemberKey, to: edge.toMemberKey,
      h0: Math.max(0, finite(edge.probability) ?? 0), excitation: 0, inhibition: 0,
    })
  }
  for (const edge of base.thoughtHead.neuronEdges ?? []) {
    const key = `${edge.fromMemberKey}\u0000${edge.toMemberKey}`
    const previous = neuronStrength.get(key) ?? {
      from: edge.fromMemberKey, to: edge.toMemberKey, h0: 0, excitation: 0, inhibition: 0,
    }
    previous.excitation += Math.max(0, finite(edge.excitation) ?? 0)
    previous.inhibition += Math.max(0, finite(edge.inhibition) ?? 0)
    neuronStrength.set(key, previous)
  }
  for (const birth of base.thoughtHead.births ?? []) {
    const addBirthEvidence = (from, to, weight) => {
      const key = `${from}\u0000${to}`
      const previous = neuronStrength.get(key) ?? { from, to, h0: 0, excitation: 0, inhibition: 0 }
      previous.excitation += Math.max(0, finite(weight) ?? 0)
      neuronStrength.set(key, previous)
    }
    for (const entry of birth.entryEvidence ?? []) addBirthEvidence(entry.memberKey, birth.id, entry.weight)
    for (const entry of birth.outgoingEvidence ?? []) addBirthEvidence(birth.id, entry.memberKey, entry.weight)
  }
  const rows = new Map()
  for (const edge of neuronStrength.values()) {
    if (!rows.has(edge.from)) rows.set(edge.from, [])
    rows.get(edge.from).push(edge)
  }
  const evidence = new Map()
  const epsilon = 1e-12
  for (const row of rows.values()) {
    const logits = row.map((edge) => Math.log(edge.h0 + edge.excitation + epsilon) - Math.log(1 + edge.inhibition))
    const maximum = Math.max(...logits)
    const masses = logits.map((logit) => Math.exp(logit - maximum))
    const massTotal = masses.reduce((sum, mass) => sum + mass, 0)
    for (let index = 0; index < row.length; index += 1) {
      const edge = row[index]
      const h = massTotal > 0 ? masses[index] / massTotal : 0
      for (const from of memberships.get(edge.from) ?? []) {
        for (const to of memberships.get(edge.to) ?? []) {
          const key = `${from.familyId}\u0000${to.familyId}`
          evidence.set(key, (evidence.get(key) ?? 0) + h * from.weight * to.weight)
        }
      }
    }
  }
  const outgoing = new Map()
  for (const [key, value] of evidence) {
    const [from] = key.split('\u0000')
    outgoing.set(from, (outgoing.get(from) ?? 0) + value)
  }
  const transitions = [...evidence].map(([key, value]) => {
    const [from, to] = key.split('\u0000')
    return {
      from,
      to,
      kind: 'family',
      probability: outgoing.get(from) > 0 ? value / outgoing.get(from) : 0,
      excitation: value,
      inhibition: null,
      trustMass: null,
    }
  }).sort((left, right) => left.from.localeCompare(right.from) || right.probability - left.probability || left.to.localeCompare(right.to))
  return {
    transitions,
    fingerprint: thoughtDigest('derived-family-graph', base.familyRun.fingerprint, base.thoughtHead.fingerprint, transitions),
  }
}

export function familyGraphNodes(base) {
  const nodes = (base.familyRun.families ?? []).map((family) => ({
    id: family.id,
    kind: 'family',
    label: boundedString(family.region, 80) || family.id,
    activation: null,
    lifecycle: boundedString(family.lifecycle, 40) || null,
    supportCount: nonNegativeInteger(family.supportCount),
    sourceCount: nonNegativeInteger(family.sourceCount),
  }))
  const knownFamilyIds = new Set(nodes.map((node) => node.id))
  for (const birth of base.thoughtHead.births ?? []) {
    for (const membership of birth.familyMemberships ?? []) {
      if (knownFamilyIds.has(membership.familyId)) continue
      knownFamilyIds.add(membership.familyId)
      nodes.push({
        id: membership.familyId,
        kind: 'family',
        label: membership.familyId,
        activation: null,
        lifecycle: 'provisional',
        region: null,
        supportCount: 1,
        sourceCount: new Set(birth.sourceGroupIds ?? []).size,
      })
    }
  }
  return nodes.sort((left, right) => left.id.localeCompare(right.id))
}

function readThoughtArtifact(store, artifactId, kind) {
  if (!artifactId) return null
  const artifact = store.getArtifact(artifactId)
  if (!artifact || artifact.kind !== kind) return null
  return artifactValue(artifact)
}

function episodeTraceArtifact(store, episode, artifactId) {
  const artifact = store.getArtifact(artifactId)
  if (
    !artifact || artifact.kind !== 'thought-trace-chunk' ||
    artifact.payload?.value?.format !== 'thought-trace-chunk' ||
    artifact.payload?.value?.version !== 1 || artifact.payload?.value?.episodeId !== episode.id
  ) throw thoughtError('Düşünce trace chunk eseri eksik veya uyumsuz.', 'INVALID_THOUGHT_TRACE', 409)
  return artifact
}

function restoreEpisodeTraceChunk(store, episode, artifactId) {
  const chunk = artifactValue(episodeTraceArtifact(store, episode, artifactId))
  if (
    chunk?.format !== 'thought-trace-chunk' || chunk.version !== 1 || chunk.episodeId !== episode.id ||
    !Array.isArray(chunk.ticks) || chunk.ticks.length > 64
  ) throw thoughtError('Düşünce trace chunk içeriği uyumsuz.', 'INVALID_THOUGHT_TRACE', 409)
  return chunk
}

function readEpisodeTickPage(store, episode, { after, limit }) {
  const selected = []
  const seen = new Set()
  for (const artifactId of episode.traceArtifactIds ?? []) {
    const artifact = episodeTraceArtifact(store, episode, artifactId)
    const encodedLastIndex = artifact.payload.value.lastIndex
    if (Number.isSafeInteger(encodedLastIndex) && encodedLastIndex <= after) continue
    const chunk = restoreArtifact(artifact, 'thought-trace-chunk')
    for (const tick of chunk.ticks) {
      if (!Number.isSafeInteger(tick?.index) || seen.has(tick.index) || tick.index <= after) continue
      seen.add(tick.index)
      selected.push(tick)
      if (selected.length > limit) return { ticks: selected.slice(0, limit), hasMore: true }
    }
  }
  return { ticks: selected, hasMore: false }
}

function readEpisodeTickManifest(store, episode) {
  const manifest = []
  let previousDigest = null
  let previousIndex = null
  for (const artifactId of episode.traceArtifactIds ?? []) {
    const chunk = restoreEpisodeTraceChunk(store, episode, artifactId)
    for (const tick of chunk.ticks) {
      if (
        !Number.isSafeInteger(tick?.index) || typeof tick?.tickDigest !== 'string' ||
        tick.previousTickDigest !== previousDigest ||
        (previousIndex !== null && tick.index <= previousIndex)
      ) throw thoughtError('Düşünce replay manifest zinciri bozuk.', 'INVALID_THOUGHT_TRACE', 409)
      manifest.push({ index: tick.index, tickDigest: tick.tickDigest })
      previousIndex = tick.index
      previousDigest = tick.tickDigest
    }
  }
  if (manifest.length !== nonNegativeInteger(episode.tickCount) || previousDigest !== (episode.lastTickDigest ?? null)) {
    throw thoughtError('Düşünce replay manifesti episode özetiyle uyuşmuyor.', 'INVALID_THOUGHT_TRACE', 409)
  }
  return manifest
}

function episodeTraceCursor(store, episode) {
  const count = nonNegativeInteger(episode?.tickCount)
  const lastTickDigest = boundedString(episode?.lastTickDigest, 128) || null
  if (count === 0) return { count: 0, lastIndex: null, lastTickDigest: null }
  const artifactIds = Array.isArray(episode?.traceArtifactIds) ? episode.traceArtifactIds : []
  const lastArtifactId = artifactIds.at(-1)
  const chunk = restoreEpisodeTraceChunk(store, episode, lastArtifactId)
  const lastTick = Array.isArray(chunk?.ticks) ? chunk.ticks.at(-1) : null
  if (
    !chunk || chunk.format !== 'thought-trace-chunk' || chunk.version !== 1 || chunk.episodeId !== episode.id ||
    !Number.isSafeInteger(lastTick?.index) || !lastTick?.tickDigest || lastTick.tickDigest !== lastTickDigest
  ) {
    throw thoughtError('Düşünce trace cursor kanıtı son kalıcı chunk ile uyuşmuyor.', 'INVALID_THOUGHT_TRACE', 409)
  }
  return { count, lastIndex: lastTick.index, lastTickDigest }
}

function publicTraceValue(value) {
  if (value instanceof Float32Array) return Array.from(value)
  if (Array.isArray(value)) return value.map(publicTraceValue)
  if (isRecord(value)) return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, publicTraceValue(item)]))
  return value
}

function engineSummary(value) {
  if (!isRecord(value)) return null
  const evaluation = isRecord(value.evaluation) ? {
    // These flags remain audit-only; the thought candidate created below is
    // deliberately not promoted by teacher evidence.
    passed: value.evaluation.passed === true,
    runtimeVerified: value.evaluation.runtimeVerified === true,
  } : null
  return {
    status: boundedString(value.status, 80) || null,
    stopReason: boundedString(value.stopReason, 80) || null,
    explanation: boundedString(value.explanation, 500) || null,
    evaluation,
    stats: isRecord(value.stats) ? strictBoundedValue(value.stats) : {},
  }
}

function crossCheckMessages(inputText, role) {
  const contract = "Tam olarak şu EventGraph alanlarını üret: format='event-graph-v1', actor/action/patient/instrument string veya null, operators string[], negated boolean, goals {relation,target}[]. Metinleri NFC, küçük harf ve kısa tut."
  return role === 'extraction'
    ? [
      { role: 'system', content: `Sen yalnız yapısal olay çıkaran yerel bir yardımcı denetçisin. ${contract} Tahmin etme; belirsizlikte confidence düşür.` },
      { role: 'user', content: `Bu girdinin beklenen olay grafiğini çıkar:\n${inputText}` },
    ]
    : [
      { role: 'system', content: `Sen ilk çıkarımı görmeyen bağımsız ve kuşkucu bir olay grafiği eleştirmenisin. ${contract} Yalnız metnin açıkça desteklediği alanları doldur.` },
      { role: 'user', content: `Bu girdiyi bağımsız olarak denetle ve olay grafiğini çıkar:\n${inputText}` },
    ]
}

function assertCrossCheckControlState(store, episodeId, signal) {
  signal?.throwIfAborted()
  const status = store.getThoughtEpisode(episodeId)?.status
  if (status === 'cancel_requested' || status === 'cancelled') {
    throw thoughtError('Düşünce bölümü iptal edildi.', 'THOUGHT_CANCELLED', 409)
  }
  if (status === 'pause_requested' || status === 'paused') {
    throw thoughtError('Düşünce bölümü duraklatıldı.', 'THOUGHT_PAUSED', 409)
  }
}

async function runOllamaCrossCheck({ store, ollama, episodeId, inputText, signal }) {
  const selectedModel = String(store.getConfig().ollama.model || '').trim()
  if (!selectedModel) {
    const error = new Error('Ollama modeli seçilmedi.')
    error.code = 'OLLAMA_MODEL_REQUIRED'
    throw error
  }
  const passes = []
  for (const role of ['extraction', 'critic']) {
    const messages = crossCheckMessages(inputText, role)
    const maxCompletionTokens = 512
    // A control request may be written by another service instance, whose
    // local controller cannot abort this process. Re-read the durable episode
    // immediately before every paid reservation so no later pass starts after
    // that pause/cancel request.
    assertCrossCheckControlState(store, episodeId, signal)
    const usageReservation = reserveBoundedTeacherCall(store, {
      messages,
      schema: crossCheckSchema,
      maxCompletionTokens,
      model: selectedModel,
      signal,
      thoughtEpisodeId: episodeId,
    })
    let result
    try {
      result = await ollama.generateStructured({
        model: selectedModel,
        messages,
        schema: crossCheckSchema,
        signal,
        retries: 0,
        options: { num_predict: maxCompletionTokens },
      })
    } catch (error) {
      settleBoundedTeacherCall(store, usageReservation, error)
      throw error
    }
    const settledUsage = settleBoundedTeacherCall(store, usageReservation, result)
    passes.push({
      role,
      graph: normalizeOllamaGraph(result.value.graph),
      confidence: probability(result.value.confidence) ?? 0,
      model: boundedString(result.model, 160),
      promptTokens: settledUsage.promptTokens,
      completionTokens: settledUsage.completionTokens,
      usageEstimated: settledUsage.usageEstimated,
    })
  }
  const agrees = JSON.stringify(passes[0].graph) === JSON.stringify(passes[1].graph)
  const confidence = Math.min(passes[0].confidence, passes[1].confidence)
  const gatePassed = agrees && confidence >= OLLAMA_CROSS_CHECK_MIN_CONFIDENCE
  const model = passes.map((pass) => pass.model).filter(Boolean).sort().join('+') || 'ollama'
  const sourceGroupId = `ollama-cross-check:${digest(model, inputText).slice(0, 32)}`
  return {
    status: gatePassed ? 'passed' : 'disagreed',
    gatePassed,
    agrees,
    confidence,
    minimumConfidence: OLLAMA_CROSS_CHECK_MIN_CONFIDENCE,
    trustWeight: OLLAMA_CROSS_CHECK_TRUST,
    graph: gatePassed ? passes[0].graph : null,
    passes,
    provenance: {
      kind: 'ollama-cross-check',
      evidenceRole: 'auxiliary-critic',
      evaluationEligible: false,
      datasetSplit: null,
      sourceGroupId,
    },
  }
}

function thoughtResidencyPolicy(config) {
  const cacheBytes = Math.max(32, Number(config.resourcePolicy?.neuronCacheMb ?? 384)) * 1024 * 1024
  const pageBytes = Math.max(64 * 1024, Number(config.resourcePolicy?.neuronPageBytes ?? 2 * 1024 * 1024))
  return {
    hotBudgetBytes: Math.max(1, Math.floor(cacheBytes * 0.75)),
    warmBudgetBytes: Math.max(1, Math.floor(cacheBytes * 0.25)),
    pageTargetBytes: pageBytes,
    maximumPageBytes: Math.max(pageBytes, pageBytes * 2),
    maximumWorkingSetNeurons: 128,
  }
}

function residencyBaseFingerprint(base) {
  return digest(
    'thought-residency-base-v2',
    base.reference.baseModelFingerprint,
    base.thoughtHead.fingerprint,
    base.reference.representationProfile ?? null,
    base.familyRun.representationSpaceId,
  )
}

function neuronSummary(neuron) {
  return {
    memberKey: neuron.memberKey ?? neuron.id,
    kind: neuron.kind ?? neuron.provenance?.kind ?? 'raw',
    text: boundedString(neuron.text ?? neuron.provenance?.surface, 2_000),
    identityDigest: neuron.identityDigest,
    relationLength: neuron.relation?.length ?? null,
  }
}

function snapshotMacroConstituentIds(base, activeNeuronIds, repository, maximum = 32) {
  const head = base.conceptMacroHead
  if (
    !head || head.format !== 'concept-macro-head' || head.version !== 1 ||
    head.representationSpaceId !== base.familyRun.representationSpaceId ||
    head.familyRunFingerprint !== base.familyRun.fingerprint || !Array.isArray(head.macros)
  ) return []
  const byId = new Map(head.macros.map((macro) => [macro.id, macro]))
  const active = new Set(activeNeuronIds)
  const output = []
  const seen = new Set()
  const expand = (macroId, depth, visiting) => {
    if (depth > 3 || visiting.has(macroId)) return null
    const macro = byId.get(macroId)
    if (!macro || !Array.isArray(macro.orderedExpansion)) return null
    visiting.add(macroId)
    const result = []
    for (const member of macro.orderedExpansion) {
      if (member?.kind === 'raw-neuron') result.push(member.id)
      else if (member?.kind === 'composite-neuron') {
        const nested = expand(member.id, depth + 1, visiting)
        if (!nested) { visiting.delete(macroId); return null }
        result.push(...nested)
      } else { visiting.delete(macroId); return null }
      if (result.length > 128) { visiting.delete(macroId); return null }
    }
    visiting.delete(macroId)
    return result
  }
  for (const macro of [...head.macros]
    .filter((candidate) => candidate.lifecycle === 'active')
    .sort((left, right) => String(left.id).localeCompare(String(right.id)))) {
    const expanded = expand(macro.id, 1, new Set())
    if (!expanded?.length || !active.has(expanded[0])) continue
    for (const id of expanded) {
      if (!repository.has(id) || seen.has(id)) continue
      seen.add(id)
      output.push(id)
      if (output.length >= maximum) return output
    }
  }
  return output
}

export function candidateWorkingSet(base, residency, signal, seed, explicitIds = [], activeNeuronIds = []) {
  const maximum = 128
  const signalQuery = residency.queryBySignal(signal, maximum)
  const nearest = signalQuery.candidates
  const distanceById = new Map(nearest.map((entry) => [entry.id, entry.distance]))
  const topology = residency.routingTopology
  if (!topology) throw thoughtError('SSD nöron yönlendirme indeksi bulunamadı.', 'NEURON_ROUTING_INDEX_MISSING', 500)
  const preview = [...topology.previewIds]
  const explicit = [...new Set(explicitIds)].filter((id) => residency.repository.has(id)).sort().slice(0, 32)
  const active = [...new Set(activeNeuronIds)].filter((id) => residency.repository.has(id)).slice(0, 96)
  const macroConstituents = snapshotMacroConstituentIds(base, active, residency.repository, 32)
  const signalIds = nearest.slice(0, 72).map((entry) => entry.id)
  const seeds = [...new Set([...active, ...explicit, ...preview, ...signalIds.slice(0, 24)])].slice(0, 128)
  const seedMembershipQuery = topology.membershipsFor(seeds, 128)
  const memberships = seedMembershipQuery.rows
  const routeProbabilityById = new Map()
  const routeQuery = topology.topEdgesFrom(seeds, {
    maximumOrigins: 32,
    maximumPerOrigin: 8,
    maximumEntries: 256,
  })
  const routeIds = routeQuery.rows.map((transition) => {
      routeProbabilityById.set(transition.toId, Math.max(
        routeProbabilityById.get(transition.toId) ?? 0,
        probability(transition.probability) ?? 0,
      ))
      return transition.toId
    })
  const familyWeights = new Map()
  for (const rows of memberships.values()) {
    for (const membership of rows) {
      familyWeights.set(membership.familyId, Math.max(
        familyWeights.get(membership.familyId) ?? 0,
        membership.weight,
      ))
    }
  }
  const familyIds = [...familyWeights]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 16)
    .map(([familyId]) => familyId)
  const familyQuery = topology.topMembersForFamilies(familyIds, {
    maximumFamilies: 16,
    maximumPerFamily: 4,
  })
  const familyMembers = familyQuery.rows.map((entry) => entry.id)
  const exploration = residency.sampleIds(`episode:${seed}`, 16)
  const sources = [explicit, preview, signalIds, routeIds, familyMembers, exploration]
  const selected = [...new Set([...active, ...macroConstituents])].slice(0, maximum)
  const seen = new Set(selected)
  const maximumSourceLength = sources.reduce((current, source) => Math.max(current, source.length), 0)
  for (let rank = 0; rank < maximumSourceLength && selected.length < maximum; rank += 1) {
    for (const source of sources) {
      const id = source[rank]
      if (!id || seen.has(id)) continue
      seen.add(id)
      selected.push(id)
      if (selected.length >= maximum) break
    }
  }
  const explicitSet = new Set(explicit)
  const previewSet = new Set(preview)
  const selectedMembershipQuery = topology.membershipsFor(selected, 128)
  const familyProbabilityById = new Map([...selectedMembershipQuery.rows].map(([id, rows]) => [
    id,
    rows.reduce((maximumWeight, membership) => Math.max(maximumWeight, membership.weight), 0),
  ]))
  const requests = selected.map((id) => {
    const distance = distanceById.get(id) ?? 1
    return {
      id,
      evidence: {
        distance,
        routeProbability: routeProbabilityById.get(id) ?? 0,
        familyProbability: Math.min(1, familyProbabilityById.get(id) ?? 0),
        observedUseProbability: explicitSet.has(id) ? 1 : previewSet.has(id) ? 0.65 : Math.max(0, 1 - distance),
      },
    }
  })
  return {
    requests,
    diagnostics: {
      mode: 'ssd-gated-working-set-v1',
      routingSemantics: 'dynamic-bounded-approximation',
      refreshCadence: 'every-tick',
      manifestFingerprint: residency.manifest.fingerprint,
      catalogNeuronCount: residency.repository.size,
      workingSetLimit: maximum,
      selectedNeuronCount: requests.length,
      nearestCandidateCount: nearest.length,
      signalIndexMethod: signalQuery.diagnostics.method,
      signalBucketProbes: signalQuery.diagnostics.bucketProbes,
      signalIndexEntriesInspected: signalQuery.diagnostics.inspectedIndexEntries,
      explicitCandidateCount: explicit.length,
      activePathCandidateCount: active.length,
      macroConstituentCandidateCount: macroConstituents.length,
      previewCandidateCount: preview.length,
      routeCandidateCount: new Set(routeIds).size,
      familyCandidateCount: new Set(familyMembers).size,
      boundedIndexEntriesInspected:
        signalQuery.diagnostics.inspectedIndexEntries +
        seedMembershipQuery.inspectedIndexEntries +
        routeQuery.inspectedIndexEntries +
        familyQuery.inspectedIndexEntries +
        selectedMembershipQuery.inspectedIndexEntries +
        exploration.length,
      boundedIndexInspectionLimit: 1_104,
    },
  }
}

export class ThoughtService {
  constructor({ store, ollama, thoughtEngine = null, onEvent = () => {} }) {
    this.store = store
    this.ollama = ollama
    this.engine = thoughtEngine
    this.onEvent = onEvent
    this.active = new Map()
    this.baseCache = new Map()
    this.residencies = new Map()
    this.residencyPromises = new Map()
    this.residencyMeta = new Map()
    this.residencyPins = new Map()
    this.baseCacheAccess = new Map()
    this.cacheClock = 0
    this.store.recoverThoughtEpisodes()
  }

  get available() { return typeof this.engine?.runEpisode === 'function' }
  get residencyAvailable() {
    return typeof this.engine?.prepareResidencyBase === 'function' &&
      typeof this.engine?.encodeResidencyPrompt === 'function'
  }

  pinnedSnapshotIds() {
    const result = new Set()
    for (const episodeId of this.active.keys()) {
      const snapshotId = this.store.getThoughtEpisode(episodeId)?.base?.snapshotId
      if (snapshotId) result.add(snapshotId)
    }
    return result
  }

  touchBaseCache(snapshotId) {
    this.cacheClock += 1
    this.baseCacheAccess.set(snapshotId, this.cacheClock)
  }

  trimBaseCache(keepSnapshotId = null) {
    const pinned = this.pinnedSnapshotIds()
    const activeSnapshotId = this.store.getActiveSnapshot()?.id ?? null
    while (this.baseCache.size > 2) {
      const candidates = [...this.baseCache.keys()]
        .filter((snapshotId) => snapshotId !== keepSnapshotId && snapshotId !== activeSnapshotId && !pinned.has(snapshotId))
        .sort((left, right) => (this.baseCacheAccess.get(left) ?? 0) - (this.baseCacheAccess.get(right) ?? 0) ||
          left.localeCompare(right))
      if (!candidates.length) break
      const victim = candidates[0]
      this.baseCache.delete(victim)
      this.baseCacheAccess.delete(victim)
    }
  }

  touchResidency(key, snapshotId) {
    this.cacheClock += 1
    this.residencyMeta.set(key, { snapshotId, lastAccess: this.cacheClock })
  }

  trimResidencies(keepKey = null) {
    const pinnedSnapshots = this.pinnedSnapshotIds()
    const activeSnapshotId = this.store.getActiveSnapshot()?.id ?? null
    while (this.residencies.size > 2) {
      const candidates = [...this.residencies.keys()]
        .filter((key) => {
          const meta = this.residencyMeta.get(key)
          return key !== keepKey && (this.residencyPins.get(key) ?? 0) === 0 &&
            meta?.snapshotId !== activeSnapshotId && !pinnedSnapshots.has(meta?.snapshotId)
        })
        .sort((left, right) => (this.residencyMeta.get(left)?.lastAccess ?? 0) -
          (this.residencyMeta.get(right)?.lastAccess ?? 0) || left.localeCompare(right))
      if (!candidates.length) break
      const victim = candidates[0]
      this.residencies.get(victim)?.repository.close()
      this.residencies.delete(victim)
      this.residencyMeta.delete(victim)
      this.residencyPins.delete(victim)
      this.residencyPromises.delete(victim)
    }
  }

  async emit(event) {
    await this.onEvent({ ...event, data: { ...(event.data ?? {}), subsystem: 'thought' } })
  }

  assertHydrationAdmission(selection) {
    const config = this.store.getConfig()
    const metadata = thoughtHydrationMetadata(this.store, selection.snapshot)
    const aggregatePayloadBytes = metadata.reduce((sum, entry) => sum + hydrationPayloadBytes(entry), 0)
    // JSON/base64 decode, Float32 materialization and the validated model clone
    // overlap briefly on first bootstrap. The 2.5x envelope is deliberately
    // conservative and is checked before getArtifact/restoreArtifact.
    const estimatedTransientBytes = Math.ceil(aggregatePayloadBytes * 2.5)
    const sample = sampleHostResources({ dataDirectory: config.dataDirectory ?? this.store.dataDirectory })
    const projectedSample = {
      ...sample,
      rssBytes: sample.rssBytes + estimatedTransientBytes,
      heapUsedBytes: sample.heapUsedBytes + estimatedTransientBytes,
      hostFreeMemoryBytes: Math.max(0, sample.hostFreeMemoryBytes - estimatedTransientBytes),
    }
    const state = config.resourcePolicy?.enabled === false
      ? { allowed: true, reasonCodes: [], memoryTier: 'disabled', diskTier: 'disabled' }
      : evaluateResourceState(config, projectedSample, {
        logicalStorageBytes: this.store.approximateStorageBytes(),
        admission: true,
      })
    if (!state.allowed) {
      throw thoughtError(
        'Nöron tabanı güvenli RAM/SSD rezervi sağlanana kadar decode edilmedi.',
        'NEURON_RESIDENCY_RESOURCE_WAIT',
        503,
        {
          retryable: true,
          phase: 'metadata-hydration-preflight',
          artifactCount: metadata.length,
          aggregatePayloadBytes,
          estimatedTransientBytes,
          rssBytes: sample.rssBytes,
          projectedRssBytes: projectedSample.rssBytes,
          hostFreeMemoryBytes: sample.hostFreeMemoryBytes,
          projectedHostFreeMemoryBytes: projectedSample.hostFreeMemoryBytes,
          reasonCodes: state.reasonCodes,
        },
      )
    }
    return {
      phase: 'metadata-hydration-preflight',
      artifactCount: metadata.length,
      aggregatePayloadBytes,
      estimatedTransientBytes,
      projectedMemoryTier: state.memoryTier,
      projectedDiskTier: state.diskTier,
      limitation: 'first bootstrap still decodes one validated source catalog before SSD paging',
    }
  }

  async ensureResidency(base) {
    if (!this.residencyAvailable) return null
    const key = residencyBaseFingerprint(base)
    const ready = this.residencies.get(key)
    if (ready) {
      this.touchResidency(key, base.reference.snapshotId)
      return ready
    }
    const pending = this.residencyPromises.get(key)
    if (pending) return pending
    const promise = (async () => {
      if (!Array.isArray(base.neurons) || !base.neurons.length) {
        throw thoughtError('SSD nöron kapısı hazırlanamadı; fiziksel I/R kataloğu boş.', 'NEURON_RESIDENCY_UNAVAILABLE', 503)
      }
      const config = this.store.getConfig()
      const prepared = this.engine.prepareResidencyBase({
        neurons: base.neurons,
        familyRun: base.familyRun,
        thoughtHead: base.thoughtHead,
        baseModelFingerprint: base.reference.baseModelFingerprint,
      })
      const runtimeModelFingerprint = prepared.model.fingerprint
      const routingTopology = createThoughtNeuronRoutingTopology({
        model: prepared.model,
        head: prepared.head,
        routingPreview: base.familyRun.routingPreview?.entries ?? [],
      })
      const summaries = base.neurons.map(neuronSummary)
      // prepareResidencyBase currently has to validate/clone the first dense
      // bootstrap catalog. Release the decoded artifact copy before encoding
      // SSD pages so both N-sized vector graphs are never retained together.
      base.neuronSummaries = summaries
      base.neurons = []
      base.neuronByKey = new Map(summaries.map((summary) => [summary.memberKey, {
        ...summary,
        relation: { length: summary.relationLength },
      }]))
      const policy = thoughtResidencyPolicy(config)
      const maximumSnapshotBytes = Math.max(
        64 * 1024 * 1024,
        Math.min(2 * 1024 * 1024 * 1024, Math.floor(config.budgets.diskMb * 1024 * 1024 / 2)),
      )
      const uniqueNeuronCount = new Set([
        ...prepared.model.neurons.map((neuron) => neuron.id),
        ...(prepared.head.births ?? []).map((birth) => birth.id),
      ]).size
      const estimatedSnapshotBytes = uniqueNeuronCount * (prepared.model.layout.cellCount * 4 + 8 * 1024) +
        Math.ceil(uniqueNeuronCount / 100) * policy.pageTargetBytes / 100 + 64 * 1024
      const sample = sampleHostResources({ dataDirectory: config.dataDirectory ?? this.store.dataDirectory })
      const resourceState = evaluateResourceState(config, sample, {
        logicalStorageBytes: this.store.approximateStorageBytes(),
        admission: true,
      })
      const currentTierBytes = await measureThoughtNeuronResidencyBytes(config.dataDirectory ?? this.store.dataDirectory)
      const tierRepositoryCapBytes = Math.min(2 * 1024 * 1024 * 1024, Math.floor(config.budgets.diskMb * MEBIBYTE / 2))
      const preservesPhysicalReserve = resourceState.usableDiskFreeBytes - estimatedSnapshotBytes >=
        config.resourcePolicy.minDiskFreeStartMb * MEBIBYTE
      if (!resourceState.allowed || !preservesPhysicalReserve ||
          estimatedSnapshotBytes > maximumSnapshotBytes || currentTierBytes + estimatedSnapshotBytes > tierRepositoryCapBytes) {
        throw thoughtError(
          'SSD nöron kapısı güvenli fiziksel rezerv sağlanana kadar hazırlanmadı.',
          'NEURON_RESIDENCY_RESOURCE_WAIT',
          503,
          {
            retryable: true,
            reasonCodes: resourceState.reasonCodes,
            estimatedSnapshotBytes: Math.ceil(estimatedSnapshotBytes),
            currentTierBytes,
            tierRepositoryCapBytes,
            usableDiskFreeBytes: resourceState.usableDiskFreeBytes,
          },
        )
      }
      const created = await createThoughtNeuronResidencySnapshot({
        dataDirectory: config.dataDirectory ?? this.store.dataDirectory,
        model: prepared.model,
        head: prepared.head,
        representationSpaceId: base.familyRun.representationSpaceId,
        representationProfile: base.reference.representationProfile ?? null,
        policy,
        maximumSnapshotBytes,
        residencyModelFingerprint: key,
      })
      const residency = await openThoughtNeuronResidency({
        dataDirectory: config.dataDirectory ?? this.store.dataDirectory,
        manifestPath: created.manifestPath,
        manifestFingerprint: created.manifest.fingerprint,
        expectedModelFingerprint: key,
        expectedRepresentationSpaceId: base.familyRun.representationSpaceId,
        expectedRepresentationProfile: base.reference.representationProfile ?? null,
        expectedThoughtModelFingerprint: runtimeModelFingerprint,
        policy,
        routingTopology,
      })
      residency.runtimeModelFingerprint = runtimeModelFingerprint
      // The repository and compact topology are now authoritative. Do not
      // leave the engine's validated dense clone reachable from this promise.
      prepared.model.neurons.length = 0
      base.diagnostics = {
        ...base.diagnostics,
        residency: {
          mode: 'ssd-gated-working-set-v1',
          manifestFingerprint: created.manifest.fingerprint,
          neuronCount: created.manifest.neuronCount,
          pageCount: created.manifest.pageCount,
          pageBytes: created.manifest.pageBytes,
          workingSetLimit: 128,
          hotBudgetBytes: policy.hotBudgetBytes,
          warmBudgetBytes: policy.warmBudgetBytes,
        },
      }
      this.residencies.set(key, residency)
      this.touchResidency(key, base.reference.snapshotId)
      this.trimResidencies(key)
      return residency
    })().catch((error) => {
      // A blocked/failed residency attempt must not turn the dense migration
      // input into an accidental permanent RAM cache. The next explicit retry
      // re-resolves the immutable snapshot.
      this.baseCache.delete(base.reference.snapshotId)
      this.baseCacheAccess.delete(base.reference.snapshotId)
      throw error
    }).finally(() => this.residencyPromises.delete(key))
    this.residencyPromises.set(key, promise)
    return promise
  }

  async withEpisodeWorkingSet(base, episode, callback, state = null, adaptiveContext = null) {
    const residency = await this.ensureResidency(base)
    if (!residency) return callback({ neurons: base.neurons, diagnostics: { mode: 'legacy-full-catalog' } })
    const branches = Array.isArray(state?.branches) ? state.branches : []
    const dominant = [...branches].sort((left, right) => (right.mass ?? 0) - (left.mass ?? 0) ||
      String(left.id).localeCompare(String(right.id)))[0]
    const promptSignal = dominant?.signal instanceof Float32Array
      ? dominant.signal
      : await this.engine.encodeResidencyPrompt(
        episode.inputText,
        base.layout,
        base.familyRun.representationSpaceId,
        adaptiveContext ?? undefined,
      )
    const activeNeuronIds = [...new Set([
      ...branches.map((branch) => branch.currentNeuronId),
      ...branches.flatMap((branch) => (branch.path ?? []).slice(-12).reverse()),
    ].filter(Boolean))]
    const selection = candidateWorkingSet(
      base,
      residency,
      promptSignal,
      episode.seed,
      episode.candidateMemberKeys ?? [],
      activeNeuronIds,
    )
    if (!selection.requests.length) {
      throw thoughtError('SSD nöron kapısı başlangıç çalışma kümesi üretemedi.', 'NEURON_WORKING_SET_EMPTY', 409)
    }
    const key = residencyBaseFingerprint(base)
    this.residencyPins.set(key, (this.residencyPins.get(key) ?? 0) + 1)
    try {
      return await residency.withCandidateLookup(selection.requests, async (workingSet) => {
        if (!workingSet.size || workingSet.size > 128) {
          throw thoughtError('SSD nöron çalışma kümesi 1..128 sınırını ihlal etti.', 'NEURON_WORKING_SET_LIMIT', 500)
        }
        return callback({
          neurons: [...workingSet.values()],
          diagnostics: {
            ...selection.diagnostics,
            tiersBeforeRun: residency.repository.getStats(),
          },
        })
      }, { maximum: 128 })
    } finally {
      const pins = (this.residencyPins.get(key) ?? 1) - 1
      if (pins > 0) this.residencyPins.set(key, pins)
      else this.residencyPins.delete(key)
      this.trimResidencies()
    }
  }

  async runEngineEpisode(request, context, base, episode) {
    const residency = await this.ensureResidency(base)
    if (!residency) return this.engine.runEpisode(request, { ...context, neurons: base.neurons })
    const requestedReplayLimit = request.replay?.verifyOnly
      ? request.replay.expectedTickCount
      : request.maxTicks
    let resume = request.resume ?? null
    let result = null
    // One deterministic tick per residency window. S_t therefore re-ranks the
    // SSD catalog before every transition, while only the selected <=128 exact
    // I/R rows are pinned in RAM for that tick.
    while (true) {
      const currentTick = nonNegativeInteger(resume?.tick)
      const targetTick = Math.min(requestedReplayLimit, currentTick + 1)
      result = await this.withEpisodeWorkingSet(base, episode, ({ neurons, diagnostics }) => (
        this.engine.runEpisode({
          ...request,
          resume,
          replay: {
            verifyOnly: true,
            expectedTickCount: targetTick,
            ...(request.replay?.adaptiveRegion
              ? { adaptiveRegion: request.replay.adaptiveRegion }
              : {}),
          },
        }, {
          ...context,
          neurons,
          modelFingerprintOverride: residency.runtimeModelFingerprint,
          baseDiagnostics: { ...base.diagnostics, residencyEpisode: diagnostics },
        })
      ), resume, (request.base?.representationProfile || request.adaptiveRegion) ? {
        representationProfile: request.base?.representationProfile ?? null,
        adaptiveRegion: request.adaptiveRegion,
        adaptiveRegionAtlas: context.adaptiveRegionAtlas,
        sourceGroupId: request.sourceGroupId,
      } : null)
      if (!isRecord(result?.episode)) return result
      const nextTick = nonNegativeInteger(result.episode.tick)
      if (result.episode.status !== 'paused' || nextTick >= requestedReplayLimit || nextTick <= currentTick) return result
      resume = result.episode
    }
  }

  resolveBase(snapshotId) {
    const selection = selectBaseSnapshot(this.store, snapshotId)
    const manifestHash = digest(
      modelManifest(this.store, selection.snapshot),
      selection.active?.id ?? null,
      selection.active?.evaluation?.passed === true,
      selection.active?.evaluation?.runtimeVerified === true,
    )
    const cached = this.baseCache.get(selection.snapshot.id)
    if (cached?.manifestHash === manifestHash) {
      this.touchBaseCache(selection.snapshot.id)
      if (cached.error) {
        throw thoughtError(
          cached.error.message,
          cached.error.code,
          cached.error.status,
          cached.error.details,
        )
      }
      return cached.value
    }
    try {
      const hydrationPreflight = this.assertHydrationAdmission(selection)
      const value = resolveEngineBase(this.store, selection)
      value.diagnostics = { ...value.diagnostics, hydrationPreflight }
      this.baseCache.set(selection.snapshot.id, { manifestHash, value })
      this.touchBaseCache(selection.snapshot.id)
      this.trimBaseCache(selection.snapshot.id)
      return value
    } catch (error) {
      // Status is polled while a replacement base is being trained. Cache
      // deterministic 4xx incompatibilities by the immutable artifact
      // manifest so a legacy 32x32 base is not decoded on every poll.
      if (Number(error?.status) >= 400 && Number(error?.status) < 500) {
        this.baseCache.set(selection.snapshot.id, {
          manifestHash,
          error: {
            message: boundedString(error?.message, 500) || 'Düşünce tabanı çözülemedi.',
            code: boundedString(error?.code, 80) || 'INVALID_THOUGHT_BASE',
            status: Number(error.status),
            details: error.details,
          },
        })
        this.touchBaseCache(selection.snapshot.id)
        this.trimBaseCache(selection.snapshot.id)
      }
      throw error
    }
  }

  async status() {
    const episodes = this.store.listThoughtEpisodes()
    const active = episodes.find((episode) => ['queued', 'running', 'pause_requested', 'cancel_requested'].includes(episode.status)) ?? null
    let researchBaseAvailable = false
    let researchBase = null
    let researchBaseError = null
    let resolvedBaseForStatus = null
    try {
      // A FamilyRun artifact by itself is not a usable thought base. Resolve
      // the complete immutable I/R catalog here so legacy 32x32 snapshots do
      // not make the UI claim that a 64x64 research base is ready.
      const resolved = this.resolveBase(null)
      resolvedBaseForStatus = resolved
      await this.ensureResidency(resolved)
      const residency = this.residencies.get(residencyBaseFingerprint(resolved)) ?? null
      researchBaseAvailable = true
      researchBase = {
        snapshotId: resolved.reference.snapshotId,
        snapshotStatus: resolved.reference.snapshotStatus,
        selectedBy: resolved.reference.selectedBy,
        representationSpaceId: resolved.familyRun.representationSpaceId,
        representationProfile: resolved.representationProfile,
        matrixSide: resolved.layout.side,
        cellCount: resolved.layout.cellCount,
        neuronCount: resolved.diagnostics?.resolvedNeuronCount ?? resolved.neuronSummaries?.length ?? resolved.neurons.length,
        familyCount: resolved.familyRun.families.length,
        commitEligible: resolved.reference.commitEligible,
        conceptMacros: {
          status: resolved.diagnostics?.conceptMacros?.status ?? 'absent',
          artifactId: resolved.reference.conceptMacroHeadArtifactId ?? null,
          fingerprint: resolved.reference.conceptMacroHeadFingerprint ?? null,
        },
        residency: resolved.diagnostics?.residency
          ? { ...resolved.diagnostics.residency, stats: residency?.repository.getStats() ?? null }
          : null,
      }
    } catch (error) {
      researchBaseError = {
        code: boundedString(error?.code, 80) || 'THOUGHT_BASE_NOT_FOUND',
        message: boundedString(error?.message, 500) || '64x64 düşünce tabanı çözülemedi.',
      }
    }
    const plasticRegionConfig = this.store.getConfig().plasticRegions
    let plasticRegionRuntime = {
      effectiveMode: 'off',
      reason: plasticRegionConfig?.enabled === false ? 'disabled-by-policy' : 'base-unavailable',
      atlasFingerprint: null,
      atlasRevision: null,
      releaseManifestArtifactId: null,
      releaseFingerprint: null,
      ephemeralAtlas: false,
    }
    if (this.available && resolvedBaseForStatus) {
      try {
        const resolvedRegionRuntime = resolvePlasticRegionRuntime({
          store: this.store,
          base: resolvedBaseForStatus,
          requestedMode: plasticRegionConfig?.mode ?? 'off',
          policy: plasticRegionConfig,
        })
        plasticRegionRuntime = {
          effectiveMode: resolvedRegionRuntime.mode,
          reason: resolvedRegionRuntime.reason,
          atlasFingerprint: resolvedRegionRuntime.atlas?.fingerprint ?? null,
          atlasRevision: resolvedRegionRuntime.atlas?.revision ?? null,
          releaseManifestArtifactId: resolvedRegionRuntime.releaseManifestArtifactId ?? null,
          releaseFingerprint: resolvedRegionRuntime.releaseFingerprint ?? null,
          ephemeralAtlas: resolvedRegionRuntime.ephemeralAtlas,
        }
      } catch (error) {
        plasticRegionRuntime = {
          effectiveMode: 'off',
          reason: boundedString(error?.code, 100) || 'plastic-region-runtime-unavailable',
          atlasFingerprint: null,
          atlasRevision: null,
          releaseManifestArtifactId: null,
          releaseFingerprint: null,
          ephemeralAtlas: false,
        }
      }
    } else if (!this.available) {
      plasticRegionRuntime.reason = 'thought-engine-unavailable'
    }
    return {
      available: this.available,
      enabled: this.store.getConfig().thought.enabled,
      engine: { available: this.available, contract: 'thoughtEngine.runEpisode(request, context)' },
      activeEpisode: toPublicThoughtEpisode(active),
      counts: {
        total: episodes.length,
        running: episodes.filter((episode) => ['queued', 'running', 'pause_requested', 'cancel_requested'].includes(episode.status)).length,
        completed: episodes.filter((episode) => episode.status === 'completed').length,
        failed: episodes.filter((episode) => episode.status === 'failed').length,
      },
      activeSnapshotId: this.store.getActiveSnapshot()?.id ?? null,
      researchBaseAvailable,
      researchBase,
      researchBaseError,
      ...(researchBaseError ? { message: researchBaseError.message } : {}),
      crossCheck: {
        trustWeight: OLLAMA_CROSS_CHECK_TRUST,
        minimumConfidence: OLLAMA_CROSS_CHECK_MIN_CONFIDENCE,
        callsPerCheck: 2,
        teacherModes: ['off', 'ollama-cross-check'],
      },
      plasticRegions: {
        enabled: plasticRegionConfig?.enabled === true,
        configuredMode: plasticRegionConfig?.mode ?? 'off',
        effectiveMode: plasticRegionRuntime.effectiveMode,
        reason: plasticRegionRuntime.reason,
        atlasFingerprint: plasticRegionRuntime.atlasFingerprint,
        atlasRevision: plasticRegionRuntime.atlasRevision,
        ephemeralAtlas: plasticRegionRuntime.ephemeralAtlas,
        requestModes: [...PLASTIC_REGION_MODES],
        runtimeWired: true,
        engineAvailable: this.available,
      },
      limits: {
        maxListLimit: MAX_LIST_LIMIT,
        maxTickPageLimit: MAX_TICK_PAGE_LIMIT,
        maxCandidatesPerTick: MAX_PUBLIC_CANDIDATES_PER_TICK,
        maxGraphNodes: MAX_GRAPH_NODES,
        maxGraphEdges: MAX_GRAPH_EDGES,
        ...this.store.getConfig().thought,
      },
    }
  }

  list(options) {
    const all = this.store.listThoughtEpisodes()
      .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))
    const items = all.slice(options.offset, options.offset + options.limit).map((episode) => toPublicThoughtEpisode(episode))
    const nextOffset = options.offset + items.length
    return {
      items,
      pagination: {
        offset: options.offset,
        limit: options.limit,
        total: all.length,
        returned: items.length,
        hasMore: nextOffset < all.length,
        nextOffset: nextOffset < all.length ? nextOffset : null,
      },
    }
  }

  detail(id) {
    const episode = this.store.getThoughtEpisode(id)
    if (!episode) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    return { episode: toPublicThoughtEpisode(episode, { detail: true }) }
  }

  ticks(id, options) {
    const episode = this.store.getThoughtEpisode(id)
    if (!episode) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    const page = readEpisodeTickPage(this.store, episode, options)
    const selected = page.ticks.map((tick) => publicTraceValue({
      ...tick,
      candidates: (tick.candidates ?? []).slice(0, options.candidateLimit),
    }))
    const nextAfter = selected.at(-1)?.index ?? options.after
    return {
      episodeId: episode.id,
      ticks: selected,
      pagination: {
        after: options.after,
        limit: options.limit,
        returned: selected.length,
        hasMore: page.hasMore,
        nextAfter: page.hasMore ? nextAfter : null,
      },
    }
  }

  graph(options = {}) {
    const level = options.level ?? 'neuron'
    const limit = options.limit ?? 48
    if (options.episodeId) {
      const episode = this.store.getThoughtEpisode(options.episodeId)
      if (!episode) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
      const stored = readThoughtArtifact(this.store, episode.graphArtifactId, 'thought-graph') ?? { nodes: [], edges: [] }
      const edges = (stored.edges ?? []).filter((edge) => level === 'family' ? edge.kind === 'family' : edge.kind !== 'family')
      const nodeIds = new Set(edges.flatMap((edge) => [edge.from, edge.to]))
      const nodes = (stored.nodes ?? []).filter((node) => (
        level === 'family' ? node.kind === 'family' || nodeIds.has(node.id) : node.kind !== 'family' || nodeIds.has(node.id)
      ))
      return graphEnvelope(level, nodes, edges, limit, {
        episodeId: episode.id,
        available: Boolean(episode.graphArtifactId),
        eventGraph: stored.eventGraph ?? null,
        evaluation: stored.evaluation ?? null,
        branchScores: stored.branchScores ?? [],
        languageOutput: stored.languageOutput ?? null,
        familyGraphFingerprint: stored.familyGraphFingerprint ?? null,
      })
    }
    const base = this.resolveBase(null)
    if (level === 'family') {
      const derived = deriveFamilyGraphView(base)
      const nodes = familyGraphNodes(base)
      const edges = derived.transitions
      return graphEnvelope(level, nodes, edges, limit, {
        snapshotId: base.reference.snapshotId,
        headRevision: base.thoughtHead.revision,
        baseModelFingerprint: base.reference.baseModelFingerprint,
        derivedFamilyGraphFingerprint: derived.fingerprint,
      })
    }
    const primaryFamily = new Map()
    for (const membership of [...(base.familyRun.memberships ?? [])].sort((left, right) => left.rank - right.rank || right.weight - left.weight)) {
      if (!primaryFamily.has(membership.memberKey)) primaryFamily.set(membership.memberKey, membership.familyId)
    }
    const nodes = [
      ...(base.neuronSummaries ?? base.neurons).map((neuron) => ({
        id: neuron.memberKey, kind: 'neuron', label: displayText(neuron.text, 160).text || neuron.memberKey,
        activation: null, familyId: primaryFamily.get(neuron.memberKey) ?? null,
      })),
      ...base.thoughtHead.births.map((birth) => ({
        id: birth.id, kind: 'born-neuron', label: birth.id, activation: null,
        familyId: birth.familyMemberships?.[0]?.familyId ?? null,
      })),
    ].sort((left, right) => left.id.localeCompare(right.id))
    const edges = [
      ...(base.familyRun.neuronTransitions ?? []).map((transition) => ({
        from: transition.fromMemberKey, to: transition.toMemberKey, kind: 'habit',
        probability: probability(transition.probability), excitation: null, inhibition: null, trustMass: null,
      })),
      ...base.thoughtHead.neuronEdges.map((edge) => ({
        from: edge.fromMemberKey, to: edge.toMemberKey, kind: 'learned-neuron',
        probability: null, excitation: edge.excitation, inhibition: edge.inhibition, trustMass: edge.trustMass,
      })),
      ...base.thoughtHead.births.flatMap((birth) => [
        ...(birth.entryEvidence ?? []).map((entry) => ({
          from: entry.memberKey, to: birth.id, kind: 'birth-entry',
          probability: null, excitation: entry.weight, inhibition: 0, trustMass: birth.trustMass,
        })),
        ...(birth.outgoingEvidence ?? []).map((entry) => ({
          from: birth.id, to: entry.memberKey, kind: 'birth-outgoing',
          probability: null, excitation: entry.weight, inhibition: 0, trustMass: birth.trustMass,
        })),
      ]),
    ].sort((left, right) => left.from.localeCompare(right.from) || left.to.localeCompare(right.to) || left.kind.localeCompare(right.kind))
    return graphEnvelope(level, nodes, edges, limit, {
      snapshotId: base.reference.snapshotId,
      headRevision: base.thoughtHead.revision,
      baseModelFingerprint: base.reference.baseModelFingerprint,
    })
  }

  async start(body, { idempotencyKey } = {}) {
    const config = this.store.getConfig()
    if (!config.thought.enabled) throw thoughtError('Düşünce çalışma katmanı kapalı.', 'THOUGHT_DISABLED', 409)
    if (!this.available) throw thoughtError('Düşünce motoru bu servise bağlanmadı.', 'THOUGHT_ENGINE_UNAVAILABLE', 503)
    const request = normalizeStartRequest(body, config)
    const resolvedIdempotencyKey = normalizeId(idempotencyKey ?? request.idempotencyKey, 'Idempotency-Key')
    const semanticRequest = { ...request, idempotencyKey: undefined }
    const requestHash = digest('thought-request-v1', semanticRequest)
    if (resolvedIdempotencyKey) {
      const existing = this.store.findThoughtEpisodeByIdempotencyKey(resolvedIdempotencyKey)
      if (existing) {
        if (existing.requestHash !== requestHash) throw thoughtError('Idempotency anahtarı farklı bir istekle kullanılmış.', 'THOUGHT_IDEMPOTENCY_CONFLICT', 409)
        return { episode: toPublicThoughtEpisode(existing, { detail: true }), duplicate: true }
      }
    }
    if (this.active.size > 0) throw thoughtError('Aynı anda yalnızca bir düşünce bölümü çalışabilir.', 'THOUGHT_BUSY', 409)
    const base = this.resolveBase(request.snapshotId)
    const regionRuntime = resolvePlasticRegionRuntime({
      store: this.store,
      base,
      requestedMode: request.requestedRegionMode,
      policy: config.plasticRegions,
    })
    const residency = await this.ensureResidency(base)
    const id = randomUUID()
    const mode = base.reference.commitEligible ? 'active' : 'research'
    const effectiveLearningMode = mode === 'active' ? request.learningMode : 'off'
    const humanEvidence = request.expectedOutcome ? {
      kind: 'human',
      confidence: 1,
      trustWeight: 1,
      sourceGroupId: request.sourceGroupId ?? `human-outcome:${digest(request.expectedOutcome, request.inputText).slice(0, 32)}`,
      evidenceRole: 'external-structural-target',
      evaluationEligible: false,
      datasetSplit: null,
    } : null
    const episodeSourceGroupId = humanEvidence?.sourceGroupId ?? request.sourceGroupId ?? (
      regionRuntime.mode === 'off' ? null : defaultAdaptiveRegionSourceGroupId(request.inputText)
    )
    const created = this.store.createThoughtEpisode({
      id,
      idempotencyKey: resolvedIdempotencyKey,
      requestHash,
      inputText: request.inputText,
      inputDigest: digest('thought-input-v1', request.inputText),
      seed: request.seed,
      maxTicks: request.maxTicks,
      strategy: request.strategy,
      learningMode: effectiveLearningMode,
      requestedLearningMode: request.learningMode,
      teacherMode: request.teacherMode,
      requestedRegionMode: request.requestedRegionMode,
      regionMode: regionRuntime.mode,
      regionModeReason: regionRuntime.reason,
      regionAtlasArtifactId: regionRuntime.atlasArtifactId,
      regionAtlasFingerprint: regionRuntime.atlas?.fingerprint ?? null,
      regionAtlasRevision: regionRuntime.atlas?.revision ?? null,
      regionGeometryFingerprint: regionRuntime.atlas?.geometryFingerprint ?? null,
      regionRepresentationSpaceId: regionRuntime.atlas?.representationSpaceId ?? null,
      regionReleaseManifestArtifactId: regionRuntime.releaseManifestArtifactId ?? null,
      regionReleaseFingerprint: regionRuntime.releaseFingerprint ?? null,
      regionEphemeralAtlas: regionRuntime.ephemeralAtlas,
      regionCompatibilityBlend: regionRuntime.adaptiveRegion?.compatibilityBlend ?? 0,
      regionMinimumTrace: regionRuntime.adaptiveRegion?.minimumTrace ?? 0,
      regionMaximumActiveGates: regionRuntime.adaptiveRegion?.maximumActiveGates ?? 0,
      crossCheckRequested: request.crossCheckRequested,
      expectedOutcome: request.expectedOutcome,
      teacherEvidence: humanEvidence,
      sourceGroupId: episodeSourceGroupId,
      candidateMemberKeys: request.candidateMemberKeys,
      mode,
      commitEligible: base.reference.commitEligible,
      commitStatus: 'not-eligible',
      base: {
        ...base.reference,
        regionAtlasArtifactId: regionRuntime.atlasArtifactId,
        regionAtlasFingerprint: regionRuntime.atlas?.fingerprint ?? null,
        regionAtlasRevision: regionRuntime.atlas?.revision ?? null,
        regionGeometryFingerprint: regionRuntime.atlas?.geometryFingerprint ?? null,
        regionReleaseManifestArtifactId: regionRuntime.releaseManifestArtifactId ?? null,
        regionReleaseFingerprint: regionRuntime.releaseFingerprint ?? null,
        ...(residency?.manifest?.fingerprint
          ? { residencyManifestFingerprint: residency.manifest.fingerprint }
          : {}),
      },
      baseDiagnostics: {
        ...base.diagnostics,
        plasticRegions: {
          mode: regionRuntime.mode,
          reason: regionRuntime.reason,
          atlasFingerprint: regionRuntime.atlas?.fingerprint ?? null,
          atlasRevision: regionRuntime.atlas?.revision ?? null,
          ephemeralAtlas: regionRuntime.ephemeralAtlas,
        },
      },
      stopReason: null,
      crossCheck: request.crossCheckRequested
        ? { status: 'pending', trustWeight: OLLAMA_CROSS_CHECK_TRUST }
        : { status: 'not-requested', trustWeight: OLLAMA_CROSS_CHECK_TRUST },
    })
    if (!created.duplicate) this.launch(created.episode.id)
    return { episode: toPublicThoughtEpisode(created.episode, { detail: true }), duplicate: created.duplicate }
  }

  launch(id) {
    if (this.active.has(id)) return this.active.get(id).promise
    const controller = new AbortController()
    const promise = this.execute(id, controller).finally(() => this.active.delete(id))
    this.active.set(id, { controller, promise })
    return promise
  }

  async execute(id, controller) {
    let episode = this.store.getThoughtEpisode(id)
    if (!episode || !['queued', 'paused'].includes(episode.status)) return
    episode = this.store.updateThoughtEpisode(id, {
      status: 'running',
      startedAt: episode.startedAt ?? new Date().toISOString(),
      finishedAt: null,
      error: null,
    }, { expectedStatuses: [episode.status] })
    await this.emit({ type: 'thought-episode-started', data: { episodeId: id, mode: episode.mode } })
    // Only the uncheckpointed window is retained in RAM. Full historical
    // traces live in immutable 64-tick artifacts and are read only by the
    // paginated audit/replay paths.
    const streamedTicks = []
    const traceChunkSize = this.store.getConfig().thought.traceChunkSize
    let traceCursor = {
      count: nonNegativeInteger(episode.tickCount),
      lastIndex: null,
      lastTickDigest: boundedString(episode.lastTickDigest, 128) || null,
    }
    try {
      const base = this.resolveBase(episode.base.snapshotId)
      if (
        base.reference.artifactManifestHash !== episode.base.artifactManifestHash ||
        base.reference.baseModelFingerprint !== episode.base.baseModelFingerprint ||
        (base.reference.representationProfile ?? null) !== (episode.base.representationProfile ?? null) ||
        base.thoughtHead.fingerprint !== episode.base.thoughtHeadFingerprint
      ) throw thoughtError('Düşünce tabanı veya başlık revizyonu değişti.', 'STALE_THOUGHT_BASE', 409)
      assertEpisodeResidencyManifest(episode, await this.ensureResidency(base))
      const regionRuntime = assertEpisodeRegionRuntime(episode, resolvePlasticRegionRuntime({
        store: this.store,
        base,
        requestedMode: episode.regionMode ?? 'off',
        policy: this.store.getConfig().plasticRegions,
      }))
      let crossCheck = episode.crossCheck
      let expectedOutcome = episode.expectedOutcome
      let teacherEvidence = episode.teacherEvidence
      if (!expectedOutcome && episode.crossCheckRequested && (crossCheck?.status ?? 'pending') === 'pending') {
        try {
          crossCheck = await runOllamaCrossCheck({
            store: this.store,
            ollama: this.ollama,
            episodeId: id,
            inputText: episode.inputText,
            signal: controller.signal,
          })
          if (crossCheck.gatePassed) {
            expectedOutcome = crossCheck.graph
            teacherEvidence = {
              kind: 'ollama-cross-checked',
              confidence: crossCheck.confidence,
              trustWeight: OLLAMA_CROSS_CHECK_TRUST,
              sourceGroupId: crossCheck.provenance.sourceGroupId,
              evidenceRole: 'auxiliary-critic',
              evaluationEligible: false,
              datasetSplit: null,
            }
          }
        } catch (error) {
          if (controller.signal.aborted || ['THOUGHT_PAUSED', 'THOUGHT_CANCELLED'].includes(error?.code)) throw error
          crossCheck = {
            status: 'unavailable',
            gatePassed: false,
            agrees: false,
            confidence: 0,
            minimumConfidence: OLLAMA_CROSS_CHECK_MIN_CONFIDENCE,
            trustWeight: OLLAMA_CROSS_CHECK_TRUST,
            graph: null,
            passes: [],
            error: { code: boundedString(error.code, 100), message: boundedString(error.message, 500) },
            provenance: {
              kind: 'ollama-cross-check', evidenceRole: 'auxiliary-critic', evaluationEligible: false,
              datasetSplit: null, sourceGroupId: null,
            },
          }
        }
        episode = this.store.updateThoughtEpisode(
          id,
          { crossCheck, expectedOutcome, teacherEvidence },
          { expectedStatuses: ['running'] },
        )
      }
      traceCursor = episodeTraceCursor(this.store, episode)
      const checkpointArtifact = readThoughtArtifact(this.store, episode.checkpointArtifactId, 'thought-episode-checkpoint')
      const checkpoint = checkpointArtifact?.state ?? checkpointArtifact ?? null
      const persistCheckpoint = ({ state, ticks: checkpointTicks } = {}) => {
        if (!state) throw thoughtError('Checkpoint tam episode durumunu taşımalı.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
        const regionBindingFingerprint = verifiedEngineRegionBindingFingerprint(
          state.adaptiveRegion,
          episode,
          regionRuntime,
        )
        const candidates = [
          ...(Array.isArray(checkpointTicks) ? checkpointTicks : []),
          ...streamedTicks,
        ]
        const rawByIndex = new Map()
        candidates.forEach((tick, index) => {
          const fallbackIndex = traceCursor.lastIndex === null
            ? traceCursor.count + index
            : traceCursor.lastIndex + index + 1
          const tickIndex = nonNegativeInteger(tick?.index ?? tick?.tick ?? tick?.step, fallbackIndex)
          if ((traceCursor.lastIndex === null || tickIndex > traceCursor.lastIndex) && !rawByIndex.has(tickIndex)) {
            rawByIndex.set(tickIndex, tick)
          }
        })
        const newTicks = normalizeTicks(
          [...rawByIndex.values()],
          Math.max(0, episode.maxTicks - traceCursor.count),
          traceCursor.lastTickDigest,
        )
        if (newTicks.length > traceChunkSize) {
          throw thoughtError('Motor 64 tick sınırında güvenli checkpoint üretmedi.', 'THOUGHT_CHECKPOINT_REQUIRED', 502)
        }
        const nextCount = traceCursor.count + newTicks.length
        const nextLastTick = newTicks.at(-1) ?? null
        const nextLastIndex = nextLastTick?.index ?? traceCursor.lastIndex
        const nextLastDigest = nextLastTick?.tickDigest ?? traceCursor.lastTickDigest
        const entries = []
        for (let offset = 0; offset < newTicks.length; offset += traceChunkSize) {
          const ticks = newTicks.slice(offset, offset + traceChunkSize)
          const value = {
            format: 'thought-trace-chunk', version: 1, episodeId: id,
            firstIndex: ticks[0]?.index ?? null, lastIndex: ticks.at(-1)?.index ?? null, ticks,
          }
          entries.push({
            field: 'traceArtifactIds',
            artifact: artifactSpec('thought-trace-chunk', value, {
              maximumBytes: this.store.getConfig().thought.maximumEpisodeBytes,
              contentParts: [id, value.firstIndex, value.lastIndex],
              metadata: { episodeId: id, tickCount: ticks.length },
            }),
          })
        }
        entries.push({
          field: 'checkpointArtifactId',
          artifact: artifactSpec('thought-episode-checkpoint', {
            format: 'thought-episode-checkpoint', version: 1, episodeId: id,
            tickCount: nextCount, state: strictBoundedValue(state),
          }, {
            maximumBytes: this.store.getConfig().thought.maximumEpisodeBytes,
            contentParts: [id, nextCount, nextLastDigest],
            metadata: { episodeId: id, tickCount: nextCount },
          }),
        })
        const attached = this.store.attachThoughtEpisodeArtifacts(id, entries, {
          tickCount: nextCount,
          lastTickDigest: nextLastDigest,
          regionBindingFingerprint,
        }, { expectedStatuses: ['running', 'pause_requested'] })
        traceCursor = { count: nextCount, lastIndex: nextLastIndex, lastTickDigest: nextLastDigest }
        streamedTicks.length = 0
        return attached
      }
      const request = {
        format: 'thought-engine-request',
        version: 1,
        episodeId: id,
        inputText: episode.inputText,
        seed: episode.seed,
        maxTicks: episode.maxTicks,
        strategy: episode.strategy === 'greedy' ? 'deterministic' : 'seeded-stochastic',
        learningMode: episode.learningMode,
        teacherMode: episode.teacherMode ?? (episode.crossCheckRequested ? 'ollama-cross-check' : 'off'),
        candidateMemberKeys: [...(episode.candidateMemberKeys ?? [])],
        expectedOutcome: expectedOutcome ?? null,
        outcomeEvidence: teacherEvidence ?? null,
        teacherEvidence: teacherEvidence ?? null,
        sourceGroupId: teacherEvidence?.sourceGroupId ?? episode.sourceGroupId ?? null,
        base: publicBase(episode.base),
        resume: checkpoint,
        resumeAudit: traceCursor.count ? {
          tickCount: traceCursor.count,
          lastTickDigest: traceCursor.lastTickDigest,
        } : null,
        ...(regionRuntime.adaptiveRegion ? { adaptiveRegion: regionRuntime.adaptiveRegion } : {}),
      }
      const context = {
        signal: controller.signal,
        familyRun: base.familyRun,
        languageState: base.languageState,
        thoughtHead: base.thoughtHead,
        conceptMacroHead: base.conceptMacroHead,
        ...(base.representationProfile ? { representationProfile: base.representationProfile } : {}),
        ...(regionRuntime.atlas ? { adaptiveRegionAtlas: regionRuntime.atlas } : {}),
        baseDiagnostics: {
          ...base.diagnostics,
          plasticRegions: {
            mode: regionRuntime.mode,
            reason: regionRuntime.reason,
            atlasFingerprint: regionRuntime.atlas?.fingerprint ?? null,
            atlasRevision: regionRuntime.atlas?.revision ?? null,
            ephemeralAtlas: regionRuntime.ephemeralAtlas,
          },
        },
        loadArtifact: (artifactId) => {
          if (!episode.base.artifactIds.includes(artifactId)) throw thoughtError('Motor taban dışı artifact okuyamaz.', 'THOUGHT_ARTIFACT_OUTSIDE_BASE', 403)
          return this.store.getArtifact(artifactId)
        },
        shouldPause: () => this.store.getThoughtEpisode(id)?.status === 'pause_requested',
        shouldCancel: () => this.store.getThoughtEpisode(id)?.status === 'cancel_requested',
        onTick: (tick) => {
          if (traceCursor.count + streamedTicks.length >= episode.maxTicks) {
            throw thoughtError('Motor tick sınırını aştı.', 'THOUGHT_ENGINE_LIMIT_EXCEEDED', 502)
          }
          if (streamedTicks.length >= traceChunkSize) {
            throw thoughtError('Motor 64 tick sınırında güvenli checkpoint üretmedi.', 'THOUGHT_CHECKPOINT_REQUIRED', 502)
          }
          streamedTicks.push(tick)
        },
        onCheckpoint: (checkpointInput) => persistCheckpoint(checkpointInput),
      }
      const result = await this.runEngineEpisode(request, context, base, episode)
      if (!isRecord(result)) throw thoughtError('Düşünce motoru sonuç nesnesi döndürmedi.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
      const current = this.store.getThoughtEpisode(id)
      if (current?.status === 'pause_requested' || result.episode?.status === 'paused') {
        persistCheckpoint({ state: result.checkpoint ?? result.episode, ticks: result.ticks })
        throw thoughtError('Düşünce bölümü duraklatıldı.', 'THOUGHT_PAUSED', 409)
      }
      if (current?.status === 'cancel_requested') throw thoughtError('Düşünce bölümü iptal edildi.', 'THOUGHT_CANCELLED', 409)
      const combinedRaw = [...streamedTicks, ...(Array.isArray(result.ticks) ? result.ticks : [])]
      const rawByIndex = new Map()
      combinedRaw.forEach((tick, index) => {
        const fallbackIndex = traceCursor.lastIndex === null
          ? traceCursor.count + index
          : traceCursor.lastIndex + index + 1
        const tickIndex = nonNegativeInteger(tick?.index ?? tick?.tick ?? tick?.step, fallbackIndex)
        if ((traceCursor.lastIndex === null || tickIndex > traceCursor.lastIndex) && !rawByIndex.has(tickIndex)) {
          rawByIndex.set(tickIndex, tick)
        }
      })
      const newTicks = normalizeTicks(
        [...rawByIndex.values()],
        Math.max(0, episode.maxTicks - traceCursor.count),
        traceCursor.lastTickDigest,
      )
      if (newTicks.length > traceChunkSize) {
        throw thoughtError('Motor sonuçtan önce 64 tick sınırında güvenli checkpoint üretmedi.', 'THOUGHT_CHECKPOINT_REQUIRED', 502)
      }
      const totalTickCount = traceCursor.count + newTicks.length
      const finalLastTick = newTicks.at(-1) ?? null
      const finalLastDigest = finalLastTick?.tickDigest ?? traceCursor.lastTickDigest
      if (totalTickCount > episode.maxTicks || totalTickCount > this.store.getConfig().thought.maximumStoredTicks) {
        throw thoughtError('Toplam düşünce izi saklama sınırını aşıyor.', 'THOUGHT_ENGINE_LIMIT_EXCEEDED', 502)
      }
      const graph = sanitizeGraph(result.graph)
      if (graph.eventGraph) {
        try {
          const intent = eventGraphToLanguageIntent(graph.eventGraph)
          graph.languageOutput = strictBoundedValue(generateTurkishSentence(
            intent,
            base.languageOutputScaffold,
            base.languageOutputHead ?? undefined,
          ))
        } catch (error) {
          graph.languageOutput = {
            text: '',
            stoppedBy: 'unmatched',
            error: {
              code: boundedString(error?.code ?? 'LANGUAGE_OUTPUT_UNAVAILABLE', 100),
              message: boundedString(error?.message ?? String(error), 500),
            },
          }
        }
      }
      const summary = engineSummary(result.episode)
      const trustedTarget = Boolean(expectedOutcome && teacherEvidence)
      let headDelta = null
      let deltaSuppressedReason = null
      if (result.headDelta && episode.learningMode === 'proposal' && episode.commitEligible && trustedTarget) {
        headDelta = normalizeHeadDelta(result.headDelta, base, episode, teacherEvidence, result.episode?.runKey)
      } else if (result.headDelta) {
        deltaSuppressedReason = !episode.commitEligible
          ? 'research-trace-only'
          : episode.learningMode !== 'proposal'
            ? 'learning-disabled'
            : 'external-outcome-not-trusted'
      }
      const checkpointValue = result.checkpoint ?? result.episode ?? null
      const regionBindingFingerprint = verifiedEngineRegionBindingFingerprint(
        result.episode?.adaptiveRegion ?? checkpointValue?.adaptiveRegion,
        episode,
        regionRuntime,
      )
      const entries = []
      for (let offset = 0; offset < newTicks.length; offset += traceChunkSize) {
        const ticks = newTicks.slice(offset, offset + traceChunkSize)
        const value = {
          format: 'thought-trace-chunk', version: 1, episodeId: id,
          firstIndex: ticks[0]?.index ?? null, lastIndex: ticks.at(-1)?.index ?? null, ticks,
        }
        entries.push({
          field: 'traceArtifactIds',
          artifact: artifactSpec('thought-trace-chunk', value, {
            maximumBytes: this.store.getConfig().thought.maximumEpisodeBytes,
            contentParts: [id, value.firstIndex, value.lastIndex],
            metadata: { episodeId: id, tickCount: ticks.length },
          }),
        })
      }
      if (graph.nodes.length || graph.edges.length || graph.eventGraph || graph.languageOutput) entries.push({
        field: 'graphArtifactId',
        artifact: artifactSpec('thought-graph', { format: 'thought-graph', version: 1, episodeId: id, ...graph }, {
          maximumBytes: 512 * 1024, contentParts: [id], metadata: { episodeId: id, nodes: graph.nodes.length, edges: graph.edges.length },
        }),
      })
      if (checkpointValue) entries.push({
        field: 'checkpointArtifactId',
        artifact: artifactSpec('thought-episode-checkpoint', {
          format: 'thought-episode-checkpoint', version: 1, episodeId: id,
          tickCount: totalTickCount, state: strictBoundedValue(checkpointValue),
        }, {
          maximumBytes: this.store.getConfig().thought.maximumEpisodeBytes,
          contentParts: [id, totalTickCount, finalLastDigest], metadata: { episodeId: id, tickCount: totalTickCount },
        }),
      })
      if (headDelta) entries.push({
        field: 'headProposalArtifactId',
        artifact: artifactSpec('thought-head-proposal', headDelta, {
          maximumBytes: this.store.getConfig().thought.maximumEpisodeBytes,
          contentParts: [id, base.thoughtHead.fingerprint, base.thoughtHead.revision],
          metadata: { episodeId: id, baseRevision: base.thoughtHead.revision },
        }),
      })
      const attached = this.store.attachThoughtEpisodeArtifacts(id, entries, {
        status: 'completed',
        finishedAt: new Date().toISOString(),
        tickCount: totalTickCount,
        lastTickDigest: finalLastDigest,
        stopReason: summary?.stopReason ?? finalLastTick?.stopReason ?? (totalTickCount >= episode.maxTicks ? 'maximum-ticks' : null),
        engineSummary: summary,
        baseDiagnostics: base.diagnostics,
        teacherEvidence,
        expectedOutcome,
        crossCheck,
        regionBindingFingerprint,
        commitStatus: headDelta ? 'pending-review' : 'not-eligible',
        deltaSuppressedReason,
        error: null,
      }, { verifyCompletionBase: true, expectedStatuses: ['running'] })
      await this.emit({
        type: 'thought-episode-completed',
        data: { episodeId: id, tickCount: attached.episode.tickCount, commitStatus: attached.episode.commitStatus },
      })
    } catch (error) {
      const current = this.store.getThoughtEpisode(id)
      const pause = error.code === 'THOUGHT_PAUSED' || error.code === 'SERVICE_STOP' ||
        error.code === 'NEURON_RESIDENCY_RESOURCE_WAIT' || current?.status === 'pause_requested'
      const cancel = error.code === 'THOUGHT_CANCELLED' || current?.status === 'cancel_requested'
      try {
        const pendingByIndex = new Map()
        ;(pause ? [] : streamedTicks).forEach((tick, index) => {
          const fallbackIndex = traceCursor.lastIndex === null
            ? traceCursor.count + index
            : traceCursor.lastIndex + index + 1
          const tickIndex = nonNegativeInteger(tick?.index ?? tick?.tick ?? tick?.step, fallbackIndex)
          if ((traceCursor.lastIndex === null || tickIndex > traceCursor.lastIndex) && !pendingByIndex.has(tickIndex)) {
            pendingByIndex.set(tickIndex, tick)
          }
        })
        const newTicks = normalizeTicks(
          [...pendingByIndex.values()],
          Math.max(0, (current?.maxTicks ?? 0) - traceCursor.count),
          traceCursor.lastTickDigest,
        )
        const finalTickCount = traceCursor.count + newTicks.length
        const finalLastDigest = newTicks.at(-1)?.tickDigest ?? traceCursor.lastTickDigest
        const entries = []
        if (newTicks.length) {
          const value = {
            format: 'thought-trace-chunk', version: 1, episodeId: id,
            firstIndex: newTicks[0].index, lastIndex: newTicks.at(-1).index, ticks: newTicks,
          }
          entries.push({
            field: 'traceArtifactIds',
            artifact: artifactSpec('thought-trace-chunk', value, {
              maximumBytes: this.store.getConfig().thought.maximumEpisodeBytes,
              contentParts: [id, value.firstIndex, value.lastIndex],
              metadata: { episodeId: id, tickCount: newTicks.length },
            }),
          })
        }
        const status = cancel ? 'cancelled' : pause ? 'paused' : 'failed'
        this.store.attachThoughtEpisodeArtifacts(id, entries, {
          status,
          tickCount: finalTickCount,
          lastTickDigest: finalLastDigest,
          finishedAt: TERMINAL_EPISODE_STATES.has(status) ? new Date().toISOString() : null,
          error: pause || cancel ? null : {
            code: boundedString(error.code ?? 'THOUGHT_ENGINE_FAILED', 100),
            message: boundedString(error.message ?? String(error), 500),
          },
        }, {
          expectedStatuses: cancel
            ? ['cancel_requested']
            : pause
              ? ['pause_requested', 'running']
              : ['running'],
        })
        await this.emit({
          type: cancel ? 'thought-episode-cancelled' : pause ? 'thought-episode-paused' : 'thought-episode-failed',
          level: cancel || pause ? 'warning' : 'error',
          data: { episodeId: id, code: error.code ?? 'THOUGHT_ENGINE_FAILED' },
        })
      } catch (persistenceError) {
        const latest = this.store.getThoughtEpisode(id)
        if (persistenceError?.code === 'THOUGHT_STATE_CONFLICT') {
          try {
            if (latest?.status === 'cancel_requested') {
              this.store.attachThoughtEpisodeArtifacts(id, [], {
                status: 'cancelled', finishedAt: new Date().toISOString(), error: null,
              }, { expectedStatuses: ['cancel_requested'] })
            } else if (latest?.status === 'pause_requested') {
              this.store.attachThoughtEpisodeArtifacts(id, [], {
                status: 'paused', finishedAt: null, error: null,
              }, { expectedStatuses: ['pause_requested'] })
            }
          } catch {
            // A second controller/worker transition won. Its fresh terminal or
            // requested state remains authoritative; never overwrite it here.
          }
        } else if (latest && !TERMINAL_EPISODE_STATES.has(latest.status)) {
          try {
            this.store.updateThoughtEpisode(id, {
              status: 'failed', finishedAt: new Date().toISOString(),
              error: { code: boundedString(persistenceError.code ?? 'THOUGHT_PERSISTENCE_FAILED', 100), message: boundedString(persistenceError.message, 500) },
            }, { expectedStatuses: [latest.status] })
          } catch {
            // A concurrent control/completion transition is newer.
          }
        }
      }
    }
  }

  pause(id) {
    const transition = this.store.requestThoughtEpisodeControl(id, 'pause')
    if (!transition) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    if (transition.changed && transition.status === 'pause_requested') {
      this.active.get(id)?.controller.abort(thoughtError('Düşünce bölümü duraklatıldı.', 'THOUGHT_PAUSED', 409))
    }
    return toPublicThoughtEpisode(transition.episode)
  }

  resume(id) {
    const episode = this.store.getThoughtEpisode(id)
    if (!episode) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    if (!this.available) throw thoughtError('Düşünce motoru bu servise bağlanmadı.', 'THOUGHT_ENGINE_UNAVAILABLE', 503)
    if (episode.status === 'pause_requested' || this.active.has(id)) {
      throw thoughtError('Duraklatma kontrol noktasına henüz ulaşmadı.', 'THOUGHT_BUSY', 409)
    }
    if (episode.status !== 'paused') return toPublicThoughtEpisode(episode)
    if (this.active.size > 0) throw thoughtError('Başka bir düşünce bölümü çalışıyor.', 'THOUGHT_BUSY', 409)
    const transition = this.store.requestThoughtEpisodeControl(id, 'resume')
    if (!transition) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    if (transition.changed) this.launch(id)
    return toPublicThoughtEpisode(transition.episode)
  }

  cancel(id) {
    const transition = this.store.requestThoughtEpisodeControl(id, 'cancel')
    if (!transition) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    if (transition.changed && transition.status === 'cancel_requested') {
      this.active.get(id)?.controller.abort(thoughtError('Düşünce bölümü iptal edildi.', 'THOUGHT_CANCELLED', 409))
    }
    return toPublicThoughtEpisode(transition.episode)
  }

  feedback(id, body) {
    if (!isRecord(body) || !['approve', 'reject'].includes(body.decision)) {
      throw thoughtError('feedback decision approve veya reject olmalı.', 'INVALID_THOUGHT_FEEDBACK', 400)
    }
    const reason = typeof body.reason === 'string' ? body.reason.slice(0, 1_000) : ''
    const episode = this.store.getThoughtEpisode(id)
    if (!episode) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    if (body.decision === 'reject') {
      const reviewed = this.store.reviewThoughtEpisode(id, { decision: 'reject', reason })
      void this.emit({ type: 'thought-head-rejected', level: 'warning', data: { episodeId: id } })
      return { episode: toPublicThoughtEpisode(reviewed.episode, { detail: true }), duplicate: reviewed.duplicate, snapshot: null }
    }
    const base = this.resolveBase(episode.base?.snapshotId)
    if (
      base.reference.baseModelFingerprint !== episode.base?.baseModelFingerprint ||
      base.thoughtHead.fingerprint !== episode.base?.thoughtHeadFingerprint ||
      base.thoughtHead.revision !== episode.base?.thoughtHeadRevision
    ) throw thoughtError('Düşüncenin model veya başlık tabanı artık güncel değil.', 'STALE_THOUGHT_BASE', 409)
    const proposal = readThoughtArtifact(this.store, episode.headProposalArtifactId, 'thought-head-proposal')
    if (!proposal) throw thoughtError('Düşünce başlığı öneri eseri bulunamadı.', 'THOUGHT_HEAD_PROPOSAL_MISSING', 409)
    const head = applyHeadDelta(base.thoughtHead, proposal, base.layout, episode)
    const nextRevision = head.revision
    const reviewed = this.store.reviewThoughtEpisode(id, {
      decision: 'approve',
      reason,
      expectedBaseSnapshotId: episode.base?.snapshotId,
      headRevision: nextRevision,
      headFingerprint: head.fingerprint,
      headArtifact: artifactSpec('thought-runtime-head', head, {
        maximumBytes: this.store.getConfig().thought.maximumHeadBytes,
        contentParts: [head.fingerprint],
        metadata: { episodeId: id, revision: nextRevision, fingerprint: head.fingerprint },
      }),
      snapshotLabel: `Düşünce başlığı adayı · ${new Date().toLocaleString('tr-TR')}`,
      metrics: this.store.getSnapshot(episode.base?.snapshotId)?.metrics ?? null,
      // Teacher or a single episode is never validation/test evidence. The
      // normal evaluation pipeline must verify this candidate before activation.
      evaluation: {
        passed: false,
        runtimeVerified: false,
        reason: 'Düşünce başlığı kullanıcı tarafından onaylandı; bağımsız değerlendirme kapısı henüz çalışmadı.',
        thoughtEpisodeId: id,
        teacherEvidenceEligibleForEvaluation: false,
      },
    })
    void this.emit({ type: 'thought-head-candidate-created', data: { episodeId: id, snapshotId: reviewed.snapshot?.id } })
    return {
      episode: toPublicThoughtEpisode(reviewed.episode, { detail: true }),
      duplicate: reviewed.duplicate,
      snapshot: reviewed.snapshot ? { id: reviewed.snapshot.id, status: reviewed.snapshot.status, parentId: reviewed.snapshot.parentId } : null,
    }
  }

  async replay(id, { signal } = {}) {
    if (!this.available) throw thoughtError('Düşünce motoru bu servise bağlanmadı.', 'THOUGHT_ENGINE_UNAVAILABLE', 503)
    if (this.active.has(id)) throw thoughtError('Çalışan bölüm aynı anda replay edilemez.', 'THOUGHT_BUSY', 409)
    const episode = this.store.getThoughtEpisode(id)
    if (!episode) throw thoughtError('Düşünce bölümü bulunamadı.', 'THOUGHT_EPISODE_NOT_FOUND', 404)
    if (!['completed', 'paused', 'cancelled', 'failed'].includes(episode.status)) {
      throw thoughtError('Düşünce bölümü replay için kararlı durumda değil.', 'THOUGHT_STATE_CONFLICT', 409)
    }
    let base = this.resolveBase(episode.base.snapshotId)
    const residencyReplay = episode.baseDiagnostics?.residency?.mode === 'ssd-gated-working-set-v1'
    const residencyManifestBound = Boolean(boundedString(episode.base?.residencyManifestFingerprint, 256))
    const useResidencyReplay = residencyReplay && residencyManifestBound
    if (!useResidencyReplay && !base.neurons.length) {
      // Episodes created before the residency rollout must replay against the
      // exact legacy full catalog; silently switching their candidate universe
      // would invalidate the deterministic hash chain.
      this.baseCache.delete(episode.base.snapshotId)
      this.baseCacheAccess.delete(episode.base.snapshotId)
      base = this.resolveBase(episode.base.snapshotId)
    }
    if (base.reference.baseModelFingerprint !== episode.base.baseModelFingerprint) {
      throw thoughtError('Replay taban modeli artık doğrulanamıyor.', 'STALE_THOUGHT_BASE', 409)
    }
    if ((base.reference.representationProfile ?? null) !== (episode.base.representationProfile ?? null)) {
      throw thoughtError('Replay temsil profili artık doğrulanamıyor.', 'STALE_THOUGHT_BASE', 409)
    }
    if (residencyManifestBound) {
      assertEpisodeResidencyManifest(episode, await this.ensureResidency(base))
    }
    const regionRuntime = assertEpisodeRegionRuntime(episode, resolvePlasticRegionRuntime({
      store: this.store,
      base,
      requestedMode: episode.regionMode ?? 'off',
      policy: pinnedRegionReplayPolicy(episode, this.store.getConfig().plasticRegions),
      verificationOnly: true,
    }))
    const replayCheckpointArtifact = readThoughtArtifact(
      this.store,
      episode.checkpointArtifactId,
      'thought-episode-checkpoint',
    )
    const replayCheckpointState = replayCheckpointArtifact?.state ?? replayCheckpointArtifact ?? null
    const replayBindingFingerprint = verifiedEngineRegionBindingFingerprint(
      replayCheckpointState?.adaptiveRegion,
      episode,
      regionRuntime,
    )
    const expected = readEpisodeTickManifest(this.store, episode)
    if (!expected.length) {
      throw thoughtError('Bu bölüm kalıcı bir tick izi taşımadığı için replay edilemez.', 'THOUGHT_REPLAY_UNAVAILABLE', 409)
    }
    const actual = []
    const actualIndexes = new Set()
    let actualLastIndex = null
    let actualLastDigest = null
    const acceptActualTick = (rawTick) => {
      const tickIndex = nonNegativeInteger(rawTick?.index ?? rawTick?.tick ?? rawTick?.step, actual.length)
      if (actualIndexes.has(tickIndex)) return
      if (actualLastIndex !== null && tickIndex <= actualLastIndex) {
        throw thoughtError('Replay motoru tickleri artan sırada üretmedi.', 'THOUGHT_ENGINE_OUTPUT_INVALID', 502)
      }
      if (actual.length >= episode.maxTicks) {
        throw thoughtError('Replay motoru kayıtlı tick bütçesini aştı.', 'THOUGHT_ENGINE_LIMIT_EXCEEDED', 502)
      }
      const indexedTick = isRecord(rawTick) ? { ...rawTick, index: tickIndex } : { index: tickIndex }
      const normalized = normalizeTicks([indexedTick], 1, actualLastDigest)[0]
      actual.push({ index: normalized.index, tickDigest: normalized.tickDigest })
      actualIndexes.add(tickIndex)
      actualLastIndex = normalized.index
      actualLastDigest = normalized.tickDigest
    }
    const request = {
      format: 'thought-engine-request', version: 1, episodeId: episode.id,
      // The original tick budget is part of the pure engine runKey. A partial
      // replay stops at expectedTickCount, but must not rewrite that budget.
      inputText: episode.inputText, seed: episode.seed, maxTicks: episode.maxTicks,
      strategy: episode.strategy === 'greedy' ? 'deterministic' : 'seeded-stochastic', learningMode: 'off',
      teacherMode: episode.teacherMode ?? (episode.crossCheckRequested ? 'ollama-cross-check' : 'off'), candidateMemberKeys: episode.candidateMemberKeys ?? [],
      expectedOutcome: episode.expectedOutcome ?? null, teacherEvidence: episode.teacherEvidence ?? null,
      outcomeEvidence: episode.teacherEvidence ?? null,
      sourceGroupId: episode.teacherEvidence?.sourceGroupId ?? episode.sourceGroupId ?? null,
      base: publicBase(episode.base), replay: {
        verifyOnly: true,
        expectedTickCount: expected.length,
        ...(regionRuntime.adaptiveRegion ? {
          adaptiveRegion: {
            mode: regionRuntime.mode,
            atlasFingerprint: regionRuntime.atlas.fingerprint,
            bindingFingerprint: replayBindingFingerprint,
          },
        } : {}),
      }, resume: null,
      ...(regionRuntime.adaptiveRegion ? { adaptiveRegion: regionRuntime.adaptiveRegion } : {}),
    }
    const replayContext = {
      signal,
      familyRun: base.familyRun,
      languageState: base.languageState,
      thoughtHead: base.thoughtHead,
      conceptMacroHead: base.conceptMacroHead,
      ...(base.representationProfile ? { representationProfile: base.representationProfile } : {}),
      ...(regionRuntime.atlas ? { adaptiveRegionAtlas: regionRuntime.atlas } : {}),
      baseDiagnostics: {
        ...base.diagnostics,
        plasticRegions: {
          mode: regionRuntime.mode,
          reason: regionRuntime.reason,
          atlasFingerprint: regionRuntime.atlas?.fingerprint ?? null,
          atlasRevision: regionRuntime.atlas?.revision ?? null,
          ephemeralAtlas: regionRuntime.ephemeralAtlas,
          verificationOnly: true,
        },
      },
      loadArtifact: (artifactId) => {
        if (!episode.base.artifactIds.includes(artifactId)) throw thoughtError('Motor taban dışı artifact okuyamaz.', 'THOUGHT_ARTIFACT_OUTSIDE_BASE', 403)
        return this.store.getArtifact(artifactId)
      },
      shouldPause: () => false,
      shouldCancel: () => Boolean(signal?.aborted),
      onTick: acceptActualTick,
    }
    const result = useResidencyReplay
      ? await this.runEngineEpisode(request, replayContext, base, episode)
      : await this.engine.runEpisode(request, { ...replayContext, neurons: base.neurons })
    if (Array.isArray(result?.ticks)) {
      if (result.ticks.length > episode.maxTicks) {
        throw thoughtError('Replay motoru sonuç tick sınırını aştı.', 'THOUGHT_ENGINE_LIMIT_EXCEEDED', 502)
      }
      for (const tick of result.ticks) acceptActualTick(tick)
    }
    let firstMismatch = null
    const length = Math.max(expected.length, actual.length)
    for (let index = 0; index < length; index += 1) {
      if (expected[index]?.tickDigest !== actual[index]?.tickDigest) {
        firstMismatch = {
          position: index,
          expectedIndex: expected[index]?.index ?? null,
          actualIndex: actual[index]?.index ?? null,
          expectedDigest: expected[index]?.tickDigest ?? null,
          actualDigest: actual[index]?.tickDigest ?? null,
        }
        break
      }
    }
    return {
      episodeId: id,
      matched: firstMismatch === null,
      comparedTicks: Math.min(expected.length, actual.length),
      expectedTickCount: expected.length,
      actualTickCount: actual.length,
      firstMismatch,
      baseModelFingerprint: episode.base.baseModelFingerprint,
      residencyManifestFingerprint: episode.base.residencyManifestFingerprint ?? null,
      engineContract: 'thoughtEngine.runEpisode(request, context)',
    }
  }

  async close() {
    for (const { controller } of this.active.values()) {
      controller.abort(thoughtError('Servis kapatılıyor.', 'SERVICE_STOP', 409))
    }
    await Promise.allSettled([...this.active.values()].map(({ promise }) => promise))
    for (const residency of this.residencies.values()) residency.repository.close()
    this.residencies.clear()
    this.residencyMeta.clear()
    this.residencyPins.clear()
    this.baseCache.clear()
    this.baseCacheAccess.clear()
  }
}

export function createThoughtService(options) {
  return new ThoughtService(options)
}
