import {
  createSynthesisConfig,
  type SynthesisCandidate,
  type SynthesisConfig,
  type SynthesisMemberRef,
  type SynthesisRun,
} from '../synthesis'
import { digestVector } from '../text-lab/engine/representation'

const DATABASE_NAME = 'noron-sentez-laboratuvari'
const DATABASE_VERSION = 1
const DECISION_STORE = 'candidate-decisions'

export type CandidateDecision = 'approved' | 'rejected'

export interface SynthesisArchiveRecord {
  id: string
  format: 'noron-sentez-karari'
  version: 1
  candidateId: string
  candidateFingerprint: string
  decision: CandidateDecision
  reviewedAt: string
  runId: string
  runFingerprint: string
  representationSpaceId: string
  analysisRunIds: string[]
  sourceDigests: string[]
  config: SynthesisConfig
  candidate: SynthesisCandidate
}

const CONFIG_KEYS = [
  'nodeSelection',
  'maximumSpanGraphemes',
  'minimumIdentitySimilarity',
  'minimumRelationSimilarity',
  'minimumCombinedSimilarity',
  'identityOverrideSimilarity',
  'identityWeight',
  'maximumEdgesPerMember',
  'maximumRetainedEdges',
  'maximumClusterMembers',
  'maximumCandidates',
  'minimumSupport',
  'minimumSources',
  'yieldEveryComparisons',
] as const satisfies ReadonlyArray<keyof SynthesisConfig>

const textEncoder = new TextEncoder()

function archiveFingerprintParts(
  parts: readonly (string | number | boolean | Float32Array)[],
): string {
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

function fail(message: string): never {
  throw new TypeError(`Sentez arşivi kaydı geçersiz: ${message}`)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value || value.includes('\u0000')) {
    fail(`${label} boş veya bozuk.`)
  }
  return value
}

function requiredMemberKey(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value) fail(`${label} boş veya bozuk.`)
  return value
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(`${label} sonlu bir sayı değil.`)
  }
  return value
}

function nonNegativeInteger(value: unknown, label: string): number {
  const number = finiteNumber(value, label)
  if (!Number.isInteger(number) || number < 0) fail(`${label} negatif olmayan bir tam sayı değil.`)
  return number
}

function uniqueStrings(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || value.length === 0) fail(`${label} boş veya dizi değil.`)
  const strings = value.map((entry, index) => requiredString(entry, `${label}[${index}]`))
  if (new Set(strings).size !== strings.length) fail(`${label} tekrar eden kimlik içeriyor.`)
  return strings
}

function validateConfig(value: unknown): SynthesisConfig {
  if (!isRecord(value)) fail('sentez ayarları nesne değil.')
  for (const key of CONFIG_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(value, key)) fail(`sentez ayarında “${key}” eksik.`)
  }
  try {
    return createSynthesisConfig(value as Partial<SynthesisConfig>)
  } catch (cause) {
    fail(cause instanceof Error ? cause.message : 'sentez ayarları doğrulanamadı.')
  }
}

function validateVector(value: unknown, label: string): Float32Array {
  if (!(value instanceof Float32Array) || value.length === 0 || value.length > 64 * 64) {
    fail(`${label} sonlu, boş olmayan ve desteklenen uzunlukta bir Float32Array değil.`)
  }
  for (const entry of value) {
    if (!Number.isFinite(entry) || entry < -1 || entry > 1) {
      fail(`${label} [-1, 1] dışında veya sonlu olmayan değer içeriyor.`)
    }
  }
  return value
}

function validateMemberRef(value: unknown, label: string): SynthesisMemberRef {
  if (!isRecord(value)) fail(`${label} nesne değil.`)
  const ref = value as unknown as SynthesisMemberRef
  const sourceDigest = requiredString(ref.sourceDigest, `${label}.sourceDigest`)
  const analysisRunId = requiredString(ref.analysisRunId, `${label}.analysisRunId`)
  const neuronId = requiredString(ref.neuronId, `${label}.neuronId`)
  requiredString(ref.sourceId, `${label}.sourceId`)
  const key = requiredMemberKey(ref.key, `${label}.key`)
  nonNegativeInteger(ref.paragraphIndex, `${label}.paragraphIndex`)
  const start = nonNegativeInteger(ref.start, `${label}.start`)
  const end = nonNegativeInteger(ref.end, `${label}.end`)
  const graphemeCount = nonNegativeInteger(ref.graphemeCount, `${label}.graphemeCount`)
  nonNegativeInteger(ref.depth, `${label}.depth`)
  if (typeof ref.text !== 'string' || !ref.text.length) fail(`${label}.text boş veya metin değil.`)
  if (end <= start || end - start !== ref.text.length) fail(`${label} kaynak aralığı metinle uyuşmuyor.`)
  if (graphemeCount < 1 || graphemeCount > ref.text.length) fail(`${label}.graphemeCount geçersiz.`)
  if (ref.parentId !== null) {
    requiredString(ref.parentId, `${label}.parentId`)
    if (ref.parentId === neuronId) fail(`${label} kendi ebeveyni olamaz.`)
  }
  if (key !== `${sourceDigest}\u0000${analysisRunId}\u0000${neuronId}`) {
    fail(`${label}.key kaynak, deney ve nöron kimlikleriyle uyuşmuyor.`)
  }
  return ref
}

function sameMemberRef(left: SynthesisMemberRef, right: SynthesisMemberRef): boolean {
  return (
    left.key === right.key &&
    left.analysisRunId === right.analysisRunId &&
    left.sourceId === right.sourceId &&
    left.sourceDigest === right.sourceDigest &&
    left.neuronId === right.neuronId &&
    left.paragraphIndex === right.paragraphIndex &&
    left.start === right.start &&
    left.end === right.end &&
    left.text === right.text &&
    left.graphemeCount === right.graphemeCount &&
    left.depth === right.depth &&
    left.parentId === right.parentId
  )
}

function validateCandidate(
  value: unknown,
  representationSpaceId: string,
  analysisRunIds: ReadonlySet<string>,
  sourceDigests: ReadonlySet<string>,
  config: SynthesisConfig,
): SynthesisCandidate {
  if (!isRecord(value)) fail('candidate nesne değil.')
  const candidate = value as unknown as SynthesisCandidate
  const fingerprint = requiredString(candidate.fingerprint, 'candidate.fingerprint')
  if (!/^[0-9a-f]{32}$/.test(fingerprint)) fail('candidate fingerprint biçimi geçersiz.')
  if (candidate.id !== `SN-${fingerprint.slice(0, 16)}`) fail('candidate id fingerprint ile uyuşmuyor.')
  if (candidate.representationSpaceId !== representationSpaceId) {
    fail('candidate temsil uzayı arşiv kaydıyla uyuşmuyor.')
  }

  if (!Array.isArray(candidate.members) || candidate.members.length === 0) {
    fail('candidate üyeleri boş veya dizi değil.')
  }
  if (!Array.isArray(candidate.independentSupports) || candidate.independentSupports.length === 0) {
    fail('candidate bağımsız destekleri boş veya dizi değil.')
  }
  const members = candidate.members.map((member, index) =>
    validateMemberRef(member, `candidate.members[${index}]`))
  const memberByKey = new Map<string, SynthesisMemberRef>()
  for (const member of members) {
    if (memberByKey.has(member.key)) fail('candidate üyelerinde tekrar eden referans var.')
    if (!analysisRunIds.has(member.analysisRunId)) fail('candidate üyesi bilinmeyen analiz deneyine bağlı.')
    if (!sourceDigests.has(member.sourceDigest)) fail('candidate üyesi bilinmeyen kaynağa bağlı.')
    memberByKey.set(member.key, member)
  }
  if (members.length > config.maximumClusterMembers) fail('candidate küme üyesi sınırını aşıyor.')

  const supports = candidate.independentSupports.map((support, index) =>
    validateMemberRef(support, `candidate.independentSupports[${index}]`))
  const supportKeys = new Set<string>()
  for (const support of supports) {
    if (supportKeys.has(support.key)) fail('candidate desteklerinde tekrar eden referans var.')
    supportKeys.add(support.key)
    const member = memberByKey.get(support.key)
    if (!member || !sameMemberRef(member, support)) {
      fail('candidate bağımsız desteği üye kümesiyle uyuşmuyor.')
    }
  }

  const medoid = validateMemberRef(candidate.medoid, 'candidate.medoid')
  const medoidMember = memberByKey.get(medoid.key)
  if (!medoidMember || !sameMemberRef(medoidMember, medoid) || !supportKeys.has(medoid.key)) {
    fail('candidate medoid referansı bağımsız destek kümesinde değil.')
  }

  const identity = validateVector(candidate.identity, 'candidate.identity')
  const relation = validateVector(candidate.relation, 'candidate.relation')
  const cellCount = identity.length + relation.length
  const side = Math.sqrt(cellCount)
  if (!Number.isInteger(side) || side < 8 || side > 64 || identity.length % side !== 0) {
    fail('candidate kimlik ve ilişki vektörü uzunlukları geçerli bir matris düzeni oluşturmuyor.')
  }
  if (candidate.identityDigest !== digestVector(identity)) fail('candidate identity digest doğrulanamadı.')
  if (candidate.identityFrozen !== true) fail('candidate kimliği kilitli değil.')
  if (candidate.contextRevision !== 0) fail('candidate başlangıç context revision değeri sıfır değil.')

  const contextSupport = nonNegativeInteger(candidate.contextSupport, 'candidate.contextSupport')
  const supportCount = nonNegativeInteger(candidate.supportCount, 'candidate.supportCount')
  const sourceCount = nonNegativeInteger(candidate.sourceCount, 'candidate.sourceCount')
  const independentSourceCount = new Set(supports.map((support) => support.sourceDigest)).size
  if (
    contextSupport !== supports.length ||
    supportCount !== supports.length ||
    sourceCount !== independentSourceCount
  ) {
    fail('candidate destek veya kaynak sayaçları referanslarla uyuşmuyor.')
  }
  if (supportCount < config.minimumSupport || sourceCount < config.minimumSources) {
    fail('candidate doğum için kaydedilmiş asgari kanıtı karşılamıyor.')
  }

  const identityCohesion = finiteNumber(candidate.identityCohesion, 'candidate.identityCohesion')
  const relationCohesion = finiteNumber(candidate.relationCohesion, 'candidate.relationCohesion')
  const birthScore = finiteNumber(candidate.birthScore, 'candidate.birthScore')
  if (identityCohesion < -1 || identityCohesion > 1) fail('candidate identity cohesion aralık dışında.')
  if (relationCohesion < -1 || relationCohesion > 1) fail('candidate relation cohesion aralık dışında.')
  if (birthScore < 0) fail('candidate birth score negatif.')

  const expectedFingerprint = archiveFingerprintParts([
    'synthesis-candidate-v1',
    representationSpaceId,
    ...members.map((member) => member.key).sort(),
    'supports',
    ...supports.map((support) => support.key).sort(),
    medoid.key,
    identity,
    candidate.identityDigest,
    relation,
  ])
  if (fingerprint !== expectedFingerprint) fail('candidate fingerprint içerikle uyuşmuyor.')
  return candidate
}

/**
 * Treats IndexedDB data as untrusted and validates every cross-reference and
 * numerical invariant before the archive can be consumed by later engines.
 */
export function validateSynthesisArchiveRecord(value: unknown): SynthesisArchiveRecord {
  if (!isRecord(value)) fail('kayıt nesne değil.')
  const record = value as unknown as SynthesisArchiveRecord
  if (record.format !== 'noron-sentez-karari' || record.version !== 1) {
    fail('format veya sürüm desteklenmiyor.')
  }
  if (record.decision !== 'approved' && record.decision !== 'rejected') {
    fail('karar approved veya rejected değil.')
  }
  const candidateFingerprint = requiredString(record.candidateFingerprint, 'candidateFingerprint')
  if (!/^[0-9a-f]{32}$/.test(candidateFingerprint)) fail('candidate fingerprint biçimi geçersiz.')
  if (record.id !== `decision:${candidateFingerprint}`) fail('kayıt id candidate fingerprint ile uyuşmuyor.')
  const runFingerprint = requiredString(record.runFingerprint, 'runFingerprint')
  if (!/^[0-9a-f]{32}$/.test(runFingerprint)) fail('run fingerprint biçimi geçersiz.')
  if (record.runId !== `SR-${runFingerprint.slice(0, 16)}`) fail('run id fingerprint ile uyuşmuyor.')
  const representationSpaceId = requiredString(record.representationSpaceId, 'representationSpaceId')
  if (!/^RS-[0-9A-F]{16}$/.test(representationSpaceId)) fail('representation space id biçimi geçersiz.')
  if (typeof record.reviewedAt !== 'string' || Number.isNaN(Date.parse(record.reviewedAt))) {
    fail('reviewedAt geçerli bir tarih değil.')
  }
  const analysisRunIds = uniqueStrings(record.analysisRunIds, 'analysisRunIds')
  const sourceDigests = uniqueStrings(record.sourceDigests, 'sourceDigests')
  const config = validateConfig(record.config)
  const candidate = validateCandidate(
    record.candidate,
    representationSpaceId,
    new Set(analysisRunIds),
    new Set(sourceDigests),
    config,
  )
  if (record.candidateId !== candidate.id || candidateFingerprint !== candidate.fingerprint) {
    fail('candidate id veya fingerprint üst kayıtla uyuşmuyor.')
  }
  return record
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    request.onupgradeneeded = () => {
      const database = request.result
      if (!database.objectStoreNames.contains(DECISION_STORE)) {
        const store = database.createObjectStore(DECISION_STORE, { keyPath: 'id' })
        store.createIndex('by-reviewed-at', 'reviewedAt')
        store.createIndex('by-candidate-fingerprint', 'candidateFingerprint', { unique: true })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Sentez arşivi açılamadı.'))
  })
}

function awaitRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Sentez arşivi işlemi tamamlanamadı.'))
  })
}

function awaitTransaction(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onabort = () => reject(transaction.error ?? new Error('Sentez arşivi işlemi durduruldu.'))
    transaction.onerror = () => reject(transaction.error ?? new Error('Sentez arşivi işlemi tamamlanamadı.'))
  })
}

export async function loadSynthesisArchive(): Promise<SynthesisArchiveRecord[]> {
  const database = await openDatabase()
  try {
    const transaction = database.transaction(DECISION_STORE, 'readonly')
    const records: unknown[] = await awaitRequest(
      transaction.objectStore(DECISION_STORE).getAll(),
    )
    return records
      .map(validateSynthesisArchiveRecord)
      .sort((left, right) => right.reviewedAt.localeCompare(left.reviewedAt))
  } finally {
    database.close()
  }
}

/** Returns only integrity-checked neurons that were explicitly approved. */
export async function loadApprovedSynthesisArchive(): Promise<SynthesisArchiveRecord[]> {
  const records = await loadSynthesisArchive()
  return records.filter((record) => record.decision === 'approved')
}

export async function saveCandidateDecision(
  run: SynthesisRun,
  candidate: SynthesisCandidate,
  decision: CandidateDecision,
): Promise<SynthesisArchiveRecord> {
  const record: SynthesisArchiveRecord = {
    id: `decision:${candidate.fingerprint}`,
    format: 'noron-sentez-karari',
    version: 1,
    candidateId: candidate.id,
    candidateFingerprint: candidate.fingerprint,
    decision,
    reviewedAt: new Date().toISOString(),
    runId: run.id,
    runFingerprint: run.fingerprint,
    representationSpaceId: run.representationSpaceId,
    analysisRunIds: [...run.analysisRunIds],
    sourceDigests: [...run.sourceDigests],
    config: { ...run.config },
    candidate,
  }
  validateSynthesisArchiveRecord(record)

  const database = await openDatabase()
  try {
    const transaction = database.transaction(DECISION_STORE, 'readwrite')
    transaction.objectStore(DECISION_STORE).put(record)
    await awaitTransaction(transaction)
    return record
  } finally {
    database.close()
  }
}
