import { createHash } from 'node:crypto'
import {
  constants as fsConstants,
  copyFile,
  mkdir,
  mkdtemp,
  open,
  readFile,
  rm,
  stat,
} from 'node:fs/promises'
import path from 'node:path'

export const NEURON_TIER_FORMAT = 'bombom-neuron-tier-manifest'
export const NEURON_TIER_LEGACY_VERSION = 1
export const NEURON_TIER_VERSION = 2
export const NEURON_UTILITY_SCALE = 1_000_000

const PAGE_MAGIC = Buffer.from('BMNRPG01', 'ascii')
const PAGE_HEADER_BYTES = PAGE_MAGIC.byteLength + 4
const RECORD_HEADER_BYTES = 12
const ROUTING_SKETCH_BITS = 128
const ROUTING_LSH_BANDS = 4
const ROUTING_LSH_BITS_PER_BAND = 16
const ROUTING_LSH_BUCKET_LIMIT = 16
const ROUTING_LSH_INSPECTION_LIMIT = 512
const RELATION_PAGE_IDENTITY_LENGTH = 512
const RELATION_PAGE_RELATION_LENGTH = 3_584
const RELATION_PAGE_COUNT = 56
const RELATION_PAGE_LENGTH = 64
const RELATION_PAGE_SKETCH_BITS = 16
const MAX_RELATION_PAGE_QUERY_COUNT = 4

export const RELATION_PAGE_SKETCH_FORMAT = 'relation-page-sign-projection-v1'
export const ADAPTIVE_RELATION_PAGE_PROFILE = 'adaptive-eighth-id-v1'

export const DEFAULT_NEURON_TIER_POLICY = Object.freeze({
  version: 1,
  hotBudgetBytes: 256 * 1024 * 1024,
  warmBudgetBytes: 64 * 1024 * 1024,
  pageTargetBytes: 2 * 1024 * 1024,
  maximumPageBytes: 4 * 1024 * 1024,
  maximumWorkingSetNeurons: 128,
  hotUtilityThreshold: 0.62,
  warmUtilityThreshold: 0.18,
  hotPromotionHits: 2,
  utilityEmaNumerator: 3,
  utilityEmaDenominator: 10,
  idleHalfLifeAccesses: 512,
  utilityWeights: Object.freeze({
    signalSimilarity: 0.35,
    routeProbability: 0.30,
    familyProbability: 0.20,
    observedUseProbability: 0.15,
  }),
})

function tierError(code, message) {
  const error = new Error(message)
  error.code = code
  return error
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function finiteProbability(value, name) {
  const number = Number(value ?? 0)
  if (!Number.isFinite(number) || number < 0 || number > 1) {
    throw new RangeError(`${name} 0 ile 1 arasında sonlu bir sayı olmalıdır.`)
  }
  return number
}

function positiveInteger(value, name, minimum = 1) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name} en az ${minimum} olan güvenli bir tam sayı olmalıdır.`)
  }
  return value
}

function canonicalValue(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('Kalıcı nöron metadata alanları sonlu olmalıdır.')
    return Object.is(value, -0) ? 0 : value
  }
  if (Array.isArray(value)) return value.map(canonicalValue)
  if (value && typeof value === 'object') {
    const output = {}
    for (const key of Object.keys(value).sort()) {
      if (value[key] !== undefined) output[key] = canonicalValue(value[key])
    }
    return output
  }
  throw new TypeError(`Kalıcı nöron metadata türü desteklenmiyor: ${typeof value}`)
}

function canonicalJson(value) {
  return JSON.stringify(canonicalValue(value))
}

function normalizedWeights(input = {}) {
  const weights = {
    ...DEFAULT_NEURON_TIER_POLICY.utilityWeights,
    ...input,
  }
  let total = 0
  for (const key of Object.keys(DEFAULT_NEURON_TIER_POLICY.utilityWeights)) {
    const value = finiteProbability(weights[key], `utilityWeights.${key}`)
    weights[key] = value
    total += value
  }
  if (Math.abs(total - 1) > 1e-9) {
    throw new RangeError('Nöron kullanılabilirlik ağırlıklarının toplamı 1 olmalıdır.')
  }
  return Object.freeze(weights)
}

export function normalizeNeuronTierPolicy(input = {}) {
  const policy = {
    ...DEFAULT_NEURON_TIER_POLICY,
    ...input,
    utilityWeights: normalizedWeights(input.utilityWeights),
  }
  positiveInteger(policy.hotBudgetBytes, 'hotBudgetBytes')
  positiveInteger(policy.warmBudgetBytes, 'warmBudgetBytes')
  positiveInteger(policy.pageTargetBytes, 'pageTargetBytes', PAGE_HEADER_BYTES + 1)
  positiveInteger(policy.maximumPageBytes, 'maximumPageBytes', policy.pageTargetBytes)
  positiveInteger(policy.maximumWorkingSetNeurons, 'maximumWorkingSetNeurons')
  positiveInteger(policy.hotPromotionHits, 'hotPromotionHits')
  positiveInteger(policy.utilityEmaNumerator, 'utilityEmaNumerator')
  positiveInteger(policy.utilityEmaDenominator, 'utilityEmaDenominator', policy.utilityEmaNumerator)
  positiveInteger(policy.idleHalfLifeAccesses, 'idleHalfLifeAccesses')
  policy.hotUtilityThreshold = finiteProbability(policy.hotUtilityThreshold, 'hotUtilityThreshold')
  policy.warmUtilityThreshold = finiteProbability(policy.warmUtilityThreshold, 'warmUtilityThreshold')
  if (policy.hotUtilityThreshold <= policy.warmUtilityThreshold) {
    throw new RangeError('hotUtilityThreshold, warmUtilityThreshold değerinden büyük olmalıdır.')
  }
  policy.version = 1
  return Object.freeze(policy)
}

/**
 * Deterministic fixed-point estimate of P(neuron will be useful soon).
 * Distance contributes only through similarity=1-distance, so moving a neuron
 * farther away can never increase its utility when all other evidence is fixed.
 */
function scoreNeuronUtilityWithPolicy(input, policy) {
  const distance = finiteProbability(input.distance ?? 1, 'distance')
  const components = {
    signalSimilarity: 1 - distance,
    routeProbability: finiteProbability(input.routeProbability, 'routeProbability'),
    familyProbability: finiteProbability(input.familyProbability, 'familyProbability'),
    observedUseProbability: finiteProbability(input.observedUseProbability, 'observedUseProbability'),
  }
  const raw = Object.entries(policy.utilityWeights)
    .reduce((sum, [key, weight]) => sum + components[key] * weight, 0)
  const fixed = Math.max(0, Math.min(NEURON_UTILITY_SCALE, Math.round(raw * NEURON_UTILITY_SCALE)))
  return { fixed, probability: fixed / NEURON_UTILITY_SCALE, components }
}

export function scoreNeuronUtility(input = {}, policyInput = DEFAULT_NEURON_TIER_POLICY) {
  return scoreNeuronUtilityWithPolicy(input, normalizeNeuronTierPolicy(policyInput))
}

function assertVector(value, expectedLength, name) {
  if (!(value instanceof Float32Array) || value.length !== expectedLength) {
    throw new TypeError(`${name} ${expectedLength} hücreli Float32Array olmalıdır.`)
  }
  for (const number of value) {
    if (!Number.isFinite(number)) throw new RangeError(`${name} yalnız sonlu değerler içermelidir.`)
  }
}

function float32Bytes(vector) {
  const bytes = Buffer.allocUnsafe(vector.byteLength)
  for (let index = 0; index < vector.length; index += 1) bytes.writeFloatLE(vector[index], index * 4)
  return bytes
}

function float32FromBytes(bytes, length) {
  if (bytes.byteLength !== length * 4) throw tierError('NEURON_RECORD_CORRUPT', 'Nöron vektör uzunluğu bozuk.')
  const result = new Float32Array(length)
  for (let index = 0; index < length; index += 1) result[index] = bytes.readFloatLE(index * 4)
  return result
}

function neuronMetadata(neuron) {
  const metadata = {}
  for (const [key, value] of Object.entries(neuron)) {
    if (
      key !== 'identity' && key !== 'relation' && key !== 'routingVector' &&
      key !== 'routingRelationDelta' && value !== undefined
    ) metadata[key] = value
  }
  return canonicalValue(metadata)
}

function routingCoordinates(modelFingerprint, cellCount) {
  const count = Math.min(ROUTING_SKETCH_BITS, cellCount)
  const selected = []
  const seen = new Set()
  for (let nonce = 0; selected.length < count; nonce += 1) {
    const bytes = createHash('sha256')
      .update('bombom-neuron-routing-sketch-v1\0')
      .update(modelFingerprint)
      .update('\0')
      .update(String(nonce))
      .digest()
    for (let offset = 0; offset + 4 <= bytes.byteLength && selected.length < count; offset += 4) {
      const coordinate = bytes.readUInt32LE(offset) % cellCount
      if (seen.has(coordinate)) continue
      seen.add(coordinate)
      selected.push(coordinate)
    }
  }
  return selected
}

function routingSketch(vector, coordinates) {
  const bytes = Buffer.alloc(Math.ceil(coordinates.length / 8))
  for (let bit = 0; bit < coordinates.length; bit += 1) {
    if (vector[coordinates[bit]] >= 0) bytes[bit >>> 3] |= 1 << (bit & 7)
  }
  return bytes.toString('hex')
}

function relationPageSketchDefinition(physicalRepresentationSpaceId, layout) {
  if (
    layout.identityLength !== RELATION_PAGE_IDENTITY_LENGTH ||
    layout.relationLength !== RELATION_PAGE_RELATION_LENGTH
  ) return null
  const coordinatesWithinPage = routingCoordinates(
    `physical-relation-pages:${physicalRepresentationSpaceId}`,
    RELATION_PAGE_LENGTH,
  ).slice(0, RELATION_PAGE_SKETCH_BITS)
  const body = {
    format: RELATION_PAGE_SKETCH_FORMAT,
    pageCount: RELATION_PAGE_COUNT,
    pageLength: RELATION_PAGE_LENGTH,
    bitCountPerPage: RELATION_PAGE_SKETCH_BITS,
    relationOffset: RELATION_PAGE_IDENTITY_LENGTH,
    relationLength: RELATION_PAGE_RELATION_LENGTH,
    coordinatesWithinPage,
  }
  return {
    ...body,
    // This fingerprint binds only physical space + fixed 512/3584 geometry.
    // Thought heads and adaptive-atlas revisions deliberately do not enter it.
    physicalFingerprint: sha256(Buffer.from(
      `bombom-relation-page-sketch-physical-v1\0${physicalRepresentationSpaceId}\0${canonicalJson(body)}`,
    )),
  }
}

function relationPageSketchValue(neuron, layout, definition) {
  const byteCount = Math.ceil(definition.bitCountPerPage / 8)
  const bytes = Buffer.alloc(byteCount * definition.pageCount)
  const routingVector = neuron.routingVector
  const relationDelta = neuron.routingRelationDelta
  for (let pageIndex = 0; pageIndex < definition.pageCount; pageIndex += 1) {
    const pageOffset = pageIndex * definition.pageLength
    for (let bit = 0; bit < definition.coordinatesWithinPage.length; bit += 1) {
      const relationIndex = pageOffset + definition.coordinatesWithinPage[bit]
      const value = routingVector !== undefined
        ? routingVector[layout.identityLength + relationIndex]
        : neuron.relation[relationIndex] + (relationDelta?.[relationIndex] ?? 0)
      if (value >= 0) bytes[pageIndex * byteCount + (bit >>> 3)] |= 1 << (bit & 7)
    }
  }
  return bytes.toString('hex')
}

function signalRelationPageSketch(signal, layout, definition, pageIndex) {
  const byteCount = Math.ceil(definition.bitCountPerPage / 8)
  const bytes = Buffer.alloc(byteCount)
  const pageOffset = layout.identityLength + pageIndex * definition.pageLength
  for (let bit = 0; bit < definition.coordinatesWithinPage.length; bit += 1) {
    if (signal[pageOffset + definition.coordinatesWithinPage[bit]] >= 0) {
      bytes[bit >>> 3] |= 1 << (bit & 7)
    }
  }
  return bytes.toString('hex')
}

function relationPageSketchAt(sketches, definition, pageIndex) {
  const hexLength = Math.ceil(definition.bitCountPerPage / 8) * 2
  return sketches.slice(pageIndex * hexLength, (pageIndex + 1) * hexLength)
}

function normalizeRelationPageIndexes(value, pageCount) {
  if (!Array.isArray(value) || !value.length) {
    throw new TypeError('Bölge duyarlı SSD sorgusu en az bir ilişki mikro-sayfası gerektirir.')
  }
  const result = [...new Set(value)]
  if (
    result.length > MAX_RELATION_PAGE_QUERY_COUNT ||
    result.some((pageIndex) => !Number.isSafeInteger(pageIndex) || pageIndex < 0 || pageIndex >= pageCount)
  ) {
    throw new RangeError(`Bölge duyarlı SSD sorgusu en fazla ${MAX_RELATION_PAGE_QUERY_COUNT} geçerli mikro-sayfa kabul eder.`)
  }
  return result.sort((left, right) => left - right)
}

function neuronRoutingSketch(neuron, layout, coordinates) {
  if (neuron.routingVector !== undefined) {
    assertVector(neuron.routingVector, layout.cellCount, `${neuron.id} yönlendirme vektörü`)
    return routingSketch(neuron.routingVector, coordinates)
  }
  if (neuron.routingRelationDelta !== undefined) {
    assertVector(neuron.routingRelationDelta, layout.relationLength, `${neuron.id} yönlendirme R yaması`)
  }
  const bytes = Buffer.alloc(Math.ceil(coordinates.length / 8))
  for (let bit = 0; bit < coordinates.length; bit += 1) {
    const coordinate = coordinates[bit]
    const value = coordinate < layout.identityLength
      ? neuron.identity[coordinate]
      : neuron.relation[coordinate - layout.identityLength] +
        (neuron.routingRelationDelta?.[coordinate - layout.identityLength] ?? 0)
    if (value >= 0) bytes[bit >>> 3] |= 1 << (bit & 7)
  }
  return bytes.toString('hex')
}

const NIBBLE_POPCOUNT = Object.freeze([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4])

function hexNibble(code) {
  if (code >= 48 && code <= 57) return code - 48
  if (code >= 97 && code <= 102) return code - 87
  return -1
}

function hammingDistance(leftHex, rightHex, bitCount) {
  if (leftHex.length !== rightHex.length) {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron yönlendirme eskizi uzunluğu uyuşmuyor.')
  }
  let differences = 0
  for (let index = 0; index < leftHex.length; index += 1) {
    const left = hexNibble(leftHex.charCodeAt(index))
    const right = hexNibble(rightHex.charCodeAt(index))
    if (left < 0 || right < 0) throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron yönlendirme eskizi hex değil.')
    differences += NIBBLE_POPCOUNT[left ^ right]
  }
  return bitCount ? differences / bitCount : 1
}

function nearestOrder(left, right) {
  return left.distance - right.distance || left.id.localeCompare(right.id)
}

function isWorse(left, right) {
  return nearestOrder(left, right) > 0
}

function heapPushWorst(heap, value) {
  heap.push(value)
  let index = heap.length - 1
  while (index > 0) {
    const parent = Math.floor((index - 1) / 2)
    if (!isWorse(heap[index], heap[parent])) break
    ;[heap[parent], heap[index]] = [heap[index], heap[parent]]
    index = parent
  }
}

function heapReplaceWorst(heap, value) {
  heap[0] = value
  let index = 0
  while (true) {
    const left = index * 2 + 1
    const right = left + 1
    if (left >= heap.length) break
    let worst = left
    if (right < heap.length && isWorse(heap[right], heap[left])) worst = right
    if (!isWorse(heap[worst], heap[index])) break
    ;[heap[index], heap[worst]] = [heap[worst], heap[index]]
    index = worst
  }
}

/** O(N log K), O(K) extra memory deterministic nearest-sketch selection. */
export function rankNeuronRoutingSketches(entries, querySketch, bitCount, maximum) {
  const limit = positiveInteger(maximum, 'maximum')
  const heap = []
  for (const entry of entries) {
    const candidate = {
      id: entry.id,
      distance: hammingDistance(entry.routingSketch, querySketch, bitCount),
    }
    if (heap.length < limit) heapPushWorst(heap, candidate)
    else if (nearestOrder(candidate, heap[0]) < 0) heapReplaceWorst(heap, candidate)
  }
  return heap.sort(nearestOrder)
}

function routingBandHex(sketchHex, band) {
  // Four 16-bit bands are spread across the complete 128-bit sketch. This
  // avoids concentrating the index on one contiguous semantic partition.
  const nibbleOffset = band * 8
  return sketchHex.slice(nibbleOffset, nibbleOffset + ROUTING_LSH_BITS_PER_BAND / 4)
}

function routingProbeKeys(querySketch) {
  const keys = []
  for (let band = 0; band < ROUTING_LSH_BANDS; band += 1) {
    const exactHex = routingBandHex(querySketch, band)
    const exact = Number.parseInt(exactHex, 16)
    keys.push(`${band}:${exactHex}`)
    for (let bit = 0; bit < ROUTING_LSH_BITS_PER_BAND; bit += 1) {
      keys.push(`${band}:${(exact ^ (1 << bit)).toString(16).padStart(4, '0')}`)
    }
  }
  return keys
}

function greatestCommonDivisor(left, right) {
  let a = Math.abs(left)
  let b = Math.abs(right)
  while (b) [a, b] = [b, a % b]
  return a
}

/**
 * Immutable compact ANN gate built once when a manifest is opened. Query cost
 * is independent of catalog N: at most 68 bucket probes and 512 entry visits.
 * Catalogs <=128 deliberately use every entry, preserving exact replay.
 */
export class NeuronRoutingSketchIndex {
  #entries
  #orderedIds
  #buckets = new Map()

  constructor(entries) {
    this.#entries = entries instanceof Map ? entries : new Map([...entries].map((entry) => [entry.id, entry]))
    this.#orderedIds = [...this.#entries.keys()]
    for (const entry of this.#entries.values()) {
      for (let band = 0; band < ROUTING_LSH_BANDS; band += 1) {
        const key = `${band}:${routingBandHex(entry.routingSketch, band)}`
        const bucket = this.#buckets.get(key) ?? []
        // Manifest ids are validated as strictly sorted. Keeping the first
        // bounded entries therefore has deterministic collision behaviour.
        if (bucket.length < ROUTING_LSH_BUCKET_LIMIT) bucket.push(entry)
        this.#buckets.set(key, bucket)
      }
    }
  }

  query(querySketch, bitCount, maximum) {
    const limit = positiveInteger(maximum, 'maximum')
    let inspectedIndexEntries = 0
    let bucketProbes = 0
    const candidates = new Map()
    if (this.#entries.size <= limit) {
      for (const entry of this.#entries.values()) {
        candidates.set(entry.id, entry)
        inspectedIndexEntries += 1
      }
    } else {
      for (const key of routingProbeKeys(querySketch)) {
        bucketProbes += 1
        for (const entry of this.#buckets.get(key) ?? []) {
          inspectedIndexEntries += 1
          candidates.set(entry.id, entry)
          if (inspectedIndexEntries >= ROUTING_LSH_INSPECTION_LIMIT) break
        }
        if (inspectedIndexEntries >= ROUTING_LSH_INSPECTION_LIMIT) break
      }
      // An out-of-distribution query may have no exact/single-bit band hit.
      // Add a fixed-size deterministic fallback without scanning the catalog.
      if (!candidates.size) {
        for (const id of this.sampleIds(`fallback:${querySketch}`, Math.min(limit, 32))) {
          candidates.set(id, this.#entries.get(id))
          inspectedIndexEntries += 1
        }
      }
    }
    return {
      candidates: rankNeuronRoutingSketches(candidates.values(), querySketch, bitCount, limit),
      diagnostics: Object.freeze({
        method: this.#entries.size <= limit ? 'exact-small-catalog' : 'lsh-multiprobe-v1',
        bucketProbes,
        inspectedIndexEntries,
        uniqueCandidates: candidates.size,
        inspectionLimit: ROUTING_LSH_INSPECTION_LIMIT,
      }),
    }
  }

  sampleIds(seed, maximum) {
    const limit = Math.min(this.#orderedIds.length, positiveInteger(maximum, 'maximum'))
    if (!limit) return []
    const bytes = Buffer.from(sha256(`bombom-neuron-sample-v1\0${String(seed)}`), 'hex')
    const length = this.#orderedIds.length
    const start = bytes.readUInt32LE(0) % length
    let step = (bytes.readUInt32LE(4) % length) || 1
    while (greatestCommonDivisor(step, length) !== 1) step = (step + 1) % length || 1
    return Array.from({ length: limit }, (_, index) => this.#orderedIds[(start + index * step) % length])
  }
}

function assertIdentityDigest(neuron, identityDigestOf) {
  if (typeof neuron.identityDigest !== 'string' || !neuron.identityDigest.trim()) {
    throw new TypeError(`${neuron.id} nöronunun sabit I digest'i eksik.`)
  }
  if (identityDigestOf && identityDigestOf(neuron.identity) !== neuron.identityDigest) {
    throw tierError('NEURON_IDENTITY_MISMATCH', `${neuron.id} sabit I digest'i vektörüyle uyuşmuyor.`)
  }
}

function encodeNeuronRecord(neuron, layout, identityDigestOf) {
  const id = String(neuron?.id ?? '').trim()
  if (!id) throw new TypeError('Nöron kimliği boş olamaz.')
  assertVector(neuron.identity, layout.identityLength, `${id} I`)
  assertVector(neuron.relation, layout.relationLength, `${id} R`)
  assertIdentityDigest(neuron, identityDigestOf)
  const metadata = Buffer.from(canonicalJson(neuronMetadata({ ...neuron, id })), 'utf8')
  const identity = float32Bytes(neuron.identity)
  const relation = float32Bytes(neuron.relation)
  const header = Buffer.allocUnsafe(RECORD_HEADER_BYTES)
  header.writeUInt32LE(metadata.byteLength, 0)
  header.writeUInt32LE(neuron.identity.length, 4)
  header.writeUInt32LE(neuron.relation.length, 8)
  return Buffer.concat([header, metadata, identity, relation])
}

function decodeNeuronRecord(bytes, entry, layout, identityDigestOf) {
  if (sha256(bytes) !== entry.recordHash) {
    throw tierError('NEURON_RECORD_CORRUPT', `${entry.id} SSD kaydının içerik özeti uyuşmuyor.`)
  }
  if (bytes.byteLength < RECORD_HEADER_BYTES) throw tierError('NEURON_RECORD_CORRUPT', `${entry.id} SSD kaydı eksik.`)
  const metadataLength = bytes.readUInt32LE(0)
  const identityLength = bytes.readUInt32LE(4)
  const relationLength = bytes.readUInt32LE(8)
  if (identityLength !== layout.identityLength || relationLength !== layout.relationLength) {
    throw tierError('NEURON_REPRESENTATION_MISMATCH', `${entry.id} I/R yerleşimi manifestle uyuşmuyor.`)
  }
  const identityStart = RECORD_HEADER_BYTES + metadataLength
  const relationStart = identityStart + identityLength * 4
  const end = relationStart + relationLength * 4
  if (end !== bytes.byteLength) throw tierError('NEURON_RECORD_CORRUPT', `${entry.id} SSD kayıt sınırları bozuk.`)
  let metadata
  try {
    metadata = JSON.parse(bytes.subarray(RECORD_HEADER_BYTES, identityStart).toString('utf8'))
  } catch {
    throw tierError('NEURON_RECORD_CORRUPT', `${entry.id} metadata kaydı okunamadı.`)
  }
  if (!metadata || metadata.id !== entry.id || metadata.identityDigest !== entry.identityDigest) {
    throw tierError('NEURON_RECORD_CORRUPT', `${entry.id} metadata/index kimliği uyuşmuyor.`)
  }
  const identity = float32FromBytes(bytes.subarray(identityStart, relationStart), identityLength)
  const relation = float32FromBytes(bytes.subarray(relationStart, end), relationLength)
  const neuron = { ...metadata, identity, relation }
  assertIdentityDigest(neuron, identityDigestOf)
  return neuron
}

function pageBytes(records) {
  const header = Buffer.alloc(PAGE_HEADER_BYTES)
  PAGE_MAGIC.copy(header, 0)
  header.writeUInt32LE(records.length, PAGE_MAGIC.byteLength)
  return Buffer.concat([header, ...records])
}

async function promoteImmutableFile(source, destination, expectedHash) {
  await mkdir(path.dirname(destination), { recursive: true })
  try {
    await copyFile(source, destination, fsConstants.COPYFILE_EXCL)
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error
  }
  const existing = await readFile(destination)
  if (sha256(existing) !== expectedHash) {
    throw tierError('NEURON_CONTENT_COLLISION', `İçerik adresli nöron dosyası bozuk veya çakışıyor: ${path.basename(destination)}`)
  }
}

function normalizedLayout(layout) {
  const identityLength = positiveInteger(layout?.identityLength, 'layout.identityLength')
  const relationLength = positiveInteger(layout?.relationLength, 'layout.relationLength')
  const cellCount = identityLength + relationLength
  if (layout.cellCount !== undefined && layout.cellCount !== cellCount) {
    throw new RangeError('Nöron I/R bölümleri layout.cellCount değerini eksiksiz bölmelidir.')
  }
  return { identityLength, relationLength, cellCount }
}

/**
 * Writes a new immutable, content-addressed snapshot. Source artifacts are not
 * changed or deleted; repeated equivalent writes resolve to the same manifest.
 */
export async function writeNeuronTierSnapshot({
  directory,
  modelFingerprint,
  // `representationSpaceId` is a deprecated input alias retained for callers
  // of the generic store. V2 persists the physical and thought-model
  // identities in separate fields and never reinterprets a V1 manifest.
  representationSpaceId,
  physicalRepresentationSpaceId = representationSpaceId,
  thoughtModelFingerprint = modelFingerprint,
  representationProfile = null,
  layout: layoutInput,
  neurons,
  policy: policyInput,
  identityDigestOf,
  includeRelationPageSketches = true,
  maximumSnapshotBytes = Number.MAX_SAFE_INTEGER,
}) {
  if (typeof directory !== 'string' || !directory.trim()) throw new TypeError('Nöron tier dizini gereklidir.')
  if (typeof modelFingerprint !== 'string' || !modelFingerprint.trim()) throw new TypeError('Model fingerprint gereklidir.')
  if (typeof physicalRepresentationSpaceId !== 'string' || !physicalRepresentationSpaceId.trim()) {
    throw new TypeError('Fiziksel temsil uzayı kimliği gereklidir.')
  }
  if (typeof thoughtModelFingerprint !== 'string' || !thoughtModelFingerprint.trim()) {
    throw new TypeError('ThoughtModel fingerprinti gereklidir.')
  }
  if (!Array.isArray(neurons) || !neurons.length) throw new RangeError('En az bir nöron gereklidir.')
  positiveInteger(maximumSnapshotBytes, 'maximumSnapshotBytes')
  const layout = normalizedLayout(layoutInput)
  const policy = normalizeNeuronTierPolicy(policyInput)
  const normalizedRepresentationProfile = representationProfile === null || representationProfile === undefined
    ? null
    : String(representationProfile).trim()
  if (normalizedRepresentationProfile !== null &&
      (!normalizedRepresentationProfile || normalizedRepresentationProfile.length > 128)) {
    throw new TypeError('Temsil profili boş veya 128 karakterden uzun olamaz.')
  }
  const sketchCoordinates = routingCoordinates(modelFingerprint, layout.cellCount)
  const relationPageSketches = includeRelationPageSketches &&
    normalizedRepresentationProfile === ADAPTIVE_RELATION_PAGE_PROFILE
    ? relationPageSketchDefinition(physicalRepresentationSpaceId, layout)
    : null
  // Sort references, not cloned vectors. First-time bootstrap necessarily
  // sees each dense neuron once, but it must not duplicate the entire catalog.
  const ordered = [...neurons].sort((left, right) =>
    String(left?.id ?? '').trim().localeCompare(String(right?.id ?? '').trim()))
  for (const neuron of ordered) {
    if (String(neuron?.id ?? '').trim() !== neuron?.id) {
      throw new TypeError('Nöron kimlikleri boşluksuz ve normalize edilmiş olmalıdır.')
    }
  }
  for (let index = 1; index < ordered.length; index += 1) {
    if (String(ordered[index - 1].id) === String(ordered[index].id)) {
      throw new TypeError(`Yinelenen nöron kimliği: ${ordered[index].id}`)
    }
  }
  const baseDirectory = path.resolve(directory)
  await mkdir(baseDirectory, { recursive: true })
  const staging = await mkdtemp(path.join(baseDirectory, '.tier-staging-'))
  const pages = []
  const index = []
  let pageBytesTotal = 0
  try {
    let group = []
    let groupBytes = PAGE_HEADER_BYTES
    const flushGroup = async () => {
      if (!group.length) return
      const records = group.map((item) => item.bytes)
      const bytes = pageBytes(records)
      const contentHash = sha256(bytes)
      const stagingPath = path.join(staging, `${contentHash}.nrpg`)
      const handle = await open(stagingPath, 'wx')
      try {
        await handle.writeFile(bytes)
        await handle.sync()
      } finally {
        await handle.close()
      }
      let offset = PAGE_HEADER_BYTES
      for (const item of group) {
        index.push({
          id: String(item.neuron.id),
          pageHash: contentHash,
          offset,
          length: item.bytes.byteLength,
          recordHash: sha256(item.bytes),
          identityDigest: item.neuron.identityDigest,
          routingSketch: neuronRoutingSketch(item.neuron, layout, sketchCoordinates),
          ...(relationPageSketches ? {
            relationPageSketches: relationPageSketchValue(item.neuron, layout, relationPageSketches),
          } : {}),
        })
        offset += item.bytes.byteLength
      }
      pages.push({
        contentHash,
        relativePath: `pages/${contentHash}.nrpg`,
        byteLength: bytes.byteLength,
        count: group.length,
        firstNeuronId: String(group[0].neuron.id),
        lastNeuronId: String(group.at(-1).neuron.id),
      })
      pageBytesTotal += bytes.byteLength
      if (pageBytesTotal > maximumSnapshotBytes) {
        throw tierError('NEURON_SNAPSHOT_BUDGET_EXCEEDED', 'Nöron SSD sayfaları ayrılan disk bütçesini aşıyor.')
      }
      group = []
      groupBytes = PAGE_HEADER_BYTES
    }
    for (const neuron of ordered) {
      const record = { neuron, bytes: encodeNeuronRecord(neuron, layout, identityDigestOf) }
      if (PAGE_HEADER_BYTES + record.bytes.byteLength > policy.maximumPageBytes) {
        throw tierError('NEURON_PAGE_LIMIT_EXCEEDED', `${neuron.id} tek başına SSD sayfa sınırını aşıyor.`)
      }
      if (group.length && groupBytes + record.bytes.byteLength > policy.pageTargetBytes) {
        await flushGroup()
      }
      if (groupBytes + record.bytes.byteLength > policy.maximumPageBytes) {
        throw tierError('NEURON_PAGE_LIMIT_EXCEEDED', `${neuron.id} deterministik SSD sayfasına sığmıyor.`)
      }
      group.push(record)
      groupBytes += record.bytes.byteLength
    }
    await flushGroup()

    const policyFingerprint = sha256(Buffer.from(canonicalJson(policy)))
    const body = {
      format: NEURON_TIER_FORMAT,
      version: NEURON_TIER_VERSION,
      modelFingerprint,
      physicalRepresentationSpaceId,
      thoughtModelFingerprint,
      ...(normalizedRepresentationProfile
        ? { representationProfile: normalizedRepresentationProfile }
        : {}),
      layout,
      floatEncoding: 'IEEE754-f32-little-endian',
      routingSketch: {
        format: 'sign-projection-v1',
        bitCount: sketchCoordinates.length,
        coordinates: sketchCoordinates,
      },
      ...(relationPageSketches ? { relationPageSketches } : {}),
      policyFingerprint,
      neuronCount: index.length,
      pageCount: pages.length,
      pageBytes: pageBytesTotal,
      pages,
      index,
    }
    const fingerprint = sha256(Buffer.from(`bombom-neuron-tier-manifest-v2\0${canonicalJson(body)}`))
    const manifest = { ...body, fingerprint }
    const manifestBytes = Buffer.from(canonicalJson(manifest), 'utf8')
    if (pageBytesTotal + manifestBytes.byteLength > maximumSnapshotBytes) {
      throw tierError('NEURON_SNAPSHOT_BUDGET_EXCEEDED', 'Nöron SSD snapshotı ayrılan disk bütçesini aşıyor.')
    }
    for (const page of pages) {
      await promoteImmutableFile(
        path.join(staging, `${page.contentHash}.nrpg`),
        path.join(baseDirectory, page.relativePath),
        page.contentHash,
      )
    }
    const stagingManifest = path.join(staging, `${fingerprint}.json`)
    const handle = await open(stagingManifest, 'wx')
    try {
      await handle.writeFile(manifestBytes)
      await handle.sync()
    } finally {
      await handle.close()
    }
    const manifestPath = path.join(baseDirectory, 'manifests', `${fingerprint}.json`)
    await promoteImmutableFile(stagingManifest, manifestPath, sha256(manifestBytes))
    return { manifest, manifestPath }
  } finally {
    await rm(staging, { recursive: true, force: true })
  }
}

function cloneNeuron(neuron) {
  return {
    ...structuredClone(Object.fromEntries(Object.entries(neuron)
      .filter(([key]) => key !== 'identity' && key !== 'relation'))),
    identity: new Float32Array(neuron.identity),
    relation: new Float32Array(neuron.relation),
  }
}

function validateRelationPageSketchDefinition(definition, layout, physicalRepresentationSpaceId) {
  if (
    layout.identityLength !== RELATION_PAGE_IDENTITY_LENGTH ||
    layout.relationLength !== RELATION_PAGE_RELATION_LENGTH ||
    definition?.format !== RELATION_PAGE_SKETCH_FORMAT ||
    definition.pageCount !== RELATION_PAGE_COUNT ||
    definition.pageLength !== RELATION_PAGE_LENGTH ||
    definition.bitCountPerPage !== RELATION_PAGE_SKETCH_BITS ||
    definition.relationOffset !== RELATION_PAGE_IDENTITY_LENGTH ||
    definition.relationLength !== RELATION_PAGE_RELATION_LENGTH ||
    !Array.isArray(definition.coordinatesWithinPage) ||
    definition.coordinatesWithinPage.length !== RELATION_PAGE_SKETCH_BITS ||
    new Set(definition.coordinatesWithinPage).size !== RELATION_PAGE_SKETCH_BITS ||
    definition.coordinatesWithinPage.some((coordinate) =>
      !Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate >= RELATION_PAGE_LENGTH)
  ) {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD fiziksel ilişki mikro-sayfası eskizi geçersiz.')
  }
  const body = {
    format: definition.format,
    pageCount: definition.pageCount,
    pageLength: definition.pageLength,
    bitCountPerPage: definition.bitCountPerPage,
    relationOffset: definition.relationOffset,
    relationLength: definition.relationLength,
    coordinatesWithinPage: definition.coordinatesWithinPage,
  }
  if (
    canonicalJson(Object.fromEntries(Object.entries(definition).filter(([key]) => key !== 'physicalFingerprint'))) !== canonicalJson(body) ||
    definition.physicalFingerprint !== sha256(Buffer.from(
      `bombom-relation-page-sketch-physical-v1\0${physicalRepresentationSpaceId}\0${canonicalJson(body)}`,
    ))
  ) {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'Fiziksel ilişki mikro-sayfası özeti temsil geometrisiyle uyuşmuyor.')
  }
  return definition
}

function validateManifest(manifest, policy) {
  if (
    manifest?.format !== NEURON_TIER_FORMAT ||
    ![NEURON_TIER_LEGACY_VERSION, NEURON_TIER_VERSION].includes(manifest.version)
  ) {
    throw tierError('NEURON_MANIFEST_INCOMPATIBLE', 'Nöron SSD manifest sürümü desteklenmiyor.')
  }
  const { fingerprint, ...body } = manifest
  const fingerprintDomain = manifest.version === NEURON_TIER_LEGACY_VERSION
    ? 'bombom-neuron-tier-manifest-v1'
    : 'bombom-neuron-tier-manifest-v2'
  const expected = sha256(Buffer.from(`${fingerprintDomain}\0${canonicalJson(body)}`))
  if (fingerprint !== expected) throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD manifest özeti uyuşmuyor.')
  if (typeof manifest.modelFingerprint !== 'string' || !manifest.modelFingerprint) {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD manifest model fingerprinti eksik.')
  }
  if (manifest.version === NEURON_TIER_LEGACY_VERSION) {
    if (typeof manifest.representationSpaceId !== 'string' || !manifest.representationSpaceId) {
      throw tierError('NEURON_MANIFEST_CORRUPT', 'V1 nöron SSD manifest temsil alanı eksik.')
    }
  } else if (
    typeof manifest.physicalRepresentationSpaceId !== 'string' || !manifest.physicalRepresentationSpaceId ||
    typeof manifest.thoughtModelFingerprint !== 'string' || !manifest.thoughtModelFingerprint ||
    (Object.hasOwn(manifest, 'representationProfile') &&
      (typeof manifest.representationProfile !== 'string' || !manifest.representationProfile.trim() ||
        manifest.representationProfile.length > 128)) ||
    Object.hasOwn(manifest, 'representationSpaceId')
  ) {
    throw tierError(
      'NEURON_MANIFEST_CORRUPT',
      'V2 nöron SSD manifesti fiziksel temsil ve ThoughtModel kimliklerini açıkça ayırmalıdır.',
    )
  }
  const layout = normalizedLayout(manifest.layout)
  if (manifest.version === NEURON_TIER_LEGACY_VERSION && (
    Object.hasOwn(manifest, 'relationPageSketches') ||
    (Array.isArray(manifest.index) && manifest.index.some((entry) => Object.hasOwn(entry ?? {}, 'relationPageSketches')))
  )) {
    throw tierError(
      'NEURON_MANIFEST_INCOMPATIBLE',
      'V1 nöron manifestindeki temsil alanları fiziksel mikro-sayfa eskizi olarak yorumlanamaz.',
    )
  }
  const relationPageDefinition = manifest.version === NEURON_TIER_VERSION && manifest.relationPageSketches !== undefined
    ? validateRelationPageSketchDefinition(
        manifest.relationPageSketches,
        layout,
        manifest.physicalRepresentationSpaceId,
      )
    : null
  const sketch = manifest.routingSketch
  if (sketch?.format !== 'sign-projection-v1' || !Number.isSafeInteger(sketch.bitCount) ||
      sketch.bitCount < 1 || sketch.bitCount > ROUTING_SKETCH_BITS ||
      !Array.isArray(sketch.coordinates) || sketch.coordinates.length !== sketch.bitCount ||
      new Set(sketch.coordinates).size !== sketch.coordinates.length ||
      sketch.coordinates.some((coordinate) => !Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate >= layout.cellCount)) {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD yönlendirme eskizi geçersiz.')
  }
  if (!Array.isArray(manifest.pages) || !Array.isArray(manifest.index) || manifest.neuronCount !== manifest.index.length) {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD manifest sayımları geçersiz.')
  }
  const pages = new Map()
  for (const page of manifest.pages) {
    if (!/^[a-f0-9]{64}$/.test(page.contentHash) || pages.has(page.contentHash)) {
      throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD sayfa kimliği geçersiz veya yineleniyor.')
    }
    if (page.relativePath !== `pages/${page.contentHash}.nrpg`) {
      throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD sayfa yolu içerik adresiyle uyuşmuyor.')
    }
    positiveInteger(page.byteLength, 'page.byteLength', PAGE_HEADER_BYTES)
    if (page.byteLength > policy.maximumPageBytes) {
      throw tierError('NEURON_PAGE_LIMIT_EXCEEDED', 'Manifestteki SSD sayfası çalışma sınırını aşıyor.')
    }
    pages.set(page.contentHash, page)
  }
  const index = new Map()
  let previousId = null
  for (const entry of manifest.index) {
    if (typeof entry.id !== 'string' || !entry.id || index.has(entry.id) || (previousId !== null && previousId.localeCompare(entry.id) >= 0)) {
      throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD indeksi sıralı ve tekil değil.')
    }
    const page = pages.get(entry.pageHash)
    if (!page || !Number.isSafeInteger(entry.offset) || !Number.isSafeInteger(entry.length) ||
        entry.offset < PAGE_HEADER_BYTES || entry.length < RECORD_HEADER_BYTES || entry.offset + entry.length > page.byteLength ||
        !/^[a-f0-9]{64}$/.test(entry.recordHash) || typeof entry.identityDigest !== 'string' ||
        typeof entry.routingSketch !== 'string' ||
        !new RegExp(`^[a-f0-9]{${Math.ceil(sketch.bitCount / 8) * 2}}$`).test(entry.routingSketch) ||
        (relationPageDefinition
          ? !new RegExp(`^[a-f0-9]{${Math.ceil(relationPageDefinition.bitCountPerPage / 8) * 2 * relationPageDefinition.pageCount}}$`).test(entry.relationPageSketches)
          : Object.hasOwn(entry, 'relationPageSketches'))) {
      throw tierError('NEURON_MANIFEST_CORRUPT', `${entry.id} SSD indeks sınırları geçersiz.`)
    }
    index.set(entry.id, entry)
    previousId = entry.id
  }
  if (manifest.pageCount !== pages.size) throw tierError('NEURON_MANIFEST_CORRUPT', 'SSD sayfa sayısı uyuşmuyor.')
  if (manifest.pageBytes !== [...pages.values()].reduce((sum, page) => sum + page.byteLength, 0)) {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'SSD sayfa byte toplamı uyuşmuyor.')
  }
  const entriesByPage = new Map()
  for (const entry of index.values()) {
    const entries = entriesByPage.get(entry.pageHash) ?? []
    entries.push(entry)
    entriesByPage.set(entry.pageHash, entries)
  }
  for (const [pageHash, page] of pages) {
    const entries = (entriesByPage.get(pageHash) ?? []).sort((left, right) => left.offset - right.offset)
    let cursor = PAGE_HEADER_BYTES
    for (const entry of entries) {
      if (entry.offset !== cursor) throw tierError('NEURON_MANIFEST_CORRUPT', 'SSD sayfa kayıtları bitişik değil.')
      cursor += entry.length
    }
    if (entries.length !== page.count || cursor !== page.byteLength) {
      throw tierError('NEURON_MANIFEST_CORRUPT', 'SSD sayfa kayıt sayısı veya sınırı uyuşmuyor.')
    }
  }
  return { layout, pages, index }
}

function manifestSummary(manifest) {
  return Object.freeze({
    format: manifest.format,
    version: manifest.version,
    fingerprint: manifest.fingerprint,
    modelFingerprint: manifest.modelFingerprint,
    physicalRepresentationSpaceId: manifest.version === NEURON_TIER_VERSION
      ? manifest.physicalRepresentationSpaceId
      : null,
    thoughtModelFingerprint: manifest.version === NEURON_TIER_VERSION
      ? manifest.thoughtModelFingerprint
      : null,
    representationProfile: manifest.version === NEURON_TIER_VERSION
      ? manifest.representationProfile ?? null
      : null,
    legacyRepresentationSpaceId: manifest.version === NEURON_TIER_LEGACY_VERSION
      ? manifest.representationSpaceId
      : null,
    representationBinding: manifest.version === NEURON_TIER_VERSION
      ? 'explicit-v2'
      : 'legacy-v1-unverified',
    relationPageProfileBinding: manifest.version === NEURON_TIER_VERSION &&
      manifest.representationProfile === ADAPTIVE_RELATION_PAGE_PROFILE
      ? 'explicit-adaptive-profile-v1'
      : 'unverified-or-not-adaptive',
    layout: Object.freeze({ ...manifest.layout }),
    floatEncoding: manifest.floatEncoding,
    routingSketch: Object.freeze({
      ...manifest.routingSketch,
      coordinates: Object.freeze([...manifest.routingSketch.coordinates]),
    }),
    relationPageSketches: manifest.version === NEURON_TIER_VERSION &&
      manifest.representationProfile === ADAPTIVE_RELATION_PAGE_PROFILE &&
      manifest.relationPageSketches
      ? Object.freeze({
          ...manifest.relationPageSketches,
          coordinatesWithinPage: Object.freeze([...manifest.relationPageSketches.coordinatesWithinPage]),
        })
      : null,
    policyFingerprint: manifest.policyFingerprint,
    neuronCount: manifest.neuronCount,
    pageCount: manifest.pageCount,
    pageBytes: manifest.pageBytes,
  })
}

function effectiveUtility(profile, logicalClock, policy) {
  if (!profile) return 0
  const idle = Math.max(0, logicalClock - profile.lastAccess)
  const divisor = 1 + Math.floor(idle / policy.idleHalfLifeAccesses)
  return Math.floor(profile.utilityFixed / divisor)
}

function cacheVictim(left, right, profiles, logicalClock, policy) {
  const leftUtility = effectiveUtility(profiles.get(left.id), logicalClock, policy)
  const rightUtility = effectiveUtility(profiles.get(right.id), logicalClock, policy)
  return leftUtility - rightUtility || left.lastAccess - right.lastAccess || left.id.localeCompare(right.id)
}

export class NeuronTierRepository {
  #baseDirectory
  #manifest
  #layout
  #pages
  #index
  #routingIndex
  #policy
  #identityDigestOf
  #hot = new Map()
  #warm = new Map()
  #profiles = new Map()
  #pins = new Map()
  #logicalClock = 0
  #operationTail = Promise.resolve()
  #closed = false
  #counters = {
    hotHits: 0,
    warmHits: 0,
    coldReads: 0,
    fullPageReads: 0,
    bytesRead: 0,
    hotPromotions: 0,
    hotDemotions: 0,
    warmPromotions: 0,
    warmDemotions: 0,
  }

  constructor({ baseDirectory, manifest, policy, identityDigestOf, restoredPolicyState }) {
    this.#baseDirectory = baseDirectory
    this.#policy = policy
    this.#identityDigestOf = identityDigestOf
    const validated = validateManifest(manifest, policy)
    // Do not retain the large index array twice. Maps below keep the same entry
    // objects while the public header remains payload-free.
    this.#manifest = manifestSummary(manifest)
    this.#layout = validated.layout
    this.#pages = validated.pages
    this.#index = validated.index
    this.#routingIndex = new NeuronRoutingSketchIndex(this.#index)
    if (restoredPolicyState) this.#restorePolicyState(restoredPolicyState)
  }

  get manifest() { return structuredClone(this.#manifest) }
  get policy() { return structuredClone(this.#policy) }
  get size() { return this.#index.size }
  has(id) { return this.#index.has(id) }

  describe(id) {
    const entry = this.#index.get(id)
    if (!entry) return undefined
    const profile = this.#profiles.get(id)
    return {
      id,
      identityDigest: entry.identityDigest,
      pageHash: entry.pageHash,
      recordBytes: entry.length,
      tier: this.tierOf(id),
      utilityFixed: effectiveUtility(profile, this.#logicalClock, this.#policy),
      accessCount: profile?.hits ?? 0,
    }
  }

  /**
   * O(N) over compact 128-bit sketches, never over dense I/R vectors. The
   * returned order is deterministic and can seed a bounded SSD working set.
   */
  rankBySignal(signal, maximum = this.#policy.maximumWorkingSetNeurons) {
    if (!signal || typeof signal.length !== 'number' || signal.length !== this.#layout.cellCount) {
      throw new TypeError(`Sinyal ${this.#layout.cellCount} hücreli olmalıdır.`)
    }
    for (let index = 0; index < signal.length; index += 1) {
      if (!Number.isFinite(signal[index])) throw new RangeError('Sinyal yalnız sonlu değerler içermelidir.')
    }
    const limit = Math.min(
      this.#policy.maximumWorkingSetNeurons,
      positiveInteger(maximum, 'maximum'),
    )
    const queryHex = routingSketch(signal, this.#manifest.routingSketch.coordinates)
    return rankNeuronRoutingSketches(
      this.#index.values(),
      queryHex,
      this.#manifest.routingSketch.bitCount,
      limit,
    )
  }

  /**
   * Production tick gate. Unlike rankBySignal (an offline exact diagnostic),
   * this never walks all manifest entries when N exceeds the working-set cap.
   */
  queryBySignal(signal, maximum = this.#policy.maximumWorkingSetNeurons) {
    if (!signal || typeof signal.length !== 'number' || signal.length !== this.#layout.cellCount) {
      throw new TypeError(`Sinyal ${this.#layout.cellCount} hücreli olmalıdır.`)
    }
    for (let index = 0; index < signal.length; index += 1) {
      if (!Number.isFinite(signal[index])) throw new RangeError('Sinyal yalnız sonlu değerler içermelidir.')
    }
    const limit = Math.min(
      this.#policy.maximumWorkingSetNeurons,
      positiveInteger(maximum, 'maximum'),
    )
    const queryHex = routingSketch(signal, this.#manifest.routingSketch.coordinates)
    return this.#routingIndex.query(queryHex, this.#manifest.routingSketch.bitCount, limit)
  }

  /**
   * Bounded region-aware seam for the fixed 512/3584 physical layout. The
   * global LSH gate still caps inspected candidates; selected 8x8 relation
   * pages only rerank that compact set and never read dense SSD records.
   */
  queryByRelationPages(signal, pageIndexes, maximum = this.#policy.maximumWorkingSetNeurons) {
    const definition = this.#manifest.relationPageSketches
    if (!definition) {
      throw tierError(
        'NEURON_RELATION_PAGE_SKETCH_UNAVAILABLE',
        'Bu manifest doğrulanmış 512/3584 ilişki mikro-sayfası eskizi taşımıyor.',
      )
    }
    if (!signal || typeof signal.length !== 'number' || signal.length !== this.#layout.cellCount) {
      throw new TypeError(`Sinyal ${this.#layout.cellCount} hücreli olmalıdır.`)
    }
    for (let index = 0; index < signal.length; index += 1) {
      if (!Number.isFinite(signal[index])) throw new RangeError('Sinyal yalnız sonlu değerler içermelidir.')
    }
    const selectedPages = normalizeRelationPageIndexes(pageIndexes, definition.pageCount)
    const limit = Math.min(
      this.#policy.maximumWorkingSetNeurons,
      positiveInteger(maximum, 'maximum'),
    )
    const queryHex = routingSketch(signal, this.#manifest.routingSketch.coordinates)
    const global = this.#routingIndex.query(
      queryHex,
      this.#manifest.routingSketch.bitCount,
      this.#policy.maximumWorkingSetNeurons,
    )
    const pageQueries = new Map(selectedPages.map((pageIndex) => [
      pageIndex,
      signalRelationPageSketch(signal, this.#layout, definition, pageIndex),
    ]))
    const candidates = global.candidates.map((candidate) => {
      const entry = this.#index.get(candidate.id)
      const relationPageDistance = selectedPages.reduce((sum, pageIndex) => sum + hammingDistance(
        relationPageSketchAt(entry.relationPageSketches, definition, pageIndex),
        pageQueries.get(pageIndex),
        definition.bitCountPerPage,
      ), 0) / selectedPages.length
      return {
        id: candidate.id,
        distance: (candidate.distance + relationPageDistance) / 2,
        globalDistance: candidate.distance,
        relationPageDistance,
      }
    }).sort(nearestOrder).slice(0, limit)
    return {
      candidates,
      diagnostics: Object.freeze({
        ...global.diagnostics,
        method: 'lsh-multiprobe-plus-relation-pages-v1',
        globalMethod: global.diagnostics.method,
        selectedPageIndexes: Object.freeze(selectedPages),
        rerankedCandidates: global.candidates.length,
        maximumSelectedPages: MAX_RELATION_PAGE_QUERY_COUNT,
        denseRecordReads: 0,
      }),
    }
  }

  sampleIds(seed, maximum) {
    return this.#routingIndex.sampleIds(seed, Math.min(
      this.#policy.maximumWorkingSetNeurons,
      positiveInteger(maximum, 'maximum'),
    ))
  }

  tierOf(id) {
    if (!this.#index.has(id)) return 'missing'
    if (this.#hot.has(id)) return 'hot'
    if (this.#warm.has(id)) return 'warm'
    return 'cold'
  }

  getStats() {
    const hotBytes = [...this.#hot.values()].reduce((sum, entry) => sum + entry.bytes, 0)
    const warmBytes = [...this.#warm.values()].reduce((sum, entry) => sum + entry.bytes.byteLength, 0)
    const residentIds = new Set([...this.#hot.keys(), ...this.#warm.keys()])
    return {
      manifestFingerprint: this.#manifest.fingerprint,
      logicalClock: this.#logicalClock,
      neuronCount: this.#index.size,
      tiers: {
        hotNeurons: this.#hot.size,
        warmNeurons: this.#warm.size,
        coldNeurons: Math.max(0, this.#index.size - residentIds.size),
      },
      resident: { hotBytes, warmBytes, totalBytes: hotBytes + warmBytes },
      limits: { hotBytes: this.#policy.hotBudgetBytes, warmBytes: this.#policy.warmBudgetBytes },
      io: { ...this.#counters },
    }
  }

  get(id, evidence) {
    return this.#exclusive(() => this.#getInternal(id, evidence))
  }

  observe(id, evidence) {
    return this.#exclusive(async () => {
      if (!this.#index.has(id)) return false
      this.#logicalClock += 1
      this.#updateProfile(id, evidence)
      this.#rebalance()
      return true
    })
  }

  advanceLogicalClock(steps = 1) {
    return this.#exclusive(async () => {
      this.#logicalClock += positiveInteger(steps, 'steps')
      this.#rebalance()
      return this.#logicalClock
    })
  }

  materializeWorkingSet(requests, options = {}) {
    return this.#exclusive(() => this.#materialize(requests, options))
  }

  verifyPageForNeuron(id) {
    return this.#exclusive(async () => {
      const entry = this.#index.get(id)
      if (!entry) return false
      await this.#readPage(entry.pageHash)
      return true
    })
  }

  withWorkingSet(requests, callback, options = {}) {
    if (typeof callback !== 'function') throw new TypeError('withWorkingSet callback gerektirir.')
    return (async () => {
      const workingSet = await this.#exclusive(async () => {
        const result = await this.#materialize(requests, options)
        for (const id of result.keys()) this.#pins.set(id, (this.#pins.get(id) ?? 0) + 1)
        return result
      })
      try {
        return await callback(workingSet)
      } finally {
        if (!this.#closed) await this.#exclusive(async () => {
          for (const id of workingSet.keys()) {
            const count = (this.#pins.get(id) ?? 1) - 1
            if (count > 0) this.#pins.set(id, count)
            else this.#pins.delete(id)
          }
          this.#rebalance()
        })
      }
    })()
  }

  exportPolicyState() {
    const body = {
      format: 'bombom-neuron-tier-policy-state',
      version: 1,
      manifestFingerprint: this.#manifest.fingerprint,
      policyFingerprint: sha256(Buffer.from(canonicalJson(this.#policy))),
      logicalClock: this.#logicalClock,
      profiles: [...this.#profiles].sort(([left], [right]) => left.localeCompare(right)).map(([id, profile]) => ({ id, ...profile })),
    }
    return { ...body, fingerprint: sha256(Buffer.from(`bombom-neuron-tier-policy-state-v1\0${canonicalJson(body)}`)) }
  }

  close() {
    this.#closed = true
    this.#hot.clear()
    this.#warm.clear()
    this.#profiles.clear()
    this.#pins.clear()
  }

  #exclusive(operation) {
    if (this.#closed) return Promise.reject(tierError('NEURON_TIER_CLOSED', 'Nöron tier deposu kapalı.'))
    const result = this.#operationTail.then(operation)
    this.#operationTail = result.catch(() => {})
    return result
  }

  #restorePolicyState(state) {
    if (state?.format !== 'bombom-neuron-tier-policy-state' || state.version !== 1) {
      throw tierError('NEURON_POLICY_STATE_INCOMPATIBLE', 'Nöron tier politika durumu uyumsuz.')
    }
    const { fingerprint, ...body } = state
    if (fingerprint !== sha256(Buffer.from(`bombom-neuron-tier-policy-state-v1\0${canonicalJson(body)}`)) ||
        state.manifestFingerprint !== this.#manifest.fingerprint ||
        state.policyFingerprint !== sha256(Buffer.from(canonicalJson(this.#policy)))) {
      throw tierError('NEURON_POLICY_STATE_MISMATCH', 'Nöron tier politika durumu farklı manifest veya politikaya ait.')
    }
    this.#logicalClock = Math.max(0, Math.trunc(state.logicalClock))
    for (const item of state.profiles ?? []) {
      if (this.#index.has(item.id)) this.#profiles.set(item.id, {
        utilityFixed: Math.max(0, Math.min(NEURON_UTILITY_SCALE, Math.trunc(item.utilityFixed))),
        hits: Math.max(0, Math.trunc(item.hits)),
        lastAccess: Math.max(0, Math.min(this.#logicalClock, Math.trunc(item.lastAccess))),
      })
    }
  }

  #updateProfile(id, evidence = {}) {
    const instant = scoreNeuronUtilityWithPolicy(evidence, this.#policy).fixed
    const previous = this.#profiles.get(id)
    const utilityFixed = previous
      ? Math.round((previous.utilityFixed * (this.#policy.utilityEmaDenominator - this.#policy.utilityEmaNumerator) +
          instant * this.#policy.utilityEmaNumerator) / this.#policy.utilityEmaDenominator)
      : instant
    const profile = { utilityFixed, hits: (previous?.hits ?? 0) + 1, lastAccess: this.#logicalClock }
    this.#profiles.set(id, profile)
    return profile
  }

  async #getInternal(id, evidence = {}) {
    const entry = this.#index.get(id)
    if (!entry) return undefined
    this.#logicalClock += 1
    const profile = this.#updateProfile(id, evidence)
    let neuron
    const hot = this.#hot.get(id)
    if (hot) {
      hot.lastAccess = this.#logicalClock
      this.#counters.hotHits += 1
      neuron = hot.neuron
    } else {
      const utility = effectiveUtility(profile, this.#logicalClock, this.#policy)
      const wantsWarm = utility >= Math.round(this.#policy.warmUtilityThreshold * NEURON_UTILITY_SCALE)
      let record
      const warm = this.#warm.get(id)
      if (warm) {
        warm.lastAccess = this.#logicalClock
        this.#counters.warmHits += 1
        record = warm.bytes
      } else {
        record = await this.#readRecord(entry)
        if (wantsWarm) {
          this.#warm.set(id, { id, bytes: Buffer.from(record), lastAccess: this.#logicalClock })
          this.#counters.warmPromotions += 1
        }
      }
      neuron = decodeNeuronRecord(record, entry, this.#layout, this.#identityDigestOf)
      if (utility >= Math.round(this.#policy.hotUtilityThreshold * NEURON_UTILITY_SCALE) &&
          profile.hits >= this.#policy.hotPromotionHits) {
        this.#hot.set(id, { id, neuron, bytes: entry.length, lastAccess: this.#logicalClock })
        this.#counters.hotPromotions += 1
      }
    }
    const result = cloneNeuron(neuron)
    this.#rebalance()
    return result
  }

  async #materialize(requests, options) {
    if (!Array.isArray(requests)) throw new TypeError('Çalışma kümesi istekleri dizi olmalıdır.')
    const maximum = Math.min(
      this.#policy.maximumWorkingSetNeurons,
      positiveInteger(options.maximum ?? this.#policy.maximumWorkingSetNeurons, 'maximum'),
    )
    const unique = []
    const seen = new Set()
    for (const request of requests) {
      const id = typeof request === 'string' ? request : String(request?.id ?? '')
      if (!id || seen.has(id) || !this.#index.has(id)) continue
      if (unique.length >= maximum) break
      seen.add(id)
      unique.push({ id, evidence: typeof request === 'string' ? {} : request.evidence ?? {} })
    }
    const result = new Map()
    for (const request of unique) {
      const neuron = await this.#getInternal(request.id, request.evidence)
      if (neuron) result.set(request.id, neuron)
    }
    return result
  }

  async #readRecord(entry) {
    const page = this.#pages.get(entry.pageHash)
    const filePath = path.join(this.#baseDirectory, page.relativePath)
    const handle = await open(filePath, 'r')
    try {
      const buffer = Buffer.allocUnsafe(entry.length)
      const { bytesRead } = await handle.read(buffer, 0, entry.length, entry.offset)
      if (bytesRead !== entry.length) throw tierError('NEURON_RECORD_CORRUPT', `${entry.id} SSD kaydı yarım okundu.`)
      this.#counters.coldReads += 1
      this.#counters.bytesRead += bytesRead
      return buffer
    } finally {
      await handle.close()
    }
  }

  async #readPage(pageHash) {
    const page = this.#pages.get(pageHash)
    const bytes = await readFile(path.join(this.#baseDirectory, page.relativePath))
    if (bytes.byteLength !== page.byteLength || sha256(bytes) !== page.contentHash ||
        !bytes.subarray(0, PAGE_MAGIC.byteLength).equals(PAGE_MAGIC) ||
        bytes.readUInt32LE(PAGE_MAGIC.byteLength) !== page.count) {
      throw tierError('NEURON_PAGE_CORRUPT', `Nöron SSD sayfası doğrulanamadı: ${pageHash}`)
    }
    this.#counters.fullPageReads += 1
    this.#counters.bytesRead += bytes.byteLength
    return bytes
  }

  #rebalance() {
    const hotThreshold = Math.round(this.#policy.hotUtilityThreshold * NEURON_UTILITY_SCALE)
    for (const [id] of this.#hot) {
      if ((this.#pins.get(id) ?? 0) === 0 && effectiveUtility(this.#profiles.get(id), this.#logicalClock, this.#policy) < hotThreshold) {
        this.#hot.delete(id)
        this.#counters.hotDemotions += 1
      }
    }
    let hotBytes = [...this.#hot.values()].reduce((sum, entry) => sum + entry.bytes, 0)
    while (hotBytes > this.#policy.hotBudgetBytes) {
      const candidates = [...this.#hot.values()].filter((entry) => (this.#pins.get(entry.id) ?? 0) === 0)
      if (!candidates.length) break
      candidates.sort((left, right) => cacheVictim(left, right, this.#profiles, this.#logicalClock, this.#policy))
      const victim = candidates[0]
      this.#hot.delete(victim.id)
      hotBytes -= victim.bytes
      this.#counters.hotDemotions += 1
    }

    const warmThreshold = Math.round(this.#policy.warmUtilityThreshold * NEURON_UTILITY_SCALE)
    for (const [id] of this.#warm) {
      if ((this.#pins.get(id) ?? 0) === 0 && effectiveUtility(this.#profiles.get(id), this.#logicalClock, this.#policy) < warmThreshold) {
        this.#warm.delete(id)
        this.#counters.warmDemotions += 1
      }
    }
    let warmBytes = [...this.#warm.values()].reduce((sum, entry) => sum + entry.bytes.byteLength, 0)
    while (warmBytes > this.#policy.warmBudgetBytes) {
      const candidates = [...this.#warm.values()].filter((entry) => (this.#pins.get(entry.id) ?? 0) === 0)
      if (!candidates.length) break
      candidates.sort((left, right) => cacheVictim(left, right, this.#profiles, this.#logicalClock, this.#policy))
      const victim = candidates[0]
      this.#warm.delete(victim.id)
      warmBytes -= victim.bytes.byteLength
      this.#counters.warmDemotions += 1
    }
  }
}

export async function openNeuronTierRepository({
  directory,
  manifestPath,
  manifestFingerprint,
  policy: policyInput,
  identityDigestOf,
  restoredPolicyState,
}) {
  if (typeof directory !== 'string' || !directory.trim()) throw new TypeError('Nöron tier dizini gereklidir.')
  const baseDirectory = path.resolve(directory)
  const resolvedManifestPath = manifestPath
    ? path.resolve(manifestPath)
    : path.join(baseDirectory, 'manifests', `${manifestFingerprint}.json`)
  const relative = path.relative(baseDirectory, resolvedManifestPath)
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw tierError('NEURON_MANIFEST_PATH_REJECTED', 'Nöron manifesti tier dizininin dışında olamaz.')
  }
  const manifestBytes = await readFile(resolvedManifestPath)
  let manifest
  try {
    manifest = JSON.parse(manifestBytes.toString('utf8'))
  } catch {
    throw tierError('NEURON_MANIFEST_CORRUPT', 'Nöron SSD manifesti JSON olarak okunamadı.')
  }
  if (manifestFingerprint && manifest.fingerprint !== manifestFingerprint) {
    throw tierError('NEURON_MANIFEST_MISMATCH', 'İstenen nöron manifest fingerprinti dosyayla uyuşmuyor.')
  }
  const repository = new NeuronTierRepository({
    baseDirectory,
    manifest,
    policy: normalizeNeuronTierPolicy(policyInput),
    identityDigestOf,
    restoredPolicyState,
  })
  // The manifest can be valid while a page was externally removed. Stat only
  // pages here: O(page-count) startup validation, never an O(neuron-count) read.
  for (const page of manifest.pages) {
    const info = await stat(path.join(baseDirectory, page.relativePath)).catch(() => null)
    if (!info?.isFile() || info.size !== page.byteLength) {
      throw tierError('NEURON_PAGE_MISSING', `Nöron SSD sayfası eksik: ${page.contentHash}`)
    }
  }
  return repository
}
