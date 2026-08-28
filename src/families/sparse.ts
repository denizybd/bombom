import type { SparseCoordinate, SparseVector } from './types'

export const FAMILY_EPSILON = 1e-12

export function clamp(value: number, minimum = 0, maximum = 1): number {
  if (!Number.isFinite(value)) return minimum
  return Math.min(maximum, Math.max(minimum, value))
}

export function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

export function hashString(value: string, seed = 0x811c9dc5): number {
  let hash = seed >>> 0
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193) >>> 0
    hash ^= hash >>> 13
  }
  return hash >>> 0
}

export function stableDigest(parts: readonly unknown[]): string {
  const states = [0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35]
  const payload = parts.map((part) => {
    if (typeof part === 'number') return Number.isFinite(part) ? Math.fround(part) : String(part)
    if (part instanceof Float32Array) return Array.from(part, Math.fround)
    return part
  })
  const text = JSON.stringify(payload)
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index)
    for (let stateIndex = 0; stateIndex < states.length; stateIndex += 1) {
      states[stateIndex] = Math.imul(
        (states[stateIndex] ^ code ^ Math.imul(index + 1, stateIndex + 17)) >>> 0,
        [0x01000193, 0x27d4eb2d, 0x165667b1, 0x9e3779b1][stateIndex],
      ) >>> 0
    }
  }
  return states.map((value) => value.toString(16).padStart(8, '0')).join('')
}

export function sparseNorm(entries: readonly SparseCoordinate[]): number {
  let sum = 0
  for (const entry of entries) sum += entry.value * entry.value
  return Math.sqrt(sum)
}

export function createSparseVector(
  dimension: number,
  values: ReadonlyMap<number, number> | readonly SparseCoordinate[],
  maximumEntries = Number.POSITIVE_INFINITY,
): SparseVector {
  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new RangeError('Seyrek vektör boyutu pozitif bir tam sayı olmalıdır.')
  }
  const merged = new Map<number, number>()
  let source: readonly SparseCoordinate[]
  if (Array.isArray(values)) source = values
  else {
    const mappedValues = values as ReadonlyMap<number, number>
    source = Array.from(mappedValues, ([index, value]) => ({ index, value }))
  }
  for (const entry of source) {
    if (
      !Number.isInteger(entry.index) ||
      entry.index < 0 ||
      entry.index >= dimension ||
      !Number.isFinite(entry.value)
    ) {
      throw new RangeError('Seyrek vektör sonlu ve geçerli koordinatlar içermelidir.')
    }
    const next = (merged.get(entry.index) ?? 0) + entry.value
    if (Math.abs(next) <= FAMILY_EPSILON) merged.delete(entry.index)
    else merged.set(entry.index, next)
  }
  let entries = [...merged.entries()].map(([index, value]) => ({ index, value }))
  entries.sort((left, right) => Math.abs(right.value) - Math.abs(left.value) || left.index - right.index)
  if (entries.length > maximumEntries) entries.length = maximumEntries
  entries.sort((left, right) => left.index - right.index)
  const norm = sparseNorm(entries)
  if (norm > FAMILY_EPSILON) {
    entries = entries.map((entry) => ({ index: entry.index, value: entry.value / norm }))
  }
  return { dimension, entries, norm: entries.length ? 1 : 0 }
}

export function validateSparseVector(vector: SparseVector): void {
  if (!Number.isInteger(vector.dimension) || vector.dimension < 1 || !Array.isArray(vector.entries)) {
    throw new TypeError('Davranış profili geçerli bir seyrek vektör değil.')
  }
  let previous = -1
  let squared = 0
  for (const entry of vector.entries) {
    if (
      !Number.isInteger(entry.index) ||
      entry.index <= previous ||
      entry.index >= vector.dimension ||
      !Number.isFinite(entry.value)
    ) throw new TypeError('Seyrek vektör koordinatları sıralı, benzersiz ve sonlu olmalıdır.')
    previous = entry.index
    squared += entry.value * entry.value
  }
  const expected = Math.sqrt(squared)
  if (!Number.isFinite(vector.norm) || Math.abs(vector.norm - expected) > 1e-5) {
    throw new TypeError('Seyrek vektör normu koordinatlarla uyuşmuyor.')
  }
}

export function sparseCosine(left: SparseVector, right: SparseVector): number {
  if (left.dimension !== right.dimension) return 0
  if (left.norm <= FAMILY_EPSILON || right.norm <= FAMILY_EPSILON) return 0
  let leftIndex = 0
  let rightIndex = 0
  let dot = 0
  while (leftIndex < left.entries.length && rightIndex < right.entries.length) {
    const a = left.entries[leftIndex]
    const b = right.entries[rightIndex]
    if (a.index === b.index) {
      dot += a.value * b.value
      leftIndex += 1
      rightIndex += 1
    } else if (a.index < b.index) leftIndex += 1
    else rightIndex += 1
  }
  return clamp(dot / (left.norm * right.norm), -1, 1)
}

export function weightedSparseMean(
  items: readonly { vector: SparseVector; weight: number }[],
  maximumEntries: number,
): SparseVector {
  const first = items.find((item) => item.weight > 0)
  if (!first) throw new RangeError('Seyrek ortalama için pozitif ağırlıklı bir profil gerekir.')
  const accumulator = new Map<number, number>()
  let total = 0
  for (const item of items) {
    if (item.vector.dimension !== first.vector.dimension) {
      throw new RangeError('Farklı davranış uzayları ortalanamaz.')
    }
    if (!Number.isFinite(item.weight) || item.weight <= 0) continue
    total += item.weight
    for (const entry of item.vector.entries) {
      accumulator.set(entry.index, (accumulator.get(entry.index) ?? 0) + entry.value * item.weight)
    }
  }
  if (total <= FAMILY_EPSILON) throw new RangeError('Seyrek ortalama ağırlığı sıfır olamaz.')
  for (const [index, value] of accumulator) accumulator.set(index, value / total)
  return createSparseVector(first.vector.dimension, accumulator, maximumEntries)
}

export function sparseVectorDigest(vector: SparseVector): string {
  return stableDigest([
    'sparse-vector-v1',
    vector.dimension,
    ...vector.entries.flatMap((entry) => [entry.index, Math.fround(entry.value)]),
  ])
}

export function normalizedPositive<T extends { score: number }>(
  values: readonly T[],
): Array<T & { weight: number }> {
  const total = values.reduce((sum, value) => sum + Math.max(0, value.score), 0)
  if (total <= FAMILY_EPSILON) return values.map((value) => ({ ...value, weight: 0 }))
  return values.map((value) => ({ ...value, weight: Math.max(0, value.score) / total }))
}
