import { THOUGHT_MATRIX_CELLS, type ThoughtReplayFrame } from './types'

const HASH_PRIMES = [0x01000193, 0x27d4eb2d, 0x165667b1, 0x9e3779b1] as const
const HASH_SEEDS = [0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35] as const

function canonical(value: unknown): unknown {
  if (value instanceof Float32Array) {
    return ['Float32Array', ...Array.from(value, (entry) => Math.fround(entry))]
  }
  if (Array.isArray(value)) return value.map(canonical)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonical(entry)]))
  }
  if (typeof value === 'number') return Number.isFinite(value) ? Math.fround(value) : String(value)
  return value
}

export function thoughtDigest(label: string, ...parts: readonly unknown[]): string {
  const text = JSON.stringify(canonical(['thought-runtime-v1', label, ...parts]))
  const states: number[] = [...HASH_SEEDS]
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index)
    for (let stateIndex = 0; stateIndex < states.length; stateIndex += 1) {
      states[stateIndex] = Math.imul(
        (states[stateIndex] ^ code ^ Math.imul(index + 1, stateIndex + 19)) >>> 0,
        HASH_PRIMES[stateIndex],
      ) >>> 0
      states[stateIndex] ^= states[stateIndex] >>> 13
    }
  }
  return states.map((state) => (state >>> 0).toString(16).padStart(8, '0')).join('')
}

/** Digest of the exact Float32 values used by the engine. */
export function thoughtVectorDigest(vector: Float32Array): string {
  if (vector.length !== THOUGHT_MATRIX_CELLS) {
    throw new RangeError(`Düşünce vektörü ${THOUGHT_MATRIX_CELLS} hücre içermelidir.`)
  }
  return thoughtDigest('vector', vector)
}

/** Digest for an I or R partition whose size is fixed by the model layout. */
export function thoughtPartitionDigest(label: 'identity' | 'relation', vector: Float32Array): string {
  if (!(vector instanceof Float32Array) || vector.length < 1 || !vector.every(Number.isFinite)) {
    throw new RangeError(`${label} bölümü boş olmayan sonlu bir Float32Array olmalıdır.`)
  }
  return thoughtDigest(label, vector)
}

export function seededThoughtUnit(seed: number, ...parts: readonly unknown[]): number {
  const digest = thoughtDigest('seeded-unit', seed | 0, ...parts)
  return Number.parseInt(digest.slice(0, 8), 16) / 0x1_0000_0000
}

export function replayFingerprint(frames: readonly ThoughtReplayFrame[]): string {
  const chained = frames.at(-1)?.chainFingerprint
  if (chained) return chained
  if (!frames.length) return thoughtDigest('replay-empty')
  return thoughtDigest('replay-legacy', frames.map((frame) => ({
    ...frame,
    chainFingerprint: undefined,
    retainedMasses: frame.retainedMasses.map(Math.fround),
  })))
}

export function deterministicWeightedSample<T>(
  values: readonly T[],
  count: number,
  seed: number,
  key: (value: T) => string,
  weight: (value: T) => number,
  salt: readonly unknown[] = [],
): T[] {
  return [...values]
    .map((value) => {
      const stableKey = key(value)
      const boundedWeight = Math.max(Number.MIN_VALUE, weight(value))
      const unit = Math.max(Number.MIN_VALUE, seededThoughtUnit(seed, ...salt, stableKey))
      return { value, stableKey, priority: -Math.log(unit) / boundedWeight }
    })
    .sort((left, right) => left.priority - right.priority || left.stableKey.localeCompare(right.stableKey))
    .slice(0, Math.max(0, count))
    .map((entry) => entry.value)
}
