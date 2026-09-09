const HASH_SEEDS = [0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35] as const
const HASH_PRIMES = [0x01000193, 0x27d4eb2d, 0x165667b1, 0x9e3779b1] as const

export function compareRegionText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

export function canonicalRegionFloat(value: number): number {
  if (!Number.isFinite(value)) throw new TypeError('Uyarlanabilir bölge değerleri sonlu olmalıdır.')
  return Object.is(value, -0) ? 0 : Math.fround(value)
}

function canonicalValue(value: unknown): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('Uyarlanabilir bölge değerleri sonlu olmalıdır.')
    if (Number.isSafeInteger(value)) return Object.is(value, -0) ? 0 : value
    return canonicalRegionFloat(value)
  }
  if (value instanceof Float32Array) return Array.from(value, canonicalRegionFloat)
  if (Array.isArray(value)) return value.map(canonicalValue)
  if (typeof value === 'object') {
    const result: Record<string, unknown> = {}
    for (const key of Object.keys(value as Record<string, unknown>).sort(compareRegionText)) {
      const child = (value as Record<string, unknown>)[key]
      if (child !== undefined) result[key] = canonicalValue(child)
    }
    return result
  }
  if (value === undefined) return null
  throw new TypeError('Uyarlanabilir bölge özeti yalnız JSON-uyumlu değerler kabul eder.')
}

export function canonicalRegionJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value))
}

export function adaptiveRegionDigest(label: string, ...parts: readonly unknown[]): string {
  const states: number[] = [...HASH_SEEDS]
  const text = canonicalRegionJson(['adaptive-regions-v1', label, ...parts])
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index)
    for (let stateIndex = 0; stateIndex < states.length; stateIndex += 1) {
      states[stateIndex] = Math.imul(
        (states[stateIndex] ^ code ^ Math.imul(index + 1, stateIndex + 17)) >>> 0,
        HASH_PRIMES[stateIndex],
      ) >>> 0
    }
  }
  return `${label}-${states.map((value) => value.toString(16).padStart(8, '0')).join('').toUpperCase()}`
}

export function immutableRegionValue<T>(value: T): Readonly<T> {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value as Record<string, unknown>)) immutableRegionValue(child)
    Object.freeze(value)
  }
  return value
}
