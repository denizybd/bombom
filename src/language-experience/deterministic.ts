import { stableDigest } from '../families/sparse.ts'

export const LANGUAGE_EPSILON = 1e-12

export function clamp(value: number, minimum = 0, maximum = 1): number {
  if (!Number.isFinite(value)) return minimum
  return Math.min(maximum, Math.max(minimum, value))
}

export function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

export function canonicalizeLanguageText(value: string): string {
  return value.replace(/\r\n?/g, '\n').normalize('NFC')
}

/**
 * This representation is intentionally surface-only. It is used for duplicate
 * protection and diagnostics, never as a replacement for a neuron's frozen I.
 */
export function normalizeSurface(value: string): string {
  return canonicalizeLanguageText(value)
    .toLocaleLowerCase('tr-TR')
    .replace(/[\p{Z}\s]+/gu, ' ')
    .trim()
}

export function languageDigest(label: string, ...parts: readonly unknown[]): string {
  return `${label}-${stableDigest(['language-experience-v1', label, ...parts]).toUpperCase()}`
}

export function seededUnit(seed: number, ...parts: readonly unknown[]): number {
  const digest = stableDigest(['seeded-unit-v1', seed | 0, ...parts])
  return Number.parseInt(digest.slice(0, 8), 16) / 0x1_0000_0000
}

export function deterministicOrder<T>(
  values: readonly T[],
  seed: number,
  key: (value: T) => string,
): T[] {
  return values
    .map((value, index) => {
      const stableKey = key(value)
      return { value, index, stableKey, rank: seededUnit(seed, stableKey) }
    })
    .sort((left, right) =>
      left.rank - right.rank ||
      compareText(left.stableKey, right.stableKey) ||
      left.index - right.index,
    )
    .map((entry) => entry.value)
}

/**
 * Returns exactly the first value that deterministicOrder would produce after
 * filtering, without allocating or sorting the complete eligible candidate set.
 */
export function deterministicFirst<T>(
  values: readonly T[],
  seed: number,
  key: (value: T) => string,
  eligible: (value: T) => boolean = () => true,
): T | undefined {
  let selected: { value: T; stableKey: string; rank: number } | undefined
  for (const value of values) {
    if (!eligible(value)) continue
    const stableKey = key(value)
    const rank = seededUnit(seed, stableKey)
    if (
      !selected ||
      rank < selected.rank ||
      (rank === selected.rank && compareText(stableKey, selected.stableKey) < 0)
    ) selected = { value, stableKey, rank }
  }
  return selected?.value
}

export function normalizePositive<T extends { score: number }>(
  values: readonly T[],
): Array<T & { probability: number }> {
  const total = values.reduce((sum, value) => sum + Math.max(0, value.score), 0)
  if (total <= LANGUAGE_EPSILON) {
    const probability = values.length ? 1 / values.length : 0
    return values.map((value) => ({ ...value, probability }))
  }
  return values.map((value) => ({
    ...value,
    probability: Math.max(0, value.score) / total,
  }))
}

export function softmax<T extends { score: number }>(
  values: readonly T[],
  temperature: number,
): Array<T & { probability: number }> {
  if (!values.length) return []
  const safeTemperature = Math.max(1e-3, temperature)
  const maximum = Math.max(...values.map((value) => value.score / safeTemperature))
  const exponentials = values.map((value) => Math.exp(value.score / safeTemperature - maximum))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return values.map((value, index) => ({
    ...value,
    probability: total > LANGUAGE_EPSILON ? exponentials[index] / total : 0,
  }))
}

export function boundedLogOdds(excitatory: number, inhibitory: number): number {
  const positive = Math.max(0, excitatory)
  const negative = Math.max(0, inhibitory)
  return Math.max(-12, Math.min(12, Math.log((positive + 0.25) / (negative + 0.25))))
}

export function deepSerializableClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}
