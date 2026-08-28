const UINT32_MAX_PLUS_ONE = 4294967296

/** Stable UTF-16 hash. It is deliberately local and has no language knowledge. */
export function hashString(value: string, seed: number): number {
  let hash = (0x811c9dc5 ^ seed) >>> 0
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
    hash ^= hash >>> 13
  }
  hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b)
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35)
  return (hash ^ (hash >>> 16)) >>> 0
}

export function hashInteger(value: number, seed: number): number {
  let hash = (Math.trunc(value) ^ seed) >>> 0
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d)
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b)
  return (hash ^ (hash >>> 16)) >>> 0
}

export function hashUnit(value: string, seed: number): number {
  return hashString(value, seed) / UINT32_MAX_PLUS_ONE
}

export function signedHash(value: string, seed: number): 1 | -1 {
  return (hashString(value, seed) & 1) === 0 ? 1 : -1
}

export function greatestCommonDivisor(first: number, second: number): number {
  let left = Math.abs(Math.trunc(first))
  let right = Math.abs(Math.trunc(second))
  while (right !== 0) {
    const remainder = left % right
    left = right
    right = remainder
  }
  return left
}

export function coprimeMultiplier(length: number, seed: number): number {
  if (length <= 2) return 1
  let candidate = (hashInteger(length, seed) % length) | 1
  while (greatestCommonDivisor(candidate, length) !== 1) {
    candidate = (candidate + 2) % length
    if (candidate === 0) candidate = 1
  }
  return candidate
}

