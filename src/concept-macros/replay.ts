import { stableDigest } from '../families/sparse.ts'
import { compareMacroText, macroDigest } from './deterministic.ts'
import { decodeMacroBase64, encodeMacroBase64 } from './packed.ts'
import type { MacroReplayBloomV1 } from './types.ts'

function bloomFingerprint(value: Omit<MacroReplayBloomV1, 'fingerprint'>): string {
  return macroDigest('MRBLOOM', value)
}

export function createMacroReplayBloom(bitCount: number, hashCount: number): MacroReplayBloomV1 {
  if (!Number.isInteger(bitCount) || bitCount < 64 || bitCount % 8 !== 0 ||
    !Number.isInteger(hashCount) || hashCount < 1 || hashCount > 16) {
    throw new RangeError('Replay Bloom boyutu/hash sayısı geçersiz.')
  }
  const base: Omit<MacroReplayBloomV1, 'fingerprint'> = {
    format: 'macro-replay-bloom-v1',
    version: 1,
    bitCount,
    hashCount,
    insertCount: 0,
    bitsBase64: encodeMacroBase64(new Uint8Array(bitCount / 8)),
  }
  return { ...base, fingerprint: bloomFingerprint(base) }
}

function bloomPositions(key: string, bitCount: number, hashCount: number): number[] {
  const digest = stableDigest(['concept-macro-replay-bloom-v1', key])
  const h1 = Number.parseInt(digest.slice(0, 8), 16) >>> 0
  const h2 = (Number.parseInt(digest.slice(8, 16), 16) | 1) >>> 0
  const h3 = Number.parseInt(digest.slice(16, 24), 16) >>> 0
  const positions: number[] = []
  for (let index = 0; index < hashCount; index += 1) {
    const mixed = (h1 + Math.imul(index + 1, h2) + Math.imul(index * index + 1, h3)) >>> 0
    positions.push(mixed % bitCount)
  }
  return positions
}

export function validateMacroReplayBloom(value: MacroReplayBloomV1): void {
  if (value.format !== 'macro-replay-bloom-v1' || value.version !== 1 ||
    !Number.isInteger(value.bitCount) || value.bitCount < 64 || value.bitCount % 8 !== 0 ||
    !Number.isInteger(value.hashCount) || value.hashCount < 1 || value.hashCount > 16 ||
    !Number.isInteger(value.insertCount) || value.insertCount < 0) {
    throw new TypeError('Geçersiz replay Bloom kaydı.')
  }
  if (decodeMacroBase64(value.bitsBase64).length !== value.bitCount / 8) {
    throw new RangeError('Replay Bloom byte boyutu uyuşmuyor.')
  }
  const { fingerprint, ...base } = value
  if (fingerprint !== bloomFingerprint(base)) throw new Error('Replay Bloom fingerprint uyuşmazlığı.')
}

export function macroReplayBloomMayContain(value: MacroReplayBloomV1, key: string): boolean {
  return createMacroReplayBloomQuery(value)(key)
}

export function createMacroReplayBloomQuery(
  value: MacroReplayBloomV1,
): (key: string) => boolean {
  validateMacroReplayBloom(value)
  const bytes = decodeMacroBase64(value.bitsBase64)
  return (key: string) => bloomPositions(key, value.bitCount, value.hashCount).every((position) =>
    (bytes[position >>> 3] & (1 << (position & 7))) !== 0)
}

export function addMacroReplayBloomKeys(
  value: MacroReplayBloomV1,
  keys: readonly string[],
): MacroReplayBloomV1 {
  validateMacroReplayBloom(value)
  const canonicalKeys = [...new Set(keys.filter(Boolean))].sort(compareMacroText)
  if (!canonicalKeys.length) return value
  const bytes = decodeMacroBase64(value.bitsBase64)
  for (const key of canonicalKeys) {
    for (const position of bloomPositions(key, value.bitCount, value.hashCount)) {
      bytes[position >>> 3] |= 1 << (position & 7)
    }
  }
  const base: Omit<MacroReplayBloomV1, 'fingerprint'> = {
    ...value,
    insertCount: value.insertCount + canonicalKeys.length,
    bitsBase64: encodeMacroBase64(bytes),
  }
  delete (base as Partial<MacroReplayBloomV1>).fingerprint
  return { ...base, fingerprint: bloomFingerprint(base) }
}

export function boundedMacroReplayLedger(
  bloom: MacroReplayBloomV1,
  currentExactKeys: readonly string[],
  incomingKeys: readonly string[],
  maximumExactKeys: number,
): { exactKeys: string[]; bloom: MacroReplayBloomV1 } {
  if (!Number.isInteger(maximumExactKeys) || maximumExactKeys < 1) {
    throw new RangeError('Exact replay ledger kapasitesi pozitif tam sayı olmalıdır.')
  }
  const all = [...new Set([...currentExactKeys, ...incomingKeys])].sort(compareMacroText)
  const exactKeys = all.slice(0, maximumExactKeys)
  const overflow = all.slice(maximumExactKeys)
  return { exactKeys, bloom: addMacroReplayBloomKeys(bloom, overflow) }
}
