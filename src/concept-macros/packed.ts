import {
  canonicalFloat,
  compareMacroText,
  macroDigest,
  meanVectors,
  mergeWeightedVectors,
  vectorNorm,
} from './deterministic.ts'
import type { MacroPackedVectorV1, MacroPrototypeGroupV1 } from './types.ts'

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
const BASE64_REVERSE = new Int16Array(128).fill(-1)
for (let index = 0; index < BASE64_ALPHABET.length; index += 1) {
  BASE64_REVERSE[BASE64_ALPHABET.charCodeAt(index)] = index
}

export function encodeMacroBase64(bytes: Uint8Array): string {
  let output = ''
  for (let index = 0; index < bytes.length; index += 3) {
    const a = bytes[index]
    const hasB = index + 1 < bytes.length
    const hasC = index + 2 < bytes.length
    const b = hasB ? bytes[index + 1] : 0
    const c = hasC ? bytes[index + 2] : 0
    output += BASE64_ALPHABET[a >>> 2]
    output += BASE64_ALPHABET[((a & 3) << 4) | (b >>> 4)]
    output += hasB ? BASE64_ALPHABET[((b & 15) << 2) | (c >>> 6)] : '='
    output += hasC ? BASE64_ALPHABET[c & 63] : '='
  }
  return output
}

export function decodeMacroBase64(value: string): Uint8Array {
  if (value.length % 4 !== 0 || /[^A-Za-z0-9+/=]/u.test(value)) {
    throw new TypeError('Geçersiz canonical base64 payload.')
  }
  const padding = value.endsWith('==') ? 2 : value.endsWith('=') ? 1 : 0
  const result = new Uint8Array(value.length / 4 * 3 - padding)
  let offset = 0
  for (let index = 0; index < value.length; index += 4) {
    const chars = [value[index], value[index + 1], value[index + 2], value[index + 3]]
    const numbers = chars.map((char, charIndex) => {
      if (char === '=') return 0
      const code = char.charCodeAt(0)
      const decoded = code < BASE64_REVERSE.length ? BASE64_REVERSE[code] : -1
      if (decoded < 0) throw new TypeError(`Geçersiz base64 karakteri: ${charIndex}`)
      return decoded
    })
    const combined = (numbers[0] << 18) | (numbers[1] << 12) | (numbers[2] << 6) | numbers[3]
    if (offset < result.length) result[offset++] = (combined >>> 16) & 255
    if (offset < result.length) result[offset++] = (combined >>> 8) & 255
    if (offset < result.length) result[offset++] = combined & 255
  }
  if (encodeMacroBase64(result) !== value) throw new TypeError('Base64 payload canonical değil.')
  return result
}

export function packMacroVector(values: ArrayLike<number>): MacroPackedVectorV1 {
  if (!Number.isInteger(values.length) || values.length < 1) {
    throw new RangeError('Packed makro vektörü pozitif boyutlu olmalıdır.')
  }
  const bytes = new Uint8Array(values.length * Float32Array.BYTES_PER_ELEMENT)
  const view = new DataView(bytes.buffer)
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) throw new TypeError('Packed makro vektörü sonlu olmalıdır.')
    view.setFloat32(index * 4, Math.fround(values[index]), true)
  }
  const dataBase64 = encodeMacroBase64(bytes)
  const base = {
    format: 'macro-packed-f32-v1' as const,
    version: 1 as const,
    dimension: values.length,
    dataBase64,
  }
  return { ...base, digest: macroDigest('MPACK', base) }
}

export function unpackMacroVector(value: MacroPackedVectorV1): Float32Array {
  validateMacroPackedVector(value)
  const bytes = decodeMacroBase64(value.dataBase64)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const result = new Float32Array(value.dimension)
  for (let index = 0; index < result.length; index += 1) {
    result[index] = view.getFloat32(index * 4, true)
  }
  return result
}

export function validateMacroPackedVector(value: MacroPackedVectorV1): void {
  if (value.format !== 'macro-packed-f32-v1' || value.version !== 1 ||
    !Number.isInteger(value.dimension) || value.dimension < 1) {
    throw new TypeError('Geçersiz packed makro vektörü.')
  }
  const bytes = decodeMacroBase64(value.dataBase64)
  if (bytes.length !== value.dimension * 4) throw new RangeError('Packed vektör byte boyutu uyuşmuyor.')
  if (value.digest !== macroDigest('MPACK', {
    format: value.format,
    version: value.version,
    dimension: value.dimension,
    dataBase64: value.dataBase64,
  })) throw new Error('Packed makro vektörü digest uyuşmazlığı.')
}

export function computeMacroPrototypeGroupFingerprint(
  group: Omit<MacroPrototypeGroupV1, 'fingerprint'>,
): string {
  return macroDigest('MPGROUP', group)
}

export function createMacroPrototypeGroup(input: {
  independenceGroupId: string
  observationCount: number
  nonZeroEffectObservationCount: number
  entryMean: ArrayLike<number>
  exitMean: ArrayLike<number>
  trajectoryMean: ArrayLike<number>
  effectMean: ArrayLike<number>
  effectDirectionMean: ArrayLike<number>
}): MacroPrototypeGroupV1 {
  const base: Omit<MacroPrototypeGroupV1, 'fingerprint'> = {
    independenceGroupId: input.independenceGroupId,
    observationCount: input.observationCount,
    nonZeroEffectObservationCount: input.nonZeroEffectObservationCount,
    entryMean: packMacroVector(input.entryMean),
    exitMean: packMacroVector(input.exitMean),
    trajectoryMean: packMacroVector(input.trajectoryMean),
    effectMean: packMacroVector(input.effectMean),
    effectDirectionMean: packMacroVector(input.effectDirectionMean),
  }
  const result = { ...base, fingerprint: computeMacroPrototypeGroupFingerprint(base) }
  validateMacroPrototypeGroup(result)
  return result
}

function mergedPackedMean(
  left: MacroPackedVectorV1,
  leftWeight: number,
  right: MacroPackedVectorV1,
  rightWeight: number,
): number[] {
  return mergeWeightedVectors(
    Array.from(unpackMacroVector(left)),
    leftWeight,
    Array.from(unpackMacroVector(right)),
    rightWeight,
  )
}

export function mergeMacroPrototypeGroups(
  groups: readonly MacroPrototypeGroupV1[],
  maximumGroups: number,
): MacroPrototypeGroupV1[] {
  if (!Number.isInteger(maximumGroups) || maximumGroups < 1) {
    throw new RangeError('Makro prototip grup kapasitesi pozitif tam sayı olmalıdır.')
  }
  const merged = new Map<string, MacroPrototypeGroupV1>()
  for (const group of [...groups].sort((left, right) =>
    compareMacroText(left.independenceGroupId, right.independenceGroupId))) {
    validateMacroPrototypeGroup(group)
    const previous = merged.get(group.independenceGroupId)
    if (!previous) {
      merged.set(group.independenceGroupId, group)
      continue
    }
    const observationCount = previous.observationCount + group.observationCount
    const nonZeroEffectObservationCount = previous.nonZeroEffectObservationCount +
      group.nonZeroEffectObservationCount
    merged.set(group.independenceGroupId, createMacroPrototypeGroup({
      independenceGroupId: group.independenceGroupId,
      observationCount,
      nonZeroEffectObservationCount,
      entryMean: mergedPackedMean(
        previous.entryMean, previous.observationCount,
        group.entryMean, group.observationCount,
      ),
      exitMean: mergedPackedMean(
        previous.exitMean, previous.observationCount,
        group.exitMean, group.observationCount,
      ),
      trajectoryMean: mergedPackedMean(
        previous.trajectoryMean, previous.observationCount,
        group.trajectoryMean, group.observationCount,
      ),
      effectMean: mergedPackedMean(
        previous.effectMean, previous.observationCount,
        group.effectMean, group.observationCount,
      ),
      effectDirectionMean: mergedPackedMean(
        previous.effectDirectionMean, previous.nonZeroEffectObservationCount,
        group.effectDirectionMean, group.nonZeroEffectObservationCount,
      ),
    }))
  }
  return [...merged.values()]
    .sort((left, right) => compareMacroText(left.independenceGroupId, right.independenceGroupId))
    .slice(0, maximumGroups)
}

export function aggregateMacroPrototypeGroups(groups: readonly MacroPrototypeGroupV1[]): {
  entry: number[]
  exit: number[]
  trajectory: number[]
  effect: number[]
  effectCohesion: number
  observationCount: number
} {
  if (!groups.length) throw new RangeError('Makro prototipi için en az bir kaynak grubu gerekir.')
  const canonical = [...groups].sort((left, right) =>
    compareMacroText(left.independenceGroupId, right.independenceGroupId))
  for (const group of canonical) validateMacroPrototypeGroup(group)
  const entry = meanVectors(canonical.map((group) => Array.from(unpackMacroVector(group.entryMean))))
  const exit = meanVectors(canonical.map((group) => Array.from(unpackMacroVector(group.exitMean))))
  const trajectory = meanVectors(
    canonical.map((group) => Array.from(unpackMacroVector(group.trajectoryMean))),
  )
  const effect = meanVectors(canonical.map((group) => Array.from(unpackMacroVector(group.effectMean))))
  const directions = meanVectors(
    canonical.map((group) => Array.from(unpackMacroVector(group.effectDirectionMean))),
  )
  const allZeroEffects = canonical.every((group) => group.nonZeroEffectObservationCount === 0)
  return {
    entry,
    exit,
    trajectory,
    effect,
    effectCohesion: allZeroEffects ? 1 : canonicalFloat(Math.min(1, vectorNorm(directions))),
    observationCount: canonical.reduce((sum, group) => sum + group.observationCount, 0),
  }
}

export function validateMacroPrototypeGroup(group: MacroPrototypeGroupV1): void {
  if (!group.independenceGroupId || !Number.isInteger(group.observationCount) ||
    group.observationCount < 1 || !Number.isInteger(group.nonZeroEffectObservationCount) ||
    group.nonZeroEffectObservationCount < 0 ||
    group.nonZeroEffectObservationCount > group.observationCount) {
    throw new Error('Makro prototip grup sayaçları bozuk.')
  }
  const vectors = [
    group.entryMean,
    group.exitMean,
    group.trajectoryMean,
    group.effectMean,
    group.effectDirectionMean,
  ]
  for (const vector of vectors) validateMacroPackedVector(vector)
  if (new Set(vectors.map((vector) => vector.dimension)).size !== 1) {
    throw new RangeError('Makro prototip grup vektör boyutları uyuşmuyor.')
  }
  const { fingerprint, ...base } = group
  if (fingerprint !== computeMacroPrototypeGroupFingerprint(base)) {
    throw new Error('Makro prototip grup fingerprint uyuşmazlığı.')
  }
}
