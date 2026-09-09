import {
  THOUGHT_MATRIX_CELLS,
  type ThoughtMatrixLayout,
  type ThoughtRuntimeConfig,
  type ThoughtVector,
} from './types'

export const THOUGHT_EPSILON = 1e-12

export function clamp(value: number, minimum = 0, maximum = 1): number {
  if (!Number.isFinite(value)) return minimum
  return Math.min(maximum, Math.max(minimum, value))
}

export function assertThoughtVector(value: Float32Array, label = 'Vektör'): void {
  if (!(value instanceof Float32Array) || value.length !== THOUGHT_MATRIX_CELLS) {
    throw new RangeError(`${label} tam ${THOUGHT_MATRIX_CELLS} Float32 hücre içermelidir.`)
  }
  assertBoundedFinite(value, label)
}

export function assertThoughtPartition(value: Float32Array, length: number, label: string): void {
  if (!(value instanceof Float32Array) || value.length !== length) {
    throw new RangeError(`${label} tam ${length} Float32 hücre içermelidir.`)
  }
  assertBoundedFinite(value, label)
}

function assertBoundedFinite(value: Float32Array, label: string): void {
  if (!value.every((entry) => Number.isFinite(entry) && entry >= -1 && entry <= 1)) {
    throw new RangeError(`${label} yalnız [-1, 1] aralığında sonlu değerler içermelidir.`)
  }
}

export function vectorNorm(value: ArrayLike<number>): number {
  let squared = 0
  for (let index = 0; index < value.length; index += 1) squared += value[index] * value[index]
  return Math.sqrt(squared)
}

export function cosineSimilarity(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) throw new RangeError('Cosine için vektör boyutları eşit olmalıdır.')
  let dot = 0
  let leftSquared = 0
  let rightSquared = 0
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index]
    leftSquared += left[index] * left[index]
    rightSquared += right[index] * right[index]
  }
  const denominator = Math.sqrt(leftSquared * rightSquared)
  return denominator > THOUGHT_EPSILON ? clamp(dot / denominator, -1, 1) : 0
}

export function positiveCompatibility(left: ArrayLike<number>, right: ArrayLike<number>): number {
  return clamp(cosineSimilarity(left, right))
}

export function projectThoughtVector(value: ArrayLike<number>, maximumAbsolute = 1): Float32Array {
  const output = new Float32Array(value.length)
  let peak = 0
  for (let index = 0; index < value.length; index += 1) peak = Math.max(peak, Math.abs(value[index]))
  const scale = peak > maximumAbsolute ? maximumAbsolute / peak : 1
  for (let index = 0; index < value.length; index += 1) {
    output[index] = Math.fround(clamp(value[index] * scale, -maximumAbsolute, maximumAbsolute))
  }
  return output
}

export function composeNeuronRepresentation(
  immutableIdentity: Float32Array,
  baseRelation: Float32Array,
  layout: ThoughtMatrixLayout,
  persistentPatch?: Float32Array,
  fastPatch?: Float32Array,
): ThoughtVector {
  assertThoughtPartition(immutableIdentity, layout.identityLength, 'Nöron kimliği I')
  assertThoughtPartition(baseRelation, layout.relationLength, 'Nöron ilişkisi R')
  if (persistentPatch) assertThoughtPartition(persistentPatch, layout.relationLength, 'Kalıcı R yaması')
  if (fastPatch) assertThoughtPartition(fastPatch, layout.relationLength, 'Hızlı R yaması')
  const output = new Float32Array(THOUGHT_MATRIX_CELLS)
  output.set(immutableIdentity, 0)
  for (let index = 0; index < layout.relationLength; index += 1) {
    output[layout.relationStart + index] = Math.fround(clamp(
      baseRelation[index] + (persistentPatch?.[index] ?? 0) + (fastPatch?.[index] ?? 0),
      -1,
      1,
    ))
  }
  return output
}

/** S(t+1)=(1-c)S(t)+cZ; the approved learning rate is exactly alpha_t=c_t. */
export function transformThoughtSignal(
  signal: ThoughtVector,
  representation: ThoughtVector,
  config: ThoughtRuntimeConfig,
  compatibilityOverride?: number,
): { signal: ThoughtVector; compatibility: number; alpha: number; changeMagnitude: number } {
  assertThoughtVector(signal, 'Sinyal S')
  assertThoughtVector(representation, 'Nöron gösterimi Z=[I|R]')
  if (compatibilityOverride !== undefined && (
    !Number.isFinite(compatibilityOverride) || compatibilityOverride < 0 || compatibilityOverride > 1
  )) throw new RangeError('Uyumluluk override değeri [0,1] aralığında olmalıdır.')
  const compatibility = compatibilityOverride ?? positiveCompatibility(signal, representation)
  const alpha = compatibility
  const mixed = new Float32Array(THOUGHT_MATRIX_CELLS)
  for (let index = 0; index < THOUGHT_MATRIX_CELLS; index += 1) {
    mixed[index] = Math.fround((1 - alpha) * signal[index] + alpha * representation[index])
  }
  const projected = projectThoughtVector(mixed, config.signalNormMaximum)
  let squaredChange = 0
  for (let index = 0; index < THOUGHT_MATRIX_CELLS; index += 1) {
    const delta = projected[index] - signal[index]
    squaredChange += delta * delta
  }
  return { signal: projected, compatibility, alpha, changeMagnitude: Math.sqrt(squaredChange) }
}

/** Only the R partition receives a branch-local fast patch; I is never writable here. */
export function updateFastRelationPatch(
  signal: ThoughtVector,
  representation: ThoughtVector,
  previousPatch: Float32Array | undefined,
  layout: ThoughtMatrixLayout,
  signedRate: number,
  eligibility: number,
  relationWriteWeights?: Float32Array,
): Float32Array {
  assertThoughtVector(signal, 'Hızlı iz sinyali')
  assertThoughtVector(representation, 'Hızlı iz Z gösterimi')
  if (previousPatch) assertThoughtPartition(previousPatch, layout.relationLength, 'Önceki hızlı R yaması')
  if (relationWriteWeights) {
    if (relationWriteWeights.length !== layout.relationLength ||
        relationWriteWeights.some((weight) => !Number.isFinite(weight) || weight < 0 || weight > 1)) {
      throw new RangeError('Hızlı R yazma kapısı R uzunluğunda [0,1] ağırlıklar taşımalıdır.')
    }
  }
  const output = new Float32Array(layout.relationLength)
  for (let index = 0; index < layout.relationLength; index += 1) {
    const coordinate = layout.relationStart + index
    const residual = signal[coordinate] - representation[coordinate]
    const update = signedRate * eligibility * residual
    output[index] = Math.fround(clamp(
      (previousPatch?.[index] ?? 0) + (relationWriteWeights ? update * relationWriteWeights[index] : update),
      -1,
      1,
    ))
  }
  return output
}

/** Removes the positive component already explained by the best known Z. */
export function unexplainedResidual(signal: ThoughtVector, bestRepresentation?: ThoughtVector): ThoughtVector {
  assertThoughtVector(signal, 'Residual sinyali')
  if (!bestRepresentation) return projectThoughtVector(signal, 1)
  assertThoughtVector(bestRepresentation, 'Residual karşılaştırma gösterimi')
  let dot = 0
  let squared = 0
  for (let index = 0; index < THOUGHT_MATRIX_CELLS; index += 1) {
    dot += signal[index] * bestRepresentation[index]
    squared += bestRepresentation[index] * bestRepresentation[index]
  }
  const scale = squared > THOUGHT_EPSILON ? clamp(dot / squared) : 0
  const residual = new Float32Array(THOUGHT_MATRIX_CELLS)
  for (let index = 0; index < THOUGHT_MATRIX_CELLS; index += 1) {
    residual[index] = Math.fround(signal[index] - scale * bestRepresentation[index])
  }
  return projectThoughtVector(residual, 1)
}

export function normalizePositive<T>(values: readonly T[], score: (value: T) => number): Array<T & { probability: number }> {
  const total = values.reduce((sum, value) => sum + Math.max(0, score(value)), 0)
  if (total <= THOUGHT_EPSILON) {
    const probability = values.length ? 1 / values.length : 0
    return values.map((value) => ({ ...value, probability }))
  }
  return values.map((value) => ({ ...value, probability: Math.max(0, score(value)) / total }))
}

export function shannonEntropy(probabilities: readonly number[]): number {
  const total = probabilities.reduce((sum, value) => sum + Math.max(0, value), 0)
  if (total <= THOUGHT_EPSILON) return 0
  let entropy = 0
  for (const raw of probabilities) {
    const probability = Math.max(0, raw) / total
    if (probability > THOUGHT_EPSILON) entropy -= probability * Math.log(probability)
  }
  return entropy
}

/** K=ceil(exp(H)), capped to the approved 1..8 adaptive branch interval. */
export function entropyBranchCount(probabilities: readonly number[], config: ThoughtRuntimeConfig): number {
  const perplexity = Math.ceil(Math.exp(shannonEntropy(probabilities)))
  return Math.min(config.maximumBranches, Math.max(config.minimumBranches, perplexity))
}
