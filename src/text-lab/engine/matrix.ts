import type { MatrixLayout } from '../domain/types'

export const MATRIX_EPSILON = 1e-8

export function clamp(value: number, minimum = -1, maximum = 1): number {
  return Math.min(maximum, Math.max(minimum, value))
}

export function vectorNorm(
  vector: ArrayLike<number>,
  start = 0,
  length = vector.length - start,
): number {
  let squared = 0
  const end = Math.min(vector.length, start + length)
  for (let index = start; index < end; index += 1) {
    const value = vector[index]
    squared += value * value
  }
  return Math.sqrt(squared)
}

export function dotProduct(
  first: ArrayLike<number>,
  second: ArrayLike<number>,
  start = 0,
  length = Math.min(first.length, second.length) - start,
): number {
  let dot = 0
  const end = Math.min(first.length, second.length, start + length)
  for (let index = start; index < end; index += 1) {
    dot += first[index] * second[index]
  }
  return dot
}

export function cosineRegion(
  first: ArrayLike<number>,
  second: ArrayLike<number>,
  start: number,
  length: number,
  firstNorm?: number,
  secondNorm?: number,
): number {
  const leftNorm = firstNorm ?? vectorNorm(first, start, length)
  const rightNorm = secondNorm ?? vectorNorm(second, start, length)
  if (leftNorm <= MATRIX_EPSILON || rightNorm <= MATRIX_EPSILON) return 0
  const value = dotProduct(first, second, start, length) / (leftNorm * rightNorm)
  return clamp(Number.isFinite(value) ? value : 0)
}

export function maxAbsNormalize(vector: Float32Array): Float32Array {
  let maximum = 0
  for (const value of vector) maximum = Math.max(maximum, Math.abs(value))
  if (maximum <= 1) return vector
  for (let index = 0; index < vector.length; index += 1) {
    vector[index] /= maximum
  }
  return vector
}

export function identityView(matrix: Float32Array, layout: MatrixLayout): Float32Array {
  return matrix.subarray(0, layout.identityLength)
}

export function relationView(matrix: Float32Array, layout: MatrixLayout): Float32Array {
  return matrix.subarray(layout.relationStart)
}

export function relationToFullMatrix(
  relation: ArrayLike<number>,
  layout: MatrixLayout,
): Float32Array {
  const matrix = new Float32Array(layout.cellCount)
  matrix.set(Array.from(relation), layout.relationStart)
  return matrix
}

export function composeMatrix(
  identity: ArrayLike<number>,
  relation: ArrayLike<number>,
  layout: MatrixLayout,
): Float32Array {
  if (identity.length !== layout.identityLength || relation.length !== layout.relationLength) {
    throw new RangeError('Kimlik veya bağlam vektörü matris maskesiyle uyuşmuyor.')
  }
  const matrix = new Float32Array(layout.cellCount)
  matrix.set(Array.from(identity), 0)
  matrix.set(Array.from(relation), layout.relationStart)
  return matrix
}
