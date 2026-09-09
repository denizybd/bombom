import { canonicalRegionFloat } from './deterministic.ts'

export const REGION_MATH_EPSILON = 1e-12

export function clampRegion(value: number, minimum = 0, maximum = 1): number {
  if (!Number.isFinite(value)) return minimum
  return Math.min(maximum, Math.max(minimum, value))
}

export function assertFiniteVector(
  values: ArrayLike<number>,
  expectedLength: number,
  minimum: number,
  maximum: number,
  label: string,
): void {
  if (values.length !== expectedLength) throw new RangeError(`${label} ${expectedLength} değer içermelidir.`)
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      throw new RangeError(`${label} değerleri ${minimum}..${maximum} aralığında ve sonlu olmalıdır.`)
    }
  }
}

export function vectorNorm(values: ArrayLike<number>): number {
  let squared = 0
  for (let index = 0; index < values.length; index += 1) squared += values[index] * values[index]
  return Math.sqrt(squared)
}

export function cosineRegionVectors(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length || left.length < 1) return 0
  let dot = 0
  let leftSquared = 0
  let rightSquared = 0
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index]
    leftSquared += left[index] * left[index]
    rightSquared += right[index] * right[index]
  }
  if (leftSquared <= REGION_MATH_EPSILON && rightSquared <= REGION_MATH_EPSILON) return 1
  if (leftSquared <= REGION_MATH_EPSILON || rightSquared <= REGION_MATH_EPSILON) return 0
  return clampRegion(dot / Math.sqrt(leftSquared * rightSquared), -1, 1)
}

export function meanRegionVectors(vectors: readonly (readonly number[])[]): number[] {
  if (!vectors.length) throw new RangeError('Vektör ortalaması için en az bir değer gerekir.')
  const dimension = vectors[0].length
  if (!dimension || vectors.some((vector) => vector.length !== dimension)) {
    throw new RangeError('Farklı boyutlu bölge vektörleri ortalanamaz.')
  }
  const result = new Array<number>(dimension).fill(0)
  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) result[index] += vector[index]
  }
  return result.map((value) => canonicalRegionFloat(value / vectors.length))
}

export function weightedRegionMean(
  vectors: readonly { readonly values: readonly number[]; readonly weight: number }[],
): number[] {
  const first = vectors.find((entry) => entry.weight > 0)
  if (!first) throw new RangeError('Ağırlıklı ortalama pozitif ağırlık ister.')
  const result = new Array<number>(first.values.length).fill(0)
  let total = 0
  for (const entry of vectors) {
    if (!Number.isFinite(entry.weight) || entry.weight < 0 || entry.values.length !== result.length) {
      throw new RangeError('Ağırlıklı bölge vektörleri geçersiz.')
    }
    if (entry.weight === 0) continue
    total += entry.weight
    for (let index = 0; index < result.length; index += 1) {
      result[index] += entry.values[index] * entry.weight
    }
  }
  if (total <= REGION_MATH_EPSILON) throw new RangeError('Ağırlıklı ortalama toplamı sıfır olamaz.')
  return result.map((value) => canonicalRegionFloat(value / total))
}
