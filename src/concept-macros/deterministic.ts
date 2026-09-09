import { stableDigest } from '../families/sparse.ts'
import type {
  MacroMemberReferenceV1,
  MacroMdlMetricsV1,
  MacroObservationV1,
  MacroSignalSignatureV1,
} from './types.ts'

export const MACRO_EPSILON = 1e-12

export function compareMacroText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

export function clampMacro(value: number, minimum = 0, maximum = 1): number {
  if (!Number.isFinite(value)) return minimum
  return Math.min(maximum, Math.max(minimum, value))
}

export function macroDigest(label: string, ...parts: readonly unknown[]): string {
  return `${label}-${stableDigest(['concept-macros-v1', label, ...parts]).toUpperCase()}`
}

export function canonicalFamilyIds(familyIds: readonly string[]): string[] {
  const result = [...new Set(familyIds.map((value) => value.trim()).filter(Boolean))]
  result.sort(compareMacroText)
  return result
}

export function canonicalMember(member: MacroMemberReferenceV1): MacroMemberReferenceV1 {
  if (!member.id.trim()) throw new TypeError('Makro üye kimliği boş olamaz.')
  if (member.kind !== 'raw-neuron' && member.kind !== 'composite-neuron') {
    throw new TypeError('Bilinmeyen makro üye türü.')
  }
  return {
    kind: member.kind,
    id: member.id.trim(),
    identityDigest: member.identityDigest?.trim() || null,
    familyIds: canonicalFamilyIds(member.familyIds),
  }
}

export function canonicalExpansion(
  expansion: readonly MacroMemberReferenceV1[],
): MacroMemberReferenceV1[] {
  return expansion.map(canonicalMember)
}

export function computeExpansionDigest(expansion: readonly MacroMemberReferenceV1[]): string {
  const canonical = canonicalExpansion(expansion)
  return macroDigest('MEXP', canonical.map((member) => [member.kind, member.id, member.identityDigest]))
}

export function computeExactSignature(expansion: readonly MacroMemberReferenceV1[]): string {
  const canonical = canonicalExpansion(expansion)
  return macroDigest('MEXACT', canonical.map((member) => [member.kind, member.id, member.identityDigest]))
}

export function computeFamilySignature(expansion: readonly MacroMemberReferenceV1[]): string {
  const canonical = canonicalExpansion(expansion)
  return macroDigest(
    'MFAMILY',
    canonical.map((member) => member.familyIds.length ? member.familyIds : ['__unassigned__']),
  )
}

export function canonicalFloat(value: number): number {
  if (!Number.isFinite(value)) throw new TypeError('Makro vektörleri yalnız sonlu değerler içerebilir.')
  return Object.is(value, -0) ? 0 : Math.fround(value)
}

export function canonicalVector(values: readonly number[]): number[] {
  return values.map(canonicalFloat)
}

export function validateSignalVector(
  values: readonly number[],
  maximumDimension: number,
  label = 'Sinyal',
): void {
  if (!Array.isArray(values) || values.length < 1 || values.length > maximumDimension) {
    throw new RangeError(`${label} boyutu 1 ile ${maximumDimension} arasında olmalıdır.`)
  }
  for (const value of values) {
    if (!Number.isFinite(value) || value < -1 || value > 1) {
      throw new RangeError(`${label} değerleri [-1,1] aralığında ve sonlu olmalıdır.`)
    }
  }
}

export function createSignalSignature(vector: readonly number[]): MacroSignalSignatureV1 {
  const canonical = canonicalVector(vector)
  return {
    dimension: canonical.length,
    vector: canonical,
    digest: macroDigest('MSIGNAL', canonical.length, canonical),
  }
}

export function meanVectors(vectors: readonly (readonly number[])[]): number[] {
  if (!vectors.length) throw new RangeError('Vektör ortalaması için en az bir değer gerekir.')
  const dimension = vectors[0].length
  if (!dimension || vectors.some((vector) => vector.length !== dimension)) {
    throw new RangeError('Farklı boyutlu makro vektörleri ortalanamaz.')
  }
  const result = new Array<number>(dimension).fill(0)
  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) result[index] += vector[index]
  }
  return result.map((value) => canonicalFloat(value / vectors.length))
}

/** Each independence group gets exactly one vote, regardless of its episode count. */
export function sourceBalancedMean(
  observations: readonly MacroObservationV1[],
  pick: (observation: MacroObservationV1) => readonly number[],
): number[] {
  if (!observations.length) throw new RangeError('Kaynak dengeli ortalama için gözlem gerekir.')
  const groups = new Map<string, MacroObservationV1[]>()
  for (const observation of observations) {
    const key = observation.evidence.independenceGroupId
    const group = groups.get(key) ?? []
    group.push(observation)
    groups.set(key, group)
  }
  return meanVectors(
    [...groups.entries()]
      .sort(([left], [right]) => compareMacroText(left, right))
      .map(([, values]) => meanVectors(values.map(pick))),
  )
}

export function subtractVectors(left: readonly number[], right: readonly number[]): number[] {
  if (left.length !== right.length || !left.length) {
    throw new RangeError('Makro etki vektörleri aynı ve pozitif boyutta olmalıdır.')
  }
  return left.map((value, index) => canonicalFloat(value - right[index]))
}

export function vectorNorm(vector: ArrayLike<number>): number {
  let squared = 0
  for (let index = 0; index < vector.length; index += 1) squared += vector[index] * vector[index]
  return Math.sqrt(squared)
}

export function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length || !left.length) return 0
  let dot = 0
  let leftSquared = 0
  let rightSquared = 0
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index]
    leftSquared += left[index] * left[index]
    rightSquared += right[index] * right[index]
  }
  if (leftSquared <= MACRO_EPSILON && rightSquared <= MACRO_EPSILON) return 1
  if (leftSquared <= MACRO_EPSILON || rightSquared <= MACRO_EPSILON) return 0
  return clampMacro(dot / Math.sqrt(leftSquared * rightSquared), -1, 1)
}

export function effectCohesion(
  observations: readonly MacroObservationV1[],
  prototype: readonly number[],
): number {
  if (!observations.length) return 0
  const total = observations.reduce(
    (sum, observation) => sum + cosineSimilarity(observation.effect, prototype),
    0,
  )
  return clampMacro(total / observations.length, 0, 1)
}

export function computeMdlMetrics(
  observationCount: number,
  motifLength: number,
  definitionOverhead: number,
): MacroMdlMetricsV1 {
  const definitionCost = motifLength + definitionOverhead
  const rawDescriptionCost = observationCount * motifLength
  const macroDescriptionCost = definitionCost + observationCount
  const compressionGain = rawDescriptionCost - macroDescriptionCost
  const compressionRatio = rawDescriptionCost > MACRO_EPSILON
    ? clampMacro(compressionGain / rawDescriptionCost, -1, 1)
    : 0
  return {
    rawDescriptionCost: canonicalFloat(rawDescriptionCost),
    macroDescriptionCost: canonicalFloat(macroDescriptionCost),
    definitionCost: canonicalFloat(definitionCost),
    compressionGain: canonicalFloat(compressionGain),
    compressionRatio: canonicalFloat(compressionRatio),
  }
}

export function mergeWeightedVectors(
  left: readonly number[],
  leftWeight: number,
  right: readonly number[],
  rightWeight: number,
): number[] {
  if (left.length !== right.length || !left.length) {
    throw new RangeError('Birleştirilen makro prototipleri aynı boyutta olmalıdır.')
  }
  const total = Math.max(0, leftWeight) + Math.max(0, rightWeight)
  if (total <= MACRO_EPSILON) return canonicalVector(left)
  return left.map((value, index) => canonicalFloat(
    (value * Math.max(0, leftWeight) + right[index] * Math.max(0, rightWeight)) / total,
  ))
}

export function squaredError(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length || !left.length) {
    throw new RangeError('Hata ölçülen vektörler aynı boyutta olmalıdır.')
  }
  let total = 0
  for (let index = 0; index < left.length; index += 1) {
    const difference = left[index] - right[index]
    total += difference * difference
  }
  return total / left.length
}

export function vectorFidelity(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length || !left.length) {
    throw new RangeError('Fidelity vektörleri aynı boyutta olmalıdır.')
  }
  const difference = Math.sqrt(left.reduce((sum, value, index) => {
    const delta = value - right[index]
    return sum + delta * delta
  }, 0))
  const scale = vectorNorm(left) + vectorNorm(right)
  if (scale <= MACRO_EPSILON) return 1
  return clampMacro(1 - difference / scale, 0, 1)
}
