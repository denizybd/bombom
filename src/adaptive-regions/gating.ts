import { validateAdaptiveRegionAtlas } from './atlas.ts'
import { canonicalRegionFloat, compareRegionText, immutableRegionValue } from './deterministic.ts'
import {
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_RELATION_LENGTH,
  extractRelationPage,
} from './geometry.ts'
import { assertFiniteVector, clampRegion, cosineRegionVectors, REGION_MATH_EPSILON } from './math.ts'
import type {
  PlasticRegionAtlasV1,
  RegionGateDistributionV1,
  RegionGateV1,
} from './types.ts'

export interface ComputeRegionGatesInput {
  readonly pageCompatibilities: ArrayLike<number>
  readonly pageBehaviors?: readonly ArrayLike<number>[]
  readonly contextScores?: Readonly<Record<string, number>>
  readonly capacityPenalties?: Readonly<Record<string, number>>
  readonly compatibilityWeight?: number
  readonly prototypeWeight?: number
  readonly contextWeight?: number
  readonly confidenceWeight?: number
  readonly temperature?: number
  readonly unmatchedLogit?: number
  readonly maximumActiveRegions?: number
}

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`${label} sonlu olmalıdır.`)
  return value
}

export function computeRegionGates(
  atlas: PlasticRegionAtlasV1,
  input: ComputeRegionGatesInput,
): RegionGateDistributionV1 {
  validateAdaptiveRegionAtlas(atlas)
  assertFiniteVector(input.pageCompatibilities, ADAPTIVE_PAGE_COUNT, -1, 1, 'Mikro-sayfa uyumu')
  const compatibilityWeight = finite(input.compatibilityWeight ?? 1, 'Uyum ağırlığı')
  const prototypeWeight = finite(input.prototypeWeight ?? 1, 'Prototip uyum ağırlığı')
  const contextWeight = finite(input.contextWeight ?? 1, 'Bağlam ağırlığı')
  const confidenceWeight = finite(input.confidenceWeight ?? 1, 'Güven ağırlığı')
  if (compatibilityWeight < 0 || prototypeWeight < 0 || contextWeight < 0 || confidenceWeight < 0) {
    throw new RangeError('Bölge kapısı katkı ağırlıkları negatif olamaz.')
  }
  const temperature = finite(input.temperature ?? 1, 'Bölge kapısı sıcaklığı')
  if (temperature <= 0) throw new RangeError('Bölge kapısı sıcaklığı pozitif olmalıdır.')
  const unmatchedLogit = finite(input.unmatchedLogit ?? 0, 'Eşleşmeyen kanal logiti')
  const maximumActiveRegions = input.maximumActiveRegions ?? 4
  if (!Number.isInteger(maximumActiveRegions) || maximumActiveRegions < 0) {
    throw new RangeError('Etkin bölge sınırı negatif olmayan tam sayı olmalıdır.')
  }
  if (input.pageBehaviors !== undefined) {
    if (!Array.isArray(input.pageBehaviors) || input.pageBehaviors.length !== ADAPTIVE_PAGE_COUNT) {
      throw new RangeError('Mikro-sayfa davranışları tam 56 sayfa taşımalıdır.')
    }
    for (const behavior of input.pageBehaviors) {
      assertFiniteVector(behavior, atlas.behaviorDimension, -1, 1, 'Mikro-sayfa davranışı')
    }
  }

  const drafts: Omit<RegionGateV1, 'probability'>[] = []
  for (const region of atlas.regions) {
    // Provisional regions are observable proposals, not active routing units.
    if (region.lifecycle !== 'active') continue
    let weightedCompatibility = 0
    let weightedPrototypeCompatibility = 0
    let capacity = 0
    for (const page of atlas.pages) {
      const weight = page.memberships.find((membership) => membership.regionId === region.id)?.weight ?? 0
      weightedCompatibility += weight * input.pageCompatibilities[page.pageIndex]
      if (input.pageBehaviors) {
        weightedPrototypeCompatibility += weight * cosineRegionVectors(
          input.pageBehaviors[page.pageIndex],
          region.prototype.values,
        )
      }
      capacity += weight
    }
    if (capacity <= REGION_MATH_EPSILON) continue
    const compatibility = clampRegion(weightedCompatibility / capacity, -1, 1)
    const prototypeCompatibility = input.pageBehaviors
      ? clampRegion(weightedPrototypeCompatibility / capacity, -1, 1)
      : 0
    const contextScore = clampRegion(input.contextScores?.[region.id] ?? 0, -1, 1)
    const capacityPenalty = Math.max(0, finite(input.capacityPenalties?.[region.id] ?? 0, 'Kapasite cezası'))
    const confidence = clampRegion((1 - Math.exp(-region.evidence.confidenceMass)) * region.evidence.cohesion)
    const logit = (
      compatibilityWeight * compatibility +
      prototypeWeight * prototypeCompatibility +
      contextWeight * contextScore +
      confidenceWeight * Math.log(confidence + 1e-9) -
      capacityPenalty
    ) / temperature
    drafts.push({
      regionId: region.id,
      compatibility: canonicalRegionFloat(compatibility),
      prototypeCompatibility: canonicalRegionFloat(prototypeCompatibility),
      contextScore: canonicalRegionFloat(contextScore),
      confidence: canonicalRegionFloat(confidence),
      capacityPenalty: canonicalRegionFloat(capacityPenalty),
      logit: canonicalRegionFloat(logit),
    })
  }

  if (!drafts.length) {
    return immutableRegionValue({ gates: [], unmatchedProbability: 1, omittedRegionProbability: 0 }) as RegionGateDistributionV1
  }
  const maximumLogit = Math.max(unmatchedLogit, ...drafts.map((draft) => draft.logit))
  const unmatchedExp = Math.exp(unmatchedLogit - maximumLogit)
  const expValues = drafts.map((draft) => Math.exp(draft.logit - maximumLogit))
  const denominator = unmatchedExp + expValues.reduce((sum, value) => sum + value, 0)
  const withProbability = drafts.map((draft, index): RegionGateV1 => ({
    ...draft,
    probability: expValues[index] / denominator,
  })).sort((left, right) => right.probability - left.probability || compareRegionText(left.regionId, right.regionId))
  const retained = withProbability.slice(0, maximumActiveRegions)
  const omittedRegionProbability = withProbability.slice(maximumActiveRegions)
    .reduce((sum, gate) => sum + gate.probability, 0)
  const unmatchedProbability = unmatchedExp / denominator + omittedRegionProbability
  return immutableRegionValue({
    gates: retained.map((gate) => ({ ...gate, probability: canonicalRegionFloat(gate.probability) })),
    unmatchedProbability: canonicalRegionFloat(unmatchedProbability),
    omittedRegionProbability: canonicalRegionFloat(omittedRegionProbability),
  }) as RegionGateDistributionV1
}

export function computePageWriteMask(
  atlas: PlasticRegionAtlasV1,
  distribution: RegionGateDistributionV1,
  minimumTrace = 0.02,
): readonly number[] {
  validateAdaptiveRegionAtlas(atlas)
  if (!Number.isFinite(minimumTrace) || minimumTrace < 0 || minimumTrace > 1) {
    throw new RangeError('Asgari mikro iz [0,1] aralığında olmalıdır.')
  }
  const total = distribution.unmatchedProbability + distribution.gates.reduce((sum, gate) => sum + gate.probability, 0)
  if (Math.abs(total - 1) > 1e-5) throw new RangeError('Bölge kapısı dağılımı 1 toplamını korumuyor.')
  const probabilities = new Map(distribution.gates.map((gate) => [gate.regionId, gate.probability]))
  const result = atlas.pages.map((page) => {
    const regional = page.memberships.reduce(
      (sum, membership) => sum + membership.weight * (probabilities.get(membership.regionId) ?? 0),
      0,
    )
    const matched = clampRegion(regional + distribution.unmatchedProbability * page.freeMass)
    return canonicalRegionFloat(minimumTrace + (1 - minimumTrace) * matched)
  })
  return immutableRegionValue(result) as readonly number[]
}

export interface RegionAwareCosineResult {
  readonly similarity: number
  readonly pageCompatibilities: readonly number[]
  readonly pageWeights: readonly number[]
  readonly gates: RegionGateDistributionV1
}

export function regionAwareCosine(
  atlas: PlasticRegionAtlasV1,
  leftRelation: ArrayLike<number>,
  rightRelation: ArrayLike<number>,
  gateInput: Omit<ComputeRegionGatesInput, 'pageCompatibilities'> = {},
): RegionAwareCosineResult {
  assertFiniteVector(leftRelation, ADAPTIVE_RELATION_LENGTH, -1, 1, 'Sol R alanı')
  assertFiniteVector(rightRelation, ADAPTIVE_RELATION_LENGTH, -1, 1, 'Sağ R alanı')
  const pageCompatibilities = Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_, pageIndex) =>
    cosineRegionVectors(extractRelationPage(leftRelation, pageIndex), extractRelationPage(rightRelation, pageIndex)))
  const gates = computeRegionGates(atlas, { ...gateInput, pageCompatibilities })
  const gateProbabilities = new Map(gates.gates.map((gate) => [gate.regionId, gate.probability]))
  let pageWeights = atlas.pages.map((page) =>
    page.freeMass * gates.unmatchedProbability + page.memberships.reduce(
      (sum, membership) => sum + membership.weight * (gateProbabilities.get(membership.regionId) ?? 0),
      0,
    ))
  if (pageWeights.every((weight) => weight <= REGION_MATH_EPSILON)) pageWeights = pageWeights.map(() => 1)
  let dot = 0
  let leftSquared = 0
  let rightSquared = 0
  for (let pageIndex = 0; pageIndex < ADAPTIVE_PAGE_COUNT; pageIndex += 1) {
    const weight = pageWeights[pageIndex]
    const leftPage = extractRelationPage(leftRelation, pageIndex)
    const rightPage = extractRelationPage(rightRelation, pageIndex)
    for (let cellIndex = 0; cellIndex < leftPage.length; cellIndex += 1) {
      dot += weight * leftPage[cellIndex] * rightPage[cellIndex]
      leftSquared += weight * leftPage[cellIndex] * leftPage[cellIndex]
      rightSquared += weight * rightPage[cellIndex] * rightPage[cellIndex]
    }
  }
  let similarity = 0
  if (leftSquared <= REGION_MATH_EPSILON && rightSquared <= REGION_MATH_EPSILON) similarity = 1
  else if (leftSquared > REGION_MATH_EPSILON && rightSquared > REGION_MATH_EPSILON) {
    similarity = clampRegion(dot / Math.sqrt(leftSquared * rightSquared), -1, 1)
  }
  return immutableRegionValue({
    similarity: canonicalRegionFloat(similarity),
    pageCompatibilities: pageCompatibilities.map(canonicalRegionFloat),
    pageWeights: pageWeights.map(canonicalRegionFloat),
    gates,
  }) as RegionAwareCosineResult
}
