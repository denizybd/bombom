import { adaptiveRegionDigest, canonicalRegionFloat, compareRegionText, immutableRegionValue } from './deterministic.ts'
import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_RELATION_LENGTH,
  extractRelationPage,
} from './geometry.ts'
import {
  assertFiniteVector,
  clampRegion,
  cosineRegionVectors,
  meanRegionVectors,
  vectorNorm,
  weightedRegionMean,
} from './math.ts'
import type {
  PlasticPageObservationV1,
  PlasticPageSketchV1,
} from './types.ts'

export const PAGE_BEHAVIOR_DIMENSION = 8 as const

export interface ComputePageObservationInput {
  readonly observationId: string
  readonly episodeId: string
  readonly sourceGroupId: string
  readonly pageIndex: number
  readonly signalRelation: ArrayLike<number>
  readonly representationRelation: ArrayLike<number>
  readonly deltaRelation?: ArrayLike<number>
  readonly confidence?: number
  readonly outcomeCredit?: number
  readonly ablationUtility?: number
}

export interface ComputeAllPageBehaviorsInput {
  readonly signalRelation: ArrayLike<number>
  readonly representationRelation: ArrayLike<number>
  readonly deltaRelation?: ArrayLike<number>
}

function nonEmpty(value: string, label: string): string {
  const result = String(value ?? '').trim()
  if (!result) throw new TypeError(`${label} boş olamaz.`)
  return result
}

function mean(values: ArrayLike<number>): number {
  let total = 0
  for (let index = 0; index < values.length; index += 1) total += values[index]
  return total / values.length
}

function difference(left: ArrayLike<number>, right: ArrayLike<number>): number[] {
  return Array.from({ length: left.length }, (_, index) => left[index] - right[index])
}

function pageBehavior(
  signalRelation: ArrayLike<number>,
  representationRelation: ArrayLike<number>,
  pageIndex: number,
  deltaRelation?: ArrayLike<number>,
): readonly number[] {
  const signalPage = extractRelationPage(signalRelation, pageIndex)
  const representationPage = extractRelationPage(representationRelation, pageIndex)
  const residual = difference(signalPage, representationPage)
  const deltaPage = deltaRelation
    ? extractRelationPage(deltaRelation, pageIndex)
    : new Float32Array(ADAPTIVE_CELLS_PER_PAGE)
  const rootCells = Math.sqrt(ADAPTIVE_CELLS_PER_PAGE)
  const residualAlignment = vectorNorm(deltaPage) > 1e-12
    ? cosineRegionVectors(residual, deltaPage)
    : 0
  return [
    clampRegion(vectorNorm(signalPage) / rootCells),
    clampRegion(vectorNorm(representationPage) / rootCells),
    cosineRegionVectors(signalPage, representationPage),
    clampRegion(vectorNorm(residual) / (2 * rootCells)),
    clampRegion(vectorNorm(deltaPage) / (2 * rootCells)),
    clampRegion(mean(signalPage), -1, 1),
    clampRegion(mean(representationPage), -1, 1),
    residualAlignment,
  ].map(canonicalRegionFloat)
}

/**
 * Computes the pre-routing 8D behavior of all 56 physical pages. The complete
 * relation vectors are validated once, so candidate routing does not repeat a
 * 3,584-cell validation for every page.
 */
export function computeAllPageBehaviors(
  input: ComputeAllPageBehaviorsInput,
): readonly (readonly number[])[] {
  assertFiniteVector(input.signalRelation, ADAPTIVE_RELATION_LENGTH, -1, 1, 'Sinyal R alanı')
  assertFiniteVector(input.representationRelation, ADAPTIVE_RELATION_LENGTH, -1, 1, 'Nöron R alanı')
  if (input.deltaRelation) assertFiniteVector(input.deltaRelation, ADAPTIVE_RELATION_LENGTH, -2, 2, 'R delta alanı')
  return immutableRegionValue(Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_, pageIndex) =>
    pageBehavior(input.signalRelation, input.representationRelation, pageIndex, input.deltaRelation))) as
    readonly (readonly number[])[]
}

export function computePageObservation(input: ComputePageObservationInput): PlasticPageObservationV1 {
  if (!Number.isInteger(input.pageIndex) || input.pageIndex < 0 || input.pageIndex >= ADAPTIVE_PAGE_COUNT) {
    throw new RangeError('Mikro-sayfa gözlem indeksi geometri dışında.')
  }
  assertFiniteVector(input.signalRelation, ADAPTIVE_RELATION_LENGTH, -1, 1, 'Sinyal R alanı')
  assertFiniteVector(input.representationRelation, ADAPTIVE_RELATION_LENGTH, -1, 1, 'Nöron R alanı')
  if (input.deltaRelation) assertFiniteVector(input.deltaRelation, ADAPTIVE_RELATION_LENGTH, -2, 2, 'R delta alanı')
  const confidence = input.confidence ?? 1
  if (!Number.isFinite(confidence) || confidence <= 0 || confidence > 1) {
    throw new RangeError('Gözlem güveni (0,1] aralığında olmalıdır.')
  }
  const outcomeCredit = input.outcomeCredit ?? 0
  const ablationUtility = input.ablationUtility ?? 0
  if (!Number.isFinite(outcomeCredit) || outcomeCredit < -1 || outcomeCredit > 1 ||
      !Number.isFinite(ablationUtility) || ablationUtility < -1 || ablationUtility > 1) {
    throw new RangeError('Sonuç kredisi ve ablation faydası [-1,1] aralığında olmalıdır.')
  }

  const behavior = pageBehavior(
    input.signalRelation,
    input.representationRelation,
    input.pageIndex,
    input.deltaRelation,
  )
  return immutableRegionValue({
    observationId: nonEmpty(input.observationId, 'Gözlem kimliği'),
    episodeId: nonEmpty(input.episodeId, 'Episode kimliği'),
    sourceGroupId: nonEmpty(input.sourceGroupId, 'Bağımsız kaynak grubu'),
    pageIndex: input.pageIndex,
    confidence: canonicalRegionFloat(confidence),
    behavior,
    outcomeCredit: canonicalRegionFloat(outcomeCredit),
    ablationUtility: canonicalRegionFloat(ablationUtility),
  }) as PlasticPageObservationV1
}

export function computeAllPageObservations(
  input: Omit<ComputePageObservationInput, 'pageIndex' | 'observationId'> & {
    readonly observationIdPrefix: string
  },
): readonly PlasticPageObservationV1[] {
  const episodeId = nonEmpty(input.episodeId, 'Episode kimliği')
  const sourceGroupId = nonEmpty(input.sourceGroupId, 'Bağımsız kaynak grubu')
  const confidence = input.confidence ?? 1
  if (!Number.isFinite(confidence) || confidence <= 0 || confidence > 1) {
    throw new RangeError('Gözlem güveni (0,1] aralığında olmalıdır.')
  }
  const outcomeCredit = input.outcomeCredit ?? 0
  const ablationUtility = input.ablationUtility ?? 0
  if (!Number.isFinite(outcomeCredit) || outcomeCredit < -1 || outcomeCredit > 1 ||
      !Number.isFinite(ablationUtility) || ablationUtility < -1 || ablationUtility > 1) {
    throw new RangeError('Sonuç kredisi ve ablation faydası [-1,1] aralığında olmalıdır.')
  }
  const behaviors = computeAllPageBehaviors(input)
  return immutableRegionValue(behaviors.map((behavior, pageIndex) => ({
    observationId: nonEmpty(`${input.observationIdPrefix}:${pageIndex}`, 'Gözlem kimliği'),
    episodeId,
    sourceGroupId,
    pageIndex,
    confidence: canonicalRegionFloat(confidence),
    behavior,
    outcomeCredit: canonicalRegionFloat(outcomeCredit),
    ablationUtility: canonicalRegionFloat(ablationUtility),
  }))) as readonly PlasticPageObservationV1[]
}

interface GroupSketch {
  readonly sourceGroupId: string
  readonly behavior: readonly number[]
  readonly outcomeCredit: number
  readonly ablationUtility: number
  readonly confidence: number
}

function groupSketch(sourceGroupId: string, observations: readonly PlasticPageObservationV1[]): GroupSketch {
  const weighted = observations.map((observation) => ({
    values: observation.behavior,
    weight: observation.confidence,
  }))
  const confidenceTotal = observations.reduce((sum, observation) => sum + observation.confidence, 0)
  return {
    sourceGroupId,
    behavior: weightedRegionMean(weighted),
    outcomeCredit: canonicalRegionFloat(observations.reduce(
      (sum, observation) => sum + observation.outcomeCredit * observation.confidence,
      0,
    ) / confidenceTotal),
    ablationUtility: canonicalRegionFloat(observations.reduce(
      (sum, observation) => sum + observation.ablationUtility * observation.confidence,
      0,
    ) / confidenceTotal),
    confidence: canonicalRegionFloat(confidenceTotal / observations.length),
  }
}

function validateObservation(observation: PlasticPageObservationV1, behaviorDimension: number): void {
  if (!observation.observationId.trim() || !observation.episodeId.trim() || !observation.sourceGroupId.trim()) {
    throw new TypeError('Mikro-sayfa gözlemi boş kimlik taşıyamaz.')
  }
  if (!Number.isInteger(observation.pageIndex) || observation.pageIndex < 0 || observation.pageIndex >= ADAPTIVE_PAGE_COUNT) {
    throw new RangeError('Mikro-sayfa gözlemi geometri dışında.')
  }
  if (observation.behavior.length !== behaviorDimension) throw new RangeError('Gözlem davranış boyutları uyuşmuyor.')
  assertFiniteVector(observation.behavior, behaviorDimension, -1, 1, 'Gözlem davranışı')
  if (!Number.isFinite(observation.confidence) || observation.confidence <= 0 || observation.confidence > 1 ||
      !Number.isFinite(observation.outcomeCredit) || observation.outcomeCredit < -1 || observation.outcomeCredit > 1 ||
      !Number.isFinite(observation.ablationUtility) || observation.ablationUtility < -1 || observation.ablationUtility > 1) {
    throw new RangeError('Mikro-sayfa gözlem ölçüleri sınır dışında.')
  }
}

/** Each independent source group receives exactly one vote per page. */
export function aggregatePageSketches(
  observations: readonly PlasticPageObservationV1[],
  options: { readonly behaviorDimension?: number; readonly maximumEvidence?: number } = {},
): readonly PlasticPageSketchV1[] {
  if (!observations.length) return Object.freeze([])
  const behaviorDimension = options.behaviorDimension ?? observations[0].behavior.length
  const maximumEvidence = options.maximumEvidence ?? 64
  if (!Number.isInteger(maximumEvidence) || maximumEvidence < 1) {
    throw new RangeError('Saklanan gözlem kanıtı sınırı pozitif tam sayı olmalıdır.')
  }
  const observationIds = new Set<string>()
  for (const observation of observations) {
    validateObservation(observation, behaviorDimension)
    if (observationIds.has(observation.observationId)) throw new TypeError('Yinelenen mikro-sayfa gözlem kimliği.')
    observationIds.add(observation.observationId)
  }
  const byPage = new Map<number, PlasticPageObservationV1[]>()
  for (const observation of observations) {
    const values = byPage.get(observation.pageIndex) ?? []
    values.push(observation)
    byPage.set(observation.pageIndex, values)
  }

  const sketches: PlasticPageSketchV1[] = []
  for (const [pageIndex, pageObservations] of [...byPage].sort(([left], [right]) => left - right)) {
    const bySource = new Map<string, PlasticPageObservationV1[]>()
    for (const observation of pageObservations) {
      const values = bySource.get(observation.sourceGroupId) ?? []
      values.push(observation)
      bySource.set(observation.sourceGroupId, values)
    }
    const groups = [...bySource.entries()]
      .sort(([left], [right]) => compareRegionText(left, right))
      .map(([sourceGroupId, values]) => groupSketch(sourceGroupId, values))
    const behaviorPrototype = meanRegionVectors(groups.map((group) => group.behavior))
    const cohesion = groups.reduce(
      (sum, group) => sum + Math.max(0, cosineRegionVectors(group.behavior, behaviorPrototype)),
      0,
    ) / groups.length
    const sortedEvidence = [...pageObservations].sort((left, right) =>
      compareRegionText(left.sourceGroupId, right.sourceGroupId) ||
      compareRegionText(left.observationId, right.observationId))
    const firstBySource: PlasticPageObservationV1[] = []
    const remainingEvidence: PlasticPageObservationV1[] = []
    const representedSources = new Set<string>()
    for (const observation of sortedEvidence) {
      if (!representedSources.has(observation.sourceGroupId)) {
        representedSources.add(observation.sourceGroupId)
        firstBySource.push(observation)
      } else remainingEvidence.push(observation)
    }
    const retainedEvidence = firstBySource.slice(0, maximumEvidence)
    if (retainedEvidence.length < maximumEvidence) {
      retainedEvidence.push(...remainingEvidence.slice(0, maximumEvidence - retainedEvidence.length))
    }
    const evidence = retainedEvidence.map((observation) => ({
      observationId: observation.observationId,
      episodeId: observation.episodeId,
      sourceGroupId: observation.sourceGroupId,
      support: canonicalRegionFloat(Math.max(0, observation.behavior[3] ?? 0) * observation.confidence),
    }))
    const sourceGroupIds = groups.map((group) => group.sourceGroupId)
    const withoutFingerprint = {
      pageIndex,
      behaviorDimension,
      behaviorPrototype,
      outcomeCredit: canonicalRegionFloat(groups.reduce((sum, group) => sum + group.outcomeCredit, 0) / groups.length),
      ablationUtility: canonicalRegionFloat(groups.reduce((sum, group) => sum + group.ablationUtility, 0) / groups.length),
      cohesion: canonicalRegionFloat(clampRegion(cohesion)),
      observationCount: pageObservations.length,
      sourceGroupIds,
      confidenceMass: canonicalRegionFloat(groups.reduce((sum, group) => sum + group.confidence, 0)),
      evidence,
    }
    sketches.push(immutableRegionValue({
      ...withoutFingerprint,
      fingerprint: adaptiveRegionDigest('PRSKETCH', withoutFingerprint),
    }) as PlasticPageSketchV1)
  }
  return immutableRegionValue(sketches) as readonly PlasticPageSketchV1[]
}
