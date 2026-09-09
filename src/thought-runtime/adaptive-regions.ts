import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_IDENTITY_LENGTH,
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_REGION_GEOMETRY,
  ADAPTIVE_RELATION_LENGTH,
  computeAllPageBehaviors,
  computeAllPageObservations,
  computePageWriteMask,
  createPlasticRegionAtlas,
  regionAwareCosine,
  relationIndexFromPage,
  validateAdaptiveRegionAtlas,
  type PlasticRegionAtlasV1,
} from '../adaptive-regions'
import { ADAPTIVE_TEXT_LAB_PROFILE_ID } from '../text-lab/domain/config'
import type { TextLabRepresentationProfileId } from '../text-lab/domain/types'
import { thoughtDigest, thoughtVectorDigest } from './deterministic'
import { assertThoughtVector, clamp, positiveCompatibility } from './math'
import type {
  ThoughtAdaptiveBranchTelemetryV1,
  ThoughtAdaptiveCandidateTelemetryV1,
  ThoughtAdaptiveCompatibilityTelemetryV1,
  ThoughtAdaptiveRegionBindingV1,
  ThoughtAdaptiveRegionContextV1,
  ThoughtAdaptiveRegionRequestV1,
  ThoughtEpisodeState,
  ThoughtMatrixLayout,
  ThoughtVector,
} from './types'

export const DEFAULT_ACTIVE_REGION_COMPATIBILITY_BLEND = 0.25
export const MAXIMUM_ACTIVE_REGION_COMPATIBILITY_BLEND = 0.5
export const DEFAULT_ACTIVE_REGION_MINIMUM_TRACE = 0.02
export const MAXIMUM_ADAPTIVE_CANDIDATE_TELEMETRY = 8

function nonEmpty(value: string | null | undefined, label: string): string {
  const result = String(value ?? '').trim()
  if (!result) throw new TypeError(`${label} boş olamaz.`)
  return result
}

function assertTargetLayout(layout: ThoughtMatrixLayout): void {
  if (
    layout.side !== ADAPTIVE_REGION_GEOMETRY.side ||
    layout.cellCount !== ADAPTIVE_REGION_GEOMETRY.cellCount ||
    layout.identityLength !== ADAPTIVE_IDENTITY_LENGTH ||
    layout.relationStart !== ADAPTIVE_IDENTITY_LENGTH ||
    layout.relationLength !== ADAPTIVE_RELATION_LENGTH
  ) {
    throw new TypeError(
      `Uyarlanabilir bölge kipi yalnız ${ADAPTIVE_IDENTITY_LENGTH}/${ADAPTIVE_RELATION_LENGTH} I/R yerleşiminde çalışır.`,
    )
  }
}

function bounded(value: number | undefined, fallback: number, maximum: number, label: string): number {
  const result = value ?? fallback
  if (!Number.isFinite(result) || result < 0 || result > maximum) {
    throw new RangeError(`${label} 0..${maximum} aralığında olmalıdır.`)
  }
  return result
}

function boundedInteger(value: number | undefined, fallback: number, maximum: number, label: string): number {
  const result = value ?? fallback
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) {
    throw new RangeError(`${label} 1..${maximum} aralığında tam sayı olmalıdır.`)
  }
  return result
}

function bindingPayload(
  atlas: PlasticRegionAtlasV1,
  mode: ThoughtAdaptiveRegionBindingV1['mode'],
  sourceGroupId: string,
  compatibilityBlend: number,
  minimumTrace: number,
  maximumActiveGates?: number,
): Omit<ThoughtAdaptiveRegionBindingV1, 'fingerprint'> {
  return {
    format: 'thought-adaptive-region-binding',
    version: 1,
    mode,
    atlasFingerprint: atlas.fingerprint,
    atlasRevision: atlas.revision,
    representationSpaceId: atlas.representationSpaceId,
    geometryFingerprint: atlas.geometryFingerprint,
    sourceGroupId,
    compatibilityBlend,
    minimumTrace,
    ...(mode === 'active' ? { maximumActiveGates } : {}),
  }
}

/**
 * Resolves an untrusted request against the trusted snapshot atlas. Returning
 * null is the complete legacy/off path: no metadata is introduced into hashes.
 */
export function resolveThoughtAdaptiveRegionContext(input: {
  request?: ThoughtAdaptiveRegionRequestV1 | null
  atlas?: PlasticRegionAtlasV1 | null
  layout: ThoughtMatrixLayout
  representationSpaceId: string
  representationProfile?: TextLabRepresentationProfileId | null
  sourceGroupId: string
}): ThoughtAdaptiveRegionContextV1 | null {
  const request = input.request
  if (!request || request.mode === 'off') return null
  if (request.mode !== 'shadow' && request.mode !== 'proposal' && request.mode !== 'active') {
    throw new TypeError(`Bilinmeyen uyarlanabilir bölge kipi: ${String(request.mode)}`)
  }
  if (input.representationProfile !== ADAPTIVE_TEXT_LAB_PROFILE_ID) {
    throw new TypeError(
      'Uyarlanabilir bölge kipi explicit adaptive-eighth-id-v1 snapshot profil kanıtı gerektirir.',
    )
  }
  assertTargetLayout(input.layout)
  const suppliedAtlas = input.atlas
  if (!suppliedAtlas) throw new TypeError(`${request.mode} kipi snapshot'a bağlı tam bir bölge atlası gerektirir.`)
  validateAdaptiveRegionAtlas(suppliedAtlas)
  // Rebuild through the canonical constructor so a mutable deserialized input
  // cannot change topology after the episode binding has been calculated.
  const atlas = createPlasticRegionAtlas({
    representationSpaceId: suppliedAtlas.representationSpaceId,
    revision: suppliedAtlas.revision,
    parentFingerprint: suppliedAtlas.parentFingerprint,
    behaviorDimension: suppliedAtlas.behaviorDimension,
    maximumMembershipsPerPage: suppliedAtlas.maximumMembershipsPerPage,
    regions: suppliedAtlas.regions,
    pages: suppliedAtlas.pages,
  })
  const requestedFingerprint = nonEmpty(request.atlasFingerprint, 'İstenen bölge atlası fingerprinti')
  if (requestedFingerprint !== atlas.fingerprint) {
    throw new TypeError('İstenen bölge atlası fingerprinti snapshot atlasıyla birebir uyuşmuyor.')
  }
  const representationSpaceId = nonEmpty(input.representationSpaceId, 'Temsil uzayı kimliği')
  if (atlas.representationSpaceId !== representationSpaceId) {
    throw new TypeError('Bölge atlası FamilyRun temsil uzayıyla birebir uyuşmuyor.')
  }
  if (atlas.geometryFingerprint !== ADAPTIVE_REGION_GEOMETRY.fingerprint) {
    throw new TypeError('Bölge atlası hedef 8/56 geometrisine ait değil.')
  }
  const compatibilityBlend = request.mode === 'active'
    ? bounded(
      request.compatibilityBlend,
      DEFAULT_ACTIVE_REGION_COMPATIBILITY_BLEND,
      MAXIMUM_ACTIVE_REGION_COMPATIBILITY_BLEND,
      'Etkin bölge uyum harmanı',
    )
    : 0
  if (request.mode === 'active' && compatibilityBlend <= 0) {
    throw new RangeError('Etkin bölge uyum harmanı sıfırdan büyük olmalıdır.')
  }
  const minimumTrace = bounded(
    request.minimumTrace,
    DEFAULT_ACTIVE_REGION_MINIMUM_TRACE,
    1,
    'Asgari mikro-sayfa izi',
  )
  if (request.mode === 'active' && minimumTrace <= 0) {
    throw new RangeError('Etkin kipte asgari mikro-sayfa izi sıfırdan büyük olmalıdır.')
  }
  const maximumActiveGates = request.mode === 'active'
    ? boundedInteger(request.maximumActiveGates, 4, ADAPTIVE_PAGE_COUNT, 'Etkin bölge kapısı sınırı')
    : undefined
  const sourceGroupId = nonEmpty(input.sourceGroupId, 'Uyarlanabilir bölge kaynak grubu')
  const withoutFingerprint = bindingPayload(
    atlas,
    request.mode,
    sourceGroupId,
    compatibilityBlend,
    minimumTrace,
    maximumActiveGates,
  )
  const binding: ThoughtAdaptiveRegionBindingV1 = {
    ...withoutFingerprint,
    fingerprint: thoughtDigest('adaptive-region-binding', withoutFingerprint),
  }
  return {
    binding,
    atlas,
    sourceGroupId,
  }
}

export function cloneThoughtAdaptiveRegionBinding(
  binding: ThoughtAdaptiveRegionBindingV1,
): ThoughtAdaptiveRegionBindingV1 {
  return { ...binding }
}

export function validateThoughtAdaptiveRegionBinding(binding: ThoughtAdaptiveRegionBindingV1): void {
  if (binding.format !== 'thought-adaptive-region-binding' || binding.version !== 1) {
    throw new TypeError('Uyarlanabilir bölge bağlama biçimi/version 1 olmalıdır.')
  }
  if (binding.mode !== 'shadow' && binding.mode !== 'proposal' && binding.mode !== 'active') {
    throw new TypeError('Uyarlanabilir bölge bağlamı non-off bir kip taşımalıdır.')
  }
  nonEmpty(binding.atlasFingerprint, 'Bağlı atlas fingerprinti')
  nonEmpty(binding.representationSpaceId, 'Bağlı temsil uzayı')
  nonEmpty(binding.geometryFingerprint, 'Bağlı geometri fingerprinti')
  nonEmpty(binding.sourceGroupId, 'Bağlı kaynak grubu')
  if (!Number.isSafeInteger(binding.atlasRevision) || binding.atlasRevision < 0) {
    throw new RangeError('Bağlı atlas revisionı negatif olmayan güvenli tam sayı olmalıdır.')
  }
  const maximumBlend = binding.mode === 'active' ? MAXIMUM_ACTIVE_REGION_COMPATIBILITY_BLEND : 0
  bounded(binding.compatibilityBlend, 0, maximumBlend, 'Bağlı uyum harmanı')
  if (binding.mode === 'active' && binding.compatibilityBlend <= 0) {
    throw new RangeError('Etkin bağlı uyum harmanı sıfırdan büyük olmalıdır.')
  }
  bounded(binding.minimumTrace, 0, 1, 'Bağlı asgari mikro-sayfa izi')
  if (binding.mode === 'active' && binding.minimumTrace <= 0) {
    throw new RangeError('Etkin bağlı asgari mikro-sayfa izi sıfırdan büyük olmalıdır.')
  }
  if (binding.mode === 'active') {
    boundedInteger(binding.maximumActiveGates, 4, ADAPTIVE_PAGE_COUNT, 'Bağlı etkin bölge kapısı sınırı')
  } else if (binding.maximumActiveGates !== undefined) {
    throw new TypeError('Shadow/proposal bağlamı etkin kapı sınırı taşıyamaz.')
  }
  const { fingerprint, ...withoutFingerprint } = binding
  if (fingerprint !== thoughtDigest('adaptive-region-binding', withoutFingerprint)) {
    throw new TypeError('Uyarlanabilir bölge binding fingerprinti içeriğiyle uyuşmuyor.')
  }
}

export function validateThoughtAdaptiveRegionContext(
  context: ThoughtAdaptiveRegionContextV1,
  layout: ThoughtMatrixLayout,
): void {
  assertTargetLayout(layout)
  validateThoughtAdaptiveRegionBinding(context.binding)
  validateAdaptiveRegionAtlas(context.atlas)
  if (
    context.binding.atlasFingerprint !== context.atlas.fingerprint ||
    context.binding.atlasRevision !== context.atlas.revision ||
    context.binding.representationSpaceId !== context.atlas.representationSpaceId ||
    context.binding.geometryFingerprint !== context.atlas.geometryFingerprint ||
    context.binding.sourceGroupId !== context.sourceGroupId
  ) {
    throw new TypeError('Uyarlanabilir bölge çalışma bağlamı bağlı atlas/kanıt kaynağıyla uyuşmuyor.')
  }
}

/** Rejects a changed/omitted atlas before any branch state is touched. */
export function assertThoughtAdaptiveRegionEpisodeBinding(
  episode: Pick<ThoughtEpisodeState, 'adaptiveRegion'>,
  context: ThoughtAdaptiveRegionContextV1 | null | undefined,
  layout: ThoughtMatrixLayout,
): void {
  const binding = episode.adaptiveRegion
  if (!binding) {
    if (context) throw new TypeError('Off episode uyarlanabilir bölge bağlamıyla sürdürülemez.')
    return
  }
  if (!context) throw new TypeError('Uyarlanabilir bölge episode’u tam atlas bağlamı olmadan sürdürülemez.')
  validateThoughtAdaptiveRegionContext(context, layout)
  validateThoughtAdaptiveRegionBinding(binding)
  if (
    context.binding.fingerprint !== binding.fingerprint ||
    context.binding.mode !== binding.mode ||
    context.binding.atlasFingerprint !== binding.atlasFingerprint ||
    context.binding.atlasRevision !== binding.atlasRevision ||
    context.binding.representationSpaceId !== binding.representationSpaceId ||
    context.binding.geometryFingerprint !== binding.geometryFingerprint ||
    context.binding.sourceGroupId !== binding.sourceGroupId ||
    context.binding.compatibilityBlend !== binding.compatibilityBlend ||
    context.binding.minimumTrace !== binding.minimumTrace ||
    context.binding.maximumActiveGates !== binding.maximumActiveGates ||
    context.atlas.fingerprint !== binding.atlasFingerprint ||
    context.atlas.revision !== binding.atlasRevision ||
    context.atlas.representationSpaceId !== binding.representationSpaceId ||
    context.atlas.geometryFingerprint !== binding.geometryFingerprint ||
    context.sourceGroupId !== binding.sourceGroupId
  ) {
    throw new TypeError('Episode/checkpoint bölge atlası bağlamı değişmiş; deterministik replay sürdürülemez.')
  }
}

function relationPartition(vector: ThoughtVector): Float32Array {
  return vector.slice(ADAPTIVE_IDENTITY_LENGTH)
}

export function evaluateThoughtAdaptiveCompatibility(
  signal: ThoughtVector,
  representation: ThoughtVector,
  layout: ThoughtMatrixLayout,
  context: ThoughtAdaptiveRegionContextV1,
): ThoughtAdaptiveCompatibilityTelemetryV1 {
  assertTargetLayout(layout)
  assertThoughtVector(signal, 'Bölge uyumu sinyali')
  assertThoughtVector(representation, 'Bölge uyumu nöron gösterimi')
  const legacyCompatibility = positiveCompatibility(signal, representation)
  const signalRelation = relationPartition(signal)
  const representationRelation = relationPartition(representation)
  const region = regionAwareCosine(
    context.atlas,
    signalRelation,
    representationRelation,
    {
      // Only pre-routing features are supplied here; the future write delta,
      // reward and ablation result cannot leak into candidate selection.
      pageBehaviors: computeAllPageBehaviors({ signalRelation, representationRelation }),
      maximumActiveRegions: context.binding.maximumActiveGates,
    },
  )
  const regionCompatibility = clamp(region.similarity)
  const blend = context.binding.compatibilityBlend
  const blendedCompatibility = clamp((1 - blend) * legacyCompatibility + blend * regionCompatibility)
  return {
    legacyCompatibility,
    regionCompatibility,
    blendedCompatibility,
    effectiveCompatibility: context.binding.mode === 'active' ? blendedCompatibility : legacyCompatibility,
    pageCompatibilities: [...region.pageCompatibilities],
    pageWeights: [...region.pageWeights],
    gates: region.gates,
  }
}

/** Expands the canonical 56-page mask to the physical 3,584-cell R ordering. */
export function adaptiveRegionRelationWriteWeights(
  context: ThoughtAdaptiveRegionContextV1,
  compatibility: ThoughtAdaptiveCompatibilityTelemetryV1,
): Float32Array {
  const pageMask = computePageWriteMask(
    context.atlas,
    compatibility.gates,
    context.binding.minimumTrace,
  )
  const weights = new Float32Array(ADAPTIVE_RELATION_LENGTH)
  for (let pageIndex = 0; pageIndex < ADAPTIVE_PAGE_COUNT; pageIndex += 1) {
    for (let cellIndex = 0; cellIndex < ADAPTIVE_CELLS_PER_PAGE; cellIndex += 1) {
      weights[relationIndexFromPage(pageIndex, cellIndex)] = pageMask[pageIndex]
    }
  }
  return weights
}

function cloneCompatibility(
  value: ThoughtAdaptiveCompatibilityTelemetryV1,
): ThoughtAdaptiveCompatibilityTelemetryV1 {
  return {
    legacyCompatibility: value.legacyCompatibility,
    regionCompatibility: value.regionCompatibility,
    blendedCompatibility: value.blendedCompatibility,
    effectiveCompatibility: value.effectiveCompatibility,
    pageCompatibilities: [...value.pageCompatibilities],
    pageWeights: [...value.pageWeights],
    gates: value.gates,
  }
}

export function createThoughtAdaptiveCandidateTelemetry(
  neuronId: string,
  value: ThoughtAdaptiveCompatibilityTelemetryV1,
): ThoughtAdaptiveCandidateTelemetryV1 {
  return { neuronId, ...cloneCompatibility(value) }
}

export function createThoughtAdaptiveBranchTelemetry(input: {
  episodeId: string
  tick: number
  branchId: string
  signal: ThoughtVector
  representation: ThoughtVector
  relationDelta: Float32Array
  context: ThoughtAdaptiveRegionContextV1
  compatibility: ThoughtAdaptiveCompatibilityTelemetryV1
  candidateCompatibilities?: readonly ThoughtAdaptiveCandidateTelemetryV1[]
}): ThoughtAdaptiveBranchTelemetryV1 {
  const { context, compatibility } = input
  if (input.relationDelta.length !== ADAPTIVE_RELATION_LENGTH) {
    throw new RangeError(`Bölge gözlem R deltası ${ADAPTIVE_RELATION_LENGTH} hücre içermelidir.`)
  }
  const pageWriteMask = computePageWriteMask(
    context.atlas,
    compatibility.gates,
    context.binding.minimumTrace,
  )
  const observationPrefix = thoughtDigest(
    'adaptive-page-observation',
    input.episodeId,
    input.tick,
    input.branchId,
    context.binding.fingerprint,
    thoughtVectorDigest(input.signal),
    thoughtVectorDigest(input.representation),
  )
  const pageObservations = computeAllPageObservations({
    observationIdPrefix: observationPrefix,
    episodeId: input.episodeId,
    sourceGroupId: context.sourceGroupId,
    signalRelation: relationPartition(input.signal),
    representationRelation: relationPartition(input.representation),
    deltaRelation: input.relationDelta,
    confidence: 1,
    outcomeCredit: 0,
    ablationUtility: 0,
  })
  return {
    format: 'thought-adaptive-region-branch-telemetry',
    version: 1,
    mode: context.binding.mode,
    atlasFingerprint: context.binding.atlasFingerprint,
    bindingFingerprint: context.binding.fingerprint,
    activeApplied: context.binding.mode === 'active',
    ...cloneCompatibility(compatibility),
    pageWriteMask: [...pageWriteMask],
    pageObservations: pageObservations.map((observation) => ({
      ...observation,
      behavior: [...observation.behavior],
    })),
    candidateCompatibilities: (input.candidateCompatibilities ?? [])
      .slice(0, MAXIMUM_ADAPTIVE_CANDIDATE_TELEMETRY)
      .map((candidate) => ({ ...candidate, ...cloneCompatibility(candidate) })),
  }
}
