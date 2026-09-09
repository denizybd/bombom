import {
  adaptiveRegionDigest,
  canonicalRegionFloat,
  canonicalRegionJson,
  compareRegionText,
  immutableRegionValue,
} from './deterministic.ts'
import {
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_REGION_GEOMETRY,
} from './geometry.ts'
import {
  ADAPTIVE_REGION_FORMAT,
  ADAPTIVE_REGION_VERSION,
  type PlasticPageAssignmentV1,
  type PlasticRegionAtlasV1,
  type PlasticRegionEvidenceV1,
  type PlasticRegionLifecycle,
  type PlasticRegionOrigin,
  type PlasticRegionPrototypeV1,
  type PlasticRegionV1,
} from './types.ts'

export const REGION_MASS_EPSILON = 1e-6
/** One complete 8×8 relation page-equivalent is required for released routing. */
export const MINIMUM_ACTIVE_REGION_CAPACITY = 1

const LIFECYCLES = new Set<PlasticRegionLifecycle>(['provisional', 'active', 'retiring', 'retired'])
const ORIGINS = new Set<PlasticRegionOrigin>(['discovery', 'split', 'merge'])
const VALIDATED_IMMUTABLE_ATLASES = new WeakSet<object>()

export interface CreatePlasticRegionAtlasInput {
  readonly representationSpaceId: string
  readonly revision?: number
  readonly parentFingerprint?: string | null
  readonly behaviorDimension: number
  readonly maximumMembershipsPerPage?: number
  readonly regions?: readonly PlasticRegionV1[]
  readonly pages?: readonly PlasticPageAssignmentV1[]
}

function nonEmpty(value: string, label: string): string {
  const result = String(value ?? '').trim()
  if (!result) throw new TypeError(`${label} boş olamaz.`)
  return result
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(`${label} negatif olmayan tam sayı olmalıdır.`)
  return value
}

function bounded(value: number, minimum: number, maximum: number, label: string): number {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} ${minimum}..${maximum} aralığında olmalıdır.`)
  }
  return canonicalRegionFloat(value)
}

function canonicalIds(values: readonly string[], label: string): string[] {
  const ids = values.map((value) => nonEmpty(value, label))
  const unique = [...new Set(ids)].sort(compareRegionText)
  if (unique.length !== ids.length) throw new TypeError(`${label} yinelenen değer içeremez.`)
  return unique
}

export function canonicalPlasticRegionEvidence(
  evidence: PlasticRegionEvidenceV1,
): PlasticRegionEvidenceV1 {
  const sourceGroupIds = canonicalIds(evidence.sourceGroupIds, 'Bölge kaynak grubu')
  const observationCount = nonNegativeInteger(evidence.observationCount, 'Bölge gözlem sayısı')
  if (observationCount < sourceGroupIds.length) {
    throw new RangeError('Bölge gözlem sayısı bağımsız kaynak sayısından az olamaz.')
  }
  return {
    observationCount,
    sourceGroupIds,
    confidenceMass: bounded(evidence.confidenceMass, 0, sourceGroupIds.length, 'Bölge güven kütlesi'),
    cohesion: bounded(evidence.cohesion, 0, 1, 'Bölge iç uyumu'),
    holdoutUtility: bounded(evidence.holdoutUtility, -1, 1, 'Bölge holdout faydası'),
    shamUtility: bounded(evidence.shamUtility, -1, 1, 'Bölge sham faydası'),
  }
}

export function canonicalPlasticRegionPrototype(
  prototype: PlasticRegionPrototypeV1,
  behaviorDimension: number,
): PlasticRegionPrototypeV1 {
  if (prototype.dimension !== behaviorDimension || prototype.values.length !== behaviorDimension) {
    throw new RangeError('Bölge prototipi atlas davranış boyutuyla uyuşmuyor.')
  }
  return {
    dimension: behaviorDimension,
    values: prototype.values.map((value) => bounded(value, -1, 1, 'Bölge prototipi')),
  }
}

function canonicalRegion(
  region: PlasticRegionV1,
  behaviorDimension: number,
  atlasRevision: number,
): PlasticRegionV1 {
  const lifecycle = region.lifecycle
  const origin = region.origin
  if (!LIFECYCLES.has(lifecycle)) throw new TypeError('Bilinmeyen plastik bölge yaşam durumu.')
  if (!ORIGINS.has(origin)) throw new TypeError('Bilinmeyen plastik bölge kökeni.')
  const bornRevision = nonNegativeInteger(region.bornRevision, 'Bölge doğum revisionı')
  if (bornRevision > atlasRevision) throw new RangeError('Bölge atlas revisionından sonra doğmuş görünemez.')
  const retiredRevision = region.retiredRevision === null
    ? null
    : nonNegativeInteger(region.retiredRevision, 'Bölge emeklilik revisionı')
  if (lifecycle === 'retired') {
    if (retiredRevision === null || retiredRevision > atlasRevision || retiredRevision < bornRevision) {
      throw new RangeError('Emekli bölge geçerli bir emeklilik revisionı taşımalıdır.')
    }
  } else if (retiredRevision !== null) {
    throw new TypeError('Canlı bölge emeklilik revisionı taşıyamaz.')
  }
  return {
    id: nonEmpty(region.id, 'Bölge kimliği'),
    lifecycle,
    origin,
    bornRevision,
    retiredRevision,
    parentRegionIds: canonicalIds(region.parentRegionIds, 'Üst bölge kimliği'),
    prototype: canonicalPlasticRegionPrototype(region.prototype, behaviorDimension),
    evidence: canonicalPlasticRegionEvidence(region.evidence),
  }
}

function emptyPages(): PlasticPageAssignmentV1[] {
  return Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_, pageIndex) => ({
    pageIndex,
    memberships: [],
    freeMass: 1,
  }))
}

function canonicalPage(page: PlasticPageAssignmentV1): PlasticPageAssignmentV1 {
  if (!Number.isInteger(page.pageIndex) || page.pageIndex < 0 || page.pageIndex >= ADAPTIVE_PAGE_COUNT) {
    throw new RangeError('Plastik mikro-sayfa indeksi geometri dışında.')
  }
  const seen = new Set<string>()
  const memberships = page.memberships.map((membership) => {
    const regionId = nonEmpty(membership.regionId, 'Mikro-sayfa bölge kimliği')
    if (seen.has(regionId)) throw new TypeError('Bir mikro-sayfa aynı bölgeyi iki kez içeremez.')
    seen.add(regionId)
    return {
      regionId,
      weight: bounded(membership.weight, REGION_MASS_EPSILON, 1, 'Bölge aidiyet ağırlığı'),
    }
  }).sort((left, right) => compareRegionText(left.regionId, right.regionId))
  return {
    pageIndex: page.pageIndex,
    memberships,
    freeMass: bounded(page.freeMass, 0, 1, 'Serbest mikro-sayfa kütlesi'),
  }
}

function atlasFingerprintPayload(atlas: Omit<PlasticRegionAtlasV1, 'fingerprint'>): unknown {
  return atlas
}

export function computePlasticRegionAtlasFingerprint(
  atlas: Omit<PlasticRegionAtlasV1, 'fingerprint'>,
): string {
  return adaptiveRegionDigest('PRATLAS', atlasFingerprintPayload(atlas))
}

export function createPlasticRegionAtlas(input: CreatePlasticRegionAtlasInput): PlasticRegionAtlasV1 {
  const representationSpaceId = nonEmpty(input.representationSpaceId, 'Temsil uzayı kimliği')
  const revision = nonNegativeInteger(input.revision ?? 0, 'Atlas revisionı')
  const parentFingerprint = input.parentFingerprint == null
    ? null
    : nonEmpty(input.parentFingerprint, 'Üst atlas özeti')
  if (revision === 0 && parentFingerprint !== null) {
    throw new TypeError('İlk atlas revisionı bir üst atlas özeti taşıyamaz.')
  }
  if (revision > 0 && parentFingerprint === null) {
    throw new TypeError('İlk revision dışındaki atlaslar üst atlas özeti taşımalıdır.')
  }
  if (!Number.isInteger(input.behaviorDimension) || input.behaviorDimension < 1 || input.behaviorDimension > 4096) {
    throw new RangeError('Atlas davranış boyutu 1..4096 arasında olmalıdır.')
  }
  const maximumMembershipsPerPage = input.maximumMembershipsPerPage ?? 3
  if (!Number.isInteger(maximumMembershipsPerPage) || maximumMembershipsPerPage < 1 || maximumMembershipsPerPage > 8) {
    throw new RangeError('Mikro-sayfa başına aidiyet sınırı 1..8 arasında olmalıdır.')
  }

  const regions = (input.regions ?? [])
    .map((region) => canonicalRegion(region, input.behaviorDimension, revision))
    .sort((left, right) => compareRegionText(left.id, right.id))
  const regionIds = new Set<string>()
  for (const region of regions) {
    if (regionIds.has(region.id)) throw new TypeError(`Yinelenen plastik bölge kimliği: ${region.id}`)
    regionIds.add(region.id)
  }
  for (const region of regions) {
    for (const parentId of region.parentRegionIds) {
      if (!regionIds.has(parentId) || parentId === region.id) {
        throw new TypeError(`${region.id} geçersiz bir üst bölge kimliği taşıyor.`)
      }
    }
  }

  const pages = (input.pages ?? emptyPages()).map(canonicalPage).sort((left, right) => left.pageIndex - right.pageIndex)
  if (pages.length !== ADAPTIVE_PAGE_COUNT) {
    throw new RangeError(`Atlas tam olarak ${ADAPTIVE_PAGE_COUNT} mikro-sayfa içermelidir.`)
  }
  const liveRegionIds = new Set(regions.filter((region) => region.lifecycle !== 'retired').map((region) => region.id))
  for (let index = 0; index < pages.length; index += 1) {
    const page = pages[index]
    if (page.pageIndex !== index) throw new TypeError('Atlas mikro-sayfaları eksiksiz ve sıralı olmalıdır.')
    if (page.memberships.length > maximumMembershipsPerPage) {
      throw new RangeError(`${page.pageIndex}. mikro-sayfa aidiyet sınırını aşıyor.`)
    }
    let total = page.freeMass
    for (const membership of page.memberships) {
      if (!liveRegionIds.has(membership.regionId)) {
        throw new TypeError(`${page.pageIndex}. mikro-sayfa bilinmeyen veya emekli bir bölgeye bağlı.`)
      }
      total += membership.weight
    }
    if (Math.abs(total - 1) > REGION_MASS_EPSILON) {
      throw new RangeError(`${page.pageIndex}. mikro-sayfa kütlesi 1 toplamını korumuyor.`)
    }
  }
  for (const region of regions) {
    if (region.lifecycle !== 'retired' && !pages.some((page) =>
      page.memberships.some((membership) => membership.regionId === region.id))) {
      throw new TypeError(`${region.id} canlı olmasına rağmen hiçbir mikro-sayfa kapasitesi taşımıyor.`)
    }
    if (region.lifecycle === 'active') {
      const capacity = pages.reduce(
        (sum, page) => sum + (page.memberships.find((membership) => membership.regionId === region.id)?.weight ?? 0),
        0,
      )
      if (capacity + REGION_MASS_EPSILON < MINIMUM_ACTIVE_REGION_CAPACITY) {
        throw new RangeError(
          `${region.id} aktif bölgesi en az bir tam mikro-sayfa eşdeğeri kapasite taşımalıdır.`,
        )
      }
    }
  }

  const withoutFingerprint: Omit<PlasticRegionAtlasV1, 'fingerprint'> = {
    format: ADAPTIVE_REGION_FORMAT,
    version: ADAPTIVE_REGION_VERSION,
    representationSpaceId,
    geometryFingerprint: ADAPTIVE_REGION_GEOMETRY.fingerprint,
    revision,
    parentFingerprint,
    behaviorDimension: input.behaviorDimension,
    maximumMembershipsPerPage,
    regions,
    pages,
  }
  const atlas: PlasticRegionAtlasV1 = {
    ...withoutFingerprint,
    fingerprint: computePlasticRegionAtlasFingerprint(withoutFingerprint),
  }
  const immutableAtlas = immutableRegionValue(atlas) as PlasticRegionAtlasV1
  VALIDATED_IMMUTABLE_ATLASES.add(immutableAtlas)
  return immutableAtlas
}

export function createInitialPlasticRegionAtlas(input: {
  readonly representationSpaceId: string
  readonly behaviorDimension: number
  readonly maximumMembershipsPerPage?: number
}): PlasticRegionAtlasV1 {
  return createPlasticRegionAtlas(input)
}

export function validateAdaptiveRegionAtlas(atlas: PlasticRegionAtlasV1): void {
  if (VALIDATED_IMMUTABLE_ATLASES.has(atlas)) return
  if (atlas.format !== ADAPTIVE_REGION_FORMAT || atlas.version !== ADAPTIVE_REGION_VERSION) {
    throw new TypeError('Bilinmeyen plastik bölge atlası biçimi.')
  }
  if (atlas.geometryFingerprint !== ADAPTIVE_REGION_GEOMETRY.fingerprint) {
    throw new TypeError('Plastik bölge atlası hedef 8/56 geometrisine ait değil.')
  }
  const rebuilt = createPlasticRegionAtlas({
    representationSpaceId: atlas.representationSpaceId,
    revision: atlas.revision,
    parentFingerprint: atlas.parentFingerprint,
    behaviorDimension: atlas.behaviorDimension,
    maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
    regions: atlas.regions,
    pages: atlas.pages,
  })
  const { fingerprint: _atlasFingerprint, ...atlasPayload } = atlas
  const { fingerprint: _rebuiltFingerprint, ...rebuiltPayload } = rebuilt
  if (canonicalRegionJson(atlasPayload) !== canonicalRegionJson(rebuiltPayload)) {
    throw new TypeError('Plastik bölge atlası kanonik biçimde sıralanmamış veya normalize edilmemiş.')
  }
  if (atlas.fingerprint !== rebuilt.fingerprint) throw new TypeError('Plastik bölge atlası özeti içerikle uyuşmuyor.')
}

export function plasticRegionCapacity(atlas: PlasticRegionAtlasV1, regionId: string): number {
  return atlas.pages.reduce(
    (sum, page) => sum + (page.memberships.find((membership) => membership.regionId === regionId)?.weight ?? 0),
    0,
  )
}

export function totalPlasticRegionCapacity(atlas: PlasticRegionAtlasV1): {
  readonly allocated: number
  readonly free: number
  readonly total: number
} {
  let allocated = 0
  let free = 0
  for (const page of atlas.pages) {
    free += page.freeMass
    allocated += page.memberships.reduce((sum, membership) => sum + membership.weight, 0)
  }
  return {
    allocated: canonicalRegionFloat(allocated),
    free: canonicalRegionFloat(free),
    total: canonicalRegionFloat(allocated + free),
  }
}

export function getPlasticPage(
  atlas: PlasticRegionAtlasV1,
  pageIndex: number,
): PlasticPageAssignmentV1 {
  if (!Number.isInteger(pageIndex) || pageIndex < 0 || pageIndex >= ADAPTIVE_PAGE_COUNT) {
    throw new RangeError('Mikro-sayfa indeksi geometri dışında.')
  }
  return atlas.pages[pageIndex]
}
