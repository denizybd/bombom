import type { MatrixLayout, TextLabConfig } from '../domain/types'
import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  resolveTextLabRepresentationProfileId,
} from '../domain/config'
import { hashInteger, hashString, signedHash } from './hash'
import { maxAbsNormalize, vectorNorm } from './matrix'
import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_PAGE_COLUMNS,
  ADAPTIVE_PAGE_COUNT,
  assertAdaptiveRelationPageLayout,
  identityPageCellIndex,
  relationPageCellIndex,
} from './relationPages'
import { segmentGraphemes } from './segment'

interface SurfaceFeature {
  key: string
  weight: number
}

const FEATURE_COORDINATE_SALTS = [0x9e3779b9, 0x7f4a7c15] as const
const FEATURE_SIGN_SALTS = [0x85ebca6b, 0xc2b2ae35] as const
const ADAPTIVE_PAGE_SALT = 0x6a09e667
const ADAPTIVE_LOCAL_SALT = 0xbb67ae85
const ADAPTIVE_PAGE_COPY_SALT = 0x3c6ef372
const ADAPTIVE_LOCAL_COPY_SALT = 0xa54ff53a

function surfaceFeatures(value: string): SurfaceFeature[] {
  const graphemes = segmentGraphemes(value).map((part) => part.segment)
  const features: SurfaceFeature[] = []

  for (let order = 1; order <= 4; order += 1) {
    const count = Math.max(0, graphemes.length - order + 1)
    if (!count) continue
    const layerWeight = 1 / Math.sqrt(count)
    for (let index = 0; index < count; index += 1) {
      const gram = graphemes.slice(index, index + order).join('')
      const relativeBucket = Math.min(7, Math.floor((index / Math.max(1, count)) * 8))
      features.push({ key: `g${order}:${gram}`, weight: layerWeight })
      features.push({ key: `p${relativeBucket}:g${order}:${gram}`, weight: layerWeight * 0.35 })
    }
  }

  if (graphemes.length) {
    features.push({ key: `^:${graphemes.slice(0, 4).join('')}`, weight: 0.8 })
    features.push({ key: `$:${graphemes.slice(-4).join('')}`, weight: 0.8 })
    features.push({ key: `len:${Math.min(31, graphemes.length)}`, weight: 0.3 })
  }
  return features
}

function encodeLegacyIdentity(
  value: string,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  const identity = new Float32Array(layout.identityLength)
  const features = surfaceFeatures(value)

  for (const feature of features) {
    for (let copy = 0; copy < FEATURE_COORDINATE_SALTS.length; copy += 1) {
      const coordinate =
        hashString(feature.key, config.hashSeed ^ FEATURE_COORDINATE_SALTS[copy]) %
        layout.identityLength
      const sign = signedHash(feature.key, config.hashSeed ^ FEATURE_SIGN_SALTS[copy])
      identity[coordinate] += (sign * feature.weight) / Math.sqrt(FEATURE_COORDINATE_SALTS.length)
    }
  }

  return maxAbsNormalize(identity)
}

/**
 * Opt-in page-aware identity encoder. Copies land on distinct 8×8 identity
 * pages, which removes page symmetry without assigning semantic labels.
 */
function encodeAdaptiveIdentity(
  value: string,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  assertAdaptiveRelationPageLayout(layout)
  const identity = new Float32Array(layout.identityLength)
  const features = surfaceFeatures(value)

  for (const feature of features) {
    const firstPage = hashString(feature.key, config.hashSeed ^ ADAPTIVE_PAGE_SALT) %
      ADAPTIVE_PAGE_COLUMNS
    const pageOffset = 1 + (
      hashString(feature.key, config.hashSeed ^ ADAPTIVE_PAGE_COPY_SALT) %
      (ADAPTIVE_PAGE_COLUMNS - 1)
    )
    const pages = [firstPage, (firstPage + pageOffset) % ADAPTIVE_PAGE_COLUMNS]
    const locals = [
      hashString(feature.key, config.hashSeed ^ ADAPTIVE_LOCAL_SALT) % ADAPTIVE_CELLS_PER_PAGE,
      hashString(feature.key, config.hashSeed ^ ADAPTIVE_LOCAL_COPY_SALT) % ADAPTIVE_CELLS_PER_PAGE,
    ]
    for (let copy = 0; copy < pages.length; copy += 1) {
      const coordinate = identityPageCellIndex(pages[copy], locals[copy])
      const sign = signedHash(feature.key, config.hashSeed ^ FEATURE_SIGN_SALTS[copy])
      identity[coordinate] += (sign * feature.weight) / Math.sqrt(pages.length)
    }
  }

  return maxAbsNormalize(identity)
}

export function encodeIdentity(
  value: string,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  return resolveTextLabRepresentationProfileId(config) === ADAPTIVE_TEXT_LAB_PROFILE_ID
    ? encodeAdaptiveIdentity(value, layout, config)
    : encodeLegacyIdentity(value, layout, config)
}

function projectLegacyIdentityToRelation(
  identity: ArrayLike<number>,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  const relation = new Float32Array(layout.relationLength)
  const identityNorm = vectorNorm(identity)
  if (identityNorm === 0) return relation

  for (let index = 0; index < identity.length; index += 1) {
    const normalizedValue = identity[index] / identityNorm
    if (normalizedValue === 0) continue
    for (let copy = 0; copy < 2; copy += 1) {
      const salt = config.hashSeed ^ (copy === 0 ? 0x27d4eb2d : 0x165667b1)
      const coordinate = hashInteger(index, salt) % layout.relationLength
      const sign = (hashInteger(index, salt ^ 0xa5a5a5a5) & 1) === 0 ? 1 : -1
      relation[coordinate] += (sign * normalizedValue) / Math.sqrt(2)
    }
  }

  return maxAbsNormalize(relation)
}

/**
 * New page-aware base projection. Every target page receives a deterministic
 * signed permutation of one identity page. The map is fixed by representation
 * inputs only and is therefore independent of learned atlas revisions.
 */
function projectAdaptiveIdentityToRelation(
  identity: ArrayLike<number>,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  assertAdaptiveRelationPageLayout(layout)
  if (identity.length !== layout.identityLength) {
    throw new RangeError('Kimlik vektörü seçilen temsil profilinin uzunluğuyla uyuşmuyor.')
  }
  const relation = new Float32Array(layout.relationLength)
  const identityNorm = vectorNorm(identity)
  if (identityNorm === 0) return relation

  for (let targetPage = 0; targetPage < ADAPTIVE_PAGE_COUNT; targetPage += 1) {
    const pageRow = Math.floor(targetPage / ADAPTIVE_PAGE_COLUMNS)
    const pageColumn = targetPage % ADAPTIVE_PAGE_COLUMNS
    const pageShift = hashInteger(pageRow, config.hashSeed ^ 0x510e527f) %
      ADAPTIVE_PAGE_COLUMNS
    const sourcePage = (pageColumn + pageShift) % ADAPTIVE_PAGE_COLUMNS
    const pageSeed = config.hashSeed ^ hashInteger(targetPage, 0x1f83d9ab)
    // Every odd value is coprime with 64, hence a page-local permutation.
    const multiplier = (hashInteger(targetPage, pageSeed ^ 0x5be0cd19) % 32) * 2 + 1
    const offset = hashInteger(targetPage, pageSeed ^ 0xcbbb9d5d) % ADAPTIVE_CELLS_PER_PAGE

    for (let localCell = 0; localCell < ADAPTIVE_CELLS_PER_PAGE; localCell += 1) {
      const sourceLocal = (
        Math.imul(localCell, multiplier) + offset
      ) % ADAPTIVE_CELLS_PER_PAGE
      const sourceIndex = identityPageCellIndex(sourcePage, sourceLocal)
      const targetIndex = relationPageCellIndex(targetPage, localCell)
      const sign = (hashInteger(localCell, pageSeed ^ 0x9b05688c) & 1) === 0 ? 1 : -1
      relation[targetIndex] = sign * identity[sourceIndex] / identityNorm / Math.sqrt(7)
    }
  }

  return maxAbsNormalize(relation)
}

export function projectIdentityToRelation(
  identity: ArrayLike<number>,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  return resolveTextLabRepresentationProfileId(config) === ADAPTIVE_TEXT_LAB_PROFILE_ID
    ? projectAdaptiveIdentityToRelation(identity, layout, config)
    : projectLegacyIdentityToRelation(identity, layout, config)
}

export function encodeSurfaceMatrix(
  value: string,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  const matrix = new Float32Array(layout.cellCount)
  matrix.set(encodeIdentity(value, layout, config), 0)
  return matrix
}
