import type { MatrixLayout, TextLabConfig } from '../domain/types'
import { hashInteger, hashString, signedHash } from './hash'
import { maxAbsNormalize, vectorNorm } from './matrix'
import { segmentGraphemes } from './segment'

interface SurfaceFeature {
  key: string
  weight: number
}

const FEATURE_COORDINATE_SALTS = [0x9e3779b9, 0x7f4a7c15] as const
const FEATURE_SIGN_SALTS = [0x85ebca6b, 0xc2b2ae35] as const

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

export function encodeIdentity(
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

export function projectIdentityToRelation(
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

export function encodeSurfaceMatrix(
  value: string,
  layout: MatrixLayout,
  config: TextLabConfig,
): Float32Array {
  const matrix = new Float32Array(layout.cellCount)
  matrix.set(encodeIdentity(value, layout, config), 0)
  return matrix
}

