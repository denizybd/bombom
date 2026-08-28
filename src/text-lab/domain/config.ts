import type { MatrixLayout, TextLabConfig } from './types'

/** Exact all-pairs v0.1 safety boundary; data is never sampled or truncated. */
export const MAX_ANALYSIS_NODES = 1600

export const DEFAULT_TEXT_LAB_CONFIG: TextLabConfig = Object.freeze({
  matrixSide: 32,
  identityRatio: 0.25,
  minimumLeafGraphemes: 4,
  hashSeed: 0x51a7c3,
  contextRetention: 0.8,
  parentTraceWeight: 0.15,
  siblingTraceWeight: 0.05,
  minimumTrace: 0.1,
  identitySimilarityWeight: 0.5,
  habitExponent: 2,
})

const CONTEXT_WEIGHT_KEYS = [
  'contextRetention',
  'parentTraceWeight',
  'siblingTraceWeight',
] as const satisfies ReadonlyArray<keyof TextLabConfig>

const CONFIG_KEYS = [
  'matrixSide',
  'identityRatio',
  'minimumLeafGraphemes',
  'hashSeed',
  ...CONTEXT_WEIGHT_KEYS,
  'minimumTrace',
  'identitySimilarityWeight',
  'habitExponent',
] as const satisfies ReadonlyArray<keyof TextLabConfig>

/**
 * Upgrades a partial legacy configuration without trusting arbitrary values.
 * Missing context weights share the remaining probability mass in the same
 * proportions as the current defaults; supplied values are never silently
 * replaced when they are invalid.
 */
export function normalizeTextLabConfig(value: unknown): TextLabConfig {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('Metin Laboratuvarı ayarları geçerli bir nesne olmalıdır.')
  }

  const source = value as Record<string, unknown>
  const normalized = { ...DEFAULT_TEXT_LAB_CONFIG }
  const hasOwn = (key: keyof TextLabConfig) => Object.prototype.hasOwnProperty.call(source, key)

  for (const key of CONFIG_KEYS) {
    if (CONTEXT_WEIGHT_KEYS.includes(key as typeof CONTEXT_WEIGHT_KEYS[number]) || !hasOwn(key)) {
      continue
    }
    if (typeof source[key] !== 'number' || !Number.isFinite(source[key])) {
      throw new RangeError(`“${key}” ayarı sonlu bir sayı olmalıdır.`)
    }
    normalized[key] = source[key]
  }

  const suppliedWeights = CONTEXT_WEIGHT_KEYS.filter(hasOwn)
  if (suppliedWeights.length > 0) {
    for (const key of suppliedWeights) {
      if (typeof source[key] !== 'number' || !Number.isFinite(source[key]) || source[key] < 0) {
        throw new RangeError(`“${key}” ayarı negatif olmayan sonlu bir sayı olmalıdır.`)
      }
      normalized[key] = source[key]
    }

    const missingWeights = CONTEXT_WEIGHT_KEYS.filter((key) => !hasOwn(key))
    const suppliedTotal = suppliedWeights.reduce((sum, key) => sum + normalized[key], 0)
    if (suppliedTotal > 1 + 1e-8) {
      throw new RangeError('Eski bağlam aktarım ağırlıklarının toplamı 1 değerini aşamaz.')
    }
    if (missingWeights.length > 0) {
      const remainder = Math.max(0, 1 - suppliedTotal)
      const defaultMissingTotal = missingWeights.reduce(
        (sum, key) => sum + DEFAULT_TEXT_LAB_CONFIG[key],
        0,
      )
      for (const key of missingWeights) {
        normalized[key] = defaultMissingTotal > 0
          ? remainder * DEFAULT_TEXT_LAB_CONFIG[key] / defaultMissingTotal
          : remainder / missingWeights.length
      }
    }
  }

  validateConfig(normalized)
  return normalized
}

export function createMatrixLayout(config: TextLabConfig): MatrixLayout {
  const side = Math.max(8, Math.trunc(config.matrixSide))
  const identityRows = Math.max(
    1,
    Math.min(side - 1, Math.round(side * config.identityRatio)),
  )
  const cellCount = side * side
  const identityLength = identityRows * side

  return {
    side,
    cellCount,
    identityRows,
    identityLength,
    relationStart: identityLength,
    relationLength: cellCount - identityLength,
  }
}

export function validateConfig(config: TextLabConfig): void {
  const numericValues = [
    config.matrixSide,
    config.identityRatio,
    config.minimumLeafGraphemes,
    config.hashSeed,
    config.contextRetention,
    config.parentTraceWeight,
    config.siblingTraceWeight,
    config.minimumTrace,
    config.identitySimilarityWeight,
    config.habitExponent,
  ]
  if (numericValues.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
    throw new RangeError('Metin Laboratuvarı ayarlarının tamamı sonlu sayı olmalıdır.')
  }
  const weights =
    config.contextRetention + config.parentTraceWeight + config.siblingTraceWeight

  if (!Number.isInteger(config.matrixSide) || config.matrixSide < 8 || config.matrixSide > 64) {
    throw new RangeError('Bu sürümde matris kenarı 8 ile 64 hücre arasında olmalıdır.')
  }
  if (!(config.identityRatio > 0 && config.identityRatio < 1)) {
    throw new RangeError('Kimlik oranı 0 ile 1 arasında olmalıdır.')
  }
  if (!Number.isInteger(config.minimumLeafGraphemes) || config.minimumLeafGraphemes < 1) {
    throw new RangeError('En küçük yaprak uzunluğu pozitif bir tam sayı olmalıdır.')
  }
  if (!Number.isInteger(config.hashSeed)) {
    throw new RangeError('Hash seed bir tam sayı olmalıdır.')
  }
  if (Math.abs(weights - 1) > 1e-8) {
    throw new RangeError('Bağlam aktarım ağırlıklarının toplamı 1 olmalıdır.')
  }
  if (
    config.contextRetention < 0 ||
    config.parentTraceWeight < 0 ||
    config.siblingTraceWeight < 0
  ) {
    throw new RangeError('Bağlam aktarım ağırlıkları negatif olamaz.')
  }
  if (!(config.minimumTrace >= 0 && config.minimumTrace <= 1)) {
    throw new RangeError('Asgari iz 0 ile 1 arasında olmalıdır.')
  }
  if (!(config.identitySimilarityWeight > 0 && config.identitySimilarityWeight <= 1)) {
    throw new RangeError('Kimlik benzerliği ağırlığı 0 ile 1 arasında olmalıdır.')
  }
  if (!(config.habitExponent > 0)) {
    throw new RangeError('Bağ kanıtı üssü pozitif olmalıdır.')
  }
}
