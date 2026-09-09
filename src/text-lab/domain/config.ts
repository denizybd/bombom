import type {
  MatrixLayout,
  TextLabConfig,
  TextLabRepresentationProfileId,
} from './types'

/** Exact all-pairs v0.1 safety boundary; data is never sampled or truncated. */
export const MAX_ANALYSIS_NODES = 1600

export const LEGACY_TEXT_LAB_PROFILE_ID = 'legacy-quarter-id-v2' as const
export const ADAPTIVE_TEXT_LAB_PROFILE_ID = 'adaptive-eighth-id-v1' as const
export const DEFAULT_TEXT_LAB_REPRESENTATION_PROFILE_ID = LEGACY_TEXT_LAB_PROFILE_ID

export interface TextLabRepresentationProfile {
  id: TextLabRepresentationProfileId
  matrixSide: 64
  identityRows: 16 | 8
  identityRatio: 0.25 | 0.125
  relationRows: 48 | 56
  relationPageSide: 8 | null
  relationPageCount: 56 | null
}

/**
 * Named physical contracts. Learned atlas state deliberately does not live in
 * this registry, so an atlas revision can never change base text encoding.
 */
export const TEXT_LAB_REPRESENTATION_PROFILES: Readonly<
  Record<TextLabRepresentationProfileId, Readonly<TextLabRepresentationProfile>>
> = Object.freeze({
  [LEGACY_TEXT_LAB_PROFILE_ID]: Object.freeze({
    id: LEGACY_TEXT_LAB_PROFILE_ID,
    matrixSide: 64,
    identityRows: 16,
    identityRatio: 0.25,
    relationRows: 48,
    relationPageSide: null,
    relationPageCount: null,
  }),
  [ADAPTIVE_TEXT_LAB_PROFILE_ID]: Object.freeze({
    id: ADAPTIVE_TEXT_LAB_PROFILE_ID,
    matrixSide: 64,
    identityRows: 8,
    identityRatio: 0.125,
    relationRows: 56,
    relationPageSide: 8,
    relationPageCount: 56,
  }),
})

export const DEFAULT_TEXT_LAB_CONFIG: TextLabConfig = Object.freeze({
  // TextLab v2 is the canonical encoder for the thought runtime. New analyses
  // therefore start in the runtime's only supported representation space.
  // Stored projects keep their explicit matrixSide, so this does not migrate
  // or rewrite legacy 32x32 analyses.
  matrixSide: 64,
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

export function resolveTextLabRepresentationProfileId(
  config: Pick<TextLabConfig, 'representationProfile'>,
): TextLabRepresentationProfileId {
  return config.representationProfile ?? DEFAULT_TEXT_LAB_REPRESENTATION_PROFILE_ID
}

export function getTextLabRepresentationProfile(
  configOrId: Pick<TextLabConfig, 'representationProfile'> | TextLabRepresentationProfileId,
): Readonly<TextLabRepresentationProfile> {
  const id = typeof configOrId === 'string'
    ? configOrId
    : resolveTextLabRepresentationProfileId(configOrId)
  if (!Object.prototype.hasOwnProperty.call(TEXT_LAB_REPRESENTATION_PROFILES, id)) {
    throw new RangeError(`Bilinmeyen Metin Laboratuvarı temsil profili: ${String(id)}`)
  }
  const profile = TEXT_LAB_REPRESENTATION_PROFILES[id]
  return profile
}

type ProfileConfigOverrides = Partial<Omit<
  TextLabConfig,
  'representationProfile' | 'matrixSide' | 'identityRatio'
>>

/** Creates an explicit profile selection without changing the legacy default. */
export function createTextLabConfigForProfile(
  profileId: TextLabRepresentationProfileId,
  overrides: ProfileConfigOverrides = {},
): TextLabConfig {
  const profile = getTextLabRepresentationProfile(profileId)
  const config: TextLabConfig = {
    ...DEFAULT_TEXT_LAB_CONFIG,
    ...overrides,
    representationProfile: profile.id,
    matrixSide: profile.matrixSide,
    identityRatio: profile.identityRatio,
  }
  validateConfig(config)
  return config
}

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
  const normalized: TextLabConfig = { ...DEFAULT_TEXT_LAB_CONFIG }
  const hasOwn = (key: keyof TextLabConfig) => Object.prototype.hasOwnProperty.call(source, key)

  if (Object.prototype.hasOwnProperty.call(source, 'representationProfile')) {
    const profile = source.representationProfile
    if (
      profile !== LEGACY_TEXT_LAB_PROFILE_ID &&
      profile !== ADAPTIVE_TEXT_LAB_PROFILE_ID
    ) {
      throw new RangeError(`Bilinmeyen Metin Laboratuvarı temsil profili: ${String(profile)}`)
    }
    const contract = getTextLabRepresentationProfile(profile)
    normalized.representationProfile = profile
    if (!hasOwn('matrixSide')) normalized.matrixSide = contract.matrixSide
    if (!hasOwn('identityRatio')) normalized.identityRatio = contract.identityRatio
  }

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
  const profileId = resolveTextLabRepresentationProfileId(config)
  if (!Object.prototype.hasOwnProperty.call(TEXT_LAB_REPRESENTATION_PROFILES, profileId)) {
    throw new RangeError(`Bilinmeyen Metin Laboratuvarı temsil profili: ${String(profileId)}`)
  }
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

  // Historical/profile-less configurations intentionally remain flexible so
  // stored 8/16/32/64-side experiments can still be opened and replayed.
  if (config.representationProfile) {
    const profile = getTextLabRepresentationProfile(config)
    const identityRows = Math.round(config.matrixSide * config.identityRatio)
    if (
      config.matrixSide !== profile.matrixSide ||
      identityRows !== profile.identityRows ||
      config.identityRatio !== profile.identityRatio
    ) {
      throw new RangeError(
        `“${profile.id}” profili ${profile.matrixSide}×${profile.matrixSide} ve ` +
        `${profile.identityRows} kimlik satırı gerektirir.`,
      )
    }
  }
}
