import type { LanguageExperienceConfig, SurfacePieceLevel } from './types'

export const DEFAULT_LANGUAGE_EXPERIENCE_CONFIG: Readonly<LanguageExperienceConfig> = Object.freeze<LanguageExperienceConfig>({
  seed: 0x243f6a88,
  trainRatio: 0.8,
  validationRatio: 0.1,
  testRatio: 0.1,
  maximumPrefixPieces: 8,
  maximumExamplesPerSource: 20_000,
  maximumPieceBindings: 4,
  minimumAlignmentScore: 0.45,
  minimumSubwordGraphemes: 3,
  includeGraphemeExperiences: false,
  includeHierarchyLevels: ['paragraph', 'sentence', 'clause', 'token', 'subword'],
  maximumOutgoingFamilyTransitions: 48,
  maximumOutgoingNeuronTransitions: 96,
  maximumRouteFamilies: 16,
  maximumRouteNeurons: 32,
  learningRate: 0.22,
  inhibitionRate: 0.12,
  evidenceDecay: 0.0005,
  evidenceCeiling: 1_000,
  learnedFamilyWeight: 0.42,
  learnedNeuronWeight: 0.5,
  frozenFamilyPriorWeight: 0.08,
  coalitionWeight: 0.2,
  membershipWeight: 0.22,
  historyWeight: 0.14,
  repetitionPenalty: 0.18,
  routeTemperature: 0.48,
  noveltyThreshold: 0.72,
  surpriseThreshold: 2.4,
  birthMinimumObservations: 3,
  birthMinimumSources: 2,
  birthMergeSurfaceSimilarity: 0.94,
  maximumBirthProposals: 2_000,
  maximumBirthObservationsPerProposal: 32,
  emergentFamilySimilarityThreshold: 0.72,
  emergentFamilyMinimumMembers: 2,
  emergentFamilyMinimumSourceGroups: 2,
  emergentFamilyUtilityThreshold: 0.005,
  maximumEmergentLanguageFamilies: 64,
  metricTopK: [1, 3, 5],
})

const LEVELS: readonly SurfacePieceLevel[] = [
  'document', 'paragraph', 'sentence', 'clause', 'token', 'subword', 'grapheme',
]

function positiveInteger(value: number): boolean {
  return Number.isInteger(value) && value > 0
}

export function validateLanguageExperienceConfig(config: LanguageExperienceConfig): void {
  if (!Number.isInteger(config.seed)) throw new TypeError('Dil deneyimi seed değeri tam sayı olmalıdır.')
  if (typeof config.includeGraphemeExperiences !== 'boolean') {
    throw new TypeError('Grapheme deneyimi seçimi boolean olmalıdır.')
  }
  const ratioTotal = config.trainRatio + config.validationRatio + config.testRatio
  if (
    !Number.isFinite(ratioTotal) || Math.abs(ratioTotal - 1) > 1e-9 ||
    config.trainRatio <= 0 || config.validationRatio <= 0 || config.testRatio <= 0
  ) throw new RangeError('Eğitim, doğrulama ve test oranları pozitif olmalı ve toplamları 1 olmalıdır.')
  const positiveIntegers: Array<keyof LanguageExperienceConfig> = [
    'maximumPrefixPieces',
    'maximumExamplesPerSource',
    'maximumPieceBindings',
    'minimumSubwordGraphemes',
    'maximumOutgoingFamilyTransitions',
    'maximumOutgoingNeuronTransitions',
    'maximumRouteFamilies',
    'maximumRouteNeurons',
    'birthMinimumObservations',
    'birthMinimumSources',
    'maximumBirthProposals',
    'maximumBirthObservationsPerProposal',
    'emergentFamilyMinimumMembers',
    'emergentFamilyMinimumSourceGroups',
    'maximumEmergentLanguageFamilies',
  ]
  if (positiveIntegers.some((key) => !positiveInteger(config[key] as number))) {
    throw new RangeError('Dil deneyimi kapasite sınırları pozitif tam sayı olmalıdır.')
  }
  const unitValues: Array<keyof LanguageExperienceConfig> = [
    'minimumAlignmentScore',
    'evidenceDecay',
    'learnedFamilyWeight',
    'learnedNeuronWeight',
    'frozenFamilyPriorWeight',
    'coalitionWeight',
    'membershipWeight',
    'historyWeight',
    'repetitionPenalty',
    'noveltyThreshold',
    'birthMergeSurfaceSimilarity',
    'emergentFamilySimilarityThreshold',
  ]
  if (unitValues.some((key) => {
    const value = config[key] as number
    return !Number.isFinite(value) || value < 0 || value > 1
  })) throw new RangeError('Dil deneyimi ağırlıkları ve birim eşikleri 0 ile 1 arasında olmalıdır.')
  if (
    !Number.isFinite(config.learningRate) || config.learningRate <= 0 ||
    !Number.isFinite(config.inhibitionRate) || config.inhibitionRate <= 0 ||
    !Number.isFinite(config.evidenceCeiling) || config.evidenceCeiling <= 0 ||
    !Number.isFinite(config.routeTemperature) || config.routeTemperature <= 0 ||
    !Number.isFinite(config.surpriseThreshold) || config.surpriseThreshold <= 0
    || !Number.isFinite(config.emergentFamilyUtilityThreshold) || config.emergentFamilyUtilityThreshold < 0
  ) throw new RangeError('Öğrenme, inhibisyon, sürpriz ve sıcaklık değerleri pozitif olmalıdır.')
  if (
    !Array.isArray(config.includeHierarchyLevels) || !config.includeHierarchyLevels.length ||
    new Set(config.includeHierarchyLevels).size !== config.includeHierarchyLevels.length ||
    config.includeHierarchyLevels.some((level) => !LEVELS.includes(level))
  ) throw new TypeError('Dil deneyimi hiyerarşi seviyeleri geçersiz veya tekrarlı.')
  if (
    !Array.isArray(config.metricTopK) || !config.metricTopK.length ||
    config.metricTopK.some((value) => !positiveInteger(value)) ||
    new Set(config.metricTopK).size !== config.metricTopK.length
  ) throw new TypeError('Top-k ölçüm seviyeleri benzersiz pozitif tam sayılar olmalıdır.')
}

export function createLanguageExperienceConfig(
  overrides: Partial<LanguageExperienceConfig> = {},
): LanguageExperienceConfig {
  const config: LanguageExperienceConfig = {
    ...DEFAULT_LANGUAGE_EXPERIENCE_CONFIG,
    ...overrides,
    includeHierarchyLevels: [...(
      overrides.includeHierarchyLevels ?? DEFAULT_LANGUAGE_EXPERIENCE_CONFIG.includeHierarchyLevels
    )],
    metricTopK: [...(overrides.metricTopK ?? DEFAULT_LANGUAGE_EXPERIENCE_CONFIG.metricTopK)],
  }
  validateLanguageExperienceConfig(config)
  return config
}
