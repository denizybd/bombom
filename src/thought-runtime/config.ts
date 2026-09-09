import { THOUGHT_MATRIX_SIDE, type ThoughtRuntimeConfig } from './types'

/** Approved v1 operating point. All limits are deterministic hard caps. */
export const DEFAULT_THOUGHT_RUNTIME_CONFIG: Readonly<ThoughtRuntimeConfig> = Object.freeze({
  matrixSide: THOUGHT_MATRIX_SIDE,
  defaultTicks: 64,
  maximumTicks: 512,
  signalNormMaximum: 1,
  repetitionWindow: 8,
  repetitionPenalty: 0.18,
  explorationReserve: 0.02,
  maximumHabitEdgesPerNeuron: 96,
  maximumTotalHabitEdges: 250_000,
  maximumCandidatePool: 128,
  maximumEvaluatedCandidates: 32,
  maximumCandidateFamilies: 8,
  minimumBranches: 1,
  maximumBranches: 8,
  minimumBranchProbability: 1e-7,
  fastEligibilityDecay: 0.9,
  fastExcitationRate: 0.05,
  fastInhibitionRate: 0.025,
  evidenceCeiling: 50,
  birthPressureMemory: 0.8,
  birthPressureThreshold: 0.65,
  birthPressureConsecutiveTicks: 3,
  minimumResidualObservations: 3,
  maximumBirthsPerEpisode: 2,
  nurseryCohesionThreshold: 0.82,
  nurseryMinimumObservations: 3,
  nurseryMinimumIndependentSources: 2,
  nurseryMinimumTrustMass: 1,
  promotionCooldownTicks: 32,
  minimumOutcomeScore: 0.8,
  outcomeGraphWeight: 0.7,
  outcomeConstraintWeight: 0.3,
  maximumFamilyMembershipsPerNeuron: 5,
  hashSeed: 0x6a09e667,
  selectionMode: 'deterministic',
})

const POSITIVE_INTEGER_KEYS: Array<keyof ThoughtRuntimeConfig> = [
  'defaultTicks', 'maximumTicks', 'repetitionWindow', 'maximumHabitEdgesPerNeuron',
  'maximumTotalHabitEdges', 'maximumCandidatePool', 'maximumEvaluatedCandidates',
  'maximumCandidateFamilies', 'minimumBranches', 'maximumBranches',
  'birthPressureConsecutiveTicks', 'minimumResidualObservations', 'maximumBirthsPerEpisode',
  'nurseryMinimumObservations', 'nurseryMinimumIndependentSources', 'promotionCooldownTicks',
  'maximumFamilyMembershipsPerNeuron',
]

const UNIT_INTERVAL_KEYS: Array<keyof ThoughtRuntimeConfig> = [
  'signalNormMaximum', 'repetitionPenalty', 'explorationReserve', 'fastEligibilityDecay',
  'fastExcitationRate', 'fastInhibitionRate',
  'birthPressureMemory', 'birthPressureThreshold', 'nurseryCohesionThreshold',
  'minimumOutcomeScore', 'outcomeGraphWeight', 'outcomeConstraintWeight',
]

export function validateThoughtRuntimeConfig(config: ThoughtRuntimeConfig): void {
  if (config.matrixSide !== THOUGHT_MATRIX_SIDE) {
    throw new RangeError('Düşünce motoru yalnız tam 64x64 S matrisiyle çalışır.')
  }
  for (const key of POSITIVE_INTEGER_KEYS) {
    const value = config[key]
    if (!Number.isInteger(value) || (value as number) < 1) {
      throw new RangeError(`${String(key)} pozitif bir tam sayı olmalıdır.`)
    }
  }
  for (const key of UNIT_INTERVAL_KEYS) {
    const value = config[key] as number
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      throw new RangeError(`${String(key)} 0 ile 1 arasında olmalıdır.`)
    }
  }
  for (const key of [
    'minimumBranchProbability', 'evidenceCeiling', 'nurseryMinimumTrustMass',
  ] as const) {
    if (!Number.isFinite(config[key]) || config[key] <= 0) {
      throw new RangeError(`${key} pozitif ve sonlu olmalıdır.`)
    }
  }
  if (config.defaultTicks > config.maximumTicks || config.maximumTicks > 512) {
    throw new RangeError('Tick bütçesi varsayılan ≤ azami ≤ 512 olmalıdır.')
  }
  if (config.maximumEvaluatedCandidates > config.maximumCandidatePool) {
    throw new RangeError('Değerlendirilen aday sayısı aday havuzunu aşamaz.')
  }
  if (config.maximumHabitEdgesPerNeuron > 96 || config.maximumTotalHabitEdges > 250_000 ||
      config.maximumCandidatePool > 128 || config.maximumEvaluatedCandidates > 32 ||
      config.maximumCandidateFamilies > 8) {
    throw new RangeError('v1 seyrek sınırları edge≤96/global≤250000/pool≤128/eval≤32/family≤8 olmalıdır.')
  }
  if (config.maximumBranches > 8 || config.minimumBranches > config.maximumBranches) {
    throw new RangeError('Uyarlanabilir dal sayısı 1–8 aralığında olmalıdır.')
  }
  if (Math.abs(config.explorationReserve - 0.02) > 1e-8) {
    throw new RangeError('v1 keşif rezervi tam 0.02 olmalıdır.')
  }
  if (Math.abs(config.signalNormMaximum - 1) > 1e-8) {
    throw new RangeError('v1 sinyal izdüşüm tavanı tam 1 olmalıdır.')
  }
  if (Math.abs(config.outcomeGraphWeight + config.outcomeConstraintWeight - 1) > 1e-8) {
    throw new RangeError('Sonuç ölçümü ağırlıkları toplamı 1 olmalıdır.')
  }
  if (config.evidenceCeiling > 50) throw new RangeError('Kanıt tavanı 50 değerini aşamaz.')
  if (!Number.isInteger(config.hashSeed)) throw new RangeError('hashSeed bir tam sayı olmalıdır.')
  if (!['deterministic', 'seeded-stochastic'].includes(config.selectionMode)) {
    throw new TypeError('selectionMode deterministic veya seeded-stochastic olmalıdır.')
  }
}

export function createThoughtRuntimeConfig(overrides: Partial<ThoughtRuntimeConfig> = {}): ThoughtRuntimeConfig {
  const config = { ...DEFAULT_THOUGHT_RUNTIME_CONFIG, ...overrides }
  validateThoughtRuntimeConfig(config)
  return config
}
