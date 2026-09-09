import { stableDigest } from '../families/sparse.ts'
import {
  CONCEPT_MACRO_EXECUTION_SEMANTICS,
  type MacroConfigV1,
} from './types.ts'

export const DEFAULT_MACRO_CONFIG: Readonly<MacroConfigV1> = Object.freeze({
  executionSemantics: CONCEPT_MACRO_EXECUTION_SEMANTICS,
  minimumMotifLength: 2,
  maximumMotifLength: 8,
  maximumHeavyHitters: 2_048,
  maximumEvidencePerCandidate: 64,
  maximumVariantsPerCandidate: 16,
  maximumPrototypeGroupsPerCandidate: 4,
  maximumComposites: 2_048,
  maximumVectorDimension: 4_096,
  maximumExpandedConstituents: 512,
  minimumProposalObservations: 3,
  minimumIndependentGroups: 2,
  minimumEquivalenceVariants: 2,
  minimumCompressionGain: 1,
  macroDefinitionOverhead: 1,
  minimumEffectCohesion: 0.82,
  minimumEntryCompatibility: 0.8,
  minimumShadowCases: 3,
  minimumShadowIndependentGroups: 2,
  minimumShadowPositiveCases: 1,
  minimumShadowNegativeCases: 1,
  minimumShadowFidelity: 0.98,
  maximumFalseActivationRate: 0.05,
  minimumActivationRecall: 0.95,
  minimumHoldoutUtility: 0,
  minimumShadowCompressionGain: 0,
  maximumNestingDepth: 3,
  maximumExactReplayKeys: 4_096,
  replayBloomBitCount: 262_144,
  replayBloomHashCount: 7,
})

export function createMacroConfig(overrides: Partial<MacroConfigV1> = {}): MacroConfigV1 {
  const config = { ...DEFAULT_MACRO_CONFIG, ...overrides }
  validateMacroConfig(config)
  return config
}

export function validateMacroConfig(config: MacroConfigV1): void {
  if (config.executionSemantics !== CONCEPT_MACRO_EXECUTION_SEMANTICS) {
    throw new TypeError('Makro yürütme semantiği terminal-anchor-v1 olmalıdır.')
  }
  const integers = [
    config.minimumMotifLength,
    config.maximumMotifLength,
    config.maximumHeavyHitters,
    config.maximumEvidencePerCandidate,
    config.maximumVariantsPerCandidate,
    config.maximumPrototypeGroupsPerCandidate,
    config.maximumComposites,
    config.maximumVectorDimension,
    config.maximumExpandedConstituents,
    config.minimumProposalObservations,
    config.minimumIndependentGroups,
    config.minimumEquivalenceVariants,
    config.maximumNestingDepth,
    config.minimumShadowCases,
    config.maximumExactReplayKeys,
    config.replayBloomBitCount,
    config.replayBloomHashCount,
    config.minimumShadowIndependentGroups,
    config.minimumShadowPositiveCases,
    config.minimumShadowNegativeCases,
  ]
  if (integers.some((value) => !Number.isInteger(value) || value < 1)) {
    throw new RangeError('Makro kapasite ve eşikleri pozitif tam sayı olmalıdır.')
  }
  if (config.minimumMotifLength < 2 || config.maximumMotifLength > 8) {
    throw new RangeError('Makro motif uzunluğu 2 ile 8 arasında olmalıdır.')
  }
  if (config.minimumMotifLength > config.maximumMotifLength) {
    throw new RangeError('En küçük motif uzunluğu en büyük uzunluğu aşamaz.')
  }
  if (config.maximumNestingDepth > 3) {
    throw new RangeError('V1 iç içe makro derinliği en fazla 3 olabilir.')
  }
  if (config.replayBloomBitCount < 64 || config.replayBloomBitCount % 8 !== 0) {
    throw new RangeError('Replay Bloom bit sayısı en az 64 ve 8 ile bölünebilir olmalıdır.')
  }
  if (config.replayBloomHashCount > 16) {
    throw new RangeError('Replay Bloom hash sayısı en fazla 16 olabilir.')
  }
  if (config.replayBloomBitCount > 16_777_216) {
    throw new RangeError('Replay Bloom bit sayısı V1 içinde en fazla 16.777.216 olabilir.')
  }
  if (config.maximumPrototypeGroupsPerCandidate > 64 ||
    config.maximumPrototypeGroupsPerCandidate > config.maximumEvidencePerCandidate) {
    throw new RangeError('Prototip kaynak grubu sınırı kanıt kapasitesini ve V1 64 sınırını aşamaz.')
  }
  if (config.maximumEvidencePerCandidate < config.minimumProposalObservations) {
    throw new RangeError('Kanıt kapasitesi öneri gözlem eşiğinden küçük olamaz.')
  }
  const units = [
    config.minimumEffectCohesion,
    config.minimumEntryCompatibility,
    config.minimumShadowFidelity,
    config.maximumFalseActivationRate,
    config.minimumActivationRecall,
  ]
  if (units.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new RangeError('Makro benzerlik ve hata eşikleri 0 ile 1 arasında olmalıdır.')
  }
  const finite = [
    config.minimumCompressionGain,
    config.macroDefinitionOverhead,
    config.minimumHoldoutUtility,
    config.minimumShadowCompressionGain,
  ]
  if (finite.some((value) => !Number.isFinite(value))) {
    throw new RangeError('Makro MDL ve shadow eşikleri sonlu olmalıdır.')
  }
  if (config.minimumCompressionGain < 0 || config.macroDefinitionOverhead < 0) {
    throw new RangeError('Makro MDL maliyetleri negatif olamaz.')
  }
}

export function computeMacroConfigFingerprint(config: MacroConfigV1): string {
  validateMacroConfig(config)
  return `MCFG-${stableDigest(['concept-macro-config-v1', config]).toUpperCase()}`
}
