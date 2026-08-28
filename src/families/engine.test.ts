import { describe, expect, it } from 'vitest'
import { DEFAULT_TEXT_LAB_CONFIG } from '../text-lab/domain/config'
import { analyzeText } from '../text-lab/engine/pipeline'
import {
  createFamilyConfig,
  FamilyCancelledError,
  FamilyValidationError,
  runFamilyLearning,
} from './engine'
import type { FamilyAnalysisInput, FamilyProgress } from './types'

const textConfig = {
  ...DEFAULT_TEXT_LAB_CONFIG,
  matrixSide: 8,
  identityRatio: 0.5,
  minimumLeafGraphemes: 1,
}

async function input(id: string, rawText: string): Promise<FamilyAnalysisInput> {
  return await analyzeText({ id, title: id, rawText }, textConfig)
}

describe('nöron aileleri uçtan uca motoru', () => {
  it('profil -> M -> aile -> G -> holdout hattını deterministik ve I/R salt-okunur çalıştırır', async () => {
    const analyses = await Promise.all([
      input('source-a', 'Ali kesti.'),
      input('source-b', 'Ali kesti!'),
      input('source-c', 'Ali kesti?'),
    ])
    const identityBefore = analyses.map((analysis) => analysis.nodes.map((node) => Array.from(node.identity)))
    const relationBefore = analyses.map((analysis) => analysis.nodes.map((node) => Array.from(node.relation)))
    const config = createFamilyConfig({
      minimumFormationSimilarity: 0.1,
      minimumMembershipScore: 0,
      minimumCoreMembership: 0,
      minimumSupport: 2,
      minimumSources: 2,
      maximumFamilies: 16,
      utilityMinimumCases: 2,
      utilityAlternativeCount: 2,
    })

    const first = await runFamilyLearning({ analyses }, config)
    const reordered = await runFamilyLearning({ analyses: [...analyses].reverse() }, config)

    expect(first.profiles.length).toBeGreaterThan(0)
    expect(first.families.length).toBeGreaterThan(0)
    expect(first.memberships.length).toBeGreaterThan(0)
    expect(first.familyTransitions.length).toBeGreaterThan(0)
    expect(first.utilityReports).toHaveLength(first.families.length)
    expect(first.stats.peakMembershipsPerNeuron).toBeLessThanOrEqual(config.maximumMembershipsPerNeuron)
    expect(first.stats.peakOutgoingFamilyTransitions).toBeLessThanOrEqual(config.maximumOutgoingFamilyTransitions)
    expect(reordered.fingerprint).toBe(first.fingerprint)
    expect(analyses.map((analysis) => analysis.nodes.map((node) => Array.from(node.identity)))).toEqual(identityBefore)
    expect(analyses.map((analysis) => analysis.nodes.map((node) => Array.from(node.relation)))).toEqual(relationBefore)
  })

  it('aynı korpusu yeniden oynatmayı yeni gelişim kanıtı saymaz', async () => {
    const analyses = await Promise.all([
      input('source-a', 'Ali kesti.'),
      input('source-b', 'Ali kesti!'),
      input('source-c', 'Ali kesti?'),
    ])
    const config = createFamilyConfig({
      minimumFormationSimilarity: 0.1,
      minimumMembershipScore: 0,
      minimumCoreMembership: 0,
      minimumSupport: 2,
      minimumSources: 2,
    })
    const first = await runFamilyLearning({ analyses }, config)
    const replay = await runFamilyLearning({ analyses, previousRun: first }, config)

    expect(replay.inputManifest.corpusFingerprint).toBe(first.inputManifest.corpusFingerprint)
    expect(replay.families.some((family) => (
      family.lifecycle === 'strengthening' || family.lifecycle === 'stable'
    ))).toBe(false)
  })

  it('iptali ortak hata tipine çevirir', async () => {
    const analysis = await input('source-a', 'Ali kesti.')
    await expect(runFamilyLearning(
      { analyses: [analysis] },
      { minimumSupport: 2, minimumSources: 2 },
      { shouldCancel: () => true },
    )).rejects.toBeInstanceOf(FamilyCancelledError)
  })

  it('hesap akışı içinde iptal edilir ve bütün progress oranlarını monoton bildirir', async () => {
    const analyses = await Promise.all([
      input('source-a', 'Ali bıçakla kesti.'),
      input('source-b', 'Ayşe makasla kesti.'),
      input('source-c', 'Mert ipi kesti.'),
    ])
    const progress: FamilyProgress[] = []
    await runFamilyLearning(
      { analyses },
      {
        minimumFormationSimilarity: 0.1,
        minimumMembershipScore: 0,
        minimumCoreMembership: 0,
        minimumSupport: 2,
        minimumSources: 2,
      },
      {
        yieldControl: async () => Promise.resolve(),
        onProgress: (entry) => progress.push(entry),
      },
    )
    expect(progress[0]).toMatchObject({ stage: 'validating', ratio: 0 })
    expect(progress.at(-1)).toMatchObject({ stage: 'complete', ratio: 1 })
    for (let index = 1; index < progress.length; index += 1) {
      expect(progress[index].ratio).toBeGreaterThanOrEqual(progress[index - 1].ratio)
      expect(progress[index].ratio).toBeGreaterThanOrEqual(0)
      expect(progress[index].ratio).toBeLessThanOrEqual(1)
    }

    let checks = 0
    await expect(runFamilyLearning(
      { analyses },
      { minimumSupport: 2, minimumSources: 2 },
      {
        yieldControl: async () => Promise.resolve(),
        shouldCancel: () => {
          checks += 1
          return checks > 6
        },
      },
    )).rejects.toBeInstanceOf(FamilyCancelledError)
    expect(checks).toBeGreaterThan(6)
  })

  it('geçersiz config sınırlarını açık doğrulama hatasıyla reddeder', () => {
    expect(() => createFamilyConfig({ maximumProfileFeatures: 257, profileDimension: 256 }))
      .toThrow('davranış uzayı boyutunu aşamaz')
    expect(() => createFamilyConfig({ minimumSources: 1 }))
      .toThrow('en az iki kaynak')
    expect(() => createFamilyConfig({ minimumMembershipScore: 0.8, minimumCoreMembership: 0.7 }))
      .toThrow('Çekirdek üyelik')
  })

  it('farklı temsil uzaylarını ve bozulmuş I/H girdilerini reddeder', async () => {
    const analyses = await Promise.all([
      input('source-a', 'Ali kesti.'),
      input('source-b', 'Ayşe kesti.'),
    ])
    const otherSpace = await analyzeText(
      { id: 'source-other', title: 'Other', rawText: 'Başka bir temsil.' },
      { ...textConfig, hashSeed: textConfig.hashSeed + 1 },
    )
    await expect(runFamilyLearning(
      { analyses: [analyses[0], otherSpace] },
    )).rejects.toMatchObject({
      name: 'FamilyValidationError',
      code: 'REPRESENTATION_SPACE_MISMATCH',
    } satisfies Partial<FamilyValidationError>)

    const corruptedIdentity = structuredClone(analyses)
    corruptedIdentity[0].nodes[0].identity[0] += 0.1
    await expect(runFamilyLearning(
      { analyses: corruptedIdentity },
    )).rejects.toMatchObject({
      name: 'FamilyValidationError',
      code: 'INVALID_ANALYSIS',
    } satisfies Partial<FamilyValidationError>)

    const corruptedHabit = structuredClone(analyses)
    corruptedHabit[0].habitProbabilities[0] = Number.NaN
    await expect(runFamilyLearning(
      { analyses: corruptedHabit },
    )).rejects.toMatchObject({
      name: 'FamilyValidationError',
      code: 'INVALID_ANALYSIS',
    } satisfies Partial<FamilyValidationError>)
  })
})
