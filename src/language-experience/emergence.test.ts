import { describe, expect, it } from 'vitest'
import { createLanguageExperienceConfig } from './config'
import { buildLanguageDataset } from './corpus'
import { discoverEmergentLanguageFamilies } from './emergence'
import { createLanguageExperienceState } from './state'
import { trainLanguageExperience } from './training'
import { createFixtureFamilyRun, fixtureSources } from './testFixtures'

describe('kendiliğinden ortaya çıkan dil aileleri', () => {
  it('elle dil etiketi vermeden geçiş imzasını kümeler ve holdout ablation faydasını kaydeder', () => {
    const run = createFixtureFamilyRun()
    const frozenBefore = structuredClone(run)
    const config = createLanguageExperienceConfig({
      includeHierarchyLevels: ['token'],
      emergentFamilyMinimumMembers: 1,
      emergentFamilyMinimumSourceGroups: 1,
      emergentFamilyUtilityThreshold: 0,
      emergentFamilySimilarityThreshold: 0.6,
    })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const trained = trainLanguageExperience(
      createLanguageExperienceState(run, config),
      run,
      dataset,
      { epochs: 4, evaluateSplits: [] },
    ).state
    const discovered = discoverEmergentLanguageFamilies(trained, run, dataset)
    const heldOutCandidate = discovered.find((family) => family.heldOutCaseCount > 0)

    expect(discovered).toEqual(trained.emergentLanguageFamilies)
    expect(discovered.length).toBeGreaterThan(0)
    expect(heldOutCandidate).toBeDefined()
    expect(heldOutCandidate?.outgoingSignature.length).toBeGreaterThan(0)
    expect(Number.isFinite(heldOutCandidate?.predictiveUtility)).toBe(true)
    expect(heldOutCandidate?.explanation).toMatch(/elle etiketlenmedi|gözleniyor/u)
    expect(run).toEqual(frozenBefore)
  })

  it('aynı kitabın çok sayıdaki chunkını çoklu bağımsız kaynak kanıtı saymaz', () => {
    const groupedSources = fixtureSources.map((source, index) => ({
      ...source,
      independenceGroupId: index < 4 ? 'book-train' : index === 4 ? 'book-validation' : 'book-test',
    }))
    const run = createFixtureFamilyRun(groupedSources)
    const config = createLanguageExperienceConfig({
      includeHierarchyLevels: ['token'],
      emergentFamilyMinimumMembers: 1,
      emergentFamilyMinimumSourceGroups: 2,
      emergentFamilyUtilityThreshold: 0,
      emergentFamilySimilarityThreshold: 0.6,
    })
    const dataset = buildLanguageDataset(run, groupedSources, config, [
      { independenceGroupId: 'book-train', split: 'train' },
      { independenceGroupId: 'book-validation', split: 'validation' },
      { independenceGroupId: 'book-test', split: 'test' },
    ])
    const trained = trainLanguageExperience(
      createLanguageExperienceState(run, config),
      run,
      dataset,
      { epochs: 4, evaluateSplits: [] },
    ).state

    expect(trained.emergentLanguageFamilies.length).toBeGreaterThan(0)
    expect(trained.emergentLanguageFamilies.every((family) => family.sourceGroupCount <= 1)).toBe(true)
    expect(trained.emergentLanguageFamilies.every((family) => family.status === 'observing')).toBe(true)
  })

  it('tekrarlanan davranış iki bağımsız eğitim kaynağı ve holdout faydası taşıyınca aileyi uygun kılar', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({
      includeHierarchyLevels: ['token'],
      emergentFamilyMinimumMembers: 1,
      emergentFamilyMinimumSourceGroups: 2,
      emergentFamilyUtilityThreshold: 0,
      emergentFamilySimilarityThreshold: 0.6,
    })
    const dataset = buildLanguageDataset(run, fixtureSources, config, fixtureSources.map((source, index) => ({
      independenceGroupId: source.sourceId,
      split: index < 4 ? 'train' as const : index === 4 ? 'validation' as const : 'test' as const,
    })))
    const trained = trainLanguageExperience(
      createLanguageExperienceState(run, config),
      run,
      dataset,
      { epochs: 4, evaluateSplits: [] },
    ).state
    const eligible = trained.emergentLanguageFamilies.filter((family) => family.status === 'eligible')

    expect(eligible.length).toBeGreaterThan(0)
    expect(eligible.every((family) => family.sourceGroupCount >= 2)).toBe(true)
    expect(eligible.every((family) => family.heldOutCaseCount > 0)).toBe(true)
    expect(eligible.every((family) => family.predictiveUtility >= 0)).toBe(true)
  })
})
