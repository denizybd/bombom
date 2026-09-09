import { describe, expect, it } from 'vitest'
import { createLanguageExperienceConfig } from '../language-experience/config'
import { buildLanguageDataset } from '../language-experience/corpus'
import { computeLanguageStateFingerprint, createLanguageExperienceState, validateLanguageExperienceState } from '../language-experience/state'
import { createFixtureFamilyRun, fixtureSources } from '../language-experience/testFixtures'
import type { LanguageDataset, LanguageExperienceState, NeuronBirthProposal } from '../language-experience/types'
import { createLanguageOutputScaffold, LANGUAGE_OUTPUT_FAMILY_IDS } from './scaffold'
import { createLanguageOutputHead, validateLanguageOutputHead } from './state'
import { trainLanguageOutputHead } from './training'

const assignments = fixtureSources.map((source, index) => ({
  independenceGroupId: source.sourceId,
  split: index < 4 ? 'train' as const : index === 4 ? 'validation' as const : 'test' as const,
}))

function fixture(): {
  familyRun: ReturnType<typeof createFixtureFamilyRun>
  dataset: LanguageDataset
  languageState: LanguageExperienceState
} {
  const familyRun = createFixtureFamilyRun()
  const config = createLanguageExperienceConfig({ includeHierarchyLevels: ['token'] })
  const dataset = buildLanguageDataset(familyRun, fixtureSources, config, assignments)
  const initial = createLanguageExperienceState(familyRun, config)
  const trainExamples = dataset.examples.filter((example) => example.split === 'train' && !example.eos)
  const observations = [...new Map(trainExamples.map((example) => [example.sourceDigest, example])).values()]
    .slice(0, 3)
    .map((example) => ({
      sourceDigest: example.sourceDigest,
      exampleId: example.id,
      surface: 'Yepyeni',
      normalizedSurface: 'yepyeni',
      granularity: 'token' as const,
      surprise: 3,
      novelty: 0.9,
    }))
  const sourceByDigest = new Map(dataset.sources.map((source) => [source.sourceDigest, source]))
  const proposal: NeuronBirthProposal = {
    id: 'NBP-manual-test',
    fingerprint: 'language-experience-proposal-proof',
    status: 'eligible',
    normalizedSurface: 'yepyeni',
    representativeSurface: 'Yepyeni',
    observationCount: observations.length,
    sourceCount: observations.length,
    sourceDigests: observations.map((observation) => observation.sourceDigest).sort(),
    sourceGroupIds: observations.map((observation) => sourceByDigest.get(observation.sourceDigest)!.independenceGroupId).sort(),
    meanSurprise: 3,
    meanNovelty: 0.9,
    observations,
    requiresExternalApproval: true,
  }
  const validationExample = dataset.examples.find((example) => example.split === 'validation' && !example.eos)!
  const validationOnly: NeuronBirthProposal = {
    ...proposal,
    id: 'NBP-validation-only',
    normalizedSurface: 'sızmamalı',
    representativeSurface: 'Sızmamalı',
    observationCount: 1,
    sourceCount: 1,
    sourceDigests: [validationExample.sourceDigest],
    sourceGroupIds: [sourceByDigest.get(validationExample.sourceDigest)!.independenceGroupId],
    observations: [{
      sourceDigest: validationExample.sourceDigest,
      exampleId: validationExample.id,
      surface: 'Sızmamalı',
      normalizedSurface: 'sızmamalı',
      granularity: 'token',
      surprise: 4,
      novelty: 1,
    }],
    status: 'observing',
  }
  const { fingerprint: _fingerprint, ...initialBase } = initial
  const stateBase: Omit<LanguageExperienceState, 'fingerprint'> = {
    ...initialBase,
    birthProposals: [proposal, validationOnly],
  }
  const languageState = { ...stateBase, fingerprint: computeLanguageStateFingerprint(stateBase) }
  validateLanguageExperienceState(languageState)
  return { familyRun, dataset, languageState }
}

describe('dil çıktı head eğitimi', () => {
  it('yalnız train örneklerinden seyrek E+/E−, üyelik ve nursery kanıtı üretir; upstream kayıtları değiştirmez', () => {
    const { familyRun, dataset, languageState } = fixture()
    const frozenRun = structuredClone(familyRun)
    const frozenDataset = structuredClone(dataset)
    const frozenLanguage = structuredClone(languageState)
    const scaffold = createLanguageOutputScaffold()
    const frozenScaffold = structuredClone(scaffold)
    const initial = createLanguageOutputHead(scaffold, familyRun)

    const result = trainLanguageOutputHead(initial, scaffold, dataset, languageState, familyRun)
    const trainSourceDigests = new Set(dataset.sources.filter((source) => source.split === 'train').map((source) => source.sourceDigest))
    const trainGroupIds = new Set(dataset.sources.filter((source) => source.split === 'train').map((source) => source.independenceGroupId))

    expect(result.replaySkipped).toBe(false)
    expect(result.processedExampleCount).toBeGreaterThan(0)
    expect(result.head.revision).toBe(1)
    expect(result.head.transitionEvidence.length).toBeGreaterThan(0)
    expect(result.head.transitionEvidence.some((edge) => edge.excitatoryEvidence > 0)).toBe(true)
    expect(result.head.transitionEvidence.some((edge) => edge.inhibitoryEvidence > 0)).toBe(true)
    expect(result.head.membershipEvidence.length).toBeGreaterThan(0)
    expect(result.head.membershipEvidence.some((membership) => membership.inhibitoryEvidence > 0)).toBe(true)
    expect(result.head.transitionEvidence.flatMap((edge) => edge.sourceDigests).every((digest) => trainSourceDigests.has(digest))).toBe(true)
    expect(result.head.membershipEvidence.flatMap((entry) => entry.sourceGroupIds).every((id) => trainGroupIds.has(id))).toBe(true)
    expect(result.head.nursery).toHaveLength(1)
    expect(result.head.nursery[0]).toMatchObject({ normalizedSurface: 'yepyeni', status: 'eligible', requiresValidation: true })
    expect(result.head.nursery[0].familyCandidates[0]?.familyId).toBe(LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder)
    expect(result.head.nursery.some((proposal) => proposal.normalizedSurface === 'sızmamalı')).toBe(false)
    expect(result.promotedNeuronCount).toBe(1)
    expect(result.provisionalNeuronCount).toBe(1)
    expect(result.head.training.provisionalNeuronPromotions).toBe(1)
    expect(result.head.outputNeurons).toHaveLength(1)
    expect(result.head.outputNeurons[0]).toMatchObject({
      normalizedSurface: 'yepyeni',
      familyId: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder,
      observationCount: 3,
      status: 'provisional',
      reversible: true,
      physicalIdentityAllocated: false,
    })
    expect(result.head.outputNeurons[0].sourceGroupIds).toHaveLength(3)
    expect(result.head.outputNeurons[0].meanFamilyConfidence).toBeCloseTo(0.6)
    validateLanguageOutputHead(result.head, scaffold)

    expect(familyRun).toEqual(frozenRun)
    expect(dataset).toEqual(frozenDataset)
    expect(languageState).toEqual(frozenLanguage)
    expect(scaffold).toEqual(frozenScaffold)
  })

  it('aynı taban ve veriyle deterministik sonuç verir, varsayılan olarak aynı dataseti ikinci kez saymaz', () => {
    const { familyRun, dataset, languageState } = fixture()
    const scaffold = createLanguageOutputScaffold()
    const initial = createLanguageOutputHead(scaffold, familyRun)
    const first = trainLanguageOutputHead(initial, scaffold, dataset, languageState, familyRun)
    const second = trainLanguageOutputHead(initial, scaffold, dataset, languageState, familyRun)

    expect(first).toEqual(second)
    const replay = trainLanguageOutputHead(first.head, scaffold, dataset, languageState, familyRun)
    expect(replay.replaySkipped).toBe(true)
    expect(replay.processedExampleCount).toBe(0)
    expect(replay.promotedNeuronCount).toBe(0)
    expect(replay.provisionalNeuronCount).toBe(1)
    expect(replay.head).toEqual(first.head)
  })

  it('aynı kaynak grubunun holdout bölümünü değiştiren veya farklı FamilyRun kullanan girdiyi reddeder', () => {
    const { familyRun, dataset, languageState } = fixture()
    const scaffold = createLanguageOutputScaffold()
    const first = trainLanguageOutputHead(
      createLanguageOutputHead(scaffold, familyRun), scaffold, dataset, languageState, familyRun,
    ).head
    const moved = structuredClone(dataset)
    const movedSource = moved.sources.find((source) => source.split === 'train')!
    movedSource.split = 'validation'
    moved.examples.filter((example) => example.sourceDigest === movedSource.sourceDigest)
      .forEach((example) => { example.split = 'validation' })
    moved.fingerprint = 'moved-dataset'
    expect(() => trainLanguageOutputHead(first, scaffold, moved, languageState, familyRun)).toThrow(/holdout bölüm/u)

    expect(() => trainLanguageOutputHead(first, scaffold, dataset, languageState, {
      ...familyRun,
      fingerprint: 'different-family-run',
    })).toThrow(/FamilyRun/u)

    const duplicateExample = structuredClone(dataset)
    duplicateExample.examples.push(structuredClone(duplicateExample.examples[0]))
    duplicateExample.fingerprint = 'duplicate-example-dataset'
    expect(() => trainLanguageOutputHead(
      createLanguageOutputHead(scaffold, familyRun), scaffold, duplicateExample, languageState, familyRun,
    )).toThrow(/dataset örneği/u)
  })

  it('holdouttan gelebilecek corruption üyesini negatif eğitim adayı olarak bile kullanmaz', () => {
    const { familyRun, dataset, languageState } = fixture()
    const isolated = structuredClone(dataset)
    const trainExample = isolated.examples.find((example) => example.split === 'train' && !example.eos)!
    trainExample.corruptionBindings = [{
      memberKey: 'holdout-only-member',
      score: 1,
      familyWeights: [],
      identityDigest: 'holdout-only-digest',
    }]
    isolated.fingerprint = 'holdout-corruption-proof'
    const scaffold = createLanguageOutputScaffold()
    const result = trainLanguageOutputHead(
      createLanguageOutputHead(scaffold, familyRun),
      scaffold,
      isolated,
      languageState,
      familyRun,
    )

    expect(result.head.transitionEvidence.some((edge) => edge.toKey === 'holdout-only-member')).toBe(false)
    expect(result.head.membershipEvidence.some((entry) => entry.memberKey === 'holdout-only-member')).toBe(false)
  })
})
