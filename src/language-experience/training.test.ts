import { describe, expect, it } from 'vitest'
import { createLanguageExperienceConfig } from './config'
import { buildLanguageDataset } from './corpus'
import { memberOriginForExample, targetMemberWeights } from './examples'
import { evaluateLanguageExperience } from './metrics'
import { createLanguageExperienceState, validateLanguageExperienceState } from './state'
import { LanguageTrainingCancelledError, trainLanguageExperience } from './training'
import { createFixtureFamilyRun, fixtureSources } from './testFixtures'

describe('self-supervised dil deneyimi', () => {
  it('teacher forcing ile doğru aile/nöron yollarını güçlendirir, rakipleri inhibe eder ve holdout verisini yazmaz', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({
      includeHierarchyLevels: ['token'],
      maximumExamplesPerSource: 100,
    })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const initial = createLanguageExperienceState(run, config)
    const initialCopy = structuredClone(initial)
    const runCopy = structuredClone(run)
    const before = evaluateLanguageExperience(initial, run, dataset, 'train')
    const result = trainLanguageExperience(initial, run, dataset, {
      epochs: 3,
      evaluateSplits: ['train', 'validation', 'test'],
    })
    const after = result.metrics.train!
    const trainDigests = new Set(dataset.sources.filter((source) => source.split === 'train').map((source) => source.sourceDigest))

    expect(initial).toEqual(initialCopy)
    expect(run).toEqual(runCopy)
    expect(result.state.revision).toBe(1)
    expect(result.state.parentFingerprint).toBe(initial.fingerprint)
    expect(result.state.familyTransitions.length).toBeGreaterThan(0)
    expect(result.state.neuronTransitions.length).toBeGreaterThan(0)
    expect(result.state.neuronTransitions.some((edge) => edge.inhibitoryEvidence > 0)).toBe(true)
    expect(result.state.neuronTransitions.some((edge) => edge.toKey.includes('/EOS/'))).toBe(true)
    expect(result.state.familyTransitions.every((edge) => edge.sourceDigests.every((digest) => trainDigests.has(digest)))).toBe(true)
    expect(result.state.neuronTransitions.every((edge) => edge.sourceDigests.every((digest) => trainDigests.has(digest)))).toBe(true)
    expect(after.negativeLogLikelihood).toBeLessThan(before.negativeLogLikelihood)
    expect(Number.isFinite(result.metrics.validation!.perplexityLike)).toBe(true)
    expect(Number.isFinite(result.metrics.test!.roleSwapSensitivity)).toBe(true)
    expect(result.metrics.train!.grammarCorruptionCaseCount).toBeGreaterThan(0)
    expect(result.metrics.train!.topK.every((metric) => metric.accuracy >= 0 && metric.accuracy <= 1)).toBe(true)
    validateLanguageExperienceState(result.state)
  })

  it('modelin önceki seçimi yanlış olsa bile sonraki bağlamı gerçek hedeften kurar', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({ includeHierarchyLevels: ['token'] })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const state = createLanguageExperienceState(run, config)
    const example = dataset.examples.find((candidate) => candidate.split === 'train' && candidate.prefixPieceIds.length > 0 && candidate.targetBindings.length)!
    const origin = memberOriginForExample(example, dataset)
    const targets = new Set(targetMemberWeights(example).map((target) => target.key))
    const trained = trainLanguageExperience(state, run, dataset, { epochs: 1, evaluateSplits: [] }).state

    expect(trained.neuronTransitions.some((edge) => (
      edge.fromKey === origin && targets.has(edge.toKey) && edge.excitatoryEvidence > 0
    ))).toBe(true)
  })

  it('yüksek sürprizli ve uyumsuz yüzeyi yalnız onay gerektiren doğum önerisine dönüştürür', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({
      includeHierarchyLevels: ['token'],
      surpriseThreshold: 0.1,
      noveltyThreshold: 0.5,
      birthMinimumObservations: 2,
      birthMinimumSources: 2,
    })
    const original = buildLanguageDataset(run, fixtureSources, config)
    const dataset = structuredClone(original)
    const perSource = new Map<string, number>()
    for (const example of dataset.examples) {
      if (example.split !== 'train' || example.eos || (perSource.get(example.sourceDigest) ?? 0) > 0) continue
      example.targetText = 'Zarg'
      example.targetNormalizedText = 'zarg'
      example.targetBindings = []
      perSource.set(example.sourceDigest, 1)
      if (perSource.size === 2) break
    }
    const result = trainLanguageExperience(
      createLanguageExperienceState(run, config),
      run,
      dataset,
      { evaluateSplits: [] },
    )
    const proposal = result.birthProposals.find((candidate) => candidate.normalizedSurface === 'zarg')

    expect(proposal).toMatchObject({
      status: 'eligible',
      observationCount: 2,
      sourceCount: 2,
      requiresExternalApproval: true,
    })
    expect(run.profiles.some((profile) => profile.text === 'Zarg')).toBe(false)
  })

  it('iptalde kısmi plastisiteyi dışarı sızdırmaz ve grup-split oynanmasını reddeder', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({ includeHierarchyLevels: ['token'] })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const initial = createLanguageExperienceState(run, config)
    const before = structuredClone(initial)

    expect(() => trainLanguageExperience(initial, run, dataset, {
      shouldCancel: () => true,
      evaluateSplits: [],
    })).toThrow(LanguageTrainingCancelledError)
    expect(initial).toEqual(before)

    const leaked = structuredClone(dataset)
    const trainSource = leaked.sources.find((source) => source.split === 'train')!
    const testSource = leaked.sources.find((source) => source.split === 'test')!
    testSource.independenceGroupId = trainSource.independenceGroupId
    expect(() => trainLanguageExperience(initial, run, leaked, { evaluateSplits: [] })).toThrow(/sızmış/u)
  })

  it('aynı kitabın iki chunkını doğum için iki bağımsız kaynak saymaz', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({
      includeHierarchyLevels: ['token'],
      surpriseThreshold: 0.1,
      noveltyThreshold: 0.5,
      birthMinimumObservations: 2,
      birthMinimumSources: 2,
    })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const trainSources = dataset.sources.filter((source) => source.split === 'train').slice(0, 2)
    trainSources.forEach((source) => { source.independenceGroupId = 'aynı-fiziksel-kitap' })
    for (const source of trainSources) {
      const example = dataset.examples.find((candidate) => (
        candidate.sourceDigest === source.sourceDigest && !candidate.eos
      ))!
      example.targetText = 'Yeniörnek'
      example.targetNormalizedText = 'yeniörnek'
      example.targetBindings = []
    }
    const result = trainLanguageExperience(
      createLanguageExperienceState(run, config), run, dataset, { evaluateSplits: [] },
    )
    const proposal = result.birthProposals.find((candidate) => candidate.normalizedSurface === 'yeniörnek')

    expect(proposal).toMatchObject({ status: 'observing', observationCount: 2, sourceCount: 1 })
  })

  it('aynı kaynak partisini açık replay izni olmadan ikinci kez ağırlıklandırmaz', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({ includeHierarchyLevels: ['token'] })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const first = trainLanguageExperience(
      createLanguageExperienceState(run, config), run, dataset, { evaluateSplits: [] },
    ).state
    const second = trainLanguageExperience(first, run, dataset, { evaluateSplits: [] }).state

    expect(second.fingerprint).toBe(first.fingerprint)
    expect(second.revision).toBe(first.revision)
    expect(second.training.processedExamples).toBe(first.training.processedExamples)
  })
})
