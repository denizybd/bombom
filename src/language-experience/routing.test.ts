import { describe, expect, it } from 'vitest'
import type { CoalitionState } from '../families/types'
import { createLanguageExperienceConfig } from './config'
import { buildLanguageDataset } from './corpus'
import { generateLanguageSequence, routeNextLanguageStep } from './routing'
import { createLanguageExperienceState } from './state'
import { trainLanguageExperience } from './training'
import { createFixtureFamilyRun, fixtureSources } from './testFixtures'

function contentCoalition(): CoalitionState {
  return {
    tick: 1,
    activeFamilyCount: 1,
    unmatchedReason: null,
    coalitions: [{
      id: 'thought-1',
      eventMemberKey: 'member:kesti',
      eventFamilyId: 'family:event',
      familyActivations: [{ familyId: 'family:content', activation: 1 }],
      bindings: [],
      memberKeys: ['member:kesti'],
      score: {
        activation: 1,
        coherence: 1,
        persistence: 0,
        goalRelevance: 0,
        conflict: 0,
        repetition: 0,
        cooldown: 0,
        total: 1,
      },
      dominance: 1,
      confidence: 1,
      temporary: true,
    }],
  }
}

describe('iki seviyeli dil yönlendirmesi', () => {
  it('geçici düşünce koalisyonunu önce aileye, sonra dondurulmuş aile üyelerine dağıtır', () => {
    const run = createFixtureFamilyRun()
    const state = createLanguageExperienceState(run, createLanguageExperienceConfig())
    const route = routeNextLanguageStep(state, run, { coalition: contentCoalition(), recentMemberKeys: [] })
    const total = route.neuronCandidates.reduce((sum, candidate) => sum + candidate.probability, 0) +
      route.eosProbability + route.unmatchedProbability

    expect(route.familyCandidates[0]?.familyId).toBe('family:content')
    expect(route.neuronCandidates.length).toBeGreaterThan(0)
    expect(route.neuronCandidates.every((candidate) => candidate.identityDigest)).toBe(true)
    expect(total).toBeCloseTo(1, 8)
  })

  it('yakın çıktı geçmişini ve EOS yolunu kullanarak seed ile tekrarlanabilir üretim yapar', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({ includeHierarchyLevels: ['token'] })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const trained = trainLanguageExperience(
      createLanguageExperienceState(run, config),
      run,
      dataset,
      { epochs: 2, evaluateSplits: [] },
    ).state
    const first = generateLanguageSequence(trained, run, { coalition: contentCoalition() }, {
      maximumSteps: 12,
      strategy: 'seeded-sample',
      seed: 991,
    })
    const second = generateLanguageSequence(trained, run, { coalition: contentCoalition() }, {
      maximumSteps: 12,
      strategy: 'seeded-sample',
      seed: 991,
    })

    expect(first).toEqual(second)
    expect(first.steps.length).toBeGreaterThan(0)
    expect(['eos', 'maximum-steps', 'unmatched']).toContain(first.stoppedBy)
    const learnedVocabulary = new Set(trained.neuronTransitions
      .filter((edge) => edge.excitatoryEvidence > 0 && !edge.toKey.includes('/EOS/'))
      .map((edge) => edge.toKey))
    expect(first.steps.flatMap((step) => step.route.neuronCandidates)
      .every((candidate) => learnedVocabulary.has(candidate.memberKey))).toBe(true)
  })

  it('aynı seed, veri ve başlangıç snapshotından bit düzeyinde aynı öğrenme durumunu üretir', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({ includeHierarchyLevels: ['token'], seed: 1234 })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const initial = createLanguageExperienceState(run, config)
    const first = trainLanguageExperience(initial, run, dataset, { epochs: 2, evaluateSplits: [] }).state
    const second = trainLanguageExperience(initial, run, dataset, { epochs: 2, evaluateSplits: [] }).state

    expect(first.fingerprint).toBe(second.fingerprint)
    expect(first).toEqual(second)
  })
})
