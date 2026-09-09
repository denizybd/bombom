import { describe, expect, it } from 'vitest'
import { consolidateThoughtEpisode, outcomeTrust } from './consolidation'
import { evaluateEventGraph, scoreEventGraph } from './evaluation'
import { createThoughtHead, createThoughtNeuron } from './model'
import { blankBranch, partition, TEST_CONFIG, TEST_LAYOUT, testModel, testNeuron, transition } from './test-fixtures'
import type { EventGraphV1, ThoughtBranchEvaluation, ThoughtEpisodeState } from './types'

function episodeWithBranch(branch = blankBranch('branch-final', 'c')): ThoughtEpisodeState {
  return {
    format: 'bombom-thought-episode', version: 1,
    episodeId: 'administrative-episode', runKey: 'semantic-run', seed: 1,
    modelFingerprint: 'filled-by-test', headRevision: 0, headFingerprint: 'filled-by-test',
    initialSignalDigest: 'signal', tick: 2, maximumTicks: 2,
    branches: [branch], status: 'stopped', replay: [], temporaryBirths: [], fingerprint: 'episode',
  }
}

function branchEvaluation(branchId: string, resultScore: number): ThoughtBranchEvaluation {
  const graph: EventGraphV1 = {
    format: 'event-graph-v1', actor: 'a', action: 'b', patient: 'c', instrument: null,
    operators: [], negated: false, goals: [],
  }
  return {
    branchId,
    eventGraph: graph,
    evaluation: {
      graphF1: resultScore,
      constraintAccuracy: resultScore,
      resultScore,
      truePositive: 1,
      falsePositive: 0,
      falseNegative: 0,
    },
  }
}

describe('sonuç kredisi ve kalıcı başlık deltası', () => {
  it('r=2*score-1 ve sondan geriye 0.90 eligibility ile exact E+ üretir', () => {
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 1: 1 })
    const c = testNeuron('c', { 2: 1 })
    const { model, head } = testModel({ neurons: [a, b, c] })
    const branch = blankBranch('branch-final', 'c')
    branch.path = ['a', 'b', 'c']
    branch.transitions = [transition(1, 'a', 'b'), transition(2, 'b', 'c')]
    const episode = episodeWithBranch(branch)
    episode.modelFingerprint = model.fingerprint
    episode.headFingerprint = head.fingerprint
    const delta = consolidateThoughtEpisode({
      episode, model, head, branchScores: [], branchEvaluations: [branchEvaluation(branch.id, 1)],
      evidence: { kind: 'human', confidence: 0.2, sourceGroupId: 'source-human' },
      learningMode: 'proposal', config: TEST_CONFIG,
    })!
    const first = delta.neuronEdges.find((edge) => edge.fromMemberKey === 'a')!
    const last = delta.neuronEdges.find((edge) => edge.fromMemberKey === 'b')!
    expect(first.excitation).toBeCloseTo(0.05 * 0.9, 12)
    expect(last.excitation).toBeCloseTo(0.05, 12)
    expect(first.excitation / last.excitation).toBeCloseTo(0.9, 12)
    expect(first.inhibition + last.inhibition).toBe(0)
    expect(first.trustMass).toBeCloseTo(0.9, 12)
    expect(last.trustMass).toBe(1)
  })

  it('aynı edge tekrarlarında trustMass dış kanıt güvenini aşmaz; updateCount ayrı kalır', () => {
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 1: 1 })
    const { model, head } = testModel({ neurons: [a, b] })
    const branch = blankBranch('branch-final', 'b')
    branch.transitions = [transition(1, 'a', 'b'), transition(2, 'a', 'b')]
    const episode = episodeWithBranch(branch)
    const delta = consolidateThoughtEpisode({
      episode, model, head, branchScores: [], branchEvaluations: [branchEvaluation(branch.id, 1)],
      evidence: { kind: 'observed', confidence: 1, sourceGroupId: 'observed-source' },
      learningMode: 'proposal', config: TEST_CONFIG,
    })!
    expect(delta.neuronEdges).toHaveLength(1)
    expect(delta.neuronEdges[0].trustMass).toBeCloseTo(0.6, 12)
    expect(delta.neuronEdges[0].updateCount).toBe(2)
    expect(delta.neuronEdges[0].excitation).toBeCloseTo(0.05 * 0.6 * 1.9, 12)
  })

  it('terfi etmemiş temp kimlikleri kalıcı edge/context deltalarından atar ve self kanıtına kredi vermez', () => {
    const a = testNeuron('a', { 0: 1 })
    const { model, head } = testModel({ neurons: [a] })
    const branch = blankBranch('branch-final', 'a')
    branch.transitions = [transition(1, 'temp-new', 'a')]
    branch.fastTraces = [{ fromNeuronId: 'a', toNeuronId: 'temp-new', excitation: 1, inhibition: 0, eligibility: 1, visits: 1 }]
    branch.fastRelationPatches = [{ neuronId: 'temp-new', delta: new Float32Array(model.layout.relationLength), eligibility: 1, visits: 1 }]
    const episode = episodeWithBranch(branch)
    const input = {
      episode, model, head, branchScores: [], branchEvaluations: [branchEvaluation(branch.id, 1)],
      learningMode: 'proposal', config: TEST_CONFIG,
    }
    const human = consolidateThoughtEpisode({
      ...input, evidence: { kind: 'human' as const, confidence: 1, sourceGroupId: 'human-source' },
    })!
    expect(human.neuronEdges).toEqual([])
    expect(human.contextPatches).toBeUndefined()
    expect(consolidateThoughtEpisode({
      ...input, evidence: { kind: 'self', confidence: 1, sourceGroupId: 'self-source' },
    })).toBeUndefined()
    expect(outcomeTrust({ kind: 'ollama-cross-checked', confidence: 1, sourceGroupId: 'ollama' })).toBe(0.25)
  })

  it('kalıcı R deltası reward, trust ve dal kütlesini korur; sıfır reward yazmaz', () => {
    const a = testNeuron('a', { 0: 1 })
    const { model, head } = testModel({ neurons: [a] })
    const relationDelta = partition(TEST_LAYOUT.relationLength, { 0: 0.8, 1: -0.4 })
    const makeDelta = (resultScore: number, kind: 'human' | 'ollama-cross-checked') => {
      const branch = blankBranch(`branch-${kind}-${resultScore}`, 'a')
      branch.mass = 0.5
      branch.fastRelationPatches = [{ neuronId: 'a', delta: relationDelta, eligibility: 1, visits: 1 }]
      const episode = episodeWithBranch(branch)
      const sibling = blankBranch(`sibling-${kind}-${resultScore}`, 'a')
      sibling.mass = 0.5
      episode.branches = [branch, sibling]
      episode.modelFingerprint = model.fingerprint
      episode.headFingerprint = head.fingerprint
      return consolidateThoughtEpisode({
        episode, model, head, branchScores: [], branchEvaluations: [
          branchEvaluation(branch.id, resultScore),
          branchEvaluation(sibling.id, resultScore),
        ],
        evidence: { kind, confidence: 1, sourceGroupId: `source-${kind}` },
        learningMode: 'proposal', config: TEST_CONFIG,
      })
    }

    const human = makeDelta(1, 'human')!.contextPatches![0]
    const ollama = makeDelta(1, 'ollama-cross-checked')!.contextPatches![0]
    expect(human.relationDelta[0]).toBeCloseTo(0.8 * 1 * 0.5, 7)
    expect(ollama.relationDelta[0]).toBeCloseTo(0.8 * 0.25 * 0.5, 7)
    expect(human.relationDelta[0] / ollama.relationDelta[0]).toBeCloseTo(4, 7)
    expect(human.relationDelta[1]).toBeCloseTo(-0.4 * 1 * 0.5, 7)

    const negative = makeDelta(0, 'human')!.contextPatches![0]
    expect(negative.relationDelta[0]).toBeCloseTo(-0.8 * 1 * 0.5, 7)
    expect(negative.relationDelta[1]).toBeCloseTo(0.4 * 1 * 0.5, 7)
    expect(makeDelta(0.5, 'human')!.contextPatches).toBeUndefined()
  })

  it('aynı semantik runKey replay’ini ikinci nursery gözlemi/trust kanıtı saymaz', () => {
    const a = testNeuron('a', { 0: 1 })
    const { model, head } = testModel({ neurons: [a] })
    const temporary = createThoughtNeuron({
      id: 'temp-idempotent',
      identity: partition(TEST_LAYOUT.identityLength, { 3: 1 }),
      relation: partition(TEST_LAYOUT.relationLength),
      familyMemberships: [{ familyId: 'nursery-provisional:first', weight: 1 }],
      provenance: { kind: 'episode-newborn', sourceId: 'semantic-run', residualCohesion: 0.95 },
    }, TEST_LAYOUT, TEST_CONFIG)
    const branch = blankBranch('branch-final', 'a')
    const episode = episodeWithBranch(branch)
    episode.temporaryBirths = [temporary]
    const evidence = { kind: 'human' as const, confidence: 1, sourceGroupId: 'source-a' }
    const evaluation = [branchEvaluation(branch.id, 1)]
    const first = consolidateThoughtEpisode({
      episode, model, head, branchScores: [], branchEvaluations: evaluation,
      evidence, learningMode: 'proposal', config: TEST_CONFIG,
    })!
    expect(first.nurseryUpserts).toHaveLength(1)
    expect(first.nurseryUpserts![0].observationCount).toBe(1)
    const headWithCandidate = createThoughtHead(model, {
      ...head,
      nursery: first.nurseryUpserts!,
    })
    const replay = consolidateThoughtEpisode({
      episode, model, head: headWithCandidate, branchScores: [], branchEvaluations: evaluation,
      evidence, learningMode: 'proposal', config: TEST_CONFIG,
    })!
    expect(replay.nurseryUpserts).toBeUndefined()
    expect(replay.births).toBeUndefined()
  })

  it('nursery kökenini baskın ilk daldan değil geçici nöronu gerçekten geçen daldan alır', () => {
    const a = testNeuron('a', { 0: 1 }, 'family-a')
    const b = testNeuron('b', { 1: 1 }, 'family-b')
    const c = testNeuron('c', { 2: 1 }, 'family-c')
    const { model, head } = testModel({ neurons: [a, b, c] })
    const temporary = createThoughtNeuron({
      id: 'temp-traversed',
      identity: partition(TEST_LAYOUT.identityLength, { 3: 1 }),
      relation: partition(TEST_LAYOUT.relationLength),
      familyMemberships: [{ familyId: 'family-b', weight: 1 }],
      provenance: {
        kind: 'episode-newborn', sourceId: 'semantic-run', residualCohesion: 0.95,
        residualObservationCount: 3,
        birthBranchIds: ['actual-branch'],
        originEvidence: [{ key: 'b', weight: 1 }],
        familyEvidence: [{ key: 'family-b', weight: 1 }],
      },
    }, TEST_LAYOUT, TEST_CONFIG)
    const dominant = blankBranch('dominant-branch', 'a')
    dominant.mass = 0.8
    const actual = blankBranch('actual-branch', 'c')
    actual.mass = 0.2
    actual.path = ['b', temporary.id, 'c']
    actual.transitions = [transition(1, 'b', temporary.id, 'family-b'), transition(2, temporary.id, 'c', 'family-c')]
    actual.fastRelationPatches = [{
      neuronId: temporary.id,
      delta: partition(TEST_LAYOUT.relationLength, { 0: 0.4 }),
      eligibility: 1,
      visits: 1,
    }]
    const episode = episodeWithBranch(dominant)
    episode.branches = [dominant, actual]
    episode.temporaryBirths = [temporary]
    const delta = consolidateThoughtEpisode({
      episode, model, head, branchScores: [],
      branchEvaluations: [branchEvaluation(dominant.id, 1), branchEvaluation(actual.id, 1)],
      evidence: { kind: 'human', confidence: 1, sourceGroupId: 'source-a' },
      learningMode: 'proposal', config: TEST_CONFIG,
    })!
    const candidate = delta.nurseryUpserts![0]
    expect(candidate.originEvidence.map((entry) => entry.key)).toEqual(['b'])
    expect(candidate.outgoingEvidence?.map((entry) => entry.key)).toEqual(['c'])
    expect(candidate.familyEvidence.map((entry) => entry.key)).toEqual(['family-b'])
    expect(candidate.relationPrototype[0]).toBeCloseTo(0.4, 7)
  })

  it('promotion entry/outgoing yolunu ve kaynak-dengeli R kanıtını doğan nörona taşır', () => {
    const b = testNeuron('b', { 1: 1 }, 'family-b')
    const c = testNeuron('c', { 2: 1 }, 'family-c')
    const { model, head: emptyHead } = testModel({ neurons: [b, c] })
    const identity = partition(TEST_LAYOUT.identityLength, { 3: 1 })
    const relation = partition(TEST_LAYOUT.relationLength)
    const temporary = createThoughtNeuron({
      id: 'temp-promote', identity, relation,
      familyMemberships: [{ familyId: 'family-b', weight: 1 }],
      provenance: {
        kind: 'episode-newborn', sourceId: 'semantic-run', residualCohesion: 0.95,
        residualObservationCount: 3,
        originEvidence: [{ key: 'b', weight: 1 }],
        familyEvidence: [{ key: 'family-b', weight: 1 }],
      },
    }, TEST_LAYOUT, TEST_CONFIG)
    const head = createThoughtHead(model, {
      ...emptyHead,
      nursery: [{
        id: 'nursery-promote', identity, relationPrototype: relation,
        identityDigest: temporary.identityDigest,
        birthEpisodeIds: ['old-run-1', 'old-run-2'], independentSourceIds: ['source-1', 'source-2'],
        observationCount: 2, residualObservationCount: 6,
        supportWeight: 2, trustMass: 1, resultScoreTotal: 2,
        originEvidence: [{ key: 'b', weight: 1 }],
        outgoingEvidence: [{ key: 'c', weight: 1 }],
        familyEvidence: [{ key: 'family-b', weight: 2 }],
        nextPromotionTick: 0, residualCohesion: 0.95,
        sourceRelationPrototypes: [
          { sourceGroupId: 'source-1', relation, observationCount: 1 },
          { sourceGroupId: 'source-2', relation, observationCount: 1 },
        ],
      }],
    })
    const branch = blankBranch('promoting-branch', 'c')
    branch.path = ['b', temporary.id, 'c']
    branch.transitions = [transition(1, 'b', temporary.id, 'family-b'), transition(2, temporary.id, 'c', 'family-c')]
    branch.fastRelationPatches = [{
      neuronId: temporary.id,
      delta: partition(TEST_LAYOUT.relationLength, { 0: 0.3 }),
      eligibility: 1,
      visits: 1,
    }]
    const episode = episodeWithBranch(branch)
    episode.temporaryBirths = [temporary]
    const delta = consolidateThoughtEpisode({
      episode, model, head, branchScores: [], branchEvaluations: [branchEvaluation(branch.id, 1)],
      evidence: { kind: 'human', confidence: 1, sourceGroupId: 'source-3' },
      learningMode: 'proposal', config: TEST_CONFIG,
    })!
    expect(delta.nurseryRemoveIds).toEqual(['nursery-promote'])
    expect(delta.births).toHaveLength(1)
    expect(delta.births![0].entryEvidence).toEqual([{ memberKey: 'b', weight: 2 }])
    expect(delta.births![0].outgoingEvidence).toEqual([{ memberKey: 'c', weight: 2 }])
    expect(delta.births![0].contextEvidence).toEqual({
      observationCount: 3,
      sourceGroupIds: ['source-1', 'source-2', 'source-3'],
    })
    expect(delta.births![0].relation[0]).toBeCloseTo(0.1, 7)
  })
})

describe('tek canonical EventGraph skoru', () => {
  it('gold null slotuna fazladan tahmini false-positive sayar ve 7 kısıtı ölçer', () => {
    const gold: EventGraphV1 = {
      format: 'event-graph-v1', actor: null, action: 'taşı', patient: null, instrument: null,
      operators: [], negated: false, goals: [],
    }
    const predicted: EventGraphV1 = { ...gold, actor: 'ada', instrument: 'kanca', operators: ['hemen'] }
    const score = scoreEventGraph(predicted, gold)
    const evaluated = evaluateEventGraph(predicted, gold, TEST_CONFIG)
    expect(score.predictedAtoms - score.matchedAtoms).toBe(3)
    expect(evaluated.falsePositive).toBe(3)
    expect(score.constraintAccuracy).toBeCloseTo(4 / 7, 12)
    expect(score.f1).toBeLessThan(1)
  })
})
