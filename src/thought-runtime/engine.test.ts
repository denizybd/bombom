import { describe, expect, it } from 'vitest'
import { createThoughtRuntimeConfig } from './config'
import { thoughtPartitionDigest } from './deterministic'
import {
  averageResidualPrototype,
  calculateResidualCohesion,
  selectThoughtProposalKeys,
  startThoughtEpisode,
  stepThoughtEpisode,
} from './engine'
import { createThoughtHead } from './model'
import { createThoughtRuntimeIndex } from './routing'
import { blankBranch, partition, TEST_CONFIG, TEST_LAYOUT, testModel, testNeuron } from './test-fixtures'

describe('saf düşünce episode çekirdeği', () => {
  it('seeded-stochastic kipte K-1 güçlü dalı koruyup yalnız bir artık-kütle keşif dalı ayırır', () => {
    const config = createThoughtRuntimeConfig({ maximumBranches: 3 })
    const proposals = [
      { stableKey: 'strongest', probability: 0.35 },
      { stableKey: 'second', probability: 0.25 },
      { stableKey: 'third', probability: 0.2 },
      { stableKey: 'fourth', probability: 0.15 },
      { stableKey: 'fifth', probability: 0.05 },
    ]
    const seeded = selectThoughtProposalKeys(proposals, 91, 7, config, 'seeded-stochastic')
    expect(seeded).toHaveLength(3)
    expect(seeded.slice(0, 2)).toEqual(['strongest', 'second'])
    expect(['third', 'fourth', 'fifth']).toContain(seeded[2])
    expect(selectThoughtProposalKeys(proposals, 91, 7, config, 'seeded-stochastic')).toEqual(seeded)
    expect(selectThoughtProposalKeys(proposals, 91, 7, config, 'deterministic')).toEqual([
      'strongest', 'second', 'third',
    ])
  })

  it('residual cohesion dallar arasındaki çapraz çiftleri de ölçer', () => {
    const left = blankBranch('left', 'a')
    const aligned = blankBranch('aligned', 'b')
    const opposed = blankBranch('opposed', 'c')
    for (const branch of [left, aligned, opposed]) {
      branch.residual.observationCount = 3
      branch.residual.weight = 3
      branch.residual.cohesionTotal = 3
      branch.residual.cohesionComparisons = 3
    }
    left.residual.cohesionUnitSum[0] = 3
    aligned.residual.cohesionUnitSum[0] = 3
    opposed.residual.cohesionUnitSum[0] = -3
    expect(calculateResidualCohesion([left, aligned])).toBeCloseTo(1, 12)
    expect(calculateResidualCohesion([left, opposed])).toBeLessThan(0.82)
  })

  it('newborn residual prototype’unu branch massinden bağımsız eşit branch ortalaması yapar', () => {
    const dominant = blankBranch('dominant', 'a')
    dominant.mass = 0.99
    dominant.residual.weight = 2
    dominant.residual.sum[0] = 2
    const minority = blankBranch('minority', 'b')
    minority.mass = 0.01
    minority.residual.weight = 4
    minority.residual.sum[1] = 4
    const prototype = averageResidualPrototype([dominant, minority])
    expect(prototype[0]).toBeCloseTo(0.5, 7)
    expect(prototype[1]).toBeCloseTo(0.5, 7)
  })

  it('yalnız gerçekten geçilen current nörona R_fast izi bırakır ve tam heatmap trace üretir', () => {
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 0: 0.8, 1: 0.6 })
    const { model, head } = testModel({ neurons: [a, b], rows: [{ fromNeuronId: 'a', edges: [{
      toNeuronId: 'b', priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0,
      trustMass: 0, updateCount: 0, independentSourceIds: [],
    }] }] })
    const identityBefore = model.neurons.map((neuron) => Array.from(neuron.identity))
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 4)
    const signal = new Float32Array(4096)
    signal[0] = 1
    const episode = startThoughtEpisode(model, head, index, signal, {
      episodeId: 'administrative-a', runKey: 'semantic-run-key', seed: 4, maximumTicks: 2, initialNeuronIds: ['a'],
    }, TEST_CONFIG)
    expect(episode.branches[0].currentNeuronId).toBe('a')
    const stepped = stepThoughtEpisode(episode, model, head, index, TEST_CONFIG)
    expect(stepped.episode.branches[0].fastRelationPatches.map((patch) => patch.neuronId)).toEqual(['a'])
    expect(stepped.tick.branches[0].beforeSignal).toHaveLength(4096)
    expect(stepped.tick.branches[0].afterSignal).toHaveLength(4096)
    expect(stepped.tick.branches[0].neuronCandidates[0]).toEqual(expect.objectContaining({
      baseH0: 1,
      q: expect.any(Number),
      repetitionPenalty: expect.any(Number),
      selectionProbability: expect.any(Number),
    }))
    expect(stepped.routes.every((route) => Math.abs(route.totalProbability - 1) <= 1e-6)).toBe(true)
    expect(model.neurons.map((neuron) => Array.from(neuron.identity))).toEqual(identityBefore)
  })

  it('idari episodeId değişse de branch/tick/replay sonucu aynı kalır ve kimlikler sabit uzunluktadır', () => {
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 0: 0.8, 1: 0.6 })
    const { model, head } = testModel({ neurons: [a, b], rows: [{ fromNeuronId: 'a', edges: [{
      toNeuronId: 'b', priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0,
      trustMass: 0, updateCount: 0, independentSourceIds: [],
    }] }] })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 17)
    const signal = new Float32Array(4096)
    signal[0] = 1
    const execute = (episodeId: string) => {
      let episode = startThoughtEpisode(model, head, index, signal, {
        episodeId, runKey: 'same-semantic-run', seed: 17, maximumTicks: 12, initialNeuronIds: ['a'],
      }, TEST_CONFIG)
      const ticks = []
      while (episode.status === 'active') {
        const step = stepThoughtEpisode(episode, model, head, index, TEST_CONFIG)
        episode = step.episode
        ticks.push(step.tick)
      }
      return { episode, ticks }
    }
    const left = execute('uuid-left')
    const right = execute('uuid-right')
    expect(left.episode.runKey).toBe(right.episode.runKey)
    expect(left.episode.fingerprint).toBe(right.episode.fingerprint)
    expect(left.ticks).toEqual(right.ticks)
    expect(left.episode.branches.every((branch) => branch.id.length < 80)).toBe(true)
  })

  it('üç tutarlı yüksek-pressure tick sonunda geçici nöron doğurur ve sonraki tick havuzuna alır', () => {
    const config = createThoughtRuntimeConfig({
      maximumBranches: 1,
      birthPressureMemory: 0,
      birthPressureThreshold: 0.5,
      birthPressureConsecutiveTicks: 3,
      minimumResidualObservations: 3,
    })
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 1: 1 })
    const { model, head } = testModel({ neurons: [a, b] })
    const index = createThoughtRuntimeIndex(model, head, config, 23)
    const signal = new Float32Array(4096)
    signal[2] = 1
    let episode = startThoughtEpisode(model, head, index, signal, {
      runKey: 'birth-run', seed: 23, maximumTicks: 4, initialNeuronIds: ['a'],
    }, config)
    for (let tick = 1; tick <= 3; tick += 1) {
      const step = stepThoughtEpisode(episode, model, head, index, config)
      episode = step.episode
      if (tick < 3) expect(episode.temporaryBirths).toHaveLength(0)
    }
    expect(episode.temporaryBirths).toHaveLength(1)
    const newbornId = episode.temporaryBirths[0].id
    expect(newbornId).toMatch(/^temp-/)
    const fourth = stepThoughtEpisode(episode, model, head, index, config)
    expect(fourth.routes.some((route) => route.neuronCandidates.some((candidate) => candidate.neuronId === newbornId))).toBe(true)
    expect(fourth.episode.branches.some((branch) => branch.path.includes(newbornId))).toBe(true)
  })

  it('doğumu gerçek katkı dalına yazar ve ilişkisiz paralel dalın residualını tüketmez', () => {
    const config = createThoughtRuntimeConfig({
      maximumBranches: 2,
      birthPressureMemory: 1,
      birthPressureThreshold: 0.5,
      birthPressureConsecutiveTicks: 3,
      minimumResidualObservations: 3,
    })
    const unrelatedOrigin = testNeuron('unrelated-origin', { 0: 1 }, 'family-unrelated')
    const contributorOrigin = testNeuron('contributor-origin', { 1: 1 }, 'family-contributor')
    const { model, head } = testModel({ neurons: [unrelatedOrigin, contributorOrigin] })
    const index = createThoughtRuntimeIndex(model, head, config, 29)
    const signal = new Float32Array(4096)
    signal[2] = 1
    const episode = startThoughtEpisode(model, head, index, signal, {
      runKey: 'branch-attribution-run', seed: 29, maximumTicks: 2,
      initialNeuronIds: [unrelatedOrigin.id],
    }, config)
    const unrelated = blankBranch('unrelated-parent', unrelatedOrigin.id)
    unrelated.mass = 0.5
    unrelated.signal = new Float32Array(signal)
    unrelated.pressure.smoothedBirth = 1
    unrelated.pressure.consecutiveBirthTicks = 1
    unrelated.residual.observationCount = 1
    unrelated.residual.weight = 1
    unrelated.residual.sum[3] = 1
    unrelated.residual.cohesionUnitSum[3] = 1
    unrelated.residual.originEvidence = [{ key: unrelatedOrigin.id, weight: 1 }]
    unrelated.residual.familyEvidence = [{ key: 'family-unrelated', weight: 1 }]

    const contributor = blankBranch('contributor-parent', contributorOrigin.id)
    contributor.mass = 0.5
    contributor.signal = new Float32Array(signal)
    contributor.pressure.smoothedBirth = 1
    contributor.pressure.consecutiveBirthTicks = 2
    contributor.residual.observationCount = 2
    contributor.residual.weight = 2
    contributor.residual.sum[2] = 2
    contributor.residual.cohesionUnitSum[2] = 2
    contributor.residual.cohesionTotal = 1
    contributor.residual.cohesionComparisons = 1
    contributor.residual.originEvidence = [{ key: contributorOrigin.id, weight: 2 }]
    contributor.residual.familyEvidence = [{ key: 'family-contributor', weight: 2 }]
    episode.branches = [unrelated, contributor]

    const step = stepThoughtEpisode(episode, model, head, index, config)
    expect(step.episode.temporaryBirths).toHaveLength(1)
    const newborn = step.episode.temporaryBirths[0]
    const contributorChild = step.episode.branches.find((branch) => branch.parentId === contributor.id)!
    const unrelatedChild = step.episode.branches.find((branch) => branch.parentId === unrelated.id)!
    expect(newborn.provenance.birthBranchIds).toEqual([contributorChild.id])
    expect(newborn.provenance.originEvidence?.map((entry) => entry.key)).toEqual([contributorOrigin.id])
    expect(newborn.provenance.familyEvidence?.map((entry) => entry.key)).toEqual(['family-contributor'])
    expect(contributorChild.residual.observationCount).toBe(0)
    expect(unrelatedChild.residual.observationCount).toBe(2)
    expect(step.tick.birthEvents.map((event) => event.branchId)).toEqual([contributorChild.id])
    expect(step.tick.branches.find((trace) => trace.branchId === unrelatedChild.id)?.birthEvents).toEqual([])
  })

  it('head birth sabit I alanındaki bir mutasyonu erişildiği tick öncesinde yakalar', () => {
    const base = testNeuron('base', { 0: 1 })
    const { model, head: emptyHead } = testModel({ neurons: [base] })
    const identity = partition(TEST_LAYOUT.identityLength, { 2: 1 })
    const relation = partition(TEST_LAYOUT.relationLength)
    const head = createThoughtHead(model, {
      ...emptyHead,
      births: [{
        format: 'born-neuron-v1', id: 'born', identity: Array.from(identity), relation: Array.from(relation),
        identityDigest: thoughtPartitionDigest('identity', identity),
        familyMemberships: [{ familyId: 'family-content', weight: 1 }],
        sourceGroupIds: ['s1', 's2'], trustMass: 1, bornAtTick: 1, nurseryId: 'nursery-born',
      }],
    })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 2)
    const signal = new Float32Array(4096)
    signal[2] = 1
    const episode = startThoughtEpisode(model, head, index, signal, {
      runKey: 'birth-integrity', seed: 2, maximumTicks: 1, initialNeuronIds: ['born'],
    }, TEST_CONFIG)
    index.neuronById.get('born')!.identity[2] = 0.5
    expect(() => stepThoughtEpisode(episode, model, head, index, TEST_CONFIG)).toThrow(/sabit I alanı/)
  })
})
