import { describe, expect, it } from 'vitest'
import { createThoughtRuntimeConfig } from './config'
import { thoughtPartitionDigest } from './deterministic'
import { calculateThoughtPressure } from './engine'
import {
  composeNeuronRepresentation,
  cosineSimilarity,
  transformThoughtSignal,
  updateFastRelationPatch,
} from './math'
import { createThoughtHead, createThoughtNeuron } from './model'
import { createThoughtRuntimeIndex, queryThoughtSignalIndex, routeThoughtBranch, withTemporaryThoughtNeurons } from './routing'
import { blankBranch, partition, TEST_CONFIG, TEST_LAYOUT, testModel, testNeuron } from './test-fixtures'
import type { ThoughtRuntimeIndex } from './routing'
import type { ThoughtNeuron } from './types'

describe('64x64 düşünce matematiği altın değerleri', () => {
  it('alpha_t değerini tam cosine uyumu yapar ve yalnız R için hızlı iz üretir', () => {
    const neuron = testNeuron('n', { 0: 0.5, 1: Math.sqrt(0.75) })
    const signal = new Float32Array(4096)
    signal[0] = 1
    const representation = composeNeuronRepresentation(neuron.identity, neuron.relation, TEST_LAYOUT)
    const identityBefore = Array.from(neuron.identity)
    const transformed = transformThoughtSignal(signal, representation, TEST_CONFIG)
    expect(transformed.compatibility).toBeCloseTo(0.5, 6)
    expect(transformed.alpha).toBe(transformed.compatibility)
    expect(transformed.signal[0]).toBeCloseTo(0.75, 6)
    expect(transformed.signal[1]).toBeCloseTo(Math.sqrt(0.75) * 0.5, 6)

    const patch = updateFastRelationPatch(signal, representation, undefined, TEST_LAYOUT, 0.05, 1)
    expect(patch).toHaveLength(TEST_LAYOUT.relationLength)
    expect(Array.from(neuron.identity)).toEqual(identityBefore)
  })

  it('p_birth değerinde unmatched kütlesini tam çarpan olarak kullanır', () => {
    const config = createThoughtRuntimeConfig({ maximumBranches: 1 })
    const base = {
      selectedHabitProbability: 0.1,
      candidatePoolCount: 8,
      maximumQ: 0.2,
      previousSmoothedBirth: 0,
      previousConsecutiveBirthTicks: 0,
    }
    const high = calculateThoughtPressure({ ...base, unmatchedProbability: 0.8 }, config)
    const low = calculateThoughtPressure({ ...base, unmatchedProbability: 0.2 }, config)
    expect(high.habitDistance).toBe(low.habitDistance)
    expect(high.representationError).toBe(low.representationError)
    expect(high.birth / low.birth).toBeCloseTo(4, 10)
  })
})

describe('iki aşamalı seyrek routing', () => {
  it('H log-softmax, q cosine ve sıralı hazard dağılımını birebir uygular', () => {
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 0: 0.5, 1: Math.sqrt(0.75) })
    const c = testNeuron('c', { 0: 0.8, 1: 0.6 })
    const { model, head } = testModel({
      neurons: [a, b, c],
      rows: [{
        fromNeuronId: 'a',
        edges: [
          { toNeuronId: 'b', priorEvidence: 3, excitationEvidence: 0, inhibitionEvidence: 0, trustMass: 0, updateCount: 0, independentSourceIds: [] },
          { toNeuronId: 'c', priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0, trustMass: 0, updateCount: 0, independentSourceIds: [] },
        ],
      }],
    })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 7)
    const branch = blankBranch('branch', 'a')
    branch.signal[0] = 1
    const route = routeThoughtBranch(model, index, branch, TEST_CONFIG)
    const rb = route.neuronCandidates.find((candidate) => candidate.neuronId === 'b')!
    const rc = route.neuronCandidates.find((candidate) => candidate.neuronId === 'c')!

    expect(index.familyGraph.transitions).toEqual([expect.objectContaining({
      fromFamilyId: 'family-content', toFamilyId: 'family-content', probability: 1,
    })])
    expect(rb.habitProbability).toBeCloseTo(0.75, 10)
    expect(rc.habitProbability).toBeCloseTo(0.25, 10)
    expect(rb.q).toBeCloseTo(cosineSimilarity(branch.signal, composeNeuronRepresentation(b.identity, b.relation, TEST_LAYOUT)), 7)
    expect(rb.q).toBeCloseTo(0.5, 6)
    expect(rb.acceptanceProbability).toBeCloseTo(1 - (1 - rb.q) ** (1 + 1), 10)
    expect(rc.reachProbability).toBeCloseTo(1 - rb.acceptanceProbability, 10)
    expect(rc.acceptanceProbability).toBeCloseTo(1 - (1 - rc.q) ** (1 + 1 / 3), 7)
    expect(route.totalProbability).toBeCloseTo(1, 10)
    expect(route.unmatchedProbability).toBeCloseTo(1 - rb.selectionProbability - rc.selectionProbability, 10)
  })

  it('H dışında kalan fakat sinyale yakın nöronu LSH ile havuza alır', () => {
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 1: 1 })
    const negativeIdentity = partition(TEST_LAYOUT.identityLength, Object.fromEntries(
      Array.from({ length: TEST_LAYOUT.identityLength }, (_, index) => [index, -1]),
    ))
    const negativeRelation = partition(TEST_LAYOUT.relationLength, Object.fromEntries(
      Array.from({ length: TEST_LAYOUT.relationLength }, (_, index) => [index, -1]),
    ))
    const c = createThoughtNeuron({
      id: 'c',
      identity: negativeIdentity,
      relation: negativeRelation,
      familyMemberships: [{ familyId: 'family-content', weight: 1 }],
      provenance: { kind: 'seed', surface: 'c' },
    }, TEST_LAYOUT, TEST_CONFIG)
    const { model, head } = testModel({
      neurons: [a, b, c],
      rows: [{ fromNeuronId: 'a', edges: [{
        toNeuronId: 'b', priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0,
        trustMass: 0, updateCount: 0, independentSourceIds: [],
      }] }],
    })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 11)
    const signal = new Float32Array(4096).fill(-1)
    expect(queryThoughtSignalIndex(index, signal, 128)).toContain('c')
    const branch = blankBranch('branch', 'a')
    branch.signal = signal
    const route = routeThoughtBranch(model, index, branch, TEST_CONFIG)
    expect(route.neuronCandidates.some((candidate) => candidate.neuronId === 'c' && candidate.baseH0 === 0)).toBe(true)
  })

  it('promotion sonrası entry/outgoing kanıtını kalıcı E+ yolları olarak derler', () => {
    const a = testNeuron('a', { 0: 1 }, 'family-a')
    const c = testNeuron('c', { 2: 1 }, 'family-c')
    const { model, head: emptyHead } = testModel({ neurons: [a, c] })
    const identity = partition(TEST_LAYOUT.identityLength, { 1: 1 })
    const relation = partition(TEST_LAYOUT.relationLength)
    const head = createThoughtHead(model, {
      ...emptyHead,
      births: [{
        format: 'born-neuron-v1', id: 'born-shortcut',
        identity: Array.from(identity), relation: Array.from(relation),
        identityDigest: thoughtPartitionDigest('identity', identity),
        familyMemberships: [{ familyId: 'family-a', weight: 1 }],
        sourceGroupIds: ['source-1', 'source-2'], trustMass: 1,
        bornAtTick: 32, nurseryId: 'nursery-shortcut',
        entryEvidence: [{ memberKey: 'a', weight: 0.8 }],
        outgoingEvidence: [{ memberKey: 'c', weight: 0.6 }],
        contextEvidence: { observationCount: 3, sourceGroupIds: ['source-1', 'source-2'] },
      }],
    })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 31)
    expect(index.edgeRows.get('a')).toEqual([expect.objectContaining({
      toNeuronId: 'born-shortcut', persistentExcitation: 0.8, baseH0: 0,
    })])
    expect(index.edgeRows.get('born-shortcut')).toEqual([expect.objectContaining({
      toNeuronId: 'c', persistentExcitation: 0.6, baseH0: 0,
    })])
    expect(index.familyGraph.transitions.some((transition) => (
      transition.fromFamilyId === 'family-a' && transition.toFamilyId === 'family-c'
    ))).toBe(true)
  })

  it('top-8 aileyi salt G’den önce kesmez; düşük-G/yüksek-sinyal ailesini birleşik logit yarışına alır', () => {
    const origin = testNeuron('origin', { 0: 1 }, 'origin-family')
    const graphTargets = Array.from({ length: 8 }, (_, index) =>
      testNeuron(`graph-${index}`, { [10 + index]: 1 }, `graph-family-${index}`))
    const signalTarget = testNeuron('signal-target', { 2: 1 }, 'z-signal-family')
    const { model, head } = testModel({
      neurons: [origin, ...graphTargets, signalTarget],
      rows: [{
        fromNeuronId: origin.id,
        edges: graphTargets.map((target) => ({
          toNeuronId: target.id,
          priorEvidence: 1,
          excitationEvidence: 0,
          inhibitionEvidence: 0,
          trustMass: 0,
          updateCount: 0,
          independentSourceIds: [],
        })),
      }],
    })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 13)
    const branch = blankBranch('branch', origin.id)
    branch.signal[2] = 1
    const route = routeThoughtBranch(model, index, branch, TEST_CONFIG, {
      candidateNeuronIds: [signalTarget.id],
    })
    expect(route.familyCandidates).toHaveLength(8)
    expect(route.familyCandidates.some((family) => family.familyId === 'z-signal-family')).toBe(true)
    expect(route.familyCandidates[0].familyId).toBe('z-signal-family')
  })

  it('bounded G satırının top-128 dışındaki kanıt kütlesini unmatched olarak korur', () => {
    const origin = testNeuron('origin', { 0: 1 }, 'origin-family')
    const targets = Array.from({ length: 96 }, (_, index) => createThoughtNeuron({
      id: `target-${index}`,
      identity: partition(TEST_LAYOUT.identityLength, { 1: 1 }),
      relation: partition(TEST_LAYOUT.relationLength),
      familyMemberships: [
        { familyId: `family-${index * 2}`, weight: 0.5 },
        { familyId: `family-${index * 2 + 1}`, weight: 0.5 },
      ],
      provenance: { kind: 'seed', surface: `target-${index}` },
    }, TEST_LAYOUT, TEST_CONFIG))
    const { model, head } = testModel({
      neurons: [origin, ...targets],
      rows: [{
        fromNeuronId: origin.id,
        edges: targets.map((target) => ({
          toNeuronId: target.id, priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0,
          trustMass: 0, updateCount: 0, independentSourceIds: [],
        })),
      }],
    })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 71)
    expect(index.familyRows.get('origin-family')).toHaveLength(128)
    expect(index.familyRowTailEvidence.get('origin-family')).toBeGreaterThan(0)
    const branch = blankBranch('branch', origin.id)
    branch.signal[1] = 1
    const route = routeThoughtBranch(model, index, branch, TEST_CONFIG)
    expect(route.unmatchedProbability).toBeGreaterThan(0)
    expect(route.totalProbability).toBeCloseTo(1, 10)
  })

  it('100K katalog etiketi ve geçici doğum varken tick sırasında global map iterasyonu yapmaz', () => {
    const a = testNeuron('a', { 0: 1 })
    const b = testNeuron('b', { 0: 0.8, 1: 0.6 })
    const newborn = testNeuron('temp-newborn', { 0: 0.9, 1: 0.4 })
    const { model, head } = testModel({ neurons: [a, b], rows: [{ fromNeuronId: 'a', edges: [{
      toNeuronId: 'b', priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0,
      trustMass: 0, updateCount: 0, independentSourceIds: [],
    }] }] })
    const base = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 3)
    const lookupOnly: ReadonlyMap<string, ThoughtNeuron> = {
      size: 100_000,
      get: (key) => base.neuronById.get(key),
      has: (key) => base.neuronById.has(key),
      entries: () => { throw new Error('global entries scan') },
      keys: () => { throw new Error('global keys scan') },
      values: () => { throw new Error('global values scan') },
      forEach: () => { throw new Error('global forEach scan') },
      [Symbol.iterator]: () => { throw new Error('global iterator scan') },
    }
    const syntheticScale = { ...base, neuronById: lookupOnly, compiledNeuronCount: 100_000 } satisfies ThoughtRuntimeIndex
    const index = withTemporaryThoughtNeurons(syntheticScale, [newborn])
    const branch = blankBranch('branch', 'a')
    branch.signal[0] = 1
    const route = routeThoughtBranch(model, index, branch, TEST_CONFIG)
    expect(route.candidatePoolCount).toBeLessThanOrEqual(128)
    expect(route.evaluatedCandidateCount).toBeLessThanOrEqual(32)
    expect(route.inspectedNeuronCount).toBeLessThanOrEqual(48)
    expect(route.neuronCandidates.some((candidate) => candidate.neuronId === 'temp-newborn')).toBe(true)
  })
})
