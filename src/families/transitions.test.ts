import { describe, expect, it } from 'vitest'
import { digestVector } from '../text-lab/engine/representation'
import { rawMemberKey } from './profile'
import { createSparseVector, sparseVectorDigest } from './sparse'
import {
  buildFamilyTransitions,
  buildSparseNeuronTransitions,
  computeRouteDistribution,
} from './transitions'
import type {
  BehaviorDiagnostics,
  BehaviorProfile,
  FamilyAnalysisInput,
  FamilyConfig,
  FamilyMembership,
  FamilyNeuronInput,
  FunctionalRegionId,
  LearnedFamily,
  SparseNeuronTransition,
} from './types'

const CONFIG: FamilyConfig = {
  profileDimension: 8,
  maximumProfileFeatures: 8,
  contextRadius: 1,
  maximumTransitionNeighbors: 2,
  maximumIndexBucketSize: 32,
  maximumCandidateNeighbors: 8,
  maximumFamilyMembers: 8,
  maximumFamilies: 16,
  maximumMembershipsPerNeuron: 3,
  minimumFormationSimilarity: 0.7,
  minimumMembershipScore: 0,
  minimumCoreMembership: 0.7,
  minimumSupport: 2,
  minimumSources: 2,
  maximumOutgoingFamilyTransitions: 2,
  runtimeActivationThreshold: 0.1,
  maximumActiveNeurons: 32,
  maximumActiveFamilies: 8,
  maximumCoalitions: 8,
  maximumRoleCandidates: 4,
  coalitionBeamWidth: 8,
  routingCandidateLimit: 8,
  routingTemperature: 0.7,
  habitRouteWeight: 0.4,
  familyRouteWeight: 0.35,
  signalRouteWeight: 0.25,
  repetitionPenaltyWeight: 0.2,
  utilityMinimumCases: 2,
  utilityAlternativeCount: 3,
  utilityEpsilon: 1e-6,
  lifecycleStabilityThreshold: 0.7,
  lifecycleUtilityThreshold: 0,
  yieldEveryOperations: 16,
  hashSeed: 1,
}

const diagnostics: BehaviorDiagnostics = {
  spanScale: 0.5,
  relativePosition: 0.5,
  depthScale: 0,
  relationStrength: 0.5,
  transitionEntropy: 0.5,
  transitionConcentration: 0.5,
  localContinuity: 0.5,
  bridgeImpact: 0.5,
  leftRightAsymmetry: 0.5,
  shortScopePotential: 0.5,
}

function node(id: string, sourceId: string, start: number): FamilyNeuronInput {
  const identity = Float32Array.of(1)
  const relation = Float32Array.of(0)
  return {
    id,
    sourceId,
    paragraphIndex: 0,
    start,
    end: start + 1,
    text: id,
    graphemeCount: 1,
    depth: 0,
    branch: 'root',
    parentId: null,
    childIds: [],
    splitBoundary: 'leaf',
    identity,
    relation,
    identityDigest: digestVector(identity),
    identityNorm: 1,
    relationNorm: 0,
    contextSupport: 0,
    contextRevision: 0,
    trace: {
      inheritedNorm: 0,
      parentIdentityCompatibility: 0,
      parentGain: 0,
      siblingId: null,
      siblingIdentityCompatibility: null,
      siblingGain: 0,
    },
  }
}

function analysis(
  sourceDigest: string,
  runId: string,
  ids: readonly string[],
  habitProbabilities: readonly number[],
): FamilyAnalysisInput {
  const nodes = ids.map((id, index) => node(id, sourceDigest, index))
  return {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId,
    createdAt: '2026-08-28T00:00:00.000Z',
    sourceDigest,
    identitySpaceId: 'IS-test',
    contextSpaceId: 'CS-test',
    representationSpaceId: 'RS-test',
    source: { id: sourceDigest, title: sourceDigest, rawText: ids.join(''), text: ids.join('') },
    config: {
      matrixSide: 8,
      identityRatio: 0.5,
      minimumLeafGraphemes: 1,
      hashSeed: 1,
      contextRetention: 0.8,
      parentTraceWeight: 0.15,
      siblingTraceWeight: 0.05,
      minimumTrace: 0.1,
      identitySimilarityWeight: 0.5,
      habitExponent: 2,
    },
    layout: {
      side: 8,
      cellCount: 64,
      identityRows: 4,
      identityLength: 32,
      relationStart: 32,
      relationLength: 32,
    },
    rootIds: [...ids],
    nodes,
    habitProbabilities: Float32Array.from(habitProbabilities),
  }
}

function profile(
  memberKey: string,
  sourceDigest: string,
  coordinate: number,
  sourceId = sourceDigest,
  independenceGroupId?: string,
): BehaviorProfile {
  const vector = createSparseVector(8, [{ index: coordinate, value: 1 }])
  const regions: FunctionalRegionId[] = [
    'content', 'event', 'role', 'operator', 'goal', 'language', 'control',
  ]
  return {
    memberKey,
    kind: 'raw',
    representationSpaceId: 'RS-test',
    behaviorSpaceId: 'BS-test',
    vector,
    diagnostics,
    regionAffinities: regions.map((region) => ({
      region,
      score: region === 'content' ? 1 : 0,
      verified: false,
      explanation: 'test',
    })),
    evidence: [{
      memberKey,
      analysisRunId: 'run-test',
      independenceGroupId,
      sourceId,
      sourceDigest,
      neuronId: memberKey,
      paragraphIndex: 0,
      start: 0,
      end: 1,
      text: memberKey,
    }],
    identityDigest: memberKey,
    text: memberKey,
    derivedFromMemberKeys: [],
  }
}

function family(id: string): LearnedFamily {
  const prototype = createSparseVector(8, [{ index: 0, value: 1 }])
  return {
    id,
    fingerprint: `fingerprint-${id}`,
    behaviorSpaceId: 'BS-test',
    region: 'content',
    regionVerified: false,
    lifecycle: 'provisional',
    lifecycleReason: 'test',
    prototype,
    prototypeDigest: sparseVectorDigest(prototype),
    memberKeys: [],
    coreMemberKeys: [],
    independentEvidence: [],
    supportCount: 2,
    sourceCount: 2,
    cohesion: 1,
    separation: 1,
    stability: 0,
    utility: null,
    predecessorFamilyIds: [],
  }
}

function membership(memberKey: string, familyId: string, weight = 1): FamilyMembership {
  return {
    memberKey,
    familyId,
    rank: 1,
    score: weight,
    weight,
    core: true,
    breakdown: {
      behaviorSimilarity: 1,
      regionCompatibility: 1,
      evidenceConfidence: 1,
      combined: weight,
    },
  }
}

describe('seyrek H, G ve iki seviyeli yönlendirme', () => {
  it('yoğun H satırını top-k seyrek geçişe çevirir ve tutulan satırı yeniden normalize eder', () => {
    const input = analysis('source-a', 'run-a', ['a', 'b', 'c', 'd'], [
      0, 0.5, 0.3, 0.2,
      0, 0, 1, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ])
    const keys = new Set(input.nodes.map((item) => rawMemberKey(input.sourceDigest, input.runId, item.id)))
    const transitions = buildSparseNeuronTransitions([input], keys, CONFIG)
    const fromA = transitions.filter((entry) => entry.fromMemberKey.endsWith('\u0000a'))

    expect(fromA).toHaveLength(2)
    expect(fromA.map((entry) => entry.toMemberKey.slice(-1))).toEqual(['b', 'c'])
    expect(fromA.reduce((sum, entry) => sum + entry.probability, 0)).toBeCloseTo(1)
    expect(transitions.filter((entry) => entry.fromMemberKey.endsWith('\u0000b'))).toHaveLength(1)
  })

  it('G çıkışlarını top-k ile sınırlar ve her kaynak aile satırını bire normalize eder', () => {
    const keys = ['a', 'b', 'c', 'd']
    const profiles = keys.map((key, index) => profile(key, 'same-source', index))
    const families = ['fa', 'fb', 'fc', 'fd'].map(family)
    const memberships = keys.map((key, index) => membership(key, families[index].id))
    const neuronTransitions: SparseNeuronTransition[] = [
      { fromMemberKey: 'a', toMemberKey: 'b', probability: 0.5 },
      { fromMemberKey: 'a', toMemberKey: 'c', probability: 0.3 },
      { fromMemberKey: 'a', toMemberKey: 'd', probability: 0.2 },
    ]

    const result = buildFamilyTransitions(
      neuronTransitions,
      profiles,
      families,
      memberships,
      { ...CONFIG, maximumOutgoingFamilyTransitions: 2 },
    )
    const outgoing = result.familyTransitions.filter((entry) => entry.fromFamilyId === 'fa')

    expect(outgoing).toHaveLength(2)
    expect(outgoing.map((entry) => entry.toFamilyId)).toEqual(['fb', 'fc'])
    expect(outgoing.reduce((sum, entry) => sum + entry.probability, 0)).toBeCloseTo(1)
    expect(result.peakOutgoingFamilyTransitions).toBe(2)
  })

  it('uzun kaynağın tekrar sayısının kısa bağımsız kaynağı ezmesini kaynak-normalizasyonuyla önler', () => {
    const profiles = [
      profile('a1', 'source-long', 0),
      profile('b1', 'source-long', 1),
      profile('a2', 'source-long', 0),
      profile('b2', 'source-long', 1),
      profile('a3', 'source-long', 0),
      profile('b3', 'source-long', 1),
      profile('c1', 'source-short', 0),
      profile('d1', 'source-short', 2),
    ]
    const memberships = [
      membership('a1', 'fa'), membership('b1', 'fb'),
      membership('a2', 'fa'), membership('b2', 'fb'),
      membership('a3', 'fa'), membership('b3', 'fb'),
      membership('c1', 'fa'), membership('d1', 'fc'),
    ]
    const result = buildFamilyTransitions(
      [
        { fromMemberKey: 'a1', toMemberKey: 'b1', probability: 1 },
        { fromMemberKey: 'a2', toMemberKey: 'b2', probability: 1 },
        { fromMemberKey: 'a3', toMemberKey: 'b3', probability: 1 },
        { fromMemberKey: 'c1', toMemberKey: 'd1', probability: 1 },
      ],
      profiles,
      [family('fa'), family('fb'), family('fc')],
      memberships,
      { ...CONFIG, maximumOutgoingFamilyTransitions: 3 },
    )
    const outgoing = result.familyTransitions.filter((entry) => entry.fromFamilyId === 'fa')

    expect(outgoing).toHaveLength(2)
    expect(outgoing.find((entry) => entry.toFamilyId === 'fb')?.probability).toBeCloseTo(0.5)
    expect(outgoing.find((entry) => entry.toFamilyId === 'fc')?.probability).toBeCloseTo(0.5)
  })

  it('aynı Ollama soyundaki geçişleri tek kaynak kanıtı olarak raporlar', () => {
    const profiles = [
      profile('a1', 'digest-a', 0, 'lesson-a', 'ollama:qwen14b'),
      profile('b1', 'digest-a', 1, 'lesson-a', 'ollama:qwen14b'),
      profile('a2', 'digest-b', 0, 'lesson-b', 'ollama:qwen14b'),
      profile('b2', 'digest-b', 1, 'lesson-b', 'ollama:qwen14b'),
      profile('c1', 'digest-c', 0, 'book-c', 'human:book-c'),
      profile('d1', 'digest-c', 2, 'book-c', 'human:book-c'),
    ]
    const memberships = [
      membership('a1', 'fa'), membership('b1', 'fb'),
      membership('a2', 'fa'), membership('b2', 'fb'),
      membership('c1', 'fa'), membership('d1', 'fc'),
    ]
    const result = buildFamilyTransitions(
      [
        { fromMemberKey: 'a1', toMemberKey: 'b1', probability: 1 },
        { fromMemberKey: 'a2', toMemberKey: 'b2', probability: 1 },
        { fromMemberKey: 'c1', toMemberKey: 'd1', probability: 1 },
      ],
      profiles,
      [family('fa'), family('fb'), family('fc')],
      memberships,
      { ...CONFIG, maximumOutgoingFamilyTransitions: 3 },
    )

    const ollamaEdge = result.familyTransitions.find((entry) => (
      entry.fromFamilyId === 'fa' && entry.toFamilyId === 'fb'
    ))
    expect(ollamaEdge?.sourceCount).toBe(1)
    expect(ollamaEdge?.supportCount).toBe(2)
  })

  it('aday bulunmadığında açık unmatched sonucu döndürür', () => {
    const current = profile('current', 'source-a', 0)
    const result = computeRouteDistribution(
      current.memberKey,
      [current],
      [],
      [],
      [],
      CONFIG,
    )

    expect(result.entries).toEqual([])
    expect(result.unmatchedProbability).toBe(1)
    expect(result.totalProbability).toBe(1)
  })

  it('H/M/G/sinyal puanlarını tek dağılımda birleştirip toplam olasılığı bire eşitler', () => {
    const profiles = [
      profile('current', 'source-a', 0),
      profile('target-b', 'source-a', 1),
      profile('target-c', 'source-a', 2),
    ]
    const memberships = [
      membership('current', 'fa'),
      membership('target-b', 'fb'),
      membership('target-c', 'fc'),
    ]
    const neuronTransitions: SparseNeuronTransition[] = [
      { fromMemberKey: 'current', toMemberKey: 'target-b', probability: 0.6 },
      { fromMemberKey: 'current', toMemberKey: 'target-c', probability: 0.4 },
    ]
    const familyTransitions = [
      {
        id: 'g-ab',
        fromFamilyId: 'fa',
        toFamilyId: 'fb',
        priorEvidence: 0.7,
        experienceEvidence: 0,
        inhibitionEvidence: 0,
        sourceCount: 2,
        supportCount: 2,
        probability: 0.7,
      },
      {
        id: 'g-ac',
        fromFamilyId: 'fa',
        toFamilyId: 'fc',
        priorEvidence: 0.3,
        experienceEvidence: 0,
        inhibitionEvidence: 0,
        sourceCount: 2,
        supportCount: 2,
        probability: 0.3,
      },
    ]
    const result = computeRouteDistribution(
      'current',
      profiles,
      memberships,
      neuronTransitions,
      familyTransitions,
      CONFIG,
      {
        signalCompatibility: new Map([['target-b', 0.9], ['target-c', 0.2]]),
        repetition: new Map([['target-b', 0.1], ['target-c', 0.8]]),
      },
    )

    expect(result.entries).toHaveLength(2)
    expect(result.entries[0].memberKey).toBe('target-b')
    expect(result.entries.reduce((sum, entry) => sum + entry.probability, 0)).toBeCloseTo(1)
    expect(result.unmatchedProbability).toBeCloseTo(0)
    expect(result.totalProbability).toBeCloseTo(1)
  })

  it('bir nöronun birden çok yumuşak aile üyeliğini G hesabında birlikte kullanır', () => {
    const profiles = [
      profile('current', 'source-a', 0),
      profile('target', 'source-a', 1),
    ]
    const memberships = [
      { ...membership('current', 'fa', 0.25), rank: 1 },
      { ...membership('current', 'fx', 0.75), rank: 2 },
      { ...membership('target', 'fb', 0.4), rank: 1 },
      { ...membership('target', 'fy', 0.6), rank: 2 },
    ]
    const result = computeRouteDistribution(
      'current',
      profiles,
      memberships,
      [],
      [
        {
          id: 'g-ab', fromFamilyId: 'fa', toFamilyId: 'fb', priorEvidence: 1,
          experienceEvidence: 0, inhibitionEvidence: 0, sourceCount: 2, supportCount: 2, probability: 1,
        },
        {
          id: 'g-xy', fromFamilyId: 'fx', toFamilyId: 'fy', priorEvidence: 1,
          experienceEvidence: 0, inhibitionEvidence: 0, sourceCount: 2, supportCount: 2, probability: 1,
        },
      ],
      CONFIG,
      { signalCompatibility: new Map([['target', 0]]) },
    )

    expect(result.entries).toHaveLength(1)
    expect(result.entries[0].breakdown.family).toBeCloseTo(0.25 * 0.4 + 0.75 * 0.6)
    expect(result.entries[0].probability).toBe(1)
  })

  it('geçiş çıkarımı sırasında iptali satır sınırında uygular', () => {
    const input = analysis('source-a', 'run-a', ['a', 'b', 'c'], [
      0, 0.5, 0.5,
      0.5, 0, 0.5,
      0.5, 0.5, 0,
    ])
    const keys = new Set(input.nodes.map((item) => rawMemberKey(input.sourceDigest, input.runId, item.id)))
    let checks = 0

    expect(() => buildSparseNeuronTransitions([input], keys, CONFIG, () => {
      checks += 1
      return checks === 2
    })).toThrow('FAMILY_CANCELLED')
    expect(checks).toBe(2)
  })
})
