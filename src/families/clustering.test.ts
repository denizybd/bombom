import { describe, expect, it } from 'vitest'
import {
  assignSoftMemberships,
  discoverFamilies,
} from './clustering'
import { createSparseVector, sparseVectorDigest } from './sparse'
import type {
  BehaviorDiagnostics,
  BehaviorProfile,
  FamilyConfig,
  FunctionalRegionId,
  IndependentEvidenceRef,
  LearnedFamily,
  SparseVector,
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
  maximumMembershipsPerNeuron: 2,
  minimumFormationSimilarity: 0.75,
  minimumMembershipScore: 0,
  minimumCoreMembership: 0.7,
  minimumSupport: 3,
  minimumSources: 2,
  maximumOutgoingFamilyTransitions: 3,
  runtimeActivationThreshold: 0.1,
  maximumActiveNeurons: 32,
  maximumActiveFamilies: 8,
  maximumCoalitions: 8,
  maximumRoleCandidates: 4,
  coalitionBeamWidth: 8,
  routingCandidateLimit: 16,
  routingTemperature: 1,
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

const ZERO_DIAGNOSTICS: BehaviorDiagnostics = {
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

function evidence(
  memberKey: string,
  sourceDigest: string,
  start = 0,
  end = 4,
  sourceId = `source-${sourceDigest}`,
  independenceGroupId?: string,
): IndependentEvidenceRef {
  return {
    memberKey,
    analysisRunId: `run-${memberKey}`,
    independenceGroupId,
    sourceId,
    sourceDigest,
    neuronId: `neuron-${memberKey}`,
    paragraphIndex: 0,
    start,
    end,
    text: memberKey,
  }
}

function profile(
  memberKey: string,
  sourceDigest: string,
  vector: SparseVector,
  range: readonly [number, number] = [0, 4],
  sourceId?: string,
  independenceGroupId?: string,
): BehaviorProfile {
  const regions: FunctionalRegionId[] = [
    'content', 'event', 'role', 'operator', 'goal', 'language', 'control',
  ]
  return {
    memberKey,
    kind: 'raw',
    representationSpaceId: 'RS-test',
    behaviorSpaceId: 'BS-test',
    vector,
    diagnostics: { ...ZERO_DIAGNOSTICS },
    regionAffinities: regions.map((region) => ({
      region,
      score: region === 'content' ? 0.9 : region === 'event' ? 0.4 : 0,
      verified: false,
      explanation: 'test',
    })),
    evidence: [evidence(
      memberKey,
      sourceDigest,
      range[0],
      range[1],
      sourceId,
      independenceGroupId,
    )],
    identityDigest: `identity-${memberKey}`,
    text: memberKey,
    derivedFromMemberKeys: [],
  }
}

function family(id: string, prototype: SparseVector, region: FunctionalRegionId = 'content'): LearnedFamily {
  return {
    id,
    fingerprint: `fingerprint-${id}`,
    behaviorSpaceId: 'BS-test',
    region,
    regionVerified: false,
    lifecycle: 'provisional',
    lifecycleReason: 'test',
    prototype,
    prototypeDigest: sparseVectorDigest(prototype),
    memberKeys: [],
    coreMemberKeys: [],
    independentEvidence: [],
    supportCount: 3,
    sourceCount: 3,
    cohesion: 1,
    separation: 1,
    stability: 0,
    utility: null,
    predecessorFamilyIds: [],
  }
}

function withStructuralHypothesis(
  value: BehaviorProfile,
  depthScale: number,
  scores: Partial<Record<FunctionalRegionId, number>>,
): BehaviorProfile {
  return {
    ...value,
    diagnostics: { ...value.diagnostics, depthScale },
    regionAffinities: value.regionAffinities.map((affinity) => ({
      ...affinity,
      score: scores[affinity.region] ?? 0,
    })),
  }
}

describe('aile keşfi ve yumuşak üyelik', () => {
  it('tek kaynağın çok sayıdaki tekrarından aile doğurmaz', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = [
      profile('a', 'same-source', shared, [0, 4]),
      profile('b', 'same-source', shared, [4, 8]),
      profile('c', 'same-source', shared, [8, 12]),
    ]

    const result = discoverFamilies(profiles, profiles, new Set(profiles.map((item) => item.memberKey)), CONFIG)

    expect(result.edges).toEqual([])
    expect(result.families).toEqual([])
  })

  it('aynı yüzey etiketi tek başına aile kurmaz; davranış vektörü uyuşmalıdır', () => {
    const left = profile('left', 'digest-left', createSparseVector(8, [{ index: 0, value: 1 }]))
    const right = profile('right', 'digest-right', createSparseVector(8, [{ index: 1, value: 1 }]))
    left.text = 'aynı-etiket'
    right.text = 'aynı-etiket'

    const result = discoverFamilies(
      [left, right],
      [left, right],
      new Set([left.memberKey, right.memberKey]),
      { ...CONFIG, minimumSupport: 2, minimumSources: 2 },
    )

    expect(result.edges).toEqual([])
    expect(result.families).toEqual([])
  })

  it('aynı kitabın farklı parça özetlerini bağımsız kaynak saymaz', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = [
      profile('chunk-a', 'digest-chunk-a', shared, [0, 4], 'book-1'),
      profile('chunk-b', 'digest-chunk-b', shared, [0, 4], 'book-1'),
      profile('other', 'digest-other', shared, [0, 4], 'book-2'),
    ]

    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      { ...CONFIG, minimumSupport: 3, minimumSources: 3 },
    )

    expect(result.families).toEqual([])
  })

  it('aynı içeriğin farklı kaynak kimliğiyle kopyalanmasını bağımsız kanıt saymaz', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = [
      profile('original', 'same-content-digest', shared, [0, 4], 'book-original'),
      profile('renamed-copy', 'same-content-digest', shared, [0, 4], 'book-copy'),
      profile('other', 'other-content-digest', shared, [0, 4], 'book-other'),
    ]

    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      { ...CONFIG, minimumSupport: 3, minimumSources: 3 },
    )

    expect(result.edges.every((edge) => !(
      edge.leftMemberKey === 'original' && edge.rightMemberKey === 'renamed-copy'
    ))).toBe(true)
    expect(result.families).toEqual([])
  })

  it('ortak sentetik üretim soyunu farklı sourceId ve digestlere rağmen tek aile kaynağı sayar', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = [
      profile('lesson-a', 'digest-a', shared, [0, 4], 'lesson-a', 'ollama:qwen14b'),
      profile('lesson-b', 'digest-b', shared, [0, 4], 'lesson-b', 'ollama:qwen14b'),
      profile('human', 'digest-human', shared, [0, 4], 'human', 'human:book'),
    ]

    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      { ...CONFIG, minimumSupport: 3, minimumSources: 3 },
    )

    expect(result.edges.some((edge) => (
      [edge.leftMemberKey, edge.rightMemberKey].includes('lesson-a') &&
      [edge.leftMemberKey, edge.rightMemberKey].includes('lesson-b')
    ))).toBe(false)
    expect(result.families).toEqual([])
  })

  it('düşük bölge farkında semantik etiket uydurmak yerine belirsiz kalır', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = ['a', 'b', 'c'].map((source) => withStructuralHypothesis(
      profile(`member-${source}`, `digest-${source}`, shared, [0, 1], `source-${source}`),
      0.8,
      { content: 0.56, operator: 0.55, event: 0.2, role: 0.2 },
    ))

    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      CONFIG,
    )

    expect(result.families).toHaveLength(1)
    expect(result.families[0].region).toBe('unknown')
    expect(result.families[0].regionVerified).toBe(false)

    const lowScoreProfiles = ['d', 'e', 'f'].map((source) => withStructuralHypothesis(
      profile(`low-${source}`, `low-digest-${source}`, shared, [0, 1], `low-source-${source}`),
      0.8,
      { content: 0.49, operator: 0.3, event: 0.2, role: 0.2 },
    ))
    const lowScore = discoverFamilies(
      lowScoreProfiles,
      lowScoreProfiles,
      new Set(lowScoreProfiles.map((item) => item.memberKey)),
      CONFIG,
    )
    expect(lowScore.families).toHaveLength(1)
    expect(lowScore.families[0].region).toBe('unknown')
  })

  it('leaf, parent ve root ölçeklerini deterministik örnekleyip tek-bölge çöküşünü önler', () => {
    const leafVector = createSparseVector(8, [{ index: 0, value: 1 }])
    const parentVector = createSparseVector(8, [{ index: 1, value: 1 }])
    const rootVector = createSparseVector(8, [{ index: 2, value: 1 }])
    const leaves = ['a', 'b', 'c'].map((source) => withStructuralHypothesis(
      profile(`leaf-${source}`, `leaf-digest-${source}`, leafVector, [0, 1], `source-${source}`),
      0.9,
      { operator: 0.9, content: 0.2, event: 0.1, role: 0.1 },
    ))
    const parents = ['a', 'b', 'c'].map((source) => withStructuralHypothesis(
      profile(`parent-${source}`, `parent-digest-${source}`, parentVector, [0, 4], `source-${source}`),
      0.4,
      { event: 0.9, content: 0.2, operator: 0.1, role: 0.1 },
    ))
    const roots = ['a', 'b', 'c'].map((source) => withStructuralHypothesis(
      profile(`root-${source}`, `root-digest-${source}`, rootVector, [0, 8], `source-${source}`),
      0,
      { content: 0.9, event: 0.2, operator: 0.1, role: 0.1 },
    ))
    const profiles = [...leaves, ...parents, ...roots]
    const leafKeys = new Set(leaves.map((item) => item.memberKey))

    const first = discoverFamilies(profiles, profiles, leafKeys, CONFIG)
    const replay = discoverFamilies([...profiles].reverse(), [...profiles].reverse(), leafKeys, CONFIG)
    const projection = (result: typeof first) => result.families.map((item) => ({
      id: item.id,
      region: item.region,
      memberKeys: item.memberKeys,
    }))

    expect(new Set(first.families.map((item) => item.region))).toEqual(
      new Set<FunctionalRegionId>(['content', 'event', 'operator']),
    )
    expect(first.families.some((item) => item.memberKeys.every((key) => key.startsWith('root-')))).toBe(true)
    expect(first.families.some((item) => item.memberKeys.every((key) => key.startsWith('parent-')))).toBe(true)
    expect(projection(replay)).toEqual(projection(first))
    expect(replay.edges).toEqual(first.edges)
  })

  it('aynı-kaynak aday kalabalığını bağımsız davranış adayından önce süzer', () => {
    const independent = createSparseVector(8, [{ index: 0, value: 1 }])
    const crowded = createSparseVector(8, [
      { index: 0, value: 0.6 },
      { index: 1, value: 0.8 },
    ])
    const current = createSparseVector(8, [
      { index: 0, value: 0.8 },
      { index: 1, value: 0.6 },
    ])
    const profiles = [
      profile('a-independent', 'digest-independent', independent, [0, 1], 'book-independent'),
      ...Array.from({ length: 6 }, (_, index) =>
        profile(`b-crowd-${index}`, `digest-crowd-${index}`, crowded, [index, index + 1], 'book-crowd')),
      profile('z-current', 'digest-current', current, [0, 1], 'book-crowd'),
    ]

    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      {
        ...CONFIG,
        maximumCandidateNeighbors: 1,
        minimumFormationSimilarity: 0.75,
        minimumSupport: 2,
        minimumSources: 2,
      },
    )
    expect(result.edges).toContainEqual({
      leftMemberKey: 'a-independent',
      rightMemberKey: 'z-current',
      similarity: 0.8,
    })
  })

  it('son davranış grafiğinde her nöronun gerçek derece sınırını uygular', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = Array.from({ length: 6 }, (_, index) =>
      profile(`member-${index}`, `digest-${index}`, shared, [0, 1], `book-${index}`))
    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      {
        ...CONFIG,
        maximumCandidateNeighbors: 1,
        minimumSupport: 2,
        minimumSources: 2,
      },
    )
    const reordered = discoverFamilies(
      [...profiles].reverse(),
      [...profiles].reverse(),
      new Set(profiles.map((item) => item.memberKey)),
      {
        ...CONFIG,
        maximumCandidateNeighbors: 1,
        minimumSupport: 2,
        minimumSources: 2,
      },
    )
    const degree = new Map<string, number>()
    for (const edge of result.edges) {
      degree.set(edge.leftMemberKey, (degree.get(edge.leftMemberKey) ?? 0) + 1)
      degree.set(edge.rightMemberKey, (degree.get(edge.rightMemberKey) ?? 0) + 1)
    }

    expect(Math.max(...degree.values())).toBeLessThanOrEqual(1)
    expect(result.edges.length).toBeGreaterThan(0)
    expect(reordered.edges).toEqual(result.edges)
  })

  it('örtüşen aynı-kaynak parçaları destek sayısını şişirmez', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = [
      profile('parent', 'source-a', shared, [0, 10]),
      profile('child', 'source-a', shared, [0, 4]),
      profile('other-b', 'source-b', shared, [0, 4]),
      profile('other-c', 'source-c', shared, [0, 4]),
    ]
    const leafKeys = new Set(profiles.map((item) => item.memberKey))

    const strict = discoverFamilies(profiles, profiles, leafKeys, { ...CONFIG, minimumSupport: 4 })
    const allowed = discoverFamilies(profiles, profiles, leafKeys, { ...CONFIG, minimumSupport: 3 })

    expect(strict.families).toEqual([])
    expect(allowed.families).toHaveLength(1)
    expect(allowed.families[0].supportCount).toBe(3)
    expect(allowed.families[0].sourceCount).toBe(3)
  })

  it('tek-bağ benzerlik zinciriyle uyumsuz uçları aynı aileye sokmaz', () => {
    const a = createSparseVector(8, [{ index: 0, value: 1 }])
    const b = createSparseVector(8, [
      { index: 0, value: Math.sqrt(3) / 2 },
      { index: 1, value: 0.5 },
    ])
    const c = createSparseVector(8, [
      { index: 0, value: 0.5 },
      { index: 1, value: Math.sqrt(3) / 2 },
    ])
    const profiles = [
      profile('a', 'source-a', a),
      profile('b', 'source-b', b),
      profile('c', 'source-c', c),
    ]
    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      { ...CONFIG, minimumSupport: 2, minimumSources: 2 },
    )

    expect(result.edges).toHaveLength(2)
    expect(result.families).toHaveLength(1)
    expect(result.families[0].memberKeys).toHaveLength(2)
  })

  it('aile üye sayısını ve toplam aile sayısını yapılandırılmış sınırlarda tutar', () => {
    const shared = createSparseVector(8, [{ index: 0, value: 1 }])
    const profiles = Array.from({ length: 8 }, (_, index) =>
      profile(`member-${index}`, `source-${index}`, shared))
    const result = discoverFamilies(
      [...profiles].reverse(),
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      {
        ...CONFIG,
        maximumFamilyMembers: 2,
        maximumFamilies: 2,
        minimumSupport: 2,
        minimumSources: 2,
      },
    )

    expect(result.families.length).toBeLessThanOrEqual(2)
    expect(result.families.every((item) => item.memberKeys.length <= 2)).toBe(true)
  })

  it('açık bir deney tavanı verilmediğinde 128 ailede yapay olarak kesmez', () => {
    const familyCount = 129
    const profiles = Array.from({ length: familyCount }, (_, familyIndex) => {
      const vector = createSparseVector(familyCount, [{ index: familyIndex, value: 1 }])
      return [
        profile(`family-${familyIndex}-a`, `digest-${familyIndex}-a`, vector, [0, 1], `source-${familyIndex}-a`),
        profile(`family-${familyIndex}-b`, `digest-${familyIndex}-b`, vector, [0, 1], `source-${familyIndex}-b`),
      ].map((item) => ({
        ...item,
        vector: { ...item.vector, dimension: familyCount },
      }))
    }).flat()
    const result = discoverFamilies(
      profiles,
      profiles,
      new Set(profiles.map((item) => item.memberKey)),
      {
        ...CONFIG,
        profileDimension: familyCount,
        maximumProfileFeatures: familyCount,
        maximumIndexBucketSize: 8,
        maximumCandidateNeighbors: 2,
        maximumFamilyMembers: 2,
        maximumFamilies: Number.MAX_SAFE_INTEGER,
        minimumFormationSimilarity: 0.99,
        minimumSupport: 2,
        minimumSources: 2,
      },
    )

    expect(result.families).toHaveLength(familyCount)
  })

  it('M üyeliklerini top-k ile sınırlar, deterministik sıralar ve her nöron için normalize eder', () => {
    const vector = createSparseVector(8, [{ index: 0, value: 1 }])
    const subject = profile('subject', 'source-a', vector)
    const families = [
      family('family-c', createSparseVector(8, [{ index: 0, value: 0.8 }, { index: 2, value: 0.2 }])),
      family('family-a', vector),
      family('family-b', createSparseVector(8, [{ index: 0, value: 0.9 }, { index: 1, value: 0.1 }])),
    ]

    const first = assignSoftMemberships([subject], families, { ...CONFIG, maximumMembershipsPerNeuron: 2 })
    const second = assignSoftMemberships([subject], [...families].reverse(), { ...CONFIG, maximumMembershipsPerNeuron: 2 })

    expect(first).toEqual(second)
    expect(first).toHaveLength(2)
    expect(first.map((membership) => membership.rank)).toEqual([1, 2])
    expect(first.reduce((sum, membership) => sum + membership.weight, 0)).toBeCloseTo(1)
    expect(first.every((membership) => Number.isFinite(membership.weight))).toBe(true)
  })
})
