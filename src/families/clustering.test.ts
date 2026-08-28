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

function evidence(memberKey: string, sourceDigest: string, start = 0, end = 4): IndependentEvidenceRef {
  return {
    memberKey,
    analysisRunId: `run-${memberKey}`,
    sourceId: `source-${sourceDigest}`,
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
    evidence: [evidence(memberKey, sourceDigest, range[0], range[1])],
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
