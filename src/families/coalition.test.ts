import { describe, expect, it } from 'vitest'
import { buildEventCoalitions } from './coalition'
import { createSparseVector, sparseVectorDigest } from './sparse'
import type {
  BehaviorDiagnostics,
  BehaviorProfile,
  CoalitionState,
  FamilyConfig,
  FamilyMembership,
  FunctionalRegionId,
  LearnedFamily,
  NeuronActivation,
} from './types'

const CONFIG: FamilyConfig = {
  profileDimension: 8,
  maximumProfileFeatures: 8,
  contextRadius: 1,
  maximumTransitionNeighbors: 2,
  maximumIndexBucketSize: 32,
  maximumCandidateNeighbors: 8,
  maximumFamilyMembers: 16,
  maximumFamilies: 16,
  maximumMembershipsPerNeuron: 3,
  minimumFormationSimilarity: 0.7,
  minimumMembershipScore: 0,
  minimumCoreMembership: 0.7,
  minimumSupport: 2,
  minimumSources: 2,
  maximumOutgoingFamilyTransitions: 4,
  runtimeActivationThreshold: 0.1,
  maximumActiveNeurons: 32,
  maximumActiveFamilies: 8,
  maximumCoalitions: 4,
  maximumRoleCandidates: 3,
  coalitionBeamWidth: 4,
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

function profile(
  memberKey: string,
  text: string,
  sourceDigest: string,
  start: number,
  end: number,
  coordinate = 0,
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
      score: 0.5,
      verified: false,
      explanation: 'test',
    })),
    evidence: [{
      memberKey,
      analysisRunId: 'run-test',
      sourceId: sourceDigest,
      sourceDigest,
      neuronId: memberKey,
      paragraphIndex: 0,
      start,
      end,
      text,
    }],
    identityDigest: memberKey,
    text,
    derivedFromMemberKeys: [],
  }
}

function family(id: string, region: FunctionalRegionId): LearnedFamily {
  const prototype = createSparseVector(8, [{ index: 0, value: 1 }])
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

describe('geçici olay koalisyonları', () => {
  it('aktif olay ailesi tohumu yoksa sahte koalisyon üretmez', () => {
    const content = profile('object', 'bıçak', 'source-a', 0, 5)
    const result = buildEventCoalitions(
      [{ memberKey: content.memberKey, activation: 1 }],
      [content],
      [family('content-family', 'content')],
      [membership(content.memberKey, 'content-family')],
      [],
      CONFIG,
    )

    expect(result.state.coalitions).toEqual([])
    expect(result.state.unmatchedReason).toContain('olay/eylem')
    expect(result.expansionCount).toBe(0)
  })

  it('bıçakla içeriğini yalnız o koalisyonda geçici araç rolüne bağlar', () => {
    const knife = profile('knife', 'bıçakla', 'source-a', 5, 12, 1)
    const cut = profile('cut', 'kesti', 'source-a', 13, 18, 0)
    const families = [
      family('event-family', 'event'),
      family('content-family', 'content'),
    ]
    const memberships = [
      membership(cut.memberKey, 'event-family'),
      membership(knife.memberKey, 'content-family'),
    ]
    const familiesBefore = structuredClone(families)
    const membershipsBefore = structuredClone(memberships)

    const result = buildEventCoalitions(
      [
        { memberKey: cut.memberKey, activation: 1 },
        { memberKey: knife.memberKey, activation: 0.9 },
      ],
      [knife, cut],
      families,
      memberships,
      [],
      { ...CONFIG, maximumRoleCandidates: 1 },
    )
    const binding = result.state.coalitions[0]?.bindings[0]

    expect(binding).toMatchObject({
      memberKey: 'knife',
      role: 'instrument-hypothesis',
      temporary: true,
      roleFamilyId: null,
    })
    expect(result.state.coalitions[0].temporary).toBe(true)
    expect(families).toEqual(familiesBefore)
    expect(memberships).toEqual(membershipsBefore)
    expect(memberships.some((item) => item.familyId.includes('instrument'))).toBe(false)
  })

  it('aile aktivasyonunu üye sayısına göre normalize ederek büyük aile yanlılığını sınırlar', () => {
    const event = profile('event', 'oldu', 'source-a', 20, 24)
    const bigMembers = Array.from({ length: 4 }, (_, index) =>
      profile(`big-${index}`, `b${index}`, 'source-a', index * 2, index * 2 + 1, 1))
    const small = profile('small', 's', 'source-a', 10, 11, 2)
    const families = [
      family('event-family', 'event'),
      family('big-family', 'content'),
      family('small-family', 'content'),
    ]
    const memberships = [
      membership(event.memberKey, 'event-family'),
      ...bigMembers.map((item) => membership(item.memberKey, 'big-family')),
      membership(small.memberKey, 'small-family'),
    ]
    const activations: NeuronActivation[] = [
      { memberKey: event.memberKey, activation: 0.8 },
      ...bigMembers.map((item) => ({ memberKey: item.memberKey, activation: 0.25 })),
      { memberKey: small.memberKey, activation: 0.5 },
    ]

    const result = buildEventCoalitions(
      activations,
      [event, ...bigMembers, small],
      families,
      memberships,
      [],
      CONFIG,
    )
    const activationsByFamily = new Map(
      result.state.coalitions[0].familyActivations.map((item) => [item.familyId, item.activation]),
    )

    expect(activationsByFamily.get('big-family')).toBeCloseTo(0.5)
    expect(activationsByFamily.get('small-family')).toBeCloseTo(0.5)
  })

  it('aday genişlemesini beam ve rol sınırlarında tutar', () => {
    const event = profile('event', 'oldu', 'source-a', 100, 104)
    const candidates = Array.from({ length: 20 }, (_, index) =>
      profile(`candidate-${index}`, `aday${index}`, 'source-a', index * 3, index * 3 + 2, (index % 6) + 1))
    const families = [family('event-family', 'event'), family('content-family', 'content')]
    const memberships = [
      membership(event.memberKey, 'event-family'),
      ...candidates.map((item) => membership(item.memberKey, 'content-family')),
    ]
    const result = buildEventCoalitions(
      [
        { memberKey: event.memberKey, activation: 1 },
        ...candidates.map((item) => ({ memberKey: item.memberKey, activation: 0.5 })),
      ],
      [event, ...candidates],
      families,
      memberships,
      [],
      { ...CONFIG, coalitionBeamWidth: 3, maximumRoleCandidates: 2 },
    )

    expect(result.peakBeamWidth).toBeLessThanOrEqual(3)
    expect(result.state.coalitions).toHaveLength(1)
    expect(result.state.coalitions[0].bindings.length).toBeLessThanOrEqual(2)
    expect(result.state.coalitions.reduce((sum, item) => sum + item.dominance, 0)).toBeCloseTo(1)
  })

  it('önceki baskın düşüncenin tekrarı cooldown üretip aynı koşuldaki puanı düşürür', () => {
    const event = profile('event', 'oldu', 'source-a', 10, 14)
    const participant = profile('participant', 'Ali', 'source-a', 0, 3, 1)
    const families = [family('event-family', 'event'), family('content-family', 'content')]
    const memberships = [
      membership(event.memberKey, 'event-family'),
      membership(participant.memberKey, 'content-family'),
    ]
    const initial = buildEventCoalitions(
      [
        { memberKey: event.memberKey, activation: 1 },
        { memberKey: participant.memberKey, activation: 1 },
      ],
      [event, participant],
      families,
      memberships,
      [],
      CONFIG,
    ).state
    const previous: CoalitionState = {
      ...initial,
      coalitions: initial.coalitions.map((coalition) => ({ ...coalition, dominance: 1 })),
    }
    const quiet = buildEventCoalitions(
      [
        { memberKey: event.memberKey, activation: 1, repetition: 0 },
        { memberKey: participant.memberKey, activation: 1, repetition: 0 },
      ],
      [event, participant],
      families,
      memberships,
      [],
      CONFIG,
      previous,
    ).state.coalitions[0]
    const repeated = buildEventCoalitions(
      [
        { memberKey: event.memberKey, activation: 1, repetition: 1 },
        { memberKey: participant.memberKey, activation: 1, repetition: 1 },
      ],
      [event, participant],
      families,
      memberships,
      [],
      CONFIG,
      previous,
    ).state.coalitions[0]

    expect(repeated.score.cooldown).toBeGreaterThan(0)
    expect(repeated.score.total).toBeLessThan(quiet.score.total)
  })
})
