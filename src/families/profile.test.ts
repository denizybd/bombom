import { describe, expect, it } from 'vitest'
import { createMatrixLayout, DEFAULT_TEXT_LAB_CONFIG } from '../text-lab/domain/config'
import { digestText, digestVector } from '../text-lab/engine/representation'
import {
  assertIdentityUnchanged,
  buildBehaviorProfiles,
  createBehaviorSpaceId,
  profileEvidenceConfidence,
  profilesFingerprint,
  selectIndependentEvidence,
  snapshotIdentityDigests,
} from './profile'
import type {
  FamilyAnalysisInput,
  FamilyConfig,
  FamilyNeuronInput,
  IndependentEvidenceRef,
} from './types'

const FAMILY_CONFIG: FamilyConfig = {
  profileDimension: 128,
  maximumProfileFeatures: 32,
  contextRadius: 2,
  maximumTransitionNeighbors: 3,
  maximumIndexBucketSize: 64,
  maximumCandidateNeighbors: 16,
  maximumFamilyMembers: 32,
  maximumFamilies: 32,
  maximumMembershipsPerNeuron: 3,
  minimumFormationSimilarity: 0.7,
  minimumMembershipScore: 0.5,
  minimumCoreMembership: 0.75,
  minimumSupport: 3,
  minimumSources: 2,
  maximumOutgoingFamilyTransitions: 4,
  runtimeActivationThreshold: 0.1,
  maximumActiveNeurons: 32,
  maximumActiveFamilies: 8,
  maximumCoalitions: 8,
  maximumRoleCandidates: 4,
  coalitionBeamWidth: 8,
  routingCandidateLimit: 16,
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
  hashSeed: 0x1234abcd,
}

const textConfig = {
  ...DEFAULT_TEXT_LAB_CONFIG,
  matrixSide: 8,
  identityRatio: 0.5,
  minimumLeafGraphemes: 1,
}

interface NodeSpec {
  id: string
  start: number
  end: number
  text: string
  identityIndex: number
  relationIndex: number
}

function vector(index: number, value = 1): Float32Array {
  const output = new Float32Array(32)
  output[index] = value
  return output
}

function makeNode(sourceId: string, spec: NodeSpec): FamilyNeuronInput {
  const identity = vector(spec.identityIndex)
  const relation = vector(spec.relationIndex, 0.75)
  return {
    id: spec.id,
    sourceId,
    paragraphIndex: 0,
    start: spec.start,
    end: spec.end,
    text: spec.text,
    graphemeCount: Array.from(new Intl.Segmenter('tr', { granularity: 'grapheme' }).segment(spec.text)).length,
    depth: 0,
    branch: 'root',
    parentId: null,
    childIds: [],
    splitBoundary: 'leaf',
    identity,
    relation,
    identityDigest: digestVector(identity),
    identityNorm: 1,
    relationNorm: 0.75,
    contextSupport: 1,
    contextRevision: 1,
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

function makeAnalysis(
  sourceId = 'source-a',
  runId = 'run-a',
  order: readonly number[] = [0, 1, 2],
): FamilyAnalysisInput {
  const rawText = 'bıçak kesti'
  const specs: NodeSpec[] = [
    { id: 'knife', start: 0, end: 5, text: 'bıçak', identityIndex: 1, relationIndex: 7 },
    { id: 'space', start: 5, end: 6, text: ' ', identityIndex: 2, relationIndex: 8 },
    { id: 'cut', start: 6, end: 11, text: 'kesti', identityIndex: 3, relationIndex: 9 },
  ]
  const canonicalNodes = specs.map((spec) => makeNode(sourceId, spec))
  const baseHabits = Float32Array.from([
    0, 0.7, 0.3,
    0.2, 0, 0.8,
    0.4, 0.6, 0,
  ])
  const nodes = order.map((index) => canonicalNodes[index])
  const habitProbabilities = new Float32Array(nodes.length * nodes.length)
  for (let row = 0; row < order.length; row += 1) {
    for (let column = 0; column < order.length; column += 1) {
      habitProbabilities[row * order.length + column] =
        baseHabits[order[row] * order.length + order[column]]
    }
  }
  const layout = createMatrixLayout(textConfig)
  return {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId,
    createdAt: '2026-08-28T00:00:00.000Z',
    sourceDigest: digestText(rawText),
    identitySpaceId: 'IS-test',
    contextSpaceId: 'CS-test',
    representationSpaceId: 'RS-test',
    source: { id: sourceId, title: sourceId, rawText, text: rawText },
    config: textConfig,
    layout,
    rootIds: nodes.map((node) => node.id),
    nodes,
    habitProbabilities,
  }
}

function bytes(value: Float32Array): number[] {
  return Array.from(new Uint8Array(value.buffer, value.byteOffset, value.byteLength))
}

function evidence(
  memberKey: string,
  sourceDigest: string,
  start: number,
  end: number,
  overrides: Partial<IndependentEvidenceRef> = {},
): IndependentEvidenceRef {
  return {
    memberKey,
    analysisRunId: `run-${memberKey}`,
    sourceId: `source-${memberKey}`,
    sourceDigest,
    neuronId: `neuron-${memberKey}`,
    paragraphIndex: 0,
    start,
    end,
    text: memberKey,
    ...overrides,
  }
}

describe('davranış profili', () => {
  it('analiz ve nöron girdi sırasından bağımsız aynı profil fingerprintini üretir', () => {
    const firstAnalysis = makeAnalysis('source-a', 'run-a', [0, 1, 2])
    const reorderedAnalysis = makeAnalysis('source-a', 'run-a', [2, 0, 1])
    const otherAnalysis = makeAnalysis('source-b', 'run-b', [1, 2, 0])
    const behaviorSpaceId = createBehaviorSpaceId('RS-test', [firstAnalysis, otherAnalysis], FAMILY_CONFIG)

    const first = buildBehaviorProfiles(
      [firstAnalysis, otherAnalysis],
      [],
      behaviorSpaceId,
      FAMILY_CONFIG,
    )
    const second = buildBehaviorProfiles(
      [otherAnalysis, reorderedAnalysis],
      [],
      createBehaviorSpaceId('RS-test', [otherAnalysis, reorderedAnalysis], FAMILY_CONFIG),
      FAMILY_CONFIG,
    )

    expect(second.profiles.map((profile) => profile.memberKey)).toEqual(
      first.profiles.map((profile) => profile.memberKey),
    )
    expect(profilesFingerprint(second.profiles)).toBe(profilesFingerprint(first.profiles))
    expect(second.profiles.map((profile) => profile.vector)).toEqual(
      first.profiles.map((profile) => profile.vector),
    )
  })

  it('profil feature sayısını sınırlar ve yedi bölgenin tamamını ayrı hipotez olarak taşır', () => {
    const analysis = makeAnalysis()
    const result = buildBehaviorProfiles(
      [analysis],
      [],
      createBehaviorSpaceId('RS-test', [analysis], FAMILY_CONFIG),
      { ...FAMILY_CONFIG, maximumProfileFeatures: 5 },
    )

    for (const profile of result.profiles) {
      expect(profile.vector.entries.length).toBeLessThanOrEqual(5)
      expect(profile.regionAffinities.map((affinity) => affinity.region)).toEqual([
        'content', 'event', 'role', 'operator', 'goal', 'language', 'control',
      ])
      expect(profile.regionAffinities.every((affinity) => affinity.verified === false)).toBe(true)
    }
  })

  it('aynı kaynağın tekrarını ve örtüşen soy aralıklarını bağımsız destek saymaz', () => {
    const selected = selectIndependentEvidence([
      evidence('parent', 'same-source', 0, 10),
      evidence('child', 'same-source', 0, 4),
      evidence('duplicate-child', 'same-source', 0, 4, { analysisRunId: 'duplicate-run' }),
      evidence('disjoint', 'same-source', 4, 8),
      evidence('other-source', 'other-source', 0, 10),
    ])

    expect(selected.map((item) => item.memberKey)).toEqual(['other-source', 'child', 'disjoint'])
    expect(new Set(selected.map((item) => item.sourceDigest)).size).toBe(2)
  })

  it('yeniden adlandırılmış aynı içerik üyelik güvenini şişirmez', () => {
    const base = buildBehaviorProfiles(
      [makeAnalysis()],
      [],
      'BS-test',
      FAMILY_CONFIG,
    ).profiles[0]
    const copied = {
      ...base,
      evidence: [
        evidence('copy-a', 'same-digest', 0, 4, { sourceId: 'book-a' }),
        evidence('copy-b', 'same-digest', 0, 4, { sourceId: 'book-renamed' }),
        evidence('copy-c', 'same-digest', 0, 4, { sourceId: 'book-third-name' }),
      ],
    }
    const independent = {
      ...base,
      evidence: [
        evidence('independent-a', 'digest-a', 0, 4, { sourceId: 'book-a' }),
        evidence('independent-b', 'digest-b', 0, 4, { sourceId: 'book-b' }),
        evidence('independent-c', 'digest-c', 0, 4, { sourceId: 'book-c' }),
      ],
    }

    expect(profileEvidenceConfidence(copied)).toBeLessThan(profileEvidenceConfidence(independent))
  })

  it('profil çıkarırken I ve R byte alanlarına dokunmaz', () => {
    const analysis = makeAnalysis()
    const identityBefore = analysis.nodes.map((node) => bytes(node.identity))
    const relationBefore = analysis.nodes.map((node) => bytes(node.relation))
    const digests = snapshotIdentityDigests([analysis])

    buildBehaviorProfiles(
      [analysis],
      [],
      createBehaviorSpaceId('RS-test', [analysis], FAMILY_CONFIG),
      FAMILY_CONFIG,
    )

    expect(analysis.nodes.map((node) => bytes(node.identity))).toEqual(identityBefore)
    expect(analysis.nodes.map((node) => bytes(node.relation))).toEqual(relationBefore)
    expect(() => assertIdentityUnchanged([analysis], digests)).not.toThrow()

    analysis.nodes[0].identity[0] = 0.25
    expect(() => assertIdentityUnchanged([analysis], digests)).toThrow('sabit kimlik')
  })

  it('iptal sinyalini profil döngüsünde gecikmeden uygular', () => {
    const analysis = makeAnalysis()
    let checks = 0
    expect(() => buildBehaviorProfiles(
      [analysis],
      [],
      createBehaviorSpaceId('RS-test', [analysis], FAMILY_CONFIG),
      FAMILY_CONFIG,
      undefined,
      () => {
        checks += 1
        return checks === 2
      },
    )).toThrow('FAMILY_CANCELLED')
    expect(checks).toBe(2)
  })
})
