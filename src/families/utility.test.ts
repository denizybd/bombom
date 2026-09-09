import { describe, expect, it } from 'vitest'
import { digestVector } from '../text-lab/engine/representation'
import { rawMemberKey } from './profile'
import { createSparseVector, sparseVectorDigest } from './sparse'
import {
  applyUtilityLifecycle,
  measureFamilyAblationUtilities,
} from './utility'
import type {
  FamilyAnalysisInput,
  FamilyConfig,
  FamilyMembership,
  FamilyNeuronInput,
  LearnedFamily,
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
  utilityMinimumCases: 3,
  utilityAlternativeCount: 1,
  utilityEpsilon: 1e-6,
  lifecycleStabilityThreshold: 0.7,
  lifecycleUtilityThreshold: 0,
  yieldEveryOperations: 16,
  hashSeed: 1,
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
  physicalSourceId = sourceDigest,
  independenceGroupId?: string,
): FamilyAnalysisInput {
  const ids = [`a-${runId}`, `b-${runId}`, `c-${runId}`]
  const nodes = ids.map((id, index) => node(id, physicalSourceId, index))
  return {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId,
    createdAt: '2026-08-28T00:00:00.000Z',
    independenceGroupId,
    sourceDigest,
    identitySpaceId: 'IS-test',
    contextSpaceId: 'CS-test',
    representationSpaceId: 'RS-test',
    source: { id: physicalSourceId, title: sourceDigest, rawText: ids.join(''), text: ids.join('') },
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
    habitProbabilities: new Float32Array(9),
  }
}

function membership(memberKey: string, familyId: string): FamilyMembership {
  return {
    memberKey,
    familyId,
    rank: 1,
    score: 1,
    weight: 1,
    core: true,
    breakdown: {
      behaviorSimilarity: 1,
      regionCompatibility: 1,
      evidenceConfidence: 1,
      combined: 1,
    },
  }
}

function family(id: string, lifecycle: LearnedFamily['lifecycle'] = 'strengthening'): LearnedFamily {
  const prototype = createSparseVector(8, [{ index: 0, value: 1 }])
  return {
    id,
    fingerprint: `fingerprint-${id}`,
    behaviorSpaceId: 'BS-test',
    region: 'content',
    regionVerified: false,
    lifecycle,
    lifecycleReason: 'test',
    prototype,
    prototypeDigest: sparseVectorDigest(prototype),
    memberKeys: [],
    coreMemberKeys: [],
    independentEvidence: [],
    supportCount: 6,
    sourceCount: 3,
    cohesion: 1,
    separation: 1,
    stability: 0.9,
    utility: null,
    predecessorFamilyIds: [],
  }
}

function membershipsFor(analyses: readonly FamilyAnalysisInput[]): FamilyMembership[] {
  return analyses.flatMap((item) => {
    const keys = item.nodes.map((entry) => rawMemberKey(item.sourceDigest, item.runId, entry.id))
    return [
      membership(keys[0], 'family-a'),
      membership(keys[1], 'family-b'),
      membership(keys[2], 'family-a'),
    ]
  })
}

function competingMemberships(analyses: readonly FamilyAnalysisInput[]): FamilyMembership[] {
  return analyses.flatMap((item) => {
    const keys = item.nodes.map((entry) => rawMemberKey(item.sourceDigest, item.runId, entry.id))
    const sequence = item.source.id === 'book-b'
      ? ['family-x', 'family-z', 'family-y']
      : ['family-x', 'family-y', 'family-z']
    return keys.map((key, index) => membership(key, sequence[index]))
  })
}

describe('kaynak-holdout aile susturma testi', () => {
  it('her foldun bütün sourceDigest kaynağını eğitimden ayırdığı pozitif faydayı ölçer', () => {
    const analyses = [
      analysis('source-a', 'run-a'),
      analysis('source-b', 'run-b'),
      analysis('source-c', 'run-c'),
    ]
    const reports = measureFamilyAblationUtilities(
      analyses,
      [family('family-b')],
      membershipsFor(analyses),
      CONFIG,
    )
    const report = reports[0]

    expect(report.mode).toBe('source-holdout-diagnostic')
    expect(report.sourceFoldCount).toBe(3)
    expect(report.folds.map((fold) => fold.heldOutSourceDigest)).toEqual([
      'source-a', 'source-b', 'source-c',
    ])
    expect(report.caseCount).toBe(6)
    expect(report.deltaUtility).toBeGreaterThan(0)
    expect(report.confidenceLow).toBeGreaterThan(0)
    expect(report.status).toBe('validated')
  })

  it('bağımsız eğitim kaynağı yetersizse fayda iddiasında bulunmaz', () => {
    const analyses = [
      analysis('source-a', 'run-a'),
      analysis('source-b', 'run-b'),
    ]
    const report = measureFamilyAblationUtilities(
      analyses,
      [family('family-b')],
      membershipsFor(analyses),
      CONFIG,
    )[0]

    expect(report.status).toBe('insufficient-evidence')
    expect(report.sourceFoldCount).toBe(0)
    expect(report.caseCount).toBe(0)
  })

  it('aynı kitabın analiz parçalarını tek holdout fold olarak birlikte tutar', () => {
    const analyses = [
      analysis('book-a-chunk-1', 'run-a1', 'book-a'),
      analysis('book-a-chunk-2', 'run-a2', 'book-a'),
      analysis('book-b', 'run-b', 'book-b'),
      analysis('book-c', 'run-c', 'book-c'),
    ]
    const report = measureFamilyAblationUtilities(
      analyses,
      [family('family-b')],
      membershipsFor(analyses),
      CONFIG,
    )[0]

    expect(report.sourceFoldCount).toBe(3)
    expect(report.folds.map((fold) => fold.heldOutSourceId)).toEqual(['book-a', 'book-b', 'book-c'])
    expect(report.folds.find((fold) => fold.heldOutSourceId === 'book-a')?.caseCount).toBe(4)
  })

  it('aynı içerik özetini farklı kaynak adıyla yeniden yüklemeyi ayrı holdout saymaz', () => {
    const analyses = [
      analysis('same-content', 'run-original', 'book-original'),
      analysis('same-content', 'run-copy', 'renamed-copy'),
      analysis('other-b', 'run-b', 'book-b'),
      analysis('other-c', 'run-c', 'book-c'),
    ]
    const report = measureFamilyAblationUtilities(
      analyses,
      [family('family-b')],
      membershipsFor(analyses),
      CONFIG,
    )[0]

    expect(report.sourceFoldCount).toBe(3)
    expect(report.folds.find((fold) => fold.heldOutSourceId === 'book-original')?.caseCount).toBe(4)
    expect(report.folds.some((fold) => fold.heldOutSourceId === 'renamed-copy')).toBe(false)
  })

  it('aynı Ollama üretim soyundaki farklı metinleri tek holdout foldta tutar', () => {
    const analyses = [
      analysis('lesson-a', 'run-a', 'lesson-a', 'ollama:qwen14b'),
      analysis('lesson-b', 'run-b', 'lesson-b', 'ollama:qwen14b'),
      analysis('book-b', 'run-c', 'book-b', 'human:book-b'),
      analysis('book-c', 'run-d', 'book-c', 'human:book-c'),
    ]
    const report = measureFamilyAblationUtilities(
      analyses,
      [family('family-b')],
      membershipsFor(analyses),
      CONFIG,
    )[0]

    expect(report.sourceFoldCount).toBe(3)
    expect(report.folds.find((fold) => fold.heldOutSourceId === 'lesson-a')?.caseCount).toBe(4)
    expect(report.folds.some((fold) => fold.heldOutSourceId === 'lesson-b')).toBe(false)
  })

  it('bir kitabın fazla chunkının eğitim geçiş oyunu ağırlaştırmasına izin vermez', () => {
    const firstChunk = analysis('book-a-chunk-1', 'run-a1', 'book-a')
    const secondChunk = analysis('book-a-chunk-2', 'run-a2', 'book-a')
    const other = analysis('book-b', 'run-b', 'book-b')
    const heldOut = analysis('book-c', 'run-c', 'book-c')
    const base = [firstChunk, other, heldOut]
    const chunked = [firstChunk, secondChunk, other, heldOut]
    const baseReport = measureFamilyAblationUtilities(
      base,
      [family('family-y')],
      competingMemberships(base),
      CONFIG,
    )[0]
    const chunkedReport = measureFamilyAblationUtilities(
      chunked,
      [family('family-y')],
      competingMemberships(chunked),
      CONFIG,
    )[0]
    const baseFold = baseReport.folds.find((fold) => fold.heldOutSourceId === 'book-c')
    const chunkedFold = chunkedReport.folds.find((fold) => fold.heldOutSourceId === 'book-c')

    expect(baseFold).toBeDefined()
    expect(chunkedFold).toBeDefined()
    expect(chunkedFold?.fullLoss).toBeCloseTo(baseFold?.fullLoss ?? Number.NaN, 12)
    expect(chunkedFold?.ablatedLoss).toBeCloseTo(baseFold?.ablatedLoss ?? Number.NaN, 12)
  })

  it('yalnız kararlı ve doğrulanmış strengthening ailesini stable durumuna geçirir', () => {
    const analyses = [
      analysis('source-a', 'run-a'),
      analysis('source-b', 'run-b'),
      analysis('source-c', 'run-c'),
    ]
    const strengthening = family('family-b')
    const provisional = family('family-provisional', 'provisional')
    const reports = measureFamilyAblationUtilities(
      analyses,
      [strengthening],
      membershipsFor(analyses),
      CONFIG,
    )
    const updated = applyUtilityLifecycle([strengthening, provisional], reports, CONFIG)

    expect(updated.find((item) => item.id === strengthening.id)?.lifecycle).toBe('stable')
    expect(updated.find((item) => item.id === provisional.id)?.lifecycle).toBe('provisional')
  })

  it('aile döngüsündeki iptal sinyalini uygular', () => {
    const analyses = [
      analysis('source-a', 'run-a'),
      analysis('source-b', 'run-b'),
      analysis('source-c', 'run-c'),
    ]
    let checks = 0
    expect(() => measureFamilyAblationUtilities(
      analyses,
      [family('family-b'), family('family-c')],
      membershipsFor(analyses),
      CONFIG,
      () => {
        checks += 1
        return checks === 2
      },
    )).toThrow('FAMILY_CANCELLED')
    expect(checks).toBe(2)
  })
})
