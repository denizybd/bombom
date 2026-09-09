import { describe, expect, it } from 'vitest'
import { applyRegionLifecycleProposal, proposeRegionBirth } from './lifecycle.ts'
import { computePageWriteMask, computeRegionGates, regionAwareCosine } from './gating.ts'
import { ADAPTIVE_RELATION_LENGTH, relationIndexFromPage } from './geometry.ts'
import { cosineRegionVectors } from './math.ts'
import { strongEvidence, testAtlas, testPrototype } from './test-fixtures.ts'

function twoRegionAtlas() {
  const first = applyRegionLifecycleProposal(testAtlas(), proposeRegionBirth(testAtlas(), {
    regionId: 'region-a',
    allocations: [{ pageIndex: 0, weight: 1 }],
    prototype: testPrototype(1, 0),
    evidence: strongEvidence('birth-a'),
  }))
  return applyRegionLifecycleProposal(first, proposeRegionBirth(first, {
    regionId: 'region-b',
    allocations: [{ pageIndex: 1, weight: 1 }],
    prototype: testPrototype(0, 1),
    evidence: strongEvidence('birth-b'),
  }))
}

describe('yumuşak bölge kapıları', () => {
  it('unmatched ile birlikte 1 toplamını korur ve budanan kütleyi unmatched kanalına aktarır', () => {
    const atlas = twoRegionAtlas()
    const compatibility = new Array(56).fill(0)
    compatibility[0] = 1
    compatibility[1] = -1
    const gates = computeRegionGates(atlas, {
      pageCompatibilities: compatibility,
      confidenceWeight: 0,
      maximumActiveRegions: 1,
      unmatchedLogit: 0,
    })
    expect(gates.gates).toHaveLength(1)
    expect(gates.gates[0].regionId).toBe('region-a')
    expect(gates.omittedRegionProbability).toBeGreaterThan(0)
    expect(gates.gates[0].probability + gates.unmatchedProbability).toBeCloseTo(1, 6)
  })

  it('m_min izini korur ve yalnız aidiyet/serbest kütle üzerinden yazma maskesi üretir', () => {
    const atlas = twoRegionAtlas()
    const compatibility = new Array(56).fill(0)
    compatibility[0] = 1
    compatibility[1] = -1
    const gates = computeRegionGates(atlas, {
      pageCompatibilities: compatibility,
      confidenceWeight: 0,
      maximumActiveRegions: 1,
    })
    const mask = computePageWriteMask(atlas, gates, 0.1)
    expect(mask).toHaveLength(56)
    expect(mask[0]).toBeCloseTo(0.1 + 0.9 * gates.gates[0].probability, 6)
    expect(mask[1]).toBeCloseTo(0.1, 6)
    expect(mask[2]).toBeCloseTo(0.1 + 0.9 * gates.unmatchedProbability, 6)
    expect(mask.every((value) => value >= 0.1 && value <= 1)).toBe(true)
  })

  it('aktif bölge bütçesi sıfırken bütün bölge kütlesini unmatched kanalında korur', () => {
    const atlas = twoRegionAtlas()
    const gates = computeRegionGates(atlas, {
      pageCompatibilities: new Array(56).fill(0),
      maximumActiveRegions: 0,
    })
    expect(gates.gates).toEqual([])
    expect(gates.unmatchedProbability).toBeCloseTo(1, 7)
    expect(gates.omittedRegionProbability).toBeGreaterThan(0)
  })

  it('boş atlası tam R cosine ile birebir uyumlu fallback olarak kullanır', () => {
    const atlas = testAtlas()
    const left = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    const right = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    left[relationIndexFromPage(0, 0)] = 1
    left[relationIndexFromPage(20, 8)] = -0.5
    right[relationIndexFromPage(0, 0)] = 0.5
    right[relationIndexFromPage(20, 8)] = -1
    const expected = cosineRegionVectors(left, right)
    const result = regionAwareCosine(atlas, left, right)
    expect(result.gates.gates).toEqual([])
    expect(result.gates.unmatchedProbability).toBe(1)
    expect(result.pageWeights.every((value) => value === 1)).toBe(true)
    expect(result.similarity).toBeCloseTo(expected, 7)
  })

  it('aynı sayfa aidiyetindeki zıt prototipleri gözlenen davranışla ayırır', () => {
    const base = testAtlas()
    const first = applyRegionLifecycleProposal(base, proposeRegionBirth(base, {
      regionId: 'prototype-positive', allocations: [
        { pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 },
      ],
      prototype: testPrototype(1, 0), evidence: strongEvidence('prototype-positive'),
    }))
    const atlas = applyRegionLifecycleProposal(first, proposeRegionBirth(first, {
      regionId: 'prototype-negative', allocations: [
        { pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 },
      ],
      prototype: testPrototype(-1, 0), evidence: strongEvidence('prototype-negative'),
    }))
    const pageBehaviors = Array.from({ length: 56 }, () => [0, 0])
    pageBehaviors[0] = [1, 0]
    pageBehaviors[1] = [1, 0]
    const gates = computeRegionGates(atlas, {
      pageCompatibilities: new Array(56).fill(0),
      pageBehaviors,
      compatibilityWeight: 0,
      confidenceWeight: 0,
      maximumActiveRegions: 2,
    })
    expect(gates.gates.map((gate) => gate.regionId)).toEqual(['prototype-positive', 'prototype-negative'])
    expect(gates.gates[0].prototypeCompatibility).toBe(1)
    expect(gates.gates[1].prototypeCompatibility).toBe(-1)
    expect(gates.gates[0].probability).toBeGreaterThan(gates.gates[1].probability)
  })

  it('provisional bölgeyi aktif yönlendirme kapısına almaz', () => {
    const proposal = proposeRegionBirth(testAtlas(), {
      regionId: 'provisional-only', allocations: [{ pageIndex: 0, weight: 0.5 }],
      prototype: testPrototype(),
      evidence: {
        evidenceId: 'weak-routing', validationAuthority: 'independent-holdout',
        observationCount: 1, sourceGroupIds: ['one'], confidenceMass: 0.2,
        cohesion: 0.2, holdoutGain: 0, shamGain: 0, codes: ['unexplained-residual'],
      },
    })
    expect(proposal.proposedAtlas.regions[0].lifecycle).toBe('provisional')
    const gates = computeRegionGates(proposal.proposedAtlas, {
      pageCompatibilities: new Array(56).fill(1),
      pageBehaviors: Array.from({ length: 56 }, () => [1, 0]),
    })
    expect(gates.gates).toEqual([])
    expect(gates.unmatchedProbability).toBe(1)
  })
})
