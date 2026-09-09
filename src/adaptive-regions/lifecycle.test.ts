import { describe, expect, it } from 'vitest'
import { totalPlasticRegionCapacity, validateAdaptiveRegionAtlas } from './atlas.ts'
import {
  applyRegionLifecycleProposal,
  proposeRegionBirth,
  proposeRegionGrow,
  proposeRegionMerge,
  proposeRegionRetire,
  proposeRegionShrink,
  proposeRegionSplit,
} from './lifecycle.ts'
import { strongEvidence, testAtlas, testPrototype } from './test-fixtures.ts'

function expectConserved(atlas: ReturnType<typeof testAtlas>): void {
  expect(totalPlasticRegionCapacity(atlas).total).toBeCloseTo(56, 5)
  for (const page of atlas.pages) {
    expect(page.freeMass + page.memberships.reduce((sum, membership) => sum + membership.weight, 0)).toBeCloseTo(1, 6)
  }
  expect(() => validateAdaptiveRegionAtlas(atlas)).not.toThrow()
}

describe('kanıtlı ve değişmez plastik bölge yaşam döngüsü', () => {
  it('doğumda serbest kütleyi tüketir, fingerprint zinciri kurar ve tabanı değiştirmez', () => {
    const base = testAtlas()
    const proposal = proposeRegionBirth(base, {
      regionId: 'region-a',
      allocations: [{ pageIndex: 0, weight: 0.5 }, { pageIndex: 8, weight: 0.5 }],
      prototype: testPrototype(),
      evidence: strongEvidence('birth'),
    })
    expect(proposal.eligible).toBe(true)
    expect(proposal.pageDeltas.map((delta) => delta.pageIndex)).toEqual([0, 8])
    expect(base.regions).toHaveLength(0)
    expect(base.pages[0].freeMass).toBe(1)
    const born = applyRegionLifecycleProposal(base, proposal)
    expect(born.parentFingerprint).toBe(base.fingerprint)
    expect(born.revision).toBe(1)
    expect(born.pages[0]).toMatchObject({ freeMass: 0.5 })
    expect(born.pages[0].memberships).toEqual([{ regionId: 'region-a', weight: 0.5 }])
    expectConserved(born)
  })

  it('zayıf kanıtı proposal olarak kaydeder fakat varsayılan olarak uygulatmaz', () => {
    const base = testAtlas()
    const proposal = proposeRegionBirth(base, {
      regionId: 'region-weak',
      allocations: [{ pageIndex: 0, weight: 0.25 }],
      prototype: testPrototype(),
      evidence: {
        evidenceId: 'weak', validationAuthority: 'independent-holdout',
        observationCount: 1, sourceGroupIds: ['one'], confidenceMass: 0.25,
        cohesion: 0.2, holdoutGain: -0.1, shamGain: 0.2, codes: ['unexplained-residual'],
      },
    })
    expect(proposal.eligible).toBe(false)
    expect(proposal.rejectionReasons).toEqual([
      'INSUFFICIENT_OBSERVATIONS',
      'INSUFFICIENT_INDEPENDENT_SOURCES',
      'INSUFFICIENT_CONFIDENCE_MASS',
      'INSUFFICIENT_COHESION',
      'SHAM_EFFECT_TOO_HIGH',
      'INSUFFICIENT_HOLDOUT_GAIN',
      'INSUFFICIENT_REGION_CAPACITY',
    ])
    expect(() => applyRegionLifecycleProposal(base, proposal)).toThrow(/Uygun olmayan/)
  })

  it('büyüme ve küçülmede aidiyet ile serbest kütleyi eşit miktarda değiştirir', () => {
    const base = testAtlas()
    const born = applyRegionLifecycleProposal(base, proposeRegionBirth(base, {
      regionId: 'region-a', allocations: [{ pageIndex: 0, weight: 0.5 }, { pageIndex: 2, weight: 0.5 }],
      prototype: testPrototype(), evidence: strongEvidence('birth'),
    }))
    const grown = applyRegionLifecycleProposal(born, proposeRegionGrow(born, {
      regionId: 'region-a',
      allocations: [{ pageIndex: 0, weight: 0.25 }, { pageIndex: 1, weight: 0.5 }],
      evidence: strongEvidence('grow', ['capacity-pressure']),
    }))
    expect(grown.pages[0].memberships[0].weight).toBeCloseTo(0.75, 7)
    expect(grown.pages[1].freeMass).toBeCloseTo(0.5, 7)
    const shrunk = applyRegionLifecycleProposal(grown, proposeRegionShrink(grown, {
      regionId: 'region-a', allocations: [{ pageIndex: 0, weight: 0.1 }],
      evidence: strongEvidence('shrink', ['low-unique-utility']),
    }))
    expect(shrunk.pages[0].memberships[0].weight).toBeCloseTo(0.65, 6)
    expect(shrunk.pages[0].freeMass).toBeCloseTo(0.35, 6)
    expectConserved(shrunk)
  })

  it('bölünmede kaynak tahsisini iki çocuğa kayıpsız aktarır ve ebeveyni emekli eder', () => {
    const base = testAtlas()
    const born = applyRegionLifecycleProposal(base, proposeRegionBirth(base, {
      regionId: 'parent',
      allocations: [{ pageIndex: 0, weight: 1 }, { pageIndex: 1, weight: 1 }],
      prototype: testPrototype(), evidence: strongEvidence('birth'),
    }))
    const splitProposal = proposeRegionSplit(born, {
      sourceRegionId: 'parent', firstChildId: 'child-a', secondChildId: 'child-b',
      firstChildPrototype: testPrototype(1, 0), secondChildPrototype: testPrototype(0, 1),
      fractions: [
        { pageIndex: 0, firstChildFraction: 0.5 },
        { pageIndex: 1, firstChildFraction: 0.5 },
      ],
      evidence: strongEvidence('split', ['bimodal-behavior']),
    })
    const split = applyRegionLifecycleProposal(born, splitProposal)
    expect(split.regions.find((region) => region.id === 'parent')).toMatchObject({ lifecycle: 'retired', retiredRevision: 2 })
    expect(split.pages[0].memberships).toEqual([
      { regionId: 'child-a', weight: expect.closeTo(0.5, 6) },
      { regionId: 'child-b', weight: expect.closeTo(0.5, 6) },
    ])
    expect(split.pages[1].memberships).toEqual([
      { regionId: 'child-a', weight: expect.closeTo(0.5, 6) },
      { regionId: 'child-b', weight: expect.closeTo(0.5, 6) },
    ])
    expectConserved(split)
  })

  it('birleşmede iki tahsisi yeni latent bölgede toplar; emeklilikte tamamını serbest bırakır', () => {
    const base = testAtlas()
    const a = applyRegionLifecycleProposal(base, proposeRegionBirth(base, {
      regionId: 'region-a', allocations: [{ pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 }],
      prototype: testPrototype(), evidence: strongEvidence('a'),
    }))
    const b = applyRegionLifecycleProposal(a, proposeRegionBirth(a, {
      regionId: 'region-b', allocations: [{ pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 }],
      prototype: testPrototype(0, 1), evidence: strongEvidence('b'),
    }))
    const merged = applyRegionLifecycleProposal(b, proposeRegionMerge(b, {
      sourceRegionIds: ['region-b', 'region-a'], mergedRegionId: 'region-merged',
      prototype: testPrototype(0.5, 0.5), evidence: strongEvidence('merge', ['redundant-behavior']),
    }))
    expect(merged.pages[0].memberships).toEqual([{ regionId: 'region-merged', weight: 1 }])
    expect(merged.regions.filter((region) => ['region-a', 'region-b'].includes(region.id)).every(
      (region) => region.lifecycle === 'retired',
    )).toBe(true)
    const retired = applyRegionLifecycleProposal(merged, proposeRegionRetire(merged, {
      regionId: 'region-merged', evidence: strongEvidence('retire', ['low-unique-utility']),
    }))
    expect(retired.pages[0]).toMatchObject({ memberships: [], freeMass: 1 })
    expect(retired.pages[1]).toMatchObject({ memberships: [], freeMass: 1 })
    expect(retired.regions.find((region) => region.id === 'region-merged')?.lifecycle).toBe('retired')
    expectConserved(retired)
  })

  it('kapasite aşımını ve stale proposal uygulamasını kapalı biçimde reddeder', () => {
    const base = testAtlas()
    expect(() => proposeRegionBirth(base, {
      regionId: 'too-large', allocations: [{ pageIndex: 0, weight: 1.1 }],
      prototype: testPrototype(), evidence: strongEvidence(),
    })).toThrow(/ağırlığı/)
    const proposal = proposeRegionBirth(base, {
      regionId: 'region-a', allocations: [{ pageIndex: 0, weight: 1 }],
      prototype: testPrototype(), evidence: strongEvidence('a'),
    })
    const changed = applyRegionLifecycleProposal(base, proposeRegionBirth(base, {
      regionId: 'region-b', allocations: [{ pageIndex: 1, weight: 1 }],
      prototype: testPrototype(), evidence: strongEvidence('b'),
    }))
    expect(() => applyRegionLifecycleProposal(changed, proposal)).toThrow(/stale/)
  })
})
