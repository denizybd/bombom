import { describe, expect, it } from 'vitest'
import {
  createInitialPlasticRegionAtlas,
  createPlasticRegionAtlas,
  plasticRegionCapacity,
  totalPlasticRegionCapacity,
  validateAdaptiveRegionAtlas,
} from './atlas.ts'
import type { PlasticPageAssignmentV1, PlasticRegionV1 } from './types.ts'

function region(id: string): PlasticRegionV1 {
  return {
    id,
    lifecycle: 'active',
    origin: 'discovery',
    bornRevision: 0,
    retiredRevision: null,
    parentRegionIds: [],
    prototype: { dimension: 2, values: id === 'region-a' ? [1, 0] : [0, 1] },
    evidence: {
      observationCount: 3,
      sourceGroupIds: ['s1', 's2', 's3'],
      confidenceMass: 1.5,
      cohesion: 0.9,
      holdoutUtility: 0.1,
      shamUtility: 0.01,
    },
  }
}

function assignedPages(reverse = false): PlasticPageAssignmentV1[] {
  const pages: PlasticPageAssignmentV1[] = Array.from({ length: 56 }, (_, pageIndex) => ({
    pageIndex,
    memberships: [],
    freeMass: 1,
  }))
  pages[0] = {
    pageIndex: 0,
    memberships: reverse
      ? [{ regionId: 'region-b', weight: 0.25 }, { regionId: 'region-a', weight: 0.5 }]
      : [{ regionId: 'region-a', weight: 0.5 }, { regionId: 'region-b', weight: 0.25 }],
    freeMass: 0.25,
  }
  pages[1] = {
    pageIndex: 1,
    memberships: reverse
      ? [{ regionId: 'region-b', weight: 0.25 }, { regionId: 'region-a', weight: 0.5 }]
      : [{ regionId: 'region-a', weight: 0.5 }, { regionId: 'region-b', weight: 0.25 }],
    freeMass: 0.25,
  }
  pages[2] = {
    pageIndex: 2,
    memberships: [{ regionId: 'region-b', weight: 0.5 }],
    freeMass: 0.5,
  }
  return reverse ? pages.reverse() : pages
}

describe('plastik bölge atlası', () => {
  it('başlangıçta 56 sayfanın tamamını serbest ve atlası çalışma anında değişmez kurar', () => {
    const atlas = createInitialPlasticRegionAtlas({
      representationSpaceId: 'adaptive-eighth-id-v1',
      behaviorDimension: 8,
    })
    expect(atlas.format).toBe('plastic-region-atlas-head')
    expect(atlas.regions).toEqual([])
    expect(atlas.pages).toHaveLength(56)
    expect(atlas.pages.every((page) => page.freeMass === 1 && page.memberships.length === 0)).toBe(true)
    expect(totalPlasticRegionCapacity(atlas)).toEqual({ allocated: 0, free: 56, total: 56 })
    expect(Object.isFrozen(atlas)).toBe(true)
    expect(Object.isFrozen(atlas.pages)).toBe(true)
    expect(Object.isFrozen(atlas.pages[0])).toBe(true)
    expect(() => validateAdaptiveRegionAtlas(atlas)).not.toThrow()
  })

  it('girdi sırası değişse de aynı kanonik fingerprinti üretir', () => {
    const left = createPlasticRegionAtlas({
      representationSpaceId: 'space',
      behaviorDimension: 2,
      regions: [region('region-b'), region('region-a')],
      pages: assignedPages(true),
    })
    const right = createPlasticRegionAtlas({
      representationSpaceId: 'space',
      behaviorDimension: 2,
      regions: [region('region-a'), region('region-b')],
      pages: assignedPages(false),
    })
    expect(left).toEqual(right)
    expect(left.fingerprint).toBe(right.fingerprint)
    expect(plasticRegionCapacity(left, 'region-a')).toBeCloseTo(1, 7)
  })

  it('sayfa kütlesi, aidiyet sınırı, emekli referansı ve fingerprint ihlalini reddeder', () => {
    const basePages = assignedPages()
    const wrongMass = basePages.map((page) => page.pageIndex === 0 ? { ...page, freeMass: 0.4 } : page)
    expect(() => createPlasticRegionAtlas({
      representationSpaceId: 'space', behaviorDimension: 2,
      regions: [region('region-a'), region('region-b')], pages: wrongMass,
    })).toThrow(/kütlesi 1/)
    expect(() => createPlasticRegionAtlas({
      representationSpaceId: 'space', behaviorDimension: 2, maximumMembershipsPerPage: 1,
      regions: [region('region-a'), region('region-b')], pages: basePages,
    })).toThrow(/aidiyet sınırını/)

    const atlas = createPlasticRegionAtlas({
      representationSpaceId: 'space', behaviorDimension: 2,
      regions: [region('region-a'), region('region-b')], pages: basePages,
    })
    expect(() => validateAdaptiveRegionAtlas({ ...atlas, fingerprint: 'tampered' })).toThrow(/özeti/)
    expect(() => validateAdaptiveRegionAtlas({
      ...atlas,
      regions: [...atlas.regions].reverse(),
      pages: [...atlas.pages].reverse(),
    })).toThrow(/kanonik/)
  })

  it('provisional fidan küçük kalabilirken active bölge için en az bir sayfa-eşdeğeri ister', () => {
    const pages: PlasticPageAssignmentV1[] = Array.from({ length: 56 }, (_, pageIndex) => ({
      pageIndex,
      memberships: pageIndex === 0 ? [{ regionId: 'region-a', weight: 0.75 }] : [],
      freeMass: pageIndex === 0 ? 0.25 : 1,
    }))
    expect(() => createPlasticRegionAtlas({
      representationSpaceId: 'space', behaviorDimension: 2,
      regions: [region('region-a')], pages,
    })).toThrow(/en az bir tam mikro-sayfa eşdeğeri/)

    const provisional = createPlasticRegionAtlas({
      representationSpaceId: 'space', behaviorDimension: 2,
      regions: [{ ...region('region-a'), lifecycle: 'provisional' }], pages,
    })
    expect(plasticRegionCapacity(provisional, 'region-a')).toBe(0.75)
  })
})
