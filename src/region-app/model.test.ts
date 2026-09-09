import { describe, expect, it } from 'vitest'
import { mergeRegionDashboard, normalizeRegionAtlas, normalizeRegionAtlasDetail, normalizeRegionStatus, normalizeSemanticProbeInspection } from './model'

describe('plastic region dashboard normalization', () => {
  it('defaults to a safe off state when the service is empty', () => {
    const status = normalizeRegionStatus({})
    expect(status.effectiveMode).toBe('off')
    expect(status.effectiveModeReason).toBe('unknown')
    expect(status.available).toBe(false)
    expect(status.representationProfile).toBe('legacy')
    expect(status.limits.maximumRegions).toBe(56)
    expect(status.release.artifactId).toBeNull()
  })

  it('fails closed on anonymous heads and deduplicates pages, regions, and memberships', () => {
    expect(normalizeRegionAtlas({ atlas: {
      format: 'plastic-region-atlas-head', version: 1, representationSpaceId: 'RS-A', regions: [], pages: [],
    } })).toBeNull()

    const atlas = normalizeRegionAtlas({ atlas: {
      format: 'plastic-region-atlas-head', version: 1,
      representationSpaceId: 'RS-A', fingerprint: 'atlas-dedup', maximumMembershipsPerPage: 3,
      regions: [{ id: 'r1' }, { id: 'r1' }],
      pages: [
        { pageIndex: 0, freeMass: .25, memberships: [{ regionId: 'r1', weight: .75 }, { regionId: 'r1', weight: .2 }] },
        { pageIndex: 0, freeMass: 1, memberships: [] },
      ],
    } })
    expect(atlas?.regions).toHaveLength(1)
    expect(atlas?.pages).toHaveLength(1)
    expect(atlas?.pages[0]?.memberships).toEqual([{ regionId: 'r1', weight: .75 }])
    expect(atlas?.regions[0]?.retiredRevision).toBeNull()
    expect(atlas?.stats).toMatchObject({ regionCount: 1, pageCount: 1, assignedMass: .75, freeMass: 55.25 })
  })

  it('bounds page memberships and sorts the physical page order', () => {
    const atlas = normalizeRegionAtlas({ atlas: {
      format: 'plastic-region-atlas-head', version: 1,
      representationSpaceId: 'RS-A', fingerprint: 'atlas-a', revision: 2,
      pages: [
        { pageIndex: 4, freeMass: -.5, memberships: [{ regionId: 'r1', weight: 4 }] },
        { pageIndex: 1, freeMass: .2, memberships: [{ regionId: 'r2', weight: .8 }] },
        { pageIndex: 90, memberships: [] },
      ],
      regions: [{ id: 'r1', evidence: { cohesion: 2 } }], stats: {},
    } })
    expect(atlas?.pages.map((page) => page.pageIndex)).toEqual([1, 4])
    expect(atlas?.pages[1]?.memberships[0]?.weight).toBe(1)
    expect(atlas?.pages[1]?.freeMass).toBe(0)
    expect(atlas?.regions[0]?.evidence.cohesion).toBe(1)
  })

  it('keeps status, atlas and immutable history separate', () => {
    const result = mergeRegionDashboard(
      {
        enabled: true, configuredMode: 'shadow', effectiveMode: 'off',
        effectiveModeReason: 'verified-region-release-required',
        release: { artifactId: 'release-a', releaseId: 'release-1', fingerprint: 'release-fp', proposalId: 'proposal-a' },
        limits: { releaseMinimumCases: 30, releaseMinimumSourceGroups: 3, releaseMinimumHoldoutGain: .01 },
      },
      { atlas: null },
      { items: [{ artifactId: 'a1', kind: 'plastic-region-shadow-report', payloadBytes: 12 }] },
    )
    expect(result.status.configuredMode).toBe('shadow')
    expect(result.status.effectiveMode).toBe('off')
    expect(result.status.release).toMatchObject({ artifactId: 'release-a', fingerprint: 'release-fp' })
    expect(result.status.limits).toMatchObject({ releaseMinimumCases: 30, releaseMinimumSourceGroups: 3, releaseMinimumHoldoutGain: .01 })
    expect(result.atlas).toBeNull()
    expect(result.artifacts).toHaveLength(1)
  })

  it('normalizes a detached proposal as an explicit inspection-only atlas detail', () => {
    const result = normalizeRegionAtlasDetail({
      artifact: { artifactId: 'proposal-artifact', kind: 'plastic-region-atlas-proposal', payloadBytes: 321 },
      proposal: {
        proposalId: 'proposal-1', action: 'birth', eligible: false,
        rejectionReasons: ['holdout-gain-insufficient'], fingerprint: 'proposal-fp', baseAtlasFingerprint: 'base-fp',
      },
      atlas: {
        format: 'plastic-region-atlas-head', version: 1,
        representationSpaceId: 'RS-A', fingerprint: 'candidate-atlas-fp', revision: 1,
        regions: [{ id: 'provisional-r1', lifecycle: 'provisional' }],
        pages: [{ pageIndex: 0, freeMass: .75, memberships: [{ regionId: 'provisional-r1', weight: .25 }] }],
      },
    })
    expect(result?.artifact?.artifactId).toBe('proposal-artifact')
    expect(result?.proposal).toMatchObject({ proposalId: 'proposal-1', action: 'birth', eligible: false })
    expect(result?.atlas.regions[0]).toMatchObject({ id: 'provisional-r1', lifecycle: 'provisional' })
  })

  it('keeps semantic labels as bounded external hypotheses without raw training rows', () => {
    const result = normalizeSemanticProbeInspection({
      available: true, atlasFingerprint: 'atlas-a', bytesRead: 1200,
      legacySkeletons: { recognizedCount: 2 },
      summary: { omittedCount: 1, items: [{
        hypothesis: { hypothesisId: 'hyp-1', namespace: 'event-role.actor', label: 'yapan=Ada' },
        status: 'calibrated', atlasFingerprint: 'atlas-a', observationCount: 80, independentSourceCount: 12,
        target: { featureId: 'opaque-region-1', responseKind: 'region-gate', footprint: [{ pageIndex: 4, weight: .8 }] },
        evidence: {
          evaluationSplit: 'test', targetBrier: .12, targetEce: .04,
          gainVsBaseline: { point: .2, low: .1, high: .3, confidenceLevel: .95 },
          gainVsSham: { point: .15, low: .08, high: .2, confidenceLevel: .95 },
          reasons: [], rawRows: [{ prompt: 'must not survive' }],
        },
        records: [{ score: 1 }], lineages: ['secret'],
      }] },
    })
    expect(result).toMatchObject({ available: true, interpretationOnly: true, routingMutationAllowed: false, releaseAuthority: false, legacySkeletonCount: 2 })
    expect(result.items[0]).toMatchObject({ status: 'calibrated', target: { featureId: 'opaque-region-1' }, evidence: { evaluationSplit: 'test', targetBrier: .12 } })
    expect(JSON.stringify(result)).not.toContain('must not survive')
    expect(JSON.stringify(result)).not.toContain('secret')
  })
})
