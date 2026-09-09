import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  activateReleasedRegionSnapshot,
  configureSemanticRegionProbes,
  enableAdaptiveRegionProposals,
  enableAdaptiveRegionShadow,
  enableReleasedAdaptiveRegions,
  loadRegionDashboard,
  loadRegionReleaseJob,
  queueRegionReleaseEvaluation,
} from './api'

afterEach(() => vi.unstubAllGlobals())

function response(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

describe('loadRegionDashboard', () => {
  it('does not request a head payload when status reports no attached atlas', async () => {
    const fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/regions/status')) return response({ effectiveMode: 'off', head: null })
      if (url.includes('/api/regions/artifacts')) return response({ items: [] })
      if (url.includes('/api/regions/probes')) return response({ available: false, summary: { items: [] } })
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetch)

    const dashboard = await loadRegionDashboard('http://example.test')
    expect(dashboard.atlas).toBeNull()
    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it('loads the exact current atlas only after status identifies a head', async () => {
    const fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/regions/status')) return response({
        available: true,
        head: { artifactId: 'head-1', fingerprint: 'atlas-1', representationSpaceId: 'RS-A' },
      })
      if (url.endsWith('/api/regions/atlas')) return response({
        available: true,
        authority: 'active',
        head: { artifactId: 'head-1', fingerprint: 'atlas-1', representationSpaceId: 'RS-A' },
        atlas: {
          format: 'plastic-region-atlas-head', version: 1,
          representationSpaceId: 'RS-A', fingerprint: 'atlas-1', regions: [], pages: [],
        },
      })
      if (url.includes('/api/regions/artifacts')) return response({ items: [] })
      if (url.includes('/api/regions/probes')) return response({
        available: true,
        atlasFingerprint: 'atlas-1',
        interpretationOnly: true,
        trainingWritesAllowed: false,
        routingMutationAllowed: false,
        releaseAuthority: false,
        summary: { items: [{
          hypothesis: { hypothesisId: 'h-1', namespace: 'event-role.actor', label: 'yapan=Ada' },
          status: 'provisional', atlasFingerprint: 'atlas-1', observationCount: 24,
          independentSourceCount: 8, target: { featureId: 'latent-1', responseKind: 'region-gate', footprint: [{ pageIndex: 2, weight: 1 }] },
          evidence: null,
        }] },
      })
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetch)

    const dashboard = await loadRegionDashboard('http://example.test')
    expect(dashboard.atlas?.fingerprint).toBe('atlas-1')
    expect(dashboard.probes.items[0]?.hypothesis.label).toBe('yapan=Ada')
    expect(dashboard.probes.routingMutationAllowed).toBe(false)
    expect(fetch).toHaveBeenCalledTimes(4)
  })

  it('fails closed instead of merging status and an atlas from different snapshots', async () => {
    const fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/regions/status')) return response({
        head: { artifactId: 'head-old', fingerprint: 'atlas-old', representationSpaceId: 'RS-A' },
      })
      if (url.includes('/api/regions/artifacts')) return response({ items: [] })
      if (url.endsWith('/api/regions/atlas')) return response({
        head: { artifactId: 'head-new', fingerprint: 'atlas-new', representationSpaceId: 'RS-A' },
        atlas: { format: 'plastic-region-atlas-head', version: 1, representationSpaceId: 'RS-A', fingerprint: 'atlas-torn', regions: [], pages: [] },
      })
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetch)

    await expect(loadRegionDashboard('http://example.test')).rejects.toMatchObject({
      code: 'REGION_DASHBOARD_SNAPSHOT_CHANGED', status: 409,
    })
    expect(fetch.mock.calls.some(([url]) => String(url).includes('/api/regions/probes'))).toBe(false)
  })

  it('enables the separate adaptive lineage first in behavior-free shadow mode', async () => {
    const fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.method).toBe('PATCH')
      expect(JSON.parse(String(init?.body))).toEqual({
        plasticRegions: {
          enabled: true,
          representationProfile: 'adaptive-eighth-id-v1',
          mode: 'shadow',
        },
      })
      return response({
        config: { plasticRegions: {} },
        enabled: true,
        configuredMode: 'shadow',
        effectiveMode: 'shadow',
        representationProfile: 'adaptive-eighth-id-v1',
      })
    })
    vi.stubGlobal('fetch', fetch)

    await expect(enableAdaptiveRegionShadow('http://example.test')).resolves.toBeUndefined()
  })

  it('enables proposal production without exposing an active-mode mutation', async () => {
    const fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(JSON.parse(String(init?.body))).toEqual({
        plasticRegions: {
          enabled: true,
          representationProfile: 'adaptive-eighth-id-v1',
          mode: 'proposal',
        },
      })
      return response({ configuredMode: 'proposal', representationProfile: 'adaptive-eighth-id-v1' })
    })
    vi.stubGlobal('fetch', fetch)

    await expect(enableAdaptiveRegionProposals('http://example.test')).resolves.toBeUndefined()
  })

  it('has a separate explicit request for an already released active snapshot', async () => {
    const fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(JSON.parse(String(init?.body))).toEqual({
        plasticRegions: {
          enabled: true,
          representationProfile: 'adaptive-eighth-id-v1',
          mode: 'active',
        },
      })
      return response({ config: { plasticRegions: { mode: 'active' } } })
    })
    vi.stubGlobal('fetch', fetch)

    await expect(enableReleasedAdaptiveRegions('http://example.test')).resolves.toBeUndefined()
  })

  it('configures semantic probes as a separate read-only opt-in', async () => {
    const fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.method).toBe('PATCH')
      expect(JSON.parse(String(init?.body))).toEqual({ plasticRegions: { semanticProbesEnabled: true } })
      return response({ config: { plasticRegions: { semanticProbesEnabled: true } } })
    })
    vi.stubGlobal('fetch', fetch)

    await expect(configureSemanticRegionProbes(true, 'http://example.test')).resolves.toBeUndefined()
  })

  it('queues server-owned release evaluation and polls its bounded public job state', async () => {
    const fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/api/regions/proposals/')) {
        expect(init?.method).toBe('POST')
        expect(init?.body).toBe('{}')
        return response({ duplicate: false, job: { id: 'release-job-1', state: 'queued', progress: 0 } }, 202)
      }
      if (url.endsWith('/api/jobs/release-job-1')) {
        return response({ ok: true, job: { id: 'release-job-1', state: 'active', progress: 0.4, stage: 'evaluate' } })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetch)

    const queued = await queueRegionReleaseEvaluation('proposal/id', 'http://example.test')
    expect(queued).toMatchObject({ id: 'release-job-1', state: 'waiting', duplicate: false })
    const running = await loadRegionReleaseJob(queued.id, 'http://example.test')
    expect(running).toMatchObject({ id: 'release-job-1', state: 'active', progress: 0.4 })
    expect(fetch).not.toHaveBeenCalledWith(expect.stringContaining('/api/queue'), expect.anything())
  })

  it('projects a rejected release separately from job completion and activates only an explicit released snapshot', async () => {
    const fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/jobs/release-job-2')) return response({ ok: true, job: {
        id: 'release-job-2', state: 'complete', progress: 1,
        releaseResult: {
          released: false, automaticActivation: false, reasonCodes: ['HOLDOUT_GAIN_TOO_LOW'],
          auditArtifactId: 'audit-2', metrics: { conservativeHoldoutGain: 0.01, conservativeShamGain: 0.02 },
        },
      } })
      if (url.endsWith('/api/snapshots/released-snapshot/activate')) {
        expect(init?.method).toBe('POST')
        expect(init?.body).toBe('{}')
        return response({ ok: true })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetch)

    const rejected = await loadRegionReleaseJob('release-job-2', 'http://example.test')
    expect(rejected?.state).toBe('complete')
    expect(rejected?.releaseResult).toMatchObject({
      released: false,
      automaticActivation: false,
      reasonCodes: ['HOLDOUT_GAIN_TOO_LOW'],
      auditArtifactId: 'audit-2',
    })
    await expect(activateReleasedRegionSnapshot('released-snapshot', 'http://example.test')).resolves.toBeUndefined()
  })
})
