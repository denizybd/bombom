import { afterEach, describe, expect, it, vi } from 'vitest'
import { TrainingApi } from './api'
import type { TeacherFeedCampaign, TrainingStatus } from './types'

const offlineStatus: TrainingStatus = {
  serviceVersion: 'test',
  state: 'idle',
  runId: null,
  startedAt: null,
  updatedAt: '2026-08-29T00:00:00.000Z',
  activeStage: null,
  progress: 0,
  queuedItems: 0,
  processedItems: 0,
  acceptedItems: 0,
  rejectedItems: 0,
  activeSnapshotId: null,
  ollama: {
    connected: false,
    endpoint: 'http://127.0.0.1:11434',
    checkedAt: '2026-08-29T00:00:00.000Z',
    models: [],
  },
  config: {
    ollamaBaseUrl: 'http://127.0.0.1:11434',
    model: '',
    automationLevel: 'proposal',
    autoRun: false,
    batchSize: 50,
    budgets: { cpuPercent: 60, ramMb: 4096, maxRunMinutes: 120, diskMb: 4096 },
  },
  stages: [],
  warnings: [],
}

afterEach(() => vi.unstubAllGlobals())

describe('TrainingApi dashboard', () => {
  it('keeps the service dashboard usable when only Ollama is offline', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => {
      const path = new URL(String(input)).pathname
      if (path === '/api/status') return Response.json(offlineStatus)
      if (path === '/api/ollama/models') {
        return Response.json({ message: 'Ollama kapalı', code: 'OLLAMA_UNAVAILABLE' }, { status: 502 })
      }
      if (path === '/api/metrics') return Response.json(null)
      return Response.json([])
    }))

    const dashboard = await new TrainingApi('http://127.0.0.1:8787/').getDashboard()

    expect(dashboard.status.ollama.connected).toBe(false)
    expect(dashboard.models).toEqual([])
    expect(dashboard.sources).toEqual([])
    expect(fetch).toHaveBeenCalledTimes(6)
  })

  it('requests a bounded family inspection page with encoded filters', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Response.json({ available: false })))

    await new TrainingApi('http://127.0.0.1:8787/').getFamilyInspection({
      snapshotId: 'aday / 1',
      offset: 50,
      limit: 25,
      memberLimit: 7,
      transitionLimit: 5,
    })

    expect(fetch).toHaveBeenCalledOnce()
    const requested = new URL(String(vi.mocked(fetch).mock.calls[0][0]))
    expect(requested.pathname).toBe('/api/families/inspection')
    expect(Object.fromEntries(requested.searchParams)).toEqual({
      snapshotId: 'aday / 1',
      offset: '50',
      limit: '25',
      memberLimit: '7',
      transitionLimit: '5',
    })
  })

  it('creates a bounded research-only teacher feed campaign without changing its contract', async () => {
    const campaign: TeacherFeedCampaign = {
      id: 'campaign-1',
      status: 'running',
      model: 'qwen2.5-coder:14b',
      seed: 20260829,
      targetAccepted: 128,
      generationBatchSize: 2,
      trainEveryAccepted: 32,
      curriculum: 'turkish-foundations-v1',
      mode: 'research-only',
      generated: 0,
      accepted: 0,
      duplicates: 0,
      rejectedQuality: 0,
      teacherRequests: 0,
      teacherTokens: 0,
      completedTrainingCycles: 0,
      nextBatchIndex: 0,
      lastSnapshotId: null,
      stopReason: null,
      continuous: null,
      createdAt: '2026-08-29T00:00:00.000Z',
      updatedAt: '2026-08-29T00:00:00.000Z',
    }
    vi.stubGlobal('fetch', vi.fn(async () => Response.json(campaign)))

    const result = await new TrainingApi().createTeacherFeedCampaign({
      model: campaign.model,
      seed: campaign.seed,
      targetAccepted: campaign.targetAccepted,
      generationBatchSize: campaign.generationBatchSize,
      trainEveryAccepted: campaign.trainEveryAccepted,
      curriculum: campaign.curriculum,
      mode: campaign.mode,
    })

    expect(result).toEqual(campaign)
    const [input, init] = vi.mocked(fetch).mock.calls[0]
    expect(new URL(String(input)).pathname).toBe('/api/teacher/feed/campaigns')
    expect(init?.method).toBe('POST')
    expect(JSON.parse(String(init?.body))).toEqual({
      model: 'qwen2.5-coder:14b',
      seed: 20260829,
      targetAccepted: 128,
      generationBatchSize: 2,
      trainEveryAccepted: 32,
      curriculum: 'turkish-foundations-v1',
      mode: 'research-only',
    })
  })

  it('sends the bounded continuous block plan without any activation authority', async () => {
    const response = {
      id: 'continuous-1', status: 'running', model: 'qwen:14b', seed: 7,
      targetAccepted: 4096, generationBatchSize: 2, trainEveryAccepted: 32,
      curriculum: 'turkish-foundations-v1', mode: 'research-only',
      generated: 0, accepted: 0, duplicates: 0, rejectedQuality: 0,
      teacherRequests: 0, teacherTokens: 0, completedTrainingCycles: 0,
      nextBatchIndex: 0, lastSnapshotId: null, stopReason: null,
      continuous: {
        blockAccepted: 128, maximumBlocks: 32, completedBlocks: 0, nextBlockAt: 128,
        minimumDiskHeadroomMb: 16, maximumSyntheticSources: 100_000,
        bootstrapSourceCount: 68, bootstrapSourceFingerprint: 'bootstrap-source-fingerprint',
        stopGate: { required: true, passed: false, status: 'waiting', snapshotId: null, benchmarkFingerprint: 'f', failedChecks: [] },
      },
      createdAt: '2026-08-29T00:00:00.000Z', updatedAt: '2026-08-29T00:00:00.000Z',
    } satisfies TeacherFeedCampaign
    vi.stubGlobal('fetch', vi.fn(async () => Response.json(response)))
    const plan = {
      model: 'qwen:14b', seed: 7, targetAccepted: 128, generationBatchSize: 2, trainEveryAccepted: 32,
      curriculum: 'turkish-foundations-v1' as const, mode: 'research-only' as const,
      continuousPlan: { blockAccepted: 128, maxBlocks: 32, minimumDiskHeadroomMb: 16, maximumSyntheticSources: 100_000 },
    }
    await new TrainingApi().createTeacherFeedCampaign(plan)
    const [, init] = vi.mocked(fetch).mock.calls[0]
    expect(JSON.parse(String(init?.body))).toEqual(plan)
    expect(JSON.parse(String(init?.body))).not.toHaveProperty('automationLevel', 'controlled')
  })

  it('lists and controls teacher feed campaigns through encoded campaign ids', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => {
      const method = String(vi.mocked(fetch).mock.calls.at(-1)?.[1]?.method ?? 'GET')
      return Response.json(method === 'GET' ? [] : { id: 'campaign / 1', status: 'paused' })
    }))
    const api = new TrainingApi()

    await api.getTeacherFeedCampaigns()
    await api.pauseTeacherFeedCampaign('campaign / 1')
    await api.resumeTeacherFeedCampaign('campaign / 1')
    await api.cancelTeacherFeedCampaign('campaign / 1')

    const paths = vi.mocked(fetch).mock.calls.map(([input]) => new URL(String(input)).pathname)
    expect(paths).toEqual([
      '/api/teacher/feed/campaigns',
      '/api/teacher/feed/campaigns/campaign%20%2F%201/pause',
      '/api/teacher/feed/campaigns/campaign%20%2F%201/resume',
      '/api/teacher/feed/campaigns/campaign%20%2F%201/cancel',
    ])
    expect(vi.mocked(fetch).mock.calls.slice(1).every(([, init]) => init?.method === 'POST' && init.body === '{}')).toBe(true)
  })
})
