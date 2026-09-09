import { afterEach, describe, expect, it, vi } from 'vitest'
import { ThoughtApi } from './api'
import type { StartThoughtEpisodeRequest } from './types'

describe('ThoughtApi start contract', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('sends canonical expectedOutcome and teacher mode', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({
      ok: true,
      episode: {
        id: 'episode-created',
        status: 'queued',
        input: { preview: 'Çocuk pencereyi kapattı.' },
        base: { snapshotId: 'snapshot-3' },
        crossCheck: { status: 'pending' },
        seed: 99,
        maxTicks: 64,
        tickCount: 0,
      },
    }), { status: 202, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)
    const api = new ThoughtApi('http://127.0.0.1:8787/')
    const request: StartThoughtEpisodeRequest = {
      prompt: 'Çocuk pencereyi kapattı.',
      snapshotId: 'snapshot-3',
      teacherMode: 'ollama-cross-check',
      learningMode: 'proposal',
      seed: 99,
      maxTicks: 64,
      expectedOutcome: {
        format: 'event-graph-v1',
        actor: 'çocuk',
        action: 'kapatmak',
        patient: 'pencere',
        instrument: null,
        operators: [],
        negated: false,
        goals: [{ relation: 'amaç', target: 'odayı kuru tutmak' }],
      },
    }

    const episode = await api.startEpisode(request)
    const [url, init] = fetchMock.mock.calls[0]!
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>

    expect(url).toBe('http://127.0.0.1:8787/api/thought/episodes')
    expect(body.teacherMode).toBe('ollama-cross-check')
    expect(body).not.toHaveProperty('crossCheck')
    expect(body.learningMode).toBe('proposal')
    expect(body.expectedOutcome).toEqual(request.expectedOutcome)
    expect(episode.id).toBe('episode-created')
    expect(episode.prompt).toBe('Çocuk pencereyi kapattı.')
  })

  it('posts approve/reject feedback using the server decision envelope', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({
      ok: true,
      duplicate: false,
      episode: {
        id: 'episode-review',
        status: 'completed',
        input: { preview: 'Bir inceleme bölümü.' },
        seed: 8,
        maxTicks: 12,
        tickCount: 12,
        learningMode: 'proposal',
        commitEligible: true,
        commitStatus: 'candidate',
        artifacts: { candidateSnapshotId: 'candidate-snapshot-9' },
      },
      snapshot: { id: 'candidate-snapshot-9', status: 'candidate', parentId: 'active-snapshot' },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)
    const api = new ThoughtApi()

    const result = await api.feedbackEpisode('episode-review', 'approve', 'İz tutarlı görünüyor.')
    const [url, init] = fetchMock.mock.calls[0]!

    expect(url).toBe('http://127.0.0.1:8787/api/thought/episodes/episode-review/feedback')
    expect(JSON.parse(String(init?.body))).toEqual({ decision: 'approve', reason: 'İz tutarlı görünüyor.' })
    expect(result.episode.commitStatus).toBe('candidate')
    expect(result.episode.candidateSnapshotId).toBe('candidate-snapshot-9')
    expect(result.snapshot?.id).toBe('candidate-snapshot-9')
  })

  it('sends an explicit reject decision without snapshot activation fields', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({
      ok: true,
      duplicate: false,
      episode: {
        id: 'episode-rejected', status: 'completed', input: { preview: 'Reddedilen öneri' },
        seed: 4, maxTicks: 8, tickCount: 8, learningMode: 'proposal', commitEligible: true, commitStatus: 'rejected',
        review: { decision: 'reject', reason: 'Döngüye girdi.', reviewedAt: '2026-08-29T10:00:00.000Z' },
      },
      snapshot: null,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await new ThoughtApi().feedbackEpisode('episode-rejected', 'reject', 'Döngüye girdi.')
    const [, init] = fetchMock.mock.calls[0]!

    expect(JSON.parse(String(init?.body))).toEqual({ decision: 'reject', reason: 'Döngüye girdi.' })
    expect(result.episode.review?.decision).toBe('reject')
    expect(result.episode.commitStatus).toBe('rejected')
    expect(result.snapshot).toBeNull()
  })

  it('reads long traces through the server bounded tick pagination contract', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
      const url = String(input)
      if (!url.includes('/ticks?')) {
        return new Response(JSON.stringify({
          ok: true,
          episode: { id: 'episode-pages', status: 'completed', input: { preview: 'Sayfalı iz' }, seed: 1, maxTicks: 2, tickCount: 2 },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const second = url.includes('after=0')
      return new Response(JSON.stringify({
        ok: true,
        ticks: [{ index: second ? 1 : 0 }],
        pagination: { hasMore: !second, nextAfter: second ? null : 0 },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)

    const episode = await new ThoughtApi().getEpisodeWithTicks('episode-pages')
    const urls = fetchMock.mock.calls.map(([url]) => String(url))

    expect(episode.ticks.map((tick) => tick.index)).toEqual([0, 1])
    expect(urls.some((url) => url.includes('after=-1') && url.includes('candidateLimit=16'))).toBe(true)
    expect(urls.some((url) => url.includes('after=0'))).toBe(true)
  })

  it('requests the selected episode graph instead of silently falling back to the global graph', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL) => new Response(JSON.stringify({
      ok: true,
      graph: { level: 'family', nodes: [], edges: [], totalNodeCount: 0, totalEdgeCount: 0, truncated: false },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await new ThoughtApi().getGraph({ level: 'family', episodeId: 'episode/özel', limit: 24 })

    expect(String(fetchMock.mock.calls[0]?.[0])).toContain('episodeId=episode%2F%C3%B6zel')
  })
})
