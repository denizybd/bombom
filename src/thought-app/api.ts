import {
  mergeEpisodeTicks,
  normalizeEpisode,
  normalizeEpisodeList,
  normalizeGraph,
  normalizeReplay,
  normalizeSnapshots,
  normalizeStatus,
  normalizeTickList,
} from './model'
import type {
  ReplayVerification,
  StartThoughtEpisodeRequest,
  ThoughtEpisode,
  ThoughtEpisodeSummary,
  ThoughtFeedbackResult,
  ThoughtGraph,
  ThoughtGraphLevel,
  ThoughtServiceStatus,
  ThoughtSnapshotOption,
  ThoughtTick,
} from './types'

export const DEFAULT_THOUGHT_SERVICE_URL = 'http://127.0.0.1:8787'

export class ThoughtApiError extends Error {
  readonly status: number
  readonly code?: string
  readonly details?: unknown

  constructor(message: string, status = 0, code?: string, details?: unknown) {
    super(message)
    this.name = 'ThoughtApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

function normalizedBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, '')
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

async function responseError(response: Response): Promise<ThoughtApiError> {
  let body: Record<string, unknown> = {}
  try {
    body = asRecord(await response.json())
  } catch {
    // A local service can intentionally answer with an empty error body.
  }
  const nested = asRecord(body.error)
  const message = typeof body.message === 'string'
    ? body.message
    : typeof body.error === 'string'
      ? body.error
      : typeof nested.message === 'string'
        ? nested.message
        : response.status === 404
          ? 'Düşünce akışı servisi bu sürümde henüz kullanılamıyor.'
          : `Düşünce servisi isteği reddetti (${response.status}).`
  return new ThoughtApiError(
    message,
    response.status,
    typeof body.code === 'string' ? body.code : typeof nested.code === 'string' ? nested.code : undefined,
    body.details ?? nested.details,
  )
}

function normalizeUnknownError(error: unknown): ThoughtApiError {
  if (error instanceof ThoughtApiError) return error
  if (isAbortError(error)) return new ThoughtApiError('İstek iptal edildi.', 0, 'ABORTED')
  if (error instanceof Error) {
    return new ThoughtApiError(
      'Yerel düşünce servisine ulaşılamadı. Eğitim servisinin 127.0.0.1:8787 adresinde açık olduğunu kontrol edin.',
      0,
      'CONNECTION_FAILED',
      error.message,
    )
  }
  return new ThoughtApiError('Beklenmeyen bir bağlantı sorunu oluştu.', 0, 'UNKNOWN')
}

export class ThoughtApi {
  readonly baseUrl: string

  constructor(baseUrl = DEFAULT_THOUGHT_SERVICE_URL) {
    this.baseUrl = normalizedBaseUrl(baseUrl)
  }

  private async request(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<unknown> {
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal,
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
          ...init.headers,
        },
      })
      if (!response.ok) throw await responseError(response)
      if (response.status === 204) return undefined
      return await response.json()
    } catch (error) {
      throw normalizeUnknownError(error)
    }
  }

  async getStatus(signal?: AbortSignal): Promise<ThoughtServiceStatus> {
    return normalizeStatus(await this.request('/api/thought/status', {}, signal))
  }

  async listEpisodes(signal?: AbortSignal): Promise<ThoughtEpisodeSummary[]> {
    return normalizeEpisodeList(await this.request('/api/thought/episodes?offset=0&limit=50', {}, signal))
  }

  async getEpisode(episodeId: string, signal?: AbortSignal): Promise<ThoughtEpisode> {
    return normalizeEpisode(await this.request(`/api/thought/episodes/${encodeURIComponent(episodeId)}`, {}, signal))
  }

  private async getTickPage(episodeId: string, after: number, limit: number, signal?: AbortSignal) {
    const query = new URLSearchParams({ after: String(after), limit: String(Math.max(1, Math.min(128, limit))), candidateLimit: '16' })
    const response = await this.request(`/api/thought/episodes/${encodeURIComponent(episodeId)}/ticks?${query}`, {}, signal)
    const record = asRecord(response)
    const pagination = asRecord(record.pagination)
    const rawNextAfter = pagination.nextAfter
    return {
      ticks: normalizeTickList(response),
      hasMore: pagination.hasMore === true,
      nextAfter: typeof rawNextAfter === 'number' && Number.isSafeInteger(rawNextAfter) ? rawNextAfter : null,
    }
  }

  async getTicks(episodeId: string, after = -1, limit = 128, signal?: AbortSignal) {
    return (await this.getTickPage(episodeId, after, limit, signal)).ticks
  }

  async getEpisodeWithTicks(episodeId: string, signal?: AbortSignal): Promise<ThoughtEpisode> {
    const episode = await this.getEpisode(episodeId, signal)
    try {
      const ticks: ThoughtTick[] = []
      let after = -1
      for (let pageIndex = 0; pageIndex < 8 && ticks.length < episode.maxTicks; pageIndex += 1) {
        const page = await this.getTickPage(episodeId, after, Math.min(128, episode.maxTicks - ticks.length), signal)
        ticks.push(...page.ticks)
        if (!page.hasMore || page.nextAfter === null || page.nextAfter <= after) break
        after = page.nextAfter
      }
      return mergeEpisodeTicks(episode, ticks)
    } catch (error) {
      if (error instanceof ThoughtApiError && (error.status === 404 || error.status === 405)) return episode
      throw error
    }
  }

  async listSnapshots(signal?: AbortSignal): Promise<ThoughtSnapshotOption[]> {
    return normalizeSnapshots(await this.request('/api/snapshots', {}, signal))
  }

  async getGraph(options: { level: ThoughtGraphLevel; episodeId?: string; limit?: number }, signal?: AbortSignal): Promise<ThoughtGraph> {
    const query = new URLSearchParams({
      level: options.level,
      limit: String(options.limit ?? 60),
    })
    if (options.episodeId) query.set('episodeId', options.episodeId)
    return normalizeGraph(await this.request(`/api/thought/graph?${query}`, {}, signal), options.level)
  }

  async startEpisode(payload: StartThoughtEpisodeRequest, signal?: AbortSignal): Promise<ThoughtEpisode> {
    const response = await this.request('/api/thought/episodes', {
      method: 'POST',
      body: JSON.stringify(payload),
    }, signal)
    return normalizeEpisode(response)
  }

  async pauseEpisode(episodeId: string, signal?: AbortSignal): Promise<ThoughtEpisode> {
    return normalizeEpisode(await this.request(`/api/thought/episodes/${encodeURIComponent(episodeId)}/pause`, {
      method: 'POST', body: '{}',
    }, signal))
  }

  async resumeEpisode(episodeId: string, signal?: AbortSignal): Promise<ThoughtEpisode> {
    return normalizeEpisode(await this.request(`/api/thought/episodes/${encodeURIComponent(episodeId)}/resume`, {
      method: 'POST', body: '{}',
    }, signal))
  }

  async cancelEpisode(episodeId: string, signal?: AbortSignal): Promise<ThoughtEpisode> {
    return normalizeEpisode(await this.request(`/api/thought/episodes/${encodeURIComponent(episodeId)}/cancel`, {
      method: 'POST', body: '{}',
    }, signal))
  }

  async feedbackEpisode(
    episodeId: string,
    decision: 'approve' | 'reject',
    reason = '',
    signal?: AbortSignal,
  ): Promise<ThoughtFeedbackResult> {
    const response = asRecord(await this.request(`/api/thought/episodes/${encodeURIComponent(episodeId)}/feedback`, {
      method: 'POST',
      body: JSON.stringify({ decision, reason }),
    }, signal))
    const snapshot = asRecord(response.snapshot)
    return {
      episode: normalizeEpisode(response),
      duplicate: response.duplicate === true,
      snapshot: Object.keys(snapshot).length ? {
        id: typeof snapshot.id === 'string' ? snapshot.id : '',
        status: typeof snapshot.status === 'string' ? snapshot.status : undefined,
        parentId: typeof snapshot.parentId === 'string' ? snapshot.parentId : snapshot.parentId === null ? null : undefined,
      } : null,
    }
  }

  async replayEpisode(episodeId: string, signal?: AbortSignal): Promise<ReplayVerification> {
    return normalizeReplay(await this.request(`/api/thought/episodes/${encodeURIComponent(episodeId)}/replay`, {
      method: 'POST', body: '{}',
    }, signal))
  }
}
