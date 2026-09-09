import type {
  ActionResponse,
  DashboardData,
  FamilyInspection,
  ModelSnapshot,
  OllamaModel,
  QuarantineItem,
  QueueItem,
  ReviewRequest,
  RollbackRequest,
  SourceTextRequest,
  TrainingConfig,
  TrainingEvent,
  TrainingMetrics,
  TrainingSource,
  TrainingStatus,
  StartRunRequest,
  TeacherGenerateRequest,
  TeacherGenerateResponse,
  TeacherFeedCampaign,
  TeacherFeedCampaignRequest,
} from './types'

export const DEFAULT_TRAINING_SERVICE_URL = 'http://127.0.0.1:8787'

export class TrainingApiError extends Error {
  readonly status: number
  readonly code?: string
  readonly details?: unknown

  constructor(message: string, status = 0, code?: string, details?: unknown) {
    super(message)
    this.name = 'TrainingApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

type EventHandlers = {
  onEvent: (event: TrainingEvent) => void
  onConnectionChange?: (connected: boolean) => void
  onError?: (error: TrainingApiError) => void
}

function normalizedBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, '')
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function errorFromUnknown(error: unknown): TrainingApiError {
  if (error instanceof TrainingApiError) return error
  if (isAbortError(error)) return new TrainingApiError('İstek iptal edildi.', 0, 'ABORTED')
  if (error instanceof Error) {
    return new TrainingApiError(
      'Yerel eğitim servisine bağlanılamadı. Servisin 127.0.0.1:8787 adresinde çalıştığını kontrol edin.',
      0,
      'CONNECTION_FAILED',
      error.message,
    )
  }
  return new TrainingApiError('Beklenmeyen bir bağlantı hatası oluştu.', 0, 'UNKNOWN')
}

async function parseError(response: Response): Promise<TrainingApiError> {
  type ErrorBody = {
    message?: string
    error?: string | { message?: string; code?: string; details?: unknown }
    code?: string
    details?: unknown
  }
  let body: ErrorBody = {}
  try {
    const parsed: unknown = await response.json()
    if (parsed && typeof parsed === 'object') body = parsed as ErrorBody
  } catch {
    // Some local-server failures intentionally have no JSON body.
  }
  const nested = body.error && typeof body.error === 'object' ? body.error : undefined
  return new TrainingApiError(
    body.message ?? (typeof body.error === 'string' ? body.error : nested?.message) ?? `Eğitim servisi isteği reddetti (${response.status}).`,
    response.status,
    body.code ?? nested?.code,
    body.details ?? nested?.details,
  )
}

export class TrainingApi {
  readonly baseUrl: string

  constructor(baseUrl = DEFAULT_TRAINING_SERVICE_URL) {
    this.baseUrl = normalizedBaseUrl(baseUrl)
  }

  private async request<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal,
        headers: {
          Accept: 'application/json',
          ...(!(init.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
          ...init.headers,
        },
      })
      if (!response.ok) throw await parseError(response)
      if (response.status === 204) return undefined as T
      return await response.json() as T
    } catch (error) {
      throw errorFromUnknown(error)
    }
  }

  getStatus(signal?: AbortSignal): Promise<TrainingStatus> {
    return this.request('/api/status', {}, signal)
  }

  getModels(signal?: AbortSignal): Promise<OllamaModel[]> {
    return this.request('/api/ollama/models', {}, signal)
  }

  getSources(signal?: AbortSignal): Promise<TrainingSource[]> {
    return this.request('/api/sources', {}, signal)
  }

  getQuarantine(signal?: AbortSignal): Promise<QuarantineItem[]> {
    return this.request('/api/quarantine', {}, signal)
  }

  getQueue(signal?: AbortSignal): Promise<QueueItem[]> {
    return this.request('/api/queue', {}, signal)
  }

  getMetrics(signal?: AbortSignal): Promise<TrainingMetrics | null> {
    return this.request('/api/metrics', {}, signal)
  }

  getSnapshots(signal?: AbortSignal): Promise<ModelSnapshot[]> {
    return this.request('/api/snapshots', {}, signal)
  }

  getFamilyInspection(
    options: { snapshotId?: string; offset?: number; limit?: number; memberLimit?: number; transitionLimit?: number } = {},
    signal?: AbortSignal,
  ): Promise<FamilyInspection> {
    const query = new URLSearchParams()
    if (options.snapshotId) query.set('snapshotId', options.snapshotId)
    if (options.offset !== undefined) query.set('offset', String(options.offset))
    if (options.limit !== undefined) query.set('limit', String(options.limit))
    if (options.memberLimit !== undefined) query.set('memberLimit', String(options.memberLimit))
    if (options.transitionLimit !== undefined) query.set('transitionLimit', String(options.transitionLimit))
    const suffix = query.size ? `?${query.toString()}` : ''
    return this.request(`/api/families/inspection${suffix}`, {}, signal)
  }

  async getDashboard(signal?: AbortSignal): Promise<DashboardData> {
    // Service health and Ollama health are different facts. The status endpoint is
    // authoritative; if Ollama is down, keep the control plane usable and present an
    // empty model list instead of incorrectly labelling the whole service offline.
    const status = await this.getStatus(signal)
    const [models, sources, quarantine, queue, metrics, snapshots] = await Promise.all([
      status.ollama.models !== undefined
        ? Promise.resolve(status.ollama.models)
        : this.getModels(signal).catch((error: unknown) => {
          if (!status.ollama.connected) return []
          throw error
        }),
      this.getSources(signal),
      this.getQuarantine(signal),
      this.getQueue(signal),
      this.getMetrics(signal),
      this.getSnapshots(signal),
    ])
    return { status, models, sources, quarantine, queue, metrics, snapshots }
  }

  startRun(payload: StartRunRequest, signal?: AbortSignal): Promise<ActionResponse> {
    return this.request('/api/run/start', { method: 'POST', body: JSON.stringify(payload) }, signal)
  }

  pauseRun(signal?: AbortSignal): Promise<ActionResponse> {
    return this.request('/api/run/pause', { method: 'POST', body: '{}' }, signal)
  }

  resumeRun(signal?: AbortSignal): Promise<ActionResponse> {
    return this.request('/api/run/resume', { method: 'POST', body: '{}' }, signal)
  }

  cancelRun(signal?: AbortSignal): Promise<ActionResponse> {
    return this.request('/api/run/cancel', { method: 'POST', body: '{}' }, signal)
  }

  updateConfig(config: TrainingConfig, signal?: AbortSignal): Promise<TrainingStatus> {
    return this.request('/api/config', { method: 'PATCH', body: JSON.stringify(config) }, signal)
  }

  addTextSource(payload: SourceTextRequest, signal?: AbortSignal): Promise<TrainingSource> {
    return this.request('/api/sources/text', { method: 'POST', body: JSON.stringify(payload) }, signal)
  }

  generateTeacherData(payload: TeacherGenerateRequest, signal?: AbortSignal): Promise<TeacherGenerateResponse> {
    return this.request('/api/teacher/generate', { method: 'POST', body: JSON.stringify(payload) }, signal)
  }

  getTeacherFeedCampaigns(signal?: AbortSignal): Promise<TeacherFeedCampaign[]> {
    return this.request('/api/teacher/feed/campaigns', {}, signal)
  }

  createTeacherFeedCampaign(
    payload: TeacherFeedCampaignRequest,
    signal?: AbortSignal,
  ): Promise<TeacherFeedCampaign> {
    return this.request('/api/teacher/feed/campaigns', {
      method: 'POST',
      body: JSON.stringify(payload),
    }, signal)
  }

  pauseTeacherFeedCampaign(id: string, signal?: AbortSignal): Promise<TeacherFeedCampaign> {
    return this.controlTeacherFeedCampaign(id, 'pause', signal)
  }

  resumeTeacherFeedCampaign(id: string, signal?: AbortSignal): Promise<TeacherFeedCampaign> {
    return this.controlTeacherFeedCampaign(id, 'resume', signal)
  }

  cancelTeacherFeedCampaign(id: string, signal?: AbortSignal): Promise<TeacherFeedCampaign> {
    return this.controlTeacherFeedCampaign(id, 'cancel', signal)
  }

  private controlTeacherFeedCampaign(
    id: string,
    action: 'pause' | 'resume' | 'cancel',
    signal?: AbortSignal,
  ): Promise<TeacherFeedCampaign> {
    return this.request(`/api/teacher/feed/campaigns/${encodeURIComponent(id)}/${action}`, {
      method: 'POST',
      body: '{}',
    }, signal)
  }

  addFileSources(files: File[], provenance: string, signal?: AbortSignal): Promise<TrainingSource[]> {
    const body = new FormData()
    for (const file of files) body.append('files', file, file.name)
    body.append('provenance', provenance)
    return this.request('/api/sources/files', { method: 'POST', body }, signal)
  }

  setSourceSelected(sourceId: string, selected: boolean, signal?: AbortSignal): Promise<TrainingSource> {
    return this.request(`/api/sources/${encodeURIComponent(sourceId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ selected }),
    }, signal)
  }

  reviewQuarantine(
    itemId: string,
    decision: 'approve' | 'reject',
    payload: ReviewRequest = {},
    signal?: AbortSignal,
  ): Promise<QuarantineItem> {
    return this.request(`/api/quarantine/${encodeURIComponent(itemId)}/${decision}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }, signal)
  }

  rollbackSnapshot(snapshotId: string, payload: RollbackRequest, signal?: AbortSignal): Promise<ActionResponse> {
    return this.request(`/api/snapshots/${encodeURIComponent(snapshotId)}/rollback`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }, signal)
  }

  async activateSnapshot(snapshotId: string, signal?: AbortSignal): Promise<{ message?: string }> {
    await this.request(`/api/snapshots/${encodeURIComponent(snapshotId)}/activate`, {
      method: 'POST',
      body: '{}',
    }, signal)
    return { message: 'Doğrulanmış aday etkin model olarak kabul edildi.' }
  }

  subscribeToEvents(handlers: EventHandlers, signal?: AbortSignal): () => void {
    let eventSource: EventSource | null = null
    let retryTimer: number | null = null
    let stopped = false
    let retry = 0

    const connect = () => {
      if (stopped || signal?.aborted) return
      eventSource = new EventSource(`${this.baseUrl}/api/events`)
      eventSource.onopen = () => {
        retry = 0
        handlers.onConnectionChange?.(true)
      }
      eventSource.onmessage = (message) => {
        try {
          handlers.onEvent(JSON.parse(message.data) as TrainingEvent)
        } catch {
          handlers.onError?.(new TrainingApiError('Canlı olay kaydı okunamadı.', 0, 'INVALID_EVENT'))
        }
      }
      eventSource.onerror = () => {
        handlers.onConnectionChange?.(false)
        eventSource?.close()
        eventSource = null
        if (stopped || signal?.aborted) return
        const delay = Math.min(30_000, 1_000 * (2 ** retry))
        retry += 1
        handlers.onError?.(new TrainingApiError('Canlı olay bağlantısı kesildi; yeniden bağlanılıyor.', 0, 'EVENT_STREAM_DISCONNECTED'))
        retryTimer = window.setTimeout(connect, delay)
      }
    }

    const stop = () => {
      stopped = true
      if (retryTimer !== null) window.clearTimeout(retryTimer)
      eventSource?.close()
      handlers.onConnectionChange?.(false)
    }

    signal?.addEventListener('abort', stop, { once: true })
    connect()
    return stop
  }
}
