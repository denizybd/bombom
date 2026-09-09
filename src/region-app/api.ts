import { mergeRegionDashboard, normalizeRegionAtlasDetail, normalizeRegionStatus } from './model'
import type {
  PlasticRegionAtlasDetail,
  PlasticRegionDashboard,
  PlasticRegionReleaseJobState,
  PlasticRegionReleaseJobView,
} from './types'

export const DEFAULT_REGION_SERVICE_URL = 'http://127.0.0.1:8787'

export class RegionApiError extends Error {
  constructor(message: string, readonly status = 0, readonly code?: string) {
    super(message)
    this.name = 'RegionApiError'
  }
}

async function request(baseUrl: string, path: string, signal?: AbortSignal): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(`${baseUrl.replace(/\/$/u, '')}${path}`, {
      headers: { Accept: 'application/json' }, signal,
    })
  } catch {
    throw new RegionApiError('Yerel eğitim servisine ulaşılamadı. Çalışma alanını başlatıp yeniden deneyin.', 0, 'CONNECTION_FAILED')
  }
  if (!response.ok) {
    let body: Record<string, unknown> = {}
    try { body = await response.json() as Record<string, unknown> } catch { /* empty response */ }
    throw new RegionApiError(
      typeof body.message === 'string' ? body.message : `Bölge servisi isteği reddetti (${response.status}).`,
      response.status,
      typeof body.code === 'string' ? body.code : undefined,
    )
  }
  return response.json()
}

async function writeRequest(
  baseUrl: string,
  path: string,
  method: 'PATCH' | 'POST',
  body: unknown,
  signal?: AbortSignal,
): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(`${baseUrl.replace(/\/$/u, '')}${path}`, {
      method,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  } catch {
    throw new RegionApiError('Yerel eğitim servisine ulaşılamadı. Çalışma alanını başlatıp yeniden deneyin.', 0, 'CONNECTION_FAILED')
  }
  if (!response.ok) {
    let payload: Record<string, unknown> = {}
    try { payload = await response.json() as Record<string, unknown> } catch { /* empty response */ }
    throw new RegionApiError(
      typeof payload.message === 'string' ? payload.message : `Bölge servisi isteği reddetti (${response.status}).`,
      response.status,
      typeof payload.code === 'string' ? payload.code : undefined,
    )
  }
  return response.json()
}

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function releaseJobState(value: unknown): PlasticRegionReleaseJobState {
  if (value === 'queued' || value === 'paused' || value === 'waiting') return 'waiting'
  if (value === 'running' || value === 'pause_requested' || value === 'cancel_requested' || value === 'active') return 'active'
  if (value === 'succeeded' || value === 'complete') return 'complete'
  if (value === 'failed') return 'failed'
  return 'cancelled'
}

function normalizeReleaseJob(value: unknown, duplicate = false): PlasticRegionReleaseJobView | null {
  const source = record(value)
  const id = typeof source.id === 'string' ? source.id : ''
  if (!id) return null
  const progress = typeof source.progress === 'number' && Number.isFinite(source.progress)
    ? Math.max(0, Math.min(1, source.progress))
    : 0
  const releaseResult = record(source.releaseResult)
  const metrics = record(releaseResult.metrics)
  const nullableFinite = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : null
  return {
    id,
    state: releaseJobState(source.state),
    progress,
    stage: typeof source.stage === 'string' ? source.stage : null,
    message: typeof source.message === 'string'
      ? source.message
      : typeof record(source.error).message === 'string'
        ? String(record(source.error).message)
        : null,
    duplicate,
    releaseResult: typeof releaseResult.released === 'boolean' ? {
      released: releaseResult.released,
      awaitingApproval: releaseResult.awaitingApproval === true,
      automaticActivation: false,
      evaluationId: typeof releaseResult.evaluationId === 'string' ? releaseResult.evaluationId : null,
      snapshotId: typeof releaseResult.snapshotId === 'string' ? releaseResult.snapshotId : null,
      releaseFingerprint: typeof releaseResult.releaseFingerprint === 'string' ? releaseResult.releaseFingerprint : null,
      auditArtifactId: typeof releaseResult.auditArtifactId === 'string' ? releaseResult.auditArtifactId : null,
      benchmarkReportArtifactId: typeof releaseResult.benchmarkReportArtifactId === 'string' ? releaseResult.benchmarkReportArtifactId : null,
      reasonCodes: Array.isArray(releaseResult.reasonCodes)
        ? releaseResult.reasonCodes.filter((entry): entry is string => typeof entry === 'string').slice(0, 32)
        : [],
      metrics: {
        baseMean: nullableFinite(metrics.baseMean),
        candidateMean: nullableFinite(metrics.candidateMean),
        shamMean: nullableFinite(metrics.shamMean),
        conservativeHoldoutGain: nullableFinite(metrics.conservativeHoldoutGain),
        conservativeShamGain: nullableFinite(metrics.conservativeShamGain),
        holdoutRegressionHigh: nullableFinite(metrics.holdoutRegressionHigh),
      },
    } : null,
  }
}

export async function loadRegionDashboard(
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<PlasticRegionDashboard> {
  const [status, artifacts] = await Promise.all([
    request(baseUrl, '/api/regions/status', signal),
    request(baseUrl, '/api/regions/artifacts?offset=0&limit=40', signal),
  ])
  const normalizedStatus = normalizeRegionStatus(status)
  const atlasEnvelope = normalizedStatus.head
    ? await request(baseUrl, '/api/regions/atlas', signal)
    : { ...record(status), atlas: null }
  const coherentStatus = normalizeRegionStatus(atlasEnvelope)
  const atlasRecord = record(atlasEnvelope)
  const atlas = record(atlasRecord.atlas)
  const atlasFingerprint = typeof atlas.fingerprint === 'string' ? atlas.fingerprint : null
  if (atlasFingerprint && coherentStatus.head?.fingerprint !== atlasFingerprint) {
    throw new RegionApiError(
      'Atlas görünümü güncellenirken snapshot değişti; karışık iki sürüm gösterilmedi. Yenileyin.',
      409,
      'REGION_DASHBOARD_SNAPSHOT_CHANGED',
    )
  }
  const probes = await request(
    baseUrl,
    `/api/regions/probes?limit=64${atlasFingerprint ? `&atlasFingerprint=${encodeURIComponent(atlasFingerprint)}` : ''}`,
    signal,
  )
  return mergeRegionDashboard(atlasEnvelope, atlasEnvelope, artifacts, probes)
}

export async function loadRegionAtlasArtifact(
  artifactId: string,
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<PlasticRegionAtlasDetail> {
  const raw = await request(
    baseUrl,
    `/api/regions/atlases/${encodeURIComponent(artifactId)}`,
    signal,
  )
  const detail = normalizeRegionAtlasDetail(raw)
  if (!detail) throw new RegionApiError('Seçilen kayıt doğrulanabilir bir atlas taşımıyor.', 409, 'INVALID_REGION_ATLAS_DETAIL')
  return detail
}

/**
 * Opts into a new representation lineage in behavior-free shadow mode. It
 * never rewrites legacy vectors or enables an atlas for live routing.
 */
export async function enableAdaptiveRegionShadow(
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<void> {
  await writeRequest(baseUrl, '/api/config', 'PATCH', {
    plasticRegions: {
      enabled: true,
      representationProfile: 'adaptive-eighth-id-v1',
      mode: 'shadow',
    },
  }, signal)
}

/** Enables proposal production only; active routing still requires release. */
export async function enableAdaptiveRegionProposals(
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<void> {
  await writeRequest(baseUrl, '/api/config', 'PATCH', {
    plasticRegions: {
      enabled: true,
      representationProfile: 'adaptive-eighth-id-v1',
      mode: 'proposal',
    },
  }, signal)
}

/**
 * Requests live routing only after the server has verified that the currently
 * active snapshot carries the exact independent release chain.
 */
export async function enableReleasedAdaptiveRegions(
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<void> {
  await writeRequest(baseUrl, '/api/config', 'PATCH', {
    plasticRegions: {
      enabled: true,
      representationProfile: 'adaptive-eighth-id-v1',
      mode: 'active',
    },
  }, signal)
}

/** Enables/disables the separate read-only diagnostic producer for later cycles. */
export async function configureSemanticRegionProbes(
  enabled: boolean,
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<void> {
  await writeRequest(baseUrl, '/api/config', 'PATCH', {
    plasticRegions: { semanticProbesEnabled: enabled },
  }, signal)
}

export async function queueRegionReleaseEvaluation(
  proposalArtifactId: string,
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<PlasticRegionReleaseJobView> {
  const raw = record(await writeRequest(
    baseUrl,
    `/api/regions/proposals/${encodeURIComponent(proposalArtifactId)}/evaluate`,
    'POST',
    {},
    signal,
  ))
  const job = normalizeReleaseJob(raw.job, raw.duplicate === true)
  if (!job) throw new RegionApiError('Release işi doğrulanabilir bir iş kimliği döndürmedi.', 409, 'INVALID_REGION_RELEASE_JOB')
  return job
}

export async function loadRegionReleaseJob(
  jobId: string,
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<PlasticRegionReleaseJobView | null> {
  const raw = record(await request(baseUrl, `/api/jobs/${encodeURIComponent(jobId)}`, signal))
  return normalizeReleaseJob(raw.job)
}

/** Human approval of the released candidate snapshot; region routing remains off. */
export async function activateReleasedRegionSnapshot(
  snapshotId: string,
  baseUrl = DEFAULT_REGION_SERVICE_URL,
  signal?: AbortSignal,
): Promise<void> {
  await writeRequest(baseUrl, `/api/snapshots/${encodeURIComponent(snapshotId)}/activate`, 'POST', {}, signal)
}
