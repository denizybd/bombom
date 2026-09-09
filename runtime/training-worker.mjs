import { randomUUID } from 'node:crypto'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createTrainingStore } from '../server/store.mjs'
import { OllamaTeacher } from '../server/ollama.mjs'
import { ResourceGovernor, sampleHostResources } from '../server/resource-policy.mjs'
import { createTeacherFeedExecutor, TEACHER_FEED_JOB_TYPE } from '../server/teacher-feed.mjs'
import { createThoughtRuntimeEngine } from '../src/thought-runtime/orchestration.ts'
import { createRealPipelineExecutor } from './core-runtime.mjs'
import { aggregateWorkerMemoryUsage } from './worker-resources.mjs'
import { runEphemeralPipeline } from './ephemeral-pipeline-runner.mjs'
import { createIndependentStructuralBenchmarkRunner } from './structural-benchmark-runner.mjs'
import {
  createPlasticRegionReleaseExecutor,
  PLASTIC_REGION_RELEASE_JOB_TYPE,
} from './plastic-region-release-runner.mjs'
import {
  PLASTIC_REGION_RENEWAL_JOB_TYPE,
  createPlasticRegionRenewalExecutor,
} from './plastic-region-renewal-runner.mjs'
import { unloadTeacherModelAtWorkerExit } from './training-worker-cleanup.mjs'

const PROTOCOL_VERSION = 1
const PARENT_LOSS_EXIT_MS = 5_000

let activeController = null
let activeToken = null
let running = false
let parentRssBytes = 0
const pendingRequests = new Map()
let peakWorkerRssBytes = 0
let peakAggregateRssBytes = 0

function serializeError(error) {
  return {
    code: String(error?.code ?? 'PIPELINE_FAILED'),
    message: String(error?.message ?? error).slice(0, 2_000),
    ...(error?.retryable === true ? { retryable: true } : {}),
    ...(Number.isFinite(error?.status) ? { status: Number(error.status) } : {}),
    ...(error?.resourceState ? { resourceState: structuredClone(error.resourceState) } : {}),
  }
}

function deserializeError(value, fallbackCode = 'JOB_CANCELLED') {
  const error = new Error(String(value?.message ?? 'Ayrı eğitim süreci durduruldu.'))
  error.code = String(value?.code ?? fallbackCode)
  if (value?.retryable === true) error.retryable = true
  if (Number.isFinite(value?.status)) error.status = Number(value.status)
  if (value?.resourceState) error.resourceState = structuredClone(value.resourceState)
  return error
}

function send(message) {
  return new Promise((resolve, reject) => {
    if (!process.connected || typeof process.send !== 'function') {
      const error = new Error('Ana eğitim servisiyle IPC bağlantısı kapalı.')
      error.code = 'JOB_WORKER_PARENT_LOST'
      reject(error)
      return
    }
    process.send(message, (error) => error ? reject(error) : resolve())
  })
}

function requestParent(action, payload) {
  const requestId = randomUUID()
  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject })
    void send({
      type: 'worker-request',
      version: PROTOCOL_VERSION,
      token: activeToken,
      requestId,
      action,
      payload,
    }).catch((error) => {
      pendingRequests.delete(requestId)
      reject(error)
    })
  })
}

function notifyParent(action, payload) {
  void send({
    type: 'worker-notification',
    version: PROTOCOL_VERSION,
    token: activeToken,
    action,
    payload,
  }).catch(() => {})
}

function captureMemoryPeak() {
  const memoryUsage = process.memoryUsage()
  const aggregate = aggregateWorkerMemoryUsage(memoryUsage, parentRssBytes)
  peakWorkerRssBytes = Math.max(peakWorkerRssBytes, memoryUsage.rss)
  peakAggregateRssBytes = Math.max(peakAggregateRssBytes, aggregate.rss)
  return aggregate
}

function workerMetrics() {
  captureMemoryPeak()
  return {
    pid: process.pid,
    peakWorkerRssBytes,
    peakAggregateRssBytes,
    finalParentRssBytes: parentRssBytes,
  }
}

async function run(message) {
  if (running) {
    const error = new Error('Ayrı eğitim süreci aynı anda yalnız bir iş çalıştırabilir.')
    error.code = 'JOB_WORKER_ALREADY_RUNNING'
    throw error
  }
  if (
    typeof message.token !== 'string' || !message.token ||
    typeof message.dataDirectory !== 'string' || !message.dataDirectory ||
    typeof message.jobId !== 'string' || !message.jobId ||
    typeof message.owner !== 'string' || !message.owner ||
    !Number.isInteger(message.attempt) || message.attempt < 1
  ) {
    const error = new Error('Ayrı eğitim süreci başlatma zarfı geçersiz.')
    error.code = 'JOB_WORKER_PROTOCOL_ERROR'
    throw error
  }
  running = true
  activeToken = message.token
  parentRssBytes = Number.isFinite(message.parentRssBytes) && message.parentRssBytes >= 0
    ? Number(message.parentRssBytes)
    : 0
  activeController = new AbortController()
  let store
  let job
  let ollama
  try {
    store = await createTrainingStore({
      dataDirectory: path.resolve(message.dataDirectory),
      requireSqlite: true,
    })
    const claimed = store.assertJobClaim(message.jobId, message.owner, message.attempt)
    job = store.getJob(message.jobId)
    if (!job || claimed?.id !== job.id) {
      const error = new Error('Claim edilmiş eğitim işi ayrı süreçte bulunamadı.')
      error.code = 'STALE_JOB_LEASE'
      throw error
    }
    if (job.state === 'cancel_requested' || job.state === 'pause_requested') {
      const error = new Error(job.state === 'cancel_requested'
        ? 'İş kullanıcı tarafından iptal edildi.'
        : 'İş kullanıcı tarafından duraklatıldı.')
      error.code = job.state === 'cancel_requested' ? 'JOB_CANCELLED' : 'JOB_PAUSED'
      throw error
    }
    const config = store.getConfig()
    const isRegionReleaseJob = job.type === PLASTIC_REGION_RELEASE_JOB_TYPE ||
      job.type === PLASTIC_REGION_RENEWAL_JOB_TYPE
    ollama = isRegionReleaseJob
      ? null
      : new OllamaTeacher(() => store.getConfig())
    const directPipelineExecutor = createRealPipelineExecutor()
    const pipelineWorkerEntry = fileURLToPath(new URL('./pipeline-worker.mjs', import.meta.url))
    const pipelineExecutor = job.type === TEACHER_FEED_JOB_TYPE
      ? (pipelineJob, pipelineContext) => runEphemeralPipeline({
          workerEntry: pipelineWorkerEntry,
          dataDirectory: store.dataDirectory,
          pipelineJob,
          context: pipelineContext,
          owner: message.owner,
          attempt: message.attempt,
          serviceParentRss: () => parentRssBytes,
        })
      : directPipelineExecutor
    const structuralBenchmarkReport = createIndependentStructuralBenchmarkRunner({
      thoughtEngine: createThoughtRuntimeEngine(),
    })
    const teacherExecutor = createTeacherFeedExecutor({ pipelineExecutor, structuralBenchmarkReport })
    const releaseExecutor = createPlasticRegionReleaseExecutor({
      thoughtEngine: createThoughtRuntimeEngine(),
    })
    const renewalExecutor = createPlasticRegionRenewalExecutor({
      thoughtEngine: createThoughtRuntimeEngine(),
    })
    const executor = job.type === PLASTIC_REGION_RELEASE_JOB_TYPE
      ? releaseExecutor
      : job.type === PLASTIC_REGION_RENEWAL_JOB_TYPE
        ? renewalExecutor
        : teacherExecutor
    const resourceGovernor = new ResourceGovernor({
      store,
      // The policy is a service-wide envelope, not a per-process allowance.
      // Add the latest parent RSS sample to this worker's RSS so 70/85/100%
      // thresholds remain authoritative after process isolation.
      sampler: (options) => {
        return sampleHostResources({
          ...options,
          memoryUsage: captureMemoryPeak(),
        })
      },
    })
    let lastSpillEventAt = 0

    const context = {
      jobId: job.id,
      store,
      ollama,
      signal: activeController.signal,
      config: structuredClone(config),
      jobClaim: Object.freeze({ owner: message.owner, attempt: message.attempt }),
      resourceBudgets: {
        cpuPercent: config.budgets.cpuPercent,
        ramMb: config.budgets.ramMb,
        maxRunMinutes: config.budgets.maxRunMinutes,
        diskMb: config.budgets.diskMb,
      },
      resourceGovernor,
      checkResources: (stage = 'runtime') => {
        const resource = resourceGovernor.assertSafe()
        if (resource.state.shouldSpill && Date.now() - lastSpillEventAt >= 30_000) {
          lastSpillEventAt = Date.now()
          notifyParent('event', {
            type: 'resource-spill-requested',
            level: 'warning',
            stage,
            data: { state: resource.state, sample: resource.sample },
          })
        }
        return resource.state
      },
      emit: (event) => requestParent('event', event),
      progress: (stage, progress) => requestParent('progress', { stage, progress }),
      checkpoint: async (stage) => {
        if (activeController.signal.aborted) {
          throw activeController.signal.reason ?? new DOMException('Aborted', 'AbortError')
        }
        const current = store.assertJobClaim(job.id, message.owner, message.attempt)
        context.checkResources(stage)
        if (current.state === 'cancel_requested' || current.state === 'pause_requested') {
          const error = new Error(current.state === 'cancel_requested'
            ? 'İş kullanıcı tarafından iptal edildi.'
            : `İş ${stage} aşamasından önce duraklatıldı.`)
          error.code = current.state === 'cancel_requested' ? 'JOB_CANCELLED' : 'JOB_PAUSED'
          activeController.abort(error)
          throw error
        }
      },
    }

    notifyParent('event', {
      type: 'job-worker-ready',
      stage: 'worker-running',
      data: { pid: process.pid, isolation: 'process' },
    })
    const result = await executor(job, context)
    await send({
      type: 'worker-terminal',
      version: PROTOCOL_VERSION,
      token: activeToken,
      ok: true,
      result,
      workerMetrics: workerMetrics(),
    })
    process.exitCode = 0
  } catch (error) {
    await send({
      type: 'worker-terminal',
      version: PROTOCOL_VERSION,
      token: activeToken ?? message.token,
      ok: false,
      error: serializeError(error),
      workerMetrics: workerMetrics(),
    }).catch(() => {})
    process.exitCode = 1
  } finally {
    for (const pending of pendingRequests.values()) {
      const error = new Error('Ayrı eğitim süreci kapanıyor.')
      error.code = 'JOB_WORKER_CLOSED'
      pending.reject(error)
    }
    pendingRequests.clear()
    await unloadTeacherModelAtWorkerExit({ job, ollama }).catch(() => {})
    try {
      store?.close()
    } finally {
      if (process.connected) process.disconnect()
    }
  }
}

process.on('message', (message) => {
  if (!message || message.version !== PROTOCOL_VERSION) return
  if (message.type === 'worker-response' && message.token === activeToken) {
    const pending = pendingRequests.get(message.requestId)
    if (!pending) return
    pendingRequests.delete(message.requestId)
    if (message.error) pending.reject(deserializeError(message.error, 'JOB_WORKER_PARENT_REQUEST_FAILED'))
    else pending.resolve(message.result)
    return
  }
  if (message.type === 'worker-abort' && message.token === activeToken) {
    activeController?.abort(deserializeError(message.reason))
    return
  }
  if (message.type === 'worker-resource-baseline' && message.token === activeToken) {
    if (Number.isFinite(message.parentRssBytes) && message.parentRssBytes >= 0) {
      parentRssBytes = Number(message.parentRssBytes)
    }
    return
  }
  if (message.type === 'worker-run') {
    void run(message)
  }
})

process.once('disconnect', () => {
  const error = new Error('Ana eğitim servisi kapandı; ayrı süreç güvenli biçimde durduruluyor.')
  error.code = 'SERVICE_STOP'
  activeController?.abort(error)
  const timer = setTimeout(() => process.exit(1), PARENT_LOSS_EXIT_MS)
  timer.unref?.()
})

process.once('SIGTERM', () => {
  const error = new Error('Ayrı eğitim süreci sonlandırılıyor.')
  error.code = 'SERVICE_STOP'
  if (activeController) activeController.abort(error)
  else process.exit(1)
})

if (process.env.BOMBOM_JOB_WORKER_CHILD !== '1' || typeof process.send !== 'function') {
  throw new Error('training-worker yalnız IPC kontrollü eğitim alt süreci olarak çalıştırılabilir.')
}

await send({ type: 'worker-ready', version: PROTOCOL_VERSION, pid: process.pid })
