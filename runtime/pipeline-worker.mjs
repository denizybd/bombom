import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { createTrainingStore } from '../server/store.mjs'
import { OllamaTeacher } from '../server/ollama.mjs'
import { ResourceGovernor, sampleHostResources } from '../server/resource-policy.mjs'
import { createRealPipelineExecutor } from './core-runtime.mjs'
import { aggregateWorkerMemoryUsage } from './worker-resources.mjs'

const VERSION = 1
let token = null
let controller = null
let ancestorRssBytes = 0
let peakWorkerRssBytes = 0
let peakAggregateRssBytes = 0
const pending = new Map()

function send(message) {
  return new Promise((resolve, reject) => {
    if (!process.connected || typeof process.send !== 'function') return reject(Object.assign(new Error('Pipeline worker IPC bağlantısı kapalı.'), { code: 'JOB_WORKER_PARENT_LOST' }))
    process.send(message, (error) => error ? reject(error) : resolve())
  })
}

function request(action, payload) {
  const requestId = randomUUID()
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject })
    void send({ type: 'worker-request', version: VERSION, token, requestId, action, payload }).catch((error) => { pending.delete(requestId); reject(error) })
  })
}

function notify(action, payload) {
  void send({ type: 'worker-notification', version: VERSION, token, action, payload }).catch(() => {})
}

function memorySample() {
  const own = process.memoryUsage()
  const aggregate = aggregateWorkerMemoryUsage(own, ancestorRssBytes)
  peakWorkerRssBytes = Math.max(peakWorkerRssBytes, own.rss)
  peakAggregateRssBytes = Math.max(peakAggregateRssBytes, aggregate.rss)
  return aggregate
}

function metrics() {
  memorySample()
  return { pid: process.pid, peakWorkerRssBytes, peakAggregateRssBytes, finalAncestorRssBytes: ancestorRssBytes }
}

function serializeError(error) {
  return {
    code: String(error?.code ?? 'PIPELINE_FAILED'),
    message: String(error?.message ?? error).slice(0, 2_000),
    ...(error?.retryable === true ? { retryable: true } : {}),
    ...(error?.resourceState ? { resourceState: structuredClone(error.resourceState) } : {}),
  }
}

function fromError(value, fallback = 'JOB_CANCELLED') {
  const error = new Error(String(value?.message ?? 'Pipeline worker durduruldu.'))
  error.code = String(value?.code ?? fallback)
  if (value?.retryable) error.retryable = true
  if (value?.resourceState) error.resourceState = structuredClone(value.resourceState)
  return error
}

async function run(message) {
  token = message.token
  ancestorRssBytes = Math.max(0, Number(message.ancestorRssBytes) || 0)
  controller = new AbortController()
  let store
  try {
    if (
      typeof message.dataDirectory !== 'string' || !message.dataDirectory ||
      typeof message.jobId !== 'string' || !message.jobId ||
      typeof message.owner !== 'string' || !message.owner ||
      !Number.isInteger(message.attempt) || message.attempt < 1 ||
      !message.pipelineJob || typeof message.pipelineJob !== 'object'
    ) throw Object.assign(new Error('Pipeline worker başlatma zarfı geçersiz.'), { code: 'JOB_WORKER_PROTOCOL_ERROR' })
    store = await createTrainingStore({ dataDirectory: path.resolve(message.dataDirectory), requireSqlite: true })
    const claimed = store.assertJobClaim(message.jobId, message.owner, message.attempt)
    if (claimed.state !== 'running') throw Object.assign(new Error('Pipeline claim artık yazılabilir durumda değil.'), { code: 'STALE_JOB_LEASE' })
    const config = store.getConfig()
    const governor = new ResourceGovernor({
      store,
      sampler: (options) => sampleHostResources({ ...options, memoryUsage: memorySample() }),
    })
    let lastSpillAt = 0
    const context = {
      jobId: message.jobId,
      store,
      ollama: new OllamaTeacher(() => store.getConfig()),
      signal: controller.signal,
      config: structuredClone(config),
      jobClaim: Object.freeze({ owner: message.owner, attempt: message.attempt }),
      resourceBudgets: {
        cpuPercent: config.budgets.cpuPercent,
        ramMb: config.budgets.ramMb,
        maxRunMinutes: config.budgets.maxRunMinutes,
        diskMb: config.budgets.diskMb,
      },
      resourceGovernor: governor,
      checkResources: (stage = 'runtime') => {
        const resource = governor.assertSafe()
        if (resource.state.shouldSpill && Date.now() - lastSpillAt >= 30_000) {
          lastSpillAt = Date.now()
          notify('event', { type: 'resource-spill-requested', level: 'warning', stage, data: { state: resource.state, sample: resource.sample } })
        }
        return resource.state
      },
      emit: (event) => request('event', event),
      progress: (stage, progress) => request('progress', { stage, progress }),
      checkpoint: async (stage) => {
        if (controller.signal.aborted) throw controller.signal.reason ?? new DOMException('Aborted', 'AbortError')
        const current = store.assertJobClaim(message.jobId, message.owner, message.attempt)
        context.checkResources(stage)
        if (current.state === 'cancel_requested' || current.state === 'pause_requested') {
          const error = new Error(current.state === 'cancel_requested' ? 'İş iptal edildi.' : `İş ${stage} öncesinde duraklatıldı.`)
          error.code = current.state === 'cancel_requested' ? 'JOB_CANCELLED' : 'JOB_PAUSED'
          controller.abort(error)
          throw error
        }
      },
    }
    notify('event', { type: 'pipeline-worker-ready', stage: 'teacher-feed:pipeline-worker-running', data: { pid: process.pid } })
    const result = await createRealPipelineExecutor()(message.pipelineJob, context)
    await send({ type: 'worker-terminal', version: VERSION, token, ok: true, result, workerMetrics: metrics() })
    process.exitCode = 0
  } catch (error) {
    await send({ type: 'worker-terminal', version: VERSION, token, ok: false, error: serializeError(error), workerMetrics: metrics() }).catch(() => {})
    process.exitCode = 1
  } finally {
    for (const waiter of pending.values()) waiter.reject(Object.assign(new Error('Pipeline worker kapanıyor.'), { code: 'JOB_WORKER_CLOSED' }))
    pending.clear()
    store?.close()
    if (process.connected) process.disconnect()
  }
}

process.on('message', (message) => {
  if (!message || message.version !== VERSION) return
  if (message.type === 'worker-response' && message.token === token) {
    const waiter = pending.get(message.requestId)
    if (!waiter) return
    pending.delete(message.requestId)
    if (message.error) waiter.reject(fromError(message.error, 'JOB_WORKER_PARENT_REQUEST_FAILED'))
    else waiter.resolve(message.result)
    return
  }
  if (message.type === 'worker-abort' && message.token === token) {
    controller?.abort(fromError(message.reason))
    return
  }
  if (message.type === 'worker-resource-baseline' && message.token === token) {
    ancestorRssBytes = Math.max(0, Number(message.ancestorRssBytes) || 0)
    return
  }
  if (message.type === 'worker-run' && token === null) void run(message)
})

process.once('disconnect', () => {
  controller?.abort(Object.assign(new Error('Teacher coordinator kapandı.'), { code: 'SERVICE_STOP' }))
  const timer = setTimeout(() => process.exit(1), 5_000)
  timer.unref?.()
})
process.once('SIGTERM', () => {
  if (controller) controller.abort(Object.assign(new Error('Pipeline worker sonlandırılıyor.'), { code: 'SERVICE_STOP' }))
  else process.exit(1)
})

if (process.env.BOMBOM_PIPELINE_WORKER_CHILD !== '1' || typeof process.send !== 'function') {
  throw new Error('pipeline-worker yalnız IPC kontrollü alt süreç olarak çalıştırılabilir.')
}
await send({ type: 'worker-ready', version: VERSION, pid: process.pid })
