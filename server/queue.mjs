import { randomUUID } from 'node:crypto'
import { fork } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { isDeepStrictEqual } from 'node:util'
import { ResourceGovernor } from './resource-policy.mjs'
import { plasticRegionSnapshotActivationGate } from '../runtime/plastic-region-cache.mjs'

const terminalStates = new Set(['succeeded', 'failed', 'cancelled'])
const ISOLATED_JOB_TYPES = new Set([
  'full-cycle',
  'teacher-feed-campaign',
  'plastic-region-release-evaluation',
  'plastic-region-release-renewal',
])
const ATOMIC_CANDIDATE_JOB_TYPES = new Set([
  'plastic-region-release-evaluation',
  'plastic-region-release-renewal',
])
const WORKER_PROTOCOL_VERSION = 1
const WORKER_HANDSHAKE_TIMEOUT_MS = 10_000
const WORKER_ABORT_GRACE_MS = 5_000
const WORKER_KILL_GRACE_MS = 5_000
const MAX_WORKER_DIAGNOSTIC_BYTES = 16 * 1024

function assertLiveSnapshotActivationPolicy(store, snapshot) {
  const liveConfig = store.getConfig()
  if (liveConfig?.automation?.level !== 'controlled') {
    const error = new Error('Etkinleştirme anında canlı otomasyon seviyesi controlled değil; aday model etkinleştirilmedi.')
    error.code = 'SNAPSHOT_ACTIVATION_POLICY_CHANGED'
    error.details = { automationLevel: liveConfig?.automation?.level ?? null }
    throw error
  }
  const regionGate = plasticRegionSnapshotActivationGate(store, snapshot)
  if (regionGate.required && !regionGate.eligible) {
    const error = new Error('Uyarlanabilir temsil adayı doğrulanmış plastik bölge release zinciri taşımıyor.')
    error.code = 'PLASTIC_REGION_RELEASE_REQUIRED'
    error.details = {
      reason: regionGate.reason ?? null,
      snapshotId: snapshot?.id ?? null,
    }
    throw error
  }
  return { eligible: true }
}

export class DurableJobQueue {
  constructor({
    store,
    executor,
    ollama,
    onEvent = () => {},
    workerEntry = process.env.BOMBOM_JOB_WORKER_ENTRY,
    workerIsolation = process.env.BOMBOM_JOB_ISOLATION === '1',
    forkImpl = fork,
  }) {
    this.store = store
    this.executor = executor
    this.ollama = ollama
    this.onEvent = onEvent
    this.owner = `worker-${process.pid}-${randomUUID()}`
    this.running = false
    this.timer = null
    this.active = new Map()
    this.nextRecoveryAt = 0
    this.resourceGovernor = new ResourceGovernor({ store })
    this.lastResourceNoticeAt = 0
    this.workerEntry = typeof workerEntry === 'string' && workerEntry.trim()
      ? path.resolve(workerEntry)
      : null
    this.workerIsolation = workerIsolation === true || workerIsolation === 'fallback'
    this.workerFallback = workerIsolation === 'fallback'
    this.forkImpl = forkImpl
  }

  start() {
    if (this.running) return
    this.running = true
    let config
    try {
      config = this.store.getConfig()
      const recovered = this.store.recoverExpiredJobs()
      if (recovered) void this.emit({ type: 'queue-recovered', level: 'warning', data: { recovered } })
    } catch (error) {
      void this.reportQueueCycleFailure(error, 'startup-recovery')
    }
    this.nextRecoveryAt = Date.now() + this.recoveryIntervalMs(config ?? { automation: { leaseMs: 5_000 } })
    this.schedule(0)
  }

  async stop({ abort = false } = {}) {
    this.running = false
    clearTimeout(this.timer)
    this.timer = null
    if (abort) for (const { controller } of this.active.values()) controller.abort(Object.assign(new Error('Servis kapatılıyor.'), { code: 'SERVICE_STOP' }))
    await Promise.allSettled([...this.active.values()].map(({ promise }) => promise))
  }

  schedule(delay) {
    if (!this.running || this.timer) return
    this.timer = setTimeout(() => {
      this.timer = null
      void this.tick()
    }, delay)
    this.timer.unref?.()
  }

  async tick() {
    if (!this.running) return
    let pollMs = 1_000
    try {
      const config = this.store.getConfig()
      pollMs = config.automation.pollMs
      if (this.active.size === 0 && Date.now() >= this.nextRecoveryAt) {
        const recovered = this.store.recoverExpiredJobs()
        if (recovered) await this.emit({ type: 'queue-recovered', level: 'warning', data: { recovered } })
        this.nextRecoveryAt = Date.now() + this.recoveryIntervalMs(config)
      }
      if (!this.store.getPipelineState().paused && this.active.size < config.automation.concurrency) {
        if (typeof this.store.hasQueuedJob !== 'function') {
          const error = new Error('Kuyruk için sınırlı iş durum dizini eksik.')
          error.code = 'STORE_JOB_INDEX_CONTRACT_MISSING'
          throw error
        }
        if (!this.store.hasQueuedJob()) return
        const resource = this.resourceGovernor.inspect({ admission: true })
        if (!resource.state.allowed) {
          if (this.store.getPipelineState().phase !== 'resource-wait') {
            this.store.setPipelineState({
              phase: 'resource-wait',
              activeJobId: null,
              message: 'Güvenli RAM/SSD rezervi bekleniyor',
            })
          }
          if (Date.now() - this.lastResourceNoticeAt >= 30_000) {
            this.lastResourceNoticeAt = Date.now()
            await this.emit({
              type: 'resource-admission-wait',
              level: 'warning',
              data: { state: resource.state, sample: resource.sample },
            })
          }
          return
        }
        const job = this.store.claimNextJob(
          this.owner,
          config.automation.leaseMs,
          this.executionLeaseMs(config),
        )
        if (job) this.launch(job)
      }
    } catch (error) {
      await this.reportQueueCycleFailure(error, 'tick')
    } finally {
      // A transient SQLite writer collision must not permanently stop polling.
      // schedule() is idempotent if another launch/completion already armed it.
      this.schedule(pollMs)
    }
  }

  async reportQueueCycleFailure(error, stage) {
    try {
      await this.emit({
        type: 'queue-cycle-failed',
        level: 'warning',
        stage,
        data: serializeError(error),
      })
    } catch {
      // The same SQLite lock can also prevent the diagnostic event. Polling is
      // still re-armed by tick()'s finally block and can report a later cycle.
    }
  }

  recoveryIntervalMs(config) {
    return Math.max(1_000, Math.min(5_000, Math.floor(config.automation.leaseMs / 2)))
  }

  executionLeaseMs(config) {
    const heartbeatMs = Number(config.automation.leaseMs)
    const maximumRunMs = Number(config.budgets.maxRunMinutes) * 60_000
    // maxRunMinutes is validated and bounded by the config contract. The fixed
    // grace absorbs claim/start latency and one final durable terminal write;
    // unlike heartbeat renewal, this deadline never slides forward.
    const graceMs = Math.max(60_000, heartbeatMs * 2)
    return Math.max(heartbeatMs, maximumRunMs + graceMs)
  }

  launch(job) {
    const controller = new AbortController()
    const promise = this.execute(job, controller).finally(() => {
      this.active.delete(job.id)
      this.schedule(0)
    })
    this.active.set(job.id, { controller, promise })
  }

  async execute(job, controller) {
    const config = this.store.getConfig()
    let lastSpillEventAt = 0
    let lastProgressWriteAt = 0
    let lastProgressStage = null
    const lastRuntimeProgressEventAt = new Map()
    this.store.setPipelineState({ phase: 'running', activeJobId: job.id, message: 'Eğitim çevrimi çalışıyor' })
    await this.emit({ type: 'job-started', jobId: job.id, jobAttempt: job.attempt, stage: 'starting' })
    const leaseTimer = setInterval(() => {
      try {
        const renewed = this.store.renewLease(job.id, this.owner, config.automation.leaseMs, job.attempt)
        if (renewed) return
        const current = this.store.getJob(job.id)
        // The isolated release worker may have atomically published its
        // candidate and terminal job record just before its IPC terminal frame.
        // That durable success is not a lost lease and must not abort the child.
        if (current?.state === 'succeeded' && ATOMIC_CANDIDATE_JOB_TYPES.has(job.type) &&
            current.result?.automaticActivation === false && typeof current.result?.snapshotId === 'string') return
        if (current?.state === 'cancel_requested' || current?.state === 'pause_requested') {
          const error = new Error(current.state === 'cancel_requested'
            ? 'İş kullanıcı tarafından iptal edildi.'
            : 'İş kullanıcı tarafından duraklatıldı.')
          error.code = current.state === 'cancel_requested' ? 'JOB_CANCELLED' : 'JOB_PAUSED'
          controller.abort(error)
          return
        }
        const executionDeadline = Date.parse(current?.executionLeaseUntil ?? '')
        if (
          current?.leaseOwner !== this.owner ||
          current?.attempt !== job.attempt ||
          (Number.isFinite(executionDeadline) && executionDeadline <= Date.now())
        ) {
          controller.abort(staleLeaseError(job, current))
        }
      } catch (error) {
        // A competing SQLite writer can transiently block this heartbeat. The
        // fixed execution lease remains authoritative; report and retry on the
        // next interval instead of letting a timer exception kill the process.
        void this.reportQueueCycleFailure(error, 'lease-renewal')
      }
    }, Math.max(1_000, Math.floor(config.automation.leaseMs / 3)))
    leaseTimer.unref?.()
    const runBudgetTimer = setTimeout(() => controller.abort(Object.assign(new Error('Çalışma süresi bütçesi aşıldı.'), { code: 'RUN_TIME_BUDGET_EXCEEDED' })), config.budgets.maxRunMinutes * 60_000)
    runBudgetTimer.unref?.()
    const context = {
      jobId: job.id,
      store: this.store,
      ollama: this.ollama,
      signal: controller.signal,
      config: structuredClone(config),
      jobClaim: Object.freeze({ owner: this.owner, attempt: job.attempt }),
      resourceBudgets: { cpuPercent: config.budgets.cpuPercent, ramMb: config.budgets.ramMb, maxRunMinutes: config.budgets.maxRunMinutes, diskMb: config.budgets.diskMb },
      resourceGovernor: this.resourceGovernor,
      checkResources: (stage = 'runtime') => {
        const resource = this.resourceGovernor.assertSafe()
        if (resource.state.shouldSpill && Date.now() - lastSpillEventAt >= 30_000) {
          lastSpillEventAt = Date.now()
          void this.emitClaimed(job, {
            type: 'resource-spill-requested',
            level: 'warning',
            stage,
            data: { state: resource.state, sample: resource.sample },
          })
        }
        return resource.state
      },
      emit: (event) => {
        const eventType = String(event?.type ?? '')
        if (eventType.endsWith('-progress')) {
          const eventKey = `${eventType}\u0000${String(event?.stage ?? '')}`
          const currentTime = Date.now()
          const previous = lastRuntimeProgressEventAt.get(eventKey) ?? 0
          if (currentTime - previous < Number(config.resourcePolicy?.progressFlushMs ?? 0)) {
            return Promise.resolve(null)
          }
          lastRuntimeProgressEventAt.set(eventKey, currentTime)
        }
        return this.emitClaimed(job, event)
      },
      progress: async (stage, progress) => {
        const boundedProgress = Math.max(0, Math.min(1, progress))
        const currentTime = Date.now()
        const flushInterval = Number(config.resourcePolicy?.progressFlushMs ?? 0)
        if (
          stage === lastProgressStage && boundedProgress > 0 && boundedProgress < 1 &&
          currentTime - lastProgressWriteAt < flushInterval
        ) return
        lastProgressWriteAt = currentTime
        lastProgressStage = stage
        this.store.updateClaimedJob(job.id, this.owner, job.attempt, { stage, progress: boundedProgress })
        this.store.setPipelineState({ phase: stage, activeJobId: job.id, message: stage })
        await this.emitClaimed(job, { type: 'job-progress', stage, progress: boundedProgress })
      },
      checkpoint: async (stage) => {
        if (controller.signal.aborted) throw controller.signal.reason ?? new DOMException('Aborted', 'AbortError')
        const current = this.store.assertJobClaim(job.id, this.owner, job.attempt)
        context.checkResources(stage)
        if (current.state === 'cancel_requested') {
          controller.abort(new Error('İş kullanıcı tarafından iptal edildi.'))
          const error = new Error('İş kullanıcı tarafından iptal edildi.')
          error.code = 'JOB_CANCELLED'
          throw error
        }
        if (current.state === 'pause_requested') {
          const error = new Error(`İş ${stage} aşamasından önce duraklatıldı.`)
          error.code = 'JOB_PAUSED'
          throw error
        }
      },
    }
    try {
      const result = await this.runExecutor(job, context, controller)
      if (result?.atomicCandidateJobCompletion === true) {
        const { atomicCandidateJobCompletion: _marker, ...publicResult } = result
        const settled = this.store.getJob(job.id)
        if (!ATOMIC_CANDIDATE_JOB_TYPES.has(job.type) || settled?.state !== 'succeeded' ||
            !isDeepStrictEqual(settled.result, publicResult)) {
          const error = new Error('Atomik aday yayını ile kalıcı iş sonucu uyuşmuyor.')
          error.code = 'ATOMIC_CANDIDATE_JOB_RESULT_MISMATCH'
          throw error
        }
        try {
          await this.emit({ type: 'job-succeeded', jobId: job.id, jobAttempt: job.attempt, data: publicResult })
        } catch (telemetryError) {
          await this.reportQueueCycleFailure(telemetryError, 'post-candidate-commit-telemetry')
        }
      } else if (result?.activationSnapshotId) {
        if (typeof this.store.completeClaimedJobWithSnapshot !== 'function') {
          const error = new Error('Atomik iş + snapshot tamamlama primitive’i eksik.')
          error.code = 'STORE_CLAIM_CONTRACT_MISSING'
          throw error
        }
        const finalResult = { ...result, committed: true }
        this.store.completeClaimedJobWithSnapshot(
          job.id,
          this.owner,
          job.attempt,
          result.activationSnapshotId,
          finalResult,
          (snapshot) => assertLiveSnapshotActivationPolicy(this.store, snapshot),
        )
        // Model/job commit is already durable and atomic. Telemetry failure
        // must not roll a successful model transition into the retry path.
        try {
          await this.emit({ type: 'snapshot-activated', jobId: job.id, jobAttempt: job.attempt, snapshotId: result.activationSnapshotId })
          await this.emit({ type: 'job-succeeded', jobId: job.id, jobAttempt: job.attempt, data: finalResult })
        } catch (telemetryError) {
          await this.reportQueueCycleFailure(telemetryError, 'post-commit-telemetry')
        }
      } else {
        const settlement = this.settleJob(job, { action: 'success', result })
        await this.emitJobSettlement(job, settlement, { result })
      }
    } catch (error) {
      const serialized = serializeError(error)
      const action = error.code === 'JOB_PAUSED' ||
        error.code === 'RAM_BUDGET_EXCEEDED' ||
        error.code === 'DISK_RESOURCE_RESERVE_EXCEEDED' ||
        error.code === 'DISK_BUDGET_EXCEEDED' ||
        error.code === 'ARTIFACT_PAYLOAD_TOO_LARGE' ||
        error.code === 'NEURON_RESIDENCY_RESOURCE_WAIT' ||
        error.code === 'RUN_TIME_BUDGET_EXCEEDED' ||
        error.code === 'JOB_WORKER_UNAVAILABLE' ||
        error.code === 'JOB_WORKER_SPAWN_FAILED' ||
        error.code === 'JOB_WORKER_HANDSHAKE_TIMEOUT'
        ? 'pause'
        : error.code === 'JOB_CANCELLED'
          ? 'cancel'
          : error.code === 'SERVICE_STOP'
            ? 'release'
            : job.attempt < job.maxAttempts && isRetryable(error)
              ? 'retry'
              : 'fail'
      let settlement
      try {
        // settleClaimedJob intentionally preserves prior diagnostics for a
        // resumable pause. Persist the concrete resource/worker reason first
        // so operators can see why the safety valve closed.
        if (action === 'pause') {
          this.store.updateClaimedJob(job.id, this.owner, job.attempt, { error: serialized })
        }
        settlement = this.settleJob(job, { action, error: serialized })
      } catch (claimError) {
        if (claimError?.code === 'STALE_JOB_LEASE') return
        throw claimError
      }
      await this.emitJobSettlement(job, settlement, { error: serialized })
    } finally {
      clearInterval(leaseTimer)
      clearTimeout(runBudgetTimer)
      const current = this.store.getJob(job.id)
      if (!current || terminalStates.has(current.state) || current.state === 'paused') {
        this.store.setPipelineState({ phase: current?.state ?? 'idle', activeJobId: null, message: current?.stage ?? 'Hazır' })
      }
    }
  }

  shouldIsolate(job) {
    return this.workerIsolation &&
      this.store.backendKind === 'sqlite' &&
      ISOLATED_JOB_TYPES.has(job.type) &&
      Boolean(this.workerEntry) &&
      existsSync(this.workerEntry)
  }

  async runExecutor(job, context, controller) {
    const requested = this.workerIsolation && ISOLATED_JOB_TYPES.has(job.type)
    if (!this.shouldIsolate(job)) {
      if (requested) {
        const reason = this.store.backendKind !== 'sqlite'
          ? 'concurrent-json-storage-is-unsafe'
          : !this.workerEntry
            ? 'worker-entry-not-configured'
            : 'worker-entry-missing'
        await context.emit({
          type: 'job-worker-fallback',
          level: 'warning',
          stage: 'worker-fallback',
          data: { reason, allowed: this.workerFallback },
        })
        if (!this.workerFallback) throw unavailableWorkerError(reason)
      }
      return this.executor(job, context)
    }
    try {
      return await this.runIsolatedExecutor(job, context, controller)
    } catch (error) {
      if (!['JOB_WORKER_SPAWN_FAILED', 'JOB_WORKER_HANDSHAKE_TIMEOUT'].includes(error?.code)) throw error
      if (!this.workerFallback) throw error
      await context.emit({
        type: 'job-worker-fallback',
        level: 'warning',
        stage: 'worker-fallback',
        data: { reason: error.code, message: error.message },
      })
      return this.executor(job, context)
    }
  }

  async runIsolatedExecutor(job, context, controller) {
    const token = randomUUID()
    const config = context.config
    const initialParentRssBytes = process.memoryUsage().rss
    const parentRssMb = Math.ceil(initialParentRssBytes / (1024 * 1024))
    const heapLimitMb = Math.max(384, Math.floor(Number(config.budgets.ramMb) - parentRssMb - 512))
    const child = this.forkImpl(this.workerEntry, [], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        BOMBOM_JOB_WORKER_CHILD: '1',
      },
      execArgv: workerExecArgv(heapLimitMb),
      serialization: 'advanced',
      stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    })
    let diagnostic = ''
    const captureDiagnostic = (chunk) => {
      diagnostic = `${diagnostic}${String(chunk)}`.slice(-MAX_WORKER_DIAGNOSTIC_BYTES)
    }
    child.stdout?.on('data', captureDiagnostic)
    child.stderr?.on('data', captureDiagnostic)

    const workerResult = new Promise((resolve, reject) => {
      let ready = false
      let terminalMessage = null
      let settled = false
      let forcedFailure = null
      let abortTimer = null
      let killTimer = null
      let resourceTimer = null
      const handshakeTimer = setTimeout(() => {
        forcedFailure = workerFailure('JOB_WORKER_HANDSHAKE_TIMEOUT', 'Ayrı eğitim süreci zamanında hazır olmadı.', diagnostic, true)
        terminateWorker('SIGTERM')
        killTimer = setTimeout(() => terminateWorker('SIGKILL'), WORKER_KILL_GRACE_MS)
        killTimer.unref?.()
      }, WORKER_HANDSHAKE_TIMEOUT_MS)
      handshakeTimer.unref?.()

      const clearTimers = () => {
        clearTimeout(handshakeTimer)
        clearTimeout(abortTimer)
        clearTimeout(killTimer)
        clearInterval(resourceTimer)
      }
      const cleanup = () => {
        clearTimers()
        controller.signal.removeEventListener('abort', onAbort)
        child.removeAllListeners('message')
        child.removeAllListeners('error')
        child.removeAllListeners('exit')
      }
      const finishResolve = (value) => {
        if (settled) return
        settled = true
        cleanup()
        resolve(value)
      }
      const finishReject = (error) => {
        if (settled) return
        settled = true
        cleanup()
        reject(error)
      }
      const terminateWorker = (signal) => {
        if (child.exitCode === null && child.signalCode === null) child.kill(signal)
      }
      const send = (message) => {
        if (!child.connected) return false
        try {
          child.send(message, (error) => {
            if (error && !settled && child.exitCode === null) captureDiagnostic(error.message)
          })
          return true
        } catch (error) {
          captureDiagnostic(error?.message ?? error)
          return false
        }
      }
      const reply = (requestId, result, error) => {
        send({
          type: 'worker-response',
          version: WORKER_PROTOCOL_VERSION,
          token,
          requestId,
          ...(error ? { error: serializeError(error) } : { result }),
        })
      }
      const onAbort = () => {
        const reason = serializeError(controller.signal.reason ?? Object.assign(new Error('İş durduruldu.'), { code: 'JOB_CANCELLED' }))
        send({ type: 'worker-abort', version: WORKER_PROTOCOL_VERSION, token, reason })
        abortTimer = setTimeout(() => terminateWorker('SIGTERM'), WORKER_ABORT_GRACE_MS)
        abortTimer.unref?.()
        killTimer = setTimeout(() => terminateWorker('SIGKILL'), WORKER_ABORT_GRACE_MS + WORKER_KILL_GRACE_MS)
        killTimer.unref?.()
      }

      controller.signal.addEventListener('abort', onAbort, { once: true })
      if (controller.signal.aborted) onAbort()

      child.on('message', async (message) => {
        if (!message || message.version !== WORKER_PROTOCOL_VERSION) return
        if (message.type === 'worker-ready') {
          if (ready || settled) return
          ready = true
          clearTimeout(handshakeTimer)
          if (controller.signal.aborted) return
          const sent = send({
            type: 'worker-run',
            version: WORKER_PROTOCOL_VERSION,
            token,
            dataDirectory: this.store.dataDirectory,
            jobId: job.id,
            owner: this.owner,
            attempt: job.attempt,
            parentRssBytes: process.memoryUsage().rss,
          })
          if (!sent) terminateWorker('SIGTERM')
          resourceTimer = setInterval(() => {
            send({
              type: 'worker-resource-baseline',
              version: WORKER_PROTOCOL_VERSION,
              token,
              parentRssBytes: process.memoryUsage().rss,
            })
          }, Math.max(250, Number(config.resourcePolicy?.sampleIntervalMs ?? 1_000)))
          resourceTimer.unref?.()
          return
        }
        if (message.token !== token) return
        if (message.type === 'worker-request' && typeof message.requestId === 'string') {
          try {
            if (message.action === 'event') {
              reply(message.requestId, await context.emit(message.payload))
            } else if (message.action === 'progress') {
              const { stage, progress } = message.payload ?? {}
              reply(message.requestId, await context.progress(stage, progress))
            } else {
              const error = new Error('Bilinmeyen ayrı süreç isteği.')
              error.code = 'JOB_WORKER_PROTOCOL_ERROR'
              throw error
            }
          } catch (error) {
            reply(message.requestId, undefined, error)
          }
          return
        }
        if (message.type === 'worker-notification' && message.action === 'event') {
          void context.emit(message.payload).catch((error) => captureDiagnostic(error?.message ?? error))
          return
        }
        if (message.type === 'worker-terminal') {
          terminalMessage = message
          abortTimer = setTimeout(() => terminateWorker('SIGTERM'), WORKER_ABORT_GRACE_MS)
          abortTimer.unref?.()
          killTimer = setTimeout(() => terminateWorker('SIGKILL'), WORKER_ABORT_GRACE_MS + WORKER_KILL_GRACE_MS)
          killTimer.unref?.()
        }
      })
      child.once('error', (error) => {
        finishReject(workerFailure('JOB_WORKER_SPAWN_FAILED', 'Ayrı eğitim süreci başlatılamadı.', `${diagnostic}\n${error.message}`, true))
      })
      child.once('exit', (code, signal) => {
        clearTimers()
        void context.emit({
          type: 'job-worker-released',
          stage: 'worker-released',
          level: terminalMessage?.ok === true && code === 0 ? 'info' : 'warning',
          data: {
            pid: child.pid ?? null,
            exitCode: code,
            signal: signal ?? null,
            memoryReleased: true,
            ...(terminalMessage?.workerMetrics ? { memory: terminalMessage.workerMetrics } : {}),
          },
        }).catch((error) => captureDiagnostic(error?.message ?? error)).finally(() => {
          if (forcedFailure) {
            finishReject(forcedFailure)
            return
          }
          if (terminalMessage?.ok === true && code === 0) {
            finishResolve(terminalMessage.result)
            return
          }
          if (terminalMessage?.ok === false) {
            finishReject(deserializeWorkerError(terminalMessage.error, diagnostic))
            return
          }
          if (controller.signal.aborted) {
            finishReject(deserializeWorkerError(serializeError(controller.signal.reason), diagnostic))
            return
          }
          finishReject(workerFailure(
            'JOB_WORKER_EXITED',
            `Ayrı eğitim süreci sonuç vermeden kapandı (kod=${code ?? 'null'}, sinyal=${signal ?? 'yok'}).`,
            diagnostic,
            true,
          ))
        })
      })
    })
    // A fork failure may reject before the durable spawn telemetry write has
    // completed. Attach a handler immediately to avoid an unhandled-rejection
    // turn while still returning the original promise to execute().
    workerResult.catch(() => {})
    try {
      await context.emit({
        type: 'job-worker-spawned',
        stage: 'worker-starting',
        data: { pid: child.pid ?? null, isolation: 'process', heapLimitMb },
      })
    } catch (error) {
      child.kill('SIGTERM')
      await workerResult.catch(() => {})
      throw error
    }
    return workerResult
  }

  settleJob(job, settlement) {
    if (typeof this.store.settleClaimedJob !== 'function') {
      const error = new Error('Atomik claimed iş sonuçlandırma primitive’i eksik.')
      error.code = 'STORE_CLAIM_CONTRACT_MISSING'
      throw error
    }
    return this.store.settleClaimedJob(
      job.id,
      this.owner,
      job.attempt,
      settlement,
    )
  }

  async emitJobSettlement(job, settlement, { result, error } = {}) {
    const event = settlement.action === 'success'
      ? { type: 'job-succeeded', data: result }
      : settlement.action === 'pause'
        ? { type: 'job-paused', level: 'warning' }
        : settlement.action === 'cancel'
          ? { type: 'job-cancelled', level: 'warning' }
          : settlement.action === 'release'
            ? { type: 'job-released', level: 'warning' }
            : settlement.action === 'retry'
              ? { type: 'job-retry-scheduled', level: 'warning', data: error }
              : { type: 'job-failed', level: 'error', data: error }
    try {
      await this.emit({ ...event, jobId: job.id, jobAttempt: job.attempt })
    } catch (telemetryError) {
      await this.reportQueueCycleFailure(telemetryError, 'post-settlement-telemetry')
    }
  }

  createJob(input) {
    return this.createJobThroughStore(input, () => this.store.createJob(input))
  }

  createExclusiveTeacherCampaign(input) {
    if (typeof this.store.createExclusiveTeacherCampaignJob !== 'function') {
      const error = new Error('Atomik tekil 14B kampanya store primitive’i eksik.')
      error.code = 'STORE_JOB_CONTRACT_MISSING'
      throw error
    }
    return this.createJobThroughStore(input, () => this.store.createExclusiveTeacherCampaignJob(input))
  }

  createJobThroughStore(input, create) {
    const existing = this.store.findJobByIdempotencyKey(input.idempotencyKey)
    // Route every duplicate through the store so type/request-hash conflicts
    // cannot bypass its atomic idempotency contract.
    if (existing) return create()
    this.store.consumeUsage('jobs', 1)
    let created
    try {
      created = create()
    } catch (error) {
      this.store.refundUsage('jobs', 1)
      throw error
    }
    if (!created.duplicate) {
      void this.emit({ type: 'job-queued', jobId: created.job.id, data: { type: created.job.type } })
      this.schedule(0)
    } else {
      this.store.refundUsage('jobs', 1)
    }
    return created
  }

  control(jobId, action) {
    if (!['pause', 'resume', 'cancel'].includes(action)) throw new TypeError('Bilinmeyen iş kontrol eylemi.')
    const candidate = jobId
      ? { id: jobId }
      : action === 'resume'
        ? this.store.findJobByState(['paused'])
        : this.currentOrNext()
    if (!candidate) return undefined
    const transition = this.store.requestJobControlTransition(candidate.id, action)
    if (!transition) return undefined
    if (action === 'resume' && transition.changed && transition.job.state === 'queued') {
      this.store.setPipelineState({ paused: false })
      this.schedule(0)
    }
    if (action === 'cancel' && transition.changed && transition.job.state === 'cancel_requested') {
      this.active.get(candidate.id)?.controller.abort(Object.assign(new Error('İş kullanıcı tarafından iptal edildi.'), { code: 'JOB_CANCELLED' }))
    }
    return transition
  }
  pause(jobId) {
    return this.control(jobId, 'pause')?.job
  }
  resume(jobId) {
    return this.control(jobId, 'resume')?.job
  }
  cancel(jobId) {
    return this.control(jobId, 'cancel')?.job
  }
  currentOrNext() {
    return this.store.findJobByState(['running', 'pause_requested', 'queued', 'paused'])
  }
  async emit(event) {
    const record = this.store.addEvent(event)
    this.onEvent(record)
    return record
  }
  async emitClaimed(job, event) {
    if (typeof this.store.addClaimedEvent !== 'function') {
      const error = new Error('Claim korumalı olay günlüğü store primitive’i eksik.')
      error.code = 'STORE_CLAIM_CONTRACT_MISSING'
      throw error
    }
    const record = this.store.addClaimedEvent(
      job.id,
      this.owner,
      job.attempt,
      event,
    )
    this.onEvent(record)
    return record
  }
}

function isRetryable(error) {
  return error?.retryable === true ||
    ['OLLAMA_TEMPORARY_ERROR', 'LEASE_EXPIRED', 'SQLITE_BUSY', 'SQLITE_LOCKED'].includes(error?.code) ||
    (error?.code === 'ERR_SQLITE_ERROR' && /locked|busy/iu.test(String(error?.message ?? '')))
}

function serializeError(error) {
  return {
    code: String(error?.code ?? 'PIPELINE_FAILED'),
    message: String(error?.message ?? error).slice(0, 2_000),
    ...(error?.retryable === true ? { retryable: true } : {}),
    ...(Number.isFinite(error?.status) ? { status: Number(error.status) } : {}),
    ...(error?.resourceState ? { resourceState: structuredClone(error.resourceState) } : {}),
  }
}

function workerExecArgv(heapLimitMb) {
  const inherited = []
  for (let index = 0; index < process.execArgv.length; index += 1) {
    const argument = process.execArgv[index]
    if (['-e', '-p', '--eval', '--print'].includes(argument)) {
      index += 1
      continue
    }
    if (
      argument.startsWith('--inspect') ||
      argument.startsWith('--test') ||
      argument.startsWith('--input-type') ||
      argument.startsWith('--eval=') ||
      argument.startsWith('--print=') ||
      argument === '--watch' ||
      argument.startsWith('--max-old-space-size')
    ) continue
    inherited.push(argument)
  }
  return [...inherited, `--max-old-space-size=${heapLimitMb}`]
}

function workerFailure(code, message, diagnostic, retryable = false) {
  const suffix = String(diagnostic ?? '').trim().slice(-2_000)
  const error = new Error(suffix ? `${message} Ayrı süreç tanısı: ${suffix}` : message)
  error.code = code
  error.retryable = retryable
  return error
}

function unavailableWorkerError(reason) {
  const error = new Error(`Ağır eğitim işi ayrı süreç olmadan başlatılmadı (${reason}).`)
  error.code = 'JOB_WORKER_UNAVAILABLE'
  error.retryable = false
  return error
}

function deserializeWorkerError(value, diagnostic) {
  const error = workerFailure(
    String(value?.code ?? 'JOB_WORKER_FAILED'),
    String(value?.message ?? 'Ayrı eğitim süreci başarısız oldu.'),
    diagnostic,
    value?.retryable === true,
  )
  if (Number.isFinite(value?.status)) error.status = Number(value.status)
  if (value?.resourceState) error.resourceState = structuredClone(value.resourceState)
  return error
}

function staleLeaseError(job, current) {
  const error = new Error('Worker lease/attempt sahipliği daha yeni bir denemeye geçti.')
  error.code = 'STALE_JOB_LEASE'
  error.jobId = job.id
  error.expectedOwner = job.leaseOwner
  error.expectedAttempt = job.attempt
  error.actualOwner = current?.leaseOwner ?? null
  error.actualAttempt = current?.attempt ?? null
  return error
}
