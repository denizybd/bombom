import assert from 'node:assert/strict'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { DurableJobQueue } from '../queue.mjs'
import { createTrainingStore } from '../store.mjs'
import { temporaryDirectory, waitFor } from './helpers.mjs'

const isolatedWorkerFixture = fileURLToPath(new URL('../../runtime/test/fixtures/isolated-worker-fixture.mjs', import.meta.url))

function processExists(pid) {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    if (error?.code === 'ESRCH') return false
    throw error
  }
}

test('SQLite full-cycle work runs in a disposable process and releases its allocation on exit', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  let inlineExecutions = 0
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => {
      inlineExecutions += 1
      throw new Error('inline executor must not run')
    },
    workerIsolation: true,
    workerEntry: isolatedWorkerFixture,
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle' })
  queue.start()

  const completed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'succeeded' ? current : undefined
  })
  assert.equal(inlineExecutions, 0)
  assert.notEqual(completed.result.isolatedPid, process.pid)
  assert.equal(completed.result.allocatedBytes, 64 * 1024 * 1024)
  assert.equal(processExists(completed.result.isolatedPid), false, 'terminal settlement waits for child exit, which releases all child memory')
  assert.ok(store.listEvents().some((event) => event.type === 'job-worker-spawned' && event.jobId === job.id))
  assert.ok(store.listEvents().some((event) => event.type === 'fixture-worker-running' && event.jobId === job.id))
  assert.ok(store.listEvents().some((event) => event.type === 'job-progress' && event.progress === 0.5))
  assert.equal(completed.progress, 1)
  assert.equal(completed.stage, 'completed')
})

for (const jobType of ['plastic-region-release-evaluation', 'plastic-region-release-renewal']) {
test(`${jobType} is always dispatched to the disposable worker`, async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  let inlineExecutions = 0
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => {
      inlineExecutions += 1
      throw new Error('release evaluator must not run in the service process')
    },
    workerIsolation: true,
    workerEntry: isolatedWorkerFixture,
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: jobType })
  queue.start()
  const completed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'succeeded' ? current : undefined
  })
  assert.equal(inlineExecutions, 0)
  assert.notEqual(completed.result.isolatedPid, process.pid)
  assert.equal(processExists(completed.result.isolatedPid), false)
  assert.ok(store.listEvents().some((event) => event.type === 'job-worker-spawned' && event.jobId === job.id))
})
}

test('isolated release worker atomically publishes a candidate and parent performs telemetry-only settlement', async (t) => {
  const previousMode = process.env.BOMBOM_ISOLATION_FIXTURE_MODE
  process.env.BOMBOM_ISOLATION_FIXTURE_MODE = 'atomic-candidate'
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  store.updateConfig({ automation: { leaseMs: 5_000 } })
  const active = store.createSnapshot({
    id: 'fixture-atomic-active', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(active.id)
  let inlineExecutions = 0
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => {
      inlineExecutions += 1
      throw new Error('atomic release fixture must run in the isolated worker')
    },
    workerIsolation: true,
    workerEntry: isolatedWorkerFixture,
  })
  t.after(async () => {
    if (previousMode === undefined) delete process.env.BOMBOM_ISOLATION_FIXTURE_MODE
    else process.env.BOMBOM_ISOLATION_FIXTURE_MODE = previousMode
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'plastic-region-release-renewal' })
  queue.start()
  const completed = await waitFor(() => {
    const current = store.getJob(job.id)
    const terminalEvent = store.listEvents().some((event) => event.type === 'job-succeeded' && event.jobId === job.id)
    return current.state === 'succeeded' && terminalEvent ? current : undefined
  })
  assert.equal(inlineExecutions, 0)
  assert.equal(completed.result.atomicCandidateJobCompletion, undefined, 'internal settlement marker must not persist')
  assert.equal(completed.result.automaticActivation, false)
  const candidate = store.getSnapshot(completed.result.snapshotId)
  assert.equal(candidate.status, 'candidate')
  assert.equal(candidate.producerJobId, job.id)
  assert.equal(store.getActiveSnapshot().id, active.id)
  assert.ok(store.listEvents().some((event) => event.type === 'job-succeeded' && event.jobId === job.id))
})

test('cancel propagates into the isolated worker and settlement waits for process teardown', async (t) => {
  const previousMode = process.env.BOMBOM_ISOLATION_FIXTURE_MODE
  process.env.BOMBOM_ISOLATION_FIXTURE_MODE = 'hang'
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => { throw new Error('inline executor must not run') },
    workerIsolation: true,
    workerEntry: isolatedWorkerFixture,
  })
  t.after(async () => {
    if (previousMode === undefined) delete process.env.BOMBOM_ISOLATION_FIXTURE_MODE
    else process.env.BOMBOM_ISOLATION_FIXTURE_MODE = previousMode
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle' })
  queue.start()
  const runningEvent = await waitFor(() => store.listEvents().find((event) => event.type === 'fixture-worker-running'))
  assert.equal(queue.cancel(job.id).state, 'cancel_requested')
  const cancelled = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'cancelled' ? current : undefined
  })
  assert.equal(cancelled.result, null)
  assert.equal(processExists(runningEvent.data.pid), false)
})

test('an unavailable isolation entry falls back explicitly before any child work starts', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  let inlineExecutions = 0
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => ({ inlineFallback: ++inlineExecutions }),
    workerIsolation: 'fallback',
    workerEntry: `${dataDirectory}/missing-worker.mjs`,
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle' })
  queue.start()
  const completed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'succeeded' ? current : undefined
  })
  assert.equal(completed.result.inlineFallback, 1)
  const fallback = store.listEvents().find((event) => event.type === 'job-worker-fallback')
  assert.equal(fallback.data.reason, 'worker-entry-missing')
  assert.equal(fallback.data.allowed, true)
})

test('production isolation pauses instead of retaining a heavy job in the parent process', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  let inlineExecutions = 0
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => ({ unsafe: ++inlineExecutions }),
    workerIsolation: true,
    workerEntry: `${dataDirectory}/missing-production-worker.mjs`,
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle' })
  queue.start()
  const paused = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'paused' ? current : undefined
  })
  assert.equal(inlineExecutions, 0)
  assert.equal(paused.error.code, 'JOB_WORKER_UNAVAILABLE')
  const diagnostic = store.listEvents().find((event) => event.type === 'job-worker-fallback')
  assert.equal(diagnostic.data.allowed, false)
})

for (const code of ['DISK_BUDGET_EXCEEDED', 'ARTIFACT_PAYLOAD_TOO_LARGE', 'NEURON_RESIDENCY_RESOURCE_WAIT']) {
  test(`${code} leaves a durable resumable pause instead of spending another heavy attempt`, async (t) => {
    const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
    const queue = new DurableJobQueue({
      store,
      ollama: {},
      executor: async () => { throw Object.assign(new Error(code), { code }) },
    })
    t.after(async () => {
      await queue.stop({ abort: true })
      store.close()
    })
    const { job } = store.createJob({ type: 'teacher-feed-campaign', maxAttempts: 3 })
    queue.start()
    const paused = await waitFor(() => {
      const current = store.getJob(job.id)
      return current.state === 'paused' ? current : undefined
    })
    assert.equal(paused.attempt, 1)
    assert.equal(paused.error.code, code)
  })
}

test('queue recovers a lease that expires after service startup', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => ({ recovered: true }),
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })

  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 3 })
  store.claimNextJob('dead-worker', 30_000)
  queue.start()
  assert.equal(store.getJob(job.id).state, 'running', 'startup must respect a lease that has not expired')

  store.updateJob(job.id, {
    leaseUntil: new Date(Date.now() - 1).toISOString(),
    executionLeaseUntil: new Date(Date.now() - 1).toISOString(),
  })
  queue.nextRecoveryAt = 0
  await queue.tick()

  const completed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'succeeded' ? current : undefined
  })
  assert.equal(completed.result.recovered, true)
  assert.equal(completed.attempt, 2)
})

test('a transient SQLite busy tick is recorded and polling continues', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => ({ afterBusyRetry: true }),
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle' })
  const originalClaim = store.claimNextJob.bind(store)
  let failOnce = true
  store.claimNextJob = (...args) => {
    if (failOnce) {
      failOnce = false
      const error = new Error('database is locked')
      error.code = 'SQLITE_BUSY'
      throw error
    }
    return originalClaim(...args)
  }

  queue.start()
  const completed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'succeeded' ? current : undefined
  })
  assert.equal(completed.result.afterBusyRetry, true)
  const diagnostic = store.listEvents().find((event) => event.type === 'queue-cycle-failed')
  assert.equal(diagnostic.stage, 'tick')
  assert.equal(diagnostic.data.code, 'SQLITE_BUSY')
})

test('a transient SQLite busy lease renewal is recorded without killing the worker', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  store.updateConfig({ automation: { leaseMs: 5_000 } })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => {
      await new Promise((resolve) => setTimeout(resolve, 1_900))
      return { survivedBusyRenewal: true }
    },
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle' })
  const originalRenewLease = store.renewLease.bind(store)
  let failOnce = true
  store.renewLease = (...args) => {
    if (failOnce) {
      failOnce = false
      const error = new Error('database is locked')
      error.code = 'SQLITE_BUSY'
      throw error
    }
    return originalRenewLease(...args)
  }

  queue.start()
  const completed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'succeeded' ? current : undefined
  })

  assert.equal(completed.result.survivedBusyRenewal, true)
  const diagnostic = store.listEvents().find((event) => event.type === 'queue-cycle-failed' && event.stage === 'lease-renewal')
  assert.equal(diagnostic.data.code, 'SQLITE_BUSY')
})

test('explicit resume recovers only a checkpointed research-only teacher campaign after a resource failure', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => ({ recovered: true }),
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })

  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload: {
      campaign: { mode: 'research-only', proposalOnly: true },
      feedState: {
        pendingGenerationBatch: null,
        pendingBatchAccounting: null,
        pendingTrainingMilestone: {
          milestoneIndex: 3,
          sourceIds: ['source-a'],
          candidateSnapshotId: 'candidate-3',
        },
      },
    },
  })
  store.updateJob(job.id, {
    state: 'failed',
    stage: 'failed',
    attempt: 2,
    finishedAt: new Date().toISOString(),
    error: { code: 'RAM_BUDGET_EXCEEDED', message: 'limit' },
  })

  const resumed = queue.resume(job.id)
  assert.equal(resumed.state, 'queued')
  assert.equal(resumed.stage, 'resource-recovery-queued')
  assert.equal(resumed.error, null)
  assert.equal(resumed.finishedAt, null)
  assert.equal(resumed.maxAttempts, 3)
  assert.equal(queue.resume(job.id).maxAttempts, 3, 'queued işe ikinci resume yeni attempt hakkı eklememeli')

  const leaseExpired = store.createJob({
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload: structuredClone(job.payload),
  }).job
  store.updateJob(leaseExpired.id, {
    state: 'failed', attempt: 2,
    error: { code: 'LEASE_EXPIRED', message: 'owner lost' },
  })
  assert.equal(queue.resume(leaseExpired.id).state, 'queued')
  assert.equal(store.getJob(leaseExpired.id).maxAttempts, 3)

  const workerOom = store.createJob({
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload: structuredClone(job.payload),
  }).job
  store.updateJob(workerOom.id, {
    state: 'failed', attempt: 2,
    error: { code: 'JOB_WORKER_EXITED', message: 'V8 OOM', retryable: true },
  })
  assert.equal(queue.resume(workerOom.id).state, 'queued')

  const nonRetryableExit = store.createJob({
    type: 'teacher-feed-campaign',
    payload: structuredClone(job.payload),
  }).job
  store.updateJob(nonRetryableExit.id, {
    state: 'failed',
    error: { code: 'JOB_WORKER_EXITED', message: 'protocol failure', retryable: false },
  })
  assert.equal(queue.resume(nonRetryableExit.id).state, 'failed')

  const unsafe = store.createJob({ type: 'full-cycle' }).job
  store.updateJob(unsafe.id, {
    state: 'failed',
    error: { code: 'RAM_BUDGET_EXCEEDED', message: 'limit' },
  })
  assert.equal(queue.resume(unsafe.id).state, 'failed')

  const wrongFailure = store.createJob({
    type: 'teacher-feed-campaign',
    payload: structuredClone(job.payload),
  }).job
  store.updateJob(wrongFailure.id, {
    state: 'failed',
    error: { code: 'SNAPSHOT_IDEMPOTENCY_CONFLICT', message: 'not a resource failure' },
  })
  assert.equal(queue.resume(wrongFailure.id).state, 'failed')

  const noCheckpoint = store.createJob({
    type: 'teacher-feed-campaign',
    payload: {
      campaign: { mode: 'research-only', proposalOnly: true },
      feedState: { pendingTrainingMilestone: null },
    },
  }).job
  store.updateJob(noCheckpoint.id, {
    state: 'failed',
    error: { code: 'LEASE_EXPIRED', message: 'owner lost' },
  })
  assert.equal(queue.resume(noCheckpoint.id).state, 'failed')
})

test('explicit resume releases only a dead heartbeat at a clean continuous benchmark boundary', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const queue = new DurableJobQueue({ store, ollama: {}, executor: async () => ({ recovered: true }) })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })

  const createInterrupted = (overrides = {}) => {
    const { job } = store.createJob({
      type: 'teacher-feed-campaign',
      maxAttempts: 2,
      payload: {
        campaign: {
          mode: 'research-only', proposalOnly: true,
          continuousPlan: { blockAccepted: 64, maxBlocks: 32 },
        },
        feedState: {
          accepted: 64,
          lastTrainedAccepted: 64,
          lastSnapshotId: 'snapshot-at-clean-boundary',
          pendingGenerationBatch: null,
          pendingBatchAccounting: null,
          pendingTrainingMilestone: null,
        },
      },
    })
    store.updateJob(job.id, {
      state: 'running',
      stage: 'teacher-feed:benchmark:full',
      attempt: 2,
      leaseOwner: 'worker-before-reboot',
      leaseUntil: new Date(Date.now() - 1_000).toISOString(),
      executionLeaseUntil: new Date(Date.now() + 60_000).toISOString(),
      error: { code: 'JOB_WORKER_EXITED', message: 'worker exited', retryable: true },
      ...overrides,
    })
    return job.id
  }

  const recoverableId = createInterrupted()
  const resumed = queue.resume(recoverableId)
  assert.equal(resumed.state, 'queued')
  assert.equal(resumed.stage, 'interrupted-benchmark-recovery-queued')
  assert.equal(resumed.error, null)
  assert.equal(resumed.leaseOwner, null)
  assert.equal(resumed.executionLeaseUntil, null)
  assert.equal(resumed.maxAttempts, 3)

  const liveHeartbeatId = createInterrupted({ leaseUntil: new Date(Date.now() + 60_000).toISOString() })
  assert.equal(queue.resume(liveHeartbeatId).state, 'running', 'canlı heartbeat taşıyan iş çalınmamalı')

  const ambiguousCallId = createInterrupted({
    payload: {
      campaign: {
        mode: 'research-only', proposalOnly: true,
        continuousPlan: { blockAccepted: 64, maxBlocks: 32 },
      },
      feedState: {
        accepted: 64,
        lastTrainedAccepted: 64,
        lastSnapshotId: 'snapshot-at-clean-boundary',
        pendingGenerationBatch: { inFlightAttempt: { requestId: 'paid-call' } },
        pendingBatchAccounting: null,
        pendingTrainingMilestone: null,
      },
    },
  })
  assert.equal(queue.resume(ambiguousCallId).state, 'running', 'açık öğretmen çağrısı sessizce yeniden oynatılmamalı')
})

test('explicit resume safely accounts and skips an interrupted continuous teacher generation call', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const queue = new DurableJobQueue({ store, ollama: {}, executor: async () => ({ recovered: true }) })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })

  const createInterrupted = (leaseUntil) => {
    const { job } = store.createJob({
      type: 'teacher-feed-campaign',
      maxAttempts: 2,
      payload: {
        campaign: {
          mode: 'research-only', proposalOnly: true,
          continuousPlan: { blockAccepted: 64, maxBlocks: 32 },
        },
        feedState: {
          promptTokens: 10,
          completionTokens: 5,
          teacherTokens: 15,
          pendingGenerationBatch: {
            version: 1,
            promptTokens: 0,
            completionTokens: 0,
            generationErrors: 0,
            inFlightAttempt: {
              attemptIndex: 0,
              requestUsageReserved: true,
              tokenUsageReserved: true,
              maxPromptTokens: 4096,
              maxCompletionTokens: 128,
              contextTokenLimit: 4096,
              promptBoundPolicy: 'ollama-num-ctx-v1',
              usageDay: '2026-08-30',
            },
          },
          pendingBatchAccounting: null,
          pendingTrainingMilestone: null,
        },
      },
    })
    store.updateJob(job.id, {
      state: 'running',
      stage: 'teacher-feed:generating',
      attempt: 2,
      leaseOwner: 'worker-before-reboot',
      leaseUntil,
      executionLeaseUntil: new Date(Date.now() + 60_000).toISOString(),
    })
    return job.id
  }

  const recoverableId = createInterrupted(new Date(Date.now() - 1_000).toISOString())
  const resumed = queue.resume(recoverableId)
  assert.equal(resumed.state, 'queued')
  assert.equal(resumed.stage, 'interrupted-generation-recovery-queued')
  assert.equal(resumed.leaseOwner, null)
  assert.equal(resumed.executionLeaseUntil, null)
  assert.equal(resumed.maxAttempts, 3)
  assert.equal(resumed.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(resumed.payload.feedState.pendingGenerationBatch.generationErrors, 1)
  assert.equal(resumed.payload.feedState.pendingGenerationBatch.usageEstimated, true)
  assert.equal(resumed.payload.feedState.pendingGenerationBatch.promptTokens, 4096)
  assert.equal(resumed.payload.feedState.pendingGenerationBatch.completionTokens, 128)
  assert.equal(resumed.payload.feedState.teacherTokens, 4239)

  const liveHeartbeatId = createInterrupted(new Date(Date.now() + 60_000).toISOString())
  const live = queue.resume(liveHeartbeatId)
  assert.equal(live.state, 'running', 'canlı heartbeat taşıyan öğretmen çağrısı çalınmamalı')
  assert.notEqual(live.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
})

test('bounded execution lease protects a CPU-bound attempt until its max-run window', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  store.updateConfig({ automation: { leaseMs: 5_000 }, budgets: { maxRunMinutes: 5 } })
  const queue = new DurableJobQueue({ store, ollama: {}, executor: async () => ({}) })
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 2 })
  const config = store.getConfig()
  const claimed = store.claimNextJob(queue.owner, config.automation.leaseMs, queue.executionLeaseMs(config))
  const attemptStarted = Date.parse(claimed.attemptStartedAt)
  const executionDeadline = Date.parse(claimed.executionLeaseUntil)
  assert.ok(executionDeadline - attemptStarted >= 6 * 60_000)

  store.updateJob(job.id, { leaseUntil: new Date(Date.now() - 1).toISOString() })
  assert.equal(store.recoverExpiredJobs(), 0, 'heartbeat starvation alone must not recover a CPU-bound live attempt')
  assert.equal(store.getJob(job.id).state, 'running')

  store.updateJob(job.id, { executionLeaseUntil: new Date(Date.now() - 1).toISOString() })
  assert.equal(store.recoverExpiredJobs(), 1)
  assert.equal(store.getJob(job.id).state, 'queued')
})

test('350ms queue admission uses the compact queued-job index instead of listing job payloads', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async () => ({ indexedAdmission: true }),
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({
    type: 'full-cycle',
    payload: { historicalDiagnostics: 'x'.repeat(512 * 1024) },
  })
  const originalListJobs = store.listJobs.bind(store)
  store.listJobs = () => { throw new Error('queue performed a full job scan') }
  try {
    queue.start()
    const completed = await waitFor(() => {
      const current = store.getJob(job.id)
      return current.state === 'succeeded' ? current : undefined
    })
    assert.equal(completed.result.indexedAdmission, true)
  } finally {
    store.listJobs = originalListJobs
  }
})

test('atomic completion rechecks the live controlled policy after inference', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  store.updateConfig({ automation: { level: 'controlled' } })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async (job, context) => {
      const candidate = context.store.createClaimedSnapshot(
        job.id,
        context.jobClaim.owner,
        context.jobClaim.attempt,
        { status: 'candidate', evaluation: { passed: true, runtimeVerified: true } },
      )
      // This simulates an operator changing the live policy after the executor's
      // preliminary decision but before the combined job/snapshot transaction.
      context.store.updateConfig({ automation: { level: 'proposal' } })
      return { snapshotId: candidate.id, activationSnapshotId: candidate.id }
    },
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 1 })
  queue.start()
  const failed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'failed' ? current : undefined
  })
  assert.equal(failed.error.code, 'SNAPSHOT_ACTIVATION_POLICY_CHANGED')
  assert.equal(store.getActiveSnapshot(), undefined)
  assert.equal(store.listSnapshots()[0].status, 'candidate')
})

test('atomic completion rejects a bare adaptive candidate even when an executor requests activation', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  store.updateConfig({ automation: { level: 'controlled' } })
  const representationProof = store.createArtifact({
    kind: 'text-lab-analysis',
    payload: { metadata: { representationProfile: 'adaptive-eighth-id-v1' } },
  })
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async (job, context) => {
      const candidate = context.store.createClaimedSnapshot(
        job.id,
        context.jobClaim.owner,
        context.jobClaim.attempt,
        {
          status: 'candidate',
          artifactIds: [representationProof.id],
          evaluation: { passed: true, runtimeVerified: true },
        },
      )
      return { snapshotId: candidate.id, activationSnapshotId: candidate.id }
    },
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 1 })
  queue.start()
  const failed = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'failed' ? current : undefined
  })
  assert.equal(failed.error.code, 'PLASTIC_REGION_RELEASE_REQUIRED')
  assert.equal(store.getActiveSnapshot(), undefined)
  assert.equal(store.listSnapshots()[0].status, 'candidate')
})

for (const operation of ['checkpoint', 'progress', 'terminal']) {
  test(`stale worker cannot overwrite a replacement attempt through ${operation}`, async (t) => {
    const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
    let release
    let entered
    const enteredPromise = new Promise((resolve) => { entered = resolve })
    const releasePromise = new Promise((resolve) => { release = resolve })
    const queue = new DurableJobQueue({
      store,
      ollama: {},
      executor: async (_job, context) => {
        entered()
        await releasePromise
        if (operation === 'checkpoint') await context.checkpoint('stale-checkpoint')
        if (operation === 'progress') await context.progress('stale-progress', 0.5)
        return { staleWorkerMustNotCommit: true }
      },
    })
    t.after(async () => {
      await queue.stop({ abort: true })
      store.close()
    })
    const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 3 })
    queue.start()
    await enteredPromise
    const first = store.getJob(job.id)
    assert.equal(first.attempt, 1)

    store.updateJob(job.id, {
      leaseUntil: new Date(Date.now() - 1).toISOString(),
      executionLeaseUntil: new Date(Date.now() - 1).toISOString(),
    })
    assert.equal(store.recoverExpiredJobs(), 1)
    const replacement = store.claimNextJob('replacement-worker', 30_000, 60_000)
    assert.equal(replacement.attempt, 2)
    release()
    await waitFor(() => queue.active.size === 0 ? true : undefined)

    const current = store.getJob(job.id)
    assert.equal(current.state, 'running')
    assert.equal(current.attempt, 2)
    assert.equal(current.leaseOwner, 'replacement-worker')
    assert.equal(current.stage, 'starting')
    assert.equal(current.result, null)
  })
}

test('cancel between controlled inference and atomic completion leaves model inactive', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const sourceInput = {
    id: 'controlled-source',
    title: 'Controlled kaynak',
    content: 'Controlled aktivasyon yarışını sınayan kaynak.',
    contentHash: 'controlled-source-hash',
    bytes: 48,
    language: 'tr',
    sourceType: 'manual',
    status: 'approved',
    selected: true,
    provenance: {},
  }
  store.backend.put('sources', sourceInput.id, sourceInput)
  const queue = new DurableJobQueue({
    store,
    ollama: {},
    executor: async (job, context) => {
      const snapshot = context.store.createClaimedSnapshot(
        job.id,
        context.jobClaim.owner,
        context.jobClaim.attempt,
        {
          status: 'candidate',
          sourceIds: [sourceInput.id],
          evaluation: { passed: true, runtimeVerified: true },
        },
      )
      return {
        snapshotId: snapshot.id,
        activationSnapshotId: snapshot.id,
        committed: false,
      }
    },
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    store.close()
  })
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 2 })
  const originalComplete = store.completeClaimedJobWithSnapshot.bind(store)
  let cancellationInjected = false
  store.completeClaimedJobWithSnapshot = (...args) => {
    if (!cancellationInjected) {
      cancellationInjected = true
      store.updateJob(job.id, { state: 'cancel_requested', stage: 'cancel-requested' })
    }
    return originalComplete(...args)
  }
  queue.start()
  const cancelled = await waitFor(() => {
    const current = store.getJob(job.id)
    return current.state === 'cancelled' ? current : undefined
  })

  assert.equal(cancelled.result, null)
  assert.equal(store.getActiveSnapshot(), undefined)
  assert.equal(store.listSnapshots().length, 1)
  assert.equal(store.listSnapshots()[0].status, 'candidate')
  assert.equal(store.getSource(sourceInput.id).processedSnapshotId ?? null, null)
})

test('a fresh cancel request wins atomically over ordinary successful completion', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const workerStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const controlStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const queue = new DurableJobQueue({
    store: workerStore,
    ollama: {},
    executor: async () => ({ shouldNotCommit: true }),
  })
  t.after(async () => {
    await queue.stop({ abort: true })
    workerStore.close()
    controlStore.close()
  })
  const { job } = workerStore.createJob({ type: 'full-cycle', maxAttempts: 2 })
  const originalSettlement = workerStore.settleClaimedJob.bind(workerStore)
  let injected = false
  workerStore.settleClaimedJob = (...args) => {
    if (!injected) {
      injected = true
      assert.equal(controlStore.requestJobControl(job.id, 'cancel').state, 'cancel_requested')
    }
    return originalSettlement(...args)
  }

  queue.start()
  const cancelled = await waitFor(() => {
    const current = workerStore.getJob(job.id)
    return current.state === 'cancelled' ? current : undefined
  })
  assert.equal(cancelled.result, null)
  assert.equal(workerStore.listEvents().some((event) => event.type === 'job-succeeded' && event.jobId === job.id), false)
  assert.equal(workerStore.listEvents().some((event) => event.type === 'job-cancelled' && event.jobId === job.id), true)
})
