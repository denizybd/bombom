import test from 'node:test'
import assert from 'node:assert/strict'
import { once } from 'node:events'
import { Worker } from 'node:worker_threads'
import path from 'node:path'
import {
  assertSingleArtifactPayloadBytes,
  createTrainingStore,
  MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES,
} from '../store.mjs'
import { normalizeTextSource } from '../source-policy.mjs'
import { temporaryDirectory } from './helpers.mjs'

test('store persists records, deduplicates content, and keeps immutable snapshot history', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  let store = await createTrainingStore({ dataDirectory, forceJson: true })
  const sourceInput = normalizeTextSource({ text: 'Ali bıçakla ekmeği kesti.', title: 'Deney' }, store.getConfig())
  const first = store.createSource(sourceInput)
  const duplicate = store.createSource({ ...sourceInput, title: 'Başka ad' })
  assert.equal(first.duplicate, false)
  assert.equal(duplicate.duplicate, true)
  assert.equal(first.source.id, duplicate.source.id)
  store.decideSource(first.source.id, 'approved', 'kontrollü veri')
  const artifact = store.createArtifact({ kind: 'test-model', payload: { weights: [1, 2] } })
  const original = store.createSnapshot({ artifactIds: [artifact.id], sourceIds: [first.source.id], metrics: { overallScore: 0.7 }, evaluation: { passed: true, runtimeVerified: true } })
  store.activateSnapshot(original.id)
  const newer = store.createSnapshot({ metrics: { overallScore: 0.8 }, evaluation: { passed: true, runtimeVerified: true } })
  store.activateSnapshot(newer.id)
  const restored = store.rollbackToSnapshot(original.id)
  assert.equal(restored.status, 'active')
  assert.notEqual(restored.id, original.id)
  assert.deepEqual(restored.artifactIds, [artifact.id])
  store.close()

  store = await createTrainingStore({ dataDirectory, forceJson: true })
  assert.equal(store.getActiveSnapshot().id, restored.id)
  assert.equal(store.listSources({ status: 'approved' }).length, 1)
  store.close()
})

test('v3 workstation budgets migrate through v6 RAM/SSD, concept and plastic-region policy without deleting records', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const first = await createTrainingStore({ dataDirectory, forceJson: true })
  first.createSource({
    title: 'Korunan kayıt', sourceType: 'manual-text', content: 'kalıcı veri',
    contentHash: 'migration-source-proof', bytes: 11,
  })
  const legacy = first.getConfig()
  legacy.version = 3
  legacy.budgets.ramMb = 8_192
  legacy.budgets.diskMb = 20_480
  delete legacy.resourcePolicy
  first.backend.setMeta('config', legacy)
  first.close()

  const migrated = await createTrainingStore({ dataDirectory, forceJson: true })
  t.after(() => migrated.close())
  assert.equal(migrated.getConfig().version, 6)
  assert.equal(migrated.getConfig().budgets.ramMb, 4_096)
  assert.equal(migrated.getConfig().budgets.diskMb, 4_096)
  assert.equal(migrated.getConfig().resourcePolicy.neuronCacheMb, 384)
  assert.equal(migrated.getConfig().conceptMacros.mode, 'proposal-shadow')
  assert.equal(migrated.getConfig().plasticRegions.mode, 'off')
  assert.equal(migrated.getConfig().plasticRegions.representationProfile, 'legacy')
  assert.equal(migrated.listSources({ includeContent: true })[0].content, 'kalıcı veri')
})

test('single-artifact write and read limits are identical so every accepted payload is decodable', () => {
  assert.equal(assertSingleArtifactPayloadBytes(MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES), MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES)
  assert.throws(
    () => assertSingleArtifactPayloadBytes(MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES + 1),
    (error) => error?.code === 'ARTIFACT_PAYLOAD_TOO_LARGE' &&
      error.maximumPayloadBytes === MAX_SINGLE_ARTIFACT_PAYLOAD_BYTES,
  )
})

test('expired durable jobs are recovered without losing attempts', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 2 })
  const claimed = store.claimNextJob('dead-worker', 5_000)
  assert.equal(claimed.id, job.id)
  store.updateJob(job.id, {
    leaseUntil: new Date(Date.now() - 1_000).toISOString(),
    executionLeaseUntil: new Date(Date.now() - 1_000).toISOString(),
  })
  assert.equal(store.recoverExpiredJobs(), 1)
  assert.equal(store.getJob(job.id).state, 'queued')
  assert.equal(store.getJob(job.id).attempt, 1)
})

test('job idempotency rejects changed request hashes and cross-type key reuse while preserving legacy retries', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())

  const first = store.createJob({
    type: 'teacher-feed-campaign', idempotencyKey: 'hashed-campaign', requestHash: 'request-a',
  })
  const repeated = store.createJob({
    type: 'teacher-feed-campaign', idempotencyKey: 'hashed-campaign', requestHash: 'request-a',
  })
  assert.equal(repeated.duplicate, true)
  assert.equal(repeated.job.id, first.job.id)
  assert.throws(
    () => store.createJob({ type: 'teacher-feed-campaign', idempotencyKey: 'hashed-campaign', requestHash: 'request-b' }),
    (error) => error?.code === 'JOB_IDEMPOTENCY_CONFLICT' && error?.status === 409 && error?.details?.reason === 'request-hash',
  )
  assert.throws(
    () => store.createJob({ type: 'full-cycle', idempotencyKey: 'hashed-campaign', requestHash: 'request-a' }),
    (error) => error?.code === 'JOB_IDEMPOTENCY_CONFLICT' && error?.details?.reason === 'job-type',
  )

  const legacy = store.createJob({ type: 'teacher-feed-campaign', idempotencyKey: 'legacy-campaign' })
  const legacyRetry = store.createJob({
    type: 'teacher-feed-campaign', idempotencyKey: 'legacy-campaign', requestHash: 'newly-known-request',
  })
  assert.equal(legacyRetry.duplicate, true)
  assert.equal(legacyRetry.job.id, legacy.job.id)
})

for (const backend of ['atomic-json', 'sqlite']) {
  test(`${backend} job indexes serve polling, state/type and idempotency lookups without a full job scan`, async (t) => {
    const store = await createTrainingStore({
      dataDirectory: await temporaryDirectory(t),
      ...(backend === 'atomic-json' ? { forceJson: true } : { requireSqlite: true }),
    })
    t.after(() => store.close())
    const terminal = store.createJob({
      id: `${backend}-terminal-job`,
      type: 'full-cycle',
      idempotencyKey: `${backend}-terminal-key`,
      payload: { diagnostic: 'x'.repeat(512 * 1024) },
      state: 'succeeded',
    }).job
    const queued = store.createJob({
      id: `${backend}-queued-job`,
      type: 'teacher-feed-campaign',
      idempotencyKey: `${backend}-queued-key`,
    }).job

    const originalList = store.backend.list.bind(store.backend)
    const originalGet = store.backend.get.bind(store.backend)
    let hydratedJobs = 0
    store.backend.list = (bucket) => {
      if (bucket === 'jobs') throw new Error('full job scan invoked')
      return originalList(bucket)
    }
    store.backend.get = (bucket, id) => {
      if (bucket === 'jobs') hydratedJobs += 1
      return originalGet(bucket, id)
    }
    try {
      assert.equal(store.hasQueuedJob(), true)
      assert.equal(hydratedJobs, 0, 'queue admission reads only the compact state index')
      assert.equal(store.findJobByIdempotencyKey(`${backend}-terminal-key`).id, terminal.id)
      assert.equal(store.findJobByState(['queued'], { types: ['teacher-feed-campaign'] }).id, queued.id)
      const claimed = store.claimNextJob('indexed-worker', 30_000, 60_000)
      assert.equal(claimed.id, queued.id)
      assert.equal(store.hasQueuedJob(), false)
      assert.equal(store.findJobByState(['running'], { types: ['teacher-feed-campaign'] }).id, queued.id)
      assert.ok(hydratedJobs <= 5, `indexed operations hydrated ${hydratedJobs} job rows`)
    } finally {
      store.backend.list = originalList
      store.backend.get = originalGet
    }

    assert.throws(() => store.backend.transaction(() => {
      const running = store.backend.get('jobs', queued.id)
      running.state = 'failed'
      store.backend.put('jobs', running.id, running)
      throw new Error('job metadata rollback proof')
    }), /job metadata rollback proof/u)
    assert.equal(store.findJobByState(['running'], { types: ['teacher-feed-campaign'] }).id, queued.id)
    assert.equal(store.findJobByState(['failed'], { types: ['teacher-feed-campaign'] }), undefined)

    if (backend === 'sqlite') {
      const statePlan = store.backend.db.prepare(
        "EXPLAIN QUERY PLAN SELECT id FROM job_metadata WHERE state = 'queued' ORDER BY created_at ASC, id ASC LIMIT 1",
      ).all()
      const idempotencyPlan = store.backend.db.prepare(
        'EXPLAIN QUERY PLAN SELECT id FROM job_metadata WHERE idempotency_key = ? ORDER BY created_at DESC, id DESC LIMIT 1',
      ).all(`${backend}-terminal-key`)
      assert.match(statePlan.map((row) => row.detail).join('\n'), /job_metadata_state_created/u)
      assert.match(idempotencyPlan.map((row) => row.detail).join('\n'), /job_metadata_idempotency_created/u)
    }
  })
}

test('SQLite startup backfills the compact job index for legacy job rows', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  let store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const { job } = store.createJob({
    id: 'legacy-indexed-job',
    type: 'teacher-feed-campaign',
    idempotencyKey: 'legacy-indexed-key',
  })
  store.close()

  const { DatabaseSync } = await import('node:sqlite')
  const database = new DatabaseSync(path.join(dataDirectory, 'autonomous-training.sqlite'))
  database.exec('DELETE FROM job_metadata')
  database.close()

  store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => store.close())
  const originalList = store.backend.list.bind(store.backend)
  store.backend.list = (bucket) => {
    if (bucket === 'jobs') throw new Error('legacy backfill fell back to full job scan')
    return originalList(bucket)
  }
  try {
    assert.equal(store.hasQueuedJob(), true)
    assert.equal(store.findJobByIdempotencyKey('legacy-indexed-key').id, job.id)
    assert.equal(store.findJobByState(['queued'], { types: ['teacher-feed-campaign'] }).id, job.id)
  } finally {
    store.backend.list = originalList
  }
})

test('atomic JSON restart rebuilds compact job indexes from existing records', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  let store = await createTrainingStore({ dataDirectory, forceJson: true })
  const { job } = store.createJob({
    id: 'json-restart-indexed-job',
    type: 'teacher-feed-campaign',
    idempotencyKey: 'json-restart-indexed-key',
  })
  store.close()

  store = await createTrainingStore({ dataDirectory, forceJson: true })
  t.after(() => store.close())
  const originalList = store.backend.list.bind(store.backend)
  store.backend.list = (bucket) => {
    if (bucket === 'jobs') throw new Error('JSON restart fell back to full job scan')
    return originalList(bucket)
  }
  try {
    assert.equal(store.hasQueuedJob(), true)
    assert.equal(store.findJobByIdempotencyKey('json-restart-indexed-key').id, job.id)
  } finally {
    store.backend.list = originalList
  }
})

test('expired cancelled teacher call is conservatively charged exactly once', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const usageDay = new Date().toISOString().slice(0, 10)
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload: {
      feedState: {
        promptTokens: 2,
        completionTokens: 3,
        teacherTokens: 5,
        pendingGenerationBatch: {
          promptTokens: 2,
          completionTokens: 3,
          generationErrors: 0,
          usageEstimated: false,
          lastError: null,
          inFlightAttempt: {
            requestUsageReserved: true,
            maxPromptTokens: 11,
            maxCompletionTokens: 17,
            usageDay,
          },
        },
      },
    },
  })
  store.claimNextJob('expired-teacher-worker', 30_000, 60_000)
  store.updateJob(job.id, {
    state: 'cancel_requested',
    leaseUntil: new Date(Date.now() - 1).toISOString(),
    executionLeaseUntil: new Date(Date.now() - 1).toISOString(),
  })

  assert.equal(store.recoverExpiredJobs(), 1)
  assert.equal(store.recoverExpiredJobs(), 0)
  const current = store.getJob(job.id)
  assert.equal(current.state, 'cancelled')
  assert.equal(current.payload.feedState.teacherTokens, 32_790)
  assert.equal(current.payload.feedState.promptTokens, 32_770)
  assert.equal(current.payload.feedState.completionTokens, 20)
  assert.equal(current.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(current.payload.feedState.pendingGenerationBatch.usageEstimated, true)
  assert.equal(current.payload.feedState.pendingGenerationBatch.generationErrors, 1)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
})

test('atomic job control cannot overwrite a concurrent teacher checkpoint', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const controlStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const workerStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => {
    controlStore.close()
    workerStore.close()
  })
  const initialFeedState = {
    promptTokens: 0,
    completionTokens: 0,
    teacherTokens: 0,
    pendingGenerationBatch: {
      promptTokens: 0,
      completionTokens: 0,
      generationErrors: 0,
      usageEstimated: false,
      inFlightAttempt: null,
    },
  }
  const { job: queued } = workerStore.createJob({
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload: { campaign: { id: 'control-race' }, feedState: initialFeedState },
  })
  const claimed = workerStore.claimNextJob('teacher-control-worker', 30_000, 60_000)
  assert.equal(claimed.id, queued.id)
  const reservedFeedState = {
    ...initialFeedState,
    pendingGenerationBatch: {
      ...initialFeedState.pendingGenerationBatch,
      inFlightAttempt: {
        requestUsageReserved: true,
        maxPromptTokens: 2,
        maxCompletionTokens: 3,
      },
    },
  }
  workerStore.updateClaimedJobPayload(claimed.id, claimed.leaseOwner, claimed.attempt, {
    feedState: reservedFeedState,
  })
  const checkpoint = {
    ...reservedFeedState,
    promptTokens: 2,
    completionTokens: 3,
    teacherTokens: 5,
    pendingGenerationBatch: {
      ...reservedFeedState.pendingGenerationBatch,
      promptTokens: 2,
      completionTokens: 3,
      inFlightAttempt: null,
    },
  }
  const commit = (phase, allowRequestedUsageFinalization = false) => workerStore.commitTeacherFeedBatch(claimed.id, {
    inputs: [],
    checkpoint,
    claim: { owner: claimed.leaseOwner, attempt: claimed.attempt },
    allowRequestedUsageFinalization,
    usageReservations: [{ kind: 'teacherTokens', amount: 5 }],
    artifact: {
      kind: 'teacher-feed-checkpoint',
      contentHash: `control-race-${phase}`,
      phase,
      payload: { teacherTokens: 5 },
    },
  })

  const originalGet = controlStore.backend.get.bind(controlStore.backend)
  let competingError
  let interleaved = false
  controlStore.backend.get = (bucket, id) => {
    const value = originalGet(bucket, id)
    if (!interleaved && bucket === 'jobs' && id === claimed.id) {
      interleaved = true
      try { commit('unsafe-worker-commit') } catch (error) { competingError = error }
    }
    return value
  }
  let requested
  try {
    requested = controlStore.requestJobControl(claimed.id, 'cancel')
  } finally {
    controlStore.backend.get = originalGet
  }

  assert.equal(requested.state, 'cancel_requested')
  assert.ok(['SQLITE_BUSY', 'ERR_SQLITE_ERROR'].includes(competingError?.code), 'control transition must own the write lock before reading the job row')
  assert.match(String(competingError?.message ?? ''), /locked|busy/u)
  assert.equal(controlStore.getJob(claimed.id).payload.feedState.teacherTokens, 0)
  assert.equal(controlStore.getUsage().teacherTokens, 0)

  commit('cancelled-call-accounted', true)
  const finalized = controlStore.getJob(claimed.id)
  assert.equal(finalized.state, 'cancel_requested')
  assert.equal(finalized.payload.feedState.teacherTokens, 5)
  assert.equal(finalized.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(controlStore.getUsage().teacherTokens, 5)
})

test('cancelling a paused teacher job closes any outstanding token debt exactly once', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const usageDay = new Date().toISOString().slice(0, 10)
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    payload: {
      feedState: {
        promptTokens: 0,
        completionTokens: 0,
        teacherTokens: 0,
        pendingGenerationBatch: {
          promptTokens: 0,
          completionTokens: 0,
          generationErrors: 0,
          usageEstimated: false,
          inFlightAttempt: {
            requestUsageReserved: true,
            tokenUsageReserved: false,
            maxPromptTokens: 11,
            maxCompletionTokens: 17,
            usageDay,
          },
        },
      },
    },
  })
  store.updateJob(job.id, { state: 'paused', stage: 'paused' })

  const cancelled = store.requestJobControl(job.id, 'cancel')
  assert.equal(cancelled.state, 'cancelled')
  assert.equal(cancelled.payload.feedState.teacherTokens, 32_785)
  assert.equal(cancelled.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(cancelled.payload.feedState.pendingGenerationBatch.usageEstimated, true)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
  assert.equal(store.requestJobControl(job.id, 'cancel').payload.feedState.teacherTokens, 32_785)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
  assert.equal(store.recoverExpiredJobs(), 0)
})

test('legacy pre-reserved teacher attempt adds only the v8 context-floor delta once', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const usageDay = new Date().toISOString().slice(0, 10)
  store.consumeUsage('teacherTokens', 28, usageDay)
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    payload: {
      feedState: {
        promptTokens: 0,
        completionTokens: 0,
        teacherTokens: 0,
        pendingGenerationBatch: {
          promptTokens: 0,
          completionTokens: 0,
          generationErrors: 0,
          usageEstimated: false,
          lastError: null,
          inFlightAttempt: {
            requestUsageReserved: true,
            tokenUsageReserved: true,
            maxPromptTokens: 11,
            maxCompletionTokens: 17,
            usageDay,
          },
        },
      },
    },
  })
  store.updateJob(job.id, { state: 'paused', stage: 'paused' })

  const cancelled = store.requestJobControl(job.id, 'cancel')
  assert.equal(cancelled.state, 'cancelled')
  assert.equal(cancelled.payload.feedState.promptTokens, 32_768)
  assert.equal(cancelled.payload.feedState.completionTokens, 17)
  assert.equal(cancelled.payload.feedState.teacherTokens, 32_785)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
  assert.equal(store.requestJobControl(job.id, 'cancel').payload.feedState.teacherTokens, 32_785)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
})

test('declared v1 feed num_ctx debt is finalized at its persisted 4096 bound', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const usageDay = new Date().toISOString().slice(0, 10)
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    payload: {
      feedState: {
        promptTokens: 0,
        completionTokens: 0,
        teacherTokens: 0,
        pendingGenerationBatch: {
          promptTokens: 0,
          completionTokens: 0,
          generationErrors: 0,
          usageEstimated: false,
          lastError: null,
          inFlightAttempt: {
            requestUsageReserved: true,
            tokenUsageReserved: false,
            maxPromptTokens: 4_096,
            contextTokenLimit: 4_096,
            promptBoundPolicy: 'ollama-num-ctx-v1',
            maxCompletionTokens: 17,
            usageDay,
          },
        },
      },
    },
  })
  store.updateJob(job.id, { state: 'paused', stage: 'paused' })

  const cancelled = store.requestJobControl(job.id, 'cancel')
  assert.equal(cancelled.state, 'cancelled')
  assert.equal(cancelled.payload.feedState.promptTokens, 4_096)
  assert.equal(cancelled.payload.feedState.completionTokens, 17)
  assert.equal(cancelled.payload.feedState.teacherTokens, 4_113)
  assert.equal(store.getUsage(usageDay).teacherTokens, 4_113)
  assert.equal(store.requestJobControl(job.id, 'cancel').payload.feedState.teacherTokens, 4_113)
  assert.equal(store.getUsage(usageDay).teacherTokens, 4_113)
})

test('safety pause finalizes a stale pre-reserved teacher attempt before releasing its claim', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const usageDay = new Date().toISOString().slice(0, 10)
  store.consumeUsage('teacherTokens', 28, usageDay)
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload: {
      campaign: { protocolVersion: 7 },
      feedState: {
        promptTokens: 0,
        completionTokens: 0,
        teacherTokens: 0,
        pendingGenerationBatch: {
          promptTokens: 0,
          completionTokens: 0,
          generationErrors: 0,
          usageEstimated: false,
          lastError: null,
          inFlightAttempt: {
            requestUsageReserved: true,
            tokenUsageReserved: true,
            maxPromptTokens: 11,
            maxCompletionTokens: 17,
            usageDay,
          },
        },
      },
    },
  })
  const claimed = store.claimNextJob('stale-protocol-worker', 30_000, 60_000)
  const settlement = store.settleClaimedJob(job.id, claimed.leaseOwner, claimed.attempt, {
    action: 'pause',
    error: { code: 'STALE_TEACHER_PROTOCOL' },
  })
  assert.equal(settlement.action, 'pause')
  assert.equal(settlement.job.state, 'paused')
  assert.equal(settlement.job.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(settlement.job.payload.feedState.teacherTokens, 32_785)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
})

test('startup reconciles an unowned legacy teacher debt before a new call can spend its quota gap', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const usageDay = new Date().toISOString().slice(0, 10)
  let store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  store.updateConfig({ budgets: { maxDailyTeacherRequests: 10, maxDailyTeacherTokens: 65_536 } })
  store.consumeUsageBatch([
    { kind: 'teacherRequests', amount: 1, day: usageDay },
    { kind: 'teacherTokens', amount: 28, day: usageDay },
  ])
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    payload: {
      campaign: { protocolVersion: 7 },
      feedState: {
        teacherRequests: 1,
        promptTokens: 0,
        completionTokens: 0,
        teacherTokens: 0,
        pendingGenerationBatch: {
          promptTokens: 0,
          completionTokens: 0,
          generationErrors: 0,
          usageEstimated: false,
          inFlightAttempt: {
            requestUsageReserved: true,
            tokenUsageReserved: true,
            maxPromptTokens: 11,
            maxCompletionTokens: 17,
            usageDay,
          },
        },
      },
    },
  })
  store.updateJob(job.id, { state: 'paused', stage: 'paused' })
  store.close()

  store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => store.close())
  const reconciled = store.getJob(job.id)
  assert.equal(reconciled.state, 'paused')
  assert.equal(reconciled.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(reconciled.payload.feedState.teacherTokens, 32_785)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
  assert.equal(store.reconcileUnownedTeacherUsage(), 0, 'startup reconciliation must be idempotent')

  assert.throws(() => store.reserveTeacherCallUsage({
    id: 'new-call-after-legacy-debt',
    usageDay,
    model: 'teacher:14b',
    maxPromptTokens: 32_768,
    maxCompletionTokens: 512,
    reservedTokens: 33_280,
  }), { code: 'BUDGET_EXCEEDED' })
  assert.equal(store.getUsage(usageDay).teacherRequests, 1)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
})

test('startup reconciles terminal legacy teacher debt exactly once before reopening quota', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const usageDay = new Date().toISOString().slice(0, 10)
  let store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  store.updateConfig({ budgets: { maxDailyTeacherRequests: 10, maxDailyTeacherTokens: 65_536 } })
  store.consumeUsageBatch([
    { kind: 'teacherRequests', amount: 1, day: usageDay },
    { kind: 'teacherTokens', amount: 28, day: usageDay },
  ])
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    payload: {
      campaign: { protocolVersion: 7 },
      feedState: {
        teacherRequests: 1,
        promptTokens: 0,
        completionTokens: 0,
        teacherTokens: 0,
        pendingGenerationBatch: {
          promptTokens: 0,
          completionTokens: 0,
          generationErrors: 0,
          usageEstimated: false,
          inFlightAttempt: {
            requestUsageReserved: true,
            tokenUsageReserved: true,
            maxPromptTokens: 11,
            maxCompletionTokens: 17,
            usageDay,
          },
        },
      },
    },
  })
  store.updateJob(job.id, {
    state: 'cancelled',
    stage: 'cancelled',
    finishedAt: new Date().toISOString(),
  })
  store.close()

  store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => store.close())
  const reconciled = store.getJob(job.id)
  assert.equal(reconciled.state, 'cancelled')
  assert.equal(reconciled.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(reconciled.payload.feedState.teacherTokens, 32_785)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
  assert.equal(store.reconcileUnownedTeacherUsage(), 0, 'terminal startup reconciliation must be idempotent')
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)

  assert.throws(() => store.reserveTeacherCallUsage({
    id: 'new-call-after-terminal-legacy-debt',
    usageDay,
    model: 'teacher:14b',
    maxPromptTokens: 32_768,
    maxCompletionTokens: 512,
    reservedTokens: 33_280,
  }), { code: 'BUDGET_EXCEEDED' })
  assert.equal(store.getUsage(usageDay).teacherRequests, 1)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)
})

test('claim reconciles a queued legacy teacher debt but startup leaves a live claimed attempt untouched', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const usageDay = new Date().toISOString().slice(0, 10)
  const store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => store.close())
  store.consumeUsage('teacherTokens', 28, usageDay)
  const legacyPayload = {
    campaign: { protocolVersion: 7 },
    feedState: {
      promptTokens: 0,
      completionTokens: 0,
      teacherTokens: 0,
      pendingGenerationBatch: {
        promptTokens: 0,
        completionTokens: 0,
        generationErrors: 0,
        usageEstimated: false,
        inFlightAttempt: {
          requestUsageReserved: true,
          tokenUsageReserved: true,
          maxPromptTokens: 11,
          maxCompletionTokens: 17,
          usageDay,
        },
      },
    },
  }
  const queued = store.createJob({ type: 'teacher-feed-campaign', payload: legacyPayload }).job
  const claimed = store.claimNextJob('legacy-claim-owner', 30_000, 60_000)
  assert.equal(claimed.id, queued.id)
  assert.equal(claimed.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(store.getUsage(usageDay).teacherTokens, 32_785)

  const liveDay = '2026-08-28'
  store.consumeUsage('teacherTokens', 28, liveDay)
  const live = store.createJob({
    type: 'teacher-feed-campaign',
    payload: structuredClone(legacyPayload),
  }).job
  store.updateJob(live.id, {
    state: 'running',
    attempt: 1,
    leaseOwner: 'live-owner',
    leaseUntil: new Date(Date.now() + 30_000).toISOString(),
    executionLeaseUntil: new Date(Date.now() + 60_000).toISOString(),
  })
  live.payload.feedState.pendingGenerationBatch.inFlightAttempt.usageDay = liveDay
  store.updateJob(live.id, { payload: live.payload })

  const observer = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => observer.close())
  const stillOwned = observer.getJob(live.id)
  assert.equal(stillOwned.state, 'running')
  assert.ok(stillOwned.payload.feedState.pendingGenerationBatch.inFlightAttempt)
  assert.equal(observer.getUsage(liveDay).teacherTokens, 28)
  assert.equal(store.assertJobClaim(live.id, 'live-owner', 1).state, 'running')
})

test('teacher feed batch commits approved research sources and durable cursor atomically', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    payload: { campaign: { id: 'campaign-1' }, feedState: { nextBatchIndex: 0 } },
  })
  const input = normalizeTextSource({
    text: 'Dün marangoz küçük masayı dikkatlice onardı.',
    title: 'Sentetik araştırma örneği',
    provenance: {
      kind: 'ollama-generated',
      campaignId: 'campaign-1',
      evaluationEligible: false,
      independenceGroupId: 'ollama-lineage:test-model',
    },
  }, store.getConfig())
  input.sourceType = 'ollama-generated'
  input.provenance.kind = 'ollama-generated'

  const committed = store.commitTeacherFeedBatch(job.id, {
    inputs: [input],
    checkpoint: { nextBatchIndex: 1, accepted: 1 },
    artifact: { kind: 'teacher-feed-batch', contentHash: 'teacher-batch-1', payload: { accepted: 1 } },
  })

  assert.equal(committed.items.length, 1)
  assert.equal(committed.items[0].duplicate, false)
  assert.equal(store.getSource(committed.items[0].source.id).status, 'approved')
  assert.equal(store.getSource(committed.items[0].source.id).selected, true)
  assert.equal(store.getSource(committed.items[0].source.id).provenance.evaluationEligible, false)
  assert.match(store.getSource(committed.items[0].source.id).decisionReason, /dondurulmuş öğretmen protokolü/u)
  assert.deepEqual(store.getJob(job.id).payload.feedState, { nextBatchIndex: 1, accepted: 1 })
  assert.equal(store.listArtifacts().filter((artifact) => artifact.kind === 'teacher-feed-batch').length, 1)
})

test('teacher request quota reservation and feed checkpoint commit atomically', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  store.updateConfig({ budgets: { maxDailyTeacherRequests: 1 } })
  const { job } = store.createJob({
    type: 'teacher-feed-campaign',
    payload: { campaign: { id: 'campaign-reservation' }, feedState: { teacherRequests: 0 } },
  })

  assert.throws(() => store.commitTeacherFeedBatch(job.id, {
    inputs: [{ sourceType: 'manual', provenance: { evaluationEligible: false } }],
    checkpoint: { teacherRequests: 1 },
    artifact: { kind: 'teacher-feed-checkpoint', contentHash: 'teacher-reservation-invalid', payload: { phase: 'must-rollback' } },
    usageReservations: [{ kind: 'teacherRequests', amount: 1 }],
  }), { code: 'INVALID_TEACHER_SOURCE' })
  assert.equal(store.getUsage().teacherRequests, 0)
  assert.deepEqual(store.getJob(job.id).payload.feedState, { teacherRequests: 0 })

  store.commitTeacherFeedBatch(job.id, {
    inputs: [],
    checkpoint: { teacherRequests: 1 },
    artifact: { kind: 'teacher-feed-checkpoint', contentHash: 'teacher-reservation-1', payload: { phase: 'reserved' } },
    usageReservations: [{ kind: 'teacherRequests', amount: 1 }],
  })
  assert.equal(store.getUsage().teacherRequests, 1)
  assert.deepEqual(store.getJob(job.id).payload.feedState, { teacherRequests: 1 })

  assert.throws(() => store.commitTeacherFeedBatch(job.id, {
    inputs: [],
    checkpoint: { teacherRequests: 2 },
    artifact: { kind: 'teacher-feed-checkpoint', contentHash: 'teacher-reservation-2', payload: { phase: 'must-rollback' } },
    usageReservations: [{ kind: 'teacherRequests', amount: 1 }],
  }), { code: 'BUDGET_EXCEEDED' })
  assert.equal(store.getUsage().teacherRequests, 1)
  assert.deepEqual(store.getJob(job.id).payload.feedState, { teacherRequests: 1 })
  assert.equal(store.listArtifacts().some((artifact) => artifact.contentHash === 'teacher-reservation-2'), false)
})

test('a bounded teacher request and token reservation succeeds or rolls back as one unit', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  store.updateConfig({ budgets: { maxDailyTeacherTokens: 0 } })

  assert.throws(() => store.consumeUsageBatch([
    { kind: 'teacherRequests', amount: 1 },
    { kind: 'teacherTokens', amount: 10 },
  ]), { code: 'BUDGET_EXCEEDED' })
  assert.equal(store.getUsage().teacherRequests, 0)
  assert.equal(store.getUsage().teacherTokens, 0)
})

test('durable teacher call reservation settles exactly once and records reported overrun', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const reservation = store.reserveTeacherCallUsage({
    id: 'teacher-call-ledger-1',
    model: 'teacher:14b',
    maxPromptTokens: 30,
    maxCompletionTokens: 10,
    reservedTokens: 40,
  })
  assert.equal(reservation.status, 'reserved')
  assert.deepEqual(store.getUsage(), {
    day: reservation.usageDay,
    jobs: 0,
    teacherRequests: 1,
    teacherTokens: 40,
  })

  const settled = store.settleTeacherCallUsage(reservation.id, {
    promptTokens: 50,
    completionTokens: 5,
    usageEstimated: false,
  })
  assert.equal(settled.status, 'settled')
  assert.equal(settled.actualTokens, 55)
  assert.equal(store.getUsage().teacherTokens, 55, 'spent usage above the reservation must not be hidden')

  const replay = store.settleTeacherCallUsage(reservation.id, {
    promptTokens: 50,
    completionTokens: 5,
    usageEstimated: false,
  })
  assert.deepEqual(replay, settled)
  assert.equal(store.getUsage().teacherTokens, 55)
  assert.throws(() => store.settleTeacherCallUsage(reservation.id, {
    promptTokens: 1,
    completionTokens: 1,
  }), { code: 'TEACHER_USAGE_SETTLEMENT_CONFLICT' })
})

test('teacher call reservation is atomic and idempotent at the daily quota', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  store.updateConfig({ budgets: { maxDailyTeacherRequests: 1, maxDailyTeacherTokens: 40 } })
  const input = {
    id: 'teacher-call-ledger-quota',
    model: 'teacher:14b',
    maxPromptTokens: 30,
    maxCompletionTokens: 10,
    reservedTokens: 40,
  }
  const first = store.reserveTeacherCallUsage(input)
  assert.deepEqual(store.reserveTeacherCallUsage(input), first)
  assert.equal(store.getUsage().teacherRequests, 1)
  assert.equal(store.getUsage().teacherTokens, 40)
  assert.throws(() => store.reserveTeacherCallUsage({ ...input, id: 'teacher-call-ledger-over-quota' }), { code: 'BUDGET_EXCEEDED' })
  assert.equal(store.getUsage().teacherRequests, 1)
  assert.equal(store.getUsage().teacherTokens, 40)
})

test('thought control and terminal artifact commit use one-way atomic state fences', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const { episode } = store.createThoughtEpisode({
    id: 'thought-control-fence',
    requestHash: 'thought-control-request',
    inputText: 'Kontrol yarışı.',
    base: { snapshotId: 'base-for-control-test' },
  })
  store.updateThoughtEpisode(episode.id, { status: 'running' }, { expectedStatuses: ['queued'] })
  const cancel = store.requestThoughtEpisodeControl(episode.id, 'cancel')
  assert.equal(cancel.changed, true)
  assert.equal(cancel.status, 'cancel_requested')

  assert.throws(() => store.attachThoughtEpisodeArtifacts(episode.id, [], {
    status: 'completed', finishedAt: new Date().toISOString(),
  }, { expectedStatuses: ['running'] }), { code: 'THOUGHT_STATE_CONFLICT' })
  assert.equal(store.getThoughtEpisode(episode.id).status, 'cancel_requested')

  store.attachThoughtEpisodeArtifacts(episode.id, [], {
    status: 'cancelled', finishedAt: new Date().toISOString(),
  }, { expectedStatuses: ['cancel_requested'] })
  const terminalNoOp = store.requestThoughtEpisodeControl(episode.id, 'pause')
  assert.equal(terminalNoOp.changed, false)
  assert.equal(terminalNoOp.status, 'cancelled')
})

test('Node SQLite backend is selected when available', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  t.after(() => store.close())
  assert.equal(store.backendKind, 'sqlite')
  assert.equal(store.backend.db.prepare('PRAGMA busy_timeout').get().timeout, 1_000)
  const source = normalizeTextSource({ text: 'SQLite kalıcı kayıt deneyi.' }, store.getConfig())
  assert.equal(store.createSource(source).duplicate, false)
  assert.equal(store.createSource(source).duplicate, true)
})

test('SQLite renewal serializes stale-read recovery races across store instances', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => store.close())
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 3 })
  const firstClaim = store.claimNextJob('worker-attempt-1', 30_000, 60_000)
  assert.equal(firstClaim.attempt, 1)
  // Legacy records can lack the fixed execution deadline. This leaves exactly
  // the historical read/recover/write window that the regression targets.
  store.updateJob(job.id, {
    leaseUntil: new Date(Date.now() - 1_000).toISOString(),
    executionLeaseUntil: null,
  })

  const controlBuffer = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT)
  const control = new Int32Array(controlBuffer)
  const worker = new Worker(`
    const { parentPort, workerData } = require('node:worker_threads')
    ;(async () => {
      let secondStore
      try {
        const { createTrainingStore } = await import(workerData.storeUrl)
        secondStore = await createTrainingStore({ dataDirectory: workerData.dataDirectory, requireSqlite: true })
        secondStore.backend.db.exec('PRAGMA busy_timeout=1000')
        const control = new Int32Array(workerData.controlBuffer)
        parentPort.postMessage({ type: 'ready' })
        Atomics.wait(control, 0, 0)
        const recoverAndClaim = () => {
          const recovered = secondStore.recoverExpiredJobs()
          const claimed = secondStore.claimNextJob('worker-attempt-2', 30_000, 60_000)
          return { ok: true, recovered, claimed: claimed && { id: claimed.id, attempt: claimed.attempt, leaseOwner: claimed.leaseOwner } }
        }
        let initial
        try {
          initial = recoverAndClaim()
        } catch (error) {
          initial = { ok: false, code: error?.code, message: String(error?.message ?? error) }
        }
        Atomics.store(control, 0, initial.ok ? 2 : 3)
        Atomics.notify(control, 0)
        while (Atomics.load(control, 0) !== 4) {
          const observed = Atomics.load(control, 0)
          Atomics.wait(control, 0, observed)
        }

        let final = initial
        if (!initial.ok) {
          secondStore.updateJob(workerData.jobId, {
            leaseUntil: new Date(Date.now() - 1000).toISOString(),
            executionLeaseUntil: null,
          })
          final = recoverAndClaim()
        }
        parentPort.postMessage({ type: 'done', initial, final })
      } catch (error) {
        parentPort.postMessage({ type: 'fatal', code: error?.code, message: String(error?.stack ?? error) })
      } finally {
        secondStore?.close()
      }
    })()
  `, {
    eval: true,
    workerData: {
      controlBuffer,
      dataDirectory,
      jobId: job.id,
      storeUrl: new URL('../store.mjs', import.meta.url).href,
    },
  })
  t.after(() => worker.terminate())
  const [ready] = await once(worker, 'message')
  assert.equal(ready.type, 'ready')
  const donePromise = new Promise((resolve, reject) => {
    const onMessage = (message) => {
      if (message.type === 'done') {
        worker.off('error', onError)
        resolve(message)
      } else if (message.type === 'fatal') {
        worker.off('error', onError)
        reject(Object.assign(new Error(message.message), { code: message.code }))
      }
    }
    const onError = (error) => {
      worker.off('message', onMessage)
      reject(error)
    }
    worker.on('message', onMessage)
    worker.once('error', onError)
  })

  const originalGet = store.backend.get.bind(store.backend)
  let interleaved = false
  store.backend.get = (bucket, id) => {
    const value = originalGet(bucket, id)
    if (!interleaved && bucket === 'jobs' && id === job.id) {
      interleaved = true
      Atomics.store(control, 0, 1)
      Atomics.notify(control, 0)
      const waitResult = Atomics.wait(control, 0, 1, 5_000)
      assert.notEqual(waitResult, 'timed-out', 'second SQLite connection must reach the competing write')
    }
    return value
  }
  let renewed
  try {
    renewed = store.renewLease(job.id, 'worker-attempt-1', 30_000, 1)
  } finally {
    store.backend.get = originalGet
    Atomics.store(control, 0, 4)
    Atomics.notify(control, 0)
  }
  assert.equal(renewed, true)
  const done = await donePromise
  assert.equal(done.initial.ok, false, 'BEGIN IMMEDIATE must exclude the competing recovery between read and write')
  assert.match(done.initial.message, /locked|busy/u)
  assert.equal(done.final.ok, true)
  assert.equal(done.final.recovered, 1)
  assert.equal(done.final.claimed.attempt, 2)

  const current = store.getJob(job.id)
  assert.equal(current.attempt, 2)
  assert.equal(current.leaseOwner, 'worker-attempt-2')
  assert.equal(current.state, 'running')
})

test('stale teacher-feed claim cannot commit sources, usage, artifact, or checkpoint', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const firstStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const secondStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => {
    firstStore.close()
    secondStore.close()
  })
  const { job } = firstStore.createJob({
    type: 'teacher-feed-campaign',
    maxAttempts: 3,
    payload: { campaign: { id: 'stale-claim-campaign' }, feedState: { accepted: 0 } },
  })
  assert.equal(firstStore.claimNextJob('teacher-worker-1', 30_000, 60_000).attempt, 1)
  secondStore.updateJob(job.id, {
    leaseUntil: new Date(Date.now() - 1_000).toISOString(),
    executionLeaseUntil: new Date(Date.now() - 1_000).toISOString(),
  })
  assert.equal(secondStore.recoverExpiredJobs(), 1)
  const replacement = secondStore.claimNextJob('teacher-worker-2', 30_000, 60_000)
  assert.equal(replacement.attempt, 2)

  const input = normalizeTextSource({
    text: 'Bahçıvan küçük fidanı dikkatlice suladı.',
    title: 'Stale claim araştırma örneği',
    provenance: {
      kind: 'ollama-generated',
      campaignId: 'stale-claim-campaign',
      evaluationEligible: false,
      independenceGroupId: 'ollama-lineage:test-model',
    },
  }, firstStore.getConfig())
  input.sourceType = 'ollama-generated'
  input.provenance.kind = 'ollama-generated'

  assert.throws(() => firstStore.commitTeacherFeedBatch(job.id, {
    inputs: [input],
    checkpoint: { accepted: 1 },
    artifact: { kind: 'teacher-feed-batch', contentHash: 'stale-claim-batch', payload: { accepted: 1 } },
    usageReservations: [{ kind: 'teacherRequests', amount: 1 }],
    claim: { owner: 'teacher-worker-1', attempt: 1 },
  }), { code: 'STALE_JOB_LEASE' })

  const current = secondStore.getJob(job.id)
  assert.equal(current.attempt, 2)
  assert.equal(current.leaseOwner, 'teacher-worker-2')
  assert.deepEqual(current.payload.feedState, { accepted: 0 })
  assert.equal(secondStore.getUsage().teacherRequests, 0)
  assert.equal(secondStore.listSources().length, 0)
  assert.equal(secondStore.listArtifacts().length, 0)
})

test('stale worker claim cannot reject a snapshot owned by a replacement attempt', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const firstStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const secondStore = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => {
    firstStore.close()
    secondStore.close()
  })
  const { job } = firstStore.createJob({ type: 'teacher-feed-campaign', maxAttempts: 3 })
  const snapshot = firstStore.createSnapshot({ status: 'candidate', evaluation: { passed: true } })
  assert.equal(firstStore.claimNextJob('snapshot-worker-1', 30_000, 60_000).attempt, 1)
  secondStore.updateJob(job.id, {
    leaseUntil: new Date(Date.now() - 1_000).toISOString(),
    executionLeaseUntil: new Date(Date.now() - 1_000).toISOString(),
  })
  assert.equal(secondStore.recoverExpiredJobs(), 1)
  assert.equal(secondStore.claimNextJob('snapshot-worker-2', 30_000, 60_000).attempt, 2)

  assert.throws(() => firstStore.updateClaimedJobPayload(
    job.id,
    'snapshot-worker-1',
    1,
    { resolvedSourceIds: ['stale-source'] },
  ), { code: 'STALE_JOB_LEASE' })
  assert.throws(() => firstStore.createClaimedArtifact(
    job.id,
    'snapshot-worker-1',
    1,
    { kind: 'stale-artifact', contentHash: 'stale-artifact-hash', payload: {} },
  ), { code: 'STALE_JOB_LEASE' })
  assert.throws(() => firstStore.createClaimedSnapshot(
    job.id,
    'snapshot-worker-1',
    1,
    { id: 'stale-attempt-snapshot', status: 'candidate' },
  ), { code: 'STALE_JOB_LEASE' })
  assert.throws(() => firstStore.activateClaimedSnapshot(
    job.id,
    'snapshot-worker-1',
    1,
    snapshot.id,
  ), { code: 'STALE_JOB_LEASE' })
  assert.throws(() => firstStore.addClaimedEvent(
    job.id,
    'snapshot-worker-1',
    1,
    { type: 'stale-stage-completed' },
  ), { code: 'STALE_JOB_LEASE' })

  assert.throws(() => firstStore.rejectSnapshot(
    snapshot.id,
    { passed: false, reason: 'stale attempt must not persist' },
    { jobId: job.id, claim: { owner: 'snapshot-worker-1', attempt: 1 } },
  ), { code: 'STALE_JOB_LEASE' })

  const current = secondStore.getSnapshot(snapshot.id)
  assert.equal(current.status, 'candidate')
  assert.deepEqual(current.evaluation, { passed: true })
  assert.equal(secondStore.getActiveSnapshot(), undefined)
  assert.equal(secondStore.getSnapshot('stale-attempt-snapshot'), undefined)
  assert.equal(secondStore.listArtifacts().length, 0)
  assert.equal(secondStore.listEvents().length, 0)
  assert.equal(secondStore.getJob(job.id).payload.resolvedSourceIds, undefined)
})

test('expired fixed execution lease is a hard fence even before takeover', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const { job } = store.createJob({ type: 'teacher-feed-campaign', maxAttempts: 3 })
  const claimed = store.claimNextJob('deadline-worker', 30_000, 60_000)
  store.updateJob(job.id, {
    leaseUntil: new Date(Date.now() + 30_000).toISOString(),
    executionLeaseUntil: new Date(Date.now() - 1).toISOString(),
  })

  assert.throws(
    () => store.assertJobClaim(job.id, 'deadline-worker', claimed.attempt),
    (error) => error?.code === 'STALE_JOB_LEASE' && error.executionLeaseExpired === true,
  )
  assert.throws(() => store.commitTeacherFeedBatch(job.id, {
    inputs: [],
    checkpoint: { accepted: 1 },
    artifact: { kind: 'teacher-feed-checkpoint', contentHash: 'expired-execution-lease', payload: {} },
    usageReservations: [{ kind: 'teacherRequests', amount: 1 }],
    claim: { owner: 'deadline-worker', attempt: claimed.attempt },
  }), { code: 'STALE_JOB_LEASE' })
  assert.deepEqual(store.getJob(job.id).payload.feedState, undefined)
  assert.equal(store.getUsage().teacherRequests, 0)
  assert.equal(store.listArtifacts().length, 0)
})

test('artifact kind+content hash index returns the full immutable artifact and deduplicates writes', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const first = store.createArtifact({ kind: 'language-model', contentHash: 'abc123', payload: { weights: [1, 2, 3] }, jobId: 'job-a' })
  const repeated = store.createArtifact({ kind: 'language-model', contentHash: 'abc123', payload: { weights: [9] }, jobId: 'job-b' })
  const otherKind = store.createArtifact({ kind: 'family-model', contentHash: 'abc123', payload: { families: 4 } })
  assert.equal(repeated.id, first.id)
  assert.deepEqual(store.findArtifactByContentHash('language-model', 'abc123').payload, { weights: [1, 2, 3] })
  assert.equal(store.findArtifactByContentHash('family-model', 'abc123').id, otherKind.id)
  assert.equal(store.listArtifacts().length, 2)
})

for (const backend of ['atomic-json', 'sqlite']) {
  test(`${backend} artifact metadata stays payload-free and preserves exact payload bytes`, async (t) => {
    const store = await createTrainingStore({
      dataDirectory: await temporaryDirectory(t),
      ...(backend === 'atomic-json' ? { forceJson: true } : { requireSqlite: true }),
    })
    t.after(() => store.close())
    const payload = {
      label: 'Türkçe içerik',
      nested: [{ value: true }, { value: null }],
      text: 'x'.repeat(128 * 1024),
    }
    const artifact = store.createArtifact({
      id: `${backend}-metadata-artifact`,
      kind: 'family-run',
      contentHash: `${backend}-metadata-hash`,
      payload,
      jobId: 'metadata-job',
    })

    const expectedBytes = Buffer.byteLength(JSON.stringify(payload))
    assert.deepEqual(store.getArtifactMetadata(artifact.id), {
      id: artifact.id,
      kind: 'family-run',
      contentHash: `${backend}-metadata-hash`,
      createdAt: artifact.createdAt,
      jobId: 'metadata-job',
      payloadBytes: expectedBytes,
    })
    assert.equal(store.listArtifactMetadata()[0].payloadBytes, expectedBytes)
    assert.equal('payload' in store.listArtifactMetadata()[0], false)
    assert.equal('payloadBytes' in store.listArtifacts()[0], false, 'legacy public list shape remains unchanged')

    store.createArtifact({
      id: `${backend}-other-kind`, kind: 'unrelated-kind', contentHash: `${backend}-other-hash`, payload: { ignored: true },
    })
    store.createArtifact({
      id: `${backend}-metadata-artifact-2`, kind: 'family-run', contentHash: `${backend}-metadata-hash-2`, payload: { newer: true },
    })
    const bounded = store.listArtifactMetadataByKind('family-run', { limit: 1 })
    assert.equal(bounded.length, 1)
    assert.equal(bounded[0].kind, 'family-run')
    assert.equal('payload' in bounded[0], false)
    assert.equal(store.countArtifactMetadataByKind('family-run'), 2)
    assert.equal(store.countArtifactMetadataByKind('unrelated-kind'), 1)
    assert.equal(store.countArtifactMetadataByKind('missing-kind'), 0)
    assert.deepEqual(store.listArtifactMetadataByKind('missing-kind'), [])
    assert.throws(() => store.listArtifactMetadataByKind('family-run', { limit: 0 }), /1\.\.4096/)
    assert.throws(() => store.countArtifactMetadataByKind(''), /geçerli kind/u)
  })
}

for (const backend of ['atomic-json', 'sqlite']) {
  test(`${backend} compresses large immutable artifact payloads transparently`, async (t) => {
    const store = await createTrainingStore({
      dataDirectory: await temporaryDirectory(t),
      ...(backend === 'atomic-json' ? { forceJson: true } : { requireSqlite: true }),
    })
    t.after(() => store.close())
    const payload = { matrix: '0.0000,'.repeat(80_000), proof: { identityDigest: 'fixed' } }
    const artifact = store.createArtifact({ kind: 'runtime-model-state', contentHash: `compressed-${backend}`, payload })
    const stored = store.backend.get('artifacts', artifact.id)
    assert.equal(stored.payload.$type, 'bombom/gzip-json-v1')
    assert.ok(stored.payload.compressedBytes < stored.payload.uncompressedBytes)
    assert.deepEqual(store.getArtifact(artifact.id).payload, payload)
    assert.deepEqual(store.findArtifactByContentHash('runtime-model-state', `compressed-${backend}`).payload, payload)
    assert.equal(store.getArtifactMetadata(artifact.id).payloadBytes, Buffer.byteLength(JSON.stringify(payload)))
  })
}

test('SQLite startup backfills metadata for artifacts written before the side table existed', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  let store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const payload = { legacy: true, text: 'eski artifact'.repeat(16_384) }
  const artifact = store.createArtifact({ id: 'legacy-artifact', kind: 'family-run', payload })
  store.close()

  const { DatabaseSync } = await import('node:sqlite')
  const database = new DatabaseSync(path.join(dataDirectory, 'autonomous-training.sqlite'))
  database.exec('DELETE FROM artifact_metadata')
  database.close()

  store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => store.close())
  assert.equal(store.getArtifactMetadata(artifact.id).payloadBytes, Buffer.byteLength(JSON.stringify(payload)))
  assert.equal(store.getArtifactMetadata(artifact.id).kind, 'family-run')
})

for (const backend of ['atomic-json', 'sqlite']) {
  test(`${backend} source metadata stays content-free and follows source decisions transactionally`, async (t) => {
    const store = await createTrainingStore({
      dataDirectory: await temporaryDirectory(t),
      ...(backend === 'atomic-json' ? { forceJson: true } : { requireSqlite: true }),
    })
    t.after(() => store.close())
    const created = store.createSource({
      id: `${backend}-source-metadata`,
      title: 'Kaynak metadata kanıtı',
      sourceType: 'manual-text',
      content: 'İçerik yalnız tam kaynak okumasında bulunmalıdır.',
      contentHash: `${backend}-source-metadata-hash`,
      bytes: 52,
      language: 'tr',
      provenance: {
        kind: 'manual',
        label: 'metadata testi',
        managedByTeacherCampaign: false,
        nested: { retained: true },
      },
    }).source

    const { content, ...expectedMetadata } = created
    assert.equal(content, 'İçerik yalnız tam kaynak okumasında bulunmalıdır.')
    assert.deepEqual(store.backend.getSourceMetadata(created.id), expectedMetadata)
    assert.deepEqual(store.listSources(), [expectedMetadata])
    assert.deepEqual(store.listSources({ includeContent: true }), [created])

    store.decideSource(created.id, 'approved', 'metadata kararı')
    assert.equal(store.backend.getSourceMetadata(created.id).status, 'approved')
    assert.equal(store.backend.getSourceMetadata(created.id).selected, true)
    store.setSourceSelected(created.id, false)
    assert.equal(store.backend.getSourceMetadata(created.id).selected, false)

    const beforeRollback = store.backend.getSourceMetadata(created.id)
    assert.throws(() => store.backend.transaction(() => {
      const source = store.backend.get('sources', created.id)
      source.status = 'rejected'
      store.backend.put('sources', source.id, source)
      throw new Error('rollback proof')
    }), /rollback proof/u)
    assert.deepEqual(store.backend.getSourceMetadata(created.id), beforeRollback)
  })
}

test('SQLite source list and hash lookup avoid the full corpus row scan', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  t.after(() => store.close())
  const content = 'çok büyük kaynak '.repeat(64 * 1024)
  const source = store.createSource({
    id: 'sqlite-source-hot-path',
    title: 'Sıcak yol kanıtı',
    sourceType: 'manual-text',
    content,
    contentHash: 'sqlite-source-hot-path-hash',
    bytes: Buffer.byteLength(content),
    provenance: { kind: 'manual', label: 'hot path' },
  }).source
  store.decideSource(source.id, 'approved', 'sıcak yol testi')

  const originalList = store.backend.list.bind(store.backend)
  const originalGet = store.backend.get.bind(store.backend)
  let sourceGets = 0
  store.backend.list = (bucket) => {
    if (bucket === 'sources') throw new Error('full source scan invoked')
    return originalList(bucket)
  }
  store.backend.get = (bucket, id) => {
    if (bucket === 'sources') sourceGets += 1
    return originalGet(bucket, id)
  }
  try {
    const listed = store.listSources({ status: 'approved' })
    assert.equal(listed.length, 1)
    assert.equal('content' in listed[0], false)
    assert.equal(sourceGets, 0, 'metadata listing must not decode even one full source row')
    assert.equal(store.getSourceContent(source.id), content)
    assert.equal(sourceGets, 0, 'streaming one content field must not hydrate the full JSON source record')

    const found = store.findSourceByHash(source.contentHash)
    assert.equal(found.id, source.id)
    assert.equal(found.content, content)
    assert.equal(sourceGets, 1, 'indexed hash lookup hydrates only the matching source row')
  } finally {
    store.backend.list = originalList
    store.backend.get = originalGet
  }

  const queryPlan = store.backend.db.prepare(
    'EXPLAIN QUERY PLAN SELECT data FROM source_metadata WHERE content_hash = ? LIMIT 1',
  ).all(source.contentHash)
  assert.match(queryPlan.map((row) => row.detail).join('\n'), /source_metadata_content_hash/u)
  assert.equal(store.listSources({ includeContent: true })[0].content, content)
})

test('SQLite direct source writes roll back the record when metadata indexing fails', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  t.after(() => store.close())
  const source = {
    id: 'atomic-source-write',
    title: 'Atomiklik kanıtı',
    sourceType: 'manual-text',
    content: 'İkinci SQLite yazımı başarısız olursa ilk yazım görünür kalmamalı.',
    contentHash: 'atomic-source-write-hash',
    bytes: 68,
    language: 'tr',
    status: 'quarantine',
    provenance: { kind: 'manual' },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    selected: false,
  }
  const originalStatement = store.backend.sourceMetadataPutStatement
  store.backend.sourceMetadataPutStatement = {
    run() { throw new Error('injected metadata failure') },
  }
  try {
    assert.throws(() => store.backend.put('sources', source.id, source), /injected metadata failure/u)
  } finally {
    store.backend.sourceMetadataPutStatement = originalStatement
  }
  assert.equal(store.backend.get('sources', source.id), undefined)
  assert.equal(store.backend.getSourceMetadata(source.id), undefined)

  store.backend.put('sources', source.id, source)
  assert.equal(store.backend.get('sources', source.id).contentHash, source.contentHash)
  assert.equal(store.backend.getSourceMetadata(source.id).contentHash, source.contentHash)
})

test('SQLite restart backfills legacy source metadata with provenance and indexed hash lookup', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  let store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  const source = store.createSource({
    id: 'legacy-source-metadata',
    title: 'Eski kaynak',
    sourceType: 'ollama-generated',
    content: 'eski kaynak içeriği '.repeat(8_192),
    contentHash: 'legacy-source-metadata-hash',
    bytes: 163_840,
    provenance: {
      kind: 'ollama-generated',
      label: 'Qwen 14B',
      managedByTeacherCampaign: true,
      evaluationEligible: false,
      independenceGroupId: 'ollama-lineage:qwen14b',
    },
  }).source
  store.decideSource(source.id, 'approved', 'eski kayıt')
  store.close()

  const { DatabaseSync } = await import('node:sqlite')
  const database = new DatabaseSync(path.join(dataDirectory, 'autonomous-training.sqlite'))
  database.exec('DELETE FROM source_metadata')
  database.close()

  store = await createTrainingStore({ dataDirectory, requireSqlite: true })
  t.after(() => store.close())
  const metadata = store.listSources({ status: 'approved' })[0]
  assert.equal(metadata.id, source.id)
  assert.equal(metadata.contentHash, source.contentHash)
  assert.equal(metadata.provenance.kind, 'ollama-generated')
  assert.equal(metadata.provenance.managedByTeacherCampaign, true)
  assert.equal(metadata.provenance.independenceGroupId, 'ollama-lineage:qwen14b')
  assert.equal('content' in metadata, false)
  assert.equal(store.findSourceByHash(source.contentHash).id, source.id)

  const row = store.backend.db.prepare('SELECT * FROM source_metadata WHERE id = ?').get(source.id)
  assert.equal(row.content_hash, source.contentHash)
  assert.equal(row.status, 'approved')
  assert.equal(row.selected, 1)
  assert.equal(row.source_type, 'ollama-generated')
  assert.equal(row.provenance_kind, 'ollama-generated')
  assert.equal(row.provenance_label, 'Qwen 14B')
  assert.equal(row.managed_by_teacher_campaign, 1)
})

test('logical disk budget covers jobs, events, snapshots, and configuration writes', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  assert.equal(store.getEffectiveDiskBudgetBytes(), 64 * 1024 * 1024)

  const constrained = store.getConfig()
  for (let iteration = 0; iteration < 4; iteration += 1) {
    constrained.budgets.diskMb = (store.approximateStorageBytes() + 128) / (1024 * 1024)
    store.backend.setMeta('config', constrained)
  }
  const large = 'x'.repeat(8_192)
  const rejected = { code: 'DISK_BUDGET_EXCEEDED' }

  assert.throws(() => store.createJob({ payload: { large } }), rejected)
  assert.throws(() => store.addEvent({ type: 'oversized-test-event', data: { large } }), rejected)
  assert.throws(() => store.createSnapshot({ label: large }), rejected)
  assert.throws(() => store.updateConfig({ sourcePolicy: { watchedDirectories: [`/tmp/${large}`] } }), rejected)

  assert.equal(store.listJobs().length, 0)
  assert.equal(store.listEvents().length, 0)
  assert.equal(store.listSnapshots().length, 0)
})

test('activation accepts only a pending candidate built on the current active snapshot', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())

  const baseline = store.createSnapshot({ evaluation: { passed: true, runtimeVerified: true } })
  store.activateSnapshot(baseline.id)
  const winner = store.createSnapshot({ label: 'Güncel aday', evaluation: { passed: true, runtimeVerified: true } })
  const stale = store.createSnapshot({ label: 'Aynı ebeveynden eski aday', evaluation: { passed: true, runtimeVerified: true } })

  store.activateSnapshot(winner.id)
  assert.throws(
    () => store.activateSnapshot(stale.id),
    (error) => error?.code === 'SNAPSHOT_PARENT_MISMATCH',
  )
  assert.equal(store.getActiveSnapshot().id, winner.id)
  assert.equal(store.getSnapshot(stale.id).status, 'candidate')
  assert.throws(
    () => store.activateSnapshot(baseline.id),
    (error) => error?.code === 'SNAPSHOT_NOT_CANDIDATE',
  )
})

test('explicit snapshot ids are idempotent only for the same immutable input frontier', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const input = {
    id: 'teacher-feed-candidate-32',
    sourceIds: [],
    artifactIds: [],
    inputManifestHash: 'manifest-a',
    evaluation: { passed: false },
  }
  const first = store.createSnapshot(input)
  const repeated = store.createSnapshot({ ...input, evaluation: { passed: false, retried: true } })
  assert.equal(repeated.id, first.id)
  assert.equal(store.listSnapshots().length, 1)
  assert.throws(
    () => store.createSnapshot({ ...input, inputManifestHash: 'manifest-b' }),
    { code: 'SNAPSHOT_IDEMPOTENCY_CONFLICT' },
  )
})

test('deselecting a source atomically rejects dependent candidates and blocks later stale activation', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({ text: 'Seçim değişikliği aday kaynağı.' }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  const candidate = store.createSnapshot({ sourceIds: [source.id], evaluation: { passed: true, runtimeVerified: true } })

  const originalPut = store.backend.put.bind(store.backend)
  store.backend.put = (bucket, id, value) => {
    if (bucket === 'snapshots' && id === candidate.id) {
      throw Object.assign(new Error('enjekte edilen snapshot yazma hatası'), { code: 'TEST_SNAPSHOT_WRITE_FAILURE' })
    }
    return originalPut(bucket, id, value)
  }
  try {
    assert.throws(() => store.setSourceSelected(source.id, false), { code: 'TEST_SNAPSHOT_WRITE_FAILURE' })
  } finally {
    store.backend.put = originalPut
  }
  assert.equal(store.getSource(source.id).selected, true, 'snapshot invalidasyonu yazılamazsa kaynak seçimi de geri alınmalı')
  assert.equal(store.getSnapshot(candidate.id).status, 'candidate')

  store.setSourceSelected(source.id, false)
  const rejected = store.getSnapshot(candidate.id)
  assert.equal(store.getSource(source.id).selected, false)
  assert.equal(rejected.status, 'rejected')
  assert.equal(rejected.evaluation.passed, false)
  assert.equal(rejected.evaluation.invalidated, true)
  assert.equal(rejected.evaluation.code, 'SOURCE_SELECTION_CHANGED')
  assert.equal(rejected.evaluation.selectionChange, 'DESELECTED')
  assert.match(rejected.evaluation.reason, /seçiminden çıkarıldı/u)

  const lateCandidate = store.createSnapshot({ sourceIds: [source.id], evaluation: { passed: true, runtimeVerified: true } })
  assert.throws(() => store.activateSnapshot(lateCandidate.id), { code: 'SNAPSHOT_SOURCE_STALE' })
})

test('activation and processed-source update roll back together on persistence failure', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({ text: 'Atomik etkinleştirme kaynağı.' }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  const candidate = store.createSnapshot({ sourceIds: [source.id], evaluation: { passed: true, runtimeVerified: true } })

  const originalSetMeta = store.backend.setMeta.bind(store.backend)
  store.backend.setMeta = (key, value) => {
    if (key === 'activeSnapshotId') throw Object.assign(new Error('enjekte edilen yazma hatası'), { code: 'TEST_WRITE_FAILURE' })
    return originalSetMeta(key, value)
  }
  try {
    assert.throws(() => store.activateSnapshot(candidate.id), (error) => error?.code === 'TEST_WRITE_FAILURE')
  } finally {
    store.backend.setMeta = originalSetMeta
  }

  assert.equal(store.getActiveSnapshot(), undefined)
  assert.equal(store.getSnapshot(candidate.id).status, 'candidate')
  assert.equal(store.getSource(source.id).processedSnapshotId, null)

  store.activateSnapshot(candidate.id)
  assert.equal(store.getActiveSnapshot().id, candidate.id)
  assert.equal(store.getSource(source.id).processedSnapshotId, candidate.id)
})

test('controlled snapshot activation and producer job success commit atomically', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({ text: 'Atomik controlled commit kaynağı.' }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 2 })
  const claim = store.claimNextJob('controlled-worker', 30_000, 60_000)
  const candidate = store.createClaimedSnapshot(job.id, claim.leaseOwner, claim.attempt, {
    status: 'candidate',
    sourceIds: [source.id],
    evaluation: { passed: true, runtimeVerified: true },
  })
  const result = { snapshotId: candidate.id, activationSnapshotId: candidate.id, committed: true }

  assert.throws(
    () => store.completeClaimedJobWithSnapshot(job.id, claim.leaseOwner, claim.attempt, candidate.id, result),
    { code: 'SNAPSHOT_ACTIVATION_GUARD_REQUIRED' },
  )
  assert.throws(
    () => store.completeClaimedJobWithSnapshot(
      job.id,
      claim.leaseOwner,
      claim.attempt,
      candidate.id,
      result,
      () => ({ eligible: false, code: 'TEST_POLICY_CHANGED' }),
    ),
    { code: 'TEST_POLICY_CHANGED' },
  )
  assert.equal(store.getJob(job.id).state, 'running')
  assert.equal(store.getSnapshot(candidate.id).status, 'candidate')
  assert.equal(store.getActiveSnapshot(), undefined)

  const originalPut = store.backend.put.bind(store.backend)
  store.backend.put = (bucket, id, value) => {
    if (bucket === 'jobs' && id === job.id && value.state === 'succeeded') {
      throw Object.assign(new Error('enjekte edilen terminal job yazma hatası'), { code: 'TEST_TERMINAL_WRITE_FAILURE' })
    }
    return originalPut(bucket, id, value)
  }
  try {
    assert.throws(
      () => store.completeClaimedJobWithSnapshot(job.id, claim.leaseOwner, claim.attempt, candidate.id, result, () => true),
      { code: 'TEST_TERMINAL_WRITE_FAILURE' },
    )
  } finally {
    store.backend.put = originalPut
  }
  assert.equal(store.getJob(job.id).state, 'running')
  assert.equal(store.getSnapshot(candidate.id).status, 'candidate')
  assert.equal(store.getActiveSnapshot(), undefined)
  assert.equal(store.getSource(source.id).processedSnapshotId, null)

  const completed = store.completeClaimedJobWithSnapshot(job.id, claim.leaseOwner, claim.attempt, candidate.id, result, () => true)
  assert.equal(completed.job.state, 'succeeded')
  assert.equal(completed.job.result.snapshotId, candidate.id)
  assert.equal(completed.snapshot.status, 'active')
  assert.equal(store.getActiveSnapshot().id, candidate.id)
  assert.equal(store.getSource(source.id).processedSnapshotId, candidate.id)
})

test('claimed release snapshot creation is fenced by the exact active snapshot and head', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const head = store.createArtifact({
    kind: 'test-release-head',
    contentHash: 'test-release-head-content',
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: { fingerprint: 'test-release-head-fingerprint' }, value: {},
    },
  })
  const active = store.createSnapshot({
    id: 'test-release-active-base', artifactIds: [head.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(active.id)
  const firstJob = store.createJob({ id: 'test-release-cas-success', type: 'test-release' }).job
  const firstClaim = store.claimNextJob('test-release-owner', 30_000, 60_000)
  const candidate = store.createClaimedSnapshotOnSnapshotHeadBase(
    firstJob.id, firstClaim.leaseOwner, firstClaim.attempt,
    { id: 'test-release-candidate', parentId: active.id, status: 'candidate', artifactIds: [head.id], sourceIds: [] },
    {
      snapshotId: active.id,
      headKind: head.kind,
      headArtifactId: head.id,
      headContentHash: head.contentHash,
      headFingerprint: 'test-release-head-fingerprint',
      staleCode: 'STALE_REGION_ATLAS_BASE',
    },
  )
  assert.equal(candidate.status, 'candidate')
  assert.equal(candidate.producerJobId, firstJob.id)
  assert.equal(store.getActiveSnapshot().id, active.id, 'publishing a release candidate must never activate it')

  store.updateJob(firstJob.id, { state: 'failed', stage: 'failed', finishedAt: new Date().toISOString() })
  const secondJob = store.createJob({ id: 'test-release-cas-stale', type: 'test-release' }).job
  const secondClaim = store.claimNextJob('test-release-owner', 30_000, 60_000)
  assert.throws(() => store.createClaimedSnapshotOnSnapshotHeadBase(
    secondJob.id, secondClaim.leaseOwner, secondClaim.attempt,
    { id: 'test-release-stale-candidate', parentId: active.id, status: 'candidate', artifactIds: [], sourceIds: [] },
    {
      snapshotId: active.id,
      headKind: head.kind,
      headArtifactId: head.id,
      headContentHash: head.contentHash,
      headFingerprint: 'fabricated-head-fingerprint',
      staleCode: 'STALE_REGION_ATLAS_BASE',
    },
  ), (error) => error?.code === 'STALE_REGION_ATLAS_BASE' && error.details?.field === 'headFingerprint')
  assert.equal(store.getSnapshot('test-release-stale-candidate'), undefined)
})

test('candidate-only release publication and producer success commit atomically without activation', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const head = store.createArtifact({
    kind: 'test-atomic-candidate-head', contentHash: 'test-atomic-candidate-head-content',
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: { fingerprint: 'test-atomic-candidate-head-fingerprint' }, value: {},
    },
  })
  const active = store.createSnapshot({
    id: 'test-atomic-candidate-active', artifactIds: [head.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(active.id)
  const job = store.createJob({ id: 'test-atomic-candidate-job', type: 'plastic-region-release-evaluation' }).job
  const claim = store.claimNextJob('test-atomic-candidate-owner', 30_000, 60_000)
  const input = {
    id: 'test-atomic-candidate', parentId: active.id, status: 'candidate', artifactIds: [head.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  }
  const expectedBase = {
    snapshotId: active.id, headKind: head.kind, headArtifactId: head.id,
    headContentHash: head.contentHash, headFingerprint: 'test-atomic-candidate-head-fingerprint',
  }
  const result = { snapshotId: input.id, released: true, automaticActivation: false }

  assert.throws(() => store.completeClaimedJobWithCandidateOnSnapshotHeadBase(
    job.id, claim.leaseOwner, claim.attempt, input, expectedBase, result,
    () => ({ eligible: false, code: 'TEST_CANDIDATE_GATE_REJECTED' }),
  ), { code: 'TEST_CANDIDATE_GATE_REJECTED' })
  assert.equal(store.getSnapshot(input.id), undefined)
  assert.equal(store.getJob(job.id).state, 'running')

  const originalPut = store.backend.put.bind(store.backend)
  store.backend.put = (bucket, id, value) => {
    if (bucket === 'jobs' && id === job.id && value.state === 'succeeded') {
      throw Object.assign(new Error('enjekte edilen aday terminal yazma hatası'), { code: 'TEST_CANDIDATE_TERMINAL_FAILURE' })
    }
    return originalPut(bucket, id, value)
  }
  try {
    assert.throws(() => store.completeClaimedJobWithCandidateOnSnapshotHeadBase(
      job.id, claim.leaseOwner, claim.attempt, input, expectedBase, result, () => true,
    ), { code: 'TEST_CANDIDATE_TERMINAL_FAILURE' })
  } finally {
    store.backend.put = originalPut
  }
  assert.equal(store.getSnapshot(input.id), undefined, 'terminal failure must roll the candidate back')
  assert.equal(store.getJob(job.id).state, 'running')

  const completed = store.completeClaimedJobWithCandidateOnSnapshotHeadBase(
    job.id, claim.leaseOwner, claim.attempt, input, expectedBase, result, () => true,
  )
  assert.equal(completed.snapshot.status, 'candidate')
  assert.equal(completed.job.state, 'succeeded')
  assert.deepEqual(completed.job.result, result)
  assert.equal(store.getActiveSnapshot().id, active.id)
})

test('rollback atomically restores the source frontier and reopens later sources', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const first = store.createSource(normalizeTextSource({ text: 'Birinci geri alma kaynağı.' }, store.getConfig())).source
  const later = store.createSource(normalizeTextSource({ text: 'İkinci ve daha yeni geri alma kaynağı.' }, store.getConfig())).source
  store.decideSource(first.id, 'approved')
  store.decideSource(later.id, 'approved')

  const target = store.createSnapshot({ sourceIds: [first.id], evaluation: { passed: true, runtimeVerified: true } })
  store.activateSnapshot(target.id)
  const newer = store.createSnapshot({ sourceIds: [first.id, later.id], evaluation: { passed: true, runtimeVerified: true } })
  store.activateSnapshot(newer.id)
  assert.equal(store.getSource(first.id).processedSnapshotId, newer.id)
  assert.equal(store.getSource(later.id).processedSnapshotId, newer.id)

  const snapshotCount = store.listSnapshots().length
  const originalSetMeta = store.backend.setMeta.bind(store.backend)
  store.backend.setMeta = (key, value) => {
    if (key === 'activeSnapshotId') throw Object.assign(new Error('enjekte edilen geri alma hatası'), { code: 'TEST_ROLLBACK_FAILURE' })
    return originalSetMeta(key, value)
  }
  try {
    assert.throws(() => store.rollbackToSnapshot(target.id), (error) => error?.code === 'TEST_ROLLBACK_FAILURE')
  } finally {
    store.backend.setMeta = originalSetMeta
  }

  assert.equal(store.listSnapshots().length, snapshotCount, 'başarısız işlem yetim geri alma adayı bırakmamalı')
  assert.equal(store.getActiveSnapshot().id, newer.id)
  assert.equal(store.getSnapshot(newer.id).status, 'active')
  assert.equal(store.getSnapshot(target.id).status, 'superseded')
  assert.equal(store.getSource(first.id).processedSnapshotId, newer.id)
  assert.equal(store.getSource(later.id).processedSnapshotId, newer.id)

  const restored = store.rollbackToSnapshot(target.id)
  assert.equal(store.getActiveSnapshot().id, restored.id)
  assert.equal(store.getSource(first.id).processedSnapshotId, restored.id)
  assert.equal(store.getSource(later.id).processedSnapshotId, null, 'hedefte olmayan kaynak yeniden eğitime açılmalı')
})

test('a TextLab representation upgrade invalidates old candidates and stays unprocessed across rollback', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const raw = normalizeTextSource({ text: 'Temsil kanıtı sonradan güçlenen kaynak.' }, store.getConfig())
  const source = store.createSource(raw).source
  store.decideSource(source.id, 'approved')

  const target = store.createSnapshot({ sourceIds: [source.id], evaluation: { passed: true, runtimeVerified: true } })
  store.activateSnapshot(target.id)
  const awaiting = store.createSnapshot({ sourceIds: [source.id], evaluation: { passed: true, runtimeVerified: true } })
  const newer = store.createSnapshot({ evaluation: { passed: true, runtimeVerified: true } })
  store.activateSnapshot(newer.id)

  store.createSource({
    ...raw,
    provenance: {
      ...raw.provenance,
      textLabManifest: {
        format: 'text-lab-recompute-proof-v1',
        sourceDigest: '0123456789abcdef',
        representationSpaceId: 'RS-0123456789ABCDEF',
        config: { matrixSide: 64 },
        identityProofs: [{ start: 0, end: raw.content.length, identityDigest: 'fedcba9876543210' }],
      },
    },
  })

  assert.equal(store.getSnapshot(awaiting.id).status, 'rejected')
  assert.equal(store.getSnapshot(awaiting.id).evaluation.code, 'SOURCE_REPRESENTATION_CHANGED')
  assert.equal(store.getSource(source.id).status, 'quarantine')
  assert.equal(store.getSource(source.id).processedSnapshotId, null)
  assert.throws(() => store.activateSnapshot(awaiting.id), { code: 'SNAPSHOT_NOT_CANDIDATE' })

  const restored = store.rollbackToSnapshot(target.id)
  assert.equal(restored.status, 'active')
  assert.equal(store.getSource(source.id).processedSnapshotId, null, 'old representation must not mark the upgraded source processed')
  store.decideSource(source.id, 'approved')
  assert.equal(store.getSource(source.id).processedSnapshotId, null, 're-approved upgraded input must remain eligible for retraining')

  const lateStaleCandidate = store.createSnapshot({
    sourceIds: [source.id],
    sourceRevisions: target.sourceRevisions,
    evaluation: { passed: true, runtimeVerified: true },
  })
  assert.throws(() => store.activateSnapshot(lateStaleCandidate.id), { code: 'SNAPSHOT_SOURCE_STALE' })
})

test('implicit queued work keeps approved sources representation-locked until its corpus is resolved', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const raw = normalizeTextSource({ text: 'Örtük iş tarafından kilitlenen kaynak.' }, store.getConfig())
  const source = store.createSource(raw).source
  store.decideSource(source.id, 'approved')
  store.setSourceSelected(source.id, false)
  store.createJob({ payload: {} })

  assert.throws(() => store.createSource({
    ...raw,
    provenance: {
      ...raw.provenance,
      textLabManifest: {
        format: 'text-lab-recompute-proof-v1',
        sourceDigest: '0123456789abcdef',
        representationSpaceId: 'RS-0123456789ABCDEF',
        config: { matrixSide: 64 },
        identityProofs: [{ start: 0, end: raw.content.length, identityDigest: 'fedcba9876543210' }],
      },
    },
  }), { code: 'SOURCE_BUSY' })
})

test('atomic JSON backend recovers after a commit flush failure and persists later writes', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  let store = await createTrainingStore({ dataDirectory, forceJson: true })
  const realFlush = store.backend.flush.bind(store.backend)
  let failOnce = true
  store.backend.flush = () => {
    if (failOnce) {
      failOnce = false
      throw Object.assign(new Error('Enjekte edilmiş disk yazma hatası.'), { code: 'TEST_FLUSH_FAILURE' })
    }
    return realFlush()
  }

  assert.throws(() => store.createJob({ id: 'must-rollback' }), { code: 'TEST_FLUSH_FAILURE' })
  assert.equal(store.backend.transactionDepth, 0)
  assert.equal(store.getJob('must-rollback'), undefined)
  store.createJob({ id: 'must-persist' })
  const trackedBytes = store.approximateStorageBytes()
  assert.ok(trackedBytes > 0)
  store.backend.flush = realFlush
  store.close()

  store = await createTrainingStore({ dataDirectory, forceJson: true })
  t.after(() => store.close())
  assert.equal(store.getJob('must-rollback'), undefined)
  assert.equal(store.getJob('must-persist').id, 'must-persist')
  assert.equal(store.approximateStorageBytes(), trackedBytes)
})
