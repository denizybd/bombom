import test from 'node:test'
import assert from 'node:assert/strict'
import { toPublicSnapshot, toPublicStatus, toPublicTeacherFeedCampaign } from '../public-api.mjs'
import { createTrainingStore } from '../store.mjs'
import { temporaryDirectory } from './helpers.mjs'

function stage(status, id) {
  return status.stages.find((item) => item.id === id)
}

test('plastic region work keeps its own public stage between language and concepts', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const { job } = store.createJob({ id: 'plastic-stage-job', type: 'full-cycle', payload: {} })
  store.updateJob(job.id, { state: 'running', stage: 'plasticRegions', progress: 0.4 })

  const status = toPublicStatus(store, { available: true, models: [] })
  const ids = status.stages.map((item) => item.id)
  assert.equal(status.activeStage, 'plastic_regions')
  assert.equal(stage(status, 'plastic_regions').state, 'active')
  assert.ok(ids.indexOf('language_experience') < ids.indexOf('plastic_regions'))
  assert.ok(ids.indexOf('plastic_regions') < ids.indexOf('concept_macros'))
})

test('completed proposal shows evaluated work but not activation or rollback as complete', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())

  const candidate = store.createSnapshot({
    id: 'proposal-candidate',
    status: 'candidate',
    evaluation: { passed: true, runtimeVerified: true },
  })
  const { job } = store.createJob({ id: 'proposal-job', type: 'full-cycle', payload: { automationLevel: 'proposal' } })
  store.updateJob(job.id, {
    state: 'succeeded',
    stage: 'completed',
    progress: 1,
    result: { snapshotId: candidate.id, committed: false, awaitingApproval: true },
  })

  let status = toPublicStatus(store, { available: true, models: [] })
  assert.equal(status.state, 'idle')
  assert.ok(status.warnings.some((warning) => warning.code === 'JSON_STORAGE_FALLBACK'))
  assert.equal(stage(status, 'evaluate').state, 'complete')
  assert.equal(stage(status, 'commit').state, 'pending')
  assert.match(stage(status, 'commit').message, /onayını bekliyor/u)
  assert.equal(stage(status, 'rollback').state, 'pending')
  assert.equal(stage(status, 'rollback').progress, 0)

  store.activateSnapshot(candidate.id)
  status = toPublicStatus(store, { available: true, models: [] })
  assert.equal(stage(status, 'commit').state, 'complete', 'manual acceptance should be visible after it really happens')
  assert.equal(stage(status, 'rollback').state, 'pending')

  const restored = store.rollbackToSnapshot(candidate.id)
  assert.ok(restored)
  status = toPublicStatus(store, { available: true, models: [] })
  assert.equal(stage(status, 'commit').state, 'complete')
  assert.equal(stage(status, 'rollback').state, 'rolled_back')
  assert.equal(stage(status, 'rollback').progress, 1)
})

test('an invalidated proposal is shown as skipped even when its old job result said awaiting approval', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const candidate = store.createSnapshot({
    status: 'candidate',
    evaluation: { passed: true, runtimeVerified: true },
  })
  const { job } = store.createJob({ id: 'invalidated-proposal-job', type: 'full-cycle' })
  store.updateJob(job.id, {
    state: 'succeeded',
    stage: 'completed',
    progress: 1,
    result: { snapshotId: candidate.id, committed: false, awaitingApproval: true },
  })
  store.rejectSnapshot(candidate.id, {
    passed: false,
    runtimeVerified: true,
    invalidated: true,
    code: 'SOURCE_REPRESENTATION_CHANGED',
  })

  const status = toPublicStatus(store, { available: true, models: [] })
  assert.equal(stage(status, 'commit').state, 'skipped')
  assert.match(stage(status, 'commit').message, /yeniden eğitim/u)
})

test('a real runtime rejection is described as insufficient evidence, not missing integration', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const runtimeArtifact = store.createArtifact({
    kind: 'runtime-model-state',
    payload: { format: 'bombom-runtime-artifact', version: 1, value: {} },
  })
  store.createSnapshot({
    id: 'real-runtime-rejection',
    status: 'rejected',
    artifactIds: [runtimeArtifact.id],
    evaluation: { passed: false, runtimeVerified: false },
  })

  const status = toPublicStatus(store, { available: true, models: [] })
  assert.ok(status.warnings.some((warning) => warning.code === 'RUNTIME_EVIDENCE_INSUFFICIENT'))
  assert.ok(!status.warnings.some((warning) => warning.code === 'RUNTIME_INTEGRATION_REQUIRED'))
})

test('status and snapshot projections use artifact metadata without opening payloads', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  t.after(() => store.close())
  const payload = { weights: 'x'.repeat(256 * 1024), families: [1, 2, 3] }
  const artifact = store.createArtifact({ kind: 'runtime-model-state', payload })
  const snapshot = store.createSnapshot({
    id: 'metadata-only-public-snapshot',
    status: 'rejected',
    artifactIds: [artifact.id],
    evaluation: { passed: false, runtimeVerified: false },
  })
  store.getArtifact = () => {
    throw new Error('public summary must not materialize an artifact payload')
  }

  const status = toPublicStatus(store, { available: true, models: [] })
  assert.ok(status.warnings.some((warning) => warning.code === 'RUNTIME_EVIDENCE_INSUFFICIENT'))
  assert.equal(toPublicSnapshot(snapshot, store).bytes, Buffer.byteLength(JSON.stringify(payload)))
})

test('teacher feed campaign projection exposes research counters without raw job internals', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const { job } = store.createJob({
    id: 'feed-public-1',
    type: 'teacher-feed-campaign',
    payload: {
      campaign: {
        model: 'teacher:14b', seed: 42, targetAccepted: 128,
        generationBatchSize: 16, trainEveryAccepted: 32,
        curriculum: 'turkish-foundations-v1', mode: 'research-only',
      },
      feedState: {
        generated: 20, accepted: 16, duplicates: 2, rejectedQuality: 2,
        teacherRequests: 2, teacherTokens: 900, completedTrainingCycles: 0,
        nextBatchIndex: 1, lastSnapshotId: null, stopReason: null,
      },
    },
  })
  const visible = toPublicTeacherFeedCampaign(job)
  assert.deepEqual({
    id: visible.id,
    status: visible.status,
    model: visible.model,
    accepted: visible.accepted,
    generated: visible.generated,
    teacherTokens: visible.teacherTokens,
  }, {
    id: 'feed-public-1', status: 'queued', model: 'teacher:14b', accepted: 16, generated: 20, teacherTokens: 900,
  })
  assert.equal('payload' in visible, false)
  assert.equal('modelIdentity' in visible, false)
  assert.equal(visible.continuous, null)
})

test('continuous teacher feed projection exposes bounded blocks and structural gate without raw evidence', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const { job } = store.createJob({
    id: 'feed-continuous-public',
    type: 'teacher-feed-campaign',
    payload: {
      campaign: {
        model: 'teacher:14b', seed: 9, targetAccepted: 512,
        generationBatchSize: 2, trainEveryAccepted: 32,
        curriculum: 'turkish-foundations-v1', mode: 'research-only',
        continuousPlan: {
          blockAccepted: 128, maxBlocks: 4, minimumDiskHeadroomBytes: 64 * 1024 * 1024,
          maximumSyntheticSources: 20_000,
          stopGate: { benchmarkFingerprint: 'benchmark-fingerprint' },
        },
      },
      feedState: {
        accepted: 128, completedTrainingCycles: 1, nextBatchIndex: 64,
        bootstrapSourceIds: ['source-a', 'source-b'],
        bootstrapSourceFingerprint: 'bootstrap-fingerprint',
        continuous: {
          completedBlocks: 1, nextBlockAt: 256, gatePassed: false, gateSnapshotId: null,
          lastGate: {
            status: 'failed', snapshotId: 'proposal-1',
            checks: [{ id: 'holdout-graph-f1', passed: false }, { id: 'no-ollama-evaluator', passed: true }],
          },
        },
      },
    },
  })
  const visible = toPublicTeacherFeedCampaign(job)
  assert.deepEqual(visible.continuous, {
    blockAccepted: 128,
    maximumBlocks: 4,
    completedBlocks: 1,
    nextBlockAt: 256,
    minimumDiskHeadroomMb: 64,
    maximumSyntheticSources: 20_000,
    bootstrapSourceCount: 2,
    bootstrapSourceFingerprint: 'bootstrap-fingerprint',
    stopGate: {
      required: true,
      passed: false,
      status: 'failed',
      valid: false,
      conclusive: false,
      completion: null,
      evaluationKind: null,
      diagnosticMeanGraphF1: null,
      holdoutGraphF1: null,
      holdoutGraphF1Low: null,
      holdoutGraphF1High: null,
      snapshotId: 'proposal-1',
      benchmarkFingerprint: 'benchmark-fingerprint',
      failedChecks: ['holdout-graph-f1'],
    },
  })
  assert.equal('bootstrapSourceIds' in visible.continuous, false)
  assert.equal('blockHistory' in visible.continuous, false)
})
