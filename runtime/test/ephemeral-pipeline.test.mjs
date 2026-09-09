import assert from 'node:assert/strict'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { build } from 'vite'
import { createTrainingStore } from '../../server/store.mjs'
import { normalizeTextSource } from '../../server/source-policy.mjs'
import { runEphemeralPipeline } from '../ephemeral-pipeline-runner.mjs'
import { restoreArtifact } from '../artifacts.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')

function processExists(pid) {
  try { process.kill(pid, 0); return true } catch (error) {
    if (error?.code === 'ESRCH') return false
    throw error
  }
}

test('successive teacher milestones use distinct disposable pipeline processes', { timeout: 30_000 }, async (t) => {
  const testRoot = await temporaryDirectory(t)
  const outputDirectory = path.join(testRoot, 'bundle')
  await build({
    configFile: path.join(repositoryRoot, 'vite.training.config.ts'),
    build: { outDir: outputDirectory, emptyOutDir: true },
    logLevel: 'silent',
  })
  const pipelineWorkerEntry = path.join(outputDirectory, 'pipeline-worker.mjs')
  const store = await createTrainingStore({ dataDirectory: path.join(testRoot, 'data'), requireSqlite: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({
    title: 'İki kilometre taşı izolasyon kanıtı',
    text: 'Ali topu masaya bıraktı. Ayşe topu masadan aldı. Çocuk iki olayı sırayla anlattı.',
  }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  const queued = store.createJob({ type: 'teacher-feed-campaign', maxAttempts: 2 }).job
  const claimed = store.claimNextJob('ephemeral-test-owner', 30_000, 5 * 60_000)
  assert.equal(claimed.id, queued.id)
  const controller = new AbortController()
  const events = []
  const context = {
    jobId: claimed.id,
    config: store.getConfig(),
    signal: controller.signal,
    emit: async (event) => { events.push(structuredClone(event)); return event },
    progress: async () => {},
  }

  for (let milestone = 1; milestone <= 2; milestone += 1) {
    const result = await runEphemeralPipeline({
      workerEntry: pipelineWorkerEntry,
      dataDirectory: store.dataDirectory,
      pipelineJob: {
        id: `${claimed.id}:proposal:${milestone}`,
        type: 'full-cycle',
        payload: {
          sourceIds: [source.id],
          resolvedSourceIds: [source.id],
          reprocess: true,
          automationLevel: 'proposal',
          candidateSnapshotId: `ephemeral-test-snapshot-${milestone}`,
          teacherTrainingMilestone: milestone,
        },
      },
      context,
      owner: claimed.leaseOwner,
      attempt: claimed.attempt,
    })
    assert.equal(result.snapshotId, `ephemeral-test-snapshot-${milestone}`)
    const snapshot = store.getSnapshot(result.snapshotId)
    const artifactKinds = new Set(snapshot.artifactIds.map((id) => store.getArtifactMetadata(id)?.kind))
    assert.equal(artifactKinds.has('concept-macro-observation-batch'), true)
    assert.equal(artifactKinds.has('concept-macro-head'), true)
    const familyArtifactId = snapshot.artifactIds.find((id) => store.getArtifactMetadata(id)?.kind === 'family-run')
    const familyRun = restoreArtifact(store.getArtifact(familyArtifactId), 'family-run')
    assert.equal(familyRun.inputManifest.dataSplit?.mode, 'declared-independent-groups')
    assert.deepEqual(familyRun.inputManifest.dataSplit?.evaluationIndependenceGroupIds, [])
    assert.ok(familyRun.inputManifest.dataSplit?.discoveryIndependenceGroupIds.length >= 1)
  }

  const started = events.filter((event) => event.type === 'pipeline-worker-spawned')
  const released = events.filter((event) => event.type === 'pipeline-worker-released')
  assert.equal(started.length, 2)
  assert.equal(released.length, 2)
  assert.notEqual(started[0].data.pid, started[1].data.pid)
  for (const [index, event] of released.entries()) {
    assert.equal(event.data.pid, started[index].data.pid)
    assert.equal(event.data.memoryReleased, true)
    assert.ok(event.data.memory.peakWorkerRssBytes > 0)
    assert.equal(processExists(event.data.pid), false)
  }
})
