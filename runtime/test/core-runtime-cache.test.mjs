import test from 'node:test'
import assert from 'node:assert/strict'
import {
  createFamilyModelLineage,
  createFamilyRunCacheKey,
  createFamilyContinuityState,
  readActiveFamilyContinuity,
  readActiveRuntimeEnvelope,
  readActiveRuntimeContext,
} from '../family-cache.mjs'
import { persistArtifact } from '../artifacts.mjs'
import { createTrainingStore } from '../../server/store.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'

function cacheInput() {
  return {
    runtimeVersion: 'test-runtime-v1',
    analyses: [
      { runId: 'run-b', sourceDigest: 'source-b' },
      { runId: 'run-a', sourceDigest: 'source-a' },
    ],
    synthesisDecisions: [{
      id: 'decision-1',
      format: 'noron-sentez-karari',
      version: 1,
      candidateId: 'candidate-1',
      candidateFingerprint: 'candidate-fingerprint-1',
      decision: 'approved',
      reviewedAt: '2026-08-29T10:00:00.000Z',
      runId: 'synthesis-run-1',
      runFingerprint: 'synthesis-fingerprint-1',
      representationSpaceId: 'representation-1',
      analysisRunIds: ['run-a', 'run-b'],
      sourceDigests: ['source-a', 'source-b'],
      config: { minimumSupport: 2 },
      candidate: {
        fingerprint: 'candidate-fingerprint-1',
        identityDigest: 'identity-1',
        relation: new Float32Array([0.25, -0.5]),
        independentSupports: [{ key: 'support-a' }, { key: 'support-b' }],
        medoid: { text: 'örnek birleşim' },
      },
    }],
    config: { minimumSources: 2 },
    previousRun: {
      id: 'family-run-a',
      fingerprint: 'family-fingerprint-a',
      families: [{ id: 'family-a', lifecycle: 'provisional' }],
      coalition: { step: 3, coalitions: [] },
    },
    lineage: {
      activeSnapshotId: 'snapshot-a',
      parentSnapshotId: null,
      rollbackOfSnapshotId: null,
      runtimeModelArtifactId: 'artifact-a',
      runtimeModelContentHash: 'model-a',
      familyRunFingerprint: 'family-fingerprint-a',
    },
  }
}

test('family cache key covers the complete reviewed synthesis decision and previous family state', () => {
  const input = cacheInput()
  const baseline = createFamilyRunCacheKey(input)

  assert.notEqual(createFamilyRunCacheKey({
    ...input,
    synthesisDecisions: [{ ...input.synthesisDecisions[0], reviewedAt: '2026-08-29T10:01:00.000Z' }],
  }), baseline, 'reviewedAt must invalidate the family artifact')

  assert.notEqual(createFamilyRunCacheKey({
    ...input,
    synthesisDecisions: [{
      ...input.synthesisDecisions[0],
      candidate: { ...input.synthesisDecisions[0].candidate, independentSupports: [{ key: 'support-c' }] },
    }],
  }), baseline, 'candidate evidence used by behavior profiles must invalidate the artifact')

  assert.notEqual(createFamilyRunCacheKey({
    ...input,
    previousRun: { ...input.previousRun, coalition: { step: 4, coalitions: [] } },
  }), baseline, 'previous FamilyRun state must invalidate the developmental result')

  assert.equal(createFamilyRunCacheKey({
    ...input,
    analyses: [...input.analyses].reverse(),
  }), baseline, 'equivalent analysis ordering should remain cacheable')

  assert.notEqual(createFamilyRunCacheKey({
    ...input,
    analyses: input.analyses.map((analysis, index) => ({
      ...analysis,
      independenceGroupId: index === 0 ? 'lineage-a' : 'lineage-b',
    })),
  }), baseline, 'source independence provenance must invalidate the family artifact')

  const splitBaseline = createFamilyRunCacheKey({
    ...input,
    discoveryIndependenceGroupIds: ['train-b', 'train-a'],
    evaluationIndependenceGroupIds: ['evaluation-a'],
  })
  assert.equal(createFamilyRunCacheKey({
    ...input,
    discoveryIndependenceGroupIds: ['train-a', 'train-b'],
    evaluationIndependenceGroupIds: ['evaluation-a'],
  }), splitBaseline, 'equivalent split declaration ordering should remain cacheable')
  assert.notEqual(splitBaseline, baseline, 'a split-aware request must not reuse a legacy all-source artifact')
  assert.notEqual(createFamilyRunCacheKey({
    ...input,
    discoveryIndependenceGroupIds: ['train-a'],
    evaluationIndependenceGroupIds: ['evaluation-a', 'train-b'],
  }), splitBaseline, 'moving a lineage from discovery to evaluation must invalidate the family artifact')
})

test('rollback creates a new model lineage and cannot reuse the abandoned branch family cache', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())

  const stable = store.createSnapshot({ id: 'stable-snapshot', status: 'candidate', artifactIds: [] })
  store.activateSnapshot(stable.id)
  const stableLineage = createFamilyModelLineage(store)
  const abandoned = store.createSnapshot({ id: 'abandoned-snapshot', status: 'candidate', artifactIds: [] })
  store.activateSnapshot(abandoned.id)
  const restored = store.rollbackToSnapshot(stable.id)
  const rollbackLineage = createFamilyModelLineage(store)

  assert.equal(rollbackLineage.activeSnapshotId, restored.id)
  assert.equal(rollbackLineage.parentSnapshotId, abandoned.id)
  assert.equal(rollbackLineage.rollbackOfSnapshotId, abandoned.id)
  assert.notDeepEqual(rollbackLineage, stableLineage)

  const input = cacheInput()
  assert.notEqual(
    createFamilyRunCacheKey({ ...input, lineage: rollbackLineage }),
    createFamilyRunCacheKey({ ...input, lineage: stableLineage }),
    'a rollback branch must not resurrect a family artifact cached before the abandoned branch',
  )
})

test('runtime model v2 hydrates its immutable FamilyRun reference without duplicating it', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const context = { store, jobId: null }
  const familyRun = {
    format: 'noron-aileleri-run',
    version: 1,
    fingerprint: 'family-reference-proof',
    families: [],
    memberships: [],
    profiles: [],
    neuronTransitions: [],
  }
  const familyArtifact = persistArtifact(context, {
    kind: 'family-run',
    contentHash: 'family-reference-hash',
    value: familyRun,
  }).artifact
  const runtimeArtifact = persistArtifact(context, {
    kind: 'runtime-model-state',
    contentHash: 'runtime-reference-hash',
    value: {
      format: 'bombom-runtime-model',
      version: 2,
      familyRunArtifactId: familyArtifact.id,
      familyRunFingerprint: familyRun.fingerprint,
      languageState: { fingerprint: 'language-proof' },
    },
  }).artifact
  const snapshot = store.createSnapshot({ status: 'candidate', artifactIds: [familyArtifact.id, runtimeArtifact.id] })
  store.activateSnapshot(snapshot.id)

  assert.equal(store.getArtifact(runtimeArtifact.id).payload.value.familyRun, undefined)
  const resolved = readActiveRuntimeContext(store)
  assert.deepEqual(resolved.model.familyRun, familyRun)
  assert.equal(createFamilyModelLineage(store).familyRunFingerprint, familyRun.fingerprint)
})

test('compact family continuity excludes the previous full profile and transition graph', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const context = { store, jobId: null }
  const familyRun = {
    id: 'family-run-large', fingerprint: 'family-continuity-proof', behaviorSpaceId: 'behavior-space',
    inputManifest: { entries: [], corpusFingerprint: 'corpus-a' },
    families: [{ id: 'family-a', behaviorSpaceId: 'behavior-space', lifecycle: 'provisional', memberKeys: ['n-1'] }],
    coalition: { step: 2, coalitions: [] },
    profiles: Array.from({ length: 10_000 }, (_, index) => ({ memberKey: `n-${index}` })),
    neuronTransitions: Array.from({ length: 20_000 }, (_, index) => ({ fromMemberKey: `n-${index}`, toMemberKey: `n-${index + 1}` })),
  }
  const familyArtifact = persistArtifact(context, { kind: 'family-run', contentHash: 'large-family-hash', value: familyRun }).artifact
  const continuityArtifact = persistArtifact(context, {
    kind: 'family-continuity-state', contentHash: 'continuity-hash', value: createFamilyContinuityState(familyRun),
  }).artifact
  const snapshot = store.createSnapshot({ status: 'candidate', artifactIds: [familyArtifact.id, continuityArtifact.id] })
  store.activateSnapshot(snapshot.id)

  const continuity = readActiveFamilyContinuity(store)
  assert.equal(continuity.fingerprint, familyRun.fingerprint)
  assert.deepEqual(continuity.families, familyRun.families)
  assert.equal(continuity.profiles, undefined)
  assert.equal(continuity.neuronTransitions, undefined)
  assert.equal(createFamilyModelLineage(store).familyRunContentHash, familyArtifact.contentHash)
})

test('runtime envelope metadata gate skips oversized legacy JSON before parsing it', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const runtimeArtifact = persistArtifact({ store, jobId: null }, {
    kind: 'runtime-model-state', contentHash: 'oversized-runtime-proof',
    value: { format: 'bombom-runtime-model', version: 1, legacyPayload: 'x'.repeat(32_000) },
  }).artifact
  const snapshot = store.createSnapshot({ status: 'candidate', artifactIds: [runtimeArtifact.id] })
  store.activateSnapshot(snapshot.id)

  assert.equal(readActiveRuntimeEnvelope(store, 1_024), null)
  assert.equal(readActiveRuntimeEnvelope(store, 64_000).format, 'bombom-runtime-model')
})
