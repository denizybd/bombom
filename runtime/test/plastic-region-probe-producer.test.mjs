import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer as createViteServer } from 'vite'
import { encodeArtifactValue } from '../artifacts.mjs'
import { createTrainingStore } from '../../server/store.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'

async function loadModules(t) {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  return Promise.all([
    vite.ssrLoadModule('/runtime/plastic-region-probe-producer.mjs'),
    vite.ssrLoadModule('/src/adaptive-regions/index.ts'),
    vite.ssrLoadModule('/server/plastic-region-semantic-inspection.mjs'),
  ])
}

function activeAtlas(regions) {
  const evidence = {
    observationCount: 8,
    sourceGroupIds: ['source-a', 'source-b'],
    confidenceMass: 2,
    cohesion: 0.9,
    holdoutUtility: 0.1,
    shamUtility: 0,
  }
  return regions.createPlasticRegionAtlas({
    representationSpaceId: 'RS-SEMANTIC-PRODUCER',
    behaviorDimension: regions.PAGE_BEHAVIOR_DIMENSION,
    maximumMembershipsPerPage: 3,
    regions: [
      {
        id: 'opaque-region-a', lifecycle: 'active', origin: 'discovery', bornRevision: 0,
        retiredRevision: null, parentRegionIds: [],
        prototype: { dimension: 8, values: [1, 0, 0, 0, 0, 0, 0, 0] }, evidence,
      },
      {
        id: 'opaque-region-b', lifecycle: 'active', origin: 'discovery', bornRevision: 0,
        retiredRevision: null, parentRegionIds: [],
        prototype: { dimension: 8, values: [0, 1, 0, 0, 0, 0, 0, 0] }, evidence,
      },
    ],
    pages: Array.from({ length: regions.ADAPTIVE_PAGE_COUNT }, (_, pageIndex) => pageIndex < 2
      ? {
          pageIndex,
          memberships: [
            { regionId: 'opaque-region-a', weight: 0.5 },
            { regionId: 'opaque-region-b', weight: 0.5 },
          ],
          freeMass: 0,
        }
      : { pageIndex, memberships: [], freeMass: 1 }),
  })
}

function attachAtlas(store, atlas) {
  const artifact = store.createArtifact({
    id: 'semantic-producer-atlas',
    kind: 'plastic-region-atlas-head',
    contentHash: 'semantic-producer-atlas-content',
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      bytes: 1,
      metadata: {
        representationSpaceId: atlas.representationSpaceId,
        fingerprint: atlas.fingerprint,
        geometryFingerprint: atlas.geometryFingerprint,
        revision: atlas.revision,
        behaviorDimension: atlas.behaviorDimension,
        maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
      },
      value: encodeArtifactValue(atlas),
    },
  })
  const snapshot = store.createSnapshot({
    id: 'semantic-producer-snapshot',
    artifactIds: [artifact.id],
    sourceIds: [],
    inputManifestHash: 'semantic-producer-input-manifest',
    evaluation: { passed: true, runtimeVerified: true },
  })
  return { artifact, snapshot }
}

function shadowResult(atlas, benchmarkCase) {
  const actorAda = benchmarkCase.expected.actor === 'Ada'
  const target = actorAda ? 0.75 : 0.05
  const sham = 0.2
  return {
    ticks: [{
      tick: 1,
      branches: [{
        branchId: 'branch-a',
        mass: 1,
        adaptiveRegion: {
          format: 'thought-adaptive-region-branch-telemetry',
          version: 1,
          mode: 'shadow',
          activeApplied: false,
          atlasFingerprint: atlas.fingerprint,
          gates: {
            gates: [
              { regionId: 'opaque-region-a', probability: target },
              { regionId: 'opaque-region-b', probability: sham },
            ],
            unmatchedProbability: 1 - target - sham,
            omittedRegionProbability: 0,
          },
        },
      }],
    }],
  }
}

test('real deterministic producer persists server-owned spec, train calibration, and isolated validation/test evidence', async (t) => {
  const [producer, regions, inspection] = await loadModules(t)
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const atlas = activeAtlas(regions)
  const { snapshot } = attachAtlas(store, atlas)
  const requests = []
  const context = {
    store,
    jobId: null,
    checkpoint: async () => {},
    progress: async () => {},
  }
  const runCase = async ({ benchmarkCase, request, atlas: requestedAtlas }) => {
    requests.push({ benchmarkCase, request })
    assert.equal(requestedAtlas.fingerprint, atlas.fingerprint)
    return shadowResult(atlas, benchmarkCase)
  }
  const first = await producer.produceAndPersistStructuralSemanticProbes({
    context,
    snapshot,
    runCase,
    evaluationPolicy: { bootstrapSamples: 200 },
  })

  assert.equal(first.status, 'calibrated')
  assert.equal(first.measuredCaseCount, 288)
  assert.equal(first.activeFeatureCount, 2)
  assert.equal(first.trainingWritesAllowed, false)
  assert.equal(first.routingMutationAllowed, false)
  assert.equal(first.releaseAuthority, false)
  assert.ok(first.artifactsWritten > 1)
  assert.equal(first.artifactIds.length, first.artifactsWritten)
  assert.ok(first.summary.items.some((item) =>
    item.hypothesis.namespace === 'event-role.actor' && item.status === 'calibrated'))
  assert.ok(first.summary.items.some((item) =>
    item.hypothesis.namespace === 'event-operator.negation' && item.status === 'unknown'))
  assert.ok(requests.every(({ request }) =>
    request.learningMode === 'off' && request.teacherMode === 'off' &&
    request.adaptiveRegion.mode === 'shadow' && request.trainingWritesAllowed === false &&
    request.routingMutationAllowed === false))
  assert.ok(store.listArtifactMetadata().some((artifact) => artifact.kind === producer.STRUCTURAL_SEMANTIC_PROBE_SPEC_ARTIFACT_KIND))
  const evaluationMetadata = store.listArtifactMetadata()
    .filter((artifact) => artifact.kind === 'plastic-region-semantic-probe-evaluation')
  assert.ok(evaluationMetadata.length > 0)
  for (const metadata of evaluationMetadata) {
    const artifact = store.getArtifact(metadata.id)
    assert.equal(artifact.payload.metadata.trainingWritesAllowed, false)
    assert.equal(artifact.payload.metadata.routingMutationAllowed, false)
    assert.equal(artifact.payload.metadata.releaseAuthority, false)
    assert.equal(artifact.payload.metadata.inputManifestHash, snapshot.inputManifestHash)
    assert.equal(artifact.payload.metadata.baseAtlasFingerprint, atlas.fingerprint)
  }
  const publicView = inspection.inspectSemanticProbeHypotheses(store, {
    atlasFingerprint: atlas.fingerprint,
    maximumItems: 16,
  })
  assert.equal(publicView.available, true)
  assert.equal(publicView.trainingWritesAllowed, false)
  assert.equal(publicView.routingMutationAllowed, false)
  assert.equal(publicView.releaseAuthority, false)
  assert.ok(publicView.summary.items.some((item) => item.status === 'calibrated'))
  assert.equal(JSON.stringify(publicView).includes('lineageFingerprint'), false)
  assert.equal(JSON.stringify(publicView).includes('scores'), false)

  const validEvaluationArtifact = store.getArtifact(evaluationMetadata[0].id)
  store.createArtifact({
    id: 'semantic-probe-store-hash-mismatch',
    kind: validEvaluationArtifact.kind,
    contentHash: 'forged-store-content-hash',
    payload: validEvaluationArtifact.payload,
  })
  const viewWithForgedEnvelope = inspection.inspectSemanticProbeHypotheses(store, {
    atlasFingerprint: atlas.fingerprint,
    maximumItems: 16,
  })
  assert.ok(viewWithForgedEnvelope.artifacts['plastic-region-semantic-probe-evaluation'].rejected >= 1)

  const artifactCount = store.listArtifactMetadata().length
  const second = await producer.produceAndPersistStructuralSemanticProbes({
    context,
    snapshot,
    runCase,
    evaluationPolicy: { bootstrapSamples: 200 },
  })
  assert.equal(second.summary.fingerprint, first.summary.fingerprint)
  assert.deepEqual(second.artifactIds, first.artifactIds)
  assert.equal(store.listArtifactMetadata().length, artifactCount, 'replay-equivalent producer run must reuse immutable artifacts')
})

test('producer returns honest unknown and writes no skeleton when exact atlas or two active features are absent', async (t) => {
  const [producer, regions] = await loadModules(t)
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const emptySnapshot = store.createSnapshot({
    id: 'semantic-no-atlas', artifactIds: [], sourceIds: [], inputManifestHash: 'manifest-no-atlas',
  })
  const noAtlas = await producer.produceAndPersistStructuralSemanticProbes({
    context: { store, jobId: null },
    snapshot: emptySnapshot,
    runCase: async () => { throw new Error('must not run') },
  })
  assert.equal(noAtlas.status, 'unknown')
  assert.equal(noAtlas.artifactsWritten, 0)
  assert.deepEqual(noAtlas.artifactIds, [])
  assert.equal(store.listArtifactMetadata().length, 0)

  const atlas = regions.createPlasticRegionAtlas({
    representationSpaceId: 'RS-NO-FEATURES', behaviorDimension: 8,
  })
  const { snapshot } = attachAtlas(store, atlas)
  const noFeatures = await producer.produceAndPersistStructuralSemanticProbes({
    context: { store, jobId: null }, snapshot,
    runCase: async () => { throw new Error('must not run') },
  })
  assert.equal(noFeatures.status, 'unknown')
  assert.equal(noFeatures.reason, 'insufficient-active-region-features')
  assert.equal(noFeatures.artifactsWritten, 0)
  assert.equal(store.listArtifactMetadata().filter((artifact) => artifact.kind.includes('semantic-probe')).length, 0)
})

test('only unapplied exact-atlas shadow telemetry is accepted', async (t) => {
  const [producer, regions] = await loadModules(t)
  const atlas = activeAtlas(regions)
  const valid = shadowResult(atlas, { expected: { actor: 'Ada' } })
  const scores = producer.extractStructuralSemanticFeatureScores(atlas, valid)
  assert.deepEqual(scores.map((item) => item.featureId), ['opaque-region-a', 'opaque-region-b'])
  assert.ok(Math.abs(scores[0].score - 0.75) < 1e-6)
  assert.ok(Math.abs(scores[1].score - 0.2) < 1e-6)

  const active = structuredClone(valid)
  active.ticks[0].branches[0].adaptiveRegion.activeApplied = true
  active.ticks[0].branches[0].adaptiveRegion.mode = 'active'
  assert.throws(
    () => producer.extractStructuralSemanticFeatureScores(atlas, active),
    (error) => error?.code === 'SEMANTIC_PROBE_ROUTING_ISOLATION_FAILED',
  )
  const wrongAtlas = structuredClone(valid)
  wrongAtlas.ticks[0].branches[0].adaptiveRegion.atlasFingerprint = 'wrong-atlas'
  assert.throws(
    () => producer.extractStructuralSemanticFeatureScores(atlas, wrongAtlas),
    (error) => error?.code === 'SEMANTIC_PROBE_ROUTING_ISOLATION_FAILED',
  )
})

test('historical empty skeletons are audit-only and never migration or persistence input', async (t) => {
  const [producer] = await loadModules(t)
  const legacy = {
    format: 'plastic-region-probe-cache',
    version: 1,
    trainingWritesAllowed: false,
    semanticAssignments: [],
    packs: [
      { id: 'morphology-diagnostic-v1', status: 'skeleton', probes: [], learnedBindings: [] },
      { id: 'mathematics-diagnostic-v1', status: 'skeleton', probes: [], learnedBindings: [] },
    ],
  }
  assert.deepEqual(producer.inspectLegacyPlasticRegionProbeSkeleton(legacy), {
    recognized: true,
    status: 'unknown',
    auditOnly: true,
    migrationAllowed: false,
    persistAgain: false,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
  })
  assert.equal(producer.inspectLegacyPlasticRegionProbeSkeleton({ ...legacy, semanticAssignments: ['fake'] }).recognized, false)
})

test('producer enforces the configured artifact bound before running the diagnostic pack', async (t) => {
  const [producer, regions] = await loadModules(t)
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  store.updateConfig({ plasticRegions: { maximumArtifactBytes: 64 * 1024 } })
  const atlas = activeAtlas(regions)
  const { snapshot } = attachAtlas(store, atlas)
  let runs = 0
  await assert.rejects(
    () => producer.produceAndPersistStructuralSemanticProbes({
      context: { store, jobId: null },
      snapshot,
      runCase: async () => { runs += 1; return shadowResult(atlas, { expected: { actor: 'Ada' } }) },
    }),
    (error) => error?.code === 'SEMANTIC_PROBE_ARTIFACT_LIMIT_EXCEEDED',
  )
  assert.equal(runs, 0)
  assert.equal(store.listArtifactMetadata().filter((artifact) => artifact.kind.includes('semantic-probe')).length, 0)
})
