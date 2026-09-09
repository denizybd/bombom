import assert from 'node:assert/strict'
import test from 'node:test'
import { createPlasticRegionAtlas, proposeRegionBirth } from '../../src/adaptive-regions/index.ts'
import { currentPlasticRegionBase, persistPlasticRegionArtifact } from '../plastic-region-cache.mjs'
import {
  PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_JOB_TYPE,
  createPlasticRegionReleaseExecutor,
  preparePlasticRegionReleaseJob,
} from '../plastic-region-release-runner.mjs'
import { createTrainingStore } from '../../server/store.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'

const REPRESENTATION_SPACE_ID = 'RS-RELEASE-RUNNER-TEST'

async function releaseFixture(t, { evaluateCase, sourceMetrics } = {}) {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const baseSnapshot = store.createSnapshot({
    id: 'release-runner-base',
    artifactIds: [],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(baseSnapshot.id)
  const baseAtlas = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
  const proposal = proposeRegionBirth(baseAtlas, {
    regionId: 'release-runner-region',
    allocations: [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })),
    prototype: { dimension: 8, values: [1, 0, 0, 0, 0, 0, 0, 0] },
    evidence: {
      evidenceId: 'release-runner-train-evidence',
      validationAuthority: 'train-only',
      observationCount: 40,
      sourceGroupIds: ['train-a', 'train-b', 'train-c'],
      confidenceMass: 3,
      cohesion: 0.9,
      holdoutGain: 0,
      shamGain: 0,
      codes: ['counterfactual-utility', 'unexplained-residual'],
    },
  })
  assert.equal(proposal.eligible, false)
  assert.deepEqual(proposal.rejectionReasons, [
    'MISSING_INDEPENDENT_EVALUATION',
    'INSUFFICIENT_HOLDOUT_GAIN',
  ])
  const producer = store.createJob({ id: 'release-runner-producer', type: 'full-cycle' }).job
  const proposalArtifact = persistPlasticRegionArtifact({ store, jobId: producer.id }, {
    kind: 'plastic-region-atlas-proposal',
    value: proposal,
    expectedBase: currentPlasticRegionBase(store),
    contentHash: 'release-runner-proposal-content',
  }).artifact
  const sourceSnapshot = store.createSnapshot({
    id: 'release-runner-source',
    parentId: baseSnapshot.id,
    status: 'candidate',
    producerJobId: producer.id,
    artifactIds: [proposalArtifact.id],
    sourceIds: [],
    sourceRevisions: {},
    inputManifestHash: 'release-runner-source-input',
    metrics: sourceMetrics ?? {},
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.updateJob(producer.id, {
    state: 'succeeded',
    stage: 'completed',
    result: { snapshotId: sourceSnapshot.id },
    finishedAt: new Date().toISOString(),
  })
  const prepared = preparePlasticRegionReleaseJob(store, proposalArtifact.id)
  const queued = store.createJob({
    type: PLASTIC_REGION_RELEASE_JOB_TYPE,
    payload: prepared.payload,
    requestHash: prepared.requestHash,
    idempotencyKey: prepared.defaultIdempotencyKey,
    maxAttempts: 1,
  }).job
  const claimed = store.claimNextJob('release-runner-owner', 60_000, 120_000)
  assert.equal(claimed.id, queued.id)
  const fakeBase = {
    reference: {
      snapshotId: sourceSnapshot.id,
      snapshotStatus: sourceSnapshot.status,
      activeAtStart: false,
      artifactManifestHash: prepared.payload.sourceArtifactManifestHash,
      baseModelFingerprint: 'release-runner-model',
      familyRunFingerprint: 'release-runner-family',
      languageStateFingerprint: null,
      thoughtHeadFingerprint: 'release-runner-head',
      thoughtHeadRevision: 0,
      conceptMacroHeadFingerprint: null,
    },
    layout: { side: 64, cellCount: 4096, identityLength: 512, relationStart: 512, relationLength: 3584 },
    representationProfile: 'adaptive-eighth-id-v1',
    neurons: [{ memberKey: 'n-1' }],
    familyRun: { fingerprint: 'release-runner-family', representationSpaceId: REPRESENTATION_SPACE_ID },
    languageState: null,
    thoughtHead: { fingerprint: 'release-runner-head', revision: 0 },
    conceptMacroHead: null,
  }
  const executor = createPlasticRegionReleaseExecutor({
    resolveBase: () => fakeBase,
    evaluateCase: evaluateCase ?? (({ variant }) => variant === 'candidate' ? 0.7 : 0.5),
  })
  const context = {
    jobId: queued.id,
    store,
    config: store.getConfig(),
    jobClaim: { owner: 'release-runner-owner', attempt: claimed.attempt },
    checkResources: () => ({}),
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }
  return { store, baseSnapshot, sourceSnapshot, proposalArtifact, prepared, queued, claimed, executor, context }
}

test('release runner measures server-owned paired rows, creates only a manual candidate, and is idempotent', async (t) => {
  const variantsByPair = new Map()
  const fixture = await releaseFixture(t, {
    evaluateCase: ({ variant, benchmarkCase, seed, benchmarkPairingKey, runtimePolicy, regionMode }) => {
      const key = `${benchmarkCase.id}:${seed}`
      const observed = variantsByPair.get(key) ?? { pairingKey: benchmarkPairingKey, variants: new Set() }
      assert.equal(benchmarkPairingKey, observed.pairingKey)
      assert.deepEqual(runtimePolicy, {
        compatibilityBlend: Math.fround(0.25),
        minimumTrace: Math.fround(0.02),
        maximumActiveGates: 4,
      })
      assert.equal(regionMode, variant === 'base' ? 'off' : 'active')
      observed.variants.add(variant)
      variantsByPair.set(key, observed)
      return variant === 'candidate' ? 0.7 : 0.5
    },
  })
  const result = await fixture.executor(fixture.queued, fixture.context)
  assert.equal(result.released, true)
  assert.equal(result.committed, false)
  assert.equal(result.awaitingApproval, true)
  assert.equal(result.automaticActivation, false)
  assert.equal(result.atomicCandidateJobCompletion, true)
  assert.equal(fixture.store.getJob(fixture.queued.id).state, 'succeeded')
  assert.equal(fixture.store.getJob(fixture.queued.id).result.atomicCandidateJobCompletion, undefined)
  assert.equal(fixture.store.getActiveSnapshot().id, fixture.baseSnapshot.id)
  assert.equal(fixture.store.getArtifactMetadata(result.auditArtifactId).kind, PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND)
  assert.ok(fixture.store.listArtifactMetadata().some((artifact) =>
    artifact.id === result.auditArtifactId && artifact.kind === PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND))
  assert.equal(variantsByPair.size, 150)
  assert.ok([...variantsByPair.values()].every((entry) => entry.variants.size === 3))

  const candidate = fixture.store.getSnapshot(result.snapshotId)
  assert.equal(candidate.status, 'candidate')
  const again = await fixture.executor(fixture.queued, fixture.context)
  assert.equal(again.snapshotId, result.snapshotId)
  assert.deepEqual(again.releaseArtifactIds, result.releaseArtifactIds)

  fixture.store.activateSnapshot(candidate.id)
  assert.equal(fixture.store.getActiveSnapshot().id, candidate.id)
})

test('release runner rejects a stale/tampered queue payload before resolving the dense base', async (t) => {
  const fixture = await releaseFixture(t)
  let resolved = 0
  const executor = createPlasticRegionReleaseExecutor({
    resolveBase: () => { resolved += 1; throw new Error('must not resolve') },
    evaluateCase: () => 0,
  })
  const tampered = {
    ...fixture.queued,
    payload: { ...fixture.queued.payload, sourceInputManifestHash: 'fabricated-input' },
  }
  await assert.rejects(
    () => executor(tampered, fixture.context),
    (error) => error?.code === 'PLASTIC_REGION_RELEASE_REQUEST_TAMPERED',
  )
  assert.equal(resolved, 0)
})

test('pre-hydration dense guard fails closed before resolveBase when metadata cannot prove a safe catalog', async (t) => {
  const fixture = await releaseFixture(t, { sourceMetrics: { neuronCount: 20_000 } })
  let resolved = 0
  const executor = createPlasticRegionReleaseExecutor({
    resolveBase: () => { resolved += 1; throw new Error('must not resolve') },
    evaluateCase: () => 0,
  })
  await assert.rejects(
    () => executor(fixture.queued, fixture.context),
    (error) => error?.code === 'JOB_PAUSED' &&
      error.details?.reasonCode === 'PLASTIC_REGION_RELEASE_DENSE_BASE_UNSUPPORTED',
  )
  assert.equal(resolved, 0)
})

test('failed benchmark persists one self-contained audit and no release candidate artifacts', async (t) => {
  const fixture = await releaseFixture(t, { evaluateCase: () => 0.5 })
  const beforeIds = new Set(fixture.store.listArtifactMetadata().map((artifact) => artifact.id))
  const result = await fixture.executor(fixture.queued, fixture.context)
  assert.equal(result.released, false)
  assert.equal(result.committed, false)
  assert.equal(result.automaticActivation, false)
  assert.equal(result.benchmarkReportArtifactId, undefined)
  const created = fixture.store.listArtifactMetadata().filter((artifact) => !beforeIds.has(artifact.id))
  assert.deepEqual(created.map((artifact) => artifact.kind), [PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND])
  assert.equal(fixture.store.listSnapshots().some((snapshot) => snapshot.id.startsWith('region-release-')), false)
})
