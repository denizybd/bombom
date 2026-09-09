import assert from 'node:assert/strict'
import test from 'node:test'
import { createPlasticRegionAtlas, proposeRegionBirth } from '../../src/adaptive-regions/index.ts'
import {
  currentPlasticRegionBase,
  persistPlasticRegionArtifact,
  readPlasticRegionRelease,
  readPlasticRegionSnapshot,
} from '../plastic-region-cache.mjs'
import {
  PLASTIC_REGION_RELEASE_JOB_TYPE,
  createPlasticRegionReleaseExecutor,
  preparePlasticRegionReleaseJob,
} from '../plastic-region-release-runner.mjs'
import {
  PLASTIC_REGION_RENEWAL_JOB_TYPE,
  createPlasticRegionRenewalExecutor,
  preparePlasticRegionRenewalJob,
} from '../plastic-region-renewal-runner.mjs'
import { verifyPlasticRegionRenewalChain } from '../plastic-region-renewal.mjs'
import { createTrainingStore } from '../../server/store.mjs'
import { resolvePlasticRegionRuntime } from '../../server/plastic-region-runtime.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'

const REPRESENTATION_SPACE_ID = 'RS-RENEWAL-RUNNER-TEST'

function fakeBase(snapshot, manifestHash) {
  return {
    reference: {
      snapshotId: snapshot.id,
      snapshotStatus: snapshot.status,
      activeAtStart: false,
      artifactManifestHash: manifestHash,
      baseModelFingerprint: `model-${snapshot.id}`,
      familyRunFingerprint: `family-${snapshot.id}`,
      languageStateFingerprint: null,
      thoughtHeadFingerprint: `head-${snapshot.id}`,
      thoughtHeadRevision: 0,
      conceptMacroHeadFingerprint: null,
    },
    layout: { side: 64, cellCount: 4_096, identityLength: 512, relationStart: 512, relationLength: 3_584 },
    representationProfile: 'adaptive-eighth-id-v1',
    neurons: [{ memberKey: 'renewal-neuron' }],
    familyRun: { fingerprint: `family-${snapshot.id}`, representationSpaceId: REPRESENTATION_SPACE_ID },
    languageState: null,
    thoughtHead: { fingerprint: `head-${snapshot.id}`, revision: 0 },
    conceptMacroHead: null,
  }
}

async function seedReleasedBase(store) {
  const baseSnapshot = store.createSnapshot({
    id: 'renewal-initial-base', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(baseSnapshot.id)
  const atlas = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
  const proposal = proposeRegionBirth(atlas, {
    regionId: 'renewal-region',
    allocations: [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })),
    prototype: { dimension: 8, values: [1, 0, 0, 0, 0, 0, 0, 0] },
    evidence: {
      evidenceId: 'renewal-initial-train', validationAuthority: 'train-only', observationCount: 40,
      sourceGroupIds: ['train-a', 'train-b', 'train-c'], confidenceMass: 3, cohesion: 0.9,
      holdoutGain: 0, shamGain: 0, codes: ['counterfactual-utility', 'unexplained-residual'],
    },
  })
  const producer = store.createJob({ id: 'renewal-initial-producer', type: 'full-cycle' }).job
  const proposalArtifact = persistPlasticRegionArtifact({ store, jobId: producer.id }, {
    kind: 'plastic-region-atlas-proposal', value: proposal,
    expectedBase: currentPlasticRegionBase(store), contentHash: 'renewal-initial-proposal-content',
  }).artifact
  const source = store.createSnapshot({
    id: 'renewal-initial-source', parentId: baseSnapshot.id, status: 'candidate', producerJobId: producer.id,
    artifactIds: [proposalArtifact.id], sourceIds: [], sourceRevisions: {}, inputManifestHash: 'renewal-initial-input',
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.updateJob(producer.id, {
    state: 'succeeded', stage: 'completed', result: { snapshotId: source.id }, finishedAt: new Date().toISOString(),
  })
  const prepared = preparePlasticRegionReleaseJob(store, proposalArtifact.id)
  const releaseJob = store.createJob({
    type: PLASTIC_REGION_RELEASE_JOB_TYPE, payload: prepared.payload, requestHash: prepared.requestHash,
    idempotencyKey: prepared.defaultIdempotencyKey, maxAttempts: 1,
  }).job
  const owner = 'renewal-initial-release-owner'
  const claim = store.claimNextJob(owner, 60_000, 120_000)
  const release = await createPlasticRegionReleaseExecutor({
    resolveBase: () => fakeBase(source, prepared.payload.sourceArtifactManifestHash),
    evaluateCase: ({ variant }) => variant === 'candidate' ? 0.7 : 0.5,
  })(releaseJob, {
    jobId: releaseJob.id, store, config: store.getConfig(), jobClaim: { owner, attempt: claim.attempt },
    checkResources: () => ({}), checkpoint: async () => {}, progress: async () => {}, emit: async () => {},
  })
  store.activateSnapshot(release.snapshotId)
  return store.getSnapshot(release.snapshotId)
}

test('unchanged-atlas renewal rebenchmarks a new model candidate and waits for explicit activation', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const active = await seedReleasedBase(store)
  const activeAtlas = readPlasticRegionSnapshot(store, active)
  const producer = store.createJob({ id: 'renewal-model-producer', type: 'full-cycle' }).job
  const source = store.createSnapshot({
    id: 'renewal-model-source', parentId: active.id, status: 'candidate', producerJobId: producer.id,
    artifactIds: [...active.artifactIds], sourceIds: [], sourceRevisions: {}, inputManifestHash: 'renewal-model-input',
    metrics: { neuronCount: 1 }, evaluation: { passed: true, runtimeVerified: true },
  })
  store.updateJob(producer.id, {
    state: 'succeeded', stage: 'completed', result: { snapshotId: source.id }, finishedAt: new Date().toISOString(),
  })
  const prepared = preparePlasticRegionRenewalJob(store, source.id)
  let tamperedResolveCalls = 0
  const tamperedExecutor = createPlasticRegionRenewalExecutor({
    resolveBase: () => {
      tamperedResolveCalls += 1
      throw new Error('tampered renewal must fail before dense source hydration')
    },
    evaluateCase: () => 0.5,
  })
  await assert.rejects(
    tamperedExecutor({
      id: 'tampered-renewal-job',
      type: PLASTIC_REGION_RENEWAL_JOB_TYPE,
      requestHash: prepared.requestHash,
      payload: {
        ...prepared.payload,
        sourceArtifactManifestHash: `${prepared.payload.sourceArtifactManifestHash}-tampered`,
      },
    }, {
      jobId: 'tampered-renewal-job', store, config: store.getConfig(),
      checkResources: () => ({}), checkpoint: async () => {}, progress: async () => {}, emit: async () => {},
    }),
    (error) => error?.code === 'PLASTIC_REGION_RENEWAL_REQUEST_TAMPERED',
  )
  assert.equal(tamperedResolveCalls, 0)
  const job = store.createJob({
    type: PLASTIC_REGION_RENEWAL_JOB_TYPE, payload: prepared.payload, requestHash: prepared.requestHash,
    idempotencyKey: prepared.defaultIdempotencyKey, maxAttempts: 1,
  }).job
  const owner = 'renewal-model-release-owner'
  const claim = store.claimNextJob(owner, 60_000, 120_000)
  const variants = new Map()
  const executor = createPlasticRegionRenewalExecutor({
    resolveBase: () => fakeBase(source, prepared.payload.sourceArtifactManifestHash),
    evaluateCase: ({ variant, benchmarkCase, seed, benchmarkPairingKey, regionMode }) => {
      const key = `${benchmarkCase.id}:${seed}`
      const entry = variants.get(key) ?? { pairingKey: benchmarkPairingKey, modes: new Map() }
      assert.equal(entry.pairingKey, benchmarkPairingKey)
      entry.modes.set(variant, regionMode)
      variants.set(key, entry)
      return variant === 'candidate' ? 0.7 : 0.5
    },
  })
  const context = {
    jobId: job.id, store, config: store.getConfig(), jobClaim: { owner, attempt: claim.attempt },
    checkResources: () => ({}), checkpoint: async () => {}, progress: async () => {}, emit: async () => {},
  }
  const result = await executor(job, context)
  assert.equal(result.renewed, true)
  assert.equal(result.committed, false)
  assert.equal(result.awaitingApproval, true)
  assert.equal(result.automaticActivation, false)
  assert.equal(result.atomicCandidateJobCompletion, true)
  assert.equal(store.getJob(job.id).state, 'succeeded')
  assert.equal(store.getJob(job.id).result.atomicCandidateJobCompletion, undefined)
  assert.equal(store.getActiveSnapshot().id, active.id)
  assert.equal(variants.size, 150)
  assert.ok([...variants.values()].every((entry) =>
    entry.modes.get('base') === 'off' && entry.modes.get('candidate') === 'active' && entry.modes.get('sham') === 'active'))

  const candidate = store.getSnapshot(result.snapshotId)
  const candidateAtlas = readPlasticRegionSnapshot(store, candidate)
  assert.equal(candidateAtlas.metadata.id, activeAtlas.metadata.id)
  assert.equal(candidateAtlas.head.fingerprint, activeAtlas.head.fingerprint)
  const repeated = await executor(job, context)
  assert.equal(repeated.snapshotId, result.snapshotId)
  assert.deepEqual(repeated.releaseArtifactIds, result.releaseArtifactIds)

  store.activateSnapshot(candidate.id)
  const verified = readPlasticRegionRelease(store, store.getActiveSnapshot())
  assert.equal(verified.compatible, true)
  assert.equal(verified.releaseKind, 'unchanged-atlas-model-renewal')
  assert.equal(verified.manifestMetadata.id, result.manifestArtifactId)
  assert.equal(verified.manifest.fingerprint, store.getSnapshot(candidate.id).evaluation.plasticRegionReleaseFingerprint)
  const activeSnapshot = store.getActiveSnapshot()
  const activeRuntime = resolvePlasticRegionRuntime({
    store,
    base: {
      ...fakeBase(activeSnapshot, prepared.payload.sourceArtifactManifestHash),
      snapshot: activeSnapshot,
      reference: {
        ...fakeBase(activeSnapshot, prepared.payload.sourceArtifactManifestHash).reference,
        activeAtStart: true,
        commitEligible: true,
      },
    },
    requestedMode: 'active',
    policy: {
      ...store.getConfig().plasticRegions,
      enabled: true,
      mode: 'active',
      representationProfile: 'adaptive-eighth-id-v1',
    },
  })
  assert.equal(activeRuntime.releaseManifestArtifactId, result.manifestArtifactId)
  assert.equal(activeRuntime.releaseFingerprint, result.releaseFingerprint)

  const secondProducer = store.createJob({ id: 'renewal-second-model-producer', type: 'full-cycle' }).job
  const secondSource = store.createSnapshot({
    id: 'renewal-second-model-source', parentId: activeSnapshot.id, status: 'candidate',
    producerJobId: secondProducer.id, artifactIds: [...activeSnapshot.artifactIds], sourceIds: [], sourceRevisions: {},
    inputManifestHash: 'renewal-second-model-input', metrics: { neuronCount: 1 },
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.updateJob(secondProducer.id, {
    state: 'succeeded', stage: 'completed', result: { snapshotId: secondSource.id }, finishedAt: new Date().toISOString(),
  })
  const secondPrepared = preparePlasticRegionRenewalJob(store, secondSource.id)
  const secondJob = store.createJob({
    type: PLASTIC_REGION_RENEWAL_JOB_TYPE, payload: secondPrepared.payload, requestHash: secondPrepared.requestHash,
    idempotencyKey: secondPrepared.defaultIdempotencyKey, maxAttempts: 1,
  }).job
  const secondOwner = 'renewal-second-model-owner'
  const secondClaim = store.claimNextJob(secondOwner, 60_000, 120_000)
  const secondResult = await createPlasticRegionRenewalExecutor({
    resolveBase: () => fakeBase(secondSource, secondPrepared.payload.sourceArtifactManifestHash),
    evaluateCase: ({ variant }) => variant === 'candidate' ? 0.7 : 0.5,
  })(secondJob, {
    jobId: secondJob.id, store, config: store.getConfig(),
    jobClaim: { owner: secondOwner, attempt: secondClaim.attempt },
    checkResources: () => ({}), checkpoint: async () => {}, progress: async () => {}, emit: async () => {},
  })
  store.activateSnapshot(secondResult.snapshotId)
  const secondVerified = readPlasticRegionRelease(store, store.getActiveSnapshot())
  assert.equal(secondVerified.compatible, true)
  assert.equal(secondVerified.releaseKind, 'unchanged-atlas-model-renewal')
  assert.equal(secondVerified.manifest.baseBinding.activeReleaseFingerprint, result.releaseFingerprint)

  const depthLimited = verifyPlasticRegionRenewalChain(
    store,
    store.getActiveSnapshot(),
    readPlasticRegionSnapshot(store, store.getActiveSnapshot()),
    store.getConfig().plasticRegions,
    new Set(Array.from({ length: 64 }, (_, index) => `prior-renewal-${index}`)),
  )
  assert.equal(depthLimited.compatible, false)
  assert.equal(depthLimited.reason, 'PLASTIC_REGION_RENEWAL_DEPTH_EXCEEDED')
})
