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
    vite.ssrLoadModule('/runtime/plastic-region-pipeline.mjs'),
    vite.ssrLoadModule('/src/adaptive-regions/index.ts'),
    vite.ssrLoadModule('/runtime/plastic-region-evidence.mjs'),
    vite.ssrLoadModule('/runtime/plastic-region-cache.mjs'),
  ])
}

function adaptiveService(mode = 'shadow') {
  return {
    enabled: true,
    mode,
    representationProfile: 'adaptive-eighth-id-v1',
    thoughtSamplesPerCycle: 4,
    thoughtTicksPerSample: 16,
    maximumObservationsPerBatch: 2_048,
    maximumObservationBatchesPerCycle: 64,
    maximumRegions: 56,
    maximumMembershipsPerPage: 3,
    maximumActiveGates: 4,
    compatibilityBlend: 0.25,
    minimumTrace: 0.02,
    maximumArtifactBytes: 4 * 1024 * 1024,
  }
}

function tickResult(pageObservations, mode = 'shadow', atlasFingerprint = 'atlas-fingerprint') {
  return {
    episode: {
      episodeId: 'episode-a',
      branches: [{
        id: 'branch-child',
        mass: 0.75,
        transitions: [{ tick: 1, fromNeuronId: 'neuron-a', toNeuronId: 'neuron-b' }],
      }],
    },
    ticks: [{
      tick: 1,
      branches: [{
        branchId: 'branch-child',
        parentBranchId: 'branch-root',
        mass: 0.75,
        macroExecution: null,
        adaptiveRegion: {
          format: 'thought-adaptive-region-branch-telemetry',
          version: 1,
          mode,
          atlasFingerprint,
          activeApplied: false,
          pageObservations,
        },
      }],
    }],
  }
}

function lifecycleEvidence(evidenceId) {
  return {
    evidenceId,
    validationAuthority: 'independent-holdout',
    observationCount: 6,
    sourceGroupIds: ['source-a', 'source-b', 'source-c'],
    confidenceMass: 3,
    cohesion: 0.9,
    holdoutGain: 0.1,
    shamGain: 0.01,
    codes: ['counterfactual-utility'],
  }
}

function lifecycleSketch(pageIndex, behavior, { residualSupport = behavior[3] ?? 0 } = {}) {
  const sourceGroupIds = ['source-a', 'source-b', 'source-c']
  return {
    pageIndex,
    behaviorDimension: behavior.length,
    behaviorPrototype: behavior,
    outcomeCredit: 0,
    ablationUtility: 0,
    cohesion: 0.9,
    observationCount: sourceGroupIds.length,
    sourceGroupIds,
    confidenceMass: sourceGroupIds.length,
    evidence: sourceGroupIds.map((sourceGroupId) => ({
      observationId: `shared-${sourceGroupId}:${pageIndex}`,
      episodeId: `episode-${sourceGroupId}`,
      sourceGroupId,
      support: residualSupport,
    })),
    fingerprint: `sketch-${pageIndex}-${behavior.join('-')}`,
  }
}

function createCompletedCandidate(store, input) {
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 1 })
  const snapshot = store.createSnapshot({
    ...input,
    producerJobId: job.id,
    producerAttempt: 1,
  })
  store.updateJob(job.id, {
    state: 'succeeded',
    stage: 'completed',
    attempt: 1,
    result: { snapshotId: snapshot.id },
    finishedAt: new Date().toISOString(),
  })
  return snapshot
}

test('runtime request and dry-run inventory are explicit, deterministic, and never silently reencode', async (t) => {
  const [pipeline] = await loadModules(t)
  assert.deepEqual(
    pipeline.resolvePlasticRegionRuntimeRequest({ plasticRegions: { enabled: true, mode: 'off', representationProfile: 'legacy' } }),
    { requested: false, mode: 'off', profile: 'legacy' },
  )
  assert.equal(
    pipeline.resolvePlasticRegionRuntimeRequest({ plasticRegions: adaptiveService('proposal') }).profile,
    'adaptive-eighth-id-v1',
  )
  assert.deepEqual(
    pipeline.resolvePlasticRegionRuntimeRequest({ plasticRegions: adaptiveService('active') }),
    { requested: true, mode: 'proposal', configuredMode: 'active', profile: 'adaptive-eighth-id-v1' },
  )

  const sources = [
    { id: 'source-b', contentHash: 'hash-b', provenance: {} },
    { id: 'source-a', contentHash: 'hash-a', provenance: {} },
  ]
  const first = pipeline.planPlasticRegionReencodeInventory(sources)
  const second = pipeline.planPlasticRegionReencodeInventory([...sources].reverse())
  assert.equal(first.fingerprint, second.fingerprint)
  assert.equal(first.dryRun, true)
  assert.equal(first.willMutate, false)
  assert.deepEqual(first.rows.map((row) => row.status), ['encode-required', 'encode-required'])

  const mismatch = pipeline.planPlasticRegionReencodeInventory([{
    id: 'legacy-import',
    contentHash: 'legacy-hash',
    provenance: {
      textLabManifest: {
        format: 'text-lab-recompute-proof-v1',
        config: { matrixSide: 64, identityRatio: 0.25 },
      },
    },
  }])
  assert.equal(mismatch.blocked, true)
  assert.equal(mismatch.rows[0].status, 'blocked-representation-mismatch')
  assert.throws(
    () => pipeline.resolveAdaptiveTextLabConfig([{
      id: 'legacy-import',
      provenance: mismatch.rows[0].provenance ?? {
        textLabManifest: {
          format: 'text-lab-recompute-proof-v1',
          config: { matrixSide: 64, identityRatio: 0.25 },
        },
      },
    }], adaptiveService()),
    (error) => error?.code === 'REPRESENTATION_SPACE_MISMATCH',
  )
})

test('only real shadow/proposal train tick telemetry becomes source-balanced observations', async (t) => {
  const [pipeline, regions] = await loadModules(t)
  const signal = new Float32Array(regions.ADAPTIVE_RELATION_LENGTH).fill(0.75)
  const representation = new Float32Array(regions.ADAPTIVE_RELATION_LENGTH).fill(0.25)
  const delta = new Float32Array(regions.ADAPTIVE_RELATION_LENGTH).fill(0.1)
  const pageObservations = regions.computeAllPageObservations({
    observationIdPrefix: 'tick-observation',
    episodeId: 'episode-a',
    sourceGroupId: 'human-book-a',
    signalRelation: signal,
    representationRelation: representation,
    deltaRelation: delta,
    confidence: 1,
    outcomeCredit: 0,
    ablationUtility: 0,
  })
  const result = tickResult(pageObservations)
  const train = pipeline.extractPlasticRegionObservationsFromThoughtResult({
    result,
    source: { split: 'train', independenceGroupId: 'human-book-a', provenance: { kind: 'human' } },
    atlasFingerprint: 'atlas-fingerprint',
    maximumObservations: 56,
  })
  assert.equal(train.transitionCount, 1)
  assert.equal(train.observations.length, 56)
  assert.ok(train.observations.every((observation) => observation.confidence === 0.75))
  assert.ok(train.observations.every((observation) => observation.outcomeCredit === 0))

  const validation = pipeline.extractPlasticRegionObservationsFromThoughtResult({
    result,
    source: { split: 'validation', independenceGroupId: 'human-book-a', provenance: { kind: 'human' } },
    maximumObservations: 56,
  })
  assert.equal(validation.excluded, true)
  assert.equal(validation.reason, 'evaluation-split-forbidden')
  assert.equal(validation.observations.length, 0)

  const ollamaPages = pageObservations.map((observation) => ({
    ...observation,
    sourceGroupId: 'not-a-lineage',
  }))
  assert.throws(
    () => pipeline.extractPlasticRegionObservationsFromThoughtResult({
      result: tickResult(ollamaPages),
      source: { split: 'train', independenceGroupId: 'not-a-lineage', provenance: { kind: 'ollama' } },
      maximumObservations: 56,
    }),
    (error) => error?.code === 'OLLAMA_LINEAGE_MISMATCH',
  )
})

test('proposal cycle is deterministic, keeps probes semantic-free, and persists no-active-base as detached trace-only', async (t) => {
  const [pipeline, regions] = await loadModules(t)
  const atlas = regions.createInitialPlasticRegionAtlas({
    representationSpaceId: 'RS-ADAPTIVE-PIPELINE',
    behaviorDimension: regions.PAGE_BEHAVIOR_DIMENSION,
    maximumMembershipsPerPage: 3,
  })
  const signal = new Float32Array(regions.ADAPTIVE_RELATION_LENGTH).fill(1)
  const representation = new Float32Array(regions.ADAPTIVE_RELATION_LENGTH)
  const delta = new Float32Array(regions.ADAPTIVE_RELATION_LENGTH).fill(0.25)
  const observations = ['source-a', 'source-b', 'source-c'].flatMap((sourceGroupId, index) =>
    regions.computeAllPageObservations({
      observationIdPrefix: `evidence-${index}`,
      episodeId: `episode-${index}`,
      sourceGroupId,
      signalRelation: signal,
      representationRelation: representation,
      deltaRelation: delta,
      confidence: 1,
      outcomeCredit: 0,
      ablationUtility: 0,
    }))
  const input = {
    mode: 'proposal',
    baseAtlas: atlas,
    representationSpaceId: atlas.representationSpaceId,
    familyRunFingerprint: 'family-run-fingerprint',
    inputManifestHash: 'manifest-hash',
    observations,
    maximumPerBatch: 2_048,
    maximumBatches: 2,
    maximumRegions: 56,
    maximumMembershipsPerPage: 3,
    maximumActiveGates: 4,
  }
  const first = pipeline.buildPlasticRegionCycle(input)
  const second = pipeline.buildPlasticRegionCycle({ ...input, observations: [...observations].reverse() })
  assert.equal(first.shadowReport.fingerprint, second.shadowReport.fingerprint)
  assert.equal(first.proposal.fingerprint, second.proposal.fingerprint)
  assert.equal(first.proposal.eligible, false, 'train-only evidence cannot self-validate a proposal')
  assert.equal(first.probeCache, null, 'ordinary cycles must not manufacture an empty semantic skeleton')

  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const persisted = pipeline.persistPlasticRegionCycle({ store, jobId: null }, first, {
    expectedBase: { snapshotId: null, atlasArtifactId: null, atlasContentHash: null, atlasFingerprint: null },
  })
  assert.equal(persisted.safetyMode, 'trace-only-no-active-base')
  assert.equal(persisted.atlasActivated, false)
  assert.ok(persisted.detachedAtlasCandidateArtifact)
  assert.ok(!persisted.attachedArtifacts.some((artifact) => artifact.kind === 'plastic-region-atlas-head'))
  assert.ok(!persisted.attachedArtifacts.some((artifact) => artifact.kind === 'plastic-region-probe-cache'))
  assert.equal(persisted.detachedAtlasCandidateArtifact.payload.metadata.activationEligible, false)
  assert.equal(persisted.detachedAtlasCandidateArtifact.payload.metadata.traceOnly, true)
  assert.equal(store.getActiveSnapshot(), undefined)
})

test('one-proposal-per-revision selector reaches every lifecycle action and train traces never self-release', async (t) => {
  const [pipeline, regions] = await loadModules(t)
  const prototype = (first, second = 0) => ({ dimension: 8, values: [first, second, 0, 0, 0, 0, 0, 0] })
  const initial = (id) => regions.createInitialPlasticRegionAtlas({
    representationSpaceId: id, behaviorDimension: 8, maximumMembershipsPerPage: 3,
  })
  const born = (atlas, regionId, allocations, regionPrototype = prototype(1)) =>
    regions.applyRegionLifecycleProposal(atlas, regions.proposeRegionBirth(atlas, {
      regionId, allocations, prototype: regionPrototype, evidence: lifecycleEvidence(`evidence-${regionId}`),
    }))
  const decide = (atlas, sketches, maximumRegions = 56) =>
    pipeline.selectPlasticRegionLifecycleProposal(atlas, sketches, 'selector-observation-batch', { maximumRegions })
  const assertDecision = (decision, action) => {
    assert.equal(decision.selectedAction, action)
    assert.equal(decision.oneProposalPerRevision, true)
    assert.equal(decision.baseRevision + 1, decision.proposal.proposedAtlas.revision)
    assert.equal(decision.proposal.eligible, false)
    assert.ok(decision.proposal.rejectionReasons.includes('MISSING_INDEPENDENT_EVALUATION'))
    assert.equal(decision.candidates.filter((candidate) => candidate.proposed && candidate.action === action).length, 1)
  }

  const birth = decide(initial('RS-SELECT-BIRTH'), [0, 1, 2, 3].map((pageIndex) =>
    lifecycleSketch(pageIndex, [1, 0, 0, 1, 0, 0, 0, 0])))
  assertDecision(birth, 'birth')
  assert.ok(!birth.proposal.rejectionReasons.includes('INSUFFICIENT_REGION_CAPACITY'))

  const growBase = born(initial('RS-SELECT-GROW'), 'grow-region', [{ pageIndex: 0, weight: 1 }])
  assertDecision(decide(growBase, [lifecycleSketch(1, [1, 0, 0, 1, 0, 0, 0, 0])], 1), 'grow')

  const shrinkBase = born(initial('RS-SELECT-SHRINK'), 'shrink-region', [
    { pageIndex: 0, weight: 1 }, { pageIndex: 1, weight: 0.5 },
  ])
  assertDecision(decide(shrinkBase, [lifecycleSketch(1, [-1, 0, 0, 0, 0, 0, 0, 0])], 1), 'shrink')

  const splitBase = born(initial('RS-SELECT-SPLIT'), 'split-region', [
    { pageIndex: 0, weight: 1 }, { pageIndex: 1, weight: 1 },
  ], prototype(0, 1))
  assertDecision(decide(splitBase, [
    lifecycleSketch(0, [1, 0, 0, 0.5, 0, 0, 0, 0]),
    lifecycleSketch(1, [-1, 0, 0, 0.5, 0, 0, 0, 0]),
  ], 2), 'split')

  let mergeBase = born(initial('RS-SELECT-MERGE'), 'merge-a', [
    { pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 },
  ])
  mergeBase = born(mergeBase, 'merge-b', [
    { pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 },
  ])
  assertDecision(decide(mergeBase, [lifecycleSketch(0, [1, 0, 0, 0, 0, 0, 0, 0])], 2), 'merge')

  const retireBase = born(initial('RS-SELECT-RETIRE'), 'retire-region', [{ pageIndex: 0, weight: 1 }])
  assertDecision(decide(retireBase, [lifecycleSketch(0, [0, 1, 0, 0, 0, 0, 0, 0])], 1), 'retire')
})

test('birth proposal does not merge contradictory pages or count one transition once per page', async (t) => {
  const [pipeline, regions] = await loadModules(t)
  const atlas = regions.createInitialPlasticRegionAtlas({
    representationSpaceId: 'RS-ADVERSARIAL-PAGES',
    behaviorDimension: 8,
  })
  const observations = ['source-a', 'source-b', 'source-c'].flatMap((sourceGroupId, sourceIndex) =>
    Array.from({ length: 56 }, (_, pageIndex) => ({
      observationId: `transition-${sourceIndex}:${pageIndex}`,
      episodeId: `episode-${sourceIndex}`,
      sourceGroupId,
      pageIndex,
      confidence: 1,
      behavior: pageIndex === 0
        ? [0, 0, 0, 0.5, 0, 1, 0, 0]
        : pageIndex === 1
          ? [0, 0, 0, 0.5, 0, -1, 0, 0]
          : [0, 0, 0, 0, 0, 0, 0, 0],
      outcomeCredit: 0,
      ablationUtility: 0,
    })))
  const cycle = pipeline.buildPlasticRegionCycle({
    mode: 'proposal',
    baseAtlas: atlas,
    representationSpaceId: atlas.representationSpaceId,
    familyRunFingerprint: 'family-adversarial',
    inputManifestHash: 'manifest-adversarial',
    observations,
    maximumPerBatch: 224,
    maximumBatches: 1,
  })
  assert.ok(cycle.proposal)
  assert.equal(cycle.proposal.pageDeltas.length, 1, 'opposite page prototypes must not share one newborn region')
  assert.equal(cycle.proposal.evidence.observationCount, 3, 'three transitions must not be inflated to six page observations')
  assert.deepEqual(cycle.proposal.evidence.codes, ['unexplained-residual'])
})

test('multi-page birth only credits sources and transitions jointly supporting every page', async (t) => {
  const [pipeline, regions] = await loadModules(t)
  const atlas = regions.createInitialPlasticRegionAtlas({
    representationSpaceId: 'RS-JOINT-PAGE-EVIDENCE',
    behaviorDimension: 8,
  })
  const transition = (sourceGroupId, sourceIndex) => Array.from({ length: 56 }, (_, pageIndex) => ({
      observationId: `transition-${sourceIndex}:${pageIndex}`,
      episodeId: `episode-${sourceIndex}`,
      sourceGroupId,
      pageIndex,
      confidence: 1,
      behavior: pageIndex === 0 || (pageIndex === 1 && sourceGroupId === 'source-a')
        ? [0, 0, 0, 0.5, 0, 1, 0, 0]
        : [0, 0, 0, 0, 0, 0, 0, 0],
      outcomeCredit: 0,
      ablationUtility: 0,
    }))
  const cycle = pipeline.buildPlasticRegionCycle({
    mode: 'proposal',
    baseAtlas: atlas,
    representationSpaceId: atlas.representationSpaceId,
    familyRunFingerprint: 'family-joint-evidence',
    inputManifestHash: 'manifest-joint-evidence',
    observations: [
      ...transition('source-a', 0),
      ...transition('source-b', 1),
      ...transition('source-c', 2),
    ],
    maximumPerBatch: 224,
    maximumBatches: 1,
  })
  assert.ok(cycle.proposal)
  assert.equal(cycle.proposal.pageDeltas.length, 2)
  assert.equal(cycle.proposal.evidence.observationCount, 1)
  assert.deepEqual(cycle.proposal.evidence.sourceGroupIds, ['source-a'])
  assert.equal(cycle.proposal.eligible, false)
  assert.ok(cycle.proposal.rejectionReasons.includes('INSUFFICIENT_INDEPENDENT_SOURCES'))
})

test('cumulative evidence deduplicates replay runKeys and carries source-balanced observations across cycles', async (t) => {
  const [pipeline, regions] = await loadModules(t)
  const atlas = regions.createInitialPlasticRegionAtlas({
    representationSpaceId: 'RS-CUMULATIVE-EVIDENCE', behaviorDimension: 8,
  })
  const transition = (sourceGroupId, episodeId, prefix, mean) => Array.from({ length: 56 }, (_, pageIndex) => ({
    observationId: `${prefix}:${pageIndex}`,
    episodeId,
    sourceGroupId,
    pageIndex,
    confidence: 1,
    behavior: [0, 0, 0, 0.5, 0, mean, 0, 0],
    outcomeCredit: 0,
    ablationUtility: 0,
  }))
  const common = {
    mode: 'shadow', baseAtlas: atlas, representationSpaceId: atlas.representationSpaceId,
    familyRunFingerprint: 'family-cumulative', inputManifestHash: 'manifest-cumulative',
    maximumPerBatch: 2_048, maximumBatches: 2, maximumCumulativeObservations: 112,
  }
  const first = pipeline.buildPlasticRegionCycle({
    ...common,
    observations: transition('source-a', 'run-a', 'transition-a', 1),
  })
  const second = pipeline.buildPlasticRegionCycle({
    ...common,
    priorEvidenceHeads: [first.evidenceHead],
    observations: [
      ...transition('source-a', 'run-a', 'transition-a', 1),
      ...transition('source-b', 'run-b', 'transition-b', 0.5),
    ],
  })
  assert.equal(first.evidenceHead.transitionCount, 1)
  assert.equal(second.evidenceHead.revision, 1)
  assert.deepEqual(second.evidenceHead.parentFingerprints, [first.evidenceHead.fingerprint])
  assert.equal(second.evidenceHead.transitionCount, 2, 'replayed run/transition must count once')
  assert.equal(second.evidenceHead.observationCount, 112)
  assert.deepEqual(second.evidenceHead.sourceGroupIds, ['source-a', 'source-b'])

  const compacted = pipeline.buildPlasticRegionCycle({
    ...common,
    maximumCumulativeObservations: 56,
    priorEvidenceHeads: [second.evidenceHead],
    observations: [],
  })
  assert.equal(compacted.evidenceHead.transitionCount, 1)
  assert.equal(compacted.evidenceHead.observationCount, 56)
  assert.ok(compacted.evidenceHead.compactedTransitionCount >= 1)
})

test('cumulative evidence builder rejects parents from another representation, atlas, or authority lineage', async (t) => {
  const [, , evidence] = await loadModules(t)
  const parent = evidence.createPlasticRegionEvidenceHead({
    representationSpaceId: 'RS-PARENT-LINEAGE',
    baseAtlasFingerprint: 'atlas-parent-lineage',
    authority: 'research-only',
    maximumObservations: 56,
    observations: [],
  })
  for (const mismatch of [
    { representationSpaceId: 'RS-OTHER', baseAtlasFingerprint: parent.baseAtlasFingerprint, authority: parent.authority },
    { representationSpaceId: parent.representationSpaceId, baseAtlasFingerprint: 'atlas-other', authority: parent.authority },
    { representationSpaceId: parent.representationSpaceId, baseAtlasFingerprint: parent.baseAtlasFingerprint, authority: 'cas-active-base' },
  ]) {
    assert.throws(
      () => evidence.createPlasticRegionEvidenceHead({
        ...mismatch,
        maximumObservations: 56,
        parentHeads: [parent],
        observations: [],
      }),
      (error) => error?.code === 'PLASTIC_REGION_EVIDENCE_LINEAGE_MISMATCH',
    )
  }
  assert.throws(
    () => evidence.readPlasticRegionEvidenceHeads({ listArtifactMetadata: () => [] }, {
      representationSpaceId: parent.representationSpaceId,
      baseAtlasFingerprint: parent.baseAtlasFingerprint,
      maximumArtifactBytes: 1_024,
      maximumTotalBytes: 512,
    }),
    (error) => error?.code === 'INVALID_PLASTIC_REGION_EVIDENCE',
    'reader total-byte budget must never be smaller than a payload the writer can accept',
  )
})

test('CAS evidence survives two non-region snapshot hops while detached forged evidence stays quarantined', async (t) => {
  const [pipeline, regions, evidence, cache] = await loadModules(t)
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())

  const initial = store.createSnapshot({
    id: 'evidence-initial', status: 'candidate', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(initial.id)
  const atlas = regions.createInitialPlasticRegionAtlas({
    representationSpaceId: 'RS-EVIDENCE-HOPS', behaviorDimension: 8, maximumMembershipsPerPage: 3,
  })
  const atlasArtifact = cache.persistPlasticRegionArtifact({ store }, {
    kind: cache.PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    value: atlas,
    expectedBase: cache.currentPlasticRegionBase(store),
  }).artifact
  const snapshotA = store.createSnapshot({
    id: 'evidence-snapshot-a', parentId: initial.id, status: 'candidate',
    artifactIds: [atlasArtifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshotA.id)
  const baseA = cache.currentPlasticRegionBase(store)
  store.listArtifactMetadata = () => {
    throw new Error('global artifact metadata scan is forbidden on the cumulative hot path')
  }
  const transition = (sourceGroupId, episodeId, prefix, value) =>
    Array.from({ length: 56 }, (_, pageIndex) => ({
      observationId: `${prefix}:${pageIndex}`,
      episodeId,
      sourceGroupId,
      pageIndex,
      confidence: 1,
      behavior: [0, 0, 0, 0.5, 0, value, 0, 0],
      outcomeCredit: 0,
      ablationUtility: 0,
    }))
  const firstCycle = pipeline.buildPlasticRegionCycle({
    mode: 'shadow', baseAtlas: atlas, representationSpaceId: atlas.representationSpaceId,
    familyRunFingerprint: 'family-evidence-hop', inputManifestHash: 'manifest-evidence-hop-a',
    evidenceAuthority: 'cas-active-base', observations: transition('source-a', 'run-a', 'transition-a', 1),
  })
  const firstPersisted = pipeline.persistPlasticRegionCycle(
    { store, jobId: null }, firstCycle, { expectedBase: baseA },
  )
  const readOptionsA = {
    representationSpaceId: atlas.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: 'cas-active-base',
    expectedBase: baseA,
  }
  assert.deepEqual(
    evidence.readPlasticRegionEvidenceHeads(store, readOptionsA),
    [],
    'a CAS-written head is still an orphan until a completed candidate journals it',
  )
  assert.ok(!cache.plasticRegionCarryForwardArtifactIds(store, snapshotA).some((id) =>
    store.getArtifactMetadata(id)?.kind === evidence.PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND))
  store.createSnapshot({
    id: 'evidence-untrusted-no-producer', parentId: snapshotA.id, status: 'candidate',
    artifactIds: [atlasArtifact.id, ...firstPersisted.attachedArtifacts.map((artifact) => artifact.id)],
    sourceIds: [], evaluation: { passed: true, runtimeVerified: true },
  })
  assert.deepEqual(
    evidence.readPlasticRegionEvidenceHeads(store, readOptionsA),
    [],
    'self-declared evaluation without an exact succeeded producer is not a completed journal',
  )
  const proposalA = createCompletedCandidate(store, {
    id: 'evidence-proposal-a', parentId: snapshotA.id, status: 'candidate',
    artifactIds: [atlasArtifact.id, ...firstPersisted.attachedArtifacts.map((artifact) => artifact.id)],
    sourceIds: [], evaluation: { passed: true, runtimeVerified: true },
  })
  const firstJournaled = evidence.readPlasticRegionEvidenceHeads(store, readOptionsA)
  assert.equal(firstJournaled.length, 1)
  assert.equal(firstJournaled[0].transitionCount, 1)

  const secondCycleOnA = pipeline.buildPlasticRegionCycle({
    mode: 'shadow', baseAtlas: atlas, representationSpaceId: atlas.representationSpaceId,
    familyRunFingerprint: 'family-evidence-hop', inputManifestHash: 'manifest-evidence-hop-b',
    evidenceAuthority: 'cas-active-base', priorEvidenceHeads: firstJournaled,
    observations: transition('source-b', 'run-b', 'transition-b', 0.5),
  })
  const secondPersisted = pipeline.persistPlasticRegionCycle(
    { store, jobId: null }, secondCycleOnA, { expectedBase: baseA },
  )
  assert.equal(
    evidence.readPlasticRegionEvidenceHeads(store, readOptionsA)[0].fingerprint,
    firstJournaled[0].fingerprint,
    'a newer unjournaled head must not shadow the last completed candidate',
  )
  createCompletedCandidate(store, {
    id: 'evidence-proposal-b', parentId: snapshotA.id, status: 'candidate',
    artifactIds: [atlasArtifact.id, ...secondPersisted.attachedArtifacts.map((artifact) => artifact.id)],
    sourceIds: [], evaluation: { passed: true, runtimeVerified: true },
  })
  const secondJournaled = evidence.readPlasticRegionEvidenceHeads(store, readOptionsA)
  assert.equal(secondJournaled.length, 1)
  assert.equal(secondJournaled[0].revision, 1)
  assert.equal(secondJournaled[0].transitionCount, 2)

  const carriedToB = cache.plasticRegionCarryForwardArtifactIds(store, snapshotA)
  const firstEvidenceId = carriedToB.find((id) =>
    store.getArtifactMetadata(id)?.kind === evidence.PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND)
  assert.ok(firstEvidenceId, 'latest journaled evidence head must be carried into the candidate')
  const snapshotB = store.createSnapshot({
    id: 'evidence-snapshot-b', parentId: snapshotA.id, status: 'candidate',
    artifactIds: carriedToB, sourceIds: [], evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshotB.id)
  const carriedToC = cache.plasticRegionCarryForwardArtifactIds(store, snapshotB)
  assert.ok(carriedToC.includes(firstEvidenceId))
  const snapshotC = store.createSnapshot({
    id: 'evidence-snapshot-c', parentId: snapshotB.id, status: 'candidate',
    artifactIds: carriedToC, sourceIds: [], evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshotC.id)

  const baseC = cache.currentPlasticRegionBase(store)
  const indexedMetadataByKind = store.listArtifactMetadataByKind.bind(store)
  store.listArtifactMetadataByKind = () => []
  const inherited = evidence.readPlasticRegionEvidenceHeads(store, {
    representationSpaceId: atlas.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: 'cas-active-base',
    expectedBase: baseC,
  })
  store.listArtifactMetadataByKind = indexedMetadataByKind
  assert.equal(inherited.length, 1)
  assert.equal(inherited[0].revision, 1)
  assert.equal(inherited[0].transitionCount, 2)

  const thirdCycle = pipeline.buildPlasticRegionCycle({
    mode: 'shadow', baseAtlas: atlas, representationSpaceId: atlas.representationSpaceId,
    familyRunFingerprint: 'family-evidence-hop', inputManifestHash: 'manifest-evidence-hop-d',
    evidenceAuthority: 'cas-active-base', priorEvidenceHeads: inherited,
    observations: transition('source-c', 'run-c', 'transition-c', 0.25),
  })
  assert.equal(thirdCycle.evidenceHead.revision, 2)
  assert.equal(thirdCycle.evidenceHead.transitionCount, 3)
  assert.deepEqual(thirdCycle.evidenceHead.sourceGroupIds, ['source-a', 'source-b', 'source-c'])
  const thirdPersisted = pipeline.persistPlasticRegionCycle(
    { store, jobId: null }, thirdCycle, { expectedBase: baseC },
  )
  assert.equal(evidence.readPlasticRegionEvidenceHeads(store, {
    representationSpaceId: atlas.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: 'cas-active-base',
    expectedBase: baseC,
  })[0].fingerprint, inherited[0].fingerprint, 'cycle D is not readable before its candidate journal exists')
  createCompletedCandidate(store, {
    id: 'evidence-proposal-d', parentId: snapshotC.id, status: 'candidate',
    artifactIds: [...carriedToC, ...thirdPersisted.attachedArtifacts.map((artifact) => artifact.id)],
    sourceIds: [], evaluation: { passed: true, runtimeVerified: true },
  })
  const continued = evidence.readPlasticRegionEvidenceHeads(store, {
    representationSpaceId: atlas.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: 'cas-active-base',
    expectedBase: baseC,
  })
  assert.equal(continued.length, 1)
  assert.equal(continued[0].fingerprint, thirdCycle.evidenceHead.fingerprint)

  const forgedValue = evidence.createPlasticRegionEvidenceHead({
    representationSpaceId: atlas.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: 'cas-active-base',
    maximumObservations: 56,
    observations: transition('source-forged', 'run-forged', 'transition-forged', -1),
  })
  const forgedArtifact = store.createArtifact({
    id: 'detached-forged-evidence',
    kind: evidence.PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
    contentHash: 'forged-non-canonical-content-hash',
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: {
        representationSpaceId: forgedValue.representationSpaceId,
        fingerprint: forgedValue.fingerprint,
        geometryFingerprint: forgedValue.geometryFingerprint,
        revision: forgedValue.revision,
        baseAtlasFingerprint: forgedValue.baseAtlasFingerprint,
        authority: forgedValue.authority,
        baseSnapshotId: baseC.snapshotId,
        baseAtlasArtifactId: baseC.atlasArtifactId,
        casBaseAtlasFingerprint: baseC.atlasFingerprint,
      },
      value: encodeArtifactValue(forgedValue),
    },
  })
  const afterForgery = evidence.readPlasticRegionEvidenceHeads(store, {
    representationSpaceId: atlas.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: 'cas-active-base',
    expectedBase: baseC,
  })
  assert.equal(afterForgery.length, 1)
  assert.equal(afterForgery[0].fingerprint, continued[0].fingerprint)

  const carriedToD = cache.plasticRegionCarryForwardArtifactIds(store, snapshotC)
  const continuedEvidenceId = carriedToD.find((id) => {
    if (store.getArtifactMetadata(id)?.kind !== evidence.PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND) return false
    return store.getArtifact(id)?.payload?.metadata?.fingerprint === continued[0].fingerprint
  })
  assert.ok(continuedEvidenceId)
  const snapshotD = store.createSnapshot({
    id: 'evidence-snapshot-e', parentId: snapshotC.id, status: 'candidate',
    artifactIds: [...carriedToD, forgedArtifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshotD.id)
  assert.throws(
    () => evidence.readPlasticRegionEvidenceHeads(store, {
      representationSpaceId: atlas.representationSpaceId,
      baseAtlasFingerprint: atlas.fingerprint,
      authority: 'cas-active-base',
      expectedBase: cache.currentPlasticRegionBase(store),
    }),
    (error) => error?.code === 'PLASTIC_REGION_EVIDENCE_READ_FAILED',
    'the same forged head must fail closed once attached to the active snapshot',
  )
})

test('trace-only persistence detects a newly appearing active base before writing', async (t) => {
  const [pipeline, regions] = await loadModules(t)
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const expectedBase = { snapshotId: null, atlasArtifactId: null, atlasContentHash: null, atlasFingerprint: null }
  const snapshot = store.createSnapshot({
    id: 'new-active-base', status: 'candidate', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  const cycle = pipeline.buildPlasticRegionCycle({
    mode: 'shadow',
    baseAtlas: regions.createInitialPlasticRegionAtlas({ representationSpaceId: 'RS-STALE', behaviorDimension: 8 }),
    representationSpaceId: 'RS-STALE',
    familyRunFingerprint: 'family-stale',
    inputManifestHash: 'manifest-stale',
    observations: [],
  })
  assert.throws(
    () => pipeline.persistPlasticRegionCycle({ store, jobId: null }, cycle, { expectedBase }),
    (error) => error?.code === 'STALE_REGION_ATLAS_BASE',
  )
  assert.equal(store.listArtifactMetadata().length, 0)
})
