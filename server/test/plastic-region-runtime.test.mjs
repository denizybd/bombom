import assert from 'node:assert/strict'
import test from 'node:test'
import { encodeArtifactValue, stableRuntimeHash } from '../../runtime/artifacts.mjs'
import {
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
} from '../../runtime/plastic-region-cache.mjs'
import {
  createPlasticRegionReleaseBenchmarkReport,
  createPlasticRegionReleaseEvidence,
  createPlasticRegionReleaseManifest,
  derivePlasticRegionReleaseMetrics,
  plasticRegionReleasePolicy,
} from '../../runtime/plastic-region-release.mjs'
import {
  PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
  PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
  plasticRegionReleaseCases,
  plasticRegionReleaseCorpusFingerprint,
} from '../../runtime/plastic-region-release-spec.mjs'
import {
  ADAPTIVE_REGION_PROFILE,
  LEGACY_REGION_PROFILE,
  canonicalPlasticRegionProfile,
  plasticRegionProfileForLayout,
  resolvePlasticRegionRuntime,
} from '../plastic-region-runtime.mjs'
import {
  ADAPTIVE_REGION_GEOMETRY,
  createPlasticRegionAtlas,
  proposeRegionBirth,
} from '../../src/adaptive-regions/index.ts'
import { createTrainingStore } from '../store.mjs'
import { temporaryDirectory } from './helpers.mjs'

const ADAPTIVE_LAYOUT = Object.freeze({
  side: 64, cellCount: 4096, identityLength: 512, relationStart: 512, relationLength: 3584,
})
const LEGACY_LAYOUT = Object.freeze({
  side: 64, cellCount: 4096, identityLength: 1024, relationStart: 1024, relationLength: 3072,
})

function base(snapshot, layout = ADAPTIVE_LAYOUT, options = {}) {
  return {
    snapshot,
    layout,
    ...(options.representationProfile === null ? {} : {
      representationProfile: options.representationProfile ?? ADAPTIVE_REGION_PROFILE,
    }),
    familyRun: { representationSpaceId: options.representationSpaceId ?? 'RS-ADAPTIVE-RUNTIME-TEST' },
    reference: {
      snapshotId: snapshot.id,
      activeAtStart: options.activeAtStart ?? false,
      commitEligible: options.commitEligible ?? false,
    },
  }
}

function artifactMetadata(kind, value, extras = {}) {
  const atlas = kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ? value.proposedAtlas : value
  return {
    representationSpaceId: atlas.representationSpaceId,
    fingerprint: value.fingerprint,
    ...(kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND ? {
      geometryFingerprint: value.geometryFingerprint,
      revision: value.revision,
      behaviorDimension: value.behaviorDimension,
      maximumMembershipsPerPage: value.maximumMembershipsPerPage,
    } : {}),
    ...(kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ? {
      geometryFingerprint: atlas.geometryFingerprint,
      revision: atlas.revision,
      proposalId: value.proposalId,
      baseAtlasFingerprint: value.baseAtlasFingerprint,
      proposedAtlasFingerprint: atlas.fingerprint,
      eligible: value.eligible,
    } : {}),
    ...(kind === PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND ? {
      geometryFingerprint: value.geometryFingerprint,
      evaluationId: value.evaluationId,
      evaluationKind: value.evaluationKind,
      proposalFingerprint: value.proposalFingerprint,
      proposedAtlasFingerprint: value.proposedAtlasFingerprint,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
      benchmarkReportArtifactId: value.benchmarkReport.artifactId,
      benchmarkReportFingerprint: value.benchmarkReport.fingerprint,
    } : {}),
    ...(kind === PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND ? {
      evaluationId: value.evaluationId,
      runnerFingerprint: value.runnerFingerprint,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
      originalProposalArtifactId: value.originalProposal.artifactId,
      usesOllama: false,
    } : {}),
    ...(kind === PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND ? {
      geometryFingerprint: value.geometryFingerprint,
      releaseId: value.releaseId,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
      benchmarkReportArtifactId: value.benchmarkReport.artifactId,
      benchmarkReportFingerprint: value.benchmarkReport.fingerprint,
    } : {}),
    ...extras,
  }
}

function createArtifact(store, id, kind, value, contentHash, extras = {}) {
  return store.createArtifact({
    id, kind, contentHash,
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: artifactMetadata(kind, value, extras),
      value: encodeArtifactValue(value),
    },
  })
}

function createReleasedAtlasArtifacts(store, representationSpaceId = 'RS-ADAPTIVE-RUNTIME-TEST', options = {}) {
  const baseAtlas = createPlasticRegionAtlas({ representationSpaceId, behaviorDimension: 8 })
  const proposal = proposeRegionBirth(baseAtlas, {
    regionId: 'released-region',
    allocations: [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })),
    prototype: { dimension: 8, values: [1, 0, 0, 0, 0, 0, 0, 0] },
    evidence: {
      evidenceId: 'released-region-evidence', validationAuthority: 'independent-holdout', observationCount: 30,
      sourceGroupIds: ['train-a', 'train-b', 'train-c'], confidenceMass: 3,
      cohesion: 0.9, holdoutGain: 0.1, shamGain: 0.01,
      codes: ['counterfactual-utility', 'unexplained-residual'],
    },
  })
  assert.equal(proposal.eligible, true)
  assert.equal(proposal.proposedAtlas.regions[0].lifecycle, 'active')
  const atlas = proposal.proposedAtlas
  const baseSnapshot = store.createSnapshot({
    id: `released-parent-${representationSpaceId}`,
    artifactIds: [],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(baseSnapshot.id)
  const atlasArtifact = createArtifact(store, 'released-atlas', PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    atlas, 'released-atlas-content', {
      activationEligible: true, candidateOnly: false, ...(options.atlasMetadata ?? {}),
    })
  const proposalArtifact = createArtifact(store, 'released-proposal', PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
    proposal, 'released-proposal-content')
  const producer = store.createJob({ id: 'released-source-producer', type: 'full-cycle' }).job
  const sourceSnapshot = store.createSnapshot({
    id: 'released-source-candidate',
    parentId: baseSnapshot.id,
    producerJobId: producer.id,
    artifactIds: [proposalArtifact.id],
    sourceIds: [],
    sourceRevisions: {},
    inputManifestHash: 'released-source-input-manifest',
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.updateJob(producer.id, {
    state: 'succeeded',
    stage: 'completed',
    result: { snapshotId: sourceSnapshot.id },
    finishedAt: new Date().toISOString(),
  })
  const sourceArtifactManifestHash = stableRuntimeHash('thought-base-artifacts-v1', [{
    id: proposalArtifact.id,
    kind: proposalArtifact.kind,
    contentHash: proposalArtifact.contentHash,
  }])
  const baseBinding = {
    activeSnapshotId: baseSnapshot.id,
    sourceSnapshotId: sourceSnapshot.id,
    sourceProducerJobId: producer.id,
    sourceInputManifestHash: sourceSnapshot.inputManifestHash,
    sourceArtifactManifestHash,
    baseModelFingerprint: 'released-base-model-fingerprint',
    familyRunFingerprint: 'released-family-run-fingerprint',
    languageStateFingerprint: null,
    thoughtHeadFingerprint: 'released-thought-head-fingerprint',
    thoughtHeadRevision: 0,
    conceptMacroHeadFingerprint: null,
    representationProfile: 'adaptive-eighth-id-v1',
    atlasArtifactId: null,
    atlasContentHash: null,
    atlasFingerprint: baseAtlas.fingerprint,
  }
  const releasePolicy = plasticRegionReleasePolicy(options.releasePolicy ?? store.getConfig().plasticRegions)
  const benchmarkCases = plasticRegionReleaseCases(30)
  const sourceGroupIds = benchmarkCases.map((item) => item.sourceGroupId).sort()
  const seeds = [0x51a7, 0x9e37, 0xc0de, 0xd1ce, 0xf17e]
  const rows = benchmarkCases.flatMap((benchmarkCase) => seeds.map((seed) => ({
    caseId: benchmarkCase.id,
    sourceGroupId: benchmarkCase.sourceGroupId,
    seed,
    baseScore: 0.5,
    candidateScore: 0.6,
    shamScore: 0.51,
  })))
  const benchmark = createPlasticRegionReleaseBenchmarkReport({
    evaluationId: 'released-benchmark',
    runnerFingerprint: options.runnerFingerprint ?? PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
    corpusFingerprint: plasticRegionReleaseCorpusFingerprint(30),
    engineFingerprint: 'released-engine-fingerprint',
    engineContractFingerprint: PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
    scorerFingerprint: PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
    pairingFingerprint: 'released-pairing-fingerprint',
    representationSpaceId,
    geometryFingerprint: atlas.geometryFingerprint,
    baseMode: 'off',
    baseBinding,
    runtimePolicy: {
      compatibilityBlend: releasePolicy.compatibilityBlend,
      minimumTrace: releasePolicy.minimumTrace,
      maximumActiveGates: releasePolicy.maximumActiveGates,
    },
    originalProposal: {
      artifactId: proposalArtifact.id,
      contentHash: proposalArtifact.contentHash,
      fingerprint: proposal.fingerprint,
      proposalId: proposal.proposalId,
    },
    evaluatedCandidateAtlasFingerprint: atlas.fingerprint,
    shamAtlasFingerprint: 'released-sham-atlas-fingerprint',
    maximumTicks: 16,
    sourceGroupIds,
    rows,
  })
  const benchmarkArtifact = createArtifact(
    store,
    'released-benchmark',
    PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
    benchmark,
    'released-benchmark-content',
  )
  const derived = derivePlasticRegionReleaseMetrics(benchmark)
  const benchmarkReport = {
    artifactId: benchmarkArtifact.id,
    contentHash: benchmarkArtifact.contentHash,
    fingerprint: benchmark.fingerprint,
  }
  const evidenceBase = {
    representationSpaceId,
    geometryFingerprint: atlas.geometryFingerprint,
    proposalId: proposal.proposalId,
    proposalFingerprint: proposal.fingerprint,
    proposedAtlasFingerprint: atlas.fingerprint,
    baseBinding,
    benchmarkReport,
  }
  const holdout = createPlasticRegionReleaseEvidence({
    ...evidenceBase, evaluationId: 'holdout-evaluation', evaluationKind: 'holdout',
    sourceGroupIds,
    metrics: derived.holdout,
  })
  const sham = createPlasticRegionReleaseEvidence({
    ...evidenceBase, evaluationId: 'sham-evaluation', evaluationKind: 'sham',
    sourceGroupIds,
    metrics: derived.sham,
  })
  const holdoutArtifact = createArtifact(store, 'released-holdout', PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
    holdout, 'released-holdout-content')
  const shamArtifact = createArtifact(store, 'released-sham', PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
    sham, 'released-sham-content')
  const manifest = createPlasticRegionReleaseManifest({
    releaseId: 'region-release-1', representationSpaceId,
    geometryFingerprint: atlas.geometryFingerprint,
    baseBinding,
    benchmarkReport,
    atlas: {
      artifactId: atlasArtifact.id,
      contentHash: options.manifestAtlasContentHash ?? atlasArtifact.contentHash,
      fingerprint: atlas.fingerprint,
      revision: atlas.revision, behaviorDimension: atlas.behaviorDimension,
      maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
    },
    proposal: {
      artifactId: proposalArtifact.id, contentHash: proposalArtifact.contentHash,
      fingerprint: proposal.fingerprint, proposalId: proposal.proposalId,
      baseAtlasFingerprint: proposal.baseAtlasFingerprint, proposedAtlasFingerprint: atlas.fingerprint,
    },
    holdoutEvidence: {
      artifactId: holdoutArtifact.id, contentHash: holdoutArtifact.contentHash,
      fingerprint: holdout.fingerprint, evaluationId: holdout.evaluationId,
    },
    shamEvidence: {
      artifactId: shamArtifact.id, contentHash: shamArtifact.contentHash,
      fingerprint: sham.fingerprint, evaluationId: sham.evaluationId,
    },
    policy: releasePolicy,
  })
  const manifestArtifact = createArtifact(store, 'released-manifest', PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
    manifest, 'released-manifest-content')
  return {
    atlas,
    proposal,
    manifest,
    baseSnapshot,
    sourceSnapshot,
    artifacts: [
      proposalArtifact,
      benchmarkArtifact,
      atlasArtifact,
      holdoutArtifact,
      shamArtifact,
      manifestArtifact,
    ],
  }
}

test('profile aliases end at the service boundary and physical layouts are explicit', () => {
  assert.equal(canonicalPlasticRegionProfile('legacy'), LEGACY_REGION_PROFILE)
  assert.equal(canonicalPlasticRegionProfile('adaptive-8x56'), ADAPTIVE_REGION_PROFILE)
  assert.equal(plasticRegionProfileForLayout(LEGACY_LAYOUT), LEGACY_REGION_PROFILE)
  assert.equal(plasticRegionProfileForLayout(ADAPTIVE_LAYOUT), ADAPTIVE_REGION_PROFILE)
  assert.equal(plasticRegionProfileForLayout({ ...ADAPTIVE_LAYOUT, relationLength: 3072 }), null)
})

test('Ollama lineages and cross-check groups can never become release evidence', () => {
  const common = {
    evaluationId: 'release-source-policy', evaluationKind: 'holdout',
    representationSpaceId: 'RS-RELEASE-SOURCE-POLICY',
    geometryFingerprint: ADAPTIVE_REGION_GEOMETRY.fingerprint,
    proposalId: 'proposal-source-policy', proposalFingerprint: 'proposal-fingerprint',
    proposedAtlasFingerprint: 'atlas-fingerprint',
    metrics: { caseCount: 30, independentSourceGroupCount: 1, observedGain: 0.1, regressionLoss: 0 },
  }
  for (const sourceGroupId of [
    'ollama-lineage:qwen14b',
    'ollama-cross-check:qwen14b',
    'ollama:qwen14b',
    'ollama-generated-qwen14b',
    'OLLAMA-CROSS-CHECK:QWEN14B',
  ]) {
    assert.throws(
      () => createPlasticRegionReleaseEvidence({ ...common, sourceGroupIds: [sourceGroupId] }),
      (error) => error?.code === 'OLLAMA_RELEASE_EVIDENCE_FORBIDDEN',
    )
  }
})

test('off is behavior-free and shadow may bind a deterministic empty atlas without mutating the snapshot', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const snapshot = store.createSnapshot({ id: 'region-shadow-base', artifactIds: [], sourceIds: [] })

  const off = resolvePlasticRegionRuntime({
    store, base: base(snapshot, LEGACY_LAYOUT), requestedMode: 'active',
    policy: { enabled: true, mode: 'off', representationProfile: 'legacy' },
  })
  assert.equal(off.mode, 'off')
  assert.equal(off.adaptiveRegion, undefined)

  const shadow = resolvePlasticRegionRuntime({
    store, base: base(snapshot), requestedMode: 'active',
    policy: {
      enabled: true, mode: 'shadow', representationProfile: 'adaptive-8x56',
      maximumMembershipsPerPage: 3,
    },
  })
  assert.equal(shadow.mode, 'shadow')
  assert.equal(shadow.ephemeralAtlas, true)
  assert.equal(shadow.atlas.regions.length, 0)
  assert.equal(shadow.atlas.pages.length, 56)
  assert.equal(shadow.atlas.geometryFingerprint, ADAPTIVE_REGION_GEOMETRY.fingerprint)
  assert.equal(shadow.adaptiveRegion.compatibilityBlend, 0)
  assert.equal(store.listArtifactMetadata().length, 0, 'runtime resolution must be read-only')
})

test('non-off never reinterprets a legacy layout and proposal requires an attached atlas', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const snapshot = store.createSnapshot({ id: 'region-mismatch-base', artifactIds: [], sourceIds: [] })
  assert.throws(
    () => resolvePlasticRegionRuntime({
      store, base: base(snapshot, LEGACY_LAYOUT), requestedMode: 'shadow',
      policy: { enabled: true, mode: 'shadow', representationProfile: 'adaptive-8x56' },
    }),
    (error) => error?.code === 'PLASTIC_REGION_REENCODE_REQUIRED',
  )
  assert.throws(
    () => resolvePlasticRegionRuntime({
      store, base: base(snapshot, ADAPTIVE_LAYOUT, { representationProfile: null }), requestedMode: 'shadow',
      policy: { enabled: true, mode: 'shadow', representationProfile: 'adaptive-8x56' },
    }),
    (error) => error?.code === 'PLASTIC_REGION_REENCODE_REQUIRED' && error.details?.baseProfile === null,
  )
  assert.throws(
    () => resolvePlasticRegionRuntime({
      store, base: base(snapshot), requestedMode: 'proposal',
      policy: { enabled: true, mode: 'proposal', representationProfile: 'adaptive-8x56' },
    }),
    (error) => error?.code === 'PLASTIC_REGION_ATLAS_REQUIRED',
  )
})

test('active mode rejects a generally valid snapshot carrying only a naked atlas', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const representationSpaceId = 'RS-ADAPTIVE-RUNTIME-TEST'
  const atlas = createPlasticRegionAtlas({ representationSpaceId, behaviorDimension: 8 })
  const artifact = createArtifact(store, 'region-runtime-atlas', PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    atlas, 'region-runtime-atlas-content', { activationEligible: true, candidateOnly: false })
  const snapshot = store.createSnapshot({
    id: 'region-active-base', artifactIds: [artifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  assert.throws(() => resolvePlasticRegionRuntime({
    store, base: base(store.getSnapshot(snapshot.id), ADAPTIVE_LAYOUT, {
      representationSpaceId, activeAtStart: true, commitEligible: true,
    }), requestedMode: 'active',
    policy: {
      ...store.getConfig().plasticRegions,
      enabled: true, mode: 'active', representationProfile: 'adaptive-8x56',
    },
  }), (error) => error?.code === 'PLASTIC_REGION_RELEASE_REQUIRED')
})

test('active mode accepts only the exact snapshot-bound proposal/holdout/sham release chain', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const representationSpaceId = 'RS-ADAPTIVE-RUNTIME-TEST'
  const activePolicy = {
    ...store.getConfig().plasticRegions,
    enabled: true, mode: 'active', representationProfile: 'adaptive-8x56',
    compatibilityBlend: 0.2, minimumTrace: 0.03,
  }
  const released = createReleasedAtlasArtifacts(store, representationSpaceId, { releasePolicy: activePolicy })
  const snapshot = store.createSnapshot({
    id: 'region-released-base', parentId: released.baseSnapshot.id,
    artifactIds: released.artifacts.map((artifact) => artifact.id),
    sourceIds: [...released.sourceSnapshot.sourceIds],
    sourceRevisions: { ...released.sourceSnapshot.sourceRevisions },
    inputManifestHash: released.sourceSnapshot.inputManifestHash,
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  const resolved = resolvePlasticRegionRuntime({
    store, base: base(store.getSnapshot(snapshot.id), ADAPTIVE_LAYOUT, {
      representationSpaceId, activeAtStart: true, commitEligible: true,
    }), requestedMode: 'active',
    policy: activePolicy,
  })
  assert.equal(resolved.mode, 'active')
  assert.equal(resolved.atlasArtifactId, 'released-atlas')
  assert.equal(resolved.atlas.fingerprint, released.atlas.fingerprint)
  assert.equal(resolved.releaseManifestArtifactId, 'released-manifest')
  assert.equal(resolved.releaseFingerprint, released.manifest.fingerprint)
  assert.deepEqual(resolved.adaptiveRegion, {
    mode: 'active', atlasFingerprint: released.atlas.fingerprint, compatibilityBlend: 0.2, minimumTrace: 0.03,
    maximumActiveGates: activePolicy.maximumActiveGates,
  })
  assert.throws(() => resolvePlasticRegionRuntime({
    store,
    base: base(store.getSnapshot(snapshot.id), ADAPTIVE_LAYOUT, {
      representationSpaceId, activeAtStart: true, commitEligible: true,
    }),
    requestedMode: 'active',
    policy: { ...activePolicy, maximumActiveGates: activePolicy.maximumActiveGates + 1 },
  }), (error) => error?.code === 'PLASTIC_REGION_RELEASE_REQUIRED' || error?.code === 'PLASTIC_REGION_RELEASE_INVALID')
})

for (const [name, options] of [
  ['candidate-only head', { atlasMetadata: { candidateOnly: true } }],
  ['stale atlas content reference', { manifestAtlasContentHash: 'stale-atlas-content' }],
  ['stale benchmark runner contract', { runnerFingerprint: 'retired-release-runner-contract' }],
]) {
  test(`active mode rejects ${name} in an otherwise complete release chain`, async (t) => {
    const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
    t.after(() => store.close())
    const released = createReleasedAtlasArtifacts(store, 'RS-ADAPTIVE-RUNTIME-TEST', options)
    const snapshot = store.createSnapshot({
      id: `tamper-${name}`, parentId: released.baseSnapshot.id,
      artifactIds: released.artifacts.map((artifact) => artifact.id),
      sourceIds: [...released.sourceSnapshot.sourceIds],
      sourceRevisions: { ...released.sourceSnapshot.sourceRevisions },
      inputManifestHash: released.sourceSnapshot.inputManifestHash,
      evaluation: { passed: true, runtimeVerified: true },
    })
    store.activateSnapshot(snapshot.id)
    assert.throws(() => resolvePlasticRegionRuntime({
      store,
      base: base(store.getSnapshot(snapshot.id), ADAPTIVE_LAYOUT, {
        activeAtStart: true, commitEligible: true,
      }),
      requestedMode: 'active',
      policy: {
        ...store.getConfig().plasticRegions,
        enabled: true, mode: 'active', representationProfile: 'adaptive-eighth-id-v1',
      },
    }), (error) => error?.code === 'PLASTIC_REGION_RELEASE_INVALID')
  })
}
