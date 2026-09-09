import assert from 'node:assert/strict'
import test from 'node:test'
import { encodeArtifactValue } from '../../runtime/artifacts.mjs'
import { PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND } from '../../runtime/plastic-region-cache.mjs'
import { createTrainingService } from '../http-server.mjs'
import { inspectPlasticRegionStatus, listPlasticRegionRecords } from '../plastic-region-inspection.mjs'
import { createTrainingStore } from '../store.mjs'
import { toPublicQueueItem } from '../public-api.mjs'
import { jsonFetch, listenService, temporaryDirectory } from './helpers.mjs'
import { createPlasticRegionAtlas } from '../../src/adaptive-regions/index.ts'

function atlas() {
  return createPlasticRegionAtlas({
    representationSpaceId: 'RS-ADAPTIVE-TEST',
    revision: 2,
    parentFingerprint: 'atlas-parent',
    behaviorDimension: 8,
    maximumMembershipsPerPage: 3,
    regions: [{
      id: 'latent-region-1', lifecycle: 'active', origin: 'discovery', bornRevision: 1,
      retiredRevision: null, parentRegionIds: [],
      prototype: { dimension: 8, values: [0.25, -0.5, 0, 0, 0, 0, 0, 0] },
      evidence: {
        observationCount: 24, sourceGroupIds: ['source-a', 'source-b', 'source-c'],
        confidenceMass: 3, cohesion: 0.91, holdoutUtility: 0.14, shamUtility: 0.01,
      },
    }],
    pages: Array.from({ length: 56 }, (_, pageIndex) => pageIndex === 0
      ? { pageIndex, memberships: [{ regionId: 'latent-region-1', weight: 1 }], freeMass: 0 }
      : { pageIndex, memberships: [], freeMass: 1 }),
  })
}

test('plastic-region HTTP surface reports configured/effective modes separately and exposes bounded atlas detail', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const value = atlas()
  const artifact = store.createArtifact({
    id: 'plastic-atlas-artifact',
    kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    contentHash: 'plastic-atlas-content',
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: {
        fingerprint: value.fingerprint,
        representationSpaceId: value.representationSpaceId,
        geometryFingerprint: value.geometryFingerprint,
        revision: value.revision,
        behaviorDimension: value.behaviorDimension,
        maximumMembershipsPerPage: value.maximumMembershipsPerPage,
      },
      value: encodeArtifactValue(value),
    },
  })
  const snapshot = store.createSnapshot({
    id: 'plastic-atlas-snapshot', status: 'candidate', artifactIds: [artifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  const candidateOnly = inspectPlasticRegionStatus(store)
  assert.equal(candidateOnly.authority, 'candidate-research')
  assert.equal(candidateOnly.routingEligible, false)
  assert.equal(candidateOnly.activationCandidateSnapshotId, null, 'a naked atlas candidate is not activation-ready')
  store.activateSnapshot(snapshot.id)
  const service = await createTrainingService({ store })
  const base = await listenService(t, service)

  const status = await jsonFetch(base, '/api/regions/status')
  assert.equal(status.response.status, 200)
  assert.equal(status.body.configuredMode, 'off')
  assert.equal(status.body.effectiveMode, 'off')
  assert.equal(status.body.effectiveModeReason, 'policy-off')
  assert.equal(status.body.runtimeWired, true)
  assert.equal(status.body.authority, 'active')
  assert.equal(status.body.routingEligible, false, 'an active atlas without release remains behavior-ineligible')
  assert.equal(status.body.available, true)
  assert.equal(status.body.head.fingerprint, value.fingerprint)

  service.store.updateConfig({ plasticRegions: {
    representationProfile: 'adaptive-eighth-id-v1',
    mode: 'shadow',
  } })
  const shadowStatus = await jsonFetch(base, '/api/regions/status')
  assert.equal(shadowStatus.body.configuredMode, 'shadow')
  assert.equal(shadowStatus.body.effectiveMode, 'shadow')
  assert.equal(shadowStatus.body.effectiveModeReason, 'configured-opt-in')

  const current = await jsonFetch(base, '/api/regions/atlas')
  assert.equal(current.response.status, 200)
  assert.equal(current.body.atlas.regions[0].id, 'latent-region-1')
  assert.equal(current.body.atlas.regions[0].evidence.sourceGroupCount, 3)
  assert.equal(current.body.atlas.pages.length, 56)
  assert.equal(current.body.atlas.stats.assignedMass, 1)
  assert.equal(current.body.atlas.stats.freeMass, 55)

  const listing = await jsonFetch(base, '/api/regions/artifacts?kind=plastic-region-atlas-head&limit=10')
  assert.equal(listing.response.status, 200)
  assert.equal(listing.body.pagination.total, 1)
  assert.equal(listing.body.items[0].artifactId, artifact.id)

  const detail = await jsonFetch(base, `/api/regions/atlases/${artifact.id}`)
  assert.equal(detail.response.status, 200)
  assert.equal(detail.body.artifact.kind, PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND)
  assert.equal(detail.body.atlas.fingerprint, value.fingerprint)

  const missing = await jsonFetch(base, '/api/regions/atlases/unknown')
  assert.equal(missing.response.status, 404)
  assert.equal(missing.body.code, 'PLASTIC_REGION_ATLAS_NOT_FOUND')
})

test('plastic-region status is an honest empty scaffold before any atlas head exists', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const status = await jsonFetch(base, '/api/regions/status')

  assert.equal(status.response.status, 200)
  assert.equal(status.body.available, false)
  assert.equal(status.body.reason, 'atlas-head-absent')
  assert.equal(status.body.representationProfile, 'legacy')
  assert.equal(status.body.counts.totalArtifacts, 0)

  const probes = await jsonFetch(base, '/api/regions/probes?limit=16')
  assert.equal(probes.response.status, 200)
  assert.equal(probes.body.available, false)
  assert.equal(probes.body.interpretationOnly, true)
  assert.equal(probes.body.routingMutationAllowed, false)
  assert.deepEqual(probes.body.summary.items, [])

  const invalid = await jsonFetch(base, '/api/regions/probes?limit=999')
  assert.equal(invalid.response.status, 400)
  assert.equal(invalid.body.code, 'INVALID_SEMANTIC_PROBE_INSPECTION')
})

test('release job projection distinguishes a completed rejection from an approved manual candidate', () => {
  const rejected = toPublicQueueItem({
    id: 'release-rejected', type: 'plastic-region-release-evaluation', state: 'succeeded',
    stage: 'completed', progress: 1, attempt: 1, createdAt: '2026-01-01', updatedAt: '2026-01-01',
    result: {
      released: false, automaticActivation: false, evaluationId: 'evaluation-1',
      auditArtifactId: 'audit-1', reasonCodes: ['HOLDOUT_GAIN_TOO_LOW'],
      metrics: { baseMean: 0.5, candidateMean: 0.51, conservativeHoldoutGain: 0.01 },
      privateRows: Array.from({ length: 200 }, () => ({ prompt: 'must-not-leak' })),
    },
  })
  assert.equal(rejected.state, 'complete')
  assert.deepEqual(rejected.releaseResult.reasonCodes, ['HOLDOUT_GAIN_TOO_LOW'])
  assert.equal(rejected.releaseResult.released, false)
  assert.equal(rejected.releaseResult.auditArtifactId, 'audit-1')
  assert.equal(JSON.stringify(rejected).includes('must-not-leak'), false)

  const approved = toPublicQueueItem({
    id: 'release-approved', type: 'plastic-region-release-evaluation', state: 'succeeded',
    stage: 'completed', progress: 1, attempt: 1, createdAt: '2026-01-01', updatedAt: '2026-01-01',
    result: { released: true, awaitingApproval: true, automaticActivation: false, snapshotId: 'candidate-1' },
  })
  assert.equal(approved.releaseResult.released, true)
  assert.equal(approved.releaseResult.snapshotId, 'candidate-1')
  assert.equal(approved.releaseResult.automaticActivation, false)

  const renewal = toPublicQueueItem({
    id: 'renewal-approved', type: 'plastic-region-release-renewal', state: 'succeeded',
    stage: 'completed', progress: 1, attempt: 1, createdAt: '2026-01-01', updatedAt: '2026-01-01',
    result: {
      renewed: true, released: true, awaitingApproval: true, automaticActivation: false,
      renewalId: 'renewal-1', snapshotId: 'candidate-2', releaseFingerprint: 'release-fp',
      evidenceFingerprint: 'evidence-fp', auditArtifactId: 'audit-2', evidenceArtifactId: 'evidence-2',
      manifestArtifactId: 'manifest-2', reasonCodes: [],
      metrics: {
        holdout: { caseCount: 150, independentSourceGroupCount: 30, observedGain: 0.2, regressionLoss: 0 },
        sham: { caseCount: 150, independentSourceGroupCount: 30, observedGain: 0, regressionLoss: 0 },
      },
      rows: [{ prompt: 'must-not-leak-renewal' }],
    },
  })
  assert.equal(renewal.releaseResult.renewed, true)
  assert.equal(renewal.releaseResult.releaseKind, 'unchanged-atlas-model-renewal')
  assert.equal(renewal.releaseResult.renewalId, 'renewal-1')
  assert.equal(renewal.releaseResult.evidenceFingerprint, 'evidence-fp')
  assert.equal(renewal.releaseResult.metrics.holdoutCaseCount, 150)
  assert.equal(renewal.releaseResult.metrics.holdoutObservedGain, 0.2)
  assert.equal(JSON.stringify(renewal).includes('must-not-leak-renewal'), false)
})

test('status/history use compact kind indexes and stale evidence cannot open proposal readiness', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const base = store.createSnapshot({
    id: 'readiness-active', status: 'candidate', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(base.id)
  store.updateConfig({ plasticRegions: {
    enabled: true, representationProfile: 'adaptive-eighth-id-v1', mode: 'shadow',
  } })
  store.createArtifact({
    id: 'stale-region-evidence', kind: 'plastic-region-evidence-head', contentHash: 'stale-hash',
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: { baseSnapshotId: 'other-base', casBaseAtlasFingerprint: null },
      value: { deliberately: 'invalid-but-never-authoritative' },
    },
  })
  store.listArtifactMetadata = () => { throw new Error('global artifact scan forbidden') }

  const status = inspectPlasticRegionStatus(store)
  assert.equal(status.counts.byKind['plastic-region-evidence-head'], 1)
  assert.equal(status.counts.source, 'kind-indexed-exact')
  assert.equal(status.readiness.shadowEvidenceAvailable, false)
  assert.equal(status.readiness.reason, 'current-base-evidence-not-found')
  const listing = listPlasticRegionRecords(store, { kind: 'plastic-region-evidence-head', offset: 0, limit: 10 })
  assert.equal(listing.items.length, 1)
  assert.equal(listing.pagination.total, 1)
})
