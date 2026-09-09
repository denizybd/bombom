import assert from 'node:assert/strict'
import test from 'node:test'
import { encodeArtifactValue } from '../artifacts.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'
import { createTrainingStore } from '../../server/store.mjs'
import {
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  currentPlasticRegionBase,
  persistPlasticRegionArtifact,
  readPlasticRegionSnapshot,
} from '../plastic-region-cache.mjs'
import { createPlasticRegionAtlas } from '../../src/adaptive-regions/index.ts'

function emptyAtlas(representationSpaceId = 'RS-PLASTIC-TEST', revision = 0, parentFingerprint = null) {
  return createPlasticRegionAtlas({
    representationSpaceId,
    revision,
    parentFingerprint,
    behaviorDimension: 8,
    maximumMembershipsPerPage: 3,
  })
}

test('plastic-region artifacts are immutable, content addressed and CAS fenced from a stale atlas base', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const initial = store.createSnapshot({
    id: 'plastic-base-0', status: 'candidate', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(initial.id)
  const base0 = currentPlasticRegionBase(store)
  assert.deepEqual(base0, {
    snapshotId: initial.id,
    atlasArtifactId: null,
    atlasContentHash: null,
    atlasFingerprint: null,
  })

  const atlas0 = emptyAtlas()
  const first = persistPlasticRegionArtifact({ store }, {
    kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    value: atlas0,
    expectedBase: base0,
  })
  const duplicate = persistPlasticRegionArtifact({ store }, {
    kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    value: atlas0,
    expectedBase: base0,
  })
  assert.equal(duplicate.artifact.id, first.artifact.id)

  const candidate = store.createSnapshot({
    id: 'plastic-base-1', parentId: initial.id, status: 'candidate',
    artifactIds: [first.artifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(candidate.id)
  const restored = readPlasticRegionSnapshot(store)
  assert.equal(restored.compatible, true)
  assert.equal(restored.head.fingerprint, atlas0.fingerprint)

  assert.throws(
    () => persistPlasticRegionArtifact({ store }, {
      kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      value: emptyAtlas('RS-PLASTIC-TEST', 1, atlas0.fingerprint),
      expectedBase: base0,
    }),
    (error) => error?.code === 'STALE_REGION_ATLAS_BASE' && error.details?.field === 'snapshotId',
  )
})

test('store CAS rejects a changed attached head even when the snapshot id is unchanged', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const head = store.createArtifact({
    kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: { fingerprint: 'atlas-current' }, value: emptyAtlas(),
    },
  })
  const snapshot = store.createSnapshot({
    id: 'cas-active', status: 'candidate', artifactIds: [head.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)

  assert.throws(
    () => store.createArtifactOnSnapshotHeadBase({ kind: 'plastic-region-shadow-report', payload: {} }, {
      snapshotId: snapshot.id,
      headKind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      headArtifactId: head.id,
      headFingerprint: 'atlas-stale',
      staleCode: 'STALE_REGION_ATLAS_BASE',
    }),
    (error) => error?.code === 'STALE_REGION_ATLAS_BASE' && error.details?.field === 'headFingerprint',
  )
  assert.equal(store.listArtifactMetadata().length, 1, 'failed CAS must not create an orphan artifact')
})

for (const scenario of ['page-content', 'metadata', 'behavior-dimension']) {
  test(`atlas reader quarantines ${scenario} tamper`, async (t) => {
    const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
    t.after(() => store.close())
    let value = scenario === 'behavior-dimension'
      ? createPlasticRegionAtlas({ representationSpaceId: 'RS-TAMPER', behaviorDimension: 2 })
      : emptyAtlas('RS-TAMPER')
    if (scenario === 'page-content') {
      value = structuredClone(value)
      value.pages[0].freeMass = 0.5
    }
    const artifact = store.createArtifact({
      id: `tampered-${scenario}`,
      kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      contentHash: `tampered-${scenario}-content`,
      payload: {
        format: 'bombom-runtime-artifact', version: 1, bytes: 1,
        metadata: {
          fingerprint: scenario === 'metadata' ? 'forged-fingerprint' : value.fingerprint,
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
      id: `tamper-snapshot-${scenario}`, status: 'candidate', artifactIds: [artifact.id], sourceIds: [],
      evaluation: { passed: true, runtimeVerified: true },
    })
    store.activateSnapshot(snapshot.id)
    const restored = readPlasticRegionSnapshot(store)
    assert.equal(restored.compatible, false)
    assert.match(restored.reason, /^atlas-head-invalid:/)
  })
}
