import assert from 'node:assert/strict'
import test from 'node:test'
import { createMacroHead } from '../../src/concept-macros/index.ts'
import { createTrainingStore } from '../../server/store.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'
import { persistArtifact } from '../artifacts.mjs'
import {
  CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
  readLatestConceptMacroHead,
} from '../concept-macro-cache.mjs'

test('concept head lookup selects the newest compatible representation, not the newest unrelated head', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const context = { store, jobId: null }
  const compatible = createMacroHead({ representationSpaceId: 'RS-A', familyRunFingerprint: 'family-a' })
  const unrelated = createMacroHead({ representationSpaceId: 'RS-B', familyRunFingerprint: 'family-b' })
  const first = persistArtifact(context, {
    kind: CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    contentHash: compatible.fingerprint,
    value: compatible,
  }).artifact
  const second = persistArtifact(context, {
    kind: CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    contentHash: unrelated.fingerprint,
    value: unrelated,
  }).artifact
  const snapshot = store.createSnapshot({
    status: 'candidate', artifactIds: [first.id, second.id],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)

  const result = readLatestConceptMacroHead(store, 'RS-A')
  assert.equal(result.head.fingerprint, compatible.fingerprint)
  assert.equal(result.artifact.id, first.id)
  assert.equal(result.quarantinedReason, null)
})

test('concept head lookup fails closed when no representation is compatible', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const context = { store, jobId: null }
  const head = createMacroHead({ representationSpaceId: 'RS-B', familyRunFingerprint: 'family-b' })
  const artifact = persistArtifact(context, {
    kind: CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    contentHash: head.fingerprint,
    value: head,
  }).artifact
  const snapshot = store.createSnapshot({
    status: 'candidate', artifactIds: [artifact.id],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  const result = readLatestConceptMacroHead(store, 'RS-A')
  assert.equal(result.head, null)
  assert.match(result.quarantinedReason, /^representation-space-mismatch:/u)
})

test('concept head lookup never inherits orphan or rejected artifacts', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const context = { store, jobId: null }
  const orphan = createMacroHead({ representationSpaceId: 'RS-A', familyRunFingerprint: 'orphan-family' })
  const rejected = createMacroHead({ representationSpaceId: 'RS-A', familyRunFingerprint: 'rejected-family' })
  persistArtifact(context, {
    kind: CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    contentHash: orphan.fingerprint,
    value: orphan,
  })
  const rejectedArtifact = persistArtifact(context, {
    kind: CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    contentHash: rejected.fingerprint,
    value: rejected,
  }).artifact
  store.createSnapshot({
    status: 'rejected', artifactIds: [rejectedArtifact.id],
    evaluation: { passed: false, runtimeVerified: false },
  })

  const result = readLatestConceptMacroHead(store, 'RS-A')
  assert.equal(result.head, null)
  assert.equal(result.quarantinedReason, 'no-trusted-snapshot-lineage')
})
