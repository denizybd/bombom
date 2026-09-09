import assert from 'node:assert/strict'
import test from 'node:test'
import { persistArtifact } from '../artifacts.mjs'
import {
  LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND,
  LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND,
  readLanguageOutputSnapshot,
} from '../language-output-cache.mjs'
import { createTrainingStore } from '../../server/store.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'

test('language output artifacts remain separate and require matching scaffold fingerprints', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const context = { store, jobId: null }
  const scaffold = { format: 'language-output-scaffold', version: 1, fingerprint: 'scaffold-a', families: [] }
  const head = { format: 'language-output-head', version: 1, fingerprint: 'head-a', scaffoldFingerprint: 'scaffold-a' }
  const scaffoldArtifact = persistArtifact(context, {
    kind: LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND, contentHash: 'scaffold-a', value: scaffold,
  }).artifact
  const headArtifact = persistArtifact(context, {
    kind: LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND, contentHash: 'head-a', value: head,
  }).artifact
  const snapshot = store.createSnapshot({ status: 'candidate', artifactIds: [scaffoldArtifact.id, headArtifact.id] })

  const resolved = readLanguageOutputSnapshot(store, snapshot)
  assert.equal(resolved.compatible, true)
  assert.deepEqual(resolved.scaffold, scaffold)
  assert.deepEqual(resolved.head, head)
})

test('a mismatched or oversized plastic head fails closed', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const context = { store, jobId: null }
  const scaffoldArtifact = persistArtifact(context, {
    kind: LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND,
    contentHash: 'scaffold-b',
    value: { format: 'language-output-scaffold', version: 1, fingerprint: 'scaffold-b', families: [] },
  }).artifact
  const headArtifact = persistArtifact(context, {
    kind: LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND,
    contentHash: 'head-b',
    value: { format: 'language-output-head', version: 1, fingerprint: 'head-b', scaffoldFingerprint: 'other' },
  }).artifact
  const snapshot = store.createSnapshot({ status: 'candidate', artifactIds: [scaffoldArtifact.id, headArtifact.id] })

  const mismatch = readLanguageOutputSnapshot(store, snapshot)
  assert.equal(mismatch.mismatch, true)
  assert.equal(mismatch.head, null)

  const bounded = readLanguageOutputSnapshot(store, snapshot, { maximumPayloadBytes: 1 })
  assert.equal(bounded.skippedHydration, true)
  assert.equal(bounded.head, null)
})
