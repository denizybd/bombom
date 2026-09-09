import assert from 'node:assert/strict'
import test from 'node:test'
import { encodeArtifactValue } from '../../runtime/artifacts.mjs'
import { createTrainingService } from '../http-server.mjs'
import { jsonFetch, listenService, temporaryDirectory } from './helpers.mjs'

function conceptHead() {
  return {
    format: 'concept-macro-head',
    version: 1,
    id: 'head-concept-test',
    fingerprint: 'head-concept-fingerprint',
    revision: 7,
    representationSpaceId: 'RS-concept-test',
    familyRunFingerprint: 'family-run-test',
    rawConstituentsRetained: true,
    stats: { acceptedObservationCount: 12 },
    macros: [{
      id: 'CMACRO-TEST',
      fingerprint: 'macro-fingerprint',
      kind: 'sequence-macro',
      lifecycle: 'shadow-validated',
      orderedExpansion: [
        { kind: 'raw-neuron', id: 'member-a', identityDigest: 'digest-a', familyIds: ['family-a'] },
        { kind: 'composite-neuron', id: 'CMACRO-CHILD', identityDigest: null, familyIds: ['family-b'] },
      ],
      observationCount: 12,
      independentGroupCount: 3,
      evidence: [
        { observationId: 'o1', sourceId: 's1', sourceDigest: 'd1', independenceGroupId: 'book-1', split: 'train', tickStart: 0, tickEnd: 2 },
        { observationId: 'o2', sourceId: 's2', sourceDigest: 'd2', independenceGroupId: 'book-2', split: 'train', tickStart: 1, tickEnd: 3 },
        { observationId: 'o3', sourceId: 's3', sourceDigest: 'd3', independenceGroupId: 'book-3', split: 'train', tickStart: 2, tickEnd: 4 },
      ],
      effectCohesion: 0.91,
      mdl: { compressionGain: 9 },
      metrics: {
        activationRecall: 0.96,
        fidelity: 0.98,
        falseActivationRate: 0.02,
        holdoutUtility: 0.08,
      },
      depth: 2,
      expansionDigest: 'expansion-test',
      physicalIdentityAllocated: false,
      rawConstituentsRetained: true,
      structuralEquivalencePassed: false,
      structuralEquivalenceAudit: null,
      shadowReportIds: ['shadow-1'],
      // Dense internals must never cross the browser API.
      entryPrototype: { vector: new Array(4_096).fill(0.25), digest: 'entry' },
      exitPrototype: { vector: new Array(4_096).fill(0.5), digest: 'exit' },
      effectPrototype: new Array(4_096).fill(0.25),
    }],
  }
}

test('concept inspection projects macro evidence without exposing dense vectors', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const head = conceptHead()
  service.store.createArtifact({
    id: 'concept-head-artifact',
    kind: 'concept-macro-head',
    contentHash: head.fingerprint,
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      metadata: { mode: 'proposal-shadow' },
      value: encodeArtifactValue(head),
    },
  })
  const snapshot = service.store.createSnapshot({
    status: 'candidate',
    artifactIds: ['concept-head-artifact'],
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(snapshot.id)
  let headReads = 0
  const getArtifact = service.store.getArtifact.bind(service.store)
  service.store.getArtifact = (id) => {
    if (id === 'concept-head-artifact') headReads += 1
    return getArtifact(id)
  }
  const base = await listenService(t, service)

  const list = await jsonFetch(base, '/api/concepts/macros')
  assert.equal(list.response.status, 200)
  assert.equal(list.body.available, true)
  assert.equal(list.body.mode, 'proposal-shadow')
  assert.equal(list.body.head.snapshotStatus, 'active')
  assert.equal(list.body.stats.observationCount, 12)
  assert.equal(list.body.stats.sourceGroupCount, 3)
  assert.equal(list.body.stats.shadowEvaluationCount, 1)
  assert.equal(list.body.macros[0].completionProbability, 0.96)
  assert.equal(list.body.macros[0].shadowFidelity, 0.98)
  assert.equal(list.body.macros[0].structuralEquivalencePassed, false)
  assert.equal(list.body.stats.structuralValidatedCount, 0)
  assert.equal(list.body.macros[0].orderedExpansion[0].memberKey, 'member-a')
  assert.equal(list.body.macros[0].orderedExpansion[1].macroId, 'CMACRO-CHILD')
  assert.doesNotMatch(JSON.stringify(list.body), /entryPrototype|exitPrototype|effectPrototype|"vector"/u)

  const detail = await jsonFetch(base, '/api/concepts/macros/CMACRO-TEST')
  assert.equal(detail.response.status, 200)
  assert.equal(detail.body.macro.evidence.length, 3)
  assert.deepEqual(detail.body.macro.sourceGroupIds, ['book-1', 'book-2', 'book-3'])
  assert.doesNotMatch(JSON.stringify(detail.body), /entryPrototype|exitPrototype|effectPrototype|"vector"/u)
  assert.equal(headReads, 1, 'poll/list/detail must reuse the immutable decoded head')
})

test('concept inspection is explicitly empty before the first macro head', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const result = await jsonFetch(base, '/api/concepts/status')
  assert.equal(result.response.status, 200)
  assert.equal(result.body.available, false)
  assert.equal(result.body.stats.totalMacros, 0)
})
