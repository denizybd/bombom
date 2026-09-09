import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { createServer as createViteServer } from 'vite'
import { encodeArtifactValue } from '../artifacts.mjs'
import { jsonFetch, waitFor } from '../../server/test/helpers.mjs'

const LAYOUT_SIDE = 64
const CELL_COUNT = LAYOUT_SIDE * LAYOUT_SIDE

function hashInteger(value, seed) {
  let hash = (Math.trunc(value) ^ seed) >>> 0
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d)
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b)
  return (hash ^ (hash >>> 16)) >>> 0
}

function textLabDigest(vector) {
  const buffer = new ArrayBuffer(4)
  const view = new DataView(buffer)
  let first = 0x811c9dc5
  let second = 0x9e3779b9
  for (let index = 0; index < vector.length; index += 1) {
    view.setFloat32(0, vector[index], true)
    const bits = view.getUint32(0, true)
    first = hashInteger(bits ^ index, first)
    second = hashInteger(bits ^ Math.imul(index + 1, 0x85ebca6b), second)
  }
  return `${first.toString(16).padStart(8, '0')}${second.toString(16).padStart(8, '0')}`
}

function runtimeArtifact(kind, value, id) {
  return {
    id,
    kind,
    contentHash: `${id}-hash`,
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      bytes: 1,
      metadata: {},
      value: encodeArtifactValue(value),
    },
  }
}

function seedActiveBase(store, encodings) {
  const identityLength = encodings[0].identityLength
  const layout = {
    side: LAYOUT_SIDE,
    cellCount: CELL_COUNT,
    identityLength,
    relationStart: identityLength,
    relationLength: CELL_COUNT - identityLength,
  }
  const sourceDigest = 'real-thought-source'
  const runId = 'real-thought-analysis'
  const nodes = encodings.map((encoding, index) => {
    assert.equal(encoding.signal.length, CELL_COUNT)
    assert.equal(encoding.identityLength, identityLength)
    assert.equal(encoding.representationSpaceId, encodings[0].representationSpaceId)
    const identity = encoding.signal.slice(0, identityLength)
    const relation = encoding.signal.slice(identityLength)
    return {
      id: `neuron-${index + 1}`,
      text: index === 0 ? 'Ali ekmeği kesti.' : 'Ece topu tuttu.',
      identity,
      relation,
      identityDigest: textLabDigest(identity),
      contextSupport: 2,
      contextRevision: 1,
    }
  })
  const memberKeys = nodes.map((node) => `${sourceDigest}\u0000${runId}\u0000${node.id}`)
  const analysis = {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId,
    sourceDigest,
    representationSpaceId: encodings[0].representationSpaceId,
    layout,
    nodes,
    habitProbabilities: new Float32Array([0, 1, 1, 0]),
  }
  const familyRun = {
    format: 'noron-aileleri-run',
    version: 1,
    id: 'real-thought-family-run',
    fingerprint: 'real-thought-family-fingerprint',
    representationSpaceId: analysis.representationSpaceId,
    behaviorSpaceId: 'real-thought-behavior-space',
    profiles: nodes.map((node, index) => ({
      memberKey: memberKeys[index],
      kind: 'raw',
      identityDigest: node.identityDigest,
      text: node.text,
      representationSpaceId: analysis.representationSpaceId,
      behaviorSpaceId: 'real-thought-behavior-space',
      vector: {},
      diagnostics: {},
      regionAffinities: [],
      evidence: [],
      derivedFromMemberKeys: [],
    })),
    families: [{
      id: 'family-event',
      region: 'event',
      lifecycle: 'stable',
      supportCount: 2,
      sourceCount: 2,
      memberKeys,
      coreMemberKeys: memberKeys,
      cohesion: 1,
      separation: 1,
      stability: 1,
    }],
    memberships: memberKeys.map((memberKey) => ({
      memberKey,
      familyId: 'family-event',
      rank: 1,
      score: 1,
      weight: 1,
      core: true,
    })),
    neuronTransitions: [
      { fromMemberKey: memberKeys[0], toMemberKey: memberKeys[1], probability: 1 },
      { fromMemberKey: memberKeys[1], toMemberKey: memberKeys[0], probability: 1 },
    ],
    familyTransitions: [],
    utilityReports: [],
    coalition: { tick: 0, activeFamilyCount: 0, coalitions: [], unmatchedReason: null },
    routingPreview: null,
  }
  const analysisArtifact = store.createArtifact(runtimeArtifact('text-lab-analysis', analysis, 'real-thought-analysis-artifact'))
  const familyArtifact = store.createArtifact(runtimeArtifact('family-run', familyRun, 'real-thought-family-artifact'))
  const snapshot = store.createSnapshot({
    id: 'real-thought-active-snapshot',
    status: 'candidate',
    label: 'Gerçek düşünce motoru entegrasyon tabanı',
    artifactIds: [analysisArtifact.id, familyArtifact.id],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  return { snapshot: store.getSnapshot(snapshot.id), memberKeys }
}

test('runtime başlangıcı gerçek 64x64 düşünce motorunu servise enjekte eder ve kısa bölüm tamamlar', { timeout: 20_000 }, async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'bombom-real-thought-'))
  const vite = await createViteServer({
    appType: 'custom',
    logLevel: 'silent',
    server: { middlewareMode: true },
  })
  let service
  t.after(async () => {
    await service?.close()
    await vite.close()
    await rm(directory, { recursive: true, force: true })
  })

  const [{ startRealTrainingService }, { encodeThoughtPrompt }] = await Promise.all([
    vite.ssrLoadModule('/runtime/start.mjs'),
    vite.ssrLoadModule('/src/thought-runtime/prompt.ts'),
  ])
  const encodings = await Promise.all([
    encodeThoughtPrompt('Ali ekmeği kesti.', { allowFlatFallback: true }),
    encodeThoughtPrompt('Ece topu tuttu.', { allowFlatFallback: true }),
  ])
  const started = await startRealTrainingService({ port: 0, dataDirectory: directory, forceJson: true })
  service = started.service
  const base = `http://127.0.0.1:${started.address.port}`
  const { snapshot, memberKeys } = seedActiveBase(service.store, encodings)

  const status = await jsonFetch(base, '/api/thought/status')
  assert.equal(status.response.status, 200)
  assert.equal(status.body.available, true)
  assert.equal(status.body.engine.available, true)
  assert.equal(service.thought.engine.config.matrixSide, LAYOUT_SIDE)
  assert.equal(service.thought.engine.config.maximumBranches, 8)

  const created = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST',
    headers: { 'Idempotency-Key': 'real-thought-engine-episode' },
    body: JSON.stringify({
      text: 'Ali ekmeği kesti.',
      snapshotId: snapshot.id,
      seed: 41,
      maxTicks: 2,
      strategy: 'greedy',
      learningMode: 'off',
      teacherMode: 'off',
      candidateMemberKeys: memberKeys,
    }),
  })
  assert.equal(created.response.status, 202, JSON.stringify(created.body))

  const episodeId = created.body.episode.id
  const completed = await waitFor(() => {
    const episode = service.store.getThoughtEpisode(episodeId)
    return ['completed', 'failed', 'cancelled'].includes(episode?.status) ? episode : null
  }, { timeoutMs: 10_000 })
  assert.equal(completed.status, 'completed', JSON.stringify(completed.error))
  assert.equal(completed.tickCount, 2)
  assert.equal(completed.engineSummary.status, 'stopped')

  const ticks = await jsonFetch(base, `/api/thought/episodes/${episodeId}/ticks?limit=8&candidateLimit=8`)
  assert.equal(ticks.response.status, 200)
  assert.equal(ticks.body.ticks.length, 2)
  for (const tick of ticks.body.ticks) {
    assert.ok(tick.replayFingerprint)
    assert.ok(tick.branches.length >= 1 && tick.branches.length <= 8)
    for (const branch of tick.branches) {
      assert.equal(branch.signal.before.length, CELL_COUNT)
      assert.equal(branch.signal.after.length, CELL_COUNT)
    }
  }

  const graph = await jsonFetch(base, `/api/thought/episodes/${episodeId}/graph?level=neuron`)
  assert.equal(graph.response.status, 200)
  assert.equal(graph.body.graph.audit.available, true)
  assert.ok(graph.body.graph.nodes.some((node) => memberKeys.includes(node.id)))

  const replay = await jsonFetch(base, `/api/thought/episodes/${episodeId}/replay`, {
    method: 'POST',
    body: '{}',
  })
  assert.equal(replay.response.status, 200, JSON.stringify(replay.body))
  assert.equal(replay.body.replay.matched, true, JSON.stringify(replay.body.replay.firstMismatch))
  assert.equal(replay.body.replay.expectedTickCount, 2)
  assert.equal(replay.body.replay.actualTickCount, 2)
})
