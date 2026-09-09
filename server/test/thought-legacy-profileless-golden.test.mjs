import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile, rename, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'

import { encodeArtifactValue } from '../../runtime/artifacts.mjs'
import { createTrainingService } from '../http-server.mjs'
import { temporaryDirectory, waitFor } from './helpers.mjs'

const LEGACY_LAYOUT = Object.freeze({
  side: 64,
  cellCount: 4_096,
  identityLength: 1_024,
  relationStart: 1_024,
  relationLength: 3_072,
})

const SOURCE_DIGEST = 'profileless-legacy-source'
const ANALYSIS_RUN_ID = 'profileless-legacy-analysis'
const NEURON_ID = 'profileless-legacy-neuron'
const MEMBER_KEY = `${SOURCE_DIGEST}\u0000${ANALYSIS_RUN_ID}\u0000${NEURON_ID}`
const REPRESENTATION_SPACE_ID = 'RS-profileless-legacy-16x48-golden'
const STATE_FILENAME = 'autonomous-training.json'
const GOLDEN_IDENTITY_BYTES_DIGEST = '1d678fdb968d7a859f65d6b6f3df77bcbf68863dc66f707df558115d840c1d54'
const GOLDEN_RELATION_BYTES_DIGEST = '687be15b9d69897047e7b6b4adf8328d115c59705561cc3b08db66b616cc29ed'
const GOLDEN_TICK_DIGESTS = Object.freeze([
  '3fe4506d471f69040d9e2056cace6db455b4fed1d9f55d85aba57d0d79699c0d',
  '5ddc9b1ad15363666a3df60cc70590ee5bb1e27e456cde6ed99f4ae0dc351843',
])

const PRE_PLASTIC_EPISODE_FIELDS = Object.freeze([
  'requestedRegionMode',
  'regionMode',
  'regionModeReason',
  'regionAtlasArtifactId',
  'regionAtlasFingerprint',
  'regionAtlasRevision',
  'regionGeometryFingerprint',
  'regionRepresentationSpaceId',
  'regionReleaseManifestArtifactId',
  'regionReleaseFingerprint',
  'regionEphemeralAtlas',
  'regionCompatibilityBlend',
  'regionMinimumTrace',
  'regionMaximumActiveGates',
  'regionBindingFingerprint',
])

const PRE_PLASTIC_BASE_FIELDS = Object.freeze([
  'regionAtlasArtifactId',
  'regionAtlasFingerprint',
  'regionAtlasRevision',
  'regionGeometryFingerprint',
  'regionReleaseManifestArtifactId',
  'regionReleaseFingerprint',
])

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

function float32BytesDigest(vector) {
  return createHash('sha256')
    .update(Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength))
    .digest('hex')
}

function runtimeArtifact(kind, value, id) {
  return {
    id,
    kind,
    contentHash: `${id}-content`,
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      bytes: 1,
      metadata: {},
      value: encodeArtifactValue(value),
    },
  }
}

function seedProfilelessLegacyBase(store) {
  const identity = new Float32Array(LEGACY_LAYOUT.identityLength)
  identity[0] = Math.fround(0.75)
  identity[257] = Math.fround(-0.375)
  identity[LEGACY_LAYOUT.identityLength - 1] = Math.fround(0.125)

  const relation = new Float32Array(LEGACY_LAYOUT.relationLength)
  relation[0] = Math.fround(-0.25)
  relation[1_537] = Math.fround(0.625)
  relation[LEGACY_LAYOUT.relationLength - 1] = Math.fround(-0.125)

  const identityDigest = textLabDigest(identity)
  const analysis = {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId: ANALYSIS_RUN_ID,
    sourceDigest: SOURCE_DIGEST,
    representationSpaceId: REPRESENTATION_SPACE_ID,
    // This omission is intentional: historical 64x64 records had no
    // representationProfile and therefore retain their exact 16/48 layout.
    layout: { ...LEGACY_LAYOUT },
    nodes: [{
      id: NEURON_ID,
      text: 'Eski profilsiz nöron değişmeden kalır.',
      identity,
      relation,
      identityDigest,
      contextSupport: 3,
      contextRevision: 1,
    }],
    habitProbabilities: new Float32Array([1]),
  }
  const familyRun = {
    format: 'noron-aileleri-run',
    version: 1,
    id: 'profileless-legacy-family-run',
    fingerprint: 'profileless-legacy-family-run-fingerprint',
    representationSpaceId: REPRESENTATION_SPACE_ID,
    behaviorSpaceId: 'BS-profileless-legacy-golden',
    profiles: [{
      memberKey: MEMBER_KEY,
      kind: 'raw',
      identityDigest,
      text: 'Eski profilsiz nöron değişmeden kalır.',
      representationSpaceId: REPRESENTATION_SPACE_ID,
      behaviorSpaceId: 'BS-profileless-legacy-golden',
      vector: {},
      diagnostics: {},
      regionAffinities: [],
      evidence: [],
      derivedFromMemberKeys: [],
    }],
    families: [{
      id: 'profileless-legacy-family',
      region: 'event',
      lifecycle: 'stable',
      supportCount: 3,
      sourceCount: 2,
      memberKeys: [MEMBER_KEY],
      coreMemberKeys: [MEMBER_KEY],
      cohesion: 1,
      separation: 1,
      stability: 1,
    }],
    memberships: [{
      memberKey: MEMBER_KEY,
      familyId: 'profileless-legacy-family',
      rank: 1,
      score: 1,
      weight: 1,
      core: true,
    }],
    neuronTransitions: [{ fromMemberKey: MEMBER_KEY, toMemberKey: MEMBER_KEY, probability: 1 }],
    familyTransitions: [],
    utilityReports: [],
  }

  const analysisArtifact = store.createArtifact(runtimeArtifact(
    'text-lab-analysis', analysis, 'profileless-legacy-analysis-artifact',
  ))
  const familyArtifact = store.createArtifact(runtimeArtifact(
    'family-run', familyRun, 'profileless-legacy-family-artifact',
  ))
  const snapshot = store.createSnapshot({
    id: 'profileless-legacy-active-snapshot',
    status: 'candidate',
    label: 'Profilsiz 16/48 altın regresyon tabanı',
    artifactIds: [analysisArtifact.id, familyArtifact.id],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  return {
    snapshot: store.getSnapshot(snapshot.id),
    identityBytesDigest: float32BytesDigest(identity),
    relationBytesDigest: float32BytesDigest(relation),
  }
}

function goldenTick(index) {
  const before = new Float32Array(LEGACY_LAYOUT.cellCount)
  const after = new Float32Array(LEGACY_LAYOUT.cellCount)
  before[index - 1] = Math.fround(index / 16)
  before[LEGACY_LAYOUT.identityLength - 1] = Math.fround(0.25 * index)
  before[LEGACY_LAYOUT.relationStart] = Math.fround(-0.125 * index)
  after[index] = Math.fround(index / 8)
  after[LEGACY_LAYOUT.identityLength - 1] = Math.fround(0.125 * index)
  after[LEGACY_LAYOUT.relationStart] = Math.fround(-0.25 * index)
  after[LEGACY_LAYOUT.cellCount - 1] = Math.fround(0.0625 * index)
  return {
    tick: index,
    branchCount: 1,
    activeBranchCount: 1,
    dominantBranchId: 'legacy-branch',
    dominantNeuronId: MEMBER_KEY,
    dominantMass: 1,
    replayFingerprint: `profileless-legacy-replay-${index}`,
    signal: {
      before,
      after,
      beforeDigest: `profileless-before-${index}`,
      afterDigest: `profileless-after-${index}`,
    },
    branches: [{
      branchId: 'legacy-branch',
      parentBranchId: index === 1 ? 'legacy-seed' : 'legacy-branch',
      mass: 1,
      beforeSignal: before,
      afterSignal: after,
      beforeSignalDigest: `profileless-branch-before-${index}`,
      afterSignalDigest: `profileless-branch-after-${index}`,
      beforeSignalNorm: 0.5,
      afterSignalNorm: 0.5,
      signalChangeMagnitude: 0.25,
      selectedFamilyId: 'profileless-legacy-family',
      selectedNeuronId: MEMBER_KEY,
      familyCandidates: [{ familyId: 'profileless-legacy-family', probability: 1 }],
      neuronCandidates: [{
        neuronId: MEMBER_KEY,
        familyId: 'profileless-legacy-family',
        rank: 1,
        habitProbability: 1,
        compatibility: 0.875,
        selectionProbability: 1,
      }],
      unmatchedProbability: 0,
      edgePressure: 0,
      birthPressure: 0,
      smoothedBirthPressure: 0,
    }],
  }
}

function createGoldenEngine(observations) {
  return {
    async runEpisode(request, context) {
      assert.equal(Object.hasOwn(request.base, 'representationProfile'), false)
      assert.equal(Object.hasOwn(request, 'adaptiveRegion'), false)
      assert.equal(Object.hasOwn(context, 'representationProfile'), false)
      assert.equal(Object.hasOwn(context, 'adaptiveRegionAtlas'), false)
      assert.equal(context.neurons.length, 1)

      const [neuron] = context.neurons
      assert.equal(neuron.memberKey, MEMBER_KEY)
      assert.ok(neuron.identity instanceof Float32Array)
      assert.ok(neuron.relation instanceof Float32Array)
      assert.equal(neuron.identity.length, LEGACY_LAYOUT.identityLength)
      assert.equal(neuron.relation.length, LEGACY_LAYOUT.relationLength)

      observations.push({
        replay: Boolean(request.replay?.verifyOnly),
        expectedTickCount: request.replay?.expectedTickCount ?? null,
        identityLength: neuron.identity.length,
        relationLength: neuron.relation.length,
        identityBytesDigest: float32BytesDigest(neuron.identity),
        relationBytesDigest: float32BytesDigest(neuron.relation),
      })

      const ticks = [goldenTick(1), goldenTick(2)]
      return {
        episode: {
          format: 'bombom-thought-episode',
          version: 1,
          episodeId: request.episodeId,
          tick: 2,
          maximumTicks: request.maxTicks,
          status: 'stopped',
          branches: [],
          replay: [],
          fingerprint: 'profileless-legacy-engine-golden',
          stopReason: 'settled',
        },
        ticks,
        graph: { nodes: [], edges: [] },
      }
    },
  }
}

async function waitCompleted(store, episodeId) {
  return waitFor(() => {
    const episode = store.getThoughtEpisode(episodeId)
    return ['completed', 'failed', 'cancelled'].includes(episode?.status) ? episode : null
  })
}

async function rewriteEpisodeAsPrePlasticAndPreResidency(dataDirectory, episodeId) {
  const filename = path.join(dataDirectory, STATE_FILENAME)
  const state = JSON.parse(await readFile(filename, 'utf8'))
  const episode = state.records?.['thought-episodes']?.[episodeId]
  assert.ok(episode, 'persisted legacy fixture episode must exist')

  for (const field of PRE_PLASTIC_EPISODE_FIELDS) delete episode[field]
  for (const field of PRE_PLASTIC_BASE_FIELDS) delete episode.base[field]
  delete episode.base.representationProfile
  delete episode.base.residencyManifestFingerprint
  delete episode.baseDiagnostics?.plasticRegions
  delete episode.baseDiagnostics?.residency
  delete episode.baseDiagnostics?.representationProfile

  const replacement = `${filename}.legacy-fixture`
  await writeFile(replacement, JSON.stringify(state), { encoding: 'utf8', mode: 0o600 })
  await rename(replacement, filename)
}

test('profileless 64x64 I=1024/R=3072 episode survives restart and replays with its pre-overlay contract', async (t) => {
  const dataDirectory = await temporaryDirectory(t, 'thought-profileless-legacy-golden-')
  const observations = []
  const engine = createGoldenEngine(observations)

  const first = await createTrainingService({ dataDirectory, forceJson: true, thoughtEngine: engine })
  const seeded = seedProfilelessLegacyBase(first.store)
  const resolved = first.thought.resolveBase(seeded.snapshot.id)

  assert.deepEqual(resolved.layout, LEGACY_LAYOUT)
  assert.equal(resolved.representationProfile, null)
  assert.equal(resolved.reference.representationProfile, undefined)
  assert.equal(resolved.diagnostics.representationProfile.explicit, false)
  assert.equal(resolved.diagnostics.representationProfile.value, null)

  const started = await first.thought.start({
    text: 'Profilsiz eski düşünce kaydı aynı biçimde yürüsün.',
    snapshotId: seeded.snapshot.id,
    seed: 1_648,
    maxTicks: 2,
    strategy: 'greedy',
    learningMode: 'off',
    teacherMode: 'off',
  })
  const completed = await waitCompleted(first.store, started.episode.id)
  assert.equal(completed.status, 'completed', JSON.stringify(completed.error))
  assert.equal(completed.tickCount, 2)
  assert.equal(completed.base.representationProfile, undefined)
  assert.equal(completed.base.residencyManifestFingerprint, undefined)
  assert.equal(seeded.identityBytesDigest, GOLDEN_IDENTITY_BYTES_DIGEST)
  assert.equal(seeded.relationBytesDigest, GOLDEN_RELATION_BYTES_DIGEST)
  assert.equal(observations.length, 1)
  assert.equal(observations[0].replay, false)
  assert.equal(observations[0].identityBytesDigest, seeded.identityBytesDigest)
  assert.equal(observations[0].relationBytesDigest, seeded.relationBytesDigest)

  const beforeRestartTicks = first.thought.ticks(completed.id, { after: -1, limit: 8, candidateLimit: 8 }).ticks
  assert.equal(beforeRestartTicks.length, 2)
  assert.deepEqual(beforeRestartTicks.map((tick) => tick.tickDigest), GOLDEN_TICK_DIGESTS)
  assert.equal(completed.lastTickDigest, GOLDEN_TICK_DIGESTS.at(-1))

  await first.close()
  await rewriteEpisodeAsPrePlasticAndPreResidency(dataDirectory, completed.id)

  const second = await createTrainingService({ dataDirectory, forceJson: true, thoughtEngine: engine })
  t.after(() => second.close())

  const restored = second.store.getThoughtEpisode(completed.id)
  assert.equal(restored.status, 'completed')
  assert.equal(restored.tickCount, 2)
  assert.equal(restored.lastTickDigest, GOLDEN_TICK_DIGESTS.at(-1))
  assert.equal(Object.hasOwn(restored.base, 'representationProfile'), false)
  assert.equal(Object.hasOwn(restored.base, 'residencyManifestFingerprint'), false)
  assert.equal(Object.hasOwn(restored, 'regionMode'), false)
  assert.equal(Object.hasOwn(restored, 'regionAtlasFingerprint'), false)
  assert.equal(Object.hasOwn(restored.baseDiagnostics ?? {}, 'plasticRegions'), false)
  assert.equal(Object.hasOwn(restored.baseDiagnostics ?? {}, 'residency'), false)

  const restartedBase = second.thought.resolveBase(restored.base.snapshotId)
  assert.deepEqual(restartedBase.layout, LEGACY_LAYOUT)
  assert.equal(restartedBase.representationProfile, null)
  assert.equal(restartedBase.reference.representationProfile, undefined)
  assert.equal(restartedBase.neurons[0].identity.length, 1_024)
  assert.equal(restartedBase.neurons[0].relation.length, 3_072)
  assert.equal(float32BytesDigest(restartedBase.neurons[0].identity), seeded.identityBytesDigest)
  assert.equal(float32BytesDigest(restartedBase.neurons[0].relation), seeded.relationBytesDigest)

  const afterRestartTicks = second.thought.ticks(restored.id, { after: -1, limit: 8, candidateLimit: 8 }).ticks
  assert.deepEqual(afterRestartTicks, beforeRestartTicks)
  assert.deepEqual(afterRestartTicks.map((tick) => tick.tickDigest), GOLDEN_TICK_DIGESTS)

  const replay = await second.thought.replay(restored.id)
  assert.deepEqual(replay, {
    episodeId: restored.id,
    matched: true,
    comparedTicks: 2,
    expectedTickCount: 2,
    actualTickCount: 2,
    firstMismatch: null,
    baseModelFingerprint: restored.base.baseModelFingerprint,
    residencyManifestFingerprint: null,
    engineContract: 'thoughtEngine.runEpisode(request, context)',
  })
  assert.equal(observations.length, 2)
  assert.equal(observations[1].replay, true)
  assert.equal(observations[1].expectedTickCount, 2)
  assert.deepEqual(observations[1], {
    ...observations[0],
    replay: true,
    expectedTickCount: 2,
  })

  const afterReplay = second.store.getThoughtEpisode(restored.id)
  assert.deepEqual(afterReplay, restored, 'verification replay must not mutate the historical episode')
})
