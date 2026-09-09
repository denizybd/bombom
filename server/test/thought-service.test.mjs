import test from 'node:test'
import assert from 'node:assert/strict'
import { createTrainingService } from '../http-server.mjs'
import { createThoughtService, deriveFamilyGraphView, familyGraphNodes, retainMaterializedEdges } from '../thought-service.mjs'
import { encodeArtifactValue, restoreArtifact } from '../../runtime/artifacts.mjs'
import { createPlasticRegionAtlas, proposeRegionBirth } from '../../src/adaptive-regions/index.ts'
import {
  currentPlasticRegionBase,
  persistPlasticRegionArtifact,
  readPlasticRegionSnapshot,
} from '../../runtime/plastic-region-cache.mjs'
import {
  PLASTIC_REGION_RELEASE_JOB_TYPE,
  createPlasticRegionReleaseExecutor,
  preparePlasticRegionReleaseJob,
} from '../../runtime/plastic-region-release-runner.mjs'
import { jsonFetch, listenService, temporaryDirectory, waitFor } from './helpers.mjs'

const MEMBER_KEY = 'source-digest\u0000analysis-run\u0000neuron-1'
const LAYOUT = {
  side: 64,
  cellCount: 4_096,
  identityLength: 2_048,
  relationStart: 2_048,
  relationLength: 2_048,
}
const ADAPTIVE_LAYOUT = {
  side: 64,
  cellCount: 4_096,
  identityLength: 512,
  relationStart: 512,
  relationLength: 3_584,
}

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

function canonicalThoughtValue(value) {
  if (value instanceof Float32Array) return ['Float32Array', ...Array.from(value, (entry) => Math.fround(entry))]
  if (Array.isArray(value)) return value.map(canonicalThoughtValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonicalThoughtValue(entry)]))
  }
  if (typeof value === 'number') return Number.isFinite(value) ? Math.fround(value) : String(value)
  return value
}

function runtimeThoughtDigest(label, ...parts) {
  const primes = [0x01000193, 0x27d4eb2d, 0x165667b1, 0x9e3779b1]
  const states = [0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35]
  const serialized = JSON.stringify(canonicalThoughtValue(['thought-runtime-v1', label, ...parts]))
  for (let index = 0; index < serialized.length; index += 1) {
    for (let stateIndex = 0; stateIndex < states.length; stateIndex += 1) {
      states[stateIndex] = Math.imul(
        (states[stateIndex] ^ serialized.charCodeAt(index) ^ Math.imul(index + 1, stateIndex + 19)) >>> 0,
        primes[stateIndex],
      ) >>> 0
      states[stateIndex] ^= states[stateIndex] >>> 13
    }
  }
  return states.map((state) => (state >>> 0).toString(16).padStart(8, '0')).join('')
}

function runtimeHeadFingerprint(head) {
  return runtimeThoughtDigest(
    'head', 1, head.revision, head.baseModelFingerprint, head.totalTicks,
    head.neuronEdges,
    head.contextPatches.map((patch) => ({
      ...patch,
      delta: runtimeThoughtDigest('relation', patch.delta),
    })),
    head.nursery.map((candidate) => ({
      ...candidate,
      identity: runtimeThoughtDigest('identity', candidate.identity),
      relationPrototype: runtimeThoughtDigest('relation', candidate.relationPrototype),
    })),
    head.births,
  )
}

function runtimeArtifact(kind, value, id = `${kind}-artifact`) {
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

function seedActiveBase(store, suffix = '') {
  const identity = new Float32Array(LAYOUT.identityLength)
  const relation = new Float32Array(LAYOUT.relationLength)
  identity[0] = 0.75
  relation[0] = -0.25
  const identityDigest = textLabDigest(identity)
  const analysis = {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId: 'analysis-run',
    sourceDigest: 'source-digest',
    representationSpaceId: 'RS-thought-test',
    layout: LAYOUT,
    nodes: [{
      id: 'neuron-1', text: 'ali ekmeği kesti', identity, relation, identityDigest,
      contextSupport: 2, contextRevision: 1,
    }],
    habitProbabilities: new Float32Array([0]),
  }
  const familyRun = {
    format: 'noron-aileleri-run', version: 1, id: 'family-run', fingerprint: `family-run-fingerprint${suffix}`,
    representationSpaceId: 'RS-thought-test', behaviorSpaceId: 'BS-thought-test',
    profiles: [{
      memberKey: MEMBER_KEY, kind: 'raw', identityDigest, text: 'ali ekmeği kesti',
      representationSpaceId: 'RS-thought-test', behaviorSpaceId: 'BS-thought-test', vector: {}, diagnostics: {},
      regionAffinities: [], evidence: [], derivedFromMemberKeys: [],
    }],
    families: [{
      id: 'family-event', region: 'event', lifecycle: 'stable', supportCount: 2, sourceCount: 2,
      memberKeys: [MEMBER_KEY], coreMemberKeys: [MEMBER_KEY], cohesion: 1, separation: 1, stability: 1,
    }],
    memberships: [{ memberKey: MEMBER_KEY, familyId: 'family-event', rank: 1, score: 1, weight: 1, core: true }],
    neuronTransitions: [], familyTransitions: [], utilityReports: [],
  }
  const analysisArtifact = store.createArtifact(runtimeArtifact('text-lab-analysis', analysis, `analysis-artifact${suffix}`))
  const familyArtifact = store.createArtifact(runtimeArtifact('family-run', familyRun, `family-artifact${suffix}`))
  const snapshot = store.createSnapshot({
    id: `active-thought-snapshot${suffix}`,
    status: 'candidate',
    label: 'Doğrulanmış düşünce tabanı',
    artifactIds: [analysisArtifact.id, familyArtifact.id],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  return { snapshot: store.getSnapshot(snapshot.id), analysis, familyRun }
}

async function seedActiveAdaptiveBase(store, suffix = '') {
  const representationSpaceId = `RS-adaptive-thought-test${suffix}`
  const identity = new Float32Array(ADAPTIVE_LAYOUT.identityLength)
  const relation = new Float32Array(ADAPTIVE_LAYOUT.relationLength)
  identity[0] = 0.75
  relation[0] = -0.25
  const identityDigest = textLabDigest(identity)
  const memberKey = `adaptive-source${suffix}\u0000adaptive-analysis${suffix}\u0000adaptive-neuron${suffix}`
  const analysis = {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId: `adaptive-analysis${suffix}`,
    sourceDigest: `adaptive-source${suffix}`,
    representationSpaceId,
    config: { representationProfile: 'adaptive-eighth-id-v1' },
    layout: ADAPTIVE_LAYOUT,
    nodes: [{
      id: `adaptive-neuron${suffix}`,
      text: 'ali ekmeği kesti',
      identity,
      relation,
      identityDigest,
      contextSupport: 2,
      contextRevision: 1,
    }],
    habitProbabilities: new Float32Array([0]),
  }
  const familyRun = {
    format: 'noron-aileleri-run', version: 1, id: `adaptive-family-run${suffix}`,
    fingerprint: `adaptive-family-run-fingerprint${suffix}`,
    representationSpaceId, behaviorSpaceId: `BS-adaptive-thought-test${suffix}`,
    profiles: [{
      memberKey, kind: 'raw', identityDigest, text: 'ali ekmeği kesti',
      representationSpaceId, behaviorSpaceId: `BS-adaptive-thought-test${suffix}`, vector: {}, diagnostics: {},
      regionAffinities: [], evidence: [], derivedFromMemberKeys: [],
    }],
    families: [{
      id: 'adaptive-family-event', region: 'event', lifecycle: 'stable', supportCount: 2, sourceCount: 2,
      memberKeys: [memberKey], coreMemberKeys: [memberKey], cohesion: 1, separation: 1, stability: 1,
    }],
    memberships: [{ memberKey, familyId: 'adaptive-family-event', rank: 1, score: 1, weight: 1, core: true }],
    neuronTransitions: [], familyTransitions: [], utilityReports: [],
  }
  const baseAtlas = createPlasticRegionAtlas({ representationSpaceId, behaviorDimension: 8 })
  const proposal = proposeRegionBirth(baseAtlas, {
    regionId: `adaptive-region${suffix}`,
    allocations: [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })),
    prototype: { dimension: 8, values: [1, 0, 0, 0, 0, 0, 0, 0] },
    evidence: {
      evidenceId: `adaptive-train-evidence${suffix}`,
      validationAuthority: 'train-only',
      observationCount: 40,
      sourceGroupIds: ['adaptive-train-a', 'adaptive-train-b', 'adaptive-train-c'],
      confidenceMass: 3,
      cohesion: 0.9,
      holdoutGain: 0,
      shamGain: 0,
      codes: ['counterfactual-utility', 'unexplained-residual'],
    },
  })
  const analysisArtifact = store.createArtifact(runtimeArtifact('text-lab-analysis', analysis, `adaptive-analysis-artifact${suffix}`))
  const familyArtifact = store.createArtifact(runtimeArtifact('family-run', familyRun, `adaptive-family-artifact${suffix}`))
  const baseSnapshot = store.createSnapshot({
    id: `adaptive-release-base${suffix}`,
    artifactIds: [],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(baseSnapshot.id)
  const producer = store.createJob({ id: `adaptive-source-producer${suffix}`, type: 'full-cycle' }).job
  const proposalArtifact = persistPlasticRegionArtifact({ store, jobId: producer.id }, {
    kind: 'plastic-region-atlas-proposal',
    value: proposal,
    expectedBase: currentPlasticRegionBase(store),
    contentHash: `adaptive-proposal-content${suffix}`,
  }).artifact
  const sourceSnapshot = store.createSnapshot({
    id: `adaptive-source-snapshot${suffix}`,
    parentId: baseSnapshot.id,
    status: 'candidate',
    producerJobId: producer.id,
    artifactIds: [analysisArtifact.id, familyArtifact.id, proposalArtifact.id],
    sourceIds: [],
    sourceRevisions: {},
    inputManifestHash: `adaptive-source-input${suffix}`,
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.updateJob(producer.id, {
    state: 'succeeded', stage: 'completed', result: { snapshotId: sourceSnapshot.id }, finishedAt: new Date().toISOString(),
  })
  const prepared = preparePlasticRegionReleaseJob(store, proposalArtifact.id)
  const queued = store.createJob({
    type: PLASTIC_REGION_RELEASE_JOB_TYPE,
    payload: prepared.payload,
    requestHash: prepared.requestHash,
    idempotencyKey: prepared.defaultIdempotencyKey,
    maxAttempts: 1,
  }).job
  const owner = `adaptive-release-owner${suffix}`
  const claimed = store.claimNextJob(owner, 60_000, 120_000)
  const fakeBase = {
    reference: {
      snapshotId: sourceSnapshot.id,
      snapshotStatus: sourceSnapshot.status,
      activeAtStart: false,
      artifactManifestHash: prepared.payload.sourceArtifactManifestHash,
      baseModelFingerprint: `adaptive-model${suffix}`,
      familyRunFingerprint: familyRun.fingerprint,
      languageStateFingerprint: null,
      thoughtHeadFingerprint: `adaptive-head${suffix}`,
      thoughtHeadRevision: 0,
      conceptMacroHeadFingerprint: null,
    },
    layout: ADAPTIVE_LAYOUT,
    representationProfile: 'adaptive-eighth-id-v1',
    neurons: [{ memberKey }],
    familyRun,
    languageState: null,
    thoughtHead: { fingerprint: `adaptive-head${suffix}`, revision: 0 },
    conceptMacroHead: null,
  }
  const executor = createPlasticRegionReleaseExecutor({
    resolveBase: () => fakeBase,
    evaluateCase: ({ variant }) => variant === 'candidate' ? 0.7 : 0.5,
  })
  const result = await executor(queued, {
    jobId: queued.id,
    store,
    config: store.getConfig(),
    jobClaim: { owner, attempt: claimed.attempt },
    checkResources: () => ({}),
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  })
  store.activateSnapshot(result.snapshotId)
  const releasedAtlas = readPlasticRegionSnapshot(store, store.getSnapshot(result.snapshotId)).head
  return {
    snapshot: store.getSnapshot(result.snapshotId),
    analysis,
    familyRun,
    atlas: releasedAtlas,
    memberKey,
  }
}

function seedLegacy32Base(store) {
  const legacyLayout = {
    side: 32,
    cellCount: 1_024,
    identityLength: 256,
    relationStart: 256,
    relationLength: 768,
  }
  const identity = new Float32Array(legacyLayout.identityLength)
  const relation = new Float32Array(legacyLayout.relationLength)
  identity[0] = 0.75
  relation[0] = -0.25
  const identityDigest = textLabDigest(identity)
  const analysis = {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId: 'legacy-analysis-run',
    sourceDigest: 'legacy-source-digest',
    representationSpaceId: 'RS-legacy-32-test',
    layout: legacyLayout,
    nodes: [{
      id: 'legacy-neuron-1', text: 'eski otuz iki', identity, relation, identityDigest,
      contextSupport: 2, contextRevision: 1,
    }],
    habitProbabilities: new Float32Array([0]),
  }
  const memberKey = 'legacy-source-digest\u0000legacy-analysis-run\u0000legacy-neuron-1'
  const familyRun = {
    format: 'noron-aileleri-run', version: 1, id: 'legacy-family-run', fingerprint: 'legacy-family-run-fingerprint',
    representationSpaceId: 'RS-legacy-32-test', behaviorSpaceId: 'BS-legacy-32-test',
    profiles: [{
      memberKey, kind: 'raw', identityDigest, text: 'eski otuz iki',
      representationSpaceId: 'RS-legacy-32-test', behaviorSpaceId: 'BS-legacy-32-test', vector: {}, diagnostics: {},
      regionAffinities: [], evidence: [], derivedFromMemberKeys: [],
    }],
    families: [{
      id: 'legacy-family', region: 'event', lifecycle: 'stable', supportCount: 2, sourceCount: 2,
      memberKeys: [memberKey], coreMemberKeys: [memberKey], cohesion: 1, separation: 1, stability: 1,
    }],
    memberships: [{ memberKey, familyId: 'legacy-family', rank: 1, score: 1, weight: 1, core: true }],
    neuronTransitions: [], familyTransitions: [], utilityReports: [],
  }
  const analysisArtifact = store.createArtifact(runtimeArtifact('text-lab-analysis', analysis, 'legacy-analysis-artifact'))
  const familyArtifact = store.createArtifact(runtimeArtifact('family-run', familyRun, 'legacy-family-artifact'))
  const snapshot = store.createSnapshot({
    id: 'legacy-active-snapshot',
    status: 'candidate',
    label: 'Eski 32x32 taban',
    artifactIds: [analysisArtifact.id, familyArtifact.id],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(snapshot.id)
  return store.getSnapshot(snapshot.id)
}

const EXPECTED_GRAPH = {
  format: 'event-graph-v1',
  actor: 'Ali',
  action: 'Kesti',
  patient: 'Ekmek',
  instrument: null,
  operators: [],
  negated: false,
  goals: [],
}

function traceTick(index = 1) {
  const before = new Float32Array(4_096)
  const after = new Float32Array(4_096)
  before[0] = Math.fround(index / 100)
  after[1] = Math.fround(index / 100)
  return {
    tick: index,
    branchCount: 1,
    activeBranchCount: 1,
    dominantBranchId: 'branch-1',
    dominantNeuronId: MEMBER_KEY,
    dominantMass: 1,
    maximumBirthPressure: 0.1,
    replayFingerprint: `replay-${index}`,
    signal: { before, after, beforeDigest: `before-${index}`, afterDigest: `after-${index}` },
    branches: [{
      branchId: 'branch-1', parentBranchId: 'branch-0', mass: 1,
      beforeSignal: Array.from(before), afterSignal: Array.from(after),
      beforeSignalDigest: `branch-before-${index}`, afterSignalDigest: `branch-after-${index}`,
      beforeSignalNorm: 0.1, afterSignalNorm: 0.2, signalChangeMagnitude: 0.1,
      selectedFamilyId: 'family-event', selectedNeuronId: MEMBER_KEY,
      familyCandidates: [{ familyId: 'family-event', probability: 1, evidence: 0.8 }],
      neuronCandidates: [{
        neuronId: MEMBER_KEY, familyId: 'family-event', rank: 1, baseH0: 0.2,
        persistentExcitation: 0.1, fastExcitation: 0.05, persistentInhibition: 0.01,
        fastInhibition: 0.02, routeEvidence: 0.3, compatibility: 0.8, repetitionPenalty: 0.04,
        q: 0.76, habitProbability: 0.7, acceptanceProbability: 0.9, reachProbability: 1,
        selectionProbability: 1,
      }],
      unmatchedProbability: 0, edgePressure: 0.1, birthPressure: 0.2, smoothedBirthPressure: 0.15,
      coalitionFamilyWeights: { 'family-event': 1 },
      roleHypotheses: [{ role: 'actor', memberKey: MEMBER_KEY, score: 0.9, temporary: true }],
      birthEvents: [{ branchId: 'branch-1', kind: 'pressure-threshold', pressure: 0.2, consecutiveTicks: 3 }],
    }],
    neuronCandidates: [{
      neuronId: MEMBER_KEY, familyId: 'family-event', rank: 1, baseH0: 0.2,
      persistentExcitation: 0.1, fastExcitation: 0.05, persistentInhibition: 0.01,
      fastInhibition: 0.02, routeEvidence: 0.3, compatibility: 0.8, repetitionPenalty: 0.04,
      q: 0.76, habitProbability: 0.7, acceptanceProbability: 0.9, reachProbability: 1,
      selectionProbability: 1,
    }],
  }
}

function adaptiveBindingFor(request, context) {
  const atlas = context.adaptiveRegionAtlas
  const binding = {
    format: 'thought-adaptive-region-binding',
    version: 1,
    mode: request.adaptiveRegion.mode,
    atlasFingerprint: atlas.fingerprint,
    atlasRevision: atlas.revision,
    representationSpaceId: atlas.representationSpaceId,
    geometryFingerprint: atlas.geometryFingerprint,
    sourceGroupId: request.sourceGroupId,
    compatibilityBlend: request.adaptiveRegion.compatibilityBlend ?? 0,
    minimumTrace: request.adaptiveRegion.minimumTrace ?? 0.02,
    ...(request.adaptiveRegion.mode === 'active' ? {
      maximumActiveGates: request.adaptiveRegion.maximumActiveGates ?? 4,
    } : {}),
  }
  return { ...binding, fingerprint: runtimeThoughtDigest('adaptive-region-binding', binding) }
}

function adaptiveTraceTick(request, binding, index = 1) {
  const base = traceTick(index)
  const gates = { gates: [], unmatchedProbability: 1, omittedRegionProbability: 0 }
  const compatibility = {
    legacyCompatibility: 0.8,
    regionCompatibility: 0.8,
    blendedCompatibility: 0.8,
    effectiveCompatibility: 0.8,
    pageCompatibilities: Array(56).fill(0),
    pageWeights: Array(56).fill(1),
    gates,
  }
  const branchTelemetry = {
    format: 'thought-adaptive-region-branch-telemetry',
    version: 1,
    mode: binding.mode,
    atlasFingerprint: binding.atlasFingerprint,
    bindingFingerprint: binding.fingerprint,
    activeApplied: binding.mode === 'active',
    ...compatibility,
    pageWriteMask: Array(56).fill(1),
    pageObservations: Array.from({ length: 56 }, (_, pageIndex) => ({
      observationId: `${binding.fingerprint}:${index}:${pageIndex}`,
      episodeId: request.episodeId,
      sourceGroupId: binding.sourceGroupId,
      pageIndex,
      confidence: 1,
      behavior: Array(8).fill(0),
      outcomeCredit: 0,
      ablationUtility: 0,
    })),
    candidateCompatibilities: [],
  }
  return {
    ...base,
    branches: [{ ...base.branches[0], adaptiveRegion: branchTelemetry }],
    adaptiveRegion: {
      format: 'thought-adaptive-region-tick-telemetry',
      version: 1,
      mode: binding.mode,
      atlasFingerprint: binding.atlasFingerprint,
      bindingFingerprint: binding.fingerprint,
      observedBranchCount: 1,
      activeApplied: binding.mode === 'active',
    },
  }
}

function resultFor(request, context, ticks = [traceTick(1)], trust = 1) {
  for (const tick of ticks) context.onTick?.(tick)
  return {
    episode: {
      format: 'bombom-thought-episode', version: 1, episodeId: request.episodeId,
      tick: ticks.at(-1)?.tick ?? 0, maximumTicks: request.maxTicks, status: 'stopped',
      branches: [], replay: [], fingerprint: `episode-${ticks.length}`, stopReason: 'settled',
    },
    ticks,
    headDelta: {
      neuronEdges: [{
        fromMemberKey: MEMBER_KEY, toMemberKey: MEMBER_KEY,
        excitation: 0.2, inhibition: 0.01, trustMass: trust, updateCount: 1,
      }],
      contextPatches: [{
        memberKey: MEMBER_KEY,
        relationDelta: new Float32Array(LAYOUT.relationLength).fill(0.001),
        trustMass: trust,
        observationCount: 1,
      }],
      stats: { totalTicks: ticks.length },
    },
    graph: {
      nodes: [{ id: MEMBER_KEY, kind: 'neuron', text: 'ali ekmeği kesti' }, { id: 'family-event', kind: 'family', label: 'event' }],
      eventGraph: {
        format: 'event-graph-v1', actor: 'ali', action: 'kesti', patient: 'ekmek', instrument: null,
        operators: [], negated: false, goals: [],
      },
      evaluation: { graphF1: 1, constraintAccuracy: 1, resultScore: 1, truePositive: 3, falsePositive: 0, falseNegative: 0 },
      familyGraph: { fingerprint: 'family-graph', transitions: [{ fromFamilyId: 'family-event', toFamilyId: 'family-event', probability: 1 }] },
      branchScores: [{ branchId: 'branch-1', transitionCount: 1, meanCompatibility: 0.8, meanGraphSupport: 1, meanFamilySupport: 1, uniqueTransitionRatio: 1, loopRatio: 0, score: 1, eligible: true, explanation: 'uyumlu' }],
    },
  }
}

function residencyTestEngine() {
  return {
    config: {},
    prepareResidencyBase({ neurons, familyRun, thoughtHead }) {
      const membershipById = new Map(familyRun.memberships.map((membership) => [membership.memberKey, [{
        familyId: membership.familyId,
        weight: membership.weight,
      }]]))
      return {
        model: {
          format: 'bombom-thought-model', version: 1, revision: 0, layout: LAYOUT,
          fingerprint: 'full-catalog-runtime-fingerprint', habitRows: [],
          families: familyRun.families.map((family) => ({ id: family.id, region: family.region })),
          neurons: neurons.map((neuron) => ({
            id: neuron.memberKey,
            identity: neuron.identity,
            relation: neuron.relation,
            identityDigest: neuron.identityDigest,
            familyMemberships: membershipById.get(neuron.memberKey) ?? [],
            provenance: { kind: 'textlab-surface', surface: neuron.text },
          })),
        },
        head: thoughtHead,
      }
    },
    async encodeResidencyPrompt() { return new Float32Array(4_096) },
    async runEpisode(request, context) { return resultFor(request, context) },
  }
}

async function waitCompleted(store, id) {
  const episode = await waitFor(() => {
    const value = store.getThoughtEpisode(id)
    return ['completed', 'failed', 'cancelled'].includes(value?.status) ? value : null
  })
  assert.equal(episode.status, 'completed', JSON.stringify(episode.error))
  return episode
}

async function assertRemoteCrossCheckControlStopsNextPass(t, action) {
  let releaseFirstPass
  let enterFirstPass
  const firstPassEntered = new Promise((resolve) => { enterFirstPass = resolve })
  const firstPassGate = new Promise((resolve) => { releaseFirstPass = resolve })
  let ollamaCalls = 0
  let engineCalls = 0
  const graph = { ...EXPECTED_GRAPH, actor: 'ali', action: 'kesti', patient: 'ekmek' }
  const ollama = {
    async generateStructured() {
      ollamaCalls += 1
      if (ollamaCalls === 1) {
        enterFirstPass()
        await firstPassGate
      }
      return { value: { graph, confidence: 0.95 }, model: 'qwen:14b', promptTokens: 1, completionTokens: 1 }
    },
    async check() { return { available: true, models: [] } },
    async listModels() { return [] },
  }
  const engine = {
    async runEpisode(request, context) {
      engineCalls += 1
      return resultFor(request, context)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, `thought-remote-${action}-`),
    forceJson: true,
    thoughtEngine: engine,
    ollama,
  })
  t.after(() => service.close())
  service.store.updateConfig({ ollama: { model: 'qwen:14b' } })
  // Construct the remote controller before an episode exists: both services
  // share durable state, but only the owner holds the active AbortController.
  const remote = createThoughtService({ store: service.store, ollama, thoughtEngine: engine })
  seedActiveBase(service.store, `-remote-${action}`)

  const started = await service.thought.start({
    text: `Uzaktan ${action} deneyi.`,
    learningMode: 'proposal',
    teacherMode: 'ollama-cross-check',
  })
  const episodeId = started.episode.id
  await firstPassEntered
  const requested = remote[action](episodeId)
  assert.equal(requested.status, `${action}_requested`)
  releaseFirstPass()

  const expectedStatus = action === 'pause' ? 'paused' : 'cancelled'
  await waitFor(() => service.store.getThoughtEpisode(episodeId)?.status === expectedStatus)
  assert.equal(ollamaCalls, 1, 'fresh durable control must stop before the critic reservation')
  assert.equal(engineCalls, 0)
  assert.equal(service.store.getUsage().teacherRequests, 1)
  assert.equal(service.store.getUsage().teacherTokens, 2)
}

test('thought status rejects a legacy 32x32 family artifact as a usable research base', async (t) => {
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t),
    forceJson: true,
    thoughtEngine: { async runEpisode() { throw new Error('Bu test motoru çalıştırmamalı.') } },
  })
  seedLegacy32Base(service.store)
  const base = await listenService(t, service)

  const status = await jsonFetch(base, '/api/thought/status')

  assert.equal(status.response.status, 200)
  assert.equal(status.body.available, true)
  assert.equal(status.body.researchBaseAvailable, false)
  assert.equal(status.body.researchBase, null)
  assert.equal(status.body.researchBaseError.code, 'REPRESENTATION_SPACE_MISMATCH')
  assert.match(status.body.message, /64x64/u)
})

test('thought runtime resolves concept macro heads only from the selected snapshot', async (t) => {
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-snapshot-macro-head-'),
    forceJson: true,
    thoughtEngine: { async runEpisode() { throw new Error('Bu test motoru çalıştırmamalı.') } },
  })
  t.after(() => service.close())
  const seeded = seedActiveBase(service.store, '-snapshot-macro')
  const headValue = (fingerprint) => ({
    format: 'concept-macro-head', version: 1, id: `head-${fingerprint}`,
    fingerprint, representationSpaceId: seeded.familyRun.representationSpaceId,
    familyRunFingerprint: seeded.familyRun.fingerprint,
  })
  const globalOnly = service.store.createArtifact(runtimeArtifact(
    'concept-macro-head', headValue('GLOBAL-RESEARCH-HEAD'), 'global-research-macro-head',
  ))

  const rawBase = service.thought.resolveBase(seeded.snapshot.id)
  assert.equal(rawBase.conceptMacroHead, null)
  assert.equal(rawBase.reference.conceptMacroHeadArtifactId, null)

  const attached = service.store.createArtifact(runtimeArtifact(
    'concept-macro-head', headValue('SNAPSHOT-ATTACHED-HEAD'), 'snapshot-attached-macro-head',
  ))
  const attachedSnapshot = service.store.createSnapshot({
    parentId: seeded.snapshot.id,
    status: 'candidate',
    label: 'Snapshot-bound concept macro head',
    artifactIds: [...seeded.snapshot.artifactIds, attached.id],
    sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  const selected = service.thought.resolveBase(attachedSnapshot.id)
  assert.equal(selected.conceptMacroHead.fingerprint, 'SNAPSHOT-ATTACHED-HEAD')
  assert.equal(selected.reference.conceptMacroHeadArtifactId, attached.id)
  assert.notEqual(selected.reference.conceptMacroHeadArtifactId, globalOnly.id)
})

test('thought service passes the snapshot macro head to the engine and persists macro execution audit', async (t) => {
  let observed = null
  const engine = {
    async runEpisode(request, context) {
      observed = { request, context }
      const tick = traceTick(1)
      tick.branches[0].macroRuntimeFingerprint = 'MACRO-RUNTIME-FP'
      tick.branches[0].macroCandidates = [{
        macroId: 'macro-1', entryCompatibility: 0.95, activationWeight: 0.9,
        selectionProbability: 0.6, duration: 2, compressedTransitionCount: 1, fidelity: 0.99,
        expandedConstituentIds: [MEMBER_KEY, MEMBER_KEY],
      }]
      tick.branches[0].macroExecution = {
        macroId: 'macro-1', macroHeadFingerprint: 'SNAPSHOT-MACRO-FP',
        runtimeFingerprint: 'MACRO-RUNTIME-FP', compiled: true,
        expandedConstituentIds: [MEMBER_KEY, MEMBER_KEY], duration: 2, compressedTransitionCount: 1,
        fidelity: 0.99, entryCompatibility: 0.95, selectionProbability: 0.6,
        creditMode: 'constituent-edges', fallbackReason: null,
      }
      tick.branches[0].macroFallbacks = [{
        macroId: 'macro-2', reason: 'entry-signal-below-threshold', detail: 'compatibility=0.1',
      }]
      return resultFor(request, context, [tick])
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-macro-audit-'),
    forceJson: true,
    thoughtEngine: engine,
  })
  t.after(() => service.close())
  const seeded = seedActiveBase(service.store, '-macro-audit')
  const macroArtifact = service.store.createArtifact(runtimeArtifact('concept-macro-head', {
    format: 'concept-macro-head', version: 1, id: 'snapshot-macro-head',
    fingerprint: 'SNAPSHOT-MACRO-FP',
    representationSpaceId: seeded.familyRun.representationSpaceId,
    familyRunFingerprint: seeded.familyRun.fingerprint,
  }, 'snapshot-macro-audit-head'))
  const snapshot = service.store.createSnapshot({
    parentId: seeded.snapshot.id, status: 'candidate', label: 'Makro audit snapshotı',
    artifactIds: [...seeded.snapshot.artifactIds, macroArtifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(snapshot.id)
  const started = await service.thought.start({ text: 'Makro audit deneyi.', learningMode: 'off' })
  await waitCompleted(service.store, started.episode.id)

  assert.equal(observed.context.conceptMacroHead.fingerprint, 'SNAPSHOT-MACRO-FP')
  assert.equal(observed.request.base.conceptMacroHeadFingerprint, 'SNAPSHOT-MACRO-FP')
  const page = service.thought.ticks(started.episode.id, { after: -1, limit: 8, candidateLimit: 8 })
  const branch = page.ticks[0].branches[0]
  assert.equal(branch.macroExecution.macroId, 'macro-1')
  assert.equal(branch.macroExecution.creditMode, 'constituent-edges')
  assert.equal(branch.macroExecution.compressedTransitionCount, 1)
  assert.deepEqual(branch.macroExecution.expandedConstituentIds, [MEMBER_KEY, MEMBER_KEY])
  assert.equal(branch.macroFallbacks[0].reason, 'entry-signal-below-threshold')
})

test('thought status migrates dense base vectors behind the bounded SSD residency gate', async (t) => {
  const engine = residencyTestEngine()
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-residency-status-'),
    forceJson: true,
    thoughtEngine: engine,
  })
  seedActiveBase(service.store, '-residency-status')
  const base = await listenService(t, service)

  const status = await jsonFetch(base, '/api/thought/status')

  assert.equal(status.response.status, 200)
  assert.equal(status.body.researchBase.residency.mode, 'ssd-gated-working-set-v1')
  const cached = [...service.thought.baseCache.values()][0].value
  assert.equal(cached.neurons.length, 0, 'status must not retain N dense I/R vectors')
  assert.equal(cached.neuronByKey.get(MEMBER_KEY).relation.length, LAYOUT.relationLength)
  assert.equal(cached.neuronByKey.get(MEMBER_KEY).relation instanceof Float32Array, false)
  const residency = [...service.thought.residencies.values()][0]
  assert.equal(residency.repository.getStats().tiers.coldNeurons, 1)
  assert.equal(residency.repository.getStats().resident.totalBytes, 0)

  const started = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST',
    body: JSON.stringify({ text: 'SSD manifest bağı deneyi.', learningMode: 'off' }),
  })
  assert.equal(started.response.status, 202)
  assert.match(started.body.episode.base.residencyManifestFingerprint, /^[a-f0-9]{16,}$/)
  await waitCompleted(service.store, started.body.episode.id)
  const replay = await service.thought.replay(started.body.episode.id)
  assert.equal(replay.matched, true)
  assert.equal(
    replay.residencyManifestFingerprint,
    started.body.episode.base.residencyManifestFingerprint,
  )
})

test('residency admission preserves the physical SSD reserve and releases dense migration input', async (t) => {
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-residency-resource-'),
    forceJson: true,
    thoughtEngine: residencyTestEngine(),
  })
  seedActiveBase(service.store, '-residency-resource')
  service.store.updateConfig({
    resourcePolicy: {
      minDiskFreeStartMb: 1_048_576,
      minDiskFreeWarnMb: 1_048_576,
    },
  })
  const base = await listenService(t, service)

  const status = await jsonFetch(base, '/api/thought/status')

  assert.equal(status.response.status, 200)
  assert.equal(status.body.researchBaseAvailable, false)
  assert.equal(status.body.researchBaseError.code, 'NEURON_RESIDENCY_RESOURCE_WAIT')
  assert.equal(service.thought.baseCache.size, 0, 'blocked residency must not pin dense I/R migration input')
  assert.equal(service.thought.residencies.size, 0)
})

test('metadata preflight rejects aggregate hydration before any vector artifact decode', async (t) => {
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-hydration-preflight-'),
    forceJson: true,
    thoughtEngine: residencyTestEngine(),
  })
  seedActiveBase(service.store, '-hydration-preflight')
  service.store.updateConfig({
    budgets: { ramMb: 512 },
    resourcePolicy: { ramSoftRatio: 0.25, ramSpillRatio: 0.5, ramHardRatio: 0.75 },
  })
  const originalMetadata = service.store.getArtifactMetadata.bind(service.store)
  const originalGetArtifact = service.store.getArtifact.bind(service.store)
  let hydratedArtifacts = 0
  service.store.getArtifactMetadata = (id) => {
    const metadata = originalMetadata(id)
    return metadata ? { ...metadata, payloadBytes: 256 * 1024 * 1024 } : metadata
  }
  service.store.getArtifact = (...args) => {
    hydratedArtifacts += 1
    return originalGetArtifact(...args)
  }

  const status = await service.thought.status()

  assert.equal(status.researchBaseAvailable, false)
  assert.equal(status.researchBaseError.code, 'NEURON_RESIDENCY_RESOURCE_WAIT')
  assert.equal(hydratedArtifacts, 0, 'preflight must run before getArtifact/Float32 hydration')
  assert.equal(service.thought.baseCache.size, 0)
})

test('thought residency/base caches use deterministic max-2 LRU eviction', async (t) => {
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-residency-lru-'),
    forceJson: true,
    thoughtEngine: residencyTestEngine(),
  })
  seedActiveBase(service.store, '-lru-1')
  await service.thought.status()
  const first = [...service.thought.residencies.values()][0]
  seedActiveBase(service.store, '-lru-2')
  await service.thought.status()
  seedActiveBase(service.store, '-lru-3')
  await service.thought.status()

  assert.equal(service.thought.residencies.size, 2)
  assert.equal(service.thought.baseCache.size, 2)
  await assert.rejects(
    first.repository.get(MEMBER_KEY, { distance: 1, routeProbability: 0, familyProbability: 0, observedUseProbability: 0 }),
    (error) => error?.code === 'NEURON_TIER_CLOSED',
  )
  assert.ok([...service.thought.residencyMeta.values()].some((meta) => meta.snapshotId === 'active-thought-snapshot-lru-3'))
})

test('active episode pins its residency while newer snapshots rotate the second cache slot', async (t) => {
  let release
  const gate = new Promise((resolve) => { release = resolve })
  const engine = residencyTestEngine()
  const immediate = engine.runEpisode
  let entered = false
  engine.runEpisode = async (...args) => {
    entered = true
    await gate
    return immediate(...args)
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-residency-pin-'),
    forceJson: true,
    thoughtEngine: engine,
  })
  const firstBase = seedActiveBase(service.store, '-pin-1')
  await service.thought.status()
  const started = await service.thought.start({ text: 'Aktif residency pini.', learningMode: 'off' })
  await waitFor(() => entered && service.thought.active.has(started.episode.id))
  const pinnedKey = [...service.thought.residencyMeta]
    .find(([, meta]) => meta.snapshotId === firstBase.snapshot.id)?.[0]

  seedActiveBase(service.store, '-pin-2')
  await service.thought.status()
  seedActiveBase(service.store, '-pin-3')
  await service.thought.status()

  assert.ok(pinnedKey)
  assert.equal(service.thought.residencies.has(pinnedKey), true)
  assert.equal(service.thought.residencies.size, 2)
  release()
  await waitFor(() => !service.thought.active.has(started.episode.id))
})

test('thought HTTP API persists bounded traces, replays deterministically, and materializes an approved head', async (t) => {
  const calls = []
  const engine = {
    async runEpisode(request, context) {
      calls.push({ request, context })
      return resultFor(request, context)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t), forceJson: true, thoughtEngine: engine,
  })
  seedActiveBase(service.store)
  const base = await listenService(t, service)

  const status = await jsonFetch(base, '/api/thought/status')
  assert.equal(status.response.status, 200)
  assert.equal(status.body.researchBaseAvailable, true)
  assert.equal(status.body.researchBase.matrixSide, 64)
  assert.equal(status.body.researchBase.cellCount, 4_096)
  assert.equal(status.body.researchBaseError, null)

  const started = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', headers: { 'Idempotency-Key': 'episode-one' },
    body: JSON.stringify({ text: 'Ali ekmeği kesti.', learningMode: 'proposal', expectedOutcome: EXPECTED_GRAPH, sourceGroupId: 'human-source-1' }),
  })
  assert.equal(started.response.status, 202)
  assert.equal(started.body.episode.status, 'queued')
  const episodeId = started.body.episode.id

  const duplicate = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', headers: { 'Idempotency-Key': 'episode-one' },
    body: JSON.stringify({ text: 'Ali ekmeği kesti.', learningMode: 'proposal', expectedOutcome: EXPECTED_GRAPH, sourceGroupId: 'human-source-1' }),
  })
  assert.equal(duplicate.response.status, 200)
  assert.equal(duplicate.body.duplicate, true)

  await waitCompleted(service.store, episodeId)
  const ticks = await jsonFetch(base, `/api/thought/episodes/${episodeId}/ticks`)
  assert.equal(ticks.response.status, 200)
  assert.equal(ticks.body.ticks.length, 1)
  assert.equal(ticks.body.ticks[0].signal.before.length, 4_096)
  assert.equal(ticks.body.ticks[0].signal.after.length, 4_096)
  assert.equal(ticks.body.ticks[0].candidates[0].contributions.baseH0, 0.2)
  assert.equal(ticks.body.ticks[0].branches.length, 1)
  assert.equal(ticks.body.ticks[0].branches[0].signal.before.length, 4_096)
  assert.equal(ticks.body.ticks[0].branches[0].signal.after.length, 4_096)
  assert.equal(ticks.body.ticks[0].branches[0].candidates[0].contributions.persistentExcitation, 0.1)
  assert.equal(ticks.body.ticks[0].branches[0].roleHypotheses[0].role, 'actor')
  assert.equal(ticks.body.ticks[0].branches[0].pressure.smoothedBirth, 0.15)
  const traceArtifact = service.store.getArtifact(service.store.getThoughtEpisode(episodeId).traceArtifactIds[0])
  assert.equal(traceArtifact.payload.value.ticks[0].branches[0].signal.before.$type, 'bombom/float32-base64-v1')
  assert.equal(traceArtifact.payload.value.ticks[0].branches[0].signal.before.length, 4_096)

  const episodeGraph = await jsonFetch(base, `/api/thought/episodes/${episodeId}/graph?level=family`)
  assert.equal(episodeGraph.body.graph.level, 'family')
  assert.equal(episodeGraph.body.graph.edges[0].kind, 'family')
  assert.equal(episodeGraph.body.graph.audit.eventGraph.action, 'kesti')
  assert.equal(episodeGraph.body.graph.audit.languageOutput.text, 'Ali ekmeği kesti.')
  assert.equal(episodeGraph.body.graph.audit.languageOutput.stoppedBy, 'eos')
  assert.equal(episodeGraph.body.graph.audit.languageOutput.steps.length, 6)

  const replay = await jsonFetch(base, `/api/thought/episodes/${episodeId}/replay`, { method: 'POST' })
  assert.equal(replay.response.status, 200)
  assert.equal(replay.body.replay.matched, true)

  const approved = await jsonFetch(base, `/api/thought/episodes/${episodeId}/feedback`, {
    method: 'POST', body: JSON.stringify({ decision: 'approve', reason: 'Denetimli insan onayı' }),
  })
  assert.equal(approved.response.status, 200)
  assert.equal(approved.body.snapshot.status, 'candidate')
  const storedEpisode = service.store.getThoughtEpisode(episodeId)
  const head = restoreArtifact(service.store.getArtifact(storedEpisode.headArtifactId), 'thought-runtime-head')
  assert.equal(head.format, 'bombom-thought-head')
  assert.equal(head.revision, 1)
  assert.equal(head.neuronEdges[0].excitation, 0.2)
  assert.equal(head.contextPatches[0].delta instanceof Float32Array, true)
  assert.equal(head.contextPatches[0].delta.length, LAYOUT.relationLength)

  const candidate = service.store.getSnapshot(storedEpisode.candidateSnapshotId)
  const verified = service.store.createSnapshot({
    parentId: service.store.getActiveSnapshot().id,
    status: 'candidate',
    label: 'Bağımsız doğrulanmış düşünce başlığı',
    artifactIds: candidate.artifactIds,
    sourceIds: candidate.sourceIds,
    sourceRevisions: candidate.sourceRevisions,
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(verified.id)
  const second = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Başlığı yeniden kullan.', learningMode: 'off' }),
  })
  await waitCompleted(service.store, second.body.episode.id)
  assert.equal(calls.at(-1).context.thoughtHead.revision, 1)
  assert.equal(calls.at(-1).context.thoughtHead.neuronEdges[0].excitation, 0.2)

  const activeWithHead = service.store.getActiveSnapshot()
  const trainingArtifactsWithoutHead = activeWithHead.artifactIds.filter((artifactId) => service.store.getArtifact(artifactId)?.kind !== 'thought-runtime-head')
  const nextTrainingSnapshot = service.store.createSnapshot({
    parentId: activeWithHead.id,
    status: 'candidate',
    label: 'Aynı FamilyRun ile yeni eğitim snapshotı',
    artifactIds: trainingArtifactsWithoutHead,
    sourceIds: activeWithHead.sourceIds,
    sourceRevisions: activeWithHead.sourceRevisions,
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(nextTrainingSnapshot.id)
  const carried = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Aynı aile fingerprint başlığını taşı.', learningMode: 'off' }),
  })
  await waitCompleted(service.store, carried.body.episode.id)
  assert.equal(calls.at(-1).context.thoughtHead.revision, 1)
  assert.equal(service.store.getThoughtEpisode(carried.body.episode.id).baseDiagnostics.headCarry.status, 'direct-family-fingerprint-carry')

  const globalFamilyGraph = await jsonFetch(base, '/api/thought/graph?level=family')
  assert.equal(globalFamilyGraph.response.status, 200)
  assert.equal(globalFamilyGraph.body.graph.level, 'family')
  assert.equal(globalFamilyGraph.body.graph.edges[0].from, 'family-event')
  assert.ok(globalFamilyGraph.body.graph.audit.derivedFamilyGraphFingerprint)

  const changedFamilyRun = { ...calls.at(-1).context.familyRun, id: 'family-run-v2', fingerprint: 'family-run-fingerprint-v2' }
  const changedFamilyArtifact = service.store.createArtifact(runtimeArtifact(
    'family-run', changedFamilyRun, 'family-artifact-v2',
  ))
  const changedFamilySnapshot = service.store.createSnapshot({
    parentId: nextTrainingSnapshot.id,
    status: 'candidate',
    label: 'Aynı temsil uzayında değişen FamilyRun',
    artifactIds: [
      ...nextTrainingSnapshot.artifactIds.filter((artifactId) => {
        const kind = service.store.getArtifact(artifactId)?.kind
        return kind !== 'family-run' && kind !== 'runtime-model-state'
      }),
      changedFamilyArtifact.id,
    ],
    sourceIds: nextTrainingSnapshot.sourceIds,
    sourceRevisions: nextTrainingSnapshot.sourceRevisions,
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(changedFamilySnapshot.id)
  const migrated = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Değişen ailelerde kesin kimlik deneyimini taşı.', learningMode: 'off' }),
  })
  await waitCompleted(service.store, migrated.body.episode.id)
  const migratedHead = calls.at(-1).context.thoughtHead
  assert.equal(migratedHead.revision, 0)
  assert.equal(migratedHead.neuronEdges.length, 1)
  assert.equal(migratedHead.neuronEdges[0].excitation, 0.2)
  assert.equal(migratedHead.contextPatches.length, 1)
  assert.equal(migratedHead.nursery.length, 0)
  assert.equal(migratedHead.births.length, 0)
  const migrationDiagnostics = service.store.getThoughtEpisode(migrated.body.episode.id).baseDiagnostics.headCarry
  assert.equal(migrationDiagnostics.status, 'filtered-exact-member-migration')
  assert.equal(migrationDiagnostics.exactMemberCount, 1)
  assert.equal(migrationDiagnostics.retainedEdgeCount, 1)
  assert.equal(migrationDiagnostics.retainedContextPatchCount, 1)
})

test('compatible FamilyRun refresh preserves self-verified nursery/birth overlays and quarantines dangling path evidence', async (t) => {
  let observedHead = null
  const engine = {
    async runEpisode(request, context) {
      observedHead = context.thoughtHead
      return resultFor(request, context)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-overlay-migration-'),
    forceJson: true,
    thoughtEngine: engine,
  })
  t.after(() => service.close())
  const seeded = seedActiveBase(service.store, '-overlay-migration')
  const sourceBase = service.thought.resolveBase(seeded.snapshot.id)
  const bornIdentity = new Float32Array(LAYOUT.identityLength)
  bornIdentity[4] = 0.75
  const bornRelation = new Float32Array(LAYOUT.relationLength)
  bornRelation[1] = 0.25
  const bornIdentityDigest = runtimeThoughtDigest('identity', bornIdentity)
  const nurseryIdentity = new Float32Array(LAYOUT.identityLength)
  nurseryIdentity[5] = 0.5
  const nurseryRelation = new Float32Array(LAYOUT.relationLength)
  nurseryRelation[2] = -0.25
  const nurseryIdentityDigest = runtimeThoughtDigest('identity', nurseryIdentity)
  const sourceHeadCore = {
    format: 'bombom-thought-head', version: 1, revision: 7,
    baseModelFingerprint: sourceBase.reference.baseModelFingerprint,
    totalTicks: 64,
    neuronEdges: [
      { fromMemberKey: 'removed-member', toMemberKey: 'born-overlay', excitation: 0.7, inhibition: 0, trustMass: 1, updateCount: 3 },
      { fromMemberKey: MEMBER_KEY, toMemberKey: 'born-overlay', excitation: 0.6, inhibition: 0, trustMass: 1, updateCount: 3 },
    ],
    contextPatches: [{ neuronId: 'born-overlay', delta: bornRelation, weight: 1, updateCount: 3 }],
    nursery: [{
      id: 'nursery-overlay', identity: nurseryIdentity, relationPrototype: nurseryRelation,
      identityDigest: nurseryIdentityDigest,
      birthEpisodeIds: ['run-1', 'run-2'], independentSourceIds: ['source-1', 'source-2'],
      observationCount: 2, residualObservationCount: 6,
      supportWeight: 1.8, trustMass: 1, resultScoreTotal: 1.8,
      originEvidence: [{ key: 'removed-member', weight: 0.4 }, { key: MEMBER_KEY, weight: 0.8 }],
      outgoingEvidence: [{ key: 'removed-member', weight: 0.3 }, { key: MEMBER_KEY, weight: 0.7 }],
      familyEvidence: [{ key: 'retired-family', weight: 1 }],
      nextPromotionTick: 96, residualCohesion: 0.9,
      sourceRelationPrototypes: [
        { sourceGroupId: 'source-1', relation: nurseryRelation, observationCount: 1 },
        { sourceGroupId: 'source-2', relation: nurseryRelation, observationCount: 1 },
      ],
    }],
    births: [{
      format: 'born-neuron-v1', id: 'born-overlay', identity: bornIdentity, relation: bornRelation,
      identityDigest: bornIdentityDigest,
      familyMemberships: [{ familyId: 'family-event', weight: 1 }],
      sourceGroupIds: ['source-1', 'source-2'], trustMass: 1, bornAtTick: 40,
      nurseryId: 'nursery-born-overlay',
      entryEvidence: [
        { memberKey: 'removed-member', weight: 0.4 },
        { memberKey: MEMBER_KEY, weight: 0.9 },
      ],
      outgoingEvidence: [
        { memberKey: 'removed-member', weight: 0.3 },
        { memberKey: MEMBER_KEY, weight: 0.8 },
      ],
      contextEvidence: { observationCount: 3, sourceGroupIds: ['source-1', 'source-2'] },
    }],
  }
  const sourceHead = { ...sourceHeadCore, fingerprint: runtimeHeadFingerprint(sourceHeadCore) }
  const headArtifact = service.store.createArtifact(runtimeArtifact(
    'thought-runtime-head', sourceHead, 'thought-overlay-source-head',
  ))
  const sourceSnapshot = service.store.createSnapshot({
    parentId: seeded.snapshot.id, status: 'candidate', label: 'Doğum overlay kaynak snapshotı',
    artifactIds: [...seeded.snapshot.artifactIds, headArtifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(sourceSnapshot.id)

  const changedFamilyRun = {
    ...seeded.familyRun,
    id: 'family-run-overlay-v2',
    fingerprint: 'family-run-overlay-fingerprint-v2',
  }
  const changedFamilyArtifact = service.store.createArtifact(runtimeArtifact(
    'family-run', changedFamilyRun, 'family-overlay-artifact-v2',
  ))
  const changedSnapshot = service.store.createSnapshot({
    parentId: sourceSnapshot.id, status: 'candidate', label: 'Uyumlu FamilyRun yenilemesi',
    artifactIds: [
      ...seeded.snapshot.artifactIds.filter((artifactId) => service.store.getArtifact(artifactId)?.kind !== 'family-run'),
      changedFamilyArtifact.id,
    ],
    sourceIds: [], evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(changedSnapshot.id)
  const started = await service.thought.start({ text: 'Overlay sürekliliğini denetle.', learningMode: 'off' })
  await waitCompleted(service.store, started.episode.id)

  assert.ok(observedHead)
  assert.equal(observedHead.revision, 0)
  assert.equal(observedHead.totalTicks, 64, JSON.stringify(
    service.store.getThoughtEpisode(started.episode.id).baseDiagnostics.headCarry,
  ))
  assert.equal(observedHead.births.length, 1)
  assert.deepEqual(observedHead.births[0].entryEvidence, [{ memberKey: MEMBER_KEY, weight: 0.9 }])
  assert.deepEqual(observedHead.births[0].outgoingEvidence, [{ memberKey: MEMBER_KEY, weight: 0.8 }])
  assert.equal(observedHead.nursery.length, 1)
  assert.deepEqual(observedHead.nursery[0].originEvidence, [{ key: MEMBER_KEY, weight: 0.8 }])
  assert.deepEqual(observedHead.nursery[0].outgoingEvidence, [{ key: MEMBER_KEY, weight: 0.7 }])
  assert.match(observedHead.nursery[0].familyEvidence[0].key, /^nursery-provisional:/u)
  assert.equal(observedHead.neuronEdges.length, 1)
  assert.equal(observedHead.contextPatches[0].neuronId, 'born-overlay')
  const diagnostics = service.store.getThoughtEpisode(started.episode.id).baseDiagnostics.headCarry
  assert.equal(diagnostics.status, 'filtered-exact-member-migration')
  assert.equal(diagnostics.retainedBirthCount, 1)
  assert.equal(diagnostics.retainedNurseryCount, 1)
  assert.equal(diagnostics.quarantinedBirthCount, 0)
  assert.equal(diagnostics.quarantinedNurseryCount, 0)
})

test('present thought signal heatmaps must contain exactly 4096 Float32 cells', async (t) => {
  const engine = {
    async runEpisode(request, context) {
      const tick = traceTick(1)
      tick.branches[0].beforeSignal = new Float32Array(4_095)
      return resultFor(request, context, [tick])
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-signal-size-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-signal-size')
  const started = await service.thought.start({ text: 'Eksik hücreli sinyali reddet.' })
  const failed = await waitFor(() => {
    const episode = service.store.getThoughtEpisode(started.episode.id)
    return episode?.status === 'failed' ? episode : null
  })
  assert.equal(failed.error.code, 'THOUGHT_ENGINE_OUTPUT_INVALID')
  assert.match(failed.error.message, /tam 4096/)
})

test('episode POST is non-blocking and 64-tick checkpoint supports pause, resume, cancel, and replay', async (t) => {
  let releasePause
  const pauseGate = new Promise((resolve) => { releasePause = resolve })
  const engine = {
    async runEpisode(request, context) {
      if (request.replay) return resultFor(request, context, Array.from({ length: 65 }, (_, index) => traceTick(index + 1)))
      if (request.inputText.includes('iptal')) {
        await new Promise((resolve, reject) => {
          context.signal.addEventListener('abort', () => reject(context.signal.reason), { once: true })
        })
      }
      if (request.resume) return resultFor(request, context, [traceTick(65)])
      const ticks = Array.from({ length: 64 }, (_, index) => traceTick(index + 1))
      for (const tick of ticks) context.onTick(tick)
      await context.onCheckpoint({
        tick: 64,
        ticks,
        state: {
          format: 'bombom-thought-episode', version: 1, episodeId: request.episodeId, tick: 64,
          maximumTicks: request.maxTicks, status: 'active', branches: [{ id: 'branch-1', signal: new Float32Array(4_096) }],
          replay: [], rngState: 123, fingerprint: 'checkpoint-64',
        },
      })
      await pauseGate
      if (await context.shouldPause()) {
        return {
          ...resultFor(request, { ...context, onTick: () => {} }, ticks),
          episode: {
            format: 'bombom-thought-episode', version: 1, episodeId: request.episodeId, tick: 64,
            maximumTicks: request.maxTicks, status: 'paused', branches: [], replay: [], fingerprint: 'checkpoint-64',
          },
        }
      }
      return resultFor(request, { ...context, onTick: () => {} }, ticks)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t), forceJson: true, thoughtEngine: engine,
  })
  seedActiveBase(service.store, '-pause')
  const base = await listenService(t, service)
  const before = Date.now()
  const start = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Uzun düşünceyi çalıştır.', maxTicks: 65 }),
  })
  assert.equal(start.response.status, 202)
  assert.ok(Date.now() - before < 500)
  const episodeId = start.body.episode.id
  await waitFor(() => service.store.getThoughtEpisode(episodeId)?.checkpointArtifactId)
  const checkpoint = restoreArtifact(
    service.store.getArtifact(service.store.getThoughtEpisode(episodeId).checkpointArtifactId),
    'thought-episode-checkpoint',
  )
  assert.equal(checkpoint.tickCount, 64)
  assert.equal(checkpoint.state.branches[0].signal instanceof Float32Array, true)

  const pause = await jsonFetch(base, `/api/thought/episodes/${episodeId}/pause`, { method: 'POST' })
  assert.equal(pause.body.episode.status, 'pause_requested')
  releasePause()
  await waitFor(() => service.store.getThoughtEpisode(episodeId)?.status === 'paused')
  await waitFor(() => service.thought.active.size === 0)
  assert.equal(service.store.getThoughtEpisode(episodeId).tickCount, 64)

  const resume = await jsonFetch(base, `/api/thought/episodes/${episodeId}/resume`, { method: 'POST' })
  assert.equal(resume.response.status, 200)
  await waitCompleted(service.store, episodeId)
  assert.equal(service.store.getThoughtEpisode(episodeId).tickCount, 65)
  const replay = await jsonFetch(base, `/api/thought/episodes/${episodeId}/replay`, { method: 'POST' })
  assert.equal(replay.body.replay.matched, true)

  const cancelling = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Bu düşünceyi iptal et.', maxTicks: 65 }),
  })
  const cancelId = cancelling.body.episode.id
  await waitFor(() => service.store.getThoughtEpisode(cancelId)?.status === 'running')
  const cancel = await jsonFetch(base, `/api/thought/episodes/${cancelId}/cancel`, { method: 'POST' })
  assert.equal(cancel.body.episode.status, 'cancel_requested')
  await waitFor(() => service.store.getThoughtEpisode(cancelId)?.status === 'cancelled')
  const emptyReplay = await jsonFetch(base, `/api/thought/episodes/${cancelId}/replay`, { method: 'POST' })
  assert.equal(emptyReplay.response.status, 409)
  assert.equal(emptyReplay.body.code, 'THOUGHT_REPLAY_UNAVAILABLE')
})

test('512-tick execution keeps only a rolling 64-tick raw window between checkpoints', async (t) => {
  const engine = {
    async runEpisode(request, context) {
      const window = []
      for (let index = 1; index <= 512; index += 1) {
        const tick = {
          tick: index,
          branchCount: 0,
          activeBranchCount: 0,
          replayFingerprint: `rolling-${index}`,
        }
        window.push(tick)
        context.onTick(tick)
        if (index % 64 === 0) {
          await context.onCheckpoint({
            tick: index,
            ticks: window.splice(0),
            state: {
              format: 'bombom-thought-episode', version: 1, episodeId: request.episodeId,
              tick: index, maximumTicks: request.maxTicks, status: index === 512 ? 'stopped' : 'active',
              branches: [], replay: [], rngState: index, fingerprint: `checkpoint-${index}`,
            },
          })
        }
      }
      return {
        episode: {
          format: 'bombom-thought-episode', version: 1, episodeId: request.episodeId,
          tick: 512, maximumTicks: request.maxTicks, status: 'stopped', branches: [], replay: [],
          fingerprint: 'episode-512', stopReason: 'maximum-ticks',
        },
        ticks: [],
        graph: { nodes: [], edges: [] },
      }
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-rolling-window-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-rolling-window')
  const started = await service.thought.start({ text: 'Tam 512 tick düşün.', maxTicks: 512 })
  const completed = await waitCompleted(service.store, started.episode.id)
  assert.equal(completed.tickCount, 512)
  assert.equal(completed.traceArtifactIds.length, 8)
  const chunkSizes = completed.traceArtifactIds.map((artifactId) => (
    restoreArtifact(service.store.getArtifact(artifactId), 'thought-trace-chunk').ticks.length
  ))
  assert.deepEqual(chunkSizes, Array(8).fill(64))
  assert.ok(completed.lastTickDigest)

  const traceIds = new Set(completed.traceArtifactIds)
  const getArtifact = service.store.getArtifact.bind(service.store)
  let traceArtifactReads = 0
  service.store.getArtifact = (artifactId) => {
    if (traceIds.has(artifactId)) traceArtifactReads += 1
    return getArtifact(artifactId)
  }
  try {
    const firstPage = service.thought.ticks(completed.id, { after: 0, limit: 1, candidateLimit: 0 })
    assert.equal(firstPage.ticks.length, 1)
    assert.equal(firstPage.pagination.hasMore, true)
    assert.equal(traceArtifactReads, 1)
  } finally {
    service.store.getArtifact = getArtifact
  }
})

test('Ollama cross-check requires two agreeing high-confidence passes and never becomes evaluation evidence', async (t) => {
  const outputs = []
  let ollamaCalls = 0
  const ollama = {
    async generateStructured() { ollamaCalls += 1; return outputs.shift() },
    async check() { return { available: true, models: [] } },
    async listModels() { return [] },
  }
  const requests = []
  const engine = {
    async runEpisode(request, context) {
      requests.push(request)
      return resultFor(request, context, [traceTick(1)], 0.25)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t), forceJson: true, thoughtEngine: engine, ollama,
  })
  service.store.updateConfig({ ollama: { model: 'qwen:14b' } })
  seedActiveBase(service.store, '-ollama')
  const base = await listenService(t, service)
  const normalized = { ...EXPECTED_GRAPH, actor: 'ali', action: 'kesti', patient: 'ekmek' }
  outputs.push(
    { value: { graph: normalized, confidence: 0.93 }, model: 'qwen:14b', promptTokens: 10, completionTokens: 5 },
    { value: { graph: normalized, confidence: 0.91 }, model: 'qwen:14b', promptTokens: 9, completionTokens: 4 },
  )
  const passing = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Ali ekmeği kesti.', learningMode: 'proposal', teacherMode: 'ollama-cross-check' }),
  })
  await waitCompleted(service.store, passing.body.episode.id)
  const passedEpisode = service.store.getThoughtEpisode(passing.body.episode.id)
  assert.ok(passedEpisode.headProposalArtifactId)
  assert.equal(passedEpisode.teacherMode, 'ollama-cross-check')
  assert.equal(requests[0].outcomeEvidence.kind, 'ollama-cross-checked')
  assert.equal(requests[0].outcomeEvidence.confidence, 0.91)
  assert.equal(requests[0].outcomeEvidence.evaluationEligible, false)
  const proposal = restoreArtifact(service.store.getArtifact(passedEpisode.headProposalArtifactId), 'thought-head-proposal')
  assert.equal(proposal.neuronEdges[0].trustMassQ, 250_000)

  outputs.push(
    { value: { graph: normalized, confidence: 0.95 }, model: 'qwen:14b', promptTokens: 1, completionTokens: 1 },
    { value: { graph: { ...normalized, action: 'aldı' }, confidence: 0.95 }, model: 'qwen:14b', promptTokens: 1, completionTokens: 1 },
  )
  const disagreeing = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'İkinci bağımsız metin.', learningMode: 'proposal', crossCheck: true }),
  })
  await waitCompleted(service.store, disagreeing.body.episode.id)
  const disagreedEpisode = service.store.getThoughtEpisode(disagreeing.body.episode.id)
  assert.equal(disagreedEpisode.crossCheck.status, 'disagreed')
  assert.equal(disagreedEpisode.headProposalArtifactId, null)
  assert.equal(disagreedEpisode.deltaSuppressedReason, 'external-outcome-not-trusted')

  const callsAfterTerminalCrossCheck = ollamaCalls
  service.store.updateThoughtEpisode(disagreedEpisode.id, { status: 'paused', finishedAt: null, error: null })
  service.thought.resume(disagreedEpisode.id)
  await waitCompleted(service.store, disagreedEpisode.id)
  assert.equal(ollamaCalls, callsAfterTerminalCrossCheck, 'a persisted disagreed cross-check must not be purchased again on resume')

  const malformed = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST',
    body: JSON.stringify({ text: 'Geçersiz hedef.', expectedOutcome: { ...EXPECTED_GRAPH, negated: 'hayır' } }),
  })
  assert.equal(malformed.response.status, 422)
  assert.equal(malformed.body.code, 'INVALID_EVENT_GRAPH')
  const contradictory = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Çelişkili öğretmen modu.', teacherMode: 'off', crossCheck: true }),
  })
  assert.equal(contradictory.response.status, 422)
  assert.equal(contradictory.body.code, 'INVALID_THOUGHT_REQUEST')
})

test('a remote thought pause stops before the next paid cross-check pass', async (t) => {
  await assertRemoteCrossCheckControlStopsNextPass(t, 'pause')
})

test('a remote thought cancel stops before the next paid cross-check pass', async (t) => {
  await assertRemoteCrossCheckControlStopsNextPass(t, 'cancel')
})

test('atomic thought usage reservation closes the durable control read-to-reserve gap', async (t) => {
  let ollamaCalls = 0
  const graph = { ...EXPECTED_GRAPH, actor: 'ali', action: 'kesti', patient: 'ekmek' }
  const ollama = {
    async generateStructured() {
      ollamaCalls += 1
      return { value: { graph, confidence: 0.95 }, model: 'qwen:14b', promptTokens: 1, completionTokens: 1 }
    },
    async check() { return { available: true, models: [] } },
    async listModels() { return [] },
  }
  const engine = {
    async runEpisode(request, context) {
      return resultFor(request, context)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-atomic-reserve-gap-'),
    forceJson: true,
    thoughtEngine: engine,
    ollama,
  })
  t.after(() => service.close())
  service.store.updateConfig({ ollama: { model: 'qwen:14b' } })
  const remote = createThoughtService({ store: service.store, ollama, thoughtEngine: engine })
  seedActiveBase(service.store, '-atomic-reserve-gap')

  const originalGetThoughtEpisode = service.store.getThoughtEpisode.bind(service.store)
  let controlInterleaved = false
  service.store.getThoughtEpisode = (episodeId) => {
    const stale = originalGetThoughtEpisode(episodeId)
    if (!controlInterleaved && stale?.status === 'running' && stale.inputText === 'Atomik rezervasyon aralığını kapat.') {
      controlInterleaved = true
      const cancelled = remote.cancel(episodeId)
      assert.equal(cancelled.status, 'cancel_requested')
    }
    return stale
  }
  let started
  try {
    started = await service.thought.start({
      text: 'Atomik rezervasyon aralığını kapat.',
      teacherMode: 'ollama-cross-check',
    })
    await waitFor(() => service.store.getThoughtEpisode(started.episode.id)?.status === 'cancelled')
  } finally {
    service.store.getThoughtEpisode = originalGetThoughtEpisode
  }

  assert.equal(controlInterleaved, true)
  assert.equal(ollamaCalls, 0)
  assert.equal(service.store.getUsage().teacherRequests, 0)
  assert.equal(service.store.getUsage().teacherTokens, 0)
})

test('missing engine is explicit and completion drops a proposal when the active base changes', async (t) => {
  const unavailable = await createTrainingService({ dataDirectory: await temporaryDirectory(t, 'thought-unavailable-'), forceJson: true })
  seedActiveBase(unavailable.store, '-unavailable')
  const unavailableBase = await listenService(t, unavailable)
  const missing = await jsonFetch(unavailableBase, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Motor yok.' }),
  })
  assert.equal(missing.response.status, 503)
  assert.equal(missing.body.code, 'THOUGHT_ENGINE_UNAVAILABLE')

  let release
  const gate = new Promise((resolve) => { release = resolve })
  const engine = {
    async runEpisode(request, context) {
      await gate
      return resultFor(request, context)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-stale-'), forceJson: true, thoughtEngine: engine,
  })
  seedActiveBase(service.store, '-stale')
  const base = await listenService(t, service)
  const started = await jsonFetch(base, '/api/thought/episodes', {
    method: 'POST', body: JSON.stringify({ text: 'Taban değişirken düşün.', learningMode: 'proposal', expectedOutcome: EXPECTED_GRAPH }),
  })
  await waitFor(() => service.store.getThoughtEpisode(started.body.episode.id)?.status === 'running')
  const current = service.store.getActiveSnapshot()
  const replacement = service.store.createSnapshot({
    parentId: current.id, status: 'candidate', label: 'Yeni doğrulanmış taban',
    artifactIds: current.artifactIds, sourceIds: current.sourceIds, sourceRevisions: current.sourceRevisions,
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(replacement.id)
  release()
  await waitCompleted(service.store, started.body.episode.id)
  const stale = service.store.getThoughtEpisode(started.body.episode.id)
  assert.equal(stale.error.code, 'STALE_THOUGHT_BASE')
  assert.equal(stale.commitEligible, false)
  assert.equal(stale.headProposalArtifactId, null)
  assert.ok(stale.traceArtifactIds.length > 0)
})

test('service restart discards uncheckpointed tail before resume and deterministic replay', async (t) => {
  const directory = await temporaryDirectory(t, 'thought-restart-')
  const interruptedEngine = {
    async runEpisode(request, context) {
      context.onTick(traceTick(1))
      await new Promise((resolve, reject) => {
        if (context.signal.aborted) reject(context.signal.reason)
        else context.signal.addEventListener('abort', () => reject(context.signal.reason), { once: true })
      })
    },
  }
  const firstService = await createTrainingService({ dataDirectory: directory, forceJson: true, thoughtEngine: interruptedEngine })
  seedActiveBase(firstService.store, '-restart')
  const started = await firstService.thought.start({ text: 'Yeniden başlatma sırasında düşün.' })
  const episodeId = started.episode.id
  await waitFor(() => firstService.store.getThoughtEpisode(episodeId)?.status === 'running')
  await firstService.close()

  const resumedEngine = {
    async runEpisode(request, context) {
      return resultFor(request, context, [traceTick(1)])
    },
  }
  const secondService = await createTrainingService({ dataDirectory: directory, forceJson: true, thoughtEngine: resumedEngine })
  t.after(() => secondService.close())
  const recovered = secondService.store.getThoughtEpisode(episodeId)
  assert.equal(recovered.status, 'paused')
  assert.equal(recovered.tickCount, 0)
  assert.deepEqual(recovered.traceArtifactIds, [])
  secondService.thought.resume(episodeId)
  await waitCompleted(secondService.store, episodeId)
  await waitFor(() => secondService.thought.active.size === 0)
  const replay = await secondService.thought.replay(episodeId)
  assert.equal(replay.matched, true)
  assert.equal(replay.expectedTickCount, 1)
})

test('fresh-service replay keeps the dense catalog for pre-residency episodes', async (t) => {
  const dataDirectory = await temporaryDirectory(t, 'thought-legacy-dense-replay-')
  let replayNeuronCount = null
  const engine = {
    async runEpisode(request, context) {
      if (request.replay) replayNeuronCount = context.neurons?.length ?? 0
      return resultFor(request, context, [traceTick(1)])
    },
  }
  const first = await createTrainingService({ dataDirectory, forceJson: true, thoughtEngine: engine })
  seedActiveBase(first.store, '-legacy-dense-replay')
  const started = await first.thought.start({ text: 'Yerleşiklik öncesi yoğun katalog replay deneyi.', maxTicks: 1 })
  await waitCompleted(first.store, started.episode.id)
  const stored = first.store.getThoughtEpisode(started.episode.id)
  const { residencyManifestFingerprint: _manifest, ...legacyBase } = stored.base
  const { residency: _residency, ...legacyDiagnostics } = stored.baseDiagnostics ?? {}
  first.store.updateThoughtEpisode(stored.id, {
    base: legacyBase,
    baseDiagnostics: legacyDiagnostics,
  })
  await first.close()

  const second = await createTrainingService({ dataDirectory, forceJson: true, thoughtEngine: engine })
  t.after(() => second.close())
  const replay = await second.thought.replay(stored.id)
  assert.equal(replay.matched, true)
  assert.equal(replayNeuronCount, 1, 'legacy replay must receive the original full dense neuron catalog')
})

test('failed partial episodes replay only their persisted deterministic tick prefix', async (t) => {
  let invocation = 0
  const engine = {
    async runEpisode(request, context) {
      invocation += 1
      if (!request.replay) {
        context.onTick(traceTick(1))
        context.onTick(traceTick(2))
        throw new Error('İki tick sonrasında kontrollü motor hatası')
      }
      assert.equal(request.maxTicks, 64)
      assert.equal(request.replay.expectedTickCount, 2)
      return resultFor(request, context, [traceTick(1), traceTick(2)])
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-partial-replay-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-partial-replay')
  const started = await service.thought.start({ text: 'Yalnız kalıcı prefixi yeniden oynat.' })
  const failed = await waitFor(() => {
    const episode = service.store.getThoughtEpisode(started.episode.id)
    return episode?.status === 'failed' ? episode : null
  })
  assert.equal(failed.tickCount, 2)
  const replay = await service.thought.replay(started.episode.id)
  assert.equal(replay.matched, true)
  assert.equal(replay.expectedTickCount, 2)
  assert.equal(replay.actualTickCount, 2)
  assert.equal(invocation, 2)
})

test('replay normalizes each streamed tick immediately instead of retaining raw heatmaps', async (t) => {
  const engine = {
    async runEpisode(request, context) {
      if (!request.replay) return resultFor(request, context, [traceTick(1)])
      const tick = traceTick(1)
      context.onTick(tick)
      tick.signal.before[0] = 0.99
      tick.branches[0].beforeSignal[0] = 0.99
      return {
        episode: {
          format: 'bombom-thought-episode', version: 1, episodeId: request.episodeId,
          tick: 1, maximumTicks: request.maxTicks, status: 'paused', branches: [], replay: [],
          fingerprint: 'streamed-replay', stopReason: 'replay-prefix',
        },
        ticks: [], graph: { nodes: [], edges: [] },
      }
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-streamed-replay-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-streamed-replay')
  const started = await service.thought.start({ text: 'Replay ısı haritasını akışlı doğrula.' })
  await waitCompleted(service.store, started.episode.id)
  const replay = await service.thought.replay(started.episode.id)
  assert.equal(replay.matched, true)
  assert.equal(replay.actualTickCount, 1)
})

test('materialized head deterministically retains the strongest 96 experience edges per origin', () => {
  const retained = retainMaterializedEdges(Array.from({ length: 100 }, (_, index) => ({
    fromMemberKey: MEMBER_KEY,
    toMemberKey: `target-${String(index).padStart(3, '0')}`,
    excitation: (index + 1) / 100,
    inhibition: index === 99 ? 0.95 : 0,
    trustMass: index === 3 ? 1 : 0.5,
    updateCount: index + 1,
  })))
  assert.equal(retained.length, 96)
  const targets = new Set(retained.map((edge) => edge.toMemberKey))
  assert.equal(targets.has('target-000'), false)
  assert.equal(targets.has('target-003'), false)
  assert.equal(targets.has('target-098'), true)
  assert.equal(targets.has('target-099'), true)
})

test('derived family graph uses row-softmax virtual H and inhibition monotonically lowers G probability', () => {
  const familyRun = {
    fingerprint: 'family-softmax-test',
    memberships: [
      { memberKey: 'origin', familyId: 'family-origin', weight: 1 },
      { memberKey: 'target-a', familyId: 'family-a', weight: 1 },
      { memberKey: 'target-b', familyId: 'family-b', weight: 1 },
    ],
    neuronTransitions: [
      { fromMemberKey: 'origin', toMemberKey: 'target-a', probability: 0.5 },
      { fromMemberKey: 'origin', toMemberKey: 'target-b', probability: 0.5 },
    ],
  }
  const head = (inhibition) => ({
    fingerprint: `head-${inhibition}`,
    births: [],
    neuronEdges: [{
      fromMemberKey: 'origin', toMemberKey: 'target-a', excitation: 0, inhibition, trustMass: 1, updateCount: 1,
    }],
  })
  const baseline = deriveFamilyGraphView({ familyRun, thoughtHead: head(0) })
  const inhibited = deriveFamilyGraphView({ familyRun, thoughtHead: head(9) })
  const probabilityFor = (graph, target) => graph.transitions.find((edge) => edge.from === 'family-origin' && edge.to === target).probability
  assert.ok(Math.abs(probabilityFor(baseline, 'family-a') - 0.5) < 1e-12)
  assert.ok(probabilityFor(inhibited, 'family-a') < probabilityFor(baseline, 'family-a'))
  assert.ok(probabilityFor(inhibited, 'family-b') > probabilityFor(baseline, 'family-b'))
  for (const graph of [baseline, inhibited]) {
    const rowSum = graph.transitions.filter((edge) => edge.from === 'family-origin').reduce((sum, edge) => sum + edge.probability, 0)
    assert.ok(Math.abs(rowSum - 1) < 1e-12)
  }
})

test('family graph exposes provisional family nodes introduced by promoted births', () => {
  const nodes = familyGraphNodes({
    familyRun: { families: [{ id: 'known-family', region: 'event', lifecycle: 'stable', supportCount: 2, sourceCount: 2 }] },
    thoughtHead: {
      births: [{
        id: 'born-1', sourceGroupIds: ['source-a', 'source-b'],
        familyMemberships: [{ familyId: 'nursery-provisional:abc', weight: 1 }],
      }],
    },
  })
  const provisional = nodes.find((node) => node.id === 'nursery-provisional:abc')
  assert.equal(provisional.lifecycle, 'provisional')
  assert.equal(provisional.region, null)
  assert.equal(provisional.sourceCount, 2)
})

test('nursery path/context evidence survives commits and promotion atomically keeps it on the born neuron', async (t) => {
  const identity = new Float32Array(LAYOUT.identityLength)
  identity[0] = 0.5
  const identityDigest = runtimeThoughtDigest('identity', identity)
  const relationA = new Float32Array(LAYOUT.relationLength)
  const relationB = new Float32Array(LAYOUT.relationLength)
  const relationC = new Float32Array(LAYOUT.relationLength)
  relationA[0] = 1
  relationB[0] = -1
  const observedEpisodeIds = []
  let invocation = 0
  const engine = {
    async runEpisode(request, context) {
      const result = resultFor(request, context)
      if (invocation === 1) {
        assert.equal(context.thoughtHead.nursery[0].residualCohesion, 0.9)
        assert.deepEqual(context.thoughtHead.nursery[0].outgoingEvidence, [{ key: MEMBER_KEY, weight: 0.4 }])
        assert.equal(context.thoughtHead.nursery[0].sourceRelationPrototypes.length, 1)
        assert.equal(context.thoughtHead.nursery[0].sourceRelationPrototypes[0].relation instanceof Float32Array, true)
        assert.equal(context.thoughtHead.nursery[0].sourceRelationPrototypes[0].relation[0], 1)
      }
      if (invocation === 2) {
        assert.deepEqual(context.thoughtHead.nursery[0].outgoingEvidence, [{ key: MEMBER_KEY, weight: 0.8 }])
      }
      const sourceIds = ['source-a', 'source-b', 'source-c'].slice(0, invocation + 1)
      const episodes = [...observedEpisodeIds, request.episodeId]
      const relationPrototype = invocation === 0 ? relationA : new Float32Array(LAYOUT.relationLength)
      const sourceRelations = [relationA, relationB, relationC].slice(0, invocation + 1)
      result.headDelta.nurseryUpserts = [{
        id: 'nursery-candidate-1', identity, relationPrototype, identityDigest,
        birthEpisodeIds: episodes,
        independentSourceIds: sourceIds,
        observationCount: invocation + 1,
        residualObservationCount: invocation + 1,
        supportWeight: 0.9 * (invocation + 1),
        trustMass: invocation + 1,
        resultScoreTotal: 0.9 * (invocation + 1),
        originEvidence: [{ key: MEMBER_KEY, weight: 0.5 * (invocation + 1) }],
        outgoingEvidence: [{ key: MEMBER_KEY, weight: 0.4 * (invocation + 1) }],
        familyEvidence: [], nextPromotionTick: 0,
        residualCohesion: 0.9,
        sourceRelationPrototypes: sourceIds.map((sourceGroupId, index) => ({
          sourceGroupId, relation: sourceRelations[index], observationCount: 1,
        })),
      }]
      if (invocation === 2) {
        result.headDelta.nurseryRemoveIds = ['nursery-candidate-1']
        result.headDelta.births = [{
          format: 'born-neuron-v1', id: 'born-promoted-1', identity, relation: relationPrototype,
          identityDigest,
          familyMemberships: [{ familyId: 'family-event', weight: 1 }],
          sourceGroupIds: sourceIds,
          trustMass: 3,
          bornAtTick: 3,
          nurseryId: 'nursery-candidate-1',
          entryEvidence: [{ memberKey: MEMBER_KEY, weight: 1.5 }],
          outgoingEvidence: [{ memberKey: MEMBER_KEY, weight: 1.2 }],
          contextEvidence: { observationCount: 3, sourceGroupIds: sourceIds },
        }]
      }
      observedEpisodeIds.push(request.episodeId)
      invocation += 1
      return result
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-nursery-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-nursery')

  const first = await service.thought.start({
    text: 'Birinci bağımsız nursery gözlemi.', learningMode: 'proposal',
    expectedOutcome: EXPECTED_GRAPH, sourceGroupId: 'source-a',
  })
  await waitCompleted(service.store, first.episode.id)
  await waitFor(() => service.thought.active.size === 0)
  service.thought.feedback(first.episode.id, { decision: 'approve' })
  const firstStored = service.store.getThoughtEpisode(first.episode.id)
  const firstCandidate = service.store.getSnapshot(firstStored.candidateSnapshotId)
  const firstVerified = service.store.createSnapshot({
    parentId: service.store.getActiveSnapshot().id, status: 'candidate', label: 'Nursery 1 doğrulandı',
    artifactIds: firstCandidate.artifactIds, sourceIds: firstCandidate.sourceIds, sourceRevisions: firstCandidate.sourceRevisions,
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(firstVerified.id)

  const second = await service.thought.start({
    text: 'İkinci bağımsız nursery gözlemi.', learningMode: 'proposal',
    expectedOutcome: EXPECTED_GRAPH, sourceGroupId: 'source-b',
  })
  await waitCompleted(service.store, second.episode.id)
  await waitFor(() => service.thought.active.size === 0)
  service.thought.feedback(second.episode.id, { decision: 'approve' })
  const secondStored = service.store.getThoughtEpisode(second.episode.id)
  const finalHead = restoreArtifact(service.store.getArtifact(secondStored.headArtifactId), 'thought-runtime-head')
  assert.equal(finalHead.nursery.length, 1)
  assert.equal(finalHead.nursery[0].residualCohesion, 0.9)
  assert.equal(finalHead.nursery[0].sourceRelationPrototypes.length, 2)
  assert.deepEqual(finalHead.nursery[0].sourceRelationPrototypes.map((entry) => entry.sourceGroupId), ['source-a', 'source-b'])
  assert.equal(finalHead.nursery[0].relationPrototype[0], 0)
  assert.equal(finalHead.nursery[0].sourceRelationPrototypes[0].relation[0], 1)
  assert.equal(finalHead.nursery[0].sourceRelationPrototypes[1].relation[0], -1)
  assert.deepEqual(finalHead.nursery[0].originEvidence, [{ key: MEMBER_KEY, weight: 1 }])
  assert.deepEqual(finalHead.nursery[0].outgoingEvidence, [{ key: MEMBER_KEY, weight: 0.8 }])

  const secondCandidate = service.store.getSnapshot(secondStored.candidateSnapshotId)
  const secondVerified = service.store.createSnapshot({
    parentId: service.store.getActiveSnapshot().id, status: 'candidate', label: 'Nursery 2 doğrulandı',
    artifactIds: secondCandidate.artifactIds, sourceIds: secondCandidate.sourceIds, sourceRevisions: secondCandidate.sourceRevisions,
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(secondVerified.id)
  const third = await service.thought.start({
    text: 'Üçüncü bağımsız nursery gözlemi ve terfi.', learningMode: 'proposal',
    expectedOutcome: EXPECTED_GRAPH, sourceGroupId: 'source-c',
  })
  await waitCompleted(service.store, third.episode.id)
  await waitFor(() => service.thought.active.size === 0)
  service.thought.feedback(third.episode.id, { decision: 'approve' })
  const promotedEpisode = service.store.getThoughtEpisode(third.episode.id)
  const promotedHead = restoreArtifact(service.store.getArtifact(promotedEpisode.headArtifactId), 'thought-runtime-head')
  assert.equal(promotedHead.nursery.length, 0)
  assert.equal(promotedHead.births.length, 1)
  assert.deepEqual(promotedHead.births[0].entryEvidence, [{ memberKey: MEMBER_KEY, weight: 1.5 }])
  assert.deepEqual(promotedHead.births[0].outgoingEvidence, [{ memberKey: MEMBER_KEY, weight: 1.2 }])
  assert.deepEqual(promotedHead.births[0].contextEvidence, {
    observationCount: 3, sourceGroupIds: ['source-a', 'source-b', 'source-c'],
  })
})

test('head delta rejects ghost member keys before they become persistent experience edges', async (t) => {
  const engine = {
    async runEpisode(request, context) {
      const result = resultFor(request, context)
      result.headDelta.neuronEdges[0].toMemberKey = 'temporary-prompt-neuron'
      return result
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-ghost-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-ghost')
  const started = await service.thought.start({
    text: 'Geçici nöronu kalıcılaştırma.', learningMode: 'proposal', expectedOutcome: EXPECTED_GRAPH,
  })
  const failed = await waitFor(() => {
    const episode = service.store.getThoughtEpisode(started.episode.id)
    return episode?.status === 'failed' ? episode : null
  })
  assert.equal(failed.error.code, 'THOUGHT_DANGLING_MEMBER_KEY')
  assert.equal(failed.headProposalArtifactId, null)
})

test('same base input and seed produce the same tick digest across distinct episode ids', async (t) => {
  const engine = { async runEpisode(request, context) { return resultFor(request, context) } }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-determinism-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-determinism')
  const first = await service.thought.start({ text: 'Aynı düşünce.', seed: 77 }, { idempotencyKey: 'determinism-a' })
  await waitCompleted(service.store, first.episode.id)
  await waitFor(() => service.thought.active.size === 0)
  const second = await service.thought.start({ text: 'Aynı düşünce.', seed: 77 }, { idempotencyKey: 'determinism-b' })
  await waitCompleted(service.store, second.episode.id)
  assert.notEqual(first.episode.id, second.episode.id)
  assert.equal(
    service.store.getThoughtEpisode(first.episode.id).lastTickDigest,
    service.store.getThoughtEpisode(second.episode.id).lastTickDigest,
  )
})

test('thought request records plastic-region modes while policy-off remains behavior-free', async (t) => {
  const engineRequests = []
  const engine = {
    async runEpisode(request, context) {
      engineRequests.push(request)
      return resultFor(request, context)
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-region-mode-'), forceJson: true, thoughtEngine: engine,
  })
  t.after(() => service.close())
  seedActiveBase(service.store, '-region-mode')

  const started = await service.thought.start({ text: 'Bölge kipini yalnız metadata olarak kaydet.', regionMode: 'active' })
  assert.equal(started.episode.requestedRegionMode, 'active')
  assert.equal(started.episode.regionMode, 'off')
  assert.equal(started.episode.regionRuntime.behaviorApplied, false)
  await waitCompleted(service.store, started.episode.id)
  assert.equal(Object.hasOwn(engineRequests[0], 'regionMode'), false)
  assert.equal(Object.hasOwn(engineRequests[0], 'adaptiveRegion'), false, 'off must not alter the legacy engine request or digest')

  await waitFor(() => service.thought.active.size === 0)
  await assert.rejects(
    () => service.thought.start({ text: 'Geçersiz bölge kipi.', regionMode: 'unknown' }),
    (error) => error?.code === 'INVALID_THOUGHT_REQUEST',
  )
  const status = await service.thought.status()
  assert.deepEqual(status.plasticRegions.requestModes, ['off', 'shadow', 'proposal', 'active'])
  assert.equal(status.plasticRegions.effectiveMode, 'off')
  assert.equal(status.plasticRegions.runtimeWired, true)
  assert.equal(status.plasticRegions.engineAvailable, true)
})

test('active 8/56 episodes persist bounded region telemetry and replay the exact atlas binding', async (t) => {
  const requests = []
  const engine = {
    async runEpisode(request, context) {
      requests.push(request)
      assert.equal(request.adaptiveRegion.mode, 'active')
      assert.equal(request.adaptiveRegion.atlasFingerprint, context.adaptiveRegionAtlas.fingerprint)
      const binding = adaptiveBindingFor(request, context)
      if (request.replay) {
        assert.deepEqual(request.replay.adaptiveRegion, {
          mode: binding.mode,
          atlasFingerprint: binding.atlasFingerprint,
          bindingFingerprint: binding.fingerprint,
        })
      }
      const tick = adaptiveTraceTick(request, binding)
      context.onTick?.(tick)
      return {
        episode: {
          format: 'bombom-thought-episode', version: 1, episodeId: request.episodeId,
          tick: 1, maximumTicks: request.maxTicks, status: 'stopped', branches: [], replay: [],
          fingerprint: `adaptive-episode-${binding.fingerprint}`, stopReason: 'settled', adaptiveRegion: binding,
        },
        ticks: [tick],
        graph: { nodes: [], edges: [] },
      }
    },
  }
  const service = await createTrainingService({
    dataDirectory: await temporaryDirectory(t, 'thought-adaptive-region-'),
    forceJson: true,
    thoughtEngine: engine,
  })
  t.after(() => service.close())
  service.store.updateConfig({
    plasticRegions: {
      enabled: true,
      mode: 'active',
      representationProfile: 'adaptive-eighth-id-v1',
    },
  })
  const seeded = await seedActiveAdaptiveBase(service.store, '-runtime')
  const started = await service.thought.start({
    text: 'Aynı fiziksel sayfalarda yapısal bir iz bırak.',
    regionMode: 'active',
    maxTicks: 1,
  })
  const completed = await waitCompleted(service.store, started.episode.id)
  assert.equal(completed.regionMode, 'active')
  assert.equal(completed.regionAtlasFingerprint, seeded.atlas.fingerprint)
  assert.match(completed.regionBindingFingerprint, /^[0-9a-f]{32}$/u)

  const page = service.thought.ticks(completed.id, { after: -1, limit: 8, candidateLimit: 8 })
  const telemetry = page.ticks[0].branches[0].adaptiveRegion
  assert.equal(telemetry.pageCompatibilities.length, 56)
  assert.equal(telemetry.pageWriteMask.length, 56)
  assert.equal(telemetry.pageObservations.length, 56)
  assert.equal(telemetry.pageObservations[0].behavior.length, 8)
  assert.equal(Object.hasOwn(telemetry, 'signalRelation'), false)
  assert.equal(page.ticks[0].adaptiveRegion.bindingFingerprint, completed.regionBindingFingerprint)

  await waitFor(() => service.thought.active.size === 0)
  const replay = await service.thought.replay(completed.id)
  assert.equal(replay.matched, true)
  assert.equal(requests.length, 2)
})
