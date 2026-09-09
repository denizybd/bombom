import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer as createViteServer } from 'vite'

async function loadRunner(t) {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  return Promise.all([
    vite.ssrLoadModule('/runtime/structural-benchmark-runner.mjs'),
    vite.ssrLoadModule('/src/thought-runtime/prompt.ts'),
    vite.ssrLoadModule('/server/structural-stop-gate.mjs'),
  ])
}

function benchmarkBase(encoding) {
  const identity = encoding.signal.slice(0, encoding.identityLength)
  const relation = encoding.signal.slice(encoding.identityLength)
  const memberKeys = [
    'benchmark-neuron-a',
    'benchmark-neuron-b',
    'benchmark-neuron-c',
    'benchmark-neuron-d',
  ]
  const familyRun = {
    format: 'noron-aileleri-run',
    version: 1,
    id: 'benchmark-family-run',
    fingerprint: 'benchmark-family-run-fingerprint',
    representationSpaceId: encoding.representationSpaceId,
    behaviorSpaceId: 'benchmark-behavior-space',
    profiles: [],
    families: [
      { id: 'benchmark-event-family', region: 'event' },
      { id: 'benchmark-control-family', region: 'control' },
    ],
    memberships: [
      ...memberKeys.slice(0, 2).map((memberKey) => ({ memberKey, familyId: 'benchmark-event-family', weight: 1 })),
      ...memberKeys.slice(2).map((memberKey) => ({ memberKey, familyId: 'benchmark-control-family', weight: 0.5 })),
    ],
    neuronTransitions: [
      { fromMemberKey: memberKeys[0], toMemberKey: memberKeys[1], probability: 1 },
      { fromMemberKey: memberKeys[2], toMemberKey: memberKeys[3], probability: 0.25 },
    ],
    familyTransitions: [
      {
        id: 'event-self', fromFamilyId: 'benchmark-event-family', toFamilyId: 'benchmark-event-family',
        priorEvidence: 1, experienceEvidence: 0, inhibitionEvidence: 0,
        sourceCount: 1, supportCount: 1, probability: 1,
      },
      {
        id: 'control-self', fromFamilyId: 'benchmark-control-family', toFamilyId: 'benchmark-control-family',
        priorEvidence: 0.25, experienceEvidence: 0, inhibitionEvidence: 0,
        sourceCount: 1, supportCount: 1, probability: 0.25,
      },
    ],
    utilityReports: [
      { familyId: 'benchmark-event-family', mode: 'source-holdout-diagnostic', status: 'validated' },
      { familyId: 'benchmark-control-family', mode: 'source-holdout-diagnostic', status: 'validated' },
    ],
    coalition: {
      tick: 0,
      activeFamilyCount: 2,
      coalitions: [
        {
          id: 'target-event-coalition', eventMemberKey: memberKeys[0], eventFamilyId: 'benchmark-event-family',
          familyActivations: [{ familyId: 'benchmark-event-family', activation: 1 }],
          bindings: [{ role: 'agent-hypothesis', memberKey: memberKeys[1], roleFamilyId: 'benchmark-event-family' }],
        },
        {
          id: 'target-nested-reference-carrier', eventMemberKey: memberKeys[2], eventFamilyId: 'benchmark-control-family',
          familyActivations: [{ familyId: 'benchmark-event-family', activation: 0.8 }],
          bindings: [{ role: 'patient-hypothesis', memberKey: memberKeys[0], roleFamilyId: 'benchmark-event-family' }],
        },
        {
          id: 'pure-control-coalition-a', eventMemberKey: memberKeys[2], eventFamilyId: 'benchmark-control-family',
          familyActivations: [{ familyId: 'benchmark-control-family', activation: 0.6 }],
          bindings: [{ role: 'agent-hypothesis', memberKey: memberKeys[3], roleFamilyId: 'benchmark-control-family' }],
        },
        {
          id: 'pure-control-coalition-b', eventMemberKey: memberKeys[3], eventFamilyId: 'benchmark-control-family',
          familyActivations: [{ familyId: 'benchmark-control-family', activation: 0.5 }],
          bindings: [{ role: 'instrument-hypothesis', memberKey: memberKeys[2], roleFamilyId: 'benchmark-control-family' }],
        },
      ],
      unmatchedReason: null,
    },
    routingPreview: null,
    warnings: [],
    stats: {},
  }
  const thoughtHead = {
    format: 'bombom-thought-head',
    version: 1,
    revision: 0,
    baseModelFingerprint: 'benchmark-base-model',
    totalTicks: 0,
    neuronEdges: [{
      fromMemberKey: memberKeys[0], toMemberKey: memberKeys[1],
      excitation: 1, inhibition: 0, trustMass: 1, updateCount: 1,
    }],
    contextPatches: [],
    nursery: [],
    births: [{
      format: 'born-neuron-v1',
      id: 'benchmark-born-neuron',
      identity: [],
      relation: [],
      identityDigest: 'benchmark-born-digest',
      familyMemberships: [
        { familyId: 'benchmark-event-family', weight: 0.6 },
        { familyId: 'benchmark-control-family', weight: 0.4 },
      ],
      sourceGroupIds: ['benchmark-source'],
      trustMass: 1,
      bornAtTick: 1,
      nurseryId: 'benchmark-nursery',
    }],
    fingerprint: '',
  }
  return {
    reference: {
      snapshotId: 'benchmark-snapshot',
      snapshotStatus: 'candidate',
      activeAtStart: false,
      artifactManifestHash: 'benchmark-artifact-manifest',
      baseModelFingerprint: 'benchmark-base-model',
    },
    familyRun,
    languageState: null,
    thoughtHead,
    neurons: memberKeys.map((memberKey, index) => ({
      memberKey,
      kind: 'raw',
      text: ['Ada kutuyu çekti.', 'Bora kutuyu taşıdı.', 'Cem kapıyı açtı.', 'Duru ipi bıraktı.'][index],
      identity: new Float32Array(identity),
      relation: new Float32Array(relation),
    })),
    layout: {
      side: 64,
      cellCount: 4_096,
      identityLength: encoding.identityLength,
      relationStart: encoding.identityLength,
      relationLength: 4_096 - encoding.identityLength,
    },
    diagnostics: {},
  }
}

function emptyGraph() {
  return {
    format: 'event-graph-v1', actor: null, action: null, patient: null,
    instrument: null, operators: [], negated: false, goals: [],
  }
}

function fakeThoughtEngine() {
  const calls = []
  return {
    calls,
    async runEpisode(request, context) {
      const variant = context.baseDiagnostics.benchmarkVariant
      calls.push({ variant, replay: request.replay?.expectedTickCount ?? null, maxTicks: request.maxTicks, signalOverride: Boolean(context.benchmarkInitialSignal) })
      const destructive = ['h0-only', 'recurrence-off', 'shuffled-edge', 'target-family-ablation', 'target-edge-ablation'].includes(variant)
      const noveltyBirth = variant === 'repeated-novelty-birth' && request.base.thoughtHeadRevision >= 3
      return {
        episode: { status: request.replay ? 'paused' : 'stopped', temporaryBirths: [], branches: [] },
        ticks: [],
        graph: { eventGraph: destructive ? emptyGraph() : structuredClone(request.expectedOutcome) },
        ...(variant === 'repeated-novelty-birth' ? {
          headDelta: {
            neuronEdges: [],
            ...(noveltyBirth ? {
              births: [{
                format: 'born-neuron-v1',
                id: `born-${request.episodeId}`,
                identity: [],
                relation: [],
                identityDigest: 'benchmark-birth',
                familyMemberships: [],
                sourceGroupIds: [request.outcomeEvidence.sourceGroupId],
                trustMass: 1,
                bornAtTick: 256,
                nurseryId: `nursery-${request.episodeId}`,
              }],
            } : {}),
            stats: { totalTicks: request.maxTicks },
          },
        } : {}),
      }
    },
  }
}

function selectiveThoughtEngine(passingSeeds = []) {
  const acceptedSeeds = new Set(passingSeeds)
  const calls = []
  return {
    calls,
    async runEpisode(request, context) {
      calls.push({ seed: request.seed, variant: context.baseDiagnostics.benchmarkVariant })
      return {
        episode: { status: 'stopped', temporaryBirths: [], branches: [] },
        ticks: [],
        graph: {
          eventGraph: acceptedSeeds.has(request.seed)
            ? structuredClone(request.expectedOutcome)
            : emptyGraph(),
        },
      }
    },
  }
}

function artifactStore(snapshot) {
  const artifacts = new Map()
  return {
    artifacts,
    getSnapshot(id) { return id === snapshot.id ? structuredClone(snapshot) : undefined },
    getArtifact(id) { return structuredClone(artifacts.get(id)) },
    listArtifacts() { return [...artifacts.values()].map(({ payload: _payload, ...metadata }) => structuredClone(metadata)) },
    createClaimedArtifact(jobId, owner, attempt, input) {
      assert.equal(jobId, 'benchmark-job')
      assert.equal(owner, 'benchmark-owner')
      assert.equal(attempt, 1)
      const existing = [...artifacts.values()].find((item) => item.kind === input.kind && item.contentHash === input.contentHash)
      if (existing) return structuredClone(existing)
      const artifact = { id: `artifact-${artifacts.size + 1}`, createdAt: '2026-08-30T00:00:00.000Z', jobId, ...structuredClone(input) }
      artifacts.set(artifact.id, artifact)
      return structuredClone(artifact)
    },
  }
}

test('production runner derives every gate metric, resumes durable chunks, and persists one final claim-fenced report', { timeout: 60_000 }, async (t) => {
  const [{
    createIndependentStructuralBenchmarkRunner,
    pairedSignedAblationDrops,
    signedSourceGroupInterval,
  }, { encodeThoughtPrompt }, { evaluateIndependentStructuralBenchmark }] = await loadRunner(t)
  const signedRows = pairedSignedAblationDrops(
    [0, 1, 2, 3, 4].map((seed) => ({ seed, caseId: 'signed-case', sourceGroupId: 'signed-group', score: 0.2 })),
    [0, 1, 2, 3, 4].map((seed) => ({ seed, caseId: 'signed-case', sourceGroupId: 'signed-group', score: 0.8 })),
  )
  assert.ok(signedRows.every((row) => Math.abs(row.score + 0.6) < 1e-12), 'negative ablation effects must not be clipped')
  const signedInterval = signedSourceGroupInterval(signedRows, 1_000)
  assert.ok(Math.abs(signedInterval.mean + 0.6) < 1e-12)
  assert.ok(signedInterval.high < 0)
  const encoding = await encodeThoughtPrompt('Ada kutuyu kancayla taşıdı.', { allowFlatFallback: true })
  const base = benchmarkBase(encoding)
  const snapshot = { id: base.reference.snapshotId, inputManifestHash: 'benchmark-input-manifest', artifactIds: [] }
  const store = artifactStore(snapshot)
  const thoughtEngine = fakeThoughtEngine()
  const runner = createIndependentStructuralBenchmarkRunner({
    thoughtEngine,
    resolveBase: () => structuredClone(base),
    maxTicks: 1,
    bootstrapSamples: 1_000,
  })
  const request = {
    store,
    snapshot,
    snapshotId: snapshot.id,
    signal: new AbortController().signal,
    jobId: 'benchmark-job',
    jobClaim: { owner: 'benchmark-owner', attempt: 1 },
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }
  let interrupted = false
  await assert.rejects(runner({
    ...request,
    checkpoint: async (label) => {
      if (!interrupted && /^teacher-feed:benchmark:full:seed:.*:persisted$/u.test(label)) {
        interrupted = true
        throw Object.assign(new Error('durable benchmark pause'), { code: 'JOB_PAUSED' })
      }
    },
  }), { code: 'JOB_PAUSED' })
  const completedChunkCalls = thoughtEngine.calls.length
  assert.equal(completedChunkCalls, 88, 'one complete seed chunk is durable before the interruption')
  assert.equal(store.artifacts.size, 1)

  const first = await runner(request)
  assert.match(first.artifactId, /^artifact-/u)
  assert.equal(first.report.usesOllama, false)
  assert.equal(first.report.evidence.episodeCount, 4_160)
  assert.equal(first.report.evidence.intervals.baselineAdvantage.seedCount, 5)
  assert.equal(first.report.evidence.intervals.baselineAdvantage.confidence, 0.95)
  assert.ok(thoughtEngine.calls.every((call) => call.replay === null), 'recurrence-off must run real ticks, not a zero-tick replay')
  assert.ok(thoughtEngine.calls.some((call) => call.variant === 'recurrence-off' && call.maxTicks === 1))
  assert.ok(thoughtEngine.calls.some((call) => call.signalOverride))
  assert.ok(thoughtEngine.calls.some((call) => call.variant === 'repeated-novelty-birth'))
  assert.ok(thoughtEngine.calls.some((call) => call.variant === 'known-repeated-negative'))
  assert.equal(thoughtEngine.calls.length, 4_160, 'the persisted seed chunk must not be recomputed on resume')
  assert.equal(first.report.evidence.ablationTargetSelection, 'snapshot-model-only-no-evaluation-data-v2')
  assert.equal(first.report.evidence.edgeShamProtocol, 'snapshot-digest-random-nontarget-edge-v1')
  assert.notEqual(first.report.evidence.shamEdge, first.report.evidence.targetEdge)
  assert.equal(first.report.evidence.targetFamilyId, 'benchmark-event-family')
  assert.equal(first.report.evidence.familyShamProtocol, 'deterministic-model-only-exact-reference-footprint-mask-v1')
  assert.equal(first.report.evidence.shamFamilyFootprintMatched, true)
  assert.deepEqual(first.report.evidence.shamFamilyFootprint, first.report.evidence.targetFamilyFootprint)
  assert.deepEqual(first.report.evidence.targetFamilyFootprint, {
    memberships: 2,
    familyTransitions: 1,
    utilityReports: 1,
    coalitions: 1,
    coalitionActivations: 1,
    coalitionBindings: 1,
    birthMemberships: 1,
    total: 8,
  })
  assert.match(first.report.evidence.shamFamilyMaskDigest, /^[a-f0-9]{64}$/u)
  assert.ok(thoughtEngine.calls.some((call) => call.variant === 'sham-family-ablation'))
  assert.equal(first.report.evidence.birthPrecisionDefinition, 'TP/(TP+FP)')
  assert.deepEqual(first.report.evidence.birthConfusion, { truePositive: 25, falsePositive: 0, falseNegative: 0, trueNegative: 25 })
  assert.equal(first.report.evidence.intervals.repeatedNoveltyBirthSensitivity.mean, 1)
  const evaluated = evaluateIndependentStructuralBenchmark(first.report, {
    expectedSnapshotId: snapshot.id,
    expectedInputManifestHash: snapshot.inputManifestHash,
  })
  assert.equal(evaluated.passed, true, JSON.stringify({
    failed: evaluated.checks.filter((check) => !check.passed),
    metrics: first.report.metrics,
    intervals: first.report.evidence.intervals,
    minimumPerturbationCosine: first.report.evidence.minimumPerturbationCosine,
  }))

  const callCount = thoughtEngine.calls.length
  const replay = await runner(request)
  assert.equal(replay.artifactId, first.artifactId)
  assert.equal(thoughtEngine.calls.length, callCount, 'immutable exact-snapshot report must be reused instead of rerunning episodes')
  assert.equal(store.artifacts.size, 96, '95 bounded observation chunks plus one immutable final report are retained')
})

test('low one-seed admission evidence skips the expensive full benchmark and is idempotent', { timeout: 30_000 }, async (t) => {
  const [{
    createIndependentStructuralBenchmarkRunner,
  }, { encodeThoughtPrompt }, {
    evaluateIndependentStructuralBenchmarkScreen,
    INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_ARTIFACT_KIND,
  }] = await loadRunner(t)
  const encoding = await encodeThoughtPrompt('Ada kutuyu kancayla taşıdı.', { allowFlatFallback: true })
  const base = benchmarkBase(encoding)
  const snapshot = { id: base.reference.snapshotId, inputManifestHash: 'benchmark-input-manifest', artifactIds: [] }
  const store = artifactStore(snapshot)
  const thoughtEngine = selectiveThoughtEngine()
  const runner = createIndependentStructuralBenchmarkRunner({
    thoughtEngine,
    resolveBase: () => structuredClone(base),
    maxTicks: 1,
    bootstrapSamples: 1_000,
  })
  const request = {
    store,
    snapshot,
    snapshotId: snapshot.id,
    signal: new AbortController().signal,
    jobId: 'benchmark-job',
    jobClaim: { owner: 'benchmark-owner', attempt: 1 },
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }
  const first = await runner(request)
  assert.equal(first.report.completion, 'diagnostic-screened-out')
  assert.equal(first.report.evidence.episodeCount, 88)
  assert.equal(thoughtEngine.calls.length, 88)
  assert.equal([...store.artifacts.values()].filter((artifact) => artifact.kind === INDEPENDENT_STRUCTURAL_BENCHMARK_SCREEN_ARTIFACT_KIND).length, 1)
  const evaluated = evaluateIndependentStructuralBenchmarkScreen(first.report, {
    expectedSnapshotId: snapshot.id,
    expectedInputManifestHash: snapshot.inputManifestHash,
  })
  assert.equal(evaluated.status, 'screened-out')
  assert.equal(evaluated.passed, false)

  const replay = await runner(request)
  assert.equal(replay.artifactId, first.artifactId)
  assert.equal(thoughtEngine.calls.length, 88, 'valid immutable screen evidence must be reused')
})

test('five-seed full coverage below CI gate persists a decisive early failure and skips every diagnostic variant', { timeout: 30_000 }, async (t) => {
  const [{
    createIndependentStructuralBenchmarkRunner,
    STRUCTURAL_BENCHMARK_SEEDS,
  }, { encodeThoughtPrompt }, {
    evaluateIndependentStructuralBenchmarkEarlyFailure,
    INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_ARTIFACT_KIND,
  }] = await loadRunner(t)
  const encoding = await encodeThoughtPrompt('Ada kutuyu kancayla taşıdı.', { allowFlatFallback: true })
  const base = benchmarkBase(encoding)
  const snapshot = { id: base.reference.snapshotId, inputManifestHash: 'benchmark-input-manifest', artifactIds: [] }
  const store = artifactStore(snapshot)
  const thoughtEngine = selectiveThoughtEngine([STRUCTURAL_BENCHMARK_SEEDS[0]])
  const runner = createIndependentStructuralBenchmarkRunner({
    thoughtEngine,
    resolveBase: () => structuredClone(base),
    maxTicks: 1,
    bootstrapSamples: 1_000,
  })
  const request = {
    store,
    snapshot,
    snapshotId: snapshot.id,
    signal: new AbortController().signal,
    jobId: 'benchmark-job',
    jobClaim: { owner: 'benchmark-owner', attempt: 1 },
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }
  const first = await runner(request)
  assert.equal(first.report.completion, 'decisive-early-failure')
  assert.equal(first.report.evidence.episodeCount, 440)
  assert.equal(first.report.evidence.skippedEpisodeCount, 3_720)
  assert.equal(thoughtEngine.calls.length, 440)
  assert.deepEqual(new Set(thoughtEngine.calls.map((call) => call.variant)), new Set(['full']))
  assert.equal([...store.artifacts.values()].filter((artifact) => artifact.kind === INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_ARTIFACT_KIND).length, 1)
  const evaluated = evaluateIndependentStructuralBenchmarkEarlyFailure(first.report, {
    expectedSnapshotId: snapshot.id,
    expectedInputManifestHash: snapshot.inputManifestHash,
  })
  assert.equal(evaluated.status, 'failed-early')
  assert.equal(evaluated.conclusive, true)
  assert.equal(evaluated.passed, false)

  const replay = await runner(request)
  assert.equal(replay.artifactId, first.artifactId)
  assert.equal(thoughtEngine.calls.length, 440, 'decisive immutable failure must be reused')
})

test('benchmark resource gates pause before hydration and reject a 100K dense fallback', async (t) => {
  const [{ createIndependentStructuralBenchmarkRunner }] = await loadRunner(t)
  const snapshot = { id: 'large-snapshot', inputManifestHash: 'large-manifest', artifactIds: ['a', 'b'] }
  const request = {
    snapshot,
    snapshotId: snapshot.id,
    signal: new AbortController().signal,
    jobId: 'benchmark-job',
    jobClaim: { owner: 'benchmark-owner', attempt: 1 },
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }
  const preflightStore = artifactStore(snapshot)
  preflightStore.getArtifactMetadata = (id) => ({
    id,
    kind: id === 'a' ? 'text-lab-analysis' : 'family-run',
    payloadBytes: 256 * 1024 * 1024,
  })
  preflightStore.getConfig = () => ({
    dataDirectory: process.cwd(),
    budgets: { ramMb: 512, diskMb: 4_096 },
    resourcePolicy: {
      enabled: true,
      ramSoftRatio: 0.25,
      ramSpillRatio: 0.5,
      ramHardRatio: 0.75,
      minHostFreeMb: 256,
      minDiskFreeStartMb: 1_024,
      minDiskFreeWarnMb: 2_048,
      walReserveMb: 64,
    },
  })
  preflightStore.approximateStorageBytes = () => 0
  let resolveCalls = 0
  const preflightRunner = createIndependentStructuralBenchmarkRunner({
    thoughtEngine: fakeThoughtEngine(),
    resolveBase: () => { resolveCalls += 1 },
    maxTicks: 1,
    bootstrapSamples: 1_000,
  })
  await assert.rejects(
    preflightRunner({ ...request, store: preflightStore }),
    (error) => error?.code === 'JOB_PAUSED' && error?.reasonCode === 'STRUCTURAL_BENCHMARK_RESOURCE_WAIT',
  )
  assert.equal(resolveCalls, 0, 'metadata gate must precede dense artifact hydration')

  const denseStore = artifactStore(snapshot)
  const denseRunner = createIndependentStructuralBenchmarkRunner({
    thoughtEngine: fakeThoughtEngine(),
    resolveBase: () => ({
      reference: { snapshotId: snapshot.id },
      layout: { side: 64, cellCount: 4_096 },
      neurons: Array.from({ length: 100_000 }, () => ({})),
    }),
    maxTicks: 1,
    bootstrapSamples: 1_000,
  })
  await assert.rejects(
    denseRunner({ ...request, store: denseStore }),
    (error) => error?.code === 'JOB_PAUSED' && error?.reasonCode === 'STRUCTURAL_BENCHMARK_DENSE_BASE_UNSUPPORTED' &&
      error?.details?.neuronCount === 100_000,
  )
})
