import { describe, expect, it } from 'vitest'
import { createThoughtCatalogFromResolved } from './catalog'
import { createThoughtHead } from './model'
import { createThoughtRuntimeEngine } from './orchestration'
import { encodeThoughtPrompt } from './prompt'
import { minimalFamilyRun } from './test-fixtures'
import type {
  EventGraphV1,
  ThoughtEngineContext,
  ThoughtEngineRequest,
  ThoughtEpisodeState,
  ThoughtResolvedNeuronInput,
  ThoughtTick,
} from './types'

async function harness(text = 'Ada kutuyu taşıdı.') {
  const prompt = await encodeThoughtPrompt(text, { identityRatio: 0.25 })
  const identityLength = prompt.identityLength
  const relationLength = 4096 - identityLength
  const positiveIdentity = prompt.signal.slice(0, identityLength)
  const positiveRelation = prompt.signal.slice(identityLength)
  const negativeIdentity = Float32Array.from(positiveIdentity, (value) => -value)
  const negativeRelation = Float32Array.from(positiveRelation, (value) => -value)
  const neurons: ThoughtResolvedNeuronInput[] = [
    {
      memberKey: 'server-a', kind: 'raw', text: 'ters yüzey',
      identity: negativeIdentity, relation: negativeRelation, identityDigest: 'upstream-digest-a',
      sourceDigest: 'source-a', analysisRunId: 'run-a', neuronId: 'node-a',
    },
    {
      memberKey: 'server-b', kind: 'raw', text,
      identity: positiveIdentity, relation: positiveRelation, identityDigest: 'upstream-digest-b',
      sourceDigest: 'source-b', analysisRunId: 'run-b', neuronId: 'node-b',
    },
  ]
  expect(positiveIdentity).toHaveLength(identityLength)
  expect(positiveRelation).toHaveLength(relationLength)
  const familyRun = minimalFamilyRun({
    ids: neurons.map((neuron) => neuron.memberKey),
    representationSpaceId: prompt.representationSpaceId,
    transitions: [{ from: 'server-a', to: 'server-b', probability: 1 }],
  })
  const catalog = createThoughtCatalogFromResolved(neurons, familyRun)
  const baseModelFingerprint = 'server-base-fingerprint'
  const thoughtHead = createThoughtHead(catalog.model, null, baseModelFingerprint)
  const base = {
    snapshotId: 'snapshot-1', snapshotStatus: 'candidate', selectedBy: 'explicit',
    activeAtStart: false, commitEligible: false,
    artifactManifestHash: 'manifest-1', baseModelFingerprint,
    familyRunFingerprint: familyRun.fingerprint, languageStateFingerprint: null,
    thoughtHeadFingerprint: thoughtHead.fingerprint, thoughtHeadRevision: thoughtHead.revision,
  }
  const request: ThoughtEngineRequest = {
    format: 'thought-engine-request', version: 1,
    episodeId: 'admin-episode', inputText: text, seed: 31, maxTicks: 3,
    strategy: 'seeded-sample', learningMode: 'off', candidateMemberKeys: ['server-b'],
    base,
  }
  const context: ThoughtEngineContext = {
    neurons, familyRun, languageState: null, thoughtHead, baseDiagnostics: { adapter: 'server-shaped' },
  }
  return { request, context, prompt, catalog }
}

describe('runEpisode canlı servis adaptörü', () => {
  it('profileless legacy istekte bölge kipinin yokluğu ile explicit off aynı run/tick digestini üretir', async () => {
    const { request, context } = await harness()
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const baseline = await engine.runEpisode({ ...request, maxTicks: 1 }, context)
    const explicitOff = await engine.runEpisode({
      ...request,
      maxTicks: 1,
      adaptiveRegion: { mode: 'off', atlasFingerprint: 'off-yolunda-yok-sayilir' },
    }, context)
    expect(explicitOff.episode).toEqual(baseline.episode)
    expect(explicitOff.ticks).toEqual(baseline.ticks)
    expect(explicitOff.episode).not.toHaveProperty('adaptiveRegion')
    expect(explicitOff.ticks[0]).not.toHaveProperty('adaptiveRegion')
  })

  it('server-shaped memberKey/I/R satırlarını üyelikle birleştirir ve explicit adayı ilk seçim unionına katar', async () => {
    const { request, context } = await harness()
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const result = await engine.runEpisode({ ...request, maxTicks: 1, strategy: 'greedy' }, context)
    expect(result.episode.branches[0].path[0]).toBe('server-b')
    expect(result.graph.catalog).toEqual(expect.arrayContaining([
      expect.objectContaining({ memberKey: 'server-a', kind: 'included' }),
      expect.objectContaining({ memberKey: 'server-b', kind: 'included' }),
    ]))
    expect(result.graph.prompt.mode).toBe('textlab-hierarchy')
    expect(result.ticks[0].branches[0].beforeSignal).toHaveLength(4096)
  })

  it('episode UUID ve UI base metadata runKey/tick hashini değiştirmez; dil-state fingerprinti değiştirir', async () => {
    const { request, context } = await harness()
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const left = await engine.runEpisode({ ...request, episodeId: 'uuid-left', maxTicks: 1 }, context)
    const right = await engine.runEpisode({
      ...request,
      episodeId: 'uuid-right',
      maxTicks: 1,
      base: { ...request.base, selectedBy: 'active', activeAtStart: true, commitEligible: true },
    }, context)
    expect(left.episode.runKey).toBe(right.episode.runKey)
    expect(left.episode.fingerprint).toBe(right.episode.fingerprint)
    expect(left.ticks).toEqual(right.ticks)

    const languageA = {
      format: 'language-experience-state', fingerprint: 'language-a',
      familyRunFingerprint: context.familyRun.fingerprint,
      representationSpaceId: context.familyRun.representationSpaceId,
      neuronTransitions: [],
    }
    const languageB = { ...languageA, fingerprint: 'language-b' }
    const withLanguageA = await engine.runEpisode({
      ...request, maxTicks: 1, base: { ...request.base, languageStateFingerprint: 'language-a' },
    }, { ...context, languageState: languageA })
    const withLanguageB = await engine.runEpisode({
      ...request, maxTicks: 1, base: { ...request.base, languageStateFingerprint: 'language-b' },
    }, { ...context, languageState: languageB })
    expect(withLanguageA.episode.runKey).not.toBe(withLanguageB.episode.runKey)
  })

  it('verifyOnly replay’i exact tick sayısında, 0 dahil, durdurur', async () => {
    const { request, context } = await harness()
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const full = await engine.runEpisode(request, context)
    const one = await engine.runEpisode({ ...request, replay: { verifyOnly: true, expectedTickCount: 1 } }, context)
    expect(one.episode.tick).toBe(1)
    expect(one.episode.status).toBe('paused')
    expect(one.ticks).toEqual(full.ticks.slice(0, 1))
    const zero = await engine.runEpisode({ ...request, replay: { verifyOnly: true, expectedTickCount: 0 } }, context)
    expect(zero.episode.tick).toBe(0)
    expect(zero.episode.status).toBe('paused')
    expect(zero.ticks).toEqual([])

    const prefix: ThoughtTick[] = []
    const paused64 = await engine.runEpisode({ ...request, maxTicks: 64 }, {
      ...context,
      onTick: (tick) => { prefix.push(tick) },
      shouldPause: () => prefix.length >= 2,
    })
    expect(paused64.episode.tick).toBe(2)
    const verifiedPrefix = await engine.runEpisode({
      ...request, maxTicks: 64, replay: { verifyOnly: true, expectedTickCount: 2 },
    }, context)
    expect(verifiedPrefix.ticks).toEqual(prefix)
  })

  it('64 tick checkpoint chunkı üretir; pause→resume izi kesintisiz çalışmayla aynıdır', async () => {
    const { request, context } = await harness()
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const checkpoints: Array<{ tick: number; state: ThoughtEpisodeState; ticks: ThoughtTick[] }> = []
    let streamedCount = 0
    const long = await engine.runEpisode({ ...request, maxTicks: 64 }, {
      ...context,
      onTick: () => { streamedCount += 1 },
      onCheckpoint: (checkpoint) => { checkpoints.push(checkpoint) },
    })
    expect(streamedCount).toBe(64)
    expect(long.ticks).toEqual([])
    expect(checkpoints.at(-1)?.tick).toBe(64)
    expect(checkpoints.at(-1)?.ticks).toHaveLength(64)
    expect(checkpoints.at(-1)?.state.status).toBe('stopped')

    const uninterrupted = await engine.runEpisode(request, context)
    const streamed: ThoughtTick[] = []
    let pausedState: ThoughtEpisodeState | null = null
    const paused = await engine.runEpisode(request, {
      ...context,
      onTick: (tick) => { streamed.push(tick) },
      shouldPause: () => streamed.length >= 1,
      onCheckpoint: (checkpoint) => { pausedState = checkpoint.state },
    })
    expect(paused.episode.status).toBe('paused')
    expect(pausedState).not.toBeNull()
    const resumed = await engine.runEpisode({ ...request, resume: pausedState }, context)
    expect([...streamed, ...resumed.ticks]).toEqual(uninterrupted.ticks)
    expect(resumed.episode.fingerprint).toBe(uninterrupted.episode.fingerprint)
  })

  it('expected graph tek başına kredi vermez; geçiş aliası teacherEvidence doğrulanınca delta üretir', async () => {
    const { request, context } = await harness()
    const expected: EventGraphV1 = {
      format: 'event-graph-v1', actor: null, action: null, patient: null, instrument: null,
      operators: [], negated: false, goals: [],
    }
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const traceOnly = await engine.runEpisode({ ...request, learningMode: 'proposal', expectedOutcome: expected }, context)
    expect(traceOnly.headDelta).toBeUndefined()
    const credited = await engine.runEpisode({
      ...request,
      learningMode: 'proposal',
      expectedOutcome: expected,
      teacherEvidence: { kind: 'human', confidence: 0.95, sourceGroupId: 'human-source' },
    }, context)
    expect(credited.headDelta?.stats?.confidence).toBe(0.95)
  })
})
