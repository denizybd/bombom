import { describe, expect, it } from 'vitest'
import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_PAGE_COUNT,
  createPlasticRegionAtlas,
  relationIndexFromPage,
  type PlasticRegionAtlasV1,
} from '../adaptive-regions'
import { ADAPTIVE_TEXT_LAB_PROFILE_ID } from '../text-lab/domain/config'
import {
  resolveThoughtAdaptiveRegionContext,
  evaluateThoughtAdaptiveCompatibility,
} from './adaptive-regions'
import { createThoughtRuntimeConfig } from './config'
import { createThoughtCatalogFromResolved } from './catalog'
import {
  cloneThoughtEpisodeState,
  resumeThoughtEpisode,
  startThoughtEpisode,
  stepThoughtEpisode,
  withEpisodeStatus,
} from './engine'
import { composeNeuronRepresentation, transformThoughtSignal } from './math'
import { createThoughtHead, createThoughtLayout, createThoughtModel, createThoughtNeuron } from './model'
import { createThoughtRuntimeEngine } from './orchestration'
import { encodeThoughtPrompt } from './prompt'
import { createThoughtRuntimeIndex } from './routing'
import { minimalFamilyRun } from './test-fixtures'
import type {
  ThoughtAdaptiveRegionContextV1,
  ThoughtHead,
  ThoughtEngineContext,
  ThoughtEngineRequest,
  ThoughtModel,
  ThoughtRuntimeConfig,
  ThoughtResolvedNeuronInput,
  ThoughtVector,
} from './types'

const REPRESENTATION_SPACE_ID = 'RS-adaptive-thought-test'
const ADAPTIVE_LAYOUT = createThoughtLayout(512)

function relationWithPages(values: Readonly<Record<number, number>>): Float32Array {
  const relation = new Float32Array(ADAPTIVE_LAYOUT.relationLength)
  for (const [rawPageIndex, value] of Object.entries(values)) {
    const pageIndex = Number(rawPageIndex)
    for (let cellIndex = 0; cellIndex < ADAPTIVE_CELLS_PER_PAGE; cellIndex += 1) {
      relation[relationIndexFromPage(pageIndex, cellIndex)] = value
    }
  }
  return relation
}

function adaptiveAtlas(
  override: {
    revision?: number
    parentFingerprint?: string | null
    holdoutUtility?: number
    representationSpaceId?: string
  } = {},
): PlasticRegionAtlasV1 {
  return createPlasticRegionAtlas({
    representationSpaceId: override.representationSpaceId ?? REPRESENTATION_SPACE_ID,
    revision: override.revision ?? 0,
    parentFingerprint: override.parentFingerprint ?? null,
    behaviorDimension: 8,
    maximumMembershipsPerPage: 3,
    regions: [{
      id: 'region-page-zero',
      lifecycle: 'active',
      origin: 'discovery',
      bornRevision: 0,
      retiredRevision: null,
      parentRegionIds: [],
      prototype: { dimension: 8, values: Array(8).fill(0) },
      evidence: {
        observationCount: 4,
        sourceGroupIds: ['source-a', 'source-b', 'source-c', 'source-d'],
        confidenceMass: 4,
        cohesion: 1,
        holdoutUtility: override.holdoutUtility ?? 0,
        shamUtility: 0,
      },
    }],
    pages: Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_, pageIndex) => pageIndex === 0
      ? { pageIndex, memberships: [{ regionId: 'region-page-zero', weight: 1 }], freeMass: 0 }
      : { pageIndex, memberships: [], freeMass: 1 }),
  })
}

function adaptiveFixture(maximumBranches = 1): {
  model: ThoughtModel
  head: ThoughtHead
  config: ThoughtRuntimeConfig
  signal: ThoughtVector
} {
  const config = createThoughtRuntimeConfig({ maximumBranches })
  const sharedRelation = relationWithPages({ 0: 0.5, 1: 0.5 })
  const neuron = (id: string, identityIndex: number) => {
    const identity = new Float32Array(ADAPTIVE_LAYOUT.identityLength)
    identity[identityIndex] = 1
    return createThoughtNeuron({
      id,
      identity,
      relation: sharedRelation,
      familyMemberships: [{ familyId: 'family-adaptive', weight: 1 }],
      provenance: { kind: 'seed', sourceId: `source-${id}`, surface: id },
    }, ADAPTIVE_LAYOUT, config)
  }
  const a = neuron('a', 0)
  const b = neuron('b', 1)
  const edge = (toNeuronId: string) => ({
    toNeuronId,
    priorEvidence: 1,
    excitationEvidence: 0,
    inhibitionEvidence: 0,
    trustMass: 0,
    updateCount: 0,
    independentSourceIds: [] as string[],
  })
  const model = createThoughtModel({
    layout: ADAPTIVE_LAYOUT,
    neurons: [a, b],
    families: [{ id: 'family-adaptive' }],
    habitRows: [
      { fromNeuronId: 'a', edges: [edge('b')] },
      { fromNeuronId: 'b', edges: [edge('a')] },
    ],
  }, config)
  const signal = new Float32Array(4096)
  signal[0] = 1
  signal[1] = maximumBranches > 1 ? 1 : 0
  for (const pageIndex of [0, 1]) {
    for (let cellIndex = 0; cellIndex < ADAPTIVE_CELLS_PER_PAGE; cellIndex += 1) {
      signal[ADAPTIVE_LAYOUT.relationStart + relationIndexFromPage(pageIndex, cellIndex)] = 1
    }
  }
  return { model, head: createThoughtHead(model), config, signal }
}

function regionContext(
  mode: 'shadow' | 'proposal' | 'active',
  fixture: Pick<ReturnType<typeof adaptiveFixture>, 'model'>,
  atlas = adaptiveAtlas(),
): ThoughtAdaptiveRegionContextV1 {
  return resolveThoughtAdaptiveRegionContext({
    request: {
      mode,
      atlasFingerprint: atlas.fingerprint,
      compatibilityBlend: mode === 'active' ? 0.25 : undefined,
      minimumTrace: 0.02,
    },
    atlas,
    layout: fixture.model.layout,
    representationSpaceId: REPRESENTATION_SPACE_ID,
    representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
    sourceGroupId: 'test-source-group',
  })!
}

function start(
  fixture: ReturnType<typeof adaptiveFixture>,
  context?: ThoughtAdaptiveRegionContextV1,
  maximumTicks = 2,
) {
  const index = createThoughtRuntimeIndex(fixture.model, fixture.head, fixture.config, 71)
  const episode = startThoughtEpisode(fixture.model, fixture.head, index, fixture.signal, {
    episodeId: 'adaptive-episode',
    runKey: 'adaptive-semantic-run',
    seed: 71,
    maximumTicks,
    initialNeuronIds: ['a'],
    ...(context ? { adaptiveRegionContext: context } : {}),
  }, fixture.config)
  return { index, episode }
}

describe('thought-runtime uyarlanabilir bölge entegrasyonu', () => {
  it('eksik kip ile explicit off kipini eski episode/tick byte yapısında birebir eş tutar', () => {
    const fixture = adaptiveFixture()
    const resolvedOff = resolveThoughtAdaptiveRegionContext({
      request: { mode: 'off', atlasFingerprint: 'ignored-in-off' },
      layout: fixture.model.layout,
      representationSpaceId: REPRESENTATION_SPACE_ID,
      sourceGroupId: 'off-source',
    })
    expect(resolvedOff).toBeNull()
    const baseline = start(fixture)
    const explicit = start(fixture, resolvedOff ?? undefined)
    expect(explicit.episode).toEqual(baseline.episode)
    expect(explicit.episode).not.toHaveProperty('adaptiveRegion')

    const baselineStep = stepThoughtEpisode(
      baseline.episode, fixture.model, fixture.head, baseline.index, fixture.config,
    )
    const explicitStep = stepThoughtEpisode(
      explicit.episode, fixture.model, fixture.head, explicit.index, fixture.config,
      { adaptiveRegionContext: resolvedOff ?? undefined },
    )
    expect(explicitStep).toEqual(baselineStep)
    expect(explicitStep.tick).not.toHaveProperty('adaptiveRegion')
    expect(explicitStep.tick.branches.every((branch) => !('adaptiveRegion' in branch))).toBe(true)
  })

  it.each(['shadow', 'proposal'] as const)('%s yalnız gözler; route, sinyal ve hızlı R izini değiştirmez', (mode) => {
    const fixture = adaptiveFixture()
    const context = regionContext(mode, fixture)
    const baseline = start(fixture)
    const observed = start(fixture, context)
    expect(observed.episode.adaptiveRegion?.mode).toBe(mode)
    expect(observed.episode.branches).toEqual(baseline.episode.branches)

    const baselineStep = stepThoughtEpisode(
      baseline.episode, fixture.model, fixture.head, baseline.index, fixture.config,
    )
    const observedStep = stepThoughtEpisode(
      observed.episode, fixture.model, fixture.head, observed.index, fixture.config,
      { adaptiveRegionContext: context },
    )
    expect(observedStep.episode.branches).toEqual(baselineStep.episode.branches)
    const [{ adaptiveRegionCandidates: _candidateTelemetry, ...observedRoute }] = observedStep.routes
    expect(observedRoute).toEqual(baselineStep.routes[0])
    const { adaptiveRegion: telemetry, ...observedTrace } = observedStep.tick.branches[0]
    const { adaptiveRegion: _baselineTelemetry, ...baselineTrace } = baselineStep.tick.branches[0]
    expect(observedTrace).toEqual(baselineTrace)
    expect(telemetry).toEqual(expect.objectContaining({ mode, activeApplied: false }))
    expect(telemetry?.pageCompatibilities).toHaveLength(56)
    expect(telemetry?.pageWeights).toHaveLength(56)
    expect(telemetry?.pageWriteMask).toHaveLength(56)
    expect(telemetry?.pageObservations).toHaveLength(56)
    expect(telemetry?.pageObservations.every((observation) => observation.behavior.length === 8)).toBe(true)
    expect(telemetry?.candidateCompatibilities.length).toBeLessThanOrEqual(8)
  })

  it('active kipte uyumu bounded harmanlar, alpha=uyum bırakır ve hızlı R yazısını sayfa kapısıyla sınırlar', () => {
    const fixture = adaptiveFixture()
    const context = regionContext('active', fixture)
    const { index, episode } = start(fixture, context)
    const current = index.neuronById.get(episode.branches[0].currentNeuronId)!
    const representation = composeNeuronRepresentation(current.identity, current.relation, fixture.model.layout)
    const compatibility = evaluateThoughtAdaptiveCompatibility(
      fixture.signal,
      representation,
      fixture.model.layout,
      context,
    )
    const transformed = transformThoughtSignal(
      fixture.signal,
      representation,
      fixture.config,
      compatibility.effectiveCompatibility,
    )
    expect(transformed.alpha).toBe(transformed.compatibility)
    expect(transformed.compatibility).toBe(compatibility.blendedCompatibility)

    const stepped = stepThoughtEpisode(
      episode, fixture.model, fixture.head, index, fixture.config,
      { adaptiveRegionContext: context },
    )
    const telemetry = stepped.tick.branches[0].adaptiveRegion!
    expect(telemetry.activeApplied).toBe(true)
    expect(telemetry.effectiveCompatibility).toBe(telemetry.blendedCompatibility)
    expect(telemetry.pageWriteMask[0]).toBeGreaterThan(telemetry.pageWriteMask[1])
    const patch = stepped.episode.branches[0].fastRelationPatches.find((entry) => entry.neuronId === current.id)!
    const meanAbsolute = (pageIndex: number) => Array.from({ length: ADAPTIVE_CELLS_PER_PAGE }, (_, cellIndex) =>
      Math.abs(patch.delta[relationIndexFromPage(pageIndex, cellIndex)]))
      .reduce((sum, value) => sum + value, 0) / ADAPTIVE_CELLS_PER_PAGE
    expect(meanAbsolute(0)).toBeGreaterThan(meanAbsolute(1))
    expect(Math.min(...telemetry.pageWriteMask)).toBeGreaterThanOrEqual(context.binding.minimumTrace - 1e-7)
  })

  it('canlı aday seçiminde aynı sayfadaki bölgeleri öğrenilmiş 8D prototiple ayırır', () => {
    const fixture = adaptiveFixture()
    const expectedBehavior = [1, 0.5, 1, 0.25, 0, 1, 0.5, 0]
    const region = (id: string, prototype: readonly number[]) => ({
      id,
      lifecycle: 'active' as const,
      origin: 'discovery' as const,
      bornRevision: 1,
      retiredRevision: null,
      parentRegionIds: [],
      prototype: { dimension: 8, values: prototype },
      evidence: {
        observationCount: 4,
        sourceGroupIds: ['source-a', 'source-b', 'source-c', 'source-d'],
        confidenceMass: 4,
        cohesion: 1,
        holdoutUtility: 0.1,
        shamUtility: 0,
      },
    })
    const atlas = createPlasticRegionAtlas({
      representationSpaceId: REPRESENTATION_SPACE_ID,
      revision: 1,
      parentFingerprint: 'prototype-routing-parent',
      behaviorDimension: 8,
      regions: [
        region('prototype-match', expectedBehavior),
        region('prototype-opposite', expectedBehavior.map((value) => -value)),
      ],
      pages: Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_, pageIndex) => pageIndex < 2
        ? {
            pageIndex,
            memberships: [
              { regionId: 'prototype-match', weight: 0.5 },
              { regionId: 'prototype-opposite', weight: 0.5 },
            ],
            freeMass: 0,
          }
        : { pageIndex, memberships: [], freeMass: 1 }),
    })
    const context = regionContext('active', fixture, atlas)
    const { index, episode } = start(fixture, context)
    const current = index.neuronById.get(episode.branches[0].currentNeuronId)!
    const representation = composeNeuronRepresentation(current.identity, current.relation, fixture.model.layout)
    const compatibility = evaluateThoughtAdaptiveCompatibility(
      fixture.signal,
      representation,
      fixture.model.layout,
      context,
    )
    expect(compatibility.gates.gates.map((gate) => gate.regionId)).toEqual([
      'prototype-match',
      'prototype-opposite',
    ])
    expect(compatibility.gates.gates[0].prototypeCompatibility).toBeGreaterThan(0.99)
    expect(compatibility.gates.gates[1].prototypeCompatibility).toBeLessThan(-0.99)
    expect(compatibility.gates.gates[0].probability).toBeGreaterThan(
      compatibility.gates.gates[1].probability,
    )
  })

  it('yanlış geometriyi, atlas fingerprintini ve temsil uzayını fail-closed reddeder', () => {
    const fixture = adaptiveFixture()
    const atlas = adaptiveAtlas()
    expect(() => resolveThoughtAdaptiveRegionContext({
      request: { mode: 'shadow', atlasFingerprint: atlas.fingerprint },
      atlas,
      layout: fixture.model.layout,
      representationSpaceId: REPRESENTATION_SPACE_ID,
      sourceGroupId: 'source',
    })).toThrow(/profil kanıtı/)
    expect(() => resolveThoughtAdaptiveRegionContext({
      request: { mode: 'shadow', atlasFingerprint: atlas.fingerprint },
      atlas,
      layout: createThoughtLayout(1024),
      representationSpaceId: REPRESENTATION_SPACE_ID,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      sourceGroupId: 'source',
    })).toThrow(/512\/3584/)
    expect(() => resolveThoughtAdaptiveRegionContext({
      request: { mode: 'shadow', atlasFingerprint: 'wrong' },
      atlas,
      layout: fixture.model.layout,
      representationSpaceId: REPRESENTATION_SPACE_ID,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      sourceGroupId: 'source',
    })).toThrow(/fingerprinti/)
    expect(() => resolveThoughtAdaptiveRegionContext({
      request: { mode: 'shadow', atlasFingerprint: atlas.fingerprint },
      atlas,
      layout: fixture.model.layout,
      representationSpaceId: 'RS-other',
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      sourceGroupId: 'source',
    })).toThrow(/temsil uzayı/)
  })

  it('atlas topolojisini episode boyunca kilitler ve aynı bağlamla replay/clone/resume deterministik kalır', () => {
    const fixture = adaptiveFixture()
    const atlas = adaptiveAtlas()
    const context = regionContext('active', fixture, atlas)
    const executeFirst = () => {
      const started = start(fixture, context, 3)
      return stepThoughtEpisode(
        started.episode, fixture.model, fixture.head, started.index, fixture.config,
        { adaptiveRegionContext: context },
      )
    }
    const left = executeFirst()
    const right = executeFirst()
    expect(right).toEqual(left)
    expect(left.episode.replay[0]).toEqual(expect.objectContaining({
      adaptiveRegionMode: 'active',
      adaptiveRegionAtlasFingerprint: atlas.fingerprint,
      adaptiveRegionBindingFingerprint: context.binding.fingerprint,
    }))

    const cloned = cloneThoughtEpisodeState(left.episode)
    expect(cloned.adaptiveRegion).not.toBe(left.episode.adaptiveRegion)
    const resumed = resumeThoughtEpisode(withEpisodeStatus(cloned, 'paused'))
    const directNext = stepThoughtEpisode(
      left.episode, fixture.model, fixture.head,
      createThoughtRuntimeIndex(fixture.model, fixture.head, fixture.config, 71),
      fixture.config, { adaptiveRegionContext: context },
    )
    const resumedNext = stepThoughtEpisode(
      resumed, fixture.model, fixture.head,
      createThoughtRuntimeIndex(fixture.model, fixture.head, fixture.config, 71),
      fixture.config, { adaptiveRegionContext: context },
    )
    expect(resumedNext).toEqual(directNext)

    const changedAtlas = adaptiveAtlas({
      revision: 1,
      parentFingerprint: atlas.fingerprint,
      holdoutUtility: 0.1,
    })
    const changedContext = regionContext('active', fixture, changedAtlas)
    expect(() => stepThoughtEpisode(
      left.episode, fixture.model, fixture.head,
      createThoughtRuntimeIndex(fixture.model, fixture.head, fixture.config, 71),
      fixture.config, { adaptiveRegionContext: changedContext },
    )).toThrow(/atlası bağlamı değişmiş/)
  })

  it('paralel dalların aktif sayfa izleri ve telemetry dizileri birbirine sızmaz', () => {
    const fixture = adaptiveFixture(2)
    const context = regionContext('active', fixture)
    const { index, episode } = start(fixture, context)
    expect(episode.branches).toHaveLength(2)
    const stepped = stepThoughtEpisode(
      episode, fixture.model, fixture.head, index, fixture.config,
      { adaptiveRegionContext: context },
    )
    expect(stepped.episode.branches.length).toBeGreaterThanOrEqual(2)
    const [leftBranch, rightBranch] = stepped.episode.branches
    expect(leftBranch.fastRelationPatches[0].delta).not.toBe(rightBranch.fastRelationPatches[0].delta)
    const rightValue = rightBranch.fastRelationPatches[0].delta[0]
    leftBranch.fastRelationPatches[0].delta[0] = -0.777
    expect(rightBranch.fastRelationPatches[0].delta[0]).toBe(rightValue)

    const [leftTrace, rightTrace] = stepped.tick.branches
    expect(leftTrace.adaptiveRegion?.pageWeights).not.toBe(rightTrace.adaptiveRegion?.pageWeights)
    const rightWeight = rightTrace.adaptiveRegion!.pageWeights[0]
    leftTrace.adaptiveRegion!.pageWeights[0] = -0.5
    expect(rightTrace.adaptiveRegion!.pageWeights[0]).toBe(rightWeight)
  })

  it('adaptive prompt profilini açık seçer; identityRatio tek başına sessiz geçiş yapmaz', async () => {
    const text = 'Kuş ağacın yanına kondu ve sonra kanatlarını kapattı.'
    const legacyEighth = await encodeThoughtPrompt(text, {
      identityRatio: 0.125,
      allowFlatFallback: false,
    })
    const adaptive = await encodeThoughtPrompt(text, {
      identityRatio: 0.125,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      allowFlatFallback: false,
    })
    const adaptiveReplay = await encodeThoughtPrompt(text, {
      identityRatio: 0.125,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      allowFlatFallback: false,
    })
    expect(legacyEighth.identityLength).toBe(512)
    expect(adaptive.identityLength).toBe(512)
    expect(adaptive.representationSpaceId).not.toBe(legacyEighth.representationSpaceId)
    expect(adaptive.analysisFingerprint).toBe(adaptiveReplay.analysisFingerprint)
    expect(adaptive.signal).toEqual(adaptiveReplay.signal)
  })

  it('serializable request/context/replay hattında atlası ve adaptive prompt uzayını birebir bağlar', async () => {
    const text = 'Ada kutuyu dikkatlice masaya taşıdı.'
    const prompt = await encodeThoughtPrompt(text, {
      identityRatio: 0.125,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      allowFlatFallback: false,
    })
    const identity = prompt.signal.slice(0, prompt.identityLength)
    const relation = prompt.signal.slice(prompt.identityLength)
    const neurons: ThoughtResolvedNeuronInput[] = [
      {
        memberKey: 'adaptive-a', kind: 'raw', text,
        identity, relation, sourceDigest: 'source-a', analysisRunId: 'run-a', neuronId: 'node-a',
      },
      {
        memberKey: 'adaptive-b', kind: 'raw', text: 'karşı örnek',
        identity: Float32Array.from(identity, (value) => -value),
        relation: Float32Array.from(relation, (value) => -value),
        sourceDigest: 'source-b', analysisRunId: 'run-b', neuronId: 'node-b',
      },
    ]
    const familyRun = minimalFamilyRun({
      ids: neurons.map((neuron) => neuron.memberKey),
      representationSpaceId: prompt.representationSpaceId,
      transitions: [{ from: 'adaptive-b', to: 'adaptive-a', probability: 1 }],
    })
    const catalog = createThoughtCatalogFromResolved(neurons, familyRun)
    const baseModelFingerprint = 'adaptive-server-base'
    const head = createThoughtHead(catalog.model, null, baseModelFingerprint)
    const atlas = adaptiveAtlas({ representationSpaceId: prompt.representationSpaceId })
    const request: ThoughtEngineRequest = {
      format: 'thought-engine-request', version: 1,
      episodeId: 'adaptive-service-episode', inputText: text,
      seed: 19, maxTicks: 1, strategy: 'deterministic', learningMode: 'off',
      sourceGroupId: 'adaptive-service-source',
      candidateMemberKeys: ['adaptive-a'],
      adaptiveRegion: { mode: 'shadow', atlasFingerprint: atlas.fingerprint },
      base: {
        snapshotId: 'snapshot-adaptive', artifactManifestHash: 'manifest-adaptive',
        baseModelFingerprint, familyRunFingerprint: familyRun.fingerprint,
        languageStateFingerprint: null, thoughtHeadFingerprint: head.fingerprint,
        thoughtHeadRevision: head.revision,
        representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      },
    }
    const context: ThoughtEngineContext = {
      neurons, familyRun, languageState: null, thoughtHead: head,
      adaptiveRegionAtlas: atlas,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
    }
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const result = await engine.runEpisode(request, context)
    expect(result.graph.prompt.representationSpaceId).toBe(prompt.representationSpaceId)
    expect(result.episode.adaptiveRegion).toEqual(expect.objectContaining({
      mode: 'shadow',
      atlasFingerprint: atlas.fingerprint,
      representationSpaceId: prompt.representationSpaceId,
      sourceGroupId: request.sourceGroupId,
    }))
    expect(result.ticks[0].adaptiveRegion?.mode).toBe('shadow')
    const renamed = await engine.runEpisode({ ...request, episodeId: 'adaptive-other-admin-id' }, context)
    expect(renamed.episode.runKey).toBe(result.episode.runKey)
    expect(renamed.episode.fingerprint).toBe(result.episode.fingerprint)
    expect(renamed.ticks).toEqual(result.ticks)
    const replayAssertion = {
      mode: result.episode.adaptiveRegion!.mode,
      atlasFingerprint: result.episode.adaptiveRegion!.atlasFingerprint,
      bindingFingerprint: result.episode.adaptiveRegion!.fingerprint,
    }
    const replay = await engine.runEpisode({
      ...request,
      replay: { verifyOnly: true, expectedTickCount: 1, adaptiveRegion: replayAssertion },
    }, context)
    expect(replay.episode.fingerprint).toBe(result.episode.fingerprint)
    const regionOff = await engine.runEpisode({
      ...request,
      adaptiveRegion: { mode: 'off' },
    }, context)
    const regionAbsent = await engine.runEpisode({
      ...request,
      adaptiveRegion: undefined,
    }, context)
    expect(regionOff.graph.prompt.representationSpaceId).toBe(prompt.representationSpaceId)
    expect(regionOff.episode).not.toHaveProperty('adaptiveRegion')
    expect(regionOff.ticks.every((tick) => !('adaptiveRegion' in tick))).toBe(true)
    expect(regionOff.ticks.every((tick) => tick.branches.every((branch) => !('adaptiveRegion' in branch)))).toBe(true)
    expect(regionOff.episode.runKey).toBe(regionAbsent.episode.runKey)
    expect(regionOff.episode.fingerprint).toBe(regionAbsent.episode.fingerprint)
    expect(regionOff.ticks).toEqual(regionAbsent.ticks)

    const nextAtlas = adaptiveAtlas({
      revision: 1,
      parentFingerprint: atlas.fingerprint,
      representationSpaceId: prompt.representationSpaceId,
    })
    const activeRequest = {
      ...request,
      adaptiveRegion: {
        mode: 'active' as const,
        atlasFingerprint: atlas.fingerprint,
        compatibilityBlend: 0.25,
        minimumTrace: 0.02,
        maximumActiveGates: 4,
      },
    }
    const nextActiveRequest = {
      ...activeRequest,
      adaptiveRegion: { ...activeRequest.adaptiveRegion, atlasFingerprint: nextAtlas.fingerprint },
    }
    const pairingKey = 'trusted-release-case-1-seed-19'
    const pairedBase = await engine.runEpisode({ ...request, adaptiveRegion: { mode: 'off' } }, {
      ...context,
      benchmarkPairingKey: pairingKey,
    })
    const pairedCandidate = await engine.runEpisode(activeRequest, {
      ...context,
      adaptiveRegionAtlas: atlas,
      benchmarkPairingKey: pairingKey,
    })
    const pairedSham = await engine.runEpisode(nextActiveRequest, {
      ...context,
      adaptiveRegionAtlas: nextAtlas,
      benchmarkPairingKey: pairingKey,
    })
    expect(pairedCandidate.episode.runKey).toBe(pairedBase.episode.runKey)
    expect(pairedSham.episode.runKey).toBe(pairedBase.episode.runKey)

    const normalActive = await engine.runEpisode(activeRequest, { ...context, adaptiveRegionAtlas: atlas })
    const normalNextActive = await engine.runEpisode(nextActiveRequest, { ...context, adaptiveRegionAtlas: nextAtlas })
    expect(normalNextActive.episode.runKey).not.toBe(normalActive.episode.runKey)
    await expect(engine.runEpisode({
      ...activeRequest,
      adaptiveRegion: { ...activeRequest.adaptiveRegion, maximumActiveGates: 3 },
      replay: {
        verifyOnly: true,
        expectedTickCount: 1,
        adaptiveRegion: {
          mode: normalActive.episode.adaptiveRegion!.mode,
          atlasFingerprint: normalActive.episode.adaptiveRegion!.atlasFingerprint,
          bindingFingerprint: normalActive.episode.adaptiveRegion!.fingerprint,
        },
      },
    }, { ...context, adaptiveRegionAtlas: atlas })).rejects.toThrow(/Replay bölge atlası|runKey/i)
    await expect(engine.runEpisode(activeRequest, {
      ...context,
      adaptiveRegionAtlas: atlas,
      benchmarkPairingKey: ' not-canonical ',
    })).rejects.toThrow(/eşleme anahtarı/)

    await expect(engine.runEpisode({
      ...request,
      base: { ...request.base, representationProfile: null },
    }, {
      ...context,
      representationProfile: null,
    })).rejects.toThrow(/profil kanıtı/)

    await expect(engine.runEpisode({
      ...request,
      base: { ...request.base, representationProfile: null },
    }, context)).rejects.toThrow(/temsil profili birebir uyuşmuyor/)
    await expect(engine.runEpisode(request, {
      ...context,
      representationProfile: null,
    })).rejects.toThrow(/temsil profili birebir uyuşmuyor/)
  })

  it('shadow/proposal çok dallı seeded keşfi gözler ama davranış tuzunu değiştirmez', async () => {
    const text = 'Ada kutuyu masaya taşıdı.'
    const prompt = await encodeThoughtPrompt(text, {
      identityRatio: 0.125,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      allowFlatFallback: false,
    })
    const ids = ['seed-a', 'seed-b', 'seed-c', 'seed-d']
    const neurons: ThoughtResolvedNeuronInput[] = ids.map((memberKey, index) => ({
      memberKey,
      kind: 'raw',
      text: `${text} ${index}`,
      identity: new Float32Array(prompt.signal.slice(0, prompt.identityLength)),
      relation: new Float32Array(prompt.signal.slice(prompt.identityLength)),
      sourceDigest: `seed-source-${index}`,
      analysisRunId: `seed-run-${index}`,
      neuronId: `seed-node-${index}`,
    }))
    const transitions = ids.flatMap((from) => ids.map((to) => ({ from, to, probability: 0.25 })))
    const familyRun = minimalFamilyRun({
      ids,
      representationSpaceId: prompt.representationSpaceId,
      transitions,
    })
    const catalog = createThoughtCatalogFromResolved(neurons, familyRun)
    const baseModelFingerprint = 'adaptive-shadow-behavior-base'
    const head = createThoughtHead(catalog.model, null, baseModelFingerprint)
    const atlas = adaptiveAtlas({ representationSpaceId: prompt.representationSpaceId })
    const request: ThoughtEngineRequest = {
      format: 'thought-engine-request',
      version: 1,
      episodeId: 'adaptive-shadow-behavior',
      inputText: text,
      seed: 0x51a7,
      maxTicks: 3,
      strategy: 'seeded-stochastic',
      learningMode: 'off',
      sourceGroupId: 'adaptive-shadow-source',
      candidateMemberKeys: ids,
      base: {
        snapshotId: 'adaptive-shadow-snapshot',
        artifactManifestHash: 'adaptive-shadow-manifest',
        baseModelFingerprint,
        familyRunFingerprint: familyRun.fingerprint,
        languageStateFingerprint: null,
        thoughtHeadFingerprint: head.fingerprint,
        thoughtHeadRevision: head.revision,
        representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      },
    }
    const context: ThoughtEngineContext = {
      neurons,
      familyRun,
      languageState: null,
      thoughtHead: head,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
    }
    const engine = createThoughtRuntimeEngine({ maximumBranches: 4 })
    const baseline = await engine.runEpisode(request, context)
    const shadow = await engine.runEpisode({
      ...request,
      adaptiveRegion: { mode: 'shadow', atlasFingerprint: atlas.fingerprint },
    }, { ...context, adaptiveRegionAtlas: atlas })
    const proposal = await engine.runEpisode({
      ...request,
      adaptiveRegion: { mode: 'proposal', atlasFingerprint: atlas.fingerprint },
    }, { ...context, adaptiveRegionAtlas: atlas })
    const shadowWithIrrelevantGateLimit = await engine.runEpisode({
      ...request,
      adaptiveRegion: { mode: 'shadow', atlasFingerprint: atlas.fingerprint, maximumActiveGates: 1 },
    }, { ...context, adaptiveRegionAtlas: atlas })
    const active = await engine.runEpisode({
      ...request,
      adaptiveRegion: { mode: 'active', atlasFingerprint: atlas.fingerprint },
    }, { ...context, adaptiveRegionAtlas: atlas })

    const dynamics = (result: typeof baseline) => ({
      runKey: result.episode.runKey,
      branches: result.episode.branches,
      temporaryBirths: result.episode.temporaryBirths,
      ticks: result.ticks.map(({ adaptiveRegion: _tickAudit, replayFingerprint: _replayAudit, ...tick }) => ({
        ...tick,
        branches: tick.branches.map(({ adaptiveRegion: _branchAudit, ...branch }) => branch),
      })),
      eventGraph: result.graph.eventGraph,
      branchScores: result.graph.branchScores,
    })
    expect(dynamics(shadow)).toEqual(dynamics(baseline))
    expect(dynamics(shadowWithIrrelevantGateLimit)).toEqual(dynamics(shadow))
    expect(dynamics(proposal)).toEqual(dynamics(baseline))
    expect(shadow.episode.fingerprint).not.toBe(baseline.episode.fingerprint)
    expect(proposal.episode.fingerprint).not.toBe(baseline.episode.fingerprint)
    expect(active.episode.runKey).not.toBe(baseline.episode.runKey)
    expect(shadow.ticks.every((tick) => tick.adaptiveRegion?.mode === 'shadow')).toBe(true)
    expect(proposal.ticks.every((tick) => tick.adaptiveRegion?.mode === 'proposal')).toBe(true)
  })
})
