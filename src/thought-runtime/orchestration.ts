import {
  LANGUAGE_BOS_MEMBER,
  LANGUAGE_EOS_MEMBER,
  type LanguageExperienceState,
} from '../language-experience/types'
import type { SparseNeuronTransition } from '../families/types'
import {
  createThoughtCatalog,
  createThoughtCatalogFromResolved,
  isResolvedThoughtNeuronArray,
  isTextLabAnalysisArray,
  isThoughtCatalog,
  isThoughtModel,
} from './catalog'
import { createThoughtRuntimeConfig, DEFAULT_THOUGHT_RUNTIME_CONFIG } from './config'
import { consolidateThoughtEpisode } from './consolidation'
import { thoughtDigest, thoughtVectorDigest } from './deterministic'
import {
  cloneThoughtEpisodeState,
  resumeThoughtEpisode,
  startThoughtEpisode,
  stepThoughtEpisode,
  withEpisodeStatus,
} from './engine'
import {
  deriveEventGraph,
  deriveLiveRoutingHints,
  evaluateThoughtBranches,
  normalizeEventGraph,
  routingHintsFromFamilyRun,
} from './evaluation'
import { createThoughtHead, createThoughtLayout, createThoughtModel } from './model'
import { vectorNorm } from './math'
import { canonicalizeThoughtPrompt, encodeThoughtPrompt } from './prompt'
import { createThoughtRuntimeIndex, withTemporaryThoughtNeurons } from './routing'
import { scoreThoughtBranches } from './structural'
import { applyThoughtMacroLearningSafety, createThoughtMacroRuntime } from './concept-macro-runtime'
import {
  assertThoughtAdaptiveRegionEpisodeBinding,
  resolveThoughtAdaptiveRegionContext,
} from './adaptive-regions'
import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  getTextLabRepresentationProfile,
} from '../text-lab/domain/config'
import type { TextLabRepresentationProfileId } from '../text-lab/domain/types'
import type {
  SparseHabitRow,
  ThoughtCatalogDiagnostic,
  ThoughtCatalogResult,
  ThoughtEngineContext,
  ThoughtEngineRequest,
  ThoughtEpisodeResult,
  ThoughtEpisodeState,
  ThoughtGraphDiagnostics,
  ThoughtHead,
  ThoughtModel,
  ThoughtNeuron,
  ThoughtResolvedNeuronInput,
  ThoughtResidencyPromptContextV1,
  ThoughtRuntimeConfig,
  ThoughtSelectionMode,
  ThoughtTick,
  ThoughtVector,
} from './types'

function defaultAdaptiveRegionSourceGroupId(inputText: string): string {
  return `prompt:${thoughtDigest(
    'adaptive-default-source-group',
    canonicalizeThoughtPrompt(inputText),
  ).slice(0, 24)}`
}

function normalizeTrustedRepresentationProfile(
  value: unknown,
  label: string,
): TextLabRepresentationProfileId | null {
  if (value === undefined || value === null) return null
  if (value !== 'legacy-quarter-id-v2' && value !== ADAPTIVE_TEXT_LAB_PROFILE_ID) {
    throw new TypeError(`${label} bilinmeyen bir temsil profili taşıyor.`)
  }
  return value
}

function validateRepresentationProfileLayout(
  profileId: TextLabRepresentationProfileId | null,
  layout: ThoughtModel['layout'],
): void {
  if (!profileId) return
  const profile = getTextLabRepresentationProfile(profileId)
  if (
    layout.side !== profile.matrixSide ||
    layout.cellCount !== profile.matrixSide * profile.matrixSide ||
    layout.identityLength !== profile.identityRows * profile.matrixSide ||
    layout.relationStart !== layout.identityLength ||
    layout.relationLength !== profile.relationRows * profile.matrixSide
  ) {
    throw new TypeError(`Trusted ${profileId} profil kanıtı model I/R yerleşimiyle uyuşmuyor.`)
  }
}

function resolveTrustedBaseRepresentationProfile(
  request: ThoughtEngineRequest,
  context: ThoughtEngineContext,
  layout: ThoughtModel['layout'],
): TextLabRepresentationProfileId | null {
  const requested = normalizeTrustedRepresentationProfile(
    request.base?.representationProfile,
    'request.base.representationProfile',
  )
  const trusted = normalizeTrustedRepresentationProfile(
    context.representationProfile,
    'context.representationProfile',
  )
  if (requested !== trusted) {
    throw new TypeError('İstek ve trusted snapshot temsil profili birebir uyuşmuyor.')
  }
  validateRepresentationProfileLayout(trusted, layout)
  return trusted
}

function validateRequest(request: ThoughtEngineRequest, config: ThoughtRuntimeConfig): void {
  if (request?.format !== 'thought-engine-request' || request.version !== 1) {
    throw new TypeError('Düşünce motoru isteği format/version 1 olmalıdır.')
  }
  if (!request.episodeId?.trim() || !request.inputText?.trim()) throw new RangeError('episodeId ve inputText boş olamaz.')
  if (!Number.isInteger(request.seed)) throw new RangeError('seed bir tam sayı olmalıdır.')
  if (!Number.isInteger(request.maxTicks) || request.maxTicks < 1 || request.maxTicks > config.maximumTicks) {
    throw new RangeError(`maxTicks 1..${config.maximumTicks} arasında tam sayı olmalıdır.`)
  }
  for (const [key, value] of Object.entries({
    snapshotId: request.base?.snapshotId,
    artifactManifestHash: request.base?.artifactManifestHash,
    baseModelFingerprint: request.base?.baseModelFingerprint,
    familyRunFingerprint: request.base?.familyRunFingerprint,
    thoughtHeadFingerprint: request.base?.thoughtHeadFingerprint,
  })) {
    if (typeof value !== 'string' || !value.trim()) throw new RangeError(`base.${key} boş olamaz.`)
  }
  normalizeTrustedRepresentationProfile(
    request.base.representationProfile,
    'base.representationProfile',
  )
  if (!Number.isSafeInteger(request.base.thoughtHeadRevision) || request.base.thoughtHeadRevision < 0) {
    throw new RangeError('base.thoughtHeadRevision negatif olmayan güvenli bir tam sayı olmalıdır.')
  }
  if ((request.candidateMemberKeys?.length ?? 0) > config.maximumCandidatePool ||
      request.candidateMemberKeys?.some((key) => typeof key !== 'string' || !key.trim())) {
    throw new RangeError(`candidateMemberKeys en fazla ${config.maximumCandidatePool} dolu kimlik taşıyabilir.`)
  }
  if (request.replay) {
    if (!Number.isInteger(request.replay.expectedTickCount) || request.replay.expectedTickCount < 0 ||
        request.replay.expectedTickCount > request.maxTicks) {
      throw new RangeError('replay.expectedTickCount 0..maxTicks arasında tam sayı olmalıdır.')
    }
  }
  const regionMode = request.adaptiveRegion?.mode
  if (regionMode !== undefined &&
      regionMode !== 'off' && regionMode !== 'shadow' && regionMode !== 'proposal' && regionMode !== 'active') {
    throw new RangeError(`Bilinmeyen uyarlanabilir bölge kipi: ${String(regionMode)}`)
  }
}

function strategyAlias(strategy: string): ThoughtSelectionMode {
  if (strategy === 'deterministic' || strategy === 'greedy') return 'deterministic'
  if (strategy === 'seeded-stochastic' || strategy === 'seeded-sample') return 'seeded-stochastic'
  throw new RangeError(`Bilinmeyen düşünce stratejisi: ${strategy}`)
}

function habitRowsFromFamilyTransitions(transitions: readonly SparseNeuronTransition[], validIds: ReadonlySet<string>): SparseHabitRow[] {
  const rows = new Map<string, SparseHabitRow>()
  for (const transition of transitions) {
    if (!validIds.has(transition.fromMemberKey) || !validIds.has(transition.toMemberKey)) continue
    const row = rows.get(transition.fromMemberKey) ?? { fromNeuronId: transition.fromMemberKey, edges: [] }
    row.edges.push({
      toNeuronId: transition.toMemberKey,
      priorEvidence: Math.max(0, transition.probability),
      excitationEvidence: 0,
      inhibitionEvidence: 0,
      trustMass: 0,
      updateCount: 0,
      independentSourceIds: [],
    })
    rows.set(transition.fromMemberKey, row)
  }
  return [...rows.values()]
}

function resolveCatalog(context: ThoughtEngineContext, config: ThoughtRuntimeConfig): ThoughtCatalogResult {
  const input = context.neurons
  if (isThoughtCatalog(input)) return input
  if (isThoughtModel(input)) return { model: input, diagnostics: [] }
  if (isTextLabAnalysisArray(input)) return createThoughtCatalog(input, context.familyRun, config)
  if (isResolvedThoughtNeuronArray(input)) return createThoughtCatalogFromResolved(input, context.familyRun, config)
  if (Array.isArray(input) && input.length && input.every((entry) => entry && typeof entry === 'object' &&
      typeof entry.id === 'string' && Array.isArray(entry.familyMemberships) && entry.provenance &&
      entry.identity instanceof Float32Array && entry.relation instanceof Float32Array)) {
    const neurons = input as ThoughtNeuron[]
    const layout = createThoughtLayout(neurons[0].identity.length)
    const validIds = new Set(neurons.map((neuron) => neuron.id))
    return {
      model: createThoughtModel({
        layout,
        neurons,
        families: context.familyRun.families.map((family) => ({ id: family.id, region: family.region })),
        habitRows: habitRowsFromFamilyTransitions(context.familyRun.neuronTransitions, validIds),
      }, config),
      diagnostics: neurons.map((neuron): ThoughtCatalogDiagnostic => ({
        memberKey: neuron.id, kind: 'included', reason: 'Hazır saf ThoughtNeuron kataloğu kullanıldı.',
      })),
    }
  }
  throw new TypeError('context.neurons ThoughtModel, ThoughtCatalog, ThoughtNeuron[] veya TextLabAnalysis[] olmalıdır.')
}

function expectedHeadBase(request: ThoughtEngineRequest, context: ThoughtEngineContext): string {
  void context
  if (!request.base.baseModelFingerprint) throw new TypeError('İstek baseModelFingerprint taşımıyor.')
  return request.base.baseModelFingerprint
}

function languageState(value: unknown): LanguageExperienceState | null {
  return value && typeof value === 'object' && (value as LanguageExperienceState).format === 'language-experience-state'
    ? value as LanguageExperienceState
    : null
}

function compileTeacherEdgeRows(
  state: LanguageExperienceState | null,
  request: ThoughtEngineRequest,
  context: ThoughtEngineContext,
  validIds: ReadonlyMap<string, ThoughtNeuron>,
  maximum: number,
): ReadonlyMap<string, readonly string[]> {
  if (!state) {
    if (request.base.languageStateFingerprint !== null) {
      throw new TypeError('İstek dil deneyimi fingerprinti taşıyor fakat snapshot state çözümlenemedi.')
    }
    return new Map()
  }
  if (state.fingerprint !== request.base.languageStateFingerprint ||
      state.familyRunFingerprint !== context.familyRun.fingerprint ||
      state.representationSpaceId !== context.familyRun.representationSpaceId) {
    throw new TypeError('Dil deneyimi öğretmen kenarları snapshot/FamilyRun gösterim alanıyla uyuşmuyor.')
  }
  const byOrigin = new Map<string, typeof state.neuronTransitions>()
  for (const edge of state.neuronTransitions) {
    if (!validIds.has(edge.fromKey) || !validIds.has(edge.toKey) ||
        edge.toKey === LANGUAGE_EOS_MEMBER || edge.toKey === LANGUAGE_BOS_MEMBER ||
        edge.excitatoryEvidence <= edge.inhibitoryEvidence) continue
    const row = byOrigin.get(edge.fromKey) ?? []
    row.push(edge)
    byOrigin.set(edge.fromKey, row)
  }
  return new Map([...byOrigin].map(([origin, row]) => [origin, row
    .sort((left, right) => right.score - left.score ||
      (right.excitatoryEvidence - right.inhibitoryEvidence) - (left.excitatoryEvidence - left.inhibitoryEvidence) ||
      left.toKey.localeCompare(right.toKey))
    .slice(0, maximum)
    .map((edge) => edge.toKey)]))
}

function activeTeacherRows(
  rows: ReadonlyMap<string, readonly string[]>,
  episode: ThoughtEpisodeState,
): Readonly<Record<string, readonly string[]>> {
  const active: Record<string, readonly string[]> = {}
  for (const branch of episode.branches) {
    const row = rows.get(branch.currentNeuronId)
    if (row?.length) active[branch.currentNeuronId] = row
  }
  return active
}

function initialNeuronIds(context: ThoughtEngineContext, model: ThoughtModel): string[] {
  const valid = new Set(model.neurons.map((neuron) => neuron.id))
  const preview = context.familyRun.routingPreview
  return (preview?.entries ?? []).map((entry) => entry.memberKey).filter((id) => valid.has(id)).slice(0, 32)
}

function sparseGraph(
  episode: ThoughtEpisodeState,
  model: ThoughtModel,
  head: ThoughtHead,
  familyGraph: ThoughtGraphDiagnostics['familyGraph'],
): Pick<ThoughtGraphDiagnostics, 'nodes' | 'edges'> {
  const neuronMass = new Map<string, number>()
  const familyIds = new Set<string>()
  for (const branch of episode.branches) {
    for (const id of branch.path) neuronMass.set(id, Math.max(neuronMass.get(id) ?? 0, branch.mass))
    for (const transition of branch.transitions) familyIds.add(transition.familyId)
  }
  const nodes = [
    ...[...neuronMass].sort(([left], [right]) => left.localeCompare(right)).map(([id, mass]) => ({ id, kind: 'neuron' as const, mass })),
    ...[...familyIds].sort().map((id) => ({ id, kind: 'family' as const })),
  ]
  const edgeMap = new Map<string, ThoughtGraphDiagnostics['edges'][number]>()
  for (const branch of episode.branches) {
    for (const transition of branch.transitions) {
      const key = `n\u0000${transition.fromNeuronId}\u0000${transition.toNeuronId}`
      edgeMap.set(key, {
        from: transition.fromNeuronId,
        to: transition.toNeuronId,
        kind: 'neuron',
        probability: transition.habitProbability,
        excitation: transition.persistentExcitation + transition.fastExcitation,
        inhibition: transition.persistentInhibition + transition.fastInhibition,
      })
    }
  }
  for (const transition of familyGraph.transitions) {
    if (!familyIds.has(transition.fromFamilyId) && !familyIds.has(transition.toFamilyId)) continue
    edgeMap.set(`f\u0000${transition.fromFamilyId}\u0000${transition.toFamilyId}`, {
      from: transition.fromFamilyId, to: transition.toFamilyId, kind: 'family', probability: transition.probability,
    })
  }
  for (const delta of head.neuronEdges) {
    const key = `n\u0000${delta.fromMemberKey}\u0000${delta.toMemberKey}`
    if (!neuronMass.has(delta.fromMemberKey) && !neuronMass.has(delta.toMemberKey)) continue
    const current = edgeMap.get(key)
    if (current) { current.excitation = (current.excitation ?? 0) + delta.excitation; current.inhibition = (current.inhibition ?? 0) + delta.inhibition }
  }
  return { nodes, edges: [...edgeMap.values()] }
}

async function emitCheckpoint(
  context: ThoughtEngineContext,
  episode: ThoughtEpisodeState,
  ticks: readonly ThoughtTick[],
): Promise<void> {
  await context.onCheckpoint?.({ tick: episode.tick, state: cloneThoughtEpisodeState(episode), ticks: structuredClone([...ticks]) })
}

async function runWithConfig(
  request: ThoughtEngineRequest,
  context: ThoughtEngineContext,
  config: ThoughtRuntimeConfig,
): Promise<ThoughtEpisodeResult> {
  validateRequest(request, config)
  if (request.base.familyRunFingerprint !== context.familyRun.fingerprint) {
    throw new TypeError('İstek FamilyRun fingerprinti çalışan katalogla uyuşmuyor.')
  }
  const selectionMode = strategyAlias(request.strategy)
  const catalog = resolveCatalog(context, config)
  const model = context.modelFingerprintOverride
    ? { ...catalog.model, fingerprint: context.modelFingerprintOverride }
    : catalog.model
  const representationProfile = resolveTrustedBaseRepresentationProfile(request, context, model.layout)
  const benchmarkPairingKey = context.benchmarkPairingKey
  if (benchmarkPairingKey !== undefined && (
    typeof benchmarkPairingKey !== 'string' || benchmarkPairingKey.length < 1 || benchmarkPairingKey.length > 256 ||
    benchmarkPairingKey !== benchmarkPairingKey.trim() || benchmarkPairingKey !== benchmarkPairingKey.normalize('NFC')
  )) {
    throw new TypeError('Trusted benchmark eşleme anahtarı kanonik ve 1..256 karakter olmalıdır.')
  }
  const adaptiveRegionContext = resolveThoughtAdaptiveRegionContext({
    request: request.adaptiveRegion,
    atlas: context.adaptiveRegionAtlas,
    layout: model.layout,
    representationSpaceId: context.familyRun.representationSpaceId,
    representationProfile,
    sourceGroupId: request.sourceGroupId?.trim() || defaultAdaptiveRegionSourceGroupId(request.inputText),
  })
  if (context.thoughtHead && (
    context.thoughtHead.fingerprint !== request.base.thoughtHeadFingerprint ||
    context.thoughtHead.revision !== request.base.thoughtHeadRevision
  )) throw new TypeError('İstek thoughtHead fingerprint/revision ile çalışan başlık uyuşmuyor.')
  const head = createThoughtHead(model, context.thoughtHead, expectedHeadBase(request, context))
  const compiledMacroRuntime = createThoughtMacroRuntime({
    macroHead: context.conceptMacroHead,
    model,
    representationSpaceId: context.familyRun.representationSpaceId,
    familyRunFingerprint: context.familyRun.fingerprint,
  })
  const requestedMacroHeadFingerprint = request.base.conceptMacroHeadFingerprint ?? null
  const actualMacroHeadFingerprint = context.conceptMacroHead?.fingerprint ?? null
  if (requestedMacroHeadFingerprint !== actualMacroHeadFingerprint) {
    throw new TypeError('İstek conceptMacroHead fingerprinti seçili snapshot makro başlığıyla uyuşmuyor.')
  }
  const macroRuntime = applyThoughtMacroLearningSafety(compiledMacroRuntime, request.learningMode)
  const prompt = await encodeThoughtPrompt(request.inputText, {
    allowFlatFallback: true,
    identityRatio: model.layout.identityLength / model.layout.cellCount,
    ...(representationProfile
      ? { representationProfile }
      : {}),
  })
  if (prompt.identityLength !== model.layout.identityLength) {
    throw new TypeError('Prompt TextLab I/R sınırı snapshot model yerleşimiyle uyuşmuyor.')
  }
  if (prompt.representationSpaceId !== context.familyRun.representationSpaceId) {
    throw new TypeError('Prompt kodlayıcısının temsil uzayı FamilyRun fiziksel nöron uzayıyla uyuşmuyor.')
  }
  const initialSignal = context.benchmarkInitialSignal ?? prompt.signal
  if (context.benchmarkInitialSignal && (
    !(initialSignal instanceof Float32Array) || initialSignal.length !== model.layout.cellCount ||
    initialSignal.some((value) => !Number.isFinite(value) || value < -1 || value > 1) ||
    vectorNorm(initialSignal) > config.signalNormMaximum + 1e-6
  )) throw new TypeError('Benchmark başlangıç sinyali snapshot 64x64/norm sözleşmesine uymuyor.')
  const semanticRunKey = thoughtDigest(
    'engine-run', thoughtVectorDigest(initialSignal), request.seed, selectionMode,
    request.maxTicks, model.fingerprint, request.base.baseModelFingerprint,
    request.base.familyRunFingerprint, request.base.languageStateFingerprint, head.revision, head.fingerprint,
    actualMacroHeadFingerprint, request.learningMode,
    [...new Set(request.candidateMemberKeys ?? [])].sort(), config,
    ...(representationProfile ? [representationProfile] : []),
    ...(benchmarkPairingKey
      ? ['trusted-benchmark-pair-v1', benchmarkPairingKey]
      : adaptiveRegionContext?.binding.mode === 'active'
        ? [adaptiveRegionContext.binding.mode, adaptiveRegionContext.binding.atlasFingerprint,
          adaptiveRegionContext.binding.fingerprint]
        : []),
  )
  const index = createThoughtRuntimeIndex(model, head, config, request.seed)
  const candidateNeuronIds = [...new Set(request.candidateMemberKeys ?? [])]
    .filter((memberKey) => index.neuronById.has(memberKey))
    .sort()
  const teacherRows = compileTeacherEdgeRows(
    languageState(context.languageState), request, context, index.neuronById, config.maximumEvaluatedCandidates,
  )
  let episode: ThoughtEpisodeState
  if (request.resume) {
    if (request.resume.runKey !== semanticRunKey || request.resume.initialSignalDigest !== thoughtVectorDigest(initialSignal)) {
      throw new TypeError('Resume checkpoint semantik input/base/seed ile uyuşmuyor.')
    }
    if (request.resume.maximumTicks !== request.maxTicks) throw new TypeError('Resume sırasında maxTicks değiştirilemez.')
    assertThoughtAdaptiveRegionEpisodeBinding(request.resume, adaptiveRegionContext, model.layout)
    episode = resumeThoughtEpisode(request.resume)
  } else {
    episode = startThoughtEpisode(model, head, index, initialSignal, {
      episodeId: request.episodeId,
      runKey: semanticRunKey,
      seed: request.seed,
      maximumTicks: request.maxTicks,
      initialNeuronIds: [...new Set([...candidateNeuronIds, ...initialNeuronIds(context, model)])]
        .slice(0, config.maximumCandidatePool),
      ...(adaptiveRegionContext ? { adaptiveRegionContext } : {}),
    }, config)
  }
  if (request.replay?.adaptiveRegion) {
    const expected = request.replay.adaptiveRegion
    if (!adaptiveRegionContext ||
        expected.mode !== adaptiveRegionContext.binding.mode ||
        expected.atlasFingerprint !== adaptiveRegionContext.binding.atlasFingerprint ||
        expected.bindingFingerprint !== adaptiveRegionContext.binding.fingerprint) {
      throw new TypeError('Replay bölge atlası iddiası etkin episode bağlamıyla uyuşmuyor.')
    }
  }
  const ticks: ThoughtTick[] = []
  const checkpointTicks: ThoughtTick[] = []
  const retainReturnedTicks = !context.onTick
  const initialHints = routingHintsFromFamilyRun(context.familyRun)
  const replayTarget = request.replay?.verifyOnly ? request.replay.expectedTickCount : null
  if (replayTarget !== null && episode.tick > replayTarget) {
    throw new TypeError('Replay checkpoint tick’i beklenen tick sayısını aşamaz.')
  }
  if (replayTarget === episode.tick && episode.status === 'active') {
    episode = withEpisodeStatus(episode, 'paused')
  }
  while (episode.status === 'active' && (replayTarget === null || episode.tick < replayTarget)) {
    if (context.signal?.aborted || await context.shouldCancel?.()) {
      episode = withEpisodeStatus(episode, 'cancelled')
      await emitCheckpoint(context, episode, checkpointTicks)
      break
    }
    if (await context.shouldPause?.()) {
      episode = withEpisodeStatus(episode, 'paused')
      await emitCheckpoint(context, episode, checkpointTicks)
      break
    }
    const liveIndex = withTemporaryThoughtNeurons(index, episode.temporaryBirths)
    const hints = deriveLiveRoutingHints(episode, model, liveIndex, initialHints)
    const stepped = stepThoughtEpisode(episode, model, head, index, config, {
      selectionMode,
      candidateNeuronIds,
      teacherEdgeNeuronIdsByOrigin: activeTeacherRows(teacherRows, episode),
      macroRuntime,
      ...(adaptiveRegionContext ? { adaptiveRegionContext } : {}),
    }, hints)
    episode = stepped.episode
    if (retainReturnedTicks) ticks.push(stepped.tick)
    checkpointTicks.push(stepped.tick)
    await context.onTick?.(stepped.tick)
    if (episode.tick % 64 === 0) {
      await emitCheckpoint(context, episode, checkpointTicks)
      checkpointTicks.length = 0
    }
  }
  if (replayTarget !== null && episode.tick === replayTarget && episode.status === 'active') {
    episode = withEpisodeStatus(episode, 'paused')
  }
  const finalIndex = withTemporaryThoughtNeurons(index, episode.temporaryBirths)
  const finalHints = deriveLiveRoutingHints(episode, model, finalIndex, initialHints)
  const eventGraph = deriveEventGraph(episode, model, finalIndex, finalHints)
  const expected = request.expectedOutcome ? normalizeEventGraph(request.expectedOutcome) : null
  const branchEvaluations = expected
    ? evaluateThoughtBranches(episode, model, finalIndex, expected, config, finalHints)
    : []
  const dominantId = [...episode.branches].sort((left, right) => right.mass - left.mass || left.id.localeCompare(right.id))[0]?.id
  const evaluation = branchEvaluations.find((entry) => entry.branchId === dominantId)?.evaluation ?? null
  const branchScores = scoreThoughtBranches(episode)
  const evidence = request.outcomeEvidence ?? request.teacherEvidence ?? null
  const headDelta = episode.status === 'stopped'
    ? consolidateThoughtEpisode({
      episode, model, head, branchScores, branchEvaluations, evidence,
      learningMode: request.learningMode, config,
    })
    : undefined
  const { signal: _signal, ...promptDiagnostics } = prompt
  const flat = sparseGraph(episode, model, head, index.familyGraph)
  return {
    episode,
    ticks,
    ...(headDelta ? { headDelta } : {}),
    graph: {
      eventGraph,
      evaluation,
      branchEvaluations,
      familyGraph: index.familyGraph,
      branchScores,
      catalog: catalog.diagnostics,
      prompt: promptDiagnostics,
      baseDiagnostics: context.baseDiagnostics,
      ...flat,
    },
  }
}

export interface ThoughtRuntimeEngine {
  readonly config: ThoughtRuntimeConfig
  runEpisode(request: ThoughtEngineRequest, context: ThoughtEngineContext): Promise<ThoughtEpisodeResult>
  prepareResidencyBase(input: {
    neurons: ThoughtResolvedNeuronInput[]
    familyRun: ThoughtEngineContext['familyRun']
    thoughtHead: ThoughtHead
    baseModelFingerprint: string
  }): { model: ThoughtModel; head: ThoughtHead }
  encodeResidencyPrompt(
    inputText: string,
    layout: ThoughtModel['layout'],
    representationSpaceId: string,
    context?: ThoughtResidencyPromptContextV1,
  ): Promise<ThoughtVector>
}

export function createThoughtRuntimeEngine(overrides: Partial<ThoughtRuntimeConfig> = {}): ThoughtRuntimeEngine {
  const config = createThoughtRuntimeConfig(overrides)
  return {
    config,
    runEpisode: (request, context) => runWithConfig(request, context, config),
    prepareResidencyBase: (input) => {
      const catalog = createThoughtCatalogFromResolved(input.neurons, input.familyRun, config)
      return {
        model: catalog.model,
        head: createThoughtHead(catalog.model, input.thoughtHead, input.baseModelFingerprint),
      }
    },
    encodeResidencyPrompt: async (inputText, layout, representationSpaceId, residencyContext) => {
      const representationProfile = normalizeTrustedRepresentationProfile(
        residencyContext?.representationProfile,
        'Residency trusted representationProfile',
      )
      validateRepresentationProfileLayout(representationProfile, layout)
      const adaptiveRegionContext = resolveThoughtAdaptiveRegionContext({
        request: residencyContext?.adaptiveRegion,
        atlas: residencyContext?.adaptiveRegionAtlas,
        layout,
        representationSpaceId,
        representationProfile,
        sourceGroupId: residencyContext?.sourceGroupId?.trim() || defaultAdaptiveRegionSourceGroupId(inputText),
      })
      const prompt = await encodeThoughtPrompt(inputText, {
        allowFlatFallback: true,
        identityRatio: layout.identityLength / layout.cellCount,
        ...(representationProfile
          ? { representationProfile }
          : {}),
      })
      if (prompt.identityLength !== layout.identityLength || prompt.representationSpaceId !== representationSpaceId) {
        throw new TypeError('Prompt SSD residency tabanıyla aynı temsil uzayında değil.')
      }
      return prompt.signal
    },
  }
}

const defaultEngine = createThoughtRuntimeEngine(DEFAULT_THOUGHT_RUNTIME_CONFIG)

/** Exact server adapter entry point. */
export function runEpisode(request: ThoughtEngineRequest, context: ThoughtEngineContext): Promise<ThoughtEpisodeResult> {
  return defaultEngine.runEpisode(request, context)
}
