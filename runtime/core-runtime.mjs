import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  DEFAULT_TEXT_LAB_CONFIG,
  analyzeText,
  digestVector,
  normalizeTextLabConfig,
  resolveTextLabRepresentationProfileId,
} from '../src/text-lab/engine/index.ts'
import {
  DEFAULT_SYNTHESIS_CONFIG,
  runSynthesis,
} from '../src/synthesis/index.ts'
import {
  DEFAULT_FAMILY_CONFIG,
  rawMemberKey,
  runFamilyLearning,
} from '../src/families/index.ts'
import * as languageEngine from '../src/language-experience/index.ts'
import * as languageOutput from '../src/language-output/index.ts'
import {
  applyMacroShadowEvaluation,
  computeExactSignature,
  computeFamilySignature,
  cosineSimilarity as macroCosineSimilarity,
  createMacroConfig,
  createMacroHead,
  evaluateMacroShadow,
  ingestMacroTraces,
  macroDigest,
  rebaseMacroHeadReference,
  transitionCompositeLifecycle,
} from '../src/concept-macros/index.ts'
import { createThoughtRuntimeEngine } from '../src/thought-runtime/orchestration.ts'
import { createThoughtCatalogFromResolved } from '../src/thought-runtime/catalog.ts'
import { createThoughtHead } from '../src/thought-runtime/model.ts'
import { projectThoughtVector } from '../src/thought-runtime/math.ts'
import {
  DEFAULT_STAGE_HANDLERS,
  createPipelineExecutor,
  evaluateCandidate,
} from '../server/pipeline.mjs'
import { canonicalOllamaIndependenceGroupId } from '../server/ollama-lineage.mjs'
import {
  findCachedArtifact,
  persistArtifact,
  restoreArtifact,
  stableRuntimeHash,
} from './artifacts.mjs'
import {
  createFamilyModelLineage,
  createFamilyRunCacheKey,
  createFamilyContinuityState,
  FAMILY_CONTINUITY_ARTIFACT_KIND,
  readActiveFamilyContinuity,
  readActiveRuntimeEnvelope,
} from './family-cache.mjs'
import {
  LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND,
  LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND,
  readLanguageOutputSnapshot,
} from './language-output-cache.mjs'
import {
  CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
  CONCEPT_MACRO_OBSERVATION_ARTIFACT_KIND,
  CONCEPT_MACRO_SHADOW_ARTIFACT_KIND,
  readLatestConceptMacroHead,
} from './concept-macro-cache.mjs'
import {
  CORPUS_RUNTIME_LIMITS,
  chunkCanonicalText,
  createDiverseSynthesisShards,
} from './corpus.mjs'
import {
  currentPlasticRegionBase,
  plasticRegionCarryForwardArtifactIds,
} from './plastic-region-cache.mjs'
import { readPlasticRegionEvidenceHeads } from './plastic-region-evidence.mjs'
import {
  STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT,
  produceAndPersistStructuralSemanticProbes,
} from './plastic-region-probe-producer.mjs'
import {
  buildPlasticRegionCycle,
  extractPlasticRegionObservationsFromThoughtResult,
  loadPlasticRegionBaseAtlas,
  persistPlasticRegionCycle,
  planPlasticRegionReencodeInventory,
  resolveAdaptiveTextLabConfig,
  resolvePlasticRegionRuntimeRequest,
} from './plastic-region-pipeline.mjs'

const RUNTIME_VERSION = 'bombom-autonomous-runtime-v1'
const MACRO_OBSERVATION_THOUGHT_ENGINE = createThoughtRuntimeEngine({
  maximumBranches: 2,
  selectionMode: 'deterministic',
})
const PLASTIC_REGION_OBSERVATION_THOUGHT_ENGINE = createThoughtRuntimeEngine({
  maximumBranches: 2,
  selectionMode: 'deterministic',
})

function appendArtifact(state, artifact) {
  if (artifact?.id && !state.artifactIds.includes(artifact.id)) state.artifactIds.push(artifact.id)
}

function budgetGuard(context, startedAt) {
  if (context.signal.aborted) return true
  context.checkResources?.('runtime-inner-loop')
  const current = context.store.getJob(context.jobId)
  if (current?.state === 'pause_requested' || current?.state === 'cancel_requested') return true
  const maximumMs = Number(context.resourceBudgets?.maxRunMinutes ?? context.store.getConfig().budgets.maxRunMinutes) * 60_000
  if (Date.now() - startedAt > maximumMs) {
    const error = new Error('Eğitim çevrimi azami çalışma süresi bütçesini aştı.')
    error.code = 'RUN_TIME_BUDGET_EXCEEDED'
    throw error
  }
  return false
}

function sourceKind(source) {
  if (source.sourceType === 'ollama-generated') return 'ollama'
  if (source.sourceType === 'local-file' || source.sourceType === 'local-upload') return 'file'
  if (source.sourceType === 'manual-text') return 'human'
  return 'unknown'
}

export function resolveGeneratorModel(provenance) {
  if (!provenance || typeof provenance !== 'object' || Array.isArray(provenance)) return undefined
  for (const value of [provenance.generationModel, provenance.modelIdentity, provenance.modelDigest, provenance.model]) {
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return undefined
}

function physicalSourceLineageKey(source) {
  const provenance = source?.provenance ?? {}
  const kind = sourceKind(source)
  if (kind === 'file') {
    const canonicalPath = typeof provenance.canonicalPath === 'string'
      ? provenance.canonicalPath.trim()
      : ''
    const locator = typeof source?.locator === 'string' ? source.locator.trim() : ''
    if (canonicalPath || locator) return ['file', canonicalPath || locator]
  }
  if (source?.sourceType === 'allowlisted-url') {
    const locator = typeof source?.locator === 'string' ? source.locator.trim() : ''
    if (locator) return ['url', locator]
  }
  const contentHash = typeof source?.contentHash === 'string'
    ? source.contentHash.trim().toLowerCase()
    : ''
  if (contentHash) return ['content', contentHash]
  const recordId = typeof source?.id === 'string' ? source.id.trim() : ''
  if (recordId) return ['record', recordId]
  throw Object.assign(new Error('Bağımsız kaynak soyu için sunucu kaynak kimliği bulunamadı.'), {
    code: 'SOURCE_LINEAGE_REQUIRED',
  })
}

/**
 * Evidence independence is derived only from server-controlled source facts.
 * A caller-provided provenance.independenceGroupId remains audit metadata and
 * cannot manufacture an additional witness. Ollama evidence is canonicalized
 * from model identity; ordinary documents use their physical locator or stable
 * content/record identity before the runtime creates internal chunks.
 */
export function resolveIndependenceGroupId(source) {
  const provenance = source?.provenance
  if (sourceKind(source) === 'ollama' || String(provenance?.kind ?? '').toLowerCase().includes('ollama')) {
    const resolvedModelName = [provenance?.generationModel, provenance?.model]
      .find((value) => typeof value === 'string' && value.trim())
    const weightsDigest = [provenance?.weightsDigest, provenance?.modelDigest]
      .find((value) => typeof value === 'string' && value.trim())
    return canonicalOllamaIndependenceGroupId({ resolvedModelName, weightsDigest })
  }
  return `source-lineage:${stableRuntimeHash('physical-source-lineage-v1', physicalSourceLineageKey(source))}`
}

function textLabManifest(source) {
  const value = source.provenance?.textLabManifest
  return value?.format === 'text-lab-recompute-proof-v1' ? value : null
}

function runtimeTextLabConfig(sources) {
  const imported = sources
    .map(textLabManifest)
    .filter(Boolean)
    .map((manifest) => normalizeTextLabConfig(manifest.config))
  if (!imported.length) return { ...DEFAULT_TEXT_LAB_CONFIG }
  const fingerprint = stableRuntimeHash(imported[0])
  if (imported.some((config) => stableRuntimeHash(config) !== fingerprint)) {
    const error = new Error('Seçili Metin Laboratuvarı kayıtları farklı I/R temsil ayarları kullanıyor; tek çevrimde karıştırılamaz.')
    error.code = 'REPRESENTATION_SPACE_MISMATCH'
    throw error
  }
  return imported[0]
}

export function resolveRuntimeTextLabSelection(sources, fullConfig) {
  const request = resolvePlasticRegionRuntimeRequest(fullConfig)
  const adaptiveRepresentationSelected = request.profile === ADAPTIVE_TEXT_LAB_PROFILE_ID
  if (!request.requested && !adaptiveRepresentationSelected) {
    return {
      config: runtimeTextLabConfig(sources),
      plasticRegionRequest: request,
      reencodePlan: null,
    }
  }
  const config = resolveAdaptiveTextLabConfig(sources, fullConfig.plasticRegions)
  if (resolveTextLabRepresentationProfileId(config) !== ADAPTIVE_TEXT_LAB_PROFILE_ID) {
    const error = new Error('Plastik bölge çevrimi canonical adaptive-eighth-id-v1 TextLab profilini üretemedi.')
    error.code = 'REPRESENTATION_SPACE_MISMATCH'
    throw error
  }
  return {
    config,
    plasticRegionRequest: request,
    reencodePlan: planPlasticRegionReencodeInventory(sources, {
      targetProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      configOverrides: Object.fromEntries(Object.entries(config).filter(([key]) =>
        !['representationProfile', 'matrixSide', 'identityRatio'].includes(key))),
    }),
  }
}

function assertImportedIdentityProof(analysis, manifest) {
  if (!manifest) return false
  const mismatch = (message) => {
    const error = new Error(message)
    error.code = 'TEXT_LAB_IDENTITY_MISMATCH'
    return error
  }
  if (
    !analysis || !analysis.source || typeof analysis.source.text !== 'string' || !Array.isArray(analysis.nodes) ||
    !Array.isArray(manifest.identityProofs) ||
    manifest.identityProofs.length < 1 ||
    manifest.identityProofs.length > 1_600
  ) {
    throw mismatch('Metin Laboratuvarı kimlik kanıtı yapısal olarak geçersiz.')
  }
  if (
    analysis.sourceDigest !== manifest.sourceDigest ||
    analysis.representationSpaceId !== manifest.representationSpaceId ||
    stableRuntimeHash(normalizeTextLabConfig(analysis.config)) !== stableRuntimeHash(normalizeTextLabConfig(manifest.config))
  ) {
    throw mismatch('Metin Laboratuvarı kaydının kaynak, ayar veya temsil parmak izi yeniden üretilemedi.')
  }
  const expected = manifest.identityProofs
    .map((proof) => {
      if (
        !proof || !Number.isInteger(proof.start) || !Number.isInteger(proof.end) ||
        proof.start < 0 || proof.end <= proof.start || proof.end > analysis.source.text.length ||
        typeof proof.identityDigest !== 'string'
      ) throw mismatch('Metin Laboratuvarı kimlik kanıtındaki düğüm aralıkları geçersiz.')
      return [proof.start, proof.end, proof.identityDigest]
    })
    .sort(compareIdentityProof)
  const actual = analysis.nodes
    .map((node) => {
      if (
        node.start < 0 || node.end <= node.start || node.end > analysis.source.text.length ||
        analysis.source.text.slice(node.start, node.end) !== node.text ||
        digestVector(node.identity) !== node.identityDigest
      ) throw mismatch('Yeniden hesaplanan sabit I alanlarından biri kendi metni veya özetiyle uyuşmuyor.')
      return [node.start, node.end, node.identityDigest]
    })
    .sort(compareIdentityProof)
  if (
    expected.length !== actual.length ||
    expected.some((proof, index) => proof[0] !== actual[index][0] || proof[1] !== actual[index][1] || proof[2] !== actual[index][2])
  ) {
    throw mismatch('Sabit I alanı özgün Metin Laboratuvarı kimlik kanıtıyla eşleşmedi; kayıt eğitime alınmadı.')
  }
  return true
}

function compareIdentityProof(left, right) {
  return left[0] - right[0] || left[1] - right[1] || String(left[2]).localeCompare(String(right[2]))
}

function toSynthesisAnalysis(analysis) {
  return {
    format: analysis.format,
    version: analysis.version,
    runId: analysis.runId,
    independenceGroupId: analysis.independenceGroupId,
    sourceDigest: analysis.sourceDigest,
    identitySpaceId: analysis.identitySpaceId,
    contextSpaceId: analysis.contextSpaceId,
    representationSpaceId: analysis.representationSpaceId,
    source: analysis.source,
    config: analysis.config,
    layout: analysis.layout,
    nodes: analysis.nodes.map((node) => ({
      id: node.id,
      sourceId: node.sourceId,
      paragraphIndex: node.paragraphIndex,
      start: node.start,
      end: node.end,
      text: node.text,
      graphemeCount: node.graphemeCount,
      depth: node.depth,
      parentId: node.parentId,
      childIds: [...node.childIds],
      splitBoundary: node.splitBoundary,
      identity: node.identity,
      relation: node.relation,
      identityDigest: node.identityDigest,
    })),
  }
}

function toFamilyAnalysis(analysis) {
  return {
    format: analysis.format,
    version: analysis.version,
    runId: analysis.runId,
    createdAt: analysis.createdAt,
    independenceGroupId: analysis.independenceGroupId,
    sourceDigest: analysis.sourceDigest,
    identitySpaceId: analysis.identitySpaceId,
    contextSpaceId: analysis.contextSpaceId,
    representationSpaceId: analysis.representationSpaceId,
    source: analysis.source,
    config: analysis.config,
    layout: analysis.layout,
    rootIds: [...analysis.rootIds],
    nodes: analysis.nodes.map((node) => ({
      id: node.id,
      sourceId: node.sourceId,
      paragraphIndex: node.paragraphIndex,
      start: node.start,
      end: node.end,
      text: node.text,
      graphemeCount: node.graphemeCount,
      depth: node.depth,
      branch: node.branch,
      parentId: node.parentId,
      childIds: [...node.childIds],
      splitBoundary: node.splitBoundary,
      identity: node.identity,
      relation: node.relation,
      identityDigest: node.identityDigest,
      identityNorm: node.identityNorm,
      relationNorm: node.relationNorm,
      contextSupport: node.contextSupport,
      contextRevision: node.contextRevision,
      trace: { ...node.trace },
    })),
    habitProbabilities: analysis.habitProbabilities,
  }
}

function synthesisDecision(run, candidate) {
  return {
    id: `AUTO-${candidate.fingerprint.slice(0, 24)}`,
    format: 'noron-sentez-karari',
    version: 1,
    candidateId: candidate.id,
    candidateFingerprint: candidate.fingerprint,
    decision: 'approved',
    reviewedAt: new Date().toISOString(),
    runId: run.id,
    runFingerprint: run.fingerprint,
    representationSpaceId: run.representationSpaceId,
    analysisRunIds: [...run.analysisRunIds],
    sourceDigests: [...run.sourceDigests],
    config: { ...run.config },
    candidate,
  }
}

function topK(metrics, k) {
  return metrics?.topK?.find((entry) => entry.k === k)?.accuracy ?? 0
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
}

function familyUtilityScore(familyRun, emergentFamilies) {
  if (emergentFamilies?.length) {
    const eligible = emergentFamilies.filter((item) => item.status === 'eligible')
    const values = (eligible.length ? eligible : emergentFamilies)
      .map((item) => Number(item.holdoutUtility ?? item.predictiveUtility ?? item.score ?? 0))
      .filter(Number.isFinite)
    if (values.length) return clamp01(values.reduce((sum, value) => sum + Math.max(0, value), 0) / values.length)
  }
  const values = familyRun.utilityReports
    .map((report) => Number(report.deltaUtility))
    .filter(Number.isFinite)
  if (!values.length) return 0
  const positive = values.reduce((sum, value) => sum + Math.max(0, value), 0) / values.length
  return clamp01(1 - Math.exp(-positive * 8))
}

function deriveEmergentLanguageFamilies(trainingResult, familyRun, dataset) {
  if (Array.isArray(trainingResult.emergentLanguageFamilies)) {
    return trainingResult.emergentLanguageFamilies
  }
  return languageEngine.discoverEmergentLanguageFamilies(trainingResult.state, familyRun, dataset)
}

function createLanguageSources(records) {
  return records.map(({ analysis, physicalSource, chunk }) => ({
    sourceId: `${physicalSource.id}:chunk:${chunk.index}`,
    // A batch/campaign produced by one synthetic model is not a collection of
    // independent witnesses. Campaign provenance can collapse every output of
    // that lineage into one evidence group while ordinary human/file sources
    // continue to use their physical document id.
    independenceGroupId: resolveIndependenceGroupId(physicalSource),
    sourceDigest: analysis.sourceDigest,
    title: `${physicalSource.title} · parça ${chunk.index + 1}`,
    text: analysis.source.text,
    provenance: {
      kind: sourceKind(physicalSource),
      locator: physicalSource.locator ?? undefined,
      generatorModel: resolveGeneratorModel(physicalSource.provenance),
      license: physicalSource.provenance?.license ?? undefined,
    },
  }))
}

function languageGroupSplitAssignments(splitSources) {
  const assignments = new Map()
  for (const source of splitSources ?? []) {
    const existing = assignments.get(source.independenceGroupId)
    if (existing && existing !== source.split) {
      const error = new Error('Aynı bağımsız kaynak grubu birden fazla veri bölümüne dağıtılamaz.')
      error.code = 'DATA_SPLIT_DRIFT'
      throw error
    }
    assignments.set(source.independenceGroupId, source.split)
  }
  return [...assignments]
    .map(([independenceGroupId, split]) => ({ independenceGroupId, split }))
    .sort((left, right) => left.independenceGroupId.localeCompare(right.independenceGroupId))
}

function splitCompatibleFamilyContinuity(previous, discoveryGroupIds, evaluationGroupIds) {
  if (!previous) return null
  const priorSplit = previous.inputManifest?.dataSplit
  if (priorSplit?.mode !== 'declared-independent-groups') return null
  const discovery = new Set(discoveryGroupIds)
  const evaluation = new Set(evaluationGroupIds)
  if (priorSplit.discoveryIndependenceGroupIds.some((id) => evaluation.has(id))) return null
  if (priorSplit.evaluationIndependenceGroupIds.some((id) => discovery.has(id))) return null
  return previous
}

function syntheticRatio(sources) {
  if (!sources.length) return 0
  return sources.filter((source) => source.sourceType === 'ollama-generated').length / sources.length
}

/**
 * Keep a small, deterministic physical I/R working set for bounded research
 * thought observations. The full TextLab corpus is still released before the
 * language stage, so concept mining cannot reintroduce the old V8 OOM pattern.
 */
function retainThoughtObservationNeurons(analyses, familyRun, maximum = 192) {
  const available = new Set()
  for (const analysis of analyses ?? []) {
    for (const node of analysis.nodes ?? []) {
      available.add(rawMemberKey(analysis.sourceDigest, analysis.runId, node.id))
    }
  }
  const desired = []
  const seen = new Set()
  const add = (key) => {
    if (typeof key !== 'string' || !key || !available.has(key) || seen.has(key) || desired.length >= maximum) return
    seen.add(key)
    desired.push(key)
  }
  for (const entry of familyRun.routingPreview?.entries ?? []) add(entry.memberKey)
  for (const edge of [...(familyRun.neuronTransitions ?? [])]
    .sort((left, right) => Number(right.probability ?? 0) - Number(left.probability ?? 0) ||
      String(left.fromMemberKey).localeCompare(String(right.fromMemberKey)) ||
      String(left.toMemberKey).localeCompare(String(right.toMemberKey)))) {
    add(edge.fromMemberKey)
    add(edge.toMemberKey)
    if (desired.length >= maximum) break
  }
  const representativeMemberships = [...(familyRun.memberships ?? [])]
    .filter((membership) => membership.rank === 1)
    .sort((left, right) => String(left.familyId).localeCompare(String(right.familyId)) ||
      Number(right.weight ?? 0) - Number(left.weight ?? 0) || String(left.memberKey).localeCompare(String(right.memberKey)))
  for (const membership of representativeMemberships) add(membership.memberKey)

  const requested = new Set(desired)
  const byKey = new Map()
  for (const analysis of analyses ?? []) {
    for (const node of analysis.nodes ?? []) {
      const memberKey = rawMemberKey(analysis.sourceDigest, analysis.runId, node.id)
      if (!requested.has(memberKey)) continue
      byKey.set(memberKey, {
        memberKey,
        kind: 'raw',
        text: node.text,
        identity: new Float32Array(node.identity),
        relation: new Float32Array(node.relation),
        identityDigest: node.identityDigest,
        sourceDigest: analysis.sourceDigest,
        analysisRunId: analysis.runId,
        neuronId: node.id,
      })
    }
  }
  return desired.map((key) => byKey.get(key)).filter(Boolean)
}

function balancedThoughtObservationSources(dataset, maximum) {
  if (maximum <= 0) return []
  const rows = (dataset.sources ?? [])
    .filter((source) => source.text?.trim())
    .map((source) => ({
      ...source,
      sampleKey: stableRuntimeHash('macro-thought-source-v1', source.independenceGroupId, source.sourceDigest, source.text),
    }))
  const groups = new Map()
  for (const row of rows) {
    const values = groups.get(row.split) ?? []
    values.push(row)
    groups.set(row.split, values)
  }
  for (const values of groups.values()) values.sort((left, right) => left.sampleKey.localeCompare(right.sampleKey))
  const selected = []
  const holdoutSlots = Math.min(2, Math.max(0, maximum - 2))
  const trainSlots = maximum - holdoutSlots
  for (const value of (groups.get('train') ?? []).splice(0, trainSlots)) selected.push(value)
  for (const split of ['validation', 'test'].slice(0, holdoutSlots)) {
    const value = groups.get(split)?.shift()
    if (value) selected.push(value)
  }
  const preferred = ['train', 'validation', 'test']
  let rank = 0
  while (selected.length < maximum) {
    let added = false
    for (const split of preferred) {
      const value = groups.get(split)?.[rank]
      if (!value || selected.some((entry) => entry.sampleKey === value.sampleKey)) continue
      selected.push(value)
      added = true
      if (selected.length >= maximum) break
    }
    if (!added) break
    rank += 1
  }
  return selected
}

function traceChainForDominantBranch(result) {
  const dominant = [...(result.episode?.branches ?? [])]
    .sort((left, right) => Number(right.mass ?? 0) - Number(left.mass ?? 0) || String(left.id).localeCompare(String(right.id)))[0]
  if (!dominant) return []
  const byChild = new Map()
  for (const tick of result.ticks ?? []) {
    for (const branch of tick.branches ?? []) byChild.set(branch.branchId, { tick: tick.tick, branch })
  }
  const chain = []
  let cursor = dominant.id
  const seen = new Set()
  while (cursor && !seen.has(cursor)) {
    seen.add(cursor)
    const entry = byChild.get(cursor)
    if (!entry) break
    chain.push(entry)
    cursor = entry.branch.parentBranchId
  }
  return chain.reverse()
}

export function macroTraceFromThoughtResult(
  result,
  source,
  familyRun,
  semanticInputDigest,
  conceptMacroHead = null,
) {
  const dominant = [...(result.episode?.branches ?? [])]
    .sort((left, right) => Number(right.mass ?? 0) - Number(left.mass ?? 0) || String(left.id).localeCompare(String(right.id)))[0]
  if (!dominant) return null
  const chain = traceChainForDominantBranch(result)
  const transitionsByTick = new Map()
  for (const transition of dominant.transitions ?? []) {
    const row = transitionsByTick.get(transition.tick) ?? []
    row.push(transition)
    transitionsByTick.set(transition.tick, row)
  }
  const neuronById = new Map([
    ...(result.__catalogNeurons ?? []),
    ...(result.episode?.temporaryBirths ?? []),
  ].map((neuron) => [neuron.id, neuron]))
  const persistentIds = new Set((result.__catalogNeurons ?? []).map((neuron) => neuron.id))
  const macroById = new Map((conceptMacroHead?.macros ?? []).map((macro) => [macro.id, macro]))
  const stableSegments = [[]]
  const breakSegment = () => {
    if (stableSegments.at(-1).length) stableSegments.push([])
  }
  for (const entry of chain) {
    const transitions = transitionsByTick.get(entry.tick) ?? []
    if (entry.branch.beforeSignal.length !== entry.branch.afterSignal.length) {
      breakSegment()
      continue
    }

    const macroExecution = entry.branch.macroExecution
    if (macroExecution) {
      const macro = macroById.get(macroExecution.macroId)
      const expandedIds = macroExecution.expandedConstituentIds ?? []
      const transitionPathIsExact = transitions.length === Math.max(0, expandedIds.length - 1) &&
        transitions.every((transition, index) =>
          transition.fromNeuronId === expandedIds[index] &&
          transition.toNeuronId === expandedIds[index + 1])
      const executionIsBound = Boolean(
        conceptMacroHead &&
        macro &&
        macro.lifecycle === 'active' &&
        macroExecution.compiled === true &&
        macroExecution.macroHeadFingerprint === conceptMacroHead.fingerprint &&
        expandedIds.length >= 2 &&
        expandedIds.every((id) => persistentIds.has(id)) &&
        transitionPathIsExact,
      )
      if (!executionIsBound) {
        breakSegment()
        continue
      }
      stableSegments.at(-1).push({
        member: {
          kind: 'composite-neuron',
          id: macro.id,
          identityDigest: macro.expansionDigest,
          familyIds: [...new Set(macro.orderedExpansion.flatMap((member) => member.familyIds))].sort(),
        },
        signalBefore: [...entry.branch.beforeSignal],
        signalAfter: [...entry.branch.afterSignal],
      })
      continue
    }

    // A raw tick has exactly one durable transition. An unmatched branch,
    // newborn boundary or unexplained multi-edge jump must terminate adjacency;
    // otherwise motifs could be fabricated across a gap.
    if (transitions.length !== 1) {
      breakSegment()
      continue
    }
    const transition = transitions[0]
    const neuron = neuronById.get(transition.fromNeuronId)
    if (!neuron) {
      breakSegment()
      continue
    }
    // Episode-local newborn ids are deliberately excluded from the durable
    // macro vocabulary. Splitting at the boundary preserves real adjacency.
    if (!persistentIds.has(neuron.id)) {
      breakSegment()
      continue
    }
    stableSegments.at(-1).push({
      member: {
        kind: 'raw-neuron',
        id: neuron.id,
        identityDigest: neuron.identityDigest ?? null,
        familyIds: (neuron.familyMemberships ?? []).map((membership) => membership.familyId).sort(),
      },
      signalBefore: [...entry.branch.beforeSignal],
      signalAfter: [...entry.branch.afterSignal],
    })
  }
  const steps = stableSegments
    .map((segment, index) => ({ segment, index }))
    .sort((left, right) => right.segment.length - left.segment.length || left.index - right.index)[0]?.segment ?? []
  if (steps.length < 2) return null
  const id = `MTRACE-${stableRuntimeHash(
    'macro-trace-v1', source.sourceId, semanticInputDigest, result.episode.runKey, steps.map((step) => step.member.id),
  ).slice(0, 32)}`
  return {
    format: 'macro-trace-v1',
    version: 1,
    id,
    representationSpaceId: familyRun.representationSpaceId,
    familyRunFingerprint: familyRun.fingerprint,
    sourceId: source.sourceId,
    sourceDigest: source.sourceDigest ?? null,
    independenceGroupId: source.independenceGroupId,
    semanticInputDigest,
    episodeId: result.episode.episodeId,
    runKey: result.episode.runKey,
    split: source.split,
    steps,
  }
}

async function collectBoundedThoughtMacroTraces(context, state, config, conceptMacroHead = null) {
  const physical = state.thoughtObservationNeurons ?? []
  if (physical.length < 2 || config.thoughtSamplesPerCycle <= 0) return { traces: [], diagnostics: [] }
  const catalog = createThoughtCatalogFromResolved(physical, state.familyRun, MACRO_OBSERVATION_THOUGHT_ENGINE.config)
  const baseModelFingerprint = stableRuntimeHash(
    RUNTIME_VERSION,
    'macro-thought-research-base-v1',
    state.familyRun.fingerprint,
    state.trainingResult.state.fingerprint,
    catalog.model.fingerprint,
  )
  const thoughtHead = createThoughtHead(catalog.model, null, baseModelFingerprint)
  const representationProfile = state.textLabConfig?.representationProfile === ADAPTIVE_TEXT_LAB_PROFILE_ID
    ? ADAPTIVE_TEXT_LAB_PROFILE_ID
    : null
  const sources = balancedThoughtObservationSources(state.dataset, config.thoughtSamplesPerCycle)
  const traces = []
  const diagnostics = []
  for (let index = 0; index < sources.length; index += 1) {
    await context.checkpoint(`concept-macro-thought-${index + 1}`)
    budgetGuard(context, state.runtimeStartedAt)
    const source = sources[index]
    const inputText = source.text.slice(0, 4_000)
    const semanticInputDigest = stableRuntimeHash(
      'macro-semantic-input-v1', source.independenceGroupId, source.sourceDigest, inputText,
    )
    const episodeId = `CME-${stableRuntimeHash('macro-thought-episode-v1', semanticInputDigest, baseModelFingerprint).slice(0, 28)}`
    const seed = Number.parseInt(stableRuntimeHash('macro-thought-seed-v1', semanticInputDigest).slice(0, 8), 16) | 0
    try {
      const result = await MACRO_OBSERVATION_THOUGHT_ENGINE.runEpisode({
        format: 'thought-engine-request',
        version: 1,
        episodeId,
        inputText,
        seed,
        maxTicks: config.thoughtTicksPerSample,
        strategy: 'deterministic',
        learningMode: 'off',
        teacherMode: 'off',
        sourceGroupId: source.independenceGroupId,
        candidateMemberKeys: catalog.model.neurons.slice(0, 32).map((neuron) => neuron.id),
        base: {
          snapshotId: `research-macro:${context.jobId}`,
          snapshotStatus: 'research',
          selectedBy: 'bounded-14b-thought-observation',
          activeAtStart: false,
          commitEligible: false,
          artifactManifestHash: state.inputManifestHash,
          baseModelFingerprint,
          familyRunFingerprint: state.familyRun.fingerprint,
          languageStateFingerprint: state.trainingResult.state.fingerprint,
          thoughtHeadFingerprint: thoughtHead.fingerprint,
          thoughtHeadRevision: thoughtHead.revision,
          conceptMacroHeadFingerprint: conceptMacroHead?.fingerprint ?? null,
          mode: 'research',
          ...(representationProfile ? { representationProfile } : {}),
        },
      }, {
        signal: context.signal,
        neurons: catalog,
        familyRun: state.familyRun,
        languageState: state.trainingResult.state,
        thoughtHead,
        conceptMacroHead,
        ...(representationProfile ? { representationProfile } : {}),
        baseDiagnostics: { mode: 'bounded-pipeline-observation', sourceId: source.sourceId },
        shouldCancel: () => budgetGuard(context, state.runtimeStartedAt),
      })
      Object.defineProperty(result, '__catalogNeurons', { value: catalog.model.neurons, enumerable: false })
      const trace = macroTraceFromThoughtResult(
        result,
        source,
        state.familyRun,
        semanticInputDigest,
        conceptMacroHead,
      )
      if (trace) traces.push(trace)
      diagnostics.push({
        sourceId: source.sourceId,
        split: source.split,
        tickCount: result.episode.tick,
        traceStepCount: trace?.steps.length ?? 0,
        branchCount: result.episode.branches.length,
        status: trace ? 'accepted' : 'too-short',
      })
    } catch (error) {
      if (context.signal.aborted) throw error
      diagnostics.push({
        sourceId: source.sourceId,
        split: source.split,
        tickCount: 0,
        traceStepCount: 0,
        branchCount: 0,
        status: 'rejected',
        code: String(error?.code ?? 'THOUGHT_OBSERVATION_FAILED').slice(0, 100),
        message: String(error?.message ?? error).slice(0, 300),
      })
    }
    await context.progress('conceptMacros', (index + 1) / Math.max(1, sources.length))
  }
  return { traces, diagnostics }
}

/**
 * Plastic-region observations are training-only and source-balanced. Multiple
 * chunks (or many prompts) from one Ollama lineage can supply more ticks, but
 * never masquerade as independent witnesses inside one cycle.
 */
export function selectPlasticRegionThoughtSources(dataset, maximum) {
  if (!Number.isInteger(maximum) || maximum < 0) {
    throw new RangeError('Plastik bölge thought örneği sınırı negatif olmayan tam sayı olmalıdır.')
  }
  if (maximum === 0) return []
  const ranked = (dataset?.sources ?? [])
    .filter((source) => source.split === 'train' && source.text?.trim())
    .map((source) => ({
      ...source,
      sampleKey: stableRuntimeHash(
        'plastic-region-thought-source-v1',
        source.independenceGroupId,
        source.sourceDigest,
        source.text,
      ),
    }))
    .sort((left, right) => left.sampleKey.localeCompare(right.sampleKey) ||
      String(left.sourceId).localeCompare(String(right.sourceId)))
  const groups = new Map()
  for (const source of ranked) {
    const values = groups.get(source.independenceGroupId) ?? []
    values.push(source)
    groups.set(source.independenceGroupId, values)
  }
  const orderedGroups = [...groups.entries()].sort(([left], [right]) =>
    stableRuntimeHash('plastic-region-source-group-v1', left)
      .localeCompare(stableRuntimeHash('plastic-region-source-group-v1', right)) ||
    String(left).localeCompare(String(right)))
  const selected = []
  // Deterministic stratified round-robin gives each group one turn before a
  // second sample is taken. Repeated chunks still retain the same group id,
  // so they add observations without manufacturing independent votes.
  for (let round = 0; selected.length < maximum; round += 1) {
    let added = false
    for (const [, sources] of orderedGroups) {
      const source = sources[round]
      if (!source) continue
      selected.push(source)
      added = true
      if (selected.length >= maximum) break
    }
    if (!added) break
  }
  return selected
}

function effectivePlasticRegionObservationBudget(serviceConfig, fullConfig) {
  const configured = serviceConfig.maximumObservationsPerBatch *
    serviceConfig.maximumObservationBatchesPerCycle
  const possible = serviceConfig.thoughtSamplesPerCycle *
    serviceConfig.thoughtTicksPerSample * 56
  // The serialized form is larger than the eight-number behavior itself.
  // Reserve one eighth of the neuron cache for this transient observation set.
  const safeBytes = Number(fullConfig.resourcePolicy?.neuronCacheMb ?? 384) * 1024 * 1024 / 8
  const estimatedObservationBytes = 512
  const resourceBound = Math.floor(safeBytes / estimatedObservationBytes / 56) * 56
  return Math.max(56, Math.min(configured, possible, resourceBound, 65_536))
}

async function collectPlasticRegionThoughtObservations(context, state, serviceConfig, atlas, expectedBase) {
  const physical = state.thoughtObservationNeurons ?? []
  if (physical.length < 2 || serviceConfig.thoughtSamplesPerCycle <= 0) {
    return { observations: [], diagnostics: [], selectedSourceCount: 0, effectiveObservationLimit: 0 }
  }
  const fullConfig = context.config ?? context.store.getConfig()
  const effectiveObservationLimit = effectivePlasticRegionObservationBudget(serviceConfig, fullConfig)
  const catalog = createThoughtCatalogFromResolved(
    physical,
    state.familyRun,
    PLASTIC_REGION_OBSERVATION_THOUGHT_ENGINE.config,
  )
  if (catalog.model.layout.identityLength !== 512 || catalog.model.layout.relationLength !== 3_584) {
    const error = new Error('Plastik bölge thought kataloğu canonical 8/56 yerleşiminde değil.')
    error.code = 'REPRESENTATION_SPACE_MISMATCH'
    throw error
  }
  const baseModelFingerprint = stableRuntimeHash(
    RUNTIME_VERSION,
    'plastic-region-thought-research-base-v1',
    state.familyRun.fingerprint,
    state.trainingResult.state.fingerprint,
    catalog.model.fingerprint,
    atlas.fingerprint,
  )
  const thoughtHead = createThoughtHead(catalog.model, null, baseModelFingerprint)
  const sources = selectPlasticRegionThoughtSources(state.dataset, serviceConfig.thoughtSamplesPerCycle)
  const observations = []
  const diagnostics = []
  for (let index = 0; index < sources.length; index += 1) {
    if (effectiveObservationLimit - observations.length < 56) break
    await context.checkpoint(`plastic-region-thought-${index + 1}`)
    budgetGuard(context, state.runtimeStartedAt)
    const source = sources[index]
    const inputText = source.text.slice(0, 4_000)
    const semanticInputDigest = stableRuntimeHash(
      'plastic-region-semantic-input-v1',
      source.independenceGroupId,
      source.sourceDigest,
      inputText,
    )
    const episodeId = `PRE-${stableRuntimeHash(
      'plastic-region-thought-episode-v1',
      semanticInputDigest,
      baseModelFingerprint,
      serviceConfig.mode,
    ).slice(0, 28)}`
    const seed = Number.parseInt(
      stableRuntimeHash('plastic-region-thought-seed-v1', semanticInputDigest).slice(0, 8),
      16,
    ) | 0
    try {
      const result = await PLASTIC_REGION_OBSERVATION_THOUGHT_ENGINE.runEpisode({
        format: 'thought-engine-request',
        version: 1,
        episodeId,
        inputText,
        seed,
        maxTicks: serviceConfig.thoughtTicksPerSample,
        strategy: 'deterministic',
        learningMode: 'off',
        teacherMode: 'off',
        sourceGroupId: source.independenceGroupId,
        candidateMemberKeys: catalog.model.neurons.slice(0, 32).map((neuron) => neuron.id),
        adaptiveRegion: {
          mode: serviceConfig.mode,
          atlasFingerprint: atlas.fingerprint,
          minimumTrace: serviceConfig.minimumTrace,
        },
        base: {
          snapshotId: expectedBase.snapshotId ?? `research-plastic-region:${context.jobId}`,
          snapshotStatus: expectedBase.snapshotId ? 'active' : 'research',
          selectedBy: 'bounded-plastic-region-thought-observation',
          activeAtStart: Boolean(expectedBase.snapshotId),
          commitEligible: false,
          artifactManifestHash: state.inputManifestHash,
          baseModelFingerprint,
          familyRunFingerprint: state.familyRun.fingerprint,
          languageStateFingerprint: state.trainingResult.state.fingerprint,
          thoughtHeadFingerprint: thoughtHead.fingerprint,
          thoughtHeadRevision: thoughtHead.revision,
          conceptMacroHeadFingerprint: null,
          mode: 'research',
          representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
        },
      }, {
        signal: context.signal,
        neurons: catalog,
        familyRun: state.familyRun,
        languageState: state.trainingResult.state,
        thoughtHead,
        conceptMacroHead: null,
        adaptiveRegionAtlas: atlas,
        representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
        baseDiagnostics: {
          mode: 'bounded-plastic-region-observation',
          sourceId: source.sourceId,
          split: source.split,
          validationEligible: false,
        },
        shouldCancel: () => budgetGuard(context, state.runtimeStartedAt),
      })
      const extracted = extractPlasticRegionObservationsFromThoughtResult({
        result,
        source,
        atlasFingerprint: atlas.fingerprint,
        maximumObservations: effectiveObservationLimit - observations.length,
      })
      observations.push(...extracted.observations)
      diagnostics.push({
        sourceId: source.sourceId,
        sourceGroupId: source.independenceGroupId,
        split: source.split,
        tickCount: result.episode.tick,
        transitionCount: extracted.transitionCount,
        observationCount: extracted.observations.length,
        status: extracted.transitionCount > 0 ? 'accepted' : 'no-stable-transition',
      })
    } catch (error) {
      if (context.signal.aborted || [
        'REPRESENTATION_SPACE_MISMATCH',
        'STALE_REGION_ATLAS_BASE',
        'OLLAMA_LINEAGE_MISMATCH',
        'INVALID_PLASTIC_REGION_TICK_OBSERVATION',
      ].includes(error?.code)) throw error
      diagnostics.push({
        sourceId: source.sourceId,
        sourceGroupId: source.independenceGroupId,
        split: source.split,
        tickCount: 0,
        transitionCount: 0,
        observationCount: 0,
        status: 'rejected',
        code: String(error?.code ?? 'PLASTIC_REGION_THOUGHT_FAILED').slice(0, 100),
        message: String(error?.message ?? error).slice(0, 300),
      })
    }
    await context.progress('plasticRegions', (index + 1) / Math.max(1, sources.length))
  }
  return {
    observations,
    diagnostics,
    selectedSourceCount: sources.length,
    effectiveObservationLimit,
  }
}

function samePlasticRegionBase(left, right) {
  return ['snapshotId', 'atlasArtifactId', 'atlasContentHash', 'atlasFingerprint']
    .every((field) => (left[field] ?? null) === (right[field] ?? null))
}

function createStructuralSemanticProbeThoughtRunner(context, state, serviceConfig, atlas, expectedBase) {
  const physical = state.thoughtObservationNeurons ?? []
  if (physical.length < 2) return null
  const catalog = createThoughtCatalogFromResolved(
    physical,
    state.familyRun,
    PLASTIC_REGION_OBSERVATION_THOUGHT_ENGINE.config,
  )
  if (catalog.model.layout.identityLength !== 512 || catalog.model.layout.relationLength !== 3_584) {
    const error = new Error('Anlamsal probe thought kataloğu canonical 8/56 yerleşiminde değil.')
    error.code = 'REPRESENTATION_SPACE_MISMATCH'
    throw error
  }
  const baseModelFingerprint = stableRuntimeHash(
    RUNTIME_VERSION,
    'plastic-region-semantic-probe-base-v1',
    state.familyRun.fingerprint,
    state.trainingResult.state.fingerprint,
    catalog.model.fingerprint,
    atlas.fingerprint,
    STRUCTURAL_SEMANTIC_PROBE_PRODUCER_FINGERPRINT,
  )
  const thoughtHead = createThoughtHead(catalog.model, null, baseModelFingerprint)
  const candidateMemberKeys = catalog.model.neurons.slice(0, 32).map((neuron) => neuron.id)
  return async ({ benchmarkCase, request, atlas: requestedAtlas }) => {
    if (requestedAtlas.fingerprint !== atlas.fingerprint || request.learningMode !== 'off' ||
        request.teacherMode !== 'off' || request.adaptiveRegion?.mode !== 'shadow' ||
        request.trainingWritesAllowed !== false || request.routingMutationAllowed !== false) {
      const error = new Error('Anlamsal probe çalıştırıcısı salt-okunur shadow sözleşmesinden saptı.')
      error.code = 'SEMANTIC_PROBE_ROUTING_ISOLATION_FAILED'
      throw error
    }
    budgetGuard(context, state.runtimeStartedAt)
    const episodeId = `SPR-${stableRuntimeHash(
      'plastic-region-semantic-probe-episode-v1',
      benchmarkCase.id,
      baseModelFingerprint,
    ).slice(0, 28)}`
    const seed = Number.parseInt(stableRuntimeHash(
      'plastic-region-semantic-probe-seed-v1',
      benchmarkCase.id,
      baseModelFingerprint,
    ).slice(0, 8), 16) | 0
    return PLASTIC_REGION_OBSERVATION_THOUGHT_ENGINE.runEpisode({
      format: 'thought-engine-request',
      version: 1,
      episodeId,
      inputText: benchmarkCase.prompt,
      seed,
      maxTicks: serviceConfig.thoughtTicksPerSample,
      strategy: 'deterministic',
      learningMode: 'off',
      teacherMode: 'off',
      sourceGroupId: benchmarkCase.sourceGroupId,
      candidateMemberKeys,
      adaptiveRegion: {
        mode: 'shadow',
        atlasFingerprint: atlas.fingerprint,
        minimumTrace: serviceConfig.minimumTrace,
      },
      base: {
        snapshotId: expectedBase.snapshotId,
        snapshotStatus: 'active',
        selectedBy: 'deterministic-structural-semantic-probe',
        activeAtStart: true,
        commitEligible: false,
        artifactManifestHash: state.inputManifestHash,
        baseModelFingerprint,
        familyRunFingerprint: state.familyRun.fingerprint,
        languageStateFingerprint: state.trainingResult.state.fingerprint,
        thoughtHeadFingerprint: thoughtHead.fingerprint,
        thoughtHeadRevision: thoughtHead.revision,
        conceptMacroHeadFingerprint: null,
        mode: 'research',
        representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      },
    }, {
      signal: context.signal,
      neurons: catalog,
      familyRun: state.familyRun,
      languageState: state.trainingResult.state,
      thoughtHead,
      conceptMacroHead: null,
      adaptiveRegionAtlas: atlas,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      baseDiagnostics: {
        mode: 'deterministic-structural-semantic-probe',
        benchmarkCaseId: benchmarkCase.id,
        validationEligible: false,
        trainingWritesAllowed: false,
      },
      shouldCancel: () => budgetGuard(context, state.runtimeStartedAt),
    })
  }
}

async function plasticRegions(_job, context, state) {
  const fullConfig = context.config ?? context.store.getConfig()
  const request = state.plasticRegionRequest ?? resolvePlasticRegionRuntimeRequest(fullConfig)
  if (!request.requested) return { verified: true, disabled: true, mode: 'off' }
  const serviceConfig = fullConfig.plasticRegions
  if (
    request.profile !== ADAPTIVE_TEXT_LAB_PROFILE_ID ||
    resolveTextLabRepresentationProfileId(state.textLabConfig) !== ADAPTIVE_TEXT_LAB_PROFILE_ID ||
    state.familyRun.representationSpaceId !== state.representationSpaceId
  ) {
    const error = new Error('Plastik bölge aşaması canonical TextLab/FamilyRun temsil bağını doğrulayamadı.')
    error.code = 'REPRESENTATION_SPACE_MISMATCH'
    throw error
  }

  const expectedBase = currentPlasticRegionBase(context.store)
  const atlas = loadPlasticRegionBaseAtlas(
    context.store,
    state.representationSpaceId,
    serviceConfig.maximumMembershipsPerPage,
  )
  const carriedRegionArtifactIds = expectedBase.atlasArtifactId
    ? plasticRegionCarryForwardArtifactIds(context.store)
    : []
  for (const artifactId of carriedRegionArtifactIds) appendArtifact(state, context.store.getArtifact(artifactId))
  if (!samePlasticRegionBase(expectedBase, currentPlasticRegionBase(context.store))) {
    const error = new Error('Plastik bölge atlas tabanı gözlem başlamadan değişti.')
    error.code = 'STALE_REGION_ATLAS_BASE'
    throw error
  }
  const collected = await collectPlasticRegionThoughtObservations(
    context,
    state,
    { ...serviceConfig, mode: request.mode },
    atlas,
    expectedBase,
  )
  const priorEvidenceHeads = readPlasticRegionEvidenceHeads(context.store, {
    representationSpaceId: state.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: expectedBase.snapshotId ? 'cas-active-base' : 'research-only',
    expectedBase,
    maximumHeads: serviceConfig.maximumEvidenceHeadsPerCycle,
    maximumArtifactBytes: serviceConfig.maximumArtifactBytes,
    maximumTotalBytes: Math.max(16 * 1024 * 1024, serviceConfig.maximumArtifactBytes),
  })
  const cycle = buildPlasticRegionCycle({
    mode: request.mode,
    baseAtlas: atlas,
    representationSpaceId: state.representationSpaceId,
    familyRunFingerprint: state.familyRun.fingerprint,
    inputManifestHash: state.inputManifestHash,
    observations: collected.observations,
    priorEvidenceHeads,
    evidenceAuthority: expectedBase.snapshotId ? 'cas-active-base' : 'research-only',
    maximumCumulativeObservations: serviceConfig.maximumCumulativeObservations,
    maximumPerBatch: serviceConfig.maximumObservationsPerBatch,
    maximumBatches: serviceConfig.maximumObservationBatchesPerCycle,
    maximumRegions: serviceConfig.maximumRegions,
    maximumMembershipsPerPage: serviceConfig.maximumMembershipsPerPage,
    maximumActiveGates: serviceConfig.maximumActiveGates,
  })
  const persisted = persistPlasticRegionCycle(context, cycle, { expectedBase })
  for (const artifact of persisted.attachedArtifacts) appendArtifact(state, artifact)
  let semanticProbes = {
    enabled: Boolean(serviceConfig.semanticProbesEnabled),
    status: 'unknown',
    reason: serviceConfig.semanticProbesEnabled ? 'probe-base-unavailable' : 'disabled',
    artifactsWritten: 0,
    artifactIds: [],
  }
  if (serviceConfig.semanticProbesEnabled && expectedBase.snapshotId) {
    const runCase = createStructuralSemanticProbeThoughtRunner(
      context,
      state,
      serviceConfig,
      atlas,
      expectedBase,
    )
    if (runCase) {
      semanticProbes = {
        enabled: true,
        ...await produceAndPersistStructuralSemanticProbes({
          context,
          snapshot: context.store.getSnapshot(expectedBase.snapshotId),
          runCase,
          maximumFeatures: Math.min(128, serviceConfig.maximumRegions),
          limits: {
            maximumRecords: 512,
            maximumFeatures: Math.min(128, serviceConfig.maximumRegions),
            maximumLineageTokensPerRecord: 8,
            maximumArtifactBytes: serviceConfig.maximumArtifactBytes,
          },
        }),
      }
      for (const artifactId of semanticProbes.artifactIds) appendArtifact(state, context.store.getArtifact(artifactId))
    } else {
      semanticProbes.reason = 'insufficient-thought-neuron-catalog'
    }
  }
  state.plasticRegionSummary = {
    mode: request.mode,
    safetyMode: persisted.safetyMode,
    observationCount: collected.observations.length,
    cumulativeObservationCount: cycle.evidenceHead.observationCount,
    cumulativeTransitionCount: cycle.evidenceHead.transitionCount,
    cycleIndependentSourceGroupCount: new Set(collected.observations.map((value) => value.sourceGroupId)).size,
    independentSourceGroupCount: cycle.evidenceHead.sourceGroupIds.length,
    proposalCreated: Boolean(cycle.proposal),
    proposalEligible: cycle.proposal?.eligible ?? false,
    atlasActivated: false,
    semanticProbeStatus: semanticProbes.status,
    semanticProbeArtifactCount: semanticProbes.artifactsWritten,
  }
  return {
    // Verification means the observation safety contract ran successfully; it
    // does not approve or activate the detached atlas candidate.
    verified: true,
    mode: request.mode,
    safetyMode: persisted.safetyMode,
    representationProfile: request.profile,
    representationSpaceId: state.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    observationCount: collected.observations.length,
    cumulativeObservationCount: cycle.evidenceHead.observationCount,
    cumulativeTransitionCount: cycle.evidenceHead.transitionCount,
    evidenceRevision: cycle.evidenceHead.revision,
    carriedRegionArtifactIds,
    regionBase: { ...expectedBase },
    observationBatchCount: cycle.observationBatches.length,
    cycleIndependentSourceGroupCount: state.plasticRegionSummary.cycleIndependentSourceGroupCount,
    independentSourceGroupCount: state.plasticRegionSummary.independentSourceGroupCount,
    selectedSourceCount: collected.selectedSourceCount,
    effectiveObservationLimit: collected.effectiveObservationLimit,
    proposalArtifactId: persisted.proposalArtifact?.id ?? null,
    atlasCandidateArtifactId: persisted.detachedAtlasCandidateArtifact?.id ?? null,
    proposalEligible: cycle.proposal?.eligible ?? false,
    atlasActivated: false,
    reencodePlanFingerprint: state.plasticRegionReencodePlan?.fingerprint ?? null,
    reencodePlanCounts: state.plasticRegionReencodePlan?.counts ?? {},
    diagnostics: collected.diagnostics,
    semanticProbes: {
      enabled: semanticProbes.enabled,
      status: semanticProbes.status,
      reason: semanticProbes.reason,
      artifactsWritten: semanticProbes.artifactsWritten,
      artifactIds: semanticProbes.artifactIds,
      summaryFingerprint: semanticProbes.summary?.fingerprint ?? null,
    },
  }
}

function conceptMacroEngineConfig(serviceConfig) {
  return createMacroConfig({
    minimumMotifLength: serviceConfig.minimumLength,
    maximumMotifLength: serviceConfig.maximumLength,
    maximumHeavyHitters: serviceConfig.maximumCandidates,
    maximumComposites: serviceConfig.maximumCandidates,
    maximumVectorDimension: 4_096,
    minimumProposalObservations: serviceConfig.proposalMinimumOccurrences,
    minimumIndependentGroups: serviceConfig.proposalMinimumSourceGroups,
    minimumEffectCohesion: serviceConfig.minimumCohesion,
    minimumEntryCompatibility: serviceConfig.minimumCompletionProbability,
    minimumActivationRecall: serviceConfig.minimumCompletionProbability,
    minimumShadowFidelity: serviceConfig.minimumShadowFidelity,
    maximumFalseActivationRate: serviceConfig.maximumFalseActivationRate,
    maximumNestingDepth: Math.min(3, serviceConfig.maximumNestedDepth),
  })
}

function conceptMacroReferenceCatalog(familyRun) {
  const familiesByMember = new Map()
  for (const membership of familyRun.memberships ?? []) {
    const values = familiesByMember.get(membership.memberKey) ?? []
    values.push(membership.familyId)
    familiesByMember.set(membership.memberKey, values)
  }
  return {
    representationSpaceId: familyRun.representationSpaceId,
    familyRunFingerprint: familyRun.fingerprint,
    members: (familyRun.profiles ?? [])
      .filter((profile) => profile.memberKey && profile.identityDigest)
      .map((profile) => ({
        id: profile.memberKey,
        identityDigest: profile.identityDigest,
        familyIds: [...new Set(familiesByMember.get(profile.memberKey) ?? [])].sort(),
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
  }
}

function estimatedMacroWindows(stepCount, minimumLength, maximumLength) {
  let windows = 0
  for (let length = minimumLength; length <= maximumLength; length += 1) {
    windows += Math.max(0, stepCount - length + 1) * 2
  }
  return windows
}

/** Enforces the service observation budget before the miner materializes windows. */
function boundedMacroTrainingTraces(traces, config, maximumObservations) {
  const output = []
  let remaining = maximumObservations
  for (const trace of [...traces].sort((left, right) => left.id.localeCompare(right.id))) {
    if (remaining <= 0) break
    const fullCost = estimatedMacroWindows(
      trace.steps.length,
      config.minimumMotifLength,
      config.maximumMotifLength,
    )
    if (fullCost <= remaining) {
      output.push(trace)
      remaining -= fullCost
      continue
    }
    let retained = 0
    for (let length = config.minimumMotifLength; length <= trace.steps.length; length += 1) {
      const cost = estimatedMacroWindows(length, config.minimumMotifLength, config.maximumMotifLength)
      if (cost > remaining) break
      retained = length
    }
    if (retained >= config.minimumMotifLength) {
      output.push({ ...trace, steps: trace.steps.slice(0, retained) })
      remaining -= estimatedMacroWindows(
        retained,
        config.minimumMotifLength,
        config.maximumMotifLength,
      )
    }
  }
  return output
}

function clippedMacroEffect(signal, effect) {
  if (signal.length !== effect.length) return [...signal]
  const mixed = Float32Array.from(signal, (value, index) =>
    Math.fround(Math.max(-1, Math.min(1, value + effect[index]))))
  return Array.from(projectThoughtVector(mixed, 1))
}

function sameFirstMacroMember(composite, step) {
  const first = composite.orderedExpansion[0]
  if (!first || !step?.member) return false
  if (composite.kind === 'sequence-macro') {
    return first.id === step.member.id && first.identityDigest === step.member.identityDigest
  }
  return first.familyIds.some((familyId) => step.member.familyIds.includes(familyId))
}

function shadowCasesForComposite(composite, traces, activationThreshold, maximumCases = 256) {
  const length = composite.orderedExpansion.length
  const cases = []
  for (const trace of [...traces].sort((left, right) => left.id.localeCompare(right.id))) {
    for (let start = 0; start + length <= trace.steps.length; start += 1) {
      const steps = trace.steps.slice(start, start + length)
      const expansion = steps.map((step) => step.member)
      const signature = composite.kind === 'sequence-macro'
        ? computeExactSignature(expansion)
        : computeFamilySignature(expansion)
      const expectedSignature = composite.kind === 'sequence-macro'
        ? composite.exactSignature
        : composite.familySignature
      const shouldActivate = signature === expectedSignature
      const entry = steps[0].signalBefore
      // terminal-anchor-v1: the last member remains current and is processed
      // by the following raw tick, therefore it must not be applied twice.
      const expandedOutput = steps.at(-1).signalBefore
      const entryCompatibility = Math.max(
        0,
        macroCosineSimilarity(entry, composite.entryPrototype.vector),
      )
      // Runtime selection may inspect the current member and current signal,
      // but never the future continuation. This makes negative-prefix cases a
      // real false-activation test instead of granting the shadow oracle.
      const macroActivated = sameFirstMacroMember(composite, steps[0]) &&
        entryCompatibility >= activationThreshold
      cases.push({
        id: macroDigest('MSHADOWCASE', composite.id, trace.id, start, length),
        split: trace.split,
        independenceGroupId: trace.independenceGroupId,
        shouldActivate,
        macroActivated,
        expandedOutput: [...expandedOutput],
        macroOutput: macroActivated
          ? clippedMacroEffect(entry, composite.effectPrototype)
          : [...expandedOutput],
        expandedCost: Math.max(1, length - 1),
        macroCost: macroActivated ? 1 : length,
      })
      if (cases.length >= maximumCases) return cases
    }
  }
  return cases
}

async function collect(job, context, state) {
  state.runtimeStartedAt = Date.now()
  const result = await DEFAULT_STAGE_HANDLERS.collect(job, context, state)
  state.syntheticRatio = syntheticRatio(state.sources)
  return {
    ...result,
    syntheticRatio: state.syntheticRatio,
    warnings: state.syntheticRatio > 0.25
      ? ['Yapay öğretmen verisi bu partinin %25’ini aşıyor; kaynak-holdout kapısı sonucu ayrıca denetleyecek.']
      : [],
  }
}

async function textLab(_job, context, state) {
  const fullConfig = context.config ?? context.store.getConfig()
  const selection = resolveRuntimeTextLabSelection(state.sources, fullConfig)
  const config = selection.config
  state.textLabConfig = config
  state.plasticRegionRequest = selection.plasticRegionRequest
  state.plasticRegionReencodePlan = selection.reencodePlan
  const chunks = state.sources.flatMap((physicalSource) =>
    chunkCanonicalText(
      physicalSource.content,
      textLabManifest(physicalSource) ? Number.MAX_SAFE_INTEGER : CORPUS_RUNTIME_LIMITS.maximumChunkCharacters,
    ).map((chunk) => ({ physicalSource, chunk })),
  )
  if (!chunks.length) throw new Error('Kaynaklarda çözümlenebilir metin parçası bulunamadı.')

  const records = []
  for (let index = 0; index < chunks.length; index += 1) {
    const { physicalSource, chunk } = chunks[index]
    const independenceGroupId = resolveIndependenceGroupId(physicalSource)
    await context.checkpoint(`text-lab-chunk-${index + 1}`)
    budgetGuard(context, state.runtimeStartedAt)
    const contentHash = stableRuntimeHash(
      RUNTIME_VERSION,
      'text-lab-analysis',
      physicalSource.id,
      physicalSource.contentHash,
      chunk.index,
      chunk.start,
      chunk.end,
      independenceGroupId,
      config,
      textLabManifest(physicalSource),
    )
    const cached = findCachedArtifact(context.store, 'text-lab-analysis', contentHash)
    let analysis
    let artifact
    if (cached) {
      artifact = cached
      analysis = restoreArtifact(cached, 'text-lab-analysis')
      assertImportedIdentityProof(analysis, textLabManifest(physicalSource))
    } else {
      const localSourceId = `${physicalSource.id}:chunk:${chunk.index}`
      analysis = await analyzeText({
        id: localSourceId,
        title: `${physicalSource.title} · parça ${chunk.index + 1}`,
        rawText: chunk.text,
      }, config, {
        shouldCancel: () => budgetGuard(context, state.runtimeStartedAt),
      })
      // The local id keeps run/neuron ids unique. Evidence independence uses
      // the provenance lineage resolved above, written after deterministic encoding.
      analysis.source.id = physicalSource.id
      for (const node of analysis.nodes) node.sourceId = physicalSource.id
      analysis.independenceGroupId = independenceGroupId
      const identityProofVerified = assertImportedIdentityProof(analysis, textLabManifest(physicalSource))
      ;({ artifact } = persistArtifact(context, {
        kind: 'text-lab-analysis',
        contentHash,
        value: analysis,
        metadata: {
          physicalSourceId: physicalSource.id,
          independenceGroupId,
          sourceDigest: analysis.sourceDigest,
          chunkIndex: chunk.index,
          chunkStart: chunk.start,
          chunkEnd: chunk.end,
          neuronCount: analysis.nodes.length,
          identityProofVerified,
          representationProfile: resolveTextLabRepresentationProfileId(config),
        },
      }))
    }
    analysis.independenceGroupId = independenceGroupId
    appendArtifact(state, artifact)
    records.push({ analysis, physicalSource, chunk })
    await context.progress('textLab', (index + 1) / chunks.length)
  }
  state.analysisRecords = records
  state.analyses = records.map((record) => record.analysis)
  const representationSpaceIds = [...new Set(state.analyses.map((analysis) => analysis.representationSpaceId))]
  if (representationSpaceIds.length !== 1) {
    const error = new Error('Tek eğitim çevrimindeki Metin Laboratuvarı analizleri farklı temsil uzayları üretti.')
    error.code = 'REPRESENTATION_SPACE_MISMATCH'
    throw error
  }
  state.representationSpaceId = representationSpaceIds[0]
  if (resolveTextLabRepresentationProfileId(selection.config) === ADAPTIVE_TEXT_LAB_PROFILE_ID) {
    const incompatible = state.analyses.find((analysis) => (
      resolveTextLabRepresentationProfileId(analysis.config) !== ADAPTIVE_TEXT_LAB_PROFILE_ID ||
      analysis.layout.side !== 64 ||
      analysis.layout.identityLength !== 512 ||
      analysis.layout.relationLength !== 3_584
    ))
    if (incompatible) {
      const error = new Error('Plastik bölge çevrimi için üretilen analiz canonical 8/56 I/R geometrisiyle uyuşmuyor.')
      error.code = 'REPRESENTATION_SPACE_MISMATCH'
      throw error
    }
  }
  state.languageSources = createLanguageSources(records)
  const languageConfig = { ...languageEngine.DEFAULT_LANGUAGE_EXPERIENCE_CONFIG }
  const active = typeof context.store.getActiveSnapshot === 'function'
    ? readActiveRuntimeEnvelope(context.store)
    : null
  const splitPreview = languageEngine.splitLanguageSources(
    state.languageSources,
    languageConfig,
    active?.languageState?.sourceGroupSplits ?? [],
  )
  const assignments = languageGroupSplitAssignments(splitPreview.sources)
  state.languageSplitAssignments = assignments
  state.discoveryIndependenceGroupIds = assignments
    .filter((assignment) => assignment.split === 'train')
    .map((assignment) => assignment.independenceGroupId)
  state.evaluationIndependenceGroupIds = assignments
    .filter((assignment) => assignment.split === 'validation' || assignment.split === 'test')
    .map((assignment) => assignment.independenceGroupId)
  return {
    verified: true,
    artifactIds: records.map((_, index) => state.artifactIds[index]).filter(Boolean),
    sourceCount: state.sources.length,
    chunkCount: chunks.length,
    neuronCount: state.analyses.reduce((sum, analysis) => sum + analysis.nodes.length, 0),
    representationSpaceId: state.analyses[0].representationSpaceId,
    discoverySourceGroupCount: state.discoveryIndependenceGroupIds.length,
    evaluationSourceGroupCount: state.evaluationIndependenceGroupIds.length,
    duplicateSourceCount: splitPreview.duplicates.length,
    lossless: true,
    ...(resolveTextLabRepresentationProfileId(selection.config) === ADAPTIVE_TEXT_LAB_PROFILE_ID
      ? {
          representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
          plasticRegionMode: selection.plasticRegionRequest.requested
            ? selection.plasticRegionRequest.mode
            : 'off',
          reencodePlanFingerprint: selection.reencodePlan.fingerprint,
        }
      : {}),
  }
}

async function synthesis(job, context, state) {
  const config = { ...DEFAULT_SYNTHESIS_CONFIG }
  const discoveryGroups = new Set(state.discoveryIndependenceGroupIds ?? [])
  const discoveryAnalyses = state.analyses.filter((analysis) =>
    discoveryGroups.has(analysis.independenceGroupId || analysis.source.id))
  const shards = createDiverseSynthesisShards(
    discoveryAnalyses,
    CORPUS_RUNTIME_LIMITS.maximumSynthesisShardAnalyses,
  ).filter((shard) => new Set(shard.map((analysis) => (
    analysis.independenceGroupId || analysis.source.id
  ))).size >= 2)
  const runs = []
  for (let index = 0; index < shards.length; index += 1) {
    await context.checkpoint(`synthesis-shard-${index + 1}`)
    budgetGuard(context, state.runtimeStartedAt)
    const input = shards[index].map(toSynthesisAnalysis)
    const contentHash = stableRuntimeHash(
      RUNTIME_VERSION,
      'synthesis-run',
      input.map((analysis) => [
        analysis.runId,
        analysis.sourceDigest,
        analysis.independenceGroupId ?? '',
      ]),
      config,
    )
    const cached = findCachedArtifact(context.store, 'synthesis-run', contentHash)
    let run
    let artifact
    if (cached) {
      artifact = cached
      run = restoreArtifact(cached, 'synthesis-run')
    } else {
      run = await runSynthesis(input, config, {
        shouldCancel: () => budgetGuard(context, state.runtimeStartedAt),
      })
      ;({ artifact } = persistArtifact(context, {
        kind: 'synthesis-run',
        contentHash,
        value: run,
        metadata: { shard: index + 1, candidateCount: run.candidates.length, stats: run.stats },
      }))
    }
    appendArtifact(state, artifact)
    runs.push(run)
    await context.progress('synthesis', (index + 1) / Math.max(1, shards.length))
  }
  const automationLevel = job.payload.automationLevel ?? (context.config ?? context.store.getConfig()).automation.level
  const candidates = new Map()
  for (const run of runs) {
    for (const candidate of run.candidates) {
      if (!candidates.has(candidate.fingerprint)) candidates.set(candidate.fingerprint, { run, candidate })
    }
  }
  state.synthesisRuns = runs
  state.synthesisDecisions = automationLevel === 'controlled'
    ? [...candidates.values()].map(({ run, candidate }) => synthesisDecision(run, candidate))
    : []
  return {
    verified: true,
    shardCount: shards.length,
    candidateCount: candidates.size,
    approvedCandidateCount: state.synthesisDecisions.length,
    comparisonCount: runs.reduce((sum, run) => sum + run.stats.comparedPairCount, 0),
    mode: automationLevel,
    note: automationLevel === 'controlled'
      ? 'Sentez adayları kaynak bağımsızlığı ve eşiklerden geçtikten sonra bu aday çevrimine katıldı.'
      : 'Sentez adayları öneri olarak saklandı; sabit kimlikli yeni nörona dönüşmeleri otomatik onaylanmadı.',
  }
}

async function families(_job, context, state) {
  const config = { ...DEFAULT_FAMILY_CONFIG }
  // Never open the previous full FamilyRun beside the current corpus. Only
  // compact developmental continuity is needed here; legacy snapshots without
  // it safely start a fresh lifecycle observation.
  const previous = splitCompatibleFamilyContinuity(
    readActiveFamilyContinuity(context.store),
    state.discoveryIndependenceGroupIds,
    state.evaluationIndependenceGroupIds,
  )
  const inputHash = createFamilyRunCacheKey({
    runtimeVersion: RUNTIME_VERSION,
    analyses: state.analyses,
    synthesisDecisions: state.synthesisDecisions,
    discoveryIndependenceGroupIds: state.discoveryIndependenceGroupIds,
    evaluationIndependenceGroupIds: state.evaluationIndependenceGroupIds,
    config,
    previousRun: previous,
    lineage: createFamilyModelLineage(context.store),
  })
  const cached = findCachedArtifact(context.store, 'family-run', inputHash)
  let familyRun
  let artifact
  if (cached) {
    artifact = cached
    familyRun = restoreArtifact(cached, 'family-run')
  } else {
    familyRun = await runFamilyLearning({
      analyses: state.analyses.map(toFamilyAnalysis),
      discoveryIndependenceGroupIds: state.discoveryIndependenceGroupIds,
      evaluationIndependenceGroupIds: state.evaluationIndependenceGroupIds,
      synthesisDecisions: state.synthesisDecisions,
      previousRun: previous,
    }, config, {
      shouldCancel: () => budgetGuard(context, state.runtimeStartedAt),
      onProgress: (event) => {
        void context.emit({ type: 'family-progress', stage: 'families', data: event })
      },
    })
    ;({ artifact } = persistArtifact(context, {
      kind: 'family-run',
      contentHash: inputHash,
      value: familyRun,
      metadata: { fingerprint: familyRun.fingerprint, stats: familyRun.stats },
    }))
  }
  appendArtifact(state, artifact)
  const continuityHash = stableRuntimeHash(
    RUNTIME_VERSION,
    FAMILY_CONTINUITY_ARTIFACT_KIND,
    familyRun.fingerprint,
  )
  const { artifact: continuityArtifact } = persistArtifact(context, {
    kind: FAMILY_CONTINUITY_ARTIFACT_KIND,
    contentHash: continuityHash,
    value: createFamilyContinuityState(familyRun),
    metadata: { familyRunFingerprint: familyRun.fingerprint, familyCount: familyRun.families.length },
  })
  appendArtifact(state, continuityArtifact)
  state.familyRun = familyRun
  state.familyRunArtifact = artifact
  const discoveryGroups = new Set(state.discoveryIndependenceGroupIds)
  state.thoughtObservationNeurons = retainThoughtObservationNeurons(
    state.analyses.filter((analysis) =>
      discoveryGroups.has(analysis.independenceGroupId || analysis.source.id)),
    familyRun,
  )
  // Language training consumes the compact language sources and FamilyRun, not
  // the full 64×64 analyses or synthesis candidates. Drop those large working
  // sets before corruption generation so V8 can reclaim them under pressure.
  state.analysisRecords = undefined
  state.analyses = undefined
  state.synthesisRuns = undefined
  state.synthesisDecisions = undefined
  return {
    verified: true,
    artifactId: artifact.id,
    fingerprint: familyRun.fingerprint,
    familyCount: familyRun.families.length,
    membershipCount: familyRun.memberships.length,
    profileCount: familyRun.profiles.length,
    distinctSourceCount: familyRun.stats.distinctSourceCount,
    warnings: familyRun.warnings,
    thoughtObservationWorkingSet: state.thoughtObservationNeurons.length,
  }
}

async function languageExperience(job, context, state) {
  const config = { ...languageEngine.DEFAULT_LANGUAGE_EXPERIENCE_CONFIG }
  // A legacy runtime checkpoint may embed an entire FamilyRun. Hydrating that
  // 100+ MiB JSON here duplicates the new FamilyRun and caused V8 OOM. Reuse a
  // compact v2 envelope when it is cheap; otherwise rebuild language evidence
  // deterministically from the complete current corpus.
  const active = readActiveRuntimeEnvelope(context.store)
  const dataset = languageEngine.buildLanguageDataset(
    state.familyRun,
    state.languageSources,
    config,
    active?.languageState?.sourceGroupSplits ?? [],
  )
  const datasetAssignments = languageGroupSplitAssignments(dataset.sources)
  if (stableRuntimeHash(datasetAssignments) !== stableRuntimeHash(state.languageSplitAssignments ?? [])) {
    const error = new Error('Aile keşfi öncesindeki veri bölümü ile dil eğitimi bölümü farklılaştı; çevrim güvenle durduruldu.')
    error.code = 'DATA_SPLIT_DRIFT'
    throw error
  }
  const previous = active?.languageState?.familyRunFingerprint === state.familyRun.fingerprint &&
    active.languageState.configFingerprint === dataset.configFingerprint
    ? active.languageState
    : languageEngine.createLanguageExperienceState(state.familyRun, config)
  const epochs = Math.max(1, Math.min(1_000, Number(job.payload.epochs ?? 1)))
  const trainingResult = languageEngine.trainLanguageExperience(previous, state.familyRun, dataset, {
    epochs,
    evaluateSplits: ['train', 'validation', 'test'],
    shouldCancel: () => budgetGuard(context, state.runtimeStartedAt),
    onProgress: (event) => {
      void context.emit({ type: 'language-progress', stage: 'languageExperience', data: event })
    },
  })
  const emergentLanguageFamilies = deriveEmergentLanguageFamilies(trainingResult, state.familyRun, dataset)
  const outputScaffold = languageOutput.createLanguageOutputScaffold()
  const activeOutput = readLanguageOutputSnapshot(context.store)
  let outputHead
  if (activeOutput.head) {
    try {
      outputHead = languageOutput.migrateLanguageOutputHead(
        activeOutput.head,
        outputScaffold,
        state.familyRun,
      )
    } catch {
      // Representation changes are a hard boundary. The old head remains in
      // its immutable snapshot while this run starts a clean plastic overlay.
      outputHead = languageOutput.createLanguageOutputHead(outputScaffold, state.familyRun)
    }
  } else {
    outputHead = languageOutput.createLanguageOutputHead(outputScaffold, state.familyRun)
  }
  const outputTraining = languageOutput.trainLanguageOutputHead(
    outputHead,
    outputScaffold,
    dataset,
    trainingResult.state,
    state.familyRun,
  )
  const scaffoldHash = stableRuntimeHash(
    RUNTIME_VERSION,
    LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND,
    outputScaffold.fingerprint,
  )
  const { artifact: outputScaffoldArtifact } = persistArtifact(context, {
    kind: LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND,
    contentHash: scaffoldHash,
    value: outputScaffold,
    metadata: {
      fingerprint: outputScaffold.fingerprint,
      familyCount: outputScaffold.families.length,
      seedNeuronCount: outputScaffold.seedNeurons.length,
    },
  })
  appendArtifact(state, outputScaffoldArtifact)
  const outputHeadHash = stableRuntimeHash(
    RUNTIME_VERSION,
    LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND,
    outputTraining.head.fingerprint,
  )
  const { artifact: outputHeadArtifact } = persistArtifact(context, {
    kind: LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND,
    contentHash: outputHeadHash,
    value: outputTraining.head,
    metadata: {
      fingerprint: outputTraining.head.fingerprint,
      revision: outputTraining.head.revision,
      scaffoldFingerprint: outputScaffold.fingerprint,
      familyRunFingerprint: state.familyRun.fingerprint,
      membershipEvidenceCount: outputTraining.head.membershipEvidence.length,
      transitionEvidenceCount: outputTraining.head.transitionEvidence.length,
      nurseryCount: outputTraining.head.nursery.length,
      provisionalNeuronCount: outputTraining.head.outputNeurons.length,
    },
  })
  appendArtifact(state, outputHeadArtifact)
  const runtimeModel = {
    format: 'bombom-runtime-model',
    version: 2,
    engineVersion: RUNTIME_VERSION,
    createdAt: new Date().toISOString(),
    corpusFingerprint: dataset.fingerprint,
    // FamilyRun is already an immutable artifact in the same snapshot. Keep a
    // content-addressed reference instead of duplicating ~100 MiB inside every
    // runtime model checkpoint. Readers hydrate this reference transparently.
    familyRunArtifactId: state.familyRunArtifact.id,
    familyRunFingerprint: state.familyRun.fingerprint,
    languageState: trainingResult.state,
    languageOutputScaffoldArtifactId: outputScaffoldArtifact.id,
    languageOutputScaffoldFingerprint: outputScaffold.fingerprint,
    languageOutputHeadArtifactId: outputHeadArtifact.id,
    languageOutputHeadFingerprint: outputTraining.head.fingerprint,
    languageOutputHeadRevision: outputTraining.head.revision,
    emergentLanguageFamilies,
    datasetStats: dataset.stats,
    datasetWarnings: dataset.warnings,
    trainingMetrics: trainingResult.metrics,
    birthProposals: trainingResult.birthProposals,
  }
  const contentHash = stableRuntimeHash(
    RUNTIME_VERSION,
    'runtime-model-state-reference-v2',
    state.familyRun.fingerprint,
    trainingResult.state.fingerprint,
    outputScaffold.fingerprint,
    outputTraining.head.fingerprint,
    dataset.fingerprint,
  )
  const { artifact } = persistArtifact(context, {
    kind: 'runtime-model-state',
    contentHash,
    value: runtimeModel,
    metadata: {
      corpusFingerprint: dataset.fingerprint,
      languageStateFingerprint: trainingResult.state.fingerprint,
      familyRunFingerprint: state.familyRun.fingerprint,
      birthProposalCount: trainingResult.birthProposals.length,
      emergentLanguageFamilyCount: emergentLanguageFamilies.length,
      languageOutputScaffoldFingerprint: outputScaffold.fingerprint,
      languageOutputHeadFingerprint: outputTraining.head.fingerprint,
      languageOutputHeadRevision: outputTraining.head.revision,
      languageOutputMembershipEvidenceCount: outputTraining.head.membershipEvidence.length,
      languageOutputTransitionEvidenceCount: outputTraining.head.transitionEvidence.length,
      languageOutputNurseryCount: outputTraining.head.nursery.length,
      languageOutputProvisionalNeuronCount: outputTraining.head.outputNeurons.length,
    },
  })
  appendArtifact(state, artifact)
  state.dataset = dataset
  state.trainingResult = trainingResult
  state.emergentLanguageFamilies = emergentLanguageFamilies
  state.languageOutputScaffold = outputScaffold
  state.languageOutputHead = outputTraining.head
  state.runtimeModelArtifact = artifact
  return {
    verified: true,
    artifactId: artifact.id,
    datasetFingerprint: dataset.fingerprint,
    examples: dataset.examples.length,
    splits: dataset.stats,
    transitionCount: trainingResult.state.familyTransitions.length + trainingResult.state.neuronTransitions.length,
    birthProposalCount: trainingResult.birthProposals.length,
    eligibleBirthProposalCount: trainingResult.birthProposals.filter((proposal) => proposal.status === 'eligible').length,
    emergentLanguageFamilyCount: emergentLanguageFamilies.length,
    languageOutputFamilyCount: outputScaffold.families.length,
    languageOutputHeadRevision: outputTraining.head.revision,
    languageOutputMembershipEvidenceCount: outputTraining.head.membershipEvidence.length,
    languageOutputTransitionEvidenceCount: outputTraining.head.transitionEvidence.length,
    languageOutputNurseryCount: outputTraining.head.nursery.length,
    languageOutputProvisionalNeuronCount: outputTraining.head.outputNeurons.length,
    languageOutputPromotedNeuronCount: outputTraining.promotedNeuronCount,
    languageOutputReplaySkipped: outputTraining.replaySkipped,
    warnings: dataset.warnings,
  }
}

function effectiveMacroObservationBudget(serviceConfig, fullConfig) {
  const matrixBytesAsJsNumbers = 4_096 * 8
  // During mining an observation and its candidate aggregate can coexist.
  // Reserve only a quarter of the dedicated neuron cache for this transient
  // working set; the configured value remains a hard upper bound.
  const cacheBytes = Number(fullConfig.resourcePolicy?.neuronCacheMb ?? 384) * 1024 * 1024
  const safeTransientBytes = Math.max(matrixBytesAsJsNumbers * 16, cacheBytes * 0.25)
  const estimatedBytesPerWindow = matrixBytesAsJsNumbers * 8
  return Math.max(16, Math.min(
    serviceConfig.maximumObservationsPerCycle,
    Math.floor(safeTransientBytes / estimatedBytesPerWindow),
    // Dense audit prototypes must also stay comfortably below the store's
    // 256 MiB single-artifact decode ceiling, even on machines with huge RAM.
    512,
  ))
}

function compactMacroObservationArtifact(mining) {
  return {
    format: 'concept-macro-observation-summary-v1',
    version: 1,
    fingerprint: mining.fingerprint,
    representationSpaceId: mining.representationSpaceId,
    familyRunFingerprint: mining.familyRunFingerprint,
    configFingerprint: mining.configFingerprint,
    processedTraceCount: mining.processedTraceKeys.length,
    processedSemanticTraceCount: mining.processedSemanticTraceKeys.length,
    diagnostics: structuredClone(mining.diagnostics),
    candidates: mining.candidates.map((candidate) => ({
      id: candidate.id,
      fingerprint: candidate.fingerprint,
      kind: candidate.kind,
      lifecycle: candidate.lifecycle,
      executionSemantics: candidate.executionSemantics,
      exactSignature: candidate.exactSignature,
      familySignature: candidate.familySignature,
      length: candidate.length,
      expansionDigest: candidate.expansionDigest,
      observationCount: candidate.observationCount,
      independentGroupCount: candidate.independentGroupCount,
      sourceCount: candidate.sourceCount,
      effectCohesion: candidate.effectCohesion,
      mdl: structuredClone(candidate.mdl),
    })),
    densePrototypesStoredInHeadOnly: true,
  }
}

async function conceptMacros(_job, context, state) {
  const fullConfig = context.config ?? context.store.getConfig()
  const serviceConfig = fullConfig.conceptMacros
  if (!serviceConfig?.enabled || serviceConfig.mode === 'off') {
    state.thoughtObservationNeurons = undefined
    state.conceptMacroSummary = {
      mode: 'off', macroCount: 0, activeCount: 0, shadowValidatedCount: 0, structuralValidatedCount: 0,
    }
    return { verified: true, disabled: true, mode: 'off' }
  }

  const effectiveObservationLimit = effectiveMacroObservationBudget(serviceConfig, fullConfig)
  const engineConfig = conceptMacroEngineConfig({
    ...serviceConfig,
    maximumCandidates: Math.min(serviceConfig.maximumCandidates, effectiveObservationLimit),
  })
  const referenceCatalog = conceptMacroReferenceCatalog(state.familyRun)
  const previous = readLatestConceptMacroHead(
    context.store,
    state.familyRun.representationSpaceId,
  )
  let head
  let rebase = null
  const warnings = []
  if (previous.head?.configFingerprint === createMacroHead({
    representationSpaceId: state.familyRun.representationSpaceId,
    familyRunFingerprint: state.familyRun.fingerprint,
  }, engineConfig).configFingerprint) {
    if (previous.head.familyRunFingerprint === state.familyRun.fingerprint) head = previous.head
    else {
      rebase = rebaseMacroHeadReference(previous.head, referenceCatalog)
      head = rebase.head
    }
  } else {
    head = createMacroHead({
      representationSpaceId: state.familyRun.representationSpaceId,
      familyRunFingerprint: state.familyRun.fingerprint,
    }, engineConfig)
    if (previous.head) warnings.push('Makro motor ayarları değişti; eski immutable head korunup bu temsil uzayında yeni soy başlatıldı.')
  }
  if (previous.quarantinedReason) {
    warnings.push(`Önceki makro head güvenli karantinaya alındı: ${previous.quarantinedReason}`)
  }

  const observation = await collectBoundedThoughtMacroTraces(context, state, serviceConfig, head)
  const trainingTraces = boundedMacroTrainingTraces(
    observation.traces.filter((trace) => trace.split === 'train'),
    engineConfig,
    effectiveObservationLimit,
  )
  const holdoutTraces = observation.traces.filter((trace) =>
    trace.split === 'validation' || trace.split === 'test')
  const ingested = ingestMacroTraces(head, trainingTraces)
  head = ingested.head

  const observationHash = stableRuntimeHash(
    RUNTIME_VERSION,
    CONCEPT_MACRO_OBSERVATION_ARTIFACT_KIND,
    ingested.mining.fingerprint,
    trainingTraces.map((trace) => trace.id),
  )
  const { artifact: observationArtifact } = persistArtifact(context, {
    kind: CONCEPT_MACRO_OBSERVATION_ARTIFACT_KIND,
    contentHash: observationHash,
    // The immutable head is the canonical dense state. Persisting the same
    // 4096D prototypes again in every observation artifact caused quadratic
    // SSD growth; this audit artifact intentionally retains only thin facts.
    value: compactMacroObservationArtifact(ingested.mining),
    metadata: {
      fingerprint: ingested.mining.fingerprint,
      representationSpaceId: state.familyRun.representationSpaceId,
      familyRunFingerprint: state.familyRun.fingerprint,
      inputTraceCount: observation.traces.length,
      trainTraceCount: trainingTraces.length,
      holdoutTraceCount: holdoutTraces.length,
      effectiveObservationLimit,
      acceptedObservationCount: ingested.mining.diagnostics.acceptedObservationCount,
      retainedCandidateCount: ingested.mining.diagnostics.retainedCandidateCount,
      densePrototypesStoredInHeadOnly: true,
      diagnostics: observation.diagnostics,
    },
  })
  appendArtifact(state, observationArtifact)

  if (serviceConfig.mode === 'proposal-shadow' || serviceConfig.mode === 'controlled') {
    for (const macro of [...head.macros]
      .filter((candidate) => candidate.lifecycle === 'eligible-proposal')
      .sort((left, right) => right.mdl.compressionGain - left.mdl.compressionGain || left.id.localeCompare(right.id))) {
      head = transitionCompositeLifecycle(head, macro.id, 'virtual-provisional')
    }
  }

  const shadowReports = []
  const shadowCandidates = head.macros
    .filter((macro) => macro.lifecycle === 'virtual-provisional')
    .sort((left, right) => right.mdl.compressionGain - left.mdl.compressionGain || left.id.localeCompare(right.id))
    .slice(0, Math.min(64, serviceConfig.maximumActiveMacros))
  for (let index = 0; index < shadowCandidates.length; index += 1) {
    const candidate = head.macros.find((macro) => macro.id === shadowCandidates[index].id)
    if (!candidate || candidate.lifecycle !== 'virtual-provisional') continue
    const cases = shadowCasesForComposite(
      candidate,
      holdoutTraces,
      head.config.minimumEntryCompatibility,
      64,
    )
    const report = evaluateMacroShadow(candidate, cases, head.config)
    const beforeFingerprint = head.fingerprint
    head = applyMacroShadowEvaluation(head, report)
    shadowReports.push(report)
    if (head.fingerprint !== beforeFingerprint) {
      const shadowHash = stableRuntimeHash(
        RUNTIME_VERSION,
        CONCEPT_MACRO_SHADOW_ARTIFACT_KIND,
        report.fingerprint,
      )
      const { artifact } = persistArtifact(context, {
        kind: CONCEPT_MACRO_SHADOW_ARTIFACT_KIND,
        contentHash: shadowHash,
        value: report,
        metadata: {
          fingerprint: report.fingerprint,
          compositeId: report.compositeId,
          passed: report.passed,
          caseCount: report.caseCount,
          activationRecall: report.activationRecall,
          fidelity: report.fidelity,
          falseActivationRate: report.falseActivationRate,
        },
      })
      appendArtifact(state, artifact)
    }
    await context.progress('conceptMacros', (index + 1) / Math.max(1, shadowCandidates.length))
  }

  const headHash = stableRuntimeHash(
    RUNTIME_VERSION,
    CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    head.fingerprint,
  )
  const { artifact: headArtifact } = persistArtifact(context, {
    kind: CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    contentHash: headHash,
    value: head,
    metadata: {
      fingerprint: head.fingerprint,
      revision: head.revision,
      mode: serviceConfig.mode,
      representationSpaceId: head.representationSpaceId,
      familyRunFingerprint: head.familyRunFingerprint,
      macroCount: head.macros.length,
      activeCount: head.macros.filter((macro) => macro.lifecycle === 'active').length,
      shadowValidatedCount: head.macros.filter((macro) => macro.lifecycle === 'shadow-validated').length,
      structuralValidatedCount: head.macros.filter((macro) => macro.structuralEquivalencePassed === true).length,
      rawConstituentsRetained: head.rawConstituentsRetained,
    },
  })
  appendArtifact(state, headArtifact)
  state.conceptMacroHead = head
  state.conceptMacroHeadArtifact = headArtifact
  state.conceptMacroSummary = {
    mode: serviceConfig.mode,
    macroCount: head.macros.length,
    observingCount: head.macros.filter((macro) => macro.lifecycle === 'observing').length,
    proposalCount: head.macros.filter((macro) => macro.lifecycle === 'eligible-proposal').length,
    virtualCount: head.macros.filter((macro) => macro.lifecycle === 'virtual-provisional').length,
    shadowValidatedCount: head.macros.filter((macro) => macro.lifecycle === 'shadow-validated').length,
    structuralValidatedCount: head.macros.filter((macro) => macro.structuralEquivalencePassed === true).length,
    activeCount: head.macros.filter((macro) => macro.lifecycle === 'active').length,
    deprecatedCount: head.macros.filter((macro) => macro.lifecycle === 'deprecated').length,
    traceCount: observation.traces.length,
    trainTraceCount: trainingTraces.length,
    holdoutTraceCount: holdoutTraces.length,
    shadowReportCount: shadowReports.length,
    effectiveObservationLimit,
  }
  state.thoughtObservationNeurons = undefined
  if (!holdoutTraces.length) warnings.push('Bu çevrimde bağımsız validation/test düşünce izi yok; makrolar etkinleştirilmedi.')
  if (state.conceptMacroSummary.shadowValidatedCount > state.conceptMacroSummary.structuralValidatedCount) {
    warnings.push('Endpoint shadow kapısını geçen kavramlar var; hızlı R/decay, residual pressure ve sonraki rota eşdeğerliği kanıtlanmadığı için runtime shortcut etkinleştirilmedi.')
  }
  return {
    verified: true,
    artifactId: headArtifact.id,
    observationArtifactId: observationArtifact.id,
    ...state.conceptMacroSummary,
    replayDuplicates: ingested.mining.diagnostics.duplicateTraceCount,
    rebaseDeprecatedCount: rebase?.deprecatedMacroIds.length ?? 0,
    warnings,
  }
}

function activateValidatedConceptMacros(head, serviceConfig) {
  let next = head
  const alreadyActive = head.macros.filter((macro) => macro.lifecycle === 'active').length
  let remaining = Math.max(0, serviceConfig.maximumActiveMacros - alreadyActive)
  const activatedIds = []
  for (const macro of [...head.macros]
    .filter((candidate) =>
      candidate.kind === 'sequence-macro' &&
      candidate.lifecycle === 'shadow-validated' &&
      candidate.structuralEquivalencePassed === true &&
      candidate.structuralEquivalenceAudit?.passed === true &&
      candidate.observationCount >= serviceConfig.activationMinimumOccurrences &&
      candidate.independentGroupCount >= serviceConfig.activationMinimumSourceGroups &&
      candidate.metrics.activationRecall >= serviceConfig.minimumCompletionProbability &&
      candidate.metrics.fidelity >= serviceConfig.minimumShadowFidelity &&
      candidate.metrics.falseActivationRate <= serviceConfig.maximumFalseActivationRate)
    .sort((left, right) =>
      right.metrics.fidelity - left.metrics.fidelity ||
      right.mdl.compressionGain - left.mdl.compressionGain ||
      left.id.localeCompare(right.id))) {
    if (remaining <= 0) break
    next = transitionCompositeLifecycle(next, macro.id, 'active')
    activatedIds.push(macro.id)
    remaining -= 1
  }
  return { head: next, activatedIds }
}

async function evaluate(job, context, state) {
  const train = state.trainingResult.metrics.train
  const validation = state.trainingResult.metrics.validation
  const test = state.trainingResult.metrics.test
  const validationTop1 = topK(validation, 1)
  const validationTop5 = topK(validation, 5)
  const testTop1 = topK(test, 1)
  const trainTop1 = topK(train, 1)
  const utility = familyUtilityScore(state.familyRun, state.emergentLanguageFamilies)
  const overallScore = clamp01(
    0.3 * validationTop1 +
    0.18 * validationTop5 +
    0.18 * (validation?.meanReciprocalRank ?? 0) +
    0.16 * (validation?.grammarCorruptionAccuracy ?? 0) +
    0.08 * (validation?.eosAccuracy ?? 0) +
    0.1 * utility,
  )
  const activeMetrics = context.store.getActiveSnapshot()?.metrics
  const metrics = {
    overallScore,
    syntaxContrastAccuracy: clamp01(validation?.grammarCorruptionAccuracy ?? 0),
    regressionRate: clamp01(Math.max(0, (activeMetrics?.overallScore ?? overallScore) - overallScore)),
    memorizationGap: clamp01(Math.max(0, trainTop1 - testTop1)),
    unseenAccuracy: clamp01(testTop1),
    holdoutTop1: validationTop1,
    holdoutTop5: validationTop5,
    grammaticalPreference: clamp01(validation?.grammarCorruptionAccuracy ?? 0),
    familyUtility: utility,
    neuronCount: state.familyRun.profiles.length,
    familyCount: state.familyRun.families.length,
    birthProposalCount: state.trainingResult.birthProposals.length,
    eligibleBirthProposalCount: state.trainingResult.birthProposals.filter((proposal) => proposal.status === 'eligible').length,
    emergentLanguageFamilyCount: state.emergentLanguageFamilies.length,
    eligibleEmergentLanguageFamilyCount: state.emergentLanguageFamilies.filter((family) => family.status === 'eligible').length,
    languageOutputFamilyCount: state.languageOutputScaffold.families.length,
    languageOutputHeadRevision: state.languageOutputHead.revision,
    languageOutputMembershipEvidenceCount: state.languageOutputHead.membershipEvidence.length,
    languageOutputTransitionEvidenceCount: state.languageOutputHead.transitionEvidence.length,
    languageOutputNurseryCount: state.languageOutputHead.nursery.length,
    languageOutputProvisionalNeuronCount: state.languageOutputHead.outputNeurons.length,
    eligibleLanguageOutputNurseryCount: state.languageOutputHead.nursery.filter((proposal) => proposal.status === 'eligible').length,
    conceptMacroCount: state.conceptMacroSummary?.macroCount ?? 0,
    conceptMacroProposalCount: state.conceptMacroSummary?.proposalCount ?? 0,
    conceptMacroVirtualCount: state.conceptMacroSummary?.virtualCount ?? 0,
    conceptMacroShadowValidatedCount: state.conceptMacroSummary?.shadowValidatedCount ?? 0,
    conceptMacroStructuralValidatedCount: state.conceptMacroSummary?.structuralValidatedCount ?? 0,
    conceptMacroActiveCount: state.conceptMacroSummary?.activeCount ?? 0,
    evaluationSampleCount: (validation?.exampleCount ?? 0) + (test?.exampleCount ?? 0),
    corpusFingerprint: state.dataset.fingerprint,
  }
  const independentGroups = Object.values(state.dataset.stats)
    .reduce((sum, split) => sum + split.independenceGroupCount, 0)
  const verified = Boolean(
    independentGroups >= 3 &&
    state.syntheticRatio <= 0.25 &&
    validation?.exampleCount > 0 &&
    test?.exampleCount > 0 &&
    validation?.boundTargetCount > 0 &&
    test?.boundTargetCount > 0 &&
    state.familyRun.families.length > 0,
  )
  const fullConfig = context.config ?? context.store.getConfig()
  const automationLevel = job.payload.automationLevel ?? fullConfig.automation.level
  const policyEvaluation = evaluateCandidate(metrics, activeMetrics, fullConfig.evaluation)
  let activatedConceptMacroIds = []
  if (
    verified &&
    policyEvaluation.passed &&
    automationLevel === 'controlled' &&
    fullConfig.conceptMacros?.mode === 'controlled' &&
    state.conceptMacroHead
  ) {
    const activation = activateValidatedConceptMacros(state.conceptMacroHead, fullConfig.conceptMacros)
    if (activation.activatedIds.length) {
      state.conceptMacroHead = activation.head
      activatedConceptMacroIds = activation.activatedIds
      const headHash = stableRuntimeHash(
        RUNTIME_VERSION,
        CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
        activation.head.fingerprint,
      )
      const { artifact } = persistArtifact(context, {
        kind: CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
        contentHash: headHash,
        value: activation.head,
        metadata: {
          fingerprint: activation.head.fingerprint,
          revision: activation.head.revision,
          mode: fullConfig.conceptMacros.mode,
          representationSpaceId: activation.head.representationSpaceId,
          familyRunFingerprint: activation.head.familyRunFingerprint,
          macroCount: activation.head.macros.length,
          activeCount: activation.head.macros.filter((macro) => macro.lifecycle === 'active').length,
          activatedConceptMacroIds,
          rawConstituentsRetained: true,
        },
      })
      appendArtifact(state, artifact)
      state.conceptMacroHeadArtifact = artifact
      metrics.conceptMacroActiveCount = activation.head.macros.filter((macro) => macro.lifecycle === 'active').length
    }
  }
  return {
    verified,
    metrics,
    evidence: {
      independentSourceGroups: independentGroups,
      validationExamples: validation?.exampleCount ?? 0,
      testExamples: test?.exampleCount ?? 0,
      grammarContrastCases: validation?.grammarCorruptionCaseCount ?? 0,
      syntheticRatio: state.syntheticRatio,
      runtimeModelArtifactId: state.runtimeModelArtifact.id,
      conceptMacroHeadArtifactId: state.conceptMacroHeadArtifact?.id ?? null,
      activatedConceptMacroIds,
    },
    warnings: [
      ...(validation?.warnings ?? []),
      ...(test?.warnings ?? []),
      ...(state.syntheticRatio > 0.25 ? ['Ollama üretimi kaynak oranı %25 güvenli otomatik kabul sınırını aşıyor.'] : []),
      ...(!verified ? ['Gerçek kaynak-holdout kanıtı henüz güvenli otomatik kabul için yeterli değil.'] : []),
    ],
  }
}

export const REAL_STAGE_HANDLERS = Object.freeze({
  collect,
  textLab,
  synthesis,
  families,
  languageExperience,
  plasticRegions,
  conceptMacros,
  evaluate,
})

export function createRealPipelineExecutor() {
  return createPipelineExecutor({ stageHandlers: REAL_STAGE_HANDLERS })
}

export { RUNTIME_VERSION }
