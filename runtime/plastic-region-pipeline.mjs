import {
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_REGION_GEOMETRY,
  PAGE_BEHAVIOR_DIMENSION,
  aggregatePageSketches,
  cosineRegionVectors,
  computeRegionGates,
  createInitialPlasticRegionAtlas,
  deriveLatentRegionId,
  plasticRegionCapacity,
  proposeRegionBirth,
  proposeRegionGrow,
  proposeRegionMerge,
  proposeRegionRetire,
  proposeRegionShrink,
  proposeRegionSplit,
  validateAdaptiveRegionAtlas,
  weightedRegionMean,
} from '../src/adaptive-regions/index.ts'
import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  createTextLabConfigForProfile,
  normalizeTextLabConfig,
  resolveTextLabRepresentationProfileId,
} from '../src/text-lab/engine/index.ts'
import {
  encodeArtifactValue,
  encodedByteLength,
  persistArtifact,
  stableRuntimeHash,
} from './artifacts.mjs'
import {
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
  PLASTIC_REGION_OBSERVATION_ARTIFACT_KIND,
  PLASTIC_REGION_PROBE_CACHE_ARTIFACT_KIND,
  PLASTIC_REGION_SHADOW_ARTIFACT_KIND,
  assertPlasticRegionArtifactValue,
  currentPlasticRegionBase,
  persistPlasticRegionArtifact,
  readPlasticRegionSnapshot,
} from './plastic-region-cache.mjs'
import {
  PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
  createPlasticRegionEvidenceHead,
} from './plastic-region-evidence.mjs'

export const PLASTIC_REGION_PIPELINE_VERSION = 'plastic-region-pipeline-v1'
export const CANONICAL_PLASTIC_REGION_PROFILE = ADAPTIVE_TEXT_LAB_PROFILE_ID

function pipelineError(code, message, details) {
  const error = new Error(message)
  error.code = code
  if (details !== undefined) error.details = details
  return error
}

function compareText(left, right) {
  return String(left).localeCompare(String(right))
}

function freezeDeep(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value) || ArrayBuffer.isView(value)) return value
  for (const child of Object.values(value)) freezeDeep(child)
  return Object.freeze(value)
}

function manifestOf(source) {
  const declared = source?.provenance?.textLabManifest
  if (declared === undefined || declared === null) return { kind: 'absent', value: null }
  if (declared?.format !== 'text-lab-recompute-proof-v1') return { kind: 'invalid', value: declared }
  return { kind: 'valid', value: declared }
}

function profileOfConfig(config) {
  return resolveTextLabRepresentationProfileId(normalizeTextLabConfig(config))
}

export function resolvePlasticRegionRuntimeRequest(config) {
  const service = config?.plasticRegions
  if (!service?.enabled || service.mode === 'off') {
    return Object.freeze({ requested: false, mode: 'off', profile: service?.representationProfile ?? 'legacy' })
  }
  if (service.mode === 'active') {
    // An already released active atlas may keep collecting train-only
    // observations, but every candidate remains a detached proposal. Training
    // never replaces or activates the current head by itself.
    if (service.representationProfile !== CANONICAL_PLASTIC_REGION_PROFILE) {
      throw pipelineError('PLASTIC_REGION_PROFILE_REQUIRED', 'Active plastik bölge eğitimi canonical 8/56 profil ister.')
    }
    return Object.freeze({
      requested: true,
      mode: 'proposal',
      configuredMode: 'active',
      profile: CANONICAL_PLASTIC_REGION_PROFILE,
    })
  }
  if (service.mode !== 'shadow' && service.mode !== 'proposal') {
    throw pipelineError('INVALID_PLASTIC_REGION_MODE', 'Plastik bölge eğitim kipi shadow veya proposal olmalıdır.')
  }
  if (service.representationProfile !== CANONICAL_PLASTIC_REGION_PROFILE) {
    throw pipelineError(
      'PLASTIC_REGION_PROFILE_REQUIRED',
      `Plastik bölge ${service.mode} kipi yalnız ${CANONICAL_PLASTIC_REGION_PROFILE} temsil profiliyle çalışır.`,
    )
  }
  return Object.freeze({ requested: true, mode: service.mode, profile: CANONICAL_PLASTIC_REGION_PROFILE })
}

/**
 * Read-only inventory. It never encodes text and never rewrites an imported
 * TextLab manifest; incompatible imports are explicit blockers.
 */
export function planPlasticRegionReencodeInventory(sources, options = {}) {
  const targetProfile = options.targetProfile ?? CANONICAL_PLASTIC_REGION_PROFILE
  if (targetProfile !== CANONICAL_PLASTIC_REGION_PROFILE) {
    throw pipelineError('PLASTIC_REGION_PROFILE_REQUIRED', 'Yeniden kodlama planının hedef profili canonical 8/56 profil olmalıdır.')
  }
  const targetConfig = createTextLabConfigForProfile(targetProfile, options.configOverrides ?? {})
  const rows = [...(sources ?? [])].map((source) => {
    const sourceId = String(source?.id ?? '').trim()
    const manifest = manifestOf(source)
    if (!sourceId) throw pipelineError('INVALID_REENCODE_SOURCE', 'Yeniden kodlama envanterinde kaynak kimliği boş olamaz.')
    if (manifest.kind === 'invalid') {
      return {
        sourceId,
        contentHash: String(source?.contentHash ?? ''),
        status: 'blocked-invalid-manifest',
        currentProfile: null,
        targetProfile,
        mutationPlanned: false,
      }
    }
    if (manifest.kind === 'absent') {
      return {
        sourceId,
        contentHash: String(source?.contentHash ?? ''),
        status: 'encode-required',
        currentProfile: null,
        targetProfile,
        mutationPlanned: false,
      }
    }
    let currentProfile = null
    try {
      currentProfile = profileOfConfig(manifest.value.config)
    } catch {
      return {
        sourceId,
        contentHash: String(source?.contentHash ?? ''),
        status: 'blocked-invalid-manifest',
        currentProfile: null,
        targetProfile,
        mutationPlanned: false,
      }
    }
    return {
      sourceId,
      contentHash: String(source?.contentHash ?? ''),
      status: currentProfile === targetProfile ? 'reuse-verified-manifest' : 'blocked-representation-mismatch',
      currentProfile,
      targetProfile,
      mutationPlanned: false,
    }
  }).sort((left, right) => compareText(left.sourceId, right.sourceId))
  const counts = rows.reduce((result, row) => {
    result[row.status] = (result[row.status] ?? 0) + 1
    return result
  }, {})
  const base = {
    format: 'plastic-region-reencode-plan',
    version: 1,
    dryRun: true,
    willMutate: false,
    targetProfile,
    targetConfig,
    rows,
    counts,
    blocked: rows.some((row) => row.status.startsWith('blocked-')),
  }
  return freezeDeep({ ...base, fingerprint: stableRuntimeHash(PLASTIC_REGION_PIPELINE_VERSION, base) })
}

/** Resolves one run-wide encoder. Imported manifests remain authoritative. */
export function resolveAdaptiveTextLabConfig(sources, serviceConfig) {
  const request = resolvePlasticRegionRuntimeRequest({ plasticRegions: serviceConfig })
  if (!request.requested && serviceConfig?.representationProfile !== CANONICAL_PLASTIC_REGION_PROFILE) return null
  const plan = planPlasticRegionReencodeInventory(sources)
  if (plan.blocked) {
    throw pipelineError(
      'REPRESENTATION_SPACE_MISMATCH',
      'İçe aktarılmış Metin Laboratuvarı manifesti canonical 8/56 temsil uzayıyla uyuşmuyor; sessiz yeniden kodlama yapılmadı.',
      { blockedSources: plan.rows.filter((row) => row.status.startsWith('blocked-')).map((row) => row.sourceId) },
    )
  }
  const imported = sources
    .map((source) => manifestOf(source))
    .filter((entry) => entry.kind === 'valid')
    .map((entry) => normalizeTextLabConfig(entry.value.config))
  if (!imported.length) return plan.targetConfig
  const authorityHash = stableRuntimeHash(imported[0])
  if (imported.some((config) => stableRuntimeHash(config) !== authorityHash)) {
    throw pipelineError(
      'REPRESENTATION_SPACE_MISMATCH',
      'İçe aktarılmış manifestler farklı encoder ayarları taşıyor; tek çevrimde karıştırılamaz.',
    )
  }
  if (resolveTextLabRepresentationProfileId(imported[0]) !== CANONICAL_PLASTIC_REGION_PROFILE) {
    throw pipelineError('REPRESENTATION_SPACE_MISMATCH', 'Manifest otoritesi canonical 8/56 profilinde değil.')
  }
  return imported[0]
}

function dominantThoughtBranch(result) {
  return [...(result?.episode?.branches ?? [])]
    .sort((left, right) => Number(right.mass ?? 0) - Number(left.mass ?? 0) || compareText(left.id, right.id))[0] ?? null
}

function dominantTraceChain(result, dominant) {
  if (!dominant) return []
  const byChild = new Map()
  for (const tick of result?.ticks ?? []) {
    for (const branch of tick.branches ?? []) byChild.set(branch.branchId, { tick: tick.tick, branch })
  }
  const chain = []
  const seen = new Set()
  let cursor = dominant.id
  while (cursor && !seen.has(cursor)) {
    seen.add(cursor)
    const entry = byChild.get(cursor)
    if (!entry) break
    chain.push(entry)
    cursor = entry.branch.parentBranchId
  }
  return chain.reverse()
}

function canonicalObservationSourceGroup(source) {
  const provenance = source?.provenance ?? {}
  const kind = String(provenance.kind ?? source?.sourceType ?? '').toLowerCase()
  const declared = String(source?.independenceGroupId ?? '').trim()
  if (!declared) throw pipelineError('PLASTIC_REGION_SOURCE_GROUP_REQUIRED', 'Düşünce gözlemi bağımsız kaynak grubu ister.')
  if (kind.includes('ollama')) {
    // The ingest layer has already canonicalized model name + optional weight
    // digest. Here we reject any attempt to replace that lineage with a
    // campaign/prompt/source id; repeated outputs from one model retain one
    // `ollama-lineage:*` independence group.
    if (!declared.startsWith('ollama-lineage:')) {
      throw pipelineError(
        'OLLAMA_LINEAGE_MISMATCH',
        'Aynı Ollama model soyu farklı bağımsız kaynak gibi gösterilemez.',
        { declared },
      )
    }
  }
  return declared
}

export function extractPlasticRegionObservationsFromThoughtResult(input) {
  const { result, source } = input
  const maximumObservations = Math.max(0, Math.trunc(input.maximumObservations ?? 2_048))
  if (source?.split !== 'train') {
    return freezeDeep({
      observations: [],
      excluded: true,
      reason: source?.split === 'validation' || source?.split === 'test'
        ? 'evaluation-split-forbidden'
        : 'non-train-split',
      transitionCount: 0,
    })
  }
  if (maximumObservations < ADAPTIVE_PAGE_COUNT) {
    throw pipelineError('PLASTIC_REGION_OBSERVATION_BUDGET_TOO_SMALL', `En az ${ADAPTIVE_PAGE_COUNT} gözlem bütçesi gerekir.`)
  }
  const sourceGroupId = canonicalObservationSourceGroup(source)
  // Thought telemetry uses the semantic run key so replay-equivalent runs
  // deduplicate even when their outer service episode ids differ.
  const telemetryEpisodeId = result.episode.runKey ?? result.episode.episodeId
  const dominant = dominantThoughtBranch(result)
  const chain = dominantTraceChain(result, dominant)
  const transitionsByTick = new Map()
  for (const transition of dominant?.transitions ?? []) {
    const values = transitionsByTick.get(transition.tick) ?? []
    values.push(transition)
    transitionsByTick.set(transition.tick, values)
  }
  const maximumTransitions = Math.floor(maximumObservations / ADAPTIVE_PAGE_COUNT)
  const observations = []
  let acceptedTransitions = 0
  for (const entry of chain) {
    if (acceptedTransitions >= maximumTransitions) break
    const transitions = transitionsByTick.get(entry.tick) ?? []
    if (entry.branch.macroExecution || transitions.length !== 1) continue
    const telemetry = entry.branch.adaptiveRegion
    if (
      telemetry?.format !== 'thought-adaptive-region-branch-telemetry' ||
      telemetry.version !== 1 ||
      (telemetry.mode !== 'shadow' && telemetry.mode !== 'proposal') ||
      telemetry.activeApplied !== false ||
      !Array.isArray(telemetry.pageObservations) ||
      telemetry.pageObservations.length !== ADAPTIVE_PAGE_COUNT
    ) continue
    if (input.atlasFingerprint && telemetry.atlasFingerprint !== input.atlasFingerprint) {
      throw pipelineError(
        'STALE_REGION_ATLAS_BASE',
        'Düşünce tick gözlemi beklenen atlas fingerprintinden farklı bir atlasla üretildi.',
      )
    }
    const pageIndexes = new Set(telemetry.pageObservations.map((observation) => observation.pageIndex))
    if (pageIndexes.size !== ADAPTIVE_PAGE_COUNT ||
        [...pageIndexes].some((pageIndex) => pageIndex < 0 || pageIndex >= ADAPTIVE_PAGE_COUNT) ||
        telemetry.pageObservations.some((observation) => (
          observation.episodeId !== telemetryEpisodeId ||
          observation.sourceGroupId !== sourceGroupId ||
          observation.outcomeCredit !== 0 ||
          observation.ablationUtility !== 0
        ))) {
      throw pipelineError(
        'INVALID_PLASTIC_REGION_TICK_OBSERVATION',
        'Düşünce tick mikro-sayfa gözlemleri episode/kaynak/sayfa sözleşmesiyle uyuşmuyor.',
        {
          pageCount: telemetry.pageObservations.length,
          distinctPageCount: pageIndexes.size,
          episodeIds: [...new Set(telemetry.pageObservations.map((value) => value.episodeId))],
          sourceGroupIds: [...new Set(telemetry.pageObservations.map((value) => value.sourceGroupId))],
          expectedEpisodeId: telemetryEpisodeId,
          expectedSourceGroupId: sourceGroupId,
        },
      )
    }
    const branchMass = Math.max(1e-6, Math.min(1, Number(entry.branch.mass ?? 0)))
    observations.push(...[...telemetry.pageObservations]
      .sort((left, right) => left.pageIndex - right.pageIndex)
      .map((observation) => {
        const pageSuffix = `:${observation.pageIndex}`
        const telemetryGroupId = observation.observationId.endsWith(pageSuffix)
          ? observation.observationId.slice(0, -pageSuffix.length)
          : observation.observationId
        return {
          ...observation,
          // Preserve one source-bound transition prefix so evidence from 56
          // pages can be deduplicated as one real tick instead of 56 events.
          observationId: `PROBS-${stableRuntimeHash(
            'source-bound-plastic-observation-v1',
            sourceGroupId,
            telemetryGroupId,
          ).slice(0, 40)}:${observation.pageIndex}`,
          behavior: [...observation.behavior],
          confidence: Math.max(1e-6, Math.min(1, observation.confidence * branchMass)),
          // Results and ablations are separate future gates. A raw train trace
          // is never allowed to invent its own holdout credit.
          outcomeCredit: 0,
          ablationUtility: 0,
        }
      }))
    acceptedTransitions += 1
  }
  return freezeDeep({
    observations,
    excluded: false,
    reason: null,
    transitionCount: acceptedTransitions,
  })
}

export function createPlasticRegionObservationBatches(input) {
  const maximumPerBatch = Math.max(ADAPTIVE_PAGE_COUNT, Math.trunc(input.maximumPerBatch ?? 2_048))
  const maximumBatches = Math.max(1, Math.trunc(input.maximumBatches ?? 64))
  const pageAlignedMaximum = Math.max(ADAPTIVE_PAGE_COUNT, Math.floor(maximumPerBatch / ADAPTIVE_PAGE_COUNT) * ADAPTIVE_PAGE_COUNT)
  const observations = [...(input.observations ?? [])]
    .sort((left, right) => compareText(left.sourceGroupId, right.sourceGroupId) || compareText(left.observationId, right.observationId))
    .slice(0, pageAlignedMaximum * maximumBatches)
  const batches = []
  for (let offset = 0; offset < observations.length; offset += pageAlignedMaximum) {
    const values = observations.slice(offset, offset + pageAlignedMaximum)
    const base = {
      format: PLASTIC_REGION_OBSERVATION_ARTIFACT_KIND,
      version: 1,
      representationSpaceId: input.representationSpaceId,
      geometryFingerprint: ADAPTIVE_REGION_GEOMETRY.fingerprint,
      familyRunFingerprint: input.familyRunFingerprint,
      inputManifestHash: input.inputManifestHash,
      batchIndex: batches.length,
      observationCount: values.length,
      sourceGroupIds: [...new Set(values.map((value) => value.sourceGroupId))].sort(compareText),
      evaluationObservationCount: 0,
      observations: values,
    }
    batches.push(Object.freeze({
      ...base,
      fingerprint: stableRuntimeHash(PLASTIC_REGION_PIPELINE_VERSION, 'observation-batch', base),
    }))
  }
  return freezeDeep(batches)
}

export function createPlasticRegionProbeSkeleton(representationSpaceId, atlasFingerprint, inputManifestHash) {
  const base = {
    format: PLASTIC_REGION_PROBE_CACHE_ARTIFACT_KIND,
    version: 1,
    representationSpaceId,
    atlasFingerprint,
    inputManifestHash,
    semanticAssignments: [],
    trainingWritesAllowed: false,
    packs: [
      { id: 'morphology-diagnostic-v1', domain: 'morphology', status: 'skeleton', probes: [], learnedBindings: [] },
      { id: 'mathematics-diagnostic-v1', domain: 'mathematics', status: 'skeleton', probes: [], learnedBindings: [] },
    ],
  }
  return freezeDeep({ ...base, fingerprint: stableRuntimeHash(PLASTIC_REGION_PIPELINE_VERSION, 'probe-cache', base) })
}

function pageCompatibilityVector(sketches) {
  const values = new Array(ADAPTIVE_PAGE_COUNT).fill(0)
  for (const sketch of sketches) values[sketch.pageIndex] = Number(sketch.behaviorPrototype[2] ?? 0)
  return values
}

const MINIMUM_CROSS_PAGE_COHESION = 0.65

function transitionEvidenceKey(reference) {
  const suffix = `:${String(reference.observationId).split(':').at(-1)}`
  const transitionId = String(reference.observationId).endsWith(suffix)
    ? String(reference.observationId).slice(0, -suffix.length)
    : String(reference.observationId)
  return `${reference.sourceGroupId}\u0000${reference.episodeId}\u0000${transitionId}`
}

function transitionEvidenceKeys(sketch) {
  return new Set(sketch.evidence
    .filter((reference) => Number(reference.support ?? 0) > 1e-6)
    .map(transitionEvidenceKey))
}

function allTransitionEvidenceKeys(sketch) {
  return new Set(sketch.evidence.map(transitionEvidenceKey))
}

function intersection(left, right) {
  return new Set([...left].filter((value) => right.has(value)))
}

function coherentBirthCluster(ranked) {
  let best = null
  for (const anchor of ranked) {
    const entries = [anchor]
    let commonTransitions = transitionEvidenceKeys(anchor.sketch)
    for (const candidate of ranked) {
      if (candidate === anchor || entries.length >= 4) continue
      if (entries.some((entry) =>
        cosineRegionVectors(entry.sketch.behaviorPrototype, candidate.sketch.behaviorPrototype) <
          MINIMUM_CROSS_PAGE_COHESION)) continue
      const shared = intersection(commonTransitions, transitionEvidenceKeys(candidate.sketch))
      if (!shared.size) continue
      entries.push(candidate)
      commonTransitions = shared
    }
    const score = entries.reduce((sum, entry) => sum + entry.score, 0)
    const key = entries.map((entry) => entry.sketch.pageIndex).join(':')
    if (!best || score > best.score + 1e-12 || (Math.abs(score - best.score) <= 1e-12 && key < best.key)) {
      best = { entries, commonTransitions, score, key }
    }
  }
  return best
}

function buildBirthProposal(atlas, sketches, observationBatchFingerprint, maximumRegions) {
  const liveCount = atlas.regions.filter((region) => region.lifecycle !== 'retired').length
  if (liveCount >= maximumRegions) return { action: 'birth', proposal: null, reason: 'region-capacity-limit', score: 0 }
  const ranked = sketches.map((sketch) => ({
    sketch,
    score: Math.max(0, Number(sketch.behaviorPrototype[3] ?? 0)) * Math.max(0, sketch.cohesion),
  })).filter((entry) => entry.score > 1e-6 && atlas.pages[entry.sketch.pageIndex].freeMass > 1e-6 &&
      atlas.pages[entry.sketch.pageIndex].memberships.length < atlas.maximumMembershipsPerPage)
    .sort((left, right) => right.score - left.score || left.sketch.pageIndex - right.sketch.pageIndex)
    .slice(0, 24)
  if (!ranked.length) return { action: 'birth', proposal: null, reason: 'no-unexplained-free-page', score: 0 }
  const cluster = coherentBirthCluster(ranked)
  const selected = cluster?.entries ?? []
  if (!selected.length) return { action: 'birth', proposal: null, reason: 'no-coherent-page-cluster', score: 0 }
  // A multi-page region may only claim evidence observed jointly on every
  // selected page. A union would let one richly observed anchor inflate a
  // co-activation that was witnessed by only one source/transition.
  const supportingTransitions = selected.length > 1
    ? new Set(cluster.commonTransitions)
    : transitionEvidenceKeys(selected[0].sketch)
  const sourceGroupIds = [...new Set([...supportingTransitions].map((key) => key.split('\u0000', 1)[0]))]
    .filter(Boolean)
    .sort(compareText)
  if (!sourceGroupIds.length) return { action: 'birth', proposal: null, reason: 'no-independent-source-evidence', score: 0 }
  const allocations = selected.map(({ sketch }) => ({
    pageIndex: sketch.pageIndex,
    weight: Math.min(0.25, atlas.pages[sketch.pageIndex].freeMass),
  }))
  const prototypeValues = weightedRegionMean(selected.map(({ sketch, score }) => ({
    values: sketch.behaviorPrototype,
    weight: score,
  })))
  const observationCount = supportingTransitions.size
  const confidenceMass = Math.min(
    sourceGroupIds.length,
    selected.reduce((sum, entry) => sum + entry.sketch.confidenceMass, 0) / selected.length,
  )
  const withinPageCohesion = selected.reduce((sum, entry) => sum + entry.sketch.cohesion, 0) / selected.length
  const crossPageCohesion = selected.reduce((sum, entry) =>
    sum + Math.max(0, cosineRegionVectors(entry.sketch.behaviorPrototype, prototypeValues)), 0) / selected.length
  const cohesion = Math.min(withinPageCohesion, crossPageCohesion)
  const evidenceId = stableRuntimeHash(
    PLASTIC_REGION_PIPELINE_VERSION,
    'birth-evidence',
    observationBatchFingerprint,
    selected.map((entry) => [entry.sketch.pageIndex, entry.sketch.fingerprint]),
    [...supportingTransitions].sort(compareText),
  )
  const regionId = deriveLatentRegionId(atlas, evidenceId)
  const proposal = proposeRegionBirth(atlas, {
    regionId,
    allocations,
    prototype: { dimension: atlas.behaviorDimension, values: prototypeValues },
    evidence: {
      evidenceId,
      validationAuthority: 'train-only',
      observationCount,
      sourceGroupIds,
      confidenceMass,
      cohesion,
      // Training traces cannot validate themselves. The resulting proposal is
      // intentionally ineligible until an external holdout/sham evaluator
      // produces a later evidence revision.
      holdoutGain: 0,
      shamGain: 0,
      codes: selected.length > 1 && cluster.commonTransitions.size
        ? ['stable-coactivation', 'unexplained-residual']
        : ['unexplained-residual'],
    },
  })
  return {
    action: 'birth', proposal, reason: null,
    score: selected.reduce((sum, entry) => sum + entry.score, 0) / selected.length,
  }
}

function regionMembershipPages(atlas, regionId) {
  return atlas.pages.flatMap((page) => {
    const membership = page.memberships.find((entry) => entry.regionId === regionId)
    return membership ? [{ page, weight: membership.weight }] : []
  })
}

function actionEvidence({
  action,
  atlas,
  targetRegionIds,
  sketches,
  observationBatchFingerprint,
  codes,
  joint = false,
  residualOnly = true,
}) {
  if (!sketches.length) return null
  const sets = sketches.map((sketch) => residualOnly
    ? transitionEvidenceKeys(sketch)
    : allTransitionEvidenceKeys(sketch))
  const transitionKeys = joint
    ? sets.slice(1).reduce((shared, values) => intersection(shared, values), sets[0])
    : new Set(sets.flatMap((values) => [...values]))
  if (!transitionKeys.size) return null
  const sourceGroupIds = [...new Set([...transitionKeys].map((key) => key.split('\u0000', 1)[0]))]
    .filter(Boolean)
    .sort(compareText)
  if (!sourceGroupIds.length) return null
  const confidenceMass = Math.min(
    sourceGroupIds.length,
    sketches.reduce((sum, sketch) => sum + sketch.confidenceMass, 0) / sketches.length,
  )
  const cohesion = Math.min(...sketches.map((sketch) => sketch.cohesion))
  const evidenceId = stableRuntimeHash(
    PLASTIC_REGION_PIPELINE_VERSION,
    `${action}-evidence`,
    atlas.fingerprint,
    observationBatchFingerprint,
    [...targetRegionIds].sort(compareText),
    sketches.map((sketch) => [sketch.pageIndex, sketch.fingerprint]),
    [...transitionKeys].sort(compareText),
  )
  return {
    evidenceId,
    validationAuthority: 'train-only',
    observationCount: transitionKeys.size,
    sourceGroupIds,
    confidenceMass,
    cohesion,
    holdoutGain: 0,
    shamGain: 0,
    codes,
  }
}

function buildGrowProposal(atlas, sketches, observationBatchFingerprint) {
  const sketchByPage = new Map(sketches.map((sketch) => [sketch.pageIndex, sketch]))
  const candidates = []
  for (const region of atlas.regions.filter((entry) => entry.lifecycle !== 'retired' && entry.lifecycle !== 'retiring')) {
    for (const sketch of sketches) {
      const page = atlas.pages[sketch.pageIndex]
      const alreadyMember = page.memberships.some((entry) => entry.regionId === region.id)
      if (page.freeMass <= 1e-6 || (!alreadyMember && page.memberships.length >= atlas.maximumMembershipsPerPage)) continue
      const similarity = cosineRegionVectors(region.prototype.values, sketch.behaviorPrototype)
      const residual = Math.max(0, Number(sketch.behaviorPrototype[3] ?? 0))
      if (similarity < 0.65 || residual <= 1e-6) continue
      candidates.push({ region, sketch, score: similarity * residual * page.freeMass })
    }
  }
  candidates.sort((left, right) => right.score - left.score ||
    compareText(left.region.id, right.region.id) || left.sketch.pageIndex - right.sketch.pageIndex)
  const best = candidates[0]
  if (!best) return { action: 'grow', proposal: null, reason: 'no-compatible-free-capacity', score: 0 }
  const evidence = actionEvidence({
    action: 'grow', atlas, targetRegionIds: [best.region.id], sketches: [best.sketch],
    observationBatchFingerprint, codes: ['stable-coactivation', 'capacity-pressure'],
  })
  if (!evidence) return { action: 'grow', proposal: null, reason: 'no-independent-source-evidence', score: best.score }
  const proposal = proposeRegionGrow(atlas, {
    regionId: best.region.id,
    allocations: [{ pageIndex: best.sketch.pageIndex, weight: Math.min(0.25, atlas.pages[best.sketch.pageIndex].freeMass) }],
    evidence,
  })
  return { action: 'grow', proposal, reason: null, score: best.score }
}

function buildShrinkProposal(atlas, sketches, observationBatchFingerprint) {
  const sketchByPage = new Map(sketches.map((sketch) => [sketch.pageIndex, sketch]))
  const candidates = []
  for (const region of atlas.regions.filter((entry) => entry.lifecycle !== 'retired' && entry.lifecycle !== 'retiring')) {
    const capacity = plasticRegionCapacity(atlas, region.id)
    for (const { page, weight } of regionMembershipPages(atlas, region.id)) {
      const sketch = sketchByPage.get(page.pageIndex)
      if (!sketch) continue
      const similarity = cosineRegionVectors(region.prototype.values, sketch.behaviorPrototype)
      const residual = Math.max(0, Number(sketch.behaviorPrototype[3] ?? 0))
      const minimumRemaining = region.lifecycle === 'active' ? 1 : 1e-6
      const releasable = capacity - minimumRemaining
      if (similarity >= 0.25 || releasable <= 1e-6) continue
      candidates.push({
        region, sketch, weight,
        release: Math.min(0.125, weight * 0.25, releasable),
        score: weight * (1 - Math.max(-1, similarity)) * (1 - Math.min(1, residual)) / 2,
      })
    }
  }
  candidates.sort((left, right) => right.score - left.score ||
    compareText(left.region.id, right.region.id) || left.sketch.pageIndex - right.sketch.pageIndex)
  const best = candidates.find((entry) => entry.release > 1e-6)
  if (!best) return { action: 'shrink', proposal: null, reason: 'no-low-utility-releasable-capacity', score: 0 }
  const evidence = actionEvidence({
    action: 'shrink', atlas, targetRegionIds: [best.region.id], sketches: [best.sketch],
    observationBatchFingerprint, codes: ['low-unique-utility'], residualOnly: false,
  })
  if (!evidence) return { action: 'shrink', proposal: null, reason: 'no-independent-source-evidence', score: best.score }
  const proposal = proposeRegionShrink(atlas, {
    regionId: best.region.id,
    allocations: [{ pageIndex: best.sketch.pageIndex, weight: best.release }],
    evidence,
  })
  return { action: 'shrink', proposal, reason: null, score: best.score }
}

function buildSplitProposal(atlas, sketches, observationBatchFingerprint, maximumRegions) {
  const liveRegions = atlas.regions.filter((entry) => entry.lifecycle !== 'retired' && entry.lifecycle !== 'retiring')
  if (liveRegions.length >= maximumRegions) {
    return { action: 'split', proposal: null, reason: 'region-capacity-limit', score: 0 }
  }
  const sketchByPage = new Map(sketches.map((sketch) => [sketch.pageIndex, sketch]))
  const candidates = []
  for (const region of liveRegions) {
    const memberships = regionMembershipPages(atlas, region.id)
    if (memberships.length < 2 || memberships.some(({ page }) =>
      page.memberships.length >= atlas.maximumMembershipsPerPage)) continue
    for (let first = 0; first < memberships.length; first += 1) {
      for (let second = first + 1; second < memberships.length; second += 1) {
        const left = sketchByPage.get(memberships[first].page.pageIndex)
        const right = sketchByPage.get(memberships[second].page.pageIndex)
        if (!left || !right) continue
        const similarity = cosineRegionVectors(left.behaviorPrototype, right.behaviorPrototype)
        if (similarity > -0.25 || !intersection(transitionEvidenceKeys(left), transitionEvidenceKeys(right)).size) continue
        candidates.push({
          region, memberships, left, right,
          score: -similarity * Math.min(memberships[first].weight, memberships[second].weight),
        })
      }
    }
  }
  candidates.sort((left, right) => right.score - left.score || compareText(left.region.id, right.region.id) ||
    left.left.pageIndex - right.left.pageIndex || left.right.pageIndex - right.right.pageIndex)
  const best = candidates[0]
  if (!best) return { action: 'split', proposal: null, reason: 'no-bimodal-region-evidence', score: 0 }
  const evidence = actionEvidence({
    action: 'split', atlas, targetRegionIds: [best.region.id], sketches: [best.left, best.right],
    observationBatchFingerprint, codes: ['bimodal-behavior'], joint: true,
  })
  if (!evidence) return { action: 'split', proposal: null, reason: 'no-joint-source-evidence', score: best.score }
  const firstChildId = deriveLatentRegionId(atlas, evidence.evidenceId, 0)
  const secondChildId = deriveLatentRegionId(atlas, evidence.evidenceId, 1)
  const fractions = best.memberships.map(({ page }) => {
    const sketch = sketchByPage.get(page.pageIndex)
    if (!sketch) return { pageIndex: page.pageIndex, firstChildFraction: 0.5 }
    const firstFit = cosineRegionVectors(sketch.behaviorPrototype, best.left.behaviorPrototype)
    const secondFit = cosineRegionVectors(sketch.behaviorPrototype, best.right.behaviorPrototype)
    return {
      pageIndex: page.pageIndex,
      firstChildFraction: Math.max(0.1, Math.min(0.9, 0.5 + 0.25 * (firstFit - secondFit))),
    }
  })
  const proposal = proposeRegionSplit(atlas, {
    sourceRegionId: best.region.id,
    firstChildId,
    secondChildId,
    firstChildPrototype: { dimension: atlas.behaviorDimension, values: best.left.behaviorPrototype },
    secondChildPrototype: { dimension: atlas.behaviorDimension, values: best.right.behaviorPrototype },
    fractions,
    evidence,
  })
  return { action: 'split', proposal, reason: null, score: best.score }
}

function buildMergeProposal(atlas, sketches, observationBatchFingerprint) {
  const liveRegions = atlas.regions.filter((entry) => entry.lifecycle !== 'retired' && entry.lifecycle !== 'retiring')
  const sketchByPage = new Map(sketches.map((sketch) => [sketch.pageIndex, sketch]))
  const candidates = []
  for (let first = 0; first < liveRegions.length; first += 1) {
    for (let second = first + 1; second < liveRegions.length; second += 1) {
      const left = liveRegions[first]
      const right = liveRegions[second]
      const similarity = cosineRegionVectors(left.prototype.values, right.prototype.values)
      if (similarity < 0.9) continue
      const leftPages = new Set(regionMembershipPages(atlas, left.id).map(({ page }) => page.pageIndex))
      const sharedPages = regionMembershipPages(atlas, right.id)
        .map(({ page }) => page.pageIndex)
        .filter((pageIndex) => leftPages.has(pageIndex) && sketchByPage.has(pageIndex))
      if (!sharedPages.length) continue
      candidates.push({
        left, right, similarity, sharedPages,
        score: similarity * sharedPages.length / Math.max(leftPages.size, regionMembershipPages(atlas, right.id).length),
      })
    }
  }
  candidates.sort((left, right) => right.score - left.score ||
    compareText(left.left.id, right.left.id) || compareText(left.right.id, right.right.id))
  const best = candidates[0]
  if (!best) return { action: 'merge', proposal: null, reason: 'no-redundant-region-pair', score: 0 }
  const relevantSketches = best.sharedPages.map((pageIndex) => sketchByPage.get(pageIndex)).filter(Boolean)
  const evidence = actionEvidence({
    action: 'merge', atlas, targetRegionIds: [best.left.id, best.right.id], sketches: relevantSketches,
    observationBatchFingerprint, codes: ['redundant-behavior', 'stable-coactivation'], residualOnly: false,
  })
  if (!evidence) return { action: 'merge', proposal: null, reason: 'no-independent-source-evidence', score: best.score }
  const leftCapacity = plasticRegionCapacity(atlas, best.left.id)
  const rightCapacity = plasticRegionCapacity(atlas, best.right.id)
  const prototype = weightedRegionMean([
    { values: best.left.prototype.values, weight: leftCapacity },
    { values: best.right.prototype.values, weight: rightCapacity },
  ])
  const mergedRegionId = deriveLatentRegionId(atlas, evidence.evidenceId)
  const proposal = proposeRegionMerge(atlas, {
    sourceRegionIds: [best.left.id, best.right.id], mergedRegionId,
    prototype: { dimension: atlas.behaviorDimension, values: prototype }, evidence,
  })
  return { action: 'merge', proposal, reason: null, score: best.score }
}

function buildRetireProposal(atlas, sketches, observationBatchFingerprint) {
  const sketchByPage = new Map(sketches.map((sketch) => [sketch.pageIndex, sketch]))
  const candidates = []
  for (const region of atlas.regions.filter((entry) => entry.lifecycle !== 'retired' && entry.lifecycle !== 'retiring')) {
    const memberships = regionMembershipPages(atlas, region.id)
    const relevant = memberships.map(({ page }) => sketchByPage.get(page.pageIndex)).filter(Boolean)
    if (!relevant.length) continue
    const capacity = plasticRegionCapacity(atlas, region.id)
    const fit = relevant.reduce((sum, sketch) =>
      sum + Math.max(0, cosineRegionVectors(region.prototype.values, sketch.behaviorPrototype)), 0) / relevant.length
    const residual = relevant.reduce((sum, sketch) =>
      sum + Math.max(0, Number(sketch.behaviorPrototype[3] ?? 0)), 0) / relevant.length
    if (capacity > 1.25 || fit > 0.1 || residual > 0.1) continue
    candidates.push({ region, relevant, score: (1 / Math.max(1, capacity)) * (1 - fit) * (1 - residual) })
  }
  candidates.sort((left, right) => right.score - left.score || compareText(left.region.id, right.region.id))
  const best = candidates[0]
  if (!best) return { action: 'retire', proposal: null, reason: 'no-low-unique-utility-region', score: 0 }
  const evidence = actionEvidence({
    action: 'retire', atlas, targetRegionIds: [best.region.id], sketches: best.relevant,
    observationBatchFingerprint, codes: ['low-unique-utility'], residualOnly: false,
  })
  if (!evidence) return { action: 'retire', proposal: null, reason: 'no-independent-source-evidence', score: best.score }
  const proposal = proposeRegionRetire(atlas, { regionId: best.region.id, evidence })
  return { action: 'retire', proposal, reason: null, score: best.score }
}

const LIFECYCLE_ACTION_ORDER = Object.freeze(['split', 'merge', 'retire', 'shrink', 'grow', 'birth'])

/** Selects at most one immutable lifecycle proposal for a base atlas revision. */
export function selectPlasticRegionLifecycleProposal(
  atlas,
  sketches,
  observationBatchFingerprint,
  { maximumRegions = 56 } = {},
) {
  validateAdaptiveRegionAtlas(atlas)
  const builders = [
    () => buildSplitProposal(atlas, sketches, observationBatchFingerprint, maximumRegions),
    () => buildMergeProposal(atlas, sketches, observationBatchFingerprint),
    () => buildRetireProposal(atlas, sketches, observationBatchFingerprint),
    () => buildShrinkProposal(atlas, sketches, observationBatchFingerprint),
    () => buildGrowProposal(atlas, sketches, observationBatchFingerprint),
    () => buildBirthProposal(atlas, sketches, observationBatchFingerprint, maximumRegions),
  ]
  const outcomes = builders.map((build) => {
    try {
      return build()
    } catch (error) {
      return {
        action: 'unknown', proposal: null, score: 0,
        reason: `capacity-or-invariant:${String(error?.message ?? error).slice(0, 160)}`,
      }
    }
  }).map((outcome, index) => ({ ...outcome, action: LIFECYCLE_ACTION_ORDER[index] }))
  const selected = outcomes.filter((outcome) => outcome.proposal)
    .sort((left, right) => right.score - left.score ||
      LIFECYCLE_ACTION_ORDER.indexOf(left.action) - LIFECYCLE_ACTION_ORDER.indexOf(right.action) ||
      compareText(left.proposal.fingerprint, right.proposal.fingerprint))[0] ?? null
  return freezeDeep({
    proposal: selected?.proposal ?? null,
    reason: selected ? null : 'no-lifecycle-action',
    selectedAction: selected?.action ?? null,
    baseRevision: atlas.revision,
    oneProposalPerRevision: true,
    candidates: outcomes.map((outcome) => ({
      action: outcome.action,
      score: outcome.score,
      proposed: Boolean(outcome.proposal),
      eligible: outcome.proposal?.eligible ?? false,
      rejectionReasons: outcome.proposal?.rejectionReasons ?? [],
      reason: outcome.reason,
    })),
  })
}

export function buildPlasticRegionCycle(input) {
  if (input.mode !== 'shadow' && input.mode !== 'proposal') {
    throw pipelineError('INVALID_PLASTIC_REGION_MODE', 'Plastik bölge çevrimi yalnız shadow veya proposal kipinde kurulabilir.')
  }
  const atlas = input.baseAtlas ?? createInitialPlasticRegionAtlas({
    representationSpaceId: input.representationSpaceId,
    behaviorDimension: PAGE_BEHAVIOR_DIMENSION,
    maximumMembershipsPerPage: input.maximumMembershipsPerPage ?? 3,
  })
  validateAdaptiveRegionAtlas(atlas)
  if (atlas.representationSpaceId !== input.representationSpaceId) {
    throw pipelineError('REGION_ATLAS_REPRESENTATION_MISMATCH', 'Atlas ile düşünce gözlemleri farklı temsil uzaylarına ait.')
  }
  if (atlas.behaviorDimension !== PAGE_BEHAVIOR_DIMENSION) {
    throw pipelineError('REGION_ATLAS_BEHAVIOR_MISMATCH', 'Atlas davranış prototipi bu pipeline sürümüyle uyuşmuyor.')
  }
  const batches = createPlasticRegionObservationBatches(input)
  const retainedObservations = batches.flatMap((batch) => batch.observations)
  const evidenceHead = createPlasticRegionEvidenceHead({
    representationSpaceId: input.representationSpaceId,
    baseAtlasFingerprint: atlas.fingerprint,
    authority: input.evidenceAuthority,
    maximumObservations: input.maximumCumulativeObservations ?? 8_192,
    parentHeads: input.priorEvidenceHeads ?? [],
    observations: retainedObservations,
  })
  const sketches = aggregatePageSketches(evidenceHead.observations, {
    behaviorDimension: PAGE_BEHAVIOR_DIMENSION,
    maximumEvidence: 64,
  })
  const gates = computeRegionGates(atlas, {
    pageCompatibilities: pageCompatibilityVector(sketches),
    pageBehaviors: Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_, pageIndex) =>
      sketches.find((sketch) => sketch.pageIndex === pageIndex)?.behaviorPrototype ??
      new Array(PAGE_BEHAVIOR_DIMENSION).fill(0)),
    maximumActiveRegions: input.maximumActiveGates ?? 4,
  })
  const combinedBatchFingerprint = stableRuntimeHash(
    batches.map((batch) => batch.fingerprint),
    evidenceHead.fingerprint,
  )
  const lifecycleDecision = input.mode === 'proposal' && evidenceHead.observations.length
    ? selectPlasticRegionLifecycleProposal(atlas, sketches, combinedBatchFingerprint, {
      maximumRegions: input.maximumRegions ?? 56,
    })
    : {
      proposal: null,
      reason: input.mode === 'shadow' ? 'shadow-mode' : 'no-observations',
      selectedAction: null,
      baseRevision: atlas.revision,
      oneProposalPerRevision: true,
      candidates: [],
    }
  const shadowBase = {
    format: PLASTIC_REGION_SHADOW_ARTIFACT_KIND,
    version: 1,
    representationSpaceId: input.representationSpaceId,
    geometryFingerprint: ADAPTIVE_REGION_GEOMETRY.fingerprint,
    baseAtlasFingerprint: atlas.fingerprint,
    inputManifestHash: input.inputManifestHash,
    observationBatchFingerprints: batches.map((batch) => batch.fingerprint),
    observationCount: retainedObservations.length,
    cumulativeObservationCount: evidenceHead.observationCount,
    cumulativeTransitionCount: evidenceHead.transitionCount,
    independentSourceGroupCount: new Set(retainedObservations.map((value) => value.sourceGroupId)).size,
    sketchCount: sketches.length,
    routingMutated: false,
    persistentModelWrites: 0,
    atlasActivated: false,
    gatePreview: gates,
    proposalCreated: Boolean(lifecycleDecision.proposal),
    proposalReason: lifecycleDecision.reason,
    lifecycleDecision: {
      selectedAction: lifecycleDecision.selectedAction,
      baseRevision: lifecycleDecision.baseRevision,
      oneProposalPerRevision: lifecycleDecision.oneProposalPerRevision,
      candidates: lifecycleDecision.candidates,
    },
  }
  const shadowReport = freezeDeep({
    ...shadowBase,
    fingerprint: stableRuntimeHash(PLASTIC_REGION_PIPELINE_VERSION, 'shadow-report', shadowBase),
  })
  return freezeDeep({
    atlas,
    observationBatches: batches,
    evidenceHead,
    sketches,
    shadowReport,
    // Historical empty probe-cache skeletons remain readable for audit, but a
    // normal region cycle no longer manufactures or persists an empty semantic
    // claim. The NEW opt-in probe producer writes only measured artifacts.
    probeCache: null,
    lifecycleDecision,
    proposal: lifecycleDecision.proposal,
    atlasCandidate: lifecycleDecision.proposal?.proposedAtlas ?? null,
  })
}

export function loadPlasticRegionBaseAtlas(store, representationSpaceId, maximumMembershipsPerPage = 3) {
  const activeSnapshot = store.getActiveSnapshot?.() ?? null
  const persisted = readPlasticRegionSnapshot(store, activeSnapshot)
  if (persisted.metadata && !persisted.compatible) {
    throw pipelineError(
      'INVALID_ACTIVE_REGION_ATLAS',
      'Etkin snapshot üzerindeki plastik bölge atlası doğrulanamadı; yeni atlasla sessizce değiştirilemez.',
      { reason: persisted.reason },
    )
  }
  if (!persisted.head) {
    return createInitialPlasticRegionAtlas({
      representationSpaceId,
      behaviorDimension: PAGE_BEHAVIOR_DIMENSION,
      maximumMembershipsPerPage,
    })
  }
  validateAdaptiveRegionAtlas(persisted.head)
  if (persisted.head.representationSpaceId !== representationSpaceId) {
    throw pipelineError(
      'REGION_ATLAS_REPRESENTATION_MISMATCH',
      'Etkin snapshot plastik bölge atlası bu çevrimin temsil uzayıyla uyuşmuyor.',
    )
  }
  return persisted.head
}

/**
 * Observation/report/proposal artifacts may join the normal candidate audit.
 * The atlas head is deliberately returned as detached so normal snapshot
 * activation cannot enable it automatically.
 */
export function persistPlasticRegionCycle(context, cycle, options = {}) {
  const expectedBase = options.expectedBase ?? currentPlasticRegionBase(context.store)
  const assertUnchangedBase = () => {
    const current = currentPlasticRegionBase(context.store)
    for (const field of ['snapshotId', 'atlasArtifactId', 'atlasContentHash', 'atlasFingerprint']) {
      if ((current[field] ?? null) !== (expectedBase[field] ?? null)) {
        throw pipelineError(
          'STALE_REGION_ATLAS_BASE',
          'Plastik bölge gözlem çevriminin dayandığı snapshot/atlas tabanı değişti.',
          { field, expected: expectedBase[field] ?? null, actual: current[field] ?? null },
        )
      }
    }
  }
  assertUnchangedBase()
  const safetyMode = expectedBase.snapshotId
    ? 'cas-active-base'
    : 'trace-only-no-active-base'

  const persistCycleArtifact = ({ kind, value, metadata }) => {
    assertUnchangedBase()
    if (expectedBase.snapshotId) {
      return persistPlasticRegionArtifact(context, {
        kind,
        value,
        expectedBase,
        contentHash: stableRuntimeHash(
          PLASTIC_REGION_PIPELINE_VERSION,
          'base-bound-artifact',
          kind,
          expectedBase,
          value,
        ),
        metadata,
      }).artifact
    }

    // There is no active lineage to compare-and-swap against. Keep the
    // evidence useful and replayable, but make the missing authority explicit:
    // these ordinary immutable artifacts may enter the candidate audit while
    // the atlas head itself remains detached and activation-ineligible.
    assertPlasticRegionArtifactValue(kind, value)
    const encoded = encodeArtifactValue(value)
    const bytes = encodedByteLength(encoded)
    const maximumBytes = Number(
      context.store.getConfig?.().plasticRegions?.maximumArtifactBytes ?? 4 * 1024 * 1024,
    )
    if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1 || bytes > maximumBytes) {
      throw pipelineError(
        'PLASTIC_REGION_ARTIFACT_TOO_LARGE',
        `${kind} plastik bölge artifacti ${maximumBytes} byte sınırını aşıyor.`,
        { bytes, maximumBytes },
      )
    }
    const contentHash = stableRuntimeHash('plastic-region-artifact-v1', kind, encoded)
    return persistArtifact(context, {
      kind,
      contentHash,
      value,
      metadata: {
        ...metadata,
        representationSpaceId: kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND
          ? value.proposedAtlas.representationSpaceId
          : value.representationSpaceId,
        fingerprint: value.fingerprint ?? contentHash,
        ...(kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ? {
          geometryFingerprint: value.proposedAtlas.geometryFingerprint,
          revision: value.proposedAtlas.revision,
          proposalId: value.proposalId,
          proposedAtlasFingerprint: value.proposedAtlas.fingerprint,
          eligible: value.eligible,
        } : {}),
        ...(kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND ? {
          geometryFingerprint: value.geometryFingerprint,
          revision: value.revision,
          behaviorDimension: value.behaviorDimension,
          maximumMembershipsPerPage: value.maximumMembershipsPerPage,
        } : {}),
        ...(kind === PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND ? {
          geometryFingerprint: value.geometryFingerprint,
          revision: value.revision,
          authority: value.authority,
        } : {}),
        baseSnapshotId: null,
        baseAtlasArtifactId: null,
        baseAtlasFingerprint: kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ||
          kind === PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND
          ? value.baseAtlasFingerprint
          : null,
        traceOnly: true,
      },
    }).artifact
  }
  const attachedArtifacts = []
  for (const batch of cycle.observationBatches) {
    attachedArtifacts.push(persistCycleArtifact({
      kind: PLASTIC_REGION_OBSERVATION_ARTIFACT_KIND,
      value: batch,
      metadata: { observationCount: batch.observationCount, evaluationObservationCount: 0 },
    }))
  }
  attachedArtifacts.push(persistCycleArtifact({
    kind: PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
    value: cycle.evidenceHead,
    metadata: {
      observationCount: cycle.evidenceHead.observationCount,
      transitionCount: cycle.evidenceHead.transitionCount,
      sourceGroupCount: cycle.evidenceHead.sourceGroupIds.length,
    },
  }))
  attachedArtifacts.push(persistCycleArtifact({
    kind: PLASTIC_REGION_SHADOW_ARTIFACT_KIND,
    value: cycle.shadowReport,
    metadata: { routingMutated: false, persistentModelWrites: 0 },
  }))
  if (cycle.probeCache) {
    // Compatibility guard for explicitly supplied, non-empty future readers;
    // buildPlasticRegionCycle itself always returns null. Empty legacy
    // skeletons are never copied forward or rewritten.
    const meaningful = Array.isArray(cycle.probeCache.semanticAssignments) &&
      cycle.probeCache.semanticAssignments.length > 0
    if (!meaningful) {
      throw pipelineError(
        'EMPTY_PLASTIC_REGION_PROBE_ARTIFACT_FORBIDDEN',
        'Boş anlamsal probe skeletonı yeniden kalıcılaştırılamaz.',
      )
    }
    attachedArtifacts.push(persistCycleArtifact({
      kind: PLASTIC_REGION_PROBE_CACHE_ARTIFACT_KIND,
      value: cycle.probeCache,
      metadata: {
        semanticAssignmentCount: cycle.probeCache.semanticAssignments.length,
        trainingWritesAllowed: false,
      },
    }))
  }

  let proposalArtifact = null
  let detachedAtlasCandidateArtifact = null
  if (cycle.proposal) {
    proposalArtifact = persistCycleArtifact({
      kind: PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
      value: cycle.proposal,
      metadata: { eligible: cycle.proposal.eligible, activationEligible: false },
    })
    attachedArtifacts.push(proposalArtifact)
    detachedAtlasCandidateArtifact = persistCycleArtifact({
      kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      value: cycle.atlasCandidate,
      metadata: {
        candidateOnly: true,
        activationEligible: false,
        proposalArtifactId: proposalArtifact.id,
      },
    })
  }
  return freezeDeep({
    expectedBase,
    safetyMode,
    attachedArtifacts,
    proposalArtifact,
    detachedAtlasCandidateArtifact,
    atlasActivated: false,
  })
}
