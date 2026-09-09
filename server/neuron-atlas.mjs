import { loadFamilyProjectionSource, selectFamilyProjectionMetadata } from './family-inspection.mjs'
import {
  createLanguageOutputScaffold,
} from '../src/language-output/scaffold.ts'
import { createBusyLanguageAtlas } from '../src/universe-app/busy-atlas.ts'
import { readLanguageOutputSnapshot } from '../runtime/language-output-cache.mjs'
import {
  PLASTIC_REGION_ARTIFACT_KINDS,
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  readPlasticRegionSnapshot,
} from '../runtime/plastic-region-cache.mjs'
import { inspectPlasticRegionStatus } from './plastic-region-inspection.mjs'
import { restoreArtifact, stableRuntimeHash } from '../runtime/artifacts.mjs'

const MAX_ATLAS_FAMILIES = 4_096
const MAX_ATLAS_NEURONS = 100_000
const MAX_ATLAS_MEMBERSHIPS = 500_000
const MAX_ATLAS_FAMILY_TRANSITIONS = 250_000
const MAX_ATLAS_NEURON_TRANSITIONS = 500_000
const MAX_ATLAS_CONCEPT_MACROS = 4_096
const MAX_ATLAS_MACRO_EXPANSION_LINKS = 32_768
const MAX_ATLAS_MACRO_EXPANSION_LINKS_PER_MACRO = 32
const MAX_TEXT_LENGTH = 160
const MAX_WARNING_COUNT = 32
const MAX_WARNING_LENGTH = 260

const atlasCacheByStore = new WeakMap()
const MEMORY_HEAVY_JOB_TYPES = new Set(['teacher-feed-campaign', 'full-cycle'])
const LANGUAGE_OUTPUT_SEED_FINGERPRINT = createLanguageOutputScaffold().fingerprint
const CONCEPT_MACRO_HEAD_KIND = 'concept-macro-head'

function atlasError(message, code, status = 409, details) {
  const error = new Error(message)
  error.code = code
  error.status = status
  if (details !== undefined) error.details = details
  return error
}

function array(value) { return Array.isArray(value) ? value : [] }
function count(value) { return Number.isSafeInteger(value) && value >= 0 ? value : 0 }
function finite(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback }
function rounded(value) { return Math.round(finite(value) * 1_000_000) / 1_000_000 }
function bounded(value, maximum) { return typeof value === 'string' ? value.slice(0, maximum) : '' }

function displayText(value) {
  const normalized = typeof value === 'string' ? value.replace(/\s+/gu, ' ').trim() : ''
  return normalized.length <= MAX_TEXT_LENGTH
    ? normalized
    : `${normalized.slice(0, MAX_TEXT_LENGTH - 1).trimEnd()}…`
}

function validateAtlasSize(run) {
  const sizes = {
    familyCount: array(run?.families).length,
    neuronCount: array(run?.profiles).length,
    membershipCount: array(run?.memberships).length,
    familyTransitionCount: array(run?.familyTransitions).length,
    neuronTransitionCount: array(run?.neuronTransitions).length,
  }
  const limits = {
    familyCount: MAX_ATLAS_FAMILIES,
    neuronCount: MAX_ATLAS_NEURONS,
    membershipCount: MAX_ATLAS_MEMBERSHIPS,
    familyTransitionCount: MAX_ATLAS_FAMILY_TRANSITIONS,
    neuronTransitionCount: MAX_ATLAS_NEURON_TRANSITIONS,
  }
  const exceeded = Object.entries(sizes).find(([key, value]) => value > limits[key])
  if (exceeded) {
    throw atlasError(
      'Nöron evreni güvenli tam-görünüm sınırını aşıyor; yeni bir LOD snapshot görünümü gerekli.',
      'NEURON_ATLAS_LIMIT_EXCEEDED',
      409,
      { sizes, limits, exceeded: exceeded[0] },
    )
  }
  return { sizes, limits }
}

function snapshotReference(store, snapshot) {
  if (!snapshot) return null
  return {
    id: bounded(snapshot.id, 256),
    name: bounded(snapshot.label, 200),
    status: bounded(snapshot.status, 40),
    createdAt: bounded(snapshot.createdAt, 64),
    active: store.getActiveSnapshot()?.id === snapshot.id,
  }
}

function artifactReference(artifact) {
  if (!artifact) return null
  return {
    id: bounded(artifact.id, 256),
    kind: bounded(artifact.kind, 80),
    contentHash: bounded(artifact.contentHash, 256) || null,
    createdAt: bounded(artifact.createdAt, 64),
    jobId: bounded(artifact.jobId, 256) || null,
  }
}

function plasticRegionArtifactMetadataForSnapshot(store, snapshot) {
  if (!snapshot) return null
  for (const artifactId of [...array(snapshot.artifactIds)].reverse()) {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (metadata?.kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND) return metadata
  }
  return null
}

/**
 * Projects only bounded allocation/evidence summaries. Behavior prototypes
 * stay inside the region artifact and never enter the 3B neuron payload.
 */
function inspectPlasticRegionsForAtlas(store, snapshot, run) {
  const metadata = plasticRegionArtifactMetadataForSnapshot(store, snapshot)
  if (!metadata) return { projection: null, warnings: [] }
  const record = readPlasticRegionSnapshot(store, snapshot)
  if (!record.head) {
    return {
      projection: null,
      warnings: [`Plastik bölge atlası karantinaya alındı: ${bounded(record.reason, MAX_WARNING_LENGTH - 38) || 'head okunamadı'}.`],
    }
  }

  const head = record.head
  const compatible = head.representationSpaceId === run.representationSpaceId
  const mismatch = compatible ? null : 'representation-space-mismatch'
  const regionConfig = store.getConfig?.().plasticRegions ?? {}
  let inspected = null
  let inspectionWarning = null
  try {
    inspected = inspectPlasticRegionStatus(store)
  } catch (error) {
    inspectionWarning = `Plastik bölge çalışma uygunluğu okunamadı: ${bounded(error?.message, MAX_WARNING_LENGTH - 47) || 'denetim başarısız'}.`
  }
  const configuredMode = bounded(inspected?.configuredMode ?? regionConfig.mode, 20) || 'off'
  const exactRuntimeBinding = Boolean(
    compatible &&
    inspected?.snapshot?.id === snapshot?.id &&
    inspected?.head?.artifactId === metadata.id &&
    inspected?.head?.fingerprint === head.fingerprint,
  )
  const effectiveMode = exactRuntimeBinding ? bounded(inspected?.effectiveMode, 20) || 'off' : 'off'
  const effectiveModeReason = !compatible
    ? 'representation-space-mismatch'
    : !inspected ? 'runtime-inspection-failed'
      : inspected.snapshot?.id !== snapshot?.id ? 'selected-snapshot-not-runtime-base'
        : inspected.head?.artifactId !== metadata.id || inspected.head?.fingerprint !== head.fingerprint ? 'selected-atlas-not-runtime-head'
          : bounded(inspected.effectiveModeReason, 160) || 'unknown'
  const release = exactRuntimeBinding && inspected?.release?.artifactId
    ? {
        eligible: true,
        artifactId: bounded(inspected.release.artifactId, 256),
        releaseId: bounded(inspected.release.releaseId, 256) || null,
        fingerprint: bounded(inspected.release.fingerprint, 256) || null,
        proposalId: bounded(inspected.release.proposalId, 256) || null,
        reason: null,
      }
    : {
        eligible: false,
        artifactId: null,
        releaseId: null,
        fingerprint: null,
        proposalId: null,
        reason: bounded(exactRuntimeBinding ? inspected?.release?.reason : effectiveModeReason, 160) || 'release-unavailable',
      }
  const regions = new Map()
  for (const region of array(head.regions).slice(0, 56)) {
    const id = bounded(region?.id, 256)
    if (!id || regions.has(id)) continue
    regions.set(id, {
      id,
      lifecycle: bounded(region?.lifecycle, 40) || 'provisional',
      origin: bounded(region?.origin, 40) || 'discovery',
      effectiveCapacity: 0,
      assignedPageCount: 0,
      dominantPageCount: 0,
      meanMembershipWeight: 0,
      globalShare: 0,
      cohesion: rounded(region?.evidence?.cohesion),
      observationCount: count(region?.evidence?.observationCount),
      sourceGroupCount: new Set(array(region?.evidence?.sourceGroupIds).filter((id) => typeof id === 'string' && id)).size,
    })
  }

  let assignedMass = 0
  let freeMass = 0
  const pages = []
  const seenPages = new Set()
  if (compatible) {
    for (const page of array(head.pages).slice(0, 56)) {
      const pageIndex = count(page?.pageIndex)
      if (pageIndex >= 56 || seenPages.has(pageIndex)) continue
      seenPages.add(pageIndex)
      const memberships = array(page?.memberships)
        .map((membership) => ({
          regionId: bounded(membership?.regionId, 256),
          weight: Math.max(0, Math.min(1, finite(membership?.weight))),
        }))
        .filter((membership) => membership.regionId && regions.has(membership.regionId) && membership.weight > 0)
        .sort((left, right) => right.weight - left.weight || left.regionId.localeCompare(right.regionId))
      const dominant = memberships[0] ?? null
      for (const membership of memberships) {
        const summary = regions.get(membership.regionId)
        summary.effectiveCapacity += membership.weight
        summary.assignedPageCount += 1
        assignedMass += membership.weight
      }
      if (dominant) regions.get(dominant.regionId).dominantPageCount += 1
      const pageFreeMass = Math.max(0, Math.min(1, finite(page?.freeMass, 1)))
      freeMass += pageFreeMass
      pages.push({
        pageIndex,
        dominantRegionId: dominant?.regionId ?? null,
        dominantWeight: rounded(dominant?.weight),
        freeMass: rounded(pageFreeMass),
        membershipCount: memberships.length,
      })
    }
  }
  pages.sort((left, right) => left.pageIndex - right.pageIndex)
  const regionRows = compatible ? [...regions.values()].map((region) => ({
    ...region,
    effectiveCapacity: rounded(region.effectiveCapacity),
    meanMembershipWeight: rounded(region.assignedPageCount ? region.effectiveCapacity / region.assignedPageCount : 0),
    globalShare: rounded(assignedMass ? region.effectiveCapacity / assignedMass : 0),
  })).sort((left, right) => right.effectiveCapacity - left.effectiveCapacity || left.id.localeCompare(right.id)) : []

  return {
    projection: {
      artifact: artifactReference(metadata),
      headFingerprint: bounded(head.fingerprint, 256) || metadata.contentHash || null,
      headRevision: count(head.revision),
      geometryFingerprint: bounded(head.geometryFingerprint, 256) || null,
      representationSpaceId: bounded(head.representationSpaceId, 256),
      configuredMode,
      effectiveMode,
      effectiveModeReason,
      runtimeWired: inspected?.runtimeWired === true,
      behaviorApplied: effectiveMode === 'active' && release.eligible,
      exactRuntimeBinding,
      release,
      compatible,
      mismatch,
      regionCount: compatible ? regionRows.length : 0,
      activeRegionCount: compatible ? regionRows.filter((region) => region.lifecycle === 'active').length : 0,
      pageCount: compatible ? pages.length : 0,
      assignedMass: compatible ? rounded(assignedMass) : 0,
      freeMass: compatible ? rounded(freeMass + Math.max(0, 56 - pages.length)) : 56,
      dominantRegionId: compatible ? regionRows[0]?.id ?? null : null,
      regions: regionRows,
      pages: compatible ? pages : [],
    },
    warnings: [
      ...(!compatible ? ['Plastik bölge atlası gösterilmedi: temsil uzayı seçili aile çalışmasıyla uyuşmuyor.'] : []),
      ...(inspectionWarning ? [inspectionWarning] : []),
    ],
  }
}

function diagnostics(profile) {
  const value = profile?.diagnostics ?? {}
  return [
    rounded(value.spanScale),
    rounded(value.relativePosition),
    rounded(value.depthScale),
    rounded(value.relationStrength),
    rounded(value.transitionEntropy),
    rounded(value.transitionConcentration),
    rounded(value.localContinuity),
    rounded(value.bridgeImpact),
    rounded(value.leftRightAsymmetry),
    rounded(value.shortScopePotential),
  ]
}

function strongestAffinity(profile) {
  const value = [...array(profile?.regionAffinities)]
    .filter((item) => typeof item?.region === 'string')
    .sort((left, right) => finite(right?.score) - finite(left?.score) || String(left.region).localeCompare(String(right.region)))[0]
  return value
    ? { region: bounded(value.region, 80), score: rounded(value.score), verified: Boolean(value.verified) }
    : null
}

function sourceCount(profile) {
  return new Set(array(profile?.evidence).map((item) => item?.sourceId).filter((id) => typeof id === 'string')).size
}

function logistic(value) {
  const limited = Math.max(-30, Math.min(30, finite(value)))
  return 1 / (1 + Math.exp(-limited))
}

function appendLanguageOutputOverlay({ store, selection, run, families, neurons, membershipTuples, familyTransitions }) {
  const persisted = readLanguageOutputSnapshot(store, selection.snapshot)
  const sourceScaffold = persisted.scaffold ?? createLanguageOutputScaffold()
  const head = persisted.head &&
    (persisted.head.familyRunFingerprint === null || persisted.head.familyRunFingerprint === run.fingerprint) &&
    (persisted.head.representationSpaceId === null || persisted.head.representationSpaceId === run.representationSpaceId)
    ? persisted.head
    : null
  const familyIndex = new Map(families.map((family) => [family.id, family.index]))
  const neuronIndex = new Map(neurons.map((neuron) => [neuron.id, neuron.index]))
  const learnedMemberKeysByFamily = new Map()
  const sourceGroupsByFamily = new Map()
  const supportByFamily = new Map()

  for (const definition of [...sourceScaffold.families].sort((left, right) => left.order - right.order || left.id.localeCompare(right.id))) {
    const index = families.length
    familyIndex.set(definition.id, index)
    const seedCount = sourceScaffold.seedNeurons.filter((neuron) => neuron.familyId === definition.id).length
    families.push({
      id: bounded(definition.id, 256),
      index,
      origin: 'manual-language-seed',
      layoutZone: 'language-surface',
      layoutPinned: true,
      layoutSlot: Math.max(0, definition.order - 1),
      parentFamilyId: null,
      label: bounded(definition.label, MAX_TEXT_LENGTH),
      region: 'language',
      regionVerified: false,
      lifecycle: 'manual-seed',
      supportCount: 0,
      sourceCount: 0,
      memberCount: seedCount,
      coreMemberCount: seedCount,
      cohesion: 0,
      separation: 0,
      stability: 1,
      utility: null,
      utilityStatus: 'manual-scaffold',
      predecessors: [],
    })
  }

  for (const seed of sourceScaffold.seedNeurons) {
    const family = familyIndex.get(seed.familyId)
    if (!Number.isInteger(family)) continue
    const index = neurons.length
    neuronIndex.set(seed.id, index)
    neurons.push({
      id: bounded(seed.id, 512),
      index,
      text: displayText(seed.surface ?? seed.normalizedSurface ?? seed.id),
      kind: 'language-output',
      sourceCount: 0,
      evidenceCount: 0,
      membershipCount: 1,
      primaryFamily: family,
      diagnostics: Array(10).fill(0),
      affinity: { region: 'language', score: rounded(seed.priorWeight), verified: false },
      origin: 'manual-language-seed',
      status: 'frozen',
    })
    membershipTuples.push([index, family, rounded(seed.priorWeight), 1, 1, 1])
  }

  if (head) {
    const membershipsByMember = new Map()
    for (const evidence of array(head.membershipEvidence)) {
      const family = familyIndex.get(evidence?.familyId)
      const neuron = neuronIndex.get(evidence?.memberKey)
      if (!Number.isInteger(family) || !Number.isInteger(neuron)) continue
      if (finite(evidence.excitatoryEvidence) <= finite(evidence.inhibitoryEvidence)) continue
      const values = membershipsByMember.get(neuron) ?? []
      values.push({ evidence, family, score: logistic(evidence.score) * Math.max(0.05, finite(evidence.classifierConfidence, 1)) })
      membershipsByMember.set(neuron, values)
      const keys = learnedMemberKeysByFamily.get(family) ?? new Set()
      keys.add(evidence.memberKey)
      learnedMemberKeysByFamily.set(family, keys)
      const groups = sourceGroupsByFamily.get(family) ?? new Set()
      for (const sourceGroupId of array(evidence.sourceGroupIds)) groups.add(sourceGroupId)
      sourceGroupsByFamily.set(family, groups)
      supportByFamily.set(family, (supportByFamily.get(family) ?? 0) + count(evidence.updateCount))
    }
    for (const [neuron, values] of membershipsByMember) {
      values.sort((left, right) => right.score - left.score || left.family - right.family)
      const total = values.reduce((sum, value) => sum + value.score, 0) || 1
      values.forEach((value, index) => {
        membershipTuples.push([
          neuron,
          value.family,
          rounded(logistic(value.evidence.score)),
          rounded(value.score / total),
          index + 1,
          0,
        ])
      })
      neurons[neuron].membershipCount += values.length
    }

    const promotedSurfaces = new Set()
    for (const outputNeuron of array(head.outputNeurons)) {
      const family = familyIndex.get(outputNeuron?.familyId)
      if (!Number.isInteger(family)) continue
      const index = neurons.length
      promotedSurfaces.add(outputNeuron.normalizedSurface)
      neurons.push({
        id: bounded(outputNeuron.id, 512),
        index,
        text: displayText(outputNeuron.representativeSurface ?? outputNeuron.normalizedSurface),
        kind: 'language-output',
        sourceCount: array(outputNeuron.sourceGroupIds).length,
        evidenceCount: count(outputNeuron.observationCount),
        membershipCount: 1,
        primaryFamily: family,
        diagnostics: Array(10).fill(0),
        affinity: { region: 'language', score: rounded(outputNeuron.meanFamilyConfidence), verified: false },
        origin: 'autonomous-language',
        status: 'provisional',
      })
      membershipTuples.push([index, family, rounded(outputNeuron.meanFamilyConfidence), 1, 1, 1])
      const keys = learnedMemberKeysByFamily.get(family) ?? new Set()
      keys.add(outputNeuron.id)
      learnedMemberKeysByFamily.set(family, keys)
      const groups = sourceGroupsByFamily.get(family) ?? new Set()
      for (const sourceGroupId of array(outputNeuron.sourceGroupIds)) groups.add(sourceGroupId)
      sourceGroupsByFamily.set(family, groups)
      supportByFamily.set(family, (supportByFamily.get(family) ?? 0) + count(outputNeuron.observationCount))
    }

    for (const proposal of array(head.nursery)) {
      if (promotedSurfaces.has(proposal?.normalizedSurface)) continue
      const candidates = array(proposal?.familyCandidates)
        .map((candidate) => ({ family: familyIndex.get(candidate?.familyId), evidence: Math.max(0, finite(candidate?.evidence)) }))
        .filter((candidate) => Number.isInteger(candidate.family) && candidate.evidence > 0)
        .sort((left, right) => right.evidence - left.evidence || left.family - right.family)
      if (!candidates.length) continue
      const index = neurons.length
      const total = candidates.reduce((sum, candidate) => sum + candidate.evidence, 0) || 1
      const primaryFamily = candidates[0].family
      neurons.push({
        id: bounded(proposal.id, 512),
        index,
        text: displayText(proposal.representativeSurface ?? proposal.normalizedSurface),
        kind: 'language-nursery',
        sourceCount: array(proposal.sourceGroupIds).length,
        evidenceCount: count(proposal.observationCount),
        membershipCount: candidates.length,
        primaryFamily,
        diagnostics: Array(10).fill(0),
        affinity: { region: 'language', score: rounded(candidates[0].evidence / total), verified: false },
        origin: 'autonomous-language',
        status: bounded(proposal.status, 40) || 'observing',
      })
      candidates.forEach((candidate, rank) => {
        membershipTuples.push([index, candidate.family, rounded(candidate.evidence / total), rounded(candidate.evidence / total), rank + 1, 0])
        const keys = learnedMemberKeysByFamily.get(candidate.family) ?? new Set()
        keys.add(proposal.id)
        learnedMemberKeysByFamily.set(candidate.family, keys)
      })
    }
  }

  for (const family of families.filter((candidate) => candidate.layoutZone === 'language-surface')) {
    const learnedCount = learnedMemberKeysByFamily.get(family.index)?.size ?? 0
    family.memberCount += learnedCount
    family.supportCount = supportByFamily.get(family.index) ?? 0
    family.sourceCount = sourceGroupsByFamily.get(family.index)?.size ?? 0
  }

  const evidenceByEdge = new Map(array(head?.transitionEvidence)
    .filter((edge) => edge?.kind === 'family-transition')
    .map((edge) => [`${edge.fromKey}\0${edge.toKey}`, edge]))
  const transitionKeys = new Set()
  const rows = new Map()
  for (const transition of sourceScaffold.manualTransitions) {
    const from = familyIndex.get(transition.fromKey)
    const to = familyIndex.get(transition.toKey)
    if (!Number.isInteger(from) || !Number.isInteger(to)) continue
    transitionKeys.add(`${transition.fromKey}\0${transition.toKey}`)
    const evidence = evidenceByEdge.get(`${transition.fromKey}\0${transition.toKey}`)
    const values = rows.get(from) ?? []
    values.push({
      from,
      to,
      prior: Math.max(0, finite(transition.priorEvidence)),
      excitation: Math.max(0, finite(evidence?.excitatoryEvidence)),
      inhibition: Math.max(0, finite(evidence?.inhibitoryEvidence)),
      sourceCount: array(evidence?.sourceGroupIds).length,
      supportCount: count(evidence?.updateCount),
    })
    rows.set(from, values)
  }
  // The frozen scaffold is only the starting grammar. Once the plastic head
  // has independent evidence for a previously absent family route, expose it
  // in the peripheral language island without changing the core force graph.
  for (const evidence of evidenceByEdge.values()) {
    const key = `${evidence.fromKey}\0${evidence.toKey}`
    if (transitionKeys.has(key)) continue
    const from = familyIndex.get(evidence.fromKey)
    const to = familyIndex.get(evidence.toKey)
    if (!Number.isInteger(from) || !Number.isInteger(to)) continue
    const values = rows.get(from) ?? []
    values.push({
      from,
      to,
      prior: 0,
      excitation: Math.max(0, finite(evidence.excitatoryEvidence)),
      inhibition: Math.max(0, finite(evidence.inhibitoryEvidence)),
      sourceCount: array(evidence.sourceGroupIds).length,
      supportCount: count(evidence.updateCount),
    })
    rows.set(from, values)
  }
  for (const values of rows.values()) {
    const logits = values.map((value) => Math.log(value.prior + value.excitation + 1e-9) - Math.log1p(value.inhibition))
    const maximum = Math.max(...logits)
    const weights = logits.map((value) => Math.exp(value - maximum))
    const total = weights.reduce((sum, value) => sum + value, 0) || 1
    values.forEach((value, index) => familyTransitions.push([
      value.from,
      value.to,
      rounded(weights[index] / total),
      rounded(value.prior),
      rounded(value.excitation),
      rounded(value.inhibition),
      value.sourceCount,
      value.supportCount,
    ]))
  }

  return {
    scaffoldFingerprint: sourceScaffold.fingerprint,
    headFingerprint: head?.fingerprint ?? null,
    headRevision: count(head?.revision),
    persisted: Boolean(persisted.scaffold),
    compatible: persisted.compatible && Boolean(head),
    mismatch: persisted.mismatch,
    familyCount: sourceScaffold.families.length,
    seedNeuronCount: sourceScaffold.seedNeurons.length,
    nurseryCount: array(head?.nursery).length,
    provisionalNeuronCount: array(head?.outputNeurons).length,
    membershipEvidenceCount: array(head?.membershipEvidence).length,
  }
}

function conceptMacroArtifactMetadataForSnapshot(store, snapshot) {
  if (!snapshot) return null
  return array(snapshot.artifactIds)
    .map((id) => store.getArtifactMetadata?.(id))
    .filter((metadata) => metadata?.kind === CONCEPT_MACRO_HEAD_KIND)
    .sort((left, right) =>
      String(right.createdAt).localeCompare(String(left.createdAt)) ||
      String(right.id).localeCompare(String(left.id)),
    )[0] ?? null
}

function macroLifecycleRank(lifecycle) {
  return ({ active: 0, 'shadow-validated': 1, 'virtual-provisional': 2, 'eligible-proposal': 3, observing: 4, deprecated: 5 })[lifecycle] ?? 6
}

function macroDisplayName(macro) {
  const type = macro?.kind === 'equivalence-concept' ? 'Eşdeğerlik kavramı' : 'Sıra makrosu'
  return `${type} · ${count(macro?.orderedExpansion?.length)} adım`
}

function emptyConceptMacroProjection(metadata = null, mismatch = null) {
  return {
    artifact: artifactReference(metadata),
    headFingerprint: null,
    headRevision: 0,
    compatible: false,
    mismatch,
    snapshotStatus: null,
    runtimeEligible: false,
    totalMacroCount: 0,
    nodeCount: 0,
    linkCount: 0,
    activeCount: 0,
    truncated: false,
    byLifecycle: [],
  }
}

/**
 * Adds a bounded, metadata-only concept layer from the exact selected
 * snapshot. This deliberately does not consult a global "latest head": a
 * macro from another lineage must never appear to belong to this model.
 */
function appendConceptMacroOverlay({ store, selection, run, neurons }) {
  const metadata = conceptMacroArtifactMetadataForSnapshot(store, selection.snapshot)
  if (!metadata) return { projection: emptyConceptMacroProjection(), links: [], warnings: [] }

  let head
  try {
    head = restoreArtifact(store.getArtifact(metadata.id), CONCEPT_MACRO_HEAD_KIND)
  } catch (error) {
    return {
      projection: emptyConceptMacroProjection(metadata, 'head-decode-failed'),
      links: [],
      warnings: [`Kavram katmanı karantinaya alındı: ${bounded(String(error?.message ?? error), MAX_WARNING_LENGTH - 34)}`],
    }
  }

  if (head?.format !== CONCEPT_MACRO_HEAD_KIND || head?.version !== 1 || !Array.isArray(head.macros)) {
    return {
      projection: emptyConceptMacroProjection(metadata, 'invalid-head-format'),
      links: [],
      warnings: ['Kavram katmanı karantinaya alındı: head biçimi doğrulanamadı.'],
    }
  }
  if (head.representationSpaceId !== run.representationSpaceId) {
    return {
      projection: emptyConceptMacroProjection(metadata, 'representation-space-mismatch'),
      links: [],
      warnings: ['Kavram katmanı gösterilmedi: temsil uzayı seçili aile çalışmasıyla uyuşmuyor.'],
    }
  }
  if (head.familyRunFingerprint !== null && head.familyRunFingerprint !== run.fingerprint) {
    return {
      projection: emptyConceptMacroProjection(metadata, 'family-run-mismatch'),
      links: [],
      warnings: ['Kavram katmanı gösterilmedi: aile çalışması soyu seçili snapshot ile uyuşmuyor.'],
    }
  }

  const snapshotStatus = bounded(selection.snapshot?.status, 40) || 'unknown'
  const snapshotActive = store.getActiveSnapshot()?.id === selection.snapshot?.id
  const runtimeEligible = snapshotActive && snapshotStatus !== 'rejected'
  const originalNeuronCount = neurons.length
  const availableNodeSlots = Math.max(0, MAX_ATLAS_NEURONS - originalNeuronCount)
  const sortedMacros = head.macros
    .filter((macro) => macro && typeof macro.id === 'string' && Array.isArray(macro.orderedExpansion))
    .sort((left, right) =>
      macroLifecycleRank(left.lifecycle) - macroLifecycleRank(right.lifecycle) ||
      count(right.observationCount) - count(left.observationCount) ||
      String(left.id).localeCompare(String(right.id)),
    )
  const selectedMacros = sortedMacros.slice(0, Math.min(MAX_ATLAS_CONCEPT_MACROS, availableNodeSlots))
  const neuronIndex = new Map(neurons.map((neuron) => [neuron.id, neuron.index]))
  const retainedMacros = []
  for (const macro of selectedMacros) {
    if (neuronIndex.has(macro.id)) continue
    const index = neurons.length
    const runtimeActive = runtimeEligible && macro.lifecycle === 'active'
    const conceptMacro = {
      macroKind: macro.kind === 'equivalence-concept' ? 'equivalence-concept' : 'sequence-macro',
      lifecycle: bounded(macro.lifecycle, 40) || 'observing',
      runtimeStatus: snapshotStatus === 'rejected'
        ? 'rejected-snapshot'
        : runtimeActive ? 'active' : snapshotActive ? 'inactive-lifecycle' : 'research-snapshot',
      runtimeActive,
      expansionLength: count(macro.orderedExpansion.length),
      depth: count(macro.depth),
      fidelity: rounded(macro.metrics?.fidelity),
      compressionGain: rounded(macro.mdl?.compressionGain),
      compressionRatio: rounded(macro.mdl?.compressionRatio),
      sourceGroupCount: count(macro.independentGroupCount),
      observationCount: count(macro.observationCount),
    }
    neurons.push({
      id: bounded(macro.id, 512),
      index,
      text: macroDisplayName(macro),
      kind: 'concept-macro',
      origin: 'concept-macro-overlay',
      status: conceptMacro.lifecycle,
      layoutZone: 'concept-surface',
      sourceCount: count(macro.sourceCount),
      evidenceCount: conceptMacro.observationCount,
      membershipCount: 0,
      primaryFamily: -1,
      diagnostics: Array(10).fill(0),
      affinity: null,
      conceptMacro,
    })
    neuronIndex.set(macro.id, index)
    retainedMacros.push({ macro, index })
  }

  const links = []
  for (const { macro, index } of retainedMacros) {
    const expansion = macro.orderedExpansion.slice(0, MAX_ATLAS_MACRO_EXPANSION_LINKS_PER_MACRO)
    for (let ordinal = 0; ordinal < expansion.length && links.length < MAX_ATLAS_MACRO_EXPANSION_LINKS; ordinal += 1) {
      const member = expansion[ordinal]
      const target = neuronIndex.get(member?.id)
      if (!Number.isInteger(target) || target === index) continue
      links.push([index, target, ordinal, member?.kind === 'composite-neuron' ? 1 : 0])
    }
    if (links.length >= MAX_ATLAS_MACRO_EXPANSION_LINKS) break
  }

  const lifecycleCounts = new Map()
  for (const { macro } of retainedMacros) {
    const lifecycle = bounded(macro.lifecycle, 40) || 'observing'
    lifecycleCounts.set(lifecycle, (lifecycleCounts.get(lifecycle) ?? 0) + 1)
  }
  const truncated = retainedMacros.length < sortedMacros.length
  const warnings = []
  if (truncated) warnings.push(`Kavram yüzeyi güvenli atlas sınırı nedeniyle ${sortedMacros.length - retainedMacros.length} düğümü göstermedi.`)
  if (snapshotStatus === 'rejected') warnings.push('Kavram head’i reddedilmiş snapshot’a aittir; hiçbir düğüm etkin çalışma makrosu olarak işaretlenmedi.')

  return {
    projection: {
      artifact: artifactReference(metadata),
      headFingerprint: bounded(head.fingerprint, 256) || metadata.contentHash || null,
      headRevision: count(head.revision),
      compatible: true,
      mismatch: null,
      snapshotStatus,
      runtimeEligible,
      totalMacroCount: sortedMacros.length,
      nodeCount: retainedMacros.length,
      linkCount: links.length,
      activeCount: retainedMacros.filter(({ macro }) => runtimeEligible && macro.lifecycle === 'active').length,
      truncated,
      byLifecycle: [...lifecycleCounts].sort(([left], [right]) => left.localeCompare(right)).map(([lifecycle, value]) => ({ lifecycle, count: value })),
    },
    links,
    warnings,
  }
}

function buildAtlas(store, selection, run, sizeState) {
  const familyIndex = new Map(run.families.map((family, index) => [family.id, index]))
  const neuronIndex = new Map(run.profiles.map((profile, index) => [profile.memberKey, index]))
  const utilityStatus = new Map(array(run.utilityReports).map((report) => [report?.familyId, report?.status]))
  const membershipsByNeuron = new Map()
  const membershipTuples = []

  for (const membership of run.memberships) {
    const neuron = neuronIndex.get(membership?.memberKey)
    const family = familyIndex.get(membership?.familyId)
    if (!Number.isInteger(neuron) || !Number.isInteger(family)) continue
    const tuple = [
      neuron,
      family,
      rounded(membership.score),
      rounded(membership.weight),
      count(membership.rank),
      membership.core ? 1 : 0,
    ]
    membershipTuples.push(tuple)
    const values = membershipsByNeuron.get(neuron) ?? []
    values.push(tuple)
    membershipsByNeuron.set(neuron, values)
  }

  const primaryFamilyByNeuron = new Int32Array(run.profiles.length).fill(-1)
  for (const [neuron, memberships] of membershipsByNeuron) {
    memberships.sort((left, right) => right[3] - left[3] || right[2] - left[2] || left[4] - right[4] || left[1] - right[1])
    primaryFamilyByNeuron[neuron] = memberships[0]?.[1] ?? -1
  }

  const families = run.families.map((family, index) => ({
    id: bounded(family.id, 256),
    index,
    region: bounded(family.region, 80) || 'unknown',
    regionVerified: Boolean(family.regionVerified),
    lifecycle: bounded(family.lifecycle, 80) || 'unknown',
    supportCount: count(family.supportCount),
    sourceCount: count(family.sourceCount),
    memberCount: array(family.memberKeys).length,
    coreMemberCount: array(family.coreMemberKeys).length,
    cohesion: rounded(family.cohesion),
    separation: rounded(family.separation),
    stability: rounded(family.stability),
    utility: family.utility === null || family.utility === undefined ? null : rounded(family.utility),
    utilityStatus: bounded(utilityStatus.get(family.id), 80) || 'missing',
    predecessors: array(family.predecessorFamilyIds).map((id) => familyIndex.get(id)).filter(Number.isInteger).slice(0, 32),
  }))

  const neurons = run.profiles.map((profile, index) => ({
    id: bounded(profile.memberKey, 512),
    index,
    text: displayText(profile.text ?? profile.evidence?.[0]?.text),
    kind: profile.kind === 'synthesis' ? 'synthesis' : 'raw',
    sourceCount: sourceCount(profile),
    evidenceCount: array(profile.evidence).length,
    membershipCount: membershipsByNeuron.get(index)?.length ?? 0,
    primaryFamily: primaryFamilyByNeuron[index],
    diagnostics: diagnostics(profile),
    affinity: strongestAffinity(profile),
  }))

  const familyTransitions = []
  for (const transition of run.familyTransitions) {
    const from = familyIndex.get(transition?.fromFamilyId)
    const to = familyIndex.get(transition?.toFamilyId)
    if (!Number.isInteger(from) || !Number.isInteger(to)) continue
    const prior = rounded(transition.priorEvidence)
    const experience = rounded(transition.experienceEvidence)
    const inhibition = rounded(transition.inhibitionEvidence)
    familyTransitions.push([
      from, to, rounded(transition.probability), prior, experience, inhibition,
      count(transition.sourceCount), count(transition.supportCount),
    ])
  }

  const neuronTransitions = []
  for (const transition of run.neuronTransitions) {
    const from = neuronIndex.get(transition?.fromMemberKey)
    const to = neuronIndex.get(transition?.toMemberKey)
    if (!Number.isInteger(from) || !Number.isInteger(to)) continue
    neuronTransitions.push([from, to, rounded(transition.probability)])
  }

  const languageOutput = appendLanguageOutputOverlay({
    store, selection, run, families, neurons, membershipTuples, familyTransitions,
  })
  const conceptMacros = appendConceptMacroOverlay({ store, selection, run, neurons })
  const plasticRegions = inspectPlasticRegionsForAtlas(store, selection.snapshot, run)
  const atlasFingerprint = stableRuntimeHash(
    'bombom-neuron-atlas-v2',
    run.fingerprint,
    selection.snapshot?.id ?? null,
    selection.snapshot?.status ?? null,
    store.getActiveSnapshot()?.id ?? null,
    languageOutput.headFingerprint,
    conceptMacros.projection.headFingerprint,
    conceptMacros.projection.artifact?.contentHash ?? null,
    plasticRegions.projection?.headFingerprint ?? null,
    plasticRegions.projection?.artifact?.contentHash ?? null,
    plasticRegions.projection?.configuredMode ?? null,
    plasticRegions.projection?.effectiveMode ?? null,
    plasticRegions.projection?.effectiveModeReason ?? null,
    plasticRegions.projection?.release?.fingerprint ?? null,
    plasticRegions.projection?.release?.reason ?? null,
    plasticRegions.projection?.behaviorApplied ?? false,
  )

  return {
    format: 'bombom-neuron-atlas',
    version: 1,
    fingerprint: atlasFingerprint,
    available: true,
    selection: {
      selectedBy: selection.selectedBy,
      snapshot: snapshotReference(store, selection.snapshot),
      artifact: artifactReference(selection.artifact),
    },
    run: {
      id: bounded(run.id, 256),
      fingerprint: bounded(run.fingerprint, 256),
      createdAt: bounded(run.createdAt, 64),
      representationSpaceId: bounded(run.representationSpaceId, 256),
      behaviorSpaceId: bounded(run.behaviorSpaceId, 256),
      corpusFingerprint: bounded(run.inputManifest?.corpusFingerprint, 256),
    },
    stats: {
      ...sizeState.sizes,
      learnedFamilyCount: sizeState.sizes.familyCount,
      familyCount: families.length,
      neuronCount: neurons.length,
      membershipCount: membershipTuples.length,
      familyTransitionCount: familyTransitions.length,
      languageOutputFamilyCount: languageOutput.familyCount,
      languageOutputSeedNeuronCount: languageOutput.seedNeuronCount,
      languageOutputNurseryCount: languageOutput.nurseryCount,
      languageOutputProvisionalNeuronCount: languageOutput.provisionalNeuronCount,
      ...(conceptMacros.projection.artifact ? {
        conceptMacroCount: conceptMacros.projection.nodeCount,
        conceptMacroTotalCount: conceptMacros.projection.totalMacroCount,
        conceptMacroActiveCount: conceptMacros.projection.activeCount,
        conceptMacroExpansionLinkCount: conceptMacros.projection.linkCount,
      } : {}),
      ...(plasticRegions.projection?.compatible ? {
        plasticRegionCount: plasticRegions.projection.regionCount,
        plasticRegionActiveCount: plasticRegions.projection.activeRegionCount,
        plasticRegionPageCount: plasticRegions.projection.pageCount,
      } : {}),
      distinctSourceCount: count(run.stats?.distinctSourceCount),
      validatedUtilityFamilyCount: families.filter((family) => family.utilityStatus === 'validated').length,
      mergeProposedFamilyCount: families.filter((family) => family.lifecycle === 'merge-proposed').length,
      priorEdgeCount: familyTransitions.filter((edge) => finite(edge[3]) > 0).length,
      experienceEdgeCount: familyTransitions.filter((edge) => finite(edge[4]) > 0).length,
      inhibitionEdgeCount: familyTransitions.filter((edge) => finite(edge[5]) > 0).length,
    },
    limits: {
      ...sizeState.limits,
      conceptMacroCount: MAX_ATLAS_CONCEPT_MACROS,
      macroExpansionLinkCount: MAX_ATLAS_MACRO_EXPANSION_LINKS,
    },
    projection: {
      semanticAxes: false,
      layout: 'deterministic-core-force-plus-peripheral-language-and-concept-surfaces-v3',
      note: '3B eksenler semantik etiket değildir; konumlar aile geçişi, yumuşak aidiyet ve davranış tanılarının kayıpsız olmayan görsel izdüşümüdür.',
      languageOutput,
      conceptMacros: conceptMacros.projection,
      ...(plasticRegions.projection ? { plasticRegions: plasticRegions.projection } : {}),
    },
    tupleSchema: {
      membership: ['neuron', 'family', 'score', 'weight', 'rank', 'core'],
      familyTransition: ['from', 'to', 'probability', 'prior', 'experience', 'inhibition', 'sourceCount', 'supportCount'],
      neuronTransition: ['from', 'to', 'probability'],
      macroExpansion: ['macroNeuron', 'memberNeuron', 'ordinal', 'nestedComposite'],
      diagnostics: ['spanScale', 'relativePosition', 'depthScale', 'relationStrength', 'transitionEntropy', 'transitionConcentration', 'localContinuity', 'bridgeImpact', 'leftRightAsymmetry', 'shortScopePotential'],
    },
    families,
    neurons,
    links: { memberships: membershipTuples, familyTransitions, neuronTransitions, macroExpansions: conceptMacros.links },
    warnings: [...array(run.warnings), ...conceptMacros.warnings, ...plasticRegions.warnings]
      .slice(0, MAX_WARNING_COUNT)
      .map((warning) => bounded(warning, MAX_WARNING_LENGTH)),
  }
}

function emptyAtlas(store, selection) {
  return {
    format: 'bombom-neuron-atlas', version: 1, available: false,
    reason: selection.snapshot ? 'Bu snapshot içinde aile çalışması bulunmuyor.' : 'Henüz 3B atlas oluşturacak aile çalışması yok.',
    selection: { selectedBy: selection.selectedBy, snapshot: snapshotReference(store, selection.snapshot), artifact: null },
    run: null,
    stats: { familyCount: 0, neuronCount: 0, membershipCount: 0, familyTransitionCount: 0, neuronTransitionCount: 0 },
    families: [], neurons: [], links: { memberships: [], familyTransitions: [], neuronTransitions: [] },
  }
}

function memoryHeavyTrainingJob(store) {
  return store.listJobs().find((job) =>
    MEMORY_HEAVY_JOB_TYPES.has(job?.type) &&
    (
      // A queued heavy job can be claimed immediately after this check. Treat
      // it as a RAM owner already so a startup-time atlas request cannot race
      // the worker and hydrate a multi-hundred-megabyte FamilyRun in-process.
      ['queued', 'running', 'pause_requested', 'cancel_requested'].includes(job?.state) ||
      (
        job?.state === 'paused' &&
        Boolean(job?.payload?.feedState?.pendingTrainingMilestone)
      )
    ),
  ) ?? null
}

function busyAtlas(store, selection, job) {
  return createBusyLanguageAtlas({
    format: 'bombom-neuron-atlas', version: 1, available: false,
    reason: 'Aile eğitimi sürerken bellek eğitim motoruna ayrıldı. 3B evren işlem tamamlanınca otomatik yenilenecek.',
    busy: true,
    busyStage: bounded(job?.stage, 160) || 'training',
    selection: {
      selectedBy: selection.selectedBy,
      snapshot: snapshotReference(store, selection.snapshot),
      artifact: artifactReference(selection.artifact),
    },
    run: null,
    stats: { familyCount: 0, neuronCount: 0, membershipCount: 0, familyTransitionCount: 0, neuronTransitionCount: 0 },
    families: [], neurons: [], links: { memberships: [], familyTransitions: [], neuronTransitions: [] },
  })
}

export function createNeuronAtlas(store, { snapshotId } = {}) {
  const metadataSelection = selectFamilyProjectionMetadata(store, snapshotId)
  const cache = atlasCacheByStore.get(store) ?? new Map()
  atlasCacheByStore.set(store, cache)
  const activeTraining = memoryHeavyTrainingJob(store)
  if (activeTraining) {
    // Drop the potentially multi-GiB JS projection before the disposable
    // pipeline worker reaches family formation. The endpoint remains useful as
    // a lightweight status response and never opens the FamilyRun payload.
    cache.clear()
    return busyAtlas(store, metadataSelection, activeTraining)
  }
  const outputArtifactLineage = (metadataSelection.snapshot?.artifactIds ?? [])
    .map((id) => store.getArtifactMetadata?.(id))
    .filter((metadata) => metadata && ['language-output-scaffold', 'language-output-head'].includes(metadata.kind))
    .map((metadata) => `${metadata.kind}:${metadata.id}:${metadata.contentHash ?? ''}`)
    .sort()
    .join('|')
  const conceptMacroArtifact = conceptMacroArtifactMetadataForSnapshot(store, metadataSelection.snapshot)
  const conceptMacroLineage = conceptMacroArtifact
    ? `${conceptMacroArtifact.kind}:${conceptMacroArtifact.id}:${conceptMacroArtifact.contentHash ?? ''}`
    : ''
  const plasticRegionKinds = new Set(PLASTIC_REGION_ARTIFACT_KINDS)
  const plasticRegionLineage = (metadataSelection.snapshot?.artifactIds ?? [])
    .map((id) => store.getArtifactMetadata?.(id))
    .filter((metadata) => metadata && plasticRegionKinds.has(metadata.kind))
    .map((metadata) => `${metadata.kind}:${metadata.id}:${metadata.contentHash ?? ''}`)
    .sort()
    .join('|')
  const plasticRegionConfig = store.getConfig?.().plasticRegions ?? {}
  const plasticRegionPolicyLineage = [
    plasticRegionConfig.enabled === false ? 'disabled' : 'enabled',
    plasticRegionConfig.mode ?? 'off',
    plasticRegionConfig.representationProfile ?? 'legacy',
  ].join(':')
  const cacheKey = [
    metadataSelection.artifact?.id ?? 'empty',
    metadataSelection.artifact?.contentHash ?? '',
    metadataSelection.snapshot?.id ?? '',
    metadataSelection.snapshot?.status ?? '',
    store.getActiveSnapshot()?.id ?? '',
    LANGUAGE_OUTPUT_SEED_FINGERPRINT,
    outputArtifactLineage,
    conceptMacroLineage,
    plasticRegionLineage,
    plasticRegionPolicyLineage,
  ].join('\0')
  if (cache.has(cacheKey)) return cache.get(cacheKey)
  const { selection, run } = loadFamilyProjectionSource(store, snapshotId, metadataSelection)
  if (!run) return emptyAtlas(store, selection)
  const result = buildAtlas(store, selection, run, validateAtlasSize(run))
  cache.set(cacheKey, result)
  while (cache.size > 2) cache.delete(cache.keys().next().value)
  return result
}
