import { decodeArtifactValue } from '../runtime/artifacts.mjs'

const HEAD_KIND = 'concept-macro-head'
const OBSERVATION_KIND = 'concept-macro-observation-batch'
const SHADOW_KIND = 'concept-macro-shadow-report'
const MAX_LIMIT = 2_000
const headCacheByStore = new WeakMap()

function boundedInteger(value, fallback, minimum, maximum) {
  const parsed = Number(value)
  return Number.isInteger(parsed) ? Math.max(minimum, Math.min(maximum, parsed)) : fallback
}

function artifactValue(artifact) {
  if (!artifact || artifact.payload?.format !== 'bombom-runtime-artifact') return null
  return decodeArtifactValue(artifact.payload.value)
}

function trustedSnapshot(store) {
  const active = store.getActiveSnapshot?.()
  if (active) return active
  return [...(store.listSnapshots?.() ?? [])]
    .filter((snapshot) =>
      snapshot.status === 'candidate' &&
      snapshot.evaluation?.passed === true &&
      snapshot.evaluation?.runtimeVerified === true)
    .sort((left, right) =>
      String(right.createdAt).localeCompare(String(left.createdAt)) ||
      String(right.id).localeCompare(String(left.id)))[0] ?? null
}

function snapshotMetadata(store, snapshot, kind) {
  if (!snapshot) return null
  for (const artifactId of [...(snapshot.artifactIds ?? [])].reverse()) {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (metadata?.kind === kind) return metadata
  }
  return null
}

function latestHead(store, snapshot = trustedSnapshot(store)) {
  const metadata = snapshotMetadata(store, snapshot, HEAD_KIND)
  if (!metadata) return { artifact: null, head: null }
  const cached = headCacheByStore.get(store)
  if (cached?.artifactId === metadata.id && cached?.contentHash === metadata.contentHash) {
    return { artifact: cached.artifact, head: cached.head }
  }
  const storedArtifact = store.getArtifact(metadata.id)
  const head = artifactValue(storedArtifact)
  // Do not pin both the encoded artifact payload and its decoded 4096D head
  // in RAM between UI polls. The cache needs only the thin artifact envelope.
  const artifact = {
    ...metadata,
    payload: { metadata: storedArtifact?.payload?.metadata ?? {} },
  }
  headCacheByStore.set(store, {
    artifactId: metadata.id,
    contentHash: metadata.contentHash,
    artifact,
    head,
  })
  return { artifact, head }
}

function array(value) {
  return Array.isArray(value) ? value : []
}

function macroList(head) {
  return array(head?.macros ?? head?.candidates ?? head?.entries)
}

function lifecycle(value) {
  return String(value?.lifecycle ?? value?.status ?? 'observing')
}

function publicExpansion(value) {
  const expansion = array(value?.orderedExpansion ?? value?.expansion ?? value?.constituents)
  return expansion.slice(0, 64).map((entry, index) => {
    if (typeof entry === 'string') return { position: index, memberKey: entry, familyId: null, macroId: null }
    const familyIds = array(entry?.familyIds).filter((item) => typeof item === 'string').slice(0, 16)
    return {
      position: Number.isInteger(entry?.position) ? entry.position : index,
      memberKey: typeof entry?.memberKey === 'string'
        ? entry.memberKey
        : entry?.kind === 'raw-neuron' && typeof entry?.id === 'string' ? entry.id : null,
      familyId: typeof entry?.familyId === 'string' ? entry.familyId : familyIds[0] ?? null,
      familyIds,
      macroId: typeof entry?.macroId === 'string'
        ? entry.macroId
        : entry?.kind === 'composite-neuron' && typeof entry?.id === 'string' ? entry.id : null,
      identityDigest: typeof entry?.identityDigest === 'string' ? entry.identityDigest : null,
      label: typeof entry?.label === 'string' ? entry.label.slice(0, 160) : null,
    }
  })
}

function metric(value, ...keys) {
  for (const key of keys) {
    const candidate = value?.[key]
    if (typeof candidate === 'number' && Number.isFinite(candidate)) return candidate
  }
  return null
}

function publicSignalSignature(value) {
  if (!value || typeof value !== 'object') return null
  return {
    dimension: Number.isInteger(value.dimension)
      ? value.dimension
      : Array.isArray(value.vector) ? value.vector.length : null,
    digest: typeof value.digest === 'string' ? value.digest : null,
  }
}

function publicMacro(value, detail = false) {
  const expansion = publicExpansion(value)
  const evidence = array(value?.evidence)
  const sourceGroupIds = [...new Set([
    ...array(value?.sourceGroupIds ?? value?.independentSourceGroupIds),
    ...evidence.map((item) => item?.independenceGroupId),
  ])]
    .filter((item) => typeof item === 'string').slice(0, detail ? 256 : 16)
  const result = {
    id: String(value?.id ?? value?.fingerprint ?? ''),
    fingerprint: String(value?.fingerprint ?? value?.id ?? ''),
    kind: String(value?.kind ?? value?.birthKind ?? 'sequence-macro'),
    lifecycle: lifecycle(value),
    label: String(value?.label ?? value?.representativeSurface ?? value?.id ?? 'Adsız makro').slice(0, 200),
    expansionLength: expansion.length,
    orderedExpansion: expansion,
    occurrenceCount: metric(value, 'occurrenceCount', 'observationCount', 'supportCount') ?? 0,
    sourceGroupCount: metric(value, 'sourceGroupCount', 'independentGroupCount') ?? sourceGroupIds.length,
    sourceGroupIds,
    completionProbability: metric(value, 'completionProbability', 'completionConfidence') ?? metric(value?.metrics, 'activationRecall'),
    pathCohesion: metric(value, 'pathCohesion', 'trajectoryCohesion', 'cohesion'),
    effectCohesion: metric(value, 'effectCohesion'),
    compressionGain: metric(value, 'compressionGain', 'mdlGain') ?? metric(value?.mdl, 'compressionGain'),
    predictiveUtility: metric(value, 'predictiveUtility', 'holdoutUtility') ?? metric(value?.metrics, 'holdoutUtility'),
    shadowFidelity: metric(value, 'shadowFidelity', 'fidelity') ?? metric(value?.metrics, 'fidelity'),
    falseActivationRate: metric(value, 'falseActivationRate') ?? metric(value?.metrics, 'falseActivationRate'),
    logicalDuration: metric(value, 'logicalDuration', 'length') ?? expansion.length,
    nestedDepth: metric(value, 'nestedDepth', 'depth') ?? 1,
    structuralEquivalencePassed: value?.structuralEquivalencePassed === true,
    structuralEquivalenceAuditId: typeof value?.structuralEquivalenceAudit?.id === 'string'
      ? value.structuralEquivalenceAudit.id
      : null,
    expansionDigest: typeof value?.expansionDigest === 'string' ? value.expansionDigest : null,
    representationSpaceId: typeof value?.representationSpaceId === 'string' ? value.representationSpaceId : null,
    familyRunFingerprint: typeof value?.familyRunFingerprint === 'string' ? value.familyRunFingerprint : null,
    fallbackSafe: value?.fallbackSafe !== false,
    physicalIdentityAllocated: value?.physicalIdentityAllocated === true,
    updatedAt: value?.updatedAt ?? null,
  }
  if (detail) {
    result.entrySignature = publicSignalSignature(value?.entrySignature ?? value?.entryBoundarySignature ?? value?.entryPrototype)
    result.exitSignature = publicSignalSignature(value?.exitSignature ?? value?.exitDistribution ?? value?.exitPrototype)
    result.predecessorMacroIds = array(value?.predecessorMacroIds).filter((item) => typeof item === 'string').slice(0, 128)
    result.shadow = value?.shadow ?? value?.shadowEvidence ?? value?.metrics ?? null
    result.evidence = evidence.slice(0, 256).map((item) => ({
      observationId: item?.observationId ?? null,
      sourceId: item?.sourceId ?? null,
      sourceDigest: item?.sourceDigest ?? null,
      independenceGroupId: item?.independenceGroupId ?? null,
      split: item?.split ?? null,
      tickStart: item?.tickStart ?? null,
      tickEnd: item?.tickEnd ?? null,
    }))
    result.explanation = typeof value?.explanation === 'string' ? value.explanation.slice(0, 2_000) : null
  }
  return result
}

export function parseConceptInspectionQuery(searchParams) {
  const lifecycleValue = searchParams.get('lifecycle')?.trim() || null
  const kindValue = searchParams.get('kind')?.trim() || null
  const query = searchParams.get('q')?.trim().toLocaleLowerCase('tr-TR').slice(0, 160) || null
  return {
    limit: boundedInteger(searchParams.get('limit'), 500, 1, MAX_LIMIT),
    offset: boundedInteger(searchParams.get('offset'), 0, 0, 1_000_000),
    lifecycle: lifecycleValue,
    kind: kindValue,
    query,
  }
}

export function inspectConceptMacros(store, options = {}) {
  const snapshot = trustedSnapshot(store)
  const { artifact: headArtifact, head } = latestHead(store, snapshot)
  const observationArtifact = snapshotMetadata(store, snapshot, OBSERVATION_KIND)
  const shadowArtifact = snapshotMetadata(store, snapshot, SHADOW_KIND)
  const allMacros = macroList(head).map((macro) => publicMacro(macro))
  const filtered = allMacros.filter((macro) => {
    if (options.lifecycle && macro.lifecycle !== options.lifecycle) return false
    if (options.kind && macro.kind !== options.kind) return false
    if (options.query) {
      const haystack = `${macro.id} ${macro.label} ${macro.orderedExpansion.map((entry) => `${entry.memberKey ?? ''} ${entry.familyId ?? ''}`).join(' ')}`.toLocaleLowerCase('tr-TR')
      if (!haystack.includes(options.query)) return false
    }
    return true
  })
  const offset = options.offset ?? 0
  const limit = options.limit ?? 500
  const counts = {}
  for (const macro of allMacros) counts[macro.lifecycle] = (counts[macro.lifecycle] ?? 0) + 1
  return {
    available: Boolean(head),
    mode: head?.mode ?? head?.safetyMode ?? headArtifact?.payload?.metadata?.mode ?? 'proposal-shadow',
    head: head ? {
      id: head.id ?? headArtifact?.id,
      fingerprint: head.fingerprint ?? headArtifact?.contentHash ?? null,
      revision: head.revision ?? 0,
      representationSpaceId: head.representationSpaceId ?? null,
      familyRunFingerprint: head.familyRunFingerprint ?? null,
      createdAt: head.createdAt ?? headArtifact?.createdAt ?? null,
      updatedAt: head.updatedAt ?? headArtifact?.createdAt ?? null,
      snapshotId: snapshot?.id ?? null,
      snapshotStatus: snapshot?.status ?? null,
    } : null,
    stats: {
      totalMacros: allMacros.length,
      lifecycleCounts: counts,
      observationCount: Number(head?.stats?.acceptedObservationCount ?? observationArtifact?.metadata?.acceptedObservationCount ?? 0),
      sourceGroupCount: new Set(allMacros.flatMap((macro) => macro.sourceGroupIds)).size,
      shadowEvaluationCount: allMacros.reduce((sum, macro) => {
        const raw = macroList(head).find((item) => String(item?.id ?? item?.fingerprint) === macro.id)
        return sum + array(raw?.shadowReportIds).length
      }, 0) || Number(Boolean(shadowArtifact)),
      activeCount: allMacros.filter((macro) => macro.lifecycle === 'active').length,
      structuralValidatedCount: allMacros.filter((macro) => macro.structuralEquivalencePassed).length,
      virtualCount: allMacros.filter((macro) => ['virtual-provisional', 'shadow-validated'].includes(macro.lifecycle)).length,
      fallbackProtectedCount: allMacros.filter((macro) => macro.fallbackSafe).length,
    },
    artifacts: {
      headArtifactId: headArtifact?.id ?? null,
      observationArtifactId: observationArtifact?.id ?? null,
      shadowArtifactId: shadowArtifact?.id ?? null,
    },
    total: filtered.length,
    offset,
    limit,
    macros: filtered.slice(offset, offset + limit),
  }
}

export function inspectConceptMacro(store, id) {
  const envelope = inspectConceptMacros(store, { limit: MAX_LIMIT })
  const { head } = latestHead(store, trustedSnapshot(store))
  const raw = macroList(head).find((macro) => String(macro?.id ?? macro?.fingerprint) === id)
  if (!raw) return null
  return { ...envelope, macros: undefined, total: undefined, macro: publicMacro(raw, true) }
}

export const CONCEPT_ARTIFACT_KINDS = Object.freeze({
  head: HEAD_KIND,
  observations: OBSERVATION_KIND,
  shadow: SHADOW_KIND,
})
