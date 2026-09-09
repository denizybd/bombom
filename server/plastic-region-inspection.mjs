import { restoreArtifact } from '../runtime/artifacts.mjs'
import {
  PLASTIC_REGION_ARTIFACT_KINDS,
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
  PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
  assertPlasticRegionArtifactMetadata,
  assertPlasticRegionArtifactValue,
  currentPlasticRegionBase,
  listPlasticRegionArtifacts,
  readPlasticRegionRelease,
  readPlasticRegionSnapshot,
} from '../runtime/plastic-region-cache.mjs'

const MAX_LIST_LIMIT = 200
const MAX_LIST_OFFSET = 4_095
const MAX_REGION_COUNT = 56
const MAX_PAGE_COUNT = 56
const MAX_BEHAVIOR_DIMENSION = 256
const ATLAS_KINDS = new Set([
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
])
const READINESS_MAXIMUM_HEADS = 32
const READINESS_MAXIMUM_BYTES = 16 * 1024 * 1024

function inspectionError(code, message, status = 400, details) {
  const error = new Error(message)
  error.code = code
  error.status = status
  if (details !== undefined) error.details = details
  return error
}

function boundedInteger(value, fallback, minimum, maximum, name) {
  if (value === null || value === undefined || value === '') return fallback
  if (!/^-?\d+$/u.test(String(value))) {
    throw inspectionError('INVALID_PLASTIC_REGION_QUERY', `${name} bir tam sayı olmalı.`)
  }
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw inspectionError('INVALID_PLASTIC_REGION_QUERY', `${name} ${minimum}–${maximum} aralığında olmalı.`)
  }
  return parsed
}

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function probability(value) {
  return Math.max(0, Math.min(1, finite(value)))
}

function strings(value, maximum = 64) {
  return Array.isArray(value)
    ? [...new Set(value.filter((item) => typeof item === 'string' && item).map((item) => item.slice(0, 256)))].slice(0, maximum)
    : []
}

function publicRegion(region) {
  const prototype = Array.isArray(region?.prototype?.values)
    ? region.prototype.values.slice(0, MAX_BEHAVIOR_DIMENSION).map(finite)
    : []
  return {
    id: String(region?.id ?? '').slice(0, 256),
    lifecycle: String(region?.lifecycle ?? 'provisional').slice(0, 40),
    origin: String(region?.origin ?? 'discovery').slice(0, 40),
    bornRevision: Number.isSafeInteger(region?.bornRevision) ? region.bornRevision : 0,
    retiredRevision: Number.isSafeInteger(region?.retiredRevision) ? region.retiredRevision : null,
    parentRegionIds: strings(region?.parentRegionIds, 56),
    prototype: {
      dimension: Number.isSafeInteger(region?.prototype?.dimension) ? region.prototype.dimension : prototype.length,
      values: prototype,
      truncated: Number(region?.prototype?.dimension ?? prototype.length) > prototype.length,
    },
    evidence: {
      observationCount: Number.isSafeInteger(region?.evidence?.observationCount) ? region.evidence.observationCount : 0,
      sourceGroupCount: strings(region?.evidence?.sourceGroupIds, 1_000).length,
      confidenceMass: finite(region?.evidence?.confidenceMass),
      cohesion: probability(region?.evidence?.cohesion),
      holdoutUtility: finite(region?.evidence?.holdoutUtility),
      shamUtility: finite(region?.evidence?.shamUtility),
    },
  }
}

function publicPage(page) {
  const memberships = Array.isArray(page?.memberships)
    ? page.memberships.slice(0, 3).map((membership) => ({
        regionId: String(membership?.regionId ?? '').slice(0, 256),
        weight: probability(membership?.weight),
      }))
    : []
  return {
    pageIndex: Number.isSafeInteger(page?.pageIndex) ? page.pageIndex : null,
    memberships,
    freeMass: probability(page?.freeMass),
  }
}

export function publicPlasticRegionAtlas(head) {
  if (!head) return null
  const regions = (head.regions ?? []).slice(0, MAX_REGION_COUNT).map(publicRegion)
  const pages = (head.pages ?? []).slice(0, MAX_PAGE_COUNT).map(publicPage)
  const lifecycleCounts = {}
  for (const region of regions) lifecycleCounts[region.lifecycle] = (lifecycleCounts[region.lifecycle] ?? 0) + 1
  return {
    format: head.format,
    version: head.version,
    representationSpaceId: head.representationSpaceId,
    geometryFingerprint: head.geometryFingerprint ?? null,
    revision: head.revision,
    parentFingerprint: head.parentFingerprint ?? null,
    behaviorDimension: head.behaviorDimension,
    maximumMembershipsPerPage: head.maximumMembershipsPerPage,
    fingerprint: head.fingerprint,
    regions,
    pages,
    stats: {
      regionCount: head.regions?.length ?? 0,
      pageCount: head.pages?.length ?? 0,
      lifecycleCounts,
      assignedMass: pages.reduce((sum, page) => sum + page.memberships.reduce((part, item) => part + item.weight, 0), 0),
      freeMass: pages.reduce((sum, page) => sum + page.freeMass, 0),
    },
    truncated: (head.regions?.length ?? 0) > regions.length || (head.pages?.length ?? 0) > pages.length,
  }
}

function metadataRow(metadata) {
  return {
    artifactId: metadata.id,
    kind: metadata.kind,
    contentHash: metadata.contentHash ?? null,
    createdAt: metadata.createdAt ?? null,
    jobId: metadata.jobId ?? null,
    payloadBytes: Number(metadata.payloadBytes ?? 0),
  }
}

function inspectCurrentEvidenceReadiness(store, config) {
  const base = currentPlasticRegionBase(store)
  if (!base.snapshotId) {
    return {
      shadowEvidenceAvailable: false,
      reason: 'active-base-required',
      evidenceArtifactId: null,
      evidenceFingerprint: null,
    }
  }
  const rows = store.listArtifactMetadataByKind(PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND, {
    limit: READINESS_MAXIMUM_HEADS,
  })
  let bytesRead = 0
  for (const metadata of rows) {
    const bytes = Number(metadata.payloadBytes ?? 0)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > config.maximumArtifactBytes ||
        bytesRead + bytes > READINESS_MAXIMUM_BYTES) continue
    bytesRead += bytes
    try {
      const artifact = store.getArtifact(metadata.id)
      const envelope = artifact?.payload?.metadata
      if (envelope?.baseSnapshotId !== base.snapshotId ||
          (envelope?.casBaseAtlasFingerprint ?? null) !== (base.atlasFingerprint ?? null)) continue
      const value = restoreArtifact(artifact, PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND)
      assertPlasticRegionArtifactValue(PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND, value)
      assertPlasticRegionArtifactMetadata(PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND, value, artifact)
      if (value.authority !== 'cas-active-base' || value.observationCount < 1) continue
      const producer = metadata.jobId ? store.getJob(metadata.jobId) : null
      if (metadata.jobId && producer?.state !== 'succeeded') continue
      return {
        shadowEvidenceAvailable: true,
        reason: null,
        evidenceArtifactId: metadata.id,
        evidenceFingerprint: value.fingerprint,
      }
    } catch {
      // Corrupt, stale or differently scoped evidence never opens the UX gate.
    }
  }
  return {
    shadowEvidenceAvailable: false,
    reason: rows.length ? 'current-base-evidence-not-found' : 'shadow-evidence-absent',
    evidenceArtifactId: null,
    evidenceFingerprint: null,
  }
}

export function parsePlasticRegionListQuery(searchParams) {
  const kind = searchParams.get('kind')?.trim() || null
  if (kind && !PLASTIC_REGION_ARTIFACT_KINDS.includes(kind)) {
    throw inspectionError('INVALID_PLASTIC_REGION_QUERY', 'Bilinmeyen plastik bölge artifact türü.')
  }
  return {
    kind,
    offset: boundedInteger(searchParams.get('offset'), 0, 0, MAX_LIST_OFFSET, 'offset'),
    limit: boundedInteger(searchParams.get('limit'), 50, 1, MAX_LIST_LIMIT, 'limit'),
  }
}

export function inspectPlasticRegionStatus(store, recordOverride = null) {
  const config = store.getConfig().plasticRegions
  const record = recordOverride ?? readPlasticRegionSnapshot(store)
  const activeSnapshotId = store.getActiveSnapshot()?.id ?? null
  const authority = !record.snapshot
    ? 'none'
    : record.snapshot.id === activeSnapshotId
      ? 'active'
      : 'candidate-research'
  const release = record.compatible
    ? readPlasticRegionRelease(store, record.snapshot)
    : { compatible: false, reason: record.reason, manifest: null, manifestMetadata: null }
  const readiness = inspectCurrentEvidenceReadiness(store, config)
  const kindCounts = Object.fromEntries(PLASTIC_REGION_ARTIFACT_KINDS.map((kind) => [
    kind,
    store.countArtifactMetadataByKind(kind),
  ]))
  const totalArtifacts = Object.values(kindCounts).reduce((sum, value) => sum + value, 0)
  let effectiveMode = config.enabled ? config.mode : 'off'
  let effectiveModeReason = config.enabled ? (config.mode === 'off' ? 'policy-off' : 'configured-opt-in') : 'disabled-by-policy'
  if (effectiveMode !== 'off' && config.representationProfile !== 'adaptive-eighth-id-v1') {
    effectiveMode = 'off'
    effectiveModeReason = 'adaptive-representation-required'
  } else if (['proposal', 'active'].includes(effectiveMode) && !record.compatible) {
    effectiveMode = 'off'
    effectiveModeReason = 'verified-atlas-required'
  } else if (effectiveMode === 'active') {
    if (store.getActiveSnapshot()?.id !== record.snapshot?.id) {
      effectiveMode = 'off'
      effectiveModeReason = 'active-snapshot-required'
    } else if (!release.compatible) {
      effectiveMode = 'off'
      effectiveModeReason = 'verified-region-release-required'
    }
  }
  return {
    enabled: config.enabled,
    configuredMode: config.mode,
    // The runtime lane exists, but remains opt-in. Disabled configuration
    // always wins and `active` can only come from an explicit valid config.
    effectiveMode,
    effectiveModeReason,
    runtimeWired: true,
    representationProfile: config.representationProfile,
    semanticProbesEnabled: config.semanticProbesEnabled === true,
    authority,
    routingEligible: authority === 'active' && record.compatible && release.compatible,
    activationCandidateSnapshotId: authority === 'candidate-research' && record.compatible && release.compatible
      ? record.snapshot.id
      : null,
    available: record.compatible,
    reason: record.reason,
    snapshot: record.snapshot ? {
      id: record.snapshot.id,
      status: record.snapshot.status,
      createdAt: record.snapshot.createdAt,
    } : null,
    head: record.head ? {
      artifactId: record.metadata.id,
      contentHash: record.metadata.contentHash ?? null,
      fingerprint: record.head.fingerprint,
      revision: record.head.revision,
      representationSpaceId: record.head.representationSpaceId,
      geometryFingerprint: record.head.geometryFingerprint ?? null,
      regionCount: record.head.regions.length,
      pageCount: record.head.pages.length,
      createdAt: record.metadata.createdAt ?? null,
    } : null,
    release: release.compatible ? {
      artifactId: release.manifestMetadata.id,
      releaseId: release.manifest.releaseId,
      fingerprint: release.manifest.fingerprint,
      proposalId: release.manifest.proposal.proposalId,
    } : { artifactId: null, reason: release.reason },
    readiness,
    counts: { totalArtifacts, byKind: kindCounts, source: 'kind-indexed-exact' },
    limits: {
      maximumObservationsPerBatch: config.maximumObservationsPerBatch,
      maximumObservationBatchesPerCycle: config.maximumObservationBatchesPerCycle,
      maximumRegions: config.maximumRegions,
      maximumMembershipsPerPage: config.maximumMembershipsPerPage,
      maximumActiveGates: config.maximumActiveGates,
      releaseMinimumCases: config.releaseMinimumCases,
      releaseMinimumSourceGroups: config.releaseMinimumSourceGroups,
      releaseMinimumHoldoutGain: config.releaseMinimumHoldoutGain,
      releaseMaximumShamGain: config.releaseMaximumShamGain,
      releaseMaximumRegressionLoss: config.releaseMaximumRegressionLoss,
      maximumArtifactBytes: config.maximumArtifactBytes,
    },
  }
}

export function listPlasticRegionRecords(store, options = {}) {
  const offset = options.offset ?? 0
  const limit = options.limit ?? 50
  const maximumPerKind = Math.min(4_096, offset + limit + 1)
  const all = options.kind
    ? store.listArtifactMetadataByKind(options.kind, { limit: maximumPerKind })
    : listPlasticRegionArtifacts(store, { maximumPerKind })
  const total = options.kind
    ? store.countArtifactMetadataByKind(options.kind)
    : PLASTIC_REGION_ARTIFACT_KINDS.reduce((sum, kind) => sum + store.countArtifactMetadataByKind(kind), 0)
  const items = all.slice(offset, offset + limit).map(metadataRow)
  const nextOffset = offset + items.length
  return {
    items,
    pagination: {
      offset,
      limit,
      total,
      returned: items.length,
      hasMore: nextOffset < total,
      nextOffset: nextOffset < total && nextOffset <= MAX_LIST_OFFSET ? nextOffset : null,
      cursorLimitReached: nextOffset < total && nextOffset > MAX_LIST_OFFSET,
    },
  }
}

export function inspectCurrentPlasticRegionAtlas(store) {
  const record = readPlasticRegionSnapshot(store)
  if (!record.metadata) return { ...inspectPlasticRegionStatus(store, record), atlas: null }
  if (!record.compatible) {
    throw inspectionError(
      'PLASTIC_REGION_ATLAS_UNAVAILABLE',
      'Etkin snapshot üzerindeki plastik bölge atlası okunamıyor.',
      409,
      { reason: record.reason, artifactId: record.metadata.id },
    )
  }
  return {
    // Reuse the exact decoded record so the public status/head and atlas can
    // never describe two snapshots across a concurrent activation boundary.
    ...inspectPlasticRegionStatus(store, record),
    atlas: publicPlasticRegionAtlas(record.head),
  }
}

export function inspectPlasticRegionAtlas(store, artifactId) {
  if (typeof artifactId !== 'string' || !artifactId || artifactId.length > 256) {
    throw inspectionError('INVALID_PLASTIC_REGION_QUERY', 'Atlas artifact kimliği geçersiz.')
  }
  const metadata = store.getArtifactMetadata(artifactId)
  if (!metadata || !ATLAS_KINDS.has(metadata.kind)) return null
  const maximumBytes = store.getConfig().plasticRegions.maximumArtifactBytes
  if (Number(metadata.payloadBytes) > maximumBytes) {
    throw inspectionError('PLASTIC_REGION_ARTIFACT_TOO_LARGE', 'Atlas artifacti güvenli görüntüleme sınırını aşıyor.', 409)
  }
  const artifact = store.getArtifact(artifactId)
  let value
  try {
    value = restoreArtifact(artifact, metadata.kind)
    assertPlasticRegionArtifactValue(metadata.kind, value)
    assertPlasticRegionArtifactMetadata(metadata.kind, value, artifact)
  } catch (error) {
    throw inspectionError(
      'PLASTIC_REGION_ATLAS_INVALID',
      'Plastik bölge atlas artifacti doğrulanamadı.',
      409,
      { artifactId, cause: String(error?.code ?? error?.message ?? error).slice(0, 240) },
    )
  }
  const atlas = metadata.kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND
    ? value
    : value.proposedAtlas
  if (!atlas || atlas.format !== PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND) {
    throw inspectionError('PLASTIC_REGION_ATLAS_INVALID', 'Öneri artifacti doğrulanabilir bir atlas taşımıyor.', 409)
  }
  return {
    artifact: metadataRow(metadata),
    proposal: metadata.kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ? {
      proposalId: value.proposalId ?? null,
      action: value.action ?? null,
      eligible: value.eligible === true,
      rejectionReasons: strings(value.rejectionReasons, 32),
      fingerprint: value.fingerprint ?? null,
      baseAtlasFingerprint: value.baseAtlasFingerprint ?? null,
    } : null,
    atlas: publicPlasticRegionAtlas(atlas),
  }
}
