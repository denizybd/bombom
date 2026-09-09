import type {
  PlasticRegionArtifactView,
  PlasticRegionAtlasView,
  PlasticRegionAtlasDetail,
  PlasticRegionDashboard,
  PlasticRegionMode,
  PlasticRegionStatus,
  PlasticRegionView,
  RegionPageView,
  SemanticProbeInspectionView,
  SemanticProbeStatus,
} from './types'

const modes = new Set<PlasticRegionMode>(['off', 'shadow', 'proposal', 'active'])
const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
const array = (value: unknown): unknown[] => Array.isArray(value) ? value : []
const text = (value: unknown): string => typeof value === 'string' ? value : ''
const finite = (value: unknown, fallback = 0): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback
const integer = (value: unknown, fallback = 0): number => {
  const result = finite(value, fallback)
  return Number.isSafeInteger(result) && result >= 0 ? result : fallback
}
const probability = (value: unknown): number => Math.min(1, Math.max(0, finite(value)))
const mode = (value: unknown): PlasticRegionMode => modes.has(value as PlasticRegionMode)
  ? value as PlasticRegionMode
  : 'off'

function normalizePage(value: unknown, maximumMemberships = 3): RegionPageView | null {
  const source = record(value)
  const pageIndex = integer(source.pageIndex, -1)
  if (pageIndex < 0 || pageIndex >= 56) return null
  const seen = new Set<string>()
  const memberships = array(source.memberships).map((entry) => {
    const item = record(entry)
    return { regionId: text(item.regionId), weight: probability(item.weight) }
  }).filter((entry) => {
    if (!entry.regionId || seen.has(entry.regionId)) return false
    seen.add(entry.regionId)
    return true
  }).slice(0, Math.max(1, Math.min(8, maximumMemberships)))
  const total = memberships.reduce((sum, entry) => sum + entry.weight, 0)
  const requestedFreeMass = typeof source.freeMass === 'number' && Number.isFinite(source.freeMass)
    ? probability(source.freeMass)
    : Math.max(0, 1 - total)
  if (total + requestedFreeMass <= 1 + 1e-6) {
    return { pageIndex, memberships, freeMass: requestedFreeMass }
  }
  const available = Math.max(0, 1 - requestedFreeMass)
  const scale = total > 0 ? available / total : 0
  return {
    pageIndex,
    memberships: memberships.map((entry) => ({ ...entry, weight: entry.weight * scale })),
    freeMass: requestedFreeMass,
  }
}

function normalizeRegion(value: unknown): PlasticRegionView | null {
  const source = record(value)
  const id = text(source.id)
  if (!id) return null
  const prototype = record(source.prototype)
  const evidence = record(source.evidence)
  return {
    id,
    lifecycle: text(source.lifecycle) || 'provisional',
    origin: text(source.origin) || 'discovery',
    bornRevision: integer(source.bornRevision),
    retiredRevision: source.retiredRevision === null || source.retiredRevision === undefined
      ? null
      : integer(source.retiredRevision),
    parentRegionIds: array(source.parentRegionIds).map(text).filter(Boolean).slice(0, 56),
    prototype: {
      dimension: integer(prototype.dimension),
      values: array(prototype.values).map((entry) => finite(entry)).slice(0, 256),
      truncated: prototype.truncated === true,
    },
    evidence: {
      observationCount: integer(evidence.observationCount),
      sourceGroupCount: integer(evidence.sourceGroupCount),
      confidenceMass: Math.max(0, finite(evidence.confidenceMass)),
      cohesion: probability(evidence.cohesion),
      holdoutUtility: finite(evidence.holdoutUtility),
      shamUtility: finite(evidence.shamUtility),
    },
  }
}

export function normalizeRegionStatus(value: unknown): PlasticRegionStatus {
  const source = record(value)
  const snapshot = record(source.snapshot)
  const head = record(source.head)
  const counts = record(source.counts)
  const limits = record(source.limits)
  const release = record(source.release)
  const readiness = record(source.readiness)
  return {
    enabled: source.enabled === true,
    configuredMode: mode(source.configuredMode),
    effectiveMode: mode(source.effectiveMode),
    effectiveModeReason: text(source.effectiveModeReason) || 'unknown',
    runtimeWired: source.runtimeWired === true,
    representationProfile: text(source.representationProfile) || 'legacy',
    semanticProbesEnabled: source.semanticProbesEnabled === true,
    authority: source.authority === 'active' || source.authority === 'candidate-research'
      ? source.authority
      : 'none',
    routingEligible: source.routingEligible === true,
    activationCandidateSnapshotId: text(source.activationCandidateSnapshotId) || null,
    available: source.available === true,
    reason: text(source.reason) || null,
    snapshot: text(snapshot.id) ? {
      id: text(snapshot.id), status: text(snapshot.status), createdAt: text(snapshot.createdAt),
    } : null,
    head: text(head.artifactId) ? {
      artifactId: text(head.artifactId),
      contentHash: text(head.contentHash) || null,
      fingerprint: text(head.fingerprint),
      revision: integer(head.revision),
      representationSpaceId: text(head.representationSpaceId),
      geometryFingerprint: text(head.geometryFingerprint) || null,
      regionCount: integer(head.regionCount),
      pageCount: integer(head.pageCount),
      createdAt: text(head.createdAt) || null,
    } : null,
    release: {
      artifactId: text(release.artifactId) || null,
      releaseId: text(release.releaseId) || null,
      fingerprint: text(release.fingerprint) || null,
      proposalId: text(release.proposalId) || null,
      reason: text(release.reason) || null,
    },
    readiness: {
      shadowEvidenceAvailable: readiness.shadowEvidenceAvailable === true,
      reason: text(readiness.reason) || null,
      evidenceArtifactId: text(readiness.evidenceArtifactId) || null,
      evidenceFingerprint: text(readiness.evidenceFingerprint) || null,
    },
    counts: {
      totalArtifacts: integer(counts.totalArtifacts),
      byKind: Object.fromEntries(Object.entries(record(counts.byKind)).map(([key, entry]) => [key, integer(entry)])),
    },
    limits: {
      maximumObservationsPerBatch: integer(limits.maximumObservationsPerBatch),
      maximumObservationBatchesPerCycle: integer(limits.maximumObservationBatchesPerCycle),
      maximumRegions: integer(limits.maximumRegions, 56),
      maximumMembershipsPerPage: integer(limits.maximumMembershipsPerPage, 3),
      maximumActiveGates: integer(limits.maximumActiveGates, 4),
      releaseMinimumCases: integer(limits.releaseMinimumCases),
      releaseMinimumSourceGroups: integer(limits.releaseMinimumSourceGroups),
      releaseMinimumHoldoutGain: probability(limits.releaseMinimumHoldoutGain),
      releaseMaximumShamGain: probability(limits.releaseMaximumShamGain),
      releaseMaximumRegressionLoss: probability(limits.releaseMaximumRegressionLoss),
      maximumArtifactBytes: integer(limits.maximumArtifactBytes),
    },
  }
}

export function normalizeRegionAtlas(value: unknown): PlasticRegionAtlasView | null {
  const source = record(value)
  const atlas = record(source.atlas)
  if (atlas.format !== 'plastic-region-atlas-head' || atlas.version !== 1) return null
  const representationSpaceId = text(atlas.representationSpaceId)
  const fingerprint = text(atlas.fingerprint)
  if (!representationSpaceId || !fingerprint) return null
  const maximumMembershipsPerPage = Math.max(1, Math.min(8, integer(atlas.maximumMembershipsPerPage, 3)))
  const seenRegions = new Set<string>()
  const regions = array(atlas.regions).map(normalizeRegion).filter((entry): entry is PlasticRegionView => {
    if (!entry || seenRegions.has(entry.id)) return false
    seenRegions.add(entry.id)
    return true
  }).slice(0, 56)
  const seenPages = new Set<number>()
  const pages = array(atlas.pages)
    .map((entry) => normalizePage(entry, maximumMembershipsPerPage))
    .filter((entry): entry is RegionPageView => {
      if (!entry || seenPages.has(entry.pageIndex)) return false
      seenPages.add(entry.pageIndex)
      const originalMass = entry.memberships.reduce((sum, membership) => sum + membership.weight, 0)
      entry.memberships = entry.memberships.filter((membership) => seenRegions.has(membership.regionId))
      const retainedMass = entry.memberships.reduce((sum, membership) => sum + membership.weight, 0)
      entry.freeMass = probability(entry.freeMass + originalMass - retainedMass)
      return true
    })
    .sort((left, right) => left.pageIndex - right.pageIndex)
  const lifecycleCounts: Record<string, number> = {}
  for (const region of regions) lifecycleCounts[region.lifecycle] = (lifecycleCounts[region.lifecycle] ?? 0) + 1
  const assignedMass = pages.reduce(
    (sum, page) => sum + page.memberships.reduce((pageSum, membership) => pageSum + membership.weight, 0),
    0,
  )
  const freeMass = pages.reduce((sum, page) => sum + page.freeMass, 0) + Math.max(0, 56 - pages.length)
  return {
    format: 'plastic-region-atlas-head',
    version: 1,
    representationSpaceId,
    geometryFingerprint: text(atlas.geometryFingerprint) || null,
    revision: integer(atlas.revision),
    parentFingerprint: text(atlas.parentFingerprint) || null,
    behaviorDimension: integer(atlas.behaviorDimension),
    maximumMembershipsPerPage,
    fingerprint,
    regions,
    pages,
    stats: {
      regionCount: regions.length,
      pageCount: pages.length,
      lifecycleCounts,
      assignedMass,
      freeMass,
    },
    truncated: atlas.truncated === true,
  }
}

export function normalizeRegionArtifacts(value: unknown): PlasticRegionArtifactView[] {
  return array(record(value).items).map((entry) => {
    const item = record(entry)
    return {
      artifactId: text(item.artifactId),
      kind: text(item.kind),
      contentHash: text(item.contentHash) || null,
      createdAt: text(item.createdAt) || null,
      jobId: text(item.jobId) || null,
      payloadBytes: integer(item.payloadBytes),
    }
  }).filter((entry) => entry.artifactId && entry.kind)
}

function normalizeProbeInterval(value: unknown) {
  const source = record(value)
  return {
    point: finite(source.point),
    low: finite(source.low),
    high: finite(source.high),
    confidenceLevel: probability(source.confidenceLevel),
  }
}

export function normalizeSemanticProbeInspection(value: unknown): SemanticProbeInspectionView {
  const source = record(value)
  const summary = record(source.summary)
  const legacy = record(source.legacySkeletons)
  const statuses = new Set<SemanticProbeStatus>(['unknown', 'provisional', 'calibrated'])
  const items = array(summary.items).map((entry) => {
    const item = record(entry)
    const hypothesis = record(item.hypothesis)
    const target = record(item.target)
    const evidence = record(item.evidence)
    const status = statuses.has(item.status as SemanticProbeStatus)
      ? item.status as SemanticProbeStatus
      : 'unknown'
    const featureId = text(target.featureId)
    const evaluationSplit: 'validation' | 'test' = evidence.evaluationSplit === 'test' ? 'test' : 'validation'
    return {
      hypothesis: {
        hypothesisId: text(hypothesis.hypothesisId),
        namespace: text(hypothesis.namespace),
        label: text(hypothesis.label),
      },
      status,
      atlasFingerprint: text(item.atlasFingerprint),
      observationCount: integer(item.observationCount),
      independentSourceCount: integer(item.independentSourceCount),
      target: featureId ? {
        featureId,
        responseKind: text(target.responseKind),
        footprint: array(target.footprint).map((footprint) => {
          const point = record(footprint)
          return { pageIndex: integer(point.pageIndex, -1), weight: probability(point.weight) }
        }).filter((point) => point.pageIndex >= 0 && point.pageIndex < 56).slice(0, 56),
      } : null,
      evidence: item.evidence && typeof item.evidence === 'object' ? {
        evaluationSplit,
        targetBrier: probability(evidence.targetBrier),
        targetEce: probability(evidence.targetEce),
        gainVsBaseline: normalizeProbeInterval(evidence.gainVsBaseline),
        gainVsSham: normalizeProbeInterval(evidence.gainVsSham),
        reasons: array(evidence.reasons).map(text).filter(Boolean).slice(0, 32),
      } : null,
    }
  }).filter((item) => item.hypothesis.hypothesisId && item.hypothesis.namespace && item.hypothesis.label).slice(0, 128)
  return {
    available: source.available === true && items.length > 0,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    releaseAuthority: false,
    atlasFingerprint: text(source.atlasFingerprint) || null,
    bytesRead: integer(source.bytesRead),
    legacySkeletonCount: integer(legacy.recognizedCount),
    items,
    omittedCount: integer(summary.omittedCount),
  }
}

export function mergeRegionDashboard(statusRaw: unknown, atlasRaw: unknown, artifactsRaw: unknown, probesRaw?: unknown): PlasticRegionDashboard {
  return {
    status: normalizeRegionStatus(statusRaw),
    atlas: normalizeRegionAtlas(atlasRaw),
    artifacts: normalizeRegionArtifacts(artifactsRaw),
    probes: normalizeSemanticProbeInspection(probesRaw),
  }
}

export function normalizeRegionAtlasDetail(value: unknown): PlasticRegionAtlasDetail | null {
  const source = record(value)
  const atlas = normalizeRegionAtlas(source)
  if (!atlas) return null
  const proposal = record(source.proposal)
  return {
    artifact: normalizeRegionArtifacts({ items: [source.artifact] })[0] ?? null,
    proposal: text(proposal.proposalId) ? {
      proposalId: text(proposal.proposalId) || null,
      action: text(proposal.action) || null,
      eligible: proposal.eligible === true,
      rejectionReasons: array(proposal.rejectionReasons).map(text).filter(Boolean).slice(0, 32),
      fingerprint: text(proposal.fingerprint) || null,
      baseAtlasFingerprint: text(proposal.baseAtlasFingerprint) || null,
    } : null,
    atlas,
  }
}
