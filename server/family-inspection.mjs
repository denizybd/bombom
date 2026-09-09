const MAX_FAMILY_LIMIT = 100
const MAX_MEMBER_LIMIT = 12
const MAX_TRANSITION_LIMIT = 12
const MAX_PREDECESSORS = 12
const MAX_MEMBER_SOURCES = 4
const MAX_TEXT_LENGTH = 240
const MAX_REASON_LENGTH = 360
const MAX_ID_LENGTH = 256

const FAMILY_ARTIFACT_KINDS = new Set(['family-run', 'runtime-model-state'])
const MEMORY_HEAVY_JOB_TYPES = new Set(['teacher-feed-campaign', 'full-cycle'])

function memoryHeavyTrainingJob(store) {
  return store.listJobs().find((job) =>
    MEMORY_HEAVY_JOB_TYPES.has(job?.type) && (
      ['running', 'pause_requested', 'cancel_requested'].includes(job?.state) ||
      (['queued', 'paused'].includes(job?.state) && Boolean(job?.payload?.feedState?.pendingTrainingMilestone))
    ),
  ) ?? null
}

function inspectionError(message, code, status) {
  const error = new Error(message)
  error.code = code
  error.status = status
  return error
}

function boundedString(value, maximum = MAX_ID_LENGTH) {
  if (typeof value !== 'string') return ''
  return value.slice(0, maximum)
}

function displayText(value, maximum = MAX_TEXT_LENGTH) {
  const normalized = typeof value === 'string' ? value.replace(/\s+/gu, ' ').trim() : ''
  if (normalized.length <= maximum) return { text: normalized, textTruncated: false }
  return { text: `${normalized.slice(0, Math.max(0, maximum - 1)).trimEnd()}…`, textTruncated: true }
}

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function count(value) {
  return Number.isInteger(value) && value >= 0 ? value : 0
}

function array(value) {
  return Array.isArray(value) ? value : []
}

function parseInteger(value, { name, fallback, minimum, maximum }) {
  if (value === undefined || value === null || value === '') return fallback
  if (!/^\d+$/u.test(String(value))) {
    throw inspectionError(`${name} negatif olmayan bir tam sayı olmalı.`, 'INVALID_FAMILY_INSPECTION_QUERY', 400)
  }
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw inspectionError(`${name} ${minimum}–${maximum} aralığında olmalı.`, 'INVALID_FAMILY_INSPECTION_QUERY', 400)
  }
  return parsed
}

export function parseFamilyInspectionQuery(searchParams) {
  const snapshotId = searchParams.get('snapshotId')
  if (snapshotId !== null && (snapshotId.length < 1 || snapshotId.length > MAX_ID_LENGTH)) {
    throw inspectionError('snapshotId uzunluğu geçersiz.', 'INVALID_FAMILY_INSPECTION_QUERY', 400)
  }
  return {
    snapshotId: snapshotId ?? undefined,
    offset: parseInteger(searchParams.get('offset'), { name: 'offset', fallback: 0, minimum: 0, maximum: 1_000_000 }),
    limit: parseInteger(searchParams.get('limit'), { name: 'limit', fallback: 50, minimum: 1, maximum: MAX_FAMILY_LIMIT }),
    memberLimit: parseInteger(searchParams.get('memberLimit'), { name: 'memberLimit', fallback: 8, minimum: 0, maximum: MAX_MEMBER_LIMIT }),
    transitionLimit: parseInteger(searchParams.get('transitionLimit'), { name: 'transitionLimit', fallback: 8, minimum: 0, maximum: MAX_TRANSITION_LIMIT }),
  }
}

function artifactMetadataForSnapshot(store, snapshot) {
  const artifacts = array(snapshot?.artifactIds)
    .map((id) => store.getArtifactMetadata(id))
    .filter((artifact) => artifact && FAMILY_ARTIFACT_KINDS.has(artifact.kind))
  if (!artifacts.length) return null
  // The dedicated artifact is smaller and is the authoritative output of the
  // family stage. The runtime model is a compatibility fallback for snapshots
  // that only retained the consolidated checkpoint.
  const selected = [...artifacts].sort((left, right) => {
    const kindRank = (artifact) => artifact.kind === 'family-run' ? 0 : 1
    return kindRank(left) - kindRank(right) || String(right.createdAt).localeCompare(String(left.createdAt))
  })[0]
  // Selection only needs the tiny side-table row. Materialize exactly one
  // immutable payload after the authoritative family artifact is known.
  return selected
}

/**
 * Resolve only side-table metadata. The 3B atlas uses this immutable hint to
 * answer cache hits without inflating the large FamilyRun payload again.
 */
export function selectFamilyProjectionMetadata(store, snapshotId) {
  if (snapshotId !== undefined) {
    const snapshot = store.getSnapshot(snapshotId)
    if (!snapshot) throw inspectionError('İstenen model sürümü bulunamadı.', 'SNAPSHOT_NOT_FOUND', 404)
    return { selectedBy: 'requested-snapshot', snapshot, artifact: artifactMetadataForSnapshot(store, snapshot) }
  }

  const snapshots = [...store.listSnapshots()].sort((left, right) =>
    String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)),
  )
  for (const snapshot of snapshots) {
    const artifact = artifactMetadataForSnapshot(store, snapshot)
    if (artifact) return { selectedBy: 'latest-snapshot', snapshot, artifact }
  }

  const artifact = store.listArtifactMetadata()
    .filter((item) => FAMILY_ARTIFACT_KINDS.has(item.kind))
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))[0]
  return artifact
    ? { selectedBy: 'latest-artifact', snapshot: null, artifact }
    : { selectedBy: 'none', snapshot: null, artifact: null }
}

function materializeProjectionSelection(store, selection) {
  return {
    ...selection,
    artifact: selection.artifact ? store.getArtifact(selection.artifact.id) : null,
  }
}

function selectInspectionTarget(store, snapshotId) {
  return materializeProjectionSelection(store, selectFamilyProjectionMetadata(store, snapshotId))
}

function familyRunFromArtifact(artifact) {
  if (!artifact) return null
  if (artifact.payload?.format !== 'bombom-runtime-artifact' || artifact.payload?.version !== 1) {
    throw inspectionError('Aile kontrol noktası biçimi doğrulanamadı.', 'INVALID_FAMILY_ARTIFACT', 409)
  }
  const value = artifact.payload.value
  const run = artifact.kind === 'family-run' ? value : value?.familyRun
  if (
    !run || run.format !== 'noron-aileleri-run' || run.version !== 1 ||
    !Array.isArray(run.families) || !Array.isArray(run.memberships) ||
    !Array.isArray(run.profiles) || !Array.isArray(run.familyTransitions)
  ) {
    throw inspectionError('Aile çalışması eksik veya uyumsuz.', 'INVALID_FAMILY_ARTIFACT', 409)
  }
  return run
}

/**
 * Internal, read-only source for other bounded projections such as the 3D
 * atlas. Callers receive the already validated FamilyRun but must never expose
 * its vectors/prototypes directly.
 */
export function loadFamilyProjectionSource(store, snapshotId, metadataSelection) {
  const selection = materializeProjectionSelection(
    store,
    metadataSelection ?? selectFamilyProjectionMetadata(store, snapshotId),
  )
  return {
    selection,
    run: selection.artifact ? familyRunFromArtifact(selection.artifact) : null,
  }
}

function snapshotReference(store, snapshot) {
  if (!snapshot) return null
  return {
    id: boundedString(snapshot.id),
    name: boundedString(snapshot.label, 200),
    status: boundedString(snapshot.status, 40),
    createdAt: boundedString(snapshot.createdAt, 64),
    active: store.getActiveSnapshot()?.id === snapshot.id,
  }
}

function artifactReference(artifact) {
  if (!artifact) return null
  return {
    id: boundedString(artifact.id),
    kind: artifact.kind,
    createdAt: boundedString(artifact.createdAt, 64),
    contentHash: artifact.contentHash ? boundedString(artifact.contentHash) : null,
    jobId: artifact.jobId ? boundedString(artifact.jobId) : null,
  }
}

function runReference(run) {
  if (!run) return null
  return {
    id: boundedString(run.id),
    fingerprint: boundedString(run.fingerprint),
    createdAt: boundedString(run.createdAt, 64),
    representationSpaceId: boundedString(run.representationSpaceId),
    behaviorSpaceId: boundedString(run.behaviorSpaceId),
  }
}

function groupedCounts(values) {
  const counts = new Map()
  for (const value of values) {
    const key = boundedString(value, 80) || 'unknown'
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return [...counts].sort(([left], [right]) => left.localeCompare(right)).map(([key, value]) => ({ key, count: value }))
}

function emptySummary() {
  return {
    familyCount: 0,
    membershipCount: 0,
    profileCount: 0,
    familyTransitionCount: 0,
    distinctSourceCount: 0,
    multiSourceFamilyCount: 0,
    utilityMeasuredFamilyCount: 0,
    validatedUtilityFamilyCount: 0,
    byLifecycle: [],
    byRegion: [],
  }
}

function publicSummary(run) {
  const reports = array(run.utilityReports)
  return {
    familyCount: run.families.length,
    membershipCount: run.memberships.length,
    profileCount: run.profiles.length,
    familyTransitionCount: run.familyTransitions.length,
    distinctSourceCount: count(run.stats?.distinctSourceCount),
    multiSourceFamilyCount: run.families.filter((family) => count(family?.sourceCount) >= 2).length,
    utilityMeasuredFamilyCount: run.families.filter((family) => finite(family?.utility) !== null).length,
    validatedUtilityFamilyCount: reports.filter((report) => report?.status === 'validated').length,
    byLifecycle: groupedCounts(run.families.map((family) => family?.lifecycle)),
    byRegion: groupedCounts(run.families.map((family) => family?.region)),
  }
}

function publicUtilityEvidence(report) {
  if (!report) return null
  return {
    status: boundedString(report.status, 80) || 'unknown',
    caseCount: count(report.caseCount),
    sourceFoldCount: count(report.sourceFoldCount),
    fullLoss: finite(report.fullLoss),
    ablatedLoss: finite(report.ablatedLoss),
    deltaUtility: finite(report.deltaUtility),
    confidenceLow: finite(report.confidenceLow),
    confidenceHigh: finite(report.confidenceHigh),
    explanation: boundedString(report.explanation, MAX_REASON_LENGTH),
  }
}

function sourceReferences(profile, sourceById) {
  const ids = []
  const seen = new Set()
  for (const evidence of array(profile?.evidence)) {
    if (typeof evidence?.sourceId !== 'string' || seen.has(evidence.sourceId)) continue
    seen.add(evidence.sourceId)
    ids.push(evidence.sourceId)
  }
  return {
    sourceCount: ids.length,
    sources: ids.slice(0, MAX_MEMBER_SOURCES).map((id) => ({
      id: boundedString(id),
      name: boundedString(sourceById.get(id)?.title ?? id, 200),
    })),
  }
}

function publicMembers(family, membershipsByFamily, profileByMember, sourceById, memberLimit) {
  if (memberLimit === 0) return []
  return array(membershipsByFamily.get(family.id))
    .sort((left, right) =>
      Number(Boolean(right?.core)) - Number(Boolean(left?.core)) ||
      (finite(right?.score) ?? -Infinity) - (finite(left?.score) ?? -Infinity) ||
      (finite(right?.weight) ?? -Infinity) - (finite(left?.weight) ?? -Infinity) ||
      String(left?.memberKey).localeCompare(String(right?.memberKey)),
    )
    .slice(0, memberLimit)
    .map((membership) => {
      const profile = profileByMember.get(membership.memberKey)
      const text = displayText(profile?.text ?? array(profile?.evidence)[0]?.text)
      return {
        memberKey: boundedString(membership.memberKey),
        kind: profile?.kind === 'synthesis' ? 'synthesis' : 'raw',
        ...text,
        softScore: finite(membership.score),
        weight: finite(membership.weight),
        rank: count(membership.rank),
        core: Boolean(membership.core),
        ...sourceReferences(profile, sourceById),
      }
    })
}

function publicTransitions(family, transitionsByFamily, familyById, transitionLimit) {
  if (transitionLimit === 0) return []
  return array(transitionsByFamily.get(family.id))
    .sort((left, right) =>
      (finite(right?.probability) ?? -Infinity) - (finite(left?.probability) ?? -Infinity) ||
      String(left?.toFamilyId).localeCompare(String(right?.toFamilyId)),
    )
    .slice(0, transitionLimit)
    .map((transition) => {
      const target = familyById.get(transition.toFamilyId)
      return {
        id: boundedString(transition.id),
        toFamilyId: boundedString(transition.toFamilyId),
        toRegion: boundedString(target?.region, 80) || 'unknown',
        toLifecycle: boundedString(target?.lifecycle, 80) || 'unknown',
        probability: finite(transition.probability),
        sourceCount: count(transition.sourceCount),
        supportCount: count(transition.supportCount),
        priorEvidence: finite(transition.priorEvidence),
        experienceEvidence: finite(transition.experienceEvidence),
        inhibitionEvidence: finite(transition.inhibitionEvidence),
      }
    })
}

function emptyResponse(selection, options, store, reason) {
  return {
    available: false,
    reason,
    selection: {
      selectedBy: selection.selectedBy,
      snapshot: snapshotReference(store, selection.snapshot),
      artifact: artifactReference(selection.artifact),
    },
    run: null,
    summary: emptySummary(),
    pagination: { offset: options.offset, limit: options.limit, total: 0, returned: 0, hasMore: false, nextOffset: null },
    limits: {
      memberLimit: options.memberLimit,
      transitionLimit: options.transitionLimit,
      maxFamilyLimit: MAX_FAMILY_LIMIT,
      maxMemberLimit: MAX_MEMBER_LIMIT,
      maxTransitionLimit: MAX_TRANSITION_LIMIT,
    },
    families: [],
  }
}

/**
 * Builds a deliberately thin projection of a FamilyRun. Profiles' identity
 * arrays, relation arrays, sparse feature vectors and family prototypes never
 * cross this boundary.
 */
export function inspectFamilies(store, options) {
  const activeTraining = memoryHeavyTrainingJob(store)
  if (activeTraining) {
    const metadataSelection = selectFamilyProjectionMetadata(store, options.snapshotId)
    return {
      ...emptyResponse(
        metadataSelection,
        options,
        store,
        'Aile eğitimi sürerken bellek eğitim motoruna ayrıldı. Denetim görünümü işlem tamamlanınca yenilenecek.',
      ),
      busy: true,
      busyStage: boundedString(activeTraining.stage, 160) || 'training',
    }
  }
  const selection = selectInspectionTarget(store, options.snapshotId)
  if (!selection.artifact) {
    return emptyResponse(
      selection,
      options,
      store,
      selection.snapshot
        ? 'Bu model sürümünde aile çalışması bulunmuyor.'
        : 'Henüz denetlenebilir bir nöron ailesi çalışması yok.',
    )
  }

  const run = familyRunFromArtifact(selection.artifact)
  const sourceById = new Map(store.listSources().map((source) => [source.id, source]))
  const profileByMember = new Map(run.profiles.map((profile) => [profile?.memberKey, profile]))
  const familyById = new Map(run.families.map((family) => [family?.id, family]))
  const utilityByFamily = new Map(array(run.utilityReports).map((report) => [report?.familyId, report]))
  const membershipsByFamily = new Map()
  for (const membership of run.memberships) {
    if (typeof membership?.familyId !== 'string') continue
    const values = membershipsByFamily.get(membership.familyId) ?? []
    values.push(membership)
    membershipsByFamily.set(membership.familyId, values)
  }
  const transitionsByFamily = new Map()
  for (const transition of run.familyTransitions) {
    if (typeof transition?.fromFamilyId !== 'string') continue
    const values = transitionsByFamily.get(transition.fromFamilyId) ?? []
    values.push(transition)
    transitionsByFamily.set(transition.fromFamilyId, values)
  }

  const page = run.families.slice(options.offset, options.offset + options.limit)
  const families = page.map((family) => ({
    id: boundedString(family?.id),
    region: boundedString(family?.region, 80) || 'unknown',
    regionVerified: Boolean(family?.regionVerified),
    lifecycle: boundedString(family?.lifecycle, 80) || 'unknown',
    lifecycleReason: boundedString(family?.lifecycleReason, MAX_REASON_LENGTH),
    supportCount: count(family?.supportCount),
    sourceCount: count(family?.sourceCount),
    memberCount: array(family?.memberKeys).length,
    coreMemberCount: array(family?.coreMemberKeys).length,
    cohesion: finite(family?.cohesion),
    separation: finite(family?.separation),
    stability: finite(family?.stability),
    utility: finite(family?.utility),
    predecessorFamilyIds: array(family?.predecessorFamilyIds).slice(0, MAX_PREDECESSORS).map((id) => boundedString(id)),
    utilityEvidence: publicUtilityEvidence(utilityByFamily.get(family?.id)),
    topMembers: publicMembers(family, membershipsByFamily, profileByMember, sourceById, options.memberLimit),
    outgoingTransitions: publicTransitions(family, transitionsByFamily, familyById, options.transitionLimit),
  }))
  const nextOffset = options.offset + families.length

  return {
    available: true,
    selection: {
      selectedBy: selection.selectedBy,
      snapshot: snapshotReference(store, selection.snapshot),
      artifact: artifactReference(selection.artifact),
    },
    run: runReference(run),
    summary: publicSummary(run),
    pagination: {
      offset: options.offset,
      limit: options.limit,
      total: run.families.length,
      returned: families.length,
      hasMore: nextOffset < run.families.length,
      nextOffset: nextOffset < run.families.length ? nextOffset : null,
    },
    limits: {
      memberLimit: options.memberLimit,
      transitionLimit: options.transitionLimit,
      maxFamilyLimit: MAX_FAMILY_LIMIT,
      maxMemberLimit: MAX_MEMBER_LIMIT,
      maxTransitionLimit: MAX_TRANSITION_LIMIT,
    },
    families,
  }
}
