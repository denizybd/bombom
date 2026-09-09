import {
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_REGION_GEOMETRY,
  PAGE_BEHAVIOR_DIMENSION,
  aggregatePageSketches,
} from '../src/adaptive-regions/index.ts'
import { restoreArtifact, stableRuntimeHash } from './artifacts.mjs'

export const PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND = 'plastic-region-evidence-head'

function evidenceError(code, message, details) {
  const error = new Error(message)
  error.code = code
  if (details !== undefined) error.details = details
  return error
}

function compareText(left, right) {
  return String(left).localeCompare(String(right))
}

function transitionKey(observation) {
  const suffix = `:${observation.pageIndex}`
  const prefix = String(observation.observationId).endsWith(suffix)
    ? String(observation.observationId).slice(0, -suffix.length)
    : String(observation.observationId)
  return `${observation.sourceGroupId}\u0000${observation.episodeId}\u0000${prefix}`
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (value && typeof value === 'object' && !ArrayBuffer.isView(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function validateObservationSet(observations) {
  if (!Array.isArray(observations)) throw evidenceError('INVALID_PLASTIC_REGION_EVIDENCE', 'Birikimli gözlemler dizi olmalı.')
  // The core aggregator performs the canonical range/dimension checks without
  // assigning semantic meaning to any behavior coordinate.
  if (observations.length) aggregatePageSketches(observations, {
    behaviorDimension: PAGE_BEHAVIOR_DIMENSION,
    maximumEvidence: 1,
  })
}

function completeTransitions(observations) {
  const byTransition = new Map()
  for (const observation of observations) {
    const key = transitionKey(observation)
    const pages = byTransition.get(key) ?? new Map()
    const previous = pages.get(observation.pageIndex)
    if (previous && canonicalJson(previous) !== canonicalJson(observation)) {
      throw evidenceError('PLASTIC_REGION_EVIDENCE_CONFLICT', 'Aynı transition/page kimliği farklı içerik taşıyor.', {
        transitionKey: key, pageIndex: observation.pageIndex,
      })
    }
    pages.set(observation.pageIndex, observation)
    byTransition.set(key, pages)
  }
  const complete = []
  let incompleteTransitionCount = 0
  for (const [key, pages] of byTransition) {
    if (pages.size !== ADAPTIVE_PAGE_COUNT ||
        Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_, index) => pages.has(index)).includes(false)) {
      incompleteTransitionCount += 1
      continue
    }
    const values = [...pages.values()].sort((left, right) => left.pageIndex - right.pageIndex)
    complete.push({
      key,
      sourceGroupId: values[0].sourceGroupId,
      episodeId: values[0].episodeId,
      observations: values,
      orderKey: stableRuntimeHash('plastic-region-transition-order-v1', key),
    })
  }
  return { complete, incompleteTransitionCount }
}

function sourceBalancedCompaction(transitions, maximumTransitions) {
  const groups = new Map()
  for (const transition of transitions) {
    const values = groups.get(transition.sourceGroupId) ?? []
    values.push(transition)
    groups.set(transition.sourceGroupId, values)
  }
  const ordered = [...groups.entries()]
    .map(([groupId, values]) => [groupId, values.sort((left, right) =>
      compareText(left.orderKey, right.orderKey) || compareText(left.key, right.key))])
    .sort(([left], [right]) =>
      compareText(stableRuntimeHash('plastic-region-evidence-group-v1', left),
        stableRuntimeHash('plastic-region-evidence-group-v1', right)) || compareText(left, right))
  const retained = []
  for (let round = 0; retained.length < maximumTransitions; round += 1) {
    let added = false
    for (const [, values] of ordered) {
      if (!values[round]) continue
      retained.push(values[round])
      added = true
      if (retained.length >= maximumTransitions) break
    }
    if (!added) break
  }
  return retained.sort((left, right) => compareText(left.key, right.key))
}

export function createPlasticRegionEvidenceHead(input) {
  const maximumObservations = Number(input.maximumObservations ?? 8_176)
  if (!Number.isSafeInteger(maximumObservations) || maximumObservations < ADAPTIVE_PAGE_COUNT) {
    throw evidenceError('INVALID_PLASTIC_REGION_EVIDENCE', 'Birikimli gözlem sınırı en az 56 olmalı.')
  }
  const parentHeads = [...(input.parentHeads ?? [])]
  for (const head of parentHeads) {
    validatePlasticRegionEvidenceHead(head)
    if (head.representationSpaceId !== input.representationSpaceId ||
        head.baseAtlasFingerprint !== input.baseAtlasFingerprint ||
        head.authority !== (input.authority === 'cas-active-base' ? 'cas-active-base' : 'research-only')) {
      throw evidenceError(
        'PLASTIC_REGION_EVIDENCE_LINEAGE_MISMATCH',
        'Birikimli kanıt parentı farklı temsil, atlas veya otorite soyuna ait.',
      )
    }
  }
  const allObservations = [
    ...parentHeads.flatMap((head) => head.observations),
    ...(input.observations ?? []),
  ]
  validateObservationSet(input.observations ?? [])
  const { complete, incompleteTransitionCount } = completeTransitions(allObservations)
  const deduplicated = new Map()
  for (const transition of complete) {
    const previous = deduplicated.get(transition.key)
    if (previous && canonicalJson(previous.observations) !== canonicalJson(transition.observations)) {
      throw evidenceError('PLASTIC_REGION_EVIDENCE_CONFLICT', 'Aynı runKey transitionı farklı içerikle yinelendi.')
    }
    deduplicated.set(transition.key, transition)
  }
  const unique = [...deduplicated.values()]
  const maximumTransitions = Math.max(1, Math.floor(maximumObservations / ADAPTIVE_PAGE_COUNT))
  const retained = sourceBalancedCompaction(unique, maximumTransitions)
  const observations = retained.flatMap((transition) => transition.observations)
  const sourceGroupIds = [...new Set(retained.map((transition) => transition.sourceGroupId))].sort(compareText)
  const runKeys = [...new Set(retained.map((transition) => transition.episodeId))].sort(compareText)
  const parentFingerprints = [...new Set(parentHeads.map((head) => head.fingerprint))].sort(compareText)
  const revision = parentHeads.length ? Math.max(...parentHeads.map((head) => head.revision)) + 1 : 0
  const payload = {
    format: PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
    version: 1,
    representationSpaceId: String(input.representationSpaceId),
    geometryFingerprint: ADAPTIVE_REGION_GEOMETRY.fingerprint,
    baseAtlasFingerprint: String(input.baseAtlasFingerprint),
    authority: input.authority === 'cas-active-base' ? 'cas-active-base' : 'research-only',
    revision,
    parentFingerprints,
    maximumObservations,
    observationCount: observations.length,
    transitionCount: retained.length,
    sourceGroupIds,
    runKeys,
    duplicateTransitionCount: Math.max(0, complete.length - unique.length),
    incompleteTransitionCount,
    compactedTransitionCount: Math.max(0, unique.length - retained.length),
    observations,
  }
  if (!payload.representationSpaceId.trim() || !payload.baseAtlasFingerprint.trim()) {
    throw evidenceError('INVALID_PLASTIC_REGION_EVIDENCE', 'Birikimli kanıt temsil ve atlas soyu ister.')
  }
  return Object.freeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-evidence-head-v1', payload),
  })
}

export function validatePlasticRegionEvidenceHead(value) {
  if (!value || typeof value !== 'object' || value.format !== PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND ||
      value.version !== 1 || !['cas-active-base', 'research-only'].includes(value.authority) ||
      !Number.isSafeInteger(value.revision) || value.revision < 0 || !Array.isArray(value.parentFingerprints) ||
      !Array.isArray(value.sourceGroupIds) || !Array.isArray(value.runKeys)) {
    throw evidenceError('INVALID_PLASTIC_REGION_EVIDENCE', 'Birikimli kanıt başlığı biçimi geçersiz.')
  }
  if (value.geometryFingerprint !== ADAPTIVE_REGION_GEOMETRY.fingerprint ||
      !String(value.representationSpaceId ?? '').trim() || !String(value.baseAtlasFingerprint ?? '').trim()) {
    throw evidenceError('INVALID_PLASTIC_REGION_EVIDENCE', 'Birikimli kanıt geometri/soy bağı geçersiz.')
  }
  validateObservationSet(value.observations)
  const rebuilt = createPlasticRegionEvidenceHead({
    representationSpaceId: value.representationSpaceId,
    baseAtlasFingerprint: value.baseAtlasFingerprint,
    authority: value.authority,
    maximumObservations: value.maximumObservations,
    observations: value.observations,
    parentHeads: [],
  })
  const expectedPayload = {
    ...rebuilt,
    revision: value.revision,
    parentFingerprints: [...value.parentFingerprints],
    duplicateTransitionCount: value.duplicateTransitionCount,
    incompleteTransitionCount: value.incompleteTransitionCount,
    compactedTransitionCount: value.compactedTransitionCount,
  }
  const { fingerprint: _expectedFingerprint, ...payload } = expectedPayload
  const expectedFingerprint = stableRuntimeHash('plastic-region-evidence-head-v1', payload)
  if (canonicalJson({ ...value, fingerprint: expectedFingerprint }) !== canonicalJson({ ...payload, fingerprint: expectedFingerprint }) ||
      value.fingerprint !== expectedFingerprint) {
    throw evidenceError('INVALID_PLASTIC_REGION_EVIDENCE', 'Birikimli kanıt sayaçları veya fingerprinti içerikle uyuşmuyor.')
  }
  return value
}

function plasticRegionEvidenceReadError(message, details) {
  return evidenceError('PLASTIC_REGION_EVIDENCE_READ_FAILED', message, details)
}

function isSnapshotAncestor(store, ancestorId, descendantId) {
  const seen = new Set()
  let cursor = descendantId ? store.getSnapshot?.(descendantId) : null
  while (cursor && !seen.has(cursor.id)) {
    if (cursor.id === ancestorId) return true
    seen.add(cursor.id)
    cursor = cursor.parentId ? store.getSnapshot?.(cursor.parentId) : null
  }
  return false
}

function candidateProducerCompleted(store, snapshot) {
  if (!snapshot?.producerJobId || !Number.isSafeInteger(snapshot.producerAttempt) || snapshot.producerAttempt < 1) {
    return false
  }
  const job = store.getJob?.(snapshot.producerJobId)
  if (!job || job.state !== 'succeeded' || job.attempt !== snapshot.producerAttempt) return false
  return job.result?.snapshotId === snapshot.id || job.result?.lastSnapshotId === snapshot.id
}

function evidenceJournalArtifactIds(store, authority, expectedBase, maximumSnapshotScan = 4_096) {
  const candidates = [...(store.listSnapshots?.() ?? [])]
    .sort((left, right) => compareText(right.createdAt, left.createdAt) || compareText(right.id, left.id))
    .slice(0, maximumSnapshotScan)
    .filter((snapshot) =>
      snapshot.status === 'candidate' && snapshot.evaluation?.passed === true &&
      snapshot.evaluation?.runtimeVerified === true && candidateProducerCompleted(store, snapshot))
    .filter((snapshot) => authority === 'cas-active-base'
      ? snapshot.parentId === expectedBase?.snapshotId &&
        (!expectedBase?.atlasArtifactId || (snapshot.artifactIds ?? []).includes(expectedBase.atlasArtifactId))
      : snapshot.parentId === null)
  return new Set(candidates.flatMap((snapshot) => snapshot.artifactIds ?? []))
}

function casEvidenceBaseMatches(store, record, artifact, value, expectedBase, attachedToCurrent, journaled) {
  const envelope = artifact.payload?.metadata
  const originAtlasMetadata = envelope?.baseAtlasArtifactId
    ? store.getArtifactMetadata?.(envelope.baseAtlasArtifactId)
    : null
  const originBase = {
    snapshotId: envelope?.baseSnapshotId ?? null,
    atlasArtifactId: envelope?.baseAtlasArtifactId ?? null,
    atlasContentHash: originAtlasMetadata?.contentHash ?? null,
    atlasFingerprint: envelope?.casBaseAtlasFingerprint ?? null,
  }
  const canonicalContentHash = stableRuntimeHash(
    'plastic-region-pipeline-v1',
    'base-bound-artifact',
    PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
    originBase,
    value,
  )
  if (!expectedBase?.snapshotId ||
      record.contentHash !== canonicalContentHash ||
      (envelope?.baseAtlasArtifactId ?? null) !== (expectedBase.atlasArtifactId ?? null) ||
      (envelope?.casBaseAtlasFingerprint ?? null) !== (expectedBase.atlasFingerprint ?? null)) return false
  if (envelope.baseSnapshotId === expectedBase.snapshotId) return attachedToCurrent || journaled

  // A head created under an earlier active snapshot may be carried through an
  // immutable snapshot chain while the exact atlas artifact remains unchanged.
  // It is trusted only when it is explicitly attached to the current active
  // snapshot and its original CAS base is a real active/superseded ancestor.
  if (!attachedToCurrent || typeof envelope.baseSnapshotId !== 'string' ||
      !isSnapshotAncestor(store, envelope.baseSnapshotId, expectedBase.snapshotId)) return false
  const origin = store.getSnapshot?.(envelope.baseSnapshotId)
  if (!origin || !['active', 'superseded'].includes(origin.status) ||
      !(origin.artifactIds ?? []).includes(envelope.baseAtlasArtifactId)) return false
  const current = store.getSnapshot?.(expectedBase.snapshotId)
  return Boolean(current && (current.artifactIds ?? []).includes(record.id))
}

/**
 * Returns validated immutable evidence records. The reader hydrates only a
 * bounded number of bytes and returns lineage leaves, so a cumulative head does
 * not repeatedly pull all of its already folded ancestors back into memory.
 */
export function readPlasticRegionEvidenceHeadRecords(store, {
  representationSpaceId,
  baseAtlasFingerprint,
  authority = 'research-only',
  expectedBase = null,
  maximumHeads = 64,
  maximumArtifactBytes = 4 * 1024 * 1024,
  maximumTotalBytes = Math.max(16 * 1024 * 1024, maximumArtifactBytes),
} = {}) {
  if (!['cas-active-base', 'research-only'].includes(authority)) {
    throw evidenceError('INVALID_PLASTIC_REGION_EVIDENCE', 'Birikimli kanıt okuma otoritesi geçersiz.')
  }
  if (!Number.isSafeInteger(maximumHeads) || maximumHeads < 1 ||
      !Number.isSafeInteger(maximumArtifactBytes) || maximumArtifactBytes < 1 ||
      !Number.isSafeInteger(maximumTotalBytes) || maximumTotalBytes < maximumArtifactBytes) {
    throw evidenceError(
      'INVALID_PLASTIC_REGION_EVIDENCE',
      'Kanıt okuma bütçesi pozitif olmalı ve toplam bütçe tek-artifact bütçesinden küçük olmamalı.',
    )
  }
  const currentSnapshot = authority === 'cas-active-base' && expectedBase?.snapshotId
    ? store.getSnapshot?.(expectedBase.snapshotId)
    : null
  const snapshotArtifactIds = currentSnapshot?.artifactIds ?? []
  const maximumAttachedArtifactIds = Math.min(16_384, Math.max(1_024, maximumHeads * 64))
  if (snapshotArtifactIds.length > maximumAttachedArtifactIds) {
    throw plasticRegionEvidenceReadError(
      'Etkin snapshot birikimli kanıt için güvenli ekli-artifact tarama sınırını aşıyor.',
      { artifactCount: snapshotArtifactIds.length, maximumAttachedArtifactIds },
    )
  }
  const attachedIds = new Set(snapshotArtifactIds)
  const journaledIds = evidenceJournalArtifactIds(store, authority, expectedBase)
  if (typeof store.listArtifactMetadataByKind !== 'function') {
    throw evidenceError(
      'STORE_ARTIFACT_KIND_INDEX_MISSING',
      'Birikimli kanıt okuyucusu sınırlı artifact-türü indeksini gerektirir; global artifact taraması yasaktır.',
    )
  }
  const maximumMetadataRecords = Math.min(4_096, Math.max(64, maximumHeads * 8))
  const indexedMetadata = store.listArtifactMetadataByKind(
    PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
    { limit: maximumMetadataRecords },
  )
    .filter((record) => attachedIds.has(record.id) || journaledIds.has(record.id))
  // Attached evidence is authoritative even if a busy detached research lane
  // has pushed it beyond the bounded newest-by-kind window. Resolve attached
  // ids directly from the active snapshot instead of widening to a global scan.
  const attachedMetadata = snapshotArtifactIds
    .map((artifactId) => store.getArtifactMetadata?.(artifactId))
    .filter((record) => record?.kind === PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND)
  const metadata = [...new Map(
    [...indexedMetadata, ...attachedMetadata].map((record) => [record.id, record]),
  ).values()]
    .sort((left, right) =>
      Number(attachedIds.has(right.id)) - Number(attachedIds.has(left.id)) ||
      Number(journaledIds.has(right.id)) - Number(journaledIds.has(left.id)) ||
      compareText(right.createdAt, left.createdAt) || compareText(right.id, left.id))
  const attachedEvidenceCount = metadata.filter((record) => attachedIds.has(record.id)).length
  if (attachedEvidenceCount > maximumHeads) {
    throw plasticRegionEvidenceReadError(
      'Etkin snapshot, izin verilen başlık sayısından fazla birikimli kanıt başlığı taşıyor.',
      { attachedEvidenceCount, maximumHeads },
    )
  }
  const result = []
  const seen = new Set()
  let hydratedBytes = 0
  for (const record of metadata) {
    if (result.length >= maximumHeads) break
    const bytes = Number(record.payloadBytes ?? 0)
    const attachedToCurrent = attachedIds.has(record.id)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > maximumArtifactBytes ||
        hydratedBytes + bytes > maximumTotalBytes) {
      if (attachedToCurrent) {
        throw plasticRegionEvidenceReadError(
          'Etkin snapshot’a bağlı birikimli kanıt başlığı okuma bütçesini aşıyor veya boyutu geçersiz.',
          { artifactId: record.id, bytes, maximumArtifactBytes, maximumTotalBytes },
        )
      }
      continue
    }
    hydratedBytes += bytes
    try {
      const artifact = store.getArtifact(record.id)
      const value = validatePlasticRegionEvidenceHead(restoreArtifact(artifact, PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND))
      const envelope = artifact.payload?.metadata
      const envelopeMismatch = value.representationSpaceId !== representationSpaceId ||
          value.baseAtlasFingerprint !== baseAtlasFingerprint || value.authority !== authority ||
          envelope?.representationSpaceId !== value.representationSpaceId ||
          envelope?.geometryFingerprint !== value.geometryFingerprint ||
          envelope?.baseAtlasFingerprint !== value.baseAtlasFingerprint ||
          envelope?.authority !== value.authority || envelope?.revision !== value.revision ||
          envelope?.fingerprint !== value.fingerprint
      if (envelopeMismatch) {
        if (attachedToCurrent) {
          throw plasticRegionEvidenceReadError(
            'Etkin snapshot’a bağlı birikimli kanıtın metadata veya temsil soyu içerikle uyuşmuyor.',
            { artifactId: record.id },
          )
        }
        continue
      }
      if (authority === 'cas-active-base') {
        if (!casEvidenceBaseMatches(
          store,
          record,
          artifact,
          value,
          expectedBase,
          attachedToCurrent,
          journaledIds.has(record.id),
        )) {
          if (attachedToCurrent) {
            throw plasticRegionEvidenceReadError(
              'Etkin snapshot’a bağlı birikimli kanıt farklı bir CAS/atlas soyuna ait.',
              { artifactId: record.id },
            )
          }
          continue
        }
      } else if (envelope.baseSnapshotId !== null || envelope.baseAtlasArtifactId !== null) continue
      if (seen.has(value.fingerprint)) continue
      seen.add(value.fingerprint)
      result.push({ value, artifact, metadata: record })
    } catch (error) {
      if (attachedToCurrent) {
        if (error?.code === 'PLASTIC_REGION_EVIDENCE_READ_FAILED') throw error
        throw plasticRegionEvidenceReadError(
          'Etkin snapshot’a bağlı birikimli kanıt doğrulanamadı.',
          { artifactId: record.id, cause: String(error?.message ?? error) },
        )
      }
      // Corrupt/stale evidence is quarantined by omission and never folded
      // into a later proposal.
    }
  }
  const parentFingerprints = new Set(result.flatMap(({ value }) => value.parentFingerprints))
  return result.filter(({ value }) => !parentFingerprints.has(value.fingerprint))
}

export function readPlasticRegionEvidenceHeads(store, options = {}) {
  return readPlasticRegionEvidenceHeadRecords(store, options).map(({ value }) => value)
}
