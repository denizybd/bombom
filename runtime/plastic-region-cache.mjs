import {
  encodeArtifactValue,
  encodedByteLength,
  findCachedArtifact,
  restoreArtifact,
  stableRuntimeHash,
} from './artifacts.mjs'
import {
  ADAPTIVE_REGION_GEOMETRY,
  PAGE_BEHAVIOR_DIMENSION,
  validateAdaptiveRegionAtlas,
  validateRegionLifecycleProposal,
} from '../src/adaptive-regions/index.ts'
import {
  PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
  validatePlasticRegionReleaseBenchmarkReport,
  validatePlasticRegionReleaseAudit,
  validatePlasticRegionReleaseEvidence,
  validatePlasticRegionReleaseManifest,
  verifyPlasticRegionReleaseChain,
} from './plastic-region-release.mjs'
import {
  PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND,
  validatePlasticRegionRenewalAudit,
  validatePlasticRegionRenewalEvidence,
  validatePlasticRegionRenewalManifest,
  verifyPlasticRegionRenewalChain,
} from './plastic-region-renewal.mjs'
import {
  PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
  readPlasticRegionEvidenceHeadRecords,
  validatePlasticRegionEvidenceHead,
} from './plastic-region-evidence.mjs'

export {
  PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
  verifyPlasticRegionReleaseChain,
} from './plastic-region-release.mjs'
export { PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND } from './plastic-region-evidence.mjs'
export {
  PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND,
  verifyPlasticRegionRenewalChain,
} from './plastic-region-renewal.mjs'

export const PLASTIC_REGION_OBSERVATION_ARTIFACT_KIND = 'plastic-region-observation-batch'
export const PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND = 'plastic-region-atlas-head'
export const PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND = 'plastic-region-atlas-proposal'
export const PLASTIC_REGION_SHADOW_ARTIFACT_KIND = 'plastic-region-shadow-report'
export const PLASTIC_REGION_PROBE_CACHE_ARTIFACT_KIND = 'plastic-region-probe-cache'

export const PLASTIC_REGION_ARTIFACT_KINDS = Object.freeze([
  PLASTIC_REGION_OBSERVATION_ARTIFACT_KIND,
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
  PLASTIC_REGION_SHADOW_ARTIFACT_KIND,
  PLASTIC_REGION_PROBE_CACHE_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
])

export const PLASTIC_REGION_MODES = Object.freeze(['off', 'shadow', 'proposal', 'active'])

const KIND_SET = new Set(PLASTIC_REGION_ARTIFACT_KINDS)
const DEFAULT_MAXIMUM_PAYLOAD_BYTES = 4 * 1024 * 1024
const VALUE_FORMAT_BY_KIND = Object.freeze({
  [PLASTIC_REGION_OBSERVATION_ARTIFACT_KIND]: PLASTIC_REGION_OBSERVATION_ARTIFACT_KIND,
  [PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND]: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  [PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND]: PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
  [PLASTIC_REGION_SHADOW_ARTIFACT_KIND]: PLASTIC_REGION_SHADOW_ARTIFACT_KIND,
  [PLASTIC_REGION_PROBE_CACHE_ARTIFACT_KIND]: PLASTIC_REGION_PROBE_CACHE_ARTIFACT_KIND,
  [PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND]: PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
  [PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND]: PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
  [PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND]: PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
  [PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND]: PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
  [PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND]: PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND,
  [PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND]: PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND,
  [PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND]: PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND,
  [PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND]: PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND,
})

function cacheError(code, message, details) {
  const error = new Error(message)
  error.code = code
  if (details !== undefined) error.details = details
  return error
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value)
}

function boundedString(value, maximum = 256) {
  return typeof value === 'string' && value.length > 0 && value.length <= maximum ? value : null
}

export function normalizePlasticRegionMode(value, fallback = 'off') {
  const mode = value ?? fallback
  if (!PLASTIC_REGION_MODES.includes(mode)) {
    throw cacheError('INVALID_PLASTIC_REGION_MODE', 'Plastik bölge kipi off, shadow, proposal veya active olmalı.')
  }
  return mode
}

export function assertPlasticRegionArtifactValue(kind, value) {
  if (!KIND_SET.has(kind)) {
    throw cacheError('INVALID_PLASTIC_REGION_ARTIFACT_KIND', `Bilinmeyen plastik bölge artifact türü: ${kind}`)
  }
  if (!isRecord(value) || value.format !== VALUE_FORMAT_BY_KIND[kind] || value.version !== 1) {
    throw cacheError(
      'INVALID_PLASTIC_REGION_ARTIFACT',
      `${kind} değeri eşleşen format ve version=1 taşımalıdır.`,
    )
  }
  const representationSpaceId = kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND
    ? value.proposedAtlas?.representationSpaceId
    : value.representationSpaceId
  if (!boundedString(representationSpaceId, 256)) {
    throw cacheError('INVALID_PLASTIC_REGION_ARTIFACT', `${kind} temsil uzayı kimliği taşımıyor.`)
  }
  if (kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND) {
    try {
      validateAdaptiveRegionAtlas(value)
      if (value.geometryFingerprint !== ADAPTIVE_REGION_GEOMETRY.fingerprint ||
          value.behaviorDimension !== PAGE_BEHAVIOR_DIMENSION) {
        throw new TypeError('Atlas canonical 8/56 geometri ve davranış boyutunda değil.')
      }
    } catch (cause) {
      throw cacheError('INVALID_PLASTIC_REGION_ATLAS', 'Plastik bölge atlası kanonik geometri/fingerprint doğrulamasını geçemedi.', {
        cause: String(cause?.message ?? cause),
      })
    }
  }
  if (kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND) {
    try {
      validateRegionLifecycleProposal(value)
    } catch (cause) {
      throw cacheError('INVALID_PLASTIC_REGION_PROPOSAL', 'Plastik bölge önerisi kanonik kanıt/fingerprint doğrulamasını geçemedi.', {
        cause: String(cause?.message ?? cause),
      })
    }
  }
  if (kind === PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND) validatePlasticRegionReleaseBenchmarkReport(value)
  if (kind === PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND) validatePlasticRegionReleaseAudit(value)
  if (kind === PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND) validatePlasticRegionReleaseEvidence(value)
  if (kind === PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND) validatePlasticRegionReleaseManifest(value)
  if (kind === PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND) validatePlasticRegionRenewalEvidence(value)
  if (kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND) validatePlasticRegionRenewalManifest(value)
  if (kind === PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND) validatePlasticRegionRenewalAudit(value)
  if (kind === PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND) validatePlasticRegionEvidenceHead(value)
  return value
}

/** Rejects a valid payload wrapped in stale or forged routing metadata. */
export function assertPlasticRegionArtifactMetadata(kind, value, artifact) {
  const metadata = artifact?.payload?.metadata
  if (!isRecord(metadata)) throw cacheError('INVALID_PLASTIC_REGION_METADATA', `${kind} metadata taşımıyor.`)
  const atlas = kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ? value.proposedAtlas : value
  const expected = {
    representationSpaceId: atlas.representationSpaceId,
    fingerprint: value.fingerprint,
  }
  if (kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND) {
    Object.assign(expected, {
      geometryFingerprint: value.geometryFingerprint,
      revision: value.revision,
      behaviorDimension: value.behaviorDimension,
      maximumMembershipsPerPage: value.maximumMembershipsPerPage,
    })
  }
  if (kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND) {
    Object.assign(expected, {
      geometryFingerprint: value.proposedAtlas.geometryFingerprint,
      revision: value.proposedAtlas.revision,
      proposalId: value.proposalId,
      baseAtlasFingerprint: value.baseAtlasFingerprint,
      proposedAtlasFingerprint: value.proposedAtlas.fingerprint,
      eligible: value.eligible,
    })
  }
  if (kind === PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND) {
    Object.assign(expected, {
      geometryFingerprint: value.geometryFingerprint,
      evaluationId: value.evaluationId,
      evaluationKind: value.evaluationKind,
      proposalFingerprint: value.proposalFingerprint,
      proposedAtlasFingerprint: value.proposedAtlasFingerprint,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
      benchmarkReportArtifactId: value.benchmarkReport.artifactId,
      benchmarkReportFingerprint: value.benchmarkReport.fingerprint,
    })
  }
  if (kind === PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND) {
    Object.assign(expected, {
      evaluationId: value.evaluationId,
      runnerFingerprint: value.runnerFingerprint,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
      originalProposalArtifactId: value.originalProposal.artifactId,
      usesOllama: false,
    })
  }
  if (kind === PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND) {
    Object.assign(expected, {
      evaluationId: value.evaluationId,
      decision: value.decision,
      baseSnapshotId: value.baseSnapshotId,
      sourceSnapshotId: value.sourceSnapshotId,
      originalProposalArtifactId: value.originalProposalArtifactId,
      originalProposalFingerprint: value.originalProposalFingerprint,
      usesOllama: false,
      automaticActivation: false,
    })
  }
  if (kind === PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND) {
    Object.assign(expected, {
      geometryFingerprint: value.geometryFingerprint,
      releaseId: value.releaseId,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
      benchmarkReportArtifactId: value.benchmarkReport.artifactId,
      benchmarkReportFingerprint: value.benchmarkReport.fingerprint,
    })
  }
  if (kind === PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND) {
    Object.assign(expected, {
      renewalId: value.renewalId,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
      atlasFingerprint: value.baseBinding.atlasFingerprint,
      runnerFingerprint: value.runnerFingerprint,
      usesOllama: false,
      automaticActivation: false,
    })
  }
  if (kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND) {
    Object.assign(expected, {
      releaseId: value.releaseId,
      releaseKind: value.releaseKind,
      activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
      sourceSnapshotId: value.baseBinding.sourceSnapshotId,
      evidenceArtifactId: value.evidence.artifactId,
      atlasFingerprint: value.baseBinding.atlasFingerprint,
      automaticActivation: false,
    })
  }
  if (kind === PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND) {
    Object.assign(expected, {
      renewalId: value.renewalId,
      decision: value.decision,
      baseSnapshotId: value.baseSnapshotId,
      sourceSnapshotId: value.sourceSnapshotId,
      atlasFingerprint: value.atlasFingerprint,
      usesOllama: false,
      automaticActivation: false,
    })
  }
  if (kind === PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND) {
    Object.assign(expected, {
      geometryFingerprint: value.geometryFingerprint,
      revision: value.revision,
      baseAtlasFingerprint: value.baseAtlasFingerprint,
      authority: value.authority,
    })
  }
  const mismatch = Object.entries(expected).find(([field, expectedValue]) => metadata[field] !== expectedValue)
  if (mismatch) {
    throw cacheError('INVALID_PLASTIC_REGION_METADATA', `${kind} metadata içeriğe bağlı değil.`, {
      field: mismatch[0], expected: mismatch[1], actual: metadata[mismatch[0]],
    })
  }
  return metadata
}

function attachedHeadMetadata(store, snapshot) {
  if (!snapshot) return null
  for (const artifactId of [...(snapshot.artifactIds ?? [])].reverse()) {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (metadata?.kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND) return metadata
  }
  return null
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

export function currentPlasticRegionBase(store) {
  const snapshot = store.getActiveSnapshot?.() ?? null
  const metadata = attachedHeadMetadata(store, snapshot)
  let fingerprint = null
  if (metadata) {
    const artifact = store.getArtifact(metadata.id)
    try {
      fingerprint = restoreArtifact(artifact, PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND)?.fingerprint ?? null
    } catch {
      fingerprint = null
    }
  }
  return {
    snapshotId: snapshot?.id ?? null,
    atlasArtifactId: metadata?.id ?? null,
    atlasContentHash: metadata?.contentHash ?? null,
    atlasFingerprint: fingerprint,
  }
}

export function assertPlasticRegionBase(store, expectedBase) {
  if (!isRecord(expectedBase) || !boundedString(expectedBase.snapshotId)) {
    throw cacheError('INVALID_PLASTIC_REGION_BASE', 'Plastik bölge yazımı açık bir base snapshot kimliği gerektirir.')
  }
  const current = currentPlasticRegionBase(store)
  const fields = [
    ['snapshotId', expectedBase.snapshotId],
    ['atlasArtifactId', expectedBase.atlasArtifactId ?? null],
    ['atlasContentHash', expectedBase.atlasContentHash ?? null],
    ['atlasFingerprint', expectedBase.atlasFingerprint ?? null],
  ]
  const mismatch = fields.find(([field, expected]) => current[field] !== expected)
  if (mismatch) {
    throw cacheError(
      'STALE_REGION_ATLAS_BASE',
      'Plastik bölge önerisinin dayandığı snapshot veya atlas başlığı artık güncel değil.',
      { field: mismatch[0], expected: mismatch[1], actual: current[mismatch[0]], current },
    )
  }
  return current
}

/**
 * Persists an immutable region artifact behind a snapshot/head CAS fence. It
 * deliberately does not attach the artifact to a snapshot and never changes
 * routing; candidate creation remains a separate, evaluated operation.
 */
export function persistPlasticRegionArtifact(context, {
  kind,
  value,
  expectedBase,
  contentHash,
  metadata = {},
}) {
  assertPlasticRegionArtifactValue(kind, value)
  const maximumBytes = Number(
    context.store.getConfig?.().plasticRegions?.maximumArtifactBytes ?? DEFAULT_MAXIMUM_PAYLOAD_BYTES,
  )
  const encoded = encodeArtifactValue(value)
  const bytes = encodedByteLength(encoded)
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1 || bytes > maximumBytes) {
    throw cacheError(
      'PLASTIC_REGION_ARTIFACT_TOO_LARGE',
      `${kind} plastik bölge artifacti ${maximumBytes} byte sınırını aşıyor.`,
      { bytes, maximumBytes },
    )
  }
  context.checkResources?.(`artifact-write:${kind}`)
  assertPlasticRegionBase(context.store, expectedBase)
  const resolvedHash = contentHash ?? stableRuntimeHash('plastic-region-artifact-v1', kind, encoded)
  const cached = findCachedArtifact(context.store, kind, resolvedHash, {
    repairIndex: !context.jobClaim,
  })
  if (cached) return { artifact: cached, value }

  const input = {
    jobId: context.jobId ?? null,
    kind,
    contentHash: resolvedHash,
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      bytes,
      metadata: {
        ...metadata,
        representationSpaceId: kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND
          ? value.proposedAtlas.representationSpaceId
          : value.representationSpaceId,
        fingerprint: value.fingerprint ?? resolvedHash,
        ...(kind === PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND ? {
          geometryFingerprint: value.geometryFingerprint,
          revision: value.revision,
          behaviorDimension: value.behaviorDimension,
          maximumMembershipsPerPage: value.maximumMembershipsPerPage,
        } : {}),
        ...(kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ? {
          geometryFingerprint: value.proposedAtlas.geometryFingerprint,
          revision: value.proposedAtlas.revision,
          proposalId: value.proposalId,
          baseAtlasFingerprint: value.baseAtlasFingerprint,
          proposedAtlasFingerprint: value.proposedAtlas.fingerprint,
          eligible: value.eligible,
        } : {}),
        ...(kind === PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND ? {
          geometryFingerprint: value.geometryFingerprint,
          evaluationId: value.evaluationId,
          evaluationKind: value.evaluationKind,
          proposalFingerprint: value.proposalFingerprint,
          proposedAtlasFingerprint: value.proposedAtlasFingerprint,
          activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
          sourceSnapshotId: value.baseBinding.sourceSnapshotId,
          sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
          benchmarkReportArtifactId: value.benchmarkReport.artifactId,
          benchmarkReportFingerprint: value.benchmarkReport.fingerprint,
        } : {}),
        ...(kind === PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND ? {
          evaluationId: value.evaluationId,
          runnerFingerprint: value.runnerFingerprint,
          activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
          sourceSnapshotId: value.baseBinding.sourceSnapshotId,
          sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
          originalProposalArtifactId: value.originalProposal.artifactId,
          usesOllama: false,
        } : {}),
        ...(kind === PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND ? {
          evaluationId: value.evaluationId,
          decision: value.decision,
          baseSnapshotId: value.baseSnapshotId,
          sourceSnapshotId: value.sourceSnapshotId,
          originalProposalArtifactId: value.originalProposalArtifactId,
          originalProposalFingerprint: value.originalProposalFingerprint,
          usesOllama: false,
          automaticActivation: false,
        } : {}),
        ...(kind === PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND ? {
          geometryFingerprint: value.geometryFingerprint,
          releaseId: value.releaseId,
          activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
          sourceSnapshotId: value.baseBinding.sourceSnapshotId,
          sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
          benchmarkReportArtifactId: value.benchmarkReport.artifactId,
          benchmarkReportFingerprint: value.benchmarkReport.fingerprint,
        } : {}),
        ...(kind === PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND ? {
          renewalId: value.renewalId,
          activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
          sourceSnapshotId: value.baseBinding.sourceSnapshotId,
          sourceArtifactManifestHash: value.baseBinding.sourceArtifactManifestHash,
          atlasFingerprint: value.baseBinding.atlasFingerprint,
          runnerFingerprint: value.runnerFingerprint,
          usesOllama: false,
          automaticActivation: false,
        } : {}),
        ...(kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND ? {
          releaseId: value.releaseId,
          releaseKind: value.releaseKind,
          activeBaseSnapshotId: value.baseBinding.activeSnapshotId,
          sourceSnapshotId: value.baseBinding.sourceSnapshotId,
          evidenceArtifactId: value.evidence.artifactId,
          atlasFingerprint: value.baseBinding.atlasFingerprint,
          automaticActivation: false,
        } : {}),
        ...(kind === PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND ? {
          renewalId: value.renewalId,
          decision: value.decision,
          baseSnapshotId: value.baseSnapshotId,
          sourceSnapshotId: value.sourceSnapshotId,
          atlasFingerprint: value.atlasFingerprint,
          usesOllama: false,
          automaticActivation: false,
        } : {}),
        ...(kind === PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND ? {
          geometryFingerprint: value.geometryFingerprint,
          revision: value.revision,
          baseAtlasFingerprint: value.baseAtlasFingerprint,
          authority: value.authority,
        } : {}),
        baseSnapshotId: expectedBase.snapshotId,
        baseAtlasArtifactId: expectedBase.atlasArtifactId ?? null,
        casBaseAtlasFingerprint: expectedBase.atlasFingerprint ?? null,
        baseAtlasFingerprint: kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND ||
          kind === PLASTIC_REGION_EVIDENCE_HEAD_ARTIFACT_KIND
          ? value.baseAtlasFingerprint
          : expectedBase.atlasFingerprint ?? null,
      },
      value: encoded,
    },
  }
  const casBase = {
    snapshotId: expectedBase.snapshotId,
    headKind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    headArtifactId: expectedBase.atlasArtifactId ?? null,
    headContentHash: expectedBase.atlasContentHash ?? null,
    headFingerprint: expectedBase.atlasFingerprint ?? null,
    staleCode: 'STALE_REGION_ATLAS_BASE',
  }
  let artifact
  if (context.jobClaim) {
    if (typeof context.store.createClaimedArtifactOnSnapshotHeadBase !== 'function') {
      throw cacheError('STORE_CAS_CONTRACT_MISSING', 'Store, claim korumalı plastik bölge CAS yazımını desteklemiyor.')
    }
    artifact = context.store.createClaimedArtifactOnSnapshotHeadBase(
      context.jobId,
      context.jobClaim.owner,
      context.jobClaim.attempt,
      input,
      casBase,
    )
  } else {
    if (typeof context.store.createArtifactOnSnapshotHeadBase !== 'function') {
      throw cacheError('STORE_CAS_CONTRACT_MISSING', 'Store, plastik bölge CAS yazımını desteklemiyor.')
    }
    artifact = context.store.createArtifactOnSnapshotHeadBase(input, casBase)
  }
  return { artifact, value }
}

export function readPlasticRegionSnapshot(store, snapshot = trustedSnapshot(store), {
  maximumPayloadBytes = Number(store.getConfig?.().plasticRegions?.maximumArtifactBytes ?? DEFAULT_MAXIMUM_PAYLOAD_BYTES),
} = {}) {
  const metadata = attachedHeadMetadata(store, snapshot)
  if (!metadata) {
    return { snapshot, metadata: null, artifact: null, head: null, compatible: false, reason: 'atlas-head-absent' }
  }
  if (Number(metadata.payloadBytes) > maximumPayloadBytes) {
    return { snapshot, metadata, artifact: null, head: null, compatible: false, reason: 'atlas-head-too-large' }
  }
  const artifact = store.getArtifact(metadata.id)
  try {
    const head = assertPlasticRegionArtifactValue(
      PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      restoreArtifact(artifact, PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND),
    )
    assertPlasticRegionArtifactMetadata(PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND, head, artifact)
    return { snapshot, metadata, artifact, head, compatible: true, reason: null }
  } catch (error) {
    return {
      snapshot,
      metadata,
      artifact: null,
      head: null,
      compatible: false,
      reason: `atlas-head-invalid:${String(error?.code ?? 'decode-failed')}`,
    }
  }
}

export function readPlasticRegionRelease(store, snapshot = trustedSnapshot(store), options = {}) {
  const { releasePolicy, ...readOptions } = options
  const atlasRecord = readPlasticRegionSnapshot(store, snapshot, readOptions)
  if (!atlasRecord.compatible) {
    return {
      compatible: false,
      reason: atlasRecord.reason,
      manifest: null,
      artifactIds: [],
      atlasRecord,
    }
  }
  const latestManifest = (snapshot?.artifactIds ?? [])
    .map((artifactId) => store.getArtifactMetadata?.(artifactId))
    .filter((metadata) => metadata && (
      metadata.kind === PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND ||
      metadata.kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND
    ))
    .sort((left, right) =>
      String(right.createdAt).localeCompare(String(left.createdAt)) ||
      Number(right.kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND) -
        Number(left.kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND) ||
      String(right.id).localeCompare(String(left.id)))[0] ?? null
  const policy = releasePolicy ?? store.getConfig?.().plasticRegions ?? {}
  return {
    ...(latestManifest?.kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND
      ? verifyPlasticRegionRenewalChain(store, snapshot, atlasRecord, policy)
      : verifyPlasticRegionReleaseChain(store, snapshot, atlasRecord, policy)),
    atlasRecord,
  }
}

export function plasticRegionCarryForwardArtifactIds(store, snapshot = store.getActiveSnapshot?.() ?? null) {
  const atlasRecord = readPlasticRegionSnapshot(store, snapshot)
  if (!atlasRecord.compatible) return []
  const release = readPlasticRegionRelease(store, snapshot)
  const artifactIds = release.compatible ? [...release.artifactIds] : [atlasRecord.metadata.id]
  const currentBase = currentPlasticRegionBase(store)
  if (snapshot?.id !== currentBase.snapshotId || atlasRecord.metadata.id !== currentBase.atlasArtifactId ||
      atlasRecord.head.fingerprint !== currentBase.atlasFingerprint) return artifactIds

  const config = store.getConfig?.().plasticRegions ?? {}
  const maximumArtifactBytes = Number(config.maximumArtifactBytes ?? DEFAULT_MAXIMUM_PAYLOAD_BYTES)
  const evidence = readPlasticRegionEvidenceHeadRecords(store, {
    representationSpaceId: atlasRecord.head.representationSpaceId,
    baseAtlasFingerprint: atlasRecord.head.fingerprint,
    authority: 'cas-active-base',
    expectedBase: currentBase,
    maximumHeads: Number(config.maximumEvidenceHeadsPerCycle ?? 64),
    maximumArtifactBytes,
    maximumTotalBytes: Math.max(16 * 1024 * 1024, maximumArtifactBytes),
  })
  for (const record of evidence) {
    if (!artifactIds.includes(record.metadata.id)) artifactIds.push(record.metadata.id)
  }
  return artifactIds
}

export function snapshotUsesAdaptiveRegionRepresentation(store, snapshot) {
  for (const artifactId of snapshot?.artifactIds ?? []) {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (!metadata) continue
    if (KIND_SET.has(metadata.kind)) return true
    if (metadata.kind !== 'text-lab-analysis') continue
    try {
      const artifact = store.getArtifact(artifactId)
      if (artifact?.payload?.metadata?.representationProfile === 'adaptive-eighth-id-v1') return true
      const analysis = restoreArtifact(artifact, 'text-lab-analysis')
      if (analysis?.config?.representationProfile === 'adaptive-eighth-id-v1') return true
    } catch {
      // An unreadable representation proof can never authorize activation.
      return true
    }
  }
  return false
}

export function plasticRegionSnapshotActivationGate(store, snapshot) {
  const required = snapshotUsesAdaptiveRegionRepresentation(store, snapshot)
  if (!required) return { required: false, eligible: true, reason: null, release: null }
  const release = readPlasticRegionRelease(store, snapshot)
  return {
    required: true,
    eligible: release.compatible,
    reason: release.compatible ? null : release.reason,
    release,
  }
}

export function listPlasticRegionArtifacts(store, { maximumPerKind = 512 } = {}) {
  if (!Number.isSafeInteger(maximumPerKind) || maximumPerKind < 1 || maximumPerKind > 4_096 ||
      typeof store.listArtifactMetadataByKind !== 'function') {
    throw cacheError(
      'INVALID_PLASTIC_REGION_ARTIFACT_QUERY',
      'Plastik bölge artifact sorgusu kind-indexed store ve 1..4096 sınırı ister.',
    )
  }
  return PLASTIC_REGION_ARTIFACT_KINDS
    .flatMap((kind) => store.listArtifactMetadataByKind(kind, { limit: maximumPerKind }))
    .sort((left, right) =>
      String(right.createdAt).localeCompare(String(left.createdAt)) ||
      String(right.id).localeCompare(String(left.id)))
}

export function isPlasticRegionArtifactKind(kind) {
  return KIND_SET.has(kind)
}
