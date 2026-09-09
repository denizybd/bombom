import { restoreArtifact, stableRuntimeHash } from './artifacts.mjs'

export const FAMILY_CONTINUITY_ARTIFACT_KIND = 'family-continuity-state'
export const DEFAULT_RUNTIME_ENVELOPE_HYDRATION_BYTES = 48 * 1024 * 1024

function activeArtifactMetadata(store, snapshot, kind) {
  if (!snapshot) return null
  for (const artifactId of [...snapshot.artifactIds].reverse()) {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (metadata?.kind === kind) return metadata
  }
  return null
}

/**
 * The previous FamilyRun can be hundreds of MiB. Family development only
 * needs its compact prototypes/lifecycle state and coalition continuity; it
 * must not hydrate the complete profile/transition graph beside the new
 * corpus. Old snapshots without this artifact safely start a fresh lifecycle
 * observation while preserving all source data and immutable snapshots.
 */
export function createFamilyContinuityState(run) {
  if (!run) return null
  return {
    format: 'family-continuity-state',
    version: 1,
    familyRunId: run.id,
    fingerprint: run.fingerprint,
    behaviorSpaceId: run.behaviorSpaceId,
    inputManifest: structuredClone(run.inputManifest),
    families: structuredClone(run.families ?? []),
    coalition: structuredClone(run.coalition ?? null),
  }
}

export function readActiveFamilyContinuity(store) {
  const snapshot = store.getActiveSnapshot()
  const metadata = activeArtifactMetadata(store, snapshot, FAMILY_CONTINUITY_ARTIFACT_KIND)
  if (!metadata) return null
  try {
    const artifact = store.getArtifact(metadata.id)
    const value = restoreArtifact(artifact, FAMILY_CONTINUITY_ARTIFACT_KIND)
    return value?.format === 'family-continuity-state' && value?.version === 1 ? value : null
  } catch {
    return null
  }
}

export function readActiveRuntimeContext(store, {
  hydrateFamilyRun = true,
  maximumRuntimePayloadBytes = Number.POSITIVE_INFINITY,
} = {}) {
  const snapshot = store.getActiveSnapshot()
  if (!snapshot) return { snapshot: null, artifact: null, model: null }
  for (const artifactId of [...snapshot.artifactIds].reverse()) {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (metadata?.kind && metadata.kind !== 'runtime-model-state') continue
    if (Number(metadata?.payloadBytes) > maximumRuntimePayloadBytes) {
      return { snapshot, artifact: null, artifactMetadata: metadata, model: null, skippedHydration: true }
    }
    const artifact = store.getArtifact(artifactId)
    if (artifact?.kind === 'runtime-model-state') {
      try {
        const restored = restoreArtifact(artifact, 'runtime-model-state')
        let model = restored
        if (hydrateFamilyRun && !restored.familyRun && restored.familyRunArtifactId) {
          const familyArtifact = store.getArtifact(restored.familyRunArtifactId)
          if (familyArtifact?.kind !== 'family-run') throw new TypeError('Runtime model FamilyRun başvurusu bulunamadı.')
          const familyRun = restoreArtifact(familyArtifact, 'family-run')
          if (restored.familyRunFingerprint && familyRun.fingerprint !== restored.familyRunFingerprint) {
            throw new TypeError('Runtime model FamilyRun parmak izi uyuşmuyor.')
          }
          model = { ...restored, familyRun }
        }
        return {
          snapshot,
          artifact,
          model,
        }
      } catch {
        return { snapshot, artifact: null, model: null }
      }
    }
  }
  return { snapshot, artifact: null, model: null }
}

export function readActiveRuntimeModel(store) {
  return readActiveRuntimeContext(store).model
}

export function readActiveRuntimeEnvelope(store, maximumRuntimePayloadBytes = DEFAULT_RUNTIME_ENVELOPE_HYDRATION_BYTES) {
  return readActiveRuntimeContext(store, {
    hydrateFamilyRun: false,
    maximumRuntimePayloadBytes,
  }).model
}

/**
 * Family learning is developmental: both the previous run and the branch on
 * which it was active can change the result. The active snapshot id is kept in
 * the lineage descriptor deliberately. A rollback creates a fresh snapshot,
 * so work cached on the abandoned branch cannot silently re-enter the new one.
 */
export function createFamilyModelLineage(store) {
  const snapshot = store.getActiveSnapshot()
  if (!snapshot) return null
  const runtimeMetadata = activeArtifactMetadata(store, snapshot, 'runtime-model-state')
  const familyMetadata = activeArtifactMetadata(store, snapshot, 'family-run')
  const continuity = readActiveFamilyContinuity(store)
  const runtimeEnvelope = readActiveRuntimeEnvelope(store)
  return {
    activeSnapshotId: snapshot.id,
    parentSnapshotId: snapshot.parentId ?? null,
    rollbackOfSnapshotId: snapshot.rollbackOf ?? null,
    inputManifestHash: snapshot.inputManifestHash ?? null,
    artifactIds: [...(snapshot.artifactIds ?? [])],
    runtimeModelArtifactId: runtimeMetadata?.id ?? null,
    runtimeModelContentHash: runtimeMetadata?.contentHash ?? null,
    familyRunArtifactId: familyMetadata?.id ?? null,
    familyRunContentHash: familyMetadata?.contentHash ?? null,
    runtimeEngineVersion: runtimeEnvelope?.engineVersion ?? null,
    runtimeCorpusFingerprint: runtimeEnvelope?.corpusFingerprint ?? null,
    familyRunFingerprint: continuity?.fingerprint ?? runtimeEnvelope?.familyRunFingerprint ?? runtimeEnvelope?.familyRun?.fingerprint ?? null,
    languageStateFingerprint: runtimeEnvelope?.languageState?.fingerprint ?? null,
  }
}

export function createFamilyRunCacheKey({
  runtimeVersion,
  analyses = [],
  synthesisDecisions = [],
  discoveryIndependenceGroupIds,
  evaluationIndependenceGroupIds,
  config,
  previousRun,
  lineage,
}) {
  const orderedAnalyses = [...analyses]
    .sort((left, right) => String(left.sourceDigest).localeCompare(String(right.sourceDigest)) || String(left.runId).localeCompare(String(right.runId)))
    .map((analysis) => [analysis.runId, analysis.sourceDigest, analysis.independenceGroupId ?? ''])
  const orderedDecisions = [...synthesisDecisions]
    .sort((left, right) => String(left.candidateFingerprint).localeCompare(String(right.candidateFingerprint)) || String(left.id).localeCompare(String(right.id)))
  return stableRuntimeHash(
    runtimeVersion,
    // v5 additionally isolates split-before-discovery runs. A cached model
    // learned from validation sources must never satisfy a train-only request.
    'family-run-cache-v5',
    orderedAnalyses,
    discoveryIndependenceGroupIds === undefined
      ? null
      : [...new Set(discoveryIndependenceGroupIds)].sort(),
    evaluationIndependenceGroupIds === undefined
      ? null
      : [...new Set(evaluationIndependenceGroupIds)].sort(),
    // Keep the complete reviewed decision record. Fields such as reviewedAt,
    // decision, support evidence and candidate vectors all reach FamilyRun
    // validation, its input manifest, or behavior-profile construction.
    orderedDecisions,
    config,
    previousRun ? stableRuntimeHash('previous-family-run-v1', previousRun) : null,
    lineage,
  )
}
