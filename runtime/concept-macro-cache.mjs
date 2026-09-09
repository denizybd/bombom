import { restoreArtifact } from './artifacts.mjs'

export const CONCEPT_MACRO_HEAD_ARTIFACT_KIND = 'concept-macro-head'
export const CONCEPT_MACRO_OBSERVATION_ARTIFACT_KIND = 'concept-macro-observation-batch'
export const CONCEPT_MACRO_SHADOW_ARTIFACT_KIND = 'concept-macro-shadow-report'

function trustedConceptSnapshots(store) {
  const active = store.getActiveSnapshot?.()
  if (active) return [active]
  return [...(store.listSnapshots?.() ?? [])]
    .filter((snapshot) =>
      snapshot.status === 'candidate' &&
      snapshot.evaluation?.passed === true &&
      snapshot.evaluation?.runtimeVerified === true)
    .sort((left, right) =>
      String(right.createdAt).localeCompare(String(left.createdAt)) ||
      String(right.id).localeCompare(String(left.id)))
}

export function readLatestConceptMacroHead(store, representationSpaceId) {
  const snapshots = trustedConceptSnapshots(store)
  if (!snapshots.length) {
    return { head: null, artifact: null, quarantinedReason: 'no-trusted-snapshot-lineage' }
  }
  let firstProblem = null
  let attachedHeadCount = 0
  for (const snapshot of snapshots) {
    for (const artifactId of [...(snapshot.artifactIds ?? [])].reverse()) {
      const metadata = store.getArtifactMetadata?.(artifactId)
      if (metadata?.kind !== CONCEPT_MACRO_HEAD_ARTIFACT_KIND) continue
      attachedHeadCount += 1
      const artifact = store.getArtifact(artifactId)
      let head
      try {
        head = restoreArtifact(artifact, CONCEPT_MACRO_HEAD_ARTIFACT_KIND)
      } catch (error) {
        firstProblem ??= { artifact, reason: `head-decode:${String(error?.message ?? error).slice(0, 240)}` }
        continue
      }
      if (head?.representationSpaceId !== representationSpaceId) {
        firstProblem ??= { artifact, reason: `representation-space-mismatch:${head?.representationSpaceId ?? 'missing'}` }
        continue
      }
      return { head, artifact, quarantinedReason: null }
    }
  }
  return {
    head: null,
    artifact: firstProblem?.artifact ?? null,
    quarantinedReason: firstProblem?.reason ?? (
      attachedHeadCount ? 'no-compatible-concept-macro-head' : 'trusted-snapshot-has-no-concept-macro-head'
    ),
  }
}

export function listConceptMacroArtifacts(store) {
  const kinds = new Set([
    CONCEPT_MACRO_HEAD_ARTIFACT_KIND,
    CONCEPT_MACRO_OBSERVATION_ARTIFACT_KIND,
    CONCEPT_MACRO_SHADOW_ARTIFACT_KIND,
  ])
  return store.listArtifactMetadata()
    .filter((artifact) => kinds.has(artifact.kind))
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))
}
