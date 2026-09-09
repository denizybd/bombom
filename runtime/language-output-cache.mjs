import { restoreArtifact } from './artifacts.mjs'

export const LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND = 'language-output-scaffold'
export const LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND = 'language-output-head'
export const DEFAULT_LANGUAGE_OUTPUT_HYDRATION_BYTES = 32 * 1024 * 1024

function artifactMetadata(store, snapshot, kind) {
  if (!snapshot) return null
  for (const artifactId of [...(snapshot.artifactIds ?? [])].reverse()) {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (metadata?.kind === kind) return metadata
  }
  return null
}

function restoreBounded(store, metadata, kind, maximumBytes) {
  if (!metadata) return { metadata: null, artifact: null, value: null, skippedHydration: false }
  if (Number(metadata.payloadBytes) > maximumBytes) {
    return { metadata, artifact: null, value: null, skippedHydration: true }
  }
  const artifact = store.getArtifact(metadata.id)
  if (!artifact || artifact.kind !== kind) {
    return { metadata, artifact: null, value: null, skippedHydration: false }
  }
  try {
    return { metadata, artifact, value: restoreArtifact(artifact, kind), skippedHydration: false }
  } catch {
    return { metadata, artifact, value: null, skippedHydration: false }
  }
}

/**
 * Reads the small, separate language-output controller artifacts without ever
 * opening a FamilyRun or runtime-model payload. A mismatch fails closed: the
 * manual scaffold may still be displayed from source, but its plastic head is
 * not allowed to route or label members.
 */
export function readLanguageOutputSnapshot(store, snapshot = store.getActiveSnapshot(), {
  maximumPayloadBytes = DEFAULT_LANGUAGE_OUTPUT_HYDRATION_BYTES,
} = {}) {
  const scaffoldRecord = restoreBounded(
    store,
    artifactMetadata(store, snapshot, LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND),
    LANGUAGE_OUTPUT_SCAFFOLD_ARTIFACT_KIND,
    maximumPayloadBytes,
  )
  const headRecord = restoreBounded(
    store,
    artifactMetadata(store, snapshot, LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND),
    LANGUAGE_OUTPUT_HEAD_ARTIFACT_KIND,
    maximumPayloadBytes,
  )
  const scaffold = scaffoldRecord.value?.format === 'language-output-scaffold' && scaffoldRecord.value?.version === 1
    ? scaffoldRecord.value
    : null
  const rawHead = headRecord.value?.format === 'language-output-head' && headRecord.value?.version === 1
    ? headRecord.value
    : null
  const compatible = Boolean(
    scaffold && rawHead &&
    rawHead.scaffoldFingerprint === scaffold.fingerprint,
  )
  return {
    snapshot,
    scaffold,
    head: compatible ? rawHead : null,
    compatible,
    scaffoldArtifact: scaffoldRecord.artifact,
    headArtifact: headRecord.artifact,
    scaffoldMetadata: scaffoldRecord.metadata,
    headMetadata: headRecord.metadata,
    skippedHydration: scaffoldRecord.skippedHydration || headRecord.skippedHydration,
    mismatch: Boolean(scaffold && rawHead && !compatible),
  }
}
