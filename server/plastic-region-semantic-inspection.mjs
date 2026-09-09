import { restoreArtifact } from '../runtime/artifacts.mjs'
import {
  SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND,
  SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND,
  SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND,
  createSemanticProbePublicSummary,
  validateSemanticProbeCalibration,
  validateSemanticProbeEvaluationEvidence,
  validateSemanticProbeObservationSet,
} from '../runtime/plastic-region-probes.mjs'

const KINDS = Object.freeze([
  SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND,
  SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND,
  SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND,
])

function inspectionError(code, message) {
  const error = new Error(message)
  error.code = code
  return error
}

function boundedInteger(value, fallback, minimum, maximum, label) {
  const result = value ?? fallback
  if (!Number.isSafeInteger(result) || result < minimum || result > maximum) {
    throw inspectionError('INVALID_SEMANTIC_PROBE_INSPECTION', `${label} ${minimum}..${maximum} aralığında tam sayı olmalı.`)
  }
  return result
}

function validator(kind) {
  if (kind === SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND) return validateSemanticProbeObservationSet
  if (kind === SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND) return validateSemanticProbeCalibration
  return validateSemanticProbeEvaluationEvidence
}

// Keep the HTTP reader independent from the heavy diagnostic producer (which
// imports the TypeScript benchmark engine). Historical empty caches are only
// recognized for audit and are never migrated into semantic evidence.
function inspectLegacyPlasticRegionProbeSkeleton(value) {
  const recognized = value?.format === 'plastic-region-probe-cache' && value?.version === 1 &&
    value?.trainingWritesAllowed === false && Array.isArray(value?.semanticAssignments) &&
    value.semanticAssignments.length === 0 && Array.isArray(value?.packs) &&
    value.packs.every((pack) => pack?.status === 'skeleton' && Array.isArray(pack.probes) &&
      pack.probes.length === 0 && Array.isArray(pack.learnedBindings) && pack.learnedBindings.length === 0)
  return { recognized }
}

function readKind(store, kind, options, budget) {
  const rows = store.listArtifactMetadataByKind(kind, { limit: options.maximumArtifactsPerKind })
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) ||
      String(right.id).localeCompare(String(left.id)))
  const values = []
  let rejected = 0
  let oversized = 0
  let skippedForBudget = 0
  for (const metadata of rows) {
    const bytes = Number(metadata.payloadBytes ?? 0)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > options.maximumPayloadBytes) {
      oversized += 1
      continue
    }
    if (budget.used + bytes > options.maximumTotalBytes) {
      skippedForBudget += 1
      continue
    }
    // Invalid artifacts still consume I/O and decode budget. Charge before
    // restore so a row of corrupt near-limit payloads cannot bypass the
    // bounded-reader contract.
    budget.used += bytes
    try {
      const artifact = store.getArtifact(metadata.id)
      const envelope = artifact?.payload?.metadata
      if (envelope?.trainingWritesAllowed !== false || envelope?.routingMutationAllowed !== false ||
          envelope?.releaseAuthority !== false ||
          (options.atlasFingerprint && envelope?.baseAtlasFingerprint !== options.atlasFingerprint)) {
        throw inspectionError('INVALID_SEMANTIC_PROBE_INSPECTION', 'Probe metadata salt-okunur exact atlas bağı taşımıyor.')
      }
      const value = restoreArtifact(artifact, kind)
      if (artifact?.contentHash !== value?.fingerprint || metadata?.contentHash !== value?.fingerprint) {
        throw inspectionError('INVALID_SEMANTIC_PROBE_INSPECTION', 'Probe store içerik özeti immutable artifact fingerprintiyle uyuşmuyor.')
      }
      validator(kind)(value)
      values.push(value)
    } catch {
      rejected += 1
    }
  }
  return { values, rejected, oversized, skippedForBudget, scanned: rows.length }
}

/**
 * Bounded, read-only server projection for UI. Raw observations, source
 * lineages and score matrices are validated but never returned.
 */
export function inspectSemanticProbeHypotheses(store, input = {}) {
  if (typeof store?.listArtifactMetadataByKind !== 'function' || typeof store?.getArtifact !== 'function') {
    throw inspectionError('SEMANTIC_PROBE_STORE_CONTRACT_MISSING', 'Probe denetimi kind-indexed artifact okuyucusu ister.')
  }
  const options = {
    atlasFingerprint: typeof input.atlasFingerprint === 'string' && input.atlasFingerprint.trim()
      ? input.atlasFingerprint.trim()
      : null,
    maximumItems: boundedInteger(input.maximumItems, 64, 1, 128, 'Probe özet sınırı'),
    maximumArtifactsPerKind: boundedInteger(input.maximumArtifactsPerKind, 64, 1, 128, 'Probe artifact tarama sınırı'),
    maximumPayloadBytes: boundedInteger(input.maximumPayloadBytes, 4 * 1024 * 1024, 1_024, 8 * 1024 * 1024, 'Probe payload sınırı'),
    maximumTotalBytes: boundedInteger(input.maximumTotalBytes, 16 * 1024 * 1024, 4_096, 32 * 1024 * 1024, 'Probe toplam okuma sınırı'),
  }
  const budget = { used: 0 }
  const loaded = Object.fromEntries(KINDS.map((kind) => [kind, readKind(store, kind, options, budget)]))
  const legacyRows = store.listArtifactMetadataByKind('plastic-region-probe-cache', {
    limit: options.maximumArtifactsPerKind,
  })
  let legacySkeletonAuditCount = 0
  for (const metadata of legacyRows) {
    const bytes = Number(metadata.payloadBytes ?? 0)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > options.maximumPayloadBytes ||
        budget.used + bytes > options.maximumTotalBytes) continue
    budget.used += bytes
    try {
      const value = restoreArtifact(store.getArtifact(metadata.id), 'plastic-region-probe-cache')
      if (inspectLegacyPlasticRegionProbeSkeleton(value).recognized) legacySkeletonAuditCount += 1
    } catch {
      // Historical corrupt/unknown cache values are neither migrated nor exposed.
    }
  }
  const summary = createSemanticProbePublicSummary({
    observations: loaded[SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND].values,
    calibrations: loaded[SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND].values,
    evaluations: loaded[SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND].values,
    maximumItems: options.maximumItems,
  })
  const counters = Object.fromEntries(KINDS.map((kind) => [kind, {
    scanned: loaded[kind].scanned,
    loaded: loaded[kind].values.length,
    rejected: loaded[kind].rejected,
    oversized: loaded[kind].oversized,
    skippedForBudget: loaded[kind].skippedForBudget,
  }]))
  return {
    format: 'plastic-region-semantic-probe-inspection',
    version: 1,
    layer: 'new-opt-in-interpretation',
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    releaseAuthority: false,
    available: summary.itemCount > 0,
    atlasFingerprint: options.atlasFingerprint,
    bytesRead: budget.used,
    legacySkeletons: {
      auditOnly: true,
      recognizedCount: legacySkeletonAuditCount,
      migrationAllowed: false,
    },
    artifacts: counters,
    summary,
  }
}
