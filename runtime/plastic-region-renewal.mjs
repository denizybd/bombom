import {
  PAGE_BEHAVIOR_DIMENSION,
  canonicalRegionFloat,
  validateAdaptiveRegionAtlas,
} from '../src/adaptive-regions/index.ts'
import { restoreArtifact, stableRuntimeHash } from './artifacts.mjs'
import {
  derivePlasticRegionReleaseMetrics,
  plasticRegionReleasePolicy,
  verifyPlasticRegionReleaseChain,
} from './plastic-region-release.mjs'
import {
  PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
  PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SEEDS,
  PLASTIC_REGION_RELEASE_SOURCE_POLICY,
  plasticRegionReleaseCases,
  plasticRegionReleaseCorpusFingerprint,
} from './plastic-region-release-spec.mjs'

export const PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND = 'plastic-region-release-renewal-evidence'
export const PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND = 'plastic-region-release-renewal-manifest'
export const PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND = 'plastic-region-release-renewal-audit'
export const PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT = stableRuntimeHash(
  'plastic-region-release-renewal-runner-v1',
  PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
)
const MAXIMUM_RENEWAL_ANCESTRY_DEPTH = 64

function renewalError(code, message, details) {
  const error = new Error(message)
  error.code = code
  if (details !== undefined) error.details = details
  return error
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value)
}

function text(value, label) {
  const result = typeof value === 'string' ? value.trim() : ''
  if (!result || result.length > 256) throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', `${label} geçersiz.`)
  return result
}

function nullableText(value, label) {
  return value === null || value === undefined ? null : text(value, label)
}

function integer(value, minimum, maximum, label) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', `${label} sınır dışında.`)
  }
  return value
}

function finite(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', `${label} sınır dışında.`)
  }
  return canonicalRegionFloat(value)
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (isRecord(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function sameCanonical(left, right) {
  return canonicalJson(left) === canonicalJson(right)
}

function runtimePolicy(value) {
  if (!isRecord(value)) throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal runtime politikası eksik.')
  return {
    compatibilityBlend: finite(value.compatibilityBlend, Number.EPSILON, 0.5, 'Renewal uyum harmanı'),
    minimumTrace: finite(value.minimumTrace, Number.EPSILON, 1, 'Renewal asgari izi'),
    maximumActiveGates: integer(value.maximumActiveGates, 1, 56, 'Renewal etkin kapı sınırı'),
  }
}

function artifactReference(value, label) {
  if (!isRecord(value)) throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', `${label} bağı eksik.`)
  return {
    artifactId: text(value.artifactId, `${label} artifact kimliği`),
    contentHash: text(value.contentHash, `${label} içerik özeti`),
    fingerprint: text(value.fingerprint, `${label} fingerprinti`),
  }
}

function baseBinding(value) {
  if (!isRecord(value)) throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal base bağı eksik.')
  return {
    activeSnapshotId: text(value.activeSnapshotId, 'Etkin base snapshot'),
    activeReleaseManifestArtifactId: text(value.activeReleaseManifestArtifactId, 'Etkin release manifest artifacti'),
    activeReleaseManifestContentHash: text(value.activeReleaseManifestContentHash, 'Etkin release manifest içerik özeti'),
    activeReleaseFingerprint: text(value.activeReleaseFingerprint, 'Etkin release fingerprinti'),
    sourceSnapshotId: text(value.sourceSnapshotId, 'Kaynak aday snapshotı'),
    sourceProducerJobId: text(value.sourceProducerJobId, 'Kaynak üretici işi'),
    sourceInputManifestHash: text(value.sourceInputManifestHash, 'Kaynak girdi manifesti'),
    sourceArtifactManifestHash: text(value.sourceArtifactManifestHash, 'Kaynak artifact manifesti'),
    sourceBaseModelFingerprint: text(value.sourceBaseModelFingerprint, 'Kaynak model fingerprinti'),
    sourceFamilyRunFingerprint: text(value.sourceFamilyRunFingerprint, 'Kaynak FamilyRun fingerprinti'),
    sourceLanguageStateFingerprint: nullableText(value.sourceLanguageStateFingerprint, 'Kaynak dil durumu fingerprinti'),
    sourceThoughtHeadFingerprint: text(value.sourceThoughtHeadFingerprint, 'Kaynak düşünce başlığı fingerprinti'),
    sourceThoughtHeadRevision: integer(value.sourceThoughtHeadRevision, 0, Number.MAX_SAFE_INTEGER, 'Kaynak düşünce başlığı revizyonu'),
    sourceConceptMacroHeadFingerprint: nullableText(value.sourceConceptMacroHeadFingerprint, 'Kaynak kavram başlığı fingerprinti'),
    representationProfile: text(value.representationProfile, 'Kaynak temsil profili'),
    atlasArtifactId: text(value.atlasArtifactId, 'Atlas artifact kimliği'),
    atlasContentHash: text(value.atlasContentHash, 'Atlas içerik özeti'),
    atlasFingerprint: text(value.atlasFingerprint, 'Atlas fingerprinti'),
    atlasRevision: integer(value.atlasRevision, 0, Number.MAX_SAFE_INTEGER, 'Atlas revizyonu'),
  }
}

function canonicalRow(value) {
  if (!isRecord(value)) throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal benchmark satırı geçersiz.')
  return {
    caseId: text(value.caseId, 'Renewal vaka kimliği'),
    sourceGroupId: text(value.sourceGroupId, 'Renewal kaynak grubu'),
    seed: integer(value.seed, 0, 0xffff_ffff, 'Renewal seed'),
    baseScore: finite(value.baseScore, 0, 1, 'Renewal base skoru'),
    candidateScore: finite(value.candidateScore, 0, 1, 'Renewal aday skoru'),
    shamScore: finite(value.shamScore, 0, 1, 'Renewal sham skoru'),
  }
}

function canonicalMetrics(value) {
  if (!isRecord(value)) throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal metrikleri eksik.')
  const pair = (entry, label) => ({
    caseCount: integer(entry?.caseCount, 1, 1_000_000, `${label} vaka sayısı`),
    independentSourceGroupCount: integer(entry?.independentSourceGroupCount, 1, 100_000, `${label} kaynak sayısı`),
    observedGain: finite(entry?.observedGain, -1, 1, `${label} kazancı`),
    regressionLoss: finite(entry?.regressionLoss, 0, 1, `${label} regresyonu`),
  })
  return { holdout: pair(value.holdout, 'Holdout'), sham: pair(value.sham, 'Sham') }
}

export function createPlasticRegionRenewalEvidence(input) {
  const sourceGroupIds = [...new Set((input.sourceGroupIds ?? []).map((id) => text(id, 'Renewal kaynak grubu')))].sort()
  if (!sourceGroupIds.length || sourceGroupIds.some((id) => id.toLowerCase().startsWith('ollama:') || id.toLowerCase().startsWith('ollama-'))) {
    throw renewalError('OLLAMA_RELEASE_EVIDENCE_FORBIDDEN', 'Renewal yalnız dondurulmuş Ollama-dışı holdout kaynağı kabul eder.')
  }
  const rows = (input.rows ?? []).map(canonicalRow)
  const payload = {
    format: PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND,
    version: 1,
    renewalId: text(input.renewalId, 'Renewal kimliği'),
    representationSpaceId: text(input.representationSpaceId, 'Renewal temsil uzayı'),
    geometryFingerprint: text(input.geometryFingerprint, 'Renewal geometri fingerprinti'),
    runnerFingerprint: text(input.runnerFingerprint, 'Renewal runner fingerprinti'),
    corpusFingerprint: text(input.corpusFingerprint, 'Renewal corpus fingerprinti'),
    engineFingerprint: text(input.engineFingerprint, 'Renewal engine fingerprinti'),
    engineContractFingerprint: text(input.engineContractFingerprint, 'Renewal engine sözleşmesi'),
    scorerFingerprint: text(input.scorerFingerprint, 'Renewal puanlayıcı fingerprinti'),
    pairingFingerprint: text(input.pairingFingerprint, 'Renewal eşleme fingerprinti'),
    sourcePolicy: PLASTIC_REGION_RELEASE_SOURCE_POLICY,
    usesOllama: false,
    automaticActivation: false,
    baseBinding: baseBinding(input.baseBinding),
    runtimePolicy: runtimePolicy(input.runtimePolicy),
    sourceGroupIds,
    seeds: [...(input.seeds ?? [])].map((seed) => integer(seed, 0, 0xffff_ffff, 'Renewal seed')),
    maximumTicks: integer(input.maximumTicks, 1, 64, 'Renewal tick sınırı'),
    candidateAtlasFingerprint: text(input.candidateAtlasFingerprint, 'Renewal aday atlası'),
    shamAtlasFingerprint: text(input.shamAtlasFingerprint, 'Renewal sham atlası'),
    rows,
    metrics: canonicalMetrics(input.metrics),
  }
  if (payload.runnerFingerprint !== PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT ||
      payload.engineContractFingerprint !== PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT ||
      payload.scorerFingerprint !== PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT ||
      payload.corpusFingerprint !== plasticRegionReleaseCorpusFingerprint(sourceGroupIds.length) ||
      !sameCanonical(payload.seeds, PLASTIC_REGION_RELEASE_SEEDS)) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal güvenilen runner/corpus/seed sözleşmesiyle uyuşmuyor.')
  }
  const expectedCases = plasticRegionReleaseCases(sourceGroupIds.length)
  const expectedGroups = expectedCases.map((item) => item.sourceGroupId).sort()
  const expectedPairs = new Set(expectedCases.flatMap((item) => payload.seeds.map((seed) => `${item.id}\u0000${item.sourceGroupId}\u0000${seed}`)))
  const actualPairs = new Set(rows.map((row) => `${row.caseId}\u0000${row.sourceGroupId}\u0000${row.seed}`))
  if (!sameCanonical(sourceGroupIds, expectedGroups) || rows.length !== expectedPairs.size ||
      actualPairs.size !== expectedPairs.size || [...actualPairs].some((key) => !expectedPairs.has(key))) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal ham vaka/seed matrisi eksik veya dondurulmuş corpus dışı.')
  }
  const derived = derivePlasticRegionReleaseMetrics({ rows, sourceGroupIds })
  if (!sameCanonical(payload.metrics, { holdout: derived.holdout, sham: derived.sham })) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal metrikleri ham vaka satırlarından yeniden üretilemiyor.')
  }
  return Object.freeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-release-renewal-evidence-v1', payload),
  })
}

export function validatePlasticRegionRenewalEvidence(value) {
  if (!isRecord(value) || value.format !== PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND || value.version !== 1) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal evidence biçimi geçersiz.')
  }
  const recreated = createPlasticRegionRenewalEvidence(value)
  if (value.fingerprint !== recreated.fingerprint || !sameCanonical(value, recreated)) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal evidence fingerprinti uyuşmuyor.')
  }
  return value
}

export function createPlasticRegionRenewalManifest(input) {
  const payload = {
    format: PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND,
    version: 1,
    releaseId: text(input.releaseId, 'Renewal release kimliği'),
    releaseKind: 'unchanged-atlas-model-renewal',
    representationSpaceId: text(input.representationSpaceId, 'Renewal temsil uzayı'),
    geometryFingerprint: text(input.geometryFingerprint, 'Renewal geometri fingerprinti'),
    automaticActivation: false,
    baseBinding: baseBinding(input.baseBinding),
    runtimePolicy: runtimePolicy(input.runtimePolicy),
    atlas: artifactReference(input.atlas, 'Renewal atlası'),
    evidence: artifactReference(input.evidence, 'Renewal evidence'),
  }
  return Object.freeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-release-renewal-manifest-v1', payload),
  })
}

export function validatePlasticRegionRenewalManifest(value) {
  if (!isRecord(value) || value.format !== PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND || value.version !== 1 ||
      value.releaseKind !== 'unchanged-atlas-model-renewal' || value.automaticActivation !== false) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal manifest biçimi geçersiz.')
  }
  const recreated = createPlasticRegionRenewalManifest(value)
  if (value.fingerprint !== recreated.fingerprint || !sameCanonical(value, recreated)) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal manifest fingerprinti uyuşmuyor.')
  }
  return value
}

export function createPlasticRegionRenewalAudit(input) {
  const payload = {
    format: PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND,
    version: 1,
    renewalId: text(input.renewalId, 'Renewal denetim kimliği'),
    representationSpaceId: text(input.representationSpaceId, 'Renewal denetim temsil uzayı'),
    decision: input.decision,
    reasonCodes: [...(input.reasonCodes ?? [])].map((code) => text(code, 'Renewal neden kodu')),
    baseSnapshotId: text(input.baseSnapshotId, 'Renewal base snapshotı'),
    sourceSnapshotId: text(input.sourceSnapshotId, 'Renewal kaynak snapshotı'),
    atlasFingerprint: text(input.atlasFingerprint, 'Renewal atlas fingerprinti'),
    evidence: input.evidence ?? null,
    releaseArtifactIds: [...(input.releaseArtifactIds ?? [])].map((id) => text(id, 'Renewal release artifacti')),
    runnerFingerprint: PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT,
    usesOllama: false,
    automaticActivation: false,
  }
  if (!['rejected', 'approved-awaiting-manual-activation'].includes(payload.decision)) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal denetim kararı geçersiz.')
  }
  return Object.freeze({ ...payload, fingerprint: stableRuntimeHash('plastic-region-release-renewal-audit-v1', payload) })
}

export function validatePlasticRegionRenewalAudit(value) {
  if (!isRecord(value) || value.format !== PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND || value.version !== 1) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal denetim biçimi geçersiz.')
  }
  const recreated = createPlasticRegionRenewalAudit(value)
  if (value.fingerprint !== recreated.fingerprint || !sameCanonical(value, recreated)) {
    throw renewalError('INVALID_PLASTIC_REGION_RENEWAL', 'Renewal denetim fingerprinti uyuşmuyor.')
  }
  return value
}

function latestAttached(store, snapshot, kind) {
  return (snapshot?.artifactIds ?? []).map((id) => store.getArtifactMetadata?.(id))
    .filter((entry) => entry?.kind === kind)
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))[0] ?? null
}

function snapshotArtifactManifest(store, snapshot) {
  const entries = (snapshot?.artifactIds ?? []).map((id) => {
    const metadata = store.getArtifactMetadata?.(id)
    if (!metadata) throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Kaynak snapshot artifact manifesti eksik.')
    return { id: metadata.id, kind: metadata.kind, contentHash: metadata.contentHash }
  }).sort((left, right) => left.id.localeCompare(right.id))
  return stableRuntimeHash('thought-base-artifacts-v1', entries)
}

function exactArtifact(store, attached, reference, kind) {
  if (!attached.has(reference.artifactId)) throw renewalError('PLASTIC_REGION_RENEWAL_STALE', `${kind} snapshotta bağlı değil.`)
  const metadata = store.getArtifactMetadata?.(reference.artifactId)
  const artifact = metadata ? store.getArtifact?.(reference.artifactId) : null
  if (!metadata || !artifact || metadata.kind !== kind || metadata.contentHash !== reference.contentHash) {
    throw renewalError('PLASTIC_REGION_RENEWAL_STALE', `${kind} immutable bağı değişmiş.`)
  }
  return { metadata, artifact }
}

function exactMetadata(artifact, expected) {
  const metadata = artifact?.payload?.metadata
  if (!isRecord(metadata)) throw renewalError('PLASTIC_REGION_RENEWAL_TAMPERED', 'Renewal artifact metadata’sı eksik.')
  const mismatch = Object.entries(expected).find(([field, value]) => metadata[field] !== value)
  if (mismatch) {
    throw renewalError('PLASTIC_REGION_RENEWAL_TAMPERED', `Renewal metadata ${mismatch[0]} alanı içerikle uyuşmuyor.`)
  }
}

function atlasRecord(store, snapshot) {
  const metadata = latestAttached(store, snapshot, 'plastic-region-atlas-head')
  if (!metadata) return null
  const artifact = store.getArtifact?.(metadata.id)
  const head = restoreArtifact(artifact, 'plastic-region-atlas-head')
  validateAdaptiveRegionAtlas(head)
  if (artifact.payload?.metadata?.fingerprint !== head.fingerprint ||
      artifact.payload?.metadata?.geometryFingerprint !== head.geometryFingerprint) {
    throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Atlas metadata/içerik bağı geçersiz.')
  }
  return { metadata, artifact, head, compatible: true, reason: null }
}

function priorRelease(store, snapshot, config, visited) {
  if (visited.size >= MAXIMUM_RENEWAL_ANCESTRY_DEPTH) {
    throw renewalError('PLASTIC_REGION_RENEWAL_DEPTH_EXCEEDED', 'Renewal release soyu güvenli doğrulama derinliğini aşıyor.')
  }
  if (visited.has(snapshot.id)) throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Renewal release soyu döngü içeriyor.')
  const nextVisited = new Set(visited).add(snapshot.id)
  const atlas = atlasRecord(store, snapshot)
  if (!atlas) return { compatible: false, reason: 'atlas-head-absent' }
  const renewal = latestAttached(store, snapshot, PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND)
  const lifecycle = latestAttached(store, snapshot, 'plastic-region-release-manifest')
  const renewalWins = renewal && (!lifecycle || String(renewal.createdAt).localeCompare(String(lifecycle.createdAt)) >= 0)
  return renewalWins
    ? verifyPlasticRegionRenewalChain(store, snapshot, atlas, config, nextVisited)
    : verifyPlasticRegionReleaseChain(store, snapshot, atlas, config)
}

export function verifyPlasticRegionRenewalChain(store, snapshot, currentAtlasRecord, config = {}, visited = new Set()) {
  const metadata = latestAttached(store, snapshot, PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND)
  if (!metadata) return { compatible: false, reason: 'renewal-manifest-absent', manifest: null, artifactIds: [] }
  try {
    const attached = new Set(snapshot?.artifactIds ?? [])
    const manifestArtifact = store.getArtifact(metadata.id)
    const manifest = validatePlasticRegionRenewalManifest(restoreArtifact(manifestArtifact, metadata.kind))
    exactMetadata(manifestArtifact, {
      representationSpaceId: manifest.representationSpaceId,
      fingerprint: manifest.fingerprint,
      releaseId: manifest.releaseId,
      releaseKind: manifest.releaseKind,
      activeBaseSnapshotId: manifest.baseBinding.activeSnapshotId,
      sourceSnapshotId: manifest.baseBinding.sourceSnapshotId,
      evidenceArtifactId: manifest.evidence.artifactId,
      atlasFingerprint: manifest.baseBinding.atlasFingerprint,
      automaticActivation: false,
    })
    const expectedPolicy = plasticRegionReleasePolicy(config)
    if (!sameCanonical(manifest.runtimePolicy, {
      compatibilityBlend: expectedPolicy.compatibilityBlend,
      minimumTrace: expectedPolicy.minimumTrace,
      maximumActiveGates: expectedPolicy.maximumActiveGates,
    })) throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Renewal deploy politikası güncel ayarlarla uyuşmuyor.')
    const binding = manifest.baseBinding
    if (snapshot?.parentId !== binding.activeSnapshotId) throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Renewal adayı ölçülen base’in doğrudan çocuğu değil.')
    const source = store.getSnapshot?.(binding.sourceSnapshotId)
    const producer = store.getJob?.(binding.sourceProducerJobId)
    const certified = new Set([
      producer?.result?.snapshotId, producer?.result?.lastSnapshotId,
      ...(producer?.result?.snapshotIds ?? []), ...(producer?.result?.trainingSnapshotIds ?? []),
    ].filter(Boolean))
    if (!source || source.status !== 'candidate' || source.parentId !== binding.activeSnapshotId ||
        source.producerJobId !== binding.sourceProducerJobId || source.inputManifestHash !== binding.sourceInputManifestHash ||
        snapshotArtifactManifest(store, source) !== binding.sourceArtifactManifestHash || producer?.state !== 'succeeded' ||
        !certified.has(source.id) || source.evaluation?.passed !== true || source.evaluation?.runtimeVerified !== true) {
      throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Renewal kaynak aday/üretici bağı geçersiz.')
    }
    if ((source.artifactIds ?? []).some((id) => !attached.has(id)) ||
        !sameCanonical(snapshot.sourceIds ?? [], source.sourceIds ?? []) ||
        !sameCanonical(snapshot.sourceRevisions ?? {}, source.sourceRevisions ?? {}) ||
        snapshot.inputManifestHash !== source.inputManifestHash) {
      throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Renewal adayı kaynak soyunu eksiksiz korumuyor.')
    }
    const activeBase = store.getSnapshot?.(binding.activeSnapshotId)
    const activeManifestMetadata = store.getArtifactMetadata?.(binding.activeReleaseManifestArtifactId)
    if (!activeBase || !(activeBase.artifactIds ?? []).includes(binding.activeReleaseManifestArtifactId) ||
        !activeManifestMetadata || activeManifestMetadata.contentHash !== binding.activeReleaseManifestContentHash) {
      throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Ölçülen etkin release immutable bağı geçersiz.')
    }
    const activeRelease = priorRelease(store, activeBase, config, visited)
    if (!activeRelease.compatible || activeRelease.manifest?.fingerprint !== binding.activeReleaseFingerprint) {
      throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Ölçülen etkin release zinciri artık doğrulanamıyor.')
    }
    const atlas = currentAtlasRecord?.head
    if (!atlas || currentAtlasRecord.metadata?.id !== binding.atlasArtifactId ||
        currentAtlasRecord.metadata?.contentHash !== binding.atlasContentHash || atlas.fingerprint !== binding.atlasFingerprint ||
        atlas.revision !== binding.atlasRevision || atlas.behaviorDimension !== PAGE_BEHAVIOR_DIMENSION ||
        atlas.representationSpaceId !== manifest.representationSpaceId || atlas.geometryFingerprint !== manifest.geometryFingerprint ||
        manifest.atlas.artifactId !== binding.atlasArtifactId || manifest.atlas.contentHash !== binding.atlasContentHash ||
        manifest.atlas.fingerprint !== binding.atlasFingerprint) {
      throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Renewal atlası ölçülen değişmez atlasla uyuşmuyor.')
    }
    const evidenceRecord = exactArtifact(store, attached, manifest.evidence, PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND)
    const evidence = validatePlasticRegionRenewalEvidence(restoreArtifact(evidenceRecord.artifact, evidenceRecord.metadata.kind))
    exactMetadata(evidenceRecord.artifact, {
      representationSpaceId: evidence.representationSpaceId,
      fingerprint: evidence.fingerprint,
      renewalId: evidence.renewalId,
      activeBaseSnapshotId: evidence.baseBinding.activeSnapshotId,
      sourceSnapshotId: evidence.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: evidence.baseBinding.sourceArtifactManifestHash,
      atlasFingerprint: evidence.baseBinding.atlasFingerprint,
      runnerFingerprint: evidence.runnerFingerprint,
      usesOllama: false,
      automaticActivation: false,
    })
    if (evidence.fingerprint !== manifest.evidence.fingerprint || !sameCanonical(evidence.baseBinding, binding) ||
        !sameCanonical(evidence.runtimePolicy, manifest.runtimePolicy) || evidence.candidateAtlasFingerprint !== atlas.fingerprint) {
      throw renewalError('PLASTIC_REGION_RENEWAL_STALE', 'Renewal evidence/manifest bağı geçersiz.')
    }
    const holdout = evidence.metrics.holdout
    const sham = evidence.metrics.sham
    if (holdout.caseCount < expectedPolicy.minimumCases ||
        holdout.independentSourceGroupCount < expectedPolicy.minimumSourceGroups ||
        holdout.observedGain < expectedPolicy.minimumHoldoutGain || holdout.regressionLoss > expectedPolicy.maximumRegressionLoss ||
        sham.caseCount < expectedPolicy.minimumCases || sham.independentSourceGroupCount < expectedPolicy.minimumSourceGroups ||
        Math.abs(sham.observedGain) > expectedPolicy.maximumShamGain || sham.regressionLoss > expectedPolicy.maximumRegressionLoss) {
      throw renewalError('PLASTIC_REGION_RENEWAL_GATE_FAILED', 'Renewal bağımsız holdout/sham kapısını geçmiyor.')
    }
    return {
      compatible: true,
      reason: null,
      manifest,
      manifestMetadata: metadata,
      evidence,
      artifactIds: [binding.atlasArtifactId, evidenceRecord.metadata.id, metadata.id],
      releaseKind: manifest.releaseKind,
    }
  } catch (error) {
    return {
      compatible: false,
      reason: String(error?.code ?? 'renewal-chain-invalid'),
      error,
      manifest: null,
      artifactIds: [],
    }
  }
}
