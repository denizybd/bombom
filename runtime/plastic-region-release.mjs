import {
  PAGE_BEHAVIOR_DIMENSION,
  canonicalRegionFloat,
  validateAdaptiveRegionAtlas,
  validateRegionLifecycleProposal,
} from '../src/adaptive-regions/index.ts'
import { restoreArtifact, stableRuntimeHash } from './artifacts.mjs'
import {
  PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
  PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SOURCE_POLICY,
  plasticRegionReleaseCases,
  plasticRegionReleaseCorpusFingerprint,
} from './plastic-region-release-spec.mjs'

export const PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND = 'plastic-region-release-evidence'
export const PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND = 'plastic-region-release-manifest'
export const PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND = 'plastic-region-release-benchmark-report'
export const PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND = 'plastic-region-release-evaluation-audit'

const RELEASE_VERSION = 1
const EVALUATION_KINDS = new Set(['holdout', 'sham'])
const RELEASE_BOOTSTRAP_SAMPLES = 1_000
const RELEASE_BOOTSTRAP_GAIN_SEED = 0x6e91
const RELEASE_BOOTSTRAP_REGRESSION_SEED = 0x71e5

function releaseError(code, message, details) {
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
  if (!result || result.length > 256) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', `${label} geçersiz.`)
  return result
}

function nullableText(value, label) {
  if (value === null || value === undefined) return null
  return text(value, label)
}

function integer(value, minimum, maximum, label) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', `${label} sınır dışında.`)
  }
  return value
}

function finite(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', `${label} sınır dışında.`)
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

function sortedUniqueStrings(values, label) {
  if (!Array.isArray(values)) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', `${label} dizi olmalı.`)
  const normalized = values.map((value) => text(value, label)).sort((left, right) => left.localeCompare(right))
  if (!normalized.length || new Set(normalized).size !== normalized.length) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', `${label} boş veya yinelenen olamaz.`)
  }
  return normalized
}

function rejectsOllamaSource(sourceGroupIds) {
  if (sourceGroupIds.some((id) => {
    const normalized = id.toLowerCase()
    return normalized.startsWith('ollama:') || normalized.startsWith('ollama-')
  })) {
    throw releaseError(
      'OLLAMA_RELEASE_EVIDENCE_FORBIDDEN',
      'Ollama üretimi validation/test veya release kanıtı olarak kullanılamaz.',
    )
  }
}

export function validatePlasticRegionReleaseAudit(value) {
  if (!isRecord(value) || value.format !== PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND || value.version !== 1 ||
      value.runnerFingerprint !== PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT || value.deterministic !== true ||
      value.usesOllama !== false || value.automaticActivation !== false ||
      value.sourcePolicy !== PLASTIC_REGION_RELEASE_SOURCE_POLICY ||
      !['rejected-before-measurement', 'rejected', 'approved-awaiting-manual-activation'].includes(value.decision)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release değerlendirme denetim kaydı güvenilen runner sözleşmesine uymuyor.')
  }
  text(value.evaluationId, 'Release denetim kimliği')
  text(value.representationSpaceId, 'Release denetim temsil uzayı')
  text(value.baseSnapshotId, 'Release denetim base snapshotı')
  text(value.sourceSnapshotId, 'Release denetim kaynak snapshotı')
  text(value.originalProposalArtifactId, 'Release denetim proposal artifacti')
  text(value.originalProposalFingerprint, 'Release denetim proposal fingerprinti')
  text(value.baseAtlasFingerprint, 'Release denetim base atlas fingerprinti')
  integer(value.caseCount, 0, 1_000_000, 'Release denetim vaka sayısı')
  integer(value.maximumTicks, 1, 512, 'Release denetim tick sınırı')
  if (!Array.isArray(value.sourceGroupIds)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release denetim kaynak grupları dizi olmalı.')
  }
  const sourceGroupIds = value.sourceGroupIds.map((id) => text(id, 'Release denetim kaynak grubu'))
  if (new Set(sourceGroupIds).size !== sourceGroupIds.length) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release denetim kaynak grupları yinelenemez.')
  }
  rejectsOllamaSource(sourceGroupIds)
  if (!Array.isArray(value.seeds) || value.seeds.some((seed) => !Number.isSafeInteger(seed))) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release denetim seed listesi geçersiz.')
  }
  if (!Array.isArray(value.reasonCodes) || value.reasonCodes.some((code) => typeof code !== 'string' || !code || code.length > 256) ||
      !Array.isArray(value.releaseArtifactIds) || value.releaseArtifactIds.some((id) => typeof id !== 'string' || !id || id.length > 256)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release denetim neden/artifact listesi geçersiz.')
  }
  const { fingerprint: _fingerprint, ...payload } = value
  const expected = stableRuntimeHash('plastic-region-release-audit-v1', payload)
  if (value.fingerprint !== expected) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release değerlendirme denetim fingerprinti uyuşmuyor.')
  }
  return value
}

function canonicalRuntimePolicy(value) {
  if (!isRecord(value)) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release runtime politikası eksik.')
  return {
    compatibilityBlend: finite(value.compatibilityBlend, Number.EPSILON, 0.5, 'Release uyum harmanı'),
    minimumTrace: finite(value.minimumTrace, Number.EPSILON, 1, 'Release asgari mikro izi'),
    maximumActiveGates: integer(value.maximumActiveGates, 1, 56, 'Release etkin kapı sınırı'),
  }
}

function canonicalBenchmarkRow(value) {
  if (!isRecord(value)) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark satırı geçersiz.')
  return {
    caseId: text(value.caseId, 'Benchmark vaka kimliği'),
    sourceGroupId: text(value.sourceGroupId, 'Benchmark kaynak grubu'),
    seed: integer(value.seed, 0, 0xffff_ffff, 'Benchmark seed'),
    baseScore: finite(value.baseScore, 0, 1, 'Benchmark base skoru'),
    candidateScore: finite(value.candidateScore, 0, 1, 'Benchmark aday skoru'),
    shamScore: finite(value.shamScore, 0, 1, 'Benchmark sham skoru'),
  }
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length)
}

function clampSigned(value) {
  return canonicalRegionFloat(Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0)))
}

function clamp01(value) {
  return canonicalRegionFloat(Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0)))
}

function percentile(values, probability) {
  if (!values.length) return 0
  const ordered = [...values].sort((left, right) => left - right)
  const position = (ordered.length - 1) * probability
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return ordered[lower]
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
}

function releaseSeededUnit(seed, key) {
  return Number.parseInt(stableRuntimeHash('plastic-region-release-bootstrap-order-v1', seed, key).slice(0, 8), 16) /
    0x1_0000_0000
}

function bootstrapSourceGroupInterval(observations, samples, bootstrapSeed) {
  const seeds = [...new Set(observations.map((item) => item.seed))].sort((left, right) => left - right)
  if (seeds.length !== 5) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release ölçümü tam beş seed gerektirir.')
  const groups = [...new Set(observations.map((item) => item.sourceGroupId))].sort()
  if (!groups.length) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release ölçümü kaynak grubu gerektirir.')
  const bySeedAndGroup = new Map()
  for (const observation of observations) {
    const key = `${observation.seed}\u0000${observation.sourceGroupId}`
    const values = bySeedAndGroup.get(key) ?? []
    values.push(observation.score)
    bySeedAndGroup.set(key, values)
  }
  const groupScore = (seed, group) => {
    const values = bySeedAndGroup.get(`${seed}\u0000${group}`)
    return values?.length ? mean(values) : null
  }
  const pointValues = seeds.flatMap((seed) => groups
    .map((group) => groupScore(seed, group))
    .filter((value) => value !== null))
  const means = []
  for (let sample = 0; sample < samples; sample += 1) {
    const seedMeans = []
    for (const seed of seeds) {
      const selected = []
      for (let draw = 0; draw < groups.length; draw += 1) {
        const unit = releaseSeededUnit(bootstrapSeed + sample * 17 + seed, `group-draw:${draw}`)
        const group = groups[Math.min(groups.length - 1, Math.floor(unit * groups.length))]
        const value = groupScore(seed, group)
        if (value !== null) selected.push(value)
      }
      if (selected.length) seedMeans.push(mean(selected))
    }
    if (seedMeans.length) means.push(mean(seedMeans))
  }
  return {
    mean: mean(pointValues),
    low: percentile(means, 0.025),
    high: percentile(means, 0.975),
    seedCount: seeds.length,
    sourceGroupCount: groups.length,
  }
}

function pairedSummary(rows, candidateField) {
  const deltas = rows.map((row) => clampSigned(row[candidateField] - row.baseScore))
  const interval = bootstrapSourceGroupInterval(rows.map((row, index) => ({
    seed: row.seed,
    sourceGroupId: row.sourceGroupId,
    score: (deltas[index] + 1) / 2,
  })), RELEASE_BOOTSTRAP_SAMPLES, RELEASE_BOOTSTRAP_GAIN_SEED)
  const losses = rows.map((row) => ({
    seed: row.seed,
    sourceGroupId: row.sourceGroupId,
    score: clamp01(row.baseScore - row[candidateField]),
  }))
  const regression = bootstrapSourceGroupInterval(
    losses,
    RELEASE_BOOTSTRAP_SAMPLES,
    RELEASE_BOOTSTRAP_REGRESSION_SEED,
  )
  return {
    pointGain: clampSigned(mean(deltas)),
    low: clampSigned(interval.low * 2 - 1),
    high: clampSigned(interval.high * 2 - 1),
    regressionPoint: clamp01(mean(losses.map((item) => item.score))),
    regressionHigh: clamp01(regression.high),
    seedCount: interval.seedCount,
    sourceGroupCount: interval.sourceGroupCount,
  }
}

export function derivePlasticRegionReleaseMetrics(value) {
  const report = value?.format === PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND
    ? validatePlasticRegionReleaseBenchmarkReport(value)
    : value
  if (!isRecord(report) || !Array.isArray(report.rows)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release benchmark satırları eksik.')
  }
  const holdout = pairedSummary(report.rows, 'candidateScore')
  const sham = pairedSummary(report.rows, 'shamScore')
  const conservativeShamGain = Math.abs(sham.low) >= Math.abs(sham.high) ? sham.low : sham.high
  return Object.freeze({
    holdout: Object.freeze({
      caseCount: report.rows.length,
      independentSourceGroupCount: report.sourceGroupIds.length,
      observedGain: holdout.low,
      regressionLoss: holdout.regressionHigh,
    }),
    sham: Object.freeze({
      caseCount: report.rows.length,
      independentSourceGroupCount: report.sourceGroupIds.length,
      observedGain: conservativeShamGain,
      regressionLoss: sham.regressionHigh,
    }),
    summary: Object.freeze({
      baseMean: canonicalRegionFloat(mean(report.rows.map((row) => row.baseScore))),
      candidateMean: canonicalRegionFloat(mean(report.rows.map((row) => row.candidateScore))),
      shamMean: canonicalRegionFloat(mean(report.rows.map((row) => row.shamScore))),
      holdout,
      sham,
    }),
  })
}

function canonicalReleaseMetrics(metrics, sourceGroupIds) {
  if (!isRecord(metrics)) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release metrikleri eksik.')
  const result = {
    caseCount: integer(metrics.caseCount, 1, 1_000_000, 'Release vaka sayısı'),
    independentSourceGroupCount: integer(
      metrics.independentSourceGroupCount,
      1,
      100_000,
      'Release bağımsız kaynak sayısı',
    ),
    observedGain: finite(metrics.observedGain, -1, 1, 'Release gözlenen kazancı'),
    regressionLoss: finite(metrics.regressionLoss, 0, 1, 'Release regresyon kaybı'),
  }
  if (result.independentSourceGroupCount !== sourceGroupIds.length || result.caseCount < sourceGroupIds.length) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release vaka/kaynak sayaçları kaynak kimlikleriyle uyuşmuyor.')
  }
  return result
}

export function createPlasticRegionReleaseEvidence(input) {
  if (!EVALUATION_KINDS.has(input?.evaluationKind)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release kanıtı holdout veya sham olmalı.')
  }
  const sourceGroupIds = sortedUniqueStrings(input.sourceGroupIds, 'Release kaynak grubu')
  rejectsOllamaSource(sourceGroupIds)
  const payload = {
    format: PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
    version: RELEASE_VERSION,
    evaluationId: text(input.evaluationId, 'Release değerlendirme kimliği'),
    evaluationKind: input.evaluationKind,
    representationSpaceId: text(input.representationSpaceId, 'Release temsil uzayı'),
    geometryFingerprint: text(input.geometryFingerprint, 'Release geometri özeti'),
    proposalId: text(input.proposalId, 'Release proposal kimliği'),
    proposalFingerprint: text(input.proposalFingerprint, 'Release proposal özeti'),
    proposedAtlasFingerprint: text(input.proposedAtlasFingerprint, 'Release atlas özeti'),
    baseBinding: canonicalBaseBinding(input.baseBinding),
    benchmarkReport: artifactReference(input.benchmarkReport, 'Release benchmark raporu'),
    sourceGroupIds,
    metrics: canonicalReleaseMetrics(input.metrics, sourceGroupIds),
  }
  return Object.freeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-release-evidence-v1', payload),
  })
}

export function validatePlasticRegionReleaseEvidence(value) {
  if (!isRecord(value) || value.format !== PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND || value.version !== 1) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release kanıt biçimi geçersiz.')
  }
  const canonical = createPlasticRegionReleaseEvidence(value)
  if (!sameCanonical(value, canonical)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release kanıtı kanonik içerik veya fingerprint ile uyuşmuyor.')
  }
  return value
}

export function plasticRegionReleasePolicy(config = {}) {
  return Object.freeze({
    minimumCases: integer(config.minimumCases ?? config.releaseMinimumCases ?? 30, 1, 1_000_000, 'Release asgari vaka sayısı'),
    minimumSourceGroups: integer(
      config.minimumSourceGroups ?? config.releaseMinimumSourceGroups ?? 3,
      2,
      100_000,
      'Release asgari kaynak sayısı',
    ),
    minimumHoldoutGain: finite(config.minimumHoldoutGain ?? config.releaseMinimumHoldoutGain ?? 0.01, 0, 1, 'Release asgari holdout kazancı'),
    maximumShamGain: finite(config.maximumShamGain ?? config.releaseMaximumShamGain ?? 0.03, 0, 1, 'Release azami sham kazancı'),
    maximumRegressionLoss: finite(
      config.maximumRegressionLoss ?? config.releaseMaximumRegressionLoss ?? 0.01,
      0,
      1,
      'Release azami regresyon kaybı',
    ),
    behaviorDimension: PAGE_BEHAVIOR_DIMENSION,
    maximumMembershipsPerPage: integer(
      config.maximumMembershipsPerPage ?? 3,
      1,
      3,
      'Release mikro-sayfa aidiyet sınırı',
    ),
    compatibilityBlend: finite(
      config.compatibilityBlend ?? 0.25,
      Number.EPSILON,
      0.5,
      'Release uyum harmanı',
    ),
    minimumTrace: finite(
      config.minimumTrace ?? 0.02,
      Number.EPSILON,
      1,
      'Release asgari mikro izi',
    ),
    maximumActiveGates: integer(
      config.maximumActiveGates ?? 4,
      1,
      56,
      'Release etkin kapı sınırı',
    ),
  })
}

function artifactReference(value, label) {
  if (!isRecord(value)) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', `${label} artifact bağı eksik.`)
  return {
    artifactId: text(value.artifactId, `${label} artifact kimliği`),
    contentHash: text(value.contentHash, `${label} içerik özeti`),
    fingerprint: text(value.fingerprint, `${label} fingerprinti`),
  }
}

function canonicalBaseBinding(value) {
  if (!isRecord(value)) throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release base bağı eksik.')
  const atlasArtifactId = nullableText(value.atlasArtifactId, 'Base atlas artifact kimliği')
  const atlasContentHash = nullableText(value.atlasContentHash, 'Base atlas içerik özeti')
  if ((atlasArtifactId === null) !== (atlasContentHash === null)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Base atlas artifact kimliği ve içerik özeti birlikte bulunmalı.')
  }
  return {
    activeSnapshotId: text(value.activeSnapshotId, 'Etkin base snapshot kimliği'),
    sourceSnapshotId: text(value.sourceSnapshotId, 'Kaynak aday snapshot kimliği'),
    sourceProducerJobId: text(value.sourceProducerJobId, 'Kaynak üretici iş kimliği'),
    sourceInputManifestHash: text(value.sourceInputManifestHash, 'Kaynak girdi manifesti'),
    sourceArtifactManifestHash: text(value.sourceArtifactManifestHash, 'Kaynak artifact manifesti'),
    baseModelFingerprint: text(value.baseModelFingerprint, 'Base model fingerprinti'),
    familyRunFingerprint: text(value.familyRunFingerprint, 'FamilyRun fingerprinti'),
    languageStateFingerprint: nullableText(value.languageStateFingerprint, 'Dil durumu fingerprinti'),
    thoughtHeadFingerprint: text(value.thoughtHeadFingerprint, 'Düşünce başlığı fingerprinti'),
    thoughtHeadRevision: integer(value.thoughtHeadRevision, 0, Number.MAX_SAFE_INTEGER, 'Düşünce başlığı revisionı'),
    conceptMacroHeadFingerprint: nullableText(value.conceptMacroHeadFingerprint, 'Kavram başlığı fingerprinti'),
    representationProfile: text(value.representationProfile, 'Temsil profili'),
    atlasArtifactId,
    atlasContentHash,
    atlasFingerprint: text(value.atlasFingerprint, 'Base atlas fingerprinti'),
  }
}

export function createPlasticRegionReleaseBenchmarkReport(input) {
  const sourceGroupIds = sortedUniqueStrings(input.sourceGroupIds, 'Benchmark kaynak grubu')
  rejectsOllamaSource(sourceGroupIds)
  if (!Array.isArray(input.rows) || !input.rows.length || input.rows.length > 10_000) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark satır sayısı 1..10000 olmalıdır.')
  }
  const rows = input.rows.map(canonicalBenchmarkRow).sort((left, right) =>
    left.sourceGroupId.localeCompare(right.sourceGroupId) || left.seed - right.seed || left.caseId.localeCompare(right.caseId))
  if (rows.some((row) => !sourceGroupIds.includes(row.sourceGroupId))) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark satırı dondurulmuş kaynak grubu dışında.')
  }
  const seeds = [...new Set(rows.map((row) => row.seed))].sort((left, right) => left - right)
  if (seeds.length !== 5 || rows.length !== sourceGroupIds.length * seeds.length) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark tam beş seed × kaynak grubu matrisi taşımalıdır.')
  }
  const rowKeys = rows.map((row) => `${row.sourceGroupId}\u0000${row.seed}`)
  if (new Set(rowKeys).size !== rowKeys.length) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark kaynak grubu/seed satırları yinelenemez.')
  }
  for (const sourceGroupId of sourceGroupIds) {
    for (const seed of seeds) {
      if (!rowKeys.includes(`${sourceGroupId}\u0000${seed}`)) {
        throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark kaynak grubu/seed matrisi eksik.')
      }
    }
  }
  const runtimePolicy = canonicalRuntimePolicy(input.runtimePolicy)
  const payload = {
    format: PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
    version: RELEASE_VERSION,
    evaluationId: text(input.evaluationId, 'Benchmark değerlendirme kimliği'),
    runnerFingerprint: text(input.runnerFingerprint, 'Benchmark runner fingerprinti'),
    corpusFingerprint: text(input.corpusFingerprint, 'Benchmark corpus fingerprinti'),
    engineFingerprint: text(input.engineFingerprint, 'Benchmark motor fingerprinti'),
    engineContractFingerprint: text(input.engineContractFingerprint, 'Benchmark motor sözleşme fingerprinti'),
    scorerFingerprint: text(input.scorerFingerprint, 'Benchmark puanlayıcı fingerprinti'),
    pairingFingerprint: text(input.pairingFingerprint, 'Benchmark eşleme fingerprinti'),
    deterministic: true,
    usesOllama: false,
    sourcePolicy: PLASTIC_REGION_RELEASE_SOURCE_POLICY,
    representationSpaceId: text(input.representationSpaceId, 'Benchmark temsil uzayı'),
    geometryFingerprint: text(input.geometryFingerprint, 'Benchmark geometri fingerprinti'),
    baseMode: input.baseMode === 'active' ? 'active' : input.baseMode === 'off' ? 'off' : (() => {
      throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark base kipi off veya active olmalıdır.')
    })(),
    baseBinding: canonicalBaseBinding(input.baseBinding),
    runtimePolicy,
    originalProposal: {
      ...artifactReference(input.originalProposal, 'İlk proposal'),
      proposalId: text(input.originalProposal?.proposalId, 'İlk proposal kimliği'),
    },
    evaluatedCandidateAtlasFingerprint: text(
      input.evaluatedCandidateAtlasFingerprint,
      'Ölçülen aday atlas fingerprinti',
    ),
    shamAtlasFingerprint: text(input.shamAtlasFingerprint, 'Sham atlas fingerprinti'),
    maximumTicks: integer(input.maximumTicks, 1, 512, 'Benchmark tick sınırı'),
    sourceGroupIds,
    seeds,
    rows,
  }
  return Object.freeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-release-benchmark-report-v1', payload),
  })
}

export function validatePlasticRegionReleaseBenchmarkReport(value) {
  if (!isRecord(value) || value.format !== PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND || value.version !== 1 ||
      value.deterministic !== true || value.usesOllama !== false) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release benchmark raporu biçimi geçersiz.')
  }
  const canonical = createPlasticRegionReleaseBenchmarkReport(value)
  if (!sameCanonical(value, canonical)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Benchmark raporu kanonik içerik veya fingerprint ile uyuşmuyor.')
  }
  return value
}

export function createPlasticRegionReleaseManifest(input) {
  const policy = plasticRegionReleasePolicy(input.policy)
  const atlas = {
    ...artifactReference(input.atlas, 'Atlas'),
    revision: integer(input.atlas?.revision, 1, Number.MAX_SAFE_INTEGER, 'Atlas revisionı'),
    behaviorDimension: integer(input.atlas?.behaviorDimension, 1, 4096, 'Atlas davranış boyutu'),
    maximumMembershipsPerPage: integer(
      input.atlas?.maximumMembershipsPerPage,
      1,
      8,
      'Atlas aidiyet sınırı',
    ),
  }
  const proposal = {
    ...artifactReference(input.proposal, 'Proposal'),
    proposalId: text(input.proposal?.proposalId, 'Proposal kimliği'),
    baseAtlasFingerprint: text(input.proposal?.baseAtlasFingerprint, 'Proposal base atlas özeti'),
    proposedAtlasFingerprint: text(input.proposal?.proposedAtlasFingerprint, 'Proposal atlas özeti'),
  }
  const holdoutEvidence = {
    ...artifactReference(input.holdoutEvidence, 'Holdout'),
    evaluationId: text(input.holdoutEvidence?.evaluationId, 'Holdout değerlendirme kimliği'),
  }
  const shamEvidence = {
    ...artifactReference(input.shamEvidence, 'Sham'),
    evaluationId: text(input.shamEvidence?.evaluationId, 'Sham değerlendirme kimliği'),
  }
  const payload = {
    format: PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
    version: RELEASE_VERSION,
    releaseId: text(input.releaseId, 'Release kimliği'),
    decision: 'approved',
    activationMode: 'active',
    representationSpaceId: text(input.representationSpaceId, 'Release temsil uzayı'),
    geometryFingerprint: text(input.geometryFingerprint, 'Release geometri özeti'),
    baseBinding: canonicalBaseBinding(input.baseBinding),
    benchmarkReport: artifactReference(input.benchmarkReport, 'Release benchmark raporu'),
    atlas,
    proposal,
    holdoutEvidence,
    shamEvidence,
    policy,
    policyFingerprint: stableRuntimeHash('plastic-region-release-policy-v1', policy),
  }
  return Object.freeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-release-manifest-v1', payload),
  })
}

export function validatePlasticRegionReleaseManifest(value) {
  if (!isRecord(value) || value.format !== PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND || value.version !== 1 ||
      value.decision !== 'approved' || value.activationMode !== 'active') {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release manifest biçimi veya kararı geçersiz.')
  }
  const canonical = createPlasticRegionReleaseManifest(value)
  if (!sameCanonical(value, canonical)) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Release manifesti kanonik içerik veya fingerprint ile uyuşmuyor.')
  }
  if (
    value.holdoutEvidence.artifactId === value.shamEvidence.artifactId ||
    value.holdoutEvidence.evaluationId === value.shamEvidence.evaluationId ||
    value.holdoutEvidence.fingerprint === value.shamEvidence.fingerprint
  ) {
    throw releaseError('INVALID_PLASTIC_REGION_RELEASE', 'Holdout ve sham kanıtları bağımsız immutable kayıtlar olmalıdır.')
  }
  return value
}

function latestAttachedMetadata(store, snapshot, kind) {
  return [...(snapshot?.artifactIds ?? [])].reverse()
    .map((id) => store.getArtifactMetadata?.(id))
    .find((metadata) => metadata?.kind === kind) ?? null
}

function exactArtifact(store, attachedIds, reference, kind) {
  if (!attachedIds.has(reference.artifactId)) {
    throw releaseError('PLASTIC_REGION_RELEASE_STALE', `${kind} release snapshotına bağlı değil.`)
  }
  const metadata = store.getArtifactMetadata?.(reference.artifactId)
  const artifact = metadata ? store.getArtifact?.(reference.artifactId) : null
  if (!metadata || !artifact || metadata.kind !== kind || metadata.contentHash !== reference.contentHash) {
    throw releaseError('PLASTIC_REGION_RELEASE_STALE', `${kind} immutable artifact bağı değişmiş.`)
  }
  return { metadata, artifact }
}

function exactMetadata(metadata, expected) {
  if (!isRecord(metadata)) throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Artifact metadata eksik.')
  for (const [field, value] of Object.entries(expected)) {
    if (metadata[field] !== value) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', `Artifact metadata ${field} alanı içerikle uyuşmuyor.`, {
        field,
        expected: value,
        actual: metadata[field],
      })
    }
  }
}

function snapshotArtifactManifest(store, snapshot) {
  const manifest = (snapshot?.artifactIds ?? []).map((artifactId) => {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (!metadata) {
      throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Kaynak snapshot artifact manifesti artık tam değil.', {
        artifactId,
      })
    }
    return { id: metadata.id, kind: metadata.kind, contentHash: metadata.contentHash }
  }).sort((left, right) => left.id.localeCompare(right.id))
  return stableRuntimeHash('thought-base-artifacts-v1', manifest)
}

function exactBaseBinding(store, snapshot, binding) {
  if (snapshot?.parentId !== binding.activeSnapshotId) {
    throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Release adayı ölçülen etkin base’in doğrudan çocuğu değil.')
  }
  const source = store.getSnapshot?.(binding.sourceSnapshotId)
  if (!source || source.status !== 'candidate' || source.parentId !== binding.activeSnapshotId ||
      source.producerJobId !== binding.sourceProducerJobId ||
      source.inputManifestHash !== binding.sourceInputManifestHash ||
      snapshotArtifactManifest(store, source) !== binding.sourceArtifactManifestHash) {
    throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Ölçülen kaynak snapshot/manifest bağı değişmiş veya eksik.')
  }
  const producer = store.getJob?.(binding.sourceProducerJobId)
  const certifiedIds = new Set([
    producer?.result?.snapshotId,
    producer?.result?.lastSnapshotId,
    ...(Array.isArray(producer?.result?.snapshotIds) ? producer.result.snapshotIds : []),
    ...(Array.isArray(producer?.result?.trainingSnapshotIds) ? producer.result.trainingSnapshotIds : []),
  ].filter(Boolean))
  if (producer?.state !== 'succeeded' || !certifiedIds.has(source.id) ||
      source.evaluation?.passed !== true || source.evaluation?.runtimeVerified !== true) {
    throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Kaynak snapshot üretici işiyle doğrulanamıyor.')
  }
  const attached = new Set(snapshot?.artifactIds ?? [])
  if ((source.artifactIds ?? []).some((artifactId) => !attached.has(artifactId))) {
    throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Release adayı kaynak artifact soyunu eksiksiz korumuyor.')
  }
  if (!sameCanonical(snapshot.sourceIds ?? [], source.sourceIds ?? []) ||
      !sameCanonical(snapshot.sourceRevisions ?? {}, source.sourceRevisions ?? {}) ||
      snapshot.inputManifestHash !== source.inputManifestHash) {
    throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Release adayı kaynak/revision/girdi soyunu birebir korumuyor.')
  }
  const activeBase = store.getSnapshot?.(binding.activeSnapshotId)
  if (!activeBase) throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Ölçülen etkin base snapshot bulunamadı.')
  if (binding.atlasArtifactId === null) {
    if (binding.atlasContentHash !== null) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Boş base atlas bağı içerik özeti taşıyamaz.')
    }
  } else {
    if (!(activeBase.artifactIds ?? []).includes(binding.atlasArtifactId)) {
      throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Ölçülen base atlas etkin base snapshota bağlı değil.')
    }
    const metadata = store.getArtifactMetadata?.(binding.atlasArtifactId)
    const artifact = metadata ? store.getArtifact?.(binding.atlasArtifactId) : null
    if (!metadata || !artifact || metadata.kind !== 'plastic-region-atlas-head' ||
        metadata.contentHash !== binding.atlasContentHash) {
      throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Ölçülen base atlas immutable bağı değişmiş.')
    }
    const baseAtlas = restoreArtifact(artifact, 'plastic-region-atlas-head')
    validateAdaptiveRegionAtlas(baseAtlas)
    if (baseAtlas.fingerprint !== binding.atlasFingerprint) {
      throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Ölçülen base atlas fingerprinti değişmiş.')
    }
  }
  return source
}

/**
 * Validates the complete immutable release chain attached to one snapshot.
 * General snapshot evaluation is intentionally not accepted as a substitute.
 */
export function verifyPlasticRegionReleaseChain(store, snapshot, atlasRecord, config = {}) {
  const manifestMetadata = latestAttachedMetadata(store, snapshot, PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND)
  if (!manifestMetadata) {
    return { compatible: false, reason: 'release-manifest-absent', manifest: null, artifactIds: [] }
  }
  try {
    const attachedIds = new Set(snapshot?.artifactIds ?? [])
    const manifestArtifact = store.getArtifact(manifestMetadata.id)
    const manifest = validatePlasticRegionReleaseManifest(
      restoreArtifact(manifestArtifact, PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND),
    )
    exactMetadata(manifestArtifact.payload?.metadata, {
      fingerprint: manifest.fingerprint,
      representationSpaceId: manifest.representationSpaceId,
      geometryFingerprint: manifest.geometryFingerprint,
      releaseId: manifest.releaseId,
      activeBaseSnapshotId: manifest.baseBinding.activeSnapshotId,
      sourceSnapshotId: manifest.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: manifest.baseBinding.sourceArtifactManifestHash,
      benchmarkReportArtifactId: manifest.benchmarkReport.artifactId,
      benchmarkReportFingerprint: manifest.benchmarkReport.fingerprint,
    })
    const expectedPolicy = plasticRegionReleasePolicy(config)
    if (!sameCanonical(manifest.policy, expectedPolicy)) {
      throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Release politikası etkin servis politikasıyla uyuşmuyor.')
    }
    exactBaseBinding(store, snapshot, manifest.baseBinding)

    const atlas = atlasRecord?.head
    const atlasArtifact = atlasRecord?.artifact
    if (!atlas || !atlasArtifact || atlasRecord.metadata?.id !== manifest.atlas.artifactId ||
        atlasRecord.metadata.contentHash !== manifest.atlas.contentHash) {
      throw releaseError('PLASTIC_REGION_RELEASE_STALE', 'Release manifesti etkin atlas başlığıyla uyuşmuyor.')
    }
    validateAdaptiveRegionAtlas(atlas)
    exactMetadata(atlasArtifact.payload?.metadata, {
      fingerprint: atlas.fingerprint,
      representationSpaceId: atlas.representationSpaceId,
      geometryFingerprint: atlas.geometryFingerprint,
      revision: atlas.revision,
      behaviorDimension: atlas.behaviorDimension,
      maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
      activationEligible: true,
    })
    if (atlasArtifact.payload?.metadata?.candidateOnly === true) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Candidate-only atlas active yönlendirmede kullanılamaz.')
    }
    if (
      atlas.fingerprint !== manifest.atlas.fingerprint ||
      atlas.revision !== manifest.atlas.revision ||
      atlas.behaviorDimension !== manifest.atlas.behaviorDimension ||
      atlas.maximumMembershipsPerPage !== manifest.atlas.maximumMembershipsPerPage ||
      atlas.behaviorDimension !== PAGE_BEHAVIOR_DIMENSION ||
      atlas.maximumMembershipsPerPage !== expectedPolicy.maximumMembershipsPerPage ||
      atlas.representationSpaceId !== manifest.representationSpaceId ||
      atlas.geometryFingerprint !== manifest.geometryFingerprint
    ) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Release atlas içeriği, geometrisi veya politikası değişmiş.')
    }

    const proposalRecord = exactArtifact(
      store,
      attachedIds,
      manifest.proposal,
      'plastic-region-atlas-proposal',
    )
    const proposal = restoreArtifact(proposalRecord.artifact, 'plastic-region-atlas-proposal')
    validateRegionLifecycleProposal(proposal)
    exactMetadata(proposalRecord.artifact.payload?.metadata, {
      fingerprint: proposal.fingerprint,
      representationSpaceId: proposal.proposedAtlas.representationSpaceId,
      geometryFingerprint: proposal.proposedAtlas.geometryFingerprint,
      revision: proposal.proposedAtlas.revision,
      proposalId: proposal.proposalId,
      baseAtlasFingerprint: proposal.baseAtlasFingerprint,
      proposedAtlasFingerprint: proposal.proposedAtlas.fingerprint,
      eligible: true,
    })
    if (!proposal.eligible || proposal.rejectionReasons.length ||
        proposal.fingerprint !== manifest.proposal.fingerprint ||
        proposal.proposalId !== manifest.proposal.proposalId ||
        proposal.baseAtlasFingerprint !== manifest.proposal.baseAtlasFingerprint ||
        proposal.proposedAtlas.fingerprint !== manifest.proposal.proposedAtlasFingerprint ||
        proposal.proposedAtlas.fingerprint !== atlas.fingerprint ||
        proposal.baseAtlasFingerprint !== atlas.parentFingerprint) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Release proposal/atlas zinciri uyuşmuyor veya uygun değil.')
    }
    const proposedRegionSet = new Set(proposal.proposedRegionIds)
    if (atlas.regions.some((region) => proposedRegionSet.has(region.id) && region.lifecycle !== 'active')) {
      throw releaseError('PLASTIC_REGION_RELEASE_GATE_FAILED', 'Release edilen yeni bölgeler active yaşam durumuna terfi etmemiş.')
    }

    const reportRecord = exactArtifact(
      store,
      attachedIds,
      manifest.benchmarkReport,
      PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
    )
    const report = validatePlasticRegionReleaseBenchmarkReport(
      restoreArtifact(reportRecord.artifact, PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND),
    )
    exactMetadata(reportRecord.artifact.payload?.metadata, {
      fingerprint: report.fingerprint,
      evaluationId: report.evaluationId,
      runnerFingerprint: report.runnerFingerprint,
      activeBaseSnapshotId: report.baseBinding.activeSnapshotId,
      sourceSnapshotId: report.baseBinding.sourceSnapshotId,
      sourceArtifactManifestHash: report.baseBinding.sourceArtifactManifestHash,
      originalProposalArtifactId: report.originalProposal.artifactId,
      usesOllama: false,
    })
    if (report.fingerprint !== manifest.benchmarkReport.fingerprint ||
        !sameCanonical(report.baseBinding, manifest.baseBinding) ||
        !sameCanonical(report.runtimePolicy, {
          compatibilityBlend: expectedPolicy.compatibilityBlend,
          minimumTrace: expectedPolicy.minimumTrace,
          maximumActiveGates: expectedPolicy.maximumActiveGates,
        })) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Ham benchmark raporu base/deploy politikasıyla uyuşmuyor.')
    }
    const expectedCases = plasticRegionReleaseCases(report.sourceGroupIds.length)
    const expectedCaseByGroup = new Map(expectedCases.map((item) => [item.sourceGroupId, item.id]))
    if (report.runnerFingerprint !== PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT ||
        report.sourcePolicy !== PLASTIC_REGION_RELEASE_SOURCE_POLICY ||
        report.engineContractFingerprint !== PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT ||
        report.scorerFingerprint !== PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT ||
        report.corpusFingerprint !== plasticRegionReleaseCorpusFingerprint(report.sourceGroupIds.length) ||
        !sameCanonical(report.sourceGroupIds, expectedCases.map((item) => item.sourceGroupId).sort()) ||
        report.rows.some((row) => expectedCaseByGroup.get(row.sourceGroupId) !== row.caseId)) {
      throw releaseError(
        'PLASTIC_REGION_RELEASE_STALE',
        'Benchmark runner/corpus/motor/puanlayıcı sözleşmesi çalışan yazılımla uyuşmuyor.',
      )
    }
    const originalProposalRecord = exactArtifact(
      store,
      attachedIds,
      report.originalProposal,
      'plastic-region-atlas-proposal',
    )
    const originalProposal = restoreArtifact(originalProposalRecord.artifact, 'plastic-region-atlas-proposal')
    validateRegionLifecycleProposal(originalProposal)
    if (originalProposal.proposalId !== report.originalProposal.proposalId ||
        originalProposal.fingerprint !== report.originalProposal.fingerprint ||
        originalProposal.baseAtlasFingerprint !== manifest.baseBinding.atlasFingerprint ||
        originalProposal.proposedAtlas.representationSpaceId !== manifest.representationSpaceId ||
        originalProposal.proposedAtlas.geometryFingerprint !== manifest.geometryFingerprint) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Ham benchmark raporunun ilk proposal bağı geçersiz.')
    }

    const holdoutRecord = exactArtifact(
      store,
      attachedIds,
      manifest.holdoutEvidence,
      PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
    )
    const shamRecord = exactArtifact(
      store,
      attachedIds,
      manifest.shamEvidence,
      PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
    )
    const holdout = validatePlasticRegionReleaseEvidence(
      restoreArtifact(holdoutRecord.artifact, PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND),
    )
    const sham = validatePlasticRegionReleaseEvidence(
      restoreArtifact(shamRecord.artifact, PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND),
    )
    for (const [record, value, reference] of [
      [holdoutRecord, holdout, manifest.holdoutEvidence],
      [shamRecord, sham, manifest.shamEvidence],
    ]) {
      exactMetadata(record.artifact.payload?.metadata, {
        fingerprint: value.fingerprint,
        representationSpaceId: value.representationSpaceId,
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
      if (value.fingerprint !== reference.fingerprint || value.evaluationId !== reference.evaluationId ||
          value.representationSpaceId !== atlas.representationSpaceId ||
          value.geometryFingerprint !== atlas.geometryFingerprint ||
          value.proposalId !== proposal.proposalId || value.proposalFingerprint !== proposal.fingerprint ||
          value.proposedAtlasFingerprint !== atlas.fingerprint ||
          !sameCanonical(value.baseBinding, manifest.baseBinding) ||
          !sameCanonical(value.benchmarkReport, manifest.benchmarkReport)) {
        throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Release değerlendirme kanıtı proposal/atlasla uyuşmuyor.')
      }
    }
    if (holdout.evaluationKind !== 'holdout' || sham.evaluationKind !== 'sham') {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Holdout/sham kanıt rolleri değişmiş.')
    }
    const derived = derivePlasticRegionReleaseMetrics(report)
    if (!sameCanonical(holdout.sourceGroupIds, report.sourceGroupIds) ||
        !sameCanonical(sham.sourceGroupIds, report.sourceGroupIds) ||
        !sameCanonical(holdout.metrics, derived.holdout) ||
        !sameCanonical(sham.metrics, derived.sham)) {
      throw releaseError('PLASTIC_REGION_RELEASE_TAMPERED', 'Release metrikleri ham vaka satırlarından yeniden üretilemiyor.')
    }
    const holdoutPass = holdout.metrics.caseCount >= expectedPolicy.minimumCases &&
      holdout.metrics.independentSourceGroupCount >= expectedPolicy.minimumSourceGroups &&
      holdout.metrics.observedGain >= expectedPolicy.minimumHoldoutGain &&
      holdout.metrics.regressionLoss <= expectedPolicy.maximumRegressionLoss
    const shamPass = sham.metrics.caseCount >= expectedPolicy.minimumCases &&
      sham.metrics.independentSourceGroupCount >= expectedPolicy.minimumSourceGroups &&
      Math.abs(sham.metrics.observedGain) <= expectedPolicy.maximumShamGain &&
      sham.metrics.regressionLoss <= expectedPolicy.maximumRegressionLoss
    if (!holdoutPass || !shamPass ||
        Math.abs(proposal.evidence.holdoutGain - holdout.metrics.observedGain) > 1e-6 ||
        Math.abs(proposal.evidence.shamGain - sham.metrics.observedGain) > 1e-6) {
      throw releaseError('PLASTIC_REGION_RELEASE_GATE_FAILED', 'Bağımsız holdout/sham release kapısı geçilmedi.', {
        holdoutPass,
        shamPass,
        proposalHoldoutGain: proposal.evidence.holdoutGain,
        observedHoldoutGain: holdout.metrics.observedGain,
        proposalShamGain: proposal.evidence.shamGain,
        observedShamGain: sham.metrics.observedGain,
        expectedPolicy,
      })
    }

    const artifactIds = [
      atlasRecord.metadata.id,
      proposalRecord.metadata.id,
      holdoutRecord.metadata.id,
      shamRecord.metadata.id,
      reportRecord.metadata.id,
      originalProposalRecord.metadata.id,
      manifestMetadata.id,
    ]
    return {
      compatible: true,
      reason: null,
      manifest,
      manifestMetadata,
      proposal,
      holdout,
      sham,
      report,
      artifactIds,
    }
  } catch (error) {
    return {
      compatible: false,
      reason: `release-chain-invalid:${String(error?.code ?? 'decode-failed')}`,
      error,
      manifest: null,
      manifestMetadata,
      artifactIds: [],
    }
  }
}
