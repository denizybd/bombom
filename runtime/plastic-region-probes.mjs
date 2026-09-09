import {
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_REGION_GEOMETRY,
} from '../src/adaptive-regions/index.ts'
import { stableRuntimeHash } from './artifacts.mjs'

/**
 * NEW, opt-in, read-only interpretation layer.
 *
 * These artifacts never become atlas ids, region features, routing weights,
 * lifecycle evidence, or training deltas. They only test an external human or
 * deterministic hypothesis against immutable adaptive-region responses.
 */
export const SEMANTIC_PROBE_LAYER = 'new-opt-in-interpretation'
export const SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND = 'plastic-region-semantic-probe-observations'
export const SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND = 'plastic-region-semantic-probe-calibration'
export const SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND = 'plastic-region-semantic-probe-evaluation'
export const SEMANTIC_PROBE_PUBLIC_SUMMARY_FORMAT = 'plastic-region-semantic-probe-public-summary'
export const SEMANTIC_PROBE_VERSION = 1
export const SEMANTIC_PROBE_REPRESENTATION_PROFILE = 'adaptive-eighth-id-v1'

const SPLITS = new Set(['train', 'validation', 'test'])
const LABEL_AUTHORITIES = new Set(['human', 'deterministic', 'ollama'])
const RESPONSE_KINDS = new Set(['region-gate'])
const HYPOTHESIS_STATUSES = new Set(['unknown', 'provisional', 'calibrated'])
const FORBIDDEN_HYPOTHESIS_NAMESPACES = new Set(['atlas', 'region', 'feature', 'page', 'neuron', 'family'])
const HARD_LIMITS = Object.freeze({
  maximumRecords: 2_048,
  maximumFeatures: 128,
  maximumLineageTokensPerRecord: 8,
  maximumArtifactBytes: 8 * 1024 * 1024,
  maximumBootstrapSamples: 5_000,
})
const DEFAULT_LIMITS = Object.freeze({
  maximumRecords: 512,
  maximumFeatures: 56,
  maximumLineageTokensPerRecord: 8,
  maximumArtifactBytes: 4 * 1024 * 1024,
})

function probeError(code, message, details) {
  const error = new Error(message)
  error.code = code
  if (details !== undefined) error.details = details
  return error
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value)
}

function canonicalFloat(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw probeError('INVALID_SEMANTIC_PROBE', `${label} ${minimum}..${maximum} aralığında olmalı.`)
  }
  const result = Math.fround(value)
  return Object.is(result, -0) ? 0 : result
}

function integer(value, minimum, maximum, label) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw probeError('INVALID_SEMANTIC_PROBE', `${label} ${minimum}..${maximum} aralığında tam sayı olmalı.`)
  }
  return value
}

function text(value, label, maximum = 256) {
  const result = typeof value === 'string' ? value.trim() : ''
  if (!result || result.length > maximum || result.includes('\u0000')) {
    throw probeError('INVALID_SEMANTIC_PROBE', `${label} boş, aşırı uzun veya geçersiz olamaz.`)
  }
  return result
}

function compareText(left, right) {
  return String(left).localeCompare(String(right))
}

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child)
    Object.freeze(value)
  }
  return value
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (isRecord(value)) {
    return `{${Object.keys(value).sort(compareText).map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function sameCanonical(left, right) {
  return canonicalJson(left) === canonicalJson(right)
}

function artifactBytes(value) {
  return Buffer.byteLength(canonicalJson(value), 'utf8')
}

function canonicalLimits(input = {}) {
  const limits = {
    maximumRecords: integer(
      input.maximumRecords ?? DEFAULT_LIMITS.maximumRecords,
      1,
      HARD_LIMITS.maximumRecords,
      'Probe kayıt sınırı',
    ),
    maximumFeatures: integer(
      input.maximumFeatures ?? DEFAULT_LIMITS.maximumFeatures,
      2,
      HARD_LIMITS.maximumFeatures,
      'Probe özellik sınırı',
    ),
    maximumLineageTokensPerRecord: integer(
      input.maximumLineageTokensPerRecord ?? DEFAULT_LIMITS.maximumLineageTokensPerRecord,
      1,
      HARD_LIMITS.maximumLineageTokensPerRecord,
      'Probe soy izi sınırı',
    ),
    maximumArtifactBytes: integer(
      input.maximumArtifactBytes ?? DEFAULT_LIMITS.maximumArtifactBytes,
      512,
      HARD_LIMITS.maximumArtifactBytes,
      'Probe artifact byte sınırı',
    ),
  }
  return deepFreeze(limits)
}

function assertArtifactBound(value, maximumArtifactBytes) {
  const bytes = artifactBytes(value)
  if (bytes > maximumArtifactBytes) {
    throw probeError('SEMANTIC_PROBE_ARTIFACT_LIMIT_EXCEEDED', 'Anlamsal probe artifacti belirlenen byte sınırını aşıyor.', {
      bytes,
      maximumArtifactBytes,
    })
  }
  return bytes
}

function canonicalHypothesis(value) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', 'Dış hipotez tanımı eksik.')
  const namespace = text(value.namespace, 'Hipotez ad alanı', 48).toLowerCase()
  const impersonatesInternalNamespace = [...FORBIDDEN_HYPOTHESIS_NAMESPACES].some((prefix) =>
    namespace === prefix || namespace.startsWith(`${prefix}.`) || namespace.startsWith(`${prefix}-`) ||
    namespace.startsWith(`${prefix}_`) || namespace.startsWith(`${prefix}/`))
  if (!/^[a-z0-9][a-z0-9._/-]*$/.test(namespace) || impersonatesInternalNamespace) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Hipotez ad alanı atlasın iç yapısını adlandıramaz.')
  }
  const label = text(value.label, 'Hipotez etiketi', 128)
  const definitionFingerprint = text(value.definitionFingerprint, 'Hipotez tanım özeti', 256)
  const schemaVersion = integer(value.schemaVersion ?? 1, 1, 1_000_000, 'Hipotez şema sürümü')
  const payload = {
    role: 'external-hypothesis',
    namespace,
    label,
    schemaVersion,
    definitionFingerprint,
  }
  return deepFreeze({
    ...payload,
    hypothesisId: stableRuntimeHash('plastic-region-semantic-hypothesis-v1', payload),
  })
}

function canonicalAtlasBinding(value) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe atlas bağı eksik.')
  const result = {
    representationProfile: text(value.representationProfile, 'Temsil profili'),
    representationSpaceId: text(value.representationSpaceId, 'Temsil uzayı'),
    geometryFingerprint: text(value.geometryFingerprint, 'Geometri özeti'),
    atlasArtifactId: text(value.atlasArtifactId, 'Atlas artifact kimliği'),
    atlasContentHash: text(value.atlasContentHash, 'Atlas içerik özeti'),
    atlasFingerprint: text(value.atlasFingerprint, 'Atlas özeti'),
    atlasRevision: integer(value.atlasRevision, 0, Number.MAX_SAFE_INTEGER, 'Atlas revizyonu'),
  }
  if (result.representationProfile !== SEMANTIC_PROBE_REPRESENTATION_PROFILE ||
      result.geometryFingerprint !== ADAPTIVE_REGION_GEOMETRY.fingerprint) {
    throw probeError(
      'SEMANTIC_PROBE_REPRESENTATION_MISMATCH',
      'Anlamsal probe yalnız açık adaptive-eighth-id-v1 8/56 temsilinde çalışır; eski temsil dönüştürülmez.',
    )
  }
  return deepFreeze(result)
}

function canonicalDatasetBinding(value) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe veri bölümü bağı eksik.')
  return deepFreeze({
    datasetId: text(value.datasetId, 'Veri kümesi kimliği'),
    partitionManifestFingerprint: text(value.partitionManifestFingerprint, 'Bölümleme manifest özeti'),
  })
}

function canonicalProducerBinding(value) {
  if (!isRecord(value) || !['trusted-deterministic-pack', 'audited-external-labels'].includes(value.mode)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe üretici bağı eksik veya desteklenmeyen kipte.')
  }
  if (typeof value.usesOllama !== 'boolean') {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe üretici Ollama kullanımı açık boolean olmalı.')
  }
  return deepFreeze({
    mode: value.mode,
    producerFingerprint: text(value.producerFingerprint, 'Probe üretici fingerprinti'),
    inputManifestHash: text(value.inputManifestHash, 'Probe girdi manifest özeti'),
    probeSpecFingerprint: text(value.probeSpecFingerprint, 'Probe tanım paketi özeti'),
    usesOllama: value.usesOllama === true,
  })
}

function canonicalLabelBinding(value) {
  if (!isRecord(value) || !LABEL_AUTHORITIES.has(value.authority)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Etiket otoritesi human, deterministic veya ollama olmalı.')
  }
  const payload = {
    authority: value.authority,
    artifactId: text(value.artifactId, 'Etiket artifact kimliği'),
    contentHash: text(value.contentHash, 'Etiket içerik özeti'),
  }
  return deepFreeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-semantic-label-binding-v1', payload),
  })
}

function sourceOrigin(source) {
  const joined = [source?.sourceType, source?.provenance?.kind, source?.provenance?.generator, source?.provenance?.model]
    .filter((value) => typeof value === 'string')
    .join(' ')
    .toLowerCase()
  if (joined.includes('ollama')) return 'ollama'
  if (joined.includes('synthetic') || joined.includes('generated')) return 'synthetic'
  if (joined.includes('deterministic')) return 'deterministic'
  return 'human-or-imported'
}

function tokenDigest(kind, value) {
  const normalized = text(value, `Kaynak ${kind} izi`, 2_048)
  return `SLT-${stableRuntimeHash('plastic-region-semantic-lineage-token-v1', kind, normalized)}`
}

/**
 * Derives lineage only from server-controlled physical source facts. A legacy
 * or caller-provided independenceGroupId is deliberately ignored.
 */
export function deriveSemanticProbeSourceLineage(source, sampleContentHash, limits = DEFAULT_LIMITS) {
  if (!isRecord(source)) throw probeError('INVALID_SEMANTIC_PROBE', 'Fiziksel kaynak kaydı eksik.')
  const origin = sourceOrigin(source)
  const tokens = []
  const add = (kind, value) => {
    if (typeof value === 'string' && value.trim()) tokens.push(tokenDigest(kind, value.trim()))
  }
  const provenance = isRecord(source.provenance) ? source.provenance : {}
  if (origin === 'ollama') {
    const modelLineage = provenance.weightsDigest ?? provenance.modelDigest ?? provenance.generationModel ?? provenance.model
    if (typeof modelLineage !== 'string' || !modelLineage.trim()) {
      throw probeError('SEMANTIC_PROBE_OLLAMA_LINEAGE_REQUIRED', 'Ollama train kaydı model/weights soyunu taşımak zorunda.')
    }
    add('ollama-model', modelLineage)
  } else {
    add('physical-locator', provenance.canonicalPath ?? source.locator)
  }
  add('source-content', source.contentHash)
  if (!tokens.length) {
    throw probeError(
      'SEMANTIC_PROBE_PHYSICAL_LINEAGE_REQUIRED',
      'Probe kaynağı istemci kayıt kimliğinden değil fiziksel locator/içerik soyundan türetilmeli.',
    )
  }
  add('sample-content', sampleContentHash)
  const uniqueTokens = [...new Set(tokens)].sort(compareText)
  const maximumTokens = integer(
    limits.maximumLineageTokensPerRecord ?? DEFAULT_LIMITS.maximumLineageTokensPerRecord,
    1,
    HARD_LIMITS.maximumLineageTokensPerRecord,
    'Probe soy izi sınırı',
  )
  if (uniqueTokens.length < 1 || uniqueTokens.length > maximumTokens) {
    throw probeError('SEMANTIC_PROBE_LINEAGE_LIMIT_EXCEEDED', 'Kaynak soyu izi eksik veya sınırı aşıyor.')
  }
  const sampleToken = tokenDigest('sample-content', sampleContentHash)
  const payload = { origin, tokens: uniqueTokens, sampleToken }
  return deepFreeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-semantic-lineage-v1', payload),
  })
}

class DisjointSet {
  constructor() {
    this.parents = new Map()
  }

  add(key) {
    if (!this.parents.has(key)) this.parents.set(key, key)
  }

  root(key) {
    this.add(key)
    const parent = this.parents.get(key)
    if (parent === key) return key
    const root = this.root(parent)
    this.parents.set(key, root)
    return root
  }

  union(left, right) {
    const leftRoot = this.root(left)
    const rightRoot = this.root(right)
    if (leftRoot === rightRoot) return
    if (compareText(leftRoot, rightRoot) <= 0) this.parents.set(rightRoot, leftRoot)
    else this.parents.set(leftRoot, rightRoot)
  }
}

function assignIndependentGroups(lineages) {
  const set = new DisjointSet()
  for (const lineage of lineages) {
    for (const token of lineage.tokens) set.add(token)
    for (let index = 1; index < lineage.tokens.length; index += 1) {
      set.union(lineage.tokens[0], lineage.tokens[index])
    }
  }
  const tokensByRoot = new Map()
  for (const lineage of lineages) {
    const root = set.root(lineage.tokens[0])
    const values = tokensByRoot.get(root) ?? new Set()
    lineage.tokens.forEach((token) => values.add(token))
    tokensByRoot.set(root, values)
  }
  const groupByToken = new Map()
  for (const values of tokensByRoot.values()) {
    const tokens = [...values].sort(compareText)
    const group = `SPG-${stableRuntimeHash('plastic-region-semantic-independent-group-v1', tokens)}`
    tokens.forEach((token) => groupByToken.set(token, group))
  }
  return lineages.map((lineage) => ({
    ...lineage,
    independenceKey: groupByToken.get(lineage.tokens[0]),
  }))
}

function canonicalFootprint(value) {
  if (!Array.isArray(value) || !value.length || value.length > ADAPTIVE_PAGE_COUNT) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Yapısal ayak izi 1..56 mikro-sayfa içermeli.')
  }
  const seen = new Set()
  const result = value.map((entry) => {
    if (!isRecord(entry)) throw probeError('INVALID_SEMANTIC_PROBE', 'Ayak izi girdisi geçersiz.')
    const pageIndex = integer(entry.pageIndex, 0, ADAPTIVE_PAGE_COUNT - 1, 'Ayak izi sayfa indeksi')
    if (seen.has(pageIndex)) throw probeError('INVALID_SEMANTIC_PROBE', 'Ayak izi sayfa indeksleri yinelenemez.')
    seen.add(pageIndex)
    return {
      pageIndex,
      weight: canonicalFloat(entry.weight, Number.EPSILON, 1, 'Ayak izi ağırlığı'),
    }
  }).sort((left, right) => left.pageIndex - right.pageIndex)
  return deepFreeze(result)
}

function canonicalFeature(value, atlasBinding) {
  if (!isRecord(value) || !RESPONSE_KINDS.has(value.responseKind)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe özelliği desteklenmeyen bir yanıt türü taşıyor.')
  }
  const featureId = text(value.featureId, 'Yapısal özellik kimliği')
  const footprint = canonicalFootprint(value.footprint)
  const footprintFingerprint = stableRuntimeHash('plastic-region-semantic-footprint-v1', footprint)
  const payload = {
    featureId,
    responseKind: value.responseKind,
    atlasFingerprint: atlasBinding.atlasFingerprint,
    footprint,
    footprintFingerprint,
  }
  return deepFreeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-semantic-feature-v1', payload),
  })
}

function canonicalFeatures(values, atlasBinding, maximumFeatures) {
  if (!Array.isArray(values) || values.length < 2 || values.length > maximumFeatures) {
    throw probeError('SEMANTIC_PROBE_FEATURE_LIMIT_EXCEEDED', 'Probe 2 ile belirlenen üst sınır arasında özellik taşımalı.')
  }
  const features = values.map((value) => canonicalFeature(value, atlasBinding))
    .sort((left, right) => compareText(left.featureId, right.featureId))
  if (new Set(features.map((feature) => feature.featureId)).size !== features.length ||
      new Set(features.map((feature) => feature.fingerprint)).size !== features.length) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe özellikleri yinelenemez.')
  }
  return deepFreeze(features)
}

function canonicalScores(rawScores, features) {
  if (!Array.isArray(rawScores) || rawScores.length !== features.length) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Her probe kaydı her yapısal özellik için bir skor taşımalı.')
  }
  const byFeature = new Map()
  for (const entry of rawScores) {
    if (!isRecord(entry)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe skor girdisi geçersiz.')
    const featureId = text(entry.featureId, 'Skor özellik kimliği')
    if (byFeature.has(featureId)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe skor özellikleri yinelenemez.')
    byFeature.set(featureId, canonicalFloat(entry.score, 0, 1, 'Probe yanıt skoru'))
  }
  if (features.some((feature) => !byFeature.has(feature.featureId))) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe skorları özellik kataloğuyla uyuşmuyor.')
  }
  return deepFreeze(features.map((feature) => byFeature.get(feature.featureId)))
}

function canonicalStoredLineage(value, maximumTokens) {
  if (!isRecord(value) || !Array.isArray(value.tokens)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Saklanan kaynak soyu geçersiz.')
  }
  const origin = text(value.origin, 'Kaynak kökeni', 32)
  if (!['ollama', 'synthetic', 'deterministic', 'human-or-imported'].includes(origin)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Kaynak kökeni geçersiz.')
  }
  const tokens = value.tokens.map((token) => text(token, 'Kaynak soy izi'))
    .sort(compareText)
  if (tokens.length < 1 || tokens.length > maximumTokens || new Set(tokens).size !== tokens.length ||
      tokens.some((token) => !token.startsWith('SLT-'))) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Kaynak soy izleri eksik, yinelenen veya sınır dışında.')
  }
  const sampleToken = text(value.sampleToken, 'Örnek içerik izi')
  if (!tokens.includes(sampleToken)) throw probeError('INVALID_SEMANTIC_PROBE', 'Örnek içerik izi kaynak soyuna bağlı değil.')
  const payload = { origin, tokens, sampleToken }
  const fingerprint = stableRuntimeHash('plastic-region-semantic-lineage-v1', payload)
  if (value.fingerprint !== undefined && value.fingerprint !== fingerprint) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Kaynak soy fingerprinti içerikle uyuşmuyor.')
  }
  return { ...payload, fingerprint }
}

function buildObservationPayload(input) {
  const limits = canonicalLimits(input.limits)
  const hypothesis = canonicalHypothesis(input.hypothesis)
  const atlasBinding = canonicalAtlasBinding(input.atlasBinding)
  const datasetBinding = canonicalDatasetBinding(input.datasetBinding)
  const split = text(input.split, 'Probe veri bölümü', 16)
  if (!SPLITS.has(split)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe veri bölümü train, validation veya test olmalı.')
  const features = canonicalFeatures(input.features, atlasBinding, limits.maximumFeatures)
  if (features.some((feature) => feature.featureId === hypothesis.label || feature.featureId === hypothesis.hypothesisId)) {
    throw probeError('SEMANTIC_LABEL_FEATURE_COLLISION', 'Dış hipotez etiketi atlas özelliğinin kimliği olamaz.')
  }
  if (!Array.isArray(input.records) || !input.records.length || input.records.length > limits.maximumRecords) {
    throw probeError('SEMANTIC_PROBE_RECORD_LIMIT_EXCEEDED', 'Probe kayıt sayısı belirlenen sınır dışında.')
  }

  const rawRecords = input.records.map((record) => {
    if (!isRecord(record)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe kaydı geçersiz.')
    const sampleContentHash = text(record.sampleContentHash, 'Örnek içerik özeti')
    const lineage = deriveSemanticProbeSourceLineage(record.source, sampleContentHash, limits)
    const labelBinding = canonicalLabelBinding(record.labelBinding)
    if (split !== 'train' && (lineage.origin === 'ollama' || labelBinding.authority === 'ollama')) {
      throw probeError(
        'OLLAMA_SEMANTIC_EVALUATION_FORBIDDEN',
        'Ollama üretimi veya Ollama etiketli kayıt validation/test probe kanıtı olamaz.',
      )
    }
    const label = record.label === true ? 1 : record.label === false ? 0 : integer(record.label, 0, 1, 'Probe etiketi')
    return {
      lineage,
      labelBinding,
      label,
      scores: canonicalScores(record.scores, features),
    }
  })

  // Exact copied samples are one observation even when two import records use
  // different paths/ids. Preserve the union of their server-derived lineage
  // tokens so a later validation/test artifact cannot hide the overlap behind
  // the alternative locator. Conflicting labels or scores fail closed.
  const originRank = new Map([
    ['human-or-imported', 0],
    ['deterministic', 1],
    ['synthetic', 2],
    ['ollama', 3],
  ])
  const rawBySample = new Map()
  for (const record of rawRecords) {
    const previous = rawBySample.get(record.lineage.sampleToken)
    if (!previous) {
      rawBySample.set(record.lineage.sampleToken, record)
      continue
    }
    if (!sameCanonical(
      { labelBinding: previous.labelBinding, label: previous.label, scores: previous.scores },
      { labelBinding: record.labelBinding, label: record.label, scores: record.scores },
    )) {
      throw probeError('SEMANTIC_PROBE_DUPLICATE_CONFLICT', 'Aynı fiziksel örnek farklı etiket veya skorlarla yineleniyor.')
    }
    const origin = originRank.get(record.lineage.origin) > originRank.get(previous.lineage.origin)
      ? record.lineage.origin
      : previous.lineage.origin
    const tokens = [...new Set([...previous.lineage.tokens, ...record.lineage.tokens])].sort(compareText)
    if (tokens.length > limits.maximumLineageTokensPerRecord) {
      throw probeError('SEMANTIC_PROBE_LINEAGE_LIMIT_EXCEEDED', 'Kopya örneklerin birleşik kaynak soyu sınırı aşıyor.')
    }
    const lineagePayload = { origin, tokens, sampleToken: previous.lineage.sampleToken }
    rawBySample.set(record.lineage.sampleToken, {
      ...previous,
      lineage: deepFreeze({
        ...lineagePayload,
        fingerprint: stableRuntimeHash('plastic-region-semantic-lineage-v1', lineagePayload),
      }),
    })
  }
  const raw = [...rawBySample.values()]
  const producerBinding = canonicalProducerBinding(input.producerBinding)
  const observedUsesOllama = raw.some((record) =>
    record.lineage.origin === 'ollama' || record.labelBinding.authority === 'ollama')
  if (producerBinding.usesOllama !== observedUsesOllama) {
    throw probeError('SEMANTIC_PROBE_PRODUCER_BINDING_MISMATCH', 'Probe üretici Ollama beyanı kaynak/etiket soyu ile uyuşmuyor.')
  }
  if (split !== 'train' && producerBinding.usesOllama) {
    throw probeError('OLLAMA_SEMANTIC_EVALUATION_FORBIDDEN', 'Ollama validation/test probe üreticisi olamaz.')
  }
  if (producerBinding.mode === 'trusted-deterministic-pack' && raw.some((record) =>
    record.labelBinding.authority !== 'deterministic' ||
    record.labelBinding.contentHash !== producerBinding.probeSpecFingerprint)) {
    throw probeError(
      'SEMANTIC_PROBE_LABEL_AUTHORITY_MISMATCH',
      'Trusted deterministik probe etiketleri immutable probe-spec artifactine bağlı olmak zorunda.',
    )
  }

  const uniqueLineages = [...new Map(raw.map((record) => [record.lineage.fingerprint, record.lineage])).values()]
    .sort((left, right) => compareText(left.fingerprint, right.fingerprint))
  const assigned = assignIndependentGroups(uniqueLineages)
  const groupByLineage = new Map(assigned.map((lineage) => [lineage.fingerprint, lineage.independenceKey]))
  const lineages = assigned.sort((left, right) => compareText(left.fingerprint, right.fingerprint))
  const deduplicated = new Map()
  for (const record of raw) {
    const observationId = stableRuntimeHash(
      'plastic-region-semantic-observation-v1',
      hypothesis.hypothesisId,
      record.lineage.sampleToken,
    )
    const payload = {
      observationId,
      lineageFingerprint: record.lineage.fingerprint,
      independenceKey: groupByLineage.get(record.lineage.fingerprint),
      sampleToken: record.lineage.sampleToken,
      labelBinding: record.labelBinding,
      label: record.label,
      scores: record.scores,
    }
    const canonical = {
      ...payload,
      fingerprint: stableRuntimeHash('plastic-region-semantic-observation-record-v1', payload),
    }
    const previous = deduplicated.get(observationId)
    if (previous && !sameCanonical(previous, canonical)) {
      throw probeError('SEMANTIC_PROBE_DUPLICATE_CONFLICT', 'Aynı fiziksel örnek farklı etiket veya skorlarla yineleniyor.')
    }
    deduplicated.set(observationId, canonical)
  }
  const records = [...deduplicated.values()].sort((left, right) => compareText(left.observationId, right.observationId))
  const usedLineageFingerprints = new Set(records.map((record) => record.lineageFingerprint))
  const retainedLineages = lineages.filter((lineage) => usedLineageFingerprints.has(lineage.fingerprint))
  const origins = Object.fromEntries(['deterministic', 'human-or-imported', 'ollama', 'synthetic'].map((origin) => [
    origin,
    retainedLineages.filter((lineage) => lineage.origin === origin).length,
  ]))
  return {
    format: SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND,
    version: SEMANTIC_PROBE_VERSION,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    atlasMutationAllowed: false,
    hypothesis,
    atlasBinding,
    datasetBinding,
    producerBinding,
    split,
    limits,
    featureCount: features.length,
    observationCount: records.length,
    independentSourceCount: new Set(records.map((record) => record.independenceKey)).size,
    lineageCount: retainedLineages.length,
    origins,
    features,
    lineages: retainedLineages,
    records,
  }
}

export function createSemanticProbeObservationSet(input) {
  const payload = buildObservationPayload(input)
  const fingerprint = stableRuntimeHash('plastic-region-semantic-observations-v1', payload)
  const value = { ...payload, fingerprint }
  const bytes = assertArtifactBound(value, payload.limits.maximumArtifactBytes)
  return deepFreeze({ ...value, bytes })
}

function validateReadOnlyEnvelope(value, format) {
  if (!isRecord(value) || value.format !== format || value.version !== SEMANTIC_PROBE_VERSION ||
      value.layer !== SEMANTIC_PROBE_LAYER || value.interpretationOnly !== true ||
      value.trainingWritesAllowed !== false || value.routingMutationAllowed !== false ||
      value.atlasMutationAllowed !== false) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe artifacti salt-okunur yorum katmanı sözleşmesini taşımıyor.')
  }
}

export function validateSemanticProbeObservationSet(value) {
  validateReadOnlyEnvelope(value, SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND)
  const limits = canonicalLimits(value.limits)
  const hypothesis = canonicalHypothesis(value.hypothesis)
  const atlasBinding = canonicalAtlasBinding(value.atlasBinding)
  const datasetBinding = canonicalDatasetBinding(value.datasetBinding)
  const producerBinding = canonicalProducerBinding(value.producerBinding)
  const split = text(value.split, 'Probe veri bölümü', 16)
  if (!SPLITS.has(split)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe veri bölümü geçersiz.')
  const features = canonicalFeatures(value.features, atlasBinding, limits.maximumFeatures)
  if (!Array.isArray(value.lineages) || !Array.isArray(value.records) || !value.records.length ||
      value.records.length > limits.maximumRecords) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe kaynak veya kayıt tablosu geçersiz.')
  }
  const canonicalLineages = value.lineages.map((lineage) => canonicalStoredLineage(
    lineage,
    limits.maximumLineageTokensPerRecord,
  ))
  if (new Set(canonicalLineages.map((lineage) => lineage.fingerprint)).size !== canonicalLineages.length) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe kaynak soyları yinelenemez.')
  }
  const assigned = assignIndependentGroups(canonicalLineages)
    .sort((left, right) => compareText(left.fingerprint, right.fingerprint))
  const lineageByFingerprint = new Map(assigned.map((lineage) => [lineage.fingerprint, lineage]))
  const records = value.records.map((record) => {
    if (!isRecord(record)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe gözlem kaydı geçersiz.')
    const observationId = text(record.observationId, 'Probe gözlem kimliği')
    const lineageFingerprint = text(record.lineageFingerprint, 'Probe kaynak soy özeti')
    const lineage = lineageByFingerprint.get(lineageFingerprint)
    if (!lineage) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe gözlemi bilinmeyen kaynak soyuna bağlı.')
    const independenceKey = text(record.independenceKey, 'Probe bağımsız kaynak anahtarı')
    const sampleToken = text(record.sampleToken, 'Probe örnek izi')
    if (independenceKey !== lineage.independenceKey || sampleToken !== lineage.sampleToken ||
        observationId !== stableRuntimeHash('plastic-region-semantic-observation-v1', hypothesis.hypothesisId, sampleToken)) {
      throw probeError('INVALID_SEMANTIC_PROBE', 'Probe kaynak bağı veya türetilmiş bağımsızlık anahtarı uyuşmuyor.')
    }
    const labelBinding = canonicalLabelBinding(record.labelBinding)
    if (split !== 'train' && (lineage.origin === 'ollama' || labelBinding.authority === 'ollama')) {
      throw probeError('OLLAMA_SEMANTIC_EVALUATION_FORBIDDEN', 'Ollama validation/test probe kanıtı olamaz.')
    }
    const label = integer(record.label, 0, 1, 'Probe etiketi')
    if (!Array.isArray(record.scores) || record.scores.length !== features.length) {
      throw probeError('INVALID_SEMANTIC_PROBE', 'Saklanan probe skor boyutu uyuşmuyor.')
    }
    const scores = record.scores.map((score) => canonicalFloat(score, 0, 1, 'Probe yanıt skoru'))
    const payload = { observationId, lineageFingerprint, independenceKey, sampleToken, labelBinding, label, scores }
    const canonical = { ...payload, fingerprint: stableRuntimeHash('plastic-region-semantic-observation-record-v1', payload) }
    if (!sameCanonical(record, canonical)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe gözlem fingerprinti uyuşmuyor.')
    return canonical
  }).sort((left, right) => compareText(left.observationId, right.observationId))
  if (new Set(records.map((record) => record.observationId)).size !== records.length) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe gözlem kimlikleri yinelenemez.')
  }
  const observedUsesOllama = assigned.some((lineage) => lineage.origin === 'ollama') ||
    records.some((record) => record.labelBinding.authority === 'ollama')
  if (producerBinding.usesOllama !== observedUsesOllama || (split !== 'train' && observedUsesOllama)) {
    throw probeError('SEMANTIC_PROBE_PRODUCER_BINDING_MISMATCH', 'Probe üretici bağı kaynak/etiket soyuyla uyuşmuyor.')
  }
  if (producerBinding.mode === 'trusted-deterministic-pack' && records.some((record) =>
    record.labelBinding.authority !== 'deterministic' ||
    record.labelBinding.contentHash !== producerBinding.probeSpecFingerprint)) {
    throw probeError(
      'SEMANTIC_PROBE_LABEL_AUTHORITY_MISMATCH',
      'Trusted deterministik probe etiketi immutable probe-spec artifactine bağlı değil.',
    )
  }
  const usedLineages = new Set(records.map((record) => record.lineageFingerprint))
  if (assigned.some((lineage) => !usedLineages.has(lineage.fingerprint))) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe artifactinde kullanılmayan kaynak soyu saklanamaz.')
  }
  const origins = Object.fromEntries(['deterministic', 'human-or-imported', 'ollama', 'synthetic'].map((origin) => [
    origin,
    assigned.filter((lineage) => lineage.origin === origin).length,
  ]))
  const payload = {
    format: SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND,
    version: SEMANTIC_PROBE_VERSION,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    atlasMutationAllowed: false,
    hypothesis,
    atlasBinding,
    datasetBinding,
    producerBinding,
    split,
    limits,
    featureCount: features.length,
    observationCount: records.length,
    independentSourceCount: new Set(records.map((record) => record.independenceKey)).size,
    lineageCount: assigned.length,
    origins,
    features,
    lineages: assigned,
    records,
  }
  const fingerprint = stableRuntimeHash('plastic-region-semantic-observations-v1', payload)
  const withoutBytes = { ...payload, fingerprint }
  const bytes = assertArtifactBound(withoutBytes, limits.maximumArtifactBytes)
  const canonical = { ...withoutBytes, bytes }
  if (!sameCanonical(value, canonical)) throw probeError('INVALID_SEMANTIC_PROBE', 'Probe gözlem artifacti kanonik içerikle uyuşmuyor.')
  return value
}

function sourceBalancedWeights(records) {
  const counts = new Map()
  for (const record of records) counts.set(record.independenceKey, (counts.get(record.independenceKey) ?? 0) + 1)
  return records.map((record) => 1 / counts.get(record.independenceKey))
}

function weightedMean(values, weights) {
  const mass = weights.reduce((sum, weight) => sum + weight, 0)
  return values.reduce((sum, value, index) => sum + value * weights[index], 0) / Math.max(Number.EPSILON, mass)
}

function pava(probabilities, weights) {
  const blocks = probabilities.map((probability, index) => ({
    start: index,
    end: index,
    mass: weights[index],
    positive: probability * weights[index],
  }))
  for (let index = 0; index < blocks.length - 1;) {
    const left = blocks[index]
    const right = blocks[index + 1]
    if (left.positive / left.mass <= right.positive / right.mass + 1e-12) {
      index += 1
      continue
    }
    blocks.splice(index, 2, {
      start: left.start,
      end: right.end,
      mass: left.mass + right.mass,
      positive: left.positive + right.positive,
    })
    if (index > 0) index -= 1
  }
  const result = Array(probabilities.length)
  for (const block of blocks) {
    const probability = block.positive / block.mass
    for (let index = block.start; index <= block.end; index += 1) result[index] = probability
  }
  return result
}

function calibratorPrediction(calibrator, rawScore) {
  const score = calibrator.polarity === 'positive' ? rawScore : 1 - rawScore
  const index = Math.min(calibrator.binCount - 1, Math.floor(score * calibrator.binCount))
  return calibrator.probabilities[index]
}

function fitCalibrator(records, weights, featureIndex, binCount) {
  const labels = records.map((record) => record.label)
  const rawScores = records.map((record) => record.scores[featureIndex])
  const baseRate = weightedMean(labels, weights)
  const positiveScores = []
  const positiveWeights = []
  const negativeScores = []
  const negativeWeights = []
  records.forEach((record, index) => {
    if (record.label === 1) {
      positiveScores.push(rawScores[index]); positiveWeights.push(weights[index])
    } else {
      negativeScores.push(rawScores[index]); negativeWeights.push(weights[index])
    }
  })
  const positiveMean = positiveScores.length ? weightedMean(positiveScores, positiveWeights) : 0.5
  const negativeMean = negativeScores.length ? weightedMean(negativeScores, negativeWeights) : 0.5
  const polarity = positiveMean >= negativeMean ? 'positive' : 'negative'
  const binMass = Array(binCount).fill(0)
  const binPositive = Array(binCount).fill(0)
  records.forEach((record, index) => {
    const score = polarity === 'positive' ? rawScores[index] : 1 - rawScores[index]
    const bin = Math.min(binCount - 1, Math.floor(score * binCount))
    binMass[bin] += weights[index]
    binPositive[bin] += weights[index] * record.label
  })
  const priorStrength = 2
  const smoothedWeights = binMass.map((mass) => mass + priorStrength)
  const probabilities = pava(
    binMass.map((mass, index) => (binPositive[index] + baseRate * priorStrength) / (mass + priorStrength)),
    smoothedWeights,
  ).map((value) => canonicalFloat(value, 0, 1, 'Kalibre olasılığı'))
  const calibrator = {
    method: 'source-balanced-isotonic-bins-v1',
    binCount,
    polarity,
    baseRate: canonicalFloat(baseRate, 0, 1, 'Kalibrasyon taban oranı'),
    probabilities,
    sourceBalancedBinMass: binMass.map((value) => canonicalFloat(value, 0, HARD_LIMITS.maximumRecords, 'Kalibrasyon bin kütlesi')),
  }
  const predictions = rawScores.map((score) => calibratorPrediction(calibrator, score))
  const brier = weightedMean(predictions.map((prediction, index) => (prediction - labels[index]) ** 2), weights)
  return {
    calibrator: deepFreeze(calibrator),
    trainingBrier: canonicalFloat(brier, 0, 1, 'Kalibrasyon Brier skoru'),
    trainingGain: canonicalFloat(baseRate * (1 - baseRate) - brier, -1, 1, 'Kalibrasyon kazancı'),
  }
}

function footprintDistance(left, right) {
  const leftWeights = new Map(left.map((entry) => [entry.pageIndex, entry.weight]))
  const rightWeights = new Map(right.map((entry) => [entry.pageIndex, entry.weight]))
  const pages = new Set([...leftWeights.keys(), ...rightWeights.keys()])
  let difference = 0
  let mass = 0
  for (const page of pages) {
    const a = leftWeights.get(page) ?? 0
    const b = rightWeights.get(page) ?? 0
    difference += Math.abs(a - b)
    mass += a + b
  }
  return canonicalFloat(difference / Math.max(Number.EPSILON, mass), 0, 1, 'Ayak izi uzaklığı')
}

function canonicalCalibrationPolicy(value = {}) {
  return deepFreeze({
    binCount: integer(value.binCount ?? 8, 2, 32, 'Kalibrasyon bin sayısı'),
    minimumTrainingObservations: integer(value.minimumTrainingObservations ?? 8, 2, HARD_LIMITS.maximumRecords, 'Asgari train gözlemi'),
    minimumTrainingIndependentSources: integer(value.minimumTrainingIndependentSources ?? 2, 1, HARD_LIMITS.maximumRecords, 'Asgari train kaynak sayısı'),
    minimumPositiveIndependentSources: integer(value.minimumPositiveIndependentSources ?? 2, 1, HARD_LIMITS.maximumRecords, 'Asgari pozitif train kaynak sayısı'),
    minimumNegativeIndependentSources: integer(value.minimumNegativeIndependentSources ?? 2, 1, HARD_LIMITS.maximumRecords, 'Asgari negatif train kaynak sayısı'),
    maximumShamFootprintDistance: canonicalFloat(value.maximumShamFootprintDistance ?? 0.2, 0, 1, 'Sham ayak izi uzaklığı'),
  })
}

function canonicalArtifactReference(value, label, expectedFingerprint) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', `${label} artifact bağı eksik.`)
  const result = {
    artifactId: text(value.artifactId, `${label} artifact kimliği`),
    contentHash: text(value.contentHash, `${label} içerik özeti`),
    fingerprint: text(value.fingerprint ?? expectedFingerprint, `${label} fingerprinti`),
  }
  if (result.contentHash !== expectedFingerprint || result.fingerprint !== expectedFingerprint) {
    throw probeError('SEMANTIC_PROBE_ARTIFACT_BINDING_MISMATCH', `${label} bağı immutable probe fingerprintiyle uyuşmuyor.`)
  }
  return deepFreeze(result)
}

function featureReference(feature) {
  return deepFreeze({
    featureId: feature.featureId,
    responseKind: feature.responseKind,
    featureFingerprint: feature.fingerprint,
    footprintFingerprint: feature.footprintFingerprint,
  })
}

export function createSemanticProbeCalibration(input) {
  const observations = validateSemanticProbeObservationSet(input.observations)
  if (observations.split !== 'train') {
    throw probeError('SEMANTIC_PROBE_TRAIN_SPLIT_REQUIRED', 'Kalibrasyon yalnız train probe artifactinden kurulabilir.')
  }
  const observationArtifact = canonicalArtifactReference(
    input.observationArtifact,
    'Train gözlemi',
    observations.fingerprint,
  )
  const policy = canonicalCalibrationPolicy(input.policy)
  const weights = sourceBalancedWeights(observations.records)
  const featureFits = observations.features.map((feature, index) => ({
    feature,
    featureIndex: index,
    ...fitCalibrator(observations.records, weights, index, policy.binCount),
  }))
  const withSham = featureFits.map((fit) => {
    const candidates = featureFits
      .filter((candidate) => candidate.feature.fingerprint !== fit.feature.fingerprint)
      .map((candidate) => ({
        candidate,
        distance: footprintDistance(fit.feature.footprint, candidate.feature.footprint),
        tie: stableRuntimeHash('plastic-region-semantic-sham-choice-v1', fit.feature.fingerprint, candidate.feature.fingerprint),
      }))
      .filter((candidate) => candidate.distance <= policy.maximumShamFootprintDistance)
      .sort((left, right) => left.distance - right.distance || compareText(left.tie, right.tie))
    return candidates.length ? { fit, sham: candidates[0].candidate, distance: candidates[0].distance } : null
  }).filter(Boolean)
  if (!withSham.length) {
    throw probeError('SEMANTIC_PROBE_SHAM_MATCH_UNAVAILABLE', 'Etiketten bağımsız, ayak izi eşleşen bir sham özellik bulunamadı.')
  }
  withSham.sort((left, right) =>
    right.fit.trainingGain - left.fit.trainingGain || compareText(left.fit.feature.fingerprint, right.fit.feature.fingerprint))
  const selected = withSham[0]
  const lineageTokens = [...new Set(observations.lineages.flatMap((lineage) => lineage.tokens))].sort(compareText)
  const sampleTokens = [...new Set(observations.records.map((record) => record.sampleToken))].sort(compareText)
  const positiveIndependentSourceCount = new Set(observations.records
    .filter((record) => record.label === 1).map((record) => record.independenceKey)).size
  const negativeIndependentSourceCount = new Set(observations.records
    .filter((record) => record.label === 0).map((record) => record.independenceKey)).size
  const status = observations.observationCount >= policy.minimumTrainingObservations &&
    observations.independentSourceCount >= policy.minimumTrainingIndependentSources &&
    positiveIndependentSourceCount >= policy.minimumPositiveIndependentSources &&
    negativeIndependentSourceCount >= policy.minimumNegativeIndependentSources
    ? 'provisional'
    : 'unknown'
  const payload = {
    format: SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND,
    version: SEMANTIC_PROBE_VERSION,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    atlasMutationAllowed: false,
    status,
    hypothesis: observations.hypothesis,
    atlasBinding: observations.atlasBinding,
    datasetBinding: observations.datasetBinding,
    producerBinding: observations.producerBinding,
    policy,
    trainingObservationArtifact: observationArtifact,
    trainingObservationCount: observations.observationCount,
    trainingIndependentSourceCount: observations.independentSourceCount,
    positiveIndependentSourceCount,
    negativeIndependentSourceCount,
    trainingLineageTokens: lineageTokens,
    trainingSampleTokens: sampleTokens,
    selection: {
      split: 'train',
      candidateCount: featureFits.length,
      labelUsedOnlyForTargetSelectionAndCalibration: true,
      shamSelectedWithoutLabels: true,
    },
    target: {
      feature: featureReference(selected.fit.feature),
      calibrator: selected.fit.calibrator,
      trainingBrier: selected.fit.trainingBrier,
      trainingGainVsBaseRate: selected.fit.trainingGain,
    },
    sham: {
      feature: featureReference(selected.sham.feature),
      calibrator: selected.sham.calibrator,
      trainingBrier: selected.sham.trainingBrier,
      trainingGainVsBaseRate: selected.sham.trainingGain,
      footprintDistance: selected.distance,
    },
  }
  const fingerprint = stableRuntimeHash('plastic-region-semantic-calibration-v1', payload)
  const value = { ...payload, fingerprint }
  const bytes = assertArtifactBound(value, observations.limits.maximumArtifactBytes)
  return deepFreeze({ ...value, bytes })
}

function validateCalibrator(value, policy) {
  if (!isRecord(value) || value.method !== 'source-balanced-isotonic-bins-v1' ||
      value.binCount !== policy.binCount || !['positive', 'negative'].includes(value.polarity) ||
      !Array.isArray(value.probabilities) || value.probabilities.length !== value.binCount ||
      !Array.isArray(value.sourceBalancedBinMass) || value.sourceBalancedBinMass.length !== value.binCount) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Kalibrasyon modeli biçimi geçersiz.')
  }
  const probabilities = value.probabilities.map((item) => canonicalFloat(item, 0, 1, 'Kalibrasyon olasılığı'))
  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index] + 1e-7 < probabilities[index - 1]) {
      throw probeError('INVALID_SEMANTIC_PROBE', 'Kalibrasyon olasılıkları monoton değil.')
    }
  }
  return deepFreeze({
    method: value.method,
    binCount: value.binCount,
    polarity: value.polarity,
    baseRate: canonicalFloat(value.baseRate, 0, 1, 'Kalibrasyon taban oranı'),
    probabilities,
    sourceBalancedBinMass: value.sourceBalancedBinMass.map((item) =>
      canonicalFloat(item, 0, HARD_LIMITS.maximumRecords, 'Kalibrasyon bin kütlesi')),
  })
}

function validateFeatureReference(value, label) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', `${label} özellik bağı geçersiz.`)
  if (!RESPONSE_KINDS.has(value.responseKind)) {
    throw probeError('INVALID_SEMANTIC_PROBE', `${label} özellik bağı desteklenmeyen yanıt türü taşıyor.`)
  }
  return deepFreeze({
    featureId: text(value.featureId, `${label} özellik kimliği`),
    responseKind: value.responseKind,
    featureFingerprint: text(value.featureFingerprint, `${label} özellik fingerprinti`),
    footprintFingerprint: text(value.footprintFingerprint, `${label} ayak izi fingerprinti`),
  })
}

function canonicalStoredTokens(value, label, maximum, prefix) {
  if (!Array.isArray(value) || !value.length || value.length > maximum) {
    throw probeError('INVALID_SEMANTIC_PROBE', `${label} eksik veya sınır dışı.`)
  }
  const tokens = value.map((item) => text(item, label))
  if (new Set(tokens).size !== tokens.length ||
      tokens.some((token, index) => token !== value[index] || (prefix && !token.startsWith(prefix))) ||
      [...tokens].sort(compareText).some((token, index) => token !== tokens[index])) {
    throw probeError('INVALID_SEMANTIC_PROBE', `${label} kanonik değil.`)
  }
  return deepFreeze(tokens)
}

function canonicalCalibrationBranch(value, label, policy, sham = false) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', `${label} kalibrasyonu eksik.`)
  const result = {
    feature: validateFeatureReference(value.feature, label),
    calibrator: validateCalibrator(value.calibrator, policy),
    trainingBrier: canonicalFloat(value.trainingBrier, 0, 1, `${label} train Brier`),
    trainingGainVsBaseRate: canonicalFloat(value.trainingGainVsBaseRate, -1, 1, `${label} train kazancı`),
  }
  if (sham) {
    result.footprintDistance = canonicalFloat(
      value.footprintDistance,
      0,
      policy.maximumShamFootprintDistance,
      'Sham ayak izi uzaklığı',
    )
  }
  return deepFreeze(result)
}

export function validateSemanticProbeCalibration(value, observations) {
  validateReadOnlyEnvelope(value, SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND)
  if (!HYPOTHESIS_STATUSES.has(value.status) || value.status === 'calibrated') {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Train kalibrasyonu yalnız unknown veya provisional olabilir.')
  }
  const policy = canonicalCalibrationPolicy(value.policy)
  const hypothesis = canonicalHypothesis(value.hypothesis)
  const atlasBinding = canonicalAtlasBinding(value.atlasBinding)
  const datasetBinding = canonicalDatasetBinding(value.datasetBinding)
  const producerBinding = canonicalProducerBinding(value.producerBinding)
  const trainingObservationArtifact = canonicalArtifactReference(
    value.trainingObservationArtifact,
    'Train gözlemi',
    value.trainingObservationArtifact?.fingerprint,
  )
  const trainingObservationCount = integer(
    value.trainingObservationCount,
    1,
    HARD_LIMITS.maximumRecords,
    'Train gözlem sayısı',
  )
  const trainingIndependentSourceCount = integer(
    value.trainingIndependentSourceCount,
    1,
    HARD_LIMITS.maximumRecords,
    'Train kaynak sayısı',
  )
  const positiveIndependentSourceCount = integer(
    value.positiveIndependentSourceCount,
    0,
    HARD_LIMITS.maximumRecords,
    'Pozitif train kaynak sayısı',
  )
  const negativeIndependentSourceCount = integer(
    value.negativeIndependentSourceCount,
    0,
    HARD_LIMITS.maximumRecords,
    'Negatif train kaynak sayısı',
  )
  if (trainingIndependentSourceCount > trainingObservationCount ||
      positiveIndependentSourceCount > trainingIndependentSourceCount ||
      negativeIndependentSourceCount > trainingIndependentSourceCount) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Train kaynak sayaçları gözlem kütlesini aşıyor.')
  }
  const trainingLineageTokens = canonicalStoredTokens(
    value.trainingLineageTokens,
    'Train kaynak soy izi',
    HARD_LIMITS.maximumRecords * HARD_LIMITS.maximumLineageTokensPerRecord,
    'SLT-',
  )
  const trainingSampleTokens = canonicalStoredTokens(
    value.trainingSampleTokens,
    'Train örnek izi',
    HARD_LIMITS.maximumRecords,
    'SLT-',
  )
  if (trainingSampleTokens.length !== trainingObservationCount) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Train örnek izi sayısı gözlem sayısıyla uyuşmuyor.')
  }
  if (!isRecord(value.selection) || value.selection.split !== 'train' ||
      value.selection.labelUsedOnlyForTargetSelectionAndCalibration !== true ||
      value.selection.shamSelectedWithoutLabels !== true) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Kalibrasyon seçim sözleşmesi geçersiz.')
  }
  const selection = deepFreeze({
    split: 'train',
    candidateCount: integer(value.selection.candidateCount, 2, HARD_LIMITS.maximumFeatures, 'Kalibrasyon aday sayısı'),
    labelUsedOnlyForTargetSelectionAndCalibration: true,
    shamSelectedWithoutLabels: true,
  })
  const target = canonicalCalibrationBranch(value.target, 'Hedef', policy)
  const sham = canonicalCalibrationBranch(value.sham, 'Sham', policy, true)
  if (target.feature.featureFingerprint === sham.feature.featureFingerprint ||
      target.calibrator.baseRate !== sham.calibrator.baseRate) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Hedef ve sham ayrı özellikler olmalı ve aynı train taban oranını taşımalı.')
  }
  const expectedStatus = trainingObservationCount >= policy.minimumTrainingObservations &&
    trainingIndependentSourceCount >= policy.minimumTrainingIndependentSources &&
    positiveIndependentSourceCount >= policy.minimumPositiveIndependentSources &&
    negativeIndependentSourceCount >= policy.minimumNegativeIndependentSources
    ? 'provisional'
    : 'unknown'
  if (value.status !== expectedStatus) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Kalibrasyon durumu kaynak ve sınıf desteğiyle uyuşmuyor.')
  }
  const payload = {
    format: SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND,
    version: SEMANTIC_PROBE_VERSION,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    atlasMutationAllowed: false,
    status: expectedStatus,
    hypothesis,
    atlasBinding,
    datasetBinding,
    producerBinding,
    policy,
    trainingObservationArtifact,
    trainingObservationCount,
    trainingIndependentSourceCount,
    positiveIndependentSourceCount,
    negativeIndependentSourceCount,
    trainingLineageTokens,
    trainingSampleTokens,
    selection,
    target,
    sham,
  }
  const fingerprint = stableRuntimeHash('plastic-region-semantic-calibration-v1', payload)
  const withoutBytes = { ...payload, fingerprint }
  const bytes = assertArtifactBound(withoutBytes, HARD_LIMITS.maximumArtifactBytes)
  const canonical = { ...withoutBytes, bytes }
  if (!sameCanonical(value, canonical)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Kalibrasyon artifacti kanonik içerik, fingerprint veya boyutla uyuşmuyor.')
  }
  if (observations) {
    const rebuilt = createSemanticProbeCalibration({
      observations,
      observationArtifact: trainingObservationArtifact,
      policy,
    })
    if (!sameCanonical(value, rebuilt)) {
      throw probeError('INVALID_SEMANTIC_PROBE', 'Kalibrasyon bağlı train gözleminden yeniden üretilemedi.')
    }
  }
  return value
}

function evaluationWeights(records) {
  return sourceBalancedWeights(records)
}

function brierAndEce(records, weights, predictions, binCount) {
  const brier = weightedMean(records.map((record, index) => (predictions[index] - record.label) ** 2), weights)
  const bins = Array.from({ length: binCount }, () => ({ mass: 0, prediction: 0, label: 0 }))
  records.forEach((record, index) => {
    const bin = Math.min(binCount - 1, Math.floor(predictions[index] * binCount))
    bins[bin].mass += weights[index]
    bins[bin].prediction += weights[index] * predictions[index]
    bins[bin].label += weights[index] * record.label
  })
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  const ece = bins.reduce((sum, bin) => {
    if (!bin.mass) return sum
    return sum + bin.mass / total * Math.abs(bin.prediction / bin.mass - bin.label / bin.mass)
  }, 0)
  return {
    brier: canonicalFloat(brier, 0, 1, 'Probe Brier skoru'),
    ece: canonicalFloat(ece, 0, 1, 'Probe kalibrasyon hatası'),
  }
}

function seededRandom(seed) {
  let state = seed >>> 0 || 0x9e3779b9
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 0x1_0000_0000
  }
}

function percentile(sorted, quantile) {
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor(quantile * sorted.length)))
  return sorted[index]
}

function groupedInterval(rows, field, samples, confidenceLevel, seed) {
  const groups = new Map()
  for (const row of rows) {
    const values = groups.get(row.independenceKey) ?? []
    values.push(row[field])
    groups.set(row.independenceKey, values)
  }
  const groupValues = [...groups.entries()].sort(([left], [right]) => compareText(left, right))
    .map(([, values]) => values.reduce((sum, value) => sum + value, 0) / values.length)
  const point = groupValues.reduce((sum, value) => sum + value, 0) / groupValues.length
  const random = seededRandom(seed)
  const distribution = []
  for (let sample = 0; sample < samples; sample += 1) {
    let total = 0
    for (let index = 0; index < groupValues.length; index += 1) {
      total += groupValues[Math.floor(random() * groupValues.length)]
    }
    distribution.push(total / groupValues.length)
  }
  distribution.sort((left, right) => left - right)
  const tail = (1 - confidenceLevel) / 2
  return deepFreeze({
    point: canonicalFloat(point, -1, 1, 'Probe aralık noktası'),
    low: canonicalFloat(percentile(distribution, tail), -1, 1, 'Probe aralık alt sınırı'),
    high: canonicalFloat(percentile(distribution, 1 - tail), -1, 1, 'Probe aralık üst sınırı'),
    confidenceLevel: canonicalFloat(confidenceLevel, 0.5, 0.999, 'Probe güven düzeyi'),
    bootstrapSamples: samples,
    independentSourceCount: groupValues.length,
  })
}

function canonicalEvaluationPolicy(value = {}) {
  return deepFreeze({
    minimumEvaluationObservations: integer(value.minimumEvaluationObservations ?? 8, 2, HARD_LIMITS.maximumRecords, 'Asgari evaluation gözlemi'),
    minimumEvaluationIndependentSources: integer(value.minimumEvaluationIndependentSources ?? 3, 1, HARD_LIMITS.maximumRecords, 'Asgari evaluation kaynak sayısı'),
    minimumPositiveIndependentSources: integer(value.minimumPositiveIndependentSources ?? 2, 1, HARD_LIMITS.maximumRecords, 'Asgari pozitif evaluation kaynak sayısı'),
    minimumNegativeIndependentSources: integer(value.minimumNegativeIndependentSources ?? 2, 1, HARD_LIMITS.maximumRecords, 'Asgari negatif evaluation kaynak sayısı'),
    minimumGainVsBaseline: canonicalFloat(value.minimumGainVsBaseline ?? 0.01, -1, 1, 'Asgari taban kazancı'),
    minimumGainVsSham: canonicalFloat(value.minimumGainVsSham ?? 0.01, -1, 1, 'Asgari sham kazancı'),
    maximumExpectedCalibrationError: canonicalFloat(value.maximumExpectedCalibrationError ?? 0.15, 0, 1, 'Azami calibration error'),
    bootstrapSamples: integer(value.bootstrapSamples ?? 1_000, 100, HARD_LIMITS.maximumBootstrapSamples, 'Bootstrap örnek sayısı'),
    confidenceLevel: canonicalFloat(value.confidenceLevel ?? 0.95, 0.5, 0.999, 'Güven düzeyi'),
    seed: integer(value.seed ?? 0x51a7, 0, 0xffff_ffff, 'Probe evaluation seed'),
  })
}

function evaluationReasons(calibrationStatus, metrics, policy) {
  const reasons = []
  if (calibrationStatus === 'unknown') reasons.push('INSUFFICIENT_TRAIN_SUPPORT')
  if (metrics.observationCount < policy.minimumEvaluationObservations) reasons.push('INSUFFICIENT_EVALUATION_OBSERVATIONS')
  if (metrics.independentSourceCount < policy.minimumEvaluationIndependentSources) reasons.push('INSUFFICIENT_INDEPENDENT_SOURCES')
  if (metrics.positiveIndependentSourceCount < policy.minimumPositiveIndependentSources) reasons.push('INSUFFICIENT_POSITIVE_SOURCES')
  if (metrics.negativeIndependentSourceCount < policy.minimumNegativeIndependentSources) reasons.push('INSUFFICIENT_NEGATIVE_SOURCES')
  if (metrics.gainVsBaseline.low < policy.minimumGainVsBaseline) reasons.push('BASELINE_GAIN_NOT_CONFIRMED')
  if (metrics.gainVsSham.low < policy.minimumGainVsSham) reasons.push('SHAM_ADVANTAGE_NOT_CONFIRMED')
  if (metrics.target.ece > policy.maximumExpectedCalibrationError) reasons.push('CALIBRATION_ERROR_TOO_HIGH')
  return reasons
}

function evaluationStatus(reasons) {
  if (!reasons.length) return 'calibrated'
  return reasons.some((reason) => reason.startsWith('INSUFFICIENT')) ? 'unknown' : 'provisional'
}

function assertEvaluationCompatibility(calibration, train, evaluation) {
  if (evaluation.split === 'train') {
    throw probeError('SEMANTIC_PROBE_EVALUATION_SPLIT_REQUIRED', 'Evaluation probe artifacti validation veya test olmalı.')
  }
  for (const [label, left, right] of [
    ['hipotez', calibration.hypothesis.hypothesisId, evaluation.hypothesis.hypothesisId],
    ['atlas', calibration.atlasBinding.atlasFingerprint, evaluation.atlasBinding.atlasFingerprint],
    ['temsil', calibration.atlasBinding.representationSpaceId, evaluation.atlasBinding.representationSpaceId],
    ['veri kümesi', calibration.datasetBinding.datasetId, evaluation.datasetBinding.datasetId],
    ['bölümleme manifesti', calibration.datasetBinding.partitionManifestFingerprint, evaluation.datasetBinding.partitionManifestFingerprint],
    ['üretici', calibration.producerBinding.producerFingerprint, evaluation.producerBinding.producerFingerprint],
    ['girdi manifesti', calibration.producerBinding.inputManifestHash, evaluation.producerBinding.inputManifestHash],
    ['probe paketi', calibration.producerBinding.probeSpecFingerprint, evaluation.producerBinding.probeSpecFingerprint],
    ['train artifact', calibration.trainingObservationArtifact.fingerprint, train.fingerprint],
  ]) {
    if (left !== right) throw probeError('SEMANTIC_PROBE_BINDING_MISMATCH', `Probe ${label} bağı train/evaluation arasında uyuşmuyor.`)
  }
  const evaluationTokens = new Set(evaluation.lineages.flatMap((lineage) => lineage.tokens))
  const leakedToken = calibration.trainingLineageTokens.find((token) => evaluationTokens.has(token))
  const evaluationSamples = new Set(evaluation.records.map((record) => record.sampleToken))
  const leakedSample = calibration.trainingSampleTokens.find((token) => evaluationSamples.has(token))
  if (leakedToken || leakedSample) {
    throw probeError('SEMANTIC_PROBE_SPLIT_LEAKAGE', 'Aynı fiziksel kaynak veya kopya içerik train ile validation/test arasında sızmış.', {
      leakedToken: leakedToken ?? null,
      leakedSample: leakedSample ?? null,
    })
  }
}

export function createSemanticProbeEvaluationEvidence(input) {
  const train = validateSemanticProbeObservationSet(input.trainingObservations)
  const evaluation = validateSemanticProbeObservationSet(input.evaluationObservations)
  const calibration = validateSemanticProbeCalibration(input.calibration, train)
  assertEvaluationCompatibility(calibration, train, evaluation)
  const evaluationArtifact = canonicalArtifactReference(
    input.evaluationArtifact,
    'Evaluation gözlemi',
    evaluation.fingerprint,
  )
  const calibrationArtifact = canonicalArtifactReference(
    input.calibrationArtifact,
    'Kalibrasyon',
    calibration.fingerprint,
  )
  const policy = canonicalEvaluationPolicy(input.policy)
  const featureIndex = new Map(evaluation.features.map((feature, index) => [feature.fingerprint, index]))
  const targetIndex = featureIndex.get(calibration.target.feature.featureFingerprint)
  const shamIndex = featureIndex.get(calibration.sham.feature.featureFingerprint)
  if (targetIndex === undefined || shamIndex === undefined ||
      evaluation.features[targetIndex].footprintFingerprint !== calibration.target.feature.footprintFingerprint ||
      evaluation.features[shamIndex].footprintFingerprint !== calibration.sham.feature.footprintFingerprint) {
    throw probeError('SEMANTIC_PROBE_FEATURE_BINDING_MISMATCH', 'Evaluation özellikleri kalibre hedef/sham ayak izleriyle uyuşmuyor.')
  }
  const weights = evaluationWeights(evaluation.records)
  const baselinePredictions = evaluation.records.map(() => calibration.target.calibrator.baseRate)
  const targetPredictions = evaluation.records.map((record) =>
    calibratorPrediction(calibration.target.calibrator, record.scores[targetIndex]))
  const shamPredictions = evaluation.records.map((record) =>
    calibratorPrediction(calibration.sham.calibrator, record.scores[shamIndex]))
  const baseline = brierAndEce(evaluation.records, weights, baselinePredictions, calibration.policy.binCount)
  const target = brierAndEce(evaluation.records, weights, targetPredictions, calibration.policy.binCount)
  const sham = brierAndEce(evaluation.records, weights, shamPredictions, calibration.policy.binCount)
  const rows = evaluation.records.map((record, index) => ({
    independenceKey: record.independenceKey,
    gainVsBaseline: (baselinePredictions[index] - record.label) ** 2 - (targetPredictions[index] - record.label) ** 2,
    gainVsSham: (shamPredictions[index] - record.label) ** 2 - (targetPredictions[index] - record.label) ** 2,
  }))
  const gainVsBaseline = groupedInterval(
    rows,
    'gainVsBaseline',
    policy.bootstrapSamples,
    policy.confidenceLevel,
    policy.seed,
  )
  const gainVsSham = groupedInterval(
    rows,
    'gainVsSham',
    policy.bootstrapSamples,
    policy.confidenceLevel,
    policy.seed ^ 0xa5a5_5a5a,
  )
  const positiveIndependentSourceCount = new Set(evaluation.records
    .filter((record) => record.label === 1).map((record) => record.independenceKey)).size
  const negativeIndependentSourceCount = new Set(evaluation.records
    .filter((record) => record.label === 0).map((record) => record.independenceKey)).size
  const metrics = {
    observationCount: evaluation.observationCount,
    independentSourceCount: evaluation.independentSourceCount,
    positiveIndependentSourceCount,
    negativeIndependentSourceCount,
    baseline,
    target,
    sham,
    gainVsBaseline,
    gainVsSham,
  }
  const reasons = evaluationReasons(calibration.status, metrics, policy)
  const status = evaluationStatus(reasons)
  const replayBinding = {
    trainObservationFingerprint: train.fingerprint,
    evaluationObservationFingerprint: evaluation.fingerprint,
    calibrationFingerprint: calibration.fingerprint,
    atlasFingerprint: evaluation.atlasBinding.atlasFingerprint,
    partitionManifestFingerprint: evaluation.datasetBinding.partitionManifestFingerprint,
    producerFingerprint: evaluation.producerBinding.producerFingerprint,
    inputManifestHash: evaluation.producerBinding.inputManifestHash,
    probeSpecFingerprint: evaluation.producerBinding.probeSpecFingerprint,
    policyFingerprint: stableRuntimeHash('plastic-region-semantic-evaluation-policy-v1', policy),
  }
  const payload = {
    format: SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND,
    version: SEMANTIC_PROBE_VERSION,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    atlasMutationAllowed: false,
    releaseAuthority: false,
    calibrationStatus: calibration.status,
    status,
    reasons,
    hypothesis: evaluation.hypothesis,
    atlasBinding: evaluation.atlasBinding,
    datasetBinding: evaluation.datasetBinding,
    producerBinding: evaluation.producerBinding,
    evaluationSplit: evaluation.split,
    policy,
    trainingObservationArtifact: calibration.trainingObservationArtifact,
    evaluationObservationArtifact: evaluationArtifact,
    calibrationArtifact,
    replayBinding,
    metrics,
  }
  const fingerprint = stableRuntimeHash('plastic-region-semantic-evaluation-v1', payload)
  const value = { ...payload, fingerprint }
  const maximumBytes = Math.min(train.limits.maximumArtifactBytes, evaluation.limits.maximumArtifactBytes)
  const bytes = assertArtifactBound(value, maximumBytes)
  return deepFreeze({ ...value, bytes })
}

export function replaySemanticProbeEvaluation(value, dependencies) {
  const rebuilt = createSemanticProbeEvaluationEvidence({
    trainingObservations: dependencies.trainingObservations,
    evaluationObservations: dependencies.evaluationObservations,
    calibration: dependencies.calibration,
    evaluationArtifact: value.evaluationObservationArtifact,
    calibrationArtifact: value.calibrationArtifact,
    policy: value.policy,
  })
  if (!sameCanonical(value, rebuilt)) {
    throw probeError('SEMANTIC_PROBE_REPLAY_MISMATCH', 'Probe evaluation aynı girdilerle yeniden üretilemedi.')
  }
  return deepFreeze({ matched: true, fingerprint: rebuilt.fingerprint, replayBinding: rebuilt.replayBinding })
}

function canonicalProbabilityMetrics(value, label) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', `${label} metrikleri eksik.`)
  return deepFreeze({
    brier: canonicalFloat(value.brier, 0, 1, `${label} Brier`),
    ece: canonicalFloat(value.ece, 0, 1, `${label} ECE`),
  })
}

function canonicalGainInterval(value, label, policy, independentSourceCount) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', `${label} güven aralığı eksik.`)
  const result = {
    point: canonicalFloat(value.point, -1, 1, `${label} nokta tahmini`),
    low: canonicalFloat(value.low, -1, 1, `${label} alt sınırı`),
    high: canonicalFloat(value.high, -1, 1, `${label} üst sınırı`),
    confidenceLevel: canonicalFloat(value.confidenceLevel, 0.5, 0.999, `${label} güven düzeyi`),
    bootstrapSamples: integer(value.bootstrapSamples, 100, HARD_LIMITS.maximumBootstrapSamples, `${label} bootstrap sayısı`),
    independentSourceCount: integer(value.independentSourceCount, 1, HARD_LIMITS.maximumRecords, `${label} kaynak sayısı`),
  }
  if (result.low > result.high || result.confidenceLevel !== policy.confidenceLevel ||
      result.bootstrapSamples !== policy.bootstrapSamples ||
      result.independentSourceCount !== independentSourceCount) {
    throw probeError('INVALID_SEMANTIC_PROBE', `${label} güven aralığı evaluation politikasıyla uyuşmuyor.`)
  }
  return deepFreeze(result)
}

function canonicalEvaluationMetrics(value, policy) {
  if (!isRecord(value)) throw probeError('INVALID_SEMANTIC_PROBE', 'Evaluation metrikleri eksik.')
  const observationCount = integer(value.observationCount, 1, HARD_LIMITS.maximumRecords, 'Evaluation gözlem sayısı')
  const independentSourceCount = integer(
    value.independentSourceCount,
    1,
    HARD_LIMITS.maximumRecords,
    'Evaluation kaynak sayısı',
  )
  const positiveIndependentSourceCount = integer(
    value.positiveIndependentSourceCount,
    0,
    HARD_LIMITS.maximumRecords,
    'Pozitif evaluation kaynak sayısı',
  )
  const negativeIndependentSourceCount = integer(
    value.negativeIndependentSourceCount,
    0,
    HARD_LIMITS.maximumRecords,
    'Negatif evaluation kaynak sayısı',
  )
  if (independentSourceCount > observationCount ||
      positiveIndependentSourceCount > independentSourceCount ||
      negativeIndependentSourceCount > independentSourceCount) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Evaluation kaynak sayaçları gözlem kütlesini aşıyor.')
  }
  return deepFreeze({
    observationCount,
    independentSourceCount,
    positiveIndependentSourceCount,
    negativeIndependentSourceCount,
    baseline: canonicalProbabilityMetrics(value.baseline, 'Taban'),
    target: canonicalProbabilityMetrics(value.target, 'Hedef'),
    sham: canonicalProbabilityMetrics(value.sham, 'Sham'),
    gainVsBaseline: canonicalGainInterval(value.gainVsBaseline, 'Tabana karşı kazanç', policy, independentSourceCount),
    gainVsSham: canonicalGainInterval(value.gainVsSham, 'Sham karşı kazanç', policy, independentSourceCount),
  })
}

export function validateSemanticProbeEvaluationEvidence(value, dependencies) {
  validateReadOnlyEnvelope(value, SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND)
  if (value.releaseAuthority !== false || !HYPOTHESIS_STATUSES.has(value.status) ||
      !['unknown', 'provisional'].includes(value.calibrationStatus)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe evaluation yetki veya durum alanı geçersiz.')
  }
  const policy = canonicalEvaluationPolicy(value.policy)
  const hypothesis = canonicalHypothesis(value.hypothesis)
  const atlasBinding = canonicalAtlasBinding(value.atlasBinding)
  const datasetBinding = canonicalDatasetBinding(value.datasetBinding)
  const producerBinding = canonicalProducerBinding(value.producerBinding)
  const evaluationSplit = text(value.evaluationSplit, 'Evaluation veri bölümü', 16)
  if (!['validation', 'test'].includes(evaluationSplit) || producerBinding.usesOllama) {
    throw probeError('OLLAMA_SEMANTIC_EVALUATION_FORBIDDEN', 'Evaluation validation/test ve Ollama’sız olmak zorunda.')
  }
  const trainingObservationArtifact = canonicalArtifactReference(
    value.trainingObservationArtifact,
    'Train gözlemi',
    value.trainingObservationArtifact?.fingerprint,
  )
  const evaluationObservationArtifact = canonicalArtifactReference(
    value.evaluationObservationArtifact,
    'Evaluation gözlemi',
    value.evaluationObservationArtifact?.fingerprint,
  )
  const calibrationArtifact = canonicalArtifactReference(
    value.calibrationArtifact,
    'Kalibrasyon',
    value.calibrationArtifact?.fingerprint,
  )
  const metrics = canonicalEvaluationMetrics(value.metrics, policy)
  const reasons = evaluationReasons(value.calibrationStatus, metrics, policy)
  const status = evaluationStatus(reasons)
  if (!Array.isArray(value.reasons) || !sameCanonical(value.reasons, reasons) || value.status !== status) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Evaluation durumu veya gerekçeleri politika ve metriklerle uyuşmuyor.')
  }
  const replayBinding = deepFreeze({
    trainObservationFingerprint: trainingObservationArtifact.fingerprint,
    evaluationObservationFingerprint: evaluationObservationArtifact.fingerprint,
    calibrationFingerprint: calibrationArtifact.fingerprint,
    atlasFingerprint: atlasBinding.atlasFingerprint,
    partitionManifestFingerprint: datasetBinding.partitionManifestFingerprint,
    producerFingerprint: producerBinding.producerFingerprint,
    inputManifestHash: producerBinding.inputManifestHash,
    probeSpecFingerprint: producerBinding.probeSpecFingerprint,
    policyFingerprint: stableRuntimeHash('plastic-region-semantic-evaluation-policy-v1', policy),
  })
  const payload = {
    format: SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND,
    version: SEMANTIC_PROBE_VERSION,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    atlasMutationAllowed: false,
    releaseAuthority: false,
    calibrationStatus: value.calibrationStatus,
    status,
    reasons,
    hypothesis,
    atlasBinding,
    datasetBinding,
    producerBinding,
    evaluationSplit,
    policy,
    trainingObservationArtifact,
    evaluationObservationArtifact,
    calibrationArtifact,
    replayBinding,
    metrics,
  }
  const fingerprint = stableRuntimeHash('plastic-region-semantic-evaluation-v1', payload)
  const withoutBytes = { ...payload, fingerprint }
  const bytes = assertArtifactBound(withoutBytes, HARD_LIMITS.maximumArtifactBytes)
  const canonical = { ...withoutBytes, bytes }
  if (!sameCanonical(value, canonical)) {
    throw probeError('INVALID_SEMANTIC_PROBE', 'Probe evaluation artifacti kanonik içerik, fingerprint veya boyutla uyuşmuyor.')
  }
  if (dependencies) replaySemanticProbeEvaluation(value, dependencies)
  return value
}

export function semanticProbeArtifactWrite(value) {
  if (value?.format === SEMANTIC_PROBE_OBSERVATIONS_ARTIFACT_KIND) validateSemanticProbeObservationSet(value)
  else if (value?.format === SEMANTIC_PROBE_CALIBRATION_ARTIFACT_KIND) validateSemanticProbeCalibration(value)
  else if (value?.format === SEMANTIC_PROBE_EVALUATION_ARTIFACT_KIND) validateSemanticProbeEvaluationEvidence(value)
  else throw probeError('INVALID_SEMANTIC_PROBE', 'Bilinmeyen probe artifact türü kalıcılaştırılamaz.')
  return deepFreeze({
    kind: value.format,
    contentHash: value.fingerprint,
    value,
    metadata: {
      layer: SEMANTIC_PROBE_LAYER,
      hypothesisId: value.hypothesis.hypothesisId,
      atlasFingerprint: value.atlasBinding.atlasFingerprint,
      split: value.split ?? value.evaluationSplit ?? 'train',
      status: value.status ?? 'unknown',
      interpretationOnly: true,
      trainingWritesAllowed: false,
      routingMutationAllowed: false,
      releaseAuthority: false,
    },
  })
}

/** Bounded public/UI projection; raw labels, lineages, and score rows never leave the artifact reader. */
export function createSemanticProbePublicSummary(input = {}) {
  const maximumItems = integer(input.maximumItems ?? 64, 1, 128, 'Probe özet sınırı')
  const observations = (input.observations ?? []).map(validateSemanticProbeObservationSet)
    .sort((left, right) => compareText(left.hypothesis.hypothesisId, right.hypothesis.hypothesisId) ||
      compareText(left.split, right.split) || compareText(left.fingerprint, right.fingerprint))
  const calibrations = (input.calibrations ?? []).map((value) => validateSemanticProbeCalibration(value))
    .sort((left, right) => compareText(left.hypothesis.hypothesisId, right.hypothesis.hypothesisId) ||
      compareText(left.fingerprint, right.fingerprint))
  const splitRank = { validation: 0, test: 1 }
  const statusRank = { unknown: 0, provisional: 1, calibrated: 2 }
  const evaluations = (input.evaluations ?? []).map((value) => validateSemanticProbeEvaluationEvidence(value))
    .sort((left, right) => compareText(left.hypothesis.hypothesisId, right.hypothesis.hypothesisId) ||
      (splitRank[left.evaluationSplit] ?? -1) - (splitRank[right.evaluationSplit] ?? -1) ||
      compareText(left.fingerprint, right.fingerprint))
  const byHypothesis = new Map()
  const summaryKey = (value) => [
    value.atlasBinding.atlasFingerprint,
    value.hypothesis.hypothesisId,
    value.datasetBinding.partitionManifestFingerprint,
    value.producerBinding.producerFingerprint,
    value.producerBinding.probeSpecFingerprint,
  ].join(':')
  for (const value of observations) {
    const key = summaryKey(value)
    const previous = byHypothesis.get(key)
    byHypothesis.set(key, {
      hypothesis: value.hypothesis,
      status: 'unknown',
      atlasFingerprint: value.atlasBinding.atlasFingerprint,
      observationCount: (previous?.observationCount ?? 0) + value.observationCount,
      independentSourceCount: Math.max(previous?.independentSourceCount ?? 0, value.independentSourceCount),
      target: null,
      evidence: null,
    })
  }
  for (const value of calibrations) {
    const key = summaryKey(value)
    const previous = byHypothesis.get(key)
    byHypothesis.set(key, {
      hypothesis: value.hypothesis,
      status: value.status,
      atlasFingerprint: value.atlasBinding.atlasFingerprint,
      observationCount: Math.max(previous?.observationCount ?? 0, value.trainingObservationCount),
      independentSourceCount: Math.max(previous?.independentSourceCount ?? 0, value.trainingIndependentSourceCount),
      target: value.target.feature,
      evidence: null,
    })
  }
  for (const value of evaluations) {
    const key = summaryKey(value)
    const previous = byHypothesis.get(key)
    const keepPreviousEvidence = previous?.evidence && (
      statusRank[previous.status] > statusRank[value.status] ||
      (statusRank[previous.status] === statusRank[value.status] &&
        (splitRank[previous.evidence.evaluationSplit] ?? -1) > (splitRank[value.evaluationSplit] ?? -1))
    )
    byHypothesis.set(key, {
      hypothesis: value.hypothesis,
      status: keepPreviousEvidence ? previous.status : value.status,
      atlasFingerprint: value.atlasBinding.atlasFingerprint,
      observationCount: Math.max(previous?.observationCount ?? 0, value.metrics.observationCount),
      independentSourceCount: Math.max(previous?.independentSourceCount ?? 0, value.metrics.independentSourceCount),
      target: previous?.target ?? null,
      evidence: keepPreviousEvidence ? previous.evidence : {
        evaluationSplit: value.evaluationSplit,
        targetBrier: value.metrics.target.brier,
        targetEce: value.metrics.target.ece,
        gainVsBaseline: value.metrics.gainVsBaseline,
        gainVsSham: value.metrics.gainVsSham,
        reasons: value.reasons,
      },
    })
  }
  const allItems = [...byHypothesis.values()].sort((left, right) =>
    compareText(left.hypothesis.namespace, right.hypothesis.namespace) ||
    compareText(left.hypothesis.label, right.hypothesis.label) ||
    compareText(left.hypothesis.hypothesisId, right.hypothesis.hypothesisId))
  const items = allItems.slice(0, maximumItems)
  const payload = {
    format: SEMANTIC_PROBE_PUBLIC_SUMMARY_FORMAT,
    version: 1,
    layer: SEMANTIC_PROBE_LAYER,
    interpretationOnly: true,
    trainingWritesAllowed: false,
    routingMutationAllowed: false,
    statuses: ['unknown', 'provisional', 'calibrated'],
    itemCount: items.length,
    omittedCount: Math.max(0, allItems.length - items.length),
    items,
  }
  return deepFreeze({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-semantic-public-summary-v1', payload),
  })
}
