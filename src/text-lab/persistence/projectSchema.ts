import {
  createMatrixLayout,
  MAX_ANALYSIS_NODES,
  normalizeTextLabConfig,
  resolveTextLabRepresentationProfileId,
  validateConfig,
} from '../domain/config'
import type { PersistedProject, TextLabAnalysis, TextLabConfig, TextSource } from '../domain/types'
import { encodeIdentity } from '../engine/encode'
import { vectorNorm } from '../engine/matrix'
import {
  createContextSpaceId,
  createIdentitySpaceId,
  createRepresentationSpaceId,
  digestText,
  digestVector,
} from '../engine/representation'

const FLOAT32_TAG = '__metin_lab_float32__'

export function stringifyProject(project: PersistedProject): string {
  return JSON.stringify(
    project,
    (_key, value: unknown) => {
      if (value instanceof Float32Array) {
        return { [FLOAT32_TAG]: Array.from(value) }
      }
      return value
    },
    2,
  )
}

export function parseProject(value: string): PersistedProject {
  const parsed = JSON.parse(value, (_key, candidate: unknown) => {
    if (
      candidate &&
      typeof candidate === 'object' &&
      FLOAT32_TAG in candidate &&
      Array.isArray((candidate as Record<string, unknown>)[FLOAT32_TAG])
    ) {
      return Float32Array.from((candidate as Record<string, number[]>)[FLOAT32_TAG])
    }
    return candidate
  })

  return validateProject(parsed)
}

export function validateProject(value: unknown): PersistedProject {
  if (!value || typeof value !== 'object') {
    throw new TypeError('Dosya geçerli bir Metin Laboratuvarı projesi değil.')
  }
  const parsed = value as Partial<Omit<PersistedProject, 'version'>> & { version?: number }

  if (
    parsed.format !== 'metin-laboratuvari-project' ||
    (parsed.version !== 1 && parsed.version !== 2) ||
    !parsed.source ||
    typeof parsed.source.rawText !== 'string' ||
    typeof parsed.source.id !== 'string' ||
    typeof parsed.source.title !== 'string' ||
    !parsed.config
  ) {
    throw new TypeError('Dosya geçerli bir Metin Laboratuvarı projesi değil.')
  }
  const config = normalizeTextLabConfig(parsed.config)
  const storageRevision = Number.isInteger(parsed.storageRevision) && (parsed.storageRevision ?? -1) >= 0
    ? parsed.storageRevision as number
    : 0
  if (parsed.version === 1) {
    return createPersistedProject(parsed.source, config, null, storageRevision)
  }
  if (parsed.analysis) {
    validateAnalysis(parsed.analysis)
    if (!analysisMatchesProject(parsed.source, config, parsed.analysis)) {
      throw new TypeError('Projedeki analiz farklı bir kaynak veya ayar sürümüne ait.')
    }
  }
  return {
    ...parsed,
    version: 2,
    storageRevision,
    source: parsed.source,
    config,
    analysis: parsed.analysis ?? null,
  } as PersistedProject
}

function validateAnalysis(analysis: TextLabAnalysis): void {
  if (!analysis || typeof analysis !== 'object') {
    throw new TypeError('Projedeki analiz verisi eksik veya bozuk.')
  }
  const count = Array.isArray(analysis.nodes) ? analysis.nodes.length : -1
  const layout = analysis.layout
  if (
    analysis.format !== 'metin-laboratuvari-analysis' ||
    analysis.version !== 2 ||
    typeof analysis.runId !== 'string' ||
    typeof analysis.sourceDigest !== 'string' ||
    typeof analysis.identitySpaceId !== 'string' ||
    typeof analysis.contextSpaceId !== 'string' ||
    typeof analysis.representationSpaceId !== 'string' ||
    count < 1 ||
    count > MAX_ANALYSIS_NODES ||
    !analysis.source ||
    typeof analysis.source.id !== 'string' ||
    typeof analysis.source.title !== 'string' ||
    typeof analysis.source.rawText !== 'string' ||
    typeof analysis.source.text !== 'string' ||
    !analysis.config ||
    !layout ||
    !Array.isArray(analysis.rootIds) ||
    analysis.rootIds.length < 1 ||
    !(analysis.similarities instanceof Float32Array) ||
    !(analysis.habitProbabilities instanceof Float32Array) ||
    analysis.similarities.length !== count * count ||
    analysis.habitProbabilities.length !== count * count
  ) {
    throw new TypeError('Projedeki analiz verisi eksik veya bozuk.')
  }
  validateConfig(analysis.config)
  const expectedLayout = createMatrixLayout(analysis.config)
  const expectedIdentitySpace = createIdentitySpaceId(analysis.config, expectedLayout)
  const expectedContextSpace = createContextSpaceId(analysis.config, expectedLayout)
  if (
    layout.side !== expectedLayout.side ||
    layout.cellCount !== expectedLayout.cellCount ||
    layout.identityRows !== expectedLayout.identityRows ||
    layout.identityLength !== expectedLayout.identityLength ||
    layout.relationStart !== expectedLayout.relationStart ||
    layout.relationLength !== expectedLayout.relationLength ||
    analysis.sourceDigest !== digestText(analysis.source.text) ||
    analysis.identitySpaceId !== expectedIdentitySpace ||
    analysis.contextSpaceId !== expectedContextSpace ||
    analysis.representationSpaceId !== createRepresentationSpaceId(
      expectedIdentitySpace,
      expectedContextSpace,
    )
  ) {
    throw new TypeError('Projedeki matris maskesi ayarlarla uyuşmuyor.')
  }

  const ids = new Set<string>()
  for (const node of analysis.nodes) {
    if (
      !node ||
      typeof node.id !== 'string' ||
      !node.id ||
      ids.has(node.id) ||
      !Number.isInteger(node.start) ||
      !Number.isInteger(node.end) ||
      node.start < 0 ||
      node.end <= node.start ||
      node.end > analysis.source.text.length ||
      analysis.source.text.slice(node.start, node.end) !== node.text ||
      !Array.isArray(node.childIds) ||
      !(node.identity instanceof Float32Array) ||
      node.identity.length !== layout.identityLength ||
      !(node.relation instanceof Float32Array) ||
      node.relation.length !== layout.relationLength ||
      typeof node.identityDigest !== 'string' ||
      node.identityDigest !== digestVector(node.identity) ||
      typeof node.surfaceKey !== 'string' ||
      !Number.isFinite(node.contextSupport) ||
      node.contextSupport < 0 ||
      !Number.isInteger(node.contextRevision) ||
      node.contextRevision < 0 ||
      !Number.isFinite(node.identityNorm) ||
      !Number.isFinite(node.relationNorm) ||
      Math.abs(node.identityNorm - vectorNorm(node.identity)) > 1e-5 ||
      Math.abs(node.relationNorm - vectorNorm(node.relation)) > 1e-5 ||
      Array.from(node.identity).some((cell) => !Number.isFinite(cell) || cell < -1 || cell > 1) ||
      Array.from(node.relation).some((cell) => !Number.isFinite(cell) || cell < -1 || cell > 1)
    ) {
      throw new TypeError('Projedeki ham nöron kayıtlarından biri geçersiz.')
    }
    ids.add(node.id)

    const expectedIdentity = encodeIdentity(node.text, layout, analysis.config)
    if (
      expectedIdentity.length !== node.identity.length ||
      expectedIdentity.some((cell, index) => cell !== node.identity[index])
    ) {
      throw new TypeError('Projedeki sabit kimlik alanlarından biri kaynak metinle uyuşmuyor.')
    }
  }

  for (const node of analysis.nodes) {
    if (
      (node.parentId !== null && !ids.has(node.parentId)) ||
      node.childIds.some((id) => !ids.has(id))
    ) {
      throw new TypeError('Projedeki bölünme ağacı eksik düğüm içeriyor.')
    }
  }
  if (analysis.rootIds.some((id) => !ids.has(id))) {
    throw new TypeError('Projedeki paragraf kökleri geçersiz.')
  }

  for (let row = 0; row < count; row += 1) {
    let habitSum = 0
    for (let column = 0; column < count; column += 1) {
      const index = row * count + column
      const similarity = analysis.similarities[index]
      const habit = analysis.habitProbabilities[index]
      if (
        !Number.isFinite(similarity) ||
        similarity < -1.000001 ||
        similarity > 1.000001 ||
        !Number.isFinite(habit) ||
        habit < 0 ||
        habit > 1.000001 ||
        Math.abs(similarity - analysis.similarities[column * count + row]) > 1e-5 ||
        (row === column && Math.abs(similarity - 1) > 1e-5) ||
        (row === column && habit !== 0)
      ) {
        throw new TypeError('Projedeki K veya H matrisi sayısal olarak geçersiz.')
      }
      habitSum += habit
    }
    if (habitSum > 1e-6 && Math.abs(habitSum - 1) > 1e-4) {
      throw new TypeError('Projedeki H satırlarından biri normalize değil.')
    }
  }

  if (
    !analysis.stats ||
    analysis.stats.neuronCount !== count ||
    analysis.stats.paragraphCount !== analysis.rootIds.length ||
    analysis.stats.uniquePairCount !== (count * (count - 1)) / 2 ||
    Object.values(analysis.stats).some((entry) => !Number.isFinite(entry))
  ) {
    throw new TypeError('Projedeki deney istatistikleri analizle uyuşmuyor.')
  }
}

export function createPersistedProject(
  source: PersistedProject['source'],
  config: PersistedProject['config'],
  analysis: TextLabAnalysis | null,
  storageRevision = 0,
): PersistedProject {
  validateConfig(config)
  if (!Number.isInteger(storageRevision) || storageRevision < 0) {
    throw new RangeError('Yerel kayıt sürümü negatif olmayan bir tam sayı olmalıdır.')
  }
  return {
    format: 'metin-laboratuvari-project',
    version: 2,
    storageRevision,
    savedAt: new Date().toISOString(),
    source: { ...source },
    config: { ...config },
    analysis: analysisMatchesProject(source, config, analysis) ? analysis : null,
  }
}

const CONFIG_KEYS: Array<keyof TextLabConfig> = [
  'matrixSide',
  'identityRatio',
  'minimumLeafGraphemes',
  'hashSeed',
  'contextRetention',
  'parentTraceWeight',
  'siblingTraceWeight',
  'minimumTrace',
  'identitySimilarityWeight',
  'habitExponent',
]

export function analysisMatchesProject(
  source: TextSource,
  config: TextLabConfig,
  analysis: TextLabAnalysis | null,
): boolean {
  if (!analysis) return true
  return (
    analysis.source.id === source.id &&
    analysis.source.title === source.title &&
    analysis.source.rawText === source.rawText &&
    resolveTextLabRepresentationProfileId(analysis.config) ===
      resolveTextLabRepresentationProfileId(config) &&
    CONFIG_KEYS.every((key) => analysis.config[key] === config[key])
  )
}
