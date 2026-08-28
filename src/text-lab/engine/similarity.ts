import type {
  MatrixLayout,
  PairSimilarityDetail,
  PipelineProgress,
  RawTextNeuron,
  TextLabConfig,
} from '../domain/types'
import { cosineRegion, dotProduct } from './matrix'

export class PipelineCancelledError extends Error {
  constructor() {
    super('İşlem kullanıcı tarafından durduruldu.')
    this.name = 'PipelineCancelledError'
  }
}

export interface SimilarityOptions {
  onProgress?: (progress: PipelineProgress) => void
  shouldCancel?: () => boolean
  yieldEveryPairs?: number
}

export interface HabitProbabilityOptions {
  onProgress?: (completedRows: number, totalRows: number) => void
  shouldCancel?: () => boolean
  yieldEveryRows?: number
}

function median(values: number[]): number {
  if (!values.length) return 1
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

export function relationConfidence(norm: number, medianRelationNorm: number): number {
  if (medianRelationNorm <= 0) return 0
  return Math.min(1, Math.max(0, norm / medianRelationNorm))
}

export function pairSimilarity(
  first: RawTextNeuron,
  second: RawTextNeuron,
  layout: MatrixLayout,
  config: TextLabConfig,
  medianRelationNorm: number,
): PairSimilarityDetail {
  const identity = cosineRegion(
    first.identity,
    second.identity,
    0,
    layout.identityLength,
    first.identityNorm,
    second.identityNorm,
  )
  const relation = cosineRegion(
    first.relation,
    second.relation,
    0,
    layout.relationLength,
    first.relationNorm,
    second.relationNorm,
  )
  const confidence =
    relationConfidence(first.relationNorm, medianRelationNorm) *
    relationConfidence(second.relationNorm, medianRelationNorm)
  const identityWeight = config.identitySimilarityWeight
  const denominator = identityWeight + (1 - identityWeight) * confidence
  const combined = denominator > 0
    ? (identityWeight * identity + (1 - identityWeight) * confidence * relation) / denominator
    : identity

  return {
    combined: Math.min(1, Math.max(-1, combined)),
    identity,
    relation,
    relationConfidence: confidence,
  }
}

function combinedPairSimilarity(
  first: RawTextNeuron,
  second: RawTextNeuron,
  layout: MatrixLayout,
  config: TextLabConfig,
  medianRelationNorm: number,
): number {
  const identity = cosineRegion(
    first.identity,
    second.identity,
    0,
    layout.identityLength,
    first.identityNorm,
    second.identityNorm,
  )
  const relation = cosineRegion(
    first.relation,
    second.relation,
    0,
    layout.relationLength,
    first.relationNorm,
    second.relationNorm,
  )
  const confidence =
    relationConfidence(first.relationNorm, medianRelationNorm) *
    relationConfidence(second.relationNorm, medianRelationNorm)
  const identityWeight = config.identitySimilarityWeight
  const denominator = identityWeight + (1 - identityWeight) * confidence
  const combined = denominator > 0
    ? (identityWeight * identity + (1 - identityWeight) * confidence * relation) / denominator
    : identity
  return Math.min(1, Math.max(-1, combined))
}

export function detailFromAnalysisPair(
  first: RawTextNeuron,
  second: RawTextNeuron,
  layout: MatrixLayout,
  config: TextLabConfig,
  medianRelationNorm: number,
): PairSimilarityDetail {
  return pairSimilarity(first, second, layout, config, medianRelationNorm)
}

function report(
  options: SimilarityOptions,
  completed: number,
  total: number,
  message: string,
) {
  options.onProgress?.({
    stage: 'comparing',
    completed,
    total,
    ratio: total === 0 ? 1 : completed / total,
    message,
  })
}

export async function computeAllPairSimilarities(
  nodes: readonly RawTextNeuron[],
  layout: MatrixLayout,
  config: TextLabConfig,
  options: SimilarityOptions = {},
): Promise<{ similarities: Float32Array; medianRelationNorm: number }> {
  if (options.shouldCancel?.()) throw new PipelineCancelledError()
  const count = nodes.length
  const total = (count * (count - 1)) / 2
  const similarities = new Float32Array(count * count)
  const medianRelationNorm = median(
    nodes.map((node) => node.relationNorm).filter((norm) => norm > 0),
  )
  for (let index = 0; index < count; index += 1) similarities[index * count + index] = 1

  const yieldEvery = Math.max(256, options.yieldEveryPairs ?? 2048)
  let completed = 0
  report(options, 0, total, 'Bütün benzersiz nöron çiftleri hazırlanıyor…')

  for (let row = 0; row < count; row += 1) {
    for (let column = row + 1; column < count; column += 1) {
      const value = combinedPairSimilarity(
        nodes[row],
        nodes[column],
        layout,
        config,
        medianRelationNorm,
      )
      similarities[row * count + column] = value
      similarities[column * count + row] = value
      completed += 1

      if (completed % yieldEvery === 0) {
        if (options.shouldCancel?.()) throw new PipelineCancelledError()
        report(options, completed, total, `${completed.toLocaleString('tr-TR')} çift karşılaştırıldı`)
        await new Promise<void>((resolve) => setTimeout(resolve, 0))
      }
    }
  }

  if (options.shouldCancel?.()) throw new PipelineCancelledError()
  report(options, total, total, 'Tam benzerlik matrisi hazır')
  return { similarities, medianRelationNorm }
}

export async function computeInitialHabitProbabilitiesAsync(
  similarities: Float32Array,
  nodes: readonly RawTextNeuron[],
  exponent: number,
  options: HabitProbabilityOptions = {},
): Promise<Float32Array> {
  if (options.shouldCancel?.()) throw new PipelineCancelledError()
  const count = nodes.length
  const probabilities = new Float32Array(count * count)
  const rowSums = new Float64Array(count)
  const totalRows = count * 2
  const yieldEvery = Math.max(1, options.yieldEveryRows ?? 8)

  for (let row = 0; row < count; row += 1) {
    for (let column = 0; column < count; column += 1) {
      if (row === column) continue
      const index = row * count + column
      const evidence = Math.pow(Math.max(0, similarities[index]), exponent)
      probabilities[index] = evidence
      rowSums[row] += evidence
    }
    options.onProgress?.(row + 1, totalRows)
    if ((row + 1) % yieldEvery === 0) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
      if (options.shouldCancel?.()) throw new PipelineCancelledError()
    }
  }

  const idToIndex = new Map(nodes.map((node, index) => [node.id, index]))
  for (let row = 0; row < count; row += 1) {
    if (rowSums[row] > 0) {
      for (let column = 0; column < count; column += 1) {
        if (row !== column) probabilities[row * count + column] /= rowSums[row]
      }
    } else {
      const node = nodes[row]
      const fallbackId = node.parentId ?? node.childIds[0] ?? null
      const fallbackIndex = fallbackId ? idToIndex.get(fallbackId) : undefined
      if (fallbackIndex !== undefined && fallbackIndex !== row) {
        probabilities[row * count + fallbackIndex] = 1
      }
    }
    options.onProgress?.(count + row + 1, totalRows)
    if ((row + 1) % yieldEvery === 0) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
      if (options.shouldCancel?.()) throw new PipelineCancelledError()
    }
  }

  if (options.shouldCancel?.()) throw new PipelineCancelledError()
  return probabilities
}

/** Fast diagnostic for tests and selected-pair UI; it still compares matching coordinates. */
export function rawRegionDot(
  first: RawTextNeuron,
  second: RawTextNeuron,
  layout: MatrixLayout,
): { identity: number; relation: number } {
  return {
    identity: dotProduct(first.identity, second.identity, 0, layout.identityLength),
    relation: dotProduct(first.relation, second.relation, 0, layout.relationLength),
  }
}
