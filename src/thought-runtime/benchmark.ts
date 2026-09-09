import { thoughtDigest } from './deterministic'
import type { EventGraphV1 } from './types'

export type { EventGraphV1 } from './types'
export { normalizeEventGraph, scoreEventGraph, type StructuralGraphScore } from './evaluation'

export type StructuralBenchmarkSplit = 'train' | 'validation' | 'test'

export interface StructuralBenchmarkCase {
  id: string
  combinationId: string
  sourceGroupId: string
  surfaceIndex: number
  split: StructuralBenchmarkSplit
  prompt: string
  expected: EventGraphV1
}

export interface StructuralBenchmarkObservation {
  seed: number
  sourceGroupId: string
  score: number
}

export interface BootstrapInterval {
  mean: number
  low: number
  high: number
  confidence: 0.95
  seedCount: number
  sourceGroupCount: number
  samples: number
}

export interface StructuralGateInput {
  holdoutGraphF1: number
  h0OnlyGraphF1: number
  noRecurrenceGraphF1: number
  shuffledEdgeGraphF1: number
  targetFamilyAblationDrop: number
  targetEdgeAblationDrop: number
  shamFamilyAblationDrop: number
  shamEdgeAblationDrop: number
  perturbationRecoveryRate: number
  knownStructureFalseBirthRate: number
  repeatedNoveltyBirthPrecision: number
  repeatedNoveltyBirthSensitivity: number
}

export interface StructuralGateCheck {
  id: string
  passed: boolean
  actual: number
  expected: string
}

export interface StructuralGateReport {
  passed: boolean
  strongestBaseline: number
  baselineAdvantage: number
  checks: StructuralGateCheck[]
}

const ACTORS = ['Ada', 'Bora', 'Cem', 'Duru'] as const
const ACTIONS = ['taşıdı', 'çekti', 'yerleştirdi'] as const
const PATIENTS = ['kırmızı kutu', 'mavi kutu', 'sarı kutu', 'yeşil kutu'] as const
const INSTRUMENTS = ['kanca', 'halat', 'robot kol'] as const

function turkishAccusative(value: string): string {
  const suffixes: Record<string, string> = {
    'kırmızı kutu': 'kırmızı kutuyu',
    'mavi kutu': 'mavi kutuyu',
    'sarı kutu': 'sarı kutuyu',
    'yeşil kutu': 'yeşil kutuyu',
  }
  return suffixes[value] ?? value
}

function turkishInstrumental(value: string): string {
  const suffixes: Record<string, string> = {
    kanca: 'kancayla',
    halat: 'halatla',
    'robot kol': 'robot kolla',
  }
  return suffixes[value] ?? `${value} ile`
}

function seededUnit(seed: number, key: string): number {
  return Number.parseInt(thoughtDigest('structural-benchmark-order', seed, key).slice(0, 8), 16) / 0x1_0000_0000
}

function benchmarkSurface(
  actor: string,
  action: string,
  patient: string,
  instrument: string,
  surfaceIndex: number,
): string {
  const patientAccusative = turkishAccusative(patient)
  const instrumental = turkishInstrumental(instrument)
  return surfaceIndex === 0
    ? `${actor} ${patientAccusative} ${instrumental} ${action}.`
    : `${instrumental[0].toLocaleUpperCase('tr-TR')}${instrumental.slice(1)} çalışan ${actor}, ${patientAccusative} ${action}.`
}

/**
 * Produces 3x4x4x3=144 combinations. A combination is assigned before either
 * surface is created, so its paraphrase can never leak across a split.
 */
export function createStructuralBenchmark(seed = 0x5eed, holdoutRatio = 0.3): StructuralBenchmarkCase[] {
  if (!Number.isFinite(holdoutRatio) || holdoutRatio < 0.3 || holdoutRatio >= 1) {
    throw new RangeError('Yapısal benchmark holdout oranı [0.30, 1) aralığında olmalıdır.')
  }
  const combinations = ACTIONS.flatMap((action) => ACTORS.flatMap((actor) => PATIENTS.flatMap((patient) =>
    INSTRUMENTS.map((instrument) => {
      const combinationId = thoughtDigest('structural-combination-v1', actor, action, patient, instrument).slice(0, 20)
      return { actor, action, patient, instrument, combinationId }
    }),
  )))
  const ordered = [...combinations].sort((left, right) =>
    seededUnit(seed, left.combinationId) - seededUnit(seed, right.combinationId) || left.combinationId.localeCompare(right.combinationId),
  )
  const holdoutCount = Math.ceil(ordered.length * holdoutRatio)
  const splitById = new Map<string, StructuralBenchmarkSplit>()
  ordered.forEach((combination, index) => {
    splitById.set(combination.combinationId, index < holdoutCount ? (index % 2 === 0 ? 'validation' : 'test') : 'train')
  })

  return combinations.flatMap(({ actor, action, patient, instrument, combinationId }) => {
    const expected: EventGraphV1 = {
      format: 'event-graph-v1',
      actor,
      action,
      patient,
      instrument,
      operators: [],
      negated: false,
      goals: [],
    }
    return [0, 1].map((surfaceIndex): StructuralBenchmarkCase => ({
      id: `SB-${combinationId}-${surfaceIndex}`,
      combinationId,
      sourceGroupId: `SBG-${combinationId}`,
      surfaceIndex,
      split: splitById.get(combinationId)!,
      prompt: benchmarkSurface(actor, action, patient, instrument, surfaceIndex),
      expected,
    }))
  }).sort((left, right) => left.id.localeCompare(right.id))
}

function percentile(values: readonly number[], probability: number): number {
  if (!values.length) return 0
  const ordered = [...values].sort((left, right) => left - right)
  const position = (ordered.length - 1) * probability
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return ordered[lower]
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
}

/** Source groups are resampled inside every run, then the five run means are averaged. */
export function bootstrapSourceGroupInterval(
  observations: readonly StructuralBenchmarkObservation[],
  samples = 2_000,
  bootstrapSeed = 0x51a7,
): BootstrapInterval {
  if (!Number.isInteger(samples) || samples < 100) throw new RangeError('Bootstrap en az 100 örnek kullanmalıdır.')
  const seeds = [...new Set(observations.map((item) => item.seed))].sort((left, right) => left - right)
  if (seeds.length !== 5) throw new RangeError('Yapısal kabul raporu tam beş bağımsız seed gerektirir.')
  const groups = [...new Set(observations.map((item) => item.sourceGroupId))].sort()
  if (!groups.length) throw new RangeError('Bootstrap için en az bir kaynak grubu gerekir.')
  const bySeedAndGroup = new Map<string, number[]>()
  for (const observation of observations) {
    if (!Number.isFinite(observation.score) || observation.score < 0 || observation.score > 1) {
      throw new RangeError('Benchmark skoru [0,1] aralığında olmalıdır.')
    }
    const key = `${observation.seed}\u0000${observation.sourceGroupId}`
    const values = bySeedAndGroup.get(key) ?? []
    values.push(observation.score)
    bySeedAndGroup.set(key, values)
  }
  const groupScore = (seed: number, group: string): number | null => {
    const values = bySeedAndGroup.get(`${seed}\u0000${group}`)
    return values?.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
  }
  const pointValues = seeds.flatMap((seed) => groups.map((group) => groupScore(seed, group)).filter((value): value is number => value !== null))
  const means: number[] = []
  for (let sample = 0; sample < samples; sample += 1) {
    const seedMeans: number[] = []
    for (const seed of seeds) {
      const selected: number[] = []
      for (let draw = 0; draw < groups.length; draw += 1) {
        const unit = seededUnit(bootstrapSeed + sample * 17 + seed, `group-draw:${draw}`)
        const group = groups[Math.min(groups.length - 1, Math.floor(unit * groups.length))]
        const value = groupScore(seed, group)
        if (value !== null) selected.push(value)
      }
      if (selected.length) seedMeans.push(selected.reduce((sum, value) => sum + value, 0) / selected.length)
    }
    if (seedMeans.length) means.push(seedMeans.reduce((sum, value) => sum + value, 0) / seedMeans.length)
  }
  return {
    mean: pointValues.reduce((sum, value) => sum + value, 0) / Math.max(1, pointValues.length),
    low: percentile(means, 0.025),
    high: percentile(means, 0.975),
    confidence: 0.95,
    seedCount: seeds.length,
    sourceGroupCount: groups.length,
    samples,
  }
}

export function evaluateStructuralGate(input: StructuralGateInput): StructuralGateReport {
  for (const [key, value] of Object.entries(input)) {
    const signed = ['targetFamilyAblationDrop', 'targetEdgeAblationDrop', 'shamFamilyAblationDrop', 'shamEdgeAblationDrop'].includes(key)
    if (!Number.isFinite(value) || value < (signed ? -1 : 0) || value > 1) {
      throw new RangeError(`${key} ${signed ? '[-1,1]' : '[0,1]'} aralığında olmalıdır.`)
    }
  }
  const strongestBaseline = Math.max(input.h0OnlyGraphF1, input.noRecurrenceGraphF1, input.shuffledEdgeGraphF1)
  const baselineAdvantage = input.holdoutGraphF1 - strongestBaseline
  const checks: StructuralGateCheck[] = [
    { id: 'holdout-graph-f1', passed: input.holdoutGraphF1 >= 0.75, actual: input.holdoutGraphF1, expected: '>= 0.75' },
    { id: 'baseline-advantage', passed: baselineAdvantage >= 0.15, actual: baselineAdvantage, expected: '>= 0.15' },
    { id: 'target-family-ablation', passed: input.targetFamilyAblationDrop >= 0.1, actual: input.targetFamilyAblationDrop, expected: '>= 0.10' },
    { id: 'target-edge-ablation', passed: input.targetEdgeAblationDrop >= 0.1, actual: input.targetEdgeAblationDrop, expected: '>= 0.10' },
    { id: 'sham-family-ablation', passed: input.shamFamilyAblationDrop <= 0.03, actual: input.shamFamilyAblationDrop, expected: '<= 0.03' },
    { id: 'sham-edge-ablation', passed: input.shamEdgeAblationDrop <= 0.03, actual: input.shamEdgeAblationDrop, expected: '<= 0.03' },
    { id: 'perturbation-recovery', passed: input.perturbationRecoveryRate >= 0.7, actual: input.perturbationRecoveryRate, expected: '>= 0.70' },
    { id: 'known-structure-false-birth', passed: input.knownStructureFalseBirthRate <= 0.05, actual: input.knownStructureFalseBirthRate, expected: '<= 0.05' },
    { id: 'novelty-birth-precision', passed: input.repeatedNoveltyBirthPrecision >= 0.8, actual: input.repeatedNoveltyBirthPrecision, expected: '>= 0.80' },
    { id: 'novelty-birth-sensitivity', passed: input.repeatedNoveltyBirthSensitivity >= 0.8, actual: input.repeatedNoveltyBirthSensitivity, expected: '>= 0.80' },
  ]
  return { passed: checks.every((check) => check.passed), strongestBaseline, baselineAdvantage, checks }
}
