import { describe, expect, it } from 'vitest'
import {
  bootstrapSourceGroupInterval,
  createStructuralBenchmark,
  evaluateStructuralGate,
  scoreEventGraph,
  type EventGraphV1,
} from './benchmark'

describe('controlled structural benchmark', () => {
  it('creates 144 combinations with two leak-free surfaces and at least 30% holdout', () => {
    const cases = createStructuralBenchmark(7)
    expect(cases).toHaveLength(288)
    const combinations = new Map<string, Set<string>>()
    for (const item of cases) {
      const splits = combinations.get(item.combinationId) ?? new Set<string>()
      splits.add(item.split)
      combinations.set(item.combinationId, splits)
    }
    expect(combinations.size).toBe(144)
    expect([...combinations.values()].every((splits) => splits.size === 1)).toBe(true)
    const holdout = [...combinations.entries()].filter(([id]) => cases.find((item) => item.combinationId === id)?.split !== 'train')
    expect(holdout.length / combinations.size).toBeGreaterThanOrEqual(0.3)
    expect(cases.filter((item) => item.split === 'validation').length).toBeGreaterThan(0)
    expect(cases.filter((item) => item.split === 'test').length).toBeGreaterThan(0)
  })

  it('scores normalized graph atoms and structural constraints separately', () => {
    const expected: EventGraphV1 = {
      format: 'event-graph-v1', actor: 'Ada', action: 'TAŞIDI.', patient: 'Kırmızı kutu', instrument: 'Kanca',
      operators: ['Hemen'], negated: false, goals: [{ relation: 'hedef', target: 'depo' }],
    }
    const exact = scoreEventGraph({ ...expected, actor: ' ada ' }, expected)
    expect(exact.f1).toBe(1)
    expect(exact.constraintAccuracy).toBe(1)
    const wrong = scoreEventGraph({ ...expected, actor: 'Bora', instrument: null }, expected)
    expect(wrong.f1).toBeLessThan(1)
    expect(wrong.constraintAccuracy).toBeLessThan(1)
  })

  it('requires five seeds and produces a deterministic source-group bootstrap interval', () => {
    const observations = [1, 2, 3, 4, 5].flatMap((seed) => ['a', 'b', 'c'].map((sourceGroupId, index) => ({
      seed, sourceGroupId, score: 0.7 + index * 0.05,
    })))
    const first = bootstrapSourceGroupInterval(observations, 200, 9)
    const second = bootstrapSourceGroupInterval(observations, 200, 9)
    expect(first).toEqual(second)
    expect(first.mean).toBeCloseTo(0.75)
    expect(first.low).toBeLessThanOrEqual(first.mean)
    expect(first.high).toBeGreaterThanOrEqual(first.mean)
  })

  it('enforces every approved structural gate', () => {
    const passing = evaluateStructuralGate({
      holdoutGraphF1: 0.81,
      h0OnlyGraphF1: 0.62,
      noRecurrenceGraphF1: 0.61,
      shuffledEdgeGraphF1: 0.6,
      targetFamilyAblationDrop: 0.12,
      targetEdgeAblationDrop: 0.11,
      shamFamilyAblationDrop: 0.02,
      shamEdgeAblationDrop: 0.02,
      perturbationRecoveryRate: 0.74,
      knownStructureFalseBirthRate: 0.04,
      repeatedNoveltyBirthPrecision: 0.84,
      repeatedNoveltyBirthSensitivity: 0.84,
    })
    expect(passing.passed).toBe(true)
    expect(passing.baselineAdvantage).toBeCloseTo(0.19)
    expect(evaluateStructuralGate({ ...{
      holdoutGraphF1: 0.81,
      h0OnlyGraphF1: 0.62,
      noRecurrenceGraphF1: 0.61,
      shuffledEdgeGraphF1: 0.6,
      targetFamilyAblationDrop: 0.12,
      targetEdgeAblationDrop: 0.11,
      shamFamilyAblationDrop: 0.02,
      shamEdgeAblationDrop: 0.02,
      perturbationRecoveryRate: 0.74,
      knownStructureFalseBirthRate: 0.04,
      repeatedNoveltyBirthPrecision: 0.84,
      repeatedNoveltyBirthSensitivity: 0.84,
    }, shamEdgeAblationDrop: 0.04 }).passed).toBe(false)
  })

  it('preserves a negative target ablation effect and rejects it without rejecting signed sham input', () => {
    const report = evaluateStructuralGate({
      holdoutGraphF1: 0.81,
      h0OnlyGraphF1: 0.62,
      noRecurrenceGraphF1: 0.61,
      shuffledEdgeGraphF1: 0.6,
      targetFamilyAblationDrop: 0.12,
      targetEdgeAblationDrop: -0.2,
      shamFamilyAblationDrop: -0.04,
      shamEdgeAblationDrop: -0.05,
      perturbationRecoveryRate: 0.74,
      knownStructureFalseBirthRate: 0.04,
      repeatedNoveltyBirthPrecision: 0.84,
      repeatedNoveltyBirthSensitivity: 0.84,
    })
    expect(report.checks.find((check) => check.id === 'target-edge-ablation')?.actual).toBe(-0.2)
    expect(report.checks.find((check) => check.id === 'target-edge-ablation')?.passed).toBe(false)
    expect(report.checks.find((check) => check.id === 'sham-edge-ablation')?.passed).toBe(true)
    expect(report.passed).toBe(false)
  })

  it('requires repeated-novelty birth sensitivity independently from precision', () => {
    const report = evaluateStructuralGate({
      holdoutGraphF1: 0.81,
      h0OnlyGraphF1: 0.62,
      noRecurrenceGraphF1: 0.61,
      shuffledEdgeGraphF1: 0.6,
      targetFamilyAblationDrop: 0.12,
      targetEdgeAblationDrop: 0.11,
      shamFamilyAblationDrop: 0.02,
      shamEdgeAblationDrop: 0.02,
      perturbationRecoveryRate: 0.74,
      knownStructureFalseBirthRate: 0.04,
      repeatedNoveltyBirthPrecision: 1,
      repeatedNoveltyBirthSensitivity: 0.79,
    })
    expect(report.checks.find((check) => check.id === 'novelty-birth-precision')?.passed).toBe(true)
    expect(report.checks.find((check) => check.id === 'novelty-birth-sensitivity')?.passed).toBe(false)
    expect(report.passed).toBe(false)
  })
})
