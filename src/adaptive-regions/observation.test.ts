import { describe, expect, it } from 'vitest'
import { relationIndexFromPage, ADAPTIVE_RELATION_LENGTH } from './geometry.ts'
import {
  aggregatePageSketches,
  computeAllPageBehaviors,
  computeAllPageObservations,
  computePageObservation,
  PAGE_BEHAVIOR_DIMENSION,
} from './observation.ts'
import type { PlasticPageObservationV1 } from './types.ts'

function observation(
  observationId: string,
  sourceGroupId: string,
  behavior: readonly number[],
): PlasticPageObservationV1 {
  return {
    observationId,
    episodeId: `episode-${observationId}`,
    sourceGroupId,
    pageIndex: 0,
    confidence: 1,
    behavior,
    outcomeCredit: 0.5,
    ablationUtility: 0.25,
  }
}

describe('mikro-sayfa gözlemi', () => {
  it('sinyal, temsil ve deltadan sekiz boyutlu sınırlandırılmış davranış çıkarır', () => {
    const signal = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    const representation = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    const delta = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    for (let cell = 0; cell < 64; cell += 1) {
      const index = relationIndexFromPage(3, cell)
      signal[index] = 1
      representation[index] = 1
      delta[index] = 0.5
    }
    const result = computePageObservation({
      observationId: 'obs', episodeId: 'ep', sourceGroupId: 'source', pageIndex: 3,
      signalRelation: signal, representationRelation: representation, deltaRelation: delta,
      outcomeCredit: 1, ablationUtility: 0.2,
    })
    expect(result.behavior).toHaveLength(PAGE_BEHAVIOR_DIMENSION)
    expect(result.behavior.slice(0, 5)).toEqual([1, 1, 1, 0, 0.25])
    expect(result.behavior.slice(5)).toEqual([1, 1, 0])
    expect(Object.isFrozen(result)).toBe(true)
  })

  it('bütün 56 sayfa için benzersiz ve seri hale getirilebilir gözlemler üretir', () => {
    const relation = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    const result = computeAllPageObservations({
      observationIdPrefix: 'obs', episodeId: 'ep', sourceGroupId: 'source',
      signalRelation: relation, representationRelation: relation,
    })
    expect(result).toHaveLength(56)
    expect(new Set(result.map((value) => value.observationId)).size).toBe(56)
    expect(() => JSON.stringify(result)).not.toThrow()
  })

  it('canlı yönlendirme davranışları ile kalıcı gözlem davranışlarını birebir eşler', () => {
    const signal = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    const representation = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    signal[relationIndexFromPage(5, 2)] = 1
    representation[relationIndexFromPage(5, 2)] = 0.25
    const behaviors = computeAllPageBehaviors({ signalRelation: signal, representationRelation: representation })
    const observations = computeAllPageObservations({
      observationIdPrefix: 'same', episodeId: 'episode', sourceGroupId: 'source',
      signalRelation: signal, representationRelation: representation,
    })
    expect(behaviors).toHaveLength(56)
    expect(behaviors).toEqual(observations.map((entry) => entry.behavior))
  })
})

describe('kaynak dengeli sayfa sketchleri', () => {
  it('aynı kaynağın yüz kez tekrarlanmasına ek oy hakkı vermez', () => {
    const small = [observation('a-1', 'source-a', [1, 0]), observation('b-1', 'source-b', [0, 1])]
    const repeated = [
      ...Array.from({ length: 100 }, (_, index) => observation(`a-${index}`, 'source-a', [1, 0])),
      observation('b-1', 'source-b', [0, 1]),
    ]
    const smallSketch = aggregatePageSketches(small, { behaviorDimension: 2 })[0]
    const repeatedSketch = aggregatePageSketches(repeated, { behaviorDimension: 2 })[0]
    expect(smallSketch.behaviorPrototype).toEqual([0.5, 0.5])
    expect(repeatedSketch.behaviorPrototype).toEqual(smallSketch.behaviorPrototype)
    expect(repeatedSketch.outcomeCredit).toBe(smallSketch.outcomeCredit)
    expect(repeatedSketch.sourceGroupIds).toEqual(['source-a', 'source-b'])
    expect(repeatedSketch.observationCount).toBe(101)
    expect(repeatedSketch.confidenceMass).toBe(2)
  })

  it('girdi sırasından bağımsız aynı sketch fingerprintini üretir', () => {
    const values = [
      observation('c', 'source-c', [-1, 0]),
      observation('a', 'source-a', [1, 0]),
      observation('b', 'source-b', [0, 1]),
    ]
    const left = aggregatePageSketches(values, { behaviorDimension: 2 })[0]
    const right = aggregatePageSketches([...values].reverse(), { behaviorDimension: 2 })[0]
    expect(left).toEqual(right)
    expect(left.fingerprint).toBe(right.fingerprint)
  })

  it('sınırlı kanıt defterinde önce farklı kaynak gruplarını temsil eder', () => {
    const values = [
      ...Array.from({ length: 20 }, (_, index) => observation(`a-${index}`, 'source-a', [1, 0])),
      observation('b', 'source-b', [0, 1]),
      observation('c', 'source-c', [-1, 0]),
    ]
    const sketch = aggregatePageSketches(values, { behaviorDimension: 2, maximumEvidence: 2 })[0]
    expect(sketch.evidence.map((entry) => entry.sourceGroupId)).toEqual(['source-a', 'source-b'])
  })
})
