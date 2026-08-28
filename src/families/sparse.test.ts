import { describe, expect, it } from 'vitest'
import {
  createSparseVector,
  normalizedPositive,
  sparseCosine,
  sparseVectorDigest,
  validateSparseVector,
  weightedSparseMean,
} from './sparse'
import type { SparseVector } from './types'

describe('seyrek aile matematiği', () => {
  it('tekrar eden koordinatları birleştirir, sıfırı siler ve top-k sınırını korur', () => {
    const vector = createSparseVector(16, [
      { index: 7, value: 2 },
      { index: 2, value: 1 },
      { index: 7, value: -0.5 },
      { index: 4, value: 3 },
      { index: 2, value: -1 },
      { index: 9, value: -4 },
    ], 2)

    expect(vector.entries.map((entry) => entry.index)).toEqual([4, 9])
    expect(vector.entries[0].value).toBeCloseTo(0.6)
    expect(vector.entries[1].value).toBeCloseTo(-0.8)
    expect(vector.norm).toBe(1)
    expect(() => validateSparseVector(vector)).not.toThrow()
  })

  it('map, koordinat ve girdi sırasından bağımsız aynı vektörü ve digest değerini üretir', () => {
    const first = createSparseVector(32, [
      { index: 8, value: -3 },
      { index: 1, value: 2 },
      { index: 17, value: 1 },
    ])
    const second = createSparseVector(32, new Map([
      [17, 1],
      [8, -3],
      [1, 2],
    ]))

    expect(second).toEqual(first)
    expect(sparseVectorDigest(second)).toBe(sparseVectorDigest(first))
  })

  it('bozuk sıralama, tekrar, sınır, finite değer ve norm kayıtlarını reddeder', () => {
    const cases: SparseVector[] = [
      { dimension: 4, entries: [{ index: 2, value: 1 }, { index: 1, value: 1 }], norm: Math.SQRT2 },
      { dimension: 4, entries: [{ index: 1, value: 1 }, { index: 1, value: 1 }], norm: Math.SQRT2 },
      { dimension: 4, entries: [{ index: 4, value: 1 }], norm: 1 },
      { dimension: 4, entries: [{ index: 1, value: Number.NaN }], norm: 1 },
      { dimension: 4, entries: [{ index: 1, value: 1 }], norm: 0.5 },
    ]

    for (const vector of cases) expect(() => validateSparseVector(vector)).toThrow()
  })

  it('kosinüsü sonlu ve [-1, 1] içinde tutar; sıfır ve farklı uzayları nötr sayar', () => {
    const positive = createSparseVector(8, [{ index: 2, value: 1 }])
    const opposite = createSparseVector(8, [{ index: 2, value: -1 }])
    const zero = createSparseVector(8, [])
    const otherSpace = createSparseVector(9, [{ index: 2, value: 1 }])

    expect(sparseCosine(positive, positive)).toBeCloseTo(1)
    expect(sparseCosine(positive, opposite)).toBeCloseTo(-1)
    expect(sparseCosine(positive, zero)).toBe(0)
    expect(sparseCosine(positive, otherSpace)).toBe(0)
  })

  it('ağırlıklı prototipi sıra bağımsız üretir ve davranış uzayı karışmasını reddeder', () => {
    const left = createSparseVector(8, [{ index: 1, value: 1 }])
    const right = createSparseVector(8, [{ index: 3, value: 1 }])
    const first = weightedSparseMean([
      { vector: left, weight: 1 },
      { vector: right, weight: 3 },
    ], 4)
    const second = weightedSparseMean([
      { vector: right, weight: 3 },
      { vector: left, weight: 1 },
    ], 4)

    expect(second.entries).toEqual(first.entries)
    expect(sparseVectorDigest(second)).toBe(sparseVectorDigest(first))
    expect(() => weightedSparseMean([
      { vector: left, weight: 1 },
      { vector: createSparseVector(9, [{ index: 1, value: 1 }]), weight: 1 },
    ], 4)).toThrow('Farklı davranış uzayları')
  })

  it('pozitif skorları toplamı bir olacak şekilde normalize eder', () => {
    const weighted = normalizedPositive([
      { id: 'a', score: 3 },
      { id: 'b', score: -2 },
      { id: 'c', score: 1 },
    ])

    expect(weighted.map((item) => item.weight)).toEqual([0.75, 0, 0.25])
    expect(weighted.reduce((sum, item) => sum + item.weight, 0)).toBeCloseTo(1)
    expect(normalizedPositive([{ score: 0 }, { score: -1 }]).map((item) => item.weight)).toEqual([0, 0])
  })
})
