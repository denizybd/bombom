import { beforeEach, describe, expect, it, vi } from 'vitest'

const digestCounter = vi.hoisted(() => ({ calls: 0 }))

vi.mock('../families/sparse', async (importOriginal) => {
  const original = await importOriginal<typeof import('../families/sparse')>()
  return {
    ...original,
    stableDigest(parts: readonly unknown[]) {
      digestCounter.calls += 1
      return original.stableDigest(parts)
    },
  }
})

import {
  compareText,
  deterministicFirst,
  deterministicOrder,
  seededUnit,
} from './deterministic'
import { attachCorruptions } from './corpus'
import { createLanguageExperienceConfig } from './config'
import type { LanguageExample, SurfaceBinding, SurfacePieceLevel } from './types'

interface Candidate {
  id: string
  key: string
  eligible: boolean
  payload: { values: number[] }
}

function legacyOrder<T>(values: readonly T[], seed: number, key: (value: T) => string): T[] {
  return [...values].sort((left, right) => {
    const leftKey = key(left)
    const rightKey = key(right)
    return seededUnit(seed, leftKey) - seededUnit(seed, rightKey) || compareText(leftKey, rightKey)
  })
}

function binding(memberKey: string): SurfaceBinding {
  return {
    memberKey,
    identityDigest: `digest:${memberKey}`,
    score: 1,
    familyWeights: [{ familyId: `family:${memberKey}`, weight: 1 }],
  }
}

function example(
  id: string,
  normalizedText: string,
  memberKeys: string[],
  overrides: Partial<LanguageExample> = {},
): LanguageExample {
  return {
    id,
    sourceId: `source:${id}`,
    sourceDigest: `source-digest:${id}`,
    split: 'train',
    granularity: 'token' as SurfacePieceLevel,
    parentPieceId: `parent:${id}`,
    prefixPieceIds: [],
    targetPieceId: `piece:${id}`,
    targetText: normalizedText,
    targetNormalizedText: normalizedText,
    targetBindings: memberKeys.map(binding),
    eos: false,
    corruptionBindings: [],
    ...overrides,
  }
}

describe('deterministik seçim', () => {
  beforeEach(() => {
    digestCounter.calls = 0
  })

  it('sıralama ve tek-geçiş seçimini eski comparator ile derin eşdeğer tutar', () => {
    const values: Candidate[] = [
      { id: 'a', key: 'zeytin', eligible: true, payload: { values: [1, 2] } },
      { id: 'b', key: 'çınar', eligible: false, payload: { values: [3] } },
      { id: 'c', key: 'armut', eligible: true, payload: { values: [4, 5] } },
      { id: 'd', key: 'üzüm', eligible: true, payload: { values: [6] } },
    ]
    const seed = 20260829
    const expectedOrder = legacyOrder(values, seed, (value) => value.key)
    const expectedFirst = legacyOrder(values.filter((value) => value.eligible), seed, (value) => value.key)[0]

    digestCounter.calls = 0
    expect(deterministicOrder(values, seed, (value) => value.key)).toEqual(expectedOrder)
    expect(digestCounter.calls).toBe(values.length)

    digestCounter.calls = 0
    expect(deterministicFirst(values, seed, (value) => value.key, (value) => value.eligible)).toEqual(expectedFirst)
    expect(digestCounter.calls).toBe(values.filter((value) => value.eligible).length)
  })

  it('eşit rank ve anahtarda kararlı ilk girdiyi korur; boş veya tamamen dışlanan havuzda sonuç üretmez', () => {
    const tied = [
      { id: 'first', key: 'same' },
      { id: 'second', key: 'same' },
    ]

    expect(deterministicFirst(tied, 7, (value) => value.key)).toBe(tied[0])
    expect(deterministicFirst([], 7, String)).toBeUndefined()
    expect(deterministicFirst(tied, 7, (value) => value.key, () => false)).toBeUndefined()
  })

  it('digest bütçesini yalnız uygun aday başına bire ve genel sıralamada eleman başına bire sınırlar', () => {
    const values = Array.from({ length: 128 }, (_, index) => ({
      id: index,
      key: `candidate-${index}`,
      eligible: index % 5 === 0,
    }))
    const eligibleCount = values.filter((value) => value.eligible).length

    deterministicFirst(values, 91, (value) => value.key, (value) => value.eligible)
    expect(digestCounter.calls).toBe(eligibleCount)

    digestCounter.calls = 0
    deterministicOrder(values, 91, (value) => value.key)
    expect(digestCounter.calls).toBe(values.length)
  })

  it('corruption havuzunda kimlik, yüzey ve yalnız-ortak-üye dışlamalarını koruyup her uygun çifti bir kez digestler', () => {
    const examples = [
      example('origin', 'aynı-yüzey', ['a']),
      example('origin', 'başka-yüzey', ['b']),
      example('same-surface', 'aynı-yüzey', ['c']),
      example('shared-only', 'ortak-üye', ['a']),
      example('valid', 'geçerli', ['d']),
      example('empty', 'yalnız', ['e'], { split: 'test', granularity: 'document' }),
      example('eos', '', [], { eos: true, targetPieceId: null }),
    ]
    const eligiblePairCount = examples.reduce((total, current) => {
      if (current.eos) return total
      return total + examples.filter((candidate) => (
        !candidate.eos &&
        candidate.targetBindings.length > 0 &&
        candidate.split === current.split &&
        candidate.granularity === current.granularity &&
        candidate.id !== current.id &&
        candidate.targetNormalizedText !== current.targetNormalizedText &&
        candidate.targetBindings.some((candidateBinding) => (
          !current.targetBindings.some((own) => own.memberKey === candidateBinding.memberKey)
        ))
      )).length
    }, 0)

    digestCounter.calls = 0
    const attached = attachCorruptions(examples, createLanguageExperienceConfig({ seed: 123 }))

    expect(attached[0].corruptionBindings).toEqual([binding('d')])
    expect(attached[5].corruptionBindings).toEqual([])
    expect(attached[6].corruptionBindings).toEqual([])
    expect(digestCounter.calls).toBe(eligiblePairCount)
  })
})
