import { describe, expect, it } from 'vitest'
import type { EventGraphV1 } from '../thought-runtime/types'
import { eventGraphToLanguageIntent } from './intent'

function graph(overrides: Partial<EventGraphV1> = {}): EventGraphV1 {
  return {
    format: 'event-graph-v1',
    actor: 'Ada',
    action: 'taşıdı',
    patient: 'kırmızı kutu',
    instrument: 'kanca',
    operators: [],
    negated: false,
    goals: [],
    ...overrides,
  }
}

describe('EventGraphV1 role gateway', () => {
  it('maps structural roles into a deterministic transient intent', () => {
    const source = graph({ goals: [{ relation: 'hedef', target: 'depo' }] })
    const before = structuredClone(source)
    const first = eventGraphToLanguageIntent(source)
    const replay = eventGraphToLanguageIntent(source)

    expect(first).toEqual(replay)
    expect(source).toEqual(before)
    expect(first.format).toBe('language-output-intent')
    expect(first.tense).toBe('past')
    expect(first.polarity).toBe('positive')
    expect(first.slots.map((slot) => slot.role)).toEqual([
      'actor', 'action', 'patient', 'instrument', 'goal',
    ])
    expect(first.slots.find((slot) => slot.role === 'action')?.candidates[0]?.surface).toBe('taşı')
  })

  it('preserves separate operator and goal witnesses instead of merging their identity', () => {
    const intent = eventGraphToLanguageIntent(graph({
      operators: ['yarın', 'hemen'],
      goals: [{ relation: 'hedef', target: 'depo' }, { relation: 'sonuç', target: 'iskele' }],
    }))
    expect(intent.tense).toBe('future')
    expect(intent.slots.filter((slot) => slot.role === 'operator')).toHaveLength(2)
    expect(intent.slots.filter((slot) => slot.role === 'goal')).toHaveLength(2)
    expect(new Set(intent.slots.flatMap((slot) => slot.candidates.map((candidate) => candidate.memberKey))).size)
      .toBe(intent.slots.length)
  })

  it('detects negative finite actions and question intent', () => {
    const intent = eventGraphToLanguageIntent(graph({ action: 'taşımadı', operators: ['mı'] }))
    expect(intent.tense).toBe('past')
    expect(intent.polarity).toBe('negative')
    expect(intent.sentenceKind).toBe('question')
    expect(intent.slots.find((slot) => slot.role === 'action')?.candidates[0]?.surface).toBe('taşı')
  })

  it('recognizes explicit operator negation and standalone terminal markers', () => {
    expect(eventGraphToLanguageIntent(graph({ operators: ['değil'] })).polarity).toBe('negative')
    expect(eventGraphToLanguageIntent(graph({ operators: ['?'] })).sentenceKind).toBe('question')
    expect(eventGraphToLanguageIntent(graph({ operators: ['!'] })).sentenceKind).toBe('exclamation')
  })
})
