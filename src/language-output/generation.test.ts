import { describe, expect, it } from 'vitest'
import type { EventGraphV1 } from '../thought-runtime/types'
import type { LanguageIntentV1, LanguageOutputHeadV1, LanguageOutputScaffoldV1 } from './types'
import { generateTurkishSentence } from './generation'
import { eventGraphToLanguageIntent } from './intent'

function graph(overrides: Partial<EventGraphV1> = {}): EventGraphV1 {
  return {
    format: 'event-graph-v1', actor: 'Ada', action: 'taşıdı', patient: 'kırmızı kutu', instrument: 'kanca',
    operators: [], negated: false, goals: [], ...overrides,
  }
}

describe('controlled Turkish generation chain', () => {
  it('emits order, morphology, punctuation and exactly one EOS trace', () => {
    const intent = eventGraphToLanguageIntent(graph())
    const before = structuredClone(intent)
    const result = generateTurkishSentence(intent)

    expect(result.text).toBe('Ada kırmızı kutuyu kancayla taşıdı.')
    expect(result.stoppedBy).toBe('eos')
    expect(result.steps.map((step) => step.familyId)).toEqual([
      '@language-output/family/word-order',
      '@language-output/family/verb-tense',
      '@language-output/family/case-marking',
      '@language-output/family/connective-negation',
      '@language-output/family/punctuation',
      '@language/EOS/family',
    ])
    expect(result.tokens.filter((token) => token.kind === 'terminal')).toHaveLength(1)
    expect(result.steps.at(-1)?.state.complete).toBe(true)
    expect(intent).toEqual(before)
  })

  it('places Turkish negation before tense and a detached question particle before punctuation', () => {
    const negative = generateTurkishSentence(eventGraphToLanguageIntent(graph({
      action: 'taşı', patient: 'kutu', instrument: null, negated: true,
    })))
    expect(negative.text).toBe('Ada kutuyu taşımadı.')
    expect(negative.tokens.filter((token) => token.familyId === '@language-output/family/connective-negation')
      .some((token) => token.surface === 'ma')).toBe(true)

    const question = generateTurkishSentence(eventGraphToLanguageIntent(graph({
      patient: 'kutu', instrument: null, operators: ['mı'],
    })))
    expect(question.text).toBe('Ada kutuyu taşıdı mı?')
  })

  it('uses dative goals and proper-name apostrophes', () => {
    const result = generateTurkishSentence(eventGraphToLanguageIntent(graph({
      patient: 'Ankara', instrument: null, goals: [{ relation: 'hedef', target: 'depo' }],
    })))
    expect(result.text).toBe("Ada Ankara'yı depoya taşıdı.")
  })

  it('does not infer a proper noun from sentence-style capitalization alone', () => {
    const result = generateTurkishSentence(eventGraphToLanguageIntent(graph({
      patient: 'Kırmızı kutu', instrument: null,
    })))
    expect(result.text).toBe('Ada Kırmızı kutuyu taşıdı.')
  })

  it('uses learned membership only as a deterministic tie-break and never mutates the head', () => {
    const intent: LanguageIntentV1 = {
      format: 'language-output-intent', version: 1, id: 'intent-tie', fingerprint: 'intent-tie-fingerprint',
      slots: [
        { role: 'actor', candidates: [
          { memberKey: 'actor-a', surface: 'Ada', score: 0.5 },
          { memberKey: 'actor-b', surface: 'Bora', score: 0.5 },
        ] },
        { role: 'action', candidates: [{ memberKey: 'action', surface: 'taşı', score: 1 }] },
      ],
      tense: 'past', polarity: 'positive', sentenceKind: 'statement',
    }
    const head = {
      fingerprint: 'head-fingerprint', scaffoldFingerprint: 'scaffold-fingerprint',
      membershipEvidence: [{ memberKey: 'actor-b', familyId: '@language-output/family/word-order', score: 0.4 }],
    } as unknown as LanguageOutputHeadV1
    const scaffold = { fingerprint: 'scaffold-fingerprint' } as LanguageOutputScaffoldV1
    const before = structuredClone(head)
    expect(generateTurkishSentence(intent, scaffold, head).text).toBe('Bora taşıdı.')
    expect(head).toEqual(before)
  })

  it('stops explicitly when an action role cannot be resolved', () => {
    const intent = eventGraphToLanguageIntent(graph({ action: null }))
    const result = generateTurkishSentence(intent)
    expect(result.stoppedBy).toBe('unmatched')
    expect(result.text).toBe('')
    expect(result.tokens).toEqual([])
  })
})
