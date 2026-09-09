import { describe, expect, it } from 'vitest'
import {
  realizeTurkishNoun,
  realizeTurkishVerb,
  turkishHighVowel,
  turkishLowVowel,
  turkishQuestionParticle,
} from './morphology'

describe('deterministic Turkish morphology', () => {
  it('applies vowel harmony and case allomorphs without a lookup model', () => {
    expect(turkishHighVowel('kapı')).toBe('ı')
    expect(turkishHighVowel('ev')).toBe('i')
    expect(turkishHighVowel('okul')).toBe('u')
    expect(turkishHighVowel('göl')).toBe('ü')
    expect(turkishLowVowel('okul')).toBe('a')
    expect(turkishLowVowel('göl')).toBe('e')

    expect(realizeTurkishNoun('kırmızı kutu', { case: 'accusative' }).surface).toBe('kırmızı kutuyu')
    expect(realizeTurkishNoun('ev', { case: 'dative' }).surface).toBe('eve')
    expect(realizeTurkishNoun('ağaç', { case: 'locative' }).surface).toBe('ağaçta')
    expect(realizeTurkishNoun('okul', { case: 'ablative' }).surface).toBe('okuldan')
    expect(realizeTurkishNoun('kanca', { case: 'instrumental' }).surface).toBe('kancayla')
    expect(realizeTurkishNoun('robot kol', { case: 'instrumental' }).surface).toBe('robot kolla')
    expect(realizeTurkishNoun('ekmek', { case: 'accusative' }).surface).toBe('ekmeği')
    expect(realizeTurkishNoun('kitap', { case: 'accusative' }).surface).toBe('kitabı')
    expect(realizeTurkishNoun('ağaç', { case: 'dative' }).surface).toBe('ağaca')
    expect(realizeTurkishNoun('bulut', { case: 'accusative', consonantMutation: 'none' }).surface).toBe('bulutu')
  })

  it('uses an apostrophe only when proper-noun evidence is explicit', () => {
    expect(realizeTurkishNoun('Ada', { case: 'accusative', properNoun: true }).surface).toBe("Ada'yı")
    expect(realizeTurkishNoun('Ankara', { case: 'dative', properNoun: true }).surface).toBe("Ankara'ya")
    expect(realizeTurkishNoun('ada', { case: 'accusative' }).surface).toBe('adayı')
  })

  it('separates negation, tense and person morphemes while preserving the surface', () => {
    expect(realizeTurkishVerb('taşı', { tense: 'past' }).surface).toBe('taşıdı')
    expect(realizeTurkishVerb('kes', { tense: 'past' }).surface).toBe('kesti')
    const negative = realizeTurkishVerb('taşı', { tense: 'past', polarity: 'negative' })
    expect(negative.surface).toBe('taşımadı')
    expect(negative.negativeMorpheme).toBe('ma')
    expect(negative.tenseMorpheme).toBe('dı')

    expect(realizeTurkishVerb('taşı', { tense: 'present-continuous' }).surface).toBe('taşıyor')
    expect(realizeTurkishVerb('bekle', { tense: 'present-continuous' }).surface).toBe('bekliyor')
    expect(realizeTurkishVerb('oku', { tense: 'present-continuous' }).surface).toBe('okuyor')
    expect(realizeTurkishVerb('taşı', { tense: 'present-continuous', polarity: 'negative' }).surface).toBe('taşımıyor')
    expect(realizeTurkishVerb('taşı', { tense: 'future' }).surface).toBe('taşıyacak')
    expect(realizeTurkishVerb('gel', { tense: 'evidential-past' }).surface).toBe('gelmiş')
    expect(realizeTurkishVerb('yap', { tense: 'aorist' }).surface).toBe('yapar')
  })

  it('harmonizes the detached question particle with the finite predicate', () => {
    expect(turkishQuestionParticle('taşıdı')).toBe('mı')
    expect(turkishQuestionParticle('geldi')).toBe('mi')
    expect(turkishQuestionParticle('okudu')).toBe('mu')
    expect(turkishQuestionParticle('gördü')).toBe('mü')
  })
})
