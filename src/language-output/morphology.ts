import type { LanguageMorphFeatures } from './types.ts'

export type TurkishCase = NonNullable<LanguageMorphFeatures['case']>
export type TurkishTense = NonNullable<LanguageMorphFeatures['tense']>
export type TurkishPolarity = NonNullable<LanguageMorphFeatures['polarity']>

export interface TurkishMorphologyRealization {
  stem: string
  morphemes: string[]
  surface: string
  features: LanguageMorphFeatures
}

export interface TurkishVerbRealization extends TurkishMorphologyRealization {
  negativeMorpheme: string | null
  tenseMorpheme: string
  personMorpheme: string
}

const VOWELS = new Set(['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü'])
const BACK_VOWELS = new Set(['a', 'ı', 'o', 'u'])
const VOICELESS_CONSONANTS = new Set(['ç', 'f', 'h', 'k', 'p', 's', 'ş', 't'])
const SOFTENED_FINALS = new Map([
  ['p', 'b'],
  ['ç', 'c'],
  ['t', 'd'],
  ['k', 'ğ'],
])

export function normalizeTurkishSurface(value: string): string {
  return String(value ?? '')
    .normalize('NFC')
    .replace(/[‘’ʼ]/gu, "'")
    .replace(/\s+/gu, ' ')
    .trim()
}

function lower(value: string): string {
  return value.toLocaleLowerCase('tr-TR')
}

function lastLetter(value: string): string {
  const letters = [...lower(value)].filter((character) => /[\p{L}]/u.test(character))
  return letters.at(-1) ?? ''
}

export function lastTurkishVowel(value: string): string | null {
  const characters = [...lower(value)]
  for (let index = characters.length - 1; index >= 0; index -= 1) {
    if (VOWELS.has(characters[index])) return characters[index]
  }
  return null
}

/** Four-way high-vowel harmony: ı / i / u / ü. */
export function turkishHighVowel(value: string): 'ı' | 'i' | 'u' | 'ü' {
  switch (lastTurkishVowel(value)) {
    case 'e':
    case 'i':
      return 'i'
    case 'o':
    case 'u':
      return 'u'
    case 'ö':
    case 'ü':
      return 'ü'
    default:
      return 'ı'
  }
}

/** Two-way low-vowel harmony: a / e. */
export function turkishLowVowel(value: string): 'a' | 'e' {
  const vowel = lastTurkishVowel(value)
  return vowel && !BACK_VOWELS.has(vowel) ? 'e' : 'a'
}

function endsWithVowel(value: string): boolean {
  return VOWELS.has(lastLetter(value))
}

function dental(value: string): 'd' | 't' {
  return VOICELESS_CONSONANTS.has(lastLetter(value)) ? 't' : 'd'
}

function pluralMorpheme(value: string): string {
  return `l${turkishLowVowel(value)}r`
}

function caseMorpheme(value: string, grammaticalCase: TurkishCase): string {
  const vowelFinal = endsWithVowel(value)
  switch (grammaticalCase) {
    case 'accusative':
      return `${vowelFinal ? 'y' : ''}${turkishHighVowel(value)}`
    case 'dative':
      return `${vowelFinal ? 'y' : ''}${turkishLowVowel(value)}`
    case 'locative':
      return `${dental(value)}${turkishLowVowel(value)}`
    case 'ablative':
      return `${dental(value)}${turkishLowVowel(value)}n`
    case 'instrumental':
      return `${vowelFinal ? 'y' : ''}l${turkishLowVowel(value)}`
    default:
      return ''
  }
}

function finalLexicalWord(value: string): string {
  return lower(value).match(/[\p{L}]+$/u)?.[0] ?? ''
}

function softenNominalStem(
  value: string,
  suffix: string,
  features: LanguageMorphFeatures,
): string {
  if (!suffix || features.properNoun || features.consonantMutation === 'none') return value
  const initial = suffix.replace(/^'/u, '')[0] ?? ''
  if (!VOWELS.has(initial)) return value

  const word = finalLexicalWord(value)
  const final = word.at(-1) ?? ''
  const softened = SOFTENED_FINALS.get(final)
  if (!softened) return value
  const vowelCount = [...word].filter((character) => VOWELS.has(character)).length
  if (features.consonantMutation !== 'soften' && vowelCount < 2) return value
  return `${value.slice(0, -1)}${softened}`
}

/**
 * Joins a nominal stem and Turkish surface suffixes without guessing a new
 * lexical identity. The caller explicitly marks proper nouns; capitalization
 * alone is not treated as semantic evidence.
 */
export function realizeTurkishNoun(
  value: string,
  features: LanguageMorphFeatures = {},
): TurkishMorphologyRealization {
  const stem = normalizeTurkishSurface(value).replace(/[.!?,;:]+$/u, '')
  const grammaticalCase = features.case ?? 'none'
  const number = features.number ?? 'singular'
  const morphemes: string[] = []
  let inflectionBase = stem

  if (number === 'plural') {
    const plural = pluralMorpheme(inflectionBase)
    morphemes.push(plural)
    inflectionBase += plural
  }

  const nominalCase = caseMorpheme(inflectionBase, grammaticalCase)
  if (nominalCase) {
    // Turkish orthography does not put an apostrophe before a plural suffix;
    // once pluralized, following case suffixes continue without one as well.
    const apostrophe = features.properNoun && number !== 'plural' ? "'" : ''
    morphemes.push(`${apostrophe}${nominalCase}`)
  }

  const surfaceStem = number === 'singular'
    ? softenNominalStem(stem, nominalCase, features)
    : stem

  return {
    stem: surfaceStem,
    morphemes,
    surface: `${surfaceStem}${morphemes.join('')}`,
    features: {
      ...features,
      lemma: features.lemma ?? stem,
      partOfSpeech: features.partOfSpeech ?? 'noun',
      number,
      case: grammaticalCase,
      consonantMutation: features.consonantMutation ?? 'auto',
    },
  }
}

function presentContinuous(lemma: string, polarity: TurkishPolarity) {
  if (polarity === 'negative') {
    return { stem: lemma, negative: 'm', tense: `${turkishHighVowel(lemma)}yor` }
  }
  const final = lastLetter(lemma)
  if (final === 'a' || final === 'e') {
    const stem = lemma.slice(0, -1)
    return { stem, negative: null, tense: `${turkishHighVowel(stem)}yor` }
  }
  if (endsWithVowel(lemma)) return { stem: lemma, negative: null, tense: 'yor' }
  return { stem: lemma, negative: null, tense: `${turkishHighVowel(lemma)}yor` }
}

function aoristMorpheme(lemma: string): string {
  if (endsWithVowel(lemma)) return 'r'
  const vowelCount = [...lower(lemma)].filter((character) => VOWELS.has(character)).length
  return vowelCount <= 1
    ? `${turkishLowVowel(lemma)}r`
    : `${turkishHighVowel(lemma)}r`
}

function tenseParts(lemma: string, tense: TurkishTense, polarity: TurkishPolarity) {
  if (tense === 'present-continuous') return presentContinuous(lemma, polarity)

  const negative = polarity === 'negative' ? `m${turkishLowVowel(lemma)}` : null
  const base = `${lemma}${negative ?? ''}`
  switch (tense) {
    case 'past':
      return { stem: lemma, negative, tense: `${dental(base)}${turkishHighVowel(base)}` }
    case 'evidential-past':
      return { stem: lemma, negative, tense: `m${turkishHighVowel(base)}ş` }
    case 'future':
      return {
        stem: lemma,
        negative,
        tense: `${endsWithVowel(base) ? 'y' : ''}${turkishLowVowel(base)}c${turkishLowVowel(base)}k`,
      }
    case 'aorist':
      return polarity === 'negative'
        ? { stem: lemma, negative, tense: 'z' }
        : { stem: lemma, negative: null, tense: aoristMorpheme(lemma) }
    default:
      return { stem: lemma, negative, tense: '' }
  }
}

function personMorpheme(value: string, tense: TurkishTense, person: 1 | 2 | 3, number: 'singular' | 'plural') {
  if (person === 3) return number === 'plural' ? pluralMorpheme(value) : ''
  if (tense === 'past') {
    if (person === 1) return number === 'plural' ? 'k' : 'm'
    return number === 'plural' ? `n${turkishHighVowel(value)}z` : 'n'
  }
  const high = turkishHighVowel(value)
  if (person === 1) return number === 'plural' ? `${high}z` : `${high}m`
  return number === 'plural' ? `s${high}n${high}z` : `s${high}n`
}

/** Deterministic finite-verb realization used by the controlled output path. */
export function realizeTurkishVerb(
  value: string,
  features: LanguageMorphFeatures = {},
): TurkishVerbRealization {
  const lemma = normalizeTurkishSurface(features.lemma ?? value)
    .replace(/[.!?,;:]+$/u, '')
    .toLocaleLowerCase('tr-TR')
  const tense = features.tense ?? 'past'
  const polarity = features.polarity ?? 'positive'
  const person = features.person ?? 3
  const number = features.number ?? 'singular'
  const parts = tenseParts(lemma, tense, polarity)
  const withoutPerson = `${parts.stem}${parts.negative ?? ''}${parts.tense}`
  const personSuffix = personMorpheme(withoutPerson, tense, person, number)

  // Future k softens before a vowel-initial first-person suffix.
  const softened = tense === 'future' && person === 1 && personSuffix && withoutPerson.endsWith('k')
    ? `${withoutPerson.slice(0, -1)}ğ`
    : withoutPerson
  const morphemes = [parts.negative, parts.tense, personSuffix].filter((item): item is string => Boolean(item))

  return {
    stem: parts.stem,
    negativeMorpheme: parts.negative,
    tenseMorpheme: parts.tense,
    personMorpheme: personSuffix,
    morphemes,
    surface: `${softened}${personSuffix}`,
    features: {
      ...features,
      lemma,
      partOfSpeech: 'verb',
      tense,
      polarity,
      person,
      number,
    },
  }
}

export function turkishQuestionParticle(value: string): 'mı' | 'mi' | 'mu' | 'mü' {
  return `m${turkishHighVowel(value)}` as 'mı' | 'mi' | 'mu' | 'mü'
}
