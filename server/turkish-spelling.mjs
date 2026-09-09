import nspell from 'nspell'
import dictionary from 'dictionary-tr'

export const TURKISH_SPELLING_POLICY_ID = 'nspell-2.1.5_dictionary-tr-2.0.0_policy-v1'

const spell = nspell(dictionary)
const WORD_PATTERN = /[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}]+)?/gu
const ACRONYM_PATTERN = /^[A-ZÇĞİÖŞÜ]{2,12}$/u
const TITLE_CASE_PATTERN = /^[A-ZÇĞİÖŞÜ][a-zçğıöşüâîû]+$/u

// Hunspell dictionaries intentionally do not try to contain every personal
// name. This small, explicit supplement covers common Turkish names that are
// absent from dictionary-tr; it is policy data, never learned from a campaign.
const SUPPLEMENTAL_PROPER_NAMES = new Set([
  'Ahmet', 'Aylin', 'Burak', 'Can', 'Cem', 'Ceren', 'Derya', 'Esra', 'Fatma',
  'Gökhan', 'Hakan', 'Hasan', 'Hüseyin', 'İbrahim', 'İrem', 'Leyla', 'Merve',
  'Murat', 'Ömer', 'Özge', 'Seda', 'Serkan', 'Sevgi', 'Sibel', 'Tolga', 'Uğur',
  'Yasemin', 'Zeynep',
])

const APOSTROPHE_SUFFIXES = new Set([
  // Simple case, genitive, instrumental, plural and relative suffixes.
  'a', 'e', 'ya', 'ye', 'ı', 'i', 'u', 'ü', 'yı', 'yi', 'yu', 'yü',
  'da', 'de', 'ta', 'te', 'dan', 'den', 'tan', 'ten',
  'ın', 'in', 'un', 'ün', 'nın', 'nin', 'nun', 'nün',
  'la', 'le', 'yla', 'yle', 'ki', 'nki', 'daki', 'deki', 'taki', 'teki',
  'lar', 'ler', 'lara', 'lere', 'ları', 'leri', 'ların', 'lerin',
  'larla', 'lerle', 'larda', 'lerde', 'lardan', 'lerden',
  // Common possessive chains used with people, places, institutions and dates.
  'm', 'ım', 'im', 'um', 'üm', 'mın', 'min', 'mun', 'mün',
  'ma', 'me', 'mı', 'mi', 'mu', 'mü', 'mda', 'mde', 'mdan', 'mden', 'mla', 'mle',
  'mız', 'miz', 'muz', 'müz', 'mıza', 'mize', 'mızda', 'mizde', 'mızdan', 'mizden',
  'n', 'na', 'ne', 'nı', 'ni', 'nu', 'nü', 'nda', 'nde', 'ndan', 'nden', 'nla', 'nle',
  'nız', 'niz', 'nuz', 'nüz', 'nıza', 'nize', 'nızda', 'nizde', 'nızdan', 'nizden',
  'sı', 'si', 'su', 'sü', 'sına', 'sine', 'sında', 'sinde', 'sından', 'sinden',
  'ımız', 'imiz', 'umuz', 'ümüz', 'ımızın', 'imizin', 'umuzun', 'ümüzün',
])

function normalizeToken(value) {
  return String(value).normalize('NFC').replace(/’/gu, "'")
}

function dictionaryCorrect(word) {
  if (!word) return false
  if (spell.correct(word)) return true
  const lower = word.toLocaleLowerCase('tr-TR')
  if (lower !== word && spell.correct(lower)) return true
  // Turkish circumflexes are preserved in careful writing but dictionary-tr
  // contains some of the same lemmas only in their modern plain-vowel form.
  const plainVowels = lower.replace(/â/gu, 'a').replace(/î/gu, 'i').replace(/û/gu, 'u')
  return plainVowels !== lower && spell.correct(plainVowels)
}

function properBase(base) {
  return dictionaryCorrect(base) || SUPPLEMENTAL_PROPER_NAMES.has(base) || ACRONYM_PATTERN.test(base)
}

function validApostropheToken(base, suffix) {
  if (!suffix || !APOSTROPHE_SUFFIXES.has(suffix.toLocaleLowerCase('tr-TR'))) return false
  // The apostrophe itself is strong orthographic evidence of a proper name.
  // Known dictionary/name entries are preferred; open-world title-case names
  // remain valid because no finite offline dictionary can enumerate people.
  return properBase(base) || TITLE_CASE_PATTERN.test(base)
}

function knownToken(token) {
  if (/\p{N}/u.test(token)) return true
  const apostrophe = token.indexOf("'")
  if (apostrophe >= 0) return validApostropheToken(token.slice(0, apostrophe), token.slice(apostrophe + 1))
  return dictionaryCorrect(token) || SUPPLEMENTAL_PROPER_NAMES.has(token) || ACRONYM_PATTERN.test(token)
}

export function tokenizeTurkishWords(text) {
  const normalized = String(text ?? '').normalize('NFC')
  return [...normalized.matchAll(WORD_PATTERN)].map((match) => ({
    token: normalizeToken(match[0]),
    index: match.index,
    end: match.index + match[0].length,
  }))
}

/**
 * Deterministic, offline Turkish spelling/morphology gate. Diagnostics are
 * bounded independently of input length; unknownCount still reports the full
 * count so truncation cannot be mistaken for acceptance.
 */
export function checkTurkishSpelling(text, { maximumDiagnostics = 8, maximumSuggestions = 2 } = {}) {
  if (!Number.isInteger(maximumDiagnostics) || maximumDiagnostics < 0 || maximumDiagnostics > 32) {
    throw new TypeError('maximumDiagnostics 0–32 arasında bir tam sayı olmalı.')
  }
  if (!Number.isInteger(maximumSuggestions) || maximumSuggestions < 0 || maximumSuggestions > 5) {
    throw new TypeError('maximumSuggestions 0–5 arasında bir tam sayı olmalı.')
  }
  const tokens = tokenizeTurkishWords(text)
  const diagnostics = []
  let unknownCount = 0
  let checkedWordCount = 0
  for (const item of tokens) {
    if (/\p{N}/u.test(item.token)) continue
    checkedWordCount += 1
    if (knownToken(item.token)) continue
    unknownCount += 1
    if (diagnostics.length >= maximumDiagnostics) continue
    const base = item.token.split("'", 1)[0]
    diagnostics.push({
      token: item.token,
      index: item.index,
      suggestions: maximumSuggestions ? spell.suggest(base).slice(0, maximumSuggestions) : [],
    })
  }
  return {
    valid: unknownCount === 0,
    policyId: TURKISH_SPELLING_POLICY_ID,
    checkedWordCount,
    unknownCount,
    diagnostics,
    diagnosticsTruncated: unknownCount > diagnostics.length,
  }
}
