import assert from 'node:assert/strict'
import test from 'node:test'
import {
  checkTurkishSpelling,
  tokenizeTurkishWords,
  TURKISH_SPELLING_POLICY_ID,
} from '../turkish-spelling.mjs'

test('offline Turkish gate rejects known pilot hallucinations', () => {
  for (const word of ['watching', 'teatride', 'meydanada', 'serayda', 'çatışkın']) {
    const result = checkTurkishSpelling(`Ali ${word} bekledi.`)
    assert.equal(result.valid, false, word)
    assert.ok(result.diagnostics.some((item) => item.token === word), word)
  }
})

test('offline Turkish gate accepts representative inflections and proper-name apostrophes', () => {
  const result = checkTurkishSpelling("Ali'nin arkadaşları Türkiye'de kütüphanede çocuklarla çalıştı.")
  assert.equal(result.valid, true)
  assert.equal(result.unknownCount, 0)
  assert.equal(result.policyId, TURKISH_SPELLING_POLICY_ID)

  for (const text of [
    "Ankara'ya kitaplarımızdan birkaçını götürdü.",
    "Zeynep'in öğretmenleri evlerimizde buluştu.",
    'Arkadaşlarımızın arabalarıyla pencereden baktık.',
  ]) assert.equal(checkTurkishSpelling(text).valid, true, text)
})

test('tokenizer keeps Unicode Turkish words and curly-apostrophe suffixes intact', () => {
  assert.deepEqual(
    tokenizeTurkishWords("Ayşe, Türkiye’de Ali’nin kitabını okudu.").map((item) => item.token),
    ['Ayşe', "Türkiye'de", "Ali'nin", 'kitabını', 'okudu'],
  )
})

test('unknown diagnostics are deterministically bounded without hiding the full count', () => {
  const text = 'Watching teatride meydanada serayda çatışkın bekledi.'
  const first = checkTurkishSpelling(text, { maximumDiagnostics: 2, maximumSuggestions: 1 })
  const replay = checkTurkishSpelling(text, { maximumDiagnostics: 2, maximumSuggestions: 1 })
  assert.deepEqual(first, replay)
  assert.equal(first.valid, false)
  assert.equal(first.unknownCount, 5)
  assert.equal(first.diagnostics.length, 2)
  assert.equal(first.diagnosticsTruncated, true)
})
