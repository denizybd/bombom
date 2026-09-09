import { describe, expect, it } from 'vitest'
import {
  classifyLanguageOutputSeed,
  computeLanguageOutputScaffoldFingerprint,
  createLanguageOutputScaffold,
  LANGUAGE_OUTPUT_FAMILY_DEFINITIONS,
  LANGUAGE_OUTPUT_FAMILY_IDS,
  LANGUAGE_OUTPUT_MANUAL_TRANSITIONS,
  LANGUAGE_ROLE_GATEWAY_ID,
  validateLanguageOutputScaffold,
} from './scaffold'

describe('elle tohumlanan dil çıktı scaffoldı', () => {
  it('altı sabit aileyi, sanal rol kapısını ve kenar-ada metadatasını deterministik kurar', () => {
    const first = createLanguageOutputScaffold()
    const second = createLanguageOutputScaffold()

    expect(first).toEqual(second)
    expect(first.fingerprint).toBe(second.fingerprint)
    expect(first.roleGateway).toMatchObject({ id: LANGUAGE_ROLE_GATEWAY_ID, virtual: true })
    expect(first.families).toHaveLength(6)
    expect(new Set(first.families.map((family) => family.id))).toEqual(new Set(Object.values(LANGUAGE_OUTPUT_FAMILY_IDS)))
    expect(first.families.map((family) => family.order).sort()).toEqual([1, 2, 3, 4, 5, 6])
    expect(first.families.every((family) => (
      family.frozen && family.acceptedSpanLevels.length > 0 &&
      family.atlas.layer === 'language-output' && family.atlas.placement === 'peripheral-island' &&
      family.atlas.radialScale > 1
    ))).toBe(true)
    expect(first.manualTransitions).toEqual(LANGUAGE_OUTPUT_MANUAL_TRANSITIONS)
    expect(first.families).toEqual(LANGUAGE_OUTPUT_FAMILY_DEFINITIONS)
    expect(Object.isFrozen(LANGUAGE_OUTPUT_FAMILY_DEFINITIONS)).toBe(true)
    expect(Object.isFrozen(LANGUAGE_OUTPUT_FAMILY_DEFINITIONS[0])).toBe(true)
    expect(Object.isFrozen(LANGUAGE_OUTPUT_FAMILY_DEFINITIONS[0].acceptedSpanLevels)).toBe(true)
    expect(Object.isFrozen(LANGUAGE_OUTPUT_MANUAL_TRANSITIONS[0])).toBe(true)
    validateLanguageOutputScaffold(first)
  })

  it('dışarı verilen kopyanın değiştirilmesi sonraki scaffoldı ve sabit tanımları kirletmez', () => {
    const mutated = createLanguageOutputScaffold()
    mutated.families[0].label = 'bozuk'
    mutated.families[0].acceptedSpanLevels.length = 0
    mutated.seedNeurons[0].features.partOfSpeech = 'punctuation'
    mutated.manualTransitions[0].priorEvidence = 99

    const fresh = createLanguageOutputScaffold()
    expect(fresh.families[0].label).toBe('Sözcük sırası')
    expect(fresh.families[0].acceptedSpanLevels.length).toBeGreaterThan(0)
    expect(fresh.seedNeurons[0].features.partOfSpeech).toBe('control')
    expect(fresh.manualTransitions[0].priorEvidence).toBe(1)
  })

  it('EOS, noktalama, bağlaç, zaman, hâl ve genel sıra kanıtını temkinli sınıflandırır', () => {
    expect(classifyLanguageOutputSeed({ surface: '', granularity: 'token', eos: true })[0]).toMatchObject({
      familyId: LANGUAGE_OUTPUT_FAMILY_IDS.eos, confidence: 1, membershipEligible: false,
    })
    expect(classifyLanguageOutputSeed({ surface: '?', granularity: 'token' })[0]).toMatchObject({
      familyId: LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, confidence: 1, membershipEligible: true,
    })
    expect(classifyLanguageOutputSeed({ surface: 'Çünkü', granularity: 'token' })[0]?.familyId)
      .toBe(LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation)
    expect(classifyLanguageOutputSeed({ surface: 'kesti', granularity: 'token' })[0]?.familyId)
      .toBe(LANGUAGE_OUTPUT_FAMILY_IDS.verbTense)
    expect(classifyLanguageOutputSeed({ surface: 'depoya', granularity: 'token' })[0]?.familyId)
      .toBe(LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking)
    expect(classifyLanguageOutputSeed({ surface: 'masa', granularity: 'token' })[0]).toMatchObject({
      familyId: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, membershipEligible: false,
    })
    expect(classifyLanguageOutputSeed({ surface: 'xyz', granularity: 'paragraph' })).toEqual([])
  })

  it('fingerprint veya sabit aile kümesi değiştirildiğinde fail closed davranır', () => {
    const scaffold = createLanguageOutputScaffold()
    expect(() => validateLanguageOutputScaffold({ ...scaffold, fingerprint: 'yanlış' })).toThrow(/fingerprint/u)
    expect(() => validateLanguageOutputScaffold({ ...scaffold, families: scaffold.families.slice(1) })).toThrow(/altı sabit aile/u)

    const { fingerprint: _fingerprint, ...tamperedBase } = structuredClone(scaffold)
    tamperedBase.families[0].label = 'Kendini geçerli gösteren bozuk aile'
    const selfSignedTampering = {
      ...tamperedBase,
      fingerprint: computeLanguageOutputScaffoldFingerprint(tamperedBase),
    }
    expect(() => validateLanguageOutputScaffold(selfSignedTampering)).toThrow(/sabit aile/u)
  })
})
