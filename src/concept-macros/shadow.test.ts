import { describe, expect, it } from 'vitest'
import { evaluateMacroShadow } from './shadow.ts'
import { createMacroHead, ingestMacroTraces } from './state.ts'
import { macroMember, macroTrace, passingShadowCases } from './test-fixtures.ts'

describe('expanded-vs-macro shadow değerlendiricisi', () => {
  it('eş çıktıda tam fidelity, sıfır false activation ve ölçülmüş maliyet kazancı üretir', () => {
    const initial = createMacroHead({
      representationSpaceId: 'space-64-v1', familyRunFingerprint: 'family-run-1',
    }, { minimumMotifLength: 2, maximumMotifLength: 2 })
    const members = [macroMember('a'), macroMember('b')]
    const head = ingestMacroTraces(initial, ['a', 'b', 'c'].map((id) => macroTrace({
      id, members, group: id,
    }))).head
    const macro = head.macros.find((value) => value.kind === 'sequence-macro')!
    const result = evaluateMacroShadow(macro, passingShadowCases(), head.config)
    expect(result).toMatchObject({
      format: 'concept-macro-shadow-report',
      caseCount: 3,
      fidelity: 1,
      falseActivationRate: 0,
      activationRecall: 1,
      holdoutUtility: 0,
      observedCompressionGain: 6,
      passed: true,
      failures: [],
    })
  })

  it('pozitif vakalarda makronun gerçekten tamamlanmasını activation recall ile zorunlu tutar', () => {
    const initial = createMacroHead({
      representationSpaceId: 'space-64-v1', familyRunFingerprint: 'family-run-1',
    }, { minimumMotifLength: 2, maximumMotifLength: 2 })
    const members = [macroMember('a'), macroMember('b')]
    const head = ingestMacroTraces(initial, ['a', 'b', 'c'].map((id) => macroTrace({
      id, members, group: id,
    }))).head
    const macro = head.macros.find((value) => value.kind === 'sequence-macro')!
    const result = evaluateMacroShadow(macro, passingShadowCases().map((value) => ({
      ...value,
      macroActivated: value.shouldActivate ? false : value.macroActivated,
    })), head.config)
    expect(result.activationRecall).toBe(0)
    expect(result.positiveCaseCount).toBe(2)
    expect(result.missedActivationCount).toBe(2)
    expect(result.failures).toContain('activation-recall-below-threshold')
    expect(result.passed).toBe(false)
  })

  it('train vakasını ve aynı vaka kimliğinin farklı içerikle tekrarını reddeder', () => {
    const initial = createMacroHead({
      representationSpaceId: 'space-64-v1', familyRunFingerprint: 'family-run-1',
    }, { minimumMotifLength: 2, maximumMotifLength: 2 })
    const members = [macroMember('a'), macroMember('b')]
    const head = ingestMacroTraces(initial, ['a', 'b', 'c'].map((id) => macroTrace({
      id, members, group: id,
    }))).head
    const macro = head.macros.find((value) => value.kind === 'sequence-macro')!
    expect(() => evaluateMacroShadow(macro, [{
      ...passingShadowCases()[0],
      split: 'train' as never,
    }], head.config)).toThrow(/validation\/test/u)
    expect(() => evaluateMacroShadow(macro, [
      passingShadowCases()[0],
      { ...passingShadowCases()[0], macroCost: 2 },
    ], head.config)).toThrow(/farklı içerik/u)
  })
})
