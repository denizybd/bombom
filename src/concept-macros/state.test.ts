import { describe, expect, it } from 'vitest'
import { computeFamilySignature } from './deterministic.ts'
import { evaluateMacroShadow } from './shadow.ts'
import {
  applyMacroShadowEvaluation,
  applyMacroStructuralEquivalenceAudit,
  createMacroStructuralEquivalenceAudit,
  createMacroHead,
  ingestMacroTraces,
  rebaseMacroHeadReference,
  transitionCompositeLifecycle,
  validateMacroHead,
} from './state.ts'
import { macroMember, macroTrace, passingShadowCases } from './test-fixtures.ts'

function trainedHead() {
  const head = createMacroHead({
    representationSpaceId: 'space-64-v1',
    familyRunFingerprint: 'family-run-1',
  }, { minimumMotifLength: 3, maximumMotifLength: 3 })
  const members = [
    macroMember('alice', ['actor']),
    macroMember('opens', ['action']),
    macroMember('door', ['object']),
  ]
  const traces = ['one', 'two', 'three'].map((id) => macroTrace({
    id,
    members,
    group: `book-${id}`,
  }))
  return { ...ingestMacroTraces(head, traces), members }
}

describe('MacroHead yaşam döngüsü ve kalıcılık', () => {
  it('ingest ve semantik replay idempotenttir; observationCount/MDL şişmez', () => {
    const { head, members } = trainedHead()
    const macro = head.macros.find((value) => value.kind === 'sequence-macro')!
    const replay = macroTrace({
      id: 'replayed-under-new-cycle',
      members,
      group: 'book-one',
      sourceDigest: 'SOURCE-DIGEST-book-one',
      semanticInputDigest: 'SEMANTIC-one',
      episodeId: 'another-episode',
      runKey: 'another-cycle',
    })
    const next = ingestMacroTraces(head, [replay])
    expect(next.head).toBe(head)
    expect(next.mining.diagnostics.duplicateTraceCount).toBe(1)
    expect(next.head.macros.find((value) => value.id === macro.id)?.observationCount)
      .toBe(macro.observationCount)
    expect(next.head.macros.find((value) => value.id === macro.id)?.mdl).toEqual(macro.mdl)
  })

  it('öneri -> sanal -> shadow doğrulanmış -> aktif sırasını zorunlu tutar', () => {
    const { head } = trainedHead()
    const candidate = head.macros.find((value) => value.kind === 'sequence-macro')!
    expect(candidate.lifecycle).toBe('eligible-proposal')
    expect(candidate.physicalIdentityAllocated).toBe(false)
    expect(candidate.rawConstituentsRetained).toBe(true)
    expect(() => transitionCompositeLifecycle(head, candidate.id, 'active')).toThrow(/Geçersiz/u)

    const provisional = transitionCompositeLifecycle(head, candidate.id, 'virtual-provisional')
    const report = evaluateMacroShadow(
      provisional.macros.find((value) => value.id === candidate.id)!,
      passingShadowCases(),
      provisional.config,
    )
    expect(report.passed).toBe(true)
    const validated = applyMacroShadowEvaluation(provisional, report)
    expect(validated.macros.find((value) => value.id === candidate.id)?.lifecycle)
      .toBe('shadow-validated')
    expect(() => transitionCompositeLifecycle(validated, candidate.id, 'active'))
      .toThrow(/yapısal/u)
    const validatedMacro = validated.macros.find((value) => value.id === candidate.id)!
    const audited = applyMacroStructuralEquivalenceAudit(validated,
      createMacroStructuralEquivalenceAudit({
        composite: validatedMacro,
        evaluator: 'unit-structural-evaluator-v1',
        evidenceDigest: 'STRUCTURAL-EVIDENCE-1',
        caseCount: 3,
        passed: true,
      }))
    const active = transitionCompositeLifecycle(audited, candidate.id, 'active')
    expect(active.macros.find((value) => value.id === candidate.id)?.lifecycle).toBe('active')
    validateMacroHead(active)

    const changedTrace = macroTrace({
      id: 'post-audit-definition-change',
      members: [
        macroMember('alice', ['actor']),
        macroMember('opens', ['action']),
        macroMember('door', ['object']),
      ],
      group: 'book-definition-change',
      signalOffset: 0.7,
    })
    const changed = ingestMacroTraces(active, [changedTrace]).head
    const changedMacro = changed.macros.find((value) => value.id === candidate.id)!
    expect(changedMacro.lifecycle).toBe('virtual-provisional')
    expect(changedMacro.structuralEquivalencePassed).toBe(false)
    expect(changedMacro.structuralEquivalenceAudit).toBeNull()
  })

  it('başarısız shadow raporu provisional kaydı aktif etmez', () => {
    const { head } = trainedHead()
    const candidate = head.macros.find((value) => value.kind === 'sequence-macro')!
    const provisional = transitionCompositeLifecycle(head, candidate.id, 'virtual-provisional')
    const report = evaluateMacroShadow(
      provisional.macros.find((value) => value.id === candidate.id)!,
      passingShadowCases().map((value) => ({
        ...value,
        macroOutput: [1 - value.expandedOutput[0], 1 - value.expandedOutput[1]],
        macroActivated: true,
      })),
      provisional.config,
    )
    expect(report.passed).toBe(false)
    const retained = applyMacroShadowEvaluation(provisional, report)
    expect(retained.macros.find((value) => value.id === candidate.id)?.lifecycle)
      .toBe('virtual-provisional')
  })

  it('ayrık shadow raporlarını kümülatif ve ağırlıklı toplar; rapor replayini tekrar saymaz', () => {
    const { head } = trainedHead()
    const candidate = head.macros.find((value) => value.kind === 'sequence-macro')!
    const provisional = transitionCompositeLifecycle(head, candidate.id, 'virtual-provisional')
    const macro = provisional.macros.find((value) => value.id === candidate.id)!
    const cases = passingShadowCases()
    const firstReport = evaluateMacroShadow(macro, cases.slice(0, 1), provisional.config)
    const afterFirst = applyMacroShadowEvaluation(provisional, firstReport)
    expect(afterFirst.macros.find((value) => value.id === candidate.id)?.lifecycle)
      .toBe('virtual-provisional')
    const secondReport = evaluateMacroShadow(macro, cases.slice(1), provisional.config)
    const afterSecond = applyMacroShadowEvaluation(afterFirst, secondReport)
    const accumulated = afterSecond.macros.find((value) => value.id === candidate.id)!
    expect(accumulated.lifecycle).toBe('shadow-validated')
    expect(accumulated.metrics).toMatchObject({
      shadowCaseCount: 3,
      holdoutUtilityCaseCount: 3,
      positiveCaseCount: 2,
      missedActivationCount: 0,
      negativeCaseCount: 1,
      falseActivationCount: 0,
      activationRecall: 1,
      falseActivationRate: 0,
      fidelity: 1,
      shadowCompressionGain: 6,
    })
    expect(accumulated.shadowReportIds).toHaveLength(2)
    expect(accumulated.shadowCaseIds).toHaveLength(3)
    expect(applyMacroShadowEvaluation(afterSecond, secondReport)).toBe(afterSecond)
  })

  it('aynı representation içindeki FamilyRun rebase exact makroyu korur, aileye bağlı kavramı yeniden doğrulamadan taşımaz', () => {
    const head = createMacroHead({
      representationSpaceId: 'space-64-v1', familyRunFingerprint: 'family-run-1',
    }, { minimumMotifLength: 3, maximumMotifLength: 3 })
    const variants = ['one', 'two', 'three'].map((prefix, index) => macroTrace({
      id: `trace-${prefix}`,
      group: `book-${index}`,
      members: [
        macroMember(`${prefix}-actor`, ['actor']),
        macroMember(`${prefix}-action`, ['action']),
        macroMember(`${prefix}-object`, ['object']),
      ],
    }))
    const trained = ingestMacroTraces(head, variants).head
    const sequence = trained.macros.find((value) => value.kind === 'sequence-macro')!
    const concept = trained.macros.find((value) => value.kind === 'equivalence-concept')!
    const rawMembers = new Map<string, { id: string; identityDigest: string; familyIds: string[] }>()
    for (const macro of trained.macros) {
      for (const variant of macro.variants) {
        for (const member of variant.orderedExpansion) {
          if (member.kind === 'raw-neuron') rawMembers.set(member.id, {
            id: member.id,
            identityDigest: member.identityDigest!,
            familyIds: member.familyIds.map((family) => `${family}-new`),
          })
        }
      }
    }
    const rebased = rebaseMacroHeadReference(trained, {
      representationSpaceId: trained.representationSpaceId,
      familyRunFingerprint: 'family-run-2',
      members: [...rawMembers.values()],
    })
    const nextSequence = rebased.head.macros.find((value) => value.id === sequence.id)!
    const nextConcept = rebased.head.macros.find((value) => value.id === concept.id)!
    expect(nextSequence.lifecycle).toBe(sequence.lifecycle)
    expect(nextSequence.expansionDigest).toBe(sequence.expansionDigest)
    expect(nextSequence.familySignature).toBe(computeFamilySignature(nextSequence.orderedExpansion))
    expect(nextConcept.lifecycle).toBe('deprecated')
    expect(rebased.familyChangedMacroIds).toContain(concept.id)
    expect(() => rebaseMacroHeadReference(trained, {
      representationSpaceId: 'another-space',
      familyRunFingerprint: 'family-run-2',
      members: [...rawMembers.values()],
    })).toThrow(/Temsil uzayı/u)
  })
})
