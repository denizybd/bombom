import { describe, expect, it } from 'vitest'
import { createMacroConfig } from './config.ts'
import { expandCompositeNeuron, validateCompositeDag } from './dag.ts'
import { computeExpansionDigest } from './deterministic.ts'
import {
  computeCompositeNeuronFingerprint,
  createMacroHead,
  ingestMacroTraces,
} from './state.ts'
import { macroMember, macroTrace } from './test-fixtures.ts'
import type { CompositeNeuronV1, MacroMemberReferenceV1 } from './types.ts'

function ingestRepeated(
  head: ReturnType<typeof createMacroHead>,
  id: string,
  members: readonly MacroMemberReferenceV1[],
) {
  return ingestMacroTraces(head, ['one', 'two', 'three'].map((suffix) => macroTrace({
    id: `${id}-${suffix}`,
    members,
    group: `${id}-book-${suffix}`,
  }))).head
}

describe('nested makro DAG koruması', () => {
  it('en fazla üç katmanı sırasını bozmadan ham nöronlara geri açar', () => {
    let head = createMacroHead({
      representationSpaceId: 'space-64-v1', familyRunFingerprint: 'family-run-1',
    }, createMacroConfig({ minimumMotifLength: 2, maximumMotifLength: 2 }))
    const a = macroMember('a')
    const b = macroMember('b')
    const c = macroMember('c')
    const d = macroMember('d')
    head = ingestRepeated(head, 'level-1', [a, b])
    const level1 = head.macros.find((macro) => macro.orderedExpansion[0]?.id === 'a')!

    head = ingestRepeated(head, 'level-2', [
      { ...macroMember(level1.id, [], 'composite-neuron'), identityDigest: level1.expansionDigest },
      c,
    ])
    const level2 = head.macros.find((macro) => macro.depth === 2)!
    head = ingestRepeated(head, 'level-3', [
      { ...macroMember(level2.id, [], 'composite-neuron'), identityDigest: level2.expansionDigest },
      d,
    ])
    const level3 = head.macros.find((macro) => macro.depth === 3)!
    expect(expandCompositeNeuron(level3.id, head.macros).map((member) => member.id))
      .toEqual(['a', 'b', 'c', 'd'])
    expect(head.macros.every((macro) => macro.rawConstituentsRetained)).toBe(true)

    const before = head.macros.length
    head = ingestRepeated(head, 'level-4', [
      { ...macroMember(level3.id, [], 'composite-neuron'), identityDigest: level3.expansionDigest },
      macroMember('e'),
    ])
    expect(head.macros).toHaveLength(before)
    expect(head.macros.every((macro) => macro.depth <= 3)).toBe(true)
  })

  it('döngüsel composite başvurularını doğrulama aşamasında reddeder', () => {
    const head = ingestRepeated(createMacroHead({
      representationSpaceId: 'space-64-v1', familyRunFingerprint: 'family-run-1',
    }, { minimumMotifLength: 2, maximumMotifLength: 2 }), 'base', [
      macroMember('a'), macroMember('b'),
    ])
    const template = head.macros.find((macro) => macro.kind === 'sequence-macro')!
    const build = (id: string, nestedId: string): CompositeNeuronV1 => {
      const orderedExpansion: MacroMemberReferenceV1[] = [
        { kind: 'composite-neuron', id: nestedId, identityDigest: `EXP-${nestedId}`, familyIds: [] },
        macroMember(`raw-${id}`),
      ]
      const base: Omit<CompositeNeuronV1, 'fingerprint'> = {
        ...template,
        id,
        orderedExpansion,
        expansionDigest: computeExpansionDigest(orderedExpansion),
        variants: [{
          exactSignature: `exact-${id}`,
          orderedExpansion,
          expansionDigest: computeExpansionDigest(orderedExpansion),
          observationCount: 3,
        }],
        depth: 2,
      }
      return { ...base, fingerprint: computeCompositeNeuronFingerprint(base) }
    }
    expect(() => validateCompositeDag([build('cycle-a', 'cycle-b'), build('cycle-b', 'cycle-a')]))
      .toThrow(/döngüsü/u)
  })
})
