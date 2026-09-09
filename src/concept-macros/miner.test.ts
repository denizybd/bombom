import { describe, expect, it } from 'vitest'
import { createMacroConfig } from './config.ts'
import { computeExactSignature } from './deterministic.ts'
import { effectiveMacroHeavyHitterCapacity, mineMacroMotifs } from './miner.ts'
import { unpackMacroVector } from './packed.ts'
import { macroMember, macroTrace } from './test-fixtures.ts'

describe('bounded kavram/makro motif madencisi', () => {
  it('aynı imzanın üst üste binen pencerelerini saymaz, ayrık tekrarlarını sayar', () => {
    const a = macroMember('a')
    const config = createMacroConfig({ minimumMotifLength: 2, maximumMotifLength: 2 })
    const overlapping = mineMacroMotifs([
      macroTrace({ id: 'overlap', members: [a, a, a] }),
    ], config)
    const aa = overlapping.candidates.find((candidate) =>
      candidate.exactSignature === computeExactSignature([a, a]))
    expect(aa?.observationCount).toBe(1)
    expect(overlapping.diagnostics.overlappingObservationCount).toBe(1)

    const separated = mineMacroMotifs([
      macroTrace({ id: 'separated', members: [a, a, macroMember('x'), a, a] }),
    ], config)
    expect(separated.candidates.find((candidate) =>
      candidate.exactSignature === computeExactSignature([a, a]))?.observationCount).toBe(2)
  })

  it('aynı episode/run replayini ve farklı cycle kimlikli aynı semantik girdiyi tek kez işler', () => {
    const members = [macroMember('a'), macroMember('b'), macroMember('c')]
    const first = macroTrace({
      id: 'original', members, group: 'book-1', sourceDigest: 'BOOK-DIGEST',
      semanticInputDigest: 'BOOK-DIGEST:paragraph-7:0-18', episodeId: 'episode-1', runKey: 'cycle-1',
    })
    const exactReplay = structuredClone(first)
    const renamedReplay = macroTrace({
      id: 'renamed', members, group: 'relabeled-book-group', sourceDigest: 'BOOK-DIGEST',
      semanticInputDigest: 'BOOK-DIGEST:paragraph-7:0-18', episodeId: 'episode-99', runKey: 'cycle-99',
    })
    const result = mineMacroMotifs([renamedReplay, exactReplay, first], createMacroConfig({
      minimumMotifLength: 3,
      maximumMotifLength: 3,
    }))
    expect(result.diagnostics.processedTraceCount).toBe(1)
    expect(result.diagnostics.duplicateTraceCount).toBe(2)
    expect(result.candidates.find((candidate) => candidate.kind === 'sequence-macro')?.observationCount).toBe(1)
  })

  it('MDL ve bağımsız kaynak kapısından sonra sıra makrosunu öneriye yükseltir', () => {
    const members = [macroMember('a'), macroMember('b'), macroMember('c')]
    const traces = ['one', 'two', 'three'].map((id) => macroTrace({ id, members, group: `book-${id}` }))
    const result = mineMacroMotifs(traces, createMacroConfig({
      minimumMotifLength: 3,
      maximumMotifLength: 3,
    }))
    const exact = result.candidates.find((candidate) => candidate.kind === 'sequence-macro')!
    expect(exact.lifecycle).toBe('eligible-proposal')
    expect(exact.observationCount).toBe(3)
    expect(exact.independentGroupCount).toBe(3)
    expect(exact.mdl).toEqual({
      rawDescriptionCost: 9,
      macroDescriptionCost: 7,
      definitionCost: 4,
      compressionGain: 2,
      compressionRatio: expect.closeTo(2 / 9, 6),
    })
  })

  it('son üyeyi uygulanmamış terminal ankrajı olarak bırakır', () => {
    const members = [macroMember('a'), macroMember('b'), macroMember('c')]
    const result = mineMacroMotifs([
      macroTrace({ id: 'terminal-anchor', members }),
    ], createMacroConfig({
      minimumMotifLength: 3,
      maximumMotifLength: 3,
    }))
    const exact = result.candidates.find((candidate) => candidate.kind === 'sequence-macro')!
    expect(exact.executionSemantics).toBe('terminal-anchor-v1')
    expect(exact.entryPrototype.vector).toEqual([0, 0])
    expect(exact.exitPrototype.vector[0]).toBeCloseTo(0.1, 7)
    expect(exact.exitPrototype.vector[1]).toBe(0)
    expect(exact.effectPrototype[0]).toBeCloseTo(0.1, 7)
    expect(exact.effectPrototype[1]).toBe(0)
  })

  it('aynı aile sırasını farklı exact yüzeylerde görmeden eşdeğerlik kavramı ilan etmez', () => {
    const families = [['actor'], ['action'], ['object']]
    const makeMembers = (prefix: string) => families.map((family, index) =>
      macroMember(`${prefix}-${index}`, family))
    const sameSurface = ['one', 'two', 'three'].map((id) => macroTrace({
      id,
      group: `book-${id}`,
      members: makeMembers('same'),
    }))
    const config = createMacroConfig({ minimumMotifLength: 3, maximumMotifLength: 3 })
    const observing = mineMacroMotifs(sameSurface, config).candidates.find((candidate) =>
      candidate.kind === 'equivalence-concept')!
    expect(observing.variants).toHaveLength(1)
    expect(observing.lifecycle).toBe('observing')

    const varied = ['one', 'two', 'three'].map((id) => macroTrace({
      id: `varied-${id}`,
      group: `book-${id}`,
      members: makeMembers(id),
    }))
    const eligible = mineMacroMotifs(varied, config).candidates.find((candidate) =>
      candidate.kind === 'equivalence-concept')!
    expect(eligible.variants).toHaveLength(3)
    expect(eligible.lifecycle).toBe('eligible-proposal')
  })

  it('Space-Saving tablosunu sınırlar ve sık motifi gürültü içinde tutar', () => {
    const frequent = [macroMember('a'), macroMember('b')]
    const traces = [
      ...Array.from({ length: 6 }, (_, index) => macroTrace({
        id: `a-frequent-${index}`,
        members: frequent,
        group: `g-${index}`,
      })),
      macroTrace({ id: 'z-rare-1', members: [macroMember('x'), macroMember('y')] }),
      macroTrace({ id: 'z-rare-2', members: [macroMember('p'), macroMember('q')] }),
    ]
    const result = mineMacroMotifs(traces, createMacroConfig({
      minimumMotifLength: 2,
      maximumMotifLength: 2,
      maximumHeavyHitters: 2,
    }))
    expect(result.candidates).toHaveLength(2)
    expect(result.candidates.some((candidate) =>
      candidate.exactSignature === computeExactSignature(frequent))).toBe(true)
    expect(result.diagnostics.heavyHitterReplacementCount).toBeGreaterThan(0)
  })

  it('64x64 prototiplerde dense ağır-vurucu RAM bütçesini otomatik sınırlar', () => {
    const config = createMacroConfig()
    expect(effectiveMacroHeavyHitterCapacity(config, 2)).toBe(config.maximumHeavyHitters)
    expect(effectiveMacroHeavyHitterCapacity(config, 4_096)).toBe(51)
  })

  it('grup içindeki bütün gözlemleri ortalar, sonra kaynak gruplarına eşit ağırlık verir', () => {
    const members = [macroMember('a'), macroMember('b')]
    const withEntry = (id: string, group: string, entry: number) => {
      const trace = macroTrace({
        id,
        members,
        group,
        sourceDigest: `SOURCE-${group}`,
        semanticInputDigest: `OCCURRENCE-${id}`,
      })
      trace.steps[0].signalBefore = [entry]
      trace.steps[0].signalAfter = [entry]
      trace.steps[1].signalBefore = [entry]
      trace.steps[1].signalAfter = [entry]
      return trace
    }
    const result = mineMacroMotifs([
      withEntry('g1-positive', 'group-1', 1),
      withEntry('g1-negative', 'group-1', -1),
      withEntry('g2-positive', 'group-2', 1),
    ], createMacroConfig({ minimumMotifLength: 2, maximumMotifLength: 2 }))
    const candidate = result.candidates.find((value) => value.kind === 'sequence-macro')!
    const group1 = candidate.prototypeGroups.find((group) =>
      group.independenceGroupId === 'group-1')!
    expect(group1.observationCount).toBe(2)
    expect(Array.from(unpackMacroVector(group1.entryMean))[0]).toBe(0)
    expect(candidate.prototypeGroupCount).toBe(2)
    expect(candidate.prototypeObservationCount).toBe(3)
    expect(candidate.entryPrototype.vector[0]).toBeCloseTo(0.5, 7)
  })

  it('iz giriş sırası değişse bile bit düzeyinde aynı batch üretir', () => {
    const members = [macroMember('a'), macroMember('b'), macroMember('c')]
    const traces = ['c', 'a', 'b'].map((id) => macroTrace({ id, members, group: id }))
    const config = createMacroConfig({ minimumMotifLength: 2, maximumMotifLength: 3 })
    expect(mineMacroMotifs(traces, config)).toEqual(mineMacroMotifs([...traces].reverse(), config))
  })
})
