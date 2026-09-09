import { describe, expect, it } from 'vitest'
import { macroReplayBloomMayContain } from './replay.ts'
import { computeMacroSemanticReplayKey, computeMacroTraceReplayKey } from './miner.ts'
import { createMacroHead, ingestMacroTraces } from './state.ts'
import { macroMember, macroTrace } from './test-fixtures.ts'

describe('bounded replay ledger', () => {
  it('exact listeyi sabit tutar, taşan bütün anahtarları Bloom özetinde korur', () => {
    const initial = createMacroHead({
      representationSpaceId: 'space-64-v1',
      familyRunFingerprint: 'family-run-1',
    }, {
      minimumMotifLength: 2,
      maximumMotifLength: 2,
      maximumExactReplayKeys: 2,
      replayBloomBitCount: 256,
      replayBloomHashCount: 3,
    })
    const members = [macroMember('a'), macroMember('b')]
    const traces = Array.from({ length: 6 }, (_, index) => macroTrace({
      id: `ledger-${index}`,
      members,
      group: `book-${index}`,
    }))
    const first = ingestMacroTraces(initial, traces)
    expect(first.head.processedTraceKeys).toHaveLength(2)
    expect(first.head.processedSemanticTraceKeys).toHaveLength(2)
    const overflowTraceKeys = first.mining.processedTraceKeys.filter((key) =>
      !first.head.processedTraceKeys.includes(key))
    const overflowSemanticKeys = first.mining.processedSemanticTraceKeys.filter((key) =>
      !first.head.processedSemanticTraceKeys.includes(key))
    expect(overflowTraceKeys).toHaveLength(4)
    expect(overflowSemanticKeys).toHaveLength(4)
    expect(overflowTraceKeys.every((key) =>
      macroReplayBloomMayContain(first.head.traceReplayBloom, key))).toBe(true)
    expect(overflowSemanticKeys.every((key) =>
      macroReplayBloomMayContain(first.head.semanticReplayBloom, key))).toBe(true)

    const replayed = ingestMacroTraces(first.head, [traces.find((trace) =>
      overflowTraceKeys.includes(computeMacroTraceReplayKey(trace)))!])
    expect(replayed.head).toBe(first.head)
    expect(replayed.mining.diagnostics.duplicateTraceCount).toBe(1)

    const semanticOriginal = traces.find((trace) =>
      overflowSemanticKeys.includes(computeMacroSemanticReplayKey(trace)))!
    const renamedSemanticReplay = {
      ...structuredClone(semanticOriginal),
      id: 'renamed-semantic-replay',
      runKey: 'brand-new-run',
      episodeId: 'brand-new-episode',
      independenceGroupId: 'relabeled-group',
    }
    const semanticReplay = ingestMacroTraces(first.head, [renamedSemanticReplay])
    expect(semanticReplay.head).toBe(first.head)
    expect(semanticReplay.mining.diagnostics.duplicateTraceCount).toBe(1)
  })

  it('aynı iz kümesinde sıra değişse de Bloom/head fingerprinti değişmez', () => {
    const makeHead = () => createMacroHead({
      representationSpaceId: 'space-64-v1', familyRunFingerprint: 'family-run-1',
    }, {
      minimumMotifLength: 2,
      maximumMotifLength: 2,
      maximumExactReplayKeys: 2,
      replayBloomBitCount: 256,
      replayBloomHashCount: 3,
    })
    const members = [macroMember('a'), macroMember('b')]
    const traces = Array.from({ length: 5 }, (_, index) => macroTrace({
      id: `deterministic-${index}`, members, group: `group-${index}`,
    }))
    expect(ingestMacroTraces(makeHead(), traces).head)
      .toEqual(ingestMacroTraces(makeHead(), [...traces].reverse()).head)
  })
})
