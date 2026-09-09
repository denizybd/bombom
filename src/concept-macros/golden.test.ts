import { describe, expect, it } from 'vitest'
import { computeMacroConfigFingerprint, createMacroConfig } from './config.ts'
import { mineMacroMotifs } from './miner.ts'
import { evaluateMacroShadow } from './shadow.ts'
import {
  createMacroHead,
  ingestMacroTraces,
  transitionCompositeLifecycle,
} from './state.ts'
import { macroMember, macroTrace, passingShadowCases } from './test-fixtures.ts'

describe('kavram makro V1 altın fingerprint sözleşmesi', () => {
  it('aynı canonical kanıt için sabit digest zinciri üretir', () => {
    const config = createMacroConfig({ minimumMotifLength: 3, maximumMotifLength: 3 })
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
    const batch = mineMacroMotifs(traces, config)
    const empty = createMacroHead({
      representationSpaceId: 'space-64-v1',
      familyRunFingerprint: 'family-run-1',
    }, { minimumMotifLength: 3, maximumMotifLength: 3 })
    const trained = ingestMacroTraces(empty, traces).head
    const sequence = trained.macros.find((macro) => macro.kind === 'sequence-macro')!
    const provisional = transitionCompositeLifecycle(trained, sequence.id, 'virtual-provisional')
    const report = evaluateMacroShadow(
      provisional.macros.find((macro) => macro.id === sequence.id)!,
      passingShadowCases(),
      provisional.config,
    )

    expect(computeMacroConfigFingerprint(config))
      .toBe('MCFG-3697E9F16F6FD1C5C133F4CFC7E819B1')
    expect(sequence.expansionDigest)
      .toBe('MEXP-55D8A2C2B28645E608310AB4F767973A')
    expect(batch.fingerprint)
      .toBe('MMININGFP-367AF4F15C8EBA0106DE4AF32823A44D')
    expect(batch.candidates.find((candidate) => candidate.kind === 'sequence-macro')?.fingerprint)
      .toBe('MCANDFP-85A8D50B808C49D725E889D5C5C7D51B')
    expect(empty.fingerprint)
      .toBe('CMHEADFP-B5B24E4BD52C90CC368AB6E13DEBB25E')
    expect(trained.fingerprint)
      .toBe('CMHEADFP-31D8A1F9878C10221234CFCF569DC910')
    expect(report.fingerprint)
      .toBe('MSHADOWFP-673E0DE96D0153821617AB17D4F31240')
  })
})
