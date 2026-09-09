import { describe, expect, it } from 'vitest'
import { createMacroConfig } from './config.ts'

describe('kavram makro yapılandırma değişmezleri', () => {
  it('motifleri 2..8, nested DAGı en fazla 3 ile sınırlar', () => {
    expect(createMacroConfig()).toMatchObject({
      minimumMotifLength: 2,
      maximumMotifLength: 8,
      maximumNestingDepth: 3,
    })
    expect(() => createMacroConfig({ minimumMotifLength: 1 })).toThrow(/2 ile 8/u)
    expect(() => createMacroConfig({ maximumMotifLength: 9 })).toThrow(/2 ile 8/u)
    expect(() => createMacroConfig({ maximumNestingDepth: 4 })).toThrow(/en fazla 3/u)
    expect(() => createMacroConfig({ maximumPrototypeGroupsPerCandidate: 65 }))
      .toThrow(/Prototip kaynak grubu/u)
  })
})
