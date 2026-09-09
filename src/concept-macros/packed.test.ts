import { describe, expect, it } from 'vitest'
import { packMacroVector, unpackMacroVector } from './packed.ts'

describe('JSON uyumlu packed Float32 prototipi', () => {
  it('canonical Float32 değerlerini kayıpsız açar ve number[] JSONundan küçük kalır', () => {
    const values = new Float32Array(4_096)
    for (let index = 0; index < values.length; index += 1) values[index] = Math.fround(index / values.length)
    const packed = packMacroVector(values)
    expect(Array.from(unpackMacroVector(packed))).toEqual(Array.from(values))
    expect(JSON.stringify(packed).length).toBeLessThan(JSON.stringify(Array.from(values)).length)
  })
})
