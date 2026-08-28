import { describe, expect, it } from 'vitest'
import { DEFAULT_TEXT_LAB_CONFIG, normalizeTextLabConfig } from './config'

describe('legacy TextLab config normalization', () => {
  it('fills missing fields and preserves a valid supplied context weight', () => {
    const normalized = normalizeTextLabConfig({
      matrixSide: 64,
      minimumLeafGraphemes: 5,
      contextRetention: 0.7,
    })

    expect(normalized.matrixSide).toBe(64)
    expect(normalized.minimumLeafGraphemes).toBe(5)
    expect(normalized.identityRatio).toBe(DEFAULT_TEXT_LAB_CONFIG.identityRatio)
    expect(normalized.contextRetention).toBe(0.7)
    expect(normalized.parentTraceWeight).toBeCloseTo(0.225)
    expect(normalized.siblingTraceWeight).toBeCloseTo(0.075)
    expect(
      normalized.contextRetention + normalized.parentTraceWeight + normalized.siblingTraceWeight,
    ).toBeCloseTo(1)
  })

  it('rejects invalid supplied legacy values instead of silently replacing them', () => {
    expect(() => normalizeTextLabConfig({ matrixSide: '64' })).toThrow('matrixSide')
    expect(() => normalizeTextLabConfig({ contextRetention: 1.2 })).toThrow('toplamı')
    expect(() => normalizeTextLabConfig({ minimumLeafGraphemes: 0 })).toThrow('yaprak')
  })
})
