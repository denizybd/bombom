import { describe, expect, it } from 'vitest'
import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  DEFAULT_TEXT_LAB_CONFIG,
  LEGACY_TEXT_LAB_PROFILE_ID,
  createMatrixLayout,
  createTextLabConfigForProfile,
  normalizeTextLabConfig,
  resolveTextLabRepresentationProfileId,
  validateConfig,
} from './config'

describe('legacy TextLab config normalization', () => {
  it('starts new TextLab v2 analyses in the thought runtime 64x64 space', () => {
    const layout = createMatrixLayout(DEFAULT_TEXT_LAB_CONFIG)

    expect(DEFAULT_TEXT_LAB_CONFIG.matrixSide).toBe(64)
    expect(layout.side).toBe(64)
    expect(layout.cellCount).toBe(4_096)
    expect(resolveTextLabRepresentationProfileId(DEFAULT_TEXT_LAB_CONFIG)).toBe(
      LEGACY_TEXT_LAB_PROFILE_ID,
    )
    expect(Object.hasOwn(DEFAULT_TEXT_LAB_CONFIG, 'representationProfile')).toBe(false)
  })

  it('creates the opt-in adaptive 8/56 layout from a named profile', () => {
    const config = createTextLabConfigForProfile(ADAPTIVE_TEXT_LAB_PROFILE_ID)
    const layout = createMatrixLayout(config)

    expect(config.representationProfile).toBe(ADAPTIVE_TEXT_LAB_PROFILE_ID)
    expect(layout).toEqual({
      side: 64,
      cellCount: 4_096,
      identityRows: 8,
      identityLength: 512,
      relationStart: 512,
      relationLength: 3_584,
    })

    const normalized = normalizeTextLabConfig({
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
    })
    expect(createMatrixLayout(normalized)).toEqual(layout)
  })

  it('fails closed when an explicit profile is paired with another physical layout', () => {
    const adaptive = createTextLabConfigForProfile(ADAPTIVE_TEXT_LAB_PROFILE_ID)
    expect(() => validateConfig({ ...adaptive, identityRatio: 0.25 })).toThrow(
      '8 kimlik satırı',
    )
    expect(() => validateConfig({ ...adaptive, matrixSide: 32 })).toThrow(
      '64×64',
    )
    expect(() => normalizeTextLabConfig({
      ...DEFAULT_TEXT_LAB_CONFIG,
      representationProfile: 'unknown-profile',
    })).toThrow('Bilinmeyen')
  })

  it('preserves an explicit legacy 32x32 space instead of migrating it', () => {
    const normalized = normalizeTextLabConfig({
      ...DEFAULT_TEXT_LAB_CONFIG,
      matrixSide: 32,
    })

    expect(normalized.matrixSide).toBe(32)
    expect(createMatrixLayout(normalized).cellCount).toBe(1_024)
    expect(Object.hasOwn(normalized, 'representationProfile')).toBe(false)
  })

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
