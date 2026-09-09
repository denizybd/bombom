import { describe, expect, it } from 'vitest'
import { createLanguageExperienceConfig } from './config'
import {
  createLanguageCheckpoint,
  createLanguageExperienceState,
  restoreLanguageCheckpoint,
  validateLanguageExperienceState,
} from './state'
import { createFixtureFamilyRun } from './testFixtures'

describe('dil deneyimi snapshot ve rollback', () => {
  it('JSON ile safça taşınabilen doğrulanmış kontrol noktası oluşturur', () => {
    const state = createLanguageExperienceState(createFixtureFamilyRun(), createLanguageExperienceConfig())
    const checkpoint = createLanguageCheckpoint(state)
    const serialized = JSON.stringify(checkpoint)
    const restored = restoreLanguageCheckpoint(JSON.parse(serialized))

    expect(restored).toEqual(state)
    expect(restored).not.toBe(state)
    validateLanguageExperienceState(restored)
  })

  it('oynanmış fingerprint veya geçiş skorunu reddeder', () => {
    const state = createLanguageExperienceState(createFixtureFamilyRun(), createLanguageExperienceConfig())
    expect(() => validateLanguageExperienceState({ ...state, revision: 4 })).toThrow(/durumu|parmak izi/u)
  })
})
