import { describe, expect, it } from 'vitest'
import { createLanguageExperienceConfig } from './config'

describe('dil deneyimi yapılandırması', () => {
  it('oran, sıcaklık, seyreklik ve hiyerarşi sınırlarını doğrular', () => {
    expect(() => createLanguageExperienceConfig({ trainRatio: 0.7 })).toThrow(/toplamları 1/u)
    expect(() => createLanguageExperienceConfig({ routeTemperature: 0 })).toThrow(/pozitif/u)
    expect(() => createLanguageExperienceConfig({ maximumOutgoingNeuronTransitions: 0 })).toThrow(/pozitif tam sayı/u)
    expect(() => createLanguageExperienceConfig({ includeHierarchyLevels: ['token', 'token'] })).toThrow(/tekrarlı/u)
  })

  it('dizi alanlarını varsayılan yapılandırmadan bağımsız kopyalar', () => {
    const first = createLanguageExperienceConfig()
    const second = createLanguageExperienceConfig()
    first.metricTopK.push(9)

    expect(second.metricTopK).not.toContain(9)
  })
})
