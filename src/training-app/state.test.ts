import { describe, expect, it } from 'vitest'
import { canStartRun, clampRatio, formatBytes, formatDuration, metricTrend, PIPELINE_ORDER } from './state'

describe('training UI state helpers', () => {
  it('clamps progress defensively', () => {
    expect(clampRatio(-1)).toBe(0)
    expect(clampRatio(0.42)).toBe(0.42)
    expect(clampRatio(4)).toBe(1)
    expect(clampRatio(Number.NaN)).toBe(0)
  })

  it('requires an idle engine and source, independently from the optional teacher', () => {
    expect(canStartRun('idle', 2)).toBe(true)
    expect(canStartRun('running', 2)).toBe(false)
    expect(canStartRun('idle', 0)).toBe(false)
  })

  it('calculates metric trends without inventing missing values', () => {
    expect(metricTrend({ current: 0.8, previous: 0.7, unit: 'ratio' })).toBe('up')
    expect(metricTrend({ current: 2, previous: 3, unit: 'count' })).toBe('down')
    expect(metricTrend({ current: null, previous: 3, unit: 'count' })).toBe('unknown')
  })

  it('formats resource values in Turkish-friendly units', () => {
    expect(formatBytes(1024)).toBe('1 KB')
    expect(formatDuration(90)).toBe('1 sa 30 dk')
  })

  it('shows the plastic-region companion stage before concept compression', () => {
    expect(PIPELINE_ORDER.indexOf('plastic_regions')).toBeGreaterThan(PIPELINE_ORDER.indexOf('language_experience'))
    expect(PIPELINE_ORDER.indexOf('plastic_regions')).toBeLessThan(PIPELINE_ORDER.indexOf('concept_macros'))
  })
})
