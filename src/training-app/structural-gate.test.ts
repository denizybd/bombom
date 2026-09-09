import { describe, expect, it } from 'vitest'
import {
  isStructuralGateEarlyStop,
  isStructuralGatePassed,
  structuralGateStatusLabel,
} from './structural-gate'

describe('structural gate UI labels', () => {
  it('labels a diagnostic screen-out without implying a completed benchmark', () => {
    const gate = { status: 'screened-out' as const, passed: false, failedChecks: [] }
    expect(structuralGateStatusLabel(gate)).toBe('ön ölçüm eşiği altında; tam benchmark çalıştırılmadı')
    expect(isStructuralGateEarlyStop(gate.status)).toBe(true)
    expect(isStructuralGatePassed(gate)).toBe(false)
  })

  it.each(['failed-early', 'decisive-early-failure'] as const)(
    'labels %s as a decisive early failure',
    (status) => {
      const gate = { status, passed: true, failedChecks: [] }
      expect(structuralGateStatusLabel(gate)).toBe('kesin erken başarısız; kalan tanısal varyantlar atlandı')
      expect(isStructuralGateEarlyStop(status)).toBe(true)
      expect(isStructuralGatePassed(gate)).toBe(false)
    },
  )

  it('keeps legacy states backwards compatible and requires status plus flag to pass', () => {
    expect(structuralGateStatusLabel({ status: 'waiting', passed: false })).toBe('İlk blok bekleniyor')
    expect(structuralGateStatusLabel({ status: 'failed', passed: false, failedChecks: ['f1'] })).toBe('Geçmedi · 1 eksik koşul')
    expect(isStructuralGatePassed({ status: 'passed', passed: true })).toBe(true)
    expect(isStructuralGatePassed({ status: 'failed', passed: true })).toBe(false)
  })
})
