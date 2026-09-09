import type { StructuralStopGateStatus } from './types'

export interface StructuralGateDisplayInput {
  status: StructuralStopGateStatus | null | undefined
  passed?: boolean
  failedChecks?: readonly string[]
}

const SCREENED_OUT_LABEL = 'ön ölçüm eşiği altında; tam benchmark çalıştırılmadı'
const DECISIVE_EARLY_FAILURE_LABEL = 'kesin erken başarısız; kalan tanısal varyantlar atlandı'

export function isStructuralGateEarlyStop(status: StructuralStopGateStatus | null | undefined): boolean {
  return status === 'screened-out' || status === 'failed-early' || status === 'decisive-early-failure'
}

export function isStructuralGatePassed(gate: StructuralGateDisplayInput): boolean {
  // Status is authoritative here. This prevents an inconsistent legacy payload
  // from presenting a deliberately stopped benchmark as a successful gate.
  return gate.status === 'passed' && gate.passed === true
}

export function structuralGateStatusLabel(gate: StructuralGateDisplayInput): string {
  if (gate.status === 'screened-out') return SCREENED_OUT_LABEL
  if (gate.status === 'failed-early' || gate.status === 'decisive-early-failure') {
    return DECISIVE_EARLY_FAILURE_LABEL
  }
  if (isStructuralGatePassed(gate)) return 'Geçti'
  if (gate.status === 'waiting') return 'İlk blok bekleniyor'

  const missing = gate.failedChecks?.length ?? 0
  if (gate.status === 'missing') return `Benchmark kanıtı eksik · ${missing} eksik koşul`
  if (gate.status === 'failed' || gate.status === 'passed') return `Geçmedi · ${missing} eksik koşul`
  if (!gate.status) return 'Bekleniyor'
  return `Geçmedi · ${gate.status}`
}
