import type { MetricValue, PipelineStageId, RunState } from './types'

export const PIPELINE_ORDER: readonly PipelineStageId[] = [
  'collect',
  'quarantine',
  'validate',
  'encode',
  'synthesize',
  'families',
  'language_experience',
  'plastic_regions',
  'concept_macros',
  'evaluate',
  'commit',
  'rollback',
]

export function clampRatio(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

export function isRunBusy(state: RunState): boolean {
  return ['starting', 'running', 'pausing', 'paused', 'cancelling'].includes(state)
}

export function canStartRun(state: RunState, sourceCount: number): boolean {
  return (state === 'idle' || state === 'failed') && sourceCount > 0
}

export function metricTrend(metric: MetricValue): 'up' | 'down' | 'same' | 'unknown' {
  if (metric.current === null || metric.previous === null) return 'unknown'
  const delta = metric.current - metric.previous
  if (Math.abs(delta) < 1e-9) return 'same'
  return delta > 0 ? 'up' : 'down'
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${Math.round(bytes)} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toLocaleString('tr-TR', { maximumFractionDigits: 1 })} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toLocaleString('tr-TR', { maximumFractionDigits: 1 })} MB`
  return `${(bytes / 1024 ** 3).toLocaleString('tr-TR', { maximumFractionDigits: 1 })} GB`
}

export function formatDuration(minutes: number): string {
  if (!Number.isFinite(minutes) || minutes < 0) return '—'
  if (minutes < 60) return `${Math.round(minutes)} dk`
  const hours = Math.floor(minutes / 60)
  const rest = Math.round(minutes % 60)
  return rest ? `${hours} sa ${rest} dk` : `${hours} sa`
}
