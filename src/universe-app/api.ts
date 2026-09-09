import type { LiveTrainingState, NeuronAtlas } from './types'
import type { StructuralStopGateStatus } from '../training-app/types'
import { createBusyLanguageAtlas } from './busy-atlas'

const API_BASE = 'http://127.0.0.1:8787'

async function json<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: 'GET', signal, headers: { Accept: 'application/json' } })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const message = typeof body?.message === 'string' ? body.message : `Yerel servis ${response.status} yanıtı verdi.`
    throw new Error(message)
  }
  return body as T
}

export async function loadNeuronAtlas(signal?: AbortSignal): Promise<NeuronAtlas> {
  const atlas = await json<NeuronAtlas>('/api/neuron-atlas', signal)
  if (atlas.format !== 'bombom-neuron-atlas' || atlas.version !== 1) {
    throw new Error('Nöron evreni veri biçimi desteklenmiyor.')
  }
  if (!atlas.available && atlas.busy) return createBusyLanguageAtlas(atlas)
  return atlas
}

interface PublicStatus {
  state?: string
  activeStage?: string | null
  progress?: number
  updatedAt?: string | null
}

interface PublicCampaign {
  status?: string
  accepted?: number
  generated?: number
  updatedAt?: string | null
  continuous?: {
    completedBlocks?: number
    stopGate?: { status?: StructuralStopGateStatus; passed?: boolean }
  }
}

export async function loadLiveTrainingState(signal?: AbortSignal): Promise<LiveTrainingState> {
  const [status, campaigns] = await Promise.all([
    json<PublicStatus>('/api/status', signal),
    json<PublicCampaign[]>('/api/teacher/feed/campaigns', signal),
  ])
  const campaign = campaigns[0] ?? null
  return {
    serviceState: status.state ?? 'unknown',
    activeStage: status.activeStage ?? null,
    progress: Number.isFinite(status.progress) ? Math.max(0, Math.min(1, Number(status.progress))) : 0,
    campaignStatus: campaign?.status ?? null,
    campaignAccepted: Number(campaign?.accepted ?? 0),
    campaignGenerated: Number(campaign?.generated ?? 0),
    completedBlocks: Number(campaign?.continuous?.completedBlocks ?? 0),
    gateStatus: campaign?.continuous?.stopGate?.status ?? null,
    gatePassed: campaign?.continuous?.stopGate?.passed === true,
    updatedAt: campaign?.updatedAt ?? status.updatedAt ?? null,
  }
}
