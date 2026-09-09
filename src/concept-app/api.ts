import type { ConceptEnvelope } from './types'

const API_BASE = 'http://127.0.0.1:8787'

export async function loadConceptMacros(signal?: AbortSignal): Promise<ConceptEnvelope> {
  const response = await fetch(`${API_BASE}/api/concepts/macros?limit=2000`, {
    signal,
    headers: { Accept: 'application/json' },
  })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(typeof body?.message === 'string' ? body.message : `Kavram servisi ${response.status} yanıtı verdi.`)
  }
  return body as ConceptEnvelope
}
