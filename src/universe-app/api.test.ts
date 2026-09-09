import { afterEach, describe, expect, it, vi } from 'vitest'
import { loadNeuronAtlas } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('loadNeuronAtlas', () => {
  it('turns a live busy response into the safe 3B language projection', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
      format: 'bombom-neuron-atlas',
      version: 1,
      available: false,
      busy: true,
      busyStage: 'teacher-feed:pipeline:families',
      reason: 'Aile eğitimi sürüyor.',
      selection: { selectedBy: 'latest-snapshot', snapshot: null, artifact: null },
      run: null,
      stats: { familyCount: 0, neuronCount: 0, membershipCount: 0, familyTransitionCount: 0, neuronTransitionCount: 0 },
      families: [],
      neurons: [],
      links: { memberships: [], familyTransitions: [], neuronTransitions: [] },
    }), { status: 200, headers: { 'content-type': 'application/json' } })))

    const atlas = await loadNeuronAtlas()
    expect(atlas).toMatchObject({ available: true, partial: true, busy: true })
    expect(atlas.families).toHaveLength(6)
    expect(atlas.neurons).toHaveLength(21)
  })
})
