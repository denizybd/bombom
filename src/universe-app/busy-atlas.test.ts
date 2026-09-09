import { describe, expect, it } from 'vitest'
import { createBusyLanguageAtlas } from './busy-atlas'
import { computeUniverseLayout } from './layout'
import type { NeuronAtlas } from './types'

function busyAtlas(): NeuronAtlas {
  return {
    format: 'bombom-neuron-atlas',
    version: 1,
    available: false,
    busy: true,
    busyStage: 'families',
    reason: 'Yoğun eğitim sürüyor.',
    selection: { selectedBy: 'latest-snapshot', snapshot: null, artifact: null },
    run: null,
    stats: { familyCount: 0, neuronCount: 0, membershipCount: 0, familyTransitionCount: 0, neuronTransitionCount: 0 },
    families: [],
    neurons: [],
    links: { memberships: [], familyTransitions: [], neuronTransitions: [] },
  }
}

describe('createBusyLanguageAtlas', () => {
  it('keeps 3B usable without opening or inventing the dense learned core', () => {
    const source = busyAtlas()
    const result = createBusyLanguageAtlas(source)
    expect(source.available).toBe(false)
    expect(source.families).toEqual([])
    expect(result).toMatchObject({ available: true, partial: true, busy: true, busyStage: 'families' })
    expect(result.families).toHaveLength(6)
    expect(result.neurons).toHaveLength(21)
    expect(result.stats).toMatchObject({ learnedFamilyCount: 0, languageOutputFamilyCount: 6, languageOutputSeedNeuronCount: 21 })
    expect(result.families.every((family) => family.layoutZone === 'language-surface')).toBe(true)
    expect(result.neurons.every((neuron) => neuron.kind === 'language-output')).toBe(true)
    expect([...computeUniverseLayout(result).familyPositions].every(Number.isFinite)).toBe(true)
  })

  it('normalizes every visible manual transition row', () => {
    const result = createBusyLanguageAtlas(busyAtlas())
    const rowTotals = new Map<number, number>()
    for (const edge of result.links.familyTransitions) rowTotals.set(edge[0], (rowTotals.get(edge[0]) ?? 0) + edge[2])
    expect([...rowTotals.values()].every((total) => Math.abs(total - 1) < 1e-12)).toBe(true)
  })
})
