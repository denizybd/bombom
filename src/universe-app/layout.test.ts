import { describe, expect, it } from 'vitest'
import { computeUniverseLayout } from './layout'
import type { NeuronAtlas } from './types'

function atlas(): NeuronAtlas {
  return {
    format: 'bombom-neuron-atlas', version: 1, available: true,
    selection: { selectedBy: 'test', snapshot: null, artifact: null }, run: null,
    stats: { familyCount: 2, neuronCount: 3, membershipCount: 4, familyTransitionCount: 2, neuronTransitionCount: 1 },
    families: [
      { id: 'a', index: 0, region: 'content', regionVerified: false, lifecycle: 'stable', supportCount: 2, sourceCount: 2, memberCount: 2, coreMemberCount: 1, cohesion: .8, separation: .4, stability: .9, utility: .1, utilityStatus: 'validated', predecessors: [] },
      { id: 'b', index: 1, region: 'event', regionVerified: false, lifecycle: 'provisional', supportCount: 2, sourceCount: 2, memberCount: 2, coreMemberCount: 1, cohesion: .7, separation: .3, stability: .2, utility: null, utilityStatus: 'missing', predecessors: [] },
    ],
    neurons: [
      { id: 'n0', index: 0, text: 'ali', kind: 'raw', sourceCount: 1, evidenceCount: 1, membershipCount: 1, primaryFamily: 0, diagnostics: Array(10).fill(.5), affinity: null },
      { id: 'n1', index: 1, text: 'kesti', kind: 'raw', sourceCount: 1, evidenceCount: 1, membershipCount: 2, primaryFamily: 1, diagnostics: Array(10).fill(.25), affinity: null },
      { id: 'n2', index: 2, text: 'bıçak', kind: 'synthesis', sourceCount: 2, evidenceCount: 2, membershipCount: 1, primaryFamily: 0, diagnostics: Array(10).fill(.75), affinity: null },
    ],
    links: {
      memberships: [[0, 0, .9, 1, 1, 1], [1, 1, .8, .7, 1, 1], [1, 0, .5, .3, 2, 0], [2, 0, .75, 1, 1, 1]],
      familyTransitions: [[0, 1, .8, .7, .1, 0, 2, 3], [1, 0, .2, .2, 0, 0, 1, 1]],
      neuronTransitions: [[0, 1, 1]],
    },
  }
}

function languageFamily(index: number, fields: Partial<NeuronAtlas['families'][number]> = {}): NeuronAtlas['families'][number] {
  return {
    id: `language-${index}`,
    index,
    origin: 'manual-language-seed',
    layoutZone: 'language-surface',
    layoutPinned: true,
    layoutSlot: index - 2,
    label: 'Dil yüzeyi',
    region: 'language',
    regionVerified: false,
    lifecycle: 'provisional',
    supportCount: 0,
    sourceCount: 0,
    memberCount: 1,
    coreMemberCount: 1,
    cohesion: 1,
    separation: 1,
    stability: 1,
    utility: null,
    utilityStatus: 'missing',
    predecessors: [],
    ...fields,
  }
}

function coordinate(values: Float32Array, index: number): [number, number, number] {
  const offset = index * 3
  return [values[offset] ?? 0, values[offset + 1] ?? 0, values[offset + 2] ?? 0]
}

function distance(left: [number, number, number], right: [number, number, number] = [0, 0, 0]) {
  return Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2])
}

describe('computeUniverseLayout', () => {
  it('is deterministic, finite and returns one coordinate triple per item', () => {
    const first = computeUniverseLayout(atlas())
    const second = computeUniverseLayout(atlas())
    expect([...first.familyPositions]).toEqual([...second.familyPositions])
    expect([...first.neuronPositions]).toEqual([...second.neuronPositions])
    expect(first.familyPositions).toHaveLength(6)
    expect(first.neuronPositions).toHaveLength(9)
    expect([...first.familyPositions, ...first.neuronPositions].every(Number.isFinite)).toBe(true)
    expect(first.radius).toBeGreaterThan(1)
  })

  it('handles an empty atlas without inventing nodes', () => {
    const empty = atlas()
    empty.families = []
    empty.neurons = []
    empty.links = { memberships: [], familyTransitions: [], neuronTransitions: [] }
    const result = computeUniverseLayout(empty)
    expect(result.familyPositions).toHaveLength(0)
    expect(result.neuronPositions).toHaveLength(0)
    expect(result.radius).toBe(15)
  })

  it('keeps the complete core projection bit-identical when a language surface is appended', () => {
    const core = atlas()
    const baseline = computeUniverseLayout(core)
    const extended = atlas()
    extended.families.push(languageFamily(2, { layoutSlot: 0 }))
    extended.stats.familyCount += 1
    extended.links.familyTransitions.push([0, 2, .95, 0, .95, 0, 1, 1])
    extended.links.memberships.push([0, 2, .99, .99, 1, 0])
    extended.stats.familyTransitionCount += 1
    extended.stats.membershipCount += 1

    const projected = computeUniverseLayout(extended)
    expect([...projected.familyPositions.slice(0, baseline.familyPositions.length)])
      .toEqual([...baseline.familyPositions])
    expect([...projected.neuronPositions.slice(0, baseline.neuronPositions.length)])
      .toEqual([...baseline.neuronPositions])

    const coreRadius = Math.max(distance(coordinate(baseline.familyPositions, 0)), distance(coordinate(baseline.familyPositions, 1)))
    expect(distance(coordinate(projected.familyPositions, 2))).toBeGreaterThan(coreRadius + 5)
  })

  it('pins manual slots while language members and autonomous satellites grow around them', () => {
    const seeded = atlas()
    seeded.families.push(languageFamily(2, { id: 'language-seed', layoutSlot: 3 }))
    seeded.stats.familyCount += 1
    const first = computeUniverseLayout(seeded)

    const grown = atlas()
    grown.families.push(
      languageFamily(2, { id: 'language-seed', layoutSlot: 3, memberCount: 2 }),
      languageFamily(3, {
        id: 'language-child',
        origin: 'autonomous-language',
        layoutPinned: false,
        layoutSlot: undefined,
        parentFamilyId: 'language-seed',
      }),
    )
    grown.neurons.push({
      id: 'language-neuron', index: 3, text: 'EOS', kind: 'raw', sourceCount: 0,
      evidenceCount: 0, membershipCount: 1, primaryFamily: 2,
      diagnostics: Array(10).fill(.5), affinity: null,
    })
    grown.links.memberships.push([3, 2, 1, 1, 1, 1])
    grown.stats.familyCount += 2
    grown.stats.neuronCount += 1
    grown.stats.membershipCount += 1
    const second = computeUniverseLayout(grown)

    const seedBefore = coordinate(first.familyPositions, 2)
    const seedAfter = coordinate(second.familyPositions, 2)
    const child = coordinate(second.familyPositions, 3)
    expect(seedAfter).toEqual(seedBefore)
    expect(distance(child, seedAfter)).toBeGreaterThan(3)
    expect(distance(child, seedAfter)).toBeLessThan(8)
    expect(distance(coordinate(second.neuronPositions, 3), seedAfter)).toBeLessThan(5)
    expect([...second.familyPositions, ...second.neuronPositions].every(Number.isFinite)).toBe(true)
  })

  it('uses a bounded explicit atlas anchor deterministically outside the core', () => {
    const extended = atlas()
    extended.families.push(languageFamily(2, { atlasAnchor: [100, -100, -100] }))
    extended.stats.familyCount += 1
    const first = computeUniverseLayout(extended)
    const second = computeUniverseLayout(extended)
    expect(coordinate(first.familyPositions, 2)).toEqual(coordinate(second.familyPositions, 2))
    const coreRadius = Math.max(distance(coordinate(first.familyPositions, 0)), distance(coordinate(first.familyPositions, 1)))
    expect(distance(coordinate(first.familyPositions, 2))).toBeGreaterThan(coreRadius)
  })

  it('places concept macros on a deterministic peripheral surface without moving the core', () => {
    const baselineAtlas = atlas()
    const baseline = computeUniverseLayout(baselineAtlas)
    const extended = atlas()
    extended.projection = {
      semanticAxes: false,
      layout: 'test-concept-surface',
      note: 'test',
      conceptMacros: {
        artifact: null, headFingerprint: 'macro-head-1', headRevision: 1, compatible: true, mismatch: null,
        snapshotStatus: 'candidate', runtimeEligible: false, totalMacroCount: 2, nodeCount: 2,
        linkCount: 1, activeCount: 0, truncated: false, byLifecycle: [{ lifecycle: 'observing', count: 2 }],
      },
    }
    extended.neurons.push(
      {
        id: 'macro-a', index: 3, text: 'Sıra makrosu', kind: 'concept-macro', origin: 'concept-macro-overlay',
        status: 'observing', layoutZone: 'concept-surface', sourceCount: 2, evidenceCount: 5,
        membershipCount: 0, primaryFamily: -1, diagnostics: Array(10).fill(0), affinity: null,
        conceptMacro: {
          macroKind: 'sequence-macro', lifecycle: 'observing', runtimeStatus: 'research-snapshot',
          runtimeActive: false, expansionLength: 2, depth: 1, fidelity: .91, compressionGain: 3,
          compressionRatio: .5, sourceGroupCount: 2, observationCount: 5,
        },
      },
      {
        id: 'macro-b', index: 4, text: 'Eşdeğerlik kavramı', kind: 'concept-macro', origin: 'concept-macro-overlay',
        status: 'shadow-validated', layoutZone: 'concept-surface', sourceCount: 3, evidenceCount: 9,
        membershipCount: 0, primaryFamily: -1, diagnostics: Array(10).fill(0), affinity: null,
        conceptMacro: {
          macroKind: 'equivalence-concept', lifecycle: 'shadow-validated', runtimeStatus: 'research-snapshot',
          runtimeActive: false, expansionLength: 3, depth: 2, fidelity: .97, compressionGain: 7,
          compressionRatio: .4, sourceGroupCount: 3, observationCount: 9,
        },
      },
    )
    extended.links.macroExpansions = [[3, 0, 0, 0]]
    extended.stats.neuronCount += 2

    const first = computeUniverseLayout(extended)
    const second = computeUniverseLayout(extended)
    expect([...first.familyPositions]).toEqual([...baseline.familyPositions])
    expect([...first.neuronPositions.slice(0, baseline.neuronPositions.length)]).toEqual([...baseline.neuronPositions])
    expect([...first.neuronPositions]).toEqual([...second.neuronPositions])
    const coreRadius = Math.max(distance(coordinate(baseline.familyPositions, 0)), distance(coordinate(baseline.familyPositions, 1)))
    expect(distance(coordinate(first.neuronPositions, 3))).toBeGreaterThan(coreRadius + 8)
    expect(distance(coordinate(first.neuronPositions, 4))).toBeGreaterThan(coreRadius + 8)
    expect(first.radius).toBeGreaterThan(distance(coordinate(first.neuronPositions, 4)))
  })
})
