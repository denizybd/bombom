import { describe, expect, it } from 'vitest'
import { buildFamilyLineData, buildMacroExpansionLineData, buildNeuronLineData, familyColor, neuronColor } from './scene-data'
import type { NeuronAtlas } from './types'

const atlas = {
  format: 'bombom-neuron-atlas', version: 1, available: true,
  selection: { selectedBy: 'test', snapshot: null, artifact: null }, run: null,
  stats: { familyCount: 2, neuronCount: 3, membershipCount: 3, familyTransitionCount: 1, neuronTransitionCount: 2 },
  families: [
    { id: 'a', index: 0, region: 'operator', regionVerified: false, lifecycle: 'stable', supportCount: 2, sourceCount: 1, memberCount: 2, coreMemberCount: 1, cohesion: .8, separation: .4, stability: .9, utility: .7, utilityStatus: 'validated', predecessors: [] },
    { id: 'b', index: 1, region: 'operator', regionVerified: false, lifecycle: 'provisional', supportCount: 1, sourceCount: 1, memberCount: 1, coreMemberCount: 1, cohesion: .6, separation: .5, stability: .5, utility: null, utilityStatus: 'missing', predecessors: [] },
  ],
  neurons: [0, 0, 1].map((primaryFamily, index) => ({ id: `n${index}`, index, text: '', kind: 'raw' as const, sourceCount: 1, evidenceCount: 1, membershipCount: 1, primaryFamily, diagnostics: [], affinity: null })),
  links: { memberships: [[0, 0, 1, 1, 0, 1]], familyTransitions: [[0, 1, .4, .2, .1, 0, 1, 1]], neuronTransitions: [[0, 1, .5], [1, 2, .25]] },
} satisfies NeuronAtlas

describe('universe scene data', () => {
  it('builds compact line buffers and filters the selected family', () => {
    const result = buildFamilyLineData(atlas, new Float32Array([0, 0, 0, 4, 2, 1]), 0)
    expect(result.count).toBe(1)
    expect([...result.positions]).toEqual([0, 0, 0, 4, 2, 1])
    expect(result.colors).toHaveLength(6)
  })

  it('limits neuron paths to the selected neighborhood', () => {
    const result = buildNeuronLineData(atlas, new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]), { kind: 'neuron', index: 0 })
    expect(result.count).toBe(1)
  })

  it('keeps family colors deterministic', () => {
    expect(familyColor(atlas, 0, 'family')).toEqual(familyColor(atlas, 0, 'family'))
    expect(familyColor(atlas, 0, 'lifecycle')).not.toEqual(familyColor(atlas, 1, 'lifecycle'))
  })

  it('draws cross-zone transitions without requiring them in the layout physics', () => {
    const languageAtlas = {
      ...atlas,
      families: [
        ...atlas.families,
        {
          ...atlas.families[0], id: 'language-surface', index: 2, region: 'language',
          origin: 'manual-language-seed', layoutZone: 'language-surface', layoutPinned: true, layoutSlot: 0,
        },
      ],
      links: { ...atlas.links, familyTransitions: [[0, 2, .7, 0, .7, 0, 1, 1]] },
    } satisfies NeuronAtlas
    const result = buildFamilyLineData(
      languageAtlas,
      new Float32Array([0, 0, 0, 4, 2, 1, 30, 10, -5]),
      null,
    )
    expect(result.count).toBe(1)
    expect([...result.positions]).toEqual([0, 0, 0, 30, 10, -5])
  })

  it('draws bounded macro expansion links and gives concept nodes their own lifecycle color', () => {
    const macroAtlas = structuredClone(atlas) as NeuronAtlas
    macroAtlas.neurons.push({
      id: 'macro-1', index: 3, text: 'Sıra makrosu', kind: 'concept-macro', origin: 'concept-macro-overlay',
      status: 'shadow-validated', layoutZone: 'concept-surface', sourceCount: 2, evidenceCount: 5,
      membershipCount: 0, primaryFamily: -1, diagnostics: [], affinity: null,
      conceptMacro: {
        macroKind: 'sequence-macro', lifecycle: 'shadow-validated', runtimeStatus: 'research-snapshot',
        runtimeActive: false, expansionLength: 2, depth: 1, fidelity: .96, compressionGain: 4,
        compressionRatio: .5, sourceGroupCount: 2, observationCount: 5,
      },
    })
    macroAtlas.links.macroExpansions = [[3, 0, 0, 0], [3, 1, 1, 0]]
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0, 10, 4, 2])
    const all = buildMacroExpansionLineData(macroAtlas, positions, null, 1)
    expect(all.count).toBe(1)
    const selected = buildMacroExpansionLineData(macroAtlas, positions, { kind: 'neuron', index: 1 })
    expect(selected.count).toBe(1)
    expect([...selected.positions]).toEqual([10, 4, 2, 1, 0, 0])
    expect(neuronColor(macroAtlas, 3, 'lifecycle')).toEqual([0.22, 0.88, 1])
  })
})
