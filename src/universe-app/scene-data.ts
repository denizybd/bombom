import type {
  ColorMode,
  FamilyTransitionTuple,
  NeuronAtlas,
  NeuronTransitionTuple,
  UniverseSelection,
} from './types'

export interface SceneLineData {
  positions: Float32Array
  colors: Float32Array
  count: number
}

export function hash32(value: string) {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

function hslToRgb(hue: number, saturation: number, lightness: number): [number, number, number] {
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation
  const segment = (hue % 360) / 60
  const x = chroma * (1 - Math.abs((segment % 2) - 1))
  let rgb: [number, number, number]
  if (segment < 1) rgb = [chroma, x, 0]
  else if (segment < 2) rgb = [x, chroma, 0]
  else if (segment < 3) rgb = [0, chroma, x]
  else if (segment < 4) rgb = [0, x, chroma]
  else if (segment < 5) rgb = [x, 0, chroma]
  else rgb = [chroma, 0, x]
  const match = lightness - chroma / 2
  return [rgb[0] + match, rgb[1] + match, rgb[2] + match]
}

const lifecycleColors: Record<string, [number, number, number]> = {
  stable: [0.20, 0.92, 0.67],
  provisional: [0.72, 0.45, 1],
  'merge-proposed': [1, 0.68, 0.25],
  retired: [0.34, 0.38, 0.48],
}

export function familyColor(atlas: NeuronAtlas, familyIndex: number, mode: ColorMode): [number, number, number] {
  const family = atlas.families[familyIndex]
  if (!family) return [0.45, 0.5, 0.6]
  if (family.origin === 'manual-language-seed' && mode === 'family') return [0.18, 0.9, 1]
  if (family.origin === 'autonomous-language' && mode === 'family') return [0.55, 0.72, 1]
  if (mode === 'lifecycle') return lifecycleColors[family.lifecycle] ?? [0.38, 0.68, 1]
  if (mode === 'utility') {
    if (family.utility === null) return [0.38, 0.42, 0.53]
    const score = Math.max(0, Math.min(1, family.utility))
    return [0.94 - score * 0.64, 0.24 + score * 0.64, 0.42 + score * 0.26]
  }
  const key = mode === 'region' ? family.region : family.id
  const hue = hash32(key) % 360
  return hslToRgb(hue, mode === 'region' ? 0.62 : 0.72, mode === 'region' ? 0.58 : 0.61)
}

export function neuronColor(atlas: NeuronAtlas, neuronIndex: number, mode: ColorMode): [number, number, number] {
  const neuron = atlas.neurons[neuronIndex]
  if (neuron?.kind !== 'concept-macro' || !neuron.conceptMacro) return familyColor(atlas, neuron?.primaryFamily ?? -1, mode)
  if (neuron.conceptMacro.runtimeStatus === 'rejected-snapshot') return [0.42, 0.45, 0.54]
  if (mode === 'lifecycle') {
    return ({
      active: [0.20, 1, 0.66],
      'shadow-validated': [0.22, 0.88, 1],
      'virtual-provisional': [0.64, 0.46, 1],
      'eligible-proposal': [1, 0.68, 0.25],
      observing: [0.48, 0.54, 0.68],
      deprecated: [0.3, 0.32, 0.4],
    } as Record<string, [number, number, number]>)[neuron.conceptMacro.lifecycle] ?? [0.62, 0.55, 1]
  }
  if (mode === 'utility') {
    const fidelity = Math.max(0, Math.min(1, neuron.conceptMacro.fidelity))
    return [0.86 - fidelity * 0.48, 0.32 + fidelity * 0.58, 0.84]
  }
  if (mode === 'region') return [0.64, 0.47, 1]
  return neuron.conceptMacro.macroKind === 'equivalence-concept' ? [1, 0.45, 0.86] : [0.53, 0.52, 1]
}

function pushVertex(target: number[], positions: Float32Array, index: number) {
  const offset = index * 3
  target.push(positions[offset] ?? 0, positions[offset + 1] ?? 0, positions[offset + 2] ?? 0)
}

function transitionColor(edge: FamilyTransitionTuple): [number, number, number] {
  const probability = Math.max(0.16, Math.min(1, edge[2] * 5))
  if (edge[5] > 0) return [1 * probability, 0.24 * probability, 0.32 * probability]
  if (edge[4] > 0) return [0.18 * probability, 1 * probability, 0.82 * probability]
  return [0.68 * probability, 0.48 * probability, 1 * probability]
}

export function buildFamilyLineData(
  atlas: NeuronAtlas,
  familyPositions: Float32Array,
  selectedFamily: number | null,
): SceneLineData {
  const positions: number[] = []
  const colors: number[] = []
  const selectedEdges = selectedFamily === null
    ? atlas.links.familyTransitions
    : atlas.links.familyTransitions.filter((edge) => edge[0] === selectedFamily || edge[1] === selectedFamily)
  for (const edge of selectedEdges) {
    if (edge[0] === edge[1] || edge[0] < 0 || edge[1] < 0) continue
    pushVertex(positions, familyPositions, edge[0])
    pushVertex(positions, familyPositions, edge[1])
    const color = transitionColor(edge)
    colors.push(...color, ...color)
  }
  return { positions: Float32Array.from(positions), colors: Float32Array.from(colors), count: positions.length / 6 }
}

function selectedNeuronEdges(edges: readonly NeuronTransitionTuple[], selection: UniverseSelection, maximum = 4_000) {
  if (!selection) return []
  if (selection.kind === 'neuron') {
    return edges
      .filter((edge) => edge[0] === selection.index || edge[1] === selection.index)
      .sort((left, right) => right[2] - left[2])
      .slice(0, maximum)
  }
  return edges
}

export function buildNeuronLineData(
  atlas: NeuronAtlas,
  neuronPositions: Float32Array,
  selection: UniverseSelection,
  maximum = 4_000,
): SceneLineData {
  const primaryFamily = atlas.neurons.map((neuron) => neuron.primaryFamily)
  let edges = selectedNeuronEdges(atlas.links.neuronTransitions, selection, maximum)
  if (selection?.kind === 'family') {
    edges = atlas.links.neuronTransitions
      .filter((edge) => primaryFamily[edge[0]] === selection.index || primaryFamily[edge[1]] === selection.index)
      .sort((left, right) => right[2] - left[2])
      .slice(0, maximum)
  }
  const positions: number[] = []
  const colors: number[] = []
  for (const edge of edges) {
    if (edge[0] === edge[1] || edge[0] < 0 || edge[1] < 0) continue
    pushVertex(positions, neuronPositions, edge[0])
    pushVertex(positions, neuronPositions, edge[1])
    const strength = Math.max(0.24, Math.min(1, edge[2] * 5))
    colors.push(0.18 * strength, 0.72 * strength, strength, 0.18 * strength, 0.72 * strength, strength)
  }
  return { positions: Float32Array.from(positions), colors: Float32Array.from(colors), count: positions.length / 6 }
}

export function buildMacroExpansionLineData(
  atlas: NeuronAtlas,
  neuronPositions: Float32Array,
  selection: UniverseSelection,
  maximum = 4_000,
): SceneLineData {
  let edges = atlas.links.macroExpansions ?? []
  if (selection?.kind === 'family') edges = []
  else if (selection?.kind === 'neuron') {
    edges = edges.filter((edge) => edge[0] === selection.index || edge[1] === selection.index)
  }
  edges = edges.slice(0, maximum)
  const positions: number[] = []
  const colors: number[] = []
  for (const [macro, member, ordinal, nested] of edges) {
    if (macro === member || macro < 0 || member < 0 || macro >= atlas.neurons.length || member >= atlas.neurons.length) continue
    pushVertex(positions, neuronPositions, macro)
    pushVertex(positions, neuronPositions, member)
    const orderFade = Math.max(0.42, 1 - ordinal * 0.065)
    const color: [number, number, number] = nested ? [1, 0.36, 0.82] : [0.52, 0.48, 1]
    colors.push(color[0] * orderFade, color[1] * orderFade, color[2] * orderFade)
    colors.push(color[0] * orderFade, color[1] * orderFade, color[2] * orderFade)
  }
  return { positions: Float32Array.from(positions), colors: Float32Array.from(colors), count: positions.length / 6 }
}

export function familyForSelection(atlas: NeuronAtlas, selection: UniverseSelection) {
  if (!selection) return null
  if (selection.kind === 'family') return selection.index
  const family = atlas.neurons[selection.index]?.primaryFamily
  return Number.isInteger(family) && Number(family) >= 0 ? Number(family) : null
}
