import { createLanguageOutputScaffold } from '../language-output/scaffold.ts'
import type {
  AtlasFamily,
  AtlasNeuron,
  FamilyTransitionTuple,
  MembershipTuple,
  NeuronAtlas,
} from './types.ts'

/**
 * Produces a tiny, fully functional 3B projection without opening the dense
 * FamilyRun artifact. It is used only while a memory-heavy worker owns RAM.
 */
export function createBusyLanguageAtlas(base: NeuronAtlas): NeuronAtlas {
  const scaffold = createLanguageOutputScaffold()
  const familyIndex = new Map<string, number>(scaffold.families.map((family, index) => [family.id, index]))
  const incoming = new Map<number, Set<number>>()
  const families: AtlasFamily[] = scaffold.families
    .slice()
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id))
    .map((definition, index) => ({
      id: definition.id,
      index,
      origin: 'manual-language-seed',
      layoutZone: 'language-surface',
      layoutPinned: true,
      layoutSlot: Math.max(0, definition.order - 1),
      parentFamilyId: null,
      label: definition.label,
      region: 'language',
      regionVerified: false,
      lifecycle: 'manual-seed',
      supportCount: 0,
      sourceCount: 0,
      memberCount: scaffold.seedNeurons.filter((neuron) => neuron.familyId === definition.id).length,
      coreMemberCount: scaffold.seedNeurons.filter((neuron) => neuron.familyId === definition.id).length,
      cohesion: 0,
      separation: 0,
      stability: 1,
      utility: null,
      utilityStatus: 'manual-scaffold',
      predecessors: [],
    }))

  // The sort above follows the stable manual order, but rebuild the index map
  // from the materialized result so no source array ordering assumption leaks.
  familyIndex.clear()
  for (const family of families) familyIndex.set(family.id, family.index)

  const neurons: AtlasNeuron[] = []
  const memberships: MembershipTuple[] = []
  for (const seed of scaffold.seedNeurons) {
    const family = familyIndex.get(seed.familyId)
    if (family === undefined) continue
    const index = neurons.length
    neurons.push({
      id: seed.id,
      index,
      text: seed.surface ?? seed.normalizedSurface ?? seed.id,
      kind: 'language-output',
      origin: 'manual-language-seed',
      status: 'frozen',
      sourceCount: 0,
      evidenceCount: 0,
      membershipCount: 1,
      primaryFamily: family,
      diagnostics: Array(10).fill(0),
      affinity: { region: 'language', score: seed.priorWeight, verified: false },
    })
    memberships.push([index, family, seed.priorWeight, 1, 1, 1])
  }

  const rows = new Map<number, Array<{ to: number; prior: number }>>()
  for (const transition of scaffold.manualTransitions) {
    const from = familyIndex.get(transition.fromKey)
    const to = familyIndex.get(transition.toKey)
    if (from === undefined || to === undefined) continue
    const row = rows.get(from) ?? []
    row.push({ to, prior: Math.max(0, transition.priorEvidence) })
    rows.set(from, row)
    const predecessors = incoming.get(to) ?? new Set<number>()
    predecessors.add(from)
    incoming.set(to, predecessors)
  }
  for (const family of families) family.predecessors = [...(incoming.get(family.index) ?? [])].sort((a, b) => a - b)

  const familyTransitions: FamilyTransitionTuple[] = []
  for (const [from, row] of rows) {
    const total = row.reduce((sum, edge) => sum + edge.prior, 0) || 1
    for (const edge of row) familyTransitions.push([from, edge.to, edge.prior / total, edge.prior, 0, 0, 0, 0])
  }

  const reason = base.reason || 'Yoğun eğitim sürerken güvenli dil çekirdeği görünümü kullanılıyor.'
  return {
    ...base,
    available: true,
    partial: true,
    busy: true,
    reason,
    run: null,
    stats: {
      familyCount: families.length,
      neuronCount: neurons.length,
      membershipCount: memberships.length,
      familyTransitionCount: familyTransitions.length,
      neuronTransitionCount: 0,
      distinctSourceCount: 0,
      learnedFamilyCount: 0,
      languageOutputFamilyCount: families.length,
      languageOutputSeedNeuronCount: neurons.length,
      languageOutputNurseryCount: 0,
      languageOutputProvisionalNeuronCount: 0,
      priorEdgeCount: familyTransitions.length,
      experienceEdgeCount: 0,
      inhibitionEdgeCount: 0,
    },
    projection: {
      semanticAxes: false,
      layout: 'safe-partial-peripheral-language-surface-v1',
      note: 'Yoğun eğitim sırasında FamilyRun açılmadan yalnız manuel dil çıkış adası gösterilir.',
      languageOutput: {
        scaffoldFingerprint: scaffold.fingerprint,
        headFingerprint: null,
        headRevision: 0,
        persisted: false,
        compatible: false,
        mismatch: false,
        familyCount: families.length,
        seedNeuronCount: neurons.length,
        nurseryCount: 0,
        provisionalNeuronCount: 0,
        membershipEvidenceCount: 0,
      },
    },
    families,
    neurons,
    links: { memberships, familyTransitions, neuronTransitions: [] },
    warnings: [reason, ...(base.warnings ?? [])].slice(0, 32),
  }
}
