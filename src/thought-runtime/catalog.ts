import { rawMemberKey } from '../families/profile'
import type { FamilyRun } from '../families/types'
import type { TextLabAnalysis } from '../text-lab/domain/types'
import { DEFAULT_THOUGHT_RUNTIME_CONFIG } from './config'
import { createThoughtLayout, createThoughtModel, createThoughtNeuron } from './model'
import type {
  SparseHabitRow,
  ThoughtCatalogDiagnostic,
  ThoughtCatalogResult,
  ThoughtFamilyMembership,
  ThoughtModel,
  ThoughtNeuron,
  ThoughtResolvedNeuronInput,
  ThoughtRuntimeConfig,
} from './types'

function membershipIndex(familyRun: FamilyRun): Map<string, ThoughtFamilyMembership[]> {
  const result = new Map<string, ThoughtFamilyMembership[]>()
  for (const membership of familyRun.memberships) {
    const values = result.get(membership.memberKey) ?? []
    values.push({ familyId: membership.familyId, weight: membership.weight })
    result.set(membership.memberKey, values)
  }
  return result
}

/**
 * Adapts restored 64x64 TextLab artifacts plus FamilyRun M/H0 into the pure
 * runtime catalog. Profiles without a physical I/R vector are explicitly
 * excluded; no behavior-profile vector is mistaken for a neuron matrix.
 */
export function createThoughtCatalog(
  analyses: readonly TextLabAnalysis[],
  familyRun: FamilyRun,
  config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG,
): ThoughtCatalogResult {
  const diagnostics: ThoughtCatalogDiagnostic[] = []
  const memberships = membershipIndex(familyRun)
  const rawProfileKeys = new Set(familyRun.profiles.filter((profile) => profile.kind === 'raw').map((profile) => profile.memberKey))
  for (const profile of familyRun.profiles) {
    if (profile.kind === 'synthesis') diagnostics.push({
      memberKey: profile.memberKey,
      kind: 'excluded',
      reason: 'Sentez profili davranış özeti taşıyor fakat restore edilmiş fiziksel I/R matrisi yok.',
    })
  }
  const compatible = analyses.filter((analysis) => {
    if (analysis.layout.side === 64 && analysis.layout.cellCount === 4096) return true
    diagnostics.push({ memberKey: analysis.runId, kind: 'excluded', reason: `TextLab analizi ${analysis.layout.side}x${analysis.layout.side}; saf motor 64x64 ister.` })
    return false
  })
  if (!compatible.length) throw new RangeError('Katalog için en az bir restore edilmiş 64x64 TextLab analizi gerekir.')
  const identityLength = compatible[0].layout.identityLength
  if (compatible.some((analysis) => analysis.layout.identityLength !== identityLength)) {
    throw new RangeError('Aynı episode içindeki TextLab I/R bölüm sınırları eşit olmalıdır.')
  }
  const layout = createThoughtLayout(identityLength)
  const neurons: ThoughtNeuron[] = []
  const seen = new Set<string>()
  for (const analysis of compatible) {
    for (const node of analysis.nodes) {
      const memberKey = rawMemberKey(analysis.sourceDigest, analysis.runId, node.id)
      if (!rawProfileKeys.has(memberKey)) continue
      if (seen.has(memberKey)) throw new TypeError(`Yinelenen fiziksel raw memberKey: ${memberKey}`)
      seen.add(memberKey)
      neurons.push(createThoughtNeuron({
        id: memberKey,
        identity: node.identity,
        relation: node.relation,
        sourceIdentityDigest: node.identityDigest,
        familyMemberships: memberships.get(memberKey),
        provenance: {
          kind: 'textlab-surface',
          sourceId: analysis.source.id || analysis.sourceDigest,
          surface: node.text,
        },
      }, layout, config))
      diagnostics.push({ memberKey, kind: 'included', reason: 'Restore edilmiş TextLab I ve R bölümleri FamilyRun M/H0 anahtarıyla eşleşti.' })
    }
  }
  for (const key of rawProfileKeys) {
    if (!seen.has(key)) diagnostics.push({ memberKey: key, kind: 'excluded', reason: 'FamilyRun raw profili için restore edilmiş 64x64 text-lab-analysis node bulunamadı.' })
  }
  if (!neurons.length) throw new RangeError('FamilyRun ile eşleşen vektörlü raw nöron bulunamadı.')
  const validIds = new Set(neurons.map((neuron) => neuron.id))
  const rows = new Map<string, SparseHabitRow>()
  for (const transition of familyRun.neuronTransitions) {
    if (!validIds.has(transition.fromMemberKey) || !validIds.has(transition.toMemberKey)) continue
    const row = rows.get(transition.fromMemberKey) ?? { fromNeuronId: transition.fromMemberKey, edges: [] }
    row.edges.push({
      toNeuronId: transition.toMemberKey,
      priorEvidence: Math.max(0, transition.probability),
      excitationEvidence: 0,
      inhibitionEvidence: 0,
      trustMass: 0,
      updateCount: 0,
      independentSourceIds: [],
    })
    rows.set(transition.fromMemberKey, row)
  }
  const model = createThoughtModel({
    layout,
    neurons,
    families: familyRun.families.map((family) => ({ id: family.id, region: family.region })),
    habitRows: [...rows.values()],
  }, config)
  return { model, diagnostics }
}

/** Adapts the physical neuron rows already restored and integrity-checked by the server. */
export function createThoughtCatalogFromResolved(
  resolved: readonly ThoughtResolvedNeuronInput[],
  familyRun: FamilyRun,
  config: ThoughtRuntimeConfig = DEFAULT_THOUGHT_RUNTIME_CONFIG,
): ThoughtCatalogResult {
  if (!resolved.length) throw new RangeError('Çözümlenmiş fiziksel nöron kataloğu boş olamaz.')
  const identityLength = resolved[0].identity.length
  const layout = createThoughtLayout(identityLength)
  const memberships = membershipIndex(familyRun)
  const neurons: ThoughtNeuron[] = []
  const diagnostics: ThoughtCatalogDiagnostic[] = []
  const seen = new Set<string>()
  for (const input of resolved) {
    const memberKey = String(input.memberKey ?? '').trim()
    if (!memberKey || seen.has(memberKey)) throw new TypeError(`Çözümlenmiş nöron memberKey boş/yinelenen: ${memberKey || '(boş)'}`)
    if (!(input.identity instanceof Float32Array) || !(input.relation instanceof Float32Array) ||
        input.identity.length !== layout.identityLength || input.relation.length !== layout.relationLength) {
      throw new RangeError(`${memberKey} I/R bölümü 64x64 ortak yerleşimle uyuşmuyor.`)
    }
    seen.add(memberKey)
    neurons.push(createThoughtNeuron({
      id: memberKey,
      identity: input.identity,
      relation: input.relation,
      ...(input.identityDigest ? { sourceIdentityDigest: input.identityDigest } : {}),
      familyMemberships: memberships.get(memberKey),
      provenance: {
        kind: input.kind === 'synthesis' ? 'synthesis-surface' : 'textlab-surface',
        sourceId: input.sourceDigest ?? input.analysisRunId ?? input.candidateFingerprint ?? memberKey,
        surface: input.text,
      },
    }, layout, config))
    diagnostics.push({
      memberKey,
      kind: 'included',
      reason: `Sunucunun çözdüğü ${input.kind} fiziksel I/R nöronu FamilyRun üyeliğiyle birleştirildi.`,
    })
  }
  const validIds = new Set(neurons.map((neuron) => neuron.id))
  const rows = new Map<string, SparseHabitRow>()
  for (const transition of familyRun.neuronTransitions) {
    if (!validIds.has(transition.fromMemberKey) || !validIds.has(transition.toMemberKey)) continue
    const row = rows.get(transition.fromMemberKey) ?? { fromNeuronId: transition.fromMemberKey, edges: [] }
    row.edges.push({
      toNeuronId: transition.toMemberKey,
      priorEvidence: Math.max(0, transition.probability),
      excitationEvidence: 0,
      inhibitionEvidence: 0,
      trustMass: 0,
      updateCount: 0,
      independentSourceIds: [],
    })
    rows.set(transition.fromMemberKey, row)
  }
  return {
    model: createThoughtModel({
      layout,
      neurons,
      families: familyRun.families.map((family) => ({ id: family.id, region: family.region })),
      habitRows: [...rows.values()],
    }, config),
    diagnostics,
  }
}

export function isThoughtModel(value: unknown): value is ThoughtModel {
  return Boolean(value && typeof value === 'object' && (value as ThoughtModel).format === 'bombom-thought-model')
}

export function isThoughtCatalog(value: unknown): value is ThoughtCatalogResult {
  return Boolean(value && typeof value === 'object' && isThoughtModel((value as ThoughtCatalogResult).model))
}

export function isTextLabAnalysisArray(value: unknown): value is TextLabAnalysis[] {
  return Array.isArray(value) && value.every((entry) => entry && typeof entry === 'object' && entry.format === 'metin-laboratuvari-analysis')
}

export function isResolvedThoughtNeuronArray(value: unknown): value is ThoughtResolvedNeuronInput[] {
  return Array.isArray(value) && value.length > 0 && value.every((entry) => Boolean(
    entry && typeof entry === 'object' && typeof entry.memberKey === 'string' &&
    (entry.kind === 'raw' || entry.kind === 'synthesis') &&
    entry.identity instanceof Float32Array && entry.relation instanceof Float32Array,
  ))
}
