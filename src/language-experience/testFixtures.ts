import { createSparseVector, sparseVectorDigest } from '../families/sparse'
import type {
  BehaviorDiagnostics,
  BehaviorProfile,
  FamilyMembership,
  FamilyRun,
  FunctionalRegionId,
  LearnedFamily,
  SparseNeuronTransition,
} from '../families/types'
import { languageDigest, normalizeSurface } from './deterministic'
import type { LanguageCorpusSource } from './types'

export const fixtureSources: LanguageCorpusSource[] = [
  { sourceId: 's1', sourceDigest: 'd1', title: 'Bir', text: 'Ali bıçak ile ekmek kesti.' },
  { sourceId: 's2', sourceDigest: 'd2', title: 'İki', text: 'Ayşe makas ile kağıt kesti.' },
  { sourceId: 's3', sourceDigest: 'd3', title: 'Üç', text: 'Mert anahtar ile kapı açtı.' },
  { sourceId: 's4', sourceDigest: 'd4', title: 'Dört', text: 'Elif düğme ile lamba açtı.' },
  { sourceId: 's5', sourceDigest: 'd5', title: 'Beş', text: 'Deniz bıçak ile kumaş kesti.' },
  { sourceId: 's6', sourceDigest: 'd6', title: 'Altı', text: 'Can anahtar ile kutu açtı.' },
]

const diagnostics: BehaviorDiagnostics = {
  spanScale: 0.2,
  relativePosition: 0.5,
  depthScale: 0.5,
  relationStrength: 0.5,
  transitionEntropy: 0.5,
  transitionConcentration: 0.5,
  localContinuity: 0.5,
  bridgeImpact: 0.5,
  leftRightAsymmetry: 0.5,
  shortScopePotential: 0.8,
}

function classify(surface: string): FunctionalRegionId {
  if (/^(kesti|açtı)$/u.test(surface)) return 'event'
  if (/^(ile)$/u.test(surface)) return 'role'
  if (/^[.!?]$/u.test(surface)) return 'operator'
  return 'content'
}

function family(id: string, region: FunctionalRegionId, memberKeys: string[]): LearnedFamily {
  const prototype = createSparseVector(8, [{ index: Math.max(0, ['content', 'event', 'role', 'operator'].indexOf(region)), value: 1 }])
  return {
    id,
    fingerprint: languageDigest('fixture-family', id),
    behaviorSpaceId: 'BS-language-fixture',
    region,
    regionVerified: false,
    lifecycle: 'stable',
    lifecycleReason: 'test fixture',
    prototype,
    prototypeDigest: sparseVectorDigest(prototype),
    memberKeys,
    coreMemberKeys: memberKeys,
    independentEvidence: [],
    supportCount: memberKeys.length,
    sourceCount: fixtureSources.length,
    cohesion: 0.8,
    separation: 0.7,
    stability: 1,
    utility: 0.1,
    predecessorFamilyIds: [],
  }
}

function tokens(source: LanguageCorpusSource): Array<{ text: string; start: number; end: number }> {
  return [...source.text.matchAll(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu)].map((match) => ({
    text: match[0],
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length,
  }))
}

export function createFixtureFamilyRun(sources: readonly LanguageCorpusSource[] = fixtureSources): FamilyRun {
  const occurrences = new Map<string, Array<{ source: LanguageCorpusSource; start: number; end: number; text: string }>>()
  for (const source of sources) {
    for (const token of tokens(source)) {
      const normalized = normalizeSurface(token.text)
      const values = occurrences.get(normalized) ?? []
      values.push({ source, ...token })
      occurrences.set(normalized, values)
    }
  }
  const profiles: BehaviorProfile[] = [...occurrences].map(([surface, values], index) => {
    const memberKey = `member:${surface}`
    const vector = createSparseVector(8, [{ index: index % 8, value: 1 }])
    const region = classify(surface)
    return {
      memberKey,
      kind: 'synthesis' as const,
      representationSpaceId: 'RS-language-fixture',
      behaviorSpaceId: 'BS-language-fixture',
      vector,
      diagnostics,
      regionAffinities: [{ region, score: 1, verified: false, explanation: 'fixture' }],
      evidence: values.map((value) => ({
        memberKey,
        analysisRunId: `run-${value.source.sourceId}`,
        sourceId: value.source.sourceId,
        sourceDigest: value.source.sourceDigest,
        neuronId: `${memberKey}:${value.start}`,
        paragraphIndex: 0,
        start: value.start,
        end: value.end,
        text: value.text,
      })),
      identityDigest: languageDigest('fixture-identity', surface),
      text: values[0].text,
      derivedFromMemberKeys: [],
    }
  }).sort((left, right) => left.memberKey.localeCompare(right.memberKey))
  const familyIds: Record<'content' | 'event' | 'role' | 'operator', string> = {
    content: 'family:content',
    event: 'family:event',
    role: 'family:role',
    operator: 'family:operator',
  }
  const memberships: FamilyMembership[] = profiles.map((profile) => {
    const region = classify(normalizeSurface(profile.text)) as keyof typeof familyIds
    return {
      memberKey: profile.memberKey,
      familyId: familyIds[region],
      rank: 1,
      score: 1,
      weight: 1,
      core: true,
      breakdown: { behaviorSimilarity: 1, regionCompatibility: 1, evidenceConfidence: 1, combined: 1 },
    }
  })
  const families = (Object.entries(familyIds) as Array<[keyof typeof familyIds, string]>).map(([region, id]) =>
    family(id, region, memberships.filter((membership) => membership.familyId === id).map((membership) => membership.memberKey)),
  )
  const counts = new Map<string, number>()
  const totals = new Map<string, number>()
  for (const source of sources) {
    const sequence = tokens(source).map((token) => `member:${normalizeSurface(token.text)}`)
    for (let index = 1; index < sequence.length; index += 1) {
      const key = `${sequence[index - 1]}\u0000${sequence[index]}`
      counts.set(key, (counts.get(key) ?? 0) + 1)
      totals.set(sequence[index - 1], (totals.get(sequence[index - 1]) ?? 0) + 1)
    }
  }
  const neuronTransitions: SparseNeuronTransition[] = [...counts].map(([key, count]) => {
    const [fromMemberKey, toMemberKey] = key.split('\u0000')
    return { fromMemberKey, toMemberKey, probability: count / (totals.get(fromMemberKey) ?? 1) }
  })
  const familyTransitions: FamilyRun['familyTransitions'] = [
    ['family:content', 'family:content', 0.35],
    ['family:content', 'family:role', 0.35],
    ['family:content', 'family:event', 0.3],
    ['family:role', 'family:content', 1],
    ['family:event', 'family:operator', 1],
  ].map(([fromFamilyId, toFamilyId, probability], index) => ({
    id: `ft-${index}`,
    fromFamilyId: String(fromFamilyId),
    toFamilyId: String(toFamilyId),
    priorEvidence: Number(probability),
    experienceEvidence: 0,
    inhibitionEvidence: 0,
    sourceCount: sources.length,
    supportCount: sources.length,
    probability: Number(probability),
  }))
  const fingerprint = languageDigest('fixture-run', sources.map((source) => [source.sourceDigest, source.text]))
  return {
    format: 'noron-aileleri-run',
    version: 1,
    id: 'family-run-language-fixture',
    fingerprint,
    createdAt: '2026-01-01T00:00:00.000Z',
    representationSpaceId: 'RS-language-fixture',
    behaviorSpaceId: 'BS-language-fixture',
    config: {} as FamilyRun['config'],
    inputManifest: { entries: [], corpusFingerprint: languageDigest('fixture-corpus', sources.map((source) => source.sourceDigest)) },
    profiles,
    families,
    memberships,
    neuronTransitions,
    familyTransitions,
    utilityReports: [],
    coalition: { tick: 0, activeFamilyCount: 0, coalitions: [], unmatchedReason: null },
    routingPreview: null,
    warnings: [],
    stats: {} as FamilyRun['stats'],
  }
}
