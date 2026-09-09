import type {
  BehaviorProfile,
  FamilyAnalysisInput,
  FamilyConfig,
  FamilyMembership,
  FamilyTransition,
  LearnedFamily,
  RouteDistribution,
  RouteEntry,
  SparseNeuronTransition,
} from './types'
import { profileSimilarity, rawMemberKey } from './profile'
import { FAMILY_EPSILON, clamp, compareText, stableDigest } from './sparse'
import { createIndependentSourceIndex } from './independence'
import type { IndependentSourceIndex } from './independence'

interface TransitionEvidence {
  priorEvidence: number
  supportKeys: Set<string>
  sourceIds: Set<string>
}

export interface FamilyTransitionBuildResult {
  neuronTransitions: SparseNeuronTransition[]
  familyTransitions: FamilyTransition[]
  peakOutgoingFamilyTransitions: number
}

export interface RouteContext {
  /** Optional compatibility of the current signal with each candidate. */
  signalCompatibility?: ReadonlyMap<string, number>
  /** Recent-expression or refractory value in [0, 1]. */
  repetition?: ReadonlyMap<string, number>
  /** Optional explicit candidate restriction. */
  candidateMemberKeys?: ReadonlySet<string>
}

function membershipIndex(
  memberships: readonly FamilyMembership[],
): Map<string, FamilyMembership[]> {
  const index = new Map<string, FamilyMembership[]>()
  for (const membership of memberships) {
    const values = index.get(membership.memberKey) ?? []
    values.push(membership)
    index.set(membership.memberKey, values)
  }
  for (const values of index.values()) {
    values.sort((left, right) => left.rank - right.rank || compareText(left.familyId, right.familyId))
  }
  return index
}

function keepStrongest<T>(
  items: readonly T[],
  limit: number,
  score: (item: T) => number,
  key: (item: T) => string,
): T[] {
  return [...items]
    .sort((left, right) => score(right) - score(left) || compareText(key(left), key(right)))
    .slice(0, limit)
}

/**
 * Reduces TextLab's dense H⁰ rows to a deterministic, bounded prior graph.
 * The source arrays are never mutated and synthesis profiles do not receive an
 * extra transition vote: they remain derived views of their raw supports.
 */
export function buildSparseNeuronTransitions(
  analyses: readonly FamilyAnalysisInput[],
  availableMemberKeys: ReadonlySet<string>,
  config: FamilyConfig,
  shouldCancel?: () => boolean,
): SparseNeuronTransition[] {
  const transitions: SparseNeuronTransition[] = []
  const orderedAnalyses = [...analyses].sort((left, right) =>
    compareText(left.sourceDigest, right.sourceDigest) || compareText(left.runId, right.runId),
  )
  for (const analysis of orderedAnalyses) {
    const count = analysis.nodes.length
    if (analysis.habitProbabilities.length !== count * count) {
      throw new RangeError('Aile yönlendirmesi için H⁰ boyutu nöron sayısıyla uyuşmuyor.')
    }
    for (let row = 0; row < count; row += 1) {
      if (shouldCancel?.()) throw new Error('FAMILY_CANCELLED')
      const from = rawMemberKey(analysis.sourceDigest, analysis.runId, analysis.nodes[row].id)
      if (!availableMemberKeys.has(from)) continue
      const candidates: SparseNeuronTransition[] = []
      for (let column = 0; column < count; column += 1) {
        if (column === row) continue
        const probability = analysis.habitProbabilities[row * count + column] ?? 0
        if (!Number.isFinite(probability) || probability <= 0) continue
        const to = rawMemberKey(analysis.sourceDigest, analysis.runId, analysis.nodes[column].id)
        if (!availableMemberKeys.has(to)) continue
        candidates.push({ fromMemberKey: from, toMemberKey: to, probability })
      }
      const retained = keepStrongest(
        candidates,
        config.maximumTransitionNeighbors,
        (entry) => entry.probability,
        (entry) => entry.toMemberKey,
      )
      const total = retained.reduce((sum, entry) => sum + entry.probability, 0)
      if (total <= FAMILY_EPSILON) continue
      for (const entry of retained) {
        transitions.push({ ...entry, probability: entry.probability / total })
      }
    }
  }
  return transitions.sort((left, right) =>
    compareText(left.fromMemberKey, right.fromMemberKey) ||
    right.probability - left.probability ||
    compareText(left.toMemberKey, right.toMemberKey),
  )
}

function sourceForProfile(
  profile: BehaviorProfile | undefined,
  sourceIndex: IndependentSourceIndex,
): string {
  const evidence = profile?.evidence[0]
  return evidence ? sourceIndex.groupKey(evidence) : ''
}

/**
 * Projects sparse H⁰ evidence through every soft membership pair. Each source
 * first normalizes its own outgoing evidence, preventing a long document from
 * overwhelming several shorter independent sources.
 */
export function buildFamilyTransitions(
  neuronTransitions: readonly SparseNeuronTransition[],
  profiles: readonly BehaviorProfile[],
  families: readonly LearnedFamily[],
  memberships: readonly FamilyMembership[],
  config: FamilyConfig,
): FamilyTransitionBuildResult {
  const profileByKey = new Map(profiles.map((profile) => [profile.memberKey, profile]))
  const sourceIndex = createIndependentSourceIndex(profiles.flatMap((profile) => profile.evidence))
  const byMember = membershipIndex(memberships)
  const familyIds = new Set(families.map((family) => family.id))
  const bySource = new Map<string, Map<string, TransitionEvidence>>()

  for (const transition of neuronTransitions) {
    const sourceId = sourceForProfile(profileByKey.get(transition.fromMemberKey), sourceIndex)
    if (
      !sourceId ||
      sourceId !== sourceForProfile(profileByKey.get(transition.toMemberKey), sourceIndex)
    ) continue
    const fromMemberships = byMember.get(transition.fromMemberKey) ?? []
    const toMemberships = byMember.get(transition.toMemberKey) ?? []
    for (const from of fromMemberships) {
      if (!familyIds.has(from.familyId) || from.weight <= 0) continue
      for (const to of toMemberships) {
        if (!familyIds.has(to.familyId) || to.weight <= 0) continue
        const contribution = transition.probability * from.weight * to.weight
        if (contribution <= FAMILY_EPSILON) continue
        const pair = `${from.familyId}\u0000${to.familyId}`
        const sourceMap = bySource.get(sourceId) ?? new Map<string, TransitionEvidence>()
        const evidence = sourceMap.get(pair) ?? {
          priorEvidence: 0,
          supportKeys: new Set<string>(),
          sourceIds: new Set<string>(),
        }
        evidence.priorEvidence += contribution
        evidence.supportKeys.add(`${transition.fromMemberKey}\u0001${transition.toMemberKey}`)
        evidence.sourceIds.add(sourceId)
        sourceMap.set(pair, evidence)
        bySource.set(sourceId, sourceMap)
      }
    }
  }

  // Normalize every source by origin family before combining sources equally.
  for (const sourceMap of bySource.values()) {
    const totals = new Map<string, number>()
    for (const [pair, evidence] of sourceMap) {
      const from = pair.split('\u0000', 1)[0]
      totals.set(from, (totals.get(from) ?? 0) + evidence.priorEvidence)
    }
    for (const [pair, evidence] of sourceMap) {
      const from = pair.split('\u0000', 1)[0]
      const total = totals.get(from) ?? 0
      evidence.priorEvidence = total > FAMILY_EPSILON ? evidence.priorEvidence / total : 0
    }
  }

  const combined = new Map<string, TransitionEvidence>()
  for (const [sourceId, sourceMap] of bySource) {
    for (const [pair, sourceEvidence] of sourceMap) {
      const evidence = combined.get(pair) ?? {
        priorEvidence: 0,
        supportKeys: new Set<string>(),
        sourceIds: new Set<string>(),
      }
      evidence.priorEvidence += sourceEvidence.priorEvidence
      sourceEvidence.supportKeys.forEach((key) => evidence.supportKeys.add(key))
      evidence.sourceIds.add(sourceId)
      combined.set(pair, evidence)
    }
  }

  const candidateByOrigin = new Map<string, FamilyTransition[]>()
  for (const [pair, evidence] of combined) {
    const [fromFamilyId, toFamilyId] = pair.split('\u0000')
    if (!familyIds.has(fromFamilyId) || !familyIds.has(toFamilyId)) continue
    const fingerprint = stableDigest(['family-transition-v1', fromFamilyId, toFamilyId])
    const transition: FamilyTransition = {
      id: `FG-${fingerprint.slice(0, 16)}`,
      fromFamilyId,
      toFamilyId,
      priorEvidence: evidence.priorEvidence / Math.max(1, evidence.sourceIds.size),
      experienceEvidence: 0,
      inhibitionEvidence: 0,
      sourceCount: evidence.sourceIds.size,
      supportCount: evidence.supportKeys.size,
      probability: 0,
    }
    const outgoing = candidateByOrigin.get(fromFamilyId) ?? []
    outgoing.push(transition)
    candidateByOrigin.set(fromFamilyId, outgoing)
  }

  const familyTransitions: FamilyTransition[] = []
  let peakOutgoingFamilyTransitions = 0
  for (const [fromFamilyId, candidates] of candidateByOrigin) {
    const retained = keepStrongest(
      candidates,
      config.maximumOutgoingFamilyTransitions,
      (entry) => Math.max(0, entry.priorEvidence + entry.experienceEvidence - entry.inhibitionEvidence),
      (entry) => entry.toFamilyId,
    )
    const total = retained.reduce(
      (sum, entry) => sum + Math.max(0, entry.priorEvidence + entry.experienceEvidence - entry.inhibitionEvidence),
      0,
    )
    if (total <= FAMILY_EPSILON) continue
    retained.forEach((entry) => familyTransitions.push({
      ...entry,
      fromFamilyId,
      probability: Math.max(0, entry.priorEvidence + entry.experienceEvidence - entry.inhibitionEvidence) / total,
    }))
    peakOutgoingFamilyTransitions = Math.max(peakOutgoingFamilyTransitions, retained.length)
  }

  familyTransitions.sort((left, right) =>
    compareText(left.fromFamilyId, right.fromFamilyId) ||
    right.probability - left.probability ||
    compareText(left.toFamilyId, right.toFamilyId),
  )
  return { neuronTransitions: [...neuronTransitions], familyTransitions, peakOutgoingFamilyTransitions }
}

function softmax(entries: Array<RouteEntry & { logit: number }>, temperature: number): RouteEntry[] {
  if (!entries.length) return []
  const safeTemperature = Math.max(1e-3, temperature)
  const maximum = Math.max(...entries.map((entry) => entry.logit / safeTemperature))
  const exponentials = entries.map((entry) => Math.exp(entry.logit / safeTemperature - maximum))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return entries.map(({ logit: _logit, ...entry }, index) => ({
    ...entry,
    probability: total > FAMILY_EPSILON ? exponentials[index] / total : 0,
  }))
}

/** Computes the unordered network's neuron-level distribution through M and G. */
export function computeRouteDistribution(
  currentMemberKey: string,
  profiles: readonly BehaviorProfile[],
  memberships: readonly FamilyMembership[],
  neuronTransitions: readonly SparseNeuronTransition[],
  familyTransitions: readonly FamilyTransition[],
  config: FamilyConfig,
  context: RouteContext = {},
): RouteDistribution {
  const profileByKey = new Map(profiles.map((profile) => [profile.memberKey, profile]))
  const currentProfile = profileByKey.get(currentMemberKey)
  if (!currentProfile) {
    return {
      currentMemberKey,
      entries: [],
      unmatchedProbability: 1,
      totalProbability: 1,
      explanation: 'Başlangıç nöronunun davranış profili bulunamadı.',
    }
  }
  const byMember = membershipIndex(memberships)
  const currentMemberships = byMember.get(currentMemberKey) ?? []
  const directHabits = new Map(
    neuronTransitions
      .filter((transition) => transition.fromMemberKey === currentMemberKey)
      .map((transition) => [transition.toMemberKey, transition.probability]),
  )
  const familyProbability = new Map<string, number>()
  for (const membership of currentMemberships) {
    for (const transition of familyTransitions) {
      if (transition.fromFamilyId !== membership.familyId) continue
      familyProbability.set(
        transition.toFamilyId,
        (familyProbability.get(transition.toFamilyId) ?? 0) + membership.weight * transition.probability,
      )
    }
  }

  const candidateKeys = new Set<string>(directHabits.keys())
  for (const membership of memberships) {
    if ((familyProbability.get(membership.familyId) ?? 0) > 0) candidateKeys.add(membership.memberKey)
  }
  candidateKeys.delete(currentMemberKey)
  if (context.candidateMemberKeys) {
    for (const key of [...candidateKeys]) if (!context.candidateMemberKeys.has(key)) candidateKeys.delete(key)
  }

  const scored = [...candidateKeys]
    .map((memberKey): (RouteEntry & { logit: number; evidence: number }) | null => {
      const profile = profileByKey.get(memberKey)
      if (!profile) return null
      const targetMemberships = byMember.get(memberKey) ?? []
      let family = 0
      let strongestTargetFamily = 0
      for (const target of targetMemberships) {
        const probability = familyProbability.get(target.familyId) ?? 0
        family += probability * target.weight
        strongestTargetFamily = Math.max(strongestTargetFamily, probability)
      }
      const habit = directHabits.get(memberKey) ?? 0
      const signal = clamp(
        context.signalCompatibility?.get(memberKey) ?? Math.max(0, profileSimilarity(currentProfile, profile)),
      )
      const repetitionPenalty = clamp(context.repetition?.get(memberKey) ?? 0)
      const combinedLogit =
        config.habitRouteWeight * habit +
        config.familyRouteWeight * family +
        config.signalRouteWeight * signal -
        config.repetitionPenaltyWeight * repetitionPenalty
      return {
        memberKey,
        probability: 0,
        targetFamilyProbability: clamp(strongestTargetFamily),
        breakdown: { habit, family, signal, repetitionPenalty, combinedLogit },
        logit: combinedLogit,
        evidence: habit + family + signal,
      }
    })
    .filter((entry): entry is RouteEntry & { logit: number; evidence: number } => entry !== null)

  const retained = keepStrongest(
    scored,
    config.routingCandidateLimit,
    (entry) => entry.logit,
    (entry) => entry.memberKey,
  )
  const hasEvidence = retained.some((entry) => entry.evidence > FAMILY_EPSILON)
  if (!retained.length || !hasEvidence) {
    return {
      currentMemberKey,
      entries: [],
      unmatchedProbability: 1,
      totalProbability: 1,
      explanation: 'Kas önseli, aile yolu veya sinyal uyumu taşıyan bir sonraki aday bulunamadı.',
    }
  }
  const entries = softmax(retained, config.routingTemperature)
    .sort((left, right) => right.probability - left.probability || compareText(left.memberKey, right.memberKey))
  const probability = entries.reduce((sum, entry) => sum + entry.probability, 0)
  return {
    currentMemberKey,
    entries,
    unmatchedProbability: Math.max(0, 1 - probability),
    totalProbability: probability + Math.max(0, 1 - probability),
    explanation: 'Adaylar H⁰ önseli, yumuşak M üyelikleri, seyrek G yolları, sinyal uyumu ve tekrar cezasıyla birlikte puanlandı.',
  }
}
