import type {
  BehaviorProfile,
  FamilyMembership,
  FamilyTransition,
  FamilyRun,
  FunctionalRegionId,
  LearnedFamily,
  SparseNeuronTransition,
} from '../families/types'
import {
  LANGUAGE_BOS_FAMILY,
  LANGUAGE_BOS_MEMBER,
  LANGUAGE_EOS_FAMILY,
  LANGUAGE_EOS_MEMBER,
} from './types'
import type {
  FamilyRouteCandidate,
  LanguageExperienceState,
  LanguageGenerationResult,
  LanguageRouteContext,
  LanguageRouteResult,
  LanguageTransitionEdge,
  NeuronRouteCandidate,
} from './types'
import {
  clamp,
  compareText,
  normalizePositive,
  seededUnit,
  softmax,
} from './deterministic'
import { assertLanguageStateCompatible } from './state'

interface WeightedOrigin {
  key: string
  weight: number
}

export interface PreparedFamilyRoutingSpace {
  familyRunFingerprint: string
  membershipsByMember: ReadonlyMap<string, FamilyMembership[]>
  membershipsByFamily: ReadonlyMap<string, FamilyMembership[]>
  familyById: ReadonlyMap<string, LearnedFamily>
  profileByKey: ReadonlyMap<string, BehaviorProfile>
  frozenFamilyByOrigin: ReadonlyMap<string, FamilyTransition[]>
  frozenNeuronByOrigin: ReadonlyMap<string, SparseNeuronTransition[]>
}

export interface PreparedLearnedRoutingSpace {
  familyByOrigin: ReadonlyMap<string, LanguageTransitionEdge[]>
  neuronByOrigin: ReadonlyMap<string, LanguageTransitionEdge[]>
  /** Members that have actually served as a teacher-forced surface target. */
  outputMemberKeys: ReadonlySet<string>
}

function membershipsByMember(run: FamilyRun): Map<string, FamilyMembership[]> {
  const index = new Map<string, FamilyMembership[]>()
  for (const membership of run.memberships) {
    const values = index.get(membership.memberKey) ?? []
    values.push(membership)
    index.set(membership.memberKey, values)
  }
  return index
}

export function prepareFamilyRoutingSpace(run: FamilyRun): PreparedFamilyRoutingSpace {
  const byMember = membershipsByMember(run)
  const byFamily = new Map<string, FamilyMembership[]>()
  for (const membership of run.memberships) {
    const values = byFamily.get(membership.familyId) ?? []
    values.push(membership)
    byFamily.set(membership.familyId, values)
  }
  const frozenFamilyByOrigin = new Map<string, FamilyTransition[]>()
  for (const transition of run.familyTransitions) {
    const values = frozenFamilyByOrigin.get(transition.fromFamilyId) ?? []
    values.push(transition)
    frozenFamilyByOrigin.set(transition.fromFamilyId, values)
  }
  const frozenNeuronByOrigin = new Map<string, SparseNeuronTransition[]>()
  for (const transition of run.neuronTransitions) {
    const values = frozenNeuronByOrigin.get(transition.fromMemberKey) ?? []
    values.push(transition)
    frozenNeuronByOrigin.set(transition.fromMemberKey, values)
  }
  return {
    familyRunFingerprint: run.fingerprint,
    membershipsByMember: byMember,
    membershipsByFamily: byFamily,
    familyById: new Map(run.families.map((family) => [family.id, family])),
    profileByKey: new Map(run.profiles.map((profile) => [profile.memberKey, profile])),
    frozenFamilyByOrigin,
    frozenNeuronByOrigin,
  }
}

function normalizedMemberships(memberships: readonly FamilyMembership[]): WeightedOrigin[] {
  return normalizePositive(memberships.map((membership) => ({
    key: membership.familyId,
    score: Math.max(0, membership.weight),
  }))).map(({ key, probability }) => ({ key, weight: probability }))
}

function currentFamilyOrigins(
  recentMemberKeys: readonly string[],
  index: ReadonlyMap<string, FamilyMembership[]>,
): WeightedOrigin[] {
  const current = recentMemberKeys.at(-1)
  if (!current) return [{ key: LANGUAGE_BOS_FAMILY, weight: 1 }]
  const values = normalizedMemberships(index.get(current) ?? [])
  return values.length ? values : [{ key: LANGUAGE_BOS_FAMILY, weight: 1 }]
}

function currentMemberOrigin(recentMemberKeys: readonly string[]): string {
  return recentMemberKeys.at(-1) ?? LANGUAGE_BOS_MEMBER
}

function edgeIndex(edges: readonly LanguageTransitionEdge[]): Map<string, LanguageTransitionEdge[]> {
  const index = new Map<string, LanguageTransitionEdge[]>()
  for (const edge of edges) {
    const values = index.get(edge.fromKey) ?? []
    values.push(edge)
    index.set(edge.fromKey, values)
  }
  return index
}

export function prepareLearnedRoutingSpace(state: LanguageExperienceState): PreparedLearnedRoutingSpace {
  return {
    familyByOrigin: edgeIndex(state.familyTransitions),
    neuronByOrigin: edgeIndex(state.neuronTransitions),
    outputMemberKeys: new Set(state.neuronTransitions
      .filter((edge) => edge.toKey !== LANGUAGE_EOS_MEMBER && edge.excitatoryEvidence > 0)
      .map((edge) => edge.toKey)),
  }
}

function coalitionActivations(context: LanguageRouteContext): Map<string, number> {
  const values = new Map<string, number>()
  for (const coalition of context.coalition?.coalitions ?? []) {
    for (const activation of coalition.familyActivations) {
      values.set(
        activation.familyId,
        Math.max(values.get(activation.familyId) ?? 0, activation.activation * coalition.dominance),
      )
    }
  }
  return values
}

function historyFamilyEvidence(
  recentMemberKeys: readonly string[],
  membershipIndex: ReadonlyMap<string, FamilyMembership[]>,
  learned: ReadonlyMap<string, LanguageTransitionEdge[]>,
): Map<string, number> {
  const output = new Map<string, number>()
  const recent = recentMemberKeys.slice(-4)
  recent.forEach((memberKey, index) => {
    const recency = (index + 1) / recent.length
    for (const origin of normalizedMemberships(membershipIndex.get(memberKey) ?? [])) {
      for (const edge of learned.get(origin.key) ?? []) {
        output.set(edge.toKey, (output.get(edge.toKey) ?? 0) + recency * origin.weight * edge.score)
      }
    }
  })
  return output
}

function familyCandidates(
  state: LanguageExperienceState,
  context: LanguageRouteContext,
  space: PreparedFamilyRoutingSpace,
  learnedSpace: PreparedLearnedRoutingSpace,
): { candidates: FamilyRouteCandidate[]; unmatchedProbability: number } {
  const learned = learnedSpace.familyByOrigin
  const origins = currentFamilyOrigins(context.recentMemberKeys, space.membershipsByMember)
  const coalition = coalitionActivations(context)
  const history = historyFamilyEvidence(context.recentMemberKeys, space.membershipsByMember, learned)
  const frozenByOrigin = space.frozenFamilyByOrigin
  const targetIds = new Set<string>([...coalition.keys(), ...history.keys()])
  for (const origin of origins) {
    for (const edge of learned.get(origin.key) ?? []) targetIds.add(edge.toKey)
    for (const transition of frozenByOrigin.get(origin.key) ?? []) targetIds.add(transition.toFamilyId)
  }
  if (context.candidateMemberKeys) {
    for (const memberKey of context.candidateMemberKeys) {
      for (const membership of space.membershipsByMember.get(memberKey) ?? []) targetIds.add(membership.familyId)
    }
  }
  const familyById = space.familyById
  const scored = [...targetIds]
    .filter((familyId) => familyId === LANGUAGE_EOS_FAMILY || familyById.has(familyId))
    .map((familyId) => {
      let learnedEvidence = 0
      let frozenPrior = 0
      for (const origin of origins) {
        learnedEvidence += origin.weight * (
          learned.get(origin.key)?.find((edge) => edge.toKey === familyId)?.score ?? 0
        )
        frozenPrior += origin.weight * (
          frozenByOrigin.get(origin.key)?.find((edge) => edge.toFamilyId === familyId)?.probability ?? 0
        )
      }
      const coalitionActivation = coalition.get(familyId) ?? 0
      const historyEvidence = history.get(familyId) ?? 0
      const score =
        state.config.learnedFamilyWeight * learnedEvidence +
        state.config.frozenFamilyPriorWeight * frozenPrior +
        state.config.coalitionWeight * coalitionActivation +
        state.config.historyWeight * historyEvidence
      return {
        familyId,
        region: familyId === LANGUAGE_EOS_FAMILY
          ? 'language-control' as const
          : (familyById.get(familyId)?.region ?? 'content') as FunctionalRegionId,
        probability: 0,
        score,
        learnedEvidence,
        frozenPrior,
        coalitionActivation,
        historyEvidence,
      }
    })
    .sort((left, right) => right.score - left.score || compareText(left.familyId, right.familyId))
    .slice(0, state.config.maximumRouteFamilies)

  if (!scored.length) return { candidates: [], unmatchedProbability: 1 }
  const withUnmatched = softmax([
    ...scored.map((candidate) => ({ kind: 'candidate' as const, candidate, score: candidate.score })),
    { kind: 'unmatched' as const, candidate: null, score: 0 },
  ], state.config.routeTemperature)
  const candidates = withUnmatched
    .filter((entry): entry is typeof entry & { kind: 'candidate'; candidate: FamilyRouteCandidate } => entry.kind === 'candidate')
    .map((entry) => ({ ...entry.candidate, probability: entry.probability }))
    .sort((left, right) => right.probability - left.probability || compareText(left.familyId, right.familyId))
  return {
    candidates,
    unmatchedProbability: withUnmatched.find((entry) => entry.kind === 'unmatched')?.probability ?? 0,
  }
}

function historyNeuronEvidence(
  recentMemberKeys: readonly string[],
  learned: ReadonlyMap<string, LanguageTransitionEdge[]>,
): Map<string, number> {
  const output = new Map<string, number>()
  const recent = recentMemberKeys.slice(-4)
  recent.forEach((memberKey, index) => {
    const recency = (index + 1) / recent.length
    for (const edge of learned.get(memberKey) ?? []) {
      output.set(edge.toKey, (output.get(edge.toKey) ?? 0) + recency * edge.score)
    }
  })
  return output
}

function memberCandidates(
  state: LanguageExperienceState,
  context: LanguageRouteContext,
  families: readonly FamilyRouteCandidate[],
  space: PreparedFamilyRoutingSpace,
  learnedSpace: PreparedLearnedRoutingSpace,
): { candidates: NeuronRouteCandidate[]; eosProbability: number; strandedProbability: number } {
  const profileByKey = space.profileByKey
  const learned = learnedSpace.neuronByOrigin
  const current = currentMemberOrigin(context.recentMemberKeys)
  const directEdges = learned.get(current) ?? []
  const directByTarget = new Map(directEdges.map((edge) => [edge.toKey, edge]))
  const history = historyNeuronEvidence(context.recentMemberKeys, learned)
  const frozen = new Map((space.frozenNeuronByOrigin.get(current) ?? [])
    .map((transition) => [transition.toMemberKey, transition.probability]))
  const explicitAllowed = context.candidateMemberKeys ? new Set(context.candidateMemberKeys) : null
  const learnedVocabulary = learnedSpace.outputMemberKeys
  const allowed = explicitAllowed || learnedVocabulary.size
    ? new Set([...(learnedVocabulary.size ? learnedVocabulary : (explicitAllowed ?? []))]
      .filter((memberKey) => !explicitAllowed || explicitAllowed.has(memberKey)))
    : null
  const recentCounts = new Map<string, number>()
  for (const memberKey of context.recentMemberKeys.slice(-8)) {
    recentCounts.set(memberKey, (recentCounts.get(memberKey) ?? 0) + 1)
  }
  const joint = new Map<string, {
    probability: number
    learnedEvidence: number
    membershipEvidence: number
    historyEvidence: number
    repetitionPenalty: number
    score: number
    familyProbability: number
  }>()
  let eosProbability = 0
  let strandedProbability = 0

  for (const family of families) {
    if (family.familyId === LANGUAGE_EOS_FAMILY) {
      eosProbability += family.probability
      continue
    }
    const local = (space.membershipsByFamily.get(family.familyId) ?? [])
      .filter((membership) => !allowed || allowed.has(membership.memberKey))
      .map((membership) => {
        const learnedEvidence = directByTarget.get(membership.memberKey)?.score ?? 0
        const historyEvidence = (history.get(membership.memberKey) ?? 0) + (frozen.get(membership.memberKey) ?? 0)
        const repetition = clamp((recentCounts.get(membership.memberKey) ?? 0) / 3)
        const score =
          state.config.learnedNeuronWeight * learnedEvidence +
          state.config.membershipWeight * membership.weight +
          state.config.historyWeight * historyEvidence -
          state.config.repetitionPenalty * repetition
        return { membership, score, learnedEvidence, historyEvidence, repetition }
      })
      .filter((candidate) => profileByKey.has(candidate.membership.memberKey))
      .sort((left, right) => right.score - left.score || compareText(left.membership.memberKey, right.membership.memberKey))
      .slice(0, state.config.maximumRouteNeurons)
    if (!local.length) {
      strandedProbability += family.probability
      continue
    }
    const conditional = softmax(local, state.config.routeTemperature)
    for (const candidate of conditional) {
      const memberKey = candidate.membership.memberKey
      const membershipEvidence = candidate.membership.weight
      const contribution = family.probability * candidate.probability
      const previous = joint.get(memberKey) ?? {
        probability: 0,
        learnedEvidence: 0,
        membershipEvidence: 0,
        historyEvidence: 0,
        repetitionPenalty: 0,
        score: 0,
        familyProbability: 0,
      }
      previous.probability += contribution
      previous.familyProbability += family.probability * membershipEvidence
      previous.learnedEvidence = Math.max(previous.learnedEvidence, candidate.learnedEvidence)
      previous.membershipEvidence = Math.max(previous.membershipEvidence, membershipEvidence)
      previous.historyEvidence = Math.max(previous.historyEvidence, candidate.historyEvidence)
      previous.repetitionPenalty = Math.max(previous.repetitionPenalty, candidate.repetition)
      previous.score = Math.max(previous.score, candidate.score)
      joint.set(memberKey, previous)
    }
  }

  const allCandidates = [...joint]
    .map(([memberKey, value]): NeuronRouteCandidate => ({
      memberKey,
      text: profileByKey.get(memberKey)?.text ?? '',
      identityDigest: profileByKey.get(memberKey)?.identityDigest ?? '',
      probability: value.probability,
      score: value.score,
      familyProbability: value.familyProbability,
      learnedEvidence: value.learnedEvidence,
      membershipEvidence: value.membershipEvidence,
      historyEvidence: value.historyEvidence,
      repetitionPenalty: value.repetitionPenalty,
    }))
    .sort((left, right) => right.probability - left.probability || compareText(left.memberKey, right.memberKey))
  const candidates = allCandidates.slice(0, state.config.maximumRouteNeurons)
  const droppedProbability = allCandidates.slice(state.config.maximumRouteNeurons)
    .reduce((sum, candidate) => sum + candidate.probability, 0)
  return { candidates, eosProbability, strandedProbability: strandedProbability + droppedProbability }
}

/** Routes through families first, then distributes each family's mass inside its members. */
export function routeNextLanguageStep(
  state: LanguageExperienceState,
  familyRun: FamilyRun,
  context: LanguageRouteContext,
): LanguageRouteResult {
  assertLanguageStateCompatible(state, familyRun)
  return routeNextLanguageStepUnchecked(
    state,
    familyRun,
    context,
    prepareFamilyRoutingSpace(familyRun),
    prepareLearnedRoutingSpace(state),
  )
}

/** Internal hot-path variant. Callers must validate state/run compatibility once. */
export function routeNextLanguageStepUnchecked(
  state: LanguageExperienceState,
  familyRun: FamilyRun,
  context: LanguageRouteContext,
  preparedFamilySpace?: PreparedFamilyRoutingSpace,
  preparedLearnedSpace?: PreparedLearnedRoutingSpace,
): LanguageRouteResult {
  const familySpace = preparedFamilySpace ?? prepareFamilyRoutingSpace(familyRun)
  const learnedSpace = preparedLearnedSpace ?? prepareLearnedRoutingSpace(state)
  const family = familyCandidates(state, context, familySpace, learnedSpace)
  const member = memberCandidates(state, context, family.candidates, familySpace, learnedSpace)
  const unmatchedProbability = clamp(family.unmatchedProbability + member.strandedProbability)
  const topFamily = family.candidates[0] ?? null
  const topMember = member.candidates[0] ?? null
  const eosWins = member.eosProbability > (topMember?.probability ?? 0)
  return {
    familyCandidates: family.candidates,
    neuronCandidates: member.candidates,
    eosProbability: member.eosProbability,
    unmatchedProbability,
    selectedFamilyId: topFamily?.familyId ?? null,
    selectedMemberKey: eosWins ? null : topMember?.memberKey ?? null,
    selectedText: eosWins ? null : topMember?.text ?? null,
    explanation: family.candidates.length
      ? 'Önce öğrenilmiş aile yolu, dondurulmuş aile önseli ve geçici düşünce koalisyonu; sonra aile içi nöron yolu ve yakın çıktı geçmişi birlikte yönlendirdi.'
      : 'Dil yolu için öğrenilmiş geçiş, aile önseli veya etkin düşünce koalisyonu bulunamadı.',
  }
}

function appendSurface(current: string, next: string): string {
  if (!current) return next
  if (/^[.,!?;:%)\]}»”’]+/u.test(next) || /[(\[{«“‘]$/u.test(current)) return `${current}${next}`
  return `${current} ${next}`
}

export function generateLanguageSequence(
  state: LanguageExperienceState,
  familyRun: FamilyRun,
  initial: Omit<LanguageRouteContext, 'recentMemberKeys'> & { recentMemberKeys?: string[] },
  options: { maximumSteps?: number; strategy?: 'greedy' | 'seeded-sample'; seed?: number } = {},
): LanguageGenerationResult {
  assertLanguageStateCompatible(state, familyRun)
  const maximumSteps = Math.max(1, Math.min(512, Math.trunc(options.maximumSteps ?? 64)))
  const strategy = options.strategy ?? 'greedy'
  const seed = options.seed ?? state.config.seed
  const recent = [...(initial.recentMemberKeys ?? [])]
  const steps: LanguageGenerationResult['steps'] = []
  let text = ''
  let stoppedBy: LanguageGenerationResult['stoppedBy'] = 'maximum-steps'
  const familySpace = prepareFamilyRoutingSpace(familyRun)
  const learnedSpace = prepareLearnedRoutingSpace(state)

  for (let index = 0; index < maximumSteps; index += 1) {
    const route = routeNextLanguageStepUnchecked(
      state,
      familyRun,
      { ...initial, recentMemberKeys: recent },
      familySpace,
      learnedSpace,
    )
    const outcomes = [
      ...route.neuronCandidates.map((candidate) => ({
        kind: 'member' as const,
        memberKey: candidate.memberKey,
        text: candidate.text,
        probability: candidate.probability,
      })),
      { kind: 'eos' as const, memberKey: null, text: null, probability: route.eosProbability },
    ].sort((left, right) => right.probability - left.probability || compareText(left.memberKey ?? LANGUAGE_EOS_MEMBER, right.memberKey ?? LANGUAGE_EOS_MEMBER))
    if (!outcomes.length || outcomes[0].probability <= 0) {
      steps.push({ index, route, emittedMemberKey: null, emittedText: null, eos: false })
      stoppedBy = 'unmatched'
      break
    }
    let selected = outcomes[0]
    if (strategy === 'seeded-sample') {
      const total = outcomes.reduce((sum, outcome) => sum + outcome.probability, 0)
      let cursor = seededUnit(seed, state.fingerprint, index, recent) * total
      selected = outcomes.find((outcome) => {
        cursor -= outcome.probability
        return cursor <= 0
      }) ?? outcomes.at(-1)!
    }
    if (selected.kind === 'eos') {
      steps.push({ index, route, emittedMemberKey: null, emittedText: null, eos: true })
      stoppedBy = 'eos'
      break
    }
    recent.push(selected.memberKey)
    if (recent.length > state.config.maximumPrefixPieces) recent.shift()
    text = appendSurface(text, selected.text)
    steps.push({ index, route, emittedMemberKey: selected.memberKey, emittedText: selected.text, eos: false })
  }
  return { text, memberKeys: steps.flatMap((step) => step.emittedMemberKey ? [step.emittedMemberKey] : []), steps, stoppedBy }
}
