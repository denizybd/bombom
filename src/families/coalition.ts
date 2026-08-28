import type {
  BehaviorProfile,
  CoalitionScoreBreakdown,
  CoalitionState,
  EventCoalition,
  EventRoleBinding,
  EventRoleKind,
  FamilyConfig,
  FamilyMembership,
  FamilyTransition,
  LearnedFamily,
  NeuronActivation,
} from './types'
import { profileSimilarity } from './profile'
import { FAMILY_EPSILON, clamp, compareText, stableDigest } from './sparse'

export interface CoalitionBuildResult {
  state: CoalitionState
  expansionCount: number
  peakBeamWidth: number
}

interface ActiveMember {
  activation: NeuronActivation
  profile: BehaviorProfile
  memberships: FamilyMembership[]
}

interface BindingCandidate {
  binding: EventRoleBinding
  activation: NeuronActivation
  profile: BehaviorProfile
  memberships: FamilyMembership[]
}

function indexMemberships(
  memberships: readonly FamilyMembership[],
): Map<string, FamilyMembership[]> {
  const result = new Map<string, FamilyMembership[]>()
  for (const membership of memberships) {
    const values = result.get(membership.memberKey) ?? []
    values.push(membership)
    result.set(membership.memberKey, values)
  }
  for (const values of result.values()) {
    values.sort((left, right) => right.weight - left.weight || compareText(left.familyId, right.familyId))
  }
  return result
}

function highestMembershipForRegion(
  member: ActiveMember,
  region: LearnedFamily['region'],
  familyById: ReadonlyMap<string, LearnedFamily>,
): FamilyMembership | null {
  return member.memberships
    .filter((membership) => familyById.get(membership.familyId)?.region === region)
    .sort((left, right) => right.weight - left.weight || compareText(left.familyId, right.familyId))[0] ?? null
}

function evidencePosition(profile: BehaviorProfile): {
  sourceDigest: string
  paragraphIndex: number
  start: number
  end: number
} | null {
  const evidence = profile.evidence[0]
  return evidence ? {
    sourceDigest: evidence.sourceDigest,
    paragraphIndex: evidence.paragraphIndex,
    start: evidence.start,
    end: evidence.end,
  } : null
}

function relativeRole(
  eventProfile: BehaviorProfile,
  candidate: ActiveMember,
  familyById: ReadonlyMap<string, LearnedFamily>,
): { role: EventRoleKind; roleFamilyId: string | null; explanation: string; fit: number } {
  const operator = highestMembershipForRegion(candidate, 'operator', familyById)
  if (operator) return {
    role: 'operator',
    roleFamilyId: operator.familyId,
    explanation: 'Operatör ailesi üyeliği, olayın kapsamını veya kesinliğini değiştiren geçici bağ önerdi.',
    fit: operator.weight,
  }

  const roleMembership = highestMembershipForRegion(candidate, 'role', familyById)
  const eventPosition = evidencePosition(eventProfile)
  const candidatePosition = evidencePosition(candidate.profile)
  const sameContext = Boolean(
    eventPosition && candidatePosition &&
    eventPosition.sourceDigest === candidatePosition.sourceDigest &&
    eventPosition.paragraphIndex === candidatePosition.paragraphIndex,
  )
  const instrumentSurface = /(?:\bile\b|(?:la|le))\s*$/iu.test(candidate.profile.text.trim())
  if (instrumentSurface && sameContext) return {
    role: 'instrument-hypothesis',
    roleFamilyId: roleMembership?.familyId ?? null,
    explanation: 'Araç benzeri yüzey ve aynı olay bağlamı yalnız bu koalisyonda geçici araç rolü önerdi; kalıcı aile üyeliği yazılmadı.',
    fit: Math.max(0.55, roleMembership?.weight ?? 0),
  }
  if (sameContext && eventPosition && candidatePosition && candidatePosition.end <= eventPosition.start) {
    return {
      role: 'agent-hypothesis',
      roleFamilyId: roleMembership?.familyId ?? null,
      explanation: 'Olaydan önceki aynı-bağlam içeriği geçici yapan adayı olarak bağlandı.',
      fit: Math.max(0.45, roleMembership?.weight ?? 0),
    }
  }
  if (sameContext && eventPosition && candidatePosition && candidatePosition.start >= eventPosition.end) {
    return {
      role: 'patient-hypothesis',
      roleFamilyId: roleMembership?.familyId ?? null,
      explanation: 'Olaydan sonraki aynı-bağlam içeriği geçici etkilenen adayı olarak bağlandı.',
      fit: Math.max(0.45, roleMembership?.weight ?? 0),
    }
  }
  return {
    role: 'participant',
    roleFamilyId: roleMembership?.familyId ?? null,
    explanation: roleMembership
      ? 'Rol ailesi yakınlığı, belirli alt rolü kanıtlamadan geçici katılımcı bağı önerdi.'
      : 'Aktif içerik olayla uyumlu bir geçici katılımcı olarak tutuldu.',
    fit: Math.max(0.3, roleMembership?.weight ?? 0),
  }
}

function familyActivationMap(
  active: readonly ActiveMember[],
  maximumActiveFamilies: number,
): Map<string, number> {
  const numerators = new Map<string, number>()
  const denominators = new Map<string, number>()
  for (const member of active) {
    for (const membership of member.memberships) {
      numerators.set(
        membership.familyId,
        (numerators.get(membership.familyId) ?? 0) + member.activation.activation * membership.weight,
      )
      denominators.set(
        membership.familyId,
        (denominators.get(membership.familyId) ?? 0) + membership.weight * membership.weight,
      )
    }
  }
  const retained = [...numerators.entries()]
    .map(([familyId, numerator]) => ({
      familyId,
      activation: clamp(numerator / Math.max(FAMILY_EPSILON, Math.sqrt(denominators.get(familyId) ?? 0))),
    }))
    .sort((left, right) => right.activation - left.activation || compareText(left.familyId, right.familyId))
    .slice(0, maximumActiveFamilies)
  return new Map(retained.map((entry) => [entry.familyId, entry.activation]))
}

function familyConnection(
  eventFamilyId: string,
  member: ActiveMember,
  transitions: readonly FamilyTransition[],
): number {
  let score = 0
  for (const membership of member.memberships) {
    const forward = transitions.find((transition) => (
      transition.fromFamilyId === eventFamilyId && transition.toFamilyId === membership.familyId
    ))?.probability ?? 0
    const backward = transitions.find((transition) => (
      transition.toFamilyId === eventFamilyId && transition.fromFamilyId === membership.familyId
    ))?.probability ?? 0
    score += membership.weight * Math.max(forward, backward)
  }
  return clamp(score)
}

function scoreBindings(
  event: ActiveMember,
  eventFamilyId: string,
  candidates: readonly BindingCandidate[],
  previous: CoalitionState | null | undefined,
  transitions: readonly FamilyTransition[],
): CoalitionScoreBreakdown {
  const members = [event, ...candidates.map((candidate) => ({
    activation: candidate.activation,
    profile: candidate.profile,
    memberships: [] as FamilyMembership[],
  }))]
  const activation = clamp(
    members.reduce((sum, member) => sum + clamp(member.activation.activation), 0) / Math.max(1, members.length),
  )
  const coherence = candidates.length
    ? clamp(candidates.reduce((sum, candidate) => {
      const behavior = Math.max(0, profileSimilarity(event.profile, candidate.profile))
      const family = familyConnection(
        eventFamilyId,
        { activation: candidate.activation, profile: candidate.profile, memberships: candidate.memberships },
        transitions,
      )
      return sum + 0.75 * behavior + 0.25 * family
    }, 0) / candidates.length)
    : 0
  const prior = previous?.coalitions.find((coalition) => coalition.eventMemberKey === event.profile.memberKey)
  const persistence = prior ? clamp(0.5 + 0.5 * prior.dominance) : 0
  const goalRelevance = clamp(
    members.reduce((sum, member) => sum + clamp(member.activation.goalRelevance ?? 0), 0) /
      Math.max(1, members.length),
  )
  const conflict = clamp(
    members.reduce((sum, member) => sum + clamp(member.activation.conflict ?? 0), 0) /
      Math.max(1, members.length),
  )
  const repetition = clamp(
    members.reduce((sum, member) => sum + clamp(member.activation.repetition ?? 0), 0) /
      Math.max(1, members.length),
  )
  const cooldown = prior ? clamp(prior.dominance * repetition) : 0
  const total =
    0.36 * activation +
    0.27 * coherence +
    0.12 * persistence +
    0.15 * goalRelevance -
    0.05 * conflict -
    0.03 * repetition -
    0.02 * cooldown
  return { activation, coherence, persistence, goalRelevance, conflict, repetition, cooldown, total }
}

function normalizeDominance(coalitions: EventCoalition[]): EventCoalition[] {
  if (!coalitions.length) return []
  const maximum = Math.max(...coalitions.map((coalition) => coalition.score.total))
  const exponentials = coalitions.map((coalition) => Math.exp((coalition.score.total - maximum) / 0.25))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return coalitions.map((coalition, index) => ({
    ...coalition,
    dominance: total > FAMILY_EPSILON ? exponentials[index] / total : 0,
  })).sort((left, right) => right.dominance - left.dominance || compareText(left.id, right.id))
}

/**
 * Builds a bounded transient workspace. No role binding is written back to M,
 * learned families, I or R.
 */
export function buildEventCoalitions(
  activations: readonly NeuronActivation[],
  profiles: readonly BehaviorProfile[],
  families: readonly LearnedFamily[],
  memberships: readonly FamilyMembership[],
  familyTransitions: readonly FamilyTransition[],
  config: FamilyConfig,
  previous?: CoalitionState | null,
): CoalitionBuildResult {
  const profileByKey = new Map(profiles.map((profile) => [profile.memberKey, profile]))
  const familyById = new Map(families.map((family) => [family.id, family]))
  const membershipsByMember = indexMemberships(memberships)
  const active: ActiveMember[] = activations
    .filter((activation) => Number.isFinite(activation.activation) && activation.activation > 0)
    .map((activation): ActiveMember | null => {
      const profile = profileByKey.get(activation.memberKey)
      if (!profile) return null
      return {
        activation: { ...activation, activation: clamp(activation.activation) },
        profile,
        memberships: membershipsByMember.get(activation.memberKey) ?? [],
      }
    })
    .filter((member): member is ActiveMember => member !== null)
    .sort((left, right) =>
      right.activation.activation - left.activation.activation || compareText(left.profile.memberKey, right.profile.memberKey),
    )
    .slice(0, config.maximumActiveNeurons)

  const familyActivations = familyActivationMap(active, config.maximumActiveFamilies)
  const eventSeeds = active
    .flatMap((member) => member.memberships
      .filter((membership) => familyById.get(membership.familyId)?.region === 'event')
      .map((membership) => ({
        member,
        membership,
        seedScore: member.activation.activation * membership.weight *
          (familyActivations.get(membership.familyId) ?? 0),
      })),
    )
    .filter((seed) => seed.seedScore >= config.runtimeActivationThreshold)
    .sort((left, right) => right.seedScore - left.seedScore || compareText(left.member.profile.memberKey, right.member.profile.memberKey))
    .slice(0, Math.min(config.maximumCoalitions, config.coalitionBeamWidth))

  if (!eventSeeds.length) {
    return {
      state: {
        tick: (previous?.tick ?? 0) + 1,
        activeFamilyCount: familyActivations.size,
        coalitions: [],
        unmatchedReason: 'Runtime eşiğini geçen bir olay/eylem ailesi tohumu bulunmadı; sahte olay çerçevesi kurulmadı.',
      },
      expansionCount: 0,
      peakBeamWidth: 0,
    }
  }

  let expansionCount = 0
  let peakBeamWidth = eventSeeds.length
  const coalitions: EventCoalition[] = []
  for (const seed of eventSeeds) {
    const candidates = active
      .filter((member) => member.profile.memberKey !== seed.member.profile.memberKey)
      .map((member): BindingCandidate => {
        const role = relativeRole(seed.member.profile, member, familyById)
        const behaviorFit = Math.max(0, profileSimilarity(seed.member.profile, member.profile))
        const connection = familyConnection(seed.membership.familyId, member, familyTransitions)
        const score = clamp(
          0.42 * member.activation.activation +
          0.25 * role.fit +
          0.2 * behaviorFit +
          0.13 * connection,
        )
        return {
          profile: member.profile,
          activation: member.activation,
          memberships: member.memberships,
          binding: {
            role: role.role,
            memberKey: member.profile.memberKey,
            roleFamilyId: role.roleFamilyId,
            score,
            temporary: true,
            explanation: role.explanation,
          },
        }
      })
      .sort((left, right) => right.binding.score - left.binding.score || compareText(left.profile.memberKey, right.profile.memberKey))
    expansionCount += candidates.length
    peakBeamWidth = Math.max(peakBeamWidth, Math.min(candidates.length, config.coalitionBeamWidth))

    // A member receives one role only; specialized roles win before generic participants.
    const roleCounts = new Map<EventRoleKind, number>()
    const selected: BindingCandidate[] = []
    for (const candidate of candidates.slice(0, config.coalitionBeamWidth)) {
      if (selected.length >= config.maximumRoleCandidates) break
      const maximumForRole = candidate.binding.role === 'participant'
        ? config.maximumRoleCandidates
        : 1
      if ((roleCounts.get(candidate.binding.role) ?? 0) >= maximumForRole) continue
      roleCounts.set(candidate.binding.role, (roleCounts.get(candidate.binding.role) ?? 0) + 1)
      selected.push(candidate)
    }

    const score = scoreBindings(
      seed.member,
      seed.membership.familyId,
      selected,
      previous,
      familyTransitions,
    )
    const bindings = selected.map((candidate) => candidate.binding)
    const memberKeys = [seed.member.profile.memberKey, ...bindings.map((binding) => binding.memberKey)]
    const fingerprint = stableDigest([
      'event-coalition-v1',
      (previous?.tick ?? 0) + 1,
      seed.member.profile.memberKey,
      seed.membership.familyId,
      ...bindings.flatMap((binding) => [binding.role, binding.memberKey]),
    ])
    coalitions.push({
      id: `KO-${fingerprint.slice(0, 16)}`,
      eventMemberKey: seed.member.profile.memberKey,
      eventFamilyId: seed.membership.familyId,
      familyActivations: [...familyActivations.entries()]
        .map(([familyId, activation]) => ({ familyId, activation }))
        .sort((left, right) => right.activation - left.activation || compareText(left.familyId, right.familyId)),
      bindings,
      memberKeys,
      score,
      dominance: 0,
      confidence: clamp(
        0.4 * score.activation +
        0.35 * score.coherence +
        0.25 * Math.min(1, bindings.length / Math.max(1, config.maximumRoleCandidates)),
      ),
      temporary: true,
    })
  }

  const retained = normalizeDominance(coalitions)
    .slice(0, config.maximumCoalitions)
  return {
    state: {
      tick: (previous?.tick ?? 0) + 1,
      activeFamilyCount: familyActivations.size,
      coalitions: retained,
      unmatchedReason: null,
    },
    expansionCount,
    peakBeamWidth,
  }
}

export const advanceCoalitions = buildEventCoalitions
