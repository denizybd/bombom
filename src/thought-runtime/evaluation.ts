import type { FamilyRun } from '../families/types'
import { clamp } from './math'
import type {
  EventGraphEvaluation,
  EventGraphV1,
  ThoughtBranch,
  ThoughtBranchEvaluation,
  ThoughtEpisodeState,
  ThoughtModel,
  ThoughtRoleHypothesis,
  ThoughtRoutingHints,
  ThoughtRuntimeConfig,
} from './types'
import type { ThoughtRuntimeIndex } from './routing'

export interface StructuralGraphScore {
  precision: number
  recall: number
  f1: number
  constraintAccuracy: number
  matchedAtoms: number
  predictedAtoms: number
  expectedAtoms: number
}

function canonicalValue(value: string | null): string | null {
  if (value === null) return null
  const normalized = value.normalize('NFC').toLocaleLowerCase('tr-TR')
    .replace(/[\p{P}\p{S}]+/gu, ' ').replace(/\s+/gu, ' ').trim()
  return normalized || null
}

export function normalizeEventGraph(graph: EventGraphV1): EventGraphV1 {
  if (!graph || graph.format !== 'event-graph-v1') throw new TypeError('Beklenen çıktı event-graph-v1 olmalıdır.')
  const operators = [...new Set((graph.operators ?? []).map((value) => canonicalValue(value)).filter((value): value is string => Boolean(value)))].sort()
  const goals = (graph.goals ?? []).map((goal) => ({ relation: canonicalValue(goal.relation), target: canonicalValue(goal.target) }))
    .filter((goal): goal is { relation: string; target: string } => Boolean(goal.relation && goal.target))
    .sort((left, right) => left.relation.localeCompare(right.relation, 'tr') || left.target.localeCompare(right.target, 'tr'))
  return {
    format: 'event-graph-v1',
    actor: canonicalValue(graph.actor), action: canonicalValue(graph.action),
    patient: canonicalValue(graph.patient), instrument: canonicalValue(graph.instrument),
    operators, negated: Boolean(graph.negated), goals,
  }
}

function graphAtoms(graph: EventGraphV1): Set<string> {
  const normalized = normalizeEventGraph(graph)
  const atoms = new Set<string>()
  for (const key of ['actor', 'action', 'patient', 'instrument'] as const) {
    const value = normalized[key]
    if (value) atoms.add(`${key}:${value}`)
  }
  for (const operator of normalized.operators) atoms.add(`operator:${operator}`)
  for (const goal of normalized.goals) atoms.add(`goal:${goal.relation}:${goal.target}`)
  atoms.add(`negated:${normalized.negated ? '1' : '0'}`)
  return atoms
}

function setEqual(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

export function scoreEventGraph(predictedInput: EventGraphV1, expectedInput: EventGraphV1): StructuralGraphScore {
  const predicted = normalizeEventGraph(predictedInput)
  const expected = normalizeEventGraph(expectedInput)
  const predictedAtoms = graphAtoms(predicted)
  const expectedAtoms = graphAtoms(expected)
  let matched = 0
  for (const atom of predictedAtoms) if (expectedAtoms.has(atom)) matched += 1
  const precision = predictedAtoms.size ? matched / predictedAtoms.size : expectedAtoms.size ? 0 : 1
  const recall = expectedAtoms.size ? matched / expectedAtoms.size : predictedAtoms.size ? 0 : 1
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
  const constraints = [
    predicted.actor === expected.actor,
    predicted.action === expected.action,
    predicted.patient === expected.patient,
    predicted.instrument === expected.instrument,
    predicted.negated === expected.negated,
    setEqual(predicted.operators, expected.operators),
    setEqual(predicted.goals.map((goal) => `${goal.relation}\u0000${goal.target}`), expected.goals.map((goal) => `${goal.relation}\u0000${goal.target}`)),
  ]
  return {
    precision, recall, f1,
    constraintAccuracy: constraints.filter(Boolean).length / constraints.length,
    matchedAtoms: matched, predictedAtoms: predictedAtoms.size, expectedAtoms: expectedAtoms.size,
  }
}

export function evaluateEventGraph(
  predictedInput: EventGraphV1,
  goldInput: EventGraphV1,
  config: ThoughtRuntimeConfig,
): EventGraphEvaluation {
  const score = scoreEventGraph(predictedInput, goldInput)
  const falsePositive = score.predictedAtoms - score.matchedAtoms
  const falseNegative = score.expectedAtoms - score.matchedAtoms
  return {
    graphF1: score.f1,
    constraintAccuracy: score.constraintAccuracy,
    resultScore: clamp(config.outcomeGraphWeight * score.f1 + config.outcomeConstraintWeight * score.constraintAccuracy),
    truePositive: score.matchedAtoms,
    falsePositive,
    falseNegative,
  }
}

function surface(index: ThoughtRuntimeIndex, memberKey: string): string | null {
  const value = index.neuronById.get(memberKey)?.provenance.surface?.trim()
  return value || null
}

function roleValue(
  role: ThoughtRoleHypothesis['role'],
  hypotheses: readonly ThoughtRoleHypothesis[],
  path: ReadonlySet<string>,
  index: ThoughtRuntimeIndex,
): string | null {
  const candidate = hypotheses.filter((entry) => entry.role === role && path.has(entry.memberKey))
    .sort((left, right) => right.score - left.score || left.memberKey.localeCompare(right.memberKey))[0]
  return candidate ? surface(index, candidate.memberKey) : null
}

/** Role assignments are reconstructed as temporary coalition hypotheses, never neuron labels. */
export function deriveEventGraph(
  episode: ThoughtEpisodeState,
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  hints: ThoughtRoutingHints = {},
): EventGraphV1 {
  const branch = [...episode.branches].sort((left, right) => right.mass - left.mass || left.id.localeCompare(right.id))[0]
  if (!branch) return { format: 'event-graph-v1', actor: null, action: null, patient: null, instrument: null, operators: [], negated: false, goals: [] }
  return deriveEventGraphForBranch(branch, model, index, hints)
}

export function deriveEventGraphForBranch(
  branch: ThoughtBranch,
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  hints: ThoughtRoutingHints = {},
): EventGraphV1 {
  const path = new Set(branch.path)
  const hypotheses = hints.roleHypotheses ?? []
  let actor = roleValue('actor', hypotheses, path, index)
  let action = roleValue('action', hypotheses, path, index)
  let patient = roleValue('patient', hypotheses, path, index)
  const instrument = roleValue('instrument', hypotheses, path, index)
  const operators: string[] = []
  const goals: Array<{ relation: string; target: string }> = []
  const content: string[] = []
  for (const neuronId of branch.path) {
    const neuron = index.neuronById.get(neuronId)
    const text = neuron?.provenance.surface?.trim()
    if (!neuron || !text) continue
    const regions = neuron.familyMemberships.map((membership) => model.families.find((family) => family.id === membership.familyId)?.region)
    if (!action && regions.includes('event')) action = text
    if (regions.includes('content')) content.push(text)
    if (regions.includes('operator')) operators.push(text)
    if (regions.includes('goal')) goals.push({ relation: 'goal', target: text })
  }
  actor ??= content[0] ?? null
  patient ??= content.find((value) => canonicalValue(value) !== canonicalValue(actor ?? '')) ?? null
  const normalizedOperators = [...new Set(operators.map((value) => canonicalValue(value)).filter((value): value is string => Boolean(value)))].sort()
  const negated = normalizedOperators.some((value) => /(?:^|\s)(?:değil|yok|hayır|not|no|never)(?:\s|$)/iu.test(value))
  return normalizeEventGraph({ format: 'event-graph-v1', actor, action, patient, instrument, operators: normalizedOperators, negated, goals })
}

export function evaluateThoughtBranches(
  episode: ThoughtEpisodeState,
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  expected: EventGraphV1,
  config: ThoughtRuntimeConfig,
  hints: ThoughtRoutingHints = {},
): ThoughtBranchEvaluation[] {
  return episode.branches.map((branch) => {
    const eventGraph = deriveEventGraphForBranch(branch, model, index, hints)
    return { branchId: branch.id, eventGraph, evaluation: evaluateEventGraph(eventGraph, expected, config) }
  }).sort((left, right) => left.branchId.localeCompare(right.branchId))
}

export function routingHintsFromFamilyRun(familyRun: FamilyRun): ThoughtRoutingHints {
  const coalitionFamilyWeights: Record<string, number> = {}
  const roleHypotheses: ThoughtRoleHypothesis[] = []
  for (const coalition of familyRun.coalition?.coalitions ?? []) {
    coalitionFamilyWeights[coalition.eventFamilyId] = Math.max(coalitionFamilyWeights[coalition.eventFamilyId] ?? 0, coalition.dominance)
    for (const activation of coalition.familyActivations) {
      coalitionFamilyWeights[activation.familyId] = Math.max(coalitionFamilyWeights[activation.familyId] ?? 0, activation.activation)
    }
    for (const binding of coalition.bindings) {
      const role = binding.role === 'agent-hypothesis' ? 'actor'
        : binding.role === 'patient-hypothesis' ? 'patient'
          : binding.role === 'instrument-hypothesis' ? 'instrument'
            : binding.role === 'operator' ? 'operator' : null
      if (role) roleHypotheses.push({ role, memberKey: binding.memberKey, score: binding.score, temporary: true })
    }
    roleHypotheses.push({ role: 'action', memberKey: coalition.eventMemberKey, score: coalition.confidence, temporary: true })
  }
  return { coalitionFamilyWeights, roleHypotheses }
}

/** Recomputes a temporary coalition from current branch/path activation each tick. */
export function deriveLiveRoutingHints(
  episode: ThoughtEpisodeState,
  model: ThoughtModel,
  index: ThoughtRuntimeIndex,
  initial: ThoughtRoutingHints = {},
): ThoughtRoutingHints {
  const coalitionFamilyWeights: Record<string, number> = {}
  for (const [familyId, prior] of Object.entries(initial.coalitionFamilyWeights ?? {})) {
    coalitionFamilyWeights[familyId] = 0.25 * clamp(prior)
  }
  for (const branch of episode.branches) {
    const recent = branch.path.slice(-8)
    recent.forEach((neuronId, position) => {
      const neuron = index.neuronById.get(neuronId)
      if (!neuron) return
      const recency = (position + 1) / recent.length
      for (const membership of neuron.familyMemberships) {
        coalitionFamilyWeights[membership.familyId] = (coalitionFamilyWeights[membership.familyId] ?? 0) +
          branch.mass * recency * membership.weight
      }
    })
  }
  const roleHypotheses: ThoughtRoleHypothesis[] = []
  const dominant = [...episode.branches].sort((left, right) => right.mass - left.mass || left.id.localeCompare(right.id))[0]
  if (dominant) {
    const contentIds: string[] = []
    for (const neuronId of dominant.path) {
      const neuron = index.neuronById.get(neuronId)
      if (!neuron) continue
      const regions = neuron.familyMemberships.map((membership) => model.families.find((family) => family.id === membership.familyId)?.region)
      const score = dominant.mass
      if (regions.includes('event')) roleHypotheses.push({ role: 'action', memberKey: neuronId, score, temporary: true })
      if (regions.includes('content')) contentIds.push(neuronId)
      if (regions.includes('operator')) roleHypotheses.push({ role: 'operator', memberKey: neuronId, score, temporary: true })
      if (regions.includes('goal')) roleHypotheses.push({ role: 'goal', memberKey: neuronId, score, temporary: true })
    }
    if (contentIds[0]) roleHypotheses.push({ role: 'actor', memberKey: contentIds[0], score: dominant.mass, temporary: true })
    if (contentIds[1]) roleHypotheses.push({ role: 'patient', memberKey: contentIds[1], score: dominant.mass, temporary: true })
  }
  for (const hypothesis of initial.roleHypotheses ?? []) {
    if (!roleHypotheses.some((entry) => entry.role === hypothesis.role && entry.memberKey === hypothesis.memberKey)) {
      roleHypotheses.push({ ...hypothesis, score: hypothesis.score * 0.25 })
    }
  }
  return { coalitionFamilyWeights, goalFamilyWeights: initial.goalFamilyWeights, roleHypotheses }
}
