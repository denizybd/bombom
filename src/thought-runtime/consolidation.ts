import { replayFingerprint, thoughtDigest, thoughtPartitionDigest } from './deterministic'
import { clamp, composeNeuronRepresentation, cosineSimilarity, projectThoughtVector } from './math'
import type {
  BornNeuronV1,
  OutcomeEvidence,
  ThoughtBranchEvaluation,
  ThoughtBranchScore,
  ThoughtEpisodeState,
  ThoughtHead,
  ThoughtHeadDelta,
  ThoughtHeadDeltaContextPatch,
  ThoughtHeadDeltaEdge,
  ThoughtModel,
  ThoughtNurseryCandidate,
  ThoughtRuntimeConfig,
} from './types'

function learningEnabled(mode: string): boolean {
  return ['learn', 'confirmed', 'supervised', 'auto', 'commit', 'proposal']
    .includes(String(mode ?? '').toLocaleLowerCase('en-US'))
}

export function outcomeTrust(evidence: OutcomeEvidence): number {
  if (evidence.kind === 'human' || evidence.kind === 'deterministic') return 1
  if (evidence.kind === 'observed') return 0.6
  if (evidence.kind === 'ollama-cross-checked') return 0.25
  return 0
}

function mergeWeight(entries: Array<{ key: string; weight: number }>, key: string, weight: number): void {
  const current = entries.find((entry) => entry.key === key)
  if (current) current.weight += weight
  else entries.push({ key, weight })
  entries.sort((left, right) => right.weight - left.weight || left.key.localeCompare(right.key))
}

function consolidateEdges(
  episode: ThoughtEpisodeState,
  _structuralScores: readonly ThoughtBranchScore[],
  branchEvaluations: readonly ThoughtBranchEvaluation[],
  trust: number,
  config: ThoughtRuntimeConfig,
  persistentIds: ReadonlySet<string>,
): ThoughtHeadDeltaEdge[] {
  const evaluationByBranch = new Map(branchEvaluations.map((entry) => [entry.branchId, entry.evaluation]))
  const deltas = new Map<string, ThoughtHeadDeltaEdge>()
  const add = (from: string, to: string, excitation: number, inhibition: number, edgeTrust: number) => {
    const key = `${from}\u0000${to}`
    const current = deltas.get(key) ?? {
      fromMemberKey: from, toMemberKey: to, excitation: 0, inhibition: 0, trustMass: 0, updateCount: 0,
    }
    current.excitation = clamp(current.excitation + excitation, 0, config.evidenceCeiling)
    current.inhibition = clamp(current.inhibition + inhibition, 0, config.evidenceCeiling)
    current.trustMass = Math.min(trust, current.trustMass + edgeTrust)
    current.updateCount += 1
    deltas.set(key, current)
  }
  for (const branch of episode.branches) {
    const evaluation = evaluationByBranch.get(branch.id)
    if (!evaluation) continue
    const reward = 2 * evaluation.resultScore - 1
    const transitionKeys = new Set<string>()
    const count = branch.transitions.length
    branch.transitions.forEach((transition, index) => {
      if (!persistentIds.has(transition.fromNeuronId) || !persistentIds.has(transition.toNeuronId)) return
      const eligibility = config.fastEligibilityDecay ** (count - 1 - index)
      const eligibleTrust = trust * eligibility
      add(
        transition.fromNeuronId,
        transition.toNeuronId,
        config.fastExcitationRate * Math.max(reward, 0) * eligibleTrust,
        config.fastInhibitionRate * Math.max(-reward, 0) * eligibleTrust,
        eligibleTrust,
      )
      transitionKeys.add(`${transition.fromNeuronId}\u0000${transition.toNeuronId}`)
    })
    for (const trace of branch.fastTraces) {
      if (!persistentIds.has(trace.fromNeuronId) || !persistentIds.has(trace.toNeuronId)) continue
      if (transitionKeys.has(`${trace.fromNeuronId}\u0000${trace.toNeuronId}`)) continue
      const eligibleTrust = trust * trace.eligibility
      const positiveEdge = trace.excitation > trace.inhibition
      if (!positiveEdge) continue
      const excitation = config.fastExcitationRate * Math.max(reward, 0) * eligibleTrust
      const inhibition = config.fastInhibitionRate * Math.max(-reward, 0) * eligibleTrust
      if (excitation > 0 || inhibition > 0) add(trace.fromNeuronId, trace.toNeuronId, excitation, inhibition, eligibleTrust)
    }
  }
  return [...deltas.values()]
    .sort((left, right) => left.fromMemberKey.localeCompare(right.fromMemberKey) || left.toMemberKey.localeCompare(right.toMemberKey))
}

function consolidateContext(
  episode: ThoughtEpisodeState,
  _structuralScores: readonly ThoughtBranchScore[],
  branchEvaluations: readonly ThoughtBranchEvaluation[],
  trust: number,
  relationLength: number,
  persistentIds: ReadonlySet<string>,
): ThoughtHeadDeltaContextPatch[] {
  const evaluationByBranch = new Map(branchEvaluations.map((entry) => [entry.branchId, entry.evaluation]))
  const accumulators = new Map<string, { sum: Float64Array; trustMass: number; observations: number }>()
  for (const branch of episode.branches) {
    const evaluation = evaluationByBranch.get(branch.id)
    if (!evaluation) continue
    const signedWeight = (2 * evaluation.resultScore - 1) * trust * branch.mass
    if (Math.abs(signedWeight) <= Number.EPSILON) continue
    for (const patch of branch.fastRelationPatches) {
      if (!persistentIds.has(patch.neuronId)) continue
      const accumulator = accumulators.get(patch.neuronId) ?? {
        sum: new Float64Array(relationLength), trustMass: 0, observations: 0,
      }
      for (let index = 0; index < relationLength; index += 1) accumulator.sum[index] += patch.delta[index] * signedWeight
      accumulator.trustMass += trust * branch.mass
      accumulator.observations += 1
      accumulators.set(patch.neuronId, accumulator)
    }
  }
  return [...accumulators].sort(([left], [right]) => left.localeCompare(right)).map(([memberKey, value]) => ({
    memberKey,
    // Keep the absolute outcome credit in the durable R delta. Normalizing by
    // |reward * trust * branch.mass| here would make a weak/Ollama observation
    // mutate R as strongly as a deterministic human observation.
    relationDelta: Array.from(value.sum, (coordinate) => Math.fround(clamp(coordinate, -1, 1))),
    trustMass: value.trustMass,
    observationCount: value.observations,
  }))
}

function candidateRepresentation(candidate: ThoughtNurseryCandidate): Float32Array {
  const output = new Float32Array(candidate.identity.length + candidate.relationPrototype.length)
  output.set(candidate.identity)
  output.set(candidate.relationPrototype, candidate.identity.length)
  return output
}

function cloneNursery(candidate: ThoughtNurseryCandidate): ThoughtNurseryCandidate {
  return {
    ...candidate,
    identity: new Float32Array(candidate.identity),
    relationPrototype: new Float32Array(candidate.relationPrototype),
    birthEpisodeIds: [...candidate.birthEpisodeIds],
    independentSourceIds: [...candidate.independentSourceIds],
    originEvidence: candidate.originEvidence.map((entry) => ({ ...entry })),
    ...(candidate.outgoingEvidence
      ? { outgoingEvidence: candidate.outgoingEvidence.map((entry) => ({ ...entry })) }
      : {}),
    familyEvidence: candidate.familyEvidence.map((entry) => ({ ...entry })),
    sourceRelationPrototypes: (candidate.sourceRelationPrototypes ?? []).map((entry) => ({
      ...entry,
      relation: new Float32Array(entry.relation),
    })),
  }
}

function temporaryTraversalEvidence(
  episode: ThoughtEpisodeState,
  neuronId: string,
  baseRelation: Float32Array,
  provenanceOrigins: readonly { key: string; weight: number }[],
  branchEvaluations: readonly ThoughtBranchEvaluation[],
  resultScore: number,
  trust: number,
): {
  originEvidence: Array<{ key: string; weight: number }>
  outgoingEvidence: Array<{ key: string; weight: number }>
  relation: Float32Array
} {
  const evaluationByBranch = new Map(branchEvaluations.map((entry) => [entry.branchId, entry.evaluation.resultScore]))
  const origins: Array<{ key: string; weight: number }> = []
  const outgoing: Array<{ key: string; weight: number }> = []
  const relationSum = new Float64Array(baseRelation.length)
  let relationWeight = 0
  let observedEntry = false

  for (const branch of episode.branches) {
    const branchScore = evaluationByBranch.get(branch.id)
    if (branchScore === undefined) continue
    const weight = Math.max(0, branch.mass * branchScore * trust)
    if (weight <= Number.EPSILON) continue
    for (const transition of branch.transitions) {
      if (transition.toNeuronId === neuronId && transition.fromNeuronId !== neuronId) {
        mergeWeight(origins, transition.fromNeuronId, weight)
        observedEntry = true
      }
      if (transition.fromNeuronId === neuronId && transition.toNeuronId !== neuronId) {
        mergeWeight(outgoing, transition.toNeuronId, weight)
      }
    }
    if (!branch.path.includes(neuronId)) continue
    const patch = branch.fastRelationPatches.find((entry) => entry.neuronId === neuronId)
    for (let index = 0; index < baseRelation.length; index += 1) {
      relationSum[index] += clamp(baseRelation[index] + (patch?.delta[index] ?? 0), -1, 1) * weight
    }
    relationWeight += weight
  }

  // A newborn may not be selected again before the episode ends. Its residual
  // origins are still exact contributors and are preferable to inventing an
  // origin from the dominant/final branch.
  if (!observedEntry) {
    const scale = Math.max(Number.EPSILON, resultScore * trust)
    for (const entry of provenanceOrigins) mergeWeight(origins, entry.key, entry.weight * scale)
  }
  const relation = new Float32Array(baseRelation.length)
  if (relationWeight > Number.EPSILON) {
    for (let index = 0; index < relation.length; index += 1) {
      relation[index] = Math.fround(clamp(relationSum[index] / relationWeight, -1, 1))
    }
  } else relation.set(baseRelation)
  return { originEvidence: origins, outgoingEvidence: outgoing, relation }
}

function recomputeSourceBalancedRelation(candidate: ThoughtNurseryCandidate): void {
  if (!candidate.sourceRelationPrototypes.length) return
  const mean = new Float32Array(candidate.relationPrototype.length)
  for (const source of candidate.sourceRelationPrototypes) {
    for (let index = 0; index < mean.length; index += 1) mean[index] += source.relation[index] / candidate.sourceRelationPrototypes.length
  }
  candidate.relationPrototype = projectThoughtVector(mean, 1)
}

function canonicalProvisionalFamily(identityDigest: string): string {
  return `nursery-provisional:${thoughtDigest('nursery-family', identityDigest).slice(0, 16)}`
}

function upsertNursery(
  episode: ThoughtEpisodeState,
  model: ThoughtModel,
  head: ThoughtHead,
  evidence: OutcomeEvidence,
  resultScore: number,
  trust: number,
  branchEvaluations: readonly ThoughtBranchEvaluation[],
  config: ThoughtRuntimeConfig,
): { upserts: ThoughtNurseryCandidate[]; removeIds: string[]; births: BornNeuronV1[]; retained: number; promoted: number } {
  const working = head.nursery.map(cloneNursery)
  const touched = new Set<string>()
  const globalTick = head.totalTicks + episode.tick
  const durableNeuronIds = new Set([
    ...model.neurons.map((neuron) => neuron.id),
    ...head.births.map((birth) => birth.id),
  ])
  for (const neuron of episode.temporaryBirths.slice(0, config.maximumBirthsPerEpisode)) {
    const traversal = temporaryTraversalEvidence(
      episode,
      neuron.id,
      neuron.relation,
      neuron.provenance.originEvidence ?? [],
      branchEvaluations,
      resultScore,
      trust,
    )
    traversal.originEvidence = traversal.originEvidence.filter((entry) => durableNeuronIds.has(entry.key))
    traversal.outgoingEvidence = traversal.outgoingEvidence.filter((entry) => durableNeuronIds.has(entry.key))
    const representation = composeNeuronRepresentation(neuron.identity, traversal.relation, model.layout)
    const match = working.map((entry) => ({ entry, similarity: cosineSimilarity(representation, candidateRepresentation(entry)) }))
      .sort((left, right) => right.similarity - left.similarity || left.entry.id.localeCompare(right.entry.id))[0]
    const residualCohesion = neuron.provenance.residualCohesion ?? 0
    if (!match || match.similarity < config.nurseryCohesionThreshold) {
      const id = `nursery-${thoughtDigest('nursery', neuron.identityDigest).slice(0, 24)}`
      const provisionalFamilyId = canonicalProvisionalFamily(neuron.identityDigest)
      const created: ThoughtNurseryCandidate = {
        id,
        identity: new Float32Array(neuron.identity),
        relationPrototype: new Float32Array(traversal.relation),
        identityDigest: neuron.identityDigest,
        birthEpisodeIds: [episode.runKey],
        independentSourceIds: [evidence.sourceGroupId],
        observationCount: 1,
        residualObservationCount: Math.max(
          config.minimumResidualObservations,
          neuron.provenance.residualObservationCount ?? config.minimumResidualObservations,
        ),
        supportWeight: resultScore,
        trustMass: trust,
        resultScoreTotal: resultScore,
        originEvidence: traversal.originEvidence,
        ...(traversal.outgoingEvidence.length ? { outgoingEvidence: traversal.outgoingEvidence } : {}),
        familyEvidence: (neuron.provenance.familyEvidence ?? neuron.familyMemberships.map((membership) => ({
          key: membership.familyId,
          weight: membership.weight,
        }))).map((membership) => ({
          key: membership.key.startsWith('nursery-provisional:') ? provisionalFamilyId : membership.key,
          weight: membership.weight * resultScore,
        })),
        nextPromotionTick: globalTick + config.promotionCooldownTicks,
        residualCohesion,
        sourceRelationPrototypes: [{ sourceGroupId: evidence.sourceGroupId, relation: new Float32Array(traversal.relation), observationCount: 1 }],
      }
      working.push(created)
      touched.add(created.id)
      continue
    }
    const current = match.entry
    if (thoughtPartitionDigest('identity', current.identity) !== current.identityDigest) {
      throw new TypeError(`Nursery I kimliği değişmiş: ${current.id}`)
    }
    // Replaying the same semantic run is idempotent evidence, not a new
    // observation/source/trust contribution.
    if (current.birthEpisodeIds.includes(episode.runKey)) continue
    let source = current.sourceRelationPrototypes.find((entry) => entry.sourceGroupId === evidence.sourceGroupId)
    if (!source) {
      source = { sourceGroupId: evidence.sourceGroupId, relation: new Float32Array(traversal.relation), observationCount: 1 }
      current.sourceRelationPrototypes.push(source)
    } else {
      const nextCount = source.observationCount + 1
      for (let index = 0; index < source.relation.length; index += 1) {
        source.relation[index] = Math.fround((source.relation[index] * source.observationCount + traversal.relation[index]) / nextCount)
      }
      source.observationCount = nextCount
    }
    recomputeSourceBalancedRelation(current)
    current.birthEpisodeIds.push(episode.runKey)
    if (!current.independentSourceIds.includes(evidence.sourceGroupId)) current.independentSourceIds.push(evidence.sourceGroupId)
    current.observationCount += 1
    current.residualObservationCount += Math.max(
      config.minimumResidualObservations,
      neuron.provenance.residualObservationCount ?? config.minimumResidualObservations,
    )
    current.supportWeight += resultScore
    current.trustMass += trust
    current.resultScoreTotal += resultScore
    current.residualCohesion = (current.residualCohesion * (current.observationCount - 1) + residualCohesion) / current.observationCount
    for (const origin of traversal.originEvidence) mergeWeight(current.originEvidence, origin.key, origin.weight)
    for (const target of traversal.outgoingEvidence) {
      current.outgoingEvidence ??= []
      mergeWeight(current.outgoingEvidence, target.key, target.weight)
    }
    const stableProvisionalFamily = current.familyEvidence.find((entry) => entry.key.startsWith('nursery-provisional:'))?.key ??
      canonicalProvisionalFamily(current.identityDigest)
    const observedFamilyEvidence = neuron.provenance.familyEvidence ?? neuron.familyMemberships.map((membership) => ({
      key: membership.familyId,
      weight: membership.weight,
    }))
    for (const membership of observedFamilyEvidence) {
      mergeWeight(
        current.familyEvidence,
        membership.key.startsWith('nursery-provisional:') ? stableProvisionalFamily : membership.key,
        membership.weight * resultScore,
      )
    }
    touched.add(current.id)
  }
  const removeIds: string[] = []
  const births: BornNeuronV1[] = []
  for (const candidate of working.filter((entry) => touched.has(entry.id))) {
    const mature = candidate.observationCount >= config.nurseryMinimumObservations &&
      candidate.independentSourceIds.length >= config.nurseryMinimumIndependentSources &&
      candidate.trustMass >= config.nurseryMinimumTrustMass &&
      candidate.residualCohesion >= config.nurseryCohesionThreshold &&
      globalTick >= candidate.nextPromotionTick
    if (!mature) continue
    const familyEvidence = candidate.familyEvidence.slice(0, config.maximumFamilyMembershipsPerNeuron)
    const familyTotal = familyEvidence.reduce((sum, entry) => sum + entry.weight, 0)
    const id = `born-${thoughtDigest('promotion', candidate.id, candidate.identityDigest, candidate.independentSourceIds).slice(0, 24)}`
    births.push({
      format: 'born-neuron-v1',
      id,
      identity: Array.from(candidate.identity),
      relation: Array.from(candidate.relationPrototype),
      identityDigest: candidate.identityDigest,
      familyMemberships: familyEvidence.map((entry) => ({ familyId: entry.key, weight: entry.weight / Math.max(Number.EPSILON, familyTotal) })),
      sourceGroupIds: [...candidate.independentSourceIds].sort(),
      trustMass: candidate.trustMass,
      bornAtTick: globalTick,
      nurseryId: candidate.id,
      ...(candidate.originEvidence.length ? {
        entryEvidence: candidate.originEvidence.map((entry) => ({ memberKey: entry.key, weight: entry.weight })),
      } : {}),
      ...(candidate.outgoingEvidence?.length ? {
        outgoingEvidence: candidate.outgoingEvidence.map((entry) => ({ memberKey: entry.key, weight: entry.weight })),
      } : {}),
      contextEvidence: {
        observationCount: candidate.sourceRelationPrototypes.reduce((sum, source) => sum + source.observationCount, 0),
        sourceGroupIds: candidate.sourceRelationPrototypes.map((source) => source.sourceGroupId).sort(),
      },
    })
    removeIds.push(candidate.id)
  }
  // A promoted candidate is still included in the delta as the exact nursery
  // proof consumed by the birth gate. applyHeadDelta materializes this upsert
  // and then removes it atomically, so it never survives beside the birth.
  const upserts = working.filter((candidate) => touched.has(candidate.id))
  return {
    upserts,
    removeIds,
    births,
    retained: upserts.length - removeIds.length,
    promoted: births.length,
  }
}

/** No expected graph or no independently supplied evidence means no persistent credit. */
export function consolidateThoughtEpisode(input: {
  episode: ThoughtEpisodeState
  model: ThoughtModel
  head: ThoughtHead
  branchScores: readonly ThoughtBranchScore[]
  branchEvaluations: readonly ThoughtBranchEvaluation[]
  evidence: OutcomeEvidence | null
  learningMode: string
  config: ThoughtRuntimeConfig
}): ThoughtHeadDelta | undefined {
  const { episode, model, head, branchScores, branchEvaluations, evidence, learningMode, config } = input
  if (!branchEvaluations.length || !evidence || !learningEnabled(learningMode)) return undefined
  if (!Number.isFinite(evidence.confidence) || evidence.confidence < 0 || evidence.confidence > 1 || !evidence.sourceGroupId.trim()) {
    throw new RangeError('Outcome evidence confidence 0..1 ve sourceGroupId dolu olmalıdır.')
  }
  if (evidence.kind === 'ollama-cross-checked' && evidence.confidence < 0.85) {
    throw new RangeError('Ollama cross-check etiketi için confidence en az 0.85 olmalıdır.')
  }
  const trust = outcomeTrust(evidence)
  if (trust <= 0) return undefined
  const persistentIds = new Set([...model.neurons.map((neuron) => neuron.id), ...head.births.map((birth) => birth.id)])
  const neuronEdges = consolidateEdges(episode, branchScores, branchEvaluations, trust, config, persistentIds)
  const contextPatches = consolidateContext(episode, branchScores, branchEvaluations, trust, model.layout.relationLength, persistentIds)
  const massByBranch = new Map(episode.branches.map((branch) => [branch.id, branch.mass]))
  const totalMass = branchEvaluations.reduce((sum, entry) => sum + (massByBranch.get(entry.branchId) ?? 0), 0)
  const outcomeScore = branchEvaluations.reduce((sum, entry) => sum + entry.evaluation.resultScore * (massByBranch.get(entry.branchId) ?? 0), 0) /
    Math.max(Number.EPSILON, totalMass)
  const graphF1 = branchEvaluations.reduce((sum, entry) => sum + entry.evaluation.graphF1 * (massByBranch.get(entry.branchId) ?? 0), 0) /
    Math.max(Number.EPSILON, totalMass)
  const constraintAccuracy = branchEvaluations.reduce((sum, entry) => sum + entry.evaluation.constraintAccuracy * (massByBranch.get(entry.branchId) ?? 0), 0) /
    Math.max(Number.EPSILON, totalMass)
  const nurseryAllowed = outcomeScore >= config.minimumOutcomeScore && trust > 0
  const nursery = nurseryAllowed
    ? upsertNursery(episode, model, head, evidence, outcomeScore, trust, branchEvaluations, config)
    : { upserts: [], removeIds: [], births: [], retained: 0, promoted: 0 }
  const credit = Math.abs(2 * outcomeScore - 1) * trust
  return {
    neuronEdges,
    ...(contextPatches.length ? { contextPatches } : {}),
    ...(nursery.upserts.length ? { nurseryUpserts: nursery.upserts } : {}),
    ...(nursery.removeIds.length ? { nurseryRemoveIds: nursery.removeIds } : {}),
    ...(nursery.births.length ? { births: nursery.births } : {}),
    stats: {
      baseHeadRevision: head.revision,
      baseHeadFingerprint: head.fingerprint,
      episodeId: episode.episodeId,
      replayFingerprint: replayFingerprint(episode.replay),
      outcomeScore,
      graphF1,
      constraintAccuracy,
      confidence: evidence.confidence,
      credit,
      totalTicks: episode.tick,
      nurseryRetained: nursery.retained,
      nurseryRejected: nurseryAllowed ? 0 : episode.temporaryBirths.length,
      nurseryPromoted: nursery.promoted,
    },
  }
}
