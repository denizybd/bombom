import { computeMacroConfigFingerprint, createMacroConfig, validateMacroConfig } from './config.ts'
import {
  canonicalExpansion,
  canonicalVector,
  compareMacroText,
  computeExactSignature,
  computeExpansionDigest,
  computeFamilySignature,
  computeMdlMetrics,
  createSignalSignature,
  macroDigest,
  meanVectors,
  subtractVectors,
  validateSignalVector,
  vectorNorm,
} from './deterministic.ts'
import { createMacroReplayBloomQuery } from './replay.ts'
import { aggregateMacroPrototypeGroups, createMacroPrototypeGroup } from './packed.ts'
import type {
  CompositeNeuronKind,
  MacroConfigV1,
  MacroEvidenceReferenceV1,
  MacroExpansionVariantV1,
  MacroMiningResultV1,
  MacroMotifCandidateV1,
  MacroObservationV1,
  MacroPrototypeGroupV1,
  MacroReplayGuardV1,
  MacroTraceStepV1,
  MacroTraceV1,
} from './types.ts'
import { CONCEPT_MACRO_EXECUTION_SEMANTICS } from './types.ts'

interface MutableVariant {
  exactSignature: string
  orderedExpansion: MacroExpansionVariantV1['orderedExpansion']
  expansionDigest: string
  observationCount: number
}

interface MutableCandidate {
  key: string
  kind: CompositeNeuronKind
  signature: string
  length: number
  estimatedOccurrences: number
  maximumEstimationError: number
  observationCount: number
  familySignature: string
  evidence: MacroEvidenceReferenceV1[]
  variants: Map<string, MutableVariant>
  prototypeGroups: Map<string, MutablePrototypeGroup>
}

interface MutablePrototypeGroup {
  independenceGroupId: string
  observationCount: number
  nonZeroEffectObservationCount: number
  entryMean: Float32Array
  exitMean: Float32Array
  trajectoryMean: Float32Array
  effectMean: Float32Array
  effectDirectionMean: Float32Array
}

interface MiningCounters {
  generatedWindowCount: number
  acceptedObservationCount: number
  overlappingObservationCount: number
  heavyHitterReplacementCount: number
}

/** Five Float32 accumulators per retained source-group; default payload budget is 16 MiB. */
export const MACRO_MINER_PROTOTYPE_CELL_BUDGET = 4_194_304
export const MACRO_DENSE_ARRAYS_PER_PROTOTYPE_GROUP = 5

export function effectiveMacroHeavyHitterCapacity(
  config: MacroConfigV1,
  vectorDimension: number,
): number {
  if (!Number.isInteger(vectorDimension) || vectorDimension < 1 ||
    vectorDimension > config.maximumVectorDimension) {
    throw new RangeError('Heavy-hitter vektör boyutu yapılandırılmış uzaya uymuyor.')
  }
  const accumulatorCellsPerCandidate = vectorDimension *
    MACRO_DENSE_ARRAYS_PER_PROTOTYPE_GROUP * config.maximumPrototypeGroupsPerCandidate
  return Math.max(1, Math.min(
    config.maximumHeavyHitters,
    Math.floor(MACRO_MINER_PROTOTYPE_CELL_BUDGET / accumulatorCellsPerCandidate),
  ))
}

export function computeMacroTraceReplayKey(
  trace: Pick<MacroTraceV1, 'episodeId' | 'runKey'>,
): string {
  return `${trace.runKey}\u0000${trace.episodeId}`
}

export function computeMacroSemanticReplayKey(
  trace: Pick<
    MacroTraceV1,
    'sourceDigest' | 'sourceId' | 'semanticInputDigest'
  >,
): string {
  return macroDigest(
    'MSEMTRACE',
    trace.sourceDigest ?? trace.sourceId,
    trace.semanticInputDigest,
  )
}

function traceFingerprint(trace: MacroTraceV1): string {
  return macroDigest('MTRACE', {
    id: trace.id,
    representationSpaceId: trace.representationSpaceId,
    familyRunFingerprint: trace.familyRunFingerprint,
    sourceId: trace.sourceId,
    sourceDigest: trace.sourceDigest ?? null,
    independenceGroupId: trace.independenceGroupId,
    episodeId: trace.episodeId,
    runKey: trace.runKey,
    semanticInputDigest: trace.semanticInputDigest,
    split: trace.split,
    steps: trace.steps.map((step) => ({
      member: canonicalExpansion([step.member])[0],
      signalBefore: canonicalVector(step.signalBefore),
      signalAfter: canonicalVector(step.signalAfter),
    })),
  })
}

export function validateMacroTrace(trace: MacroTraceV1, config: MacroConfigV1): void {
  if (trace.format !== 'macro-trace-v1' || trace.version !== 1) {
    throw new TypeError('Yalnız MacroTraceV1 izleri işlenebilir.')
  }
  const required = [
    trace.id,
    trace.representationSpaceId,
    trace.sourceId,
    trace.independenceGroupId,
    trace.episodeId,
    trace.runKey,
    trace.semanticInputDigest,
  ]
  if (required.some((value) => !value.trim())) throw new TypeError('Makro iz kanıt kimlikleri boş olamaz.')
  if (trace.split !== 'train' && trace.split !== 'validation' && trace.split !== 'test') {
    throw new TypeError('Makro iz veri bölümü tanınmıyor.')
  }
  let dimension: number | null = null
  for (const step of trace.steps) {
    canonicalExpansion([step.member])
    validateSignalVector(step.signalBefore, config.maximumVectorDimension, 'Giriş sinyali')
    validateSignalVector(step.signalAfter, config.maximumVectorDimension, 'Çıkış sinyali')
    if (step.signalBefore.length !== step.signalAfter.length) {
      throw new RangeError('Bir adımın giriş ve çıkış sinyal boyutları aynı olmalıdır.')
    }
    dimension ??= step.signalBefore.length
    if (step.signalBefore.length !== dimension) {
      throw new RangeError('Aynı izdeki bütün sinyaller aynı boyutta olmalıdır.')
    }
  }
}

function createObservation(
  trace: MacroTraceV1,
  steps: readonly MacroTraceStepV1[],
  start: number,
  kind: CompositeNeuronKind,
): MacroObservationV1 {
  const orderedExpansion = canonicalExpansion(steps.map((step) => step.member))
  const exactSignature = computeExactSignature(orderedExpansion)
  const familySignature = computeFamilySignature(orderedExpansion)
  const expansionDigest = computeExpansionDigest(orderedExpansion)
  const entry = canonicalVector(steps[0].signalBefore)
  // A path [a,b,c] compiles the transitions a→b→c. Its output is the
  // signal immediately before c is processed, so the runtime can leave c as
  // the current terminal anchor without applying c twice on the next tick.
  const executableSteps = steps.slice(0, -1)
  const exit = canonicalVector(steps[steps.length - 1].signalBefore)
  const trajectory = meanVectors(
    executableSteps.map((step) => subtractVectors(step.signalAfter, step.signalBefore)),
  )
  const effect = subtractVectors(exit, entry)
  const end = start + steps.length
  const signature = kind === 'sequence-macro' ? exactSignature : familySignature
  const id = macroDigest('MOBS', kind, computeMacroTraceReplayKey(trace), start, end, signature)
  const semanticOccurrenceKey = macroDigest(
    'MSEMOCC',
    kind,
    signature,
    computeMacroSemanticReplayKey(trace),
    start,
    end,
  )
  const evidence: MacroEvidenceReferenceV1 = {
    observationId: id,
    sourceId: trace.sourceId,
    sourceDigest: trace.sourceDigest ?? null,
    independenceGroupId: trace.independenceGroupId,
    episodeId: trace.episodeId,
    runKey: trace.runKey,
    semanticInputDigest: trace.semanticInputDigest,
    semanticOccurrenceKey,
    split: trace.split,
    tickStart: start,
    tickEnd: end,
  }
  const base: Omit<MacroObservationV1, 'fingerprint'> = {
    format: 'macro-observation-v1',
    version: 1,
    id,
    kind,
    executionSemantics: CONCEPT_MACRO_EXECUTION_SEMANTICS,
    exactSignature,
    familySignature,
    orderedExpansion,
    expansionDigest,
    entrySignature: createSignalSignature(entry),
    exitSignature: createSignalSignature(exit),
    trajectory,
    effect,
    evidence,
  }
  return { ...base, fingerprint: macroDigest('MOBSFP', base) }
}

function retainDiverseEvidence(
  evidence: readonly MacroEvidenceReferenceV1[],
  maximum: number,
): MacroEvidenceReferenceV1[] {
  const unique = new Map<string, MacroEvidenceReferenceV1>()
  for (const value of evidence) unique.set(value.observationId, value)
  const sorted = [...unique.values()].sort((left, right) =>
    compareMacroText(left.independenceGroupId, right.independenceGroupId) ||
    compareMacroText(left.observationId, right.observationId),
  )
  const firstByGroup: MacroEvidenceReferenceV1[] = []
  const remainder: MacroEvidenceReferenceV1[] = []
  const groups = new Set<string>()
  for (const value of sorted) {
    if (!groups.has(value.independenceGroupId)) {
      groups.add(value.independenceGroupId)
      firstByGroup.push(value)
    } else remainder.push(value)
  }
  const retained = firstByGroup.slice(0, maximum)
  if (retained.length < maximum) {
    remainder.sort((left, right) => compareMacroText(left.observationId, right.observationId))
    retained.push(...remainder.slice(0, maximum - retained.length))
  }
  return retained.sort((left, right) => compareMacroText(left.observationId, right.observationId))
}

function updateOnlinePrototype(
  target: Float32Array,
  values: readonly number[],
  previousGroupCount: number,
): void {
  const nextGroupCount = previousGroupCount + 1
  for (let index = 0; index < target.length; index += 1) {
    target[index] = Math.fround(target[index] + (values[index] - target[index]) / nextGroupCount)
  }
}

/**
 * Every observation updates its retained independence-group mean. Finalization
 * gives each retained group one equal vote, regardless of its observation count.
 */
function updateCandidatePrototypes(
  candidate: MutableCandidate,
  observation: MacroObservationV1,
  maximumGroups: number,
): void {
  const groupId = observation.evidence.independenceGroupId
  let group = candidate.prototypeGroups.get(groupId)
  if (!group) {
    if (candidate.prototypeGroups.size >= maximumGroups) return
    const dimension = observation.entrySignature.dimension
    group = {
      independenceGroupId: groupId,
      observationCount: 0,
      nonZeroEffectObservationCount: 0,
      entryMean: new Float32Array(dimension),
      exitMean: new Float32Array(dimension),
      trajectoryMean: new Float32Array(dimension),
      effectMean: new Float32Array(dimension),
      effectDirectionMean: new Float32Array(dimension),
    }
    candidate.prototypeGroups.set(groupId, group)
  }
  updateOnlinePrototype(group.entryMean, observation.entrySignature.vector, group.observationCount)
  updateOnlinePrototype(group.exitMean, observation.exitSignature.vector, group.observationCount)
  updateOnlinePrototype(group.trajectoryMean, observation.trajectory, group.observationCount)
  updateOnlinePrototype(group.effectMean, observation.effect, group.observationCount)
  const norm = vectorNorm(observation.effect)
  if (norm > 1e-12) {
    updateOnlinePrototype(
      group.effectDirectionMean,
      observation.effect.map((value) => value / norm),
      group.nonZeroEffectObservationCount,
    )
    group.nonZeroEffectObservationCount += 1
  }
  group.observationCount += 1
}

function weakestCandidate(candidates: ReadonlyMap<string, MutableCandidate>): MutableCandidate {
  return [...candidates.values()].sort((left, right) =>
    left.estimatedOccurrences - right.estimatedOccurrences ||
    left.observationCount - right.observationCount ||
    // On an exact tie evict the lexically larger key, preserving canonical small keys.
    compareMacroText(right.key, left.key),
  )[0]
}

function updateVariant(
  candidate: MutableCandidate,
  observation: MacroObservationV1,
  config: MacroConfigV1,
): void {
  const existing = candidate.variants.get(observation.exactSignature)
  if (existing) {
    existing.observationCount += 1
    return
  }
  if (candidate.variants.size >= config.maximumVariantsPerCandidate) {
    const weakest = [...candidate.variants.values()].sort((left, right) =>
      left.observationCount - right.observationCount ||
      compareMacroText(right.exactSignature, left.exactSignature),
    )[0]
    candidate.variants.delete(weakest.exactSignature)
  }
  candidate.variants.set(observation.exactSignature, {
    exactSignature: observation.exactSignature,
    orderedExpansion: observation.orderedExpansion,
    expansionDigest: observation.expansionDigest,
    observationCount: 1,
  })
}

function addObservation(
  candidates: Map<string, MutableCandidate>,
  observation: MacroObservationV1,
  config: MacroConfigV1,
  effectiveCapacity: number,
  counters: MiningCounters,
): void {
  const signature = observation.kind === 'sequence-macro'
    ? observation.exactSignature
    : observation.familySignature
  const key = `${observation.kind}\u0000${signature}`
  let candidate = candidates.get(key)
  if (!candidate) {
    let inheritedCount = 0
    if (candidates.size >= effectiveCapacity) {
      const weakest = weakestCandidate(candidates)
      inheritedCount = weakest.estimatedOccurrences
      candidates.delete(weakest.key)
      counters.heavyHitterReplacementCount += 1
    }
    candidate = {
      key,
      kind: observation.kind,
      signature,
      length: observation.orderedExpansion.length,
      estimatedOccurrences: inheritedCount + 1,
      maximumEstimationError: inheritedCount,
      observationCount: 0,
      familySignature: observation.familySignature,
      evidence: [],
      variants: new Map(),
      prototypeGroups: new Map(),
    }
    candidates.set(key, candidate)
  } else candidate.estimatedOccurrences += 1

  candidate.observationCount += 1
  if (compareMacroText(observation.familySignature, candidate.familySignature) < 0) {
    candidate.familySignature = observation.familySignature
  }
  candidate.evidence = retainDiverseEvidence(
    [...candidate.evidence, observation.evidence],
    config.maximumEvidencePerCandidate,
  )
  updateCandidatePrototypes(candidate, observation, config.maximumPrototypeGroupsPerCandidate)
  updateVariant(candidate, observation, config)
  counters.acceptedObservationCount += 1
}

function uniqueCount(values: readonly string[]): number {
  return new Set(values).size
}

function candidateEligible(
  candidate: Pick<
    MacroMotifCandidateV1,
    'kind' | 'observationCount' | 'independentGroupCount' | 'variants' | 'mdl' | 'effectCohesion'
  >,
  config: MacroConfigV1,
): boolean {
  return candidate.observationCount >= config.minimumProposalObservations &&
    candidate.independentGroupCount >= config.minimumIndependentGroups &&
    candidate.mdl.compressionGain >= config.minimumCompressionGain &&
    candidate.effectCohesion >= config.minimumEffectCohesion &&
    (candidate.kind !== 'equivalence-concept' ||
      candidate.variants.length >= config.minimumEquivalenceVariants)
}

export function meetsMacroProposalThresholds(
  candidate: Pick<
    MacroMotifCandidateV1,
    'kind' | 'observationCount' | 'independentGroupCount' | 'variants' | 'mdl' | 'effectCohesion'
  >,
  config: MacroConfigV1,
): boolean {
  validateMacroConfig(config)
  return candidateEligible(candidate, config)
}

function finalizeCandidate(
  candidate: MutableCandidate,
  config: MacroConfigV1,
): MacroMotifCandidateV1 {
  const evidence = [...candidate.evidence].sort((left, right) =>
    compareMacroText(left.observationId, right.observationId))
  const variants: MacroExpansionVariantV1[] = [...candidate.variants.values()]
    .map((variant) => ({
      exactSignature: variant.exactSignature,
      orderedExpansion: canonicalExpansion(variant.orderedExpansion),
      expansionDigest: variant.expansionDigest,
      observationCount: variant.observationCount,
    }))
    .sort((left, right) =>
      right.observationCount - left.observationCount ||
      compareMacroText(left.exactSignature, right.exactSignature),
    )
  const representative = variants[0]
  const mutableGroups = [...candidate.prototypeGroups.values()].sort((left, right) =>
    compareMacroText(left.independenceGroupId, right.independenceGroupId))
  if (!representative || !evidence.length || !mutableGroups.length) {
    throw new Error('Boş bir ağır-vurucu makro adayı sonuçlandırılamaz.')
  }
  const prototypeGroups: MacroPrototypeGroupV1[] = mutableGroups.map((group) =>
    createMacroPrototypeGroup({
      independenceGroupId: group.independenceGroupId,
      observationCount: group.observationCount,
      nonZeroEffectObservationCount: group.nonZeroEffectObservationCount,
      entryMean: group.entryMean,
      exitMean: group.exitMean,
      trajectoryMean: group.trajectoryMean,
      effectMean: group.effectMean,
      effectDirectionMean: group.effectDirectionMean,
    }))
  const prototype = aggregateMacroPrototypeGroups(prototypeGroups)
  const mdl = computeMdlMetrics(
    candidate.observationCount,
    candidate.length,
    config.macroDefinitionOverhead,
  )
  const preliminary = {
    kind: candidate.kind,
    observationCount: candidate.observationCount,
    independentGroupCount: uniqueCount(evidence.map((item) => item.independenceGroupId)),
    variants,
    mdl,
    effectCohesion: prototype.effectCohesion,
  }
  const id = macroDigest('MCAND', candidate.kind, candidate.signature)
  const base: Omit<MacroMotifCandidateV1, 'fingerprint'> = {
    format: 'macro-motif-candidate-v1',
    version: 1,
    id,
    kind: candidate.kind,
    executionSemantics: CONCEPT_MACRO_EXECUTION_SEMANTICS,
    lifecycle: candidateEligible(preliminary, config) ? 'eligible-proposal' : 'observing',
    exactSignature: candidate.kind === 'sequence-macro' ? candidate.signature : null,
    familySignature: candidate.kind === 'equivalence-concept'
      ? candidate.signature
      : candidate.familySignature,
    length: candidate.length,
    representativeExpansion: representative.orderedExpansion,
    expansionDigest: representative.expansionDigest,
    variants,
    entryPrototype: createSignalSignature(prototype.entry),
    exitPrototype: createSignalSignature(prototype.exit),
    trajectoryPrototype: prototype.trajectory,
    effectPrototype: prototype.effect,
    effectCohesion: preliminary.effectCohesion,
    prototypeGroups,
    prototypeGroupCount: prototypeGroups.length,
    prototypeObservationCount: prototype.observationCount,
    evidence,
    observationCount: candidate.observationCount,
    retainedEvidenceCount: evidence.length,
    estimatedOccurrences: candidate.estimatedOccurrences,
    maximumEstimationError: candidate.maximumEstimationError,
    independentGroupCount: preliminary.independentGroupCount,
    sourceCount: uniqueCount(evidence.map((item) => item.sourceId)),
    episodeCount: uniqueCount(evidence.map((item) => item.episodeId)),
    runKeyCount: uniqueCount(evidence.map((item) => item.runKey)),
    mdl,
  }
  return { ...base, fingerprint: macroDigest('MCANDFP', base) }
}

/**
 * Deterministic bounded Space-Saving motif miner.
 *
 * Traces are sorted before streaming, duplicate episode/run pairs are rejected,
 * and overlapping occurrences of the same signature in one trace are counted once.
 */
export function mineMacroMotifs(
  traces: readonly MacroTraceV1[],
  configInput: MacroConfigV1 = createMacroConfig(),
  alreadyProcessedTraceKeys: readonly string[] = [],
  alreadyProcessedSemanticTraceKeys: readonly string[] = [],
  replayGuard?: MacroReplayGuardV1,
): MacroMiningResultV1 {
  const config = createMacroConfig(configInput)
  const configFingerprint = computeMacroConfigFingerprint(config)
  const alreadyProcessed = new Set(alreadyProcessedTraceKeys)
  const alreadyProcessedSemantic = new Set(alreadyProcessedSemanticTraceKeys)
  const traceBloomContains = replayGuard
    ? createMacroReplayBloomQuery(replayGuard.traceBloom)
    : () => false
  const semanticBloomContains = replayGuard
    ? createMacroReplayBloomQuery(replayGuard.semanticBloom)
    : () => false
  const candidates = new Map<string, MutableCandidate>()
  const counters: MiningCounters = {
    generatedWindowCount: 0,
    acceptedObservationCount: 0,
    overlappingObservationCount: 0,
    heavyHitterReplacementCount: 0,
  }
  let duplicateTraceCount = 0
  const processedKeys: string[] = []
  const processedSemanticKeys: string[] = []
  const acceptedTraces: MacroTraceV1[] = []
  const seen = new Map<string, string>()
  const seenSemantic = new Set<string>()

  const sortedTraces = traces.map((trace) => {
    validateMacroTrace(trace, config)
    return {
      trace,
      key: computeMacroTraceReplayKey(trace),
      semanticKey: computeMacroSemanticReplayKey(trace),
      fingerprint: traceFingerprint(trace),
    }
  }).sort((left, right) =>
    compareMacroText(left.trace.independenceGroupId, right.trace.independenceGroupId) ||
    compareMacroText(left.semanticKey, right.semanticKey) ||
    compareMacroText(left.key, right.key) ||
    compareMacroText(left.trace.id, right.trace.id) ||
    compareMacroText(left.fingerprint, right.fingerprint),
  )
  for (const prepared of sortedTraces) {
    const { trace, key, fingerprint, semanticKey } = prepared
    if (alreadyProcessed.has(key) || alreadyProcessedSemantic.has(semanticKey) ||
      traceBloomContains(key) || semanticBloomContains(semanticKey) ||
      seenSemantic.has(semanticKey)) {
      duplicateTraceCount += 1
      continue
    }
    const previous = seen.get(key)
    if (previous) {
      if (previous !== fingerprint) {
        throw new Error(`Aynı episode/runKey için farklı makro izleri verildi: ${key}`)
      }
      duplicateTraceCount += 1
      continue
    }
    seen.set(key, fingerprint)
    seenSemantic.add(semanticKey)
    processedKeys.push(key)
    processedSemanticKeys.push(semanticKey)
    acceptedTraces.push(trace)
  }

  const referenceTrace = acceptedTraces[0]
  for (const trace of acceptedTraces) {
    if (trace.representationSpaceId !== referenceTrace.representationSpaceId) {
      throw new Error('Tek makro batch içinde farklı temsil uzayları karıştırılamaz.')
    }
    if (trace.familyRunFingerprint !== referenceTrace.familyRunFingerprint) {
      throw new Error('Tek makro batch içinde farklı FamilyRun fingerprintleri karıştırılamaz.')
    }
    if (trace.steps[0] && referenceTrace.steps[0] &&
      trace.steps[0].signalBefore.length !== referenceTrace.steps[0].signalBefore.length) {
      throw new Error('Tek makro batch içinde farklı sinyal boyutları karıştırılamaz.')
    }
  }

  const vectorDimension = referenceTrace?.steps[0]?.signalBefore.length ?? 1
  const effectiveCapacity = effectiveMacroHeavyHitterCapacity(config, vectorDimension)

  for (const trace of acceptedTraces) {
    const overlapEnd = new Map<string, number>()
    for (let length = config.minimumMotifLength; length <= config.maximumMotifLength; length += 1) {
      for (let start = 0; start + length <= trace.steps.length; start += 1) {
        counters.generatedWindowCount += 1
        const steps = trace.steps.slice(start, start + length)
        const exact = createObservation(trace, steps, start, 'sequence-macro')
        const observationPairs: Array<{ observation: MacroObservationV1; key: string }> = [{
          observation: exact,
          key: `sequence-macro\u0000${exact.exactSignature}`,
        }]
        if (steps.every((step) => step.member.familyIds.length > 0)) {
          const family = createObservation(trace, steps, start, 'equivalence-concept')
          observationPairs.push({
            observation: family,
            key: `equivalence-concept\u0000${family.familySignature}`,
          })
        }
        for (const pair of observationPairs) {
          const priorEnd = overlapEnd.get(pair.key) ?? -1
          if (start < priorEnd) {
            counters.overlappingObservationCount += 1
            continue
          }
          overlapEnd.set(pair.key, start + length)
          addObservation(candidates, pair.observation, config, effectiveCapacity, counters)
        }
      }
    }
  }

  const finalized = [...candidates.values()]
    .map((candidate) => finalizeCandidate(candidate, config))
    .sort((left, right) =>
      Number(right.lifecycle === 'eligible-proposal') - Number(left.lifecycle === 'eligible-proposal') ||
      right.mdl.compressionGain - left.mdl.compressionGain ||
      right.observationCount - left.observationCount ||
      compareMacroText(left.id, right.id),
    )
  const diagnostics = {
    inputTraceCount: traces.length,
    processedTraceCount: acceptedTraces.length,
    duplicateTraceCount,
    generatedWindowCount: counters.generatedWindowCount,
    acceptedObservationCount: counters.acceptedObservationCount,
    overlappingObservationCount: counters.overlappingObservationCount,
    heavyHitterReplacementCount: counters.heavyHitterReplacementCount,
    retainedCandidateCount: finalized.length,
  }
  const base: Omit<MacroMiningResultV1, 'fingerprint'> = {
    format: 'concept-macro-observation-batch',
    version: 1,
    representationSpaceId: referenceTrace?.representationSpaceId ?? null,
    familyRunFingerprint: referenceTrace?.familyRunFingerprint ?? null,
    configFingerprint,
    processedTraceKeys: processedKeys.sort(compareMacroText),
    processedSemanticTraceKeys: processedSemanticKeys.sort(compareMacroText),
    candidates: finalized,
    diagnostics,
  }
  return { ...base, fingerprint: macroDigest('MMININGFP', base) }
}
