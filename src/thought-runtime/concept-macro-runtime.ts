import {
  CONCEPT_MACRO_EXECUTION_SEMANTICS,
  expandCompositeNeuron,
  macroDigest,
  validateMacroHead,
  type CompositeNeuronV1,
  type MacroHeadV1,
  type MacroMemberReferenceV1,
} from '../concept-macros/index.ts'
import { assertThoughtIdentityIntegrity } from './model'
import { clamp, cosineSimilarity, projectThoughtVector } from './math'
import type { ThoughtModel, ThoughtVector } from './types'

export const THOUGHT_MACRO_RUNTIME_FORMAT = 'thought-concept-macro-runtime-v1' as const
export const THOUGHT_MACRO_MINIMUM_ENTRY_COMPATIBILITY = 0.8
export const THOUGHT_MACRO_MAXIMUM_CANDIDATES = 8
export const THOUGHT_MACRO_MAXIMUM_PROBABILITY_MASS = 0.6

export type ThoughtMacroFallbackReason =
  | 'snapshot-head-absent'
  | 'invalid-macro-head'
  | 'representation-space-mismatch'
  | 'family-run-fingerprint-mismatch'
  | 'model-identity-integrity-failed'
  | 'dag-validation-failed'
  | 'inactive-lifecycle'
  | 'shadow-gate-regressed'
  | 'expanded-duration-out-of-range'
  | 'raw-constituent-missing'
  | 'identity-digest-mismatch'
  | 'family-membership-mismatch'
  | 'compiled-vector-dimension-mismatch'
  | 'entry-signal-below-threshold'
  | 'candidate-runtime-mismatch'
  | 'compiled-effect-invalid'
  | 'persistent-learning-requires-raw-path'
  | 'continuation-neuron-unproven'
  | 'equivalence-runtime-unimplemented'

export interface ThoughtMacroFallbackV1 {
  macroId: string | null
  reason: ThoughtMacroFallbackReason
  detail: string
}

export interface ThoughtCompiledMacroV1 {
  macroId: string
  macroFingerprint: string
  macroHeadFingerprint: string
  kind: CompositeNeuronV1['kind']
  originNeuronId: string
  terminalNeuronId: string
  expandedConstituentIds: string[]
  expandedConstituentDigests: string[]
  duration: number
  compressedTransitionCount: number
  depth: number
  fidelity: number
  effectCohesion: number
  activationRecall: number
  falseActivationRate: number
  minimumEntryCompatibility: number
  entryPrototype: Float32Array
  effectPrototype: Float32Array
  fingerprint: string
}

export interface ThoughtRankedMacroCandidateV1 extends ThoughtCompiledMacroV1 {
  entryCompatibility: number
  activationWeight: number
  selectionProbability: number
}

export interface ThoughtMacroRuntimeV1 {
  format: typeof THOUGHT_MACRO_RUNTIME_FORMAT
  version: 1
  enabled: boolean
  representationSpaceId: string
  familyRunFingerprint: string
  macroHeadFingerprint: string | null
  fingerprint: string
  activeMacroCount: number
  candidatesByOrigin: ReadonlyMap<string, readonly ThoughtCompiledMacroV1[]>
  fallbacksByOrigin: ReadonlyMap<string, readonly ThoughtMacroFallbackV1[]>
  globalFallbacks: readonly ThoughtMacroFallbackV1[]
}

export interface ThoughtMacroRankingV1 {
  runtimeFingerprint: string
  candidates: ThoughtRankedMacroCandidateV1[]
  rawProbabilityScale: number
  macroProbabilityMass: number
  fallbacks: ThoughtMacroFallbackV1[]
}

export interface ThoughtMacroApplicationV1 {
  applied: boolean
  signal: ThoughtVector
  signalChangeMagnitude: number
  fallbackReason: ThoughtMacroFallbackReason | null
}

function fallback(
  macroId: string | null,
  reason: ThoughtMacroFallbackReason,
  detail: string,
): ThoughtMacroFallbackV1 {
  return { macroId, reason, detail }
}

function disabledRuntime(input: {
  representationSpaceId: string
  familyRunFingerprint: string
  macroHeadFingerprint?: string | null
  reason: ThoughtMacroFallbackV1
}): ThoughtMacroRuntimeV1 {
  const globalFallbacks = [input.reason]
  return {
    format: THOUGHT_MACRO_RUNTIME_FORMAT,
    version: 1,
    enabled: false,
    representationSpaceId: input.representationSpaceId,
    familyRunFingerprint: input.familyRunFingerprint,
    macroHeadFingerprint: input.macroHeadFingerprint ?? null,
    fingerprint: macroDigest(
      'THOUGHT-MACRO-RUNTIME',
      input.representationSpaceId,
      input.familyRunFingerprint,
      input.macroHeadFingerprint ?? null,
      globalFallbacks,
    ),
    activeMacroCount: 0,
    candidatesByOrigin: new Map(),
    fallbacksByOrigin: new Map(),
    globalFallbacks,
  }
}

function exactFamilyIds(
  neuron: ThoughtModel['neurons'][number],
  member: MacroMemberReferenceV1,
): boolean {
  const actual = [...new Set(neuron.familyMemberships.map((membership) => membership.familyId))].sort()
  const expected = [...new Set(member.familyIds)].sort()
  return expected.length === actual.length && expected.every((familyId, index) => familyId === actual[index])
}

function vectorDimensionIsValid(composite: CompositeNeuronV1, dimension: number): boolean {
  return composite.entryPrototype.dimension === dimension &&
    composite.entryPrototype.vector.length === dimension &&
    composite.exitPrototype.dimension === dimension &&
    composite.exitPrototype.vector.length === dimension &&
    composite.effectPrototype.length === dimension &&
    composite.entryPrototype.vector.every((value) => Number.isFinite(value) && value >= -1 && value <= 1) &&
    composite.effectPrototype.every((value) => Number.isFinite(value) && value >= -2 && value <= 2)
}

function activeGateRegressed(composite: CompositeNeuronV1, head: MacroHeadV1): boolean {
  return composite.metrics.shadowCaseCount < head.config.minimumShadowCases ||
    !composite.structuralEquivalencePassed ||
    new Set(composite.shadowCaseEvidence.map((value) => value.independenceGroupId)).size <
      head.config.minimumShadowIndependentGroups ||
    composite.metrics.positiveCaseCount < head.config.minimumShadowPositiveCases ||
    composite.metrics.negativeCaseCount < head.config.minimumShadowNegativeCases ||
    composite.metrics.fidelity < head.config.minimumShadowFidelity ||
    composite.metrics.falseActivationRate > head.config.maximumFalseActivationRate ||
    composite.metrics.activationRecall < head.config.minimumActivationRecall ||
    composite.metrics.holdoutUtility < head.config.minimumHoldoutUtility ||
    composite.metrics.shadowCompressionGain < head.config.minimumShadowCompressionGain ||
    composite.effectCohesion < head.config.minimumEffectCohesion
}

function compileOne(
  composite: CompositeNeuronV1,
  head: MacroHeadV1,
  model: ThoughtModel,
  neuronById: ReadonlyMap<string, ThoughtModel['neurons'][number]>,
): { compiled?: ThoughtCompiledMacroV1; fallback?: ThoughtMacroFallbackV1 } {
  if (composite.lifecycle !== 'active') {
    return { fallback: fallback(composite.id, 'inactive-lifecycle', `lifecycle=${composite.lifecycle}`) }
  }
  if (composite.kind !== 'sequence-macro') {
    return {
      fallback: fallback(
        composite.id,
        'equivalence-runtime-unimplemented',
        'Equivalence kavramı varyant-aware runtime seçicisi tamamlanana kadar yalnız shadow overlay olarak kalır.',
      ),
    }
  }
  if (composite.executionSemantics !== CONCEPT_MACRO_EXECUTION_SEMANTICS) {
    return {
      fallback: fallback(
        composite.id,
        'continuation-neuron-unproven',
        'Makro before(first)→before(last) terminal-anchor-v1 yürütme kanıtını taşımıyor.',
      ),
    }
  }
  if (!composite.structuralEquivalencePassed || !composite.structuralEquivalenceAudit?.passed) {
    return {
      fallback: fallback(
        composite.id,
        'shadow-gate-regressed',
        'Makro hızlı R, eligibility/decay, residual pressure ve sonraki rota eşdeğerliğini kanıtlayan yapısal denetimi taşımıyor.',
      ),
    }
  }
  if (activeGateRegressed(composite, head)) {
    return { fallback: fallback(composite.id, 'shadow-gate-regressed', 'Aktif makro güncel shadow/MDL güvenlik eşiklerini karşılamıyor.') }
  }
  let expanded: MacroMemberReferenceV1[]
  try {
    expanded = expandCompositeNeuron(
      composite.id,
      head.macros,
      Math.min(3, head.config.maximumNestingDepth),
      head.config.maximumExpandedConstituents,
    )
  } catch (error) {
    return { fallback: fallback(composite.id, 'dag-validation-failed', error instanceof Error ? error.message : String(error)) }
  }
  if (expanded.length < 2 || expanded.length > head.config.maximumExpandedConstituents) {
    return { fallback: fallback(composite.id, 'expanded-duration-out-of-range', `duration=${expanded.length}`) }
  }
  for (const member of expanded) {
    const neuron = neuronById.get(member.id)
    if (!neuron) {
      return { fallback: fallback(composite.id, 'raw-constituent-missing', member.id) }
    }
    if (!member.identityDigest || member.identityDigest !== neuron.identityDigest) {
      return { fallback: fallback(composite.id, 'identity-digest-mismatch', member.id) }
    }
    if (!exactFamilyIds(neuron, member)) {
      return { fallback: fallback(composite.id, 'family-membership-mismatch', member.id) }
    }
  }
  if (!vectorDimensionIsValid(composite, model.layout.cellCount)) {
    return { fallback: fallback(composite.id, 'compiled-vector-dimension-mismatch', `expected=${model.layout.cellCount}`) }
  }
  const ids = expanded.map((member) => member.id)
  const digests = expanded.map((member) => member.identityDigest!)
  const payload = {
    macroId: composite.id,
    macroFingerprint: composite.fingerprint,
    macroHeadFingerprint: head.fingerprint,
    kind: composite.kind,
    originNeuronId: ids[0],
    terminalNeuronId: ids.at(-1)!,
    expandedConstituentIds: ids,
    expandedConstituentDigests: digests,
    duration: ids.length,
    compressedTransitionCount: ids.length - 1,
    depth: composite.depth,
    fidelity: composite.metrics.fidelity,
    effectCohesion: composite.effectCohesion,
    activationRecall: composite.metrics.activationRecall,
    falseActivationRate: composite.metrics.falseActivationRate,
    minimumEntryCompatibility: head.config.minimumEntryCompatibility,
    entryPrototype: Float32Array.from(composite.entryPrototype.vector),
    effectPrototype: Float32Array.from(composite.effectPrototype),
  }
  return {
    compiled: {
      ...payload,
      fingerprint: macroDigest('THOUGHT-COMPILED-MACRO', {
        ...payload,
        entryPrototype: Array.from(payload.entryPrototype),
        effectPrototype: Array.from(payload.effectPrototype),
      }),
    },
  }
}

/**
 * Compiles only active, snapshot-bound macros. Any integrity failure disables
 * the affected shortcut (or the whole head) and leaves the raw router intact.
 */
export function createThoughtMacroRuntime(input: {
  macroHead: MacroHeadV1 | null | undefined
  model: ThoughtModel
  representationSpaceId: string
  familyRunFingerprint: string
}): ThoughtMacroRuntimeV1 {
  const reference = {
    representationSpaceId: input.representationSpaceId,
    familyRunFingerprint: input.familyRunFingerprint,
  }
  if (!input.macroHead) {
    return disabledRuntime({
      ...reference,
      reason: fallback(null, 'snapshot-head-absent', 'Seçili snapshot concept-macro-head taşımıyor.'),
    })
  }
  const semanticHead = input.macroHead as MacroHeadV1 & {
    config?: { executionSemantics?: string }
    macros?: Array<{ executionSemantics?: string }>
  }
  if (
    semanticHead.config?.executionSemantics !== CONCEPT_MACRO_EXECUTION_SEMANTICS ||
    !Array.isArray(semanticHead.macros) ||
    semanticHead.macros.some((macro) => macro.executionSemantics !== CONCEPT_MACRO_EXECUTION_SEMANTICS)
  ) {
    return disabledRuntime({
      ...reference,
      macroHeadFingerprint: input.macroHead.fingerprint ?? null,
      reason: fallback(
        null,
        'continuation-neuron-unproven',
        'Legacy makro head terminal-anchor-v1 yürütme kanıtı taşımıyor.',
      ),
    })
  }
  try {
    validateMacroHead(input.macroHead)
  } catch (error) {
    return disabledRuntime({
      ...reference,
      macroHeadFingerprint: input.macroHead.fingerprint ?? null,
      reason: fallback(null, 'invalid-macro-head', error instanceof Error ? error.message : String(error)),
    })
  }
  if (input.macroHead.representationSpaceId !== input.representationSpaceId) {
    return disabledRuntime({
      ...reference,
      macroHeadFingerprint: input.macroHead.fingerprint,
      reason: fallback(null, 'representation-space-mismatch', 'Makro head ve Thought modeli farklı temsil uzaylarında.'),
    })
  }
  if (input.macroHead.familyRunFingerprint !== input.familyRunFingerprint) {
    return disabledRuntime({
      ...reference,
      macroHeadFingerprint: input.macroHead.fingerprint,
      reason: fallback(null, 'family-run-fingerprint-mismatch', 'Makro head seçili FamilyRun fingerprintine ait değil.'),
    })
  }
  try {
    assertThoughtIdentityIntegrity(input.model)
  } catch (error) {
    return disabledRuntime({
      ...reference,
      macroHeadFingerprint: input.macroHead.fingerprint,
      reason: fallback(null, 'model-identity-integrity-failed', error instanceof Error ? error.message : String(error)),
    })
  }
  const neuronById = new Map(input.model.neurons.map((neuron) => [neuron.id, neuron]))
  const candidatesByOrigin = new Map<string, ThoughtCompiledMacroV1[]>()
  const fallbacksByOrigin = new Map<string, ThoughtMacroFallbackV1[]>()
  const globalFallbacks: ThoughtMacroFallbackV1[] = []
  for (const composite of [...input.macroHead.macros].sort((left, right) => left.id.localeCompare(right.id))) {
    const result = compileOne(composite, input.macroHead, input.model, neuronById)
    const directOrigin = composite.orderedExpansion[0]?.id
    if (result.fallback) {
      if (directOrigin) {
        const row = fallbacksByOrigin.get(directOrigin) ?? []
        row.push(result.fallback)
        fallbacksByOrigin.set(directOrigin, row)
      } else globalFallbacks.push(result.fallback)
      continue
    }
    const compiled = result.compiled!
    const row = candidatesByOrigin.get(compiled.originNeuronId) ?? []
    row.push(compiled)
    candidatesByOrigin.set(compiled.originNeuronId, row)
  }
  for (const row of candidatesByOrigin.values()) {
    row.sort((left, right) => right.fidelity - left.fidelity || left.macroId.localeCompare(right.macroId))
  }
  for (const row of fallbacksByOrigin.values()) {
    row.sort((left, right) => (left.macroId ?? '').localeCompare(right.macroId ?? '') || left.reason.localeCompare(right.reason))
  }
  const compiled = [...candidatesByOrigin.values()].flat()
  const fingerprint = macroDigest(
    'THOUGHT-MACRO-RUNTIME',
    input.macroHead.fingerprint,
    input.representationSpaceId,
    input.familyRunFingerprint,
    compiled.map((candidate) => candidate.fingerprint),
    [...fallbacksByOrigin.entries()],
    globalFallbacks,
  )
  return {
    format: THOUGHT_MACRO_RUNTIME_FORMAT,
    version: 1,
    enabled: compiled.length > 0,
    representationSpaceId: input.representationSpaceId,
    familyRunFingerprint: input.familyRunFingerprint,
    macroHeadFingerprint: input.macroHead.fingerprint,
    fingerprint,
    activeMacroCount: compiled.length,
    candidatesByOrigin,
    fallbacksByOrigin,
    globalFallbacks,
  }
}

/**
 * Compiled effects cannot reconstruct each constituent's exact fast R patch.
 * Persistent-learning episodes therefore keep the full raw path.
 */
export function applyThoughtMacroLearningSafety(
  runtime: ThoughtMacroRuntimeV1,
  learningMode: string,
): ThoughtMacroRuntimeV1 {
  if (String(learningMode).toLocaleLowerCase('en-US') === 'off') return runtime
  const globalFallbacks: ThoughtMacroFallbackV1[] = [
    ...runtime.globalFallbacks.map((entry) => ({ ...entry })),
    fallback(
      null,
      'persistent-learning-requires-raw-path',
      'Kalıcı öğrenmede constituent fast R mikro-izleri yalnız ham yürütmeyle hesaplanır.',
    ),
  ]
  return {
    ...runtime,
    enabled: false,
    fingerprint: macroDigest(
      'THOUGHT-MACRO-LEARNING-SAFETY',
      runtime.fingerprint,
      String(learningMode).toLocaleLowerCase('en-US'),
    ),
    activeMacroCount: 0,
    candidatesByOrigin: new Map(),
    fallbacksByOrigin: new Map(),
    globalFallbacks,
  }
}

/** Ranks with only the current member and S_t. No future route/path is read. */
export function rankThoughtMacroCandidates(
  runtime: ThoughtMacroRuntimeV1,
  currentNeuronId: string,
  signal: ThoughtVector,
): ThoughtMacroRankingV1 {
  const fallbacks = [
    ...runtime.globalFallbacks,
    ...(runtime.fallbacksByOrigin.get(currentNeuronId) ?? []),
  ].map((entry) => ({ ...entry }))
  if (!runtime.enabled) {
    return {
      runtimeFingerprint: runtime.fingerprint,
      candidates: [],
      rawProbabilityScale: 1,
      macroProbabilityMass: 0,
      fallbacks,
    }
  }
  const eligible: Array<ThoughtCompiledMacroV1 & { entryCompatibility: number; activationWeight: number }> = []
  for (const candidate of runtime.candidatesByOrigin.get(currentNeuronId) ?? []) {
    if (candidate.macroHeadFingerprint !== runtime.macroHeadFingerprint) {
      fallbacks.push(fallback(candidate.macroId, 'candidate-runtime-mismatch', 'Aday macro head fingerprinti runtime ile uyuşmuyor.'))
      continue
    }
    const entryCompatibility = Math.max(0, cosineSimilarity(signal, candidate.entryPrototype))
    if (entryCompatibility < candidate.minimumEntryCompatibility) {
      fallbacks.push(fallback(
        candidate.macroId,
        'entry-signal-below-threshold',
        `compatibility=${entryCompatibility.toFixed(6)}`,
      ))
      continue
    }
    const reliability = candidate.fidelity * candidate.effectCohesion * candidate.activationRecall *
      (1 - candidate.falseActivationRate)
    const activationWeight = Math.max(0, entryCompatibility ** 2 * reliability)
    if (activationWeight > 0) eligible.push({ ...candidate, entryCompatibility, activationWeight })
  }
  eligible.sort((left, right) => right.activationWeight - left.activationWeight || left.macroId.localeCompare(right.macroId))
  const retained = eligible.slice(0, THOUGHT_MACRO_MAXIMUM_CANDIDATES)
  const weightTotal = retained.reduce((sum, candidate) => sum + candidate.activationWeight, 0)
  const macroProbabilityMass = weightTotal > 0
    ? Math.min(THOUGHT_MACRO_MAXIMUM_PROBABILITY_MASS, weightTotal / (0.5 + weightTotal))
    : 0
  const candidates = retained.map((candidate): ThoughtRankedMacroCandidateV1 => ({
    ...candidate,
    selectionProbability: weightTotal > 0
      ? macroProbabilityMass * candidate.activationWeight / weightTotal
      : 0,
  }))
  return {
    runtimeFingerprint: runtime.fingerprint,
    candidates,
    rawProbabilityScale: 1 - macroProbabilityMass,
    macroProbabilityMass,
    fallbacks: fallbacks
      .sort((left, right) => (left.macroId ?? '').localeCompare(right.macroId ?? '') || left.reason.localeCompare(right.reason))
      .slice(0, THOUGHT_MACRO_MAXIMUM_CANDIDATES),
  }
}

/** Applies the shadow-validated one-step delta; failure returns S_t unchanged. */
export function applyThoughtMacroEffect(
  runtime: ThoughtMacroRuntimeV1,
  candidate: ThoughtRankedMacroCandidateV1,
  signal: ThoughtVector,
): ThoughtMacroApplicationV1 {
  if (
    !runtime.enabled || candidate.macroHeadFingerprint !== runtime.macroHeadFingerprint ||
    candidate.entryPrototype.length !== signal.length || candidate.effectPrototype.length !== signal.length ||
    !runtime.candidatesByOrigin.get(candidate.originNeuronId)?.some((entry) => (
      entry.macroId === candidate.macroId && entry.fingerprint === candidate.fingerprint
    ))
  ) {
    return {
      applied: false,
      signal: new Float32Array(signal),
      signalChangeMagnitude: 0,
      fallbackReason: 'candidate-runtime-mismatch',
    }
  }
  const mixed = new Float32Array(signal.length)
  let squaredChange = 0
  for (let index = 0; index < signal.length; index += 1) {
    const value = signal[index] + candidate.effectPrototype[index]
    if (!Number.isFinite(value)) {
      return {
        applied: false,
        signal: new Float32Array(signal),
        signalChangeMagnitude: 0,
        fallbackReason: 'compiled-effect-invalid',
      }
    }
    mixed[index] = Math.fround(clamp(value, -1, 1))
  }
  const output = projectThoughtVector(mixed, 1)
  for (let index = 0; index < output.length; index += 1) {
    const difference = output[index] - signal[index]
    squaredChange += difference * difference
  }
  return {
    applied: true,
    signal: output,
    signalChangeMagnitude: Math.sqrt(squaredChange),
    fallbackReason: null,
  }
}
