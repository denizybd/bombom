import { describe, expect, it } from 'vitest'
import {
  applyMacroShadowEvaluation,
  applyMacroStructuralEquivalenceAudit,
  computeCompositeNeuronFingerprint,
  computeMacroHeadFingerprint,
  createMacroHead,
  createMacroStructuralEquivalenceAudit,
  evaluateMacroShadow,
  ingestMacroTraces,
  transitionCompositeLifecycle,
  type MacroHeadV1,
  type MacroMemberReferenceV1,
  type MacroTraceStepV1,
  type MacroTraceV1,
} from '../concept-macros/index.ts'
import { passingShadowCases } from '../concept-macros/test-fixtures.ts'
import {
  applyThoughtMacroLearningSafety,
  applyThoughtMacroEffect,
  createThoughtMacroRuntime,
  rankThoughtMacroCandidates,
} from './concept-macro-runtime'
import { startThoughtEpisode, stepThoughtEpisode } from './engine'
import { createThoughtRuntimeIndex } from './routing'
import { TEST_CONFIG, testModel, testNeuron } from './test-fixtures'
import type { ThoughtNeuron } from './types'

const SPACE = 'thought-macro-test-space'
const FAMILY_RUN = 'thought-macro-family-run'

function vector(entries: Readonly<Record<number, number>>): number[] {
  const result = new Array<number>(4_096).fill(0)
  for (const [index, value] of Object.entries(entries)) result[Number(index)] = value
  return result
}

function members(neurons: readonly ThoughtNeuron[]): MacroMemberReferenceV1[] {
  return neurons.map((neuron) => ({
    kind: 'raw-neuron',
    id: neuron.id,
    identityDigest: neuron.identityDigest,
    familyIds: neuron.familyMemberships.map((membership) => membership.familyId),
  }))
}

function trace(id: string, expansion: readonly MacroMemberReferenceV1[]): MacroTraceV1 {
  const signals = [
    vector({ 0: 1 }),
    vector({ 0: 1 }),
    vector({ 0: 1 }),
    vector({ 0: 1 }),
  ]
  const steps: MacroTraceStepV1[] = expansion.map((member, index) => ({
    member,
    signalBefore: signals[index],
    signalAfter: signals[index + 1],
  }))
  return {
    format: 'macro-trace-v1', version: 1, id,
    representationSpaceId: SPACE,
    familyRunFingerprint: FAMILY_RUN,
    sourceId: `source-${id}`,
    sourceDigest: `source-digest-${id}`,
    independenceGroupId: `group-${id}`,
    episodeId: `episode-${id}`,
    runKey: `run-${id}`,
    semanticInputDigest: `semantic-${id}`,
    split: 'train',
    steps,
  }
}

function activeHead(neurons: readonly ThoughtNeuron[]): MacroHeadV1 {
  const initial = createMacroHead({
    representationSpaceId: SPACE,
    familyRunFingerprint: FAMILY_RUN,
  }, { minimumMotifLength: 3, maximumMotifLength: 3 })
  const expansion = members(neurons)
  const ingested = ingestMacroTraces(initial, ['a', 'b', 'c'].map((id) => trace(id, expansion))).head
  const candidate = ingested.macros.find((macro) => (
    macro.kind === 'sequence-macro' && macro.orderedExpansion.map((member) => member.id).join(',') === 'a,b,c'
  ))!
  const provisional = transitionCompositeLifecycle(ingested, candidate.id, 'virtual-provisional')
  const report = evaluateMacroShadow(
    provisional.macros.find((macro) => macro.id === candidate.id)!,
    passingShadowCases(),
    provisional.config,
  )
  const validated = applyMacroShadowEvaluation(provisional, report)
  const shadowValidated = validated.macros.find((macro) => macro.id === candidate.id)!
  const audited = applyMacroStructuralEquivalenceAudit(validated, createMacroStructuralEquivalenceAudit({
    composite: shadowValidated,
    evaluator: 'thought-runtime-parity-fixture-v1',
    evidenceDigest: 'fixture-raw-vs-compiled-branch-state',
    caseCount: 3,
    passed: true,
  }))
  return transitionCompositeLifecycle(audited, candidate.id, 'active')
}

function fixture() {
  const neurons = [
    testNeuron('a', { 0: 1 }),
    testNeuron('b', { 0: 1 }),
    testNeuron('c', { 0: 1 }),
  ]
  const { model, head } = testModel({
    neurons,
    rows: [
      { fromNeuronId: 'a', edges: [{
        toNeuronId: 'b', priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0,
        trustMass: 0, updateCount: 0, independentSourceIds: [],
      }] },
      { fromNeuronId: 'b', edges: [{
        toNeuronId: 'c', priorEvidence: 1, excitationEvidence: 0, inhibitionEvidence: 0,
        trustMass: 0, updateCount: 0, independentSourceIds: [],
      }] },
    ],
  })
  return { neurons, model, head, macroHead: activeHead(neurons) }
}

describe('Thought active concept macro runtime', () => {
  it('yalnız aktif snapshot makrosunu current member + S_t ile sıralar ve compiled etkiyi uygular', () => {
    const { model, macroHead } = fixture()
    const runtime = createThoughtMacroRuntime({
      macroHead, model, representationSpaceId: SPACE, familyRunFingerprint: FAMILY_RUN,
    })
    expect(runtime.enabled).toBe(true)
    expect(runtime.activeMacroCount).toBe(1)
    const input = Float32Array.from(vector({ 0: 1 }))
    const ranking = rankThoughtMacroCandidates(runtime, 'a', input)
    expect(ranking.candidates).toHaveLength(1)
    expect(ranking.candidates[0]).toMatchObject({
      originNeuronId: 'a', terminalNeuronId: 'c', duration: 3,
      expandedConstituentIds: ['a', 'b', 'c'],
    })
    const applied = applyThoughtMacroEffect(runtime, ranking.candidates[0], input)
    expect(applied.applied).toBe(true)
    expect(applied.signal[0]).toBe(1)
    expect(applied.signal[2]).toBe(0)
  })

  it('representation/family/digest uyumsuzluğunda raw fallback bırakır', () => {
    const { model, macroHead } = fixture()
    const wrongFamily = createThoughtMacroRuntime({
      macroHead, model, representationSpaceId: SPACE, familyRunFingerprint: 'other-family-run',
    })
    expect(wrongFamily.enabled).toBe(false)
    expect(wrongFamily.globalFallbacks[0].reason).toBe('family-run-fingerprint-mismatch')

    const corrupted = {
      ...model,
      neurons: model.neurons.map((neuron, index) => index === 1
        ? { ...neuron, identity: Float32Array.from(neuron.identity, (value, coordinate) => coordinate === 0 ? value + 0.1 : value) }
        : neuron),
    }
    const wrongIdentity = createThoughtMacroRuntime({
      macroHead, model: corrupted, representationSpaceId: SPACE, familyRunFingerprint: FAMILY_RUN,
    })
    expect(wrongIdentity.enabled).toBe(false)
    expect(wrongIdentity.globalFallbacks[0].reason).toBe('model-identity-integrity-failed')

    const { executionSemantics: _configSemantics, ...legacyConfig } = macroHead.config
    const legacy = {
      ...macroHead,
      config: legacyConfig,
      macros: macroHead.macros.map(({ executionSemantics: _macroSemantics, ...macro }) => macro),
    } as unknown as MacroHeadV1
    const quarantined = createThoughtMacroRuntime({
      macroHead: legacy, model, representationSpaceId: SPACE, familyRunFingerprint: FAMILY_RUN,
    })
    expect(quarantined.enabled).toBe(false)
    expect(quarantined.globalFallbacks[0].reason).toBe('continuation-neuron-unproven')

    const { fingerprint: _compositeFingerprint, ...sequenceBase } = macroHead.macros[0]
    const unauditedEquivalenceBase = {
      ...sequenceBase,
      kind: 'equivalence-concept' as const,
      exactSignature: null,
      structuralEquivalencePassed: false,
      structuralEquivalenceAudit: null,
    }
    const unauditedEquivalence = {
      ...unauditedEquivalenceBase,
      fingerprint: computeCompositeNeuronFingerprint(unauditedEquivalenceBase),
    }
    const equivalenceAudit = createMacroStructuralEquivalenceAudit({
      composite: unauditedEquivalence,
      evaluator: 'equivalence-dispatch-fixture-v1',
      evidenceDigest: 'fixture-equivalence-kind-dispatch',
      caseCount: 1,
      passed: true,
    })
    const equivalenceBase = {
      ...unauditedEquivalenceBase,
      structuralEquivalencePassed: true,
      structuralEquivalenceAudit: equivalenceAudit,
    }
    const equivalence = {
      ...equivalenceBase,
      fingerprint: computeCompositeNeuronFingerprint(equivalenceBase),
    }
    const { fingerprint: _headFingerprint, ...headBase } = macroHead
    const equivalenceHeadBase = { ...headBase, macros: [equivalence] }
    const equivalenceHead = {
      ...equivalenceHeadBase,
      fingerprint: computeMacroHeadFingerprint(equivalenceHeadBase),
    }
    const unsupported = createThoughtMacroRuntime({
      macroHead: equivalenceHead, model, representationSpaceId: SPACE, familyRunFingerprint: FAMILY_RUN,
    })
    expect(unsupported.enabled).toBe(false)
    expect(unsupported.fallbacksByOrigin.get('a')).toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: 'equivalence-runtime-unimplemented' }),
    ]))
  })

  it('kalıcı öğrenme kipinde compiled shortcutı kapatır ve audit fallback üretir', () => {
    const { model, head, macroHead } = fixture()
    const compiled = createThoughtMacroRuntime({
      macroHead, model, representationSpaceId: SPACE, familyRunFingerprint: FAMILY_RUN,
    })
    const safe = applyThoughtMacroLearningSafety(compiled, 'proposal')
    const ranking = rankThoughtMacroCandidates(safe, 'a', Float32Array.from(vector({ 0: 1 })))
    expect(ranking.candidates).toEqual([])
    expect(ranking.rawProbabilityScale).toBe(1)
    expect(ranking.fallbacks).toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: 'persistent-learning-requires-raw-path' }),
    ]))
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 91)
    const episode = startThoughtEpisode(model, head, index, Float32Array.from(vector({ 0: 1 })), {
      episodeId: 'macro-learning-safety', runKey: 'macro-learning-safety-run', seed: 91,
      maximumTicks: 1, initialNeuronIds: ['a'],
    }, TEST_CONFIG)
    const stepped = stepThoughtEpisode(episode, model, head, index, TEST_CONFIG, {
      selectionMode: 'deterministic', macroRuntime: safe,
    })
    expect(stepped.tick.branches[0].macroExecution).toBeNull()
    expect(stepped.tick.branches[0].macroFallbacks).toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: 'persistent-learning-requires-raw-path' }),
    ]))
    expect(stepped.episode.branches[0].path).toEqual(['a', 'b'])
  })

  it('engine seçilen makroyu tek tickte uygular, constituent yol/edge kredisini ve deterministik tracei korur', () => {
    const { model, head, macroHead } = fixture()
    const runtime = createThoughtMacroRuntime({
      macroHead, model, representationSpaceId: SPACE, familyRunFingerprint: FAMILY_RUN,
    })
    const index = createThoughtRuntimeIndex(model, head, TEST_CONFIG, 77)
    const initial = Float32Array.from(vector({ 0: 1 }))
    const start = () => startThoughtEpisode(model, head, index, initial, {
      episodeId: 'macro-engine-test', runKey: 'macro-engine-run', seed: 77,
      maximumTicks: 2, initialNeuronIds: ['a'],
    }, TEST_CONFIG)
    const first = stepThoughtEpisode(start(), model, head, index, TEST_CONFIG, {
      selectionMode: 'deterministic', macroRuntime: runtime,
    })
    const replay = stepThoughtEpisode(start(), model, head, index, TEST_CONFIG, {
      selectionMode: 'deterministic', macroRuntime: runtime,
    })
    const rawFirst = stepThoughtEpisode(start(), model, head, index, TEST_CONFIG, {
      selectionMode: 'deterministic',
    })
    const rawSecond = stepThoughtEpisode(rawFirst.episode, model, head, index, TEST_CONFIG, {
      selectionMode: 'deterministic',
    })
    expect(first.tick.branches[0].macroExecution).toMatchObject({
      compiled: true,
      expandedConstituentIds: ['a', 'b', 'c'],
      duration: 3,
      compressedTransitionCount: 2,
      creditMode: 'constituent-edges',
      fallbackReason: null,
    })
    expect(first.tick.branches[0].macroProbabilityMass).toBeGreaterThan(0)
    expect(first.tick.branches[0].rawProbabilityScale + first.tick.branches[0].macroProbabilityMass)
      .toBeCloseTo(1, 12)
    expect(first.episode.branches[0].path).toEqual(['a', 'b', 'c'])
    expect(first.episode.branches[0].currentNeuronId).toBe('c')
    expect(first.episode.branches[0].transitions.map((edge) => [edge.fromNeuronId, edge.toNeuronId]))
      .toEqual([['a', 'b'], ['b', 'c']])
    expect(first.episode.branches[0].fastTraces.map((edge) => [edge.fromNeuronId, edge.toNeuronId]))
      .toEqual([['a', 'b'], ['b', 'c']])
    expect(first.episode.branches[0].currentNeuronId).toBe(rawSecond.episode.branches[0].currentNeuronId)
    expect(first.episode.branches[0].path).toEqual(rawSecond.episode.branches[0].path)
    expect(Array.from(first.episode.branches[0].signal))
      .toEqual(Array.from(rawSecond.episode.branches[0].signal))
    expect(replay.tick.replayFingerprint).toBe(first.tick.replayFingerprint)
    expect(replay.episode.fingerprint).toBe(first.episode.fingerprint)
  })
})
