import type {
  MacroMemberReferenceV1,
  MacroShadowCaseV1,
  MacroTraceV1,
  MacroTraceStepV1,
} from './types.ts'

export function macroMember(
  id: string,
  familyIds: readonly string[] = [],
  kind: MacroMemberReferenceV1['kind'] = 'raw-neuron',
): MacroMemberReferenceV1 {
  return {
    kind,
    id,
    identityDigest: kind === 'raw-neuron' ? `IDENTITY-${id}` : `COMPOSITE-${id}`,
    familyIds: [...familyIds],
  }
}

export function macroStep(
  member: MacroMemberReferenceV1,
  signalBefore: readonly number[] = [0, 0],
  signalAfter: readonly number[] = [0, 0],
): MacroTraceStepV1 {
  return { member, signalBefore: [...signalBefore], signalAfter: [...signalAfter] }
}

export function macroTrace(input: {
  id: string
  members: readonly MacroMemberReferenceV1[]
  group?: string
  sourceId?: string
  sourceDigest?: string
  episodeId?: string
  runKey?: string
  semanticInputDigest?: string
  familyRunFingerprint?: string | null
  split?: MacroTraceV1['split']
  signalOffset?: number
}): MacroTraceV1 {
  const offset = input.signalOffset ?? 0
  return {
    format: 'macro-trace-v1',
    version: 1,
    id: input.id,
    representationSpaceId: 'space-64-v1',
    familyRunFingerprint: input.familyRunFingerprint ?? 'family-run-1',
    sourceId: input.sourceId ?? `source-${input.group ?? input.id}`,
    sourceDigest: input.sourceDigest ?? `SOURCE-DIGEST-${input.group ?? input.id}`,
    independenceGroupId: input.group ?? `group-${input.id}`,
    episodeId: input.episodeId ?? `episode-${input.id}`,
    runKey: input.runKey ?? `run-${input.id}`,
    semanticInputDigest: input.semanticInputDigest ?? `SEMANTIC-${input.id}`,
    split: input.split ?? 'train',
    steps: input.members.map((member, index) => macroStep(
      member,
      [Math.min(1, offset + index * 0.05), Math.max(-1, -offset)],
      [Math.min(1, offset + (index + 1) * 0.05), Math.max(-1, -offset)],
    )),
  }
}

export function passingShadowCases(): MacroShadowCaseV1[] {
  return [
    {
      id: 'shadow-a', split: 'validation', independenceGroupId: 'shadow-book-a', shouldActivate: true, macroActivated: true,
      expandedOutput: [0.25, 0.75], macroOutput: [0.25, 0.75], expectedOutput: [0.25, 0.75],
      expandedCost: 3, macroCost: 1,
    },
    {
      id: 'shadow-b', split: 'validation', independenceGroupId: 'shadow-book-b', shouldActivate: false, macroActivated: false,
      expandedOutput: [0.5, 0.5], macroOutput: [0.5, 0.5], expectedOutput: [0.5, 0.5],
      expandedCost: 3, macroCost: 1,
    },
    {
      id: 'shadow-c', split: 'test', independenceGroupId: 'shadow-book-c', shouldActivate: true, macroActivated: true,
      expandedOutput: [0.75, 0.25], macroOutput: [0.75, 0.25], expectedOutput: [0.75, 0.25],
      expandedCost: 3, macroCost: 1,
    },
  ]
}
