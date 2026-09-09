import type { FamilyRun, FunctionalRegionId } from '../families/types'
import { createThoughtRuntimeConfig } from './config'
import { createThoughtHead, createThoughtLayout, createThoughtModel, createThoughtNeuron } from './model'
import type {
  SparseHabitRow,
  ThoughtBranch,
  ThoughtHead,
  ThoughtModel,
  ThoughtNeuron,
  ThoughtTransitionRecord,
} from './types'

export const TEST_LAYOUT = createThoughtLayout(64)
export const TEST_CONFIG = createThoughtRuntimeConfig({ maximumBranches: 1 })

export function partition(length: number, entries: Readonly<Record<number, number>> = {}): Float32Array {
  const result = new Float32Array(length)
  for (const [rawIndex, value] of Object.entries(entries)) result[Number(rawIndex)] = value
  return result
}

export function testNeuron(
  id: string,
  identityEntries: Readonly<Record<number, number>>,
  familyId = 'family-content',
  regionSurface = id,
): ThoughtNeuron {
  return createThoughtNeuron({
    id,
    identity: partition(TEST_LAYOUT.identityLength, identityEntries),
    relation: partition(TEST_LAYOUT.relationLength),
    familyMemberships: [{ familyId, weight: 1 }],
    provenance: { kind: 'seed', sourceId: `source-${id}`, surface: regionSurface },
  }, TEST_LAYOUT, TEST_CONFIG)
}

export function testModel(input: {
  neurons: ThoughtNeuron[]
  rows?: SparseHabitRow[]
  regions?: Record<string, FunctionalRegionId>
}): { model: ThoughtModel; head: ThoughtHead } {
  const familyIds = [...new Set(input.neurons.flatMap((neuron) => neuron.familyMemberships.map((item) => item.familyId)))]
  const model = createThoughtModel({
    layout: TEST_LAYOUT,
    neurons: input.neurons,
    families: familyIds.map((id) => ({ id, region: input.regions?.[id] })),
    habitRows: input.rows ?? [],
  }, TEST_CONFIG)
  return { model, head: createThoughtHead(model) }
}

export function transition(
  tick: number,
  fromNeuronId: string,
  toNeuronId: string,
  familyId = 'family-content',
): ThoughtTransitionRecord {
  return {
    tick, fromNeuronId, toNeuronId, familyId,
    compatibility: 1,
    baseH0: 1,
    persistentExcitation: 0,
    fastExcitation: 0,
    persistentInhibition: 0,
    fastInhibition: 0,
    habitLogit: 0,
    habitProbability: 1,
    membershipWeight: 1,
    routeEvidence: 1,
    normalizedHabit: 1,
    q: 1,
    acceptanceProbability: 1,
    selectionProbability: 1,
    signalChangeMagnitude: 0,
  }
}

export function blankBranch(id: string, currentNeuronId: string): ThoughtBranch {
  return {
    id,
    parentId: null,
    currentNeuronId,
    signal: new Float32Array(4096),
    mass: 1,
    cumulativeLogProbability: 0,
    path: [currentNeuronId],
    transitions: [],
    fastTraces: [],
    fastRelationPatches: [],
    pressure: {
      habitDistance: 0,
      representationError: 0,
      edge: 0,
      birth: 0,
      smoothedBirth: 0,
      consecutiveBirthTicks: 0,
    },
    residual: {
      sum: new Float32Array(4096),
      weight: 0,
      observationCount: 0,
      pressureTotal: 0,
      cohesionTotal: 0,
      cohesionComparisons: 0,
      cohesionUnitSum: new Float32Array(4096),
      originEvidence: [],
      familyEvidence: [],
    },
    status: 'active',
  }
}

export function minimalFamilyRun(input: {
  ids: string[]
  representationSpaceId: string
  fingerprint?: string
  familyId?: string
  transitions?: Array<{ from: string; to: string; probability: number }>
  region?: FunctionalRegionId
}): FamilyRun {
  const familyId = input.familyId ?? 'family-content'
  return {
    format: 'noron-aileleri-run',
    version: 1,
    id: 'family-run-test',
    fingerprint: input.fingerprint ?? 'family-run-fingerprint-test',
    createdAt: '2000-01-01T00:00:00.000Z',
    representationSpaceId: input.representationSpaceId,
    behaviorSpaceId: 'behavior-space-test',
    config: {},
    inputManifest: { entries: [], corpusFingerprint: 'corpus-test' },
    profiles: input.ids.map((memberKey) => ({ memberKey, kind: 'raw' })),
    families: [{ id: familyId, region: input.region ?? 'content', lifecycle: 'stable' }],
    memberships: input.ids.map((memberKey) => ({ memberKey, familyId, weight: 1, rank: 0 })),
    neuronTransitions: (input.transitions ?? []).map((edge) => ({
      fromMemberKey: edge.from, toMemberKey: edge.to, probability: edge.probability,
    })),
    familyTransitions: [],
    utilityReports: [],
    coalition: { tick: 0, activeFamilyCount: 0, coalitions: [], unmatchedReason: null },
    routingPreview: null,
    warnings: [],
    stats: {},
  } as unknown as FamilyRun
}
