import type {
  NewbornPhase,
  ReplayVerification,
  TeacherMode,
  ThoughtBranch,
  ThoughtAdaptiveBranchTelemetry,
  ThoughtAdaptiveGateDistribution,
  ThoughtAdaptiveTickTelemetry,
  ThoughtCandidate,
  ThoughtCoalition,
  ThoughtEpisode,
  ThoughtEpisodeState,
  ThoughtEpisodeSummary,
  ThoughtGraph,
  ThoughtGraphLevel,
  ThoughtMatrix,
  ThoughtNewborn,
  ThoughtRegionMode,
  ThoughtRegionRuntime,
  ThoughtServiceStatus,
  ThoughtSnapshotOption,
  ThoughtTick,
} from './types'

const MATRIX_SIDE = 64 as const
const MATRIX_CELL_COUNT = MATRIX_SIDE * MATRIX_SIDE

type AnyRecord = Record<string, unknown>

function asRecord(value: unknown): AnyRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as AnyRecord
    : {}
}

function first(record: AnyRecord, keys: readonly string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key]
  }
  return undefined
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : typeof value === 'number' ? String(value) : fallback
}

function asFinite(value: unknown, fallback = 0): number {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' && value.trim() ? Number(value) : Number.NaN
  return Number.isFinite(parsed) ? parsed : fallback
}

function nullableFinite(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const parsed = asFinite(value, Number.NaN)
  return Number.isFinite(parsed) ? parsed : null
}

function asBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value
  if (value === 1 || value === '1' || value === 'true') return true
  if (value === 0 || value === '0' || value === 'false') return false
  return fallback
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function boundedNumericArray(value: unknown, maximum: number): number[] {
  const source = Array.isArray(value)
    ? value
    : ArrayBuffer.isView(value)
      ? Array.from(value as unknown as ArrayLike<number>)
      : Object.entries(asRecord(value))
        .filter(([key]) => /^(?:0|[1-9]\d*)$/u.test(key))
        .sort(([left], [right]) => Number(left) - Number(right))
        .map(([, entry]) => entry)
  return source.slice(0, maximum).flatMap((entry) => {
    const parsed = nullableFinite(entry)
    return parsed === null ? [] : [parsed]
  })
}

function flattenNumeric(value: unknown, target: number[]): void {
  if (target.length >= MATRIX_CELL_COUNT) return
  if (Array.isArray(value)) {
    for (const child of value) {
      flattenNumeric(child, target)
      if (target.length >= MATRIX_CELL_COUNT) break
    }
    return
  }
  if (ArrayBuffer.isView(value)) {
    for (const child of Array.from(value as unknown as ArrayLike<number>)) {
      flattenNumeric(child, target)
      if (target.length >= MATRIX_CELL_COUNT) break
    }
    return
  }
  if (value !== null && typeof value === 'object') {
    const record = asRecord(value)
    const nested = first(record, ['values', 'cells', 'data', 'matrix', 'heatmap', 'vector'])
    if (nested !== undefined && nested !== value) flattenNumeric(nested, target)
    return
  }
  const numeric = nullableFinite(value)
  if (numeric !== null) target.push(numeric)
}

export function clamp01(value: number | null | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

export function normalizeMatrix(input: unknown): ThoughtMatrix {
  const record = asRecord(input)
  const raw = Array.isArray(input)
    ? input
    : first(record, ['values', 'cells', 'data', 'matrix', 'heatmap', 'vector'])
  const values: number[] = []
  flattenNumeric(raw, values)
  if (values.length > MATRIX_CELL_COUNT) values.length = MATRIX_CELL_COUNT
  while (values.length < MATRIX_CELL_COUNT) values.push(0)

  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  let sum = 0
  for (const value of values) {
    min = Math.min(min, value)
    max = Math.max(max, value)
    sum += value
  }
  return {
    side: MATRIX_SIDE,
    values,
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
    mean: values.length ? sum / values.length : 0,
  }
}

function normalizeState(value: unknown): ThoughtEpisodeState {
  const normalized = asString(value).toLowerCase().replaceAll('_', '-')
  if (['queued', 'pending', 'waiting', 'created'].includes(normalized)) return 'queued'
  if (['running', 'active', 'processing', 'started', 'pause-requested', 'cancel-requested'].includes(normalized)) return 'running'
  if (['paused', 'suspended'].includes(normalized)) return 'paused'
  if (['completed', 'complete', 'finished', 'succeeded', 'success'].includes(normalized)) return 'completed'
  if (['failed', 'error'].includes(normalized)) return 'failed'
  if (['cancelled', 'canceled', 'aborted'].includes(normalized)) return 'cancelled'
  return 'unknown'
}

function normalizeTeacherMode(value: unknown): TeacherMode {
  if (value === true) return 'ollama-cross-check'
  if (value === false || value === null) return 'off'
  if (typeof value === 'object') {
    const record = asRecord(value)
    const status = asString(first(record, ['status', 'state'])).toLowerCase()
    return status && !['not-requested', 'off', 'disabled'].includes(status) ? 'ollama-cross-check' : 'off'
  }
  const normalized = asString(value).toLowerCase()
  if (['ollama-cross-check', 'cross-check', 'guided', 'guide', 'teacher', 'on', 'active', 'observe', 'observer'].includes(normalized)) return 'ollama-cross-check'
  return 'off'
}

function normalizeLearningMode(value: unknown): 'off' | 'proposal' {
  return asString(value).toLowerCase() === 'proposal' ? 'proposal' : 'off'
}

function normalizeRegionMode(value: unknown): ThoughtRegionMode {
  const normalized = asString(value).toLowerCase()
  return ['shadow', 'proposal', 'active'].includes(normalized) ? normalized as ThoughtRegionMode : 'off'
}

function normalizeRegionRuntime(value: unknown): ThoughtRegionRuntime | null {
  const source = asRecord(value)
  if (!Object.keys(source).length) return null
  const releaseFingerprint = asString(source.releaseFingerprint) || null
  return {
    behaviorApplied: asBoolean(source.behaviorApplied) && Boolean(releaseFingerprint),
    observationsEnabled: asBoolean(source.observationsEnabled),
    reason: asString(source.reason, 'unknown'),
    atlasArtifactId: asString(source.atlasArtifactId) || null,
    atlasFingerprint: asString(source.atlasFingerprint) || null,
    atlasRevision: source.atlasRevision === null || source.atlasRevision === undefined
      ? null : Math.max(0, Math.round(asFinite(source.atlasRevision))),
    geometryFingerprint: asString(source.geometryFingerprint) || null,
    representationSpaceId: asString(source.representationSpaceId) || null,
    ephemeralAtlas: asBoolean(source.ephemeralAtlas),
    compatibilityBlend: clamp01(nullableFinite(source.compatibilityBlend)),
    minimumTrace: clamp01(nullableFinite(source.minimumTrace)),
    bindingFingerprint: asString(source.bindingFingerprint) || null,
    releaseManifestArtifactId: asString(source.releaseManifestArtifactId) || null,
    releaseFingerprint,
  }
}

function normalizeAdaptiveGates(value: unknown): ThoughtAdaptiveGateDistribution {
  const source = asRecord(value)
  return {
    gates: asArray(source.gates).slice(0, 8).flatMap((entry) => {
      const gate = asRecord(entry)
      const regionId = asString(gate.regionId)
      return regionId ? [{
        regionId,
        compatibility: nullableFinite(gate.compatibility),
        prototypeCompatibility: nullableFinite(gate.prototypeCompatibility),
        contextScore: nullableFinite(gate.contextScore),
        confidence: nullableFinite(gate.confidence),
        probability: nullableFinite(gate.probability),
      }] : []
    }),
    unmatchedProbability: nullableFinite(source.unmatchedProbability),
    omittedRegionProbability: nullableFinite(source.omittedRegionProbability),
  }
}

function normalizeAdaptiveBranchTelemetry(value: unknown): ThoughtAdaptiveBranchTelemetry | null {
  const source = asRecord(value)
  const mode = normalizeRegionMode(source.mode)
  const atlasFingerprint = asString(source.atlasFingerprint)
  const bindingFingerprint = asString(source.bindingFingerprint)
  if (mode === 'off' || !atlasFingerprint || !bindingFingerprint) return null
  return {
    mode,
    atlasFingerprint,
    bindingFingerprint,
    activeApplied: mode === 'active' && asBoolean(source.activeApplied),
    legacyCompatibility: nullableFinite(source.legacyCompatibility),
    regionCompatibility: nullableFinite(source.regionCompatibility),
    blendedCompatibility: nullableFinite(source.blendedCompatibility),
    effectiveCompatibility: nullableFinite(source.effectiveCompatibility),
    pageCompatibilities: boundedNumericArray(source.pageCompatibilities, 56),
    pageWeights: boundedNumericArray(source.pageWeights, 56),
    pageWriteMask: boundedNumericArray(source.pageWriteMask, 56),
    gates: normalizeAdaptiveGates(source.gates),
  }
}

function normalizeAdaptiveTickTelemetry(value: unknown): ThoughtAdaptiveTickTelemetry | null {
  const source = asRecord(value)
  const mode = normalizeRegionMode(source.mode)
  const atlasFingerprint = asString(source.atlasFingerprint)
  const bindingFingerprint = asString(source.bindingFingerprint)
  if (mode === 'off' || !atlasFingerprint || !bindingFingerprint) return null
  return {
    mode,
    atlasFingerprint,
    bindingFingerprint,
    observedBranchCount: Math.max(0, Math.min(8, Math.round(asFinite(source.observedBranchCount)))),
    activeApplied: mode === 'active' && asBoolean(source.activeApplied),
  }
}

function normalizeStructuralOutcome(value: unknown): import('./types').ExpectedStructuralOutcome | null {
  const record = asRecord(value)
  if (record.format !== 'event-graph-v1') return null
  const nullableText = (entry: unknown): string | null => {
    const text = asString(entry).trim()
    return text || null
  }
  return {
    format: 'event-graph-v1',
    actor: nullableText(record.actor),
    action: nullableText(record.action),
    patient: nullableText(record.patient),
    instrument: nullableText(record.instrument),
    operators: asArray(record.operators).map((entry) => asString(entry).trim()).filter(Boolean),
    negated: asBoolean(record.negated),
    goals: asArray(record.goals).flatMap((entry) => {
      const goal = asRecord(entry)
      const relation = asString(goal.relation).trim()
      const target = asString(goal.target).trim()
      return relation && target ? [{ relation, target }] : []
    }),
  }
}

function normalizeNewbornPhase(value: unknown): NewbornPhase {
  const normalized = asString(value).toLowerCase().replaceAll('_', '-')
  if (!normalized || ['none', 'idle', 'not-triggered'].includes(normalized)) return 'none'
  if (['watching', 'observing', 'accumulating'].includes(normalized)) return 'watching'
  if (['candidate', 'proposed', 'proposal'].includes(normalized)) return 'candidate'
  if (['eligible', 'ready'].includes(normalized)) return 'eligible'
  if (['born', 'created', 'committed', 'pressure-threshold'].includes(normalized)) return 'born'
  if (['quarantined', 'quarantine', 'held'].includes(normalized)) return 'quarantined'
  if (['rejected', 'discarded'].includes(normalized)) return 'rejected'
  return 'unknown'
}

function normalizeNewborn(input: unknown): ThoughtNewborn | null {
  if (input === undefined || input === null || input === false) return null
  if (input === true) return { phase: 'candidate', confidence: null }
  const record = asRecord(input)
  const phase = normalizeNewbornPhase(first(record, ['phase', 'state', 'status', 'lifecycle', 'kind']))
  return {
    phase,
    id: asString(first(record, ['id', 'neuronId', 'candidateId'])) || undefined,
    label: asString(first(record, ['label', 'name', 'title'])) || undefined,
    reason: asString(first(record, ['reason', 'message', 'explanation'])) || undefined,
    confidence: nullableFinite(first(record, ['confidence', 'score', 'probability', 'pressure', 'birthPressure'])),
  }
}

function normalizeRoleHypotheses(input: unknown): ThoughtCoalition['roleHypotheses'] {
  if (Array.isArray(input)) {
    return input.map((value) => {
      const item = asRecord(value)
      return {
        role: asString(first(item, ['role', 'type', 'name']), 'rol'),
        target: asString(first(item, ['target', 'label', 'value', 'text', 'memberKey', 'neuronId', 'familyId']), 'henüz çıkarılmadı'),
        score: nullableFinite(first(item, ['score', 'confidence', 'probability'])),
      }
    })
  }
  const record = asRecord(input)
  return Object.entries(record).flatMap(([role, raw]) => {
    if (raw === null || raw === undefined || raw === '') return []
    if (typeof raw === 'string' || typeof raw === 'number') return [{ role, target: String(raw), score: null }]
    const item = asRecord(raw)
    return [{
      role,
      target: asString(first(item, ['target', 'label', 'value', 'neuronId']), 'henüz çıkarılmadı'),
      score: nullableFinite(first(item, ['score', 'confidence', 'probability'])),
    }]
  })
}

function normalizeCoalition(input: unknown, fallbackRoles?: unknown): ThoughtCoalition | null {
  if ((input === undefined || input === null || input === false) && fallbackRoles === undefined) return null
  const record = asRecord(input)
  const members = asArray(first(record, ['members', 'memberIds', 'memberKeys', 'neurons', 'families']))
  const memberIds: string[] = []
  const memberLabels: string[] = []
  members.forEach((member, index) => {
    if (typeof member === 'string' || typeof member === 'number') {
      memberIds.push(String(member))
      memberLabels.push(String(member))
      return
    }
    const item = asRecord(member)
    const id = asString(first(item, ['id', 'memberKey', 'neuronId', 'familyId']), `üye-${index + 1}`)
    memberIds.push(id)
    memberLabels.push(asString(first(item, ['label', 'name', 'text']), id))
  })
  return {
    id: asString(first(record, ['id', 'coalitionId']), 'coalition'),
    label: asString(first(record, ['label', 'name', 'title']), 'Baskın koalisyon'),
    score: nullableFinite(first(record, ['score', 'strength', 'total', 'dominance'])),
    agreement: nullableFinite(first(record, ['agreement', 'cohesion', 'consensus', 'confidence'])),
    dominantFamilyId: asString(first(record, ['dominantFamilyId', 'familyId'])) || undefined,
    memberIds,
    memberLabels,
    roleHypotheses: normalizeRoleHypotheses(first(record, ['roleHypotheses', 'roles', 'roleBindings']) ?? fallbackRoles),
  }
}

function normalizeCandidate(input: unknown, index: number): ThoughtCandidate {
  const record = asRecord(input)
  const scores = asRecord(first(record, ['contributions', 'scores', 'breakdown', 'components']))
  const readScore = (keys: string[]) => nullableFinite(first(record, keys) ?? first(scores, keys))
  const persistentExcitation = readScore(['persistentExcitation', 'experience', 'experienceScore', 'E'])
  const persistentInhibition = readScore(['persistentInhibition'])
  const fastExcitation = readScore(['fastExcitation', 'fastBranchTrace', 'fastTrace', 'trace', 'e'])
  const fastInhibition = readScore(['fastInhibition'])
  const id = asString(first(record, ['id', 'candidateId', 'memberKey', 'neuronId', 'targetNeuronId']), `aday-${index + 1}`)
  const outcome = asString(first(record, ['outcome', 'state', 'status'])).toLowerCase()
  return {
    id,
    label: asString(first(record, ['label', 'name', 'text', 'targetLabel', 'memberKey']), id),
    neuronId: asString(first(record, ['memberKey', 'neuronId', 'targetNeuronId'])) || undefined,
    familyId: asString(first(record, ['familyId', 'targetFamilyId'])) || undefined,
    rank: Math.max(1, Math.round(asFinite(first(record, ['rank', 'order']), index + 1))),
    probability: readScore(['probability', 'prob', 'p']),
    total: readScore(['total', 'score', 'combined', 'finalScore']),
    h0Prior: readScore(['h0Prior', 'baseH0', 'H0', 'h0', 'prior', 'habit', 'habitScore', 'habitProbability']),
    persistentExperience: readScore(['persistentExperience']) ?? (persistentExcitation === null
      ? persistentInhibition === null ? null : -persistentInhibition
      : persistentExcitation - (persistentInhibition ?? 0)),
    branchTrace: readScore(['branchTrace']) ?? (fastExcitation === null
      ? fastInhibition === null ? null : -fastInhibition
      : fastExcitation - (fastInhibition ?? 0)),
    family: readScore(['family', 'familyScore', 'routeEvidence', 'membershipWeight', 'G', 'coalition', 'coalitionScore']),
    signalQ: readScore(['signalQ', 'q', 'signal', 'signalScore', 'similarity', 'compatibility', 'signalCompatibility']),
    repetitionPenalty: readScore(['repetitionPenalty', 'repeatPenalty', 'loopPenalty', 'repetition']),
    habit: readScore(['habit', 'habitScore', 'h0', 'prior']),
    signal: readScore(['signal', 'signalScore', 'similarity', 'compatibility']),
    identity: readScore(['identity', 'identityScore']),
    context: readScore(['context', 'contextScore']),
    coalition: readScore(['coalition', 'coalitionScore']),
    edge: readScore(['edge', 'edgeScore', 'transition']),
    novelty: readScore(['novelty', 'unmatched', 'noveltyScore']),
    selected: asBoolean(first(record, ['selected', 'winner', 'chosen']), ['selected', 'chosen', 'winner'].includes(outcome)),
  }
}

function normalizeCandidates(input: unknown): ThoughtCandidate[] {
  return asArray(input)
    .map(normalizeCandidate)
    .sort((left, right) => left.rank - right.rank)
}

function normalizeBranch(input: unknown, index: number): ThoughtBranch {
  const record = asRecord(input)
  const diagnostics = asRecord(first(record, ['diagnostics', 'metrics']))
  const pressure = asRecord(first(record, ['pressure']))
  const readMetric = (keys: string[]) => nullableFinite(first(record, keys) ?? first(diagnostics, keys) ?? first(pressure, keys))
  const id = asString(first(record, ['id', 'branchId', 'targetNeuronId']), `dal-${index + 1}`)
  const state = asString(first(record, ['state', 'status'])).toLowerCase()
  const coalitionFamilyWeights = asArray(first(record, ['coalitionFamilyWeights']))
  const derivedCoalition = coalitionFamilyWeights.length ? {
    id: `${id}-coalition`,
    label: 'Geçici aile koalisyonu',
    members: coalitionFamilyWeights.map((entry) => {
      const weight = asRecord(entry)
      const familyId = asString(first(weight, ['familyId']), 'aile')
      return { familyId, label: familyId }
    }),
    dominance: coalitionFamilyWeights.reduce<number>((maximum, entry) =>
      Math.max(maximum, asFinite(first(asRecord(entry), ['weight']), 0)), 0),
  } : undefined
  return {
    id,
    rank: Math.max(1, Math.round(asFinite(first(record, ['rank', 'order']), index + 1))),
    label: asString(first(record, ['label', 'name', 'title', 'targetLabel', 'selectedNeuronId', 'currentMemberKey']), `Dal ${index + 1}`),
    targetNeuronId: asString(first(record, ['selectedNeuronId', 'currentMemberKey', 'currentNeuronId', 'targetNeuronId', 'neuronId'])) || undefined,
    targetFamilyId: asString(first(record, ['selectedFamilyId', 'targetFamilyId', 'familyId'])) || undefined,
    probability: readMetric(['probability', 'prob', 'weight', 'mass']),
    selected: asBoolean(first(record, ['selected', 'winner', 'chosen']), state === 'selected' || state === 'winner'),
    pruned: asBoolean(first(record, ['pruned', 'discarded']), state === 'pruned' || state === 'discarded'),
    candidates: normalizeCandidates(first(record, ['candidates', 'neuronCandidates', 'candidateBreakdown', 'rankedCandidates'])),
    coalition: normalizeCoalition(first(record, ['coalition', 'winningCoalition']) ?? derivedCoalition, first(record, ['roleHypotheses', 'roles', 'roleBindings'])),
    unmatched: readMetric(['unmatched', 'unmatchedScore', 'unmatchedProbability']),
    edge: readMetric(['edge', 'edgePressure', 'edgeScore', 'edgeEvidence']),
    birthPressure: readMetric(['smoothedBirthPressure', 'birthPressure', 'neurogenesisPressure', 'birthScore', 'smoothedBirth', 'birth']),
    newborn: normalizeNewborn(first(record, ['newborn', 'birth', 'birthProposal']) ?? asArray(first(record, ['birthEvents']))[0]),
    signalBefore: first(record, ['signalBefore', 'beforeSignal', 'inputSignal', 'matrixBefore']) === undefined
      ? null
      : normalizeMatrix(first(record, ['signalBefore', 'beforeSignal', 'inputSignal', 'matrixBefore'])),
    signalAfter: first(record, ['signalAfter', 'afterSignal', 'outputSignal', 'matrixAfter', 'signal']) === undefined
      ? null
      : normalizeMatrix(first(record, ['signalAfter', 'afterSignal', 'outputSignal', 'matrixAfter', 'signal'])),
    adaptiveRegion: normalizeAdaptiveBranchTelemetry(record.adaptiveRegion),
  }
}

export function normalizeTick(input: unknown, fallbackIndex = 0): ThoughtTick {
  const record = asRecord(input)
  const signal = asRecord(first(record, ['signal', 'matrix']))
  const diagnostics = asRecord(first(record, ['diagnostics', 'metrics', 'evaluation']))
  const pressure = asRecord(first(record, ['pressure', 'birthPressure']))
  const emission = asRecord(first(record, ['emission']))
  let branches = asArray(first(record, ['branches', 'branchStates', 'candidateBranches', 'paths']))
    .slice(0, 8)
    .map(normalizeBranch)
    .sort((left, right) => left.rank - right.rank)
  const readMetric = (keys: string[]) => nullableFinite(first(record, keys) ?? first(diagnostics, keys))
  const explicitSelectedBranchId = asString(first(record, ['selectedBranchId', 'winnerBranchId', 'dominantBranchId']))
  if (explicitSelectedBranchId) {
    branches = branches.map((branch) => branch.id === explicitSelectedBranchId ? { ...branch, selected: true } : branch)
  }
  const selectedBranch = branches.find((branch) => branch.selected)
  const rawBefore = first(record, ['signalBefore', 'before', 'matrixBefore']) ?? first(signal, ['before', 'input'])
  const rawAfter = first(record, ['signalAfter', 'after', 'matrixAfter']) ?? first(signal, ['after', 'output', 'current'])
  return {
    index: Math.max(0, Math.round(asFinite(first(record, ['index', 'tick', 'tickIndex', 'step']), fallbackIndex))),
    at: asString(first(record, ['at', 'createdAt', 'timestamp'])) || undefined,
    label: asString(first(record, ['label', 'title', 'phase']) ?? first(emission, ['text']), `Tik ${fallbackIndex + 1}`),
    activeNeuronId: asString(first(record, ['activeNeuronId', 'neuronId', 'currentNeuronId', 'dominantMemberKey', 'toMemberKey'])) || undefined,
    activeNeuronLabel: asString(first(record, ['activeNeuronLabel', 'neuronLabel', 'currentNeuronLabel']) ?? first(emission, ['text'])) || undefined,
    signalBefore: rawBefore === undefined ? selectedBranch?.signalBefore ?? normalizeMatrix(undefined) : normalizeMatrix(rawBefore),
    signalAfter: rawAfter === undefined ? selectedBranch?.signalAfter ?? normalizeMatrix(undefined) : normalizeMatrix(rawAfter),
    branches,
    selectedBranchId: explicitSelectedBranchId || selectedBranch?.id,
    candidates: normalizeCandidates(first(record, ['candidates', 'candidateBreakdown', 'rankedCandidates'])),
    coalition: normalizeCoalition(first(record, ['coalition', 'winningCoalition']) ?? asArray(first(record, ['coalitions']))[0], first(record, ['roleHypotheses', 'roles', 'roleBindings'])),
    unmatched: readMetric(['unmatched', 'unmatchedScore', 'unmatchedProbability']),
    edge: readMetric(['edge', 'edgeScore', 'edgeEvidence']),
    birthPressure: readMetric(['birthPressure', 'maximumBirthPressure', 'neurogenesisPressure', 'birthScore']) ?? nullableFinite(first(pressure, ['smoothed', 'instantaneous', 'representationError'])),
    newborn: normalizeNewborn(first(record, ['newborn', 'birth', 'birthProposal']) ?? asArray(first(record, ['birthEvents']))[0]),
    structuralScore: readMetric(['structuralScore', 'structureScore', 'structural']),
    trust: readMetric(['trust', 'trustScore', 'confidence']),
    note: asString(first(record, ['note', 'message', 'explanation', 'stopReason'])) || undefined,
    adaptiveRegion: normalizeAdaptiveTickTelemetry(record.adaptiveRegion),
  }
}

function unwrapEpisode(input: unknown): AnyRecord {
  const record = asRecord(input)
  return asRecord(first(record, ['episode', 'data']) ?? record)
}

export function normalizeEpisodeSummary(input: unknown, fallbackIndex = 0): ThoughtEpisodeSummary {
  const record = unwrapEpisode(input)
  const config = asRecord(first(record, ['config', 'settings', 'request']))
  const inputInfo = asRecord(first(record, ['input']))
  const base = asRecord(first(record, ['base']))
  const engine = asRecord(first(record, ['engine']))
  const metrics = asRecord(first(record, ['metrics', 'evaluation', 'scores']) ?? first(engine, ['stats']))
  const outcomeEvidence = asRecord(first(record, ['outcomeEvidence', 'teacherEvidence']))
  const artifacts = asRecord(first(record, ['artifacts']))
  const reviewRecord = asRecord(first(record, ['review']))
  const ticks = asArray(first(record, ['ticks', 'steps']))
  const read = (keys: string[]) => first(record, keys) ?? first(config, keys)
  const readMetric = (keys: string[]) => nullableFinite(first(record, keys) ?? first(metrics, keys))
  const tickCount = Math.max(0, Math.round(asFinite(first(record, ['tickCount', 'completedTicks']), ticks.length)))
  return {
    id: asString(first(record, ['id', 'episodeId', 'runId']), `bolum-${fallbackIndex + 1}`),
    prompt: asString(first(record, ['prompt', 'inputText', 'text']) ?? first(inputInfo, ['preview', 'text']), 'İsimsiz düşünce'),
    snapshotId: asString(read(['snapshotId', 'modelSnapshotId']) ?? first(base, ['snapshotId'])) || null,
    snapshotName: asString(first(record, ['snapshotName', 'modelSnapshotName'])) || undefined,
    teacherMode: normalizeTeacherMode(read(['teacherMode', 'teacher', 'guidanceMode']) ?? first(record, ['crossCheck'])),
    learningMode: normalizeLearningMode(first(record, ['learningMode'])),
    regionMode: normalizeRegionMode(first(record, ['regionMode'])),
    requestedRegionMode: normalizeRegionMode(first(record, ['requestedRegionMode'])),
    regionRuntime: normalizeRegionRuntime(first(record, ['regionRuntime'])),
    mode: asString(first(record, ['mode']), 'research'),
    commitEligible: asBoolean(first(record, ['commitEligible']), false),
    commitStatus: asString(first(record, ['commitStatus']), 'not-eligible'),
    review: Object.keys(reviewRecord).length ? {
      decision: asString(first(reviewRecord, ['decision']), 'unknown'),
      reason: asString(first(reviewRecord, ['reason'])),
      reviewedAt: asString(first(reviewRecord, ['reviewedAt', 'at'])) || undefined,
    } : null,
    candidateSnapshotId: asString(first(record, ['candidateSnapshotId']) ?? first(artifacts, ['candidateSnapshotId'])) || null,
    seed: Math.round(asFinite(read(['seed', 'randomSeed']), 0)),
    maxTicks: Math.max(1, Math.round(asFinite(read(['maxTicks', 'tickBudget', 'ticks']), Math.max(1, tickCount)))),
    tickCount,
    currentTick: Math.max(0, Math.round(asFinite(first(record, ['currentTick', 'tickIndex']), Math.max(0, tickCount - 1)))),
    state: normalizeState(first(record, ['state', 'status'])),
    createdAt: asString(first(record, ['createdAt', 'startedAt'])) || undefined,
    updatedAt: asString(first(record, ['updatedAt', 'finishedAt'])) || undefined,
    resultText: asString(first(record, ['resultText', 'output', 'result']) ?? first(engine, ['explanation'])) || undefined,
    stopReason: asString(first(record, ['stopReason', 'reason'])) || undefined,
    structuralScore: readMetric(['structuralScore', 'structureScore', 'outcomeScore', 'graphF1']),
    trust: readMetric(['trust', 'trustScore', 'confidence', 'credit']) ??
      nullableFinite(first(outcomeEvidence, ['trustWeight', 'confidence'])),
  }
}

export function normalizeEpisode(input: unknown): ThoughtEpisode {
  const record = unwrapEpisode(input)
  const summary = normalizeEpisodeSummary(record)
  const ticks = asArray(first(record, ['ticks', 'steps']))
    .map(normalizeTick)
    .sort((left, right) => left.index - right.index)
  return {
    ...summary,
    tickCount: Math.max(summary.tickCount, ticks.length),
    currentTick: ticks.length ? ticks[ticks.length - 1]!.index : summary.currentTick,
    ticks,
    warnings: asArray(first(record, ['warnings', 'notices']))
      .map((item) => typeof item === 'string' ? item : asString(first(asRecord(item), ['message', 'title'])))
      .filter(Boolean),
  }
}

export function normalizeEpisodeList(input: unknown): ThoughtEpisodeSummary[] {
  const record = asRecord(input)
  return asArray(first(record, ['episodes', 'items', 'data']) ?? input)
    .map(normalizeEpisodeSummary)
}

export function normalizeTickList(input: unknown): ThoughtTick[] {
  const record = asRecord(input)
  return asArray(first(record, ['ticks', 'items', 'data']) ?? input)
    .map(normalizeTick)
    .sort((left, right) => left.index - right.index)
}

export function normalizeStatus(input: unknown): ThoughtServiceStatus {
  const outer = asRecord(input)
  const record = asRecord(first(outer, ['status', 'data']) ?? outer)
  const counts = asRecord(first(record, ['counts']))
  const limits = asRecord(first(record, ['limits']))
  return {
    available: asBoolean(first(record, ['available', 'ready', 'ok']), true),
    enabled: first(record, ['enabled']) === undefined ? undefined : asBoolean(first(record, ['enabled'])),
    researchBaseAvailable: first(record, ['researchBaseAvailable']) === undefined ? undefined : asBoolean(first(record, ['researchBaseAvailable'])),
    activeSnapshotId: asString(first(record, ['activeSnapshotId'])) || null,
    maximumTicks: nullableFinite(first(limits, ['maximumTicks'])),
    version: asString(first(record, ['version', 'serviceVersion'])) || undefined,
    state: asString(first(record, ['state', 'status'])) || undefined,
    activeEpisodes: Math.max(0, Math.round(asFinite(first(record, ['activeEpisodes', 'runningEpisodes', 'active']) ?? first(counts, ['running']), 0))),
    queuedEpisodes: Math.max(0, Math.round(asFinite(first(record, ['queuedEpisodes', 'queued']) ?? first(counts, ['queued']), 0))),
    message: asString(first(record, ['message', 'detail'])) || undefined,
  }
}

export function normalizeSnapshots(input: unknown): ThoughtSnapshotOption[] {
  const record = asRecord(input)
  return asArray(first(record, ['snapshots', 'items', 'data']) ?? input).map((value, index) => {
    const item = asRecord(value)
    const state = asString(first(item, ['state', 'status']))
    const id = asString(first(item, ['id', 'snapshotId']), `snapshot-${index + 1}`)
    return {
      id,
      name: asString(first(item, ['name', 'label', 'title']), id),
      active: asBoolean(first(item, ['active', 'isActive']), state === 'active'),
      commitEligible: asBoolean(first(item, ['commitEligible']), false),
      state: state || undefined,
      createdAt: asString(first(item, ['createdAt', 'timestamp'])) || undefined,
    }
  })
}

export function normalizeReplay(input: unknown): ReplayVerification {
  const outer = asRecord(input)
  const record = asRecord(first(outer, ['replay', 'verification', 'data']) ?? outer)
  const matchesRaw = first(record, ['matches', 'matched', 'match', 'identical', 'deterministic'])
  const mismatch = asRecord(first(record, ['firstMismatch']))
  return {
    ok: asBoolean(first(outer, ['ok', 'accepted']), true),
    matches: matchesRaw === undefined ? null : asBoolean(matchesRaw),
    message: asString(first(record, ['message', 'summary']), matchesRaw === true ? 'Tekrar aynı akışı üretti.' : 'Tekrar doğrulaması tamamlandı.'),
    expectedHash: asString(first(record, ['expectedHash', 'originalHash']) ?? first(mismatch, ['expectedDigest'])) || undefined,
    actualHash: asString(first(record, ['actualHash', 'replayHash']) ?? first(mismatch, ['actualDigest'])) || undefined,
    firstMismatchTick: nullableFinite(first(record, ['firstMismatchTick', 'mismatchTick']) ?? first(mismatch, ['expectedIndex', 'actualIndex', 'position'])),
  }
}

export function normalizeGraph(input: unknown, requestedLevel: ThoughtGraphLevel): ThoughtGraph {
  const outer = asRecord(input)
  const record = asRecord(first(outer, ['graph', 'data']) ?? outer)
  const audit = asRecord(first(record, ['audit']))
  const evaluation = asRecord(first(audit, ['evaluation']))
  const languageOutput = asRecord(first(audit, ['languageOutput']))
  const languageError = asRecord(first(languageOutput, ['error']))
  const nodes = asArray(first(record, ['nodes', 'vertices', 'items'])).map((value, index) => {
    const item = asRecord(value)
    const id = asString(first(item, ['id', 'nodeId', 'familyId', 'neuronId']), `dugum-${index + 1}`)
    return {
      id,
      label: asString(first(item, ['label', 'name', 'text']), id),
      kind: asString(first(item, ['kind', 'type', 'region'])) || undefined,
      weight: nullableFinite(first(item, ['weight', 'score', 'activation', 'strength'])),
    }
  })
  const edges = asArray(first(record, ['edges', 'links', 'transitions'])).map((value, index) => {
    const item = asRecord(value)
    const from = asString(first(item, ['from', 'source', 'fromId', 'sourceId', 'fromFamilyId', 'fromMemberKey']), '?')
    const to = asString(first(item, ['to', 'target', 'toId', 'targetId', 'toFamilyId', 'toMemberKey']), '?')
    const probability = nullableFinite(first(item, ['weight', 'probability', 'score', 'strength']))
    const excitation = nullableFinite(first(item, ['excitation']))
    const inhibition = nullableFinite(first(item, ['inhibition']))
    return {
      id: asString(first(item, ['id', 'edgeId']), `${from}->${to}-${index}`),
      from,
      to,
      label: asString(first(item, ['label', 'type', 'relation', 'kind'])) || undefined,
      weight: probability ?? (excitation !== null ? excitation - (inhibition ?? 0) : inhibition !== null ? -inhibition : null),
    }
  }).sort((left, right) => (right.weight ?? Number.NEGATIVE_INFINITY) - (left.weight ?? Number.NEGATIVE_INFINITY))
  const levelValue = asString(first(record, ['level', 'graphLevel']))
  return {
    level: levelValue === 'family' || levelValue === 'neuron' ? levelValue : requestedLevel,
    nodes,
    edges,
    totalNodeCount: Math.max(nodes.length, Math.round(asFinite(first(record, ['totalNodeCount', 'nodeCount', 'totalNodes']), nodes.length))),
    totalEdgeCount: Math.max(edges.length, Math.round(asFinite(first(record, ['totalEdgeCount', 'edgeCount', 'totalEdges']), edges.length))),
    truncated: asBoolean(first(record, ['truncated', 'hasMore']), false),
    episodeId: asString(first(audit, ['episodeId'])) || null,
    structuralOutcome: normalizeStructuralOutcome(first(audit, ['eventGraph', 'structuralOutcome'])),
    structuralEvaluation: Object.keys(evaluation).length ? {
      graphF1: nullableFinite(first(evaluation, ['graphF1'])),
      constraintAccuracy: nullableFinite(first(evaluation, ['constraintAccuracy'])),
      resultScore: nullableFinite(first(evaluation, ['resultScore', 'score'])),
    } : null,
    languageOutput: Object.keys(languageOutput).length ? {
      text: asString(first(languageOutput, ['text'])),
      stoppedBy: asString(first(languageOutput, ['stoppedBy']), 'unmatched'),
      scaffoldFingerprint: asString(first(languageOutput, ['scaffoldFingerprint'])) || null,
      headFingerprint: asString(first(languageOutput, ['headFingerprint'])) || null,
      steps: asArray(first(languageOutput, ['steps'])).slice(0, 12).map((value) => {
        const step = asRecord(value)
        const token = asRecord(first(step, ['emittedToken']))
        const state = asRecord(first(step, ['state']))
        return {
          familyId: asString(first(step, ['familyId'])),
          emittedSurface: asString(first(token, ['surface'])) || null,
          explanation: asString(first(step, ['explanation'])),
          complete: asBoolean(first(state, ['complete']), false),
        }
      }),
      error: asString(first(languageError, ['message'])) || null,
    } : null,
  }
}

export function mergeEpisodeTicks(episode: ThoughtEpisode, ticks: ThoughtTick[]): ThoughtEpisode {
  if (!ticks.length) return episode
  const byIndex = new Map<number, ThoughtTick>()
  episode.ticks.forEach((tick) => byIndex.set(tick.index, tick))
  ticks.forEach((tick) => byIndex.set(tick.index, tick))
  const merged = [...byIndex.values()].sort((left, right) => left.index - right.index)
  return {
    ...episode,
    ticks: merged,
    tickCount: Math.max(episode.tickCount, merged.length),
    currentTick: merged.at(-1)?.index ?? episode.currentTick,
  }
}

export function statusLabel(state: ThoughtEpisodeState): string {
  return ({
    queued: 'Sırada',
    running: 'Akış sürüyor',
    paused: 'Duraklatıldı',
    completed: 'Tamamlandı',
    failed: 'Hata',
    cancelled: 'İptal edildi',
    unknown: 'Durum bekleniyor',
  } satisfies Record<ThoughtEpisodeState, string>)[state]
}

export function teacherModeLabel(mode: TeacherMode): string {
  return mode === 'off' ? 'Öğretmensiz' : 'Ollama karşılaştırmalı'
}

export function formatRatio(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  if (value >= 0 && value <= 1) return `%${Math.round(value * 100)}`
  return value.toLocaleString('tr-TR', { maximumFractionDigits: 3 })
}

export function scoreBar(value: number | null | undefined): number {
  if (value === null || value === undefined || !Number.isFinite(value)) return 0
  if (value >= 0 && value <= 1) return value
  return clamp01((value + 1) / 2)
}

export function canRequestLearningProposal(status: ThoughtServiceStatus | null, selectedSnapshotId: string): boolean {
  if (!status || status.available === false || status.enabled === false || !status.activeSnapshotId) return false
  return !selectedSnapshotId || selectedSnapshotId === status.activeSnapshotId
}
