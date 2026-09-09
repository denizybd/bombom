import type { FamilyRun } from '../families/types'
import {
  boundedLogOdds,
  compareText,
  deepSerializableClone,
  normalizePositive,
} from '../language-experience/deterministic'
import { targetMemberWeights } from '../language-experience/examples'
import { validateLanguageExperienceState } from '../language-experience/state'
import {
  LANGUAGE_EOS_MEMBER,
  type LanguageDataset,
  type LanguageExample,
  type LanguageExperienceState,
  type SurfacePiece,
} from '../language-experience/types'
import {
  classifyLanguageOutputSeed,
  LANGUAGE_OUTPUT_FAMILY_IDS,
  LANGUAGE_ROLE_GATEWAY_ID,
  validateLanguageOutputScaffold,
} from './scaffold'
import {
  computeLanguageOutputNurseryFingerprint,
  computeLanguageOutputProvisionalNeuronFingerprint,
  finalizeLanguageOutputHead,
  sortLanguageOutputEdges,
  sortLanguageOutputMemberships,
  validateLanguageOutputHead,
} from './state'
import type {
  LanguageOutputEvidenceEdge,
  LanguageOutputEvidenceKind,
  LanguageOutputFamilyCandidateEvidence,
  LanguageOutputFamilyId,
  LanguageOutputFamilyRunReference,
  LanguageOutputHeadV1,
  LanguageOutputLearningConfig,
  LanguageOutputMembershipEvidence,
  LanguageOutputNurseryProposal,
  LanguageOutputProvisionalNeuronV1,
  LanguageOutputScaffoldV1,
  LanguageOutputSeedClassification,
  LanguageOutputTrainingOptions,
  LanguageOutputTrainingResult,
} from './types'

function addUnique(values: string[], value: string): void {
  if (!values.includes(value)) {
    values.push(value)
    values.sort(compareText)
  }
}

function evidenceUtility(value: { score: number; updateCount: number }): number {
  return Math.abs(value.score) + 0.05 * Math.log1p(value.updateCount)
}

class EdgeStore {
  readonly values: LanguageOutputEvidenceEdge[]
  private readonly byKey = new Map<string, LanguageOutputEvidenceEdge>()
  private readonly byRow = new Map<string, LanguageOutputEvidenceEdge[]>()

  constructor(initial: readonly LanguageOutputEvidenceEdge[], private readonly maximumOutgoing: number) {
    this.values = deepSerializableClone([...initial])
    for (const edge of this.values) this.index(edge)
  }

  private edgeKey(kind: LanguageOutputEvidenceKind, fromKey: string, toKey: string): string {
    return `${kind}\0${fromKey}\0${toKey}`
  }

  private rowKey(kind: LanguageOutputEvidenceKind, fromKey: string): string {
    return `${kind}\0${fromKey}`
  }

  private index(edge: LanguageOutputEvidenceEdge): void {
    this.byKey.set(this.edgeKey(edge.kind, edge.fromKey, edge.toKey), edge)
    const rowKey = this.rowKey(edge.kind, edge.fromKey)
    const row = this.byRow.get(rowKey) ?? []
    row.push(edge)
    this.byRow.set(rowKey, row)
  }

  decay(kind: LanguageOutputEvidenceKind, fromKey: string, amount: number): void {
    if (amount <= 0) return
    for (const edge of this.byRow.get(this.rowKey(kind, fromKey)) ?? []) {
      edge.excitatoryEvidence *= 1 - amount
      edge.inhibitoryEvidence *= 1 - amount
      edge.score = boundedLogOdds(edge.excitatoryEvidence, edge.inhibitoryEvidence)
    }
  }

  update(input: {
    kind: LanguageOutputEvidenceKind
    fromKey: string
    toKey: string
    excitation: number
    inhibition: number
    sourceDigest: string
    sourceGroupId: string
    ceiling: number
  }): void {
    const key = this.edgeKey(input.kind, input.fromKey, input.toKey)
    let edge = this.byKey.get(key)
    if (!edge) {
      edge = {
        kind: input.kind,
        fromKey: input.fromKey,
        toKey: input.toKey,
        excitatoryEvidence: 0,
        inhibitoryEvidence: 0,
        score: 0,
        updateCount: 0,
        sourceDigests: [],
        sourceGroupIds: [],
      }
      this.values.push(edge)
      this.index(edge)
    }
    edge.excitatoryEvidence = Math.min(input.ceiling, edge.excitatoryEvidence + Math.max(0, input.excitation))
    edge.inhibitoryEvidence = Math.min(input.ceiling, edge.inhibitoryEvidence + Math.max(0, input.inhibition))
    edge.updateCount += 1
    addUnique(edge.sourceDigests, input.sourceDigest)
    addUnique(edge.sourceGroupIds, input.sourceGroupId)
    edge.score = boundedLogOdds(edge.excitatoryEvidence, edge.inhibitoryEvidence)
    this.prune(input.kind, input.fromKey)
  }

  private prune(kind: LanguageOutputEvidenceKind, fromKey: string): void {
    const rowKey = this.rowKey(kind, fromKey)
    const row = this.byRow.get(rowKey) ?? []
    if (row.length <= this.maximumOutgoing) return
    row.sort((left, right) =>
      evidenceUtility(right) - evidenceUtility(left) || compareText(left.toKey, right.toKey),
    )
    const removed = new Set(row.splice(this.maximumOutgoing))
    for (const edge of removed) this.byKey.delete(this.edgeKey(edge.kind, edge.fromKey, edge.toKey))
    for (let index = this.values.length - 1; index >= 0; index -= 1) {
      if (removed.has(this.values[index])) this.values.splice(index, 1)
    }
  }
}

class MembershipStore {
  readonly values: LanguageOutputMembershipEvidence[]
  private readonly byKey = new Map<string, LanguageOutputMembershipEvidence>()
  private readonly byMember = new Map<string, LanguageOutputMembershipEvidence[]>()

  constructor(initial: readonly LanguageOutputMembershipEvidence[], private readonly maximumPerMember: number) {
    this.values = deepSerializableClone([...initial])
    for (const membership of this.values) this.index(membership)
  }

  private key(memberKey: string, familyId: LanguageOutputFamilyId): string {
    return `${memberKey}\0${familyId}`
  }

  private index(membership: LanguageOutputMembershipEvidence): void {
    this.byKey.set(this.key(membership.memberKey, membership.familyId), membership)
    const values = this.byMember.get(membership.memberKey) ?? []
    values.push(membership)
    this.byMember.set(membership.memberKey, values)
  }

  update(input: {
    memberKey: string
    familyId: LanguageOutputFamilyId
    excitation: number
    inhibition: number
    confidence: number
    sourceDigest: string
    sourceGroupId: string
    ceiling: number
  }): void {
    const key = this.key(input.memberKey, input.familyId)
    let membership = this.byKey.get(key)
    if (!membership) {
      membership = {
        memberKey: input.memberKey,
        familyId: input.familyId,
        excitatoryEvidence: 0,
        inhibitoryEvidence: 0,
        score: 0,
        updateCount: 0,
        sourceDigests: [],
        sourceGroupIds: [],
        classifierConfidence: 0,
      }
      this.values.push(membership)
      this.index(membership)
    }
    membership.excitatoryEvidence = Math.min(input.ceiling, membership.excitatoryEvidence + Math.max(0, input.excitation))
    membership.inhibitoryEvidence = Math.min(input.ceiling, membership.inhibitoryEvidence + Math.max(0, input.inhibition))
    membership.updateCount += 1
    membership.classifierConfidence = Math.max(membership.classifierConfidence, input.confidence)
    addUnique(membership.sourceDigests, input.sourceDigest)
    addUnique(membership.sourceGroupIds, input.sourceGroupId)
    membership.score = boundedLogOdds(membership.excitatoryEvidence, membership.inhibitoryEvidence)
    this.prune(input.memberKey)
  }

  private prune(memberKey: string): void {
    const memberships = this.byMember.get(memberKey) ?? []
    if (memberships.length <= this.maximumPerMember) return
    memberships.sort((left, right) =>
      evidenceUtility(right) - evidenceUtility(left) || compareText(left.familyId, right.familyId),
    )
    const removed = new Set(memberships.splice(this.maximumPerMember))
    for (const membership of removed) this.byKey.delete(this.key(membership.memberKey, membership.familyId))
    for (let index = this.values.length - 1; index >= 0; index -= 1) {
      if (removed.has(this.values[index])) this.values.splice(index, 1)
    }
  }
}

function assertTrainingCompatibility(
  head: LanguageOutputHeadV1,
  scaffold: LanguageOutputScaffoldV1,
  dataset: LanguageDataset,
  languageState: LanguageExperienceState,
  familyRun?: LanguageOutputFamilyRunReference,
): void {
  validateLanguageOutputScaffold(scaffold)
  validateLanguageOutputHead(head, scaffold)
  validateLanguageExperienceState(languageState)
  if (
    dataset?.format !== 'language-experience-dataset' || dataset.version !== 1 || !dataset.fingerprint ||
    languageState?.format !== 'language-experience-state' || languageState.version !== 1 ||
    dataset.familyRunFingerprint !== languageState.familyRunFingerprint ||
    dataset.representationSpaceId !== languageState.representationSpaceId ||
    dataset.behaviorSpaceId !== languageState.behaviorSpaceId ||
    dataset.configFingerprint !== languageState.configFingerprint
  ) throw new TypeError('Dil çıktı eğitimi için dataset ve language-experience durumu uyumsuz.')
  validateDatasetStructure(dataset)
  if (familyRun && (
    familyRun.format !== 'noron-aileleri-run' || familyRun.version !== 1 ||
    familyRun.fingerprint !== dataset.familyRunFingerprint ||
    familyRun.representationSpaceId !== dataset.representationSpaceId
  )) throw new TypeError('Dil çıktı eğitimi verilen FamilyRun ile dataset arasında uyuşmazlık buldu.')
  if (
    (head.representationSpaceId !== null && head.representationSpaceId !== dataset.representationSpaceId) ||
    (head.familyRunFingerprint !== null && head.familyRunFingerprint !== dataset.familyRunFingerprint)
  ) throw new TypeError('Dil çıktı başlığı farklı bir FamilyRun veya temsil uzayına bağlı.')
  const assignments = new Map(head.sourceGroupSplits.map((assignment) => [assignment.independenceGroupId, assignment.split]))
  const languageAssignments = new Map(languageState.sourceGroupSplits.map((assignment) => (
    [assignment.independenceGroupId, assignment.split]
  )))
  for (const source of dataset.sources) {
    const previous = assignments.get(source.independenceGroupId)
    const languagePrevious = languageAssignments.get(source.independenceGroupId)
    if ((previous && previous !== source.split) || (languagePrevious && languagePrevious !== source.split)) {
      throw new TypeError('Dil çıktı eğitiminde bağımsız kaynak grubu önceki holdout bölümünden taşınamaz.')
    }
  }
}

function validateDatasetStructure(dataset: LanguageDataset): void {
  if (!Array.isArray(dataset.sources) || !Array.isArray(dataset.pieces) || !Array.isArray(dataset.examples)) {
    throw new TypeError('Dil çıktı dataset dizileri eksik.')
  }
  const validSplits = new Set(['train', 'validation', 'test'])
  const sourceIds = new Set<string>()
  const sourceByDigest = new Map<string, LanguageDataset['sources'][number]>()
  const groupSplits = new Map<string, string>()
  for (const source of dataset.sources) {
    const previousSplit = groupSplits.get(source.independenceGroupId)
    if (
      !source.sourceId || sourceIds.has(source.sourceId) || !source.sourceDigest ||
      sourceByDigest.has(source.sourceDigest) || !source.independenceGroupId ||
      !validSplits.has(source.split) || (previousSplit !== undefined && previousSplit !== source.split)
    ) throw new TypeError('Dil çıktı dataset kaynakları bozuk veya holdout grubu sızıyor.')
    sourceIds.add(source.sourceId)
    sourceByDigest.set(source.sourceDigest, source)
    groupSplits.set(source.independenceGroupId, source.split)
  }
  const pieceById = new Map<string, LanguageDataset['pieces'][number]>()
  for (const piece of dataset.pieces) {
    const source = sourceByDigest.get(piece.sourceDigest)
    if (!piece.id || pieceById.has(piece.id) || !source || source.sourceId !== piece.sourceId) {
      throw new TypeError('Dil çıktı dataset yüzey parçaları kaynağıyla uyuşmuyor.')
    }
    pieceById.set(piece.id, piece)
  }
  const exampleIds = new Set<string>()
  for (const example of dataset.examples) {
    const source = sourceByDigest.get(example.sourceDigest)
    const parent = pieceById.get(example.parentPieceId)
    const target = example.targetPieceId === null ? null : pieceById.get(example.targetPieceId)
    const prefix = example.prefixPieceIds.map((id) => pieceById.get(id))
    if (
      !example.id || exampleIds.has(example.id) || !source || source.sourceId !== example.sourceId ||
      source.split !== example.split || !parent || parent.sourceDigest !== example.sourceDigest ||
      prefix.some((piece) => !piece || piece.sourceDigest !== example.sourceDigest) ||
      (example.eos ? example.targetPieceId !== null : !target) ||
      (target !== null && target !== undefined && (
        target.sourceDigest !== example.sourceDigest || target.level !== example.granularity
      ))
    ) throw new TypeError('Dil çıktı dataset örneği kaynak/split/yüzey zinciriyle uyuşmuyor.')
    exampleIds.add(example.id)
  }
}

function normalizedClassifications(
  classifications: readonly LanguageOutputSeedClassification[],
  minimum: number,
): Array<LanguageOutputSeedClassification & { probability: number }> {
  return normalizePositive(classifications
    .filter((classification) => classification.confidence >= minimum)
    .map((classification) => ({ ...classification, score: classification.confidence })))
    .map(({ score: _score, ...classification }) => classification)
}

function bestBinding(piece: SurfacePiece | undefined): string | null {
  return piece?.bindings.slice().sort((left, right) =>
    right.score - left.score || compareText(left.memberKey, right.memberKey),
  )[0]?.memberKey ?? null
}

function classifyPiece(
  piece: SurfacePiece | undefined,
  minimum: number,
): Array<LanguageOutputSeedClassification & { probability: number }> {
  if (!piece) return []
  return normalizedClassifications(classifyLanguageOutputSeed({
    surface: piece.text,
    normalizedSurface: piece.normalizedText,
    granularity: piece.level,
  }), minimum)
}

interface MemberSurface {
  surface: string
  normalizedSurface: string
  granularity: LanguageExample['granularity']
  score: number
}

function surfaceByMember(dataset: LanguageDataset): Map<string, MemberSurface> {
  const values = new Map<string, MemberSurface>()
  for (const example of dataset.examples) {
    if (example.split !== 'train' || example.eos) continue
    for (const binding of example.targetBindings) {
      const candidate = {
        surface: example.targetText,
        normalizedSurface: example.targetNormalizedText,
        granularity: example.granularity,
        score: binding.score,
      }
      const current = values.get(binding.memberKey)
      if (
        !current || candidate.score > current.score ||
        (candidate.score === current.score && compareText(candidate.normalizedSurface, current.normalizedSurface) < 0)
      ) values.set(binding.memberKey, candidate)
    }
  }
  return values
}

function targetMemberDistribution(example: LanguageExample): Array<{ key: string; probability: number }> {
  if (example.eos) return [{ key: LANGUAGE_EOS_MEMBER, probability: 1 }]
  return normalizePositive(targetMemberWeights(example).map((target) => ({ key: target.key, score: target.weight })))
    .map(({ key, probability }) => ({ key, probability }))
}

function bindingDistribution(bindings: LanguageExample['corruptionBindings']): Array<{ key: string; probability: number }> {
  return normalizePositive(bindings.map((binding) => ({ key: binding.memberKey, score: binding.score })))
    .map(({ key, probability }) => ({ key, probability }))
}

function mergeSplitAssignments(head: LanguageOutputHeadV1, dataset: LanguageDataset) {
  return [...new Map([
    ...head.sourceGroupSplits.map((assignment) => [assignment.independenceGroupId, { ...assignment }] as const),
    ...dataset.sources.map((source) => [source.independenceGroupId, {
      independenceGroupId: source.independenceGroupId,
      split: source.split,
    }] as const),
  ]).values()].sort((left, right) => compareText(left.independenceGroupId, right.independenceGroupId))
}

function canonicalNursery(proposal: Omit<LanguageOutputNurseryProposal, 'fingerprint'>): LanguageOutputNurseryProposal {
  const base: Omit<LanguageOutputNurseryProposal, 'fingerprint'> = {
    ...proposal,
    sourceDigests: [...new Set(proposal.sourceDigests)].sort(compareText),
    sourceGroupIds: [...new Set(proposal.sourceGroupIds)].sort(compareText),
    exampleIds: [...new Set(proposal.exampleIds)].sort(compareText),
    sourceProposalIds: [...new Set(proposal.sourceProposalIds)].sort(compareText),
    familyCandidates: [...proposal.familyCandidates]
      .sort((left, right) => right.evidence - left.evidence || compareText(left.familyId, right.familyId)),
  }
  return { ...base, fingerprint: computeLanguageOutputNurseryFingerprint(base) }
}

function updateNursery(
  previous: readonly LanguageOutputNurseryProposal[],
  languageState: LanguageExperienceState,
  dataset: LanguageDataset,
  scaffold: LanguageOutputScaffoldV1,
  learningConfig: LanguageOutputLearningConfig,
): { nursery: LanguageOutputNurseryProposal[]; updates: number } {
  const trainExamples = new Map(dataset.examples.filter((example) => example.split === 'train').map((example) => [example.id, example]))
  const trainSources = new Map(dataset.sources.filter((source) => source.split === 'train').map((source) => [source.sourceDigest, source]))
  const bySurface = new Map(previous.map((proposal) => [proposal.normalizedSurface, deepSerializableClone(proposal)]))
  let updates = 0
  for (const sourceProposal of [...languageState.birthProposals].sort((left, right) => compareText(left.id, right.id))) {
    const existing = bySurface.get(sourceProposal.normalizedSurface)
    const seenExamples = new Set(existing?.exampleIds ?? [])
    const observations = sourceProposal.observations
      .filter((observation) => (
        trainExamples.has(observation.exampleId) && trainSources.has(observation.sourceDigest) && !seenExamples.has(observation.exampleId)
      ))
      .sort((left, right) => compareText(left.exampleId, right.exampleId))
    if (!observations.length) continue
    const familyEvidence = new Map<LanguageOutputFamilyId, { evidence: number; observationCount: number }>()
    for (const candidate of existing?.familyCandidates ?? []) {
      familyEvidence.set(candidate.familyId, { evidence: candidate.evidence, observationCount: candidate.observationCount })
    }
    const sourceDigests = [...(existing?.sourceDigests ?? [])]
    const sourceGroupIds = [...(existing?.sourceGroupIds ?? [])]
    const exampleIds = [...(existing?.exampleIds ?? [])]
    let surpriseTotal = (existing?.meanSurprise ?? 0) * (existing?.observationCount ?? 0)
    let noveltyTotal = (existing?.meanNovelty ?? 0) * (existing?.observationCount ?? 0)
    for (const observation of observations) {
      const source = trainSources.get(observation.sourceDigest)!
      const classifications = classifyLanguageOutputSeed({
        surface: observation.surface,
        normalizedSurface: observation.normalizedSurface,
        granularity: observation.granularity,
      })
      for (const classification of classifications) {
        const current = familyEvidence.get(classification.familyId) ?? { evidence: 0, observationCount: 0 }
        current.evidence += classification.confidence
        current.observationCount += 1
        familyEvidence.set(classification.familyId, current)
      }
      addUnique(sourceDigests, observation.sourceDigest)
      addUnique(sourceGroupIds, source.independenceGroupId)
      addUnique(exampleIds, observation.exampleId)
      surpriseTotal += observation.surprise
      noveltyTotal += observation.novelty
    }
    const observationCount = exampleIds.length
    const familyCandidates: LanguageOutputFamilyCandidateEvidence[] = [...familyEvidence]
      .map(([familyId, value]) => ({ familyId, ...value }))
      .filter((candidate) => scaffold.families.some((family) => family.id === candidate.familyId))
      .sort((left, right) => right.evidence - left.evidence || compareText(left.familyId, right.familyId))
    if (!familyCandidates.length) continue
    const status = observationCount >= learningConfig.nurseryMinimumObservations &&
      sourceGroupIds.length >= learningConfig.nurseryMinimumSourceGroups
      ? 'eligible' as const
      : 'observing' as const
    const proposal = canonicalNursery({
      id: existing?.id ?? `LON-${sourceProposal.id.replace(/^NBP-/u, '')}`,
      status,
      normalizedSurface: sourceProposal.normalizedSurface,
      representativeSurface: existing?.representativeSurface ?? sourceProposal.representativeSurface,
      observationCount,
      sourceDigests,
      sourceGroupIds,
      exampleIds,
      meanSurprise: surpriseTotal / Math.max(1, observationCount),
      meanNovelty: noveltyTotal / Math.max(1, observationCount),
      familyCandidates,
      sourceProposalIds: [...(existing?.sourceProposalIds ?? []), sourceProposal.id],
      requiresValidation: true,
    })
    bySurface.set(proposal.normalizedSurface, proposal)
    updates += observations.length
  }
  return {
    nursery: [...bySurface.values()].sort((left, right) =>
      Number(right.status === 'eligible') - Number(left.status === 'eligible') ||
      right.observationCount - left.observationCount || compareText(left.id, right.id),
    ),
    updates,
  }
}

function promoteNurseryNeurons(
  previous: readonly LanguageOutputProvisionalNeuronV1[],
  nursery: readonly LanguageOutputNurseryProposal[],
  config: LanguageOutputLearningConfig,
  nextRevision: number,
): { neurons: LanguageOutputProvisionalNeuronV1[]; promotions: number } {
  const previousBySurface = new Map(previous.map((neuron) => [
    neuron.normalizedSurface,
    deepSerializableClone(neuron),
  ]))
  const neurons: LanguageOutputProvisionalNeuronV1[] = []
  for (const proposal of nursery) {
    if (
      proposal.status !== 'eligible' ||
      proposal.observationCount < config.nurseryMinimumObservations ||
      proposal.sourceGroupIds.length < config.nurseryMinimumSourceGroups
    ) continue
    const strongest = [...proposal.familyCandidates].sort((left, right) => (
      right.evidence - left.evidence || compareText(left.familyId, right.familyId)
    ))[0]
    if (!strongest) continue
    const meanFamilyConfidence = strongest.evidence / Math.max(1, proposal.observationCount)
    if (meanFamilyConfidence < config.classifierMinimumConfidence) continue
    const existing = previousBySurface.get(proposal.normalizedSurface)
    const base: Omit<LanguageOutputProvisionalNeuronV1, 'fingerprint'> = {
      id: existing?.id ?? `LOPN-${proposal.id.replace(/^LON-/u, '')}`,
      status: 'provisional',
      kind: 'surface-controller',
      normalizedSurface: proposal.normalizedSurface,
      representativeSurface: proposal.representativeSurface,
      familyId: strongest.familyId,
      familyEvidence: strongest.evidence,
      meanFamilyConfidence,
      observationCount: proposal.observationCount,
      sourceDigests: [...proposal.sourceDigests],
      sourceGroupIds: [...proposal.sourceGroupIds],
      exampleIds: [...proposal.exampleIds],
      sourceProposalIds: [...proposal.sourceProposalIds],
      promotedAtRevision: existing?.promotedAtRevision ?? nextRevision,
      reversible: true,
      physicalIdentityAllocated: false,
    }
    neurons.push({ ...base, fingerprint: computeLanguageOutputProvisionalNeuronFingerprint(base) })
  }
  neurons.sort((left, right) => (
    right.sourceGroupIds.length - left.sourceGroupIds.length ||
    right.observationCount - left.observationCount ||
    compareText(left.normalizedSurface, right.normalizedSurface) || compareText(left.id, right.id)
  ))
  const bounded = neurons.slice(0, config.maximumProvisionalNeurons)
  return {
    neurons: bounded,
    promotions: bounded.filter((neuron) => !previousBySurface.has(neuron.normalizedSurface)).length,
  }
}

/**
 * Learns only from the language-experience train split. The scaffold and
 * FamilyRun are read-only; all plastic evidence is returned in a new head.
 */
export function trainLanguageOutputHead(
  head: LanguageOutputHeadV1,
  scaffold: LanguageOutputScaffoldV1,
  dataset: LanguageDataset,
  languageState: LanguageExperienceState,
  familyRun?: Pick<FamilyRun, 'format' | 'version' | 'fingerprint' | 'representationSpaceId'>,
  options: LanguageOutputTrainingOptions = {},
): LanguageOutputTrainingResult {
  assertTrainingCompatibility(head, scaffold, dataset, languageState, familyRun)
  if (head.trainedDatasetFingerprints.includes(dataset.fingerprint) && options.allowDatasetReplay !== true) {
    return {
      head: deepSerializableClone(head),
      processedExampleCount: 0,
      nurseryProposalCount: head.nursery.length,
      provisionalNeuronCount: head.outputNeurons.length,
      promotedNeuronCount: 0,
      replaySkipped: true,
    }
  }
  const edgeStore = new EdgeStore(head.transitionEvidence, head.config.maximumOutgoingEdges)
  const membershipStore = new MembershipStore(head.membershipEvidence, head.config.maximumMembershipsPerMember)
  const pieceById = new Map(dataset.pieces.map((piece) => [piece.id, piece]))
  const sourceByDigest = new Map(dataset.sources.map((source) => [source.sourceDigest, source]))
  const trainingExamples = dataset.examples
    .filter((example) => example.split === 'train' && scaffold.families.some((family) => (
      family.acceptedSpanLevels.includes(example.granularity)
    )))
    .sort((left, right) => compareText(left.sourceDigest, right.sourceDigest) || compareText(left.id, right.id))
  const trainMemberKeys = new Set(trainingExamples.flatMap((example) => (
    example.targetBindings.map((binding) => binding.memberKey)
  )))
  const memberSurfaces = surfaceByMember(dataset)
  let positiveEdgeUpdates = 0
  let negativeEdgeUpdates = 0
  let membershipUpdates = 0
  let skippedUnclassified = 0

  for (const example of trainingExamples) {
    const source = sourceByDigest.get(example.sourceDigest)
    if (!source || source.split !== 'train') throw new TypeError('Train örneği train dışı kaynak taşıyor.')
    const sourceGroupId = source.independenceGroupId
    const previousPiece = pieceById.get(example.prefixPieceIds.at(-1) ?? '')
    const memberOrigin = bestBinding(previousPiece) ?? LANGUAGE_ROLE_GATEWAY_ID
    const familyOrigins = previousPiece
      ? classifyPiece(previousPiece, head.config.classifierMinimumConfidence)
      : [{
          familyId: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder,
          confidence: 1,
          membershipEligible: false,
          reason: 'BOS rol kapısı.',
          probability: 1,
        }]
    const familyTargets = normalizedClassifications(classifyLanguageOutputSeed({
      surface: example.targetText,
      normalizedSurface: example.targetNormalizedText,
      granularity: example.granularity,
      eos: example.eos,
    }), head.config.classifierMinimumConfidence)
    const memberTargets = targetMemberDistribution(example)
    const corruptionTargets = bindingDistribution(example.corruptionBindings.filter((binding) => (
      trainMemberKeys.has(binding.memberKey)
    )))
    if (!familyTargets.length) skippedUnclassified += 1

    edgeStore.decay('member-transition', memberOrigin, head.config.evidenceDecay)
    for (const target of memberTargets) {
      edgeStore.update({
        kind: 'member-transition', fromKey: memberOrigin, toKey: target.key,
        excitation: head.config.learningRate * target.probability, inhibition: 0,
        sourceDigest: example.sourceDigest, sourceGroupId, ceiling: head.config.evidenceCeiling,
      })
      positiveEdgeUpdates += 1
    }
    for (const corrupt of corruptionTargets) {
      edgeStore.update({
        kind: 'member-transition', fromKey: memberOrigin, toKey: corrupt.key,
        excitation: 0, inhibition: head.config.inhibitionRate * corrupt.probability,
        sourceDigest: example.sourceDigest, sourceGroupId, ceiling: head.config.evidenceCeiling,
      })
      negativeEdgeUpdates += 1
    }

    for (const origin of familyOrigins) {
      const originKey = previousPiece ? origin.familyId : LANGUAGE_ROLE_GATEWAY_ID
      edgeStore.decay('family-transition', originKey, head.config.evidenceDecay)
      for (const target of familyTargets) {
        edgeStore.update({
          kind: 'family-transition', fromKey: originKey, toKey: target.familyId,
          excitation: head.config.learningRate * origin.probability * target.probability,
          inhibition: 0, sourceDigest: example.sourceDigest, sourceGroupId,
          ceiling: head.config.evidenceCeiling,
        })
        positiveEdgeUpdates += 1
      }
      const corruptFamilies = new Map<LanguageOutputFamilyId, number>()
      for (const corrupt of corruptionTargets) {
        const surface = memberSurfaces.get(corrupt.key)
        if (!surface) continue
        for (const classification of normalizedClassifications(classifyLanguageOutputSeed({
          surface: surface.surface,
          normalizedSurface: surface.normalizedSurface,
          granularity: surface.granularity,
        }), head.config.classifierMinimumConfidence)) {
          corruptFamilies.set(
            classification.familyId,
            (corruptFamilies.get(classification.familyId) ?? 0) + corrupt.probability * classification.probability,
          )
        }
      }
      for (const [familyId, weight] of [...corruptFamilies].sort(([left], [right]) => compareText(left, right))) {
        if (familyTargets.some((target) => target.familyId === familyId)) continue
        edgeStore.update({
          kind: 'family-transition', fromKey: originKey, toKey: familyId,
          excitation: 0, inhibition: head.config.inhibitionRate * origin.probability * weight,
          sourceDigest: example.sourceDigest, sourceGroupId, ceiling: head.config.evidenceCeiling,
        })
        negativeEdgeUpdates += 1
      }
    }

    if (!example.eos) {
      for (const binding of [...example.targetBindings].sort((left, right) => compareText(left.memberKey, right.memberKey))) {
        for (const classification of familyTargets.filter((candidate) => candidate.membershipEligible)) {
          membershipStore.update({
            memberKey: binding.memberKey,
            familyId: classification.familyId,
            excitation: head.config.learningRate * binding.score * classification.probability,
            inhibition: 0,
            confidence: classification.confidence,
            sourceDigest: example.sourceDigest,
            sourceGroupId,
            ceiling: head.config.evidenceCeiling,
          })
          membershipUpdates += 1
          if (classification.confidence >= 0.9) {
            for (const alternative of scaffold.families
              .filter((family) => family.emitsSurface && family.id !== classification.familyId && family.id !== LANGUAGE_OUTPUT_FAMILY_IDS.eos)
              .sort((left, right) => compareText(left.id, right.id))) {
              membershipStore.update({
                memberKey: binding.memberKey,
                familyId: alternative.id,
                excitation: 0,
                inhibition: head.config.inhibitionRate * 0.2 * binding.score * classification.probability,
                confidence: classification.confidence,
                sourceDigest: example.sourceDigest,
                sourceGroupId,
                ceiling: head.config.evidenceCeiling,
              })
              membershipUpdates += 1
            }
          }
        }
      }
    }
  }

  const nurseryResult = updateNursery(head.nursery, languageState, dataset, scaffold, head.config)
  const boundedNursery = nurseryResult.nursery.slice(0, head.config.maximumNurseryProposals)
  const promotionResult = promoteNurseryNeurons(
    head.outputNeurons,
    boundedNursery,
    head.config,
    head.revision + 1,
  )
  const {
    fingerprint: _fingerprint,
    parentFingerprint: _parentFingerprint,
    revision: _revision,
    ...headBase
  } = head
  const next = finalizeLanguageOutputHead(head, scaffold, {
    ...headBase,
    representationSpaceId: head.representationSpaceId ?? dataset.representationSpaceId,
    familyRunFingerprint: head.familyRunFingerprint ?? dataset.familyRunFingerprint,
    transitionEvidence: sortLanguageOutputEdges(edgeStore.values),
    membershipEvidence: sortLanguageOutputMemberships(membershipStore.values),
    nursery: boundedNursery,
    outputNeurons: promotionResult.neurons,
    sourceGroupSplits: mergeSplitAssignments(head, dataset),
    trainedDatasetFingerprints: [...head.trainedDatasetFingerprints, dataset.fingerprint],
    training: {
      revision: head.training.revision,
      processedExamples: head.training.processedExamples + trainingExamples.length,
      positiveEdgeUpdates: head.training.positiveEdgeUpdates + positiveEdgeUpdates,
      negativeEdgeUpdates: head.training.negativeEdgeUpdates + negativeEdgeUpdates,
      membershipUpdates: head.training.membershipUpdates + membershipUpdates,
      nurseryUpdates: head.training.nurseryUpdates + nurseryResult.updates,
      provisionalNeuronPromotions: head.training.provisionalNeuronPromotions + promotionResult.promotions,
      skippedUnclassified: head.training.skippedUnclassified + skippedUnclassified,
      lastDatasetFingerprint: dataset.fingerprint,
    },
  })
  return {
    head: next,
    processedExampleCount: trainingExamples.length,
    nurseryProposalCount: boundedNursery.length,
    provisionalNeuronCount: promotionResult.neurons.length,
    promotedNeuronCount: promotionResult.promotions,
    replaySkipped: false,
  }
}
