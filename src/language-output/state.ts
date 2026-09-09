import { stableDigest } from '../families/sparse'
import type { FamilyRun } from '../families/types'
import { boundedLogOdds, compareText } from '../language-experience/deterministic'
import {
  LANGUAGE_OUTPUT_FAMILY_IDS,
  LANGUAGE_ROLE_GATEWAY_ID,
  validateLanguageOutputScaffold,
} from './scaffold'
import type {
  LanguageOutputEvidenceEdge,
  LanguageOutputHeadV1,
  LanguageOutputLearningConfig,
  LanguageOutputMembershipEvidence,
  LanguageOutputNurseryProposal,
  LanguageOutputProvisionalNeuronV1,
  LanguageOutputScaffoldV1,
} from './types'

export const DEFAULT_LANGUAGE_OUTPUT_LEARNING_CONFIG: Readonly<LanguageOutputLearningConfig> = Object.freeze({
  learningRate: 0.18,
  inhibitionRate: 0.07,
  evidenceDecay: 0.0005,
  evidenceCeiling: 1_000,
  maximumOutgoingEdges: 96,
  maximumMembershipsPerMember: 6,
  maximumNurseryProposals: 2_048,
  maximumProvisionalNeurons: 2_048,
  classifierMinimumConfidence: 0.55,
  nurseryMinimumObservations: 3,
  nurseryMinimumSourceGroups: 2,
})

function outputDigest(label: string, ...parts: readonly unknown[]): string {
  return `${label}-${stableDigest(['language-output-v1', label, ...parts]).toUpperCase()}`
}

export function computeLanguageOutputConfigFingerprint(config: LanguageOutputLearningConfig): string {
  return outputDigest('language-output-config', config)
}

export function createLanguageOutputLearningConfig(
  overrides: Partial<LanguageOutputLearningConfig> = {},
): LanguageOutputLearningConfig {
  const config = { ...DEFAULT_LANGUAGE_OUTPUT_LEARNING_CONFIG, ...overrides }
  validateLanguageOutputLearningConfig(config)
  return config
}

export function validateLanguageOutputLearningConfig(config: LanguageOutputLearningConfig): void {
  const positive = [
    config.learningRate,
    config.inhibitionRate,
    config.evidenceCeiling,
  ]
  if (positive.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError('Dil çıktı öğrenme oranları ve kanıt tavanı pozitif olmalıdır.')
  }
  const units = [config.evidenceDecay, config.classifierMinimumConfidence]
  if (units.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new RangeError('Dil çıktı birim eşikleri 0 ile 1 arasında olmalıdır.')
  }
  const integers = [
    config.maximumOutgoingEdges,
    config.maximumMembershipsPerMember,
    config.maximumNurseryProposals,
    config.maximumProvisionalNeurons,
    config.nurseryMinimumObservations,
    config.nurseryMinimumSourceGroups,
  ]
  if (integers.some((value) => !Number.isInteger(value) || value < 1)) {
    throw new RangeError('Dil çıktı kapasite ve doğum eşikleri pozitif tam sayı olmalıdır.')
  }
}

export function sortLanguageOutputEdges(
  edges: readonly LanguageOutputEvidenceEdge[],
): LanguageOutputEvidenceEdge[] {
  return [...edges]
    .map((edge) => ({
      ...edge,
      sourceDigests: [...edge.sourceDigests].sort(compareText),
      sourceGroupIds: [...edge.sourceGroupIds].sort(compareText),
    }))
    .sort((left, right) =>
      compareText(left.kind, right.kind) || compareText(left.fromKey, right.fromKey) ||
      right.score - left.score || compareText(left.toKey, right.toKey),
    )
}

export function sortLanguageOutputMemberships(
  memberships: readonly LanguageOutputMembershipEvidence[],
): LanguageOutputMembershipEvidence[] {
  return [...memberships]
    .map((membership) => ({
      ...membership,
      sourceDigests: [...membership.sourceDigests].sort(compareText),
      sourceGroupIds: [...membership.sourceGroupIds].sort(compareText),
    }))
    .sort((left, right) => compareText(left.memberKey, right.memberKey) ||
      right.score - left.score || compareText(left.familyId, right.familyId))
}

export function computeLanguageOutputNurseryFingerprint(
  proposal: Omit<LanguageOutputNurseryProposal, 'fingerprint'>,
): string {
  return outputDigest('language-output-nursery', proposal)
}

export function computeLanguageOutputProvisionalNeuronFingerprint(
  neuron: Omit<LanguageOutputProvisionalNeuronV1, 'fingerprint'>,
): string {
  return outputDigest('language-output-provisional-neuron', neuron)
}

function sortedNursery(nursery: readonly LanguageOutputNurseryProposal[]): LanguageOutputNurseryProposal[] {
  return [...nursery]
    .map((proposal) => ({
      ...proposal,
      sourceDigests: [...proposal.sourceDigests].sort(compareText),
      sourceGroupIds: [...proposal.sourceGroupIds].sort(compareText),
      exampleIds: [...proposal.exampleIds].sort(compareText),
      sourceProposalIds: [...proposal.sourceProposalIds].sort(compareText),
      familyCandidates: [...proposal.familyCandidates]
        .map((candidate) => ({ ...candidate }))
        .sort((left, right) => right.evidence - left.evidence || compareText(left.familyId, right.familyId)),
    }))
    .sort((left, right) =>
      Number(right.status === 'eligible') - Number(left.status === 'eligible') ||
      right.observationCount - left.observationCount || compareText(left.id, right.id),
    )
}

function sortedOutputNeurons(
  neurons: readonly LanguageOutputProvisionalNeuronV1[],
): LanguageOutputProvisionalNeuronV1[] {
  return [...neurons]
    .map((neuron) => ({
      ...neuron,
      sourceDigests: [...neuron.sourceDigests].sort(compareText),
      sourceGroupIds: [...neuron.sourceGroupIds].sort(compareText),
      exampleIds: [...neuron.exampleIds].sort(compareText),
      sourceProposalIds: [...neuron.sourceProposalIds].sort(compareText),
    }))
    .sort((left, right) => compareText(left.familyId, right.familyId) ||
      compareText(left.normalizedSurface, right.normalizedSurface) || compareText(left.id, right.id))
}

function headPayload(head: Omit<LanguageOutputHeadV1, 'fingerprint'>): unknown {
  return head
}

export function computeLanguageOutputHeadFingerprint(
  head: Omit<LanguageOutputHeadV1, 'fingerprint'>,
): string {
  return outputDigest('language-output-head', headPayload(head))
}

function familyReference(familyRun?: Pick<FamilyRun, 'format' | 'version' | 'fingerprint' | 'representationSpaceId'>): {
  representationSpaceId: string | null
  familyRunFingerprint: string | null
} {
  if (!familyRun) return { representationSpaceId: null, familyRunFingerprint: null }
  if (
    familyRun.format !== 'noron-aileleri-run' || familyRun.version !== 1 ||
    !familyRun.fingerprint || !familyRun.representationSpaceId
  ) throw new TypeError('Dil çıktı başlığı için geçerli bir FamilyRun başvurusu gerekir.')
  return {
    representationSpaceId: familyRun.representationSpaceId,
    familyRunFingerprint: familyRun.fingerprint,
  }
}

export function createLanguageOutputHead(
  scaffold: LanguageOutputScaffoldV1,
  familyRun?: Pick<FamilyRun, 'format' | 'version' | 'fingerprint' | 'representationSpaceId'>,
  configOverrides: Partial<LanguageOutputLearningConfig> = {},
): LanguageOutputHeadV1 {
  validateLanguageOutputScaffold(scaffold)
  const config = createLanguageOutputLearningConfig(configOverrides)
  const reference = familyReference(familyRun)
  const configFingerprint = computeLanguageOutputConfigFingerprint(config)
  const base: Omit<LanguageOutputHeadV1, 'fingerprint'> = {
    format: 'language-output-head',
    version: 1,
    id: `LOH-${outputDigest('language-output-head-id', scaffold.fingerprint, reference, configFingerprint).slice(-20)}`,
    parentFingerprint: null,
    revision: 0,
    scaffoldFingerprint: scaffold.fingerprint,
    ...reference,
    config,
    configFingerprint,
    transitionEvidence: [],
    membershipEvidence: [],
    nursery: [],
    outputNeurons: [],
    sourceGroupSplits: [],
    trainedDatasetFingerprints: [],
    training: {
      revision: 0,
      processedExamples: 0,
      positiveEdgeUpdates: 0,
      negativeEdgeUpdates: 0,
      membershipUpdates: 0,
      nurseryUpdates: 0,
      provisionalNeuronPromotions: 0,
      skippedUnclassified: 0,
      lastDatasetFingerprint: null,
    },
  }
  const head = { ...base, fingerprint: computeLanguageOutputHeadFingerprint(base) }
  validateLanguageOutputHead(head, scaffold)
  return head
}

function validateEvidence(
  edge: Pick<LanguageOutputEvidenceEdge, 'excitatoryEvidence' | 'inhibitoryEvidence' | 'score' | 'updateCount' | 'sourceDigests' | 'sourceGroupIds'>,
  label: string,
  evidenceCeiling: number,
): void {
  if (
    !Number.isFinite(edge.excitatoryEvidence) || edge.excitatoryEvidence < 0 ||
    edge.excitatoryEvidence > evidenceCeiling ||
    !Number.isFinite(edge.inhibitoryEvidence) || edge.inhibitoryEvidence < 0 ||
    edge.inhibitoryEvidence > evidenceCeiling ||
    !Number.isFinite(edge.score) || Math.abs(edge.score - boundedLogOdds(edge.excitatoryEvidence, edge.inhibitoryEvidence)) > 1e-9 ||
    !Number.isInteger(edge.updateCount) || edge.updateCount < 0 ||
    new Set(edge.sourceDigests).size !== edge.sourceDigests.length ||
    new Set(edge.sourceGroupIds).size !== edge.sourceGroupIds.length ||
    edge.sourceDigests.some((value) => !value) || edge.sourceGroupIds.some((value) => !value)
  ) throw new TypeError(`${label} kanıt kaydı bozuk.`)
}

function validateNursery(
  nursery: readonly LanguageOutputNurseryProposal[],
  head: LanguageOutputHeadV1,
  familyIds: ReadonlySet<string>,
): void {
  if (nursery.length > head.config.maximumNurseryProposals) {
    throw new RangeError('Dil çıktı nursery yapılandırılmış kapasiteyi aşıyor.')
  }
  const ids = new Set<string>()
  const surfaces = new Set<string>()
  for (const proposal of nursery) {
    const { fingerprint: _fingerprint, ...base } = proposal
    if (
      !proposal.id || ids.has(proposal.id) || !proposal.normalizedSurface ||
      surfaces.has(proposal.normalizedSurface) || !proposal.representativeSurface ||
      (proposal.status !== 'observing' && proposal.status !== 'eligible') ||
      !Number.isInteger(proposal.observationCount) || proposal.observationCount < proposal.exampleIds.length ||
      new Set(proposal.sourceDigests).size !== proposal.sourceDigests.length ||
      new Set(proposal.sourceGroupIds).size !== proposal.sourceGroupIds.length ||
      new Set(proposal.exampleIds).size !== proposal.exampleIds.length ||
      new Set(proposal.sourceProposalIds).size !== proposal.sourceProposalIds.length ||
      proposal.observationCount !== proposal.exampleIds.length ||
      !proposal.sourceDigests.length || !proposal.sourceGroupIds.length || !proposal.sourceProposalIds.length ||
      !Number.isFinite(proposal.meanSurprise) || proposal.meanSurprise < 0 ||
      !Number.isFinite(proposal.meanNovelty) || proposal.meanNovelty < 0 || proposal.meanNovelty > 1 ||
      proposal.requiresValidation !== true ||
      !proposal.familyCandidates.length ||
      new Set(proposal.familyCandidates.map((candidate) => candidate.familyId)).size !== proposal.familyCandidates.length ||
      proposal.familyCandidates.some((candidate) => (
        !familyIds.has(candidate.familyId) || !Number.isFinite(candidate.evidence) || candidate.evidence <= 0 ||
        !Number.isInteger(candidate.observationCount) || candidate.observationCount < 1 ||
        candidate.observationCount > proposal.observationCount
      )) ||
      proposal.status !== (
        proposal.observationCount >= head.config.nurseryMinimumObservations &&
        proposal.sourceGroupIds.length >= head.config.nurseryMinimumSourceGroups
          ? 'eligible'
          : 'observing'
      ) ||
      computeLanguageOutputNurseryFingerprint(base) !== proposal.fingerprint
    ) throw new TypeError('Dil çıktı nursery önerisi bozuk veya fingerprinti uyuşmuyor.')
    ids.add(proposal.id)
    surfaces.add(proposal.normalizedSurface)
  }
}

function sameStringEvidence(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) return false
  const sortedLeft = [...left].sort(compareText)
  const sortedRight = [...right].sort(compareText)
  return sortedLeft.every((value, index) => value === sortedRight[index])
}

function validateOutputNeurons(
  neurons: readonly LanguageOutputProvisionalNeuronV1[],
  head: LanguageOutputHeadV1,
  familyIds: ReadonlySet<string>,
): void {
  if (!Array.isArray(neurons) || neurons.length > head.config.maximumProvisionalNeurons) {
    throw new RangeError('Dil çıktı provisional nöron kapasitesi aşıldı.')
  }
  const ids = new Set<string>()
  const surfaces = new Set<string>()
  const nurseryBySurface = new Map(head.nursery.map((proposal) => [proposal.normalizedSurface, proposal]))
  for (const neuron of neurons) {
    const proposal = nurseryBySurface.get(neuron.normalizedSurface)
    const familyCandidate = proposal?.familyCandidates.find((candidate) => candidate.familyId === neuron.familyId)
    const { fingerprint: _fingerprint, ...base } = neuron
    if (
      !neuron.id || ids.has(neuron.id) || surfaces.has(neuron.normalizedSurface) ||
      neuron.status !== 'provisional' || neuron.kind !== 'surface-controller' ||
      !neuron.normalizedSurface || !neuron.representativeSurface || !familyIds.has(neuron.familyId) ||
      !Number.isFinite(neuron.familyEvidence) || neuron.familyEvidence <= 0 ||
      !Number.isFinite(neuron.meanFamilyConfidence) || neuron.meanFamilyConfidence < 0 ||
      neuron.meanFamilyConfidence > 1 ||
      Math.abs(neuron.meanFamilyConfidence - neuron.familyEvidence / neuron.observationCount) > 1e-9 ||
      !Number.isInteger(neuron.observationCount) || neuron.observationCount < head.config.nurseryMinimumObservations ||
      neuron.observationCount !== neuron.exampleIds.length ||
      new Set(neuron.sourceDigests).size !== neuron.sourceDigests.length ||
      new Set(neuron.sourceGroupIds).size !== neuron.sourceGroupIds.length ||
      new Set(neuron.exampleIds).size !== neuron.exampleIds.length ||
      new Set(neuron.sourceProposalIds).size !== neuron.sourceProposalIds.length ||
      !neuron.sourceDigests.length || neuron.sourceGroupIds.length < head.config.nurseryMinimumSourceGroups ||
      !neuron.sourceProposalIds.length ||
      !Number.isInteger(neuron.promotedAtRevision) || neuron.promotedAtRevision < 1 ||
      neuron.promotedAtRevision > head.revision || neuron.reversible !== true ||
      neuron.physicalIdentityAllocated !== false || neuron.meanFamilyConfidence < head.config.classifierMinimumConfidence ||
      proposal?.status !== 'eligible' || !familyCandidate ||
      proposal.representativeSurface !== neuron.representativeSurface ||
      proposal.observationCount !== neuron.observationCount ||
      !sameStringEvidence(proposal.sourceDigests, neuron.sourceDigests) ||
      !sameStringEvidence(proposal.sourceGroupIds, neuron.sourceGroupIds) ||
      !sameStringEvidence(proposal.exampleIds, neuron.exampleIds) ||
      !sameStringEvidence(proposal.sourceProposalIds, neuron.sourceProposalIds) ||
      Math.abs(familyCandidate.evidence - neuron.familyEvidence) > 1e-9 ||
      familyCandidate.observationCount > neuron.observationCount ||
      computeLanguageOutputProvisionalNeuronFingerprint(base) !== neuron.fingerprint
    ) throw new TypeError('Dil çıktı provisional nöronu veya kaynak kanıtı bozuk.')
    ids.add(neuron.id)
    surfaces.add(neuron.normalizedSurface)
  }
}

export function validateLanguageOutputHead(
  head: LanguageOutputHeadV1,
  scaffold: LanguageOutputScaffoldV1,
): void {
  validateLanguageOutputScaffold(scaffold)
  if (
    head?.format !== 'language-output-head' || head.version !== 1 || !head.id || !head.fingerprint ||
    head.scaffoldFingerprint !== scaffold.fingerprint || !Number.isInteger(head.revision) || head.revision < 0 ||
    (head.revision === 0 ? head.parentFingerprint !== null : !head.parentFingerprint) ||
    head.training?.revision !== head.revision ||
    head.configFingerprint !== computeLanguageOutputConfigFingerprint(head.config) ||
    (head.representationSpaceId === null) !== (head.familyRunFingerprint === null)
  ) throw new TypeError('Dil çıktı başlığı eksik veya scaffold ile uyumsuz.')
  validateLanguageOutputLearningConfig(head.config)
  const familyIds = new Set<string>(Object.values(LANGUAGE_OUTPUT_FAMILY_IDS))
  const validFamilyOrigins = new Set<string>([LANGUAGE_ROLE_GATEWAY_ID, ...familyIds])
  const edgeKeys = new Set<string>()
  const outgoing = new Map<string, number>()
  for (const edge of head.transitionEvidence) {
    const key = `${edge.kind}\0${edge.fromKey}\0${edge.toKey}`
    const row = `${edge.kind}\0${edge.fromKey}`
    if (
      !edge.fromKey || !edge.toKey || edgeKeys.has(key) ||
      (edge.kind !== 'family-transition' && edge.kind !== 'member-transition') ||
      (edge.kind === 'family-transition' && (
        !validFamilyOrigins.has(edge.fromKey) || !familyIds.has(edge.toKey)
      ))
    ) throw new TypeError('Dil çıktı geçiş kanıtı bozuk veya yineleniyor.')
    validateEvidence(edge, 'Dil çıktı geçiş', head.config.evidenceCeiling)
    edgeKeys.add(key)
    outgoing.set(row, (outgoing.get(row) ?? 0) + 1)
  }
  if ([...outgoing.values()].some((count) => count > head.config.maximumOutgoingEdges)) {
    throw new RangeError('Dil çıktı seyrek geçiş satırı kapasiteyi aşıyor.')
  }
  const membershipKeys = new Set<string>()
  const membershipCounts = new Map<string, number>()
  for (const membership of head.membershipEvidence) {
    const key = `${membership.memberKey}\0${membership.familyId}`
    if (
      !membership.memberKey || !familyIds.has(membership.familyId) || membershipKeys.has(key) ||
      !Number.isFinite(membership.classifierConfidence) || membership.classifierConfidence < 0 ||
      membership.classifierConfidence > 1
    ) throw new TypeError('Dil çıktı üyelik kanıtı bozuk veya yineleniyor.')
    validateEvidence(membership, 'Dil çıktı üyelik', head.config.evidenceCeiling)
    membershipKeys.add(key)
    membershipCounts.set(membership.memberKey, (membershipCounts.get(membership.memberKey) ?? 0) + 1)
  }
  if ([...membershipCounts.values()].some((count) => count > head.config.maximumMembershipsPerMember)) {
    throw new RangeError('Dil çıktı üye başına aile kapasitesi aşılıyor.')
  }
  validateNursery(head.nursery, head, familyIds)
  validateOutputNeurons(head.outputNeurons, head, familyIds)
  if (
    !Array.isArray(head.sourceGroupSplits) ||
    new Set(head.sourceGroupSplits.map((assignment) => assignment.independenceGroupId)).size !== head.sourceGroupSplits.length ||
    head.sourceGroupSplits.some((assignment) => !assignment.independenceGroupId || !['train', 'validation', 'test'].includes(assignment.split)) ||
    new Set(head.trainedDatasetFingerprints).size !== head.trainedDatasetFingerprints.length ||
    head.trainedDatasetFingerprints.some((value) => !value) ||
    (head.training.lastDatasetFingerprint !== null &&
      !head.trainedDatasetFingerprints.includes(head.training.lastDatasetFingerprint))
  ) throw new TypeError('Dil çıktı kaynak ayrımı veya eğitim geçmişi bozuk.')
  const integerStats = [
    head.training.processedExamples,
    head.training.positiveEdgeUpdates,
    head.training.negativeEdgeUpdates,
    head.training.membershipUpdates,
    head.training.nurseryUpdates,
    head.training.provisionalNeuronPromotions,
    head.training.skippedUnclassified,
  ]
  if (integerStats.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new TypeError('Dil çıktı eğitim sayaçları bozuk.')
  }
  const { fingerprint: _fingerprint, ...base } = head
  if (computeLanguageOutputHeadFingerprint(base) !== head.fingerprint) {
    throw new TypeError('Dil çıktı başlığı fingerprint doğrulaması başarısız.')
  }
}

export function finalizeLanguageOutputHead(
  previous: LanguageOutputHeadV1,
  scaffold: LanguageOutputScaffoldV1,
  nextValues: Omit<LanguageOutputHeadV1, 'fingerprint' | 'parentFingerprint' | 'revision'>,
): LanguageOutputHeadV1 {
  validateLanguageOutputHead(previous, scaffold)
  if (
    nextValues.id !== previous.id || nextValues.scaffoldFingerprint !== previous.scaffoldFingerprint ||
    nextValues.configFingerprint !== previous.configFingerprint ||
    (previous.representationSpaceId !== null && nextValues.representationSpaceId !== previous.representationSpaceId) ||
    (previous.familyRunFingerprint !== null && nextValues.familyRunFingerprint !== previous.familyRunFingerprint)
  ) throw new TypeError('Dil çıktı revizyonu kimliği, scaffoldı, ayarı veya bağlı FamilyRun uzayını değiştiremez.')
  const revision = previous.revision + 1
  const base: Omit<LanguageOutputHeadV1, 'fingerprint'> = {
    ...nextValues,
    config: { ...nextValues.config },
    transitionEvidence: sortLanguageOutputEdges(nextValues.transitionEvidence),
    membershipEvidence: sortLanguageOutputMemberships(nextValues.membershipEvidence),
    nursery: sortedNursery(nextValues.nursery),
    outputNeurons: sortedOutputNeurons(nextValues.outputNeurons),
    sourceGroupSplits: [...nextValues.sourceGroupSplits]
      .map((assignment) => ({ ...assignment }))
      .sort((left, right) => compareText(left.independenceGroupId, right.independenceGroupId)),
    trainedDatasetFingerprints: [...new Set(nextValues.trainedDatasetFingerprints)].sort(compareText),
    parentFingerprint: previous.fingerprint,
    revision,
    training: { ...nextValues.training, revision },
  }
  const head = { ...base, fingerprint: computeLanguageOutputHeadFingerprint(base) }
  validateLanguageOutputHead(head, scaffold)
  return head
}

/**
 * Carries a language-output overlay across FamilyRun revisions without treating
 * approximate family similarity as identity. Only exact member keys survive;
 * control/EOS endpoints remain virtual scaffold keys.
 */
export function migrateLanguageOutputHead(
  head: LanguageOutputHeadV1,
  scaffold: LanguageOutputScaffoldV1,
  newFamilyRun: Pick<FamilyRun, 'format' | 'version' | 'fingerprint' | 'representationSpaceId' | 'profiles'>,
): LanguageOutputHeadV1 {
  validateLanguageOutputHead(head, scaffold)
  if (
    newFamilyRun?.format !== 'noron-aileleri-run' || newFamilyRun.version !== 1 ||
    !newFamilyRun.fingerprint || !newFamilyRun.representationSpaceId || !Array.isArray(newFamilyRun.profiles) ||
    new Set(newFamilyRun.profiles.map((profile) => profile.memberKey)).size !== newFamilyRun.profiles.length ||
    newFamilyRun.profiles.some((profile) => !profile.memberKey)
  ) throw new TypeError('Dil çıktı migration için geçerli bir FamilyRun gerekir.')
  if (!head.representationSpaceId || !head.familyRunFingerprint) {
    throw new TypeError('Bağlanmamış dil çıktı başlığı FamilyRun migration ile taşınamaz.')
  }
  if (head.representationSpaceId !== newFamilyRun.representationSpaceId) {
    throw new TypeError('Dil çıktı başlığı farklı temsil uzayına otomatik taşınamaz.')
  }
  if (head.familyRunFingerprint === newFamilyRun.fingerprint) return deepCloneHead(head)
  const exactMembers = new Set(newFamilyRun.profiles.map((profile) => profile.memberKey))
  const validFrom = (key: string) => key === LANGUAGE_ROLE_GATEWAY_ID || exactMembers.has(key)
  const validTo = (key: string) => key === '@language/EOS/member' || exactMembers.has(key)
  const revision = head.revision + 1
  const {
    fingerprint: _oldFingerprint,
    parentFingerprint: _oldParent,
    revision: _oldRevision,
    ...headBase
  } = deepCloneHead(head)
  const base: Omit<LanguageOutputHeadV1, 'fingerprint'> = {
    ...headBase,
    parentFingerprint: head.fingerprint,
    revision,
    familyRunFingerprint: newFamilyRun.fingerprint,
    transitionEvidence: sortLanguageOutputEdges(head.transitionEvidence.filter((edge) => (
      edge.kind === 'family-transition' || (validFrom(edge.fromKey) && validTo(edge.toKey))
    ))),
    membershipEvidence: sortLanguageOutputMemberships(head.membershipEvidence.filter((membership) => (
      exactMembers.has(membership.memberKey)
    ))),
    nursery: sortedNursery(head.nursery),
    outputNeurons: sortedOutputNeurons(head.outputNeurons),
    training: { ...head.training, revision },
  }
  const migrated = { ...base, fingerprint: computeLanguageOutputHeadFingerprint(base) }
  validateLanguageOutputHead(migrated, scaffold)
  return migrated
}

function deepCloneHead(head: LanguageOutputHeadV1): LanguageOutputHeadV1 {
  return JSON.parse(JSON.stringify(head)) as LanguageOutputHeadV1
}
