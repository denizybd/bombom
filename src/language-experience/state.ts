import type { FamilyRun } from '../families/types'
import { validateLanguageExperienceConfig } from './config'
import {
  boundedLogOdds,
  compareText,
  deepSerializableClone,
  languageDigest,
} from './deterministic'
import type {
  LanguageCheckpoint,
  LanguageExperienceConfig,
  LanguageExperienceState,
  LanguageTransitionEdge,
  NeuronBirthProposal,
} from './types'

function statePayload(state: Omit<LanguageExperienceState, 'fingerprint'>): unknown {
  return state
}

export function computeLanguageStateFingerprint(
  state: Omit<LanguageExperienceState, 'fingerprint'>,
): string {
  return languageDigest('state', statePayload(state))
}

export function createLanguageExperienceState(
  familyRun: FamilyRun,
  config: LanguageExperienceConfig,
): LanguageExperienceState {
  validateLanguageExperienceConfig(config)
  if (familyRun?.format !== 'noron-aileleri-run' || familyRun.version !== 1) {
    throw new TypeError('Dil deneyimi durumu yalnız doğrulanmış Nöron Aileleri v1 modeliyle kurulabilir.')
  }
  const configFingerprint = languageDigest('config', config)
  const base: Omit<LanguageExperienceState, 'fingerprint'> = {
    format: 'language-experience-state',
    version: 1,
    id: `LE-${languageDigest('state-id', familyRun.fingerprint, configFingerprint).slice(-20)}`,
    parentFingerprint: null,
    revision: 0,
    representationSpaceId: familyRun.representationSpaceId,
    behaviorSpaceId: familyRun.behaviorSpaceId,
    familyRunFingerprint: familyRun.fingerprint,
    config: deepSerializableClone(config),
    configFingerprint,
    familyTransitions: [],
    neuronTransitions: [],
    birthProposals: [],
    emergentLanguageFamilies: [],
    sourceGroupSplits: [],
    trainedSourceFingerprints: [],
    training: {
      revision: 0,
      epochs: 0,
      processedExamples: 0,
      correctFamilyStrengthenings: 0,
      correctNeuronStrengthenings: 0,
      inhibitedFamilyAlternatives: 0,
      inhibitedNeuronAlternatives: 0,
      unboundTargets: 0,
      lastDatasetFingerprint: null,
    },
  }
  return { ...base, fingerprint: computeLanguageStateFingerprint(base) }
}

function validateEdges(edges: readonly LanguageTransitionEdge[], label: string, maximumOutgoing: number): void {
  const seen = new Set<string>()
  const outgoing = new Map<string, number>()
  for (const edge of edges) {
    const key = `${edge.fromKey}\u0000${edge.toKey}`
    if (
      !edge.fromKey || !edge.toKey || seen.has(key) ||
      !Number.isFinite(edge.excitatoryEvidence) || edge.excitatoryEvidence < 0 ||
      !Number.isFinite(edge.inhibitoryEvidence) || edge.inhibitoryEvidence < 0 ||
      !Number.isInteger(edge.updateCount) || edge.updateCount < 0 ||
      !Number.isInteger(edge.sourceCount) || edge.sourceCount < 0 ||
      !Array.isArray(edge.sourceDigests) || new Set(edge.sourceDigests).size !== edge.sourceDigests.length ||
      edge.sourceCount !== edge.sourceDigests.length ||
      !Number.isFinite(edge.score) || Math.abs(edge.score - boundedLogOdds(edge.excitatoryEvidence, edge.inhibitoryEvidence)) > 1e-9
    ) throw new TypeError(`${label} geçiş tablosu bozuk veya yinelenen kenar içeriyor.`)
    seen.add(key)
    outgoing.set(edge.fromKey, (outgoing.get(edge.fromKey) ?? 0) + 1)
  }
  if ([...outgoing.values()].some((count) => count > maximumOutgoing)) {
    throw new RangeError(`${label} geçiş tablosu yapılandırılmış seyrek çıkış sınırını aşıyor.`)
  }
}

function validateBirthProposals(proposals: readonly NeuronBirthProposal[]): void {
  const ids = new Set<string>()
  for (const proposal of proposals) {
    const sources = new Set(proposal.sourceDigests)
    const sourceGroups = new Set(proposal.sourceGroupIds)
    if (
      !proposal.id || !proposal.fingerprint || ids.has(proposal.id) ||
      (proposal.status !== 'observing' && proposal.status !== 'eligible') ||
      !proposal.normalizedSurface || !proposal.representativeSurface ||
      !Number.isInteger(proposal.observationCount) || proposal.observationCount < proposal.observations.length ||
      !Array.isArray(proposal.sourceDigests) || sources.size !== proposal.sourceDigests.length ||
      !Array.isArray(proposal.sourceGroupIds) || sourceGroups.size !== proposal.sourceGroupIds.length ||
      proposal.sourceCount !== sourceGroups.size || proposal.requiresExternalApproval !== true ||
      !Number.isFinite(proposal.meanSurprise) || !Number.isFinite(proposal.meanNovelty)
    ) throw new TypeError('Nöron doğum önerisi bozuk; kalıcı kimlik üretimine izin verilmedi.')
    ids.add(proposal.id)
  }
}

function validateEmergentFamilies(families: LanguageExperienceState['emergentLanguageFamilies']): void {
  const ids = new Set<string>()
  for (const family of families) {
    if (
      !family.id || !family.fingerprint || ids.has(family.id) ||
      (family.status !== 'observing' && family.status !== 'eligible') ||
      !Array.isArray(family.memberKeys) || !family.memberKeys.length ||
      new Set(family.memberKeys).size !== family.memberKeys.length ||
      !Array.isArray(family.sourceGroupIds) || new Set(family.sourceGroupIds).size !== family.sourceGroupIds.length ||
      family.sourceGroupCount !== family.sourceGroupIds.length ||
      !Number.isInteger(family.transitionEvidenceCount) || family.transitionEvidenceCount < 0 ||
      !Number.isInteger(family.heldOutCaseCount) || family.heldOutCaseCount < 0 ||
      !Number.isFinite(family.fullHeldOutLoss) || !Number.isFinite(family.ablatedHeldOutLoss) ||
      !Number.isFinite(family.predictiveUtility) || !Array.isArray(family.outgoingSignature)
    ) throw new TypeError('Ortaya çıkan dil ailesi kanıt kaydı eksik veya bozuk.')
    ids.add(family.id)
  }
}

export function validateLanguageExperienceState(state: LanguageExperienceState): void {
  if (
    state?.format !== 'language-experience-state' || state.version !== 1 ||
    !state.id || !state.fingerprint || !state.representationSpaceId || !state.behaviorSpaceId ||
    !state.familyRunFingerprint || !state.configFingerprint ||
    !Number.isInteger(state.revision) || state.revision < 0 ||
    state.training?.revision !== state.revision ||
    !Number.isInteger(state.training.epochs) || state.training.epochs < 0 ||
    !Number.isInteger(state.training.processedExamples) || state.training.processedExamples < 0 ||
    !Number.isInteger(state.training.correctFamilyStrengthenings) || state.training.correctFamilyStrengthenings < 0 ||
    !Number.isInteger(state.training.correctNeuronStrengthenings) || state.training.correctNeuronStrengthenings < 0 ||
    !Number.isInteger(state.training.inhibitedFamilyAlternatives) || state.training.inhibitedFamilyAlternatives < 0 ||
    !Number.isInteger(state.training.inhibitedNeuronAlternatives) || state.training.inhibitedNeuronAlternatives < 0 ||
    !Number.isInteger(state.training.unboundTargets) || state.training.unboundTargets < 0 ||
    state.configFingerprint !== languageDigest('config', state.config)
  ) throw new TypeError('Dil deneyimi durumu eksik veya uyumsuz.')
  validateLanguageExperienceConfig(state.config)
  validateEdges(state.familyTransitions, 'Aile', state.config.maximumOutgoingFamilyTransitions)
  validateEdges(state.neuronTransitions, 'Nöron', state.config.maximumOutgoingNeuronTransitions)
  validateBirthProposals(state.birthProposals)
  validateEmergentFamilies(state.emergentLanguageFamilies)
  if (
    !Array.isArray(state.trainedSourceFingerprints) ||
    new Set(state.trainedSourceFingerprints).size !== state.trainedSourceFingerprints.length
  ) throw new TypeError('Eğitim kaynağı geçmişi yinelenen veya bozuk kayıt içeriyor.')
  if (
    !Array.isArray(state.sourceGroupSplits) ||
    new Set(state.sourceGroupSplits.map((assignment) => assignment.independenceGroupId)).size !== state.sourceGroupSplits.length ||
    state.sourceGroupSplits.some((assignment) => (
      !assignment.independenceGroupId || !['train', 'validation', 'test'].includes(assignment.split)
    ))
  ) throw new TypeError('Bağımsız kaynak grubu holdout geçmişi yinelenen veya bozuk kayıt içeriyor.')
  const { fingerprint: _fingerprint, ...base } = state
  if (computeLanguageStateFingerprint(base) !== state.fingerprint) {
    throw new TypeError('Dil deneyimi durumunun içerik parmak izi doğrulanamadı.')
  }
}

export function assertLanguageStateCompatible(
  state: LanguageExperienceState,
  familyRun: FamilyRun,
): void {
  validateLanguageExperienceState(state)
  if (
    state.representationSpaceId !== familyRun.representationSpaceId ||
    state.behaviorSpaceId !== familyRun.behaviorSpaceId ||
    state.familyRunFingerprint !== familyRun.fingerprint
  ) throw new TypeError('Dil deneyimi durumu farklı veya değişmiş bir aile modeliyle kullanılamaz.')
}

export function finalizeLanguageState(
  previous: LanguageExperienceState,
  nextValues: Omit<LanguageExperienceState, 'fingerprint' | 'parentFingerprint' | 'revision'>,
): LanguageExperienceState {
  validateLanguageExperienceState(previous)
  if (
    nextValues.id !== previous.id ||
    nextValues.representationSpaceId !== previous.representationSpaceId ||
    nextValues.behaviorSpaceId !== previous.behaviorSpaceId ||
    nextValues.familyRunFingerprint !== previous.familyRunFingerprint ||
    nextValues.configFingerprint !== previous.configFingerprint
  ) throw new TypeError('Dil deneyimi revizyonu model kimliğini, uzayını veya yapılandırmasını değiştiremez.')
  const base: Omit<LanguageExperienceState, 'fingerprint'> = {
    ...deepSerializableClone(nextValues),
    parentFingerprint: previous.fingerprint,
    revision: previous.revision + 1,
    training: {
      ...deepSerializableClone(nextValues.training),
      revision: previous.revision + 1,
    },
  }
  const state = { ...base, fingerprint: computeLanguageStateFingerprint(base) }
  validateLanguageExperienceState(state)
  return state
}

export function createLanguageCheckpoint(state: LanguageExperienceState): LanguageCheckpoint {
  validateLanguageExperienceState(state)
  const checkpoint: LanguageCheckpoint = {
    format: 'language-experience-checkpoint',
    version: 1,
    id: `LC-${languageDigest('checkpoint', state.fingerprint).slice(-20)}`,
    stateFingerprint: state.fingerprint,
    state: deepSerializableClone(state),
  }
  return checkpoint
}

export function restoreLanguageCheckpoint(checkpoint: LanguageCheckpoint): LanguageExperienceState {
  if (
    checkpoint?.format !== 'language-experience-checkpoint' || checkpoint.version !== 1 ||
    !checkpoint.id || checkpoint.stateFingerprint !== checkpoint.state?.fingerprint
  ) throw new TypeError('Dil deneyimi kontrol noktası eksik veya bozuk.')
  validateLanguageExperienceState(checkpoint.state)
  return deepSerializableClone(checkpoint.state)
}

export function sortTransitionEdges(edges: readonly LanguageTransitionEdge[]): LanguageTransitionEdge[] {
  return [...edges]
    .map((edge) => ({ ...edge, sourceDigests: [...edge.sourceDigests].sort(compareText) }))
    .sort((left, right) =>
      compareText(left.fromKey, right.fromKey) || right.score - left.score || compareText(left.toKey, right.toKey),
    )
}
