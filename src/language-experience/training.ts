import type { FamilyRun } from '../families/types'
import {
  LANGUAGE_EOS_FAMILY,
  LANGUAGE_EOS_MEMBER,
} from './types'
import type {
  DatasetSplit,
  LanguageDataset,
  LanguageExperienceState,
  LanguageTrainingResult,
  LanguageTransitionEdge,
  NeuronBirthProposal,
  TrainLanguageOptions,
} from './types'
import {
  boundedLogOdds,
  clamp,
  compareText,
  deepSerializableClone,
  deterministicOrder,
  languageDigest,
} from './deterministic'
import {
  familyOriginsForExample,
  memberOriginForExample,
  prepareLanguagePieceIndex,
  targetFamilyWeights,
  targetMemberWeights,
  teacherForcedRecentMembers,
} from './examples'
import { evaluateLanguageExperience } from './metrics'
import { discoverEmergentLanguageFamilies } from './emergence'
import {
  prepareFamilyRoutingSpace,
  routeNextLanguageStepUnchecked,
} from './routing'
import type { PreparedLearnedRoutingSpace } from './routing'
import {
  assertLanguageStateCompatible,
  finalizeLanguageState,
  sortTransitionEdges,
} from './state'
import { assertFamilyIdentitySnapshot, snapshotFamilyIdentities } from './adapter'

export class LanguageTrainingCancelledError extends Error {
  readonly code = 'LANGUAGE_TRAINING_CANCELLED'

  constructor() {
    super('Dil deneyimi eğitimi kullanıcı tarafından durduruldu; kısmi durum kaydedilmedi.')
    this.name = 'LanguageTrainingCancelledError'
  }
}

class EdgeStore {
  readonly edges: LanguageTransitionEdge[]
  private readonly byOrigin = new Map<string, LanguageTransitionEdge[]>()
  readonly positiveTargets = new Set<string>()

  constructor(initial: readonly LanguageTransitionEdge[], private readonly maximumOutgoing: number) {
    this.edges = initial.map((edge) => ({ ...edge, sourceDigests: [...edge.sourceDigests] }))
    for (const edge of this.edges) {
      const values = this.byOrigin.get(edge.fromKey) ?? []
      values.push(edge)
      this.byOrigin.set(edge.fromKey, values)
      if (edge.excitatoryEvidence > 0) this.positiveTargets.add(edge.toKey)
    }
  }

  get index(): ReadonlyMap<string, LanguageTransitionEdge[]> {
    return this.byOrigin
  }

  update(
    fromKey: string,
    toKey: string,
    sourceDigest: string,
    excitation: number,
    inhibition: number,
    ceiling: number,
  ): void {
    const outgoing = this.byOrigin.get(fromKey) ?? []
    let edge = outgoing.find((candidate) => candidate.toKey === toKey)
    if (!edge) {
      edge = {
        fromKey,
        toKey,
        excitatoryEvidence: 0,
        inhibitoryEvidence: 0,
        updateCount: 0,
        sourceCount: 0,
        sourceDigests: [],
        score: 0,
      }
      outgoing.push(edge)
      this.edges.push(edge)
      this.byOrigin.set(fromKey, outgoing)
    }
    edge.excitatoryEvidence = Math.min(ceiling, edge.excitatoryEvidence + Math.max(0, excitation))
    edge.inhibitoryEvidence = Math.min(ceiling, edge.inhibitoryEvidence + Math.max(0, inhibition))
    edge.updateCount += 1
    if (!edge.sourceDigests.includes(sourceDigest)) {
      edge.sourceDigests.push(sourceDigest)
      edge.sourceDigests.sort(compareText)
      edge.sourceCount = edge.sourceDigests.length
    }
    edge.score = boundedLogOdds(edge.excitatoryEvidence, edge.inhibitoryEvidence)
    if (edge.excitatoryEvidence > 0) this.positiveTargets.add(edge.toKey)
    this.prune(fromKey)
  }

  decayOrigin(fromKey: string, decay: number): void {
    if (decay <= 0) return
    for (const edge of this.byOrigin.get(fromKey) ?? []) {
      edge.excitatoryEvidence *= 1 - decay
      edge.inhibitoryEvidence *= 1 - decay
      edge.score = boundedLogOdds(edge.excitatoryEvidence, edge.inhibitoryEvidence)
    }
  }

  private prune(fromKey: string): void {
    const outgoing = this.byOrigin.get(fromKey) ?? []
    if (outgoing.length <= this.maximumOutgoing) return
    outgoing.sort((left, right) => {
      const leftUtility = Math.abs(left.score) + 0.05 * Math.log1p(left.updateCount)
      const rightUtility = Math.abs(right.score) + 0.05 * Math.log1p(right.updateCount)
      return rightUtility - leftUtility || compareText(left.toKey, right.toKey)
    })
    const removed = outgoing.splice(this.maximumOutgoing)
    const removedSet = new Set(removed)
    for (let index = this.edges.length - 1; index >= 0; index -= 1) {
      if (removedSet.has(this.edges[index])) this.edges.splice(index, 1)
    }
    for (const target of new Set(removed.map((edge) => edge.toKey))) {
      if (!this.edges.some((edge) => edge.toKey === target && edge.excitatoryEvidence > 0)) {
        this.positiveTargets.delete(target)
      }
    }
  }
}

function assertDatasetIntegrity(dataset: LanguageDataset, state: LanguageExperienceState, run: FamilyRun): void {
  if (
    dataset?.format !== 'language-experience-dataset' || dataset.version !== 1 || !dataset.fingerprint ||
    dataset.familyRunFingerprint !== run.fingerprint ||
    dataset.representationSpaceId !== state.representationSpaceId ||
    dataset.behaviorSpaceId !== state.behaviorSpaceId ||
    dataset.configFingerprint !== state.configFingerprint
  ) throw new TypeError('Dil veri kümesi durum, ayar veya aile modeliyle uyumlu değil.')
  const groupSplits = new Map<string, DatasetSplit>()
  const previousGroupSplits = new Map(state.sourceGroupSplits.map((assignment) => [
    assignment.independenceGroupId,
    assignment.split,
  ]))
  const sourceSplits = new Map<string, DatasetSplit>()
  for (const source of dataset.sources) {
    const groupSplit = groupSplits.get(source.independenceGroupId)
    if (groupSplit && groupSplit !== source.split) {
      throw new TypeError('Aynı fiziksel kaynağın parçaları farklı holdout bölümlerine sızmış.')
    }
    groupSplits.set(source.independenceGroupId, source.split)
    const previousSplit = previousGroupSplits.get(source.independenceGroupId)
    if (previousSplit && previousSplit !== source.split) {
      throw new TypeError('Bağımsız kaynak grubu önceki revizyondan farklı bir holdout bölümüne taşınamaz.')
    }
    const sourceSplit = sourceSplits.get(source.sourceDigest)
    if (sourceSplit && sourceSplit !== source.split) {
      throw new TypeError('Aynı kaynak özeti farklı holdout bölümlerine sızmış.')
    }
    sourceSplits.set(source.sourceDigest, source.split)
  }
  const exampleIds = new Set<string>()
  for (const example of dataset.examples) {
    if (exampleIds.has(example.id) || sourceSplits.get(example.sourceDigest) !== example.split) {
      throw new TypeError('Dil örnekleri yinelenen kimlik veya kaynak-bölüm sızıntısı içeriyor.')
    }
    exampleIds.add(example.id)
  }
}

function targetProbability(
  route: ReturnType<typeof routeNextLanguageStepUnchecked>,
  targets: ReadonlySet<string>,
  kind: 'family' | 'neuron',
): number {
  if (kind === 'family') {
    return route.familyCandidates.reduce(
      (sum, candidate) => sum + (targets.has(candidate.familyId) ? candidate.probability : 0),
      0,
    )
  }
  let probability = targets.has(LANGUAGE_EOS_MEMBER) ? route.eosProbability : 0
  for (const candidate of route.neuronCandidates) {
    if (targets.has(candidate.memberKey)) probability += candidate.probability
  }
  return clamp(probability)
}

function bigrams(value: string): Set<string> {
  if (value.length < 2) return new Set([value])
  return new Set(Array.from({ length: value.length - 1 }, (_, index) => value.slice(index, index + 2)))
}

function surfaceSimilarity(left: string, right: string): number {
  if (left === right) return 1
  const a = bigrams(left)
  const b = bigrams(right)
  const intersection = [...a].filter((value) => b.has(value)).length
  const union = new Set([...a, ...b]).size
  return union ? intersection / union : 0
}

function recordBirthObservation(
  proposals: NeuronBirthProposal[],
  example: LanguageDataset['examples'][number],
  surprise: number,
  novelty: number,
  state: LanguageExperienceState,
  sourceGroupId: string,
): void {
  if (
    example.eos || !example.targetNormalizedText ||
    surprise < state.config.surpriseThreshold || novelty < state.config.noveltyThreshold
  ) return
  let proposal = proposals.find((candidate) => (
    surfaceSimilarity(candidate.normalizedSurface, example.targetNormalizedText) >=
      state.config.birthMergeSurfaceSimilarity
  ))
  if (!proposal) {
    if (proposals.length >= state.config.maximumBirthProposals) return
    const id = `NBP-${languageDigest('birth-id', example.targetNormalizedText).slice(-20)}`
    proposal = {
      id,
      fingerprint: '',
      status: 'observing',
      normalizedSurface: example.targetNormalizedText,
      representativeSurface: example.targetText,
      observationCount: 0,
      sourceCount: 0,
      sourceDigests: [],
      sourceGroupIds: [],
      meanSurprise: 0,
      meanNovelty: 0,
      observations: [],
      requiresExternalApproval: true,
    }
    proposals.push(proposal)
  }
  const previousCount = proposal.observationCount
  proposal.observationCount += 1
  proposal.meanSurprise = (proposal.meanSurprise * previousCount + surprise) / proposal.observationCount
  proposal.meanNovelty = (proposal.meanNovelty * previousCount + novelty) / proposal.observationCount
  if (!proposal.sourceDigests.includes(example.sourceDigest)) {
    proposal.sourceDigests.push(example.sourceDigest)
    proposal.sourceDigests.sort(compareText)
  }
  if (!proposal.sourceGroupIds.includes(sourceGroupId)) {
    proposal.sourceGroupIds.push(sourceGroupId)
    proposal.sourceGroupIds.sort(compareText)
    proposal.sourceCount = proposal.sourceGroupIds.length
  }
  if (proposal.observations.length < state.config.maximumBirthObservationsPerProposal) {
    proposal.observations.push({
      sourceDigest: example.sourceDigest,
      exampleId: example.id,
      surface: example.targetText,
      normalizedSurface: example.targetNormalizedText,
      granularity: example.granularity,
      surprise,
      novelty,
    })
  }
  proposal.status = (
    proposal.observationCount >= state.config.birthMinimumObservations &&
    proposal.sourceCount >= state.config.birthMinimumSources
  ) ? 'eligible' : 'observing'
  proposal.fingerprint = languageDigest('birth-proposal',
    proposal.normalizedSurface,
    proposal.observationCount,
    proposal.sourceDigests,
    proposal.sourceGroupIds,
    proposal.meanSurprise,
    proposal.meanNovelty,
  )
}

/**
 * Performs source-holdout self-supervision. Every next context is the observed
 * target (teacher forcing), never the model's sampled output.
 */
export function trainLanguageExperience(
  previous: LanguageExperienceState,
  familyRun: FamilyRun,
  dataset: LanguageDataset,
  options: TrainLanguageOptions = {},
): LanguageTrainingResult {
  assertLanguageStateCompatible(previous, familyRun)
  assertDatasetIntegrity(dataset, previous, familyRun)
  const identitySnapshot = snapshotFamilyIdentities(familyRun)
  const epochs = options.epochs ?? 1
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 1_000) {
    throw new RangeError('Dil deneyimi epoch sayısı 1 ile 1000 arasında bir tam sayı olmalıdır.')
  }
  const working = deepSerializableClone(previous)
  const familyEdges = new EdgeStore(working.familyTransitions, working.config.maximumOutgoingFamilyTransitions)
  const neuronEdges = new EdgeStore(working.neuronTransitions, working.config.maximumOutgoingNeuronTransitions)
  working.familyTransitions = familyEdges.edges
  working.neuronTransitions = neuronEdges.edges
  const proposals = working.birthProposals.map((proposal) => deepSerializableClone(proposal))
  const pieceIndex = prepareLanguagePieceIndex(dataset)
  const familySpace = prepareFamilyRoutingSpace(familyRun)
  const learnedSpace: PreparedLearnedRoutingSpace = {
    familyByOrigin: familyEdges.index,
    neuronByOrigin: neuronEdges.index,
    outputMemberKeys: neuronEdges.positiveTargets,
  }
  const sourceFingerprintByDigest = new Map(dataset.sources.map((source) => [
    source.sourceDigest,
    source.canonicalFingerprint,
  ]))
  const alreadyTrained = new Set(previous.trainedSourceFingerprints)
  const trainingExamples = dataset.examples.filter((example) => (
    example.split === 'train' && (
      options.allowSourceReplay === true ||
      !alreadyTrained.has(sourceFingerprintByDigest.get(example.sourceDigest) ?? '')
    )
  ))
  const sourceGroupByDigest = new Map(dataset.sources.map((source) => [
    source.sourceDigest,
    source.independenceGroupId,
  ]))
  const total = trainingExamples.length * epochs
  let processed = 0
  let familyStrengthenings = 0
  let neuronStrengthenings = 0
  let familyInhibitions = 0
  let neuronInhibitions = 0
  let unbound = 0
  const mergedSourceGroupSplits = [...new Map([
    ...working.sourceGroupSplits.map((assignment) => [assignment.independenceGroupId, assignment] as const),
    ...dataset.sources.map((source) => [source.independenceGroupId, {
      independenceGroupId: source.independenceGroupId,
      split: source.split,
    }] as const),
  ]).values()].sort((left, right) => compareText(left.independenceGroupId, right.independenceGroupId))

  if (!trainingExamples.length) {
    const hasNewAssignments = mergedSourceGroupSplits.length !== previous.sourceGroupSplits.length
    let state = deepSerializableClone(previous)
    if (hasNewAssignments) {
      const {
        fingerprint: _fingerprint,
        parentFingerprint: _parent,
        revision: _revision,
        ...previousBase
      } = previous
      state = finalizeLanguageState(previous, {
        ...previousBase,
        sourceGroupSplits: mergedSourceGroupSplits,
      })
    }
    const requested = options.evaluateSplits ?? ['validation', 'test']
    const metrics = Object.fromEntries(requested.map((split) => [
      split,
      evaluateLanguageExperience(state, familyRun, dataset, split),
    ])) as LanguageTrainingResult['metrics']
    assertFamilyIdentitySnapshot(familyRun, identitySnapshot)
    return {
      state,
      metrics,
      birthProposals: deepSerializableClone(previous.birthProposals),
      emergentLanguageFamilies: deepSerializableClone(previous.emergentLanguageFamilies),
    }
  }

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const ordered = deterministicOrder(
      trainingExamples,
      working.config.seed + previous.revision + epoch,
      (example) => example.id,
    )
    for (const example of ordered) {
      if (options.shouldCancel?.()) throw new LanguageTrainingCancelledError()
      const recentMemberKeys = teacherForcedRecentMembers(example, dataset, pieceIndex)
      const route = routeNextLanguageStepUnchecked(
        working, familyRun, { recentMemberKeys }, familySpace, learnedSpace,
      )
      const familyOrigins = familyOriginsForExample(example, dataset, pieceIndex)
      const memberOrigin = memberOriginForExample(example, dataset, pieceIndex)
      const familyTargets = targetFamilyWeights(example)
      const memberTargets = targetMemberWeights(example)
      const correctFamilyKeys = new Set(familyTargets.map((target) => target.key))
      const correctMemberKeys = new Set(memberTargets.map((target) => target.key))
      const familyProbability = targetProbability(route, correctFamilyKeys, 'family')
      const memberProbability = targetProbability(route, correctMemberKeys, 'neuron')
      const surprise = -Math.log(Math.max(1e-12, memberProbability))
      const novelty = example.eos ? 0 : 1 - Math.max(0, ...example.targetBindings.map((binding) => binding.score))

      if (!familyTargets.length || !memberTargets.length) unbound += 1
      for (const origin of familyOrigins) {
        familyEdges.decayOrigin(origin.key, working.config.evidenceDecay)
        for (const target of familyTargets) {
          familyEdges.update(
            origin.key,
            target.key,
            example.sourceDigest,
            working.config.learningRate * origin.weight * target.weight * Math.max(0.1, 1 - familyProbability),
            0,
            working.config.evidenceCeiling,
          )
          familyStrengthenings += 1
        }
        for (const wrong of route.familyCandidates) {
          if (correctFamilyKeys.has(wrong.familyId)) continue
          familyEdges.update(
            origin.key,
            wrong.familyId,
            example.sourceDigest,
            0,
            working.config.inhibitionRate * origin.weight * wrong.probability,
            working.config.evidenceCeiling,
          )
          familyInhibitions += 1
        }
      }
      neuronEdges.decayOrigin(memberOrigin, working.config.evidenceDecay)
      for (const target of memberTargets) {
        neuronEdges.update(
          memberOrigin,
          target.key,
          example.sourceDigest,
          working.config.learningRate * target.weight * Math.max(0.1, 1 - memberProbability),
          0,
          working.config.evidenceCeiling,
        )
        neuronStrengthenings += 1
      }
      for (const wrong of route.neuronCandidates) {
        if (correctMemberKeys.has(wrong.memberKey)) continue
        neuronEdges.update(
          memberOrigin,
          wrong.memberKey,
          example.sourceDigest,
          0,
          working.config.inhibitionRate * wrong.probability,
          working.config.evidenceCeiling,
        )
        neuronInhibitions += 1
      }
      if (!correctMemberKeys.has(LANGUAGE_EOS_MEMBER) && route.eosProbability > 0) {
        neuronEdges.update(
          memberOrigin,
          LANGUAGE_EOS_MEMBER,
          example.sourceDigest,
          0,
          working.config.inhibitionRate * route.eosProbability,
          working.config.evidenceCeiling,
        )
        neuronInhibitions += 1
      }
      if (!correctFamilyKeys.has(LANGUAGE_EOS_FAMILY)) {
        const eosFamily = route.familyCandidates.find((candidate) => candidate.familyId === LANGUAGE_EOS_FAMILY)
        if (eosFamily) {
          for (const origin of familyOrigins) {
            familyEdges.update(
              origin.key,
              LANGUAGE_EOS_FAMILY,
              example.sourceDigest,
              0,
              working.config.inhibitionRate * origin.weight * eosFamily.probability,
              working.config.evidenceCeiling,
            )
            familyInhibitions += 1
          }
        }
      }
      recordBirthObservation(
        proposals,
        example,
        surprise,
        novelty,
        working,
        sourceGroupByDigest.get(example.sourceDigest) ?? example.sourceId,
      )
      processed += 1
      if (processed === total || processed % 128 === 0) {
        options.onProgress?.({
          epoch: epoch + 1,
          processed,
          total,
          message: 'Teacher-forcing dil deneyimleri işleniyor; yalnız eğitim bölümü plastisiteye yazılıyor.',
        })
      }
    }
  }

  const trainFingerprints = dataset.sources
    .filter((source) => source.split === 'train')
    .map((source) => source.canonicalFingerprint)
  const {
    fingerprint: _workingFingerprint,
    parentFingerprint: _workingParent,
    revision: _workingRevision,
    ...workingBase
  } = working
  const nextValues: Omit<LanguageExperienceState, 'fingerprint' | 'parentFingerprint' | 'revision'> = {
    ...workingBase,
    familyTransitions: sortTransitionEdges(familyEdges.edges),
    neuronTransitions: sortTransitionEdges(neuronEdges.edges),
    birthProposals: proposals.sort((left, right) =>
      Number(right.status === 'eligible') - Number(left.status === 'eligible') ||
      right.meanSurprise - left.meanSurprise || compareText(left.id, right.id),
    ),
    emergentLanguageFamilies: [],
    trainedSourceFingerprints: [...new Set([...working.trainedSourceFingerprints, ...trainFingerprints])].sort(compareText),
    sourceGroupSplits: mergedSourceGroupSplits,
    training: {
      revision: working.training.revision,
      epochs: working.training.epochs + epochs,
      processedExamples: working.training.processedExamples + processed,
      correctFamilyStrengthenings: working.training.correctFamilyStrengthenings + familyStrengthenings,
      correctNeuronStrengthenings: working.training.correctNeuronStrengthenings + neuronStrengthenings,
      inhibitedFamilyAlternatives: working.training.inhibitedFamilyAlternatives + familyInhibitions,
      inhibitedNeuronAlternatives: working.training.inhibitedNeuronAlternatives + neuronInhibitions,
      unboundTargets: working.training.unboundTargets + unbound,
      lastDatasetFingerprint: dataset.fingerprint,
    },
  }
  const provisionalState = finalizeLanguageState(previous, nextValues)
  const emergentLanguageFamilies = discoverEmergentLanguageFamilies(provisionalState, familyRun, dataset)
  const state = finalizeLanguageState(previous, { ...nextValues, emergentLanguageFamilies })
  assertFamilyIdentitySnapshot(familyRun, identitySnapshot)
  const requested = options.evaluateSplits ?? ['validation', 'test']
  const metrics = Object.fromEntries(requested.map((split) => [
    split,
    evaluateLanguageExperience(state, familyRun, dataset, split),
  ])) as LanguageTrainingResult['metrics']
  return {
    state,
    metrics,
    birthProposals: deepSerializableClone(state.birthProposals),
    emergentLanguageFamilies: deepSerializableClone(state.emergentLanguageFamilies),
  }
}
