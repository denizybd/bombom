import type { FamilyRun } from '../families/types'
import { LANGUAGE_EOS_FAMILY, LANGUAGE_EOS_MEMBER } from './types'
import type {
  EmergentLanguageFamilyEvidence,
  LanguageDataset,
  LanguageExperienceState,
  LanguageRouteResult,
} from './types'
import { compareText, languageDigest } from './deterministic'
import {
  prepareLanguagePieceIndex,
  targetMemberWeights,
  teacherForcedRecentMembers,
} from './examples'
import {
  prepareFamilyRoutingSpace,
  prepareLearnedRoutingSpace,
  routeNextLanguageStepUnchecked,
} from './routing'
import { assertLanguageStateCompatible } from './state'

interface OriginSignature {
  memberKey: string
  signature: Map<string, number>
  transitionEvidenceCount: number
  sourceDigests: Set<string>
}

function normalizeSignature(values: Map<string, number>): Map<string, number> {
  const total = [...values.values()].reduce((sum, value) => sum + Math.max(0, value), 0)
  if (total <= 1e-12) return new Map()
  return new Map([...values].map(([key, value]) => [key, Math.max(0, value) / total]))
}

function cosine(left: ReadonlyMap<string, number>, right: ReadonlyMap<string, number>): number {
  let dot = 0
  let leftSquared = 0
  let rightSquared = 0
  for (const value of left.values()) leftSquared += value * value
  for (const value of right.values()) rightSquared += value * value
  for (const [key, value] of left) dot += value * (right.get(key) ?? 0)
  const denominator = Math.sqrt(leftSquared * rightSquared)
  return denominator > 1e-12 ? dot / denominator : 0
}

function buildOriginSignatures(state: LanguageExperienceState, run: FamilyRun): OriginSignature[] {
  const memberships = new Map<string, Array<{ familyId: string; weight: number }>>()
  for (const membership of run.memberships) {
    const values = memberships.get(membership.memberKey) ?? []
    values.push({ familyId: membership.familyId, weight: membership.weight })
    memberships.set(membership.memberKey, values)
  }
  const byOrigin = new Map<string, OriginSignature>()
  const knownMembers = new Set(run.profiles.map((profile) => profile.memberKey))
  for (const edge of state.neuronTransitions) {
    if (!knownMembers.has(edge.fromKey)) continue
    const strength = Math.max(0, edge.score) * Math.log1p(edge.excitatoryEvidence)
    if (strength <= 0) continue
    const origin = byOrigin.get(edge.fromKey) ?? {
      memberKey: edge.fromKey,
      signature: new Map<string, number>(),
      transitionEvidenceCount: 0,
      sourceDigests: new Set<string>(),
    }
    const targets = edge.toKey === LANGUAGE_EOS_MEMBER
      ? [{ familyId: LANGUAGE_EOS_FAMILY, weight: 1 }]
      : memberships.get(edge.toKey) ?? []
    for (const target of targets) {
      origin.signature.set(
        target.familyId,
        (origin.signature.get(target.familyId) ?? 0) + strength * Math.max(0, target.weight),
      )
    }
    origin.transitionEvidenceCount += edge.updateCount
    edge.sourceDigests.forEach((digest) => origin.sourceDigests.add(digest))
    byOrigin.set(edge.fromKey, origin)
  }
  return [...byOrigin.values()]
    .map((origin) => ({ ...origin, signature: normalizeSignature(origin.signature) }))
    .filter((origin) => origin.signature.size > 0)
    .sort((left, right) =>
      right.transitionEvidenceCount - left.transitionEvidenceCount || compareText(left.memberKey, right.memberKey),
    )
}

function completeLinkClusters(
  signatures: readonly OriginSignature[],
  threshold: number,
): OriginSignature[][] {
  const clusters: OriginSignature[][] = []
  for (const signature of signatures) {
    const compatible = clusters
      .map((cluster, index) => ({
        index,
        minimum: Math.min(...cluster.map((member) => cosine(signature.signature, member.signature))),
      }))
      .filter((candidate) => candidate.minimum >= threshold)
      .sort((left, right) => right.minimum - left.minimum || left.index - right.index)[0]
    if (compatible) clusters[compatible.index].push(signature)
    else clusters.push([signature])
  }
  return clusters
}

function targetProbability(route: LanguageRouteResult, targets: ReadonlySet<string>): number {
  let probability = targets.has(LANGUAGE_EOS_MEMBER) ? route.eosProbability : 0
  for (const candidate of route.neuronCandidates) {
    if (targets.has(candidate.memberKey)) probability += candidate.probability
  }
  return Math.max(1e-12, Math.min(1, probability))
}

function aggregateSignature(cluster: readonly OriginSignature[]): Array<{ targetFamilyId: string; weight: number }> {
  const values = new Map<string, number>()
  let totalWeight = 0
  for (const origin of cluster) {
    const originWeight = Math.max(1, origin.transitionEvidenceCount)
    totalWeight += originWeight
    for (const [targetFamilyId, weight] of origin.signature) {
      values.set(targetFamilyId, (values.get(targetFamilyId) ?? 0) + originWeight * weight)
    }
  }
  return [...values]
    .map(([targetFamilyId, weight]) => ({ targetFamilyId, weight: weight / Math.max(1, totalWeight) }))
    .sort((left, right) => right.weight - left.weight || compareText(left.targetFamilyId, right.targetFamilyId))
}

/**
 * Discovers language-function candidates from learned next-piece behavior and
 * measures each by held-out loss ablation. It never writes into FamilyRun.
 */
export function discoverEmergentLanguageFamilies(
  state: LanguageExperienceState,
  run: FamilyRun,
  dataset: LanguageDataset,
): EmergentLanguageFamilyEvidence[] {
  assertLanguageStateCompatible(state, run)
  if (
    dataset.familyRunFingerprint !== run.fingerprint ||
    dataset.configFingerprint !== state.configFingerprint
  ) throw new TypeError('Ortaya çıkan dil aileleri farklı veri veya aile uzayında ölçülemez.')
  const sourceGroupByDigest = new Map(dataset.sources.map((source) => [source.sourceDigest, source.independenceGroupId]))
  const signatures = buildOriginSignatures(state, run)
  const clusters = completeLinkClusters(signatures, state.config.emergentFamilySimilarityThreshold)
    .sort((left, right) => {
      const leftEvidence = left.reduce((sum, origin) => sum + origin.transitionEvidenceCount, 0)
      const rightEvidence = right.reduce((sum, origin) => sum + origin.transitionEvidenceCount, 0)
      return rightEvidence - leftEvidence || compareText(left[0].memberKey, right[0].memberKey)
    })
    .slice(0, state.config.maximumEmergentLanguageFamilies)
  const heldOut = dataset.examples.filter((example) => example.split !== 'train')
  const fullRouteCache = new Map<string, LanguageRouteResult>()
  const familySpace = prepareFamilyRoutingSpace(run)
  const fullLearnedSpace = prepareLearnedRoutingSpace(state)
  const pieceIndex = prepareLanguagePieceIndex(dataset)

  return clusters.map((cluster) => {
    const memberKeys = cluster.map((origin) => origin.memberKey).sort(compareText)
    const memberSet = new Set(memberKeys)
    const sourceGroupIds = [...new Set(cluster.flatMap((origin) =>
      [...origin.sourceDigests].map((digest) => sourceGroupByDigest.get(digest)).filter((id): id is string => Boolean(id)),
    ))].sort(compareText)
    const ablatedState: LanguageExperienceState = {
      ...state,
      neuronTransitions: state.neuronTransitions.filter((edge) => !memberSet.has(edge.fromKey)),
    }
    const ablatedLearnedSpace = {
      ...prepareLearnedRoutingSpace(ablatedState),
      outputMemberKeys: fullLearnedSpace.outputMemberKeys,
    }
    let heldOutCaseCount = 0
    let fullHeldOutLoss = 0
    let ablatedHeldOutLoss = 0
    for (const example of heldOut) {
      const recentMemberKeys = teacherForcedRecentMembers(example, dataset, pieceIndex)
      if (!memberSet.has(recentMemberKeys.at(-1) ?? '')) continue
      const targets = new Set(targetMemberWeights(example).map((target) => target.key))
      if (!targets.size) continue
      const full = fullRouteCache.get(example.id) ?? routeNextLanguageStepUnchecked(
        state, run, { recentMemberKeys }, familySpace, fullLearnedSpace,
      )
      fullRouteCache.set(example.id, full)
      const ablated = routeNextLanguageStepUnchecked(
        ablatedState, run, { recentMemberKeys }, familySpace, ablatedLearnedSpace,
      )
      fullHeldOutLoss += -Math.log(targetProbability(full, targets))
      ablatedHeldOutLoss += -Math.log(targetProbability(ablated, targets))
      heldOutCaseCount += 1
    }
    if (heldOutCaseCount) {
      fullHeldOutLoss /= heldOutCaseCount
      ablatedHeldOutLoss /= heldOutCaseCount
    }
    const predictiveUtility = heldOutCaseCount ? ablatedHeldOutLoss - fullHeldOutLoss : 0
    const transitionEvidenceCount = cluster.reduce((sum, origin) => sum + origin.transitionEvidenceCount, 0)
    const status = (
      memberKeys.length >= state.config.emergentFamilyMinimumMembers &&
      sourceGroupIds.length >= state.config.emergentFamilyMinimumSourceGroups &&
      heldOutCaseCount > 0 &&
      predictiveUtility >= state.config.emergentFamilyUtilityThreshold
    ) ? 'eligible' as const : 'observing' as const
    const outgoingSignature = aggregateSignature(cluster)
    const fingerprint = languageDigest('emergent-language-family',
      memberKeys,
      sourceGroupIds,
      transitionEvidenceCount,
      heldOutCaseCount,
      predictiveUtility,
      outgoingSignature,
    )
    return {
      id: `ELF-${languageDigest('emergent-language-family-id', memberKeys).slice(-20)}`,
      fingerprint,
      status,
      memberKeys,
      sourceGroupCount: sourceGroupIds.length,
      sourceGroupIds,
      transitionEvidenceCount,
      heldOutCaseCount,
      fullHeldOutLoss,
      ablatedHeldOutLoss,
      predictiveUtility,
      outgoingSignature,
      explanation: status === 'eligible'
        ? 'Bu küme elle etiketlenmedi; bağımsız eğitim gruplarında tekrarlanan geçiş davranışı, holdout sonraki-parça kaybını ablation testinde ölçülebilir biçimde düşürdü.'
        : 'Dil işlevi adayı gözleniyor; üye, bağımsız kaynak grubu veya holdout fayda kanıtı henüz eşiği geçmedi.',
    }
  }).sort((left, right) =>
    Number(right.status === 'eligible') - Number(left.status === 'eligible') ||
    right.predictiveUtility - left.predictiveUtility ||
    right.transitionEvidenceCount - left.transitionEvidenceCount ||
    compareText(left.id, right.id),
  )
}
