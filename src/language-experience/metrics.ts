import type { FamilyRun } from '../families/types'
import { LANGUAGE_EOS_MEMBER } from './types'
import type {
  DatasetSplit,
  LanguageDataset,
  LanguageExperienceState,
  LanguageMetrics,
  LanguageRouteResult,
} from './types'
import { compareText } from './deterministic'
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

function routeDistribution(route: LanguageRouteResult): Map<string, number> {
  return new Map([
    ...route.neuronCandidates.map((candidate): [string, number] => [candidate.memberKey, candidate.probability]),
    [LANGUAGE_EOS_MEMBER, route.eosProbability],
  ])
}

function probabilityForTargets(route: LanguageRouteResult, targets: ReadonlySet<string>): number {
  let probability = targets.has(LANGUAGE_EOS_MEMBER) ? route.eosProbability : 0
  for (const candidate of route.neuronCandidates) {
    if (targets.has(candidate.memberKey)) probability += candidate.probability
  }
  return Math.min(1, probability)
}

function targetRank(route: LanguageRouteResult, targets: ReadonlySet<string>): number | null {
  const ordered = [
    ...route.neuronCandidates.map((candidate) => ({ key: candidate.memberKey, probability: candidate.probability })),
    { key: LANGUAGE_EOS_MEMBER, probability: route.eosProbability },
  ].sort((left, right) => right.probability - left.probability || compareText(left.key, right.key))
  const rank = ordered.findIndex((candidate) => targets.has(candidate.key))
  return rank < 0 ? null : rank + 1
}

function totalVariation(left: LanguageRouteResult, right: LanguageRouteResult): number {
  const a = routeDistribution(left)
  const b = routeDistribution(right)
  const keys = new Set([...a.keys(), ...b.keys()])
  let sum = 0
  for (const key of keys) sum += Math.abs((a.get(key) ?? 0) - (b.get(key) ?? 0))
  return Math.min(1, 0.5 * sum)
}

/** Metrics never update state; validation/test examples remain true holdouts for the language head. */
export function evaluateLanguageExperience(
  state: LanguageExperienceState,
  familyRun: FamilyRun,
  dataset: LanguageDataset,
  split: DatasetSplit,
): LanguageMetrics {
  assertLanguageStateCompatible(state, familyRun)
  if (
    dataset.familyRunFingerprint !== familyRun.fingerprint ||
    dataset.representationSpaceId !== state.representationSpaceId ||
    dataset.behaviorSpaceId !== state.behaviorSpaceId ||
    dataset.configFingerprint !== state.configFingerprint
  ) throw new TypeError('Dil ölçümü farklı bir veri, ayar veya aile uzayıyla çalıştırılamaz.')

  const examples = dataset.examples.filter((example) => example.split === split)
  const topHits = new Map(state.config.metricTopK.map((k) => [k, 0]))
  let evaluable = 0
  let reciprocalRank = 0
  let nll = 0
  let boundTargetCount = 0
  let corruptionCases = 0
  let corruptionWins = 0
  let corruptionMargin = 0
  let roleSwapCases = 0
  let roleSwapSensitivity = 0
  let eosCases = 0
  let eosWins = 0
  const familySpace = prepareFamilyRoutingSpace(familyRun)
  const learnedSpace = prepareLearnedRoutingSpace(state)
  const pieceIndex = prepareLanguagePieceIndex(dataset)

  for (const example of examples) {
    const recentMemberKeys = teacherForcedRecentMembers(example, dataset, pieceIndex)
    const route = routeNextLanguageStepUnchecked(
      state, familyRun, { recentMemberKeys }, familySpace, learnedSpace,
    )
    const targetWeights = targetMemberWeights(example)
    const targets = new Set(targetWeights.map((target) => target.key))
    if (targets.size) {
      evaluable += 1
      if (!example.eos) boundTargetCount += 1
      const probability = Math.max(1e-12, probabilityForTargets(route, targets))
      nll += -Math.log(probability)
      const rank = targetRank(route, targets)
      if (rank !== null) {
        reciprocalRank += 1 / rank
        for (const k of state.config.metricTopK) if (rank <= k) topHits.set(k, (topHits.get(k) ?? 0) + 1)
      }
    }
    if (example.eos) {
      eosCases += 1
      if (route.eosProbability > (route.neuronCandidates[0]?.probability ?? 0)) eosWins += 1
    }
    if (example.corruptionBindings.length && example.targetBindings.length) {
      const correct = probabilityForTargets(route, new Set(example.targetBindings.map((binding) => binding.memberKey)))
      const corrupt = probabilityForTargets(route, new Set(example.corruptionBindings.map((binding) => binding.memberKey)))
      corruptionCases += 1
      corruptionMargin += correct - corrupt
      if (correct > corrupt) corruptionWins += 1
    }
    if (
      example.granularity === 'token' &&
      recentMemberKeys.length >= 2 &&
      recentMemberKeys.at(-1) !== recentMemberKeys.at(-2)
    ) {
      const swapped = [...recentMemberKeys]
      const last = swapped.length - 1
      ;[swapped[last - 1], swapped[last]] = [swapped[last], swapped[last - 1]]
      const swappedRoute = routeNextLanguageStepUnchecked(
        state, familyRun, { recentMemberKeys: swapped }, familySpace, learnedSpace,
      )
      roleSwapCases += 1
      roleSwapSensitivity += totalVariation(route, swappedRoute)
    }
  }

  const warnings: string[] = []
  if (!examples.length) warnings.push(`${split} bölümünde ölçülebilir örnek yok.`)
  if (evaluable < examples.length) warnings.push(`${examples.length - evaluable} bağlanmamış hedef, sıralama ölçümlerinin dışında kaldı.`)
  if (!corruptionCases) warnings.push('Dilbilgisel bozulma ayrımı için karşıt hedef çifti bulunamadı.')
  if (!roleSwapCases) warnings.push('Rol/sıra değişimine duyarlılık için iki üyeli token bağlamı bulunamadı.')
  const meanNll = evaluable ? nll / evaluable : 0
  return {
    split,
    metricScope: 'language-head-source-holdout-over-frozen-family-space',
    exampleCount: examples.length,
    boundTargetCount,
    topK: state.config.metricTopK.map((k) => ({ k, accuracy: evaluable ? (topHits.get(k) ?? 0) / evaluable : 0 })),
    meanReciprocalRank: evaluable ? reciprocalRank / evaluable : 0,
    negativeLogLikelihood: meanNll,
    perplexityLike: Math.exp(Math.min(20, meanNll)),
    grammarCorruptionAccuracy: corruptionCases ? corruptionWins / corruptionCases : 0,
    grammarCorruptionMargin: corruptionCases ? corruptionMargin / corruptionCases : 0,
    grammarCorruptionCaseCount: corruptionCases,
    roleSwapSensitivity: roleSwapCases ? roleSwapSensitivity / roleSwapCases : 0,
    roleSwapCaseCount: roleSwapCases,
    eosAccuracy: eosCases ? eosWins / eosCases : 0,
    warnings,
  }
}
