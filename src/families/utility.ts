import type {
  FamilyAnalysisInput,
  FamilyConfig,
  FamilyMembership,
  FamilyUtilityReport,
  HoldoutFoldMetric,
  LearnedFamily,
} from './types'
import { rawMemberKey } from './profile'
import { FAMILY_EPSILON, clamp, compareText } from './sparse'
import { createIndependentSourceIndex } from './independence'
import type { IndependentSourceIndex } from './independence'

interface PredictionCase {
  sourceId: string
  sourceDigest: string
  currentMemberKey: string
  targetMemberKey: string
  candidateMemberKeys: string[]
}

interface FamilyProbability {
  fromFamilyId: string
  toFamilyId: string
  probability: number
}

interface MembershipWeight {
  familyId: string
  weight: number
}

function sourceIdentity(analysis: FamilyAnalysisInput) {
  return {
    independenceGroupId: analysis.independenceGroupId,
    sourceId: analysis.source.id,
    sourceDigest: analysis.sourceDigest,
  }
}

function membershipIndex(
  memberships: readonly FamilyMembership[],
  ablatedFamilyId?: string,
): Map<string, MembershipWeight[]> {
  const grouped = new Map<string, MembershipWeight[]>()
  for (const membership of memberships) {
    if (membership.familyId === ablatedFamilyId || membership.weight <= 0) continue
    const group = grouped.get(membership.memberKey) ?? []
    group.push({ familyId: membership.familyId, weight: membership.weight })
    grouped.set(membership.memberKey, group)
  }
  for (const [memberKey, group] of grouped) {
    group.sort((left, right) => compareText(left.familyId, right.familyId))
    const total = group.reduce((sum, entry) => sum + entry.weight, 0)
    grouped.set(memberKey, total > FAMILY_EPSILON
      ? group.map((entry) => ({ ...entry, weight: entry.weight / total }))
      : [])
  }
  return grouped
}

function leafPairs(analysis: FamilyAnalysisInput): Array<[string, string]> {
  const paragraphs = new Map<number, FamilyAnalysisInput['nodes']>()
  for (const node of analysis.nodes) {
    if (node.childIds.length > 0 && node.splitBoundary !== 'leaf') continue
    const paragraph = paragraphs.get(node.paragraphIndex) ?? []
    paragraph.push(node)
    paragraphs.set(node.paragraphIndex, paragraph)
  }
  const pairs: Array<[string, string]> = []
  for (const leaves of paragraphs.values()) {
    leaves.sort((left, right) => left.start - right.start || left.end - right.end || compareText(left.id, right.id))
    for (let index = 0; index + 1 < leaves.length; index += 1) {
      pairs.push([
        rawMemberKey(analysis.sourceDigest, analysis.runId, leaves[index].id),
        rawMemberKey(analysis.sourceDigest, analysis.runId, leaves[index + 1].id),
      ])
    }
  }
  return pairs
}

function trainingFamilyProbabilities(
  analyses: readonly FamilyAnalysisInput[],
  memberships: ReadonlyMap<string, MembershipWeight[]>,
  heldOutSourceGroup: string | null,
  sourceIndex: IndependentSourceIndex,
  ablatedFamilyId?: string,
): FamilyProbability[] {
  const bySource = new Map<string, Map<string, { fromFamilyId: string; toFamilyId: string; value: number }>>()
  for (const analysis of analyses) {
    const sourceGroup = sourceIndex.groupKey(sourceIdentity(analysis))
    if (heldOutSourceGroup !== null && sourceGroup === heldOutSourceGroup) continue
    const sourceEvidence = bySource.get(sourceGroup) ?? new Map()
    for (const [fromMemberKey, toMemberKey] of leafPairs(analysis)) {
      const fromMemberships = memberships.get(fromMemberKey) ?? []
      const toMemberships = memberships.get(toMemberKey) ?? []
      for (const from of fromMemberships) {
        if (from.familyId === ablatedFamilyId) continue
        for (const to of toMemberships) {
          if (to.familyId === ablatedFamilyId) continue
          const key = `${from.familyId}\u0000${to.familyId}`
          const previous = sourceEvidence.get(key)
          sourceEvidence.set(key, {
            fromFamilyId: from.familyId,
            toFamilyId: to.familyId,
            value: (previous?.value ?? 0) + from.weight * to.weight,
          })
        }
      }
    }
    bySource.set(sourceGroup, sourceEvidence)
  }

  const combined = new Map<string, { fromFamilyId: string; toFamilyId: string; value: number }>()
  for (const sourceEvidence of bySource.values()) {
    // Every physical source contributes one normalized vote, regardless of
    // how many runtime chunks were needed to process the book.
    const sourceTotal = [...sourceEvidence.values()].reduce((sum, entry) => sum + entry.value, 0)
    if (sourceTotal <= FAMILY_EPSILON) continue
    for (const [key, entry] of sourceEvidence) {
      const previous = combined.get(key)
      combined.set(key, {
        ...entry,
        value: (previous?.value ?? 0) + entry.value / sourceTotal,
      })
    }
  }
  const byFrom = new Map<string, Array<{ toFamilyId: string; value: number }>>()
  for (const entry of combined.values()) {
    const row = byFrom.get(entry.fromFamilyId) ?? []
    row.push({ toFamilyId: entry.toFamilyId, value: entry.value })
    byFrom.set(entry.fromFamilyId, row)
  }
  const probabilities: FamilyProbability[] = []
  for (const [fromFamilyId, row] of byFrom) {
    const total = row.reduce((sum, entry) => sum + Math.max(0, entry.value), 0)
    if (total <= FAMILY_EPSILON) continue
    for (const entry of row) probabilities.push({
      fromFamilyId,
      toFamilyId: entry.toFamilyId,
      probability: Math.max(0, entry.value) / total,
    })
  }
  return probabilities.sort((left, right) =>
    compareText(left.fromFamilyId, right.fromFamilyId) ||
    right.probability - left.probability ||
    compareText(left.toFamilyId, right.toFamilyId),
  )
}

function buildCases(
  analysis: FamilyAnalysisInput,
  membershipKeys: ReadonlySet<string>,
  alternativeCount: number,
): PredictionCase[] {
  const paragraphs = new Map<number, FamilyAnalysisInput['nodes']>()
  for (const node of analysis.nodes) {
    if (node.childIds.length > 0 && node.splitBoundary !== 'leaf') continue
    const paragraph = paragraphs.get(node.paragraphIndex) ?? []
    paragraph.push(node)
    paragraphs.set(node.paragraphIndex, paragraph)
  }
  const cases: PredictionCase[] = []
  for (const leaves of paragraphs.values()) {
    leaves.sort((left, right) => left.start - right.start || left.end - right.end || compareText(left.id, right.id))
    const keys = leaves.map((leaf) => rawMemberKey(analysis.sourceDigest, analysis.runId, leaf.id))
    for (let index = 0; index + 1 < keys.length; index += 1) {
      const currentMemberKey = keys[index]
      const targetMemberKey = keys[index + 1]
      if (!membershipKeys.has(currentMemberKey) || !membershipKeys.has(targetMemberKey)) continue
      const alternatives: string[] = [targetMemberKey]
      for (let distance = 2; alternatives.length < alternativeCount + 1 && distance < keys.length; distance += 1) {
        const right = keys[index + distance]
        const left = keys[index - distance + 1]
        if (right && right !== currentMemberKey && membershipKeys.has(right) && !alternatives.includes(right)) alternatives.push(right)
        if (left && left !== currentMemberKey && membershipKeys.has(left) && !alternatives.includes(left)) alternatives.push(left)
      }
      if (alternatives.length < 2) continue
      cases.push({
        sourceId: analysis.source.id,
        sourceDigest: analysis.sourceDigest,
        currentMemberKey,
        targetMemberKey,
        candidateMemberKeys: alternatives,
      })
    }
  }
  return cases
}

function candidateScore(
  current: readonly MembershipWeight[],
  candidate: readonly MembershipWeight[],
  transitionIndex: ReadonlyMap<string, number>,
): number {
  let score = 0
  for (const from of current) {
    for (const to of candidate) {
      score += from.weight * (transitionIndex.get(`${from.familyId}\u0000${to.familyId}`) ?? 0) * to.weight
    }
  }
  return Math.max(0, score)
}

function predictionLoss(
  cases: readonly PredictionCase[],
  memberships: ReadonlyMap<string, MembershipWeight[]>,
  transitions: readonly FamilyProbability[],
  epsilon: number,
): number {
  if (!cases.length) return 0
  const transitionIndex = new Map(
    transitions.map((transition) => [
      `${transition.fromFamilyId}\u0000${transition.toFamilyId}`,
      transition.probability,
    ]),
  )
  let loss = 0
  for (const predictionCase of cases) {
    const current = memberships.get(predictionCase.currentMemberKey) ?? []
    const rawScores = predictionCase.candidateMemberKeys.map((candidateKey) => candidateScore(
      current,
      memberships.get(candidateKey) ?? [],
      transitionIndex,
    ))
    const total = rawScores.reduce((sum, value) => sum + value, 0)
    const targetIndex = predictionCase.candidateMemberKeys.indexOf(predictionCase.targetMemberKey)
    const probability = total > FAMILY_EPSILON
      ? rawScores[targetIndex] / total
      : 1 / predictionCase.candidateMemberKeys.length
    loss -= Math.log(Math.max(epsilon, probability))
  }
  return loss / cases.length
}

function confidenceInterval(values: readonly number[]): { low: number; high: number } {
  if (!values.length) return { low: 0, high: 0 }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  if (values.length === 1) return { low: mean, high: mean }
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
  const margin = 1.96 * Math.sqrt(variance / values.length)
  return { low: mean - margin, high: mean + margin }
}

/**
 * Measures predictive utility on whole held-out physical sources. Every
 * analysis chunk with the same source id is held out together. Families were
 * discovered before this diagnostic, so the result is deliberately not called
 * causal or semantic proof.
 */
export function measureFamilyAblationUtilities(
  analyses: readonly FamilyAnalysisInput[],
  families: readonly LearnedFamily[],
  membershipsInput: readonly FamilyMembership[],
  config: FamilyConfig,
  shouldCancel?: () => boolean,
  onFamily?: (completed: number, total: number) => void,
): FamilyUtilityReport[] {
  const sourceIndex = createIndependentSourceIndex(analyses.map(sourceIdentity))
  const sourceGroups = [...new Set(analyses.map((analysis) =>
    sourceIndex.groupKey(sourceIdentity(analysis))))].sort(compareText)
  const representativeDigest = new Map<string, string>()
  const representativeSourceId = new Map<string, string>()
  for (const analysis of [...analyses].sort((left, right) => compareText(left.sourceDigest, right.sourceDigest))) {
    const group = sourceIndex.groupKey(sourceIdentity(analysis))
    if (!representativeDigest.has(group)) {
      representativeDigest.set(group, analysis.sourceDigest)
      representativeSourceId.set(group, sourceIndex.representativeSourceId(sourceIdentity(analysis)))
    }
  }
  sourceGroups.sort((left, right) =>
    compareText(representativeSourceId.get(left) ?? left, representativeSourceId.get(right) ?? right) ||
    compareText(left, right),
  )
  const fullMemberships = membershipIndex(membershipsInput)
  const membershipKeys = new Set(fullMemberships.keys())
  const casesBySource = new Map<string, PredictionCase[]>()
  for (const analysis of analyses) {
    const cases = buildCases(analysis, membershipKeys, config.utilityAlternativeCount)
    const sourceGroup = sourceIndex.groupKey(sourceIdentity(analysis))
    const groupCases = casesBySource.get(sourceGroup) ?? []
    groupCases.push(...cases)
    casesBySource.set(sourceGroup, groupCases)
  }
  const reports: FamilyUtilityReport[] = []
  families.forEach((family, familyIndex) => {
    if (shouldCancel?.()) throw new Error('FAMILY_CANCELLED')
    const ablatedMemberships = membershipIndex(membershipsInput, family.id)
    const folds: HoldoutFoldMetric[] = []
    for (const heldOutSourceGroup of sourceGroups) {
      const cases = casesBySource.get(heldOutSourceGroup) ?? []
      if (!cases.length || sourceGroups.length - 1 < config.minimumSources) continue
      const fullTransitions = trainingFamilyProbabilities(
        analyses,
        fullMemberships,
        heldOutSourceGroup,
        sourceIndex,
      )
      const ablatedTransitions = trainingFamilyProbabilities(
        analyses,
        ablatedMemberships,
        heldOutSourceGroup,
        sourceIndex,
        family.id,
      )
      const fullLoss = predictionLoss(cases, fullMemberships, fullTransitions, config.utilityEpsilon)
      const ablatedLoss = predictionLoss(cases, ablatedMemberships, ablatedTransitions, config.utilityEpsilon)
      folds.push({
        heldOutSourceId: representativeSourceId.get(heldOutSourceGroup) ?? heldOutSourceGroup,
        heldOutSourceDigest: representativeDigest.get(heldOutSourceGroup) ?? heldOutSourceGroup,
        caseCount: cases.length,
        fullLoss,
        ablatedLoss,
        deltaUtility: ablatedLoss - fullLoss,
      })
    }
    const caseCount = folds.reduce((sum, fold) => sum + fold.caseCount, 0)
    const weighted = (selector: (fold: HoldoutFoldMetric) => number) => caseCount
      ? folds.reduce((sum, fold) => sum + selector(fold) * fold.caseCount, 0) / caseCount
      : 0
    const fullLoss = weighted((fold) => fold.fullLoss)
    const ablatedLoss = weighted((fold) => fold.ablatedLoss)
    const deltaUtility = ablatedLoss - fullLoss
    const interval = confidenceInterval(folds.map((fold) => fold.deltaUtility))
    const sufficient = caseCount >= config.utilityMinimumCases && folds.length >= 2
    const validated = sufficient && deltaUtility > config.lifecycleUtilityThreshold && interval.low > 0
    reports.push({
      familyId: family.id,
      mode: 'source-holdout-diagnostic',
      status: !sufficient
        ? 'insufficient-evidence'
        : validated
          ? 'validated'
          : 'neutral-or-harmful',
      caseCount,
      sourceFoldCount: folds.length,
      fullLoss,
      ablatedLoss,
      deltaUtility,
      confidenceLow: interval.low,
      confidenceHigh: interval.high,
      folds,
      explanation: !sufficient
        ? 'Bağımsız kaynak ve gizli komşuluk vakası kararlı bir fayda yargısı için yetersiz.'
        : validated
          ? 'Aile susturulduğunda aynı aday evrenindeki kaynak-holdout tahmin kaybı güvenilir biçimde arttı.'
          : 'Aile susturma kaybı pozitif ve kararlı değil; bu aile henüz işlevsel olarak doğrulanmadı.',
    })
    onFamily?.(familyIndex + 1, families.length)
  })
  return reports.sort((left, right) => compareText(left.familyId, right.familyId))
}

/**
 * Measures a discovery-trained family's utility on a separately declared
 * evaluation corpus. Family probabilities are fitted only from discovery
 * analyses; evaluation analyses contribute cases and out-of-sample soft
 * memberships, never prototypes or transition evidence.
 */
export function measureFamilyAblationUtilitiesOnEvaluation(
  discoveryAnalyses: readonly FamilyAnalysisInput[],
  evaluationAnalyses: readonly FamilyAnalysisInput[],
  families: readonly LearnedFamily[],
  membershipsInput: readonly FamilyMembership[],
  config: FamilyConfig,
  shouldCancel?: () => boolean,
  onFamily?: (completed: number, total: number) => void,
): FamilyUtilityReport[] {
  const discoverySourceIndex = createIndependentSourceIndex(discoveryAnalyses.map(sourceIdentity))
  const evaluationSourceIndex = createIndependentSourceIndex(evaluationAnalyses.map(sourceIdentity))
  const evaluationSourceGroups = [...new Set(evaluationAnalyses.map((analysis) =>
    evaluationSourceIndex.groupKey(sourceIdentity(analysis))))]
  const representativeDigest = new Map<string, string>()
  const representativeSourceId = new Map<string, string>()
  for (const analysis of [...evaluationAnalyses].sort((left, right) =>
    compareText(left.sourceDigest, right.sourceDigest) || compareText(left.runId, right.runId))) {
    const group = evaluationSourceIndex.groupKey(sourceIdentity(analysis))
    if (!representativeDigest.has(group)) {
      representativeDigest.set(group, analysis.sourceDigest)
      representativeSourceId.set(group, evaluationSourceIndex.representativeSourceId(sourceIdentity(analysis)))
    }
  }
  evaluationSourceGroups.sort((left, right) =>
    compareText(representativeSourceId.get(left) ?? left, representativeSourceId.get(right) ?? right) ||
    compareText(left, right),
  )

  const fullMemberships = membershipIndex(membershipsInput)
  const membershipKeys = new Set(fullMemberships.keys())
  const casesBySource = new Map<string, PredictionCase[]>()
  for (const analysis of evaluationAnalyses) {
    const sourceGroup = evaluationSourceIndex.groupKey(sourceIdentity(analysis))
    const cases = casesBySource.get(sourceGroup) ?? []
    cases.push(...buildCases(analysis, membershipKeys, config.utilityAlternativeCount))
    casesBySource.set(sourceGroup, cases)
  }

  const reports: FamilyUtilityReport[] = []
  families.forEach((family, familyIndex) => {
    if (shouldCancel?.()) throw new Error('FAMILY_CANCELLED')
    const ablatedMemberships = membershipIndex(membershipsInput, family.id)
    const fullTransitions = trainingFamilyProbabilities(
      discoveryAnalyses,
      fullMemberships,
      null,
      discoverySourceIndex,
    )
    const ablatedTransitions = trainingFamilyProbabilities(
      discoveryAnalyses,
      ablatedMemberships,
      null,
      discoverySourceIndex,
      family.id,
    )
    const folds: HoldoutFoldMetric[] = []
    for (const sourceGroup of evaluationSourceGroups) {
      const cases = casesBySource.get(sourceGroup) ?? []
      if (!cases.length) continue
      const fullLoss = predictionLoss(cases, fullMemberships, fullTransitions, config.utilityEpsilon)
      const ablatedLoss = predictionLoss(cases, ablatedMemberships, ablatedTransitions, config.utilityEpsilon)
      folds.push({
        heldOutSourceId: representativeSourceId.get(sourceGroup) ?? sourceGroup,
        heldOutSourceDigest: representativeDigest.get(sourceGroup) ?? sourceGroup,
        caseCount: cases.length,
        fullLoss,
        ablatedLoss,
        deltaUtility: ablatedLoss - fullLoss,
      })
    }
    const caseCount = folds.reduce((sum, fold) => sum + fold.caseCount, 0)
    const weighted = (selector: (fold: HoldoutFoldMetric) => number) => caseCount
      ? folds.reduce((sum, fold) => sum + selector(fold) * fold.caseCount, 0) / caseCount
      : 0
    const fullLoss = weighted((fold) => fold.fullLoss)
    const ablatedLoss = weighted((fold) => fold.ablatedLoss)
    const deltaUtility = ablatedLoss - fullLoss
    const interval = confidenceInterval(folds.map((fold) => fold.deltaUtility))
    const sufficient = caseCount >= config.utilityMinimumCases && folds.length >= 2
    const validated = sufficient && deltaUtility > config.lifecycleUtilityThreshold && interval.low > 0
    reports.push({
      familyId: family.id,
      mode: 'declared-evaluation-diagnostic',
      status: !sufficient
        ? 'insufficient-evidence'
        : validated
          ? 'validated'
          : 'neutral-or-harmful',
      caseCount,
      sourceFoldCount: folds.length,
      fullLoss,
      ablatedLoss,
      deltaUtility,
      confidenceLow: interval.low,
      confidenceHigh: interval.high,
      folds,
      explanation: !sufficient
        ? 'Ayrı değerlendirme kaynakları ve gizli komşuluk vakaları kararlı bir fayda yargısı için yetersiz.'
        : validated
          ? 'Keşif verisinden öğrenilen aile susturulduğunda ayrı değerlendirme korpusundaki tahmin kaybı güvenilir biçimde arttı.'
          : 'Ayrı değerlendirme korpusunda aile susturma kaybı pozitif ve kararlı değil; aile henüz işlevsel olarak doğrulanmadı.',
    })
    onFamily?.(familyIndex + 1, families.length)
  })
  return reports.sort((left, right) => compareText(left.familyId, right.familyId))
}

export function applyUtilityLifecycle(
  families: readonly LearnedFamily[],
  reports: readonly FamilyUtilityReport[],
  config: FamilyConfig,
): LearnedFamily[] {
  const reportByFamily = new Map(reports.map((report) => [report.familyId, report]))
  return families.map((family) => {
    const report = reportByFamily.get(family.id)
    if (!report) return family
    if (
      family.lifecycle === 'strengthening' &&
      family.stability >= config.lifecycleStabilityThreshold &&
      report.status === 'validated' &&
      report.deltaUtility >= config.lifecycleUtilityThreshold
    ) {
      return {
        ...family,
        lifecycle: 'stable',
        lifecycleReason: 'Önceki koşuya göre kararlı kaldı ve kaynak-holdout susturma testi pozitif faydayı doğruladı.',
        utility: report.deltaUtility,
      }
    }
    return {
      ...family,
      utility: report.deltaUtility,
      lifecycleReason: family.lifecycle === 'provisional'
        ? `${family.lifecycleReason} ${report.explanation}`
        : family.lifecycleReason,
    }
  })
}

export function familyUtilityCaseCount(reports: readonly FamilyUtilityReport[]): number {
  return reports.reduce((sum, report) => sum + report.caseCount, 0)
}

export function boundedUtility(value: number): number {
  return clamp(value, -20, 20)
}
