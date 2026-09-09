import type {
  BehaviorProfile,
  FamilyConfig,
  FamilyMembership,
  FamilyRun,
  FunctionalRegionId,
  IndependentEvidenceRef,
  LearnedFamily,
  MembershipScoreBreakdown,
} from './types'
import {
  profileEvidenceConfidence,
  profileRegionScore,
  selectIndependentEvidence,
} from './profile'
import {
  clamp,
  compareText,
  normalizedPositive,
  sparseCosine,
  sparseVectorDigest,
  stableDigest,
  weightedSparseMean,
} from './sparse'
import { createIndependentSourceIndex } from './independence'
import type { IndependentSourceIndex } from './independence'

export interface BehaviorEdge {
  leftMemberKey: string
  rightMemberKey: string
  similarity: number
}

export interface FamilyDiscoveryStats {
  candidateComparisonCount: number
  retainedBehaviorEdgeCount: number
}

export interface FamilyDiscoveryResult {
  families: LearnedFamily[]
  memberships: FamilyMembership[]
  edges: BehaviorEdge[]
  stats: FamilyDiscoveryStats
}

const STATIC_CLASSIFIABLE_REGIONS = ['content', 'event', 'role', 'operator'] as const
export const STATIC_REGION_MINIMUM_SCORE = 0.5
export const STATIC_REGION_MINIMUM_MARGIN = 0.08

type StructuralSeedBand = 'root' | 'shallow-parent' | 'middle-parent' | 'deep-parent'

const STRUCTURAL_SEED_BANDS: readonly StructuralSeedBand[] = [
  'root',
  'shallow-parent',
  'middle-parent',
  'deep-parent',
]
const MAXIMUM_ADDITIONAL_MULTISCALE_SEEDS = 4_096

function independentSourceId(
  profile: BehaviorProfile,
  sourceIndex: IndependentSourceIndex,
): string {
  const evidence = profile.evidence[0]
  return evidence ? sourceIndex.groupKey(evidence) : ''
}

function edgeKey(left: string, right: string): string {
  return left < right ? `${left}\u0001${right}` : `${right}\u0001${left}`
}

function compareEdge(left: BehaviorEdge, right: BehaviorEdge): number {
  return (
    right.similarity - left.similarity ||
    compareText(left.leftMemberKey, right.leftMemberKey) ||
    compareText(left.rightMemberKey, right.rightMemberKey)
  )
}

function boundedBehaviorEdges(
  profiles: readonly BehaviorProfile[],
  config: FamilyConfig,
  sourceIndex: IndependentSourceIndex,
  shouldCancel?: () => boolean,
): { edges: BehaviorEdge[]; comparisons: number } {
  const ordered = [...profiles].sort((left, right) => compareText(left.memberKey, right.memberKey))
  const buckets = new Map<string, string[]>()
  const profileByKey = new Map(ordered.map((profile) => [profile.memberKey, profile]))
  const retainedByMember = new Map<string, BehaviorEdge[]>()
  let comparisons = 0

  const retain = (memberKey: string, edge: BehaviorEdge) => {
    const bucket = retainedByMember.get(memberKey) ?? []
    bucket.push(edge)
    bucket.sort(compareEdge)
    if (bucket.length > config.maximumCandidateNeighbors) bucket.length = config.maximumCandidateNeighbors
    retainedByMember.set(memberKey, bucket)
  }

  for (const profile of ordered) {
    if (shouldCancel?.()) throw new Error('FAMILY_CANCELLED')
    const shared = new Map<string, number>()
    for (const coordinate of profile.vector.entries) {
      const bucketKey = `${coordinate.index}:${coordinate.value >= 0 ? '+' : '-'}`
      const bucket = buckets.get(bucketKey) ?? []
      for (const candidateKey of bucket) {
        shared.set(candidateKey, (shared.get(candidateKey) ?? 0) + Math.abs(coordinate.value))
      }
    }
    const candidates = [...shared.entries()]
      .filter(([candidateKey]) => {
        const candidate = profileByKey.get(candidateKey)
        return candidate && independentSourceId(candidate, sourceIndex) !== independentSourceId(profile, sourceIndex)
      })
      .sort((left, right) => right[1] - left[1] || compareText(left[0], right[0]))
      .slice(0, config.maximumCandidateNeighbors * 4)
    for (const [candidateKey] of candidates) {
      const candidate = profileByKey.get(candidateKey)
      if (!candidate) continue
      comparisons += 1
      const similarity = sparseCosine(profile.vector, candidate.vector)
      if (similarity < config.minimumFormationSimilarity) continue
      const [leftMemberKey, rightMemberKey] = profile.memberKey < candidate.memberKey
        ? [profile.memberKey, candidate.memberKey]
        : [candidate.memberKey, profile.memberKey]
      const edge = { leftMemberKey, rightMemberKey, similarity }
      retain(leftMemberKey, edge)
      retain(rightMemberKey, edge)
    }
    for (const coordinate of profile.vector.entries) {
      const bucketKey = `${coordinate.index}:${coordinate.value >= 0 ? '+' : '-'}`
      const bucket = buckets.get(bucketKey) ?? []
      bucket.push(profile.memberKey)
      if (bucket.length > config.maximumIndexBucketSize) bucket.shift()
      buckets.set(bucketKey, bucket)
    }
  }

  const unique = new Map<string, BehaviorEdge>()
  for (const memberEdges of retainedByMember.values()) {
    for (const edge of memberEdges) unique.set(edgeKey(edge.leftMemberKey, edge.rightMemberKey), edge)
  }
  // The union of endpoint-local top-k lists is still O(nk), but a popular hub
  // can exceed k in that union. This deterministic pass makes the advertised
  // per-member graph bound real without materializing all n² pairs.
  const degree = new Map<string, number>()
  const edges: BehaviorEdge[] = []
  for (const edge of [...unique.values()].sort(compareEdge)) {
    if (
      (degree.get(edge.leftMemberKey) ?? 0) >= config.maximumCandidateNeighbors ||
      (degree.get(edge.rightMemberKey) ?? 0) >= config.maximumCandidateNeighbors
    ) continue
    edges.push(edge)
    degree.set(edge.leftMemberKey, (degree.get(edge.leftMemberKey) ?? 0) + 1)
    degree.set(edge.rightMemberKey, (degree.get(edge.rightMemberKey) ?? 0) + 1)
  }
  return { edges, comparisons }
}

class CompleteLinkClusters {
  private readonly parent = new Map<string, string>()
  private readonly members = new Map<string, string[]>()

  constructor(
    memberKeys: readonly string[],
    private readonly profiles: ReadonlyMap<string, BehaviorProfile>,
    private readonly config: FamilyConfig,
  ) {
    for (const key of memberKeys) {
      this.parent.set(key, key)
      this.members.set(key, [key])
    }
  }

  private root(key: string): string {
    const parent = this.parent.get(key)
    if (!parent) throw new Error(`Bilinmeyen davranış kümesi üyesi: ${key}`)
    if (parent === key) return key
    const root = this.root(parent)
    this.parent.set(key, root)
    return root
  }

  private compatible(left: readonly string[], right: readonly string[]): boolean {
    for (const leftKey of left) {
      const leftProfile = this.profiles.get(leftKey)
      if (!leftProfile) return false
      for (const rightKey of right) {
        const rightProfile = this.profiles.get(rightKey)
        if (!rightProfile) return false
        if (sparseCosine(leftProfile.vector, rightProfile.vector) < this.config.minimumFormationSimilarity) {
          return false
        }
      }
    }
    return true
  }

  union(leftKey: string, rightKey: string): boolean {
    const leftRoot = this.root(leftKey)
    const rightRoot = this.root(rightKey)
    if (leftRoot === rightRoot) return true
    const left = this.members.get(leftRoot) ?? []
    const right = this.members.get(rightRoot) ?? []
    if (left.length + right.length > this.config.maximumFamilyMembers) return false
    if (!this.compatible(left, right)) return false
    const [root, child] = leftRoot < rightRoot ? [leftRoot, rightRoot] : [rightRoot, leftRoot]
    const merged = [...(this.members.get(root) ?? []), ...(this.members.get(child) ?? [])].sort(compareText)
    this.parent.set(child, root)
    this.members.set(root, merged)
    this.members.delete(child)
    return true
  }

  groups(): string[][] {
    return [...this.members.values()]
      .filter((members) => members.length > 1)
      .map((members) => [...members].sort(compareText))
      .sort((left, right) => compareText(left[0], right[0]))
  }
}

function averageSimilarity(profiles: readonly BehaviorProfile[]): number {
  if (profiles.length < 2) return 1
  let sum = 0
  let count = 0
  for (let left = 0; left < profiles.length; left += 1) {
    for (let right = left + 1; right < profiles.length; right += 1) {
      sum += sparseCosine(profiles[left].vector, profiles[right].vector)
      count += 1
    }
  }
  return count ? sum / count : 1
}

function chooseRegion(profiles: readonly BehaviorProfile[]): FunctionalRegionId {
  if (!profiles.length) return 'unknown'
  const ranked = STATIC_CLASSIFIABLE_REGIONS
    .map((region) => ({
      region,
      score: profiles.reduce((sum, profile) => sum + profileRegionScore(profile, region), 0) /
        Math.max(1, profiles.length),
    }))
    .sort((left, right) => right.score - left.score || compareText(left.region, right.region))
  const top = ranked[0]
  const runnerUp = ranked[1]
  if (
    !top ||
    top.score < STATIC_REGION_MINIMUM_SCORE ||
    !runnerUp ||
    top.score - runnerUp.score < STATIC_REGION_MINIMUM_MARGIN
  ) return 'unknown'
  return top.region
}

function structuralSeedBand(profile: BehaviorProfile): StructuralSeedBand {
  if (profile.diagnostics.depthScale <= 0) return 'root'
  if (profile.diagnostics.depthScale <= 0.25) return 'shallow-parent'
  if (profile.diagnostics.depthScale <= 0.6) return 'middle-parent'
  return 'deep-parent'
}

function interleaveSourceGroups(
  groups: ReadonlyMap<string, readonly BehaviorProfile[]>,
): BehaviorProfile[] {
  const sourceGroups = [...groups.keys()].sort(compareText)
  const cursors = new Map(sourceGroups.map((sourceGroup) => [sourceGroup, 0]))
  const result: BehaviorProfile[] = []
  let progressed = true
  while (progressed) {
    progressed = false
    for (const sourceGroup of sourceGroups) {
      const profiles = groups.get(sourceGroup) ?? []
      const cursor = cursors.get(sourceGroup) ?? 0
      const profile = profiles[cursor]
      if (!profile) continue
      result.push(profile)
      cursors.set(sourceGroup, cursor + 1)
      progressed = true
    }
  }
  return result
}

/**
 * Keeps every leaf seed for backwards coverage, then admits a bounded,
 * source-balanced sample of roots and internal parents at several depths.
 * This prevents the family learner from mistaking "is a short leaf" for a
 * universal semantic role while retaining deterministic memory bounds.
 */
function multiScaleSeedProfiles(
  rawProfiles: readonly BehaviorProfile[],
  leafMemberKeys: ReadonlySet<string>,
  config: FamilyConfig,
  sourceIndex: IndependentSourceIndex,
): BehaviorProfile[] {
  const ordered = [...rawProfiles].sort((left, right) => compareText(left.memberKey, right.memberKey))
  const leaves = ordered.filter((profile) => leafMemberKeys.has(profile.memberKey))
  const parents = ordered.filter((profile) => !leafMemberKeys.has(profile.memberKey))
  if (!parents.length) return leaves

  const byBand = new Map<StructuralSeedBand, Map<string, BehaviorProfile[]>>()
  for (const profile of parents) {
    const band = structuralSeedBand(profile)
    const sourceGroup = independentSourceId(profile, sourceIndex) || profile.memberKey
    const bySource = byBand.get(band) ?? new Map<string, BehaviorProfile[]>()
    const profiles = bySource.get(sourceGroup) ?? []
    profiles.push(profile)
    bySource.set(sourceGroup, profiles)
    byBand.set(band, bySource)
  }
  for (const bySource of byBand.values()) {
    for (const profiles of bySource.values()) {
      profiles.sort((left, right) => compareText(left.memberKey, right.memberKey))
    }
  }

  const minimumBudget = Math.max(config.minimumSupport, config.minimumSources) * STRUCTURAL_SEED_BANDS.length
  const corpusBudget = Math.ceil(Math.sqrt(Math.max(1, rawProfiles.length)) * STRUCTURAL_SEED_BANDS.length)
  const budget = Math.min(
    parents.length,
    MAXIMUM_ADDITIONAL_MULTISCALE_SEEDS,
    Math.max(minimumBudget, corpusBudget),
  )
  const queues = new Map(STRUCTURAL_SEED_BANDS.map((band) => [
    band,
    interleaveSourceGroups(byBand.get(band) ?? new Map()),
  ]))
  const cursors = new Map(STRUCTURAL_SEED_BANDS.map((band) => [band, 0]))
  const selected: BehaviorProfile[] = []
  let progressed = true
  while (selected.length < budget && progressed) {
    progressed = false
    for (const band of STRUCTURAL_SEED_BANDS) {
      if (selected.length >= budget) break
      const queue = queues.get(band) ?? []
      const cursor = cursors.get(band) ?? 0
      const profile = queue[cursor]
      if (!profile) continue
      selected.push(profile)
      cursors.set(band, cursor + 1)
      progressed = true
    }
  }
  return [...leaves, ...selected].sort((left, right) => compareText(left.memberKey, right.memberKey))
}

function evidenceProfiles(
  members: readonly BehaviorProfile[],
  evidence: readonly IndependentEvidenceRef[],
): BehaviorProfile[] {
  const keys = new Set(evidence.map((item) => item.memberKey))
  const selected = members.filter((profile) => keys.has(profile.memberKey))
  return selected.length ? selected : [...members]
}

/** Prevents one large book split into many chunks from dominating a family prototype. */
function sourceBalancedProfileWeights(
  profiles: readonly BehaviorProfile[],
  sourceIndex: IndependentSourceIndex,
): number[] {
  const sourcesByProfile = profiles.map((profile) => [...new Set(
    profile.evidence.map((item) => sourceIndex.groupKey(item)),
  )])
  const counts = new Map<string, number>()
  for (const sources of sourcesByProfile) {
    for (const source of sources) counts.set(source, (counts.get(source) ?? 0) + 1)
  }
  return sourcesByProfile.map((sources) => {
    if (!sources.length) return 1
    return sources.reduce((sum, source) => sum + 1 / Math.max(1, counts.get(source) ?? 1), 0) / sources.length
  })
}

function priorMatch(
  prototype: LearnedFamily['prototype'],
  behaviorSpaceId: string,
  region: FunctionalRegionId,
  memberKeys: readonly string[],
  previousRun: FamilyRun | null | undefined,
): { predecessor: LearnedFamily | null; stability: number } {
  if (!previousRun) return { predecessor: null, stability: 0 }
  let best: LearnedFamily | null = null
  let bestStability = 0
  const members = new Set(memberKeys)
  for (const candidate of previousRun.families) {
    if (
      candidate.behaviorSpaceId !== behaviorSpaceId ||
      candidate.region !== region ||
      candidate.lifecycle === 'retired'
    ) continue
    const similarity = Math.max(0, sparseCosine(prototype, candidate.prototype))
    const priorMembers = new Set(candidate.memberKeys)
    let intersection = 0
    for (const member of members) if (priorMembers.has(member)) intersection += 1
    const union = new Set([...members, ...priorMembers]).size
    const jaccard = union ? intersection / union : 0
    const stability = 0.7 * similarity + 0.3 * jaccard
    if (stability > bestStability || (stability === bestStability && candidate.id < (best?.id ?? '\uffff'))) {
      best = candidate
      bestStability = stability
    }
  }
  return { predecessor: best, stability: bestStability }
}

function buildFamily(
  memberKeys: readonly string[],
  profileByKey: ReadonlyMap<string, BehaviorProfile>,
  config: FamilyConfig,
  sourceIndex: IndependentSourceIndex,
  previousRun: FamilyRun | null | undefined,
): LearnedFamily | null {
  const members = memberKeys
    .map((key) => profileByKey.get(key))
    .filter((profile): profile is BehaviorProfile => profile !== undefined)
  const evidence = selectIndependentEvidence(members.flatMap((profile) => profile.evidence))
  const sourceCount = new Set(evidence.map((item) => sourceIndex.groupKey(item))).size
  if (evidence.length < config.minimumSupport || sourceCount < config.minimumSources) return null
  const coreProfiles = evidenceProfiles(members, evidence)
  const sourceBalancedWeights = sourceBalancedProfileWeights(coreProfiles, sourceIndex)
  const prototype = weightedSparseMean(
    coreProfiles.map((profile, index) => ({ vector: profile.vector, weight: sourceBalancedWeights[index] })),
    config.maximumProfileFeatures,
  )
  const prototypeDigest = sparseVectorDigest(prototype)
  const region = chooseRegion(coreProfiles)
  const fingerprint = stableDigest([
    'learned-family-v1',
    members[0].behaviorSpaceId,
    region,
    prototypeDigest,
    ...memberKeys,
    ...evidence.map((item) => [
      sourceIndex.groupKey(item),
      item.sourceDigest,
      item.paragraphIndex,
      item.start,
      item.end,
    ].join(':')),
  ])
  const prior = priorMatch(prototype, members[0].behaviorSpaceId, region, memberKeys, previousRun)
  const strengthening = Boolean(
    prior.predecessor && prior.stability >= config.lifecycleStabilityThreshold,
  )
  return {
    id: `AF-${fingerprint.slice(0, 16)}`,
    fingerprint,
    behaviorSpaceId: members[0].behaviorSpaceId,
    region,
    regionVerified: false,
    lifecycle: strengthening ? 'strengthening' : 'provisional',
    lifecycleReason: strengthening
      ? 'Önceki koşudaki aile prototipi ve üyeleriyle yeterli kararlılık gösterdi; held-out fayda bekleniyor.'
      : 'Bağımsız statik kanıttan doğdu; sonraki korpus koşusu ve held-out fayda henüz gerekli.',
    prototype,
    prototypeDigest,
    memberKeys: [...memberKeys].sort(compareText),
    coreMemberKeys: coreProfiles.map((profile) => profile.memberKey).sort(compareText),
    independentEvidence: evidence,
    supportCount: evidence.length,
    sourceCount,
    cohesion: averageSimilarity(coreProfiles),
    separation: 1,
    stability: prior.stability,
    utility: null,
    predecessorFamilyIds: prior.predecessor ? [prior.predecessor.id] : [],
  }
}

function updateSeparationAndProposals(
  input: readonly LearnedFamily[],
  config: FamilyConfig,
): LearnedFamily[] {
  const families = input.map((family) => ({ ...family }))
  const mergeThreshold = Math.max(0.92, config.minimumFormationSimilarity + 0.2)
  for (let index = 0; index < families.length; index += 1) {
    let nearest = 0
    let mergeCandidate: LearnedFamily | null = null
    for (let otherIndex = 0; otherIndex < families.length; otherIndex += 1) {
      if (index === otherIndex) continue
      const similarity = sparseCosine(families[index].prototype, families[otherIndex].prototype)
      if (similarity > nearest) nearest = similarity
      if (
        families[index].region === families[otherIndex].region &&
        similarity >= mergeThreshold &&
        (!mergeCandidate || families[otherIndex].id < mergeCandidate.id)
      ) mergeCandidate = families[otherIndex]
    }
    families[index].separation = clamp(1 - nearest)
    if (mergeCandidate && families[index].id < mergeCandidate.id) {
      families[index].lifecycle = 'merge-proposed'
      families[index].lifecycleReason = `Davranış prototipi ${mergeCandidate.id} ile aşırı yakın; birleştirme yalnız fayda testi sonrası önerildi.`
      families[index].predecessorFamilyIds = [...new Set([
        ...families[index].predecessorFamilyIds,
        mergeCandidate.id,
      ])].sort(compareText)
    } else if (
      families[index].supportCount >= config.minimumSupport * 2 &&
      families[index].cohesion < config.minimumFormationSimilarity + 0.03
    ) {
      families[index].lifecycle = 'split-proposed'
      families[index].lifecycleReason = 'Destek çokluğu yanında sınırda iç uyum görüldü; iki alt prototip için bölünme deneyi önerildi.'
    }
  }
  return families
}

export function scoreFamilyMembership(
  profile: BehaviorProfile,
  family: LearnedFamily,
): MembershipScoreBreakdown {
  const behaviorSimilarity = Math.max(0, sparseCosine(profile.vector, family.prototype))
  const regionCompatibility = profileRegionScore(profile, family.region)
  const evidenceConfidence = profile.kind === 'synthesis'
    ? profileEvidenceConfidence(profile)
    : 0.5 + 0.5 * profileEvidenceConfidence(profile)
  return {
    behaviorSimilarity,
    regionCompatibility,
    evidenceConfidence,
    combined: clamp(
      0.72 * behaviorSimilarity +
      0.18 * regionCompatibility +
      0.1 * evidenceConfidence,
    ),
  }
}

export function assignSoftMemberships(
  profiles: readonly BehaviorProfile[],
  families: readonly LearnedFamily[],
  config: FamilyConfig,
): FamilyMembership[] {
  const memberships: FamilyMembership[] = []
  for (const profile of [...profiles].sort((left, right) => compareText(left.memberKey, right.memberKey))) {
    const candidates = families
      .map((family) => ({ family, breakdown: scoreFamilyMembership(profile, family) }))
      .filter((candidate) => candidate.breakdown.combined >= config.minimumMembershipScore)
      .sort((left, right) =>
        right.breakdown.combined - left.breakdown.combined || compareText(left.family.id, right.family.id),
      )
      .slice(0, config.maximumMembershipsPerNeuron)
      .map((candidate) => ({ ...candidate, score: candidate.breakdown.combined }))
    const normalized = normalizedPositive(candidates)
    normalized.forEach((candidate, index) => memberships.push({
      memberKey: profile.memberKey,
      familyId: candidate.family.id,
      rank: index + 1,
      score: candidate.score,
      weight: candidate.weight,
      core: candidate.score >= config.minimumCoreMembership,
      breakdown: candidate.breakdown,
    }))
  }
  return memberships.sort((left, right) =>
    compareText(left.memberKey, right.memberKey) || left.rank - right.rank || compareText(left.familyId, right.familyId),
  )
}

export function discoverFamilies(
  rawProfiles: readonly BehaviorProfile[],
  allProfiles: readonly BehaviorProfile[],
  leafMemberKeys: ReadonlySet<string>,
  config: FamilyConfig,
  previousRun?: FamilyRun | null,
  shouldCancel?: () => boolean,
): FamilyDiscoveryResult {
  const sourceIndex = createIndependentSourceIndex(rawProfiles.flatMap((profile) => profile.evidence))
  const seedProfiles = multiScaleSeedProfiles(rawProfiles, leafMemberKeys, config, sourceIndex)
  const profileByKey = new Map(seedProfiles.map((profile) => [profile.memberKey, profile]))
  const edgeResult = boundedBehaviorEdges(seedProfiles, config, sourceIndex, shouldCancel)
  const touched = new Set<string>()
  for (const edge of edgeResult.edges) {
    touched.add(edge.leftMemberKey)
    touched.add(edge.rightMemberKey)
  }
  const clusters = new CompleteLinkClusters([...touched].sort(compareText), profileByKey, config)
  for (const edge of edgeResult.edges) {
    if (shouldCancel?.()) throw new Error('FAMILY_CANCELLED')
    clusters.union(edge.leftMemberKey, edge.rightMemberKey)
  }
  const eligibleFamilies = clusters.groups()
    .map((group) => buildFamily(group, profileByKey, config, sourceIndex, previousRun))
    .filter((family): family is LearnedFamily => family !== null)
    .sort((left, right) =>
      right.supportCount - left.supportCount ||
      right.cohesion - left.cohesion ||
      compareText(left.fingerprint, right.fingerprint),
    )
  // maximumFamilies remains an opt-in experiment/resource budget. The
  // production default is Number.MAX_SAFE_INTEGER, so corpus evidence—not a
  // historical 128 constant—sets the natural upper bound.
  let families = eligibleFamilies.length > config.maximumFamilies
    ? eligibleFamilies.slice(0, config.maximumFamilies)
    : eligibleFamilies
  families = updateSeparationAndProposals(families, config)
  const memberships = assignSoftMemberships(allProfiles, families, config)
  // Soft M assignments are deliberately kept outside the evidence cluster.
  // Re-inserting them here would bypass complete-link and capacity guarantees.
  return {
    families,
    memberships,
    edges: edgeResult.edges,
    stats: {
      candidateComparisonCount: edgeResult.comparisons,
      retainedBehaviorEdgeCount: edgeResult.edges.length,
    },
  }
}
