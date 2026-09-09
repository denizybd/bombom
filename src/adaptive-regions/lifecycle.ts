import {
  createPlasticRegionAtlas,
  MINIMUM_ACTIVE_REGION_CAPACITY,
  plasticRegionCapacity,
  REGION_MASS_EPSILON,
  validateAdaptiveRegionAtlas,
} from './atlas.ts'
import {
  adaptiveRegionDigest,
  canonicalRegionFloat,
  canonicalRegionJson,
  compareRegionText,
  immutableRegionValue,
} from './deterministic.ts'
import { ADAPTIVE_PAGE_COUNT } from './geometry.ts'
import type {
  PlasticPageAssignmentV1,
  PlasticRegionAtlasV1,
  PlasticRegionEvidenceV1,
  PlasticRegionMembershipV1,
  PlasticRegionPrototypeV1,
  PlasticRegionV1,
  RegionAllocationChangeV1,
  RegionEvidenceCode,
  RegionLifecycleAction,
  RegionLifecycleEvidenceV1,
  RegionLifecyclePolicyV1,
  RegionLifecycleProposalV1,
  RegionPageDeltaV1,
  RegionProposalRejection,
  RegionSplitFractionV1,
} from './types.ts'
import { ADAPTIVE_REGION_PROPOSAL_FORMAT } from './types.ts'

const EVIDENCE_CODES = new Set<RegionEvidenceCode>([
  'unexplained-residual',
  'stable-coactivation',
  'capacity-pressure',
  'counterfactual-utility',
  'bimodal-behavior',
  'redundant-behavior',
  'low-unique-utility',
])
const LIFECYCLE_ACTIONS = new Set<RegionLifecycleAction>([
  'birth', 'grow', 'shrink', 'split', 'merge', 'retire',
])
const VALIDATION_AUTHORITIES = new Set(['train-only', 'independent-holdout'])

export const DEFAULT_REGION_LIFECYCLE_POLICY: RegionLifecyclePolicyV1 = Object.freeze({
  minimumObservationCount: 3,
  minimumIndependentSources: 3,
  minimumConfidenceMass: 1,
  minimumCohesion: 0.65,
  minimumConstructiveHoldoutGain: 0.01,
  maximumShamGain: 0.03,
  maximumNonRegressionLoss: 0.01,
})

interface MutablePage {
  pageIndex: number
  memberships: PlasticRegionMembershipV1[]
  freeMass: number
}

function bounded(value: number, minimum: number, maximum: number, label: string): number {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} ${minimum}..${maximum} aralığında olmalıdır.`)
  }
  return canonicalRegionFloat(value)
}

function nonEmpty(value: string, label: string): string {
  const result = String(value ?? '').trim()
  if (!result) throw new TypeError(`${label} boş olamaz.`)
  return result
}

function canonicalIds(values: readonly string[], label: string): string[] {
  const result = values.map((value) => nonEmpty(value, label)).sort(compareRegionText)
  if (new Set(result).size !== result.length) throw new TypeError(`${label} yinelenen değer içeremez.`)
  return result
}

export function canonicalRegionLifecycleEvidence(
  evidence: RegionLifecycleEvidenceV1,
): RegionLifecycleEvidenceV1 {
  if (!Number.isInteger(evidence.observationCount) || evidence.observationCount < 1) {
    throw new RangeError('Yaşam döngüsü kanıtı en az bir gözlem içermelidir.')
  }
  const sourceGroupIds = canonicalIds(evidence.sourceGroupIds, 'Kanıt kaynak grubu')
  if (!sourceGroupIds.length || evidence.observationCount < sourceGroupIds.length) {
    throw new RangeError('Kanıt gözlem sayısı bağımsız kaynak sayısından az olamaz.')
  }
  const confidenceMass = bounded(evidence.confidenceMass, 0, sourceGroupIds.length, 'Kanıt güven kütlesi')
  const codes = [...new Set(evidence.codes)]
    .sort(compareRegionText) as RegionEvidenceCode[]
  if (!codes.length || codes.some((code) => !EVIDENCE_CODES.has(code))) {
    throw new TypeError('Yaşam döngüsü kanıtı en az bir bilinen yapısal gerekçe ister.')
  }
  if (!VALIDATION_AUTHORITIES.has(evidence.validationAuthority)) {
    throw new TypeError('Yaşam döngüsü kanıtı train-only veya independent-holdout otoritesi taşımalıdır.')
  }
  return {
    evidenceId: nonEmpty(evidence.evidenceId, 'Kanıt kimliği'),
    validationAuthority: evidence.validationAuthority,
    observationCount: evidence.observationCount,
    sourceGroupIds,
    confidenceMass,
    cohesion: bounded(evidence.cohesion, 0, 1, 'Kanıt iç uyumu'),
    holdoutGain: bounded(evidence.holdoutGain, -1, 1, 'Holdout kazancı'),
    shamGain: bounded(evidence.shamGain, -1, 1, 'Sham kazancı'),
    codes,
  }
}

export function validateRegionLifecyclePolicy(policy: RegionLifecyclePolicyV1): void {
  if (!Number.isInteger(policy.minimumObservationCount) || policy.minimumObservationCount < 1 ||
      !Number.isInteger(policy.minimumIndependentSources) || policy.minimumIndependentSources < 1) {
    throw new RangeError('Yaşam döngüsü gözlem ve kaynak eşikleri pozitif tam sayı olmalıdır.')
  }
  bounded(policy.minimumConfidenceMass, 0, Number.MAX_SAFE_INTEGER, 'Asgari güven kütlesi')
  bounded(policy.minimumCohesion, 0, 1, 'Asgari iç uyum')
  bounded(policy.minimumConstructiveHoldoutGain, -1, 1, 'Asgari yapıcı holdout kazancı')
  bounded(policy.maximumShamGain, 0, 1, 'Azami sham kazancı')
  bounded(policy.maximumNonRegressionLoss, 0, 1, 'Azami regresyon')
}

export function canonicalRegionLifecyclePolicy(
  policy: RegionLifecyclePolicyV1 = DEFAULT_REGION_LIFECYCLE_POLICY,
): RegionLifecyclePolicyV1 {
  validateRegionLifecyclePolicy(policy)
  return {
    minimumObservationCount: policy.minimumObservationCount,
    minimumIndependentSources: policy.minimumIndependentSources,
    minimumConfidenceMass: canonicalRegionFloat(policy.minimumConfidenceMass),
    minimumCohesion: canonicalRegionFloat(policy.minimumCohesion),
    minimumConstructiveHoldoutGain: canonicalRegionFloat(policy.minimumConstructiveHoldoutGain),
    maximumShamGain: canonicalRegionFloat(policy.maximumShamGain),
    maximumNonRegressionLoss: canonicalRegionFloat(policy.maximumNonRegressionLoss),
  }
}

export function evaluateRegionLifecycleEvidence(
  action: RegionLifecycleAction,
  rawEvidence: RegionLifecycleEvidenceV1,
  policy: RegionLifecyclePolicyV1 = DEFAULT_REGION_LIFECYCLE_POLICY,
): readonly RegionProposalRejection[] {
  policy = canonicalRegionLifecyclePolicy(policy)
  const evidence = canonicalRegionLifecycleEvidence(rawEvidence)
  const reasons: RegionProposalRejection[] = []
  if (evidence.validationAuthority !== 'independent-holdout') reasons.push('MISSING_INDEPENDENT_EVALUATION')
  if (evidence.observationCount < policy.minimumObservationCount) reasons.push('INSUFFICIENT_OBSERVATIONS')
  if (evidence.sourceGroupIds.length < policy.minimumIndependentSources) reasons.push('INSUFFICIENT_INDEPENDENT_SOURCES')
  if (evidence.confidenceMass < policy.minimumConfidenceMass) reasons.push('INSUFFICIENT_CONFIDENCE_MASS')
  if (evidence.cohesion < policy.minimumCohesion) reasons.push('INSUFFICIENT_COHESION')
  if (Math.abs(evidence.shamGain) > policy.maximumShamGain) reasons.push('SHAM_EFFECT_TOO_HIGH')
  if (action === 'birth' || action === 'grow' || action === 'split') {
    if (evidence.holdoutGain < policy.minimumConstructiveHoldoutGain) reasons.push('INSUFFICIENT_HOLDOUT_GAIN')
  } else if (evidence.holdoutGain < -policy.maximumNonRegressionLoss) {
    reasons.push('HOLDOUT_REGRESSION_TOO_HIGH')
  }
  return immutableRegionValue(reasons) as readonly RegionProposalRejection[]
}

function regionEvidenceFromLifecycle(evidence: RegionLifecycleEvidenceV1): PlasticRegionEvidenceV1 {
  return {
    observationCount: evidence.observationCount,
    sourceGroupIds: [...evidence.sourceGroupIds],
    confidenceMass: evidence.confidenceMass,
    cohesion: evidence.cohesion,
    holdoutUtility: evidence.holdoutGain,
    shamUtility: evidence.shamGain,
  }
}

function mutablePages(atlas: PlasticRegionAtlasV1): MutablePage[] {
  return atlas.pages.map((page) => ({
    pageIndex: page.pageIndex,
    memberships: page.memberships.map((membership) => ({ ...membership })),
    freeMass: page.freeMass,
  }))
}

function setMembership(page: MutablePage, regionId: string, weight: number): void {
  const index = page.memberships.findIndex((membership) => membership.regionId === regionId)
  if (weight <= REGION_MASS_EPSILON) {
    if (index >= 0) page.memberships.splice(index, 1)
    return
  }
  if (index >= 0) page.memberships[index] = { regionId, weight }
  else page.memberships.push({ regionId, weight })
}

function canonicalAllocationChanges(
  values: readonly RegionAllocationChangeV1[],
  label: string,
): RegionAllocationChangeV1[] {
  if (!values.length) throw new RangeError(`${label} en az bir mikro-sayfa değişikliği ister.`)
  const pages = new Set<number>()
  const result = values.map((value) => {
    if (!Number.isInteger(value.pageIndex) || value.pageIndex < 0 || value.pageIndex >= ADAPTIVE_PAGE_COUNT) {
      throw new RangeError(`${label} mikro-sayfa indeksi geometri dışında.`)
    }
    if (pages.has(value.pageIndex)) throw new TypeError(`${label} aynı mikro-sayfayı iki kez içeremez.`)
    pages.add(value.pageIndex)
    return { pageIndex: value.pageIndex, weight: bounded(value.weight, REGION_MASS_EPSILON, 1, `${label} ağırlığı`) }
  })
  return result.sort((left, right) => left.pageIndex - right.pageIndex)
}

function findLiveRegion(atlas: PlasticRegionAtlasV1, regionId: string): PlasticRegionV1 {
  const id = nonEmpty(regionId, 'Bölge kimliği')
  const region = atlas.regions.find((candidate) => candidate.id === id)
  if (!region) throw new TypeError(`Bilinmeyen plastik bölge: ${id}`)
  if (region.lifecycle === 'retired' || region.lifecycle === 'retiring') {
    throw new TypeError(`Canlı olmayan plastik bölge değiştirilemez: ${id}`)
  }
  return region
}

function assertNewRegionId(atlas: PlasticRegionAtlasV1, rawId: string): string {
  const id = nonEmpty(rawId, 'Yeni bölge kimliği')
  if (atlas.regions.some((region) => region.id === id)) throw new TypeError(`Plastik bölge kimliği zaten var: ${id}`)
  return id
}

export function deriveLatentRegionId(
  atlas: PlasticRegionAtlasV1,
  evidenceId: string,
  ordinal = 0,
): string {
  if (!Number.isInteger(ordinal) || ordinal < 0) throw new RangeError('Bölge sıra numarası negatif olmayan tam sayı olmalıdır.')
  return adaptiveRegionDigest('PREGION', atlas.fingerprint, nonEmpty(evidenceId, 'Kanıt kimliği'), ordinal)
}

function nextAtlas(
  atlas: PlasticRegionAtlasV1,
  regions: readonly PlasticRegionV1[],
  pages: readonly PlasticPageAssignmentV1[],
): PlasticRegionAtlasV1 {
  return createPlasticRegionAtlas({
    representationSpaceId: atlas.representationSpaceId,
    revision: atlas.revision + 1,
    parentFingerprint: atlas.fingerprint,
    behaviorDimension: atlas.behaviorDimension,
    maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
    regions,
    pages,
  })
}

function changedPages(
  before: PlasticRegionAtlasV1,
  after: PlasticRegionAtlasV1,
): RegionPageDeltaV1[] {
  const deltas: RegionPageDeltaV1[] = []
  for (let pageIndex = 0; pageIndex < ADAPTIVE_PAGE_COUNT; pageIndex += 1) {
    if (canonicalRegionJson(before.pages[pageIndex]) !== canonicalRegionJson(after.pages[pageIndex])) {
      deltas.push({ pageIndex, before: before.pages[pageIndex], after: after.pages[pageIndex] })
    }
  }
  return deltas
}

function createProposal(input: {
  atlas: PlasticRegionAtlasV1
  proposedAtlas: PlasticRegionAtlasV1
  action: RegionLifecycleAction
  targetRegionIds: readonly string[]
  proposedRegionIds: readonly string[]
  evidence: RegionLifecycleEvidenceV1
  policy?: RegionLifecyclePolicyV1
}): RegionLifecycleProposalV1 {
  const evidence = canonicalRegionLifecycleEvidence(input.evidence)
  const policy = canonicalRegionLifecyclePolicy(input.policy)
  const targetRegionIds = canonicalIds(input.targetRegionIds, 'Hedef bölge kimliği')
  const proposedRegionIds = canonicalIds(input.proposedRegionIds, 'Önerilen bölge kimliği')
  const rejectionReasons = [
    ...evaluateRegionLifecycleEvidence(input.action, evidence, policy),
    ...evaluateProposalCapacity(input.action, input.proposedAtlas, proposedRegionIds),
  ]
  const pageDeltas = changedPages(input.atlas, input.proposedAtlas)
  if (!pageDeltas.length) throw new TypeError('Yaşam döngüsü önerisi en az bir mikro-sayfayı değiştirmelidir.')
  const proposalId = adaptiveRegionDigest(
    'PRPROPOSALID',
    input.action,
    input.atlas.fingerprint,
    input.proposedAtlas.fingerprint,
    targetRegionIds,
    proposedRegionIds,
    evidence,
  )
  const payload = {
    format: ADAPTIVE_REGION_PROPOSAL_FORMAT,
    version: 1 as const,
    action: input.action,
    proposalId,
    baseAtlasFingerprint: input.atlas.fingerprint,
    targetRegionIds,
    proposedRegionIds,
    evidence,
    policy,
    eligible: rejectionReasons.length === 0,
    rejectionReasons,
    pageDeltas,
    proposedAtlas: input.proposedAtlas,
  }
  return immutableRegionValue({
    ...payload,
    fingerprint: adaptiveRegionDigest('PRPROPOSAL', {
      ...payload,
      proposedAtlas: input.proposedAtlas.fingerprint,
    }),
  }) as RegionLifecycleProposalV1
}

function evaluateProposalCapacity(
  action: RegionLifecycleAction,
  proposedAtlas: PlasticRegionAtlasV1,
  proposedRegionIds: readonly string[],
): RegionProposalRejection[] {
  if (!['birth', 'grow', 'split', 'merge'].includes(action)) return []
  return proposedRegionIds.some((regionId) =>
    plasticRegionCapacity(proposedAtlas, regionId) + REGION_MASS_EPSILON < MINIMUM_ACTIVE_REGION_CAPACITY)
    ? ['INSUFFICIENT_REGION_CAPACITY']
    : []
}

/**
 * Canonically verifies a persisted proposal without trusting its fingerprint,
 * eligibility flag, rejection list, embedded atlas, or page deltas.
 */
export function validateRegionLifecycleProposal(proposal: RegionLifecycleProposalV1): void {
  if (!proposal || typeof proposal !== 'object' ||
      proposal.format !== ADAPTIVE_REGION_PROPOSAL_FORMAT || proposal.version !== 1 ||
      !LIFECYCLE_ACTIONS.has(proposal.action)) {
    throw new TypeError('Bilinmeyen plastik bölge yaşam döngüsü önerisi biçimi.')
  }
  validateAdaptiveRegionAtlas(proposal.proposedAtlas)
  const evidence = canonicalRegionLifecycleEvidence(proposal.evidence)
  const policy = canonicalRegionLifecyclePolicy(proposal.policy)
  const targetRegionIds = canonicalIds(proposal.targetRegionIds, 'Hedef bölge kimliği')
  const proposedRegionIds = canonicalIds(proposal.proposedRegionIds, 'Önerilen bölge kimliği')
  const rejectionReasons = [
    ...evaluateRegionLifecycleEvidence(proposal.action, evidence, policy),
    ...evaluateProposalCapacity(proposal.action, proposal.proposedAtlas, proposedRegionIds),
  ]
  if (proposal.eligible !== (rejectionReasons.length === 0) ||
      canonicalRegionJson(proposal.rejectionReasons) !== canonicalRegionJson(rejectionReasons)) {
    throw new TypeError('Plastik bölge önerisinin eligibility/rejection kanıtı içerikle uyuşmuyor.')
  }
  if (!proposal.baseAtlasFingerprint?.trim() ||
      proposal.proposedAtlas.parentFingerprint !== proposal.baseAtlasFingerprint) {
    throw new TypeError('Plastik bölge önerisinin base/proposed atlas zinciri uyuşmuyor.')
  }
  if (!Array.isArray(proposal.pageDeltas) || proposal.pageDeltas.length < 1) {
    throw new TypeError('Plastik bölge önerisi değişen mikro-sayfa kanıtı taşımıyor.')
  }
  const pageIndexes = new Set<number>()
  const pageDeltas = proposal.pageDeltas.map((delta) => {
    const pageIndex = delta?.pageIndex
    if (!Number.isInteger(pageIndex) || pageIndex < 0 || pageIndex >= ADAPTIVE_PAGE_COUNT ||
        pageIndexes.has(pageIndex) || delta.before?.pageIndex !== pageIndex || delta.after?.pageIndex !== pageIndex) {
      throw new TypeError('Plastik bölge önerisinin mikro-sayfa deltaları eksik, yinelenen veya sırasız.')
    }
    pageIndexes.add(pageIndex)
    if (canonicalRegionJson(delta.after) !== canonicalRegionJson(proposal.proposedAtlas.pages[pageIndex]) ||
        canonicalRegionJson(delta.before) === canonicalRegionJson(delta.after)) {
      throw new TypeError('Plastik bölge önerisinin mikro-sayfa deltası proposed atlasla uyuşmuyor.')
    }
    const totalBefore = Number(delta.before.freeMass) + delta.before.memberships.reduce(
      (sum: number, membership: PlasticRegionMembershipV1) => sum + Number(membership.weight), 0,
    )
    if (!Number.isFinite(totalBefore) || Math.abs(totalBefore - 1) > REGION_MASS_EPSILON) {
      throw new TypeError('Plastik bölge önerisinin önceki mikro-sayfa kütlesi korunmuyor.')
    }
    return delta
  })
  if (pageDeltas.some((delta, index) => index > 0 && pageDeltas[index - 1].pageIndex >= delta.pageIndex)) {
    throw new TypeError('Plastik bölge önerisinin mikro-sayfa deltaları kanonik sırada değil.')
  }
  const proposalId = adaptiveRegionDigest(
    'PRPROPOSALID',
    proposal.action,
    proposal.baseAtlasFingerprint,
    proposal.proposedAtlas.fingerprint,
    targetRegionIds,
    proposedRegionIds,
    evidence,
  )
  if (proposal.proposalId !== proposalId) {
    throw new TypeError('Plastik bölge proposalId özeti içerikle uyuşmuyor.')
  }
  const payload = {
    format: ADAPTIVE_REGION_PROPOSAL_FORMAT,
    version: 1 as const,
    action: proposal.action,
    proposalId,
    baseAtlasFingerprint: proposal.baseAtlasFingerprint,
    targetRegionIds,
    proposedRegionIds,
    evidence,
    policy,
    eligible: rejectionReasons.length === 0,
    rejectionReasons,
    pageDeltas,
    proposedAtlas: proposal.proposedAtlas,
  }
  const { fingerprint: _fingerprint, ...actualPayload } = proposal
  if (canonicalRegionJson(actualPayload) !== canonicalRegionJson(payload)) {
    throw new TypeError('Plastik bölge önerisi kanonik içerikle uyuşmuyor.')
  }
  const fingerprint = adaptiveRegionDigest('PRPROPOSAL', {
    ...payload,
    proposedAtlas: proposal.proposedAtlas.fingerprint,
  })
  if (proposal.fingerprint !== fingerprint) {
    throw new TypeError('Plastik bölge önerisi fingerprinti içerikle uyuşmuyor.')
  }
}

export interface ProposeRegionBirthInput {
  readonly regionId: string
  readonly allocations: readonly RegionAllocationChangeV1[]
  readonly prototype: PlasticRegionPrototypeV1
  readonly evidence: RegionLifecycleEvidenceV1
  readonly policy?: RegionLifecyclePolicyV1
}

export function proposeRegionBirth(
  atlas: PlasticRegionAtlasV1,
  input: ProposeRegionBirthInput,
): RegionLifecycleProposalV1 {
  validateAdaptiveRegionAtlas(atlas)
  const regionId = assertNewRegionId(atlas, input.regionId)
  const evidence = canonicalRegionLifecycleEvidence(input.evidence)
  const policy = canonicalRegionLifecyclePolicy(input.policy)
  const allocatedCapacity = input.allocations.reduce((sum, allocation) => sum + allocation.weight, 0)
  const lifecycle: PlasticRegionV1['lifecycle'] = evaluateRegionLifecycleEvidence('birth', evidence, policy).length ||
    allocatedCapacity + REGION_MASS_EPSILON < MINIMUM_ACTIVE_REGION_CAPACITY
    ? 'provisional'
    : 'active'
  const pages = mutablePages(atlas)
  for (const allocation of canonicalAllocationChanges(input.allocations, 'Bölge doğumu')) {
    const page = pages[allocation.pageIndex]
    if (page.freeMass + REGION_MASS_EPSILON < allocation.weight) {
      throw new RangeError(`${allocation.pageIndex}. mikro-sayfada doğum için yeterli serbest kütle yok.`)
    }
    page.freeMass -= allocation.weight
    setMembership(page, regionId, allocation.weight)
  }
  const region: PlasticRegionV1 = {
    id: regionId,
    lifecycle,
    origin: 'discovery',
    bornRevision: atlas.revision + 1,
    retiredRevision: null,
    parentRegionIds: [],
    prototype: input.prototype,
    evidence: regionEvidenceFromLifecycle(evidence),
  }
  const proposedAtlas = nextAtlas(atlas, [...atlas.regions, region], pages)
  return createProposal({
    atlas,
    proposedAtlas,
    action: 'birth',
    targetRegionIds: [],
    proposedRegionIds: [regionId],
    evidence,
    policy,
  })
}

export interface ProposeRegionResizeInput {
  readonly regionId: string
  readonly allocations: readonly RegionAllocationChangeV1[]
  readonly evidence: RegionLifecycleEvidenceV1
  readonly policy?: RegionLifecyclePolicyV1
}

export function proposeRegionGrow(
  atlas: PlasticRegionAtlasV1,
  input: ProposeRegionResizeInput,
): RegionLifecycleProposalV1 {
  validateAdaptiveRegionAtlas(atlas)
  const region = findLiveRegion(atlas, input.regionId)
  const pages = mutablePages(atlas)
  for (const allocation of canonicalAllocationChanges(input.allocations, 'Bölge büyümesi')) {
    const page = pages[allocation.pageIndex]
    if (page.freeMass + REGION_MASS_EPSILON < allocation.weight) {
      throw new RangeError(`${allocation.pageIndex}. mikro-sayfada büyüme için yeterli serbest kütle yok.`)
    }
    const previous = page.memberships.find((membership) => membership.regionId === region.id)?.weight ?? 0
    page.freeMass -= allocation.weight
    setMembership(page, region.id, previous + allocation.weight)
  }
  const evidence = canonicalRegionLifecycleEvidence(input.evidence)
  const policy = canonicalRegionLifecyclePolicy(input.policy)
  const nextCapacity = pages.reduce(
    (sum, page) => sum + (page.memberships.find((membership) => membership.regionId === region.id)?.weight ?? 0),
    0,
  )
  const shouldActivate = region.lifecycle === 'provisional' &&
    evaluateRegionLifecycleEvidence('grow', evidence, policy).length === 0 &&
    nextCapacity + REGION_MASS_EPSILON >= MINIMUM_ACTIVE_REGION_CAPACITY
  const regions = shouldActivate
    ? atlas.regions.map((candidate): PlasticRegionV1 => candidate.id === region.id
      ? { ...candidate, lifecycle: 'active' }
      : candidate)
    : atlas.regions
  const proposedAtlas = nextAtlas(atlas, regions, pages)
  return createProposal({
    atlas,
    proposedAtlas,
    action: 'grow',
    targetRegionIds: [region.id],
    proposedRegionIds: [region.id],
    evidence,
    policy,
  })
}

export function proposeRegionShrink(
  atlas: PlasticRegionAtlasV1,
  input: ProposeRegionResizeInput,
): RegionLifecycleProposalV1 {
  validateAdaptiveRegionAtlas(atlas)
  const region = findLiveRegion(atlas, input.regionId)
  const pages = mutablePages(atlas)
  let released = 0
  for (const allocation of canonicalAllocationChanges(input.allocations, 'Bölge küçülmesi')) {
    const page = pages[allocation.pageIndex]
    const previous = page.memberships.find((membership) => membership.regionId === region.id)?.weight ?? 0
    if (previous + REGION_MASS_EPSILON < allocation.weight) {
      throw new RangeError(`${allocation.pageIndex}. mikro-sayfada bölgenin bırakabileceği kadar aidiyet yok.`)
    }
    setMembership(page, region.id, previous - allocation.weight)
    page.freeMass += allocation.weight
    released += allocation.weight
  }
  if (plasticRegionCapacity(atlas, region.id) - released <= REGION_MASS_EPSILON) {
    throw new RangeError('Küçülme bölgenin tüm kapasitesini bırakamaz; emeklilik önerisi kullanılmalıdır.')
  }
  const proposedAtlas = nextAtlas(atlas, atlas.regions, pages)
  return createProposal({
    atlas,
    proposedAtlas,
    action: 'shrink',
    targetRegionIds: [region.id],
    proposedRegionIds: [region.id],
    evidence: input.evidence,
    policy: input.policy,
  })
}

export interface ProposeRegionSplitInput {
  readonly sourceRegionId: string
  readonly firstChildId: string
  readonly secondChildId: string
  readonly firstChildPrototype: PlasticRegionPrototypeV1
  readonly secondChildPrototype: PlasticRegionPrototypeV1
  readonly fractions: readonly RegionSplitFractionV1[]
  readonly evidence: RegionLifecycleEvidenceV1
  readonly policy?: RegionLifecyclePolicyV1
}

export function proposeRegionSplit(
  atlas: PlasticRegionAtlasV1,
  input: ProposeRegionSplitInput,
): RegionLifecycleProposalV1 {
  validateAdaptiveRegionAtlas(atlas)
  const source = findLiveRegion(atlas, input.sourceRegionId)
  const firstChildId = assertNewRegionId(atlas, input.firstChildId)
  const secondChildId = assertNewRegionId(atlas, input.secondChildId)
  if (firstChildId === secondChildId) throw new TypeError('Bölünme çocukları farklı kimlikler taşımalıdır.')
  const fractionMap = new Map<number, number>()
  for (const fraction of input.fractions) {
    if (!Number.isInteger(fraction.pageIndex) || fraction.pageIndex < 0 || fraction.pageIndex >= ADAPTIVE_PAGE_COUNT ||
        fractionMap.has(fraction.pageIndex)) {
      throw new RangeError('Bölünme kesirleri benzersiz ve geçerli mikro-sayfalar taşımalıdır.')
    }
    fractionMap.set(fraction.pageIndex, bounded(fraction.firstChildFraction, 0, 1, 'İlk çocuk payı'))
  }
  const sourcePages = atlas.pages.filter((page) =>
    page.memberships.some((membership) => membership.regionId === source.id))
  if (fractionMap.size !== sourcePages.length || sourcePages.some((page) => !fractionMap.has(page.pageIndex))) {
    throw new TypeError('Bölünme, kaynak bölgenin sahip olduğu her mikro-sayfa için açık bir kesir ister.')
  }
  const pages = mutablePages(atlas)
  let firstCapacity = 0
  let secondCapacity = 0
  for (const sourcePage of sourcePages) {
    const page = pages[sourcePage.pageIndex]
    const sourceWeight = page.memberships.find((membership) => membership.regionId === source.id)!.weight
    const firstWeight = sourceWeight * fractionMap.get(page.pageIndex)!
    const secondWeight = sourceWeight - firstWeight
    setMembership(page, source.id, 0)
    setMembership(page, firstChildId, firstWeight)
    setMembership(page, secondChildId, secondWeight)
    firstCapacity += firstWeight
    secondCapacity += secondWeight
  }
  if (firstCapacity <= REGION_MASS_EPSILON || secondCapacity <= REGION_MASS_EPSILON) {
    throw new RangeError('Bölünmenin iki çocuğu da pozitif kapasite almalıdır.')
  }
  const revision = atlas.revision + 1
  const retiredSource: PlasticRegionV1 = { ...source, lifecycle: 'retired', retiredRevision: revision }
  const lifecycleEvidence = canonicalRegionLifecycleEvidence(input.evidence)
  const policy = canonicalRegionLifecyclePolicy(input.policy)
  const childLifecycle: PlasticRegionV1['lifecycle'] = evaluateRegionLifecycleEvidence(
    'split', lifecycleEvidence, policy,
  ).length || firstCapacity + REGION_MASS_EPSILON < MINIMUM_ACTIVE_REGION_CAPACITY ||
    secondCapacity + REGION_MASS_EPSILON < MINIMUM_ACTIVE_REGION_CAPACITY
    ? 'provisional'
    : 'active'
  const regionEvidence = regionEvidenceFromLifecycle(lifecycleEvidence)
  const children: PlasticRegionV1[] = [
    {
      id: firstChildId,
      lifecycle: childLifecycle,
      origin: 'split',
      bornRevision: revision,
      retiredRevision: null,
      parentRegionIds: [source.id],
      prototype: input.firstChildPrototype,
      evidence: regionEvidence,
    },
    {
      id: secondChildId,
      lifecycle: childLifecycle,
      origin: 'split',
      bornRevision: revision,
      retiredRevision: null,
      parentRegionIds: [source.id],
      prototype: input.secondChildPrototype,
      evidence: regionEvidence,
    },
  ]
  const regions = atlas.regions.map((region) => region.id === source.id ? retiredSource : region).concat(children)
  const proposedAtlas = nextAtlas(atlas, regions, pages)
  return createProposal({
    atlas,
    proposedAtlas,
    action: 'split',
    targetRegionIds: [source.id],
    proposedRegionIds: [firstChildId, secondChildId],
    evidence: input.evidence,
    policy,
  })
}

export interface ProposeRegionMergeInput {
  readonly sourceRegionIds: readonly string[]
  readonly mergedRegionId: string
  readonly prototype: PlasticRegionPrototypeV1
  readonly evidence: RegionLifecycleEvidenceV1
  readonly policy?: RegionLifecyclePolicyV1
}

export function proposeRegionMerge(
  atlas: PlasticRegionAtlasV1,
  input: ProposeRegionMergeInput,
): RegionLifecycleProposalV1 {
  validateAdaptiveRegionAtlas(atlas)
  const sourceIds = canonicalIds(input.sourceRegionIds, 'Birleşme kaynak bölgesi')
  if (sourceIds.length < 2) throw new RangeError('Bölge birleşmesi en az iki kaynak ister.')
  const sources = sourceIds.map((id) => findLiveRegion(atlas, id))
  const mergedRegionId = assertNewRegionId(atlas, input.mergedRegionId)
  const pages = mutablePages(atlas)
  for (const page of pages) {
    let mergedWeight = 0
    for (const source of sources) {
      mergedWeight += page.memberships.find((membership) => membership.regionId === source.id)?.weight ?? 0
      setMembership(page, source.id, 0)
    }
    setMembership(page, mergedRegionId, mergedWeight)
  }
  const revision = atlas.revision + 1
  const lifecycleEvidence = canonicalRegionLifecycleEvidence(input.evidence)
  const policy = canonicalRegionLifecyclePolicy(input.policy)
  const mergedLifecycle: PlasticRegionV1['lifecycle'] = evaluateRegionLifecycleEvidence(
    'merge', lifecycleEvidence, policy,
  ).length ? 'provisional' : 'active'
  const sourceSet = new Set(sourceIds)
  const regions = atlas.regions.map((region): PlasticRegionV1 =>
    sourceSet.has(region.id) ? { ...region, lifecycle: 'retired', retiredRevision: revision } : region)
  regions.push({
    id: mergedRegionId,
    lifecycle: mergedLifecycle,
    origin: 'merge',
    bornRevision: revision,
    retiredRevision: null,
    parentRegionIds: sourceIds,
    prototype: input.prototype,
    evidence: regionEvidenceFromLifecycle(lifecycleEvidence),
  })
  const proposedAtlas = nextAtlas(atlas, regions, pages)
  return createProposal({
    atlas,
    proposedAtlas,
    action: 'merge',
    targetRegionIds: sourceIds,
    proposedRegionIds: [mergedRegionId],
    evidence: input.evidence,
    policy,
  })
}

export interface ProposeRegionRetireInput {
  readonly regionId: string
  readonly evidence: RegionLifecycleEvidenceV1
  readonly policy?: RegionLifecyclePolicyV1
}

export function proposeRegionRetire(
  atlas: PlasticRegionAtlasV1,
  input: ProposeRegionRetireInput,
): RegionLifecycleProposalV1 {
  validateAdaptiveRegionAtlas(atlas)
  const source = findLiveRegion(atlas, input.regionId)
  const pages = mutablePages(atlas)
  for (const page of pages) {
    const membership = page.memberships.find((candidate) => candidate.regionId === source.id)
    if (!membership) continue
    page.freeMass += membership.weight
    setMembership(page, source.id, 0)
  }
  const revision = atlas.revision + 1
  const regions = atlas.regions.map((region): PlasticRegionV1 =>
    region.id === source.id ? { ...region, lifecycle: 'retired', retiredRevision: revision } : region)
  const proposedAtlas = nextAtlas(atlas, regions, pages)
  return createProposal({
    atlas,
    proposedAtlas,
    action: 'retire',
    targetRegionIds: [source.id],
    proposedRegionIds: [],
    evidence: input.evidence,
    policy: input.policy,
  })
}

export function applyRegionLifecycleProposal(
  atlas: PlasticRegionAtlasV1,
  proposal: RegionLifecycleProposalV1,
): PlasticRegionAtlasV1 {
  validateAdaptiveRegionAtlas(atlas)
  if (proposal.format !== ADAPTIVE_REGION_PROPOSAL_FORMAT || proposal.version !== 1) {
    throw new TypeError('Bilinmeyen plastik bölge yaşam döngüsü önerisi biçimi.')
  }
  if (proposal.baseAtlasFingerprint !== atlas.fingerprint ||
      proposal.proposedAtlas.parentFingerprint !== atlas.fingerprint) {
    throw new TypeError('Yaşam döngüsü önerisi stale veya farklı bir atlas tabanına ait.')
  }
  validateAdaptiveRegionAtlas(proposal.proposedAtlas)
  if (proposal.proposedAtlas.revision !== atlas.revision + 1) {
    throw new TypeError('Yaşam döngüsü önerisi atlas revisionını tam bir adım ilerletmelidir.')
  }
  const expectedPageDeltas = changedPages(atlas, proposal.proposedAtlas)
  if (canonicalRegionJson(expectedPageDeltas) !== canonicalRegionJson(proposal.pageDeltas)) {
    throw new TypeError('Yaşam döngüsü önerisinin mikro-sayfa deltaları içerikle uyuşmuyor.')
  }
  const expectedProposalId = adaptiveRegionDigest(
    'PRPROPOSALID',
    proposal.action,
    atlas.fingerprint,
    proposal.proposedAtlas.fingerprint,
    proposal.targetRegionIds,
    proposal.proposedRegionIds,
    proposal.evidence,
  )
  if (expectedProposalId !== proposal.proposalId) {
    throw new TypeError('Yaşam döngüsü öneri kimliği içerikle uyuşmuyor.')
  }
  const expectedRejections = [
    ...evaluateRegionLifecycleEvidence(proposal.action, proposal.evidence, proposal.policy),
    ...evaluateProposalCapacity(proposal.action, proposal.proposedAtlas, proposal.proposedRegionIds),
  ]
  if (canonicalRegionJson(expectedRejections) !== canonicalRegionJson(proposal.rejectionReasons) ||
      proposal.eligible !== (expectedRejections.length === 0)) {
    throw new TypeError('Yaşam döngüsü önerisinin uygunluk kararı yeniden üretilemiyor.')
  }
  const expectedFingerprint = adaptiveRegionDigest('PRPROPOSAL', {
    format: proposal.format,
    version: proposal.version,
    action: proposal.action,
    proposalId: proposal.proposalId,
    baseAtlasFingerprint: proposal.baseAtlasFingerprint,
    targetRegionIds: proposal.targetRegionIds,
    proposedRegionIds: proposal.proposedRegionIds,
    evidence: proposal.evidence,
    policy: proposal.policy,
    eligible: proposal.eligible,
    rejectionReasons: proposal.rejectionReasons,
    pageDeltas: proposal.pageDeltas,
    proposedAtlas: proposal.proposedAtlas.fingerprint,
  })
  if (expectedFingerprint !== proposal.fingerprint) throw new TypeError('Yaşam döngüsü önerisi özeti uyuşmuyor.')
  if (!proposal.eligible) {
    throw new TypeError(`Uygun olmayan yaşam döngüsü önerisi uygulanamaz: ${proposal.rejectionReasons.join(', ')}`)
  }
  return proposal.proposedAtlas
}
