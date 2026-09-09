import { getHeapStatistics } from 'node:v8'
import {
  createInitialPlasticRegionAtlas,
  createPlasticRegionAtlas,
  proposeRegionBirth,
  proposeRegionGrow,
  proposeRegionMerge,
  proposeRegionRetire,
  proposeRegionShrink,
  proposeRegionSplit,
  validateAdaptiveRegionAtlas,
  validateRegionLifecycleProposal,
} from '../src/adaptive-regions/index.ts'
import { resolveThoughtEngineBase } from '../server/thought-service.mjs'
import { evaluateResourceState, MEBIBYTE, sampleHostResources } from '../server/resource-policy.mjs'
import { persistArtifact, restoreArtifact, stableRuntimeHash } from './artifacts.mjs'
import {
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
  assertPlasticRegionArtifactMetadata,
  currentPlasticRegionBase,
  persistPlasticRegionArtifact,
  plasticRegionSnapshotActivationGate,
  readPlasticRegionRelease,
  readPlasticRegionSnapshot,
} from './plastic-region-cache.mjs'
import {
  PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
  createPlasticRegionReleaseBenchmarkReport,
  createPlasticRegionReleaseEvidence,
  createPlasticRegionReleaseManifest,
  derivePlasticRegionReleaseMetrics,
  plasticRegionReleasePolicy,
  validatePlasticRegionReleaseAudit,
} from './plastic-region-release.mjs'
import {
  PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT,
  PLASTIC_REGION_RELEASE_DEFAULT_MAXIMUM_TICKS,
  PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
  PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT,
  PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_RUNNER_VERSION,
  PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SEEDS,
  PLASTIC_REGION_RELEASE_SOURCE_POLICY,
  plasticRegionReleaseCases,
  plasticRegionReleaseCorpusFingerprint,
  scorePlasticRegionReleaseGraph,
} from './plastic-region-release-spec.mjs'

export const PLASTIC_REGION_RELEASE_JOB_TYPE = 'plastic-region-release-evaluation'
export { PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND }
export { PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT, PLASTIC_REGION_RELEASE_RUNNER_VERSION }

const RELEASE_SEEDS = PLASTIC_REGION_RELEASE_SEEDS
const DEFAULT_CASE_COUNT = PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT
const MAXIMUM_CASE_COUNT = PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT
const DEFAULT_MAXIMUM_TICKS = PLASTIC_REGION_RELEASE_DEFAULT_MAXIMUM_TICKS
const MAXIMUM_DENSE_NEURONS = 16_384
const MAXIMUM_DENSE_VECTOR_BYTES = 256 * MEBIBYTE
const BASE_ARTIFACT_KINDS = new Set([
  'family-run',
  'runtime-model-state',
  'text-lab-analysis',
  'synthesis-run',
  'thought-runtime-head',
])
const JOB_PAYLOAD_KEYS = Object.freeze([
  'base',
  'proposalArtifactId',
  'proposalContentHash',
  'proposalFingerprint',
  'runnerFingerprint',
  'sourceSnapshotId',
  'sourceInputManifestHash',
  'sourceArtifactManifestHash',
  'sourceProducerJobId',
])

function releaseRunnerError(code, message, status = 409, details) {
  const error = new Error(message)
  error.code = code
  error.status = status
  if (details !== undefined) error.details = details
  return error
}

function canonicalJson(value) {
  if (ArrayBuffer.isView(value)) return canonicalJson(Array.from(value))
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function sameCanonical(left, right) {
  return canonicalJson(left) === canonicalJson(right)
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length)
}

function clampSigned(value) {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0))
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
}

function abortIfRequested(signal) {
  if (!signal?.aborted) return
  throw signal.reason ?? new DOMException('Aborted', 'AbortError')
}

function exactKeys(value, expected) {
  return value && typeof value === 'object' && !Array.isArray(value) &&
    sameCanonical(Object.keys(value).sort(), [...expected].sort())
}

function findProposalArtifact(store, reference) {
  const direct = store.getArtifact?.(reference)
  if (direct?.kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND) return direct
  const metadata = (store.listArtifactMetadata?.() ?? [])
    .filter((entry) => entry.kind === PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND)
    .slice(0, 4_096)
  const matches = []
  for (const entry of metadata) {
    const artifact = store.getArtifact(entry.id)
    try {
      const proposal = restoreArtifact(artifact, PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND)
      if (proposal?.proposalId === reference) matches.push(artifact)
    } catch {
      // A corrupt unrelated proposal cannot become the requested authority.
    }
  }
  if (matches.length > 1) {
    throw releaseRunnerError(
      'PLASTIC_REGION_PROPOSAL_AMBIGUOUS',
      'Aynı öneri kimliğine bağlı birden fazla artifact bulundu; immutable artifact kimliğiyle yeniden deneyin.',
    )
  }
  return matches[0] ?? null
}

function verifiedProposalRecord(store, reference) {
  const artifact = findProposalArtifact(store, String(reference ?? '').trim())
  if (!artifact) {
    throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_NOT_FOUND', 'Plastik bölge önerisi bulunamadı.', 404)
  }
  const metadata = store.getArtifactMetadata?.(artifact.id)
  if (!metadata || metadata.kind !== artifact.kind || metadata.contentHash !== artifact.contentHash) {
    throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Önerinin immutable artifact başlığı uyuşmuyor.')
  }
  let proposal
  try {
    proposal = restoreArtifact(artifact, PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND)
    validateRegionLifecycleProposal(proposal)
    assertPlasticRegionArtifactMetadata(PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND, proposal, artifact)
  } catch (cause) {
    throw releaseRunnerError(
      'PLASTIC_REGION_PROPOSAL_TAMPERED',
      'Öneri içeriği, fingerprinti veya yönlendirme metadata’sı doğrulanamadı.',
      409,
      { cause: String(cause?.message ?? cause) },
    )
  }
  const sourceSnapshots = (store.listSnapshots?.() ?? []).filter((snapshot) =>
    snapshot.status === 'candidate' &&
    snapshot.producerJobId === artifact.jobId &&
    (snapshot.artifactIds ?? []).includes(artifact.id))
  if (sourceSnapshots.length !== 1) {
    throw releaseRunnerError(
      'PLASTIC_REGION_PROPOSAL_PROVENANCE_INVALID',
      'Öneri tam bir aday snapshot ve üretici iş zincirine bağlanamıyor.',
      409,
      { candidateCount: sourceSnapshots.length },
    )
  }
  const sourceSnapshot = sourceSnapshots[0]
  const producer = store.getJob?.(sourceSnapshot.producerJobId)
  const certifiedIds = new Set([
    producer?.result?.snapshotId,
    producer?.result?.lastSnapshotId,
    ...(Array.isArray(producer?.result?.snapshotIds) ? producer.result.snapshotIds : []),
    ...(Array.isArray(producer?.result?.trainingSnapshotIds) ? producer.result.trainingSnapshotIds : []),
  ].filter(Boolean))
  if (producer?.state !== 'succeeded' || !certifiedIds.has(sourceSnapshot.id) ||
      sourceSnapshot.evaluation?.passed !== true || sourceSnapshot.evaluation?.runtimeVerified !== true) {
    throw releaseRunnerError(
      'PLASTIC_REGION_PROPOSAL_PROVENANCE_INVALID',
      'Öneriyi üreten adayın gerçek motor ve genel değerlendirme kapıları tamamlanmamış.',
      409,
      { producerState: producer?.state ?? null, sourceSnapshotId: sourceSnapshot.id },
    )
  }
  return { artifact, metadata, proposal, sourceSnapshot, producer }
}

function currentBaseAtlas(store, proposal, expectedBase) {
  const active = store.getActiveSnapshot?.()
  if (!active || active.id !== expectedBase.snapshotId) {
    throw releaseRunnerError(
      'PLASTIC_REGION_RELEASE_ACTIVE_BASE_REQUIRED',
      'Release değerlendirmesi kalıcı ve etkin bir base snapshot gerektirir.',
    )
  }
  const record = readPlasticRegionSnapshot(store, active)
  if (record.metadata && !record.compatible) {
    throw releaseRunnerError('PLASTIC_REGION_RELEASE_BASE_INVALID', 'Etkin plastik bölge atlası doğrulanamadı.', 409, {
      reason: record.reason,
    })
  }
  const atlas = record.head ?? createInitialPlasticRegionAtlas({
    representationSpaceId: proposal.proposedAtlas.representationSpaceId,
    behaviorDimension: proposal.proposedAtlas.behaviorDimension,
    maximumMembershipsPerPage: proposal.proposedAtlas.maximumMembershipsPerPage,
  })
  validateAdaptiveRegionAtlas(atlas)
  if (atlas.fingerprint !== proposal.baseAtlasFingerprint ||
      atlas.representationSpaceId !== proposal.proposedAtlas.representationSpaceId) {
    throw releaseRunnerError(
      'STALE_REGION_ATLAS_BASE',
      'Önerinin dayandığı atlas artık etkin atlasla aynı değil; öneri güncel tabanda yeniden üretilmeli.',
      409,
      { expected: proposal.baseAtlasFingerprint, actual: atlas.fingerprint },
    )
  }
  return atlas
}

function prepareRecord(store, proposalReference) {
  const record = verifiedProposalRecord(store, proposalReference)
  const base = currentPlasticRegionBase(store)
  if (!base.snapshotId) {
    throw releaseRunnerError(
      'PLASTIC_REGION_RELEASE_ACTIVE_BASE_REQUIRED',
      'İlk kalıcı plastik bölge release’i için önce doğrulanmış etkin bir snapshot gerekir.',
    )
  }
  if (record.sourceSnapshot.parentId !== base.snapshotId) {
    throw releaseRunnerError(
      'STALE_REGION_ATLAS_BASE',
      'Öneri güncel etkin snapshotın doğrudan adayı değil.',
      409,
      { expectedParentId: base.snapshotId, actualParentId: record.sourceSnapshot.parentId },
    )
  }
  currentBaseAtlas(store, record.proposal, base)
  return { ...record, base }
}

function sourceArtifactManifestHash(store, snapshot) {
  const manifest = (snapshot?.artifactIds ?? []).map((artifactId) => {
    const metadata = store.getArtifactMetadata?.(artifactId)
    if (!metadata) {
      throw releaseRunnerError(
        'PLASTIC_REGION_PROPOSAL_PROVENANCE_INVALID',
        'Kaynak adayın immutable artifact manifesti eksik.',
        409,
        { artifactId },
      )
    }
    return { id: metadata.id, kind: metadata.kind, contentHash: metadata.contentHash }
  }).sort((left, right) => left.id.localeCompare(right.id))
  return stableRuntimeHash('thought-base-artifacts-v1', manifest)
}

export function preparePlasticRegionReleaseJob(store, proposalReference) {
  const record = prepareRecord(store, proposalReference)
  const sourceInputManifestHash = String(record.sourceSnapshot.inputManifestHash ?? '').trim()
  if (!sourceInputManifestHash) {
    throw releaseRunnerError(
      'PLASTIC_REGION_PROPOSAL_PROVENANCE_INVALID',
      'Kaynak aday exact girdi manifesti taşımıyor.',
    )
  }
  const payload = Object.freeze({
    base: Object.freeze({ ...record.base }),
    proposalArtifactId: record.artifact.id,
    proposalContentHash: record.metadata.contentHash,
    proposalFingerprint: record.proposal.fingerprint,
    runnerFingerprint: PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
    sourceSnapshotId: record.sourceSnapshot.id,
    sourceInputManifestHash,
    sourceArtifactManifestHash: sourceArtifactManifestHash(store, record.sourceSnapshot),
    sourceProducerJobId: record.sourceSnapshot.producerJobId,
  })
  return Object.freeze({
    payload,
    requestHash: stableRuntimeHash('plastic-region-release-job-request-v1', payload),
    defaultIdempotencyKey: `plastic-region-release:${stableRuntimeHash(payload).slice(0, 48)}`,
    proposal: record.proposal,
    sourceSnapshot: record.sourceSnapshot,
  })
}

function validateQueuedRequest(job, store) {
  if (job?.type !== PLASTIC_REGION_RELEASE_JOB_TYPE) {
    throw releaseRunnerError('INVALID_PLASTIC_REGION_RELEASE_JOB', 'Release runner yanlış iş türüyle çağrıldı.', 400)
  }
  if (!exactKeys(job.payload, JOB_PAYLOAD_KEYS)) {
    throw releaseRunnerError(
      'INVALID_PLASTIC_REGION_RELEASE_JOB',
      'Release işi yalnız sunucunun dondurduğu öneri/base alanlarını taşıyabilir.',
      400,
    )
  }
  const prepared = preparePlasticRegionReleaseJob(store, job.payload.proposalArtifactId)
  if (!sameCanonical(job.payload, prepared.payload) || job.requestHash !== prepared.requestHash) {
    throw releaseRunnerError(
      'PLASTIC_REGION_RELEASE_REQUEST_TAMPERED',
      'Kuyruktaki release isteği güncel immutable öneri/base sözleşmesiyle uyuşmuyor.',
    )
  }
  return prepared
}

function pageRegionWeight(page, regionId) {
  return page.memberships.find((item) => item.regionId === regionId)?.weight ?? 0
}

function lifecycleEvidence(proposal, holdoutGain, shamGain) {
  return {
    ...proposal.evidence,
    validationAuthority: 'independent-holdout',
    holdoutGain: clampSigned(holdoutGain),
    shamGain: clampSigned(shamGain),
  }
}

function positiveDeltas(proposal, regionId, direction) {
  return proposal.pageDeltas.map((delta) => {
    const before = pageRegionWeight(delta.before, regionId)
    const after = pageRegionWeight(delta.after, regionId)
    const weight = direction === 'increase' ? after - before : before - after
    if (!(weight > 0)) {
      throw releaseRunnerError(
        'PLASTIC_REGION_PROPOSAL_TAMPERED',
        `${proposal.action} önerisi pozitif ve yeniden üretilebilir sayfa deltası taşımıyor.`,
      )
    }
    return { pageIndex: delta.pageIndex, weight }
  })
}

function lifecycleInput(baseAtlas, proposal, evidence, overrides = {}) {
  const target = proposal.targetRegionIds[0]
  const proposed = proposal.proposedRegionIds[0]
  if (proposal.action === 'birth') {
    const region = proposal.proposedAtlas.regions.find((item) => item.id === proposed)
    if (!region || proposal.proposedRegionIds.length !== 1 || proposal.targetRegionIds.length) {
      throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Birth önerisi tek bir yeni bölge taşımıyor.')
    }
    return {
      regionId: overrides.regionId ?? proposed,
      allocations: overrides.allocations ?? positiveDeltas(proposal, proposed, 'increase'),
      prototype: overrides.prototype ?? region.prototype,
      evidence,
      policy: proposal.policy,
    }
  }
  if (proposal.action === 'grow' || proposal.action === 'shrink') {
    if (!target || proposal.targetRegionIds.length !== 1 || proposal.proposedRegionIds.length !== 1 || proposed !== target) {
      throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', `${proposal.action} önerisinin hedef kimliği geçersiz.`)
    }
    return {
      regionId: overrides.regionId ?? target,
      allocations: overrides.allocations ?? positiveDeltas(
        proposal,
        target,
        proposal.action === 'grow' ? 'increase' : 'decrease',
      ),
      evidence,
      policy: proposal.policy,
    }
  }
  if (proposal.action === 'split') {
    if (!target || proposal.targetRegionIds.length !== 1 || proposal.proposedRegionIds.length !== 2) {
      throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Split önerisi bir kaynak ve iki çocuk taşımalıdır.')
    }
    const [firstChildId, secondChildId] = proposal.proposedRegionIds
    const first = proposal.proposedAtlas.regions.find((item) => item.id === firstChildId)
    const second = proposal.proposedAtlas.regions.find((item) => item.id === secondChildId)
    if (!first || !second) throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Split çocukları atlas içinde eksik.')
    const fractions = baseAtlas.pages.flatMap((page) => {
      const sourceWeight = pageRegionWeight(page, target)
      if (!(sourceWeight > 0)) return []
      const after = proposal.proposedAtlas.pages[page.pageIndex]
      const firstWeight = pageRegionWeight(after, firstChildId)
      const secondWeight = pageRegionWeight(after, secondChildId)
      if (Math.abs(firstWeight + secondWeight - sourceWeight) > 1e-6) {
        throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Split çocukları kaynak sayfa kütlesini korumuyor.')
      }
      return [{ pageIndex: page.pageIndex, firstChildFraction: firstWeight / sourceWeight }]
    })
    return {
      sourceRegionId: target,
      firstChildId: overrides.firstChildId ?? firstChildId,
      secondChildId: overrides.secondChildId ?? secondChildId,
      firstChildPrototype: overrides.firstChildPrototype ?? first.prototype,
      secondChildPrototype: overrides.secondChildPrototype ?? second.prototype,
      fractions,
      evidence,
      policy: proposal.policy,
    }
  }
  if (proposal.action === 'merge') {
    if (proposal.targetRegionIds.length < 2 || proposal.proposedRegionIds.length !== 1) {
      throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Merge önerisi en az iki kaynak ve bir hedef taşımalıdır.')
    }
    const region = proposal.proposedAtlas.regions.find((item) => item.id === proposed)
    if (!region) throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Merge hedef bölgesi atlas içinde eksik.')
    return {
      sourceRegionIds: proposal.targetRegionIds,
      mergedRegionId: overrides.mergedRegionId ?? proposed,
      prototype: overrides.prototype ?? region.prototype,
      evidence,
      policy: proposal.policy,
    }
  }
  if (proposal.action === 'retire') {
    if (!target || proposal.targetRegionIds.length !== 1 || proposal.proposedRegionIds.length) {
      throw releaseRunnerError('PLASTIC_REGION_PROPOSAL_TAMPERED', 'Retire önerisi tek bir hedef taşımalıdır.')
    }
    return { regionId: overrides.regionId ?? target, evidence, policy: proposal.policy }
  }
  throw releaseRunnerError('PLASTIC_REGION_RELEASE_ACTION_UNSUPPORTED', `Bilinmeyen yaşam döngüsü eylemi: ${proposal.action}`)
}

export function rebuildPlasticRegionReleaseProposal(baseAtlas, proposal, evidence, overrides = {}) {
  const input = lifecycleInput(baseAtlas, proposal, evidence, overrides)
  if (proposal.action === 'birth') return proposeRegionBirth(baseAtlas, input)
  if (proposal.action === 'grow') return proposeRegionGrow(baseAtlas, input)
  if (proposal.action === 'shrink') return proposeRegionShrink(baseAtlas, input)
  if (proposal.action === 'split') return proposeRegionSplit(baseAtlas, input)
  if (proposal.action === 'merge') return proposeRegionMerge(baseAtlas, input)
  return proposeRegionRetire(baseAtlas, input)
}

function atlasBehaviorSignature(atlas) {
  return {
    representationSpaceId: atlas.representationSpaceId,
    behaviorDimension: atlas.behaviorDimension,
    maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
    regions: atlas.regions.map((region) => ({
      id: region.id,
      lifecycle: region.lifecycle,
      prototype: region.prototype,
      confidenceMass: region.evidence.confidenceMass,
      cohesion: region.evidence.cohesion,
    })),
    pages: atlas.pages.map((page) => ({
      pageIndex: page.pageIndex,
      freeMass: page.freeMass,
      memberships: page.memberships,
    })),
  }
}

function lifecycleFootprint(baseAtlas, nextAtlas) {
  return baseAtlas.pages.flatMap((before, pageIndex) => {
    const after = nextAtlas.pages[pageIndex]
    const ids = [...new Set([
      ...before.memberships.map((item) => item.regionId),
      ...after.memberships.map((item) => item.regionId),
    ])]
    const changedMass = ids.reduce((sum, id) =>
      sum + Math.abs(pageRegionWeight(after, id) - pageRegionWeight(before, id)), 0)
    const freeMassDelta = Math.abs(after.freeMass - before.freeMass)
    return changedMass + freeMassDelta > 1e-8
      ? [{ pageIndex, changedMass, freeMassDelta }]
      : []
  })
}

export function createPlasticRegionReleaseEvaluationCandidate(baseAtlas, proposal, policy) {
  void policy
  const allowedPreReleaseReasons = new Set([
    'MISSING_INDEPENDENT_EVALUATION',
    'INSUFFICIENT_HOLDOUT_GAIN',
    'SHAM_EFFECT_TOO_HIGH',
  ])
  const blockingReasons = proposal.rejectionReasons.filter((reason) => !allowedPreReleaseReasons.has(reason))
  if (blockingReasons.length) {
    throw releaseRunnerError(
      'PLASTIC_REGION_PROPOSAL_TRAIN_EVIDENCE_INSUFFICIENT',
      'Holdout dışındaki eğitim kanıtı yaşam döngüsü eşiğini geçmedi; pahalı release ölçümü başlatılmadı.',
      409,
      { rejectionReasons: blockingReasons },
    )
  }
  const reconstructed = rebuildPlasticRegionReleaseProposal(baseAtlas, proposal, proposal.evidence)
  if (!sameCanonical(reconstructed, proposal)) {
    throw releaseRunnerError(
      'PLASTIC_REGION_PROPOSAL_TAMPERED',
      'Yaşam döngüsü önerisi base atlas üzerinden birebir yeniden üretilemedi.',
    )
  }
  const proposedIds = new Set(proposal.proposedRegionIds)
  const transientAtlas = createPlasticRegionAtlas({
    representationSpaceId: proposal.proposedAtlas.representationSpaceId,
    revision: proposal.proposedAtlas.revision,
    parentFingerprint: proposal.proposedAtlas.parentFingerprint,
    behaviorDimension: proposal.proposedAtlas.behaviorDimension,
    maximumMembershipsPerPage: proposal.proposedAtlas.maximumMembershipsPerPage,
    regions: proposal.proposedAtlas.regions.map((region) => proposedIds.has(region.id)
      ? { ...region, lifecycle: 'active' }
      : region),
    pages: proposal.proposedAtlas.pages,
  })
  return Object.freeze({
    ...proposal,
    proposedAtlas: transientAtlas,
    evaluatorOnly: true,
  })
}

function signedPrototypePermutation(values) {
  if (values.length < 2) return values.map((value) => -value)
  return values.map((_, index) => {
    const source = values[(index + 3) % values.length]
    return index % 2 === 0 ? -source : source
  })
}

function deterministicAlternativeAllocations(baseAtlas, allocations, proposalFingerprint) {
  const excluded = new Set(allocations.map((item) => item.pageIndex))
  const available = baseAtlas.pages
    .filter((page) => !excluded.has(page.pageIndex))
    .map((page) => page.pageIndex)
    .sort((left, right) => stableRuntimeHash('plastic-region-sham-page-v1', proposalFingerprint, left)
      .localeCompare(stableRuntimeHash('plastic-region-sham-page-v1', proposalFingerprint, right)) || left - right)
  const consumed = new Set()
  return allocations.map((allocation) => {
    const pageIndex = available.find((candidate) => {
      if (consumed.has(candidate)) return false
      const page = baseAtlas.pages[candidate]
      return page.freeMass + 1e-6 >= allocation.weight &&
        page.memberships.length < baseAtlas.maximumMembershipsPerPage
    })
    if (pageIndex === undefined) {
      throw releaseRunnerError(
        'PLASTIC_REGION_SHAM_CONSTRUCTION_FAILED',
        'Eş kapasiteli sham için öneriden bağımsız serbest sayfa bulunamadı.',
      )
    }
    consumed.add(pageIndex)
    return { pageIndex, weight: allocation.weight }
  })
}

function matchedMergeDonors(baseAtlas, proposal) {
  const excluded = new Set(proposal.targetRegionIds)
  const targetWeights = baseAtlas.pages.map((page) => proposal.targetRegionIds.reduce(
    (sum, regionId) => sum + pageRegionWeight(page, regionId), 0,
  ))
  const candidates = baseAtlas.regions
    .filter((region) => region.lifecycle !== 'retired' && !excluded.has(region.id))
    .sort((left, right) => left.id.localeCompare(right.id))
  let visits = 0
  const selected = []
  const accumulated = new Float64Array(baseAtlas.pages.length)
  const search = (offset) => {
    visits += 1
    if (visits > 16_384) return null
    if (selected.length >= 2 && accumulated.every((weight, index) => Math.abs(weight - targetWeights[index]) <= 1e-6)) {
      return [...selected]
    }
    for (let index = offset; index < candidates.length; index += 1) {
      const candidate = candidates[index]
      const weights = baseAtlas.pages.map((page) => pageRegionWeight(page, candidate.id))
      if (!weights.some((weight) => weight > 0) || weights.some((weight, pageIndex) =>
        accumulated[pageIndex] + weight > targetWeights[pageIndex] + 1e-6)) continue
      selected.push(candidate.id)
      weights.forEach((weight, pageIndex) => { accumulated[pageIndex] += weight })
      const match = search(index + 1)
      if (match) return match
      weights.forEach((weight, pageIndex) => { accumulated[pageIndex] -= weight })
      selected.pop()
    }
    return null
  }
  return search(0)
}

export function createPlasticRegionReleaseSham(baseAtlas, proposal, candidateProposal, releasePolicy) {
  const tentativeHoldout = Math.max(
    Number(proposal.policy.minimumConstructiveHoldoutGain),
    Number(releasePolicy.minimumHoldoutGain),
  )
  const tentativeEvidence = lifecycleEvidence(proposal, tentativeHoldout, 0)
  const original = lifecycleInput(baseAtlas, proposal, tentativeEvidence)
  if (proposal.action === 'split') {
    return proposeRegionSplit(baseAtlas, {
      ...original,
      firstChildId: `PRSHAM-${stableRuntimeHash(proposal.fingerprint, 'split-a').slice(0, 20).toUpperCase()}`,
      secondChildId: `PRSHAM-${stableRuntimeHash(proposal.fingerprint, 'split-b').slice(0, 20).toUpperCase()}`,
      firstChildPrototype: {
        dimension: original.firstChildPrototype.dimension,
        values: signedPrototypePermutation(original.firstChildPrototype.values),
      },
      secondChildPrototype: {
        dimension: original.secondChildPrototype.dimension,
        values: signedPrototypePermutation(original.secondChildPrototype.values),
      },
    })
  }
  if (proposal.action === 'merge') {
    const sourceRegionIds = matchedMergeDonors(baseAtlas, proposal)
    if (!sourceRegionIds) {
      throw releaseRunnerError(
        'PLASTIC_REGION_SHAM_CONSTRUCTION_FAILED',
        'Merge için kaynaklardan bağımsız, aynı sayfa/kütle ayak izli canlı bölge kümesi bulunamadı.',
      )
    }
    return proposeRegionMerge(baseAtlas, {
      ...original,
      sourceRegionIds,
      mergedRegionId: `PRSHAM-${stableRuntimeHash(proposal.fingerprint, 'merge').slice(0, 24).toUpperCase()}`,
      prototype: { dimension: original.prototype.dimension, values: signedPrototypePermutation(original.prototype.values) },
    })
  }
  if (proposal.action === 'retire') {
    const targetWeights = baseAtlas.pages.map((page) => pageRegionWeight(page, original.regionId))
    const donor = baseAtlas.regions.filter((region) => region.lifecycle !== 'retired' && region.id !== original.regionId)
      .find((region) => baseAtlas.pages.every((page, pageIndex) =>
        Math.abs(pageRegionWeight(page, region.id) - targetWeights[pageIndex]) <= 1e-6))
    if (!donor) throw releaseRunnerError('PLASTIC_REGION_SHAM_CONSTRUCTION_FAILED', 'Retire için eş ayak izli canlı donor bulunamadı.')
    return proposeRegionRetire(baseAtlas, { ...original, regionId: donor.id })
  }
  if (proposal.action === 'grow' || proposal.action === 'shrink') {
    const allocations = proposal.action === 'grow'
      ? deterministicAlternativeAllocations(baseAtlas, original.allocations, proposal.fingerprint)
      : original.allocations
    const donor = baseAtlas.regions.filter((region) => region.lifecycle !== 'retired' && region.id !== original.regionId)
      .find((region) => allocations.every((allocation) => {
        const page = baseAtlas.pages[allocation.pageIndex]
        return proposal.action === 'grow'
          ? page.freeMass + 1e-6 >= allocation.weight
          : pageRegionWeight(page, region.id) + 1e-6 >= allocation.weight
      }))
    if (!donor) throw releaseRunnerError('PLASTIC_REGION_SHAM_CONSTRUCTION_FAILED', `${proposal.action} için eş kapasiteli donor bulunamadı.`)
    return proposal.action === 'grow'
      ? proposeRegionGrow(baseAtlas, { ...original, regionId: donor.id, allocations })
      : proposeRegionShrink(baseAtlas, { ...original, regionId: donor.id, allocations })
  }
  if (proposal.action !== 'birth') {
    throw releaseRunnerError('PLASTIC_REGION_RELEASE_ACTION_UNSUPPORTED', `Sham eylemi desteklenmiyor: ${proposal.action}`)
  }
  const mapped = deterministicAlternativeAllocations(baseAtlas, original.allocations, proposal.fingerprint)
  const shamId = `PRSHAM-${stableRuntimeHash(proposal.fingerprint, 'footprint-sham').slice(0, 24).toUpperCase()}`
  const sham = proposeRegionBirth(baseAtlas, {
    ...original,
    regionId: shamId,
    allocations: mapped,
    prototype: {
      dimension: original.prototype.dimension,
      values: signedPrototypePermutation(original.prototype.values),
    },
    evidence: { ...original.evidence, shamGain: 0 },
  })
  if (!sham.eligible) {
    throw releaseRunnerError('PLASTIC_REGION_SHAM_CONSTRUCTION_FAILED', 'Eş büyüklükte sham atlas güvenli biçimde kurulamadı.')
  }
  return sham
}

function frozenHoldoutCases(caseCount) {
  const cases = plasticRegionReleaseCases(caseCount)
  const byGroup = new Map()
  for (const item of cases) {
    if (!byGroup.has(item.sourceGroupId)) byGroup.set(item.sourceGroupId, item)
  }
  const selected = [...byGroup.values()].slice(0, caseCount)
  if (selected.length !== caseCount || new Set(selected.map((item) => item.sourceGroupId)).size !== caseCount) {
    throw releaseRunnerError('INVALID_PLASTIC_REGION_RELEASE_CORPUS', 'Dondurulmuş release holdout grupları eksik.', 500)
  }
  return selected
}

export function plasticRegionReleaseMetadataHydrationPreflight(store, snapshot) {
  const config = store.getConfig()
  const metadata = (snapshot.artifactIds ?? [])
    .map((id) => store.getArtifactMetadata?.(id))
    .filter((entry) => entry && BASE_ARTIFACT_KINDS.has(entry.kind))
  const aggregatePayloadBytes = metadata.reduce((sum, entry) => sum + Math.max(0, Number(entry.payloadBytes) || 0), 0)
  const declaredNeuronCount = Number(
    snapshot?.metrics?.neuronCount ??
    snapshot?.metrics?.profileCount ??
    snapshot?.metrics?.familyProfileCount,
  )
  const declaredDenseVectorBytes = Number.isSafeInteger(declaredNeuronCount) && declaredNeuronCount >= 0
    ? declaredNeuronCount * 4_096 * Float32Array.BYTES_PER_ELEMENT
    : null
  if (declaredDenseVectorBytes !== null && (
    declaredNeuronCount > MAXIMUM_DENSE_NEURONS || declaredDenseVectorBytes > MAXIMUM_DENSE_VECTOR_BYTES
  )) {
    const error = releaseRunnerError(
      'JOB_PAUSED',
      'Release benchmark kataloğu metadata aşamasında güvenli dense sınırı aştı; büyük payload RAM’e açılmadı.',
      409,
      {
        reasonCode: 'PLASTIC_REGION_RELEASE_DENSE_BASE_UNSUPPORTED',
        neuronCount: declaredNeuronCount,
        denseVectorBytes: declaredDenseVectorBytes,
        maximumNeurons: MAXIMUM_DENSE_NEURONS,
        maximumDenseVectorBytes: MAXIMUM_DENSE_VECTOR_BYTES,
      },
    )
    error.retryable = true
    throw error
  }
  if (declaredDenseVectorBytes === null && aggregatePayloadBytes > MAXIMUM_DENSE_VECTOR_BYTES) {
    const error = releaseRunnerError(
      'JOB_PAUSED',
      'Eski metadata nöron sayısını kanıtlamıyor ve immutable payload üst sınırı dense RAM bütçesini aşıyor; decode edilmedi.',
      409,
      {
        reasonCode: 'PLASTIC_REGION_RELEASE_DENSE_BASE_UNPROVEN',
        aggregatePayloadBytes,
        maximumDenseVectorBytes: MAXIMUM_DENSE_VECTOR_BYTES,
      },
    )
    error.retryable = true
    throw error
  }
  const estimatedTransientBytes = Math.ceil(aggregatePayloadBytes * 2.5)
  const sample = sampleHostResources({ dataDirectory: config.dataDirectory ?? store.dataDirectory })
  const projected = {
    ...sample,
    rssBytes: sample.rssBytes + estimatedTransientBytes,
    heapUsedBytes: sample.heapUsedBytes + estimatedTransientBytes,
    hostFreeMemoryBytes: Math.max(0, sample.hostFreeMemoryBytes - estimatedTransientBytes),
  }
  const state = config.resourcePolicy?.enabled === false
    ? { allowed: true, reasonCodes: [], memoryTier: 'disabled' }
    : evaluateResourceState(config, projected, {
        logicalStorageBytes: store.approximateStorageBytes(),
        admission: true,
      })
  const heapCeilingBytes = Math.min(getHeapStatistics().heap_size_limit, 4 * 1024 * MEBIBYTE)
  if (!state.allowed || projected.heapUsedBytes >= heapCeilingBytes) {
    const error = releaseRunnerError(
      'JOB_PAUSED',
      'Release benchmark tabanı güvenli RAM rezervi oluşana kadar açılmadı.',
      409,
      {
        reasonCode: 'PLASTIC_REGION_RELEASE_RESOURCE_WAIT',
        aggregatePayloadBytes,
        estimatedTransientBytes,
        reasonCodes: [...(state.reasonCodes ?? []), ...(projected.heapUsedBytes < heapCeilingBytes ? [] : ['PROCESS_HEAP_4G_FAILSAFE'])],
      },
    )
    error.retryable = true
    throw error
  }
  return {
    aggregatePayloadBytes,
    estimatedTransientBytes,
    declaredNeuronCount: Number.isSafeInteger(declaredNeuronCount) ? declaredNeuronCount : null,
    declaredDenseVectorBytes,
    memoryTier: state.memoryTier,
    heapCeilingBytes,
  }
}

export function plasticRegionReleaseDenseBaseGuard(base) {
  const neuronCount = base.neurons.length
  const denseVectorBytes = neuronCount * base.layout.cellCount * Float32Array.BYTES_PER_ELEMENT
  if (neuronCount > MAXIMUM_DENSE_NEURONS || denseVectorBytes > MAXIMUM_DENSE_VECTOR_BYTES) {
    const error = releaseRunnerError(
      'JOB_PAUSED',
      'Release benchmarkı bu büyüklükteki kataloğu henüz SSD sayfalı değerlendiremiyor; aday etkinleşmeden kapalı tutuldu.',
      409,
      {
        reasonCode: 'PLASTIC_REGION_RELEASE_DENSE_BASE_UNSUPPORTED',
        neuronCount,
        denseVectorBytes,
        maximumNeurons: MAXIMUM_DENSE_NEURONS,
        maximumDenseVectorBytes: MAXIMUM_DENSE_VECTOR_BYTES,
      },
    )
    error.retryable = true
    throw error
  }
  return { neuronCount, denseVectorBytes, maximumNeurons: MAXIMUM_DENSE_NEURONS }
}

function thoughtRequestBase(base) {
  return {
    snapshotId: base.reference.snapshotId,
    snapshotStatus: base.reference.snapshotStatus,
    selectedBy: 'plastic-region-release-evaluator',
    activeAtStart: base.reference.activeAtStart,
    commitEligible: false,
    artifactManifestHash: base.reference.artifactManifestHash,
    baseModelFingerprint: base.reference.baseModelFingerprint,
    familyRunFingerprint: base.familyRun.fingerprint,
    languageStateFingerprint: base.languageState?.fingerprint ?? null,
    thoughtHeadFingerprint: base.thoughtHead.fingerprint,
    thoughtHeadRevision: base.thoughtHead.revision,
    conceptMacroHeadFingerprint: base.conceptMacroHead?.fingerprint ?? null,
    mode: 'independent-plastic-region-release-benchmark',
    ...(base.representationProfile ? { representationProfile: base.representationProfile } : {}),
  }
}

export async function evaluatePlasticRegionReleaseCase({
  thoughtEngine,
  base,
  atlas,
  regionMode,
  runtimePolicy,
  benchmarkPairingKey,
  variant,
  benchmarkCase,
  seed,
  maximumTicks,
  signal,
}) {
  abortIfRequested(signal)
  const result = await thoughtEngine.runEpisode({
    format: 'thought-engine-request',
    version: 1,
    episodeId: `region-release-${stableRuntimeHash(
      PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
      base.reference.snapshotId,
      variant,
      benchmarkCase.id,
      seed,
    ).slice(0, 32)}`,
    inputText: benchmarkCase.prompt,
    seed,
    maxTicks: maximumTicks,
    strategy: 'seeded-stochastic',
    learningMode: 'off',
    teacherMode: 'off',
    sourceGroupId: benchmarkCase.sourceGroupId,
    ...(regionMode === 'active' ? {
      adaptiveRegion: {
        mode: 'active',
        atlasFingerprint: atlas.fingerprint,
        compatibilityBlend: runtimePolicy.compatibilityBlend,
        minimumTrace: runtimePolicy.minimumTrace,
        maximumActiveGates: runtimePolicy.maximumActiveGates,
      },
    } : {}),
    base: thoughtRequestBase(base),
    expectedOutcome: benchmarkCase.expected,
  }, {
    signal,
    neurons: base.neurons,
    familyRun: base.familyRun,
    languageState: base.languageState,
    thoughtHead: base.thoughtHead,
    conceptMacroHead: base.conceptMacroHead,
    ...(regionMode === 'active' ? { adaptiveRegionAtlas: atlas } : {}),
    benchmarkPairingKey,
    representationProfile: base.representationProfile,
    baseDiagnostics: {
      evaluator: 'plastic-region-release',
      runnerFingerprint: PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
      variant,
    },
  })
  abortIfRequested(signal)
  return scorePlasticRegionReleaseGraph(result.graph.eventGraph, benchmarkCase.expected)
}

function auditValue(input) {
  const payload = {
    format: PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
    version: 1,
    runnerFingerprint: PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
    deterministic: true,
    usesOllama: false,
    automaticActivation: false,
    sourcePolicy: PLASTIC_REGION_RELEASE_SOURCE_POLICY,
    evaluationId: input.evaluationId,
    representationSpaceId: input.representationSpaceId,
    decision: input.decision,
    reasonCodes: [...input.reasonCodes],
    baseSnapshotId: input.baseSnapshotId,
    sourceSnapshotId: input.sourceSnapshotId,
    originalProposalArtifactId: input.originalProposalArtifactId,
    originalProposalFingerprint: input.originalProposalFingerprint,
    baseAtlasFingerprint: input.baseAtlasFingerprint,
    caseCount: input.caseCount,
    sourceGroupIds: [...input.sourceGroupIds],
    seeds: [...RELEASE_SEEDS],
    maximumTicks: input.maximumTicks,
    footprint: input.footprint ?? null,
    metrics: input.metrics ?? null,
    benchmarkReport: input.benchmarkReport ?? null,
    resourceSafety: input.resourceSafety ?? null,
    releaseArtifactIds: [...(input.releaseArtifactIds ?? [])],
  }
  return Object.freeze(validatePlasticRegionReleaseAudit({
    ...payload,
    fingerprint: stableRuntimeHash('plastic-region-release-audit-v1', payload),
  }))
}

function persistAudit(context, value) {
  return persistArtifact(context, {
    kind: PLASTIC_REGION_RELEASE_AUDIT_ARTIFACT_KIND,
    contentHash: stableRuntimeHash(
      PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
      'audit',
      value.baseSnapshotId,
      value.originalProposalFingerprint,
      value,
    ),
    value,
    metadata: {
      fingerprint: value.fingerprint,
      evaluationId: value.evaluationId,
      representationSpaceId: value.representationSpaceId,
      decision: value.decision,
      baseSnapshotId: value.baseSnapshotId,
      sourceSnapshotId: value.sourceSnapshotId,
      originalProposalArtifactId: value.originalProposalArtifactId,
      originalProposalFingerprint: value.originalProposalFingerprint,
      usesOllama: false,
      automaticActivation: false,
    },
  }).artifact
}

function regionArtifact(context, kind, value, expectedBase, metadata = {}) {
  const persisted = persistPlasticRegionArtifact(context, {
    kind,
    value,
    expectedBase,
    contentHash: stableRuntimeHash(
      PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
      'base-bound-release-artifact',
      kind,
      expectedBase,
      value,
    ),
    metadata: {
      ...metadata,
      runnerFingerprint: PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
      usesOllama: false,
    },
  }).artifact
  let restored
  try {
    restored = restoreArtifact(persisted, kind)
    assertPlasticRegionArtifactMetadata(kind, restored, persisted)
  } catch (cause) {
    throw releaseRunnerError(
      'PLASTIC_REGION_RELEASE_CACHE_TAMPERED',
      'Release artifact cache kaydı içerik/metadata doğrulamasını geçemedi.',
      409,
      { kind, artifactId: persisted?.id ?? null, cause: String(cause?.message ?? cause) },
    )
  }
  const envelope = persisted.payload?.metadata
  if (!sameCanonical(restored, value) ||
      envelope?.baseSnapshotId !== expectedBase.snapshotId ||
      (envelope?.baseAtlasArtifactId ?? null) !== (expectedBase.atlasArtifactId ?? null) ||
      (envelope?.casBaseAtlasFingerprint ?? null) !== (expectedBase.atlasFingerprint ?? null)) {
    throw releaseRunnerError(
      'PLASTIC_REGION_RELEASE_CACHE_TAMPERED',
      'Release artifact cache kaydı exact base veya değer bağıyla uyuşmuyor.',
      409,
      { kind, artifactId: persisted.id },
    )
  }
  return persisted
}

function artifactReference(artifact, value) {
  return { artifactId: artifact.id, contentHash: artifact.contentHash, fingerprint: value.fingerprint }
}

function releaseGate(policy, proposal, holdout, sham) {
  const reasons = []
  if (!proposal.eligible) reasons.push(...proposal.rejectionReasons.map((reason) => `proposal:${reason}`))
  if (holdout.metrics.caseCount < policy.minimumCases) reasons.push('holdout:insufficient-cases')
  if (holdout.metrics.independentSourceGroupCount < policy.minimumSourceGroups) reasons.push('holdout:insufficient-source-groups')
  if (holdout.metrics.observedGain < policy.minimumHoldoutGain) reasons.push('holdout:gain-below-threshold')
  if (holdout.metrics.regressionLoss > policy.maximumRegressionLoss) reasons.push('holdout:regression-too-high')
  if (sham.metrics.caseCount < policy.minimumCases) reasons.push('sham:insufficient-cases')
  if (sham.metrics.independentSourceGroupCount < policy.minimumSourceGroups) reasons.push('sham:insufficient-source-groups')
  if (Math.abs(sham.metrics.observedGain) > policy.maximumShamGain) reasons.push('sham:effect-too-high')
  if (sham.metrics.regressionLoss > policy.maximumRegressionLoss) reasons.push('sham:regression-too-high')
  return [...new Set(reasons)]
}

function releaseCandidatePublication(context, prepared, artifactIds, release, auditArtifact) {
  if (typeof context.store.completeClaimedJobWithCandidateOnSnapshotHeadBase !== 'function' || !context.jobClaim) {
    throw releaseRunnerError(
      'STORE_CAS_CONTRACT_MISSING',
      'Release adayı için claim ve snapshot/head CAS primitive’i gerekli.',
      500,
    )
  }
  const source = prepared.sourceSnapshot
  const candidateId = `region-release-${stableRuntimeHash(
    prepared.payload,
    release.fingerprint,
  ).slice(0, 32)}`
  return {
    input: {
      id: candidateId,
      parentId: prepared.payload.base.snapshotId,
      status: 'candidate',
      label: `${source.label ?? 'Aday'} · plastik bölge release`,
      metrics: source.metrics,
      artifactIds: [...new Set([...(source.artifactIds ?? []), ...artifactIds, auditArtifact.id])],
      sourceIds: [...(source.sourceIds ?? [])],
      sourceRevisions: { ...(source.sourceRevisions ?? {}) },
      inputManifestHash: source.inputManifestHash,
      evaluation: {
        ...(source.evaluation ?? {}),
        passed: true,
        runtimeVerified: true,
        plasticRegionReleasePassed: true,
        plasticRegionReleaseFingerprint: release.fingerprint,
        plasticRegionReleaseRunnerFingerprint: PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
        plasticRegionReleaseAutomaticActivation: false,
      },
    },
    expectedBase: {
      snapshotId: prepared.payload.base.snapshotId,
      headKind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      headArtifactId: prepared.payload.base.atlasArtifactId,
      headContentHash: prepared.payload.base.atlasContentHash,
      headFingerprint: prepared.payload.base.atlasFingerprint,
      staleCode: 'STALE_REGION_ATLAS_BASE',
    },
  }
}

export function createPlasticRegionReleaseExecutor({
  thoughtEngine = null,
  resolveBase = resolveThoughtEngineBase,
  evaluateCase = evaluatePlasticRegionReleaseCase,
  caseCount = DEFAULT_CASE_COUNT,
  maximumTicks = DEFAULT_MAXIMUM_TICKS,
} = {}) {
  if (typeof resolveBase !== 'function' || typeof evaluateCase !== 'function') {
    throw new TypeError('Release base çözümleyicisi ve vaka ölçücüsü fonksiyon olmalıdır.')
  }
  // Dependency injection exists only for Node's isolated test harness. The
  // production service/worker cannot stamp a custom score function with the
  // trusted release runner contract.
  if (evaluateCase !== evaluatePlasticRegionReleaseCase && process.env.NODE_TEST_CONTEXT === undefined) {
    throw new TypeError('Üretim release runnerı yalnız dondurulmuş gerçek düşünce motoru ölçücüsünü kabul eder.')
  }
  if (!Number.isInteger(caseCount) || caseCount < DEFAULT_CASE_COUNT || caseCount > MAXIMUM_CASE_COUNT) {
    throw new RangeError(`Release holdout vaka sayısı ${DEFAULT_CASE_COUNT}..${MAXIMUM_CASE_COUNT} olmalıdır.`)
  }
  if (!Number.isInteger(maximumTicks) || maximumTicks < 1 || maximumTicks > 64) {
    throw new RangeError('Release vaka tick sınırı 1..64 olmalıdır.')
  }

  return async function plasticRegionReleaseExecutor(job, context) {
    const alreadyCompleted = context.store.getJob?.(job?.id)
    if (alreadyCompleted?.state === 'succeeded' && alreadyCompleted.requestHash === job?.requestHash &&
        alreadyCompleted.result?.released === true && alreadyCompleted.result?.automaticActivation === false) {
      return alreadyCompleted.result
    }
    if (evaluateCase === evaluatePlasticRegionReleaseCase && typeof thoughtEngine?.runEpisode !== 'function') {
      throw releaseRunnerError(
        'THOUGHT_ENGINE_UNAVAILABLE',
        'Plastik bölge release değerlendirmesi gerçek Thought engine bağlanmadan çalışamaz.',
        503,
      )
    }
    const prepared = validateQueuedRequest(job, context.store)
    const { proposal, sourceSnapshot } = prepared
    const policy = plasticRegionReleasePolicy(context.config?.plasticRegions ?? context.store.getConfig().plasticRegions)
    if (policy.minimumCases > caseCount) {
      throw releaseRunnerError(
        'PLASTIC_REGION_RELEASE_POLICY_UNSUPPORTED',
        'Etkin release politikası runnerın dondurulmuş vaka bütçesinden büyük; eşik düşürülmeden runner genişletilmeli.',
        409,
        { minimumCases: policy.minimumCases, caseCount },
      )
    }
    const evaluationId = `PREVAL-${stableRuntimeHash(
      PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
      prepared.payload,
    ).slice(0, 40).toUpperCase()}`
    const baseAtlas = currentBaseAtlas(context.store, proposal, prepared.payload.base)
    let evaluationCandidate
    try {
      evaluationCandidate = createPlasticRegionReleaseEvaluationCandidate(baseAtlas, proposal, policy)
    } catch (error) {
      if (error?.code !== 'PLASTIC_REGION_PROPOSAL_TRAIN_EVIDENCE_INSUFFICIENT' &&
          error?.code !== 'PLASTIC_REGION_RELEASE_ACTION_UNSUPPORTED') throw error
      const audit = auditValue({
        evaluationId,
        representationSpaceId: proposal.proposedAtlas.representationSpaceId,
        decision: 'rejected-before-measurement',
        reasonCodes: error.details?.rejectionReasons ?? [error.code],
        baseSnapshotId: prepared.payload.base.snapshotId,
        sourceSnapshotId: sourceSnapshot.id,
        originalProposalArtifactId: prepared.payload.proposalArtifactId,
        originalProposalFingerprint: proposal.fingerprint,
        baseAtlasFingerprint: baseAtlas.fingerprint,
        caseCount: 0,
        sourceGroupIds: [],
        maximumTicks,
      })
      const auditArtifact = persistAudit(context, audit)
      await context.emit?.({
        type: 'plastic-region-release-rejected',
        level: 'warning',
        stage: 'region-release:preflight',
        data: { evaluationId, proposalArtifactId: prepared.payload.proposalArtifactId, reasonCodes: audit.reasonCodes },
      })
      await context.progress?.('region-release:rejected', 1)
      return {
        released: false,
        committed: false,
        automaticActivation: false,
        evaluationId,
        auditArtifactId: auditArtifact.id,
        reasonCodes: audit.reasonCodes,
      }
    }

    await context.checkpoint?.('region-release:metadata-preflight')
    const hydration = plasticRegionReleaseMetadataHydrationPreflight(context.store, sourceSnapshot)
    context.checkResources?.('region-release:hydrate-base')
    const base = resolveBase(context.store, sourceSnapshot.id)
    if (base.reference.snapshotId !== sourceSnapshot.id ||
        base.reference.artifactManifestHash !== prepared.payload.sourceArtifactManifestHash ||
        sourceSnapshot.inputManifestHash !== prepared.payload.sourceInputManifestHash ||
        sourceSnapshot.producerJobId !== prepared.payload.sourceProducerJobId) {
      throw releaseRunnerError('INVALID_PLASTIC_REGION_RELEASE_BASE', 'Release düşünce tabanı exact kaynak snapshota bağlanamadı.')
    }
    if (base.layout.side !== 64 || base.layout.cellCount !== 4_096 ||
        base.representationProfile !== 'adaptive-eighth-id-v1' || !base.neurons.length) {
      throw releaseRunnerError(
        'INVALID_PLASTIC_REGION_RELEASE_BASE',
        'Release değerlendirmesi açık adaptive 8/56 profilli, dolu 64×64 düşünce tabanı ister.',
      )
    }
    const dense = plasticRegionReleaseDenseBaseGuard(base)
    const runtimePolicy = Object.freeze({
      compatibilityBlend: policy.compatibilityBlend,
      minimumTrace: policy.minimumTrace,
      maximumActiveGates: policy.maximumActiveGates,
    })
    const baseMode = prepared.payload.base.atlasArtifactId ? 'active' : 'off'
    if (baseMode === 'active') {
      const baseRelease = readPlasticRegionRelease(context.store, context.store.getSnapshot(prepared.payload.base.snapshotId), {
        releasePolicy: context.config?.plasticRegions ?? context.store.getConfig().plasticRegions,
      })
      if (!baseRelease.compatible) {
        throw releaseRunnerError(
          'PLASTIC_REGION_RELEASE_BASE_INVALID',
          'Mevcut atlas baseline olarak ölçülmeden önce kendi immutable release zincirini doğrulamalıdır.',
          409,
          { reason: baseRelease.reason },
        )
      }
    }
    const baseBinding = Object.freeze({
      activeSnapshotId: prepared.payload.base.snapshotId,
      sourceSnapshotId: sourceSnapshot.id,
      sourceProducerJobId: prepared.payload.sourceProducerJobId,
      sourceInputManifestHash: prepared.payload.sourceInputManifestHash,
      sourceArtifactManifestHash: prepared.payload.sourceArtifactManifestHash,
      baseModelFingerprint: base.reference.baseModelFingerprint,
      familyRunFingerprint: base.reference.familyRunFingerprint,
      languageStateFingerprint: base.reference.languageStateFingerprint ?? null,
      thoughtHeadFingerprint: base.reference.thoughtHeadFingerprint,
      thoughtHeadRevision: base.reference.thoughtHeadRevision,
      conceptMacroHeadFingerprint: base.reference.conceptMacroHeadFingerprint ?? null,
      representationProfile: base.representationProfile,
      atlasArtifactId: prepared.payload.base.atlasArtifactId ?? null,
      atlasContentHash: prepared.payload.base.atlasContentHash ?? null,
      atlasFingerprint: baseAtlas.fingerprint,
    })
    const shamProposal = createPlasticRegionReleaseSham(baseAtlas, proposal, evaluationCandidate, policy)
    validateRegionLifecycleProposal(shamProposal)
    if (!shamProposal.eligible) {
      throw releaseRunnerError(
        'PLASTIC_REGION_SHAM_CONSTRUCTION_FAILED',
        'Eş ayak izli sham önerisi yaşam döngüsü kapılarından geçemedi.',
        409,
        { rejectionReasons: shamProposal.rejectionReasons },
      )
    }
    const candidateFootprint = lifecycleFootprint(baseAtlas, evaluationCandidate.proposedAtlas)
    const shamFootprint = lifecycleFootprint(baseAtlas, shamProposal.proposedAtlas)
    const normalizedFootprint = (items) => items.map(({ changedMass, freeMassDelta }) => ({
      changedMass,
      freeMassDelta,
    })).sort((left, right) => canonicalJson(left).localeCompare(canonicalJson(right)))
    if (candidateFootprint.length !== shamFootprint.length ||
        !sameCanonical(normalizedFootprint(candidateFootprint), normalizedFootprint(shamFootprint))) {
      throw releaseRunnerError('PLASTIC_REGION_SHAM_CONSTRUCTION_FAILED', 'Sham atlas adayla aynı kapasite ayak izini taşımıyor.')
    }

    const cases = frozenHoldoutCases(caseCount)
    const corpusFingerprint = plasticRegionReleaseCorpusFingerprint(caseCount)
    const scorerFingerprint = PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT
    const engineFingerprint = stableRuntimeHash(
      'plastic-region-release-engine-v1',
      thoughtEngine?.config ?? { evaluator: 'injected-server-owned' },
      base.reference.baseModelFingerprint,
      base.reference.familyRunFingerprint,
      base.reference.thoughtHeadFingerprint,
      base.reference.thoughtHeadRevision,
      base.reference.conceptMacroHeadFingerprint ?? null,
      base.representationProfile,
    )
    const pairingFingerprint = stableRuntimeHash(
      'plastic-region-release-pairing-v1',
      evaluationId,
      cases.map((item) => [item.id, item.sourceGroupId]),
      RELEASE_SEEDS,
      maximumTicks,
    )
    const rows = []
    const total = cases.length * RELEASE_SEEDS.length
    let completed = 0
    await context.emit?.({
      type: 'plastic-region-release-started',
      stage: 'region-release:benchmark',
      data: {
        evaluationId,
        proposalArtifactId: prepared.payload.proposalArtifactId,
        caseCount: total,
        sourceGroupCount: cases.length,
        usesOllama: false,
        automaticActivation: false,
      },
    })
    for (const benchmarkCase of cases) {
      for (const seed of RELEASE_SEEDS) {
        abortIfRequested(context.signal)
        const scores = {}
        const benchmarkPairingKey = stableRuntimeHash(
          'plastic-region-release-case-pair-v1',
          pairingFingerprint,
          benchmarkCase.id,
          benchmarkCase.sourceGroupId,
          seed,
        )
        for (const [variant, atlas, regionMode] of [
          ['base', baseAtlas, baseMode],
          ['candidate', evaluationCandidate.proposedAtlas, 'active'],
          ['sham', shamProposal.proposedAtlas, 'active'],
        ]) {
          const score = await evaluateCase({
            thoughtEngine,
            base,
            atlas,
            regionMode,
            runtimePolicy,
            benchmarkPairingKey,
            variant,
            benchmarkCase,
            seed,
            maximumTicks,
            signal: context.signal,
          })
          if (!Number.isFinite(score) || score < 0 || score > 1) {
            throw releaseRunnerError('INVALID_PLASTIC_REGION_RELEASE_MEASUREMENT', 'Yerel vaka ölçücüsü [0,1] dışında skor üretti.', 500)
          }
          scores[variant] = score
        }
        rows.push({
          caseId: benchmarkCase.id,
          sourceGroupId: benchmarkCase.sourceGroupId,
          seed,
          baseScore: scores.base,
          candidateScore: scores.candidate,
          shamScore: scores.sham,
        })
        completed += 1
        if (completed % 8 === 0 || completed === total) {
          await context.checkpoint?.(`region-release:benchmark:${completed}`)
          await context.progress?.('region-release:benchmark', completed / total)
        }
      }
    }
    const sourceGroupIds = [...new Set(rows.map((row) => row.sourceGroupId))].sort()
    const benchmarkReport = createPlasticRegionReleaseBenchmarkReport({
      evaluationId,
      runnerFingerprint: PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT,
      corpusFingerprint,
      engineFingerprint,
      engineContractFingerprint: PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
      scorerFingerprint,
      pairingFingerprint,
      representationSpaceId: proposal.proposedAtlas.representationSpaceId,
      geometryFingerprint: proposal.proposedAtlas.geometryFingerprint,
      baseMode,
      baseBinding,
      runtimePolicy,
      originalProposal: {
        ...artifactReference(context.store.getArtifact(prepared.payload.proposalArtifactId), proposal),
        proposalId: proposal.proposalId,
      },
      evaluatedCandidateAtlasFingerprint: evaluationCandidate.proposedAtlas.fingerprint,
      shamAtlasFingerprint: shamProposal.proposedAtlas.fingerprint,
      maximumTicks,
      sourceGroupIds,
      rows,
    })
    const derivedMetrics = derivePlasticRegionReleaseMetrics(benchmarkReport)
    const holdoutMetrics = derivedMetrics.holdout
    const shamMetrics = derivedMetrics.sham
    const finalProposal = rebuildPlasticRegionReleaseProposal(
      baseAtlas,
      proposal,
      lifecycleEvidence(proposal, holdoutMetrics.observedGain, shamMetrics.observedGain),
    )
    const finalFootprint = lifecycleFootprint(baseAtlas, finalProposal.proposedAtlas)
    const evaluationFootprint = lifecycleFootprint(baseAtlas, evaluationCandidate.proposedAtlas)
    const behaviorEquivalent = sameCanonical(
      atlasBehaviorSignature(finalProposal.proposedAtlas),
      atlasBehaviorSignature(evaluationCandidate.proposedAtlas),
    )
    const metrics = {
      baseMean: mean(rows.map((row) => row.baseScore)),
      candidateMean: mean(rows.map((row) => row.candidateScore)),
      shamMean: mean(rows.map((row) => row.shamScore)),
      holdout: derivedMetrics.summary.holdout,
      sham: derivedMetrics.summary.sham,
      conservativeHoldoutGain: holdoutMetrics.observedGain,
      conservativeShamGain: shamMetrics.observedGain,
      holdoutRegressionHigh: holdoutMetrics.regressionLoss,
      shamRegressionHigh: shamMetrics.regressionLoss,
    }
    const reasonCodes = releaseGate(policy, finalProposal, { metrics: holdoutMetrics }, { metrics: shamMetrics })
    if (reasonCodes.length) {
      const audit = auditValue({
        evaluationId,
        representationSpaceId: proposal.proposedAtlas.representationSpaceId,
        decision: 'rejected',
        reasonCodes,
        baseSnapshotId: prepared.payload.base.snapshotId,
        sourceSnapshotId: sourceSnapshot.id,
        originalProposalArtifactId: prepared.payload.proposalArtifactId,
        originalProposalFingerprint: proposal.fingerprint,
        baseAtlasFingerprint: baseAtlas.fingerprint,
        caseCount: rows.length,
        sourceGroupIds,
        maximumTicks,
        footprint: { candidate: candidateFootprint, sham: shamFootprint, behaviorEquivalent },
        metrics,
        benchmarkReport,
        resourceSafety: { hydration, dense },
        releaseArtifactIds: [],
      })
      const auditArtifact = persistAudit(context, audit)
      await context.emit?.({
        type: 'plastic-region-release-rejected',
        level: 'warning',
        stage: 'region-release:decision',
        data: { evaluationId, proposalArtifactId: prepared.payload.proposalArtifactId, reasonCodes },
      })
      await context.progress?.('region-release:rejected', 1)
      return {
        released: false,
        committed: false,
        automaticActivation: false,
        evaluationId,
        auditArtifactId: auditArtifact.id,
        reasonCodes,
        metrics,
      }
    }
    if (!behaviorEquivalent) {
      throw releaseRunnerError(
        'PLASTIC_REGION_RELEASE_BEHAVIOR_DRIFT',
        'Ölçülen geçici atlas ile release atlasının yönlendirme davranışı aynı değil.',
        500,
      )
    }

    await context.checkpoint?.('region-release:persist-proposal')
    const benchmarkReportArtifact = regionArtifact(
      context,
      PLASTIC_REGION_RELEASE_BENCHMARK_ARTIFACT_KIND,
      benchmarkReport,
      prepared.payload.base,
      { sourcePolicy: PLASTIC_REGION_RELEASE_SOURCE_POLICY, automaticActivation: false },
    )
    const benchmarkReportReference = artifactReference(benchmarkReportArtifact, benchmarkReport)
    const holdout = createPlasticRegionReleaseEvidence({
      evaluationId: `${evaluationId}-HOLDOUT`,
      evaluationKind: 'holdout',
      representationSpaceId: finalProposal.proposedAtlas.representationSpaceId,
      geometryFingerprint: finalProposal.proposedAtlas.geometryFingerprint,
      proposalId: finalProposal.proposalId,
      proposalFingerprint: finalProposal.fingerprint,
      proposedAtlasFingerprint: finalProposal.proposedAtlas.fingerprint,
      baseBinding,
      benchmarkReport: benchmarkReportReference,
      sourceGroupIds,
      metrics: holdoutMetrics,
    })
    const sham = createPlasticRegionReleaseEvidence({
      evaluationId: `${evaluationId}-SHAM`,
      evaluationKind: 'sham',
      representationSpaceId: finalProposal.proposedAtlas.representationSpaceId,
      geometryFingerprint: finalProposal.proposedAtlas.geometryFingerprint,
      proposalId: finalProposal.proposalId,
      proposalFingerprint: finalProposal.fingerprint,
      proposedAtlasFingerprint: finalProposal.proposedAtlas.fingerprint,
      baseBinding,
      benchmarkReport: benchmarkReportReference,
      sourceGroupIds,
      metrics: shamMetrics,
    })
    const finalProposalArtifact = regionArtifact(
      context,
      PLASTIC_REGION_ATLAS_PROPOSAL_ARTIFACT_KIND,
      finalProposal,
      prepared.payload.base,
      {
        eligible: true,
        activationEligible: true,
        supersedesProposalArtifactId: prepared.payload.proposalArtifactId,
      },
    )
    const holdoutArtifact = regionArtifact(
      context,
      PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
      holdout,
      prepared.payload.base,
      { sourcePolicy: PLASTIC_REGION_RELEASE_SOURCE_POLICY },
    )
    const shamArtifact = regionArtifact(
      context,
      PLASTIC_REGION_RELEASE_EVIDENCE_ARTIFACT_KIND,
      sham,
      prepared.payload.base,
      { sourcePolicy: PLASTIC_REGION_RELEASE_SOURCE_POLICY, footprintMatched: true },
    )
    const atlasArtifact = regionArtifact(
      context,
      PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      finalProposal.proposedAtlas,
      prepared.payload.base,
      {
        candidateOnly: false,
        activationEligible: true,
        proposalArtifactId: finalProposalArtifact.id,
      },
    )
    const releaseId = `PRREL-${stableRuntimeHash(evaluationId, finalProposal.fingerprint).slice(0, 40).toUpperCase()}`
    const manifest = createPlasticRegionReleaseManifest({
      releaseId,
      representationSpaceId: finalProposal.proposedAtlas.representationSpaceId,
      geometryFingerprint: finalProposal.proposedAtlas.geometryFingerprint,
      baseBinding,
      benchmarkReport: benchmarkReportReference,
      atlas: {
        ...artifactReference(atlasArtifact, finalProposal.proposedAtlas),
        revision: finalProposal.proposedAtlas.revision,
        behaviorDimension: finalProposal.proposedAtlas.behaviorDimension,
        maximumMembershipsPerPage: finalProposal.proposedAtlas.maximumMembershipsPerPage,
      },
      proposal: {
        ...artifactReference(finalProposalArtifact, finalProposal),
        proposalId: finalProposal.proposalId,
        baseAtlasFingerprint: finalProposal.baseAtlasFingerprint,
        proposedAtlasFingerprint: finalProposal.proposedAtlas.fingerprint,
      },
      holdoutEvidence: {
        ...artifactReference(holdoutArtifact, holdout),
        evaluationId: holdout.evaluationId,
      },
      shamEvidence: {
        ...artifactReference(shamArtifact, sham),
        evaluationId: sham.evaluationId,
      },
      policy,
    })
    const manifestArtifact = regionArtifact(
      context,
      PLASTIC_REGION_RELEASE_MANIFEST_ARTIFACT_KIND,
      manifest,
      prepared.payload.base,
      { activationMode: 'active', automaticActivation: false },
    )
    const releaseArtifactIds = [
      benchmarkReportArtifact.id,
      finalProposalArtifact.id,
      holdoutArtifact.id,
      shamArtifact.id,
      atlasArtifact.id,
      manifestArtifact.id,
    ]
    const audit = auditValue({
      evaluationId,
      representationSpaceId: proposal.proposedAtlas.representationSpaceId,
      decision: 'approved-awaiting-manual-activation',
      reasonCodes: [],
      baseSnapshotId: prepared.payload.base.snapshotId,
      sourceSnapshotId: sourceSnapshot.id,
      originalProposalArtifactId: prepared.payload.proposalArtifactId,
      originalProposalFingerprint: proposal.fingerprint,
      baseAtlasFingerprint: baseAtlas.fingerprint,
      caseCount: rows.length,
      sourceGroupIds,
      maximumTicks,
      footprint: { candidate: candidateFootprint, sham: shamFootprint, behaviorEquivalent },
      metrics,
      resourceSafety: { hydration, dense },
      releaseArtifactIds,
    })
    const auditArtifact = persistAudit(context, audit)
    const publication = releaseCandidatePublication(
      context,
      prepared,
      releaseArtifactIds,
      manifest,
      auditArtifact,
    )
    const finalResult = {
      released: true,
      committed: false,
      awaitingApproval: true,
      automaticActivation: false,
      evaluationId,
      snapshotId: publication.input.id,
      releaseFingerprint: manifest.fingerprint,
      auditArtifactId: auditArtifact.id,
      releaseArtifactIds,
      metrics,
    }
    await context.checkpoint?.('region-release:candidate-cas')
    await context.progress?.('region-release:candidate-cas', 0.99)
    const publicationCommit = context.store.completeClaimedJobWithCandidateOnSnapshotHeadBase(
      context.jobId,
      context.jobClaim.owner,
      context.jobClaim.attempt,
      publication.input,
      publication.expectedBase,
      finalResult,
      (candidate) => {
        const gate = plasticRegionSnapshotActivationGate(context.store, candidate)
        return gate.required && gate.eligible && gate.release?.manifest?.fingerprint === manifest.fingerprint
          ? true
          : {
              eligible: false,
              code: 'PLASTIC_REGION_RELEASE_CHAIN_INVALID',
              message: 'Release zinciri aday snapshot üzerinde yeniden doğrulanamadı.',
              details: { reason: gate.reason },
            }
      },
    )
    if (publicationCommit.snapshot.id !== finalResult.snapshotId || publicationCommit.job.state !== 'succeeded') {
      throw releaseRunnerError('PLASTIC_REGION_RELEASE_ATOMIC_PUBLICATION_FAILED', 'Release adayı atomik yayın doğrulamasını geçemedi.', 500)
    }
    return { ...finalResult, atomicCandidateJobCompletion: true }
  }
}
