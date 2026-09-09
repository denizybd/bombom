import {
  createPlasticRegionAtlas,
  validateAdaptiveRegionAtlas,
} from '../src/adaptive-regions/index.ts'
import { resolveThoughtEngineBase } from '../server/thought-service.mjs'
import { persistArtifact, restoreArtifact, stableRuntimeHash } from './artifacts.mjs'
import {
  PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
  assertPlasticRegionArtifactMetadata,
  currentPlasticRegionBase,
  persistPlasticRegionArtifact,
  plasticRegionSnapshotActivationGate,
  readPlasticRegionRelease,
  readPlasticRegionSnapshot,
} from './plastic-region-cache.mjs'
import {
  PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND,
  PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT,
  createPlasticRegionRenewalAudit,
  createPlasticRegionRenewalEvidence,
  createPlasticRegionRenewalManifest,
} from './plastic-region-renewal.mjs'
import { derivePlasticRegionReleaseMetrics, plasticRegionReleasePolicy } from './plastic-region-release.mjs'
import {
  PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT,
  PLASTIC_REGION_RELEASE_DEFAULT_MAXIMUM_TICKS,
  PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
  PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT,
  PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
  PLASTIC_REGION_RELEASE_SEEDS,
  plasticRegionReleaseCases,
  plasticRegionReleaseCorpusFingerprint,
} from './plastic-region-release-spec.mjs'
import {
  evaluatePlasticRegionReleaseCase,
  plasticRegionReleaseDenseBaseGuard,
  plasticRegionReleaseMetadataHydrationPreflight,
} from './plastic-region-release-runner.mjs'

export const PLASTIC_REGION_RENEWAL_JOB_TYPE = 'plastic-region-release-renewal'

const PAYLOAD_KEYS = Object.freeze([
  'base',
  'sourceSnapshotId',
  'sourceProducerJobId',
  'sourceInputManifestHash',
  'sourceArtifactManifestHash',
  'activeReleaseManifestArtifactId',
  'activeReleaseManifestContentHash',
  'activeReleaseFingerprint',
  'runnerFingerprint',
])

function runnerError(code, message, status = 409, details) {
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

function exactKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value) &&
    sameCanonical(Object.keys(value).sort(), [...keys].sort())
}

function abortIfRequested(signal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError')
}

function sourceArtifactManifestHash(store, snapshot) {
  const entries = (snapshot.artifactIds ?? []).map((id) => {
    const metadata = store.getArtifactMetadata?.(id)
    if (!metadata) throw runnerError('PLASTIC_REGION_RENEWAL_SOURCE_INVALID', 'Kaynak aday artifact manifesti eksik.')
    return { id: metadata.id, kind: metadata.kind, contentHash: metadata.contentHash }
  }).sort((left, right) => left.id.localeCompare(right.id))
  return stableRuntimeHash('thought-base-artifacts-v1', entries)
}

function certifiedSource(store, sourceSnapshotId) {
  const source = store.getSnapshot?.(String(sourceSnapshotId ?? '').trim())
  if (!source || source.status !== 'candidate') {
    throw runnerError('PLASTIC_REGION_RENEWAL_SOURCE_NOT_FOUND', 'Renewal için candidate model snapshotı bulunamadı.', 404)
  }
  const active = store.getActiveSnapshot?.()
  if (!active || source.parentId !== active.id) {
    throw runnerError('STALE_REGION_ATLAS_BASE', 'Renewal kaynağı güncel etkin modelin doğrudan adayı olmalıdır.')
  }
  const producer = store.getJob?.(source.producerJobId)
  const certified = new Set([
    producer?.result?.snapshotId, producer?.result?.lastSnapshotId,
    ...(producer?.result?.snapshotIds ?? []), ...(producer?.result?.trainingSnapshotIds ?? []),
  ].filter(Boolean))
  if (producer?.state !== 'succeeded' || !certified.has(source.id) ||
      source.evaluation?.passed !== true || source.evaluation?.runtimeVerified !== true ||
      typeof source.inputManifestHash !== 'string' || !source.inputManifestHash) {
    throw runnerError('PLASTIC_REGION_RENEWAL_SOURCE_INVALID', 'Renewal kaynağı tamamlanmış üretici ve bağımsız genel değerlendirme taşımıyor.')
  }
  return { source, active, producer }
}

function activeReleaseManifestRecord(store, snapshot, release) {
  const records = (snapshot.artifactIds ?? []).map((id) => store.getArtifactMetadata?.(id))
    .filter((metadata) => metadata && (
      metadata.kind === 'plastic-region-release-manifest' ||
      metadata.kind === PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND
    ))
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)) || String(right.id).localeCompare(String(left.id)))
  for (const metadata of records) {
    try {
      const value = restoreArtifact(store.getArtifact(metadata.id), metadata.kind)
      if (value?.fingerprint === release.manifest?.fingerprint) return { metadata, value }
    } catch {
      // Only the exact verified release manifest may authorize renewal.
    }
  }
  throw runnerError('PLASTIC_REGION_RENEWAL_BASE_INVALID', 'Etkin release manifest artifacti doğrulanamadı.')
}

export function preparePlasticRegionRenewalJob(store, sourceSnapshotId) {
  const { source, active, producer } = certifiedSource(store, sourceSnapshotId)
  const release = readPlasticRegionRelease(store, active)
  if (!release.compatible) {
    throw runnerError('PLASTIC_REGION_RENEWAL_BASE_INVALID', 'Renewal yalnız doğrulanmış etkin bir plastik bölge release’i üzerinde çalışır.', 409, {
      reason: release.reason,
    })
  }
  const activeAtlas = readPlasticRegionSnapshot(store, active)
  const sourceAtlas = readPlasticRegionSnapshot(store, source)
  if (!activeAtlas.compatible || !sourceAtlas.compatible ||
      sourceAtlas.metadata?.id !== activeAtlas.metadata?.id ||
      sourceAtlas.metadata?.contentHash !== activeAtlas.metadata?.contentHash ||
      sourceAtlas.head?.fingerprint !== activeAtlas.head?.fingerprint) {
    throw runnerError(
      'PLASTIC_REGION_RENEWAL_ATLAS_CHANGED',
      'Renewal yalnız atlas artifacti byte düzeyinde değişmeden yeni model adayına taşındığında kullanılabilir.',
    )
  }
  const currentBase = currentPlasticRegionBase(store)
  if (currentBase.snapshotId !== active.id || currentBase.atlasArtifactId !== activeAtlas.metadata.id ||
      currentBase.atlasFingerprint !== activeAtlas.head.fingerprint) {
    throw runnerError('STALE_REGION_ATLAS_BASE', 'Renewal etkin atlas başlığı güncel değil.')
  }
  const manifest = activeReleaseManifestRecord(store, active, release)
  const payload = Object.freeze({
    base: Object.freeze({ ...currentBase }),
    sourceSnapshotId: source.id,
    sourceProducerJobId: producer.id,
    sourceInputManifestHash: source.inputManifestHash,
    sourceArtifactManifestHash: sourceArtifactManifestHash(store, source),
    activeReleaseManifestArtifactId: manifest.metadata.id,
    activeReleaseManifestContentHash: manifest.metadata.contentHash,
    activeReleaseFingerprint: release.manifest.fingerprint,
    runnerFingerprint: PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT,
  })
  return Object.freeze({
    payload,
    requestHash: stableRuntimeHash('plastic-region-release-renewal-job-v1', payload),
    defaultIdempotencyKey: `plastic-region-renewal:${stableRuntimeHash(payload).slice(0, 48)}`,
    sourceSnapshot: source,
    activeSnapshot: active,
    atlas: activeAtlas.head,
  })
}

function validateQueuedRequest(job, store) {
  if (job?.type !== PLASTIC_REGION_RENEWAL_JOB_TYPE || !exactKeys(job.payload, PAYLOAD_KEYS)) {
    throw runnerError('INVALID_PLASTIC_REGION_RENEWAL_JOB', 'Renewal işi yalnız sunucunun dondurduğu alanları taşıyabilir.', 400)
  }
  const prepared = preparePlasticRegionRenewalJob(store, job.payload.sourceSnapshotId)
  if (!sameCanonical(job.payload, prepared.payload) || job.requestHash !== prepared.requestHash) {
    throw runnerError('PLASTIC_REGION_RENEWAL_REQUEST_TAMPERED', 'Renewal kuyruğu immutable source/base sözleşmesiyle uyuşmuyor.')
  }
  return prepared
}

function changedPrototype(values, salt) {
  const changed = values.map((_, index) => {
    const source = values[(index + 3) % values.length]
    return index % 2 === 0 ? -source : source
  })
  if (sameCanonical(changed, values)) {
    const index = Number.parseInt(stableRuntimeHash('renewal-sham-prototype-index', salt).slice(0, 8), 16) % values.length
    changed[index] = values[index] >= 0 ? Math.max(-1, values[index] - 0.5) : Math.min(1, values[index] + 0.5)
  }
  return changed
}

function renewalShamAtlas(atlas) {
  if (!atlas.regions.some((region) => region.lifecycle === 'active')) {
    throw runnerError('PLASTIC_REGION_RENEWAL_SHAM_UNAVAILABLE', 'Boş/etkinsiz atlas için model renewal release’i üretilemez.')
  }
  const sham = createPlasticRegionAtlas({
    representationSpaceId: atlas.representationSpaceId,
    revision: atlas.revision,
    parentFingerprint: atlas.parentFingerprint,
    behaviorDimension: atlas.behaviorDimension,
    maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
    regions: atlas.regions.map((region) => region.lifecycle === 'active' ? {
      ...region,
      prototype: {
        dimension: region.prototype.dimension,
        values: changedPrototype(region.prototype.values, `${atlas.fingerprint}:${region.id}`),
      },
    } : region),
    pages: atlas.pages,
  })
  if (sham.fingerprint === atlas.fingerprint) {
    throw runnerError('PLASTIC_REGION_RENEWAL_SHAM_UNAVAILABLE', 'Deterministik sham atlas etkin atlasla aynı kaldı.')
  }
  return sham
}

function artifactReference(artifact, value) {
  return { artifactId: artifact.id, contentHash: artifact.contentHash, fingerprint: value.fingerprint }
}

function persistRegionArtifact(context, kind, value, expectedBase, metadata) {
  const artifact = persistPlasticRegionArtifact(context, {
    kind,
    value,
    expectedBase,
    contentHash: stableRuntimeHash(
      PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT,
      'base-bound-renewal-artifact',
      kind,
      expectedBase,
      value,
    ),
    metadata,
  }).artifact
  const restored = restoreArtifact(artifact, kind)
  assertPlasticRegionArtifactMetadata(kind, restored, artifact)
  if (!sameCanonical(restored, value) || artifact.payload?.metadata?.baseSnapshotId !== expectedBase.snapshotId ||
      artifact.payload?.metadata?.baseAtlasArtifactId !== expectedBase.atlasArtifactId ||
      artifact.payload?.metadata?.casBaseAtlasFingerprint !== expectedBase.atlasFingerprint) {
    throw runnerError('PLASTIC_REGION_RENEWAL_CACHE_TAMPERED', 'Renewal artifacti exact base bağıyla uyuşmuyor.')
  }
  return artifact
}

function persistAudit(context, value) {
  return persistArtifact(context, {
    kind: PLASTIC_REGION_RENEWAL_AUDIT_ARTIFACT_KIND,
    contentHash: stableRuntimeHash(PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT, 'audit', value),
    value,
    metadata: {
      representationSpaceId: value.representationSpaceId,
      fingerprint: value.fingerprint,
      renewalId: value.renewalId,
      decision: value.decision,
      baseSnapshotId: value.baseSnapshotId,
      sourceSnapshotId: value.sourceSnapshotId,
      atlasFingerprint: value.atlasFingerprint,
      usesOllama: false,
      automaticActivation: false,
    },
  }).artifact
}

function renewalReasons(policy, metrics) {
  const reasons = []
  if (metrics.holdout.caseCount < policy.minimumCases) reasons.push('holdout:insufficient-cases')
  if (metrics.holdout.independentSourceGroupCount < policy.minimumSourceGroups) reasons.push('holdout:insufficient-source-groups')
  if (metrics.holdout.observedGain < policy.minimumHoldoutGain) reasons.push('holdout:gain-below-threshold')
  if (metrics.holdout.regressionLoss > policy.maximumRegressionLoss) reasons.push('holdout:regression-too-high')
  if (metrics.sham.caseCount < policy.minimumCases) reasons.push('sham:insufficient-cases')
  if (metrics.sham.independentSourceGroupCount < policy.minimumSourceGroups) reasons.push('sham:insufficient-source-groups')
  if (Math.abs(metrics.sham.observedGain) > policy.maximumShamGain) reasons.push('sham:effect-too-high')
  if (metrics.sham.regressionLoss > policy.maximumRegressionLoss) reasons.push('sham:regression-too-high')
  return reasons
}

export function createPlasticRegionRenewalExecutor({
  thoughtEngine = null,
  resolveBase = resolveThoughtEngineBase,
  evaluateCase = evaluatePlasticRegionReleaseCase,
  caseCount = PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT,
  maximumTicks = PLASTIC_REGION_RELEASE_DEFAULT_MAXIMUM_TICKS,
} = {}) {
  if (typeof resolveBase !== 'function' || typeof evaluateCase !== 'function') {
    throw new TypeError('Renewal base çözümleyicisi ve vaka ölçücüsü fonksiyon olmalıdır.')
  }
  // Dependency injection exists only for Node's isolated test harness. The
  // production service/worker cannot stamp a custom score function with the
  // trusted renewal runner contract.
  if (evaluateCase !== evaluatePlasticRegionReleaseCase && process.env.NODE_TEST_CONTEXT === undefined) {
    throw new TypeError('Üretim renewal runnerı yalnız dondurulmuş gerçek düşünce motoru ölçücüsünü kabul eder.')
  }
  if (!Number.isInteger(caseCount) || caseCount < PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT ||
      caseCount > PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT || !Number.isInteger(maximumTicks) ||
      maximumTicks < 1 || maximumTicks > 64) {
    throw new RangeError('Renewal runner vaka/tick sınırı geçersiz.')
  }
  return async function executePlasticRegionRenewal(job, context) {
    const alreadyCompleted = context.store.getJob?.(job?.id)
    if (alreadyCompleted?.state === 'succeeded' && alreadyCompleted.requestHash === job?.requestHash &&
        alreadyCompleted.result?.renewed === true && alreadyCompleted.result?.automaticActivation === false) {
      return alreadyCompleted.result
    }
    if (evaluateCase === evaluatePlasticRegionReleaseCase && typeof thoughtEngine?.runEpisode !== 'function') {
      throw runnerError('THOUGHT_ENGINE_UNAVAILABLE', 'Renewal değerlendirmesi gerçek düşünce motoru gerektirir.', 503)
    }
    const prepared = validateQueuedRequest(job, context.store)
    const { sourceSnapshot, atlas } = prepared
    const policy = plasticRegionReleasePolicy(context.config?.plasticRegions ?? context.store.getConfig().plasticRegions)
    if (policy.minimumCases > caseCount) throw runnerError('PLASTIC_REGION_RENEWAL_POLICY_UNSUPPORTED', 'Renewal vaka bütçesi etkin release politikasını karşılamıyor.')
    const hydration = plasticRegionReleaseMetadataHydrationPreflight(context.store, sourceSnapshot)
    const sourceBase = await resolveBase(context.store, sourceSnapshot.id, { strictActive: false, allowCandidate: true })
    if (sourceBase.reference.snapshotId !== sourceSnapshot.id || sourceBase.reference.snapshotStatus !== 'candidate' ||
        sourceBase.reference.artifactManifestHash !== prepared.payload.sourceArtifactManifestHash ||
        sourceBase.representationProfile !== 'adaptive-eighth-id-v1' || sourceBase.layout?.cellCount !== 4_096 ||
        !sourceBase.neurons?.length || sourceBase.familyRun?.representationSpaceId !== atlas.representationSpaceId) {
      throw runnerError('PLASTIC_REGION_RENEWAL_SOURCE_INVALID', 'Renewal kaynak modeli exact 64×64 adaptive source bağıyla çözülemedi.')
    }
    const dense = plasticRegionReleaseDenseBaseGuard(sourceBase)
    const shamAtlas = renewalShamAtlas(atlas)
    validateAdaptiveRegionAtlas(shamAtlas)
    const runtimePolicy = Object.freeze({
      compatibilityBlend: policy.compatibilityBlend,
      minimumTrace: policy.minimumTrace,
      maximumActiveGates: policy.maximumActiveGates,
    })
    const baseBinding = Object.freeze({
      activeSnapshotId: prepared.payload.base.snapshotId,
      activeReleaseManifestArtifactId: prepared.payload.activeReleaseManifestArtifactId,
      activeReleaseManifestContentHash: prepared.payload.activeReleaseManifestContentHash,
      activeReleaseFingerprint: prepared.payload.activeReleaseFingerprint,
      sourceSnapshotId: sourceSnapshot.id,
      sourceProducerJobId: prepared.payload.sourceProducerJobId,
      sourceInputManifestHash: prepared.payload.sourceInputManifestHash,
      sourceArtifactManifestHash: prepared.payload.sourceArtifactManifestHash,
      sourceBaseModelFingerprint: sourceBase.reference.baseModelFingerprint,
      sourceFamilyRunFingerprint: sourceBase.reference.familyRunFingerprint,
      sourceLanguageStateFingerprint: sourceBase.reference.languageStateFingerprint ?? null,
      sourceThoughtHeadFingerprint: sourceBase.reference.thoughtHeadFingerprint,
      sourceThoughtHeadRevision: sourceBase.reference.thoughtHeadRevision,
      sourceConceptMacroHeadFingerprint: sourceBase.reference.conceptMacroHeadFingerprint ?? null,
      representationProfile: sourceBase.representationProfile,
      atlasArtifactId: prepared.payload.base.atlasArtifactId,
      atlasContentHash: prepared.payload.base.atlasContentHash,
      atlasFingerprint: atlas.fingerprint,
      atlasRevision: atlas.revision,
    })
    const cases = plasticRegionReleaseCases(caseCount)
    const renewalId = `PRRENEW-${stableRuntimeHash(PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT, prepared.payload).slice(0, 40).toUpperCase()}`
    const pairingFingerprint = stableRuntimeHash('plastic-region-renewal-pairing-v1', renewalId, maximumTicks, cases, PLASTIC_REGION_RELEASE_SEEDS)
    const engineFingerprint = stableRuntimeHash(
      'plastic-region-renewal-engine-v1',
      thoughtEngine?.config ?? { evaluator: 'injected-server-owned' },
      sourceBase.reference.baseModelFingerprint,
      sourceBase.reference.familyRunFingerprint,
      sourceBase.reference.thoughtHeadFingerprint,
      sourceBase.reference.thoughtHeadRevision,
      sourceBase.reference.conceptMacroHeadFingerprint ?? null,
      sourceBase.representationProfile,
    )
    const rows = []
    let completed = 0
    const total = cases.length * PLASTIC_REGION_RELEASE_SEEDS.length
    for (const benchmarkCase of cases) {
      for (const seed of PLASTIC_REGION_RELEASE_SEEDS) {
        abortIfRequested(context.signal)
        const benchmarkPairingKey = stableRuntimeHash('plastic-region-renewal-pair-v1', pairingFingerprint, benchmarkCase.id, seed)
        const scores = {}
        for (const [variant, variantAtlas, regionMode] of [
          ['base', atlas, 'off'],
          ['candidate', atlas, 'active'],
          ['sham', shamAtlas, 'active'],
        ]) {
          scores[variant] = await evaluateCase({
            thoughtEngine,
            base: sourceBase,
            atlas: variantAtlas,
            regionMode,
            runtimePolicy,
            benchmarkPairingKey,
            variant,
            benchmarkCase,
            seed,
            maximumTicks,
            signal: context.signal,
          })
          if (!Number.isFinite(scores[variant]) || scores[variant] < 0 || scores[variant] > 1) {
            throw runnerError('INVALID_PLASTIC_REGION_RENEWAL_MEASUREMENT', 'Renewal vaka skoru [0,1] dışında.', 500)
          }
        }
        rows.push({
          caseId: benchmarkCase.id,
          sourceGroupId: benchmarkCase.sourceGroupId,
          seed,
          baseScore: Math.fround(scores.base),
          candidateScore: Math.fround(scores.candidate),
          shamScore: Math.fround(scores.sham),
        })
        completed += 1
        if (completed % 8 === 0 || completed === total) {
          await context.checkpoint?.(`region-renewal:benchmark:${completed}`)
          await context.progress?.('region-renewal:benchmark', completed / total)
        }
      }
    }
    const sourceGroupIds = cases.map((item) => item.sourceGroupId).sort()
    const derived = derivePlasticRegionReleaseMetrics({ rows, sourceGroupIds })
    const evidence = createPlasticRegionRenewalEvidence({
      renewalId,
      representationSpaceId: atlas.representationSpaceId,
      geometryFingerprint: atlas.geometryFingerprint,
      runnerFingerprint: PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT,
      corpusFingerprint: plasticRegionReleaseCorpusFingerprint(caseCount),
      engineFingerprint,
      engineContractFingerprint: PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
      scorerFingerprint: PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
      pairingFingerprint,
      baseBinding,
      runtimePolicy,
      sourceGroupIds,
      seeds: PLASTIC_REGION_RELEASE_SEEDS,
      maximumTicks,
      candidateAtlasFingerprint: atlas.fingerprint,
      shamAtlasFingerprint: shamAtlas.fingerprint,
      rows,
      metrics: { holdout: derived.holdout, sham: derived.sham },
    })
    const reasonCodes = renewalReasons(policy, evidence.metrics)
    if (reasonCodes.length) {
      const audit = createPlasticRegionRenewalAudit({
        renewalId,
        representationSpaceId: atlas.representationSpaceId,
        decision: 'rejected',
        reasonCodes,
        baseSnapshotId: prepared.payload.base.snapshotId,
        sourceSnapshotId: sourceSnapshot.id,
        atlasFingerprint: atlas.fingerprint,
        evidence,
        releaseArtifactIds: [],
      })
      const auditArtifact = persistAudit(context, audit)
      return {
        renewed: false,
        released: false,
        committed: false,
        automaticActivation: false,
        renewalId,
        auditArtifactId: auditArtifact.id,
        evidenceFingerprint: evidence.fingerprint,
        reasonCodes,
        metrics: evidence.metrics,
      }
    }
    const evidenceArtifact = persistRegionArtifact(
      context,
      PLASTIC_REGION_RENEWAL_EVIDENCE_ARTIFACT_KIND,
      evidence,
      prepared.payload.base,
      { runnerFingerprint: PLASTIC_REGION_RENEWAL_RUNNER_FINGERPRINT, usesOllama: false },
    )
    const manifest = createPlasticRegionRenewalManifest({
      releaseId: `PRREL-RENEW-${stableRuntimeHash(renewalId, evidence.fingerprint).slice(0, 36).toUpperCase()}`,
      representationSpaceId: atlas.representationSpaceId,
      geometryFingerprint: atlas.geometryFingerprint,
      baseBinding,
      runtimePolicy,
      atlas: {
        artifactId: prepared.payload.base.atlasArtifactId,
        contentHash: prepared.payload.base.atlasContentHash,
        fingerprint: atlas.fingerprint,
      },
      evidence: artifactReference(evidenceArtifact, evidence),
    })
    const manifestArtifact = persistRegionArtifact(
      context,
      PLASTIC_REGION_RENEWAL_MANIFEST_ARTIFACT_KIND,
      manifest,
      prepared.payload.base,
      { activationMode: 'active', automaticActivation: false },
    )
    const releaseArtifactIds = [evidenceArtifact.id, manifestArtifact.id]
    const audit = createPlasticRegionRenewalAudit({
      renewalId,
      representationSpaceId: atlas.representationSpaceId,
      decision: 'approved-awaiting-manual-activation',
      reasonCodes: [],
      baseSnapshotId: prepared.payload.base.snapshotId,
      sourceSnapshotId: sourceSnapshot.id,
      atlasFingerprint: atlas.fingerprint,
      evidence: artifactReference(evidenceArtifact, evidence),
      releaseArtifactIds,
    })
    const auditArtifact = persistAudit(context, audit)
    if (typeof context.store.completeClaimedJobWithCandidateOnSnapshotHeadBase !== 'function' || !context.jobClaim) {
      throw runnerError('STORE_CAS_CONTRACT_MISSING', 'Renewal adayı claim-fenced snapshot CAS gerektirir.', 500)
    }
    const candidateId = `region-renewal-${stableRuntimeHash(prepared.payload, manifest.fingerprint).slice(0, 32)}`
    const candidateInput = {
      id: candidateId,
      parentId: prepared.payload.base.snapshotId,
      status: 'candidate',
      label: `${sourceSnapshot.label ?? 'Model adayı'} · değişmeyen atlas renewal`,
      metrics: sourceSnapshot.metrics,
      artifactIds: [...new Set([...(sourceSnapshot.artifactIds ?? []), ...releaseArtifactIds, auditArtifact.id])],
      sourceIds: [...(sourceSnapshot.sourceIds ?? [])],
      sourceRevisions: { ...(sourceSnapshot.sourceRevisions ?? {}) },
      inputManifestHash: sourceSnapshot.inputManifestHash,
      evaluation: {
        ...(sourceSnapshot.evaluation ?? {}),
        passed: true,
        runtimeVerified: true,
        plasticRegionReleasePassed: true,
        plasticRegionReleaseFingerprint: manifest.fingerprint,
        plasticRegionReleaseKind: manifest.releaseKind,
        plasticRegionReleaseAutomaticActivation: false,
      },
    }
    const expectedBase = {
      snapshotId: prepared.payload.base.snapshotId,
      headKind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
      headArtifactId: prepared.payload.base.atlasArtifactId,
      headContentHash: prepared.payload.base.atlasContentHash,
      headFingerprint: prepared.payload.base.atlasFingerprint,
      staleCode: 'STALE_REGION_ATLAS_BASE',
    }
    const finalResult = {
      renewed: true,
      released: true,
      committed: false,
      awaitingApproval: true,
      automaticActivation: false,
      renewalId,
      snapshotId: candidateId,
      releaseFingerprint: manifest.fingerprint,
      evidenceFingerprint: evidence.fingerprint,
      auditArtifactId: auditArtifact.id,
      evidenceArtifactId: evidenceArtifact.id,
      manifestArtifactId: manifestArtifact.id,
      releaseArtifactIds,
      metrics: evidence.metrics,
    }
    await context.checkpoint?.('region-renewal:candidate-cas')
    await context.progress?.('region-renewal:candidate-cas', 0.99)
    const publicationCommit = context.store.completeClaimedJobWithCandidateOnSnapshotHeadBase(
      context.jobId,
      context.jobClaim.owner,
      context.jobClaim.attempt,
      candidateInput,
      expectedBase,
      finalResult,
      (candidate) => {
        const gate = plasticRegionSnapshotActivationGate(context.store, candidate)
        return gate.required && gate.eligible && gate.release?.manifest?.fingerprint === manifest.fingerprint
          ? true
          : {
              eligible: false,
              code: 'PLASTIC_REGION_RENEWAL_CHAIN_INVALID',
              message: 'Renewal zinciri candidate üzerinde doğrulanamadı.',
              details: { reason: gate.reason },
            }
      },
    )
    if (publicationCommit.snapshot.id !== candidateId || publicationCommit.job.state !== 'succeeded') {
      throw runnerError('PLASTIC_REGION_RENEWAL_ATOMIC_PUBLICATION_FAILED', 'Renewal adayı atomik yayın doğrulamasını geçemedi.', 500)
    }
    return { ...finalResult, atomicCandidateJobCompletion: true }
  }
}
