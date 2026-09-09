import { MEBIBYTE, evaluateResourceState, sampleHostResources } from './resource-policy.mjs'

const STAGES = ['collect', 'quarantine', 'validate', 'encode', 'synthesize', 'families', 'language_experience', 'plastic_regions', 'concept_macros', 'evaluate', 'commit', 'rollback']

export function toPublicConfig(config) {
  return {
    ollamaBaseUrl: config.ollama.baseUrl,
    model: config.ollama.model,
    automationLevel: config.automation.level,
    autoRun: config.automation.autoRun,
    batchSize: config.automation.batchSize,
    conceptMacros: structuredClone(config.conceptMacros),
    plasticRegions: structuredClone(config.plasticRegions),
    budgets: {
      cpuPercent: config.budgets.cpuPercent,
      ramMb: config.budgets.ramMb,
      maxRunMinutes: config.budgets.maxRunMinutes,
      diskMb: config.budgets.diskMb,
    },
  }
}

export function fromPublicConfig(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Object.assign(new Error('Geçersiz eğitim yapılandırması.'), { code: 'INVALID_CONFIG' })
  if ('ollamaBaseUrl' in value || 'automationLevel' in value || 'batchSize' in value || 'autoRun' in value) {
    return {
      ollama: { baseUrl: value.ollamaBaseUrl, model: value.model },
      automation: { level: value.automationLevel, autoRun: value.autoRun, batchSize: value.batchSize },
      budgets: value.budgets,
      ...(value.conceptMacros !== undefined ? { conceptMacros: value.conceptMacros } : {}),
      ...(value.plasticRegions !== undefined ? { plasticRegions: value.plasticRegions } : {}),
    }
  }
  return value
}

function publicStage(stage) {
  const mapping = {
    starting: 'collect', queued: 'collect', recovered: 'collect',
    collect: 'collect', textLab: 'encode', synthesis: 'synthesize', families: 'families',
    languageExperience: 'language_experience', plasticRegions: 'plastic_regions',
    evaluate: 'evaluate', 'evaluation-gate': 'evaluate',
    conceptMacros: 'concept_macros', thoughtObservation: 'concept_macros',
    'candidate-snapshot': 'commit', commit: 'commit', completed: 'commit', failed: 'commit',
  }
  return mapping[stage] ?? (STAGES.includes(stage) ? stage : 'collect')
}

function runState(job, pipeline) {
  if (pipeline.paused || job?.state === 'paused' || job?.state === 'pause_requested') return 'paused'
  if (job?.state === 'queued') return 'starting'
  if (job?.state === 'running') return 'running'
  if (job?.state === 'cancel_requested') return 'cancelling'
  if (job?.state === 'failed') return 'failed'
  return 'idle'
}

function latestJob(jobs) {
  return [...jobs].sort((left, right) =>
    String(right.updatedAt ?? right.createdAt ?? '').localeCompare(String(left.updatedAt ?? left.createdAt ?? '')) ||
    String(right.id).localeCompare(String(left.id)),
  )[0]
}

function completedCommitState(store, job) {
  const snapshot = job?.result?.snapshotId ? store.getSnapshot(job.result.snapshotId) : null
  if (snapshot && ['active', 'superseded'].includes(snapshot.status)) {
    return { state: 'complete', progress: 1, message: 'Aday doğrulandı ve etkinleştirildi.' }
  }
  if (job?.result?.committed === true) {
    return { state: 'complete', progress: 1, message: 'Aday doğrulandı ve etkinleştirildi.' }
  }
  if (snapshot?.status === 'rejected') {
    return { state: 'skipped', progress: 0, message: snapshot.evaluation?.invalidated ? 'Adayın kaynak temsili değiştiği için yeniden eğitim gerekiyor.' : 'Aday etkinleştirme kapısından geçmedi.' }
  }
  if (snapshot?.status === 'candidate' || job?.result?.awaitingApproval === true) {
    return { state: 'pending', progress: 0, message: 'Doğrulanmış aday insan onayını bekliyor.' }
  }
  return { state: 'skipped', progress: 0, message: 'Aday etkinleştirilmedi.' }
}

function publicStageStatus({ id, index, job, activeStage, activeIndex, rollbackPerformed, store }) {
  if (id === 'rollback') {
    return rollbackPerformed
      ? { state: 'rolled_back', progress: 1, message: 'Doğrulanmış eski sürüme geri dönüldü.' }
      : { state: 'pending', progress: 0 }
  }
  if (!job) return { state: 'pending', progress: 0 }
  if (job.state === 'succeeded') {
    if (id === 'commit') return completedCommitState(store, job)
    return index <= STAGES.indexOf('evaluate')
      ? { state: 'complete', progress: 1 }
      : { state: 'pending', progress: 0 }
  }
  if (job.state === 'failed' && id === activeStage) {
    return { state: 'failed', progress: Math.max(0, Math.min(1, job.progress ?? 0)), ...(job.stage ? { message: job.stage } : {}) }
  }
  if (index < activeIndex) return { state: 'complete', progress: 1 }
  if (index === activeIndex && ['running', 'pause_requested', 'cancel_requested'].includes(job.state)) {
    return { state: 'active', progress: Math.max(0, Math.min(1, job.progress ?? 0)), ...(job.stage ? { message: job.stage } : {}) }
  }
  return { state: 'pending', progress: 0 }
}

export function toPublicStatus(store, ollamaStatus = {}) {
  const config = store.getConfig()
  const pipeline = store.getPipelineState()
  const jobs = store.listJobs()
  const activeJob = (pipeline.activeJobId && store.getJob(pipeline.activeJobId)) || jobs.find((item) => ['running', 'pause_requested', 'cancel_requested', 'queued', 'paused'].includes(item.state))
  // Keep the last finished cycle visible while idle; its commit state is
  // derived from the candidate snapshot rather than from generic job success.
  const job = activeJob || latestJob(jobs)
  const sources = store.listSources()
  const activeStage = job ? publicStage(job.stage) : null
  const activeIndex = activeStage ? STAGES.indexOf(activeStage) : -1
  const state = runState(activeJob, pipeline)
  const snapshots = store.listSnapshots()
  const activeSnapshot = store.getActiveSnapshot()
  const rollbackPerformed = !activeJob && activeSnapshot?.evaluation?.rollback === true
  const lastRejected = [...snapshots].sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt))).find((snapshot) => snapshot.status === 'rejected')
  const warnings = []
  let resources = null
  try {
    const sample = sampleHostResources({ dataDirectory: config.dataDirectory })
    const resourceState = evaluateResourceState(config, sample, {
      logicalStorageBytes: store.approximateStorageBytes(),
      admission: true,
    })
    resources = {
      memoryTier: resourceState.memoryTier,
      diskTier: resourceState.diskTier,
      processRssMb: Math.round(sample.rssBytes / MEBIBYTE),
      ramLimitMb: config.budgets.ramMb,
      hostFreeMb: Math.round(sample.hostFreeMemoryBytes / MEBIBYTE),
      diskFreeMb: Math.round(sample.diskFreeBytes / MEBIBYTE),
      usableDiskFreeMb: Math.round(resourceState.usableDiskFreeBytes / MEBIBYTE),
      logicalStorageMb: Math.round(store.approximateStorageBytes() / MEBIBYTE),
      logicalDiskLimitMb: config.budgets.diskMb,
      canStart: resourceState.canStart,
      shouldSpill: resourceState.shouldSpill,
      reasonCodes: resourceState.reasonCodes,
    }
    if (resourceState.diskTier === 'warning' || resourceState.diskTier === 'admission-blocked') {
      warnings.push({
        id: 'physical-disk-reserve',
        severity: resourceState.diskTier === 'admission-blocked' ? 'critical' : 'warning',
        title: 'SSD güvenli rezervi azalıyor',
        message: resourceState.diskTier === 'admission-blocked'
          ? 'Yeni eğitim bloğu başlatılmayacak; mevcut checkpoint ve kurtarma yazıları için rezerv korunuyor.'
          : 'Tamamlanan veriler sıkıştırılıyor; SSD boşluğu yeni blok eşiğine yaklaşıyor.',
        createdAt: new Date().toISOString(),
        code: resourceState.reasonCodes[0] ?? 'PHYSICAL_DISK_WARNING_RESERVE_LOW',
      })
    }
  } catch (error) {
    warnings.push({ id: 'resource-sample-failed', severity: 'warning', title: 'Kaynak ölçümü alınamadı', message: String(error?.message ?? error).slice(0, 500), createdAt: new Date().toISOString(), code: 'RESOURCE_SAMPLE_FAILED' })
  }
  if (!ollamaStatus.available) warnings.push({ id: 'ollama-offline', severity: 'warning', title: 'Ollama bağlantısı yok', message: ollamaStatus.error || 'Yerel öğretmen modeline ulaşılamıyor.', createdAt: new Date().toISOString(), code: 'OLLAMA_OFFLINE' })
  if (store.backendKind === 'atomic-json') warnings.push({ id: 'json-storage-fallback', severity: 'warning', title: 'Sınırlı yedek depolama', message: 'node:sqlite bulunamadı; atomik JSON yedeği 64 MiB ile sınırlıdır. Büyük korpus için SQLite destekli Node.js kullanın.', createdAt: new Date().toISOString(), code: 'JSON_STORAGE_FALLBACK' })
  if (lastRejected?.evaluation?.runtimeVerified === false) {
    const artifactKinds = new Set((lastRejected.artifactIds ?? []).map((id) => store.getArtifactMetadata(id)?.kind).filter(Boolean))
    const realRuntimeRan = artifactKinds.has('runtime-model-state')
    warnings.push(realRuntimeRan
      ? { id: `runtime-${lastRejected.id}`, severity: 'info', title: 'Etkinleştirme kanıtı yetersiz', message: 'Gerçek öğrenme motorları çalıştı; ancak bağımsız gerçek kaynak, sentetik oranı veya kalite kapılarından en az biri karşılanmadığı için aday etkinleştirilmedi.', createdAt: lastRejected.rejectedAt ?? lastRejected.createdAt, code: 'RUNTIME_EVIDENCE_INSUFFICIENT' }
      : { id: `runtime-${lastRejected.id}`, severity: 'info', title: 'Gerçek motor entegrasyonu bekleniyor', message: 'Son çevrim kalıcı modele yazılmadı; bir veya daha fazla aşama yalnız entegrasyon çıktısı üretti.', createdAt: lastRejected.rejectedAt ?? lastRejected.createdAt, code: 'RUNTIME_INTEGRATION_REQUIRED' })
  }
  return {
    serviceVersion: '1.0.0',
    state,
    runId: job?.id ?? null,
    startedAt: job?.startedAt ?? null,
    updatedAt: pipeline.updatedAt,
    activeStage,
    progress: job?.progress ?? 0,
    queuedItems: jobs.filter((item) => ['queued', 'paused'].includes(item.state)).length,
    processedItems: sources.filter((source) => source.processedSnapshotId).length,
    acceptedItems: sources.filter((source) => source.status === 'approved').length,
    rejectedItems: sources.filter((source) => source.status === 'rejected').length,
    activeSnapshotId: activeSnapshot?.id ?? null,
    ...(job?.error?.message ? { lastError: job.error.message } : {}),
    ollama: {
      connected: Boolean(ollamaStatus.available),
      endpoint: config.ollama.baseUrl,
      checkedAt: new Date().toISOString(),
      models: (ollamaStatus.models ?? []).map(toPublicModel),
      ...(ollamaStatus.version ? { version: ollamaStatus.version } : {}),
      ...(!ollamaStatus.available && ollamaStatus.error ? { message: ollamaStatus.error } : {}),
    },
    config: toPublicConfig(config),
    resources,
    stages: STAGES.map((id, index) => ({
      id,
      ...publicStageStatus({ id, index, job, activeStage, activeIndex, rollbackPerformed, store }),
    })),
    warnings,
  }
}

function sourceKind(source) {
  if (source.sourceType === 'ollama-generated') return 'ollama_generated'
  if (['local-file', 'local-upload'].includes(source.sourceType)) return 'file'
  return 'pasted_text'
}
function sourceState(source) {
  if (source.status === 'quarantine') return 'quarantined'
  if (source.status === 'approved') return 'ready'
  if (source.status === 'rejected') return 'disabled'
  return 'failed'
}
function provenanceLabel(source) {
  const provenance = source.provenance ?? {}
  if (typeof provenance.label === 'string') return provenance.label
  if (provenance.kind === 'ollama-generated') return `Ollama · ${provenance.model ?? 'yerel model'}`
  if (provenance.kind === 'local-file') return provenance.canonicalPath ?? 'İzinli yerel dosya'
  if (provenance.kind === 'local-upload') return provenance.filename ?? 'Tarayıcıdan seçilen dosya'
  return provenance.kind === 'manual' ? 'Kullanıcı tarafından sağlandı' : String(provenance.kind ?? 'Kaynak geçmişi belirtilmedi')
}

export function toPublicSource(source) {
  const contentSummary = source.contentSummary
  return {
    id: source.id,
    name: source.title,
    kind: sourceKind(source),
    state: sourceState(source),
    createdAt: source.createdAt,
    updatedAt: source.updatedAt,
    bytes: source.bytes,
    itemCount: Number.isSafeInteger(contentSummary?.itemCount)
      ? contentSummary.itemCount
      : source.content ? source.content.split(/(?<=[.!?])\s+/u).filter(Boolean).length : 1,
    selected: source.status === 'approved' && source.selected !== false,
    provenance: provenanceLabel(source),
    contentFingerprint: source.contentHash,
    ...(source.decisionReason && source.status === 'rejected' ? { error: source.decisionReason } : {}),
  }
}

export function toPublicQuarantine(source) {
  const provenance = provenanceLabel(source)
  const contentSummary = source.contentSummary
  return {
    id: source.id,
    sourceId: source.id,
    sourceName: source.title,
    createdAt: source.createdAt,
    preview: typeof contentSummary?.preview === 'string'
      ? contentSummary.preview
      : String(source.content ?? '').replace(/\s+/gu, ' ').slice(0, 220),
    language: source.language,
    characterCount: Number.isSafeInteger(contentSummary?.characterCount)
      ? contentSummary.characterCount
      : String(source.content ?? '').length,
    duplicateScore: 0,
    state: source.status === 'quarantine' ? 'pending' : source.status,
    checks: [
      { id: 'non-empty', label: 'İçerik', state: source.bytes > 0 ? 'pass' : 'fail', message: source.bytes > 0 ? 'İçerik boş değil.' : 'İçerik boş.' },
      { id: 'dedupe', label: 'Tekrar', state: 'pass', message: 'İçerik parmak izi benzersiz.' },
      { id: 'provenance', label: 'Kaynak geçmişi', state: provenance ? 'pass' : 'warning', message: provenance || 'Kaynak geçmişi eksik.' },
    ],
    provenance,
  }
}

export function toPublicQueueItem(job) {
  const state = job.state === 'queued' || job.state === 'paused' ? 'waiting' : job.state === 'running' || job.state === 'pause_requested' || job.state === 'cancel_requested' ? 'active' : job.state === 'succeeded' ? 'complete' : job.state === 'failed' ? 'failed' : 'cancelled'
  const regionReleaseJob = job.type === 'plastic-region-release-evaluation' || job.type === 'plastic-region-release-renewal'
  const releaseResult = regionReleaseJob && job.result && typeof job.result === 'object'
    ? {
        released: job.result.released === true,
        renewed: job.type === 'plastic-region-release-renewal' ? job.result.renewed === true : null,
        awaitingApproval: job.result.awaitingApproval === true,
        automaticActivation: false,
        evaluationId: typeof job.result.evaluationId === 'string' ? job.result.evaluationId.slice(0, 256) : null,
        renewalId: typeof job.result.renewalId === 'string' ? job.result.renewalId.slice(0, 256) : null,
        releaseKind: job.type === 'plastic-region-release-renewal' ? 'unchanged-atlas-model-renewal' : 'lifecycle-proposal',
        snapshotId: typeof job.result.snapshotId === 'string' ? job.result.snapshotId.slice(0, 256) : null,
        releaseFingerprint: typeof job.result.releaseFingerprint === 'string' ? job.result.releaseFingerprint.slice(0, 256) : null,
        evidenceFingerprint: typeof job.result.evidenceFingerprint === 'string' ? job.result.evidenceFingerprint.slice(0, 256) : null,
        auditArtifactId: typeof job.result.auditArtifactId === 'string' ? job.result.auditArtifactId.slice(0, 256) : null,
        evidenceArtifactId: typeof job.result.evidenceArtifactId === 'string' ? job.result.evidenceArtifactId.slice(0, 256) : null,
        manifestArtifactId: typeof job.result.manifestArtifactId === 'string' ? job.result.manifestArtifactId.slice(0, 256) : null,
        benchmarkReportArtifactId: typeof job.result.benchmarkReportArtifactId === 'string' ? job.result.benchmarkReportArtifactId.slice(0, 256) : null,
        reasonCodes: Array.isArray(job.result.reasonCodes)
          ? [...new Set(job.result.reasonCodes.filter((value) => typeof value === 'string' && value).map((value) => value.slice(0, 160)))].slice(0, 32)
          : [],
        metrics: (() => {
          const source = job.result.metrics && typeof job.result.metrics === 'object' ? job.result.metrics : {}
          const holdout = source.holdout && typeof source.holdout === 'object' ? source.holdout : {}
          const sham = source.sham && typeof source.sham === 'object' ? source.sham : {}
          const finite = (value) => typeof value === 'number' && Number.isFinite(value) ? value : null
          return {
            baseMean: finite(source.baseMean),
            candidateMean: finite(source.candidateMean),
            shamMean: finite(source.shamMean),
            conservativeHoldoutGain: finite(source.conservativeHoldoutGain),
            conservativeShamGain: finite(source.conservativeShamGain),
            holdoutRegressionHigh: finite(source.holdoutRegressionHigh),
            holdoutCaseCount: Number.isSafeInteger(holdout.caseCount) ? holdout.caseCount : null,
            holdoutSourceGroupCount: Number.isSafeInteger(holdout.independentSourceGroupCount)
              ? holdout.independentSourceGroupCount
              : null,
            holdoutObservedGain: finite(holdout.observedGain),
            holdoutRegressionLoss: finite(holdout.regressionLoss),
            shamObservedGain: finite(sham.observedGain),
            shamRegressionLoss: finite(sham.regressionLoss),
          }
        })(),
      }
    : null
  return {
    id: job.id,
    runId: job.id,
    stage: publicStage(job.stage),
    state,
    label: job.type === 'full-cycle'
      ? 'Tam eğitim çevrimi'
      : job.type === 'teacher-feed-campaign'
        ? '14B araştırma besleme kampanyası'
        : job.type,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    attempts: job.attempt,
    progress: job.progress,
    ...(job.error?.message ? { message: job.error.message } : {}),
    ...(releaseResult ? { releaseResult } : {}),
  }
}

export function toPublicTeacherFeedCampaign(job) {
  const campaign = job?.payload?.campaign ?? {}
  const state = job?.payload?.feedState ?? {}
  const totals = state.totals ?? {}
  const legacyTarget = Number(campaign.batchCount ?? 0) * Number(campaign.batchSize ?? 0)
  const status = job?.state === 'paused'
    ? 'paused'
    : job?.state === 'pause_requested'
      ? 'pausing'
      : job?.state === 'cancel_requested'
        ? 'cancelling'
        : job?.state === 'cancelled'
          ? 'cancelled'
          : job?.state === 'failed'
            ? 'failed'
            : job?.state === 'succeeded'
              ? 'completed'
              : job?.state === 'queued'
                ? 'queued'
                : 'running'
  const continuous = campaign.continuousPlan && state.continuous ? {
    blockAccepted: Number(campaign.continuousPlan.blockAccepted ?? 0),
    maximumBlocks: Number(campaign.continuousPlan.maxBlocks ?? 0),
    completedBlocks: Number(state.continuous.completedBlocks ?? 0),
    nextBlockAt: Number(state.continuous.nextBlockAt ?? 0),
    minimumDiskHeadroomMb: Number(campaign.continuousPlan.minimumDiskHeadroomBytes ?? 0) / (1024 * 1024),
    maximumSyntheticSources: Number(campaign.continuousPlan.maximumSyntheticSources ?? 0),
    bootstrapSourceCount: Array.isArray(state.bootstrapSourceIds) ? state.bootstrapSourceIds.length : 0,
    bootstrapSourceFingerprint: Array.isArray(state.bootstrapSourceIds) && typeof state.bootstrapSourceFingerprint === 'string'
      ? state.bootstrapSourceFingerprint
      : null,
    stopGate: {
      required: true,
      passed: state.continuous.gatePassed === true,
      status: state.continuous.lastGate?.status ?? 'waiting',
      valid: state.continuous.lastGate?.valid === true,
      conclusive: state.continuous.lastGate?.conclusive === true,
      completion: typeof state.continuous.lastGate?.completion === 'string' ? state.continuous.lastGate.completion : null,
      evaluationKind: typeof state.continuous.lastGate?.evaluationKind === 'string' ? state.continuous.lastGate.evaluationKind : null,
      diagnosticMeanGraphF1: Number.isFinite(state.continuous.lastGate?.diagnosticMeanGraphF1) ? state.continuous.lastGate.diagnosticMeanGraphF1 : null,
      holdoutGraphF1: Number.isFinite(state.continuous.lastGate?.holdoutGraphF1) ? state.continuous.lastGate.holdoutGraphF1 : null,
      holdoutGraphF1Low: Number.isFinite(state.continuous.lastGate?.holdoutGraphF1Low) ? state.continuous.lastGate.holdoutGraphF1Low : null,
      holdoutGraphF1High: Number.isFinite(state.continuous.lastGate?.holdoutGraphF1High) ? state.continuous.lastGate.holdoutGraphF1High : null,
      snapshotId: state.continuous.gateSnapshotId ?? state.continuous.lastGate?.snapshotId ?? null,
      benchmarkFingerprint: campaign.continuousPlan.stopGate?.benchmarkFingerprint ?? null,
      failedChecks: Array.isArray(state.continuous.lastGate?.checks)
        ? state.continuous.lastGate.checks.filter((check) => check?.passed !== true).map((check) => String(check?.id ?? '')).filter(Boolean).slice(0, 40)
        : [],
    },
  } : null
  return {
    id: job.id,
    status,
    model: campaign.model ?? campaign.modelIdentity?.name ?? '',
    seed: Number(campaign.seed ?? 0),
    targetAccepted: Number(campaign.targetAccepted ?? legacyTarget),
    generationBatchSize: Number(campaign.generationBatchSize ?? campaign.batchSize ?? 0),
    trainEveryAccepted: Number(campaign.trainEveryAccepted ?? campaign.batchSize ?? 0),
    curriculum: campaign.curriculum ?? campaign.curriculumId ?? 'turkish-foundations-v1',
    mode: 'research-only',
    generationMode: campaign.generationMode ?? 'controlled',
    lanes: state.lanes ? {
      controlled: {
        generated: Number(state.lanes.controlled?.generated ?? 0),
        accepted: Number(state.lanes.controlled?.accepted ?? 0),
        duplicates: Number(state.lanes.controlled?.duplicates ?? 0),
        rejectedQuality: Number(state.lanes.controlled?.rejectedQuality ?? 0),
        teacherRequests: Number(state.lanes.controlled?.teacherRequests ?? 0),
        teacherTokens: Number(state.lanes.controlled?.teacherTokens ?? 0),
      },
      discovery: {
        generated: Number(state.lanes.discovery?.generated ?? 0),
        accepted: Number(state.lanes.discovery?.accepted ?? 0),
        duplicates: Number(state.lanes.discovery?.duplicates ?? 0),
        rejectedQuality: Number(state.lanes.discovery?.rejectedQuality ?? 0),
        teacherRequests: Number(state.lanes.discovery?.teacherRequests ?? 0),
        teacherTokens: Number(state.lanes.discovery?.teacherTokens ?? 0),
      },
    } : null,
    generated: Number(state.generated ?? totals.candidates ?? 0),
    accepted: Number(state.accepted ?? totals.inserted ?? 0),
    duplicates: Number(state.duplicates ?? totals.duplicates ?? 0),
    rejectedQuality: Number(state.rejectedQuality ?? totals.rejected ?? 0),
    teacherRequests: Number(state.teacherRequests ?? totals.generationCalls ?? 0),
    teacherTokens: Number(state.teacherTokens ?? (Number(totals.promptTokens ?? 0) + Number(totals.completionTokens ?? 0))),
    completedTrainingCycles: Number(state.completedTrainingCycles ?? state.pipelineCycles ?? 0),
    nextBatchIndex: Number(state.nextBatchIndex ?? 0),
    lastSnapshotId: state.lastSnapshotId ?? null,
    stopReason: typeof state.stopReason === 'string'
      ? state.stopReason
      : state.stopReason?.message ?? job?.error?.message ?? null,
    continuous,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
  }
}

const metricValue = (current, previous, unit = 'ratio') => ({ current: finiteOrNull(current), previous: finiteOrNull(previous), unit })
const finiteOrNull = (value) => typeof value === 'number' && Number.isFinite(value) ? value : null

function mappedMetrics(metrics, previous = {}) {
  return {
    holdoutTop1: metricValue(metrics?.holdoutTop1 ?? metrics?.overallScore, previous?.holdoutTop1 ?? previous?.overallScore),
    holdoutTop5: metricValue(metrics?.holdoutTop5, previous?.holdoutTop5),
    unseenCombinationAccuracy: metricValue(metrics?.unseenCombinationAccuracy ?? metrics?.unseenAccuracy, previous?.unseenCombinationAccuracy ?? previous?.unseenAccuracy),
    grammaticalPreference: metricValue(metrics?.grammaticalPreference ?? metrics?.syntaxContrastAccuracy, previous?.grammaticalPreference ?? previous?.syntaxContrastAccuracy),
    familyUtility: metricValue(metrics?.familyUtility, previous?.familyUtility),
    neuronCount: metricValue(metrics?.neuronCount, previous?.neuronCount, 'count'),
    familyCount: metricValue(metrics?.familyCount, previous?.familyCount, 'count'),
    birthProposalCount: metricValue(metrics?.birthProposalCount, previous?.birthProposalCount, 'count'),
    eligibleBirthProposalCount: metricValue(metrics?.eligibleBirthProposalCount, previous?.eligibleBirthProposalCount, 'count'),
    emergentLanguageFamilyCount: metricValue(metrics?.emergentLanguageFamilyCount, previous?.emergentLanguageFamilyCount, 'count'),
    eligibleEmergentLanguageFamilyCount: metricValue(metrics?.eligibleEmergentLanguageFamilyCount, previous?.eligibleEmergentLanguageFamilyCount, 'count'),
    conceptMacroCount: metricValue(metrics?.conceptMacroCount, previous?.conceptMacroCount, 'count'),
    conceptMacroProposalCount: metricValue(metrics?.conceptMacroProposalCount, previous?.conceptMacroProposalCount, 'count'),
    conceptMacroVirtualCount: metricValue(metrics?.conceptMacroVirtualCount, previous?.conceptMacroVirtualCount, 'count'),
    conceptMacroShadowValidatedCount: metricValue(metrics?.conceptMacroShadowValidatedCount, previous?.conceptMacroShadowValidatedCount, 'count'),
    conceptMacroActiveCount: metricValue(metrics?.conceptMacroActiveCount, previous?.conceptMacroActiveCount, 'count'),
  }
}

export function toPublicMetrics(store) {
  const snapshots = store.listSnapshots().sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
  const current = snapshots.find((snapshot) => snapshot.metrics)
  if (!current) return null
  const previous = snapshots.find((snapshot) => snapshot.id !== current.id && snapshot.metrics)
  const mapped = mappedMetrics(current.metrics, previous?.metrics)
  const committed = snapshots.filter((snapshot) => ['active', 'superseded'].includes(snapshot.status)).length
  const rollbackCount = snapshots.filter((snapshot) => snapshot.rollbackOf).length
  return {
    measuredAt: current.evaluation?.evaluatedAt ?? current.createdAt,
    ...mapped,
    committedBatches: metricValue(committed, Math.max(0, committed - 1), 'count'),
    rollbackRate: metricValue(snapshots.length ? rollbackCount / snapshots.length : 0, null),
    evaluationSampleCount: Number(current.metrics?.evaluationSampleCount ?? current.sourceIds?.length ?? 0),
    baselineSnapshotId: current.parentId,
  }
}

export function toPublicSnapshot(snapshot, store) {
  const previous = snapshot.parentId ? store.getSnapshot(snapshot.parentId) : null
  const metrics = mappedMetrics(snapshot.metrics, previous?.metrics)
  const size = (snapshot.artifactIds ?? []).reduce((sum, id) => {
    const artifact = store.getArtifactMetadata(id)
    return sum + (artifact?.payloadBytes ?? 0)
  }, 0)
  return {
    id: snapshot.id,
    name: snapshot.label,
    createdAt: snapshot.createdAt,
    state: snapshot.status === 'active' || snapshot.status === 'superseded' ? 'committed' : snapshot.status === 'rejected' ? 'rejected' : snapshot.status === 'candidate' ? 'candidate' : 'rolled_back',
    runId: null,
    parentId: snapshot.parentId,
    bytes: size,
    corpusFingerprint: snapshot.metrics?.corpusFingerprint ?? snapshot.inputManifestHash ?? 'yok',
    metrics: {
      holdoutTop1: metrics.holdoutTop1,
      unseenCombinationAccuracy: metrics.unseenCombinationAccuracy,
      grammaticalPreference: metrics.grammaticalPreference,
      familyUtility: metrics.familyUtility,
    },
    note: snapshot.evaluation?.reason,
    rollbackToken: `ROLLBACK:${snapshot.id}`,
  }
}

export function toPublicModel(model) {
  return {
    name: model.name,
    ...(model.digest ? { digest: model.digest } : {}),
    parameterSize: model.details?.parameter_size,
    quantization: model.details?.quantization_level,
    sizeBytes: model.size,
    modifiedAt: model.modifiedAt,
  }
}

export function toPublicEvent(event) {
  const type = String(event.type ?? 'event')
  const messages = {
    'job-started': 'Eğitim işi başladı.', 'job-progress': 'Eğitim aşaması ilerliyor.', 'job-succeeded': 'Eğitim işi tamamlandı.',
    'job-failed': 'Eğitim işi başarısız oldu.', 'job-queued': 'Yeni eğitim işi kuyruğa alındı.', 'job-paused': 'Eğitim işi duraklatıldı.',
    'job-cancelled': 'Eğitim işi iptal edildi.', 'sources-quarantined': 'Yeni kaynaklar karantinaya alındı.',
    'source-approved': 'Kaynak onaylandı.', 'source-rejected': 'Kaynak reddedildi.', 'teacher-sources-quarantined': 'Öğretmen örnekleri karantinaya alındı.',
    'snapshot-activated': 'Aday model kalite kapısını geçti ve etkinleştirildi.', 'candidate-quarantined': 'Aday model kalite kapısında reddedildi.',
    'candidate-awaiting-approval': 'Aday model kalite kapısını geçti ve insan onayını bekliyor.',
    'snapshot-rolled-back': 'Model önceki sağlam sürüme geri alındı.', 'config-updated': 'Eğitim ayarları güncellendi.',
    'thought-episode-started': 'Düşünce bölümü başladı.', 'thought-episode-completed': 'Düşünce bölümü tamamlandı.',
    'thought-episode-paused': 'Düşünce bölümü güvenli kontrol noktasında duraklatıldı.',
    'thought-episode-cancelled': 'Düşünce bölümü iptal edildi.', 'thought-episode-failed': 'Düşünce bölümü başarısız oldu.',
    'thought-head-rejected': 'Düşünce öğrenme önerisi reddedildi.',
    'thought-head-candidate-created': 'Düşünce öğrenme önerisi bağımsız değerlendirme için adaylaştırıldı.',
    'teacher-feed-campaign-created': '14B araştırma besleme kampanyası kuyruğa alındı.',
    'teacher-feed-batch-completed': '14B cümle partisi kalite kapısından ve araştırma eğitiminden geçti.',
    'teacher-feed-structural-gate-passed': 'Bağımsız deterministik yapısal benchmark bütün durdurma kapılarını geçti.',
    'teacher-feed-structural-gate-not-passed': 'Yapısal benchmark kapısı geçilmedi; güvenli eğitim bir sonraki sınırlı bloğa ilerliyor.',
    'teacher-feed-model-unloaded': '14B öğretmen modeli ağır deterministik aşamadan önce bellekten çıkarıldı.',
    'teacher-feed-model-unload-warning': 'Öğretmen modeli bellekten çıkarılamadı; güvenli pipeline denetimleri devam ediyor.',
    'teacher-feed-campaign-pause-requested': '14B besleme kampanyası için duraklatma istendi.',
    'teacher-feed-campaign-resume-requested': '14B besleme kampanyası checkpoint’ten sürdürülüyor.',
    'teacher-feed-campaign-cancel-requested': '14B besleme kampanyası için iptal istendi.',
  }
  return {
    id: String(event.id),
    type,
    at: event.at,
    level: event.level === 'info' && type.includes('succeeded') ? 'success' : event.level,
    ...(event.stage ? { stage: publicStage(event.stage) } : {}),
    message: messages[type] ?? type.replaceAll('-', ' '),
    details: event.data ?? {},
  }
}
