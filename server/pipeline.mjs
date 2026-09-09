import { createHash } from 'node:crypto'
import { teacherLessonSchema } from './schema.mjs'
import { normalizeTextSource, sha256 } from './source-policy.mjs'
import { sourceRepresentationRevision } from './store.mjs'
import { reserveBoundedTeacherCall, settleBoundedTeacherCall } from './teacher-usage.mjs'
import { canonicalOllamaIndependenceGroupId } from './ollama-lineage.mjs'
import {
  assertPlasticRegionBase,
  plasticRegionCarryForwardArtifactIds,
  plasticRegionSnapshotActivationGate,
  readPlasticRegionSnapshot,
} from '../runtime/plastic-region-cache.mjs'

const stableHash = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex')

export function evaluateCandidate(metrics, baseline, policy) {
  const checks = [
    { name: 'overall-score', passed: metrics.overallScore >= policy.minOverallScore, actual: metrics.overallScore, expected: `>= ${policy.minOverallScore}` },
    { name: 'syntax-contrast', passed: metrics.syntaxContrastAccuracy >= policy.minSyntaxContrastAccuracy, actual: metrics.syntaxContrastAccuracy, expected: `>= ${policy.minSyntaxContrastAccuracy}` },
    { name: 'regression-rate', passed: metrics.regressionRate <= policy.maxRegressionRate, actual: metrics.regressionRate, expected: `<= ${policy.maxRegressionRate}` },
    { name: 'memorization-gap', passed: metrics.memorizationGap <= policy.maxMemorizationGap, actual: metrics.memorizationGap, expected: `<= ${policy.maxMemorizationGap}` },
  ]
  if (baseline?.overallScore !== undefined) {
    checks.push({
      name: 'active-baseline',
      passed: metrics.overallScore >= baseline.overallScore - policy.baselineTolerance,
      actual: metrics.overallScore,
      expected: `>= ${(baseline.overallScore - policy.baselineTolerance).toFixed(4)}`,
    })
  }
  return { passed: checks.every((check) => check.passed), checks, metrics, baseline: baseline ?? null, evaluatedAt: new Date().toISOString() }
}

function partitionSources(sources) {
  const partitions = { train: [], validation: [], test: [] }
  const synthetic = []
  for (const source of sources) {
    if (source.sourceType === 'ollama-generated') {
      synthetic.push(source.id)
      continue
    }
    const bucket = Number.parseInt(source.contentHash.slice(0, 8), 16) % 10
    partitions[bucket === 0 ? 'test' : bucket === 1 ? 'validation' : 'train'].push(source.id)
  }
  // Tiny batches still keep at least one real document holdout whenever possible.
  // Synthetic teacher material is appended only after the fallback moves so it
  // can never become validation/test evidence in this preliminary manifest either.
  if (!partitions.test.length && partitions.train.length >= 3) partitions.test.push(partitions.train.pop())
  if (!partitions.validation.length && partitions.train.length >= 2) partitions.validation.push(partitions.train.pop())
  partitions.train.push(...synthetic)
  return partitions
}

function estimateExperiences(sources) {
  let sentences = 0
  let pieces = 0
  for (const source of sources) {
    const chunks = source.content.split(/(?<=[.!?])\s+/u).filter(Boolean)
    sentences += chunks.length
    pieces += chunks.reduce((total, sentence) => total + Math.max(0, sentence.trim().split(/\s+/u).length - 1), 0)
  }
  return { sentences, nextPieceExperiences: pieces }
}

function createPipelineArtifact(context, input) {
  if (context.jobClaim) {
    if (typeof context.store.createClaimedArtifact !== 'function') {
      const error = new Error('Claim korumalı artifact store primitive’i eksik.')
      error.code = 'STORE_CLAIM_CONTRACT_MISSING'
      throw error
    }
    return context.store.createClaimedArtifact(
      context.jobId,
      context.jobClaim.owner,
      context.jobClaim.attempt,
      input,
    )
  }
  return context.store.createArtifact(input)
}

export const DEFAULT_STAGE_HANDLERS = {
  async collect(job, context, state) {
    const config = context.config ?? context.store.getConfig()
    const frozenIds = Array.isArray(job.payload.sourceIds)
      ? job.payload.sourceIds
      : Array.isArray(job.payload.resolvedSourceIds)
        ? job.payload.resolvedSourceIds
        : null
    const requestedIds = frozenIds ? new Set(frozenIds) : null
    let approved = context.store.listSources({ status: 'approved', includeContent: true })
      .filter((source) => (
        source.status === 'approved' &&
        source.selected === true &&
        (!requestedIds || requestedIds.has(source.id))
      ))
    if (!requestedIds) {
      approved = approved
        .filter((source) => job.payload.reprocess === true || !source.processedSnapshotId)
        .slice(0, Number(job.payload.batchSize ?? config.automation.batchSize))
    }
    const bytes = approved.reduce((sum, source) => sum + source.bytes, 0)
    if (!approved.length) {
      const error = new Error('İşlenecek onaylı ve yeni kaynak bulunamadı.')
      error.code = 'NO_APPROVED_SOURCES'
      throw error
    }
    if (bytes > config.budgets.maxBatchBytes) {
      const error = new Error('Seçilen kaynak partisi byte bütçesini aşıyor.')
      error.code = 'BATCH_TOO_LARGE'
      throw error
    }
    state.sources = approved
    state.sourceRevisions = Object.fromEntries(approved.map((source) => [source.id, sourceRepresentationRevision(source)]))
    const resolvedCorpus = {
      resolvedSourceIds: approved.map((source) => source.id),
      resolvedSourceRevisions: state.sourceRevisions,
    }
    if (context.jobClaim) {
      if (typeof context.store.updateClaimedJobPayload !== 'function') {
        const error = new Error('Claim korumalı iş payload store primitive’i eksik.')
        error.code = 'STORE_CLAIM_CONTRACT_MISSING'
        throw error
      }
      context.store.updateClaimedJobPayload(
        context.jobId,
        context.jobClaim.owner,
        context.jobClaim.attempt,
        resolvedCorpus,
      )
    } else {
      const persistedJob = context.store.getJob?.(context.jobId)
      if (persistedJob) {
        context.store.updateJob(context.jobId, {
          payload: { ...(persistedJob.payload ?? {}), ...resolvedCorpus },
        })
      }
    }
    state.partitions = partitionSources(approved)
    state.inputManifestHash = stableHash(approved.map(({ id, contentHash, bytes, provenance }) => ({
      id,
      contentHash,
      bytes,
      // A TextLab proof can select a different matrix/segmentation space for
      // byte-identical text, so it is part of the immutable run input.
      textLabManifest: provenance?.textLabManifest ?? null,
    })))
    return { sourceCount: approved.length, bytes, partitions: state.partitions, inputManifestHash: state.inputManifestHash, verified: true }
  },

  async textLab(_job, context, state) {
    const artifact = createPipelineArtifact(context, {
      jobId: context.jobId, kind: 'text-lab-input-manifest', contentHash: state.inputManifestHash,
      payload: { sourceIds: state.sources.map((source) => source.id), partitions: state.partitions },
    })
    state.artifactIds.push(artifact.id)
    return { artifactId: artifact.id, verified: false, integrationRequired: 'text-lab' }
  },

  async synthesis(_job, context, state) {
    const artifact = createPipelineArtifact(context, { jobId: context.jobId, kind: 'synthesis-request', payload: { inputManifestHash: state.inputManifestHash, upstreamArtifactIds: [...state.artifactIds] } })
    state.artifactIds.push(artifact.id)
    return { artifactId: artifact.id, verified: false, integrationRequired: 'neuron-synthesis' }
  },

  async families(_job, context, state) {
    const artifact = createPipelineArtifact(context, { jobId: context.jobId, kind: 'family-rebuild-request', payload: { inputManifestHash: state.inputManifestHash, upstreamArtifactIds: [...state.artifactIds] } })
    state.artifactIds.push(artifact.id)
    return { artifactId: artifact.id, verified: false, integrationRequired: 'neuron-families' }
  },

  async languageExperience(_job, context, state) {
    const counts = estimateExperiences(state.sources)
    const artifact = createPipelineArtifact(context, {
      jobId: context.jobId, kind: 'self-supervised-language-experiences',
      payload: { partitions: state.partitions, ...counts, target: 'next-piece-prediction', sourceIds: state.sources.map((source) => source.id) },
    })
    state.artifactIds.push(artifact.id)
    return { artifactId: artifact.id, ...counts, verified: false, integrationRequired: 'language-experience-engine' }
  },

  async plasticRegions(_job, context, state) {
    const artifact = createPipelineArtifact(context, {
      jobId: context.jobId,
      kind: 'plastic-region-integration-request',
      payload: {
        inputManifestHash: state.inputManifestHash,
        upstreamArtifactIds: [...state.artifactIds],
        safetyMode: 'observation-only',
      },
    })
    state.artifactIds.push(artifact.id)
    return { artifactId: artifact.id, verified: false, integrationRequired: 'plastic-region-pipeline' }
  },

  async conceptMacros(_job, context, state) {
    const artifact = createPipelineArtifact(context, {
      jobId: context.jobId,
      kind: 'concept-macro-observation-request',
      payload: {
        inputManifestHash: state.inputManifestHash,
        upstreamArtifactIds: [...state.artifactIds],
        safetyMode: 'proposal-shadow',
      },
    })
    state.artifactIds.push(artifact.id)
    return {
      artifactId: artifact.id,
      verified: false,
      integrationRequired: 'concept-macro-engine',
    }
  },

  async evaluate(job, _context, state) {
    const supplied = job.payload.evaluationMetrics
    if (supplied && typeof supplied === 'object') {
      return { metrics: normalizeMetrics(supplied), verified: job.payload.allowExternalEvaluation === true }
    }
    const experiences = estimateExperiences(state.sources)
    const coverage = Math.min(0.45, Math.log10(experiences.nextPieceExperiences + 10) / 10)
    return {
      metrics: normalizeMetrics({ overallScore: 0.2 + coverage, syntaxContrastAccuracy: 0.35 + coverage / 2, regressionRate: 0, memorizationGap: 0.6 }),
      verified: false,
      note: 'Bu ölçümler yalnız arayüz akışını sınamak içindir; model kabulü yapamaz.',
    }
  },
}

export const BASE_PIPELINE_STAGES = Object.freeze([
  'collect',
  'textLab',
  'synthesis',
  'families',
  'languageExperience',
  'conceptMacros',
  'evaluate',
])

export function resolvePipelineStages(config) {
  const stages = [...BASE_PIPELINE_STAGES]
  const plastic = config?.plasticRegions
  if (plastic?.enabled && plastic.mode !== 'off') {
    stages.splice(stages.indexOf('conceptMacros'), 0, 'plasticRegions')
  }
  return stages
}

function normalizeMetrics(value) {
  const metric = (name, fallback) => typeof value[name] === 'number' && Number.isFinite(value[name]) ? Math.max(0, Math.min(1, value[name])) : fallback
  const output = {
    overallScore: value.overallScore === undefined ? metric('holdoutTop1', 0) : metric('overallScore', 0),
    syntaxContrastAccuracy: value.syntaxContrastAccuracy === undefined ? metric('grammaticalPreference', 0) : metric('syntaxContrastAccuracy', 0),
    regressionRate: metric('regressionRate', 1),
    memorizationGap: metric('memorizationGap', 1),
    unseenAccuracy: value.unseenAccuracy === undefined ? metric('unseenCombinationAccuracy', 0) : metric('unseenAccuracy', 0),
  }
  for (const key of ['holdoutTop1', 'holdoutTop5', 'unseenCombinationAccuracy', 'grammaticalPreference', 'familyUtility']) {
    if (typeof value[key] === 'number' && Number.isFinite(value[key])) output[key] = Math.max(0, Math.min(1, value[key]))
  }
  for (const key of [
    'neuronCount',
    'familyCount',
    'birthProposalCount',
    'eligibleBirthProposalCount',
    'emergentLanguageFamilyCount',
    'eligibleEmergentLanguageFamilyCount',
    'conceptMacroCount',
    'conceptMacroProposalCount',
    'conceptMacroVirtualCount',
    'conceptMacroShadowValidatedCount',
    'conceptMacroActiveCount',
    'evaluationSampleCount',
  ]) {
    if (typeof value[key] === 'number' && Number.isFinite(value[key]) && value[key] >= 0) output[key] = value[key]
  }
  if (typeof value.corpusFingerprint === 'string') output.corpusFingerprint = value.corpusFingerprint.slice(0, 256)
  return output
}

/**
 * Creates a replaceable pipeline executor. A real runtime can replace individual
 * stage handlers, or the service can receive an entirely custom async executor.
 * A handler must explicitly return `verified: true`; unverified placeholder
 * artifacts can never become the active model.
 */
export function createPipelineExecutor({ stageHandlers = {} } = {}) {
  const handlers = { ...DEFAULT_STAGE_HANDLERS, ...stageHandlers }
  return async function pipelineExecutor(job, context) {
    const state = { sources: [], sourceRevisions: {}, partitions: null, inputManifestHash: null, artifactIds: [], stageResults: {} }
    const stages = resolvePipelineStages(context.config ?? context.store.getConfig())
    for (let index = 0; index < stages.length; index += 1) {
      const stage = stages[index]
      await context.checkpoint(stage)
      await context.progress(stage, index / (stages.length + 1))
      const result = await handlers[stage](job, context, state)
      state.stageResults[stage] = result
      await context.emit({ type: 'stage-completed', stage, data: result })
    }

    await context.checkpoint('evaluation-gate')
    const active = context.store.getActiveSnapshot()
    if (state.stageResults.textLab?.representationProfile === 'adaptive-eighth-id-v1') {
      const activeAtlas = readPlasticRegionSnapshot(context.store, active)
      if (activeAtlas.compatible &&
          activeAtlas.head.representationSpaceId === state.stageResults.textLab.representationSpaceId) {
        for (const artifactId of plasticRegionCarryForwardArtifactIds(context.store, active)) {
          if (!state.artifactIds.includes(artifactId)) state.artifactIds.push(artifactId)
        }
      }
    }
    const metrics = normalizeMetrics(state.stageResults.evaluate?.metrics ?? {})
    const evaluation = evaluateCandidate(metrics, active?.metrics, (context.config ?? context.store.getConfig()).evaluation)
    const allRuntimeStagesVerified = stages
      .every((stage) => state.stageResults[stage]?.verified === true)
    evaluation.runtimeVerified = allRuntimeStagesVerified
    await context.checkpoint('candidate-snapshot')
    const snapshotInput = {
      ...(typeof job.payload.candidateSnapshotId === 'string' && job.payload.candidateSnapshotId
        ? { id: job.payload.candidateSnapshotId }
        : {}),
      // Bind the candidate to the exact active base observed before region
      // artifacts were produced/carried. If another worker activates a newer
      // snapshot concurrently, this candidate stays on the old parent and can
      // never be activated as though its artifacts belonged to the new base.
      parentId: state.stageResults.plasticRegions?.regionBase?.snapshotId ?? active?.id ?? null,
      status: 'candidate', metrics, evaluation,
      artifactIds: state.artifactIds, sourceIds: state.sources.map((source) => source.id), inputManifestHash: state.inputManifestHash,
      sourceRevisions: state.sourceRevisions,
      label: job.payload.label,
    }
    if (state.stageResults.plasticRegions?.regionBase) {
      assertPlasticRegionBase(context.store, state.stageResults.plasticRegions.regionBase)
    }
    let candidate
    if (context.jobClaim) {
      if (typeof context.store.createClaimedSnapshot !== 'function') {
        const error = new Error('Claim korumalı snapshot store primitive’i eksik.')
        error.code = 'STORE_CLAIM_CONTRACT_MISSING'
        throw error
      }
      candidate = context.store.createClaimedSnapshot(
        context.jobId,
        context.jobClaim.owner,
        context.jobClaim.attempt,
        snapshotInput,
      )
    } else {
      candidate = context.store.createSnapshot(snapshotInput)
    }

    let committed = false
    let activationSnapshotId = null
    const automationLevel = job.payload.automationLevel ?? (context.config ?? context.store.getConfig()).automation.level
    const plasticRegionActivation = plasticRegionSnapshotActivationGate(context.store, candidate)
    const activationEligible = !plasticRegionActivation.required || plasticRegionActivation.eligible
    if (evaluation.passed && allRuntimeStagesVerified && automationLevel === 'controlled' && activationEligible) {
      await context.checkpoint('commit')
      if (context.jobClaim) {
        if (typeof context.store.completeClaimedJobWithSnapshot !== 'function') {
          const error = new Error('Atomik iş + snapshot tamamlama primitive’i eksik.')
          error.code = 'STORE_CLAIM_CONTRACT_MISSING'
          throw error
        }
        activationSnapshotId = candidate.id
        await context.emit({ type: 'snapshot-activation-ready', stage: 'commit', snapshotId: candidate.id })
      } else {
        context.store.activateSnapshot(candidate.id)
        committed = true
        await context.emit({ type: 'snapshot-activated', stage: 'commit', snapshotId: candidate.id })
      }
    } else if (evaluation.passed && allRuntimeStagesVerified) {
      await context.emit({
        type: activationEligible ? 'candidate-awaiting-approval' : 'candidate-region-release-required',
        stage: 'commit',
        snapshotId: candidate.id,
        ...(activationEligible ? {} : { reason: plasticRegionActivation.reason }),
      })
    } else {
      context.store.rejectSnapshot(candidate.id, {
        ...evaluation,
        reason: evaluation.passed ? 'Bir veya daha fazla çalışma aşaması doğrulanmış gerçek çıktı üretmedi.' : 'Değerlendirme kapısı başarısız.',
      }, context.jobClaim ? { jobId: context.jobId, claim: context.jobClaim } : undefined)
      await context.emit({ type: 'candidate-quarantined', level: 'warning', stage: 'commit', snapshotId: candidate.id })
    }
    await context.progress('completed', 1)
    return {
      snapshotId: candidate.id,
      committed,
      activationSnapshotId,
      awaitingApproval: evaluation.passed && allRuntimeStagesVerified &&
        automationLevel !== 'controlled' && activationEligible,
      activationBlockedReason: activationEligible ? null : 'plastic-region-release-required',
      evaluation,
      allRuntimeStagesVerified,
      artifactIds: state.artifactIds,
    }
  }
}

export async function generateTeacherSources({ topic, objective = '', diversity = 'balanced', count, language = 'tr', model }, context) {
  if (typeof topic !== 'string' || topic.trim().length < 2 || topic.length > 500) {
    const error = new Error('Konu 2–500 karakter arasında olmalı.')
    error.code = 'INVALID_TOPIC'
    throw error
  }
  if (!Number.isInteger(count) || count < 1 || count > 1_000) {
    const error = new Error('Üretim sayısı 1–1000 arasında bir tam sayı olmalı.')
    error.code = 'INVALID_COUNT'
    throw error
  }
  const selectedModel = String(model || context.store.getConfig().ollama.model || '').trim()
  if (!selectedModel) {
    const error = new Error('Ollama modeli seçilmedi.')
    error.code = 'OLLAMA_MODEL_REQUIRED'
    throw error
  }
  const proposals = []
  let promptTokens = 0
  let completionTokens = 0
  let resolvedModel = selectedModel
  let offset = 0
  while (offset < count) {
    if (offset > 0) {
      // Give response-close/cancel events a turn before purchasing the next
      // chunk. A synchronously resolving local provider must not starve the
      // abort event and run the entire legacy multi-part request after its
      // client has disconnected.
      await new Promise((resolve) => setTimeout(resolve, 0))
      context.signal?.throwIfAborted()
    }
    const chunkSize = Math.min(24, count - offset)
    const schema = structuredClone(teacherLessonSchema)
    schema.properties.lessons.maxItems = chunkSize
    schema.properties.lessons.minItems = chunkSize
    const messages = [
      { role: 'system', content: `Sen yalnızca kontrollü eğitim örnekleri üreten bir öğretmensin. ${language} dilinde, verilen konuyla ilgili tam ${chunkSize} kısa ve bağımsız örnek oluştur. Çıktın yalnız verilen JSON şemasına uysun. Gerçekmiş gibi kaynak veya alıntı uydurma.` },
      { role: 'user', content: `Konu: ${topic.trim()}\nAmaç: ${String(objective || 'bağlamsal çeşitlilik').slice(0, 500)}\nÇeşitlilik: ${diversity}. Farklı özne, nesne, araç, olumsuzluk, soru ve sözdizimi örnekleri kullan. Bu parti daha önceki örneklerden farklı olmalı; parti başlangıcı ${offset + 1}.` },
    ]
    const maxCompletionTokens = Math.min(4_096, Math.max(512, chunkSize * 128))
    const usageReservation = reserveBoundedTeacherCall(context.store, {
      messages,
      schema,
      maxCompletionTokens,
      model: selectedModel,
      signal: context.signal,
    })
    let result
    try {
      result = await context.ollama.generateStructured({
        model: selectedModel,
        signal: context.signal,
        schema,
        messages,
        retries: 0,
        options: { num_predict: maxCompletionTokens },
      })
    } catch (error) {
      settleBoundedTeacherCall(context.store, usageReservation, error)
      throw error
    }
    const settledUsage = settleBoundedTeacherCall(context.store, usageReservation, result)
    promptTokens += settledUsage.promptTokens
    completionTokens += settledUsage.completionTokens
    resolvedModel = result.model
    for (const [localIndex, lesson] of result.value.lessons.entries()) {
      const index = offset + localIndex
      const normalized = normalizeTextSource({
        text: lesson.text,
        title: `Ollama deneyi: ${topic.trim().slice(0, 80)} #${index + 1}`,
        language,
        provenance: {
          kind: 'ollama-generated', model: result.model, topic: topic.trim(), objective: String(objective).slice(0, 500), diversity,
          lessonKind: lesson.kind, rationale: lesson.rationale, generationIndex: index, promptSchemaHash: sha256(JSON.stringify(schema)),
          // Outputs from one teacher lineage are correlated observations, not
          // independent evidence for birth/promotion thresholds.
          independenceGroupId: canonicalOllamaIndependenceGroupId({ resolvedModelName: result.model }),
          evaluationEligible: false,
        },
      }, context.store.getConfig())
      normalized.sourceType = 'ollama-generated'
      normalized.provenance.kind = 'ollama-generated'
      proposals.push(normalized)
    }
    offset += chunkSize
  }
  const records = context.store.createSources(proposals).map((created) => ({ ...created, source: { ...created.source, content: undefined } }))
  return {
    items: records.map(({ source, duplicate }) => ({ source, duplicate })),
    generated: records.length,
    inserted: records.filter((item) => !item.duplicate).length,
    duplicates: records.filter((item) => item.duplicate).length,
    model: resolvedModel,
    usage: { promptTokens, completionTokens },
  }
}
