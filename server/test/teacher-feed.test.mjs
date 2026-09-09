import assert from 'node:assert/strict'
import test from 'node:test'
import {
  canonicalizeTeacherText,
  createTeacherDuplicateSignature,
  createTeacherFeedCampaignInput,
  createTeacherFeedExecutor,
  createTeacherFeedLessonSchema,
  evaluateTeacherText,
  filterTeacherLessons,
  materializeTeacherLesson,
  mixSeed,
  normalizeTeacherFeedCampaignRequest,
  planTeacherFeedBatch,
  teacherFeedCandidateSnapshotId,
  teacherFeedGenerationLane,
  teacherNearDuplicateScore,
  TEACHER_FEED_HYBRID_PROTOCOL_FINGERPRINT,
  TEACHER_FEED_PROTOCOL_FINGERPRINT,
  TEACHER_FEED_PROTOCOL_VERSION,
  TEACHER_SURFACE_COMPOSITION_POLICY_ID,
  TURKISH_FOUNDATIONS_V1,
} from '../teacher-feed.mjs'
import {
  createTeacherDiscoveryAttempt,
  createTeacherDiscoverySchema,
  TEACHER_DISCOVERY_PROTOCOL_FINGERPRINT,
} from '../teacher-free-discovery.mjs'
import { createTeacherChoiceBoard } from '../teacher-feed-catalog.mjs'
import { canonicalOllamaIndependenceGroupId } from '../ollama-lineage.mjs'
import { OLLAMA_DIVERSE_MAX_NUM_PREDICT } from '../ollama.mjs'
import { TURKISH_SPELLING_POLICY_ID } from '../turkish-spelling.mjs'
import { defaultConfig } from '../config.mjs'
import { createTrainingStore } from '../store.mjs'
import { TEACHER_CONTEXT_TOKEN_LIMIT, TEACHER_FEED_CONTEXT_TOKEN_LIMIT } from '../teacher-usage.mjs'
import {
  INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT,
  INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
} from '../structural-stop-gate.mjs'
import { temporaryDirectory } from './helpers.mjs'

const MODEL = {
  name: 'qwen2.5-coder:14b',
  digest: 'sha256:model-fixture',
  size: 9_000_000_000,
  modifiedAt: '2026-08-20T12:00:00.000Z',
  details: { family: 'qwen2', parameter_size: '14.8B', quantization_level: 'Q4_K_M' },
}

function passingStructuralReport(snapshotId, inputManifestHash = 'manifest-fixed') {
  const ci = (mean, low, high, sourceGroupCount = 45) => ({
    mean, low, high, confidence: 0.95, seedCount: 5, sourceGroupCount, samples: 2_000,
  })
  return {
    format: 'bombom-independent-structural-benchmark',
    version: 1,
    benchmarkId: 'controlled-event-graph-144-v1',
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    origin: 'independent-deterministic',
    deterministic: true,
    usesOllama: false,
    snapshotId,
    inputManifestHash,
    evidence: {
      combinationCount: 144,
      surfacesPerCombination: 2,
      splitPolicy: 'combination-before-surface-v1',
      holdoutRatio: 0.31,
      holdoutCombinationCount: 45,
      seedCount: 5,
      sourceGroupCount: 45,
      bootstrapConfidence: 0.95,
      bootstrapSamples: 2_000,
      intervals: {
        holdoutGraphF1: ci(0.81, 0.80, 0.82),
        h0OnlyGraphF1: ci(0.62, 0.61, 0.63),
        noRecurrenceGraphF1: ci(0.61, 0.60, 0.62),
        shuffledEdgeGraphF1: ci(0.60, 0.59, 0.61),
        baselineAdvantage: ci(0.19, 0.17, 0.21),
        targetFamilyAblationDrop: ci(0.12, 0.11, 0.13),
        targetEdgeAblationDrop: ci(0.11, 0.105, 0.12),
        shamFamilyAblationDrop: ci(0.02, 0.015, 0.025),
        shamEdgeAblationDrop: ci(0.02, 0.015, 0.025),
        perturbationRecoveryRate: ci(0.74, 0.72, 0.76),
        knownStructureFalseBirthRate: ci(0.04, 0.03, 0.05),
        repeatedNoveltyBirthPrecision: ci(0.84, 0.81, 0.87, 10),
        repeatedNoveltyBirthSensitivity: ci(0.84, 0.81, 0.87, 5),
      },
      ablationTargetSelection: 'snapshot-model-only-no-evaluation-data-v2',
      edgeShamProtocol: 'snapshot-digest-random-nontarget-edge-v1',
      targetEdge: 'target-a\u0000target-b',
      shamEdge: 'sham-a\u0000sham-b',
      familyShamProtocol: 'deterministic-model-only-exact-reference-footprint-mask-v1',
      targetFamilyFootprint: { memberships: 2, familyTransitions: 1, utilityReports: 0, coalitions: 0, coalitionActivations: 0, coalitionBindings: 0, birthMemberships: 0, total: 3 },
      shamFamilyFootprint: { memberships: 2, familyTransitions: 1, utilityReports: 0, coalitions: 0, coalitionActivations: 0, coalitionBindings: 0, birthMemberships: 0, total: 3 },
      shamFamilyFootprintMatched: true,
      shamFamilyMaskDigest: 'a'.repeat(64),
      noRecurrenceProtocol: 'full-tick-zero-persistent-transition-evidence-v2',
      birthPrecisionDefinition: 'TP/(TP+FP)',
      birthNegativeSource: 'frozen-train-calibration-repeated-four-observations-v2',
      eligibleNoveltyCaseCount: 5,
      knownBirthPrecisionNegativeCaseCount: 5,
      birthConfusion: { truePositive: 21, falsePositive: 4, falseNegative: 4, trueNegative: 21 },
    },
    evaluationPolicy: {
      sourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
      validationOllamaExampleCount: 0,
      testOllamaExampleCount: 0,
      splitFrozenBeforeTraining: true,
    },
    metrics: {
      holdoutGraphF1: 0.81,
      h0OnlyGraphF1: 0.62,
      noRecurrenceGraphF1: 0.61,
      shuffledEdgeGraphF1: 0.60,
      targetFamilyAblationDrop: 0.12,
      targetEdgeAblationDrop: 0.11,
      shamFamilyAblationDrop: 0.02,
      shamEdgeAblationDrop: 0.02,
      perturbationRecoveryRate: 0.74,
      knownStructureFalseBirthRate: 0.04,
      repeatedNoveltyBirthPrecision: 0.84,
      repeatedNoveltyBirthSensitivity: 0.84,
    },
  }
}

function decisiveEarlyFailureReport(snapshotId, inputManifestHash = 'manifest-fixed') {
  return {
    format: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_FORMAT,
    version: 1,
    benchmarkId: 'controlled-event-graph-144-v1',
    benchmarkFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT,
    runnerVersion: 1,
    runnerFingerprint: 'b'.repeat(64),
    earlyFailurePolicyFingerprint: INDEPENDENT_STRUCTURAL_BENCHMARK_EARLY_FAILURE_POLICY_FINGERPRINT,
    origin: 'independent-deterministic',
    deterministic: true,
    usesOllama: false,
    snapshotId,
    inputManifestHash,
    completion: 'decisive-early-failure',
    decisiveCheck: {
      id: 'holdout-graph-f1',
      threshold: { direction: 'minimum', boundary: 'ci.low', value: 0.75 },
      interval: { mean: 0.27, low: 0.24, high: 0.30, confidence: 0.95, seedCount: 5, sourceGroupCount: 44, samples: 2_000 },
      passed: false,
    },
    evidence: {
      combinationCount: 144,
      surfacesPerCombination: 2,
      splitPolicy: 'combination-before-surface-v1',
      holdoutRatio: 44 / 144,
      holdoutCombinationCount: 44,
      seedCount: 5,
      sourceGroupCount: 44,
      episodeCount: 440,
      maximumTicks: 64,
      observationDigest: 'c'.repeat(64),
      observationChunkDigests: Array.from({ length: 5 }, (_, index) => index.toString(16).padStart(64, '0')),
      completedVariant: 'full',
      bootstrapConfidence: 0.95,
      bootstrapSamples: 2_000,
      skippedVariants: Array.from({ length: 9 }, (_, index) => `variant-${index}`),
      skippedEpisodeCount: 3_720,
    },
    evaluationPolicy: {
      sourcePolicy: 'controlled-benchmark-only-no-ollama-v1',
      validationOllamaExampleCount: 0,
      testOllamaExampleCount: 0,
      splitFrozenBeforeTraining: true,
    },
  }
}

function choicesFor(request, { count, start = 0 } = {}) {
  const choiceSchema = request.schema.properties.choices
  assert.equal(choiceSchema.items.type, 'integer')
  assert.equal(choiceSchema.uniqueItems, true)
  const requestedCount = count ?? choiceSchema.maxItems
  return choiceSchema.items.enum.slice(start, start + requestedCount)
}

function boardForPlan(plan, seed = mixSeed(plan.batchSeed, 'attempt', 0), size = 16) {
  return createTeacherChoiceBoard(plan, {
    seed,
    size,
    evaluate: (text) => evaluateTeacherText(text, {
      unit: plan.unit,
      form: plan.form,
      axes: plan.axes,
    }),
  })
}

function fixtureStore(job) {
  const sources = new Map()
  const byHash = new Map()
  const commits = []
  const usage = []
  const usageLimits = { teacherRequests: Number.POSITIVE_INFINITY, teacherTokens: Number.POSITIVE_INFINITY }
  const reserveUsage = (reservations) => {
    const additions = new Map()
    for (const reservation of reservations) {
      additions.set(reservation.kind, (additions.get(reservation.kind) ?? 0) + reservation.amount)
    }
    for (const [kind, amount] of additions) {
      const consumed = usage.filter((item) => item.kind === kind).reduce((sum, item) => sum + item.amount, 0)
      if (consumed + amount > (usageLimits[kind] ?? Number.POSITIVE_INFINITY)) {
        const error = new Error(`${kind} günlük bütçesi aşıldı.`)
        error.code = 'BUDGET_EXCEEDED'
        throw error
      }
    }
    for (const reservation of reservations) usage.push({ kind: reservation.kind, amount: reservation.amount })
  }
  return {
    commits,
    usage,
    usageLimits,
    sources,
    getConfig: () => defaultConfig('/tmp/no-write'),
    getJob: () => structuredClone(job),
    getSource: (id) => structuredClone(sources.get(id)),
    listSources: ({ includeContent } = {}) => [...sources.values()].map((source) => includeContent ? structuredClone(source) : { ...structuredClone(source), content: undefined }),
    getSnapshot: () => undefined,
    consumeUsage: (kind, amount) => reserveUsage([{ kind, amount }]),
    commitTeacherFeedBatch(_jobId, { inputs, checkpoint, artifact, usageReservations = [], usageSettlements = [] }) {
      reserveUsage(usageReservations)
      for (const settlement of usageSettlements) {
        const reserved = usage
          .filter((item) => item.kind === settlement.kind)
          .reduce((sum, item) => sum + item.amount, 0)
        if (reserved < settlement.reservedAmount) {
          const error = new Error('Öğretmen token rezervasyonu bulunamadı.')
          error.code = 'TEACHER_USAGE_RESERVATION_MISSING'
          throw error
        }
        const adjustment = settlement.actualAmount - settlement.reservedAmount
        if (adjustment) usage.push({ kind: settlement.kind, amount: adjustment })
      }
      const items = []
      for (const input of inputs) {
        const previous = byHash.get(input.contentHash)
        if (previous) {
          items.push({ source: structuredClone(previous), duplicate: true })
          continue
        }
        const source = { ...structuredClone(input), status: 'approved', selected: true }
        sources.set(source.id, source)
        byHash.set(source.contentHash, source)
        items.push({ source: structuredClone(source), duplicate: false })
      }
      job.payload.feedState = structuredClone(checkpoint)
      commits.push({ inputs: structuredClone(inputs), checkpoint: structuredClone(checkpoint), artifact: structuredClone(artifact) })
      return { items, artifact, job: structuredClone(job) }
    },
  }
}

function fixtureContext(job, store, ollama) {
  return {
    jobId: job.id,
    store,
    ollama,
    config: store.getConfig(),
    signal: new AbortController().signal,
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }
}

test('campaign request freezes model identity and creates deterministic initial state', () => {
  const request = { model: MODEL.name, seed: 42, targetAccepted: 12, generationBatchSize: 2, trainEveryAccepted: 8, curriculum: 'turkish-foundations-v1', mode: 'research-only' }
  const first = normalizeTeacherFeedCampaignRequest(request, MODEL, 'campaign-a')
  const second = normalizeTeacherFeedCampaignRequest(request, MODEL, 'campaign-a')
  assert.deepEqual(first, second)
  assert.equal(first.campaign.curriculum, 'turkish-foundations-v1')
  assert.equal(first.campaign.targetAccepted, 12)
  assert.equal(first.campaign.generationBatchSize, 2)
  assert.equal(first.campaign.trainEveryAccepted, 8)
  assert.equal(first.campaign.mode, 'research-only')
  assert.equal(first.campaign.proposalOnly, true)
  assert.equal(first.campaign.protocolVersion, TEACHER_FEED_PROTOCOL_VERSION)
  assert.equal(first.campaign.protocolFingerprint, TEACHER_FEED_PROTOCOL_FINGERPRINT)
  assert.equal(first.campaign.protocol.contextTokenLimit, TEACHER_FEED_CONTEXT_TOKEN_LIMIT)
  assert.equal(first.campaign.protocol.spellingPolicyId, TURKISH_SPELLING_POLICY_ID)
  assert.equal(first.feedState.protocolFingerprint, TEACHER_FEED_PROTOCOL_FINGERPRINT)
  assert.equal(first.campaign.modelIdentity.digest, MODEL.digest)
  assert.match(first.campaign.modelIdentity.fingerprint, /^[a-f0-9]{64}$/u)
  assert.equal(first.feedState.nextBatchIndex, 0)

  const defaults = normalizeTeacherFeedCampaignRequest({ model: MODEL.name, targetAccepted: 12 }, MODEL, 'campaign-defaults')
  assert.equal(defaults.campaign.generationBatchSize, 2)
  assert.throws(
    () => normalizeTeacherFeedCampaignRequest({ model: MODEL.name, targetAccepted: 12, generationBatchSize: 3 }, MODEL, 'campaign-too-wide'),
    { code: 'INVALID_TEACHER_CAMPAIGN' },
  )
  const plan = planTeacherFeedBatch({ seed: request.seed, batchIndex: 0, campaignId: 'campaign-a' })
  const board = boardForPlan(plan)
  const schema = createTeacherFeedLessonSchema(plan.unit, 2, { board })
  assert.deepEqual(schema.required, ['choices'])
  assert.equal(schema.additionalProperties, false)
  assert.equal(schema.properties.choices.minItems, 1)
  assert.equal(schema.properties.choices.maxItems, 2)
  assert.equal(schema.properties.choices.uniqueItems, true)
  assert.equal(schema.properties.choices.items.type, 'integer')
  assert.deepEqual(schema.properties.choices.items.enum, board.choices.map((choice) => choice.id))
  assert.throws(
    () => createTeacherFeedLessonSchema(plan.unit, 2),
    { code: 'TEACHER_CHOICE_BOARD_REQUIRED' },
  )

  const input = createTeacherFeedCampaignInput(request, MODEL, 'campaign-a')
  assert.equal(input.type, 'teacher-feed-campaign')
  assert.equal(input.idempotencyKey, 'teacher-feed:campaign-a')
})

test('hybrid mode adds a replayable free discovery protocol without changing the controlled default', () => {
  const controlled = normalizeTeacherFeedCampaignRequest({ model: MODEL.name, targetAccepted: 4 }, MODEL, 'controlled-default')
  assert.equal(controlled.campaign.generationMode, 'controlled')
  assert.equal(controlled.campaign.protocolFingerprint, TEACHER_FEED_PROTOCOL_FINGERPRINT)
  assert.equal(controlled.feedState.lanes, undefined)

  const hybrid = normalizeTeacherFeedCampaignRequest({ model: MODEL.name, targetAccepted: 4, generationMode: 'hybrid' }, MODEL, 'hybrid-default')
  assert.equal(hybrid.campaign.protocolFingerprint, TEACHER_FEED_HYBRID_PROTOCOL_FINGERPRINT)
  assert.equal(hybrid.campaign.protocol.discovery.id, 'turkish-free-surface-discovery-v1')
  assert.deepEqual(hybrid.feedState.lanes.controlled, hybrid.feedState.lanes.discovery)
  assert.equal(teacherFeedGenerationLane('hybrid', 0), 'controlled')
  assert.equal(teacherFeedGenerationLane('hybrid', 1), 'discovery')
  assert.equal(teacherFeedGenerationLane('controlled', 1), 'controlled')

  const plan = planTeacherFeedBatch({ seed: 123, batchIndex: 1, campaignId: 'hybrid-default' })
  const first = createTeacherDiscoveryAttempt({ plan, count: 2, attempt: 0, seed: 99 })
  const replay = createTeacherDiscoveryAttempt({ plan, count: 2, attempt: 0, seed: 99 })
  assert.deepEqual(first, replay)
  assert.match(first.fingerprint, /^[a-f0-9]{64}$/u)
  assert.equal(first.maxCompletionTokens, OLLAMA_DIVERSE_MAX_NUM_PREDICT)
  assert.match(TEACHER_DISCOVERY_PROTOCOL_FINGERPRINT, /^[a-f0-9]{64}$/u)
  assert.deepEqual(createTeacherDiscoverySchema(2).required, ['sentences'])
  assert.equal(first.schema.properties.sentences.maxItems, 2)
})

test('seeded curriculum covers each unit once per epoch and replays exactly', () => {
  const first = Array.from({ length: 12 }, (_, batchIndex) => planTeacherFeedBatch({ seed: 77, batchIndex, campaignId: 'c' }))
  const replay = Array.from({ length: 12 }, (_, batchIndex) => planTeacherFeedBatch({ seed: 77, batchIndex, campaignId: 'c' }))
  assert.deepEqual(first.map(({ batchId, batchSeed, unit }) => ({ batchId, batchSeed, unit: unit.id })), replay.map(({ batchId, batchSeed, unit }) => ({ batchId, batchSeed, unit: unit.id })))
  assert.equal(new Set(first.map((item) => item.unit.id)).size, 12)
  assert.ok(first.every((item) => item.unit.forms.includes(item.form)))
  assert.notDeepEqual(first.map((item) => item.unit.id), Array.from({ length: 12 }, (_, batchIndex) => planTeacherFeedBatch({ seed: 78, batchIndex, campaignId: 'c' }).unit.id))
  const independentCampaign = Array.from({ length: 12 }, (_, batchIndex) => planTeacherFeedBatch({ seed: 77, batchIndex, campaignId: 'another-campaign' }))
  assert.deepEqual(
    first.map(({ batchSeed, unit, form, axes }) => ({ batchSeed, unit: unit.id, form, axes })),
    independentCampaign.map(({ batchSeed, unit, form, axes }) => ({ batchSeed, unit: unit.id, form, axes })),
    'same user seed must replay the same curriculum sequence across campaign ids',
  )
  assert.notEqual(first[0].batchId, independentCampaign[0].batchId, 'campaign identity still namespaces durable batch ids')
})

test('tense witness accepts only reviewed finite verbs across Turkish final letters', () => {
  const tense = TURKISH_FOUNDATIONS_V1.find((unit) => unit.id === 'tense-aspect')
  assert.ok(tense)
  for (const surface of ['Ayşe dün fideleri suladı.', 'Mert şimdi toprağı hazırlıyor.', 'Ece yarın panoyu ölçecek.', 'Deniz daima topu yuvarlar.']) {
    assert.equal(tense.requiredSurface.test(surface), true, surface)
  }
  for (const nonWitness of ['Ayşe duvar önünde.', 'Mert pazarda.', 'Ece bahar gününde.']) {
    assert.equal(tense.requiredSurface.test(nonWitness), false, nonWitness)
  }
})

test('quality and duplicate filters are deterministic and reject malformed or near-identical text', () => {
  const plan = planTeacherFeedBatch({ seed: 11, batchIndex: 0, campaignId: 'quality' })
  const board = boardForPlan(plan)
  const choice = board.choices[0]
  const materialized = materializeTeacherLesson(choice.id, { board })
  const lesson = { ...materialized, choiceId: choice.id, skill: plan.unit.id, form: plan.form }
  const good = materialized.text
  const malformed = canonicalizeTeacherText('  bu bir örnek cümle   ')
  assert.equal(evaluateTeacherText(malformed, { unit: plan.unit, form: plan.form, axes: plan.axes }).accepted, false)

  const signature = createTeacherDuplicateSignature(good)
  const near = createTeacherDuplicateSignature(good.replace(/[.!?]$/u, '!'))
  assert.ok(teacherNearDuplicateScore(signature, near) >= 0.86)
  const result = filterTeacherLessons([
    lesson,
    structuredClone(lesson),
  ], { unit: plan.unit, form: plan.form, axes: plan.axes })
  assert.equal(result.accepted.length, 1)
  assert.equal(result.rejected.length, 1)
  assert.ok(result.rejected[0].reasons.includes('exact-duplicate'))
})

test('teacher text evaluation exposes the deterministic Turkish spelling rejection', () => {
  const invalid = evaluateTeacherText('Ali watching sırasında kapıda sessizce bekledi.')
  assert.equal(invalid.accepted, false)
  assert.ok(invalid.reasons.includes('turkish-spelling'))
  assert.equal(invalid.spellingUnknownCount, 1)
  assert.deepEqual(invalid.spellingDiagnostics.map((item) => item.token), ['watching'])

  const valid = evaluateTeacherText("Ali'nin arkadaşları Türkiye'de kütüphanede çalıştı.")
  assert.equal(valid.accepted, true)
  assert.equal(valid.spellingUnknownCount, 0)

  const plans = Array.from({ length: TURKISH_FOUNDATIONS_V1.length }, (_, batchIndex) => (
    planTeacherFeedBatch({ seed: 42, batchIndex, campaignId: 'catalog-spelling' })
  ))
  for (const plan of plans) {
    const board = boardForPlan(plan, mixSeed(plan.batchSeed, 'catalog-spelling'))
    for (const choice of board.choices) {
      const quality = evaluateTeacherText(choice.text, { unit: plan.unit, form: plan.form, axes: plan.axes })
      assert.equal(quality.reasons.includes('turkish-spelling'), false, `${choice.text}: ${quality.spellingDiagnostics.map((item) => item.token).join(', ')}`)
      assert.equal(quality.spellingUnknownCount, 0)
    }
  }
})

test('negative surface gate accepts finite negation without confusing ordinary -ma nouns', () => {
  const negativeUnit = TURKISH_FOUNDATIONS_V1.find((item) => item.id === 'negation')
  const falsePositive = evaluateTeacherText('Ali bugün kırmızı elma aldı.', { unit: negativeUnit, form: 'negative' })
  assert.ok(falsePositive.reasons.includes('missing-surface-feature'))
  assert.ok(falsePositive.reasons.includes('missing-form-feature'))

  const progressive = evaluateTeacherText('Ali bugün eski kapıyı sessizce açmıyor.', { unit: negativeUnit, form: 'negative' })
  assert.equal(progressive.accepted, true)

  for (const sentence of ['Ali bugün eski kapıyı sessizce açmadı.', 'Ece sabah ağır kutuyu depoya taşımadı.']) {
    const past = evaluateTeacherText(sentence, { unit: negativeUnit, form: 'negative' })
    assert.equal(past.accepted, true, `${sentence}: ${past.reasons.join(', ')}`)
  }
})

test('requested length, curriculum surface, and server-owned form are all acceptance gates', () => {
  const unit = TURKISH_FOUNDATIONS_V1.find((item) => item.id === 'locative-context')
  const axes = { length: 'orta: 9–15 kelime' }
  const tooShort = evaluateTeacherText('Otobüs duraklarında bekliyoruz.', { unit, form: 'contrast', axes })
  assert.equal(tooShort.accepted, false)
  assert.ok(tooShort.reasons.includes('missing-form-feature'))
  assert.ok(tooShort.reasons.includes('surface-length'))

  const wrongSkill = evaluateTeacherText('Gelişim taşıma şirketlerinin yoğunluğunu hissediyorum.', { unit, form: 'contrast', axes })
  assert.ok(wrongSkill.reasons.includes('missing-surface-feature'))

  const valid = evaluateTeacherText('Ali bugün depoda kutuyu sayıyor ama Ece dışarıda sessizce bekliyor.', { unit, form: 'contrast', axes })
  assert.equal(valid.accepted, true)
})

test('closed choice boards replay deterministically and preserve selection provenance', () => {
  const plan = planTeacherFeedBatch({ seed: 42, batchIndex: 2, campaignId: 'board-replay' })
  const seed = mixSeed(plan.batchSeed, 'attempt', 0)
  const board = boardForPlan(plan, seed)
  const replay = boardForPlan(plan, seed)
  assert.deepEqual(board, replay)
  assert.match(board.id, /^choice-board-[a-f0-9]{24}$/u)
  assert.match(board.fingerprint, /^[a-f0-9]{64}$/u)
  assert.ok(board.choices.length >= 2)
  assert.deepEqual(board.choices.map((choice) => choice.id), Array.from({ length: board.choices.length }, (_, index) => index))
  assert.equal(new Set(board.choices.map((choice) => choice.semanticCardDigest)).size, board.choices.length)

  const choice = board.choices[0]
  const materialized = materializeTeacherLesson(choice.id, { board })
  assert.equal(materialized.text, choice.text)
  assert.deepEqual(materialized.surfaceReasons, [])
  assert.equal(materialized.semanticCardDigest, choice.semanticCardDigest)
  assert.equal(materialized.surfaceAssembly.policyId, TEACHER_SURFACE_COMPOSITION_POLICY_ID)
  assert.equal(materialized.surfaceAssembly.boardId, board.id)
  assert.equal(materialized.surfaceAssembly.boardFingerprint, board.fingerprint)
  assert.equal(materialized.surfaceAssembly.choiceId, choice.id)
  assert.equal(materialized.surfaceAssembly.rawModelSelection, choice.id)
  assert.deepEqual(materialized.surfaceAssembly.resolvedSelection, choice.selection)
  assert.deepEqual(materialized.surfaceAssembly.semanticCard, choice.semanticCard)
  assert.deepEqual(materialized.surfaceAssembly.renderPlan, choice.renderPlan)
  assert.equal(materialized.surfaceAssembly.surfaceOrigin, 'server-vetted-catalog')
  assert.equal(materialized.surfaceAssembly.selectionOrigin, 'ollama-closed-choice')
  assert.match(materialized.surfaceAssembly.finalTextDigest, /^[a-f0-9]{64}$/u)

  assert.deepEqual(materializeTeacherLesson('0', { board }), { text: '', surfaceReasons: ['invalid-choice-id'] })
  assert.deepEqual(materializeTeacherLesson(board.choices.length + 10, { board }), { text: '', surfaceReasons: ['unknown-choice-id'] })
})

test('executor stores provenance atomically and invokes only proposal pipeline cycles', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 123, targetAccepted: 4, generationBatchSize: 2, trainEveryAccepted: 4, model: MODEL.name, curriculum: 'turkish-foundations-v1', mode: 'research-only' }, MODEL, 'campaign-live')
  const job = { id: 'job-1', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const calls = []
  const attemptsAtModelBoundary = []
  const reservedTeacherTokensAtModelBoundary = []
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      calls.push(request)
      attemptsAtModelBoundary.push(structuredClone(job.payload.feedState.pendingGenerationBatch.inFlightAttempt))
      reservedTeacherTokensAtModelBoundary.push(
        store.usage
          .filter((item) => item.kind === 'teacherTokens')
          .reduce((sum, item) => sum + item.amount, 0),
      )
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 20, completionTokens: 40 }
    },
  }
  const pipelineJobs = []
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => {
    pipelineJobs.push(pipelineJob)
    return { snapshotId: 'candidate-1', committed: false, awaitingApproval: true, evaluation: { passed: true } }
  } })
  const result = await executor(job, fixtureContext(job, store, ollama))

  assert.ok(calls.length >= 1)
  assert.equal(result.teacherRequests, calls.length)
  const batchSeed = planTeacherFeedBatch({ seed: 123, batchIndex: 0, campaignId: 'campaign-live' }).batchSeed
  assert.equal(calls[0].seed, mixSeed(batchSeed, 'attempt', 0))
  assert.ok(calls.every((call) => Number.isInteger(call.maxCompletionTokens) && call.maxCompletionTokens >= 128 && call.maxCompletionTokens <= 256))
  assert.equal(attemptsAtModelBoundary[0].contextTokenLimit, TEACHER_FEED_CONTEXT_TOKEN_LIMIT)
  assert.equal(attemptsAtModelBoundary[0].maxPromptTokens, TEACHER_FEED_CONTEXT_TOKEN_LIMIT)
  assert.equal(attemptsAtModelBoundary[0].promptBoundPolicy, 'ollama-num-ctx-v1')
  assert.equal(
    reservedTeacherTokensAtModelBoundary[0],
    TEACHER_FEED_CONTEXT_TOKEN_LIMIT + calls[0].maxCompletionTokens,
  )
  assert.equal(result.accepted, 4)
  assert.equal(result.completedTrainingCycles, 1)
  assert.equal(result.proposalOnly, true)
  assert.equal(pipelineJobs.length, 1)
  assert.equal(pipelineJobs[0].payload.automationLevel, 'proposal')
  assert.equal(pipelineJobs[0].payload.sourceIds.length, 4)
  assert.equal(pipelineJobs[0].payload.candidateSnapshotId, teacherFeedCandidateSnapshotId('campaign-live', 1))
  for (const source of store.sources.values()) {
    assert.equal(source.sourceType, 'ollama-generated')
    assert.equal(source.provenance.kind, 'ollama-selected-server-rendered')
    assert.equal(source.provenance.managedByTeacherCampaign, true)
    assert.equal(source.provenance.campaignId, 'campaign-live')
    assert.ok(Number.isInteger(source.provenance.batchIndex) && source.provenance.batchIndex >= 0)
    assert.equal(source.provenance.evaluationEligible, false)
    assert.equal(source.provenance.independenceGroupId, canonicalOllamaIndependenceGroupId({
      resolvedModelName: MODEL.name,
      weightsDigest: MODEL.digest,
    }))
    assert.equal(source.provenance.modelIdentity, payload.campaign.modelIdentity.fingerprint)
    assert.equal(source.provenance.teacherFeedProtocolVersion, TEACHER_FEED_PROTOCOL_VERSION)
    assert.equal(source.provenance.teacherFeedProtocolFingerprint, TEACHER_FEED_PROTOCOL_FINGERPRINT)
    assert.equal(source.provenance.spellingPolicyId, TURKISH_SPELLING_POLICY_ID)
    assert.equal(source.provenance.labelOrigin, 'server-curriculum')
    assert.equal(source.provenance.surfaceCompositionPolicyId, TEACHER_SURFACE_COMPOSITION_POLICY_ID)
    assert.equal(Object.keys(source.provenance).length, 32, 'source provenance must stay within the durable 32-field contract')
    assert.equal(typeof source.provenance.receivedAt, 'string')
    assert.equal(Object.hasOwn(source.provenance, 'labelOrigin'), true)
    assert.equal(Object.hasOwn(source.provenance, 'qualityScore'), true)
    assert.equal(Object.hasOwn(source.provenance, 'proposalOnly'), true, 'the tail field must not be silently truncated')
    assert.ok(source.provenance.surfaceTrace.length <= 1_000, 'bounded surface trace must not exceed 1000 characters')
    const surfaceTrace = JSON.parse(source.provenance.surfaceTrace)
    assert.equal(surfaceTrace.policyId, TEACHER_SURFACE_COMPOSITION_POLICY_ID)
    assert.match(surfaceTrace.catalogFingerprint, /^[a-f0-9]{64}$/u)
    assert.match(surfaceTrace.boardFingerprint, /^[a-f0-9]{64}$/u)
    assert.ok(Number.isInteger(surfaceTrace.choiceId))
    assert.equal(typeof surfaceTrace.resolvedSelection.primaryCardId, 'string')
    assert.match(surfaceTrace.semanticCardDigest, /^[a-f0-9]{64}$/u)
    assert.match(surfaceTrace.semanticIdentityDigest, /^[a-f0-9]{64}$/u)
    assert.equal(surfaceTrace.surfaceOrigin, 'server-vetted-catalog')
    assert.equal(surfaceTrace.selectionOrigin, 'ollama-closed-choice')
    assert.equal(surfaceTrace.finalTextDigest.length, 64)
    assert.equal(typeof source.provenance.curriculumInstruction, 'string')
    assert.equal(Object.hasOwn(source.provenance, 'rationale'), false)
    assert.equal(source.provenance.validatedLessonForm, true)
    assert.equal(source.provenance.validatedLengthRange, true)
    assert.ok(Number.isInteger(source.provenance.actualWordCount))
  }
  const storedBatch = store.commits.find((commit) => commit.artifact.phase === 'batch-stored')
  assert.ok(storedBatch)
  assert.equal(storedBatch.artifact.payload.protocolFingerprint, TEACHER_FEED_PROTOCOL_FINGERPRINT)
  assert.equal(storedBatch.artifact.payload.spellingPolicyId, TURKISH_SPELLING_POLICY_ID)
  assert.equal(storedBatch.artifact.payload.acceptedChoiceEvidence.length, storedBatch.inputs.length)
  assert.ok(storedBatch.artifact.payload.acceptedChoiceEvidence.every((item) => Number.isInteger(item.choiceId) && item.semanticCard))
  assert.equal(job.payload.feedState.pendingTrainingMilestone, null)
  assert.equal(job.payload.feedState.accepted, 4)
  assert.equal(job.payload.feedState.sourceIds.length, 4)
})

test('hybrid executor alternates controlled and discovery batches and keeps their evidence separate', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 123,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    generationMode: 'hybrid',
  }, MODEL, 'campaign-hybrid')
  const job = { id: 'job-hybrid', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const calls = []
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      calls.push(request)
      if (request.schema.properties.choices) {
        return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 20, completionTokens: 40 }
      }
      assert.ok(request.schema.properties.sentences)
      return {
        value: { sentences: ['Bugün usta bunu dikkatle inceler.', 'Yarın görevli ona yeni paketi verir.'] },
        model: MODEL.name,
        promptTokens: 24,
        completionTokens: 48,
      }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => ({
    snapshotId: 'hybrid-candidate', committed: false, awaitingApproval: true, evaluation: { passed: true },
  }) })
  const result = await executor(job, fixtureContext(job, store, ollama))
  assert.equal(result.accepted, 4)
  assert.equal(result.generationMode, 'hybrid')
  assert.equal(result.lanes.controlled.accepted, 2)
  assert.equal(result.lanes.discovery.accepted, 2)
  assert.equal(calls.length, 2)
  assert.ok(calls[0].schema.properties.choices)
  assert.ok(calls[1].schema.properties.sentences)

  const batches = store.commits.filter((commit) => commit.artifact.phase === 'batch-stored')
  assert.equal(batches[0].artifact.payload.generationLane, 'controlled')
  assert.equal(batches[0].artifact.payload.acceptedChoiceEvidence.length, 2)
  assert.equal(batches[0].artifact.payload.acceptedDiscoveryEvidence.length, 0)
  assert.equal(batches[1].artifact.payload.generationLane, 'discovery')
  assert.equal(batches[1].artifact.payload.acceptedChoiceEvidence.length, 0)
  assert.equal(batches[1].artifact.payload.acceptedDiscoveryEvidence.length, 2)
  const discoverySources = [...store.sources.values()].filter((source) => source.provenance.kind === 'ollama-authored-discovery')
  assert.equal(discoverySources.length, 2)
  assert.ok(discoverySources.every((source) => source.provenance.evaluationEligible === false && source.provenance.proposalOnly === true))
})

test('teacher model is released immediately before a real milestone pipeline and unload failure is only a warning', async () => {
  for (const shouldFail of [false, true]) {
    const payload = normalizeTeacherFeedCampaignRequest({
      seed: shouldFail ? 0x201 : 0x200,
      targetAccepted: 2,
      generationBatchSize: 2,
      trainEveryAccepted: 2,
      model: MODEL.name,
    }, MODEL, `campaign-unload-${shouldFail}`)
    const job = { id: `job-unload-${shouldFail}`, type: 'teacher-feed-campaign', payload }
    const store = fixtureStore(job)
    const order = []
    const events = []
    const ollama = {
      async listModels() { return [MODEL] },
      async generateDiverseStructured(request) {
        order.push('generate')
        return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
      },
      async unload(model) {
        order.push(`unload:${model}`)
        if (shouldFail) throw Object.assign(new Error('Ollama kapalı'), { code: 'OLLAMA_UNAVAILABLE' })
        return { model, unloaded: true }
      },
    }
    const context = fixtureContext(job, store, ollama)
    context.emit = async (event) => { events.push(structuredClone(event)) }
    const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => {
      order.push('pipeline')
      return { snapshotId: pipelineJob.payload.candidateSnapshotId, committed: false, awaitingApproval: false, evaluation: { passed: false } }
    } })
    const result = await executor(job, context)
    assert.equal(result.accepted, 2)
    assert.deepEqual(order, ['generate', `unload:${MODEL.name}`, 'pipeline'])
    assert.ok(events.some((event) => event.type === (shouldFail ? 'teacher-feed-model-unload-warning' : 'teacher-feed-model-unloaded')))
  }
})

test('a partial schema-valid batch is preserved and only the missing lesson is retried', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 124, targetAccepted: 2, generationBatchSize: 2, trainEveryAccepted: 2, model: MODEL.name }, MODEL, 'campaign-partial')
  const job = { id: 'job-partial', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const requestedCounts = []
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      requestedCounts.push(request.schema.properties.choices.maxItems)
      return { value: { choices: choicesFor(request, { count: 1 }) }, model: MODEL.name, promptTokens: 2, completionTokens: 3 }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => ({
    snapshotId: pipelineJob.payload.candidateSnapshotId,
    committed: false,
    awaitingApproval: false,
    evaluation: { passed: false },
  }) })
  const result = await executor(job, fixtureContext(job, store, ollama))
  assert.equal(result.accepted, 2)
  assert.deepEqual(requestedCounts, [2, 1])
})

test('pause after a partial 14B result resumes from the durable candidate without replaying it', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 125, targetAccepted: 2, generationBatchSize: 2, trainEveryAccepted: 2, model: MODEL.name }, MODEL, 'campaign-partial-pause')
  const job = { id: 'job-partial-pause', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let modelCalls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      modelCalls += 1
      return { value: { choices: choicesFor(request, { count: 1 }) }, model: MODEL.name, promptTokens: 2, completionTokens: 3 }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => ({
    snapshotId: pipelineJob.payload.candidateSnapshotId,
    committed: false,
    awaitingApproval: false,
    evaluation: { passed: false },
  }) })
  const pausingContext = fixtureContext(job, store, ollama)
  pausingContext.checkpoint = async (label) => {
    if (label.endsWith(':1')) throw Object.assign(new Error('pause boundary'), { code: 'JOB_PAUSED' })
  }
  await assert.rejects(executor(job, pausingContext), { code: 'JOB_PAUSED' })
  assert.equal(modelCalls, 1)
  assert.equal(job.payload.feedState.pendingGenerationBatch.accepted.length, 1)
  assert.equal(job.payload.feedState.teacherRequests, 1)
  assert.equal(job.payload.feedState.teacherTokens, 5)
  assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0), 5)

  const resumed = await executor(job, fixtureContext(job, store, ollama))
  assert.equal(resumed.accepted, 2)
  assert.equal(modelCalls, 2)
  assert.equal(job.payload.feedState.pendingGenerationBatch, null)
  assert.equal(job.payload.feedState.teacherRequests, 2)
  assert.equal(resumed.teacherTokens, 10)
  assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0), 10)
})

test('a token quota rejection happens before Ollama I/O and preserves the deterministic seed', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 127,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    model: MODEL.name,
  }, MODEL, 'campaign-token-quota')
  const plan = planTeacherFeedBatch({ seed: 127, batchIndex: 0, campaignId: 'campaign-token-quota' })
  const firstSeed = mixSeed(plan.batchSeed, 'attempt', 0)
  const job = { id: 'job-token-quota', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  store.usageLimits.teacherTokens = 0
  const modelSeeds = []
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      modelSeeds.push(request.seed)
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 2, completionTokens: 3 }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => ({
    snapshotId: pipelineJob.payload.candidateSnapshotId,
    committed: false,
    awaitingApproval: false,
    evaluation: { passed: false },
  }) })

  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  const stopped = job.payload.feedState
  assert.deepEqual(modelSeeds, [])
  assert.equal(stopped.stopReason.code, 'TEACHER_FEED_BUDGET_STOP')
  assert.equal(stopped.teacherRequests, 0)
  assert.equal(stopped.teacherTokens, 0)
  assert.equal(stopped.promptTokens, 0)
  assert.equal(stopped.completionTokens, 0)
  assert.equal(stopped.pendingGenerationBatch.inFlightAttempt.requestUsageReserved, false)
  assert.equal(stopped.pendingGenerationBatch.inFlightAttempt.tokenUsageReserved, false)
  assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').length, 0)

  store.usageLimits.teacherTokens = Number.POSITIVE_INFINITY
  const resumed = await executor(job, fixtureContext(job, store, ollama))
  assert.equal(resumed.accepted, 1)
  assert.deepEqual(modelSeeds, [firstSeed])
  assert.equal(resumed.teacherTokens, 5)
  assert.equal(
    store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0),
    resumed.teacherTokens,
  )
})

test('a legacy ambiguous in-flight attempt is context-floored and never replayed after restart', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 126, targetAccepted: 1, generationBatchSize: 1, trainEveryAccepted: 1, model: MODEL.name }, MODEL, 'campaign-inflight')
  const plan = planTeacherFeedBatch({ seed: 126, batchIndex: 0, campaignId: 'campaign-inflight' })
  payload.feedState.dedupInitialized = true
  payload.feedState.teacherRequests = 1
  payload.feedState.pendingGenerationBatch = {
    version: 1,
    batchId: plan.batchId,
    batchIndex: 0,
    desiredCount: 1,
    accepted: [],
    rejected: [],
    calls: 1,
    promptTokens: 0,
    completionTokens: 0,
    generationErrors: 0,
    usageEstimated: false,
    lastError: null,
    inFlightAttempt: {
      attemptIndex: 0,
      attemptSeed: mixSeed(plan.batchSeed, 'attempt', 0),
      requestedCount: 1,
      maxCompletionTokens: 128,
      requestUsageReserved: true,
    },
  }
  const job = { id: 'job-inflight', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const modelSeeds = []
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      modelSeeds.push(request.seed)
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => ({
    snapshotId: pipelineJob.payload.candidateSnapshotId,
    committed: false,
    awaitingApproval: false,
    evaluation: { passed: false },
  }) })
  const result = await executor(job, fixtureContext(job, store, ollama))
  assert.deepEqual(modelSeeds, [mixSeed(plan.batchSeed, 'attempt', 1)])
  assert.equal(result.teacherRequests, 2)
  assert.ok(result.teacherTokens >= TEACHER_CONTEXT_TOKEN_LIMIT + 130)
  assert.ok(store.commits.some((commit) => commit.artifact.phase === 'generation-interrupted-attempt-recovered'))
})

test('checkpoint resumes a committed pending batch without asking Ollama again', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 9, targetAccepted: 4, generationBatchSize: 2, trainEveryAccepted: 4, model: MODEL.name }, MODEL, 'campaign-resume')
  const job = { id: 'job-resume', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let ollamaCalls = 0
  const ollama = { async listModels() { return [MODEL] }, async generateDiverseStructured(request) {
    ollamaCalls += 1
    return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
  } }
  let pipelineCalls = 0
  const failing = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    pipelineCalls += 1
    throw Object.assign(new Error('geçici pipeline hatası'), { retryable: true })
  } })
  await assert.rejects(failing(job, fixtureContext(job, store, ollama)), /geçici pipeline/u)
  assert.ok(job.payload.feedState.pendingTrainingMilestone)
  assert.equal(ollamaCalls, 2)

  const candidateSnapshotId = job.payload.feedState.pendingTrainingMilestone.candidateSnapshotId
  const candidateSourceIds = [...job.payload.feedState.pendingTrainingMilestone.sourceIds]
  store.getSnapshot = (id) => id === candidateSnapshotId
    ? { id, status: 'rejected', sourceIds: candidateSourceIds, evaluation: { passed: false } }
    : undefined
  const resumed = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    pipelineCalls += 1
    throw new Error('persisted snapshot varken pipeline tekrar çalışmamalı')
  } })
  const result = await resumed(job, fixtureContext(job, store, ollama))
  assert.equal(ollamaCalls, 2, 'resume must not regenerate atomically stored batches')
  assert.equal(pipelineCalls, 1, 'resume recognizes an already persisted candidate/rejected snapshot')
  assert.equal(result.completedTrainingCycles, 1)
  assert.equal(result.lastSnapshotId, candidateSnapshotId)
  assert.equal(job.payload.feedState.pendingTrainingMilestone, null)
})

test('training runs at accepted milestones and final remainder rebuilds all campaign sources', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 31, targetAccepted: 5, generationBatchSize: 2, trainEveryAccepted: 4, model: MODEL.name }, MODEL, 'campaign-milestones')
  const job = { id: 'job-milestones', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 2, completionTokens: 3 }
    },
  }
  const sourceCounts = []
  const candidateIds = []
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => {
    sourceCounts.push(pipelineJob.payload.sourceIds.length)
    candidateIds.push(pipelineJob.payload.candidateSnapshotId)
    return { snapshotId: pipelineJob.payload.candidateSnapshotId, committed: false, awaitingApproval: false, evaluation: { passed: false } }
  } })
  const result = await executor(job, fixtureContext(job, store, ollama))
  assert.equal(result.accepted, 5)
  assert.ok(result.generated >= 5)
  assert.equal(result.completedTrainingCycles, 2)
  assert.deepEqual(sourceCounts, [4, 5], 'the final remainder rebuild uses every accepted campaign source')
  assert.deepEqual(candidateIds, [teacherFeedCandidateSnapshotId('campaign-milestones', 1), teacherFeedCandidateSnapshotId('campaign-milestones', 2)])
  assert.ok(job.payload.feedState.nextBatchIndex >= 3)
})

test('continuous research campaign stops early only on a recomputed independent structural gate', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x14b,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 3, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-pass')
  assert.equal(payload.campaign.targetAccepted, 12)
  assert.equal(payload.campaign.continuousPlan.maximumAccepted, 12)
  assert.equal(payload.campaign.proposalOnly, true)
  const job = { id: 'job-continuous-pass', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const snapshots = new Map()
  store.getSnapshot = (id) => structuredClone(snapshots.get(id))
  let modelCalls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      modelCalls += 1
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 2, completionTokens: 3 }
    },
  }
  const pipelineJobs = []
  const executor = createTeacherFeedExecutor({
    pipelineExecutor: async (pipelineJob) => {
      pipelineJobs.push(structuredClone(pipelineJob))
      snapshots.set(pipelineJob.payload.candidateSnapshotId, {
        id: pipelineJob.payload.candidateSnapshotId,
        status: 'candidate',
        sourceIds: [...pipelineJob.payload.sourceIds],
        inputManifestHash: 'manifest-fixed',
        artifactIds: [],
        evaluation: { passed: false, runtimeVerified: true },
      })
      return {
        snapshotId: pipelineJob.payload.candidateSnapshotId,
        committed: false,
        awaitingApproval: false,
        evaluation: { passed: false },
      }
    },
    structuralBenchmarkReport: ({ snapshotId }) => passingStructuralReport(snapshotId),
  })
  const result = await executor(job, fixtureContext(job, store, ollama))
  assert.equal(result.accepted, 4, 'later blocks must not be purchased after the objective gate passes')
  assert.equal(result.continuous.completedBlocks, 1)
  assert.equal(result.stopGatePassed, true)
  assert.equal(result.continuous.lastGate.status, 'passed')
  assert.equal(result.continuous.gateSnapshotId, pipelineJobs[0].payload.candidateSnapshotId)
  assert.equal(pipelineJobs.length, 1)
  assert.equal(modelCalls, 2)
  assert.ok(store.commits.some((commit) => commit.artifact.kind === 'teacher-feed-continuous-block'))
})

test('decisive early benchmark failure closes the block honestly and resumes Ollama generation', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x170,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 3, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-early-failure')
  const job = { id: 'job-continuous-early-failure', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const snapshots = new Map()
  store.getSnapshot = (id) => structuredClone(snapshots.get(id))
  let modelCalls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      modelCalls += 1
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  const executor = createTeacherFeedExecutor({
    pipelineExecutor: async (pipelineJob) => {
      snapshots.set(pipelineJob.payload.candidateSnapshotId, {
        id: pipelineJob.payload.candidateSnapshotId,
        status: 'candidate',
        sourceIds: [...pipelineJob.payload.sourceIds],
        inputManifestHash: 'manifest-fixed',
        artifactIds: [],
      })
      return { snapshotId: pipelineJob.payload.candidateSnapshotId, committed: false, awaitingApproval: false, evaluation: { passed: false } }
    },
    structuralBenchmarkReport: ({ snapshotId }) => ({
      artifactId: 'early-failure-artifact',
      report: decisiveEarlyFailureReport(snapshotId),
    }),
  })
  const context = fixtureContext(job, store, ollama)
  context.checkpoint = async (stage) => {
    if (job.payload.feedState.continuous.completedBlocks === 1 && String(stage).startsWith('teacher-feed:generate:')) {
      throw Object.assign(new Error('first post-gate generation observed'), { code: 'JOB_PAUSED' })
    }
  }

  await assert.rejects(executor(job, context), { code: 'JOB_PAUSED' })
  assert.equal(job.payload.feedState.accepted, 4)
  assert.equal(job.payload.feedState.continuous.completedBlocks, 1)
  assert.equal(job.payload.feedState.continuous.nextBlockAt, 8)
  assert.equal(job.payload.feedState.continuous.lastGate.status, 'failed-early')
  assert.equal(job.payload.feedState.continuous.lastGate.passed, false)
  assert.equal(job.payload.feedState.continuous.lastGate.conclusive, true)
  assert.equal(job.payload.feedState.continuous.lastGate.completion, 'decisive-early-failure')
  assert.equal(job.payload.feedState.continuous.lastGate.holdoutGraphF1, 0.27)
  assert.equal(modelCalls, 2, 'the first block was generated once and the next block was entered without replaying it')
})

test('continuous campaign freezes selected approved corpus without counting it as newly accepted', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x14a,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 2, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-bootstrap')
  const job = { id: 'job-continuous-bootstrap', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const priorSource = {
    id: 'prior-human-source',
    content: 'Deniz sabah mutfakta ekmeği kesti.',
    contentHash: 'prior-human-source-hash',
    status: 'approved',
    selected: true,
    sourceType: 'manual',
    provenance: {
      kind: 'human-authored',
      independenceGroupId: 'human:prior-corpus',
      evaluationEligible: true,
    },
  }
  store.sources.set(priorSource.id, structuredClone(priorSource))
  const snapshots = new Map()
  store.getSnapshot = (id) => structuredClone(snapshots.get(id))
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  const pipelineJobs = []
  const executor = createTeacherFeedExecutor({
    pipelineExecutor: async (pipelineJob) => {
      pipelineJobs.push(structuredClone(pipelineJob))
      snapshots.set(pipelineJob.payload.candidateSnapshotId, {
        id: pipelineJob.payload.candidateSnapshotId,
        status: 'candidate',
        sourceIds: [...pipelineJob.payload.sourceIds],
        inputManifestHash: 'manifest-fixed',
        artifactIds: [],
      })
      return {
        snapshotId: pipelineJob.payload.candidateSnapshotId,
        committed: false,
        awaitingApproval: false,
        evaluation: { passed: false },
      }
    },
    structuralBenchmarkReport: ({ snapshotId }) => passingStructuralReport(snapshotId),
  })

  const result = await executor(job, fixtureContext(job, store, ollama))

  assert.equal(result.accepted, 4, 'bootstrap corpus must not inflate newly accepted examples')
  assert.deepEqual(result.feedState.bootstrapSourceIds, [priorSource.id])
  assert.match(result.feedState.bootstrapSourceFingerprint, /^[a-f0-9]{64}$/u)
  assert.equal(pipelineJobs.length, 1)
  assert.equal(pipelineJobs[0].payload.sourceIds.length, 5)
  assert.ok(pipelineJobs[0].payload.sourceIds.includes(priorSource.id))
  assert.deepEqual(store.sources.get(priorSource.id), priorSource, 'frontier must preserve source content and provenance')
  const frontier = store.commits.find((commit) => commit.artifact.phase === 'corpus-frontier-frozen')
  assert.ok(frontier, 'the frozen frontier must be durable before generation')
  assert.equal(frontier.checkpoint.accepted, 0)
  assert.deepEqual(frontier.checkpoint.sourceIds, [priorSource.id])
})

test('continuous campaign never treats missing benchmark evidence as success and pauses at its hard block bound', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x14c,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 2, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-bounded')
  const job = { id: 'job-continuous-bounded', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  let pipelineCalls = 0
  const executor = createTeacherFeedExecutor({
    pipelineExecutor: async (pipelineJob) => {
      pipelineCalls += 1
      return { snapshotId: pipelineJob.payload.candidateSnapshotId, committed: false, awaitingApproval: false, evaluation: { passed: true } }
    },
    structuralBenchmarkReport: () => null,
  })
  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  assert.equal(job.payload.feedState.accepted, 8)
  assert.equal(job.payload.feedState.continuous.completedBlocks, 2)
  assert.equal(job.payload.feedState.continuous.gatePassed, false)
  assert.equal(job.payload.feedState.continuous.lastGate.status, 'missing')
  assert.equal(job.payload.feedState.stopReason.code, 'CONTINUOUS_MAX_BLOCKS_REACHED')
  assert.equal(pipelineCalls, 2)
  assert.equal(store.commits.filter((commit) => commit.artifact.kind === 'teacher-feed-continuous-block').length, 2)
})

test('continuous benchmark control interruption propagates instead of becoming failed evidence', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x14f,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 2, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-benchmark-pause')
  const job = { id: 'job-continuous-benchmark-pause', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  const executor = createTeacherFeedExecutor({
    pipelineExecutor: async (pipelineJob) => ({
      snapshotId: pipelineJob.payload.candidateSnapshotId,
      committed: false,
      awaitingApproval: false,
      evaluation: { passed: false },
    }),
    structuralBenchmarkReport: async () => {
      throw Object.assign(new Error('benchmark duraklatıldı'), { code: 'JOB_PAUSED' })
    },
  })

  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  assert.equal(job.payload.feedState.continuous.completedBlocks, 0)
  assert.equal(store.commits.some((commit) => commit.artifact.kind === 'teacher-feed-continuous-block'), false)
})

test('continuous benchmark resource and worker safety failures propagate without advancing the block', async () => {
  const safetyCodes = [
    'RAM_BUDGET_EXCEEDED',
    'DISK_RESOURCE_RESERVE_EXCEEDED',
    'DISK_BUDGET_EXCEEDED',
    'ARTIFACT_PAYLOAD_TOO_LARGE',
    'NEURON_RESIDENCY_RESOURCE_WAIT',
    'JOB_WORKER_UNAVAILABLE',
    'JOB_WORKER_SPAWN_FAILED',
    'JOB_WORKER_HANDSHAKE_TIMEOUT',
  ]
  for (const [index, code] of safetyCodes.entries()) {
    const payload = normalizeTeacherFeedCampaignRequest({
      seed: 0x160 + index,
      targetAccepted: 4,
      generationBatchSize: 2,
      trainEveryAccepted: 4,
      model: MODEL.name,
      continuousPlan: { blockAccepted: 4, maxBlocks: 2, minimumDiskHeadroomMb: 4 },
    }, MODEL, `campaign-benchmark-safety-${index}`)
    const job = { id: `job-benchmark-safety-${index}`, type: 'teacher-feed-campaign', payload }
    const store = fixtureStore(job)
    let generationCalls = 0
    let pipelineCalls = 0
    const ollama = {
      async listModels() { return [MODEL] },
      async generateDiverseStructured(request) {
        generationCalls += 1
        return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
      },
    }
    const executor = createTeacherFeedExecutor({
      pipelineExecutor: async (pipelineJob) => {
        pipelineCalls += 1
        return {
          snapshotId: pipelineJob.payload.candidateSnapshotId,
          committed: false,
          awaitingApproval: false,
          evaluation: { passed: false },
        }
      },
      structuralBenchmarkReport: async () => {
        throw Object.assign(new Error(`benchmark safety: ${code}`), { code })
      },
    })

    await assert.rejects(executor(job, fixtureContext(job, store, ollama)), (error) => error?.code === code)
    assert.equal(job.payload.feedState.accepted, 4, code)
    assert.equal(job.payload.feedState.continuous.completedBlocks, 0, code)
    assert.equal(generationCalls, 2, code)
    assert.equal(pipelineCalls, 1, code)
    assert.equal(store.commits.some((commit) => commit.artifact.kind === 'teacher-feed-continuous-block'), false, code)
  }
})

test('continuous campaign resumes from its durable block boundary without regenerating the completed block', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x14e,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 3, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-resume')
  const job = { id: 'job-continuous-resume', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const snapshots = new Map()
  store.getSnapshot = (id) => structuredClone(snapshots.get(id))
  let generationCalls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      generationCalls += 1
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  const pipelineExecutor = async (pipelineJob) => {
    snapshots.set(pipelineJob.payload.candidateSnapshotId, {
      id: pipelineJob.payload.candidateSnapshotId,
      status: 'candidate',
      sourceIds: [...pipelineJob.payload.sourceIds],
      inputManifestHash: 'manifest-fixed',
      artifactIds: [],
    })
    return { snapshotId: pipelineJob.payload.candidateSnapshotId, committed: false, awaitingApproval: false, evaluation: { passed: false } }
  }
  const first = createTeacherFeedExecutor({
    pipelineExecutor,
    structuralBenchmarkReport: () => null,
  })
  const interruptedContext = fixtureContext(job, store, ollama)
  interruptedContext.checkpoint = async (stage) => {
    if (job.payload.feedState.continuous.completedBlocks === 1 && String(stage).startsWith('teacher-feed:generate:')) {
      throw Object.assign(new Error('güvenli blok sınırında servis durdu'), { code: 'JOB_PAUSED' })
    }
  }
  await assert.rejects(first(job, interruptedContext), { code: 'JOB_PAUSED' })
  assert.equal(job.payload.feedState.accepted, 4)
  assert.equal(job.payload.feedState.continuous.completedBlocks, 1)
  assert.equal(generationCalls, 2)

  const resumed = createTeacherFeedExecutor({
    pipelineExecutor,
    structuralBenchmarkReport: ({ blockIndex, snapshotId }) => blockIndex === 2 ? passingStructuralReport(snapshotId) : null,
  })
  const result = await resumed(job, fixtureContext(job, store, ollama))
  assert.equal(result.accepted, 8)
  assert.equal(result.continuous.completedBlocks, 2)
  assert.equal(result.stopGatePassed, true)
  assert.equal(generationCalls, 4, 'the four sources from the completed first block must not be regenerated')
})

test('continuous campaign pauses before model I/O when logical disk headroom is exhausted', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x14d,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 2, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-disk')
  const job = { id: 'job-continuous-disk', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  store.approximateStorageBytes = () => 63 * 1024 * 1024
  store.getEffectiveDiskBudgetBytes = () => 64 * 1024 * 1024
  let modelAccessed = false
  const ollama = {
    async listModels() { modelAccessed = true; return [MODEL] },
    async generateDiverseStructured() { modelAccessed = true; throw new Error('unreachable') },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => { throw new Error('unreachable') } })
  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  assert.equal(modelAccessed, false)
  assert.equal(job.payload.feedState.stopReason.code, 'CONTINUOUS_DISK_HEADROOM_LOW')
})

test('continuous campaign pauses before model I/O when physical block admission is unsafe', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 0x14c,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 2, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-continuous-physical-reserve')
  const job = { id: 'job-continuous-physical-reserve', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let modelAccessed = false
  const ollama = {
    async listModels() { modelAccessed = true; return [MODEL] },
    async generateDiverseStructured() { modelAccessed = true; throw new Error('unreachable') },
  }
  const context = fixtureContext(job, store, ollama)
  const inspectCalls = []
  context.resourceGovernor = {
    inspect(options) {
      inspectCalls.push(options)
      return {
        sample: { diskFreeBytes: 8 * 1024 * 1024 * 1024 },
        state: {
          canStart: false,
          allowed: false,
          reasonCodes: ['PHYSICAL_DISK_START_RESERVE_LOW'],
          usableDiskFreeBytes: 7.5 * 1024 * 1024 * 1024,
        },
      }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => { throw new Error('unreachable') } })
  await assert.rejects(executor(job, context), { code: 'JOB_PAUSED' })
  assert.deepEqual(inspectCalls, [{ admission: true }])
  assert.equal(modelAccessed, false)
  assert.equal(job.payload.feedState.stopReason.code, 'CONTINUOUS_PHYSICAL_RESOURCE_RESERVE_LOW')
  assert.deepEqual(job.payload.feedState.stopReason.resourceState.reasonCodes, ['PHYSICAL_DISK_START_RESERVE_LOW'])
})

test('a pending training milestone excludes sources deselected before resume', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 32,
    targetAccepted: 2,
    generationBatchSize: 2,
    trainEveryAccepted: 2,
    model: MODEL.name,
  }, MODEL, 'campaign-deselected-milestone')
  const job = { id: 'job-deselected-milestone', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  let pipelineCalls = 0
  const first = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    pipelineCalls += 1
    throw new Error('kilometre taşını yeniden başlatmak için durdur')
  } })
  await assert.rejects(first(job, fixtureContext(job, store, ollama)), /yeniden başlatmak/u)
  assert.ok(job.payload.feedState.pendingTrainingMilestone)
  for (const source of store.sources.values()) source.selected = false

  const resumed = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    pipelineCalls += 1
    throw new Error('seçimden çıkarılan kaynak pipeline\'a girmemeli')
  } })
  const result = await resumed(job, fixtureContext(job, store, ollama))
  assert.equal(pipelineCalls, 1)
  assert.equal(result.completedTrainingCycles, 0)
  assert.equal(job.payload.feedState.pendingTrainingMilestone, null)
})

test('a cached milestone snapshot with a deselected source is rejected and deterministically replanned', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 33,
    targetAccepted: 2,
    generationBatchSize: 2,
    trainEveryAccepted: 2,
    model: MODEL.name,
  }, MODEL, 'campaign-stale-cached-milestone')
  const job = { id: 'job-stale-cached-milestone', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      return { value: { choices: choicesFor(request) }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  let pipelineCalls = 0
  const first = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    pipelineCalls += 1
    throw new Error('aday snapshot yazıldıktan sonraki kesinti')
  } })
  await assert.rejects(first(job, fixtureContext(job, store, ollama)), /sonraki kesinti/u)
  const pending = job.payload.feedState.pendingTrainingMilestone
  assert.ok(pending)

  const deselectedId = pending.sourceIds[0]
  store.sources.get(deselectedId).selected = false
  let cachedSnapshot = {
    id: pending.candidateSnapshotId,
    status: 'candidate',
    sourceIds: [...pending.sourceIds],
    evaluation: { passed: true },
  }
  store.getSnapshot = (id) => id === cachedSnapshot.id ? structuredClone(cachedSnapshot) : undefined
  store.rejectSnapshot = (id, evaluation) => {
    if (id !== cachedSnapshot.id) return undefined
    cachedSnapshot = { ...cachedSnapshot, status: 'rejected', evaluation: structuredClone(evaluation) }
    return structuredClone(cachedSnapshot)
  }

  let replannedJob = null
  const resumed = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => {
    pipelineCalls += 1
    replannedJob = structuredClone(pipelineJob)
    return {
      snapshotId: pipelineJob.payload.candidateSnapshotId,
      committed: false,
      awaitingApproval: false,
      evaluation: { passed: false },
    }
  } })
  const result = await resumed(job, fixtureContext(job, store, ollama))
  assert.equal(pipelineCalls, 2)
  assert.equal(cachedSnapshot.status, 'rejected')
  assert.equal(cachedSnapshot.evaluation.passed, false)
  assert.equal(cachedSnapshot.evaluation.invalidated, true)
  assert.equal(cachedSnapshot.evaluation.code, 'SOURCE_SELECTION_CHANGED')
  assert.equal(cachedSnapshot.evaluation.selectionChange, 'DESELECTED')
  assert.deepEqual(cachedSnapshot.evaluation.snapshotSourceIds, pending.sourceIds)
  assert.deepEqual(cachedSnapshot.evaluation.selectedApprovedSourceIds, pending.sourceIds.slice(1))
  assert.deepEqual(replannedJob.payload.sourceIds, pending.sourceIds.slice(1))
  assert.notEqual(replannedJob.payload.candidateSnapshotId, cachedSnapshot.id)
  assert.match(replannedJob.payload.candidateSnapshotId, new RegExp(`^${cachedSnapshot.id}-[a-f0-9]{16}$`, 'u'))
  assert.equal(result.lastSnapshotId, replannedJob.payload.candidateSnapshotId)
  assert.equal(result.completedTrainingCycles, 1)
})

test('failed structured outputs are charged and durably counted before campaign failure', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 93,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    maxGenerationCallsPerBatch: 2,
    model: MODEL.name,
  }, MODEL, 'campaign-invalid-output')
  const job = { id: 'job-invalid-output', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let generationCalls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured() {
      generationCalls += 1
      const error = new Error('Şema dışı çıktı')
      error.code = 'SCHEMA_VALIDATION_FAILED'
      error.promptTokens = 4
      error.completionTokens = 2
      throw error
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => { throw new Error('unreachable') } })
  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'SCHEMA_VALIDATION_FAILED' })
  assert.equal(generationCalls, 2)
  assert.equal(job.payload.feedState.teacherRequests, 2)
  assert.equal(job.payload.feedState.teacherTokens, 12)
  assert.equal(job.payload.feedState.promptTokens, 8)
  assert.equal(job.payload.feedState.completionTokens, 4)
  assert.equal(store.usage.filter((item) => item.kind === 'teacherRequests').length, 2)
  assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0), 12)
  assert.ok(store.commits.some((commit) => commit.artifact.phase === 'generation-attempts-failed'))
})

test('a missing error token component is conservatively completed from the persisted bound', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 931,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    maxGenerationCallsPerBatch: 2,
    model: MODEL.name,
  }, MODEL, 'campaign-partial-error-usage')
  const job = { id: 'job-partial-error-usage', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured() {
      const error = new Error('Eksik sayaçlı şema hatası')
      error.code = 'OLLAMA_INVALID_JSON'
      error.promptTokens = 7
      throw error
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => { throw new Error('unreachable') } })

  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  const state = job.payload.feedState
  assert.equal(state.teacherRequests, 2)
  assert.equal(state.promptTokens, 14)
  assert.equal(state.completionTokens, 256)
  assert.equal(state.teacherTokens, 270)
  assert.equal(state.pendingGenerationBatch.usageEstimated, true)
  assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0), 270)
  const failed = store.commits.filter((commit) => commit.artifact.phase === 'generation-attempts-failed')
  assert.equal(failed.length, 2)
  assert.ok(failed.every((commit) => commit.artifact.payload.usageEstimated === true))
})

test('a successful response with missing counters is conservatively charged before acceptance', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 932,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    model: MODEL.name,
  }, MODEL, 'campaign-success-missing-usage')
  const job = { id: 'job-success-missing-usage', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      return { value: { choices: choicesFor(request) }, model: MODEL.name }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => ({
    snapshotId: pipelineJob.payload.candidateSnapshotId,
    committed: false,
    awaitingApproval: false,
    evaluation: { passed: false },
  }) })

  const result = await executor(job, fixtureContext(job, store, ollama))
  const state = job.payload.feedState
  assert.equal(result.accepted, 1)
  assert.equal(state.teacherRequests, 1)
  assert.ok(state.promptTokens > 0)
  assert.equal(state.completionTokens, 128)
  assert.equal(state.teacherTokens, state.promptTokens + state.completionTokens)
  assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0), state.teacherTokens)
  const batch = store.commits.find((commit) => commit.artifact.phase === 'batch-stored')
  assert.equal(batch.artifact.payload.usageEstimated, true)
})

test('coercible non-numeric response counters fall back to the persisted conservative bounds', async () => {
  for (const [index, invalidCounter] of [null, '', false].entries()) {
    const campaignId = `campaign-invalid-counter-${index}`
    const payload = normalizeTeacherFeedCampaignRequest({
      seed: 940 + index,
      targetAccepted: 1,
      generationBatchSize: 1,
      trainEveryAccepted: 1,
      model: MODEL.name,
    }, MODEL, campaignId)
    const job = { id: `job-invalid-counter-${index}`, type: 'teacher-feed-campaign', payload }
    const store = fixtureStore(job)
    const ollama = {
      async listModels() { return [MODEL] },
      async generateDiverseStructured(request) {
        return {
          value: { choices: choicesFor(request) },
          model: MODEL.name,
          promptTokens: invalidCounter,
          completionTokens: invalidCounter,
          promptTokensReported: true,
          completionTokensReported: true,
        }
      },
    }
    const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => ({
      snapshotId: pipelineJob.payload.candidateSnapshotId,
      committed: false,
      awaitingApproval: false,
      evaluation: { passed: false },
    }) })

    const result = await executor(job, fixtureContext(job, store, ollama))
    const state = job.payload.feedState
    assert.equal(result.accepted, 1)
    assert.ok(state.promptTokens > 0)
    assert.equal(state.completionTokens, 128)
    assert.equal(state.teacherTokens, state.promptTokens + state.completionTokens)
    assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0), state.teacherTokens)
    const batch = store.commits.find((commit) => commit.artifact.phase === 'batch-stored')
    assert.equal(batch.artifact.payload.usageEstimated, true)
  }
})

test('campaign batch rejection summary records an out-of-board model choice', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 94,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    maxGenerationCallsPerBatch: 2,
    model: MODEL.name,
  }, MODEL, 'campaign-spelling-summary')
  const job = { id: 'job-spelling-summary', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let call = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      call += 1
      const allowed = request.schema.properties.choices.items.enum
      const choices = call === 1 ? [Math.max(...allowed) + 1] : choicesFor(request)
      return { value: { choices }, model: MODEL.name, promptTokens: 1, completionTokens: 1 }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (pipelineJob) => ({
    snapshotId: pipelineJob.payload.candidateSnapshotId,
    committed: false,
    awaitingApproval: false,
    evaluation: { passed: false },
  }) })
  const result = await executor(job, fixtureContext(job, store, ollama))
  assert.equal(result.accepted, 1)
  assert.equal(result.rejectedQuality, 1)
  const batch = store.commits.find((commit) => commit.artifact.phase === 'batch-stored')
  assert.equal(batch.artifact.payload.rejectionReasons['unknown-choice-id'], 1)
  assert.equal(batch.artifact.payload.rejectedDiagnostics[0].text, '')
  assert.ok(batch.artifact.payload.rejectedDiagnostics[0].reasons.includes('unknown-choice-id'))
  assert.equal(batch.artifact.payload.rejectedDiagnostics[0].surfaceAssembly, null)
})

test('frozen Ollama model drift creates a durable safe pause before generation', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 5, targetAccepted: 2, generationBatchSize: 2, trainEveryAccepted: 2, model: MODEL.name }, MODEL, 'campaign-stale')
  const job = { id: 'job-stale', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let generated = false
  const ollama = {
    async listModels() { return [{ ...MODEL, digest: 'sha256:changed-model' }] },
    async generateDiverseStructured() { generated = true; throw new Error('unreachable') },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => ({ committed: false }) })
  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  assert.equal(generated, false)
  assert.equal(job.payload.feedState.stopReason.code, 'STALE_TEACHER_MODEL')
  assert.equal(job.payload.feedState.accepted, 0)
  assert.ok(store.commits.some((commit) => commit.artifact.kind === 'teacher-feed-checkpoint' && commit.artifact.phase === 'safety-stop'))
})

test('a successful response from a different model is charged but never accepted', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 211,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    model: MODEL.name,
  }, MODEL, 'campaign-response-model-drift')
  const job = { id: 'job-response-model-drift', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let modelCalls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      modelCalls += 1
      return {
        value: { choices: choicesFor(request) },
        model: 'unexpected-model:14b',
        promptTokens: 2,
        completionTokens: 3,
      }
    },
  }
  let pipelineCalls = 0
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    pipelineCalls += 1
    throw new Error('model kimliği uyuşmazken pipeline çalışmamalı')
  } })

  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  const stopped = job.payload.feedState
  assert.equal(modelCalls, 1)
  assert.equal(pipelineCalls, 0)
  assert.equal(store.sources.size, 0)
  assert.equal(stopped.accepted, 0)
  assert.equal(stopped.teacherRequests, 1)
  assert.equal(stopped.teacherTokens, 5)
  assert.equal(stopped.pendingGenerationBatch.accepted.length, 0)
  assert.equal(stopped.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(stopped.stopReason.code, 'STALE_TEACHER_MODEL')
  assert.equal(stopped.stopReason.expectedModel, MODEL.name)
  assert.equal(stopped.stopReason.actualModel, 'unexpected-model:14b')
  assert.equal(store.usage.filter((item) => item.kind === 'teacherTokens').reduce((sum, item) => sum + item.amount, 0), 5)
})

test('cancel during a successful Ollama call records tokens but discards returned lessons', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  t.after(() => store.close())
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 212,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    model: MODEL.name,
  }, MODEL, 'campaign-cancel-during-call')
  const { job: queued } = store.createJob({
    id: 'job-cancel-during-call',
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload,
  })
  const job = store.claimNextJob('teacher-cancel-worker', 30_000, 60_000)
  assert.equal(job.id, queued.id)
  let pipelineCalls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured(request) {
      store.updateJob(job.id, { state: 'cancel_requested', stage: 'cancel-requested' })
      return {
        value: { choices: choicesFor(request) },
        model: MODEL.name,
        promptTokens: 2,
        completionTokens: 3,
      }
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    pipelineCalls += 1
    throw new Error('cancelled model output must never reach the training pipeline')
  } })
  const context = {
    jobId: job.id,
    jobClaim: { owner: job.leaseOwner, attempt: job.attempt },
    store,
    ollama,
    config: store.getConfig(),
    signal: new AbortController().signal,
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }

  await assert.rejects(executor(job, context), { code: 'JOB_CANCELLED' })
  const current = store.getJob(job.id)
  assert.equal(current.state, 'cancel_requested')
  assert.equal(current.payload.feedState.accepted, 0)
  assert.equal(current.payload.feedState.teacherRequests, 1)
  assert.equal(current.payload.feedState.teacherTokens, 5)
  assert.equal(current.payload.feedState.pendingGenerationBatch.accepted.length, 0)
  assert.equal(current.payload.feedState.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(store.getUsage().teacherRequests, 1)
  assert.equal(store.getUsage().teacherTokens, 5)
  assert.equal(store.listSources().length, 0)
  assert.equal(pipelineCalls, 0)
  assert.ok(store.listArtifacts().some((artifact) => artifact.kind === 'teacher-feed-checkpoint'))
})

test('aborted Ollama call with no response charges the persisted conservative token bound', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  t.after(() => store.close())
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 213,
    targetAccepted: 1,
    generationBatchSize: 1,
    trainEveryAccepted: 1,
    model: MODEL.name,
  }, MODEL, 'campaign-abort-no-usage')
  const { job: queued } = store.createJob({
    id: 'job-abort-no-usage',
    type: 'teacher-feed-campaign',
    maxAttempts: 2,
    payload,
  })
  const job = store.claimNextJob('teacher-abort-worker', 30_000, 60_000)
  assert.equal(job.id, queued.id)
  const controller = new AbortController()
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured() {
      store.updateJob(job.id, { state: 'cancel_requested', stage: 'cancel-requested' })
      const error = Object.assign(new Error('İş kullanıcı tarafından iptal edildi.'), { code: 'JOB_CANCELLED' })
      controller.abort(error)
      throw error
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => {
    throw new Error('aborted output must never reach the training pipeline')
  } })
  const context = {
    jobId: job.id,
    jobClaim: { owner: job.leaseOwner, attempt: job.attempt },
    store,
    ollama,
    config: store.getConfig(),
    signal: controller.signal,
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }

  await assert.rejects(executor(job, context), { code: 'JOB_CANCELLED' })
  const state = store.getJob(job.id).payload.feedState
  assert.equal(state.accepted, 0)
  assert.equal(state.teacherRequests, 1)
  assert.ok(state.promptTokens > 0)
  assert.equal(state.completionTokens, 128)
  assert.equal(state.teacherTokens, state.promptTokens + state.completionTokens)
  assert.equal(state.pendingGenerationBatch.inFlightAttempt, null)
  assert.equal(state.pendingGenerationBatch.usageEstimated, true)
  assert.equal(store.getUsage().teacherTokens, state.teacherTokens)
  assert.equal(store.listSources().length, 0)
})

test('Ollama timeout is counted once and durably pauses instead of repeating the expensive call', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 95, targetAccepted: 1, generationBatchSize: 1, trainEveryAccepted: 1, model: MODEL.name }, MODEL, 'campaign-timeout')
  const job = { id: 'job-timeout', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let calls = 0
  const ollama = {
    async listModels() { return [MODEL] },
    async generateDiverseStructured() {
      calls += 1
      const error = new Error('Ollama zaman aşımı')
      error.code = 'OLLAMA_TIMEOUT'
      error.timeoutMs = 300_000
      error.completionTokens = 128
      error.usageEstimated = true
      throw error
    },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => { throw new Error('unreachable') } })
  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  assert.equal(calls, 1)
  assert.equal(job.payload.feedState.teacherRequests, 1)
  assert.ok(job.payload.feedState.promptTokens > 0)
  assert.equal(job.payload.feedState.completionTokens, 128)
  assert.equal(
    job.payload.feedState.teacherTokens,
    job.payload.feedState.promptTokens + job.payload.feedState.completionTokens,
  )
  assert.equal(job.payload.feedState.stopReason.code, 'TEACHER_FEED_OLLAMA_TIMEOUT')
  const failed = store.commits.find((commit) => commit.artifact.phase === 'generation-attempts-failed')
  assert.equal(failed.artifact.payload.usageEstimated, true)
  assert.ok(store.commits.some((commit) => commit.artifact.phase === 'safety-stop'))
})

test('stale teacher protocol creates a durable safe pause before model access', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({ seed: 6, targetAccepted: 2, generationBatchSize: 2, trainEveryAccepted: 2, model: MODEL.name }, MODEL, 'campaign-stale-protocol')
  payload.campaign.protocolFingerprint = 'stale-protocol'
  payload.feedState.protocolFingerprint = 'stale-protocol'
  const job = { id: 'job-stale-protocol', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let modelAccessed = false
  const ollama = {
    async listModels() { modelAccessed = true; return [MODEL] },
    async generateDiverseStructured() { modelAccessed = true; throw new Error('unreachable') },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => ({ committed: false }) })
  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  assert.equal(modelAccessed, false)
  assert.equal(job.payload.feedState.stopReason.code, 'STALE_TEACHER_PROTOCOL')
  assert.equal(job.payload.feedState.stopReason.expectedProtocolFingerprint, TEACHER_FEED_PROTOCOL_FINGERPRINT)
  assert.equal(job.payload.feedState.stopReason.actualProtocolFingerprint, 'stale-protocol')
  assert.ok(store.commits.some((commit) => commit.artifact.kind === 'teacher-feed-checkpoint' && commit.artifact.phase === 'safety-stop'))
})

test('stale continuous structural gate protocol pauses before model access', async () => {
  const payload = normalizeTeacherFeedCampaignRequest({
    seed: 7,
    targetAccepted: 4,
    generationBatchSize: 2,
    trainEveryAccepted: 4,
    model: MODEL.name,
    continuousPlan: { blockAccepted: 4, maxBlocks: 2, minimumDiskHeadroomMb: 4 },
  }, MODEL, 'campaign-stale-structural-gate')
  payload.campaign.continuousPlan.stopGate.benchmarkFingerprint = 'stale-structural-gate'
  const job = { id: 'job-stale-structural-gate', type: 'teacher-feed-campaign', payload }
  const store = fixtureStore(job)
  let modelAccessed = false
  const ollama = {
    async listModels() { modelAccessed = true; return [MODEL] },
    async generateDiverseStructured() { modelAccessed = true; throw new Error('unreachable') },
  }
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async () => ({ committed: false }) })
  await assert.rejects(executor(job, fixtureContext(job, store, ollama)), { code: 'JOB_PAUSED' })
  assert.equal(modelAccessed, false)
  assert.equal(job.payload.feedState.stopReason.code, 'STALE_TEACHER_PROTOCOL')
  assert.equal(job.payload.feedState.stopReason.actualStructuralBenchmarkFingerprint, 'stale-structural-gate')
  assert.equal(job.payload.feedState.stopReason.expectedStructuralBenchmarkFingerprint, INDEPENDENT_STRUCTURAL_BENCHMARK_FINGERPRINT)
})

test('non-campaign jobs pass through unchanged', async () => {
  const sentinel = { ok: true }
  let received
  const executor = createTeacherFeedExecutor({ pipelineExecutor: async (job, context) => { received = { job, context }; return sentinel } })
  const job = { type: 'full-cycle', payload: {} }
  const context = { marker: 1 }
  assert.equal(await executor(job, context), sentinel)
  assert.deepEqual(received, { job, context })
})
