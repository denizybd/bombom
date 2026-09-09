import test from 'node:test'
import assert from 'node:assert/strict'
import { createPipelineExecutor, DEFAULT_STAGE_HANDLERS, evaluateCandidate, generateTeacherSources } from '../pipeline.mjs'
import { defaultConfig } from '../config.mjs'
import { createTrainingStore } from '../store.mjs'
import { normalizeTextSource } from '../source-policy.mjs'
import { temporaryDirectory } from './helpers.mjs'
import { currentPlasticRegionBase } from '../../runtime/plastic-region-cache.mjs'

test('evaluation gate blocks regression even if the absolute score passes', () => {
  const policy = defaultConfig('/tmp/no-write').evaluation
  const result = evaluateCandidate({ overallScore: 0.7, syntaxContrastAccuracy: 0.8, regressionRate: 0.1, memorizationGap: 0.1 }, { overallScore: 0.69 }, policy)
  assert.equal(result.passed, false)
  assert.equal(result.checks.find((check) => check.name === 'regression-rate').passed, false)
})

test('evaluation gate accepts a non-regressing unseen candidate', () => {
  const policy = defaultConfig('/tmp/no-write').evaluation
  const result = evaluateCandidate({ overallScore: 0.72, syntaxContrastAccuracy: 0.75, regressionRate: 0.01, memorizationGap: 0.2 }, { overallScore: 0.7 }, policy)
  assert.equal(result.passed, true)
})

test('legacy teacher validation and a pre-aborted signal consume no quota', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  let modelCalls = 0
  const ollama = { async generateStructured() { modelCalls += 1; throw new Error('must not run') } }
  const input = { topic: 'temel eylemler', count: 1, language: 'tr' }

  await assert.rejects(generateTeacherSources(input, { store, ollama }), { code: 'OLLAMA_MODEL_REQUIRED' })
  assert.equal(modelCalls, 0)
  assert.equal(store.getUsage().teacherRequests, 0)
  assert.equal(store.getUsage().teacherTokens, 0)

  store.updateConfig({ ollama: { model: 'teacher:14b' } })
  const controller = new AbortController()
  const abortError = Object.assign(new Error('önceden iptal'), { code: 'JOB_CANCELLED' })
  controller.abort(abortError)
  await assert.rejects(generateTeacherSources(input, { store, ollama, signal: controller.signal }), (error) => error === abortError)
  assert.equal(modelCalls, 0)
  assert.equal(store.getUsage().teacherRequests, 0)
  assert.equal(store.getUsage().teacherTokens, 0)
})

test('legacy multi-part teacher generation yields so a disconnect abort wins before the next reservation', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  store.updateConfig({
    ollama: { model: 'teacher:14b' },
    budgets: { maxDailyTeacherRequests: 10, maxDailyTeacherTokens: 1_000_000 },
  })
  const controller = new AbortController()
  const disconnect = Object.assign(new Error('istemci kapandı'), { code: 'CLIENT_DISCONNECTED' })
  let modelCalls = 0
  const ollama = {
    async generateStructured({ schema }) {
      modelCalls += 1
      if (modelCalls === 1) setTimeout(() => controller.abort(disconnect), 0)
      const requested = schema.properties.lessons.maxItems
      return {
        value: {
          lessons: Array.from({ length: requested }, (_, index) => ({
            text: `Kişi ${index + 1} aracı dikkatlice kullandı.`,
            kind: 'paraphrase',
            rationale: 'Parçalar arası bağlantı kesintisi deneyi',
          })),
        },
        model: 'teacher:14b',
        promptTokens: 10,
        completionTokens: 5,
      }
    },
  }

  await assert.rejects(generateTeacherSources({
    topic: 'parçalı yerel üretim', count: 48, language: 'tr',
  }, { store, ollama, signal: controller.signal }), (error) => error === disconnect)
  assert.equal(modelCalls, 1)
  assert.equal(store.getUsage().teacherRequests, 1)
  assert.equal(store.getUsage().teacherTokens, 15)
  assert.equal(store.listSources().length, 0)
})

test('proposal mode keeps a verified passing candidate pending; controlled mode may commit', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const created = store.createSource(normalizeTextSource({ text: 'Ali ekmeği dikkatlice kesti.' }, store.getConfig())).source
  store.decideSource(created.id, 'approved')
  const verified = async () => ({ verified: true })
  const executor = createPipelineExecutor({ stageHandlers: {
    textLab: verified, synthesis: verified, families: verified, languageExperience: verified, conceptMacros: verified,
    evaluate: async () => ({ verified: true, metrics: {
      holdoutTop1: 0.8, holdoutTop5: 0.94, grammaticalPreference: 0.8,
      regressionRate: 0, memorizationGap: 0.1, unseenCombinationAccuracy: 0.75,
      familyUtility: 0.22, neuronCount: 150, familyCount: 12,
      evaluationSampleCount: 48, corpusFingerprint: 'corpus-v1',
    } }),
  } })
  const context = {
    jobId: 'pipeline-test', store, signal: new AbortController().signal,
    checkpoint: async () => {}, progress: async () => {}, emit: async () => {},
  }
  const proposal = await executor({ payload: {}, type: 'full-cycle' }, context)
  assert.equal(proposal.awaitingApproval, true)
  assert.equal(store.getSnapshot(proposal.snapshotId).status, 'candidate')
  assert.equal(store.getSnapshot(proposal.snapshotId).metrics.holdoutTop5, 0.94)
  assert.equal(store.getSnapshot(proposal.snapshotId).metrics.corpusFingerprint, 'corpus-v1')
  assert.equal(store.getActiveSnapshot(), undefined)

  store.updateConfig({ automation: { level: 'controlled' } })
  const stillProposal = await executor({ payload: { reprocess: true, automationLevel: 'proposal' }, type: 'full-cycle' }, context)
  assert.equal(stillProposal.awaitingApproval, true, 'a running proposal must not be escalated by a later config edit')
  assert.equal(store.getActiveSnapshot(), undefined)

  const controlled = await executor({ payload: { reprocess: true, automationLevel: 'controlled' }, type: 'full-cycle' }, context)
  assert.equal(controlled.committed, true)
  assert.equal(store.getActiveSnapshot().id, controlled.snapshotId)
})

test('controlled automation cannot activate an adaptive candidate without a region release chain', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({ text: 'Adaptive aday release kapısı deneyi.' }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  const config = defaultConfig('/tmp/adaptive-auto-activation-gate')
  config.plasticRegions = {
    ...config.plasticRegions,
    mode: 'off',
    representationProfile: 'adaptive-eighth-id-v1',
  }
  const verified = async () => ({ verified: true })
  const executor = createPipelineExecutor({ stageHandlers: {
    textLab: async (_job, context, state) => {
      const artifact = context.store.createArtifact({
        kind: 'text-lab-analysis', contentHash: 'adaptive-analysis-content',
        payload: {
          format: 'bombom-runtime-artifact', version: 1, bytes: 1,
          metadata: { representationProfile: 'adaptive-eighth-id-v1' }, value: {},
        },
      })
      state.artifactIds.push(artifact.id)
      return { verified: true, representationProfile: 'adaptive-eighth-id-v1', representationSpaceId: 'RS-AUTO-GATE' }
    },
    synthesis: verified,
    families: verified,
    languageExperience: verified,
    conceptMacros: verified,
    evaluate: async () => ({ verified: true, metrics: passingMetrics }),
  } })
  const result = await executor({
    payload: { sourceIds: [source.id], automationLevel: 'controlled' }, type: 'full-cycle',
  }, {
    jobId: 'adaptive-auto-gate', store, config, signal: new AbortController().signal,
    checkpoint: async () => {}, progress: async () => {}, emit: async () => {},
  })
  assert.equal(result.committed, false)
  assert.equal(result.activationSnapshotId, null)
  assert.equal(result.activationBlockedReason, 'plastic-region-release-required')
  assert.equal(store.getSnapshot(result.snapshotId).status, 'candidate')
  assert.equal(store.getActiveSnapshot(), undefined)
})

test('candidate snapshot remains bound to the exact region base when active state changes at the final checkpoint', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({ text: 'Son CAS kapısı için eğitim kaynağı.' }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  const baseSnapshot = store.createSnapshot({
    id: 'region-final-cas-base', status: 'candidate', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  store.activateSnapshot(baseSnapshot.id)
  const regionBase = currentPlasticRegionBase(store)
  const config = defaultConfig('/tmp/adaptive-final-base-gate')
  config.plasticRegions = {
    ...config.plasticRegions,
    mode: 'shadow',
    representationProfile: 'adaptive-eighth-id-v1',
  }
  const verified = async () => ({ verified: true })
  const executor = createPipelineExecutor({ stageHandlers: {
    textLab: async () => ({
      verified: true, representationProfile: 'adaptive-eighth-id-v1', representationSpaceId: 'RS-FINAL-CAS',
    }),
    synthesis: verified,
    families: verified,
    languageExperience: verified,
    plasticRegions: async () => ({ verified: true, regionBase }),
    conceptMacros: verified,
    evaluate: async () => ({ verified: true, metrics: passingMetrics }),
  } })
  let replacement
  await assert.rejects(
    executor({ payload: { sourceIds: [source.id] }, type: 'full-cycle' }, {
      jobId: 'adaptive-final-cas', store, config, signal: new AbortController().signal,
      checkpoint: async (stage) => {
        if (stage !== 'candidate-snapshot' || replacement) return
        replacement = store.createSnapshot({
          id: 'region-final-cas-replacement', parentId: baseSnapshot.id,
          status: 'candidate', artifactIds: [], sourceIds: [],
          evaluation: { passed: true, runtimeVerified: true },
        })
        store.activateSnapshot(replacement.id)
      },
      progress: async () => {}, emit: async () => {},
    }),
    (error) => error?.code === 'STALE_REGION_ATLAS_BASE',
  )
  assert.equal(store.getActiveSnapshot().id, replacement.id)
  assert.equal(store.listSnapshots().some((snapshot) => snapshot.producerJobId === 'adaptive-final-cas'), false)
})

test('explicit corpus rebuild retains previously processed selected sources', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const first = store.createSource(normalizeTextSource({ text: 'Birinci kalıcı korpus.' }, store.getConfig())).source
  const second = store.createSource(normalizeTextSource({ text: 'İkinci yeni korpus.' }, store.getConfig())).source
  store.decideSource(first.id, 'approved')
  store.decideSource(second.id, 'approved')
  store.markSourcesProcessed([first.id], 'older-snapshot')
  const state = {}
  const result = await DEFAULT_STAGE_HANDLERS.collect({ payload: { sourceIds: [first.id, second.id] } }, { store }, state)
  assert.equal(result.sourceCount, 2)
  assert.deepEqual(new Set(state.sources.map((source) => source.id)), new Set([first.id, second.id]))
})

test('explicit source ids cannot re-enable deselected or unapproved sources', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const selected = store.createSource(normalizeTextSource({ text: 'Seçili ve onaylı eğitim kaynağı.' }, store.getConfig())).source
  const deselected = store.createSource(normalizeTextSource({ text: 'Seçimi sonradan kaldırılan eğitim kaynağı.' }, store.getConfig())).source
  const quarantined = store.createSource(normalizeTextSource({ text: 'Henüz onaylanmamış eğitim kaynağı.' }, store.getConfig())).source
  store.decideSource(selected.id, 'approved')
  store.decideSource(deselected.id, 'approved')
  store.setSourceSelected(deselected.id, false)

  const state = {}
  const result = await DEFAULT_STAGE_HANDLERS.collect({
    payload: { sourceIds: [selected.id, deselected.id, quarantined.id] },
  }, { store }, state)

  assert.equal(result.sourceCount, 1)
  assert.deepEqual(state.sources.map((source) => source.id), [selected.id])
  assert.equal(state.sources[0].status, 'approved')
  assert.equal(state.sources[0].selected, true)
})

test('preliminary corpus manifest never places Ollama sources in validation or test', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const inputs = Array.from({ length: 4 }, (_, index) => normalizeTextSource({
    text: `Benzersiz ${index} numaralı kaynak metni burada bulunur.`,
  }, store.getConfig()))
  inputs[3].sourceType = 'ollama-generated'
  inputs[3].provenance = { ...inputs[3].provenance, kind: 'ollama-generated', model: 'teacher:14b' }
  const sources = inputs.map((input) => store.createSource(input).source)
  for (const source of sources) store.decideSource(source.id, 'approved')

  const state = {}
  const result = await DEFAULT_STAGE_HANDLERS.collect(
    { payload: { sourceIds: sources.map((source) => source.id) } },
    { store, jobId: 'missing-job' },
    state,
  )

  assert.ok(result.partitions.train.includes(sources[3].id))
  assert.ok(!result.partitions.validation.includes(sources[3].id))
  assert.ok(!result.partitions.test.includes(sources[3].id))
})

test('pipeline fails closed when a claimed store primitive is missing', async (t) => {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({ text: 'Eksik claim sözleşmesi deneyi.' }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  store.updateClaimedJobPayload = undefined
  await assert.rejects(
    DEFAULT_STAGE_HANDLERS.collect(
      { payload: { sourceIds: [source.id] } },
      { store, jobId: 'claimed-job', jobClaim: { owner: 'worker', attempt: 1 } },
      {},
    ),
    { code: 'STORE_CLAIM_CONTRACT_MISSING' },
  )
})

async function claimedPipelineFixture(t) {
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const source = store.createSource(normalizeTextSource({ text: 'Kasiyer küçük paketi dikkatlice taşıdı.' }, store.getConfig())).source
  store.decideSource(source.id, 'approved')
  const { job } = store.createJob({
    type: 'full-cycle',
    maxAttempts: 3,
    payload: { sourceIds: [source.id], reprocess: true },
  })
  const claim = store.claimNextJob('pipeline-worker-1', 30_000, 60_000)
  const replaceAttempt = () => {
    store.updateJob(job.id, {
      leaseUntil: new Date(Date.now() - 1).toISOString(),
      executionLeaseUntil: new Date(Date.now() - 1).toISOString(),
    })
    assert.equal(store.recoverExpiredJobs(), 1)
    const replacement = store.claimNextJob('pipeline-worker-2', 30_000, 60_000)
    assert.equal(replacement.attempt, 2)
    return replacement
  }
  return { store, source, job: claim, claim: { owner: claim.leaseOwner, attempt: claim.attempt }, replaceAttempt }
}

const passingMetrics = {
  holdoutTop1: 0.8,
  grammaticalPreference: 0.8,
  regressionRate: 0,
  memorizationGap: 0.1,
  unseenCombinationAccuracy: 0.75,
  evaluationSampleCount: 48,
}

test('pipeline cannot persist an artifact after its claimed attempt is replaced', async (t) => {
  const fixture = await claimedPipelineFixture(t)
  let replaced = false
  const executor = createPipelineExecutor({ stageHandlers: {
    synthesis: async () => ({ verified: true }),
    families: async () => ({ verified: true }),
    languageExperience: async () => ({ verified: true }),
    conceptMacros: async () => ({ verified: true }),
    evaluate: async () => ({ verified: true, metrics: passingMetrics }),
  } })
  const context = {
    jobId: fixture.job.id,
    jobClaim: fixture.claim,
    store: fixture.store,
    signal: new AbortController().signal,
    checkpoint: async () => {},
    progress: async (stage) => {
      if (stage === 'textLab' && !replaced) {
        replaced = true
        fixture.replaceAttempt()
      }
    },
    emit: async () => {},
  }

  await assert.rejects(executor(fixture.job, context), { code: 'STALE_JOB_LEASE' })
  assert.equal(fixture.store.listArtifacts().length, 0)
  assert.equal(fixture.store.listSnapshots().length, 0)
  assert.equal(fixture.store.getJob(fixture.job.id).attempt, 2)
})

test('pipeline cannot create a candidate snapshot after its claimed attempt is replaced', async (t) => {
  const fixture = await claimedPipelineFixture(t)
  const verified = async () => ({ verified: true })
  const executor = createPipelineExecutor({ stageHandlers: {
    textLab: verified,
    synthesis: verified,
    families: verified,
    languageExperience: verified,
    conceptMacros: verified,
    evaluate: async () => ({ verified: true, metrics: passingMetrics }),
  } })
  let replaced = false
  const context = {
    jobId: fixture.job.id,
    jobClaim: fixture.claim,
    store: fixture.store,
    signal: new AbortController().signal,
    checkpoint: async (stage) => {
      if (stage === 'candidate-snapshot' && !replaced) {
        replaced = true
        fixture.replaceAttempt()
      }
    },
    progress: async () => {},
    emit: async () => {},
  }

  await assert.rejects(executor(fixture.job, context), { code: 'STALE_JOB_LEASE' })
  assert.equal(fixture.store.listSnapshots().length, 0)
})

test('atomic completion cannot activate a candidate or mark sources processed after replacement', async (t) => {
  const fixture = await claimedPipelineFixture(t)
  fixture.job.payload.automationLevel = 'controlled'
  const verified = async () => ({ verified: true })
  const executor = createPipelineExecutor({ stageHandlers: {
    textLab: verified,
    synthesis: verified,
    families: verified,
    languageExperience: verified,
    conceptMacros: verified,
    evaluate: async () => ({ verified: true, metrics: passingMetrics }),
  } })
  let replaced = false
  const context = {
    jobId: fixture.job.id,
    jobClaim: fixture.claim,
    store: fixture.store,
    signal: new AbortController().signal,
    checkpoint: async (stage) => {
      if (stage === 'commit' && !replaced) {
        replaced = true
        fixture.replaceAttempt()
      }
    },
    progress: async () => {},
    emit: async () => {},
  }

  const result = await executor(fixture.job, context)
  assert.equal(result.activationSnapshotId, result.snapshotId)
  assert.throws(() => fixture.store.completeClaimedJobWithSnapshot(
    fixture.job.id,
    fixture.claim.owner,
    fixture.claim.attempt,
    result.activationSnapshotId,
    { ...result, committed: true },
    () => true,
  ), { code: 'STALE_JOB_LEASE' })
  assert.equal(fixture.store.getActiveSnapshot(), undefined)
  assert.equal(fixture.store.listSnapshots().length, 1)
  assert.equal(fixture.store.listSnapshots()[0].status, 'candidate')
  assert.equal(fixture.store.listSnapshots()[0].producerJobId, fixture.job.id)
  assert.equal(fixture.store.listSnapshots()[0].producerAttempt, 1)
  assert.equal(fixture.store.getSource(fixture.source.id).processedSnapshotId, null)
})

for (const requestedState of ['pause_requested', 'cancel_requested']) {
  test(`pipeline cannot activate a candidate after ${requestedState}`, async (t) => {
    const fixture = await claimedPipelineFixture(t)
    fixture.job.payload.automationLevel = 'controlled'
    const verified = async () => ({ verified: true })
    const executor = createPipelineExecutor({ stageHandlers: {
      textLab: verified,
      synthesis: verified,
      families: verified,
      languageExperience: verified,
      conceptMacros: verified,
      evaluate: async () => ({ verified: true, metrics: passingMetrics }),
    } })
    let requested = false
    const context = {
      jobId: fixture.job.id,
      jobClaim: fixture.claim,
      store: fixture.store,
      signal: new AbortController().signal,
      checkpoint: async (stage) => {
        if (stage === 'commit' && !requested) {
          requested = true
          fixture.store.updateJob(fixture.job.id, { state: requestedState })
        }
      },
      progress: async () => {},
      emit: async () => {},
    }

    const result = await executor(fixture.job, context)
    assert.throws(
      () => fixture.store.completeClaimedJobWithSnapshot(
        fixture.job.id,
        fixture.claim.owner,
        fixture.claim.attempt,
        result.activationSnapshotId,
        { ...result, committed: true },
        () => true,
      ),
      (error) => error?.code === 'STALE_JOB_LEASE' && error.actualState === requestedState,
    )
    assert.equal(fixture.store.getActiveSnapshot(), undefined)
    assert.equal(fixture.store.listSnapshots().length, 1)
    assert.equal(fixture.store.listSnapshots()[0].status, 'candidate')
    assert.equal(fixture.store.getSource(fixture.source.id).processedSnapshotId, null)
  })
}
