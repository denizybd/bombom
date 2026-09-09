import test from 'node:test'
import assert from 'node:assert/strict'
import { request as httpRequest } from 'node:http'
import { createTrainingService } from '../http-server.mjs'
import { temporaryDirectory, listenService, jsonFetch, waitFor } from './helpers.mjs'
import { encodeArtifactValue } from '../../runtime/artifacts.mjs'
import { PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND } from '../../runtime/plastic-region-cache.mjs'
import { createPlasticRegionAtlas } from '../../src/adaptive-regions/index.ts'

function createCertifiedCandidate(store, input, { resultField = 'snapshotId' } = {}) {
  const { job } = store.createJob({ type: 'full-cycle', maxAttempts: 1 })
  const snapshot = store.createSnapshot({
    ...input,
    producerJobId: job.id,
    producerAttempt: 1,
  })
  store.updateJob(job.id, {
    state: 'succeeded',
    stage: 'completed',
    attempt: 1,
    result: { [resultField]: snapshot.id },
    finishedAt: new Date().toISOString(),
  })
  return snapshot
}

test('UI contract supports quarantine, queue, gated candidate, metrics, and rollback', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  let result = await jsonFetch(base, '/api/sources/text', { method: 'POST', body: JSON.stringify({ text: 'Ali bıçakla ekmeği kesti.', title: 'Mikro korpus' }) })
  assert.equal(result.response.status, 201)
  const sourceId = result.body.id
  result = await jsonFetch(base, '/api/quarantine')
  assert.equal(result.body.length, 1)
  result = await jsonFetch(base, `/api/quarantine/${sourceId}/approve`, { method: 'POST', body: '{}' })
  assert.equal(result.body.state, 'approved')

  result = await jsonFetch(base, '/api/run/start', { method: 'POST', headers: { 'Idempotency-Key': 'integration-cycle-1' }, body: '{}' })
  assert.equal(result.response.status, 202)
  const jobId = result.body.job.id
  const finalJob = await waitFor(async () => {
    const queue = await jsonFetch(base, '/api/queue')
    const job = queue.body.find((item) => item.id === jobId)
    return ['complete', 'failed'].includes(job?.state) ? job : undefined
  })
  assert.equal(finalJob.state, 'complete')
  assert.equal(service.store.getJob(jobId).result.committed, false, 'placeholder stages must never activate a model')
  const snapshots = await jsonFetch(base, '/api/snapshots')
  assert.equal(snapshots.body[0].state, 'rejected')
  const metrics = await jsonFetch(base, '/api/metrics')
  assert.equal(metrics.body.evaluationSampleCount, 1)

  const idempotent = await jsonFetch(base, '/api/run/start', { method: 'POST', headers: { 'Idempotency-Key': 'integration-cycle-1' }, body: '{}' })
  assert.equal(idempotent.response.status, 200)
  assert.equal(idempotent.body.duplicate, true)
})

test('source and quarantine dashboards use compact summaries without decoding source bodies', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), requireSqlite: true })
  const base = await listenService(t, service)
  const text = 'Ali ekmeği kesti.\n\nVeli masayı sildi?'
  const created = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ text, title: 'Özet sıcak yolu' }),
  })
  assert.equal(created.response.status, 201)

  const originalList = service.store.backend.list.bind(service.store.backend)
  const originalGet = service.store.backend.get.bind(service.store.backend)
  service.store.backend.list = (bucket) => {
    if (bucket === 'sources') throw new Error('full source scan invoked')
    return originalList(bucket)
  }
  service.store.backend.get = (bucket, id) => {
    if (bucket === 'sources') throw new Error('full source decode invoked')
    return originalGet(bucket, id)
  }
  try {
    const sources = await jsonFetch(base, '/api/sources')
    assert.equal(sources.response.status, 200)
    assert.equal(sources.body[0].itemCount, 2)

    const quarantine = await jsonFetch(base, '/api/quarantine')
    assert.equal(quarantine.response.status, 200)
    assert.equal(quarantine.body[0].preview, 'Ali ekmeği kesti. Veli masayı sildi?')
    assert.equal(quarantine.body[0].characterCount, text.length)
  } finally {
    service.store.backend.list = originalList
    service.store.backend.get = originalGet
  }
})

test('teacher output is schema-checked, deduplicated, and stays in quarantine', async (t) => {
  let chatCalls = 0
  const fetchImpl = async (url) => {
    if (url.endsWith('/api/tags')) return Response.json({ models: [{ name: 'teacher:8b', size: 42 }] })
    chatCalls += 1
    return Response.json({
      model: 'teacher:8b', prompt_eval_count: 12, eval_count: 10,
      message: { content: JSON.stringify({ lessons: [
        { text: 'Mert makasla kumaşı kesti.', kind: 'role_swap', rationale: 'Yeni birleşim' },
        { text: 'Mert makasla kumaşı kesti.', kind: 'role_swap', rationale: 'Tekrar denetimi' },
      ] }) },
    })
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  service.store.updateConfig({ ollama: { model: 'teacher:8b' } })
  const base = await listenService(t, service)
  const generated = await jsonFetch(base, '/api/teacher/generate', { method: 'POST', body: JSON.stringify({ topic: 'araç ve eylem ilişkisi', count: 2 }) })
  assert.equal(generated.response.status, 201)
  assert.equal(generated.body.generatedCount, 2)
  assert.equal(generated.body.rejectedCount, 1)
  assert.equal(chatCalls, 1)
  const usage = service.store.getUsage()
  assert.equal(usage.teacherRequests, 1)
  assert.equal(usage.teacherTokens, 22, 'successful legacy generation must settle its crash-safe reservation to strict reported usage')
  const quarantine = await jsonFetch(base, '/api/quarantine')
  assert.equal(quarantine.body.length, 1)
  assert.match(quarantine.body[0].provenance, /Ollama/)
  assert.equal(quarantine.body[0].state, 'pending')
})

test('legacy teacher usage records a provider overrun and blocks the next call before I/O', async (t) => {
  let chatCalls = 0
  const fetchImpl = async (url) => {
    if (url.endsWith('/api/tags')) return Response.json({ models: [{ name: 'teacher:14b', size: 42 }] })
    chatCalls += 1
    return Response.json({
      model: 'teacher:14b',
      prompt_eval_count: 100_000,
      eval_count: 10,
      message: { content: JSON.stringify({ lessons: [
        { text: 'Ece sabah kitabı açtı.', kind: 'paraphrase', rationale: 'Sayaç taşması deneyi' },
      ] }) },
    })
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  service.store.updateConfig({
    ollama: { model: 'teacher:14b' },
    budgets: { maxDailyTeacherRequests: 2, maxDailyTeacherTokens: 40_000 },
  })
  const base = await listenService(t, service)
  const body = JSON.stringify({ topic: 'günlük eylemler', count: 1 })
  const first = await jsonFetch(base, '/api/teacher/generate', { method: 'POST', body })
  assert.equal(first.response.status, 201)
  assert.equal(service.store.getUsage().teacherTokens, 100_010)

  const second = await jsonFetch(base, '/api/teacher/generate', { method: 'POST', body })
  assert.equal(second.body.code, 'BUDGET_EXCEEDED')
  assert.equal(chatCalls, 1, 'the overrun must close the quota before another model call')
  assert.equal(service.store.getUsage().teacherRequests, 1)
  assert.equal(service.store.getUsage().teacherTokens, 100_010)
})

test('completed request body followed by client disconnect aborts the active teacher call', async (t) => {
  let chatCalls = 0
  let entered
  const enteredCall = new Promise((resolve) => { entered = resolve })
  let observedAbort = false
  const fetchImpl = async (url, init) => {
    if (url.endsWith('/api/tags')) return Response.json({ models: [{ name: 'teacher:14b', size: 42 }] })
    chatCalls += 1
    entered()
    return new Promise((_resolve, reject) => {
      const abort = () => {
        observedAbort = true
        reject(init.signal.reason)
      }
      if (init.signal.aborted) abort()
      else init.signal.addEventListener('abort', abort, { once: true })
    })
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  service.store.updateConfig({ ollama: { model: 'teacher:14b' } })
  const base = await listenService(t, service)
  const url = new URL('/api/teacher/generate', base)
  const body = JSON.stringify({ topic: 'bağlantı kesintisi', count: 1_000 })
  const request = httpRequest({
    hostname: url.hostname,
    port: url.port,
    path: url.pathname,
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
  })
  request.on('error', () => {})
  request.end(body)
  await enteredCall
  request.destroy()
  await waitFor(() => observedAbort || undefined)
  assert.equal(chatCalls, 1, 'disconnect must stop before a second teacher chunk')
  assert.equal(service.store.getUsage().teacherRequests, 1)
  assert.ok(service.store.getUsage().teacherTokens > 0, 'the already-issued ambiguous call remains conservatively charged')
})

test('concurrent teacher campaign starts atomically admit only one job', async (t) => {
  let tagCalls = 0
  let releaseTagCalls
  const tagCallBarrier = new Promise((resolve) => { releaseTagCalls = resolve })
  const fetchImpl = async (url) => {
    assert.ok(url.endsWith('/api/tags'))
    tagCalls += 1
    if (tagCalls === 2) releaseTagCalls()
    await tagCallBarrier
    return Response.json({ models: [{
      name: 'teacher:14b', digest: 'sha256:concurrent-14b', size: 9_000_000_000,
      modified_at: '2026-08-20T12:00:00.000Z',
      details: { parameter_size: '14B', quantization_level: 'Q4_K_M' },
    }] })
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  // Keep the winner queued so execution telemetry cannot obscure the admission
  // assertions. This is a queue pause, not a teacher-campaign record.
  service.store.setPipelineState({ paused: true })
  const base = await listenService(t, service)
  const body = JSON.stringify({
    model: 'teacher:14b', seed: 20260829, targetAccepted: 16,
    generationBatchSize: 2, trainEveryAccepted: 16,
    curriculum: 'turkish-foundations-v1', mode: 'research-only',
    continuousPlan: {
      blockAccepted: 16, maxBlocks: 2,
      minimumDiskHeadroomMb: 512, maximumSyntheticSources: 100_000,
    },
  })

  const responses = await Promise.all([
    jsonFetch(base, '/api/teacher/feed/campaigns', {
      method: 'POST', headers: { 'Idempotency-Key': 'concurrent-campaign-a' }, body,
    }),
    jsonFetch(base, '/api/teacher/feed/campaigns', {
      method: 'POST', headers: { 'Idempotency-Key': 'concurrent-campaign-b' }, body,
    }),
  ])
  const winner = responses.find((item) => item.response.status === 202)
  const loser = responses.find((item) => item.response.status === 409)
  assert.ok(winner)
  assert.equal(loser?.body?.code, 'TEACHER_CAMPAIGN_BUSY')
  assert.equal(tagCalls, 2, 'both requests must cross the former pre-insert race window')

  const campaigns = service.store.listJobs().filter((job) => job.type === 'teacher-feed-campaign')
  assert.equal(campaigns.length, 1)
  assert.equal(campaigns[0].id, winner.body.id)
  assert.equal(service.store.getUsage().jobs, 1, 'the rejected contender must refund its provisional job quota')
  assert.equal(service.store.listEvents().filter((event) => event.type === 'job-queued').length, 1)
  assert.equal(service.store.listEvents().filter((event) => event.type === 'teacher-feed-campaign-created').length, 1)
})

test('bounded 14B feed campaign is durable, auto-gates synthetic sources, and never activates its proposal', async (t) => {
  const fetchImpl = async (url, init) => {
    if (url.endsWith('/api/tags')) return Response.json({ models: [{ name: 'teacher:14b', digest: 'sha256:fixed-14b', size: 9_000_000_000, modified_at: '2026-08-20T12:00:00.000Z', details: { parameter_size: '14B', quantization_level: 'Q4_K_M' } }] })
    if (url.endsWith('/api/generate')) return Response.json({ model: 'teacher:14b', done: true, done_reason: 'unload' })
    const request = JSON.parse(init.body)
    const choiceSchema = request.format.properties.choices
    const count = choiceSchema.maxItems
    const allowedChoices = choiceSchema.items.enum
    assert.ok(Array.isArray(allowedChoices))
    assert.ok(allowedChoices.length >= count)
    return Response.json({
      model: 'teacher:14b', prompt_eval_count: 40, eval_count: 160,
      message: { content: JSON.stringify({ choices: allowedChoices.slice(0, count) }) },
    })
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  const base = await listenService(t, service)
  const created = await jsonFetch(base, '/api/teacher/feed/campaigns', {
    method: 'POST',
    headers: { 'Idempotency-Key': 'http-feed-campaign-1' },
    body: JSON.stringify({
      model: 'teacher:14b', seed: 20260829, targetAccepted: 16,
      generationBatchSize: 2, trainEveryAccepted: 16,
      curriculum: 'turkish-foundations-v1', mode: 'research-only',
    }),
  })
  assert.equal(created.response.status, 202)
  const campaignId = created.body.id
  await waitFor(async () => {
    const job = service.store.getJob(campaignId)
    return ['succeeded', 'failed', 'paused'].includes(job?.state) ? job : undefined
  }, { timeoutMs: 10_000 })

  const [campaign] = (await jsonFetch(base, '/api/teacher/feed/campaigns')).body
  assert.equal(campaign.id, campaignId)
  assert.equal(campaign.status, 'completed')
  assert.equal(campaign.accepted, 16)
  assert.equal(campaign.completedTrainingCycles, 1)
  const campaignSources = service.store.listSources({ status: 'approved', includeContent: true })
    .filter((source) => source.provenance?.campaignId === campaignId)
  assert.equal(campaignSources.length, 16)
  assert.equal(new Set(campaignSources.map((source) => source.provenance.independenceGroupId)).size, 1)
  assert.ok(campaignSources.every((source) => source.provenance.evaluationEligible === false && source.provenance.managedByTeacherCampaign === true))
  assert.equal(service.store.getActiveSnapshot(), undefined)
  assert.equal(service.store.listSnapshots().at(-1)?.status, 'rejected')

  const repeated = await jsonFetch(base, '/api/teacher/feed/campaigns', {
    method: 'POST', headers: { 'Idempotency-Key': 'http-feed-campaign-1' },
    body: JSON.stringify({
      model: 'teacher:14b', seed: 20260829, targetAccepted: 16,
      generationBatchSize: 2, trainEveryAccepted: 16,
      curriculum: 'turkish-foundations-v1', mode: 'research-only',
    }),
  })
  assert.equal(repeated.response.status, 200)
  assert.equal(repeated.body.id, campaignId)
  assert.match(service.store.getJob(campaignId).requestHash, /^[a-f0-9]{64}$/u)

  const conflictingRetry = await jsonFetch(base, '/api/teacher/feed/campaigns', {
    method: 'POST', headers: { 'Idempotency-Key': 'http-feed-campaign-1' },
    body: JSON.stringify({
      model: 'teacher:14b', seed: 20260830, targetAccepted: 16,
      generationBatchSize: 2, trainEveryAccepted: 16,
      curriculum: 'turkish-foundations-v1', mode: 'research-only',
    }),
  })
  assert.equal(conflictingRetry.response.status, 409)
  assert.equal(conflictingRetry.body.code, 'TEACHER_CAMPAIGN_IDEMPOTENCY_CONFLICT')
  assert.equal(service.store.listJobs().filter((job) => job.idempotencyKey === 'http-feed-campaign-1').length, 1)

  const controlEventsBefore = service.store.listEvents().filter((event) =>
    event.type === 'teacher-feed-campaign-cancel-requested' && event.jobId === campaignId).length
  service.store.updateJob(campaignId, { state: 'running', stage: 'running', finishedAt: null })
  const originalControl = service.queue.control.bind(service.queue)
  service.queue.control = (id, action) => {
    service.store.updateJob(id, { state: 'succeeded', stage: 'completed', finishedAt: new Date().toISOString() })
    return originalControl(id, action)
  }
  let racedCancel
  try {
    racedCancel = await jsonFetch(base, `/api/teacher/feed/campaigns/${campaignId}/cancel`, {
      method: 'POST', body: '{}',
    })
  } finally {
    service.queue.control = originalControl
  }
  assert.equal(racedCancel.response.status, 200)
  assert.equal(racedCancel.body.status, 'completed')
  const controlEventsAfter = service.store.listEvents().filter((event) =>
    event.type === 'teacher-feed-campaign-cancel-requested' && event.jobId === campaignId).length
  assert.equal(controlEventsAfter, controlEventsBefore, 'a stale pre-read must not turn a terminal no-op into misleading telemetry')

  const unsafe = await jsonFetch(base, '/api/teacher/feed/campaigns', {
    method: 'POST', body: JSON.stringify({
      model: 'teacher:14b', seed: 1, targetAccepted: 16, generationBatchSize: 2,
      trainEveryAccepted: 16, curriculum: 'turkish-foundations-v1', mode: 'research-only',
      qualityThreshold: 0,
    }),
  })
  assert.equal(unsafe.response.status, 400)
  assert.equal(unsafe.body.code, 'INVALID_TEACHER_CAMPAIGN')
})

test('teacher feed campaign API reports a resource-waiting job as queued, not running', async (t) => {
  const fetchImpl = async (url) => {
    if (url.endsWith('/api/tags')) return Response.json({ models: [{ name: 'teacher:14b', digest: 'sha256:fixed-14b', size: 9_000_000_000 }] })
    throw new Error(`Unexpected Ollama request: ${url}`)
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  const base = await listenService(t, service)
  service.store.setPipelineState({ paused: true, phase: 'paused' })
  const created = await jsonFetch(base, '/api/teacher/feed/campaigns', {
    method: 'POST', headers: { 'Idempotency-Key': 'queued-campaign' },
    body: JSON.stringify({
      model: 'teacher:14b', seed: 19, targetAccepted: 4,
      generationBatchSize: 2, trainEveryAccepted: 4,
      curriculum: 'turkish-foundations-v1', mode: 'research-only',
    }),
  })
  assert.equal(created.response.status, 202)
  assert.equal(created.body.status, 'queued')
  assert.equal(service.store.getJob(created.body.id).state, 'queued')
})

test('continuous 14B API runs bounded blocks, keeps synthetic data out of holdout, and pauses without independent evidence', async (t) => {
  const fetchImpl = async (url, init) => {
    if (url.endsWith('/api/tags')) return Response.json({ models: [{ name: 'qwen:14b', digest: 'sha256:qwen-fixed', size: 9_000_000_000, modified_at: '2026-08-20T12:00:00.000Z', details: { parameter_size: '14B', quantization_level: 'Q4_K_M' } }] })
    if (url.endsWith('/api/generate')) return Response.json({ model: 'qwen:14b', done: true, done_reason: 'unload' })
    const request = JSON.parse(init.body)
    const choices = request.format.properties.choices.items.enum.slice(0, request.format.properties.choices.maxItems)
    return Response.json({
      model: 'qwen:14b', prompt_eval_count: 20, eval_count: 20,
      message: { content: JSON.stringify({ choices }) },
    })
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  const base = await listenService(t, service)
  const created = await jsonFetch(base, '/api/teacher/feed/campaigns', {
    method: 'POST',
    body: JSON.stringify({
      model: 'qwen:14b', seed: 77, targetAccepted: 4,
      generationBatchSize: 2, trainEveryAccepted: 4,
      curriculum: 'turkish-foundations-v1', mode: 'research-only',
      continuousPlan: { blockAccepted: 4, maxBlocks: 1, minimumDiskHeadroomMb: 4, maximumSyntheticSources: 100 },
    }),
  })
  assert.equal(created.response.status, 202)
  assert.equal(created.body.continuous.blockAccepted, 4)
  const campaignId = created.body.id
  const terminal = await waitFor(async () => {
    const job = service.store.getJob(campaignId)
    return ['paused', 'succeeded', 'failed'].includes(job?.state) ? job : undefined
  }, { timeoutMs: 10_000 })
  assert.equal(terminal.state, 'paused')

  const campaign = (await jsonFetch(base, '/api/teacher/feed/campaigns')).body.find((item) => item.id === campaignId)
  assert.equal(campaign.status, 'paused')
  assert.equal(campaign.accepted, 4)
  assert.equal(campaign.continuous.completedBlocks, 1)
  assert.equal(campaign.continuous.stopGate.passed, false)
  assert.equal(campaign.continuous.stopGate.status, 'missing')
  assert.match(campaign.stopReason, /benchmark/u)
  assert.equal(service.store.getActiveSnapshot(), undefined)

  const manifestArtifact = service.store.listArtifacts().find((artifact) => artifact.kind === 'text-lab-input-manifest')
  const manifest = service.store.getArtifact(manifestArtifact.id)
  assert.equal(manifest.payload.partitions.validation.length, 0)
  assert.equal(manifest.payload.partitions.test.length, 0)
  assert.equal(manifest.payload.partitions.train.length, 4)

  const oversized = await jsonFetch(base, '/api/teacher/feed/campaigns', {
    method: 'POST', body: JSON.stringify({
      model: 'qwen:14b', seed: 78, targetAccepted: 2_000, generationBatchSize: 2,
      trainEveryAccepted: 32, curriculum: 'turkish-foundations-v1', mode: 'research-only',
      continuousPlan: { blockAccepted: 2_000, maxBlocks: 100 },
    }),
  })
  assert.equal(oversized.response.status, 400)
  assert.equal(oversized.body.code, 'INVALID_TEACHER_CAMPAIGN')
})

test('server rejects non-local browser origins', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const { response, body } = await jsonFetch(base, '/api/status', { headers: { Origin: 'https://attacker.example' } })
  assert.equal(response.status, 403)
  assert.equal(body.code, 'ORIGIN_NOT_ALLOWED')
})

test('browser clients cannot mint a forged snapshot or evaluation result', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const { response, body } = await jsonFetch(base, '/api/snapshots', {
    method: 'POST',
    body: JSON.stringify({
      label: 'Sahte başarılı aday',
      evaluation: { passed: true, runtimeVerified: true },
      metrics: { overallScore: 1, syntaxContrastAccuracy: 1 },
    }),
  })
  assert.equal(response.status, 404)
  assert.equal(body.code, 'NOT_FOUND')
  assert.equal(service.store.listSnapshots().length, 0)
})

test('browser-facing responses exactly expose status, model, config, selection, and multipart contracts', async (t) => {
  const fetchImpl = async (url) => {
    if (url.endsWith('/api/tags')) return Response.json({ models: [{ name: 'qwen:local', size: 1234, modified_at: '2026-01-01', details: { parameter_size: '8B', quantization_level: 'Q4' } }] })
    throw new Error('Unexpected Ollama call')
  }
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true, fetchImpl })
  const base = await listenService(t, service)
  const status = await jsonFetch(base, '/api/status')
  assert.equal(status.body.serviceVersion, '1.0.0')
  assert.equal(status.body.ollama.connected, true)
  assert.deepEqual(status.body.ollama.models, [{ name: 'qwen:local', parameterSize: '8B', quantization: 'Q4', sizeBytes: 1234, modifiedAt: '2026-01-01' }])
  assert.equal(status.body.config.autoRun, false)
  assert.ok(Array.isArray(status.body.stages))
  const models = await jsonFetch(base, '/api/ollama/models')
  assert.deepEqual(models.body[0], { name: 'qwen:local', parameterSize: '8B', quantization: 'Q4', sizeBytes: 1234, modifiedAt: '2026-01-01' })

  const form = new FormData()
  form.append('files', new Blob(['Bir dosya eğitim örneğidir.'], { type: 'text/plain' }), 'ornek.txt')
  form.append('provenance', 'Tarayıcı test seçicisi')
  const uploadResponse = await fetch(`${base}/api/sources/files`, { method: 'POST', body: form })
  assert.equal(uploadResponse.status, 201)
  const uploaded = await uploadResponse.json()
  assert.equal(uploaded[0].kind, 'file')
  assert.equal(uploaded[0].state, 'quarantined')
  const reviewed = await jsonFetch(base, `/api/quarantine/${uploaded[0].id}/approve`, { method: 'POST', body: '{}' })
  assert.equal(reviewed.body.state, 'approved')
  const selected = await jsonFetch(base, `/api/sources/${uploaded[0].id}`, { method: 'PATCH', body: JSON.stringify({ selected: false }) })
  assert.equal(selected.body.selected, false)

  const config = { ...status.body.config, model: 'qwen:local', autoRun: true, batchSize: 64, budgets: { cpuPercent: 50, ramMb: 2048, maxRunMinutes: 30, diskMb: 4096 } }
  const updated = await jsonFetch(base, '/api/config', { method: 'PATCH', body: JSON.stringify(config) })
  assert.equal(updated.body.config.autoRun, true)
  assert.equal(updated.body.config.model, 'qwen:local')
})

test('selecting an approved source triggers one content-addressed automatic cycle', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const status = await jsonFetch(base, '/api/status')
  await jsonFetch(base, '/api/config', {
    method: 'PATCH',
    body: JSON.stringify({ ...status.body.config, autoRun: false, batchSize: 1 }),
  })
  const created = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ name: 'Otomatik seçim', text: 'Seçilince bir kez sıraya girer.' }),
  })
  await jsonFetch(base, `/api/quarantine/${created.body.id}/approve`, { method: 'POST', body: '{}' })
  await jsonFetch(base, `/api/sources/${created.body.id}`, { method: 'PATCH', body: JSON.stringify({ selected: false }) })
  await jsonFetch(base, '/api/config', {
    method: 'PATCH',
    body: JSON.stringify({ ...status.body.config, autoRun: true, batchSize: 1 }),
  })
  assert.equal((await jsonFetch(base, '/api/queue')).body.length, 0)

  await jsonFetch(base, `/api/sources/${created.body.id}`, { method: 'PATCH', body: JSON.stringify({ selected: true }) })
  await waitFor(async () => (await jsonFetch(base, '/api/queue')).body.length === 1 ? true : undefined)
  await waitFor(async () => {
    const [job] = (await jsonFetch(base, '/api/queue')).body
    return job?.state === 'complete' ? true : undefined
  })

  await jsonFetch(base, `/api/sources/${created.body.id}`, { method: 'PATCH', body: JSON.stringify({ selected: false }) })
  await jsonFetch(base, `/api/sources/${created.body.id}`, { method: 'PATCH', body: JSON.stringify({ selected: true }) })
  assert.equal((await jsonFetch(base, '/api/queue')).body.length, 1, 'the same corpus must not queue twice')
})

test('a completed rejected cycle counts only representation-new sources toward the next automatic batch', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const status = await jsonFetch(base, '/api/status')
  await jsonFetch(base, '/api/config', {
    method: 'PATCH',
    body: JSON.stringify({ ...status.body.config, autoRun: false, batchSize: 2 }),
  })

  const addApproved = async (index) => {
    const created = await jsonFetch(base, '/api/sources/text', {
      method: 'POST',
      body: JSON.stringify({ name: `Yeni kaynak ${index}`, text: `Birbirinden farklı ${index} numaralı otomatik eğitim kaynağı.` }),
    })
    await jsonFetch(base, `/api/quarantine/${created.body.id}/approve`, { method: 'POST', body: '{}' })
    return created.body.id
  }
  await addApproved(1)
  await addApproved(2)
  await jsonFetch(base, '/api/config', {
    method: 'PATCH',
    body: JSON.stringify({ ...status.body.config, autoRun: true, batchSize: 2 }),
  })
  await waitFor(async () => service.store.listJobs()[0]?.state === 'succeeded' ? true : undefined)
  const firstJob = service.store.listJobs()[0]
  assert.equal(Object.keys(firstJob.payload.resolvedSourceRevisions).length, 2)
  assert.equal(firstJob.result.committed, false)

  await addApproved(3)
  assert.equal(service.store.listJobs().length, 1, 'tek yeni temsil ikinci partiyi erken tetiklememeli')
  await addApproved(4)
  await waitFor(async () => service.store.listJobs().length === 2 ? true : undefined)
})

test('manual proposal activation is gated and marks its corpus processed', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const created = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ name: 'Onay korpusu', text: 'Bu kayıt güvenli aday aktivasyonunu sınar.', provenance: 'test' }),
  })
  const sourceId = created.body.id
  await jsonFetch(base, `/api/quarantine/${sourceId}/approve`, { method: 'POST', body: '{}' })
  const uncertified = service.store.createSnapshot({
    status: 'candidate',
    sourceIds: [sourceId],
    metrics: { overallScore: 0.79 },
    evaluation: { passed: true, runtimeVerified: true },
  })
  const blocked = await jsonFetch(base, `/api/snapshots/${uncertified.id}/activate`, { method: 'POST', body: '{}' })
  assert.equal(blocked.response.status, 409)
  assert.equal(blocked.body.code, 'SNAPSHOT_PRODUCER_UNVERIFIED')

  const { job: incompleteProducer } = service.store.createJob({ type: 'full-cycle', maxAttempts: 2 })
  service.store.updateJob(incompleteProducer.id, {
    state: 'failed',
    stage: 'failed',
    attempt: 1,
    result: null,
    finishedAt: new Date().toISOString(),
  })
  const orphan = service.store.createSnapshot({
    status: 'candidate',
    sourceIds: [sourceId],
    metrics: { overallScore: 0.795 },
    evaluation: { passed: true, runtimeVerified: true },
    producerJobId: incompleteProducer.id,
    producerAttempt: 1,
  })
  const orphanBlocked = await jsonFetch(base, `/api/snapshots/${orphan.id}/activate`, { method: 'POST', body: '{}' })
  assert.equal(orphanBlocked.response.status, 409)
  assert.equal(orphanBlocked.body.code, 'SNAPSHOT_PRODUCER_INCOMPLETE')

  const snapshot = createCertifiedCandidate(service.store, {
    status: 'candidate',
    sourceIds: [sourceId],
    metrics: { overallScore: 0.8 },
    evaluation: { passed: true, runtimeVerified: true },
  })
  const activated = await jsonFetch(base, `/api/snapshots/${snapshot.id}/activate`, { method: 'POST', body: '{}' })
  assert.equal(activated.response.status, 200)
  assert.equal(service.store.getActiveSnapshot().id, snapshot.id)
  assert.equal(service.store.getSource(sourceId).processedSnapshotId, snapshot.id)

  const firstSibling = createCertifiedCandidate(service.store, {
    status: 'candidate', sourceIds: [sourceId], metrics: { overallScore: 0.81 },
    evaluation: { passed: true, runtimeVerified: true },
  }, { resultField: 'lastSnapshotId' })
  const staleSibling = createCertifiedCandidate(service.store, {
    status: 'candidate', sourceIds: [sourceId], metrics: { overallScore: 0.82 },
    evaluation: { passed: true, runtimeVerified: true },
  })
  assert.equal((await jsonFetch(base, `/api/snapshots/${firstSibling.id}/activate`, { method: 'POST', body: '{}' })).response.status, 200)
  const staleActivation = await jsonFetch(base, `/api/snapshots/${staleSibling.id}/activate`, { method: 'POST', body: '{}' })
  assert.equal(staleActivation.response.status, 409)
  assert.equal(staleActivation.body.code, 'SNAPSHOT_PARENT_MISMATCH')
  const repeatedActivation = await jsonFetch(base, `/api/snapshots/${firstSibling.id}/activate`, { method: 'POST', body: '{}' })
  assert.equal(repeatedActivation.response.status, 409)
  assert.equal(repeatedActivation.body.code, 'SNAPSHOT_NOT_CANDIDATE')
})

test('manual activation rejects an adaptive atlas candidate without the exact release chain', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const atlas = createPlasticRegionAtlas({
    representationSpaceId: 'RS-MANUAL-REGION-GATE', behaviorDimension: 8, maximumMembershipsPerPage: 3,
  })
  const artifact = service.store.createArtifact({
    kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    contentHash: 'manual-region-atlas-content',
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: {
        representationSpaceId: atlas.representationSpaceId,
        fingerprint: atlas.fingerprint,
        geometryFingerprint: atlas.geometryFingerprint,
        revision: atlas.revision,
        behaviorDimension: atlas.behaviorDimension,
        maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
        activationEligible: false,
        candidateOnly: true,
      },
      value: encodeArtifactValue(atlas),
    },
  })
  const candidate = createCertifiedCandidate(service.store, {
    status: 'candidate', artifactIds: [artifact.id], sourceIds: [],
    metrics: { overallScore: 0.8 },
    evaluation: { passed: true, runtimeVerified: true },
  })
  const blocked = await jsonFetch(base, `/api/snapshots/${candidate.id}/activate`, { method: 'POST', body: '{}' })
  assert.equal(blocked.response.status, 409)
  assert.equal(blocked.body.code, 'PLASTIC_REGION_RELEASE_REQUIRED')
  assert.equal(service.store.getSnapshot(candidate.id).status, 'candidate')
  assert.equal(service.store.getActiveSnapshot(), undefined)
})

test('adaptive rollback stays fail-closed instead of activating a release-invalid wrapper snapshot', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const atlas = createPlasticRegionAtlas({
    representationSpaceId: 'RS-ADAPTIVE-ROLLBACK-GATE', behaviorDimension: 8, maximumMembershipsPerPage: 3,
  })
  const artifact = service.store.createArtifact({
    kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND,
    contentHash: 'adaptive-rollback-atlas-content',
    payload: {
      format: 'bombom-runtime-artifact', version: 1, bytes: 1,
      metadata: {
        representationSpaceId: atlas.representationSpaceId,
        fingerprint: atlas.fingerprint,
        geometryFingerprint: atlas.geometryFingerprint,
        revision: atlas.revision,
        behaviorDimension: atlas.behaviorDimension,
        maximumMembershipsPerPage: atlas.maximumMembershipsPerPage,
        representationProfile: 'adaptive-eighth-id-v1',
        activationEligible: true,
        candidateOnly: false,
      },
      value: encodeArtifactValue(atlas),
    },
  })
  const adaptive = service.store.createSnapshot({
    id: 'adaptive-rollback-target', status: 'candidate', artifactIds: [artifact.id], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(adaptive.id)
  const successor = service.store.createSnapshot({
    id: 'adaptive-rollback-successor', status: 'candidate', artifactIds: [], sourceIds: [],
    evaluation: { passed: true, runtimeVerified: true },
  })
  service.store.activateSnapshot(successor.id)
  const snapshotCount = service.store.listSnapshots().length

  const blocked = await jsonFetch(base, `/api/snapshots/${adaptive.id}/rollback`, {
    method: 'POST',
    body: JSON.stringify({ confirmToken: `ROLLBACK:${adaptive.id}` }),
  })
  assert.equal(blocked.response.status, 409)
  assert.equal(blocked.body.code, 'PLASTIC_REGION_ROLLBACK_RENEWAL_REQUIRED')
  assert.equal(service.store.getActiveSnapshot().id, successor.id)
  assert.equal(service.store.listSnapshots().length, snapshotCount, 'rejected rollback must not create a wrapper snapshot')
})

test('release evaluation rejects fabricated client metrics before proposal lookup', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const response = await jsonFetch(base, '/api/regions/proposals/nonexistent/evaluate', {
    method: 'POST',
    body: JSON.stringify({ metrics: { holdoutGain: 1, shamGain: 0 } }),
  })
  assert.equal(response.response.status, 400)
  assert.equal(response.body.code, 'INVALID_PLASTIC_REGION_RELEASE_REQUEST')
  assert.equal(service.store.listJobs().length, 0)
})

test('unchanged-atlas renewal rejects fabricated client benchmark values before snapshot lookup', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const response = await jsonFetch(base, '/api/regions/snapshots/nonexistent/renew-release', {
    method: 'POST',
    body: JSON.stringify({ holdoutGain: 1, shamGain: 0 }),
  })
  assert.equal(response.response.status, 400)
  assert.equal(response.body.code, 'INVALID_PLASTIC_REGION_RENEWAL_REQUEST')
  assert.equal(service.store.listJobs().length, 0)
})

test('config cannot enable active plastic regions without an exact released active snapshot', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const response = await jsonFetch(base, '/api/config', {
    method: 'PATCH',
    body: JSON.stringify({
      plasticRegions: {
        mode: 'active',
        representationProfile: 'adaptive-eighth-id-v1',
      },
    }),
  })
  assert.equal(response.response.status, 409)
  assert.equal(response.body.code, 'PLASTIC_REGION_RELEASE_REQUIRED')
  assert.equal(service.store.getConfig().plasticRegions.mode, 'off')
  assert.equal(service.store.getConfig().plasticRegions.representationProfile, 'legacy')
})

test('TextLab import keeps its bounded identity proof and upgrades an exact duplicate', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const text = 'Aynı sabit kimlikli kaynak.'
  const original = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ name: 'Önce ham', text, provenance: 'manual' }),
  })
  await jsonFetch(base, `/api/quarantine/${original.body.id}/approve`, { method: 'POST', body: '{}' })
  const manifest = {
    format: 'text-lab-recompute-proof-v1',
    originalSourceId: 'browser-source-1',
    sourceDigest: '0123456789abcdef',
    representationSpaceId: 'RS-0123456789ABCDEF',
    config: {
      matrixSide: 64, identityRatio: 0.25, minimumLeafGraphemes: 4, hashSeed: 1,
      contextRetention: 0.8, parentTraceWeight: 0.15, siblingTraceWeight: 0.05,
      minimumTrace: 0.1, identitySimilarityWeight: 0.5, habitExponent: 2,
    },
    identityProofs: [{ start: 0, end: text.length, identityDigest: 'fedcba9876543210' }],
  }
  service.store.markSourcesProcessed([original.body.id], 'old-snapshot')
  const duplicate = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ name: 'Sonra arşiv', text, provenance: 'TextLab', analysisManifest: manifest }),
  })
  assert.equal(duplicate.body.id, original.body.id)
  const upgraded = service.store.getSource(original.body.id)
  assert.deepEqual(upgraded.provenance.textLabManifest, manifest)
  assert.equal(upgraded.provenance.kind, 'text-lab-archive')
  assert.equal(upgraded.status, 'quarantine', 'a stronger representation proof requires a fresh human decision')
  assert.equal(upgraded.selected, false)
  assert.equal(upgraded.decidedAt, null)
  assert.equal(upgraded.processedSnapshotId, null)

  const compatibleAlias = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({
      name: 'Aynı kanıtın başka arşiv kaydı', text, provenance: 'TextLab',
      analysisManifest: { ...manifest, originalSourceId: 'browser-source-2' },
    }),
  })
  assert.equal(compatibleAlias.response.status, 201)
  assert.equal(compatibleAlias.body.id, original.body.id)

  const conflicting = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({
      name: 'Çelişen temsil', text, provenance: 'TextLab',
      analysisManifest: { ...manifest, config: { ...manifest.config, hashSeed: 2 } },
    }),
  })
  assert.equal(conflicting.response.status, 409)
  assert.equal(conflicting.body.code, 'TEXT_LAB_MANIFEST_CONFLICT')
})

test('TextLab manifest is rejected at intake when config or proof ranges are malformed', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const text = 'Sınırları doğrulanan kaynak.'
  const manifest = {
    format: 'text-lab-recompute-proof-v1',
    originalSourceId: 'browser-source-1',
    sourceDigest: '0123456789abcdef',
    representationSpaceId: 'RS-0123456789ABCDEF',
    config: {
      matrixSide: 64, identityRatio: 0.25, minimumLeafGraphemes: 4, hashSeed: 1,
      contextRetention: 0.8, parentTraceWeight: 0.15, siblingTraceWeight: 0.05,
      minimumTrace: 0.1, identitySimilarityWeight: 0.5, habitExponent: 2,
    },
    identityProofs: [{ start: 0, end: text.length + 1, identityDigest: 'fedcba9876543210' }],
  }
  const outside = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ name: 'Taşan kanıt', text, provenance: 'TextLab', analysisManifest: manifest }),
  })
  assert.equal(outside.response.status, 422)
  assert.equal(outside.body.code, 'INVALID_TEXT_LAB_MANIFEST')
  assert.equal(service.store.listSources().length, 0)

  const unknownConfig = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({
      name: 'Fazla ayar', text, provenance: 'TextLab',
      analysisManifest: {
        ...manifest,
        config: { ...manifest.config, injectedSetting: { unexpectedly: 'nested' } },
        identityProofs: [{ start: 0, end: text.length, identityDigest: 'fedcba9876543210' }],
      },
    }),
  })
  assert.equal(unknownConfig.response.status, 422)
  assert.equal(unknownConfig.body.code, 'INVALID_TEXT_LAB_MANIFEST')
  assert.equal(service.store.listSources().length, 0)
})

test('TextLab archive accepts only explicit profile/layout pairs while profileless legacy manifests stay byte-compatible', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const config = {
    matrixSide: 64, identityRatio: 0.125, minimumLeafGraphemes: 4, hashSeed: 1,
    contextRetention: 0.8, parentTraceWeight: 0.15, siblingTraceWeight: 0.05,
    minimumTrace: 0.1, identitySimilarityWeight: 0.5, habitExponent: 2,
  }
  const manifestFor = (text, representationProfile) => ({
    format: 'text-lab-recompute-proof-v1',
    originalSourceId: `browser-${text.length}-${representationProfile ?? 'legacy'}`,
    sourceDigest: '0123456789abcdef',
    representationSpaceId: 'RS-0123456789ABCDEF',
    config: representationProfile ? { ...config, representationProfile } : { ...config, identityRatio: 0.25 },
    identityProofs: [{ start: 0, end: text.length, identityDigest: 'fedcba9876543210' }],
  })

  const adaptiveText = 'Açık sekizde bir kimlik profili.'
  const adaptive = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({
      name: 'Adaptive arşiv', text: adaptiveText, provenance: 'TextLab',
      analysisManifest: manifestFor(adaptiveText, 'adaptive-eighth-id-v1'),
    }),
  })
  assert.equal(adaptive.response.status, 201)
  assert.equal(
    service.store.getSource(adaptive.body.id).provenance.textLabManifest.config.representationProfile,
    'adaptive-eighth-id-v1',
  )

  const mismatchedText = 'Profille ölçüsü uyuşmayan kayıt.'
  const mismatchedManifest = manifestFor(mismatchedText, 'adaptive-eighth-id-v1')
  mismatchedManifest.config.identityRatio = 0.25
  const mismatched = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ name: 'Uyuşmayan arşiv', text: mismatchedText, provenance: 'TextLab', analysisManifest: mismatchedManifest }),
  })
  assert.equal(mismatched.response.status, 422)
  assert.equal(mismatched.body.code, 'INVALID_TEXT_LAB_MANIFEST')

  const legacyText = 'Profilsiz tarihsel arşiv.'
  const legacyManifest = manifestFor(legacyText, null)
  const legacy = await jsonFetch(base, '/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({ name: 'Tarihsel arşiv', text: legacyText, provenance: 'TextLab', analysisManifest: legacyManifest }),
  })
  assert.equal(legacy.response.status, 201)
  assert.equal(
    Object.hasOwn(service.store.getSource(legacy.body.id).provenance.textLabManifest.config, 'representationProfile'),
    false,
  )
})
