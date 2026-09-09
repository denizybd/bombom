import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { request as createHttpRequest } from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { startRealTrainingService } from '../dist-training-service/training-service.mjs'

const model = String(process.argv[2] || 'qwen2.5-coder:14b').trim()
const requestedCount = Number(process.argv[3] || 4)
assert.ok(model, 'Öğretmen model adı boş olamaz.')
assert.ok(Number.isInteger(requestedCount) && requestedCount >= 4 && requestedCount <= 48, 'Smoke örnek sayısı 4–48 arasında olmalı.')

const temporary = await mkdtemp(path.join(os.tmpdir(), 'bombom-ollama-teacher-'))
const keepData = process.env.BOMBOM_KEEP_SMOKE_DATA === '1'
let service
const startedAt = Date.now()

try {
  const started = await startRealTrainingService({
    port: 0,
    dataDirectory: temporary,
  })
  service = started.service
  service.store.updateConfig({
    ollama: { model, timeoutMs: 300_000, retries: 1 },
    automation: { autoRun: false, level: 'proposal' },
    budgets: { maxRunMinutes: 20 },
  })
  const base = `http://127.0.0.1:${started.address.port}`
  // Node fetch has a finite Undici response-header timeout. A cold 14B model can
  // legitimately need longer before the local service has a JSON response, so
  // the smoke client deliberately uses a no-timeout loopback HTTP request.
  const request = (pathname, init = {}) => new Promise((resolve, reject) => {
    const body = init.body ?? null
    const headers = {
      Accept: 'application/json',
      ...(body ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } : {}),
      ...(init.headers ?? {}),
    }
    const outgoing = createHttpRequest(`${base}${pathname}`, { method: init.method ?? 'GET', headers }, (response) => {
      const chunks = []
      response.on('data', (chunk) => chunks.push(chunk))
      response.on('error', reject)
      response.on('end', () => {
        const raw = Buffer.concat(chunks).toString('utf8')
        let value = null
        try { value = raw ? JSON.parse(raw) : null } catch (cause) {
          reject(Object.assign(new Error('Yerel servis geçerli JSON döndürmedi.'), { cause, raw: raw.slice(0, 500) }))
          return
        }
        if ((response.statusCode ?? 500) < 200 || (response.statusCode ?? 500) >= 300) {
          reject(Object.assign(new Error(value?.message ?? value?.error?.message ?? `HTTP ${response.statusCode}`), { value }))
          return
        }
        resolve(value)
      })
    })
    outgoing.on('error', reject)
    outgoing.end(body)
  })

  console.log(`[1/5] ${model} yerel Ollama üzerinde doğrulanıyor…`)
  const models = await request('/api/ollama/models')
  assert.ok(models.some((item) => item.name === model), `${model} Ollama model listesinde bulunamadı.`)

  console.log(`[2/5] ${requestedCount} yapılandırılmış Türkçe mikro-paragraf üretiliyor…`)
  const generated = await request('/api/teacher/generate', {
    method: 'POST',
    body: JSON.stringify({
      model,
      language: 'tr',
      diversity: 'focused',
      count: requestedCount,
      topic: 'Türkçe cümlelerde özne, araç, nesne, eylem ve neden-sonuç örüntüleri',
      objective: 'Her örnek tam dört kısa cümlelik bağımsız bir mikro-paragraf olsun. İlk cümle “Önce” ile bir özne-araç-nesne-eylem ilişkisi kursun; ikinci cümle “Sonra” ile sonucu anlatsın; üçüncü cümle “Çünkü” ile gerekçe versin; dördüncü cümle kısa bir soru veya olumsuz karşı örnek olsun. İskeleti örnekler arasında koru, kişi, araç, nesne, yer ve fiilleri değiştir. Böylece aynı işlev farklı sözcüklerle tekrar edilsin.',
    }),
  })
  assert.ok(generated.accepted)
  assert.ok(generated.quarantineItemIds.length >= 3, '14B en az üç benzersiz kaynak üretemedi.')

  console.log(`[3/5] ${generated.quarantineItemIds.length} yeni örnek karantinadan kontrollü biçimde onaylanıyor…`)
  for (const sourceId of generated.quarantineItemIds) {
    await request(`/api/quarantine/${encodeURIComponent(sourceId)}/approve`, {
      method: 'POST',
      body: JSON.stringify({ reason: 'Yalıtılmış 14B besleme smoke testi' }),
    })
  }

  console.log('[4/5] Gerçek TextLab → sentez → aile → dil deneyimi hattı çalıştırılıyor…')
  const startedRun = await request('/api/run/start', {
    method: 'POST',
    body: JSON.stringify({
      sourceIds: generated.quarantineItemIds,
      automationLevel: 'proposal',
      epochs: 1,
      label: `Yalıtılmış ${model} öğretmen smoke testi`,
    }),
  })
  const jobId = startedRun.job?.id
  assert.ok(jobId, 'Eğitim işi kimliği dönmedi.')
  const deadline = Date.now() + 10 * 60_000
  let job
  let lastStage = ''
  while (Date.now() < deadline) {
    // The real family stage is CPU-heavy. Poll the co-located durable record
    // instead of opening hundreds of loopback sockets while that stage yields.
    job = service.store.getJob(jobId)
    if (job.stage !== lastStage) {
      lastStage = job.stage
      console.log(`      aşama: ${job.stage} · %${Math.round(Number(job.progress ?? 0) * 100)}`)
    }
    if (['succeeded', 'failed', 'cancelled'].includes(job.state)) break
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  assert.ok(job, 'Eğitim işi okunamadı.')
  assert.equal(job.state, 'succeeded', job.error?.message ?? job.message ?? 'Eğitim çevrimi tamamlanamadı.')

  console.log('[5/5] Aile kanıtları ve sentetik veri güvenlik kapısı denetleniyor…')
  const inspection = await request('/api/families/inspection?limit=100&memberLimit=4&transitionLimit=4')
  assert.equal(inspection.available, true, inspection.reason)
  assert.ok(inspection.summary.profileCount > 0, '14B metni nöron profillerine ulaşmadı.')
  assert.ok(inspection.summary.familyCount > 0, '14B örnekleri otomatik aile oluşturamadı.')

  const snapshots = service.store.listSnapshots()
  assert.equal(snapshots.length, 1, 'Yalıtılmış test tam bir aday sürüm üretmeliydi.')
  const snapshot = snapshots[0]
  assert.equal(service.store.getActiveSnapshot(), undefined, 'Tamamı sentetik korpus etkin modele dönüşmemeliydi.')
  assert.equal(snapshot.status, 'rejected', 'Tamamı sentetik aday güvenlik kapısında reddedilmeliydi.')
  assert.equal(snapshot.evaluation?.runtimeVerified, false, 'Sentetik holdout gerçek kaynak kanıtı sayılmamalıydı.')

  const sampleFamilies = inspection.families.slice(0, 5).map((family) => ({
    id: family.id,
    regionHypothesis: family.region,
    lifecycle: family.lifecycle,
    sources: family.sourceCount,
    support: family.supportCount,
    cohesion: family.cohesion,
    utilityStatus: family.utilityEvidence?.status ?? 'not-measured',
    topMembers: family.topMembers.slice(0, 3).map((member) => member.text),
  }))
  console.log(JSON.stringify({
    passed: true,
    isolation: { persistentStoreTouched: false, temporaryDataRemoved: !keepData },
    teacher: {
      model,
      requested: requestedCount,
      generated: generated.generatedCount,
      inserted: generated.quarantineItemIds.length,
      duplicates: generated.rejectedCount,
      usage: service.store.getUsage(),
    },
    learned: inspection.summary,
    sampleFamilies,
    safetyGate: {
      corpusSyntheticRatio: 1,
      snapshotStatus: snapshot.status,
      activeSnapshotId: null,
      runtimeVerified: snapshot.evaluation?.runtimeVerified,
      reason: snapshot.evaluation?.reason ?? 'Tamamı sentetik korpus gerçek holdout kanıtı sayılmadı.',
    },
    elapsedSeconds: Math.round((Date.now() - startedAt) / 100) / 10,
    temporaryDataDirectory: keepData ? temporary : undefined,
  }, null, 2))
} finally {
  await service?.close()
  if (!keepData) await rm(temporary, { recursive: true, force: true })
}
