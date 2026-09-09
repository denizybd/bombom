import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { startRealTrainingService } from '../dist-training-service/training-service.mjs'

const corpus = [
  'Kedi bahçede kırmızı topu yuvarladı. Köpek bahçede kırmızı topu yakaladı. Çocuk iki hayvanı dikkatle izledi.',
  'Kuş odada küçük çanı salladı. Çocuk odada küçük çanı duydu. Öğretmen bu olayı defterine yazdı.',
  'Deniz masada sıcak çorbayı karıştırdı. Ece masada sıcak çorbayı tattı. Aşçı iki kişiye gülümseyerek baktı.',
  'Robot atölyede metal kutuyu taşıdı. Usta atölyede metal kutuyu açtı. Öğrenci yapılan hareketleri sırayla anlattı.',
  'Doktor klinikte yeni raporu okudu. Hemşire klinikte yeni raporu sakladı. Hasta sonucu sakin biçimde dinledi.',
  'Mert parkta mavi bisikleti sürdü. Ayşe parkta mavi bisikleti durdurdu. Arkadaşları bu kısa yarışı izledi.',
]

const temporary = await mkdtemp(path.join(os.tmpdir(), 'bombom-real-runtime-'))
let service
try {
  const started = await startRealTrainingService({
    port: 0,
    dataDirectory: temporary,
    forceJson: true,
  })
  service = started.service
  const base = `http://127.0.0.1:${started.address.port}`
  const request = async (pathname, init = {}) => {
    const response = await fetch(`${base}${pathname}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
    })
    const value = await response.json()
    if (!response.ok) throw Object.assign(new Error(value.message ?? 'İstek başarısız.'), { value })
    return value
  }

  const sourceIds = []
  for (const [index, text] of corpus.entries()) {
    const source = await request('/api/sources/text', {
      method: 'POST',
      body: JSON.stringify({ name: `Kontrollü kaynak ${index + 1}`, text, provenance: 'Runtime smoke korpusu' }),
    })
    sourceIds.push(source.id)
    await request(`/api/quarantine/${source.id}/approve`, { method: 'POST', body: '{}' })
  }
  await request('/api/run/start', {
    method: 'POST',
    body: JSON.stringify({ sourceIds, automationLevel: 'proposal', epochs: 1, label: 'Gerçek motor smoke testi' }),
  })

  const deadline = Date.now() + 120_000
  let completed
  while (Date.now() < deadline) {
    const queue = await request('/api/queue')
    completed = queue.find((item) => item.state === 'complete' || item.state === 'failed' || item.state === 'cancelled')
    if (completed) break
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  assert.ok(completed, 'Gerçek eğitim çevrimi zamanında tamamlanmadı.')
  assert.equal(completed.state, 'complete', completed.message)
  const snapshots = await request('/api/snapshots')
  const artifacts = await request('/api/artifacts')
  assert.ok(snapshots.length >= 1)
  assert.ok(artifacts.artifacts.some((artifact) => artifact.kind === 'runtime-model-state'))
  const storedSnapshot = service.store.getSnapshot(snapshots[0].id)
  assert.equal(storedSnapshot.evaluation.runtimeVerified, true, 'Bütün gerçek motor aşamaları doğrulanmış olmalı.')
  assert.ok(Number.isInteger(storedSnapshot.metrics.birthProposalCount))
  assert.ok(Number.isInteger(storedSnapshot.metrics.emergentLanguageFamilyCount))
  console.log(JSON.stringify({
    jobState: completed.state,
    snapshotState: snapshots[0].state,
    artifactKinds: [...new Set(artifacts.artifacts.map((artifact) => artifact.kind))],
    metrics: {
      ...snapshots[0].metrics,
      birthProposalCount: storedSnapshot.metrics.birthProposalCount,
      eligibleBirthProposalCount: storedSnapshot.metrics.eligibleBirthProposalCount,
      emergentLanguageFamilyCount: storedSnapshot.metrics.emergentLanguageFamilyCount,
      eligibleEmergentLanguageFamilyCount: storedSnapshot.metrics.eligibleEmergentLanguageFamilyCount,
    },
    evaluationChecks: storedSnapshot.evaluation.checks,
  }, null, 2))
} finally {
  await service?.close()
  await rm(temporary, { recursive: true, force: true })
}
