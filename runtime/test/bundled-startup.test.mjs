import assert from 'node:assert/strict'
import test from 'node:test'
import { spawn } from 'node:child_process'
import { access, mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { build } from 'vite'

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')

test('üretim paketi yalnız bir servis başlangıç noktası çalıştırır', { timeout: 30_000 }, async (t) => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), 'bombom-bundled-startup-'))
  const outputDirectory = path.join(temporaryRoot, 'bundle')
  const dataDirectory = path.join(temporaryRoot, 'data')
  t.after(async () => rm(temporaryRoot, { recursive: true, force: true }))

  await build({
    configFile: path.join(repositoryRoot, 'vite.training.config.ts'),
    build: { outDir: outputDirectory, emptyOutDir: true },
    logLevel: 'silent',
  })
  await access(path.join(outputDirectory, 'training-worker.mjs'))
  await access(path.join(outputDirectory, 'pipeline-worker.mjs'))
  await access(path.join(outputDirectory, 'index.aff'))
  await access(path.join(outputDirectory, 'index.dic'))

  const child = spawn(process.execPath, [path.join(outputDirectory, 'training-service.mjs')], {
    cwd: repositoryRoot,
    env: {
      ...process.env,
      BOMBOM_TRAINING_PORT: '0',
      BOMBOM_TRAINING_DATA: dataDirectory,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  t.after(() => {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM')
  })

  let stdout = ''
  let stderr = ''
  child.stdout.setEncoding('utf8')
  child.stderr.setEncoding('utf8')
  child.stdout.on('data', (chunk) => { stdout += chunk })
  child.stderr.on('data', (chunk) => { stderr += chunk })

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`Paketlenmiş servis başlamadı.\n${stdout}\n${stderr}`)), 10_000)
    const inspect = () => {
      if (!stdout.includes('gerçek motorlarla hazır')) return
      clearTimeout(timeout)
      resolve()
    }
    child.stdout.on('data', inspect)
    child.once('exit', (code, signal) => {
      if (!stdout.includes('gerçek motorlarla hazır')) {
        clearTimeout(timeout)
        reject(new Error(`Servis erken kapandı (${code ?? signal}).\n${stdout}\n${stderr}`))
      }
    })
  })

  await new Promise((resolve) => setTimeout(resolve, 150))
  assert.equal(child.exitCode, null, stderr)
  assert.doesNotMatch(stderr, /EADDRINUSE|Unhandled|uncaught/iu)
  assert.equal((stdout.match(/gerçek motorlarla hazır/gu) ?? []).length, 1)

  const port = Number(/http:\/\/127\.0\.0\.1:(\d+)/u.exec(stdout)?.[1])
  assert.ok(Number.isInteger(port) && port > 0, stdout)
  const request = async (pathname, init = {}) => {
    const response = await fetch(`http://127.0.0.1:${port}${pathname}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
    })
    const body = await response.json()
    assert.equal(response.ok, true, JSON.stringify(body))
    return body
  }
  const source = await request('/api/sources/text', {
    method: 'POST',
    body: JSON.stringify({
      name: 'Paket izolasyon kanıtı',
      text: 'Ali topu masaya bıraktı. Ayşe topu masadan aldı. Çocuk iki eylemi sırayla anlattı.',
      provenance: 'Deterministik paket testi',
    }),
  })
  await request(`/api/quarantine/${source.id}/approve`, { method: 'POST', body: '{}' })
  const queued = await request('/api/jobs', {
    method: 'POST',
    body: JSON.stringify({ type: 'full-cycle', payload: { sourceIds: [source.id], automationLevel: 'proposal', epochs: 1 } }),
  })
  const deadline = Date.now() + 30_000
  let completed
  while (Date.now() < deadline) {
    const jobs = await request('/api/jobs')
    completed = jobs.jobs.find((job) => job.id === queued.job.id && ['succeeded', 'failed', 'cancelled', 'paused'].includes(job.state))
    if (completed) break
    await new Promise((resolve) => setTimeout(resolve, 50))
  }
  assert.equal(completed?.state, 'succeeded', JSON.stringify(completed?.error ?? null))
  const eventLog = await request('/api/pipeline/events')
  const spawned = eventLog.events.find((event) => event.type === 'job-worker-spawned' && event.jobId === queued.job.id)
  const ready = eventLog.events.find((event) => event.type === 'job-worker-ready' && event.jobId === queued.job.id)
  const released = eventLog.events.find((event) => event.type === 'job-worker-released' && event.jobId === queued.job.id)
  assert.equal(spawned?.data?.isolation, 'process')
  assert.ok(Number.isInteger(ready?.data?.pid))
  assert.equal(released?.data?.pid, ready.data.pid)
  assert.equal(released?.data?.memoryReleased, true)
  assert.ok(released?.data?.memory?.peakWorkerRssBytes > 0)
  assert.ok(released?.data?.memory?.peakAggregateRssBytes >= released.data.memory.peakWorkerRssBytes)
  assert.throws(() => process.kill(ready.data.pid, 0), (error) => error?.code === 'ESRCH')

  child.kill('SIGTERM')
  await new Promise((resolve) => child.once('exit', resolve))
})
