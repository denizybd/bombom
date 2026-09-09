import assert from 'node:assert/strict'
import test from 'node:test'
import {
  TEACHER_WORKER_UNLOAD_TIMEOUT_MS,
  unloadTeacherModelAtWorkerExit,
} from '../training-worker-cleanup.mjs'

const teacherJob = (model = 'qwen2.5-coder:14b') => ({
  type: 'teacher-feed-campaign',
  payload: {
    campaign: {
      model,
      modelIdentity: { name: model },
    },
  },
})

test('worker exit unload uses a fresh live signal and the exact teacher model', async () => {
  const alreadyAborted = AbortSignal.abort(new Error('job signal'))
  let received
  const result = await unloadTeacherModelAtWorkerExit({
    job: { ...teacherJob(), signal: alreadyAborted },
    ollama: {
      async unload(model, options) {
        received = { model, signal: options.signal }
        return { model, unloaded: true }
      },
    },
  })

  assert.equal(result.attempted, true)
  assert.equal(result.unloaded, true)
  assert.equal(received.model, 'qwen2.5-coder:14b')
  assert.notEqual(received.signal, alreadyAborted)
  assert.equal(received.signal.aborted, false)
})

test('worker exit unload is teacher-only and does not invoke inference or quota paths', async () => {
  let calls = 0
  const result = await unloadTeacherModelAtWorkerExit({
    job: { type: 'full-cycle', payload: {} },
    ollama: { async unload() { calls += 1 } },
  })

  assert.deepEqual(result, { attempted: false })
  assert.equal(calls, 0)
})

test('worker exit unload swallows failures and independently bounds a stuck release', async () => {
  let cleanupSignal
  const started = Date.now()
  const result = await unloadTeacherModelAtWorkerExit({
    job: teacherJob(),
    timeoutMs: 20,
    ollama: {
      unload(_model, { signal }) {
        cleanupSignal = signal
        return new Promise(() => {})
      },
    },
  })

  assert.equal(result.attempted, true)
  assert.equal(result.unloaded, false)
  assert.equal(result.errorCode, 'OLLAMA_CLEANUP_TIMEOUT')
  assert.equal(cleanupSignal.aborted, true)
  assert.ok(Date.now() - started < 1_000)
  assert.ok(TEACHER_WORKER_UNLOAD_TIMEOUT_MS < 5_000)

  const rejected = await unloadTeacherModelAtWorkerExit({
    job: teacherJob(),
    ollama: { async unload() { throw Object.assign(new Error('offline'), { code: 'OLLAMA_OFFLINE' }) } },
  })
  assert.equal(rejected.unloaded, false)
  assert.equal(rejected.errorCode, 'OLLAMA_OFFLINE')
})
