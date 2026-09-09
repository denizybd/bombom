const VERSION = 1
let token = null
let requestSequence = 0
const pending = new Map()

function send(message) {
  return new Promise((resolve, reject) => {
    process.send(message, (error) => error ? reject(error) : resolve())
  })
}

function request(action, payload) {
  const requestId = `fixture-${++requestSequence}`
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject })
    void send({ type: 'worker-request', version: VERSION, token, requestId, action, payload }).catch(reject)
  })
}

process.on('message', (message) => {
  if (!message || message.version !== VERSION) return
  if (message.type === 'worker-response' && message.token === token) {
    const waiter = pending.get(message.requestId)
    if (!waiter) return
    pending.delete(message.requestId)
    if (message.error) waiter.reject(Object.assign(new Error(message.error.message), { code: message.error.code }))
    else waiter.resolve(message.result)
    return
  }
  if (message.type === 'worker-abort' && message.token === token) {
    void send({
      type: 'worker-terminal', version: VERSION, token, ok: false,
      error: { code: message.reason?.code ?? 'JOB_CANCELLED', message: message.reason?.message ?? 'cancelled' },
    }).finally(() => {
      process.disconnect()
      process.exitCode = 1
    })
    return
  }
  if (message.type !== 'worker-run') return
  token = message.token
  void (async () => {
    const allocation = Buffer.alloc(64 * 1024 * 1024, 0x5a)
    await request('event', { type: 'fixture-worker-running', data: { pid: process.pid } })
    await request('progress', { stage: 'fixture-stage', progress: 0.5 })
    if (process.env.BOMBOM_ISOLATION_FIXTURE_MODE === 'hang') return
    if (process.env.BOMBOM_ISOLATION_FIXTURE_MODE === 'atomic-candidate') {
      const { createTrainingStore } = await import('../../../server/store.mjs')
      const store = await createTrainingStore({ dataDirectory: message.dataDirectory, requireSqlite: true })
      try {
        const active = store.getActiveSnapshot()
        const snapshotId = `fixture-atomic-candidate-${message.jobId}`
        const result = {
          released: true,
          committed: false,
          awaitingApproval: true,
          automaticActivation: false,
          snapshotId,
        }
        store.completeClaimedJobWithCandidateOnSnapshotHeadBase(
          message.jobId,
          message.owner,
          message.attempt,
          { id: snapshotId, parentId: active.id, status: 'candidate', artifactIds: [], sourceIds: [] },
          {
            snapshotId: active.id,
            headKind: 'fixture-absent-head',
            headArtifactId: null,
            headContentHash: null,
            headFingerprint: null,
          },
          result,
          () => true,
        )
        // Keep the child alive past the shortest heartbeat interval. The
        // parent must recognize durable atomic success instead of treating the
        // cleared lease as ownership loss.
        await new Promise((resolve) => setTimeout(resolve, 1_900))
        await send({
          type: 'worker-terminal', version: VERSION, token, ok: true,
          result: { ...result, atomicCandidateJobCompletion: true },
        })
      } finally {
        store.close()
      }
      process.disconnect()
      process.exitCode = 0
      return
    }
    await send({
      type: 'worker-terminal', version: VERSION, token, ok: true,
      result: { isolatedPid: process.pid, allocatedBytes: allocation.byteLength },
    })
    process.disconnect()
    process.exitCode = 0
  })().catch(async (error) => {
    await send({ type: 'worker-terminal', version: VERSION, token, ok: false, error: { code: error.code ?? 'FIXTURE_FAILED', message: error.message } }).catch(() => {})
    process.disconnect()
    process.exitCode = 1
  })
})

await send({ type: 'worker-ready', version: VERSION, pid: process.pid })
