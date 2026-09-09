import { randomUUID } from 'node:crypto'
import { fork } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'

const VERSION = 1
const HANDSHAKE_MS = 10_000
const ABORT_GRACE_MS = 5_000
const MAX_DIAGNOSTIC_BYTES = 16 * 1024

/**
 * Run exactly one deterministic training milestone in a disposable process.
 * The coordinator retains teacher state; all large analysis/family/language
 * graphs die with this child before Ollama is allowed to load again.
 */
export async function runEphemeralPipeline({
  workerEntry,
  dataDirectory,
  pipelineJob,
  context,
  owner,
  attempt,
  serviceParentRss = () => 0,
  forkImpl = fork,
}) {
  const resolvedEntry = path.resolve(workerEntry)
  if (!existsSync(resolvedEntry)) throw unavailableError('pipeline-worker-entry-missing')
  const token = randomUUID()
  const ancestorRss = () => Math.max(0, Number(serviceParentRss()) || 0) + process.memoryUsage().rss
  const ramMb = Number(context.config.budgets.ramMb)
  const heapLimitMb = Math.max(384, Math.floor(ramMb - Math.ceil(ancestorRss() / (1024 * 1024)) - 512))
  const child = forkImpl(resolvedEntry, [], {
    cwd: process.cwd(),
    env: { ...process.env, BOMBOM_PIPELINE_WORKER_CHILD: '1' },
    execArgv: safeExecArgv(heapLimitMb),
    serialization: 'advanced',
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
  })
  let diagnostic = ''
  const capture = (chunk) => { diagnostic = `${diagnostic}${String(chunk)}`.slice(-MAX_DIAGNOSTIC_BYTES) }
  child.stdout?.on('data', capture)
  child.stderr?.on('data', capture)

  const resultPromise = new Promise((resolve, reject) => {
    let terminal = null
    let ready = false
    let settled = false
    let resourceTimer = null
    let killTimer = null
    const handshakeTimer = setTimeout(() => {
      if (!ready) {
        child.kill('SIGTERM')
        killTimer = setTimeout(() => child.kill('SIGKILL'), ABORT_GRACE_MS)
        killTimer.unref?.()
      }
    }, HANDSHAKE_MS)
    handshakeTimer.unref?.()

    const cleanup = () => {
      clearTimeout(handshakeTimer)
      clearTimeout(killTimer)
      clearInterval(resourceTimer)
      context.signal.removeEventListener('abort', onAbort)
      child.removeAllListeners('message')
      child.removeAllListeners('error')
      child.removeAllListeners('exit')
    }
    const done = (error, value) => {
      if (settled) return
      settled = true
      cleanup()
      if (error) reject(error)
      else resolve(value)
    }
    const send = (message) => {
      if (!child.connected) return false
      try { child.send(message, (error) => { if (error) capture(error.message) }); return true } catch (error) { capture(error.message); return false }
    }
    const respond = (requestId, result, error) => send({
      type: 'worker-response', version: VERSION, token, requestId,
      ...(error ? { error: serializeError(error) } : { result }),
    })
    const onAbort = () => {
      send({ type: 'worker-abort', version: VERSION, token, reason: serializeError(context.signal.reason) })
      killTimer = setTimeout(() => child.kill('SIGKILL'), ABORT_GRACE_MS)
      killTimer.unref?.()
    }
    context.signal.addEventListener('abort', onAbort, { once: true })
    if (context.signal.aborted) onAbort()

    child.on('message', async (message) => {
      if (!message || message.version !== VERSION) return
      if (message.type === 'worker-ready') {
        if (ready || settled) return
        ready = true
        clearTimeout(handshakeTimer)
        if (context.signal.aborted) return
        send({
          type: 'worker-run', version: VERSION, token,
          dataDirectory,
          jobId: context.jobId,
          owner,
          attempt,
          pipelineJob,
          ancestorRssBytes: ancestorRss(),
        })
        resourceTimer = setInterval(() => send({
          type: 'worker-resource-baseline', version: VERSION, token,
          ancestorRssBytes: ancestorRss(),
        }), Math.max(250, Number(context.config.resourcePolicy?.sampleIntervalMs ?? 1_000)))
        resourceTimer.unref?.()
        return
      }
      if (message.token !== token) return
      if (message.type === 'worker-request') {
        try {
          if (message.action === 'event') respond(message.requestId, await context.emit(message.payload))
          else if (message.action === 'progress') respond(message.requestId, await context.progress(message.payload?.stage, message.payload?.progress))
          else throw Object.assign(new Error('Bilinmeyen pipeline worker isteği.'), { code: 'JOB_WORKER_PROTOCOL_ERROR' })
        } catch (error) { respond(message.requestId, undefined, error) }
        return
      }
      if (message.type === 'worker-notification' && message.action === 'event') {
        void context.emit(message.payload).catch((error) => capture(error.message))
        return
      }
      if (message.type === 'worker-terminal') terminal = message
    })
    child.once('error', (error) => done(workerError('JOB_WORKER_SPAWN_FAILED', error.message, diagnostic, true)))
    child.once('exit', (code, signal) => {
      clearTimeout(handshakeTimer)
      clearTimeout(killTimer)
      clearInterval(resourceTimer)
      void context.emit({
        type: 'pipeline-worker-released',
        stage: 'teacher-feed:pipeline-worker-released',
        level: terminal?.ok === true && code === 0 ? 'info' : 'warning',
        data: {
          pid: child.pid ?? null,
          exitCode: code,
          signal: signal ?? null,
          memoryReleased: true,
          ...(terminal?.workerMetrics ? { memory: terminal.workerMetrics } : {}),
        },
      }).catch((error) => capture(error.message)).finally(() => {
        if (!ready) return done(workerError('JOB_WORKER_HANDSHAKE_TIMEOUT', 'Pipeline worker hazır olmadı.', diagnostic, false))
        if (terminal?.ok === true && code === 0) return done(null, terminal.result)
        if (terminal?.ok === false) return done(deserializeError(terminal.error, diagnostic))
        if (context.signal.aborted) return done(deserializeError(serializeError(context.signal.reason), diagnostic))
        return done(workerError('JOB_WORKER_EXITED', `Pipeline worker sonuçsuz kapandı (kod=${code ?? 'null'}, sinyal=${signal ?? 'yok'}).`, diagnostic, true))
      })
    })
  })
  resultPromise.catch(() => {})
  try {
    await context.emit({
      type: 'pipeline-worker-spawned',
      stage: 'teacher-feed:pipeline-worker-starting',
      data: { pid: child.pid ?? null, isolation: 'ephemeral-process', heapLimitMb },
    })
  } catch (error) {
    child.kill('SIGTERM')
    await resultPromise.catch(() => {})
    throw error
  }
  return resultPromise
}

function serializeError(error) {
  return {
    code: String(error?.code ?? 'PIPELINE_FAILED'),
    message: String(error?.message ?? error).slice(0, 2_000),
    ...(error?.retryable === true ? { retryable: true } : {}),
    ...(error?.resourceState ? { resourceState: structuredClone(error.resourceState) } : {}),
  }
}

function deserializeError(value, diagnostic) {
  const error = workerError(value?.code ?? 'JOB_WORKER_FAILED', value?.message ?? 'Pipeline worker başarısız oldu.', diagnostic, value?.retryable === true)
  if (value?.resourceState) error.resourceState = structuredClone(value.resourceState)
  return error
}

function workerError(code, message, diagnostic, retryable) {
  const detail = String(diagnostic ?? '').trim().slice(-2_000)
  const error = new Error(detail ? `${message} Ayrı süreç tanısı: ${detail}` : message)
  error.code = String(code)
  error.retryable = retryable === true
  return error
}

function unavailableError(reason) {
  const error = new Error(`Deterministik kilometre taşı ayrı süreç olmadan başlatılmadı (${reason}).`)
  error.code = 'JOB_WORKER_UNAVAILABLE'
  return error
}

function safeExecArgv(heapLimitMb) {
  const result = []
  for (let index = 0; index < process.execArgv.length; index += 1) {
    const argument = process.execArgv[index]
    if (['-e', '-p', '--eval', '--print'].includes(argument)) { index += 1; continue }
    if (
      argument.startsWith('--inspect') || argument.startsWith('--test') ||
      argument.startsWith('--input-type') || argument.startsWith('--eval=') ||
      argument.startsWith('--print=') || argument === '--watch' ||
      argument.startsWith('--max-old-space-size')
    ) continue
    result.push(argument)
  }
  return [...result, `--max-old-space-size=${heapLimitMb}`]
}
