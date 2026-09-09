import { TEACHER_FEED_JOB_TYPE } from '../server/teacher-feed.mjs'

// The queue gives a terminal worker five seconds before SIGTERM. Keep the
// independent cleanup deadline below that grace period so the SQLite handle
// and IPC channel still close normally even when Ollama does not respond.
export const TEACHER_WORKER_UNLOAD_TIMEOUT_MS = 4_000

function cleanupModel(job) {
  const campaign = job?.payload?.campaign
  return String(campaign?.modelIdentity?.name ?? campaign?.model ?? '').trim()
}

function cleanupTimeoutError(timeoutMs) {
  const error = new Error(`Ollama worker kapanış boşaltması ${timeoutMs} ms içinde tamamlanmadı.`)
  error.code = 'OLLAMA_CLEANUP_TIMEOUT'
  return error
}

/**
 * Best-effort lifecycle cleanup for an isolated teacher worker.
 *
 * This deliberately receives no job AbortSignal. Pause/cancel/service-stop
 * abort the inference signal, but model release needs its own short lifetime.
 * The timeout race also bounds a broken/mocked unload implementation that
 * ignores AbortSignal. Calling OllamaTeacher.unload is quota-free by contract.
 */
export async function unloadTeacherModelAtWorkerExit({
  job,
  ollama,
  timeoutMs = TEACHER_WORKER_UNLOAD_TIMEOUT_MS,
} = {}) {
  if (job?.type !== TEACHER_FEED_JOB_TYPE || typeof ollama?.unload !== 'function') {
    return { attempted: false }
  }

  const boundedTimeoutMs = Math.min(
    TEACHER_WORKER_UNLOAD_TIMEOUT_MS,
    Math.max(1, Math.trunc(Number(timeoutMs) || TEACHER_WORKER_UNLOAD_TIMEOUT_MS)),
  )
  const controller = new AbortController()
  let timeout
  const deadline = new Promise((_, reject) => {
    timeout = setTimeout(() => {
      const error = cleanupTimeoutError(boundedTimeoutMs)
      controller.abort(error)
      reject(error)
    }, boundedTimeoutMs)
  })

  const model = cleanupModel(job)
  try {
    const release = await Promise.race([
      Promise.resolve().then(() => ollama.unload(model, { signal: controller.signal })),
      deadline,
    ])
    return {
      attempted: true,
      model: String(release?.model ?? model),
      unloaded: release?.unloaded !== false,
    }
  } catch (error) {
    // Cleanup must never replace the real job settlement or keep the worker
    // alive. Returning bounded diagnostics makes this helper easy to test
    // without introducing another durable event during shutdown.
    return {
      attempted: true,
      model,
      unloaded: false,
      errorCode: String(error?.code ?? error?.name ?? 'OLLAMA_UNLOAD_FAILED').slice(0, 120),
    }
  } finally {
    clearTimeout(timeout)
  }
}
