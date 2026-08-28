import { FamilyCancelledError, runFamilyLearning } from './engine'
import type { FamilyWorkerRequest, FamilyWorkerResponse } from './workerProtocol'

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<FamilyWorkerRequest>) => void) | null
  postMessage: (message: FamilyWorkerResponse) => void
}

const cancelled = new Set<string>()
let activeRunToken: string | null = null

function post(message: FamilyWorkerResponse): void {
  scope.postMessage(message)
}

scope.onmessage = (event) => {
  const request = event.data
  if (request.type === 'cancel') {
    cancelled.add(request.runToken)
    return
  }
  if (activeRunToken && activeRunToken !== request.runToken) cancelled.add(activeRunToken)
  activeRunToken = request.runToken
  cancelled.delete(request.runToken)
  const runToken = request.runToken
  void runFamilyLearning(request.input, request.config, {
    shouldCancel: () => cancelled.has(runToken) || activeRunToken !== runToken,
    onProgress: (progress) => post({ type: 'progress', runToken, progress }),
  })
    .then((run) => {
      if (cancelled.has(runToken) || activeRunToken !== runToken) {
        post({ type: 'cancelled', runToken })
        return
      }
      post({ type: 'result', runToken, run })
    })
    .catch((cause: unknown) => {
      const error = cause instanceof Error ? cause : new Error('Bilinmeyen bir nöron aileleri hatası oluştu.')
      if (
        error instanceof FamilyCancelledError ||
        error.name === 'FamilyCancelledError' ||
        error.message === 'FAMILY_CANCELLED' ||
        cancelled.has(runToken)
      ) {
        post({ type: 'cancelled', runToken })
        return
      }
      post({
        type: 'error',
        runToken,
        name: error.name,
        message: error.message,
        code: 'code' in error && typeof error.code === 'string' ? error.code : undefined,
      })
    })
    .finally(() => {
      cancelled.delete(runToken)
      if (activeRunToken === runToken) activeRunToken = null
    })
}

