import { analyzeText, PipelineCancelledError } from '../engine'
import type { WorkerRequest, WorkerResponse } from './protocol'

const cancelled = new Set<string>()
const scope = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null
  postMessage: (message: WorkerResponse, transfer?: Transferable[]) => void
}

function post(message: WorkerResponse, transfer?: Transferable[]) {
  scope.postMessage(message, transfer)
}

scope.onmessage = (event) => {
  const request = event.data
  if (request.type === 'cancel') {
    cancelled.add(request.requestId)
    return
  }

  const { requestId } = request
  cancelled.delete(requestId)
  void analyzeText(request.source, request.config, {
    shouldCancel: () => cancelled.has(requestId),
    onProgress: (progress) => post({ type: 'progress', requestId, progress }),
  })
    .then((analysis) => {
      if (cancelled.has(requestId)) {
        post({ type: 'cancelled', requestId })
        return
      }
      const transfer: Transferable[] = [
        analysis.similarities.buffer,
        analysis.habitProbabilities.buffer,
        ...analysis.nodes.flatMap((node) => [node.identity.buffer, node.relation.buffer]),
      ]
      post({ type: 'result', requestId, analysis }, transfer)
    })
    .catch((error: unknown) => {
      if (error instanceof PipelineCancelledError || cancelled.has(requestId)) {
        post({ type: 'cancelled', requestId })
        return
      }
      post({
        type: 'error',
        requestId,
        message: error instanceof Error ? error.message : 'Bilinmeyen bir işlem hatası oluştu.',
      })
    })
    .finally(() => cancelled.delete(requestId))
}
