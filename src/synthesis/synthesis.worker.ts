import { runSynthesis } from './engine'
import type { SynthesisWorkerRequest, SynthesisWorkerResponse } from './workerProtocol'

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<SynthesisWorkerRequest>) => void) | null
  postMessage: (message: SynthesisWorkerResponse, transfer?: Transferable[]) => void
}

function post(message: SynthesisWorkerResponse, transfer?: Transferable[]): void {
  scope.postMessage(message, transfer)
}

scope.onmessage = (event) => {
  if (event.data.type !== 'run') return
  const { analyses, config } = event.data
  void runSynthesis(analyses, config, {
    onProgress: (progress) => post({ type: 'progress', progress }),
  })
    .then((run) => {
      const transfer = run.candidates.flatMap((candidate) => [
        candidate.identity.buffer,
        candidate.relation.buffer,
      ])
      post({ type: 'result', run }, transfer)
    })
    .catch((cause: unknown) => {
      const error = cause instanceof Error ? cause : new Error('Bilinmeyen bir sentez hatası oluştu.')
      post({
        type: 'error',
        name: error.name,
        message: error.message,
        code: 'code' in error && typeof error.code === 'string' ? error.code : undefined,
      })
    })
}
