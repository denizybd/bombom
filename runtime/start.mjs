import { realpathSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { createTrainingService } from '../server/index.mjs'
import { createThoughtRuntimeEngine } from '../src/thought-runtime/orchestration.ts'
import { createRealPipelineExecutor } from './core-runtime.mjs'
import { createIndependentStructuralBenchmarkRunner } from './structural-benchmark-runner.mjs'

export async function startRealTrainingService(options = {}) {
  const pipelineExecutor = options.pipelineExecutor ?? createRealPipelineExecutor()
  const thoughtEngine = options.thoughtEngine ?? createThoughtRuntimeEngine()
  const structuralBenchmarkReport = options.structuralBenchmarkReport ??
    createIndependentStructuralBenchmarkRunner({ thoughtEngine })
  const customExecutionDependency = Boolean(
    options.queue || options.store || options.jobExecutor || options.pipelineExecutor ||
    options.ollama || options.fetchImpl || options.stageHandlers ||
    options.structuralBenchmarkReport || options.forceJson,
  )
  const isolationEnabled = options.workerIsolation !== false &&
    process.env.BOMBOM_JOB_ISOLATION !== '0' &&
    !customExecutionDependency
  const previousWorkerEntry = process.env.BOMBOM_JOB_WORKER_ENTRY
  const previousIsolation = process.env.BOMBOM_JOB_ISOLATION
  if (isolationEnabled) {
    process.env.BOMBOM_JOB_WORKER_ENTRY = previousWorkerEntry || fileURLToPath(new URL('./training-worker.mjs', import.meta.url))
    process.env.BOMBOM_JOB_ISOLATION = '1'
  }
  let service
  try {
    service = await createTrainingService({
      ...options,
      pipelineExecutor,
      thoughtEngine,
      structuralBenchmarkReport,
    })
  } finally {
    if (previousWorkerEntry === undefined) delete process.env.BOMBOM_JOB_WORKER_ENTRY
    else process.env.BOMBOM_JOB_WORKER_ENTRY = previousWorkerEntry
    if (previousIsolation === undefined) delete process.env.BOMBOM_JOB_ISOLATION
    else process.env.BOMBOM_JOB_ISOLATION = previousIsolation
  }
  const port = Number(options.port ?? process.env.BOMBOM_TRAINING_PORT ?? 8787)
  const address = await service.listen({ port, host: '127.0.0.1' })
  return { service, address }
}

const isEntrypoint = (() => {
  if (!process.argv[1]) return false
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url))
  } catch {
    return false
  }
})()
if (isEntrypoint) {
  const { service, address } = await startRealTrainingService({
    dataDirectory: process.env.BOMBOM_TRAINING_DATA,
  })
  console.log(`Otonom Eğitim Servisi http://127.0.0.1:${address.port} adresinde gerçek motorlarla hazır.`)
  const shutdown = async () => {
    await service.close()
    process.exit(0)
  }
  process.once('SIGINT', shutdown)
  process.once('SIGTERM', shutdown)
}
