import type { PipelineProgress, TextLabAnalysis, TextLabConfig, TextSource } from '../domain/types'

export type WorkerRequest =
  | {
      type: 'analyze'
      requestId: string
      source: TextSource
      config: TextLabConfig
    }
  | {
      type: 'cancel'
      requestId: string
    }

export type WorkerResponse =
  | {
      type: 'progress'
      requestId: string
      progress: PipelineProgress
    }
  | {
      type: 'result'
      requestId: string
      analysis: TextLabAnalysis
    }
  | {
      type: 'cancelled'
      requestId: string
    }
  | {
      type: 'error'
      requestId: string
      message: string
    }

