import type {
  SynthesisAnalysisInput,
  SynthesisConfig,
  SynthesisProgress,
  SynthesisRun,
} from './types'

export interface SynthesisWorkerRequest {
  type: 'run'
  analyses: SynthesisAnalysisInput[]
  config: SynthesisConfig
}

export type SynthesisWorkerResponse =
  | { type: 'progress'; progress: SynthesisProgress }
  | { type: 'result'; run: SynthesisRun }
  | { type: 'error'; name: string; message: string; code?: string }
