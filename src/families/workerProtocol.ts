import type {
  FamilyConfig,
  FamilyProgress,
  FamilyRun,
  FamilyRunInput,
} from './types'

export type FamilyWorkerRequest =
  | {
      type: 'run'
      runToken: string
      input: FamilyRunInput
      config?: Partial<FamilyConfig>
    }
  | {
      type: 'cancel'
      runToken: string
    }

export type FamilyWorkerResponse =
  | { type: 'progress'; runToken: string; progress: FamilyProgress }
  | { type: 'result'; runToken: string; run: FamilyRun }
  | { type: 'cancelled'; runToken: string }
  | { type: 'error'; runToken: string; name: string; message: string; code?: string }

