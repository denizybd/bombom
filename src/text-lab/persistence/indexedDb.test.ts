import { describe, expect, it } from 'vitest'
import {
  ProjectWriteConflictError,
  resolveNextStorageRevision,
} from './indexedDb'

describe('TextLab storage revisions', () => {
  it('creates revision one only for an absent record', () => {
    expect(resolveNextStorageRevision(null, 0)).toBe(1)
  })

  it('increments an exact compare-and-swap revision', () => {
    expect(resolveNextStorageRevision(8, 8)).toBe(9)
  })

  it('rejects a stale tab snapshot', () => {
    expect(() => resolveNextStorageRevision(9, 8)).toThrow(ProjectWriteConflictError)
    expect(() => resolveNextStorageRevision(1, 0)).toThrow('başka bir sekmede')
  })
})
