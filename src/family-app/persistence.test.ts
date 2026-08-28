import { beforeAll, describe, expect, it } from 'vitest'
import { DEFAULT_TEXT_LAB_CONFIG } from '../text-lab/domain/config'
import { analyzeText } from '../text-lab/engine/pipeline'
import { runFamilyLearning } from '../families/engine'
import type { FamilyRun } from '../families/types'
import {
  FamilyProjectWriteConflictError,
  type FamilyProject,
  parseFamilyProject,
  resolveNextFamilyStorageRevision,
  stringifyFamilyProject,
  validateFamilyProject,
  validateFamilyRun,
} from './persistence'

function cloneRun(run: FamilyRun): FamilyRun {
  return structuredClone(run)
}

describe('Nöron Aileleri arşiv şeması', () => {
  let run: FamilyRun
  let project: FamilyProject

  beforeAll(async () => {
    const config = {
      ...DEFAULT_TEXT_LAB_CONFIG,
      matrixSide: 8,
      identityRatio: 0.5,
      minimumLeafGraphemes: 1,
    }
    const analyses = await Promise.all([
      analyzeText({ id: 'source-a', title: 'A', rawText: 'Ali kesti.' }, config),
      analyzeText({ id: 'source-b', title: 'B', rawText: 'Ali kesti!' }, config),
      analyzeText({ id: 'source-c', title: 'C', rawText: 'Ali kesti?' }, config),
    ])
    run = await runFamilyLearning(
      { analyses },
      {
        minimumFormationSimilarity: 0.1,
        minimumMembershipScore: 0,
        minimumCoreMembership: 0,
        minimumSupport: 2,
        minimumSources: 2,
        maximumFamilies: 16,
        utilityMinimumCases: 2,
        utilityAlternativeCount: 2,
      },
      { yieldControl: async () => Promise.resolve() },
    )
    project = {
      format: 'noron-aileleri-project',
      version: 1,
      id: `family-project:${run.fingerprint}`,
      name: 'Test aile modeli',
      storageRevision: 1,
      savedAt: '2026-08-28T12:00:00.000Z',
      run,
    }
  })

  it('gerçek motor koşusunu ve JSON round-trip projesini kayıpsız doğrular', () => {
    expect(validateFamilyRun(run)).toBe(run)
    const restored = parseFamilyProject(stringifyFamilyProject(project))
    expect(restored).toEqual(project)
    expect(validateFamilyProject(restored)).toBe(restored)
  })

  it('bozuk seyrek normu ve var olmayan aileye üyeliği reddeder', () => {
    const wrongNorm = cloneRun(run)
    wrongNorm.profiles[0].vector.norm = 0.25
    expect(() => validateFamilyRun(wrongNorm)).toThrow('normu')

    const danglingMembership = cloneRun(run)
    danglingMembership.memberships[0].familyId = 'missing-family'
    expect(() => validateFamilyRun(danglingMembership)).toThrow('yumuşak aile üyeliği')
  })

  it('G satırlarının normalize olmasını zorunlu tutar', () => {
    expect(run.familyTransitions.length).toBeGreaterThan(0)
    const corrupted = cloneRun(run)
    const origin = corrupted.familyTransitions[0].fromFamilyId
    for (const transition of corrupted.familyTransitions) {
      if (transition.fromFamilyId === origin) transition.probability *= 0.5
    }
    expect(() => validateFamilyRun(corrupted)).toThrow('satır bazında normalize')
  })

  it('seyrek nöron geçişindeki bozuk ref ve olasılığı reddeder', () => {
    expect(run.neuronTransitions.length).toBeGreaterThan(0)
    const corrupted = cloneRun(run)
    corrupted.neuronTransitions[0].toMemberKey = 'missing-profile'
    corrupted.neuronTransitions[0].probability = Number.NaN
    expect(() => validateFamilyRun(corrupted)).toThrow('nöron geçiş')
  })

  it('compare-and-swap revision ile eski sekme yazımını reddeder', () => {
    expect(resolveNextFamilyStorageRevision(null, 0)).toBe(1)
    expect(resolveNextFamilyStorageRevision(4, 4)).toBe(5)
    expect(() => resolveNextFamilyStorageRevision(5, 4)).toThrow(FamilyProjectWriteConflictError)
  })
})
