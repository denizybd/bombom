import { describe, expect, it } from 'vitest'
import { createFixtureFamilyRun } from '../language-experience/testFixtures'
import { boundedLogOdds } from '../language-experience/deterministic'
import { createLanguageOutputScaffold, LANGUAGE_OUTPUT_FAMILY_IDS } from './scaffold'
import {
  computeLanguageOutputHeadFingerprint,
  computeLanguageOutputNurseryFingerprint,
  computeLanguageOutputProvisionalNeuronFingerprint,
  createLanguageOutputHead,
  finalizeLanguageOutputHead,
  migrateLanguageOutputHead,
  validateLanguageOutputHead,
} from './state'
import type {
  LanguageOutputHeadV1,
  LanguageOutputNurseryProposal,
  LanguageOutputProvisionalNeuronV1,
} from './types'

describe('plastik dil çıktı başlığı', () => {
  it('aynı scaffold ve FamilyRun ile bit düzeyinde aynı boş başlığı kurar', () => {
    const scaffold = createLanguageOutputScaffold()
    const familyRun = createFixtureFamilyRun()
    const first = createLanguageOutputHead(scaffold, familyRun)
    const second = createLanguageOutputHead(scaffold, familyRun)

    expect(first).toEqual(second)
    expect(first.revision).toBe(0)
    expect(first.familyRunFingerprint).toBe(familyRun.fingerprint)
    expect(first.representationSpaceId).toBe(familyRun.representationSpaceId)
    expect(first.transitionEvidence).toEqual([])
    expect(first.membershipEvidence).toEqual([])
    expect(first.outputNeurons).toEqual([])
    validateLanguageOutputHead(first, scaffold)
  })

  it('bozuk fingerprint ve kanıt skoru kabul etmez', () => {
    const scaffold = createLanguageOutputScaffold()
    const head = createLanguageOutputHead(scaffold, createFixtureFamilyRun())
    expect(() => validateLanguageOutputHead({ ...head, fingerprint: 'bozuk' }, scaffold)).toThrow(/fingerprint/u)

    const malformed: LanguageOutputHeadV1 = {
      ...head,
      transitionEvidence: [{
        kind: 'family-transition',
        fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder,
        toKey: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense,
        excitatoryEvidence: 1,
        inhibitoryEvidence: 0,
        score: boundedLogOdds(1, 0) + 1,
        updateCount: 1,
        sourceDigests: ['d1'],
        sourceGroupIds: ['s1'],
      }],
    }
    expect(() => validateLanguageOutputHead(malformed, scaffold)).toThrow(/kanıt/u)

    const overCeiling = structuredClone(malformed)
    overCeiling.transitionEvidence[0].excitatoryEvidence = head.config.evidenceCeiling + 1
    overCeiling.transitionEvidence[0].score = boundedLogOdds(
      overCeiling.transitionEvidence[0].excitatoryEvidence,
      overCeiling.transitionEvidence[0].inhibitoryEvidence,
    )
    expect(() => validateLanguageOutputHead(overCeiling, scaffold)).toThrow(/kanıt/u)
  })

  it('FamilyRun migrationında yalnız exact üyeleri/kenarları taşır ve farklı temsil uzayını reddeder', () => {
    const scaffold = createLanguageOutputScaffold()
    const familyRun = createFixtureFamilyRun()
    const memberKeys = familyRun.profiles.slice(0, 3).map((profile) => profile.memberKey)
    const initial = createLanguageOutputHead(scaffold, familyRun)
    const evidence = (fromKey: string, toKey: string) => ({
      kind: 'member-transition' as const,
      fromKey,
      toKey,
      excitatoryEvidence: 1,
      inhibitoryEvidence: 0,
      score: boundedLogOdds(1, 0),
      updateCount: 1,
      sourceDigests: ['d1'],
      sourceGroupIds: ['s1'],
    })
    const withEvidenceBase = {
      ...initial,
      transitionEvidence: [
        evidence(memberKeys[0], memberKeys[1]),
        evidence(memberKeys[1], memberKeys[2]),
        {
          ...evidence(LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, LANGUAGE_OUTPUT_FAMILY_IDS.verbTense),
          kind: 'family-transition' as const,
        },
      ],
      membershipEvidence: memberKeys.map((memberKey) => ({
        memberKey,
        familyId: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense,
        excitatoryEvidence: 1,
        inhibitoryEvidence: 0,
        score: boundedLogOdds(1, 0),
        updateCount: 1,
        sourceDigests: ['d1'],
        sourceGroupIds: ['s1'],
        classifierConfidence: 0.9,
      })),
    }
    const { fingerprint: _fingerprint, ...base } = withEvidenceBase
    const seeded = {
      ...base,
      fingerprint: '',
    } as LanguageOutputHeadV1
    // Use the public migration path only with a valid starting fingerprint.
    seeded.fingerprint = computeLanguageOutputHeadFingerprint(base)
    validateLanguageOutputHead(seeded, scaffold)

    const nextRun = {
      ...familyRun,
      fingerprint: 'family-run-next',
      profiles: familyRun.profiles.filter((profile) => profile.memberKey !== memberKeys[2]),
    }
    const first = migrateLanguageOutputHead(seeded, scaffold, nextRun)
    const second = migrateLanguageOutputHead(seeded, scaffold, nextRun)
    const exact = new Set(nextRun.profiles.map((profile) => profile.memberKey))

    expect(first).toEqual(second)
    expect(first.revision).toBe(seeded.revision + 1)
    expect(first.parentFingerprint).toBe(seeded.fingerprint)
    expect(first.familyRunFingerprint).toBe(nextRun.fingerprint)
    expect(first.membershipEvidence.every((membership) => exact.has(membership.memberKey))).toBe(true)
    expect(first.transitionEvidence.filter((edge) => edge.kind === 'member-transition')).toHaveLength(1)
    expect(first.transitionEvidence.some((edge) => edge.kind === 'family-transition')).toBe(true)

    expect(() => migrateLanguageOutputHead(seeded, scaffold, {
      ...nextRun,
      representationSpaceId: 'different-space',
    })).toThrow(/temsil uzay/u)
  })

  it('provisional çıktı nöronu kaynak/aile kanıtıyla kalır ve FamilyRun migrationında korunur', () => {
    const scaffold = createLanguageOutputScaffold()
    const familyRun = createFixtureFamilyRun()
    const initial = createLanguageOutputHead(scaffold, familyRun)
    const nurseryBase: Omit<LanguageOutputNurseryProposal, 'fingerprint'> = {
      id: 'LON-proof',
      status: 'eligible',
      normalizedSurface: 'yepyeni',
      representativeSurface: 'Yepyeni',
      observationCount: 3,
      sourceDigests: ['digest-a', 'digest-b', 'digest-c'],
      sourceGroupIds: ['book-a', 'book-b'],
      exampleIds: ['example-a', 'example-b', 'example-c'],
      meanSurprise: 3,
      meanNovelty: 0.9,
      familyCandidates: [{
        familyId: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder,
        evidence: 1.8,
        observationCount: 3,
      }],
      sourceProposalIds: ['NBP-proof'],
      requiresValidation: true,
    }
    const nursery = {
      ...nurseryBase,
      fingerprint: computeLanguageOutputNurseryFingerprint(nurseryBase),
    }
    const neuronBase: Omit<LanguageOutputProvisionalNeuronV1, 'fingerprint'> = {
      id: 'LOPN-proof',
      status: 'provisional',
      kind: 'surface-controller',
      normalizedSurface: nursery.normalizedSurface,
      representativeSurface: nursery.representativeSurface,
      familyId: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder,
      familyEvidence: 1.8,
      meanFamilyConfidence: 0.6,
      observationCount: 3,
      sourceDigests: [...nursery.sourceDigests],
      sourceGroupIds: [...nursery.sourceGroupIds],
      exampleIds: [...nursery.exampleIds],
      sourceProposalIds: [...nursery.sourceProposalIds],
      promotedAtRevision: 1,
      reversible: true,
      physicalIdentityAllocated: false,
    }
    const outputNeuron = {
      ...neuronBase,
      fingerprint: computeLanguageOutputProvisionalNeuronFingerprint(neuronBase),
    }
    const {
      fingerprint: _initialFingerprint,
      parentFingerprint: _initialParent,
      revision: _initialRevision,
      ...initialBase
    } = initial
    const populated = finalizeLanguageOutputHead(initial, scaffold, {
      ...initialBase,
      nursery: [nursery],
      outputNeurons: [outputNeuron],
    })
    const migrated = migrateLanguageOutputHead(populated, scaffold, {
      ...familyRun,
      fingerprint: 'family-run-provisional-migration',
    })

    expect(migrated.outputNeurons).toEqual(populated.outputNeurons)
    expect(migrated.outputNeurons[0]).toMatchObject({
      familyId: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder,
      sourceGroupIds: ['book-a', 'book-b'],
      reversible: true,
      physicalIdentityAllocated: false,
    })
    expect(migrated.nursery).toEqual(populated.nursery)
    validateLanguageOutputHead(migrated, scaffold)
  })
})
