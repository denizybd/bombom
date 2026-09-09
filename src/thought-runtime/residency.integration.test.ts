// @ts-nocheck -- Vitest runs this Node-only bridge test outside the browser type surface.
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { createInitialPlasticRegionAtlas } from '../adaptive-regions'
import { ADAPTIVE_TEXT_LAB_PROFILE_ID } from '../text-lab/domain/config'
import {
  createThoughtNeuronResidencySnapshot,
  openThoughtNeuronResidency,
} from '../../runtime/thought-neuron-residency.mjs'
import { createThoughtCatalogFromResolved } from './catalog'
import { createThoughtHead } from './model'
import { createThoughtRuntimeEngine } from './orchestration'
import { encodeThoughtPrompt } from './prompt'
import { minimalFamilyRun } from './test-fixtures'
import type {
  ThoughtEngineContext,
  ThoughtEngineRequest,
  ThoughtEpisodeResult,
  ThoughtEpisodeState,
  ThoughtResolvedNeuronInput,
  ThoughtTick,
} from './types'

const directories: string[] = []

afterEach(async () => {
  await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })))
})

describe('SSD-gated thought residency', () => {
  it('restores only the bounded candidate set and preserves deterministic engine output', async () => {
    const text = 'Ada kutuyu taşıdı.'
    const prompt = await encodeThoughtPrompt(text, { identityRatio: 0.25 })
    const identity = prompt.signal.slice(0, prompt.identityLength)
    const relation = prompt.signal.slice(prompt.identityLength)
    const neurons: ThoughtResolvedNeuronInput[] = [
      {
        memberKey: 'near', kind: 'raw', text,
        identity, relation, identityDigest: 'source-near',
        sourceDigest: 'source-near', analysisRunId: 'run-near', neuronId: 'node-near',
      },
      {
        memberKey: 'far', kind: 'raw', text: 'ters',
        identity: Float32Array.from(identity, (value) => -value),
        relation: Float32Array.from(relation, (value) => -value),
        identityDigest: 'source-far',
        sourceDigest: 'source-far', analysisRunId: 'run-far', neuronId: 'node-far',
      },
    ]
    const familyRun = minimalFamilyRun({
      ids: ['near', 'far'],
      representationSpaceId: prompt.representationSpaceId,
      transitions: [{ from: 'far', to: 'near', probability: 1 }],
    })
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const catalog = createThoughtCatalogFromResolved(neurons, familyRun, engine.config)
    const baseModelFingerprint = 'residency-base'
    const head = createThoughtHead(catalog.model, null, baseModelFingerprint)
    const prepared = engine.prepareResidencyBase({ neurons, familyRun, thoughtHead: head, baseModelFingerprint })
    const directory = await mkdtemp(path.join(tmpdir(), 'bombom-residency-'))
    directories.push(directory)
    const policy = {
      hotBudgetBytes: 128 * 1024,
      warmBudgetBytes: 64 * 1024,
      pageTargetBytes: 64 * 1024,
      maximumPageBytes: 128 * 1024,
      maximumWorkingSetNeurons: 2,
    }
    const created = await createThoughtNeuronResidencySnapshot({
      dataDirectory: directory,
      model: prepared.model,
      head: prepared.head,
      representationSpaceId: familyRun.representationSpaceId,
      policy,
      residencyModelFingerprint: `${baseModelFingerprint}:${head.fingerprint}`,
    })
    expect(created.manifest.physicalRepresentationSpaceId).toBe(familyRun.representationSpaceId)
    expect(created.manifest.thoughtModelFingerprint).toBe(prepared.model.fingerprint)
    const residency = await openThoughtNeuronResidency({
      dataDirectory: directory,
      manifestPath: created.manifestPath,
      manifestFingerprint: created.manifest.fingerprint,
      expectedModelFingerprint: `${baseModelFingerprint}:${head.fingerprint}`,
      expectedRepresentationSpaceId: familyRun.representationSpaceId,
      expectedThoughtModelFingerprint: prepared.model.fingerprint,
      policy,
    })
    const signal = await engine.encodeResidencyPrompt(text, prepared.model.layout, familyRun.representationSpaceId)
    const ranked = residency.rankBySignal(signal, 2)
    expect(ranked[0]).toMatchObject({ id: 'near', distance: 0 })
    expect(residency.repository.getStats().resident.totalBytes).toBe(0)

    const base = {
      snapshotId: 'snapshot-residency', snapshotStatus: 'candidate', selectedBy: 'explicit',
      activeAtStart: false, commitEligible: false,
      artifactManifestHash: 'manifest-residency', baseModelFingerprint,
      familyRunFingerprint: familyRun.fingerprint, languageStateFingerprint: null,
      thoughtHeadFingerprint: head.fingerprint, thoughtHeadRevision: head.revision,
    }
    const request: ThoughtEngineRequest = {
      format: 'thought-engine-request', version: 1, episodeId: 'residency-episode',
      inputText: text, seed: 17, maxTicks: 3, strategy: 'deterministic', learningMode: 'off',
      candidateMemberKeys: ['near'], base,
    }
    const fullContext: ThoughtEngineContext = {
      neurons, familyRun, languageState: null, thoughtHead: head,
    }
    const full = await engine.runEpisode(request, fullContext)
    let resume: ThoughtEpisodeState | null = null
    let paged: ThoughtEpisodeResult | null = null
    const pagedTicks: ThoughtTick[] = []
    for (let targetTick = 1; targetTick <= request.maxTicks; targetTick += 1) {
      paged = await residency.withCandidateLookup(
        ranked.map((entry) => ({
          id: entry.id,
          evidence: {
            distance: entry.distance,
            routeProbability: entry.id === 'near' ? 1 : 0,
            familyProbability: 1,
            observedUseProbability: 1 - entry.distance,
          },
        })),
        (workingSet: Map<string, unknown>) => engine.runEpisode({
          ...request,
          resume,
          replay: { verifyOnly: true, expectedTickCount: targetTick },
        }, {
          ...fullContext,
          neurons: [...workingSet.values()] as ThoughtEngineContext['neurons'],
          modelFingerprintOverride: prepared.model.fingerprint,
        }),
        { maximum: 2 },
      )
      pagedTicks.push(...paged.ticks)
      resume = paged.episode
    }

    expect(paged!.episode.fingerprint).toBe(full.episode.fingerprint)
    expect(pagedTicks).toEqual(full.ticks)
    expect(residency.repository.getStats().tiers.coldNeurons).toBeGreaterThanOrEqual(0)
    residency.repository.close()
  })

  it('8/56 snapshotta trusted non-off atlas bağını SSD ön-kodlayıcısına taşır; legacy yol sessiz profile geçmez', async () => {
    const text = 'Mina taşı dikkatlice kutunun yanına bıraktı.'
    const prompt = await encodeThoughtPrompt(text, {
      identityRatio: 0.125,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      allowFlatFallback: false,
    })
    const identity = prompt.signal.slice(0, prompt.identityLength)
    const relation = prompt.signal.slice(prompt.identityLength)
    const neurons: ThoughtResolvedNeuronInput[] = [
      {
        memberKey: 'adaptive-near', kind: 'raw', text,
        identity, relation, identityDigest: 'adaptive-source-near',
        sourceDigest: 'adaptive-source-near', analysisRunId: 'adaptive-run-near', neuronId: 'adaptive-node-near',
      },
      {
        memberKey: 'adaptive-far', kind: 'raw', text: 'ters',
        identity: Float32Array.from(identity, (value) => -value),
        relation: Float32Array.from(relation, (value) => -value),
        identityDigest: 'adaptive-source-far',
        sourceDigest: 'adaptive-source-far', analysisRunId: 'adaptive-run-far', neuronId: 'adaptive-node-far',
      },
    ]
    const familyRun = minimalFamilyRun({
      ids: ['adaptive-near', 'adaptive-far'],
      representationSpaceId: prompt.representationSpaceId,
      transitions: [{ from: 'adaptive-far', to: 'adaptive-near', probability: 1 }],
    })
    const engine = createThoughtRuntimeEngine({ maximumBranches: 1 })
    const catalog = createThoughtCatalogFromResolved(neurons, familyRun, engine.config)
    const baseModelFingerprint = 'adaptive-residency-base'
    const head = createThoughtHead(catalog.model, null, baseModelFingerprint)
    const prepared = engine.prepareResidencyBase({ neurons, familyRun, thoughtHead: head, baseModelFingerprint })
    const atlas = createInitialPlasticRegionAtlas({
      representationSpaceId: familyRun.representationSpaceId,
      behaviorDimension: 8,
    })
    const regionRequest = { mode: 'shadow' as const, atlasFingerprint: atlas.fingerprint }
    const directory = await mkdtemp(path.join(tmpdir(), 'bombom-adaptive-residency-'))
    directories.push(directory)
    const policy = {
      hotBudgetBytes: 128 * 1024,
      warmBudgetBytes: 64 * 1024,
      pageTargetBytes: 64 * 1024,
      maximumPageBytes: 128 * 1024,
      maximumWorkingSetNeurons: 2,
    }
    const created = await createThoughtNeuronResidencySnapshot({
      dataDirectory: directory,
      model: prepared.model,
      head: prepared.head,
      representationSpaceId: familyRun.representationSpaceId,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      policy,
      residencyModelFingerprint: `${baseModelFingerprint}:${head.fingerprint}`,
    })
    expect(created.manifest.representationProfile).toBe(ADAPTIVE_TEXT_LAB_PROFILE_ID)
    expect(created.manifest.relationPageSketches?.pageCount).toBe(56)
    await expect(openThoughtNeuronResidency({
      dataDirectory: directory,
      manifestPath: created.manifestPath,
      manifestFingerprint: created.manifest.fingerprint,
      expectedModelFingerprint: `${baseModelFingerprint}:${head.fingerprint}`,
      expectedRepresentationSpaceId: familyRun.representationSpaceId,
      expectedRepresentationProfile: null,
      expectedThoughtModelFingerprint: prepared.model.fingerprint,
      policy,
    })).rejects.toMatchObject({ code: 'NEURON_TIER_PROFILE_MISMATCH' })
    const residency = await openThoughtNeuronResidency({
      dataDirectory: directory,
      manifestPath: created.manifestPath,
      manifestFingerprint: created.manifest.fingerprint,
      expectedModelFingerprint: `${baseModelFingerprint}:${head.fingerprint}`,
      expectedRepresentationSpaceId: familyRun.representationSpaceId,
      expectedRepresentationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      expectedThoughtModelFingerprint: prepared.model.fingerprint,
      policy,
    })

    await expect(engine.encodeResidencyPrompt(
      text,
      prepared.model.layout,
      familyRun.representationSpaceId,
    )).rejects.toThrow(/aynı temsil uzayında/)
    await expect(engine.encodeResidencyPrompt(
      text,
      prepared.model.layout,
      familyRun.representationSpaceId,
      { adaptiveRegion: { mode: 'off' }, adaptiveRegionAtlas: atlas },
    )).rejects.toThrow(/aynı temsil uzayında/)
    const offSignal = await engine.encodeResidencyPrompt(
      text,
      prepared.model.layout,
      familyRun.representationSpaceId,
      { representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID, adaptiveRegion: { mode: 'off' } },
    )
    expect(offSignal).toEqual(prompt.signal)
    const signal = await engine.encodeResidencyPrompt(
      text,
      prepared.model.layout,
      familyRun.representationSpaceId,
      {
        representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
        adaptiveRegion: regionRequest,
        adaptiveRegionAtlas: atlas,
        sourceGroupId: 'adaptive-residency-source',
      },
    )
    expect(signal).toEqual(prompt.signal)
    const ranked = residency.rankBySignal(signal, 2)
    expect(ranked[0]).toMatchObject({ id: 'adaptive-near', distance: 0 })

    const request: ThoughtEngineRequest = {
      format: 'thought-engine-request', version: 1,
      episodeId: 'adaptive-residency-episode', inputText: text,
      seed: 41, maxTicks: 1, strategy: 'deterministic', learningMode: 'off',
      sourceGroupId: 'adaptive-residency-source', candidateMemberKeys: ['adaptive-near'],
      adaptiveRegion: regionRequest,
      base: {
        snapshotId: 'snapshot-adaptive-residency', snapshotStatus: 'candidate', selectedBy: 'explicit',
        activeAtStart: false, commitEligible: false,
        artifactManifestHash: 'manifest-adaptive-residency', baseModelFingerprint,
        familyRunFingerprint: familyRun.fingerprint, languageStateFingerprint: null,
        thoughtHeadFingerprint: head.fingerprint, thoughtHeadRevision: head.revision,
        representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
      },
    }
    const fullContext: ThoughtEngineContext = {
      neurons, familyRun, languageState: null, thoughtHead: head,
      adaptiveRegionAtlas: atlas,
      representationProfile: ADAPTIVE_TEXT_LAB_PROFILE_ID,
    }
    const full = await engine.runEpisode(request, fullContext)
    const paged = await residency.withCandidateLookup(
      ranked.map((entry) => ({
        id: entry.id,
        evidence: {
          distance: entry.distance,
          routeProbability: entry.id === 'adaptive-near' ? 1 : 0,
          familyProbability: 1,
          observedUseProbability: 1 - entry.distance,
        },
      })),
      (workingSet: Map<string, unknown>) => engine.runEpisode(request, {
        ...fullContext,
        neurons: [...workingSet.values()] as ThoughtEngineContext['neurons'],
        modelFingerprintOverride: prepared.model.fingerprint,
      }),
      { maximum: 2 },
    )
    expect(paged.episode.fingerprint).toBe(full.episode.fingerprint)
    expect(paged.ticks).toEqual(full.ticks)
    expect(paged.ticks[0].adaptiveRegion?.mode).toBe('shadow')
    residency.repository.close()
  })
})
