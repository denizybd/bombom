import { describe, expect, it } from 'vitest'
import {
  canRequestLearningProposal,
  normalizeEpisode,
  normalizeGraph,
  normalizeReplay,
  normalizeStatus,
  normalizeTick,
} from './model'

describe('thought UI response normalization', () => {
  it('unwraps the public episode envelope and nested server fields', () => {
    const episode = normalizeEpisode({
      ok: true,
      episode: {
        id: 'episode-7',
        status: 'completed',
        input: { preview: 'Kedi pencereyi dikkatle açtı.' },
        base: { snapshotId: 'snapshot-research-2' },
        crossCheck: { status: 'passed' },
        learningMode: 'proposal',
        mode: 'active',
        commitEligible: true,
        commitStatus: 'pending-review',
        artifacts: { candidateSnapshotId: null, headProposalAvailable: true },
        seed: 77,
        maxTicks: 64,
        tickCount: 3,
        engine: { stats: { outcomeScore: .81, confidence: .73 } },
      },
    })

    expect(episode.id).toBe('episode-7')
    expect(episode.prompt).toBe('Kedi pencereyi dikkatle açtı.')
    expect(episode.snapshotId).toBe('snapshot-research-2')
    expect(episode.teacherMode).toBe('ollama-cross-check')
    expect(episode.learningMode).toBe('proposal')
    expect(episode.commitEligible).toBe(true)
    expect(episode.commitStatus).toBe('pending-review')
    expect(episode.structuralScore).toBe(.81)
    expect(episode.trust).toBe(.73)
  })

  it('always produces bounded 64x64 heatmaps for a tick', () => {
    const rows = Array.from({ length: 64 }, (_, y) => Array.from({ length: 64 }, (_, x) => (x - y) / 64))
    const tick = normalizeTick({
      index: 4,
      signalBefore: rows,
      signalAfter: { values: rows.flat().map((value) => value * .5) },
    })

    expect(tick.signalBefore.side).toBe(64)
    expect(tick.signalBefore.values).toHaveLength(4096)
    expect(tick.signalAfter.values).toHaveLength(4096)
    expect(tick.signalBefore.values[65]).toBe(0)
  })

  it('keeps each routing contribution separate and accepts runtime aliases', () => {
    const tick = normalizeTick({
      candidates: [{
        memberKey: 'n-42',
        score: .71,
        contributions: {
          baseH0: .62,
          persistentExcitation: .31,
          fastExcitation: .18,
          routeEvidence: .44,
          q: .79,
          repetitionPenalty: .12,
        },
      }],
    })
    const candidate = tick.candidates[0]

    expect(candidate).toMatchObject({
      id: 'n-42',
      h0Prior: .62,
      persistentExperience: .31,
      branchTrace: .18,
      family: .44,
      signalQ: .79,
      repetitionPenalty: .12,
      total: .71,
    })
  })

  it('reads the runtime branch trace, role hypotheses and birth event without inventing labels', () => {
    const before = Array.from({ length: 4096 }, (_, index) => index === 3 ? .8 : 0)
    const after = Array.from({ length: 4096 }, (_, index) => index === 4 ? .6 : 0)
    const tick = normalizeTick({
      tick: 5,
      dominantBranchId: 'branch-a',
      maximumBirthPressure: .72,
      branches: [{
        branchId: 'branch-a',
        mass: .64,
        selectedNeuronId: 'neuron-next',
        selectedFamilyId: 'family-action',
        beforeSignal: before,
        afterSignal: after,
        unmatchedProbability: .21,
        edgePressure: .33,
        smoothedBirthPressure: .72,
        neuronCandidates: [{
          neuronId: 'neuron-next',
          familyId: 'family-action',
          rank: 1,
          baseH0: .2,
          persistentExcitation: .3,
          persistentInhibition: .04,
          fastExcitation: .1,
          fastInhibition: .02,
          routeEvidence: .4,
          q: .7,
          repetitionPenalty: .05,
          selectionProbability: .64,
        }],
        coalitionFamilyWeights: [
          { familyId: 'family-action', weight: .77 },
          { familyId: 'family-content', weight: .41 },
        ],
        roleHypotheses: [{ role: 'actor', memberKey: 'neuron-actor', score: .88, temporary: true }],
        birthEvents: [{ kind: 'pressure-threshold', neuronId: 'newborn-1', pressure: .72 }],
      }],
    })

    expect(tick.selectedBranchId).toBe('branch-a')
    expect(tick.signalBefore.values[3]).toBeCloseTo(.8)
    expect(tick.signalAfter.values[4]).toBeCloseTo(.6)
    expect(tick.branches[0]).toMatchObject({ targetNeuronId: 'neuron-next', targetFamilyId: 'family-action', edge: .33, birthPressure: .72 })
    expect(tick.branches[0]?.candidates[0]).toMatchObject({ h0Prior: .2, persistentExperience: .26, branchTrace: .08, family: .4, signalQ: .7 })
    expect(tick.branches[0]?.coalition?.memberIds).toEqual(['family-action', 'family-content'])
    expect(tick.branches[0]?.coalition?.roleHypotheses[0]).toMatchObject({ role: 'actor', target: 'neuron-actor', score: .88 })
    expect(tick.branches[0]?.newborn).toMatchObject({ phase: 'born', id: 'newborn-1' })
  })

  it('keeps bounded adaptive-region telemetry and replay lineage read-only', () => {
    const pageWriteMask = Object.fromEntries(Array.from({ length: 60 }, (_, index) => [String(index), index === 4 ? .8 : 0]))
    const episode = normalizeEpisode({ episode: {
      id: 'episode-regions', status: 'completed', input: { preview: 'Ada kutuyu taşıdı.' },
      regionMode: 'shadow', requestedRegionMode: 'active',
      regionRuntime: {
        behaviorApplied: false, observationsEnabled: true, reason: 'configured-opt-in',
        atlasArtifactId: 'atlas-artifact', atlasFingerprint: 'atlas-fingerprint', atlasRevision: 7,
        bindingFingerprint: 'binding-fingerprint', releaseFingerprint: 'release-fingerprint',
        compatibilityBlend: .25, minimumTrace: .02,
      },
      ticks: [{
        index: 1,
        adaptiveRegion: {
          mode: 'shadow', atlasFingerprint: 'atlas-fingerprint', bindingFingerprint: 'binding-fingerprint',
          observedBranchCount: 1, activeApplied: false,
        },
        branches: [{
          id: 'branch-region', mass: 1,
          adaptiveRegion: {
            mode: 'shadow', atlasFingerprint: 'atlas-fingerprint', bindingFingerprint: 'binding-fingerprint',
            activeApplied: false, legacyCompatibility: .6, regionCompatibility: .8,
            blendedCompatibility: .65, effectiveCompatibility: .6,
            pageCompatibilities: Array.from({ length: 60 }, () => .5),
            pageWeights: Array.from({ length: 56 }, () => 1 / 56), pageWriteMask,
            gates: {
              gates: Array.from({ length: 10 }, (_, index) => ({
                regionId: `region-${index}`, compatibility: .7, prototypeCompatibility: .8,
                contextScore: .2, confidence: .9, probability: index === 0 ? .7 : .02,
              })),
              unmatchedProbability: .14, omittedRegionProbability: .04,
            },
          },
        }],
      }],
    } })

    expect(episode).toMatchObject({
      regionMode: 'shadow', requestedRegionMode: 'active',
      regionRuntime: { behaviorApplied: false, atlasRevision: 7, bindingFingerprint: 'binding-fingerprint' },
    })
    expect(episode.ticks[0]?.adaptiveRegion).toMatchObject({ mode: 'shadow', observedBranchCount: 1, activeApplied: false })
    expect(episode.ticks[0]?.branches[0]?.adaptiveRegion?.gates.gates).toHaveLength(8)
    expect(episode.ticks[0]?.branches[0]?.adaptiveRegion?.gates.gates[0]).toMatchObject({ prototypeCompatibility: .8 })
    expect(episode.ticks[0]?.branches[0]?.adaptiveRegion?.pageWriteMask).toHaveLength(56)
    expect(episode.ticks[0]?.branches[0]?.adaptiveRegion?.pageWriteMask[4]).toBeCloseTo(.8)
    expect(episode.ticks[0]?.branches[0]?.adaptiveRegion?.pageCompatibilities).toHaveLength(56)
  })

  it('normalizes sparse graph counts and strongest-edge weight', () => {
    const graph = normalizeGraph({
      ok: true,
      graph: {
        level: 'family',
        nodeCount: 14,
        edgeCount: 28,
        truncated: true,
        nodes: [{ familyId: 'f-a', name: 'Eylem ailesi' }, { familyId: 'f-b', name: 'Araç ailesi' }],
        edges: [{ fromFamilyId: 'f-a', toFamilyId: 'f-b', probability: .84 }],
        audit: {
          episodeId: 'episode-graph',
          eventGraph: {
            format: 'event-graph-v1', actor: 'Ada', action: 'taşıdı', patient: 'kutu', instrument: 'kanca',
            operators: [], negated: false, goals: [],
          },
          evaluation: { graphF1: .8, constraintAccuracy: .9, resultScore: .83 },
          languageOutput: {
            text: 'Ada kutuyu kancayla taşıdı.', stoppedBy: 'eos',
            scaffoldFingerprint: 'scaffold-a', headFingerprint: 'head-a',
            steps: [{ familyId: '@language-output/family/word-order', emittedToken: { surface: 'Ada' }, state: { complete: false }, explanation: 'Roller sıralandı.' }],
          },
        },
      },
    }, 'family')

    expect(graph.totalNodeCount).toBe(14)
    expect(graph.totalEdgeCount).toBe(28)
    expect(graph.edges[0]).toMatchObject({ from: 'f-a', to: 'f-b', weight: .84 })
    expect(graph.episodeId).toBe('episode-graph')
    expect(graph.structuralOutcome?.action).toBe('taşıdı')
    expect(graph.structuralEvaluation).toEqual({ graphF1: .8, constraintAccuracy: .9, resultScore: .83 })
    expect(graph.languageOutput).toMatchObject({ text: 'Ada kutuyu kancayla taşıdı.', stoppedBy: 'eos', headFingerprint: 'head-a' })
    expect(graph.languageOutput?.steps[0]).toMatchObject({ emittedSurface: 'Ada', complete: false })
    expect(graph.truncated).toBe(true)
  })

  it('reads the real replay mismatch and nested service status counts', () => {
    const replay = normalizeReplay({
      ok: true,
      replay: {
        matched: false,
        firstMismatch: { position: 3, expectedIndex: 3, actualIndex: 4, expectedDigest: 'a', actualDigest: 'b' },
      },
    })
    const status = normalizeStatus({
      available: true,
      enabled: true,
      researchBaseAvailable: false,
      activeSnapshotId: 'snapshot-active',
      counts: { running: 2 },
      limits: { maximumTicks: 384 },
    })

    expect(replay.matches).toBe(false)
    expect(replay.firstMismatchTick).toBe(3)
    expect(replay.expectedHash).toBe('a')
    expect(status.activeEpisodes).toBe(2)
    expect(status.researchBaseAvailable).toBe(false)
    expect(status.activeSnapshotId).toBe('snapshot-active')
    expect(status.maximumTicks).toBe(384)
    expect(canRequestLearningProposal(status, 'snapshot-active')).toBe(true)
    expect(canRequestLearningProposal(status, 'older-snapshot')).toBe(false)
  })
})
