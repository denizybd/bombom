import { beforeAll, describe, expect, it } from 'vitest'
import { createMatrixLayout } from '../text-lab/domain/config'
import type { RawTextNeuron, TextLabAnalysis } from '../text-lab/domain/types'
import {
  createContextSpaceId,
  createIdentitySpaceId,
  createRepresentationSpaceId,
  digestText,
  digestVector,
} from '../text-lab/engine/representation'
import { runSynthesis } from '../synthesis'
import type { SynthesisConfig } from '../synthesis'
import {
  type SynthesisArchiveRecord,
  validateSynthesisArchiveRecord,
} from './persistence'

const TEST_CONFIG: Partial<SynthesisConfig> = {
  nodeSelection: 'all',
  minimumIdentitySimilarity: 0.75,
  minimumRelationSimilarity: -1,
  minimumCombinedSimilarity: 0.7,
  identityOverrideSimilarity: 0.98,
  identityWeight: 0.9,
  minimumSupport: 3,
  minimumSources: 2,
  maximumClusterMembers: 32,
  maximumEdgesPerMember: 8,
  maximumRetainedEdges: 128,
  maximumCandidates: 32,
  yieldEveryComparisons: 100,
}

function neuron(id: string, sourceId: string, identity: number[]): RawTextNeuron {
  const identityVector = Float32Array.from([
    ...identity,
    ...Array(Math.max(0, 32 - identity.length)).fill(0),
  ])
  const relation = Float32Array.from([1, ...Array(31).fill(0)])
  return {
    id,
    sourceId,
    paragraphIndex: 0,
    start: 0,
    end: 4,
    text: sourceId.repeat(4),
    graphemeCount: 4,
    depth: 0,
    branch: 'root',
    parentId: null,
    childIds: [],
    splitBoundary: 'leaf',
    identity: identityVector,
    relation,
    identityDigest: digestVector(identityVector),
    surfaceKey: `surface-${id}`,
    identityNorm: Math.hypot(...identityVector),
    relationNorm: 1,
    contextSupport: 1,
    contextRevision: 0,
    trace: {
      inheritedNorm: 0,
      parentIdentityCompatibility: 0,
      parentGain: 0,
      siblingId: null,
      siblingIdentityCompatibility: null,
      siblingGain: 0,
    },
  }
}

function analysis(sourceId: string, runId: string, node: RawTextNeuron): TextLabAnalysis {
  const rawText = sourceId.repeat(4)
  const config = {
    matrixSide: 8,
    identityRatio: 0.5,
    minimumLeafGraphemes: 4,
    hashSeed: 1,
    contextRetention: 0.8,
    parentTraceWeight: 0.15,
    siblingTraceWeight: 0.05,
    minimumTrace: 0.1,
    identitySimilarityWeight: 0.5,
    habitExponent: 2,
  }
  const layout = createMatrixLayout(config)
  const identitySpaceId = createIdentitySpaceId(config, layout)
  const contextSpaceId = createContextSpaceId(config, layout)
  return {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId,
    createdAt: '2026-08-28T00:00:00.000Z',
    sourceDigest: digestText(rawText),
    identitySpaceId,
    contextSpaceId,
    representationSpaceId: createRepresentationSpaceId(identitySpaceId, contextSpaceId),
    source: { id: sourceId, title: sourceId, rawText, text: rawText },
    config,
    layout,
    rootIds: [node.id],
    nodes: [node],
    similarities: Float32Array.of(1),
    habitProbabilities: Float32Array.of(0),
    stats: {
      characterCount: 4,
      graphemeCount: 4,
      paragraphCount: 1,
      neuronCount: 1,
      uniquePairCount: 0,
      matrixBytes: node.identity.byteLength + node.relation.byteLength,
      similarityBytes: 8,
      durationMs: 0,
      medianRelationNorm: 1,
    },
  }
}

function cloneRecord(record: SynthesisArchiveRecord): SynthesisArchiveRecord {
  return structuredClone(record)
}

describe('sentez arşivi güven sınırı', () => {
  let validRecord: SynthesisArchiveRecord

  beforeAll(async () => {
    const run = await runSynthesis(
      [
        analysis('a', 'run-a', neuron('a1', 'a', [1, 0, 0])),
        analysis('b', 'run-b', neuron('b1', 'b', [0.99, 0.01, 0])),
        analysis('c', 'run-c', neuron('c1', 'c', [0.98, 0.02, 0])),
      ],
      TEST_CONFIG,
      { yieldControl: async () => Promise.resolve() },
    )
    const candidate = run.candidates[0]
    validRecord = {
      id: `decision:${candidate.fingerprint}`,
      format: 'noron-sentez-karari',
      version: 1,
      candidateId: candidate.id,
      candidateFingerprint: candidate.fingerprint,
      decision: 'approved',
      reviewedAt: '2026-08-28T01:02:03.000Z',
      runId: run.id,
      runFingerprint: run.fingerprint,
      representationSpaceId: run.representationSpaceId,
      analysisRunIds: [...run.analysisRunIds],
      sourceDigests: [...run.sourceDigests],
      config: { ...run.config },
      candidate,
    }
  })

  it('motorun ürettiği eksiksiz kaydı kabul eder', () => {
    expect(validateSynthesisArchiveRecord(validRecord)).toBe(validRecord)
    expect(validateSynthesisArchiveRecord(cloneRecord(validRecord))).toEqual(validRecord)
  })

  it('üst kayıt formatını, kararı ve fingerprint kimlik bağlarını doğrular', () => {
    const wrongVersion = cloneRecord(validRecord)
    Reflect.set(wrongVersion, 'version', 2)
    expect(() => validateSynthesisArchiveRecord(wrongVersion)).toThrow('format veya sürüm')

    const wrongDecision = cloneRecord(validRecord)
    Reflect.set(wrongDecision, 'decision', 'pending')
    expect(() => validateSynthesisArchiveRecord(wrongDecision)).toThrow('karar')

    const wrongRecordId = cloneRecord(validRecord)
    wrongRecordId.id = 'decision:başka'
    expect(() => validateSynthesisArchiveRecord(wrongRecordId)).toThrow('kayıt id')

    const wrongRunId = cloneRecord(validRecord)
    wrongRunId.runId = 'SR-0000000000000000'
    expect(() => validateSynthesisArchiveRecord(wrongRunId)).toThrow('run id')
  })

  it('kimlik digestini, bütün finite hücreleri ve matris uzunluklarını doğrular', () => {
    const changedIdentity = cloneRecord(validRecord)
    changedIdentity.candidate.identity[0] -= 0.125
    expect(() => validateSynthesisArchiveRecord(changedIdentity)).toThrow('identity digest')

    const changedRelation = cloneRecord(validRecord)
    changedRelation.candidate.relation[0] = Number.NaN
    expect(() => validateSynthesisArchiveRecord(changedRelation)).toThrow('sonlu olmayan')

    const wrongLengths = cloneRecord(validRecord)
    wrongLengths.candidate.identity = new Float32Array(31)
    wrongLengths.candidate.identityDigest = digestVector(wrongLengths.candidate.identity)
    expect(() => validateSynthesisArchiveRecord(wrongLengths)).toThrow('matris düzeni')
  })

  it('candidate fingerprintini vektörler ve referanslardan yeniden hesaplar', () => {
    const corrupted = cloneRecord(validRecord)
    corrupted.candidate.relation[0] = 0.75
    expect(() => validateSynthesisArchiveRecord(corrupted)).toThrow('fingerprint içerikle uyuşmuyor')
  })

  it('tekrar eden ve üst kayıt dışında kalan üye referanslarını reddeder', () => {
    const duplicateMember = cloneRecord(validRecord)
    duplicateMember.candidate.members.push(structuredClone(duplicateMember.candidate.members[0]))
    expect(() => validateSynthesisArchiveRecord(duplicateMember)).toThrow('tekrar eden referans')

    const unknownSource = cloneRecord(validRecord)
    const member = unknownSource.candidate.members[0]
    member.sourceDigest = 'bilinmeyen-kaynak'
    member.key = `${member.sourceDigest}\u0000${member.analysisRunId}\u0000${member.neuronId}`
    expect(() => validateSynthesisArchiveRecord(unknownSource)).toThrow('bilinmeyen kaynağa')
  })

  it('destek, medoid, sayaç ve doğum değişmezlerini doğrular', () => {
    const missingMedoidSupport = cloneRecord(validRecord)
    missingMedoidSupport.candidate.independentSupports =
      missingMedoidSupport.candidate.independentSupports.filter(
        (support) => support.key !== missingMedoidSupport.candidate.medoid.key,
      )
    expect(() => validateSynthesisArchiveRecord(missingMedoidSupport)).toThrow('medoid')

    const wrongCount = cloneRecord(validRecord)
    wrongCount.candidate.supportCount += 1
    expect(() => validateSynthesisArchiveRecord(wrongCount)).toThrow('sayaçları')

    const revisedContext = cloneRecord(validRecord)
    Reflect.set(revisedContext.candidate, 'contextRevision', 1)
    expect(() => validateSynthesisArchiveRecord(revisedContext)).toThrow('context revision')

    const unfrozenIdentity = cloneRecord(validRecord)
    Reflect.set(unfrozenIdentity.candidate, 'identityFrozen', false)
    expect(() => validateSynthesisArchiveRecord(unfrozenIdentity)).toThrow('kilitli değil')
  })

  it('eksik veya geçersiz sentez ayarlarını varsayılanla sessizce doldurmaz', () => {
    const missingConfig = cloneRecord(validRecord)
    Reflect.deleteProperty(missingConfig.config, 'minimumSupport')
    expect(() => validateSynthesisArchiveRecord(missingConfig)).toThrow('minimumSupport')

    const invalidConfig = cloneRecord(validRecord)
    invalidConfig.config.minimumSources = 1
    expect(() => validateSynthesisArchiveRecord(invalidConfig)).toThrow('en az iki')
  })
})
