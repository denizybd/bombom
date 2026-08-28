import { describe, expect, it } from 'vitest'
import type {
  RawTextNeuron,
  TextLabAnalysis,
} from '../text-lab/domain/types'
import { createMatrixLayout } from '../text-lab/domain/config'
import {
  createContextSpaceId,
  createIdentitySpaceId,
  createRepresentationSpaceId,
  digestText,
  digestVector,
} from '../text-lab/engine/representation'
import {
  runSynthesis,
  SynthesisCancelledError,
  SynthesisValidationError,
} from './engine'
import type { SynthesisConfig } from './types'

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
  yieldEveryComparisons: 8,
}

function neuron(
  id: string,
  sourceId: string,
  identity: number[],
  relation: number[] = [1, 0],
  span: { start?: number; end?: number; parentId?: string | null; childIds?: string[] } = {},
): RawTextNeuron {
  const start = span.start ?? 0
  const end = span.end ?? start + 4
  const paddedIdentity = Float32Array.from([
    ...identity,
    ...Array(Math.max(0, 32 - identity.length)).fill(0),
  ])
  const paddedRelation = Float32Array.from([
    ...relation,
    ...Array(Math.max(0, 32 - relation.length)).fill(0),
  ])
  return {
    id,
    sourceId,
    paragraphIndex: 0,
    start,
    end,
    text: sourceId.repeat(end - start),
    graphemeCount: end - start,
    depth: span.parentId ? 1 : 0,
    branch: span.parentId ? 'left' : 'root',
    parentId: span.parentId ?? null,
    childIds: span.childIds ?? [],
    splitBoundary: span.childIds?.length ? 'grapheme' : 'leaf',
    identity: paddedIdentity,
    relation: paddedRelation,
    identityDigest: digestVector(paddedIdentity),
    surfaceKey: `surface-${id}`,
    identityNorm: Math.hypot(...paddedIdentity),
    relationNorm: Math.hypot(...paddedRelation),
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

function analysis(
  sourceId: string,
  _sourceDigest: string,
  runId: string,
  nodes: RawTextNeuron[],
  hashSeed = 1,
): TextLabAnalysis {
  const rawTextLength = Math.max(1, ...nodes.map((node) => node.end))
  const rawText = sourceId.repeat(rawTextLength)
  const config = {
    matrixSide: 8,
    identityRatio: 0.5,
    minimumLeafGraphemes: 4,
    hashSeed,
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
    createdAt: '2026-08-26T00:00:00.000Z',
    sourceDigest: digestText(rawText),
    identitySpaceId,
    contextSpaceId,
    representationSpaceId: createRepresentationSpaceId(
      identitySpaceId,
      contextSpaceId,
    ),
    source: { id: sourceId, title: sourceId, rawText, text: rawText },
    config,
    layout,
    rootIds: nodes.filter((node) => !node.parentId).map((node) => node.id),
    nodes,
    similarities: new Float32Array(nodes.length * nodes.length),
    habitProbabilities: new Float32Array(nodes.length * nodes.length),
    stats: {
      characterCount: rawText.length,
      graphemeCount: rawText.length,
      paragraphCount: 1,
      neuronCount: nodes.length,
      uniquePairCount: (nodes.length * (nodes.length - 1)) / 2,
      matrixBytes: 0,
      similarityBytes: 0,
      durationMs: 0,
      medianRelationNorm: 1,
    },
  }
}

function bytes(value: Float32Array): number[] {
  return Array.from(new Uint8Array(value.buffer, value.byteOffset, value.byteLength))
}

describe('nöron sentez motoru', () => {
  it('farklı temsil uzayı fingerprintlerini kesin olarak reddeder', async () => {
    const first = analysis('a', 'digest-a', 'run-a', [neuron('a1', 'a', [1, 0, 0])], 1)
    const second = analysis('b', 'digest-b', 'run-b', [neuron('b1', 'b', [1, 0, 0])], 2)

    await expect(runSynthesis([first, second], TEST_CONFIG)).rejects.toMatchObject({
      name: 'SynthesisValidationError',
      code: 'REPRESENTATION_SPACE_MISMATCH',
    } satisfies Partial<SynthesisValidationError>)
  })

  it('kimlik vektörü digestinden sonra değiştirilmiş analizi reddeder', async () => {
    const corrupted = analysis(
      'a',
      'digest-a',
      'run-a',
      [neuron('a1', 'a', [1, 0, 0])],
    )
    corrupted.nodes[0].identity[0] = 0.25

    await expect(runSynthesis([corrupted], TEST_CONFIG)).rejects.toMatchObject({
      name: 'SynthesisValidationError',
      code: 'INVALID_VECTOR',
    } satisfies Partial<SynthesisValidationError>)
  })

  it('benzer yüzeyleri farklı kaynaklardan tek adayda toplar', async () => {
    const inputs = [
      analysis('a', 'digest-a', 'run-a', [neuron('a1', 'a', [1, 0, 0], [1, 0])]),
      analysis('b', 'digest-b', 'run-b', [neuron('b1', 'b', [0.98, 0.02, 0], [0.9, 0.1])]),
      analysis('c', 'digest-c', 'run-c', [neuron('c1', 'c', [0.96, 0.04, 0], [0.8, 0.2])]),
    ]

    const run = await runSynthesis(inputs, TEST_CONFIG)

    expect(run.stats.comparedPairCount).toBe(3)
    expect(run.edges).toHaveLength(3)
    expect(run.candidates).toHaveLength(1)
    expect(run.candidates[0]).toMatchObject({
      supportCount: 3,
      sourceCount: 3,
      identityFrozen: true,
    })
  })

  it('tek kaynaktaki tekrarlar yeni nöron doğurmaz', async () => {
    const singleSource = analysis('a', 'digest-a', 'run-a', [
      neuron('a1', 'a', [1, 0, 0], [1, 0], { start: 0, end: 4 }),
      neuron('a2', 'a', [0.99, 0.01, 0], [1, 0], { start: 4, end: 8 }),
      neuron('a3', 'a', [0.98, 0.02, 0], [1, 0], { start: 8, end: 12 }),
    ])

    const run = await runSynthesis([singleSource], TEST_CONFIG)

    expect(run.stats.comparedPairCount).toBe(0)
    expect(run.edges).toEqual([])
    expect(run.candidates).toEqual([])
  })

  it('eşiği geçen bağları akıtırken bellekteki komşuluğu sınırlar', async () => {
    const makeSource = (suffix: string) => analysis(
      suffix,
      `digest-${suffix}`,
      `run-${suffix}`,
      [0, 1, 2].map((index) => neuron(
        `${suffix}${index}`,
        suffix,
        [1, index * 0.005, 0],
        [1, 0],
        { start: index * 4, end: index * 4 + 4 },
      )),
    )
    let streamed = 0
    const run = await runSynthesis(
      [makeSource('a'), makeSource('b'), makeSource('c')],
      {
        ...TEST_CONFIG,
        maximumEdgesPerMember: 1,
        maximumRetainedEdges: 5,
      },
      { onEdge: () => { streamed += 1 } },
    )

    expect(streamed).toBe(27)
    expect(run.stats.thresholdEdgeCount).toBe(27)
    expect(run.edges.length).toBeLessThanOrEqual(5)
    expect(run.stats.peakEdgesPerMember).toBe(1)
  })

  it('medoid kimliğini ortalama almadan byte-byte korur', async () => {
    const expectedMedoid = Float32Array.from([0.95, 0.05, 0, ...Array(29).fill(0)])
    const inputs = [
      analysis('a', 'digest-a', 'run-a', [neuron('a1', 'a', [1, 0, 0])]),
      analysis('b', 'digest-b', 'run-b', [neuron('b1', 'b', [...expectedMedoid])]),
      analysis('c', 'digest-c', 'run-c', [neuron('c1', 'c', [0.7, 0.3, 0])]),
    ]

    const run = await runSynthesis(inputs, TEST_CONFIG)
    const candidate = run.candidates[0]

    expect(candidate.medoid.neuronId).toBe('b1')
    expect(bytes(candidate.identity)).toEqual(bytes(expectedMedoid))
    expect(candidate.identity).not.toBe(inputs[1].nodes[0].identity)
    expect(candidate.identityDigest).toBe(digestVector(candidate.identity))
    expect(candidate.contextRevision).toBe(0)
    expect(candidate.contextSupport).toBe(candidate.independentSupports.length)
    expect(candidate.contextSupport).toBe(candidate.supportCount)
  })

  it('aynı kaynaktaki örtüşen soyları bağımsız destek saymaz', async () => {
    const parent = neuron('a-parent', 'a', [1, 0, 0], [1, 0], {
      start: 0,
      end: 8,
      childIds: ['a-child'],
    })
    const child = neuron('a-child', 'a', [1, 0, 0], [1, 0], {
      start: 0,
      end: 4,
      parentId: 'a-parent',
    })
    const other = neuron('b1', 'b', [1, 0, 0])

    const run = await runSynthesis(
      [
        analysis('a', 'digest-a', 'run-a', [parent, child]),
        analysis('b', 'digest-b', 'run-b', [other]),
      ],
      TEST_CONFIG,
    )

    expect(run.edges).toHaveLength(2)
    expect(run.candidates).toEqual([])
  })

  it('tek-bağ zinciriyle birbirine benzemeyen uçları aynı adayda birleştirmez', async () => {
    const thirtyDegrees = [Math.sqrt(3) / 2, 0.5, 0]
    const sixtyDegrees = [0.5, Math.sqrt(3) / 2, 0]
    const run = await runSynthesis(
      [
        analysis('a', 'digest-a', 'run-a', [neuron('a1', 'a', [1, 0, 0])]),
        analysis('b', 'digest-b', 'run-b', [neuron('b1', 'b', thirtyDegrees)]),
        analysis('c', 'digest-c', 'run-c', [neuron('c1', 'c', sixtyDegrees)]),
      ],
      TEST_CONFIG,
    )

    // A-B and B-C pass, while A-C does not. Complete-link compatibility
    // therefore permits a two-member cluster but never a three-member birth.
    expect(run.edges).toHaveLength(2)
    expect(run.candidates).toEqual([])
  })

  it('girdi sırasından bağımsızdır ve akış içinde iptal edilebilir', async () => {
    const inputs = [
      analysis('a', 'digest-a', 'run-a', [
        neuron('a2', 'a', [0.99, 0.01, 0], [1, 0], { start: 4, end: 8 }),
        neuron('a1', 'a', [1, 0, 0], [1, 0], { start: 0, end: 4 }),
      ]),
      analysis('b', 'digest-b', 'run-b', [neuron('b1', 'b', [0.98, 0.02, 0])]),
      analysis('c', 'digest-c', 'run-c', [neuron('c1', 'c', [0.97, 0.03, 0])]),
    ]
    const deterministicConfig = { ...TEST_CONFIG, minimumSupport: 3, yieldEveryComparisons: 1 }
    const first = await runSynthesis(inputs, deterministicConfig)
    const second = await runSynthesis(
      [...inputs].reverse().map((entry) => ({ ...entry, nodes: [...entry.nodes].reverse() })),
      deterministicConfig,
    )

    expect(second.fingerprint).toBe(first.fingerprint)
    expect(second.edges.map((edge) => edge.fingerprint)).toEqual(
      first.edges.map((edge) => edge.fingerprint),
    )
    expect(second.candidates.map((candidate) => candidate.fingerprint)).toEqual(
      first.candidates.map((candidate) => candidate.fingerprint),
    )
    expect(second.candidates.map((candidate) => candidate.identityDigest)).toEqual(
      first.candidates.map((candidate) => candidate.identityDigest),
    )

    let cancellationChecks = 0
    await expect(
      runSynthesis(inputs, deterministicConfig, {
        shouldCancel: () => {
          cancellationChecks += 1
          return cancellationChecks > 4
        },
        yieldControl: async () => Promise.resolve(),
      }),
    ).rejects.toBeInstanceOf(SynthesisCancelledError)
  })
})
