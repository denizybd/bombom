import { describe, expect, it } from 'vitest'
import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  DEFAULT_TEXT_LAB_CONFIG,
  createTextLabConfigForProfile,
} from '../domain/config'
import { analyzeText } from './pipeline'
import { applyContextObservation } from './context'
import { cosineRegion } from './matrix'
import { digestVector } from './representation'
import { detailFromAnalysisPair } from './similarity'

describe('raw text pipeline', () => {
  const config = {
    ...DEFAULT_TEXT_LAB_CONFIG,
    matrixSide: 16,
    minimumLeafGraphemes: 6,
  }

  it('keeps repeated occurrences separate with equal identity and contextual relation', async () => {
    const analysis = await analyzeText(
      {
        id: 'source-test',
        title: 'Bağlam deneyi',
        rawText: 'finans artar.\n\nfinans düşer.',
      },
      config,
      { yieldEveryPairs: 100_000 },
    )
    const occurrences = analysis.nodes.filter((node) => node.text === 'finans ')

    expect(occurrences).toHaveLength(2)
    expect(Array.from(occurrences[0].identity)).toEqual(
      Array.from(occurrences[1].identity),
    )
    expect(Array.from(occurrences[0].relation)).not.toEqual(
      Array.from(occurrences[1].relation),
    )
  })

  it('computes exact symmetric K and row-normalized initial H', async () => {
    const analysis = await analyzeText(
      {
        id: 'source-test',
        title: 'Küçük deney',
        rawText: 'Piyasa değişir. Finans etkilenir.',
      },
      config,
      { yieldEveryPairs: 100_000 },
    )
    const count = analysis.nodes.length

    for (let row = 0; row < count; row += 1) {
      expect(analysis.similarities[row * count + row]).toBe(1)
      for (let column = 0; column < count; column += 1) {
        expect(analysis.similarities[row * count + column]).toBeCloseTo(
          analysis.similarities[column * count + row],
          6,
        )
        const diagnostic = detailFromAnalysisPair(
          analysis.nodes[row],
          analysis.nodes[column],
          analysis.layout,
          analysis.config,
          analysis.stats.medianRelationNorm,
        )
        expect(analysis.similarities[row * count + column]).toBeCloseTo(
          diagnostic.combined,
          6,
        )
      }
      const sum = analysis.habitProbabilities
        .slice(row * count, (row + 1) * count)
        .reduce((total, value) => total + value, 0)
      expect(sum === 0 || Math.abs(sum - 1) < 1e-5).toBe(true)
    }
  })

  it('preserves the identity mask while context propagates', async () => {
    const analysis = await analyzeText(
      { id: 'source-test', title: 'Kimlik', rawText: 'Bir bağlam başka bağlam.' },
      config,
      { yieldEveryPairs: 100_000 },
    )
    const child = analysis.nodes.find((node) => node.parentId)
    expect(child).toBeDefined()
    expect(child!.relationNorm).toBeGreaterThan(0)
    const selfIdentity = cosineRegion(
      child!.identity,
      child!.identity,
      0,
      analysis.layout.identityLength,
      child!.identityNorm,
      child!.identityNorm,
    )
    expect(selfIdentity).toBeCloseTo(1, 6)
    expect([...child!.identity, ...child!.relation].every((value) => value >= -1 && value <= 1)).toBe(true)
  })

  it('keeps identity byte-stable while plastic context learns observations', async () => {
    const analysis = await analyzeText(
      { id: 'source-plastic', title: 'Plastik bağlam', rawText: 'Sabit kimlik değişen bağlam.' },
      config,
      { yieldEveryPairs: 100_000 },
    )
    const node = analysis.nodes.find((candidate) => candidate.parentId)!
    const identitySnapshot = Float32Array.from(node.identity)
    const identityDigest = node.identityDigest
    const observation = Float32Array.from(node.relation, (value, index) =>
      index % 2 === 0 ? Math.min(1, value + 0.2) : Math.max(-1, value - 0.2),
    )

    const alpha = applyContextObservation(node, observation, 0.9)

    expect(alpha).toBeGreaterThan(0)
    expect(Array.from(node.identity)).toEqual(Array.from(identitySnapshot))
    expect(node.identityDigest).toBe(identityDigest)
    expect(digestVector(node.identity)).toBe(identityDigest)
    expect(node.contextRevision).toBeGreaterThan(1)
  })

  it('reports monotonic overall progress across pipeline stages', async () => {
    const ratios: number[] = []
    await analyzeText(
      { id: 'source-test', title: 'İlerleme', rawText: 'Bir metin ilerler. Başka bir iz kalır.' },
      config,
      {
        yieldEveryPairs: 2,
        onProgress: (progress) => ratios.push(progress.ratio),
      },
    )

    expect(ratios[0]).toBe(0)
    expect(ratios.at(-1)).toBe(1)
    expect(ratios.every((ratio, index) => index === 0 || ratio >= ratios[index - 1])).toBe(true)
  })

  it('honors cancellation before allocating the full similarity result', async () => {
    await expect(
      analyzeText(
        { id: 'source-test', title: 'İptal', rawText: 'İptal edilecek kısa bir metin.' },
        config,
        { shouldCancel: () => true },
      ),
    ).rejects.toThrow('durduruldu')
  })

  it('honors cancellation while initial H rows are being produced', async () => {
    let cancel = false
    await expect(
      analyzeText(
        {
          id: 'source-test',
          title: 'H iptali',
          rawText: 'Birinci parça ilerler. İkinci parça iz bırakır. Üçüncü parça tamamlanır.',
        },
        config,
        {
          shouldCancel: () => cancel,
          onProgress: (progress) => {
            if (progress.stage === 'connecting' && progress.completed > 0) cancel = true
          },
          yieldEveryPairs: 100_000,
        },
      ),
    ).rejects.toThrow('durduruldu')
  })

  it('is deterministic for the same source, settings and seed', async () => {
    const source = { id: 'source-test', title: 'Tekrar', rawText: 'Aynı deney aynı izi bırakır.' }
    const first = await analyzeText(source, config, { yieldEveryPairs: 100_000 })
    const second = await analyzeText(source, config, { yieldEveryPairs: 100_000 })

    expect(first.nodes.map((node) => node.id)).toEqual(second.nodes.map((node) => node.id))
    expect(first.nodes.map((node) => [Array.from(node.identity), Array.from(node.relation)])).toEqual(
      second.nodes.map((node) => [Array.from(node.identity), Array.from(node.relation)]),
    )
    expect(Array.from(first.similarities)).toEqual(Array.from(second.similarities))
    expect(Array.from(first.habitProbabilities)).toEqual(Array.from(second.habitProbabilities))
  })

  it('builds a deterministic opt-in 8/56 analysis without changing the legacy default', async () => {
    const adaptive = createTextLabConfigForProfile(ADAPTIVE_TEXT_LAB_PROFILE_ID, {
      minimumLeafGraphemes: 6,
    })
    const source = {
      id: 'source-adaptive-profile',
      title: 'Uyarlanabilir temsil',
      rawText: 'Aynı fiziksel sayfalar, farklı öğrenilmiş işlevler taşıyabilir.',
    }
    const first = await analyzeText(source, adaptive, { yieldEveryPairs: 100_000 })
    const second = await analyzeText(source, adaptive, { yieldEveryPairs: 100_000 })

    expect(first.layout.identityLength).toBe(512)
    expect(first.layout.relationLength).toBe(3_584)
    expect(first.representationSpaceId).toBe(second.representationSpaceId)
    expect(first.nodes.map((node) => node.identityDigest)).toEqual(
      second.nodes.map((node) => node.identityDigest),
    )
    expect(first.representationSpaceId).not.toBe(
      (await analyzeText(source, DEFAULT_TEXT_LAB_CONFIG, {
        yieldEveryPairs: 100_000,
      })).representationSpaceId,
    )
  })
})
