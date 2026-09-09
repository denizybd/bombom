import { describe, expect, it } from 'vitest'
import { DEFAULT_TEXT_LAB_CONFIG } from '../domain/config'
import { analyzeText } from '../engine/pipeline'
import {
  analysisMatchesProject,
  createPersistedProject,
  parseProject,
  stringifyProject,
  validateProject,
} from './projectSchema'

describe('project schema', () => {
  it('normalizes a partial version-one config before scheduling a rebuild', () => {
    const restored = validateProject({
      format: 'metin-laboratuvari-project',
      version: 1,
      savedAt: '2026-01-01T00:00:00.000Z',
      source: { id: 'legacy', title: 'Eski metin', rawText: 'Korunacak ham metin.' },
      config: { matrixSide: 64, contextRetention: 0.7 },
      analysis: { legacy: true },
    })

    expect(restored.source.rawText).toBe('Korunacak ham metin.')
    expect(restored.config.matrixSide).toBe(64)
    expect(Object.hasOwn(restored.config, 'representationProfile')).toBe(false)
    expect(restored.config.parentTraceWeight + restored.config.siblingTraceWeight).toBeCloseTo(0.3)
    expect(restored.analysis).toBeNull()
  })

  it('round-trips typed analysis arrays without losing values', async () => {
    const source = { id: 'source-1', title: 'Kısa', rawText: 'Bir kısa metin.' }
    const config = { ...DEFAULT_TEXT_LAB_CONFIG, matrixSide: 8, minimumLeafGraphemes: 6 }
    const analysis = await analyzeText(source, config, { yieldEveryPairs: 100_000 })
    const restored = parseProject(stringifyProject(createPersistedProject(source, config, analysis)))

    expect(restored.analysis?.similarities).toBeInstanceOf(Float32Array)
    expect(Array.from(restored.analysis?.similarities ?? [])).toEqual(
      Array.from(analysis.similarities),
    )
    expect(Array.from(restored.analysis?.nodes[0].identity ?? [])).toEqual(
      Array.from(analysis.nodes[0].identity),
    )
    expect(Array.from(restored.analysis?.nodes[0].relation ?? [])).toEqual(
      Array.from(analysis.nodes[0].relation),
    )
    expect(analysisMatchesProject(restored.source, restored.config, restored.analysis)).toBe(true)
    expect(
      analysisMatchesProject(
        { ...restored.source, rawText: `${restored.source.rawText} değişti` },
        restored.config,
        restored.analysis,
      ),
    ).toBe(false)
    expect(
      analysisMatchesProject(
        restored.source,
        { ...restored.config, minimumLeafGraphemes: restored.config.minimumLeafGraphemes + 1 },
        restored.analysis,
      ),
    ).toBe(false)
    expect(analysisMatchesProject(
      { ...restored.source, title: 'Başka başlık' },
      restored.config,
      restored.analysis,
    )).toBe(false)
  })

  it('never persists or accepts an analysis after its source/config snapshot changes', async () => {
    const source = { id: 'source-stale', title: 'Özgün', rawText: 'Finans yükseldi.' }
    const config = { ...DEFAULT_TEXT_LAB_CONFIG, matrixSide: 8, minimumLeafGraphemes: 6 }
    const analysis = await analyzeText(source, config, { yieldEveryPairs: 100_000 })
    const changedSource = { ...source, rawText: `${source.rawText} Piyasa sakin.` }

    expect(createPersistedProject(changedSource, config, analysis, 7)).toMatchObject({
      storageRevision: 7,
      analysis: null,
    })
    expect(() => validateProject({
      ...createPersistedProject(changedSource, config, null, 7),
      analysis,
    })).toThrow('farklı bir kaynak veya ayar')
  })

  it('rejects an empty or numerically corrupted cached analysis', async () => {
    const source = { id: 'source-1', title: 'Kısa', rawText: 'Doğrulanacak metin.' }
    const config = { ...DEFAULT_TEXT_LAB_CONFIG, matrixSide: 8, minimumLeafGraphemes: 6 }
    const analysis = await analyzeText(source, config, { yieldEveryPairs: 100_000 })
    const project = createPersistedProject(source, config, analysis)

    expect(() => validateProject({
      ...project,
      analysis: {
        ...analysis,
        nodes: [],
        rootIds: [],
        similarities: new Float32Array(),
        habitProbabilities: new Float32Array(),
      },
    })).toThrow('analiz')

    const corrupted = createPersistedProject(source, config, {
      ...analysis,
      similarities: Float32Array.from(analysis.similarities),
    })
    corrupted.analysis!.similarities[0] = Number.NaN
    expect(() => validateProject(corrupted)).toThrow('K veya H')

    const changedIdentity = createPersistedProject(source, config, {
      ...analysis,
      nodes: analysis.nodes.map((node, index) => index === 0
        ? { ...node, identity: Float32Array.from(node.identity) }
        : node),
    })
    changedIdentity.analysis!.nodes[0].identity[0] += 0.01
    expect(() => validateProject(changedIdentity)).toThrow('ham nöron')
  })
})
