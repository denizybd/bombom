import { describe, expect, it } from 'vitest'
import { createLanguageExperienceConfig } from './config'
import { buildLanguageDataset, splitLanguageSources } from './corpus'
import { compareText, seededUnit } from './deterministic'
import { createFixtureFamilyRun, fixtureSources } from './testFixtures'
import type { LanguageExample, SurfaceBinding } from './types'

function cloneBindings(bindings: readonly SurfaceBinding[]): SurfaceBinding[] {
  return bindings.map((binding) => ({
    ...binding,
    familyWeights: binding.familyWeights.map((entry) => ({ ...entry })),
  }))
}

function legacyCorruptionBindings(
  examples: readonly LanguageExample[],
  seed: number,
): Map<string, SurfaceBinding[]> {
  const pools = new Map<string, LanguageExample[]>()
  for (const example of examples) {
    if (example.eos || !example.targetBindings.length) continue
    const key = `${example.split}\u0000${example.granularity}`
    const values = pools.get(key) ?? []
    values.push(example)
    pools.set(key, values)
  }
  for (const values of pools.values()) values.sort((left, right) => compareText(left.id, right.id))

  return new Map(examples.map((example) => {
    if (example.eos) return [example.id, []]
    const pool = pools.get(`${example.split}\u0000${example.granularity}`) ?? []
    const alternatives = pool
      .filter((candidate) => (
        candidate.id !== example.id &&
        candidate.targetNormalizedText !== example.targetNormalizedText &&
        candidate.targetBindings.some((binding) => (
          !example.targetBindings.some((own) => own.memberKey === binding.memberKey)
        ))
      ))
      .sort((left, right) => {
        const leftKey = `${example.id}\u0000${left.id}`
        const rightKey = `${example.id}\u0000${right.id}`
        return seededUnit(seed, leftKey) - seededUnit(seed, rightKey) || compareText(leftKey, rightKey)
      })
    return [example.id, cloneBindings(alternatives[0]?.targetBindings ?? [])]
  }))
}

describe('kaynak-atomik dil veri kümesi', () => {
  it('aynı fiziksel kitaba ait chunkları tek holdout bölümünde tutar', () => {
    const chunked = Array.from({ length: 8 }, (_, index) => ({
      sourceId: `book-${Math.floor(index / 2)}:chunk-${index}`,
      sourceDigest: `digest-${index}`,
      independenceGroupId: `book-${Math.floor(index / 2)}`,
      title: `Chunk ${index}`,
      text: `Benzersiz ${index} numaralı bölüm metni.`,
    }))
    const config = createLanguageExperienceConfig()
    const result = splitLanguageSources(chunked, config)
    const splitsByGroup = new Map<string, Set<string>>()
    for (const source of result.sources) {
      const values = splitsByGroup.get(source.independenceGroupId) ?? new Set<string>()
      values.add(source.split)
      splitsByGroup.set(source.independenceGroupId, values)
    }

    expect([...splitsByGroup.values()].every((values) => values.size === 1)).toBe(true)
    expect(new Set(result.sources.map((source) => source.split))).toEqual(new Set(['train', 'validation', 'test']))
  })

  it('aynı grupta aynı sourceId ile farklı chunk özetlerini yanlış duplicate saymaz', () => {
    const sources = [
      { sourceId: 'book', sourceDigest: 'chunk-a', independenceGroupId: 'book', title: 'A', text: 'Birinci benzersiz parça.' },
      { sourceId: 'book', sourceDigest: 'chunk-b', independenceGroupId: 'book', title: 'B', text: 'İkinci benzersiz parça.' },
      { sourceId: 'other', sourceDigest: 'chunk-c', independenceGroupId: 'other', title: 'C', text: 'Üçüncü benzersiz parça.' },
    ]
    const result = splitLanguageSources(sources, createLanguageExperienceConfig())

    expect(result.sources.filter((source) => source.independenceGroupId === 'book')).toHaveLength(2)
    expect(result.duplicates).toHaveLength(0)
  })

  it('kanonik tekrarları dışlar, örnekleri yalnız kaynak splitine koyar ve chunk/grup istatistiği verir', () => {
    const duplicate = { ...fixtureSources[0], sourceId: 'copy', sourceDigest: 'copy-digest' }
    const sources = [...fixtureSources, duplicate]
    const run = createFixtureFamilyRun(sources)
    const dataset = buildLanguageDataset(run, sources, createLanguageExperienceConfig())
    const splitByDigest = new Map(dataset.sources.map((source) => [source.sourceDigest, source.split]))

    expect(dataset.duplicates.some((record) => record.reason === 'exact-canonical-duplicate')).toBe(true)
    expect(dataset.examples.every((example) => splitByDigest.get(example.sourceDigest) === example.split)).toBe(true)
    for (const split of ['train', 'validation', 'test'] as const) {
      expect(dataset.stats[split].sourceCount).toBe(dataset.stats[split].chunkCount)
      expect(dataset.stats[split].independenceGroupCount).toBeLessThanOrEqual(dataset.stats[split].chunkCount)
    }
  })

  it('girdi sırasından bağımsız aynı split ve veri parmak izini üretir', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({ seed: 77 })
    const first = buildLanguageDataset(run, fixtureSources, config)
    const second = buildLanguageDataset(run, [...fixtureSources].reverse(), config)

    expect(first.fingerprint).toBe(second.fingerprint)
    expect(first.sources.map((source) => [source.sourceDigest, source.split])).toEqual(
      second.sources.map((source) => [source.sourceDigest, source.split]),
    )
  })

  it('corruption bağlarını eski tam-sıralama algoritmasıyla derin eşdeğer üretir', () => {
    const run = createFixtureFamilyRun()
    const config = createLanguageExperienceConfig({ seed: 20260829 })
    const dataset = buildLanguageDataset(run, fixtureSources, config)
    const expected = legacyCorruptionBindings(dataset.examples, config.seed)

    expect(dataset.examples.map((example) => ({
      id: example.id,
      corruptionBindings: example.corruptionBindings,
    }))).toEqual(dataset.examples.map((example) => ({
      id: example.id,
      corruptionBindings: expected.get(example.id) ?? [],
    })))
  })

  it('yeni partiler eklense bile önceki bağımsız grup holdout atamalarını korur', () => {
    const config = createLanguageExperienceConfig({ seed: 31 })
    const first = splitLanguageSources(fixtureSources.slice(0, 4), config)
    const assignments = first.sources.map((source) => ({
      independenceGroupId: source.independenceGroupId,
      split: source.split,
    }))
    const expanded = splitLanguageSources(fixtureSources, config, assignments)
    const firstByGroup = new Map(assignments.map((assignment) => [assignment.independenceGroupId, assignment.split]))

    expect(expanded.sources
      .filter((source) => firstByGroup.has(source.independenceGroupId))
      .every((source) => firstByGroup.get(source.independenceGroupId) === source.split)).toBe(true)
  })

  it('Ollama ve diğer sentetik kaynakları validation/test yerine yalnız train bölümünde tutar', () => {
    const human = Array.from({ length: 3 }, (_, index) => ({
      sourceId: `human-${index}`,
      sourceDigest: `human-digest-${index}`,
      independenceGroupId: `human-${index}`,
      title: `Gerçek ${index}`,
      text: `İnsan kaynağından gelen benzersiz ${index} numaralı metin.`,
      provenance: { kind: 'human' as const },
    }))
    const synthetic = [
      {
        sourceId: 'ollama-source', sourceDigest: 'ollama-digest', independenceGroupId: 'ollama-group',
        title: 'Ollama', text: 'Ollama tarafından üretilmiş benzersiz eğitim metni.',
        provenance: { kind: 'ollama' as const },
      },
      {
        sourceId: 'generated-source', sourceDigest: 'generated-digest', independenceGroupId: 'generated-group',
        title: 'Üretilmiş', text: 'Başka bir üretici tarafından hazırlanmış benzersiz eğitim metni.',
        provenance: { kind: 'generated' as const },
      },
    ]
    const result = splitLanguageSources(
      [...human, ...synthetic],
      createLanguageExperienceConfig(),
      [{ independenceGroupId: 'ollama-group', split: 'test' }],
    )

    expect(result.sources.filter((source) => ['ollama', 'generated'].includes(source.provenance?.kind ?? 'unknown'))
      .every((source) => source.split === 'train')).toBe(true)
    expect(new Set(result.sources.filter((source) => source.provenance?.kind === 'human').map((source) => source.split)))
      .toEqual(new Set(['train', 'validation', 'test']))
    expect(result.warnings).toContain('1 eski sentetik holdout ataması güvenli eğitim atamasıyla değiştirildi.')
  })
})
