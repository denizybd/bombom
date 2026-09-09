import { describe, expect, it } from 'vitest'
import { createIndependentSourceIndex } from './independence'

describe('bağımsız fiziksel kaynak indeksi', () => {
  it('aynı kitabın chunklarını ve yeniden adlandırılmış kopyayı geçişli olarak tek grupta tutar', () => {
    const refs = [
      { sourceId: 'book-a', sourceDigest: 'chunk-a1' },
      { sourceId: 'book-a', sourceDigest: 'shared-copy' },
      { sourceId: 'book-renamed', sourceDigest: 'shared-copy' },
      { sourceId: 'book-renamed', sourceDigest: 'chunk-a3' },
      { sourceId: 'book-b', sourceDigest: 'chunk-b1' },
    ]
    const index = createIndependentSourceIndex(refs)

    expect(index.groupCount).toBe(2)
    expect(index.groupKey(refs[0])).toBe(index.groupKey(refs[3]))
    expect(index.groupKey(refs[0])).not.toBe(index.groupKey(refs[4]))
    expect(index.representativeSourceId(refs[2])).toBe('book-a')
  })

  it('girdi sırası değişse bile aynı kararlı grup anahtarlarını üretir', () => {
    const refs = [
      { sourceId: 'book-a', sourceDigest: 'chunk-a1' },
      { sourceId: 'book-a', sourceDigest: 'chunk-a2' },
      { sourceId: 'book-b', sourceDigest: 'chunk-b1' },
    ]
    const first = createIndependentSourceIndex(refs)
    const second = createIndependentSourceIndex([...refs].reverse())

    expect(refs.map((ref) => first.groupKey(ref))).toEqual(refs.map((ref) => second.groupKey(ref)))
    expect(first.groupCount).toBe(second.groupCount)
  })

  it('aynı Ollama üretim soyundaki farklı metinleri tek bağımsız tanık sayar', () => {
    const refs = [
      { independenceGroupId: 'ollama-lineage:qwen14b', sourceId: 'lesson-a', sourceDigest: 'digest-a' },
      { independenceGroupId: 'ollama-lineage:qwen14b', sourceId: 'lesson-b', sourceDigest: 'digest-b' },
      { independenceGroupId: 'human:book', sourceId: 'book', sourceDigest: 'digest-book' },
    ]
    const index = createIndependentSourceIndex(refs)

    expect(index.groupCount).toBe(2)
    expect(index.groupKey(refs[0])).toBe(index.groupKey(refs[1]))
    expect(index.groupKey(refs[0])).not.toBe(index.groupKey(refs[2]))
  })

  it('açık üretim soyu anahtarını yeni çıktılar eklendiğinde kararlı tutar', () => {
    const firstRef = {
      independenceGroupId: 'ollama-lineage:qwen14b',
      sourceId: 'lesson-a',
      sourceDigest: 'digest-a',
    }
    const secondRef = {
      independenceGroupId: 'ollama-lineage:qwen14b',
      sourceId: 'lesson-b',
      sourceDigest: 'digest-b',
    }

    const first = createIndependentSourceIndex([firstRef])
    const expanded = createIndependentSourceIndex([firstRef, secondRef])

    expect(expanded.groupKey(firstRef)).toBe(first.groupKey(firstRef))
    expect(expanded.groupKey(secondRef)).toBe(first.groupKey(firstRef))
  })

  it('açık soylar arasında aynı digest kopyasını yine geçişli olarak birleştirir', () => {
    const refs = [
      { independenceGroupId: 'lineage-a', sourceId: 'source-a', sourceDigest: 'shared' },
      { independenceGroupId: 'lineage-b', sourceId: 'source-b', sourceDigest: 'shared' },
      { independenceGroupId: 'lineage-b', sourceId: 'source-c', sourceDigest: 'other' },
    ]
    const index = createIndependentSourceIndex(refs)

    expect(index.groupCount).toBe(1)
    expect(new Set(refs.map((ref) => index.groupKey(ref))).size).toBe(1)
  })
})
