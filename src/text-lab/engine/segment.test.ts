import { describe, expect, it } from 'vitest'
import {
  buildSpanForest,
  canonicalizeSource,
  chooseSplit,
  findParagraphRanges,
  SpanNodeLimitError,
} from './segment'

describe('text segmentation', () => {
  it('normalizes line endings and Unicode without lowercasing Turkish text', () => {
    const source = canonicalizeSource({
      id: 'source-1',
      title: 'Deneme',
      rawText: 'I\u0307stanbul\r\nÇALIŞIR',
    })

    expect(source.text).toBe('İstanbul\nÇALIŞIR')
  })

  it('keeps paragraph ranges separate without discarding internal whitespace', () => {
    const text = 'Birinci satır.\nDevamı.\n\n  İkinci paragraf!  '
    const ranges = findParagraphRanges(text)

    expect(ranges.map(({ start, end }) => text.slice(start, end))).toEqual([
      'Birinci satır.\nDevamı.',
      '  İkinci paragraf!  ',
    ])
  })

  it('prefers a sentence boundary and deterministically chooses the left tie', () => {
    const split = chooseSplit('Bir cümle. İki cümle.', 3)
    expect(split?.boundary).toBe('sentence')
    expect(split?.offset).toBe('Bir cümle. '.length)
  })

  it('builds a lossless binary forest with children inside their parent', () => {
    const source = canonicalizeSource({
      id: 'source-1',
      title: 'Deneme',
      rawText: 'Finans yükseldi. Piyasa sakin.\n\nİkinci paragraf.',
    })
    const forest = buildSpanForest(source, 4)
    const byId = new Map(forest.nodes.map((node) => [node.id, node]))

    expect(forest.rootIds).toHaveLength(2)
    for (const node of forest.nodes) {
      expect(source.text.slice(node.start, node.end)).toBe(node.text)
      if (node.childIds.length) {
        const [left, right] = node.childIds.map((id) => byId.get(id)!)
        expect(left.parentId).toBe(node.id)
        expect(right.parentId).toBe(node.id)
        expect(left.start).toBe(node.start)
        expect(left.end).toBe(right.start)
        expect(right.end).toBe(node.end)
        expect(left.text + right.text).toBe(node.text)
      }
    }
  })

  it('stops explicitly instead of truncating when the safety boundary is exceeded', () => {
    const source = canonicalizeSource({ id: 'source-1', title: 'Büyük', rawText: 'a b c d e f g h' })
    expect(() => buildSpanForest(source, 1, 3)).toThrow(SpanNodeLimitError)
  })

  it('namespaces neuron ids so separate sources can coexist without collisions', () => {
    const first = buildSpanForest(
      canonicalizeSource({ id: 'SRC-FIRST', title: 'Bir', rawText: 'Aynı metin.' }),
      4,
    )
    const second = buildSpanForest(
      canonicalizeSource({ id: 'SRC-SECOND', title: 'İki', rawText: 'Aynı metin.' }),
      4,
    )

    expect(first.nodes[0].id).toBe('HN-SRC-FIRST-000001')
    expect(second.nodes[0].id).toBe('HN-SRC-SECOND-000001')
    expect(new Set([...first.nodes, ...second.nodes].map((node) => node.id)).size)
      .toBe(first.nodes.length + second.nodes.length)
  })
})
