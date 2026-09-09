import { describe, expect, it } from 'vitest'
import { createLanguageExperienceConfig } from './config'
import { buildSurfaceHierarchy } from './hierarchy'

describe('hiyerarşik dil yüzeyi', () => {
  it('kaynak aralıklarını document > paragraph > sentence > clause > token > subword olarak korur', () => {
    const source = {
      sourceId: 'kitap-1:chunk-1',
      sourceDigest: 'digest-1',
      independenceGroupId: 'kitap-1',
      title: 'Deneme',
      text: 'Ali uzunekmek kesti, sonra durdu.\n\nİkinci paragraf geldi.',
    }
    const config = createLanguageExperienceConfig({ minimumSubwordGraphemes: 3 })
    const pieces = buildSurfaceHierarchy(source, config)
    const byId = new Map(pieces.map((piece) => [piece.id, piece]))
    const root = pieces.find((piece) => piece.level === 'document')!

    expect(root.text).toBe(source.text)
    expect(root.childIds).toHaveLength(2)
    expect(pieces.some((piece) => piece.level === 'sentence')).toBe(true)
    expect(pieces.some((piece) => piece.level === 'clause')).toBe(true)
    expect(pieces.some((piece) => piece.level === 'subword' && piece.parentId)).toBe(true)
    for (const piece of pieces) {
      expect(source.text.slice(piece.start, piece.end)).toBe(piece.text)
      for (const childId of piece.childIds) {
        const child = byId.get(childId)!
        expect(child.parentId).toBe(piece.id)
        expect(child.start).toBeGreaterThanOrEqual(piece.start)
        expect(child.end).toBeLessThanOrEqual(piece.end)
      }
    }
  })

  it('aynı seed ve kaynak için tamamen deterministiktir ve girdi metnini değiştirmez', () => {
    const source = { sourceId: 's', sourceDigest: 'd', title: 'T', text: 'Cafe\u0301 güzel.' }
    const before = structuredClone(source)
    const config = createLanguageExperienceConfig({ includeGraphemeExperiences: true })
    const first = buildSurfaceHierarchy(source, config)
    const second = buildSurfaceHierarchy(source, config)

    expect(first).toEqual(second)
    expect(source).toEqual(before)
    expect(first[0].text).toContain('Café')
    expect(first.some((piece) => piece.level === 'grapheme')).toBe(true)
  })
})
