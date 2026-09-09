import type {
  LanguageCorpusSource,
  LanguageExperienceConfig,
  SurfacePiece,
  SurfacePieceLevel,
} from './types'
import {
  canonicalizeLanguageText,
  compareText,
  languageDigest,
  normalizeSurface,
} from './deterministic'

interface Range {
  start: number
  end: number
}

const sentenceSegmenter = new Intl.Segmenter('tr', { granularity: 'sentence' })
const wordSegmenter = new Intl.Segmenter('tr', { granularity: 'word' })
const graphemeSegmenter = new Intl.Segmenter('tr', { granularity: 'grapheme' })

function pieceId(
  sourceDigest: string,
  level: SurfacePieceLevel,
  start: number,
  end: number,
): string {
  return `LP-${languageDigest('piece', sourceDigest, level, start, end).slice(-20)}`
}

function nonWhitespaceRanges(value: string, offset = 0): Range[] {
  const ranges: Range[] = []
  const expression = /\S(?:[\s\S]*?\S)?(?=\n[\t ]*\n+|$)/gu
  for (const match of value.matchAll(expression)) {
    const start = offset + (match.index ?? 0)
    ranges.push({ start, end: start + match[0].length })
  }
  return ranges
}

function sentenceRanges(value: string, offset: number): Range[] {
  return Array.from(sentenceSegmenter.segment(value), (entry) => {
    const leading = entry.segment.match(/^\s*/u)?.[0].length ?? 0
    const trailing = entry.segment.match(/\s*$/u)?.[0].length ?? 0
    return {
      start: offset + entry.index + leading,
      end: offset + entry.index + entry.segment.length - trailing,
    }
  }).filter((range) => range.end > range.start)
}

function clauseRanges(value: string, offset: number): Range[] {
  const ranges: Range[] = []
  let start = 0
  const boundary = /[,;:]+|\s+[—–-]\s+/gu
  for (const match of value.matchAll(boundary)) {
    const rawEnd = (match.index ?? 0) + match[0].length
    const raw = value.slice(start, rawEnd)
    const leading = raw.match(/^\s*/u)?.[0].length ?? 0
    const trailing = raw.match(/\s*$/u)?.[0].length ?? 0
    if (rawEnd - trailing > start + leading) {
      ranges.push({ start: offset + start + leading, end: offset + rawEnd - trailing })
    }
    start = rawEnd
  }
  const raw = value.slice(start)
  const leading = raw.match(/^\s*/u)?.[0].length ?? 0
  const trailing = raw.match(/\s*$/u)?.[0].length ?? 0
  if (value.length - trailing > start + leading) {
    ranges.push({ start: offset + start + leading, end: offset + value.length - trailing })
  }
  return ranges
}

function tokenRanges(value: string, offset: number): Range[] {
  return Array.from(wordSegmenter.segment(value), (entry) => ({
    start: offset + entry.index,
    end: offset + entry.index + entry.segment.length,
    text: entry.segment,
  }))
    .filter((entry) => /\S/u.test(entry.text))
    .map(({ start, end }) => ({ start, end }))
}

function graphemeRanges(value: string, offset: number): Range[] {
  return Array.from(graphemeSegmenter.segment(value), (entry) => ({
    start: offset + entry.index,
    end: offset + entry.index + entry.segment.length,
  }))
}

function bisectGraphemes(value: string, offset: number): Range[] {
  const graphemes = Array.from(graphemeSegmenter.segment(value))
  if (graphemes.length < 2) return [{ start: offset, end: offset + value.length }]
  const cut = graphemes[Math.floor(graphemes.length / 2)]?.index ?? Math.ceil(value.length / 2)
  return [
    { start: offset, end: offset + cut },
    { start: offset + cut, end: offset + value.length },
  ].filter((range) => range.end > range.start)
}

function isLexicalToken(value: string): boolean {
  return /[\p{L}\p{N}]/u.test(value)
}

/** Builds a deterministic containment tree; it does not allocate any neuron. */
export function buildSurfaceHierarchy(
  sourceInput: LanguageCorpusSource,
  config: LanguageExperienceConfig,
): SurfacePiece[] {
  const source = { ...sourceInput, text: canonicalizeLanguageText(sourceInput.text) }
  if (!source.sourceId || !source.sourceDigest || !source.text.trim()) {
    throw new TypeError('Dil korpusu kaynağının kimliği, özeti ve metni zorunludur.')
  }
  const pieces: SurfacePiece[] = []
  const byId = new Map<string, SurfacePiece>()

  const add = (
    level: SurfacePieceLevel,
    start: number,
    end: number,
    parentId: string | null,
  ): SurfacePiece => {
    if (start < 0 || end <= start || end > source.text.length) {
      throw new RangeError('Yüzey parçasının kaynak aralığı geçersiz.')
    }
    const id = pieceId(source.sourceDigest, level, start, end)
    const existing = byId.get(id)
    if (existing) {
      if (existing.parentId !== parentId) throw new Error('Yüzey hiyerarşisinde çakışan ebeveyn bulundu.')
      return existing
    }
    const text = source.text.slice(start, end)
    const piece: SurfacePiece = {
      id,
      sourceId: source.sourceId,
      sourceDigest: source.sourceDigest,
      level,
      start,
      end,
      text,
      normalizedText: normalizeSurface(text),
      parentId,
      childIds: [],
      bindings: [],
    }
    pieces.push(piece)
    byId.set(id, piece)
    if (parentId) byId.get(parentId)?.childIds.push(id)
    return piece
  }

  const addSubwordBranch = (range: Range, parentId: string): void => {
    const subword = add('subword', range.start, range.end, parentId)
    const subwordText = source.text.slice(subword.start, subword.end)
    const parts = graphemeRanges(subwordText, subword.start)
    if (parts.length > config.minimumSubwordGraphemes) {
      for (const childRange of bisectGraphemes(subwordText, subword.start)) {
        addSubwordBranch(childRange, subword.id)
      }
      return
    }
    if (config.includeGraphemeExperiences && parts.length > 1) {
      for (const graphemeRange of parts) add('grapheme', graphemeRange.start, graphemeRange.end, subword.id)
    }
  }

  const document = add('document', 0, source.text.length, null)
  for (const paragraphRange of nonWhitespaceRanges(source.text)) {
    const paragraph = add('paragraph', paragraphRange.start, paragraphRange.end, document.id)
    const paragraphText = source.text.slice(paragraph.start, paragraph.end)
    for (const sentenceRange of sentenceRanges(paragraphText, paragraph.start)) {
      const sentence = add('sentence', sentenceRange.start, sentenceRange.end, paragraph.id)
      const sentenceText = source.text.slice(sentence.start, sentence.end)
      for (const clauseRange of clauseRanges(sentenceText, sentence.start)) {
        const clause = add('clause', clauseRange.start, clauseRange.end, sentence.id)
        const clauseText = source.text.slice(clause.start, clause.end)
        for (const tokenRange of tokenRanges(clauseText, clause.start)) {
          const token = add('token', tokenRange.start, tokenRange.end, clause.id)
          const tokenText = source.text.slice(token.start, token.end)
          const tokenGraphemes = graphemeRanges(tokenText, token.start)
          if (isLexicalToken(tokenText) && tokenGraphemes.length > config.minimumSubwordGraphemes) {
            const subwordRanges = bisectGraphemes(tokenText, token.start)
            for (const subwordRange of subwordRanges) addSubwordBranch(subwordRange, token.id)
          } else if (config.includeGraphemeExperiences && tokenGraphemes.length > 1) {
            for (const graphemeRange of tokenGraphemes) add('grapheme', graphemeRange.start, graphemeRange.end, token.id)
          }
        }
      }
    }
  }

  for (const piece of pieces) {
    piece.childIds.sort((leftId, rightId) => {
      const left = byId.get(leftId)!
      const right = byId.get(rightId)!
      return left.start - right.start || left.end - right.end || compareText(left.id, right.id)
    })
  }
  return pieces.sort((left, right) =>
    left.start - right.start || right.end - left.end || compareText(left.level, right.level),
  )
}
