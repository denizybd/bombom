import type {
  CanonicalSource,
  NodeBranch,
  RawTextNeuron,
  SplitBoundary,
  TextSource,
} from '../domain/types'

interface GraphemePart {
  segment: string
  index: number
}

interface DraftNode extends Omit<
  RawTextNeuron,
  | 'identity'
  | 'relation'
  | 'identityDigest'
  | 'surfaceKey'
  | 'identityNorm'
  | 'relationNorm'
  | 'contextSupport'
  | 'contextRevision'
  | 'trace'
> {}

export class SpanNodeLimitError extends RangeError {
  constructor(public readonly maximumNodes: number) {
    super(
      `Bu kaynak ${maximumNodes.toLocaleString('tr-TR')} ham nöronluk güvenli deney sınırını aşıyor. Metni küçültün veya yaprak eşiğini büyütün; hiçbir düğüm gizlice silinmedi.`,
    )
    this.name = 'SpanNodeLimitError'
  }
}

const graphemeSegmenter = new Intl.Segmenter('tr', { granularity: 'grapheme' })

export function canonicalizeSource(source: TextSource): CanonicalSource {
  return {
    ...source,
    text: source.rawText.replace(/\r\n?/g, '\n').normalize('NFC'),
  }
}

export function segmentGraphemes(value: string): GraphemePart[] {
  return Array.from(graphemeSegmenter.segment(value), (entry) => ({
    segment: entry.segment,
    index: entry.index,
  }))
}

export function countGraphemes(value: string): number {
  return Array.from(graphemeSegmenter.segment(value)).length
}

export function findParagraphRanges(text: string): Array<{ start: number; end: number }> {
  const ranges: Array<{ start: number; end: number }> = []
  const separator = /\n[\t ]*\n+/g
  let start = 0

  for (const match of text.matchAll(separator)) {
    const end = match.index ?? start
    if (/\S/u.test(text.slice(start, end))) ranges.push({ start, end })
    start = end + match[0].length
  }
  if (/\S/u.test(text.slice(start))) ranges.push({ start, end: text.length })
  return ranges
}

function internalBoundaries(value: string, expression: RegExp): number[] {
  const boundaries: number[] = []
  for (const match of value.matchAll(expression)) {
    const position = (match.index ?? 0) + match[0].length
    if (position > 0 && position < value.length) boundaries.push(position)
  }
  return boundaries
}

function nearestToMiddle(candidates: number[], length: number): number | null {
  if (!candidates.length) return null
  const middle = length / 2
  return [...candidates].sort((left, right) => {
    const distance = Math.abs(left - middle) - Math.abs(right - middle)
    return distance === 0 ? left - right : distance
  })[0]
}

export function chooseSplit(
  value: string,
  minimumLeafGraphemes: number,
): { offset: number; boundary: Exclude<SplitBoundary, 'leaf'> } | null {
  const graphemes = segmentGraphemes(value)
  if (graphemes.length <= minimumLeafGraphemes) return null

  const minimumUnits = Math.min(minimumLeafGraphemes, Math.max(1, Math.floor(graphemes.length / 3)))
  const safeStart = graphemes[minimumUnits]?.index ?? 1
  const safeEnd = graphemes[graphemes.length - minimumUnits]?.index ?? value.length - 1
  const safe = (positions: number[]) =>
    positions.filter((position) => position >= safeStart && position <= safeEnd)

  const sentence = nearestToMiddle(
    safe(internalBoundaries(value, /[.!?…]+(?:["'»”’\])}]*)[\t ]*/gu)),
    value.length,
  )
  if (sentence !== null) return { offset: sentence, boundary: 'sentence' }

  const space = nearestToMiddle(safe(internalBoundaries(value, /\s+/gu)), value.length)
  if (space !== null) return { offset: space, boundary: 'space' }

  const graphemePositions = graphemes.slice(1).map((part) => part.index)
  const grapheme = nearestToMiddle(safe(graphemePositions), value.length)
  return grapheme === null ? null : { offset: grapheme, boundary: 'grapheme' }
}

export function buildSpanForest(
  source: CanonicalSource,
  minimumLeafGraphemes: number,
  maximumNodes = Number.POSITIVE_INFINITY,
): { nodes: DraftNode[]; rootIds: string[] } {
  const nodes: DraftNode[] = []
  const rootIds: string[] = []

  const createNode = (
    paragraphIndex: number,
    start: number,
    end: number,
    depth: number,
    branch: NodeBranch,
    parentId: string | null,
  ): string => {
    if (nodes.length >= maximumNodes) throw new SpanNodeLimitError(maximumNodes)
    // The source id is part of the neuron id on purpose. Local sequence numbers
    // are deterministic, while the source namespace prevents collisions when
    // multiple texts live in the same laboratory archive.
    const id = `HN-${source.id}-${String(nodes.length + 1).padStart(6, '0')}`
    const text = source.text.slice(start, end)
    const graphemeCount = countGraphemes(text)
    const draft: DraftNode = {
      id,
      sourceId: source.id,
      paragraphIndex,
      start,
      end,
      text,
      graphemeCount,
      depth,
      branch,
      parentId,
      childIds: [],
      splitBoundary: 'leaf',
    }
    nodes.push(draft)

    const split = chooseSplit(text, minimumLeafGraphemes)
    if (!split) return id
    draft.splitBoundary = split.boundary
    const cut = start + split.offset
    const leftId = createNode(paragraphIndex, start, cut, depth + 1, 'left', id)
    const rightId = createNode(paragraphIndex, cut, end, depth + 1, 'right', id)
    draft.childIds = [leftId, rightId]
    return id
  }

  findParagraphRanges(source.text).forEach((range, paragraphIndex) => {
    rootIds.push(createNode(paragraphIndex, range.start, range.end, 0, 'root', null))
  })

  return { nodes, rootIds }
}
