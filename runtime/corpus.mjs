const MAX_CHUNK_CHARACTERS = 2_400
const MIN_SOFT_BOUNDARY = Math.floor(MAX_CHUNK_CHARACTERS * 0.55)

function safeCodeUnitBoundary(text, index) {
  let cut = Math.max(1, Math.min(text.length, index))
  const previous = text.charCodeAt(cut - 1)
  const next = text.charCodeAt(cut)
  if (previous >= 0xd800 && previous <= 0xdbff && next >= 0xdc00 && next <= 0xdfff) cut += 1
  // Keep combining marks and variation selectors attached to their base.
  while (cut < text.length && /[\p{M}\uFE0E\uFE0F]/u.test(text[cut])) cut += 1
  return cut
}

function preferredCut(window) {
  const minimum = Math.min(MIN_SOFT_BOUNDARY, Math.max(1, window.length - 1))
  const patterns = [/\n[\t ]*\n+/gu, /[.!?…]+(?:["'»”’\])}]*)[\t ]+/gu, /\s+/gu]
  for (const pattern of patterns) {
    let selected = 0
    for (const match of window.matchAll(pattern)) {
      const end = (match.index ?? 0) + match[0].length
      if (end >= minimum && end < window.length) selected = end
    }
    if (selected) return selected
  }
  return window.length
}

/** Lossless, deterministic chunks. Concatenating chunk.text reproduces input. */
export function chunkCanonicalText(input, maximumCharacters = MAX_CHUNK_CHARACTERS) {
  const text = String(input ?? '').replace(/\r\n?/g, '\n').normalize('NFC')
  if (!text.trim()) return []
  const chunks = []
  let start = 0
  while (start < text.length) {
    const hardEnd = Math.min(text.length, start + maximumCharacters)
    const local = text.slice(start, hardEnd)
    const relativeCut = hardEnd === text.length ? local.length : preferredCut(local)
    const end = safeCodeUnitBoundary(text, start + relativeCut)
    chunks.push({ index: chunks.length, start, end, text: text.slice(start, end) })
    start = end
  }
  if (chunks.map((chunk) => chunk.text).join('') !== text) {
    throw new Error('Korpus parçalaması kayıpsızlık denetimini geçemedi.')
  }
  return chunks
}

/**
 * Each shard contains at most one chunk from an evidence lineage per round.
 * This keeps the exact synthesizer bounded and prevents one book or one
 * synthetic campaign/model lineage from looking like many witnesses.
 */
export function createDiverseSynthesisShards(analyses, maximumShardSize = 8) {
  const bySource = new Map()
  for (const analysis of analyses) {
    const sourceGroup = analysis.independenceGroupId || analysis.source.id
    const values = bySource.get(sourceGroup) ?? []
    values.push(analysis)
    bySource.set(sourceGroup, values)
  }
  for (const values of bySource.values()) {
    values.sort((left, right) => left.runId.localeCompare(right.runId))
  }
  const sourceIds = [...bySource.keys()].sort()
  const shards = []
  let round = 0
  while ([...bySource.values()].some((values) => values.length)) {
    const shard = []
    for (let offset = 0; offset < sourceIds.length && shard.length < maximumShardSize; offset += 1) {
      const sourceId = sourceIds[(round + offset) % sourceIds.length]
      const next = bySource.get(sourceId)?.shift()
      if (next) shard.push(next)
    }
    // If the number of sources is small, continue filling from them while
    // retaining explicit physical-source identity in the synthesis engine.
    for (const sourceId of sourceIds) {
      while (shard.length < maximumShardSize && (bySource.get(sourceId)?.length ?? 0) > 0) {
        shard.push(bySource.get(sourceId).shift())
      }
    }
    if (!shard.length) break
    shards.push(shard)
    round += 1
  }
  return shards
}

export const CORPUS_RUNTIME_LIMITS = Object.freeze({
  maximumChunkCharacters: MAX_CHUNK_CHARACTERS,
  maximumSynthesisShardAnalyses: 8,
})
