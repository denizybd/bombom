import assert from 'node:assert/strict'
import test from 'node:test'
import {
  chunkCanonicalText,
  createDiverseSynthesisShards,
} from '../corpus.mjs'

test('korpus parçalaması hiçbir karakteri kaybetmez', () => {
  const source = `${'Bir cümle. '.repeat(400)}A\u0308🙂${'x'.repeat(2_800)}`
  const canonical = source.normalize('NFC')
  const chunks = chunkCanonicalText(source, 240)
  assert.ok(chunks.length > 3)
  assert.equal(chunks.map((chunk) => chunk.text).join(''), canonical)
  assert.equal(chunks[0].start, 0)
  assert.equal(chunks.at(-1).end, canonical.length)
  for (let index = 1; index < chunks.length; index += 1) {
    assert.equal(chunks[index - 1].end, chunks[index].start)
  }
})

test('sentez shardları bütün analizleri tam bir kez taşır', () => {
  const analyses = ['A', 'A', 'A', 'B', 'B', 'C'].map((sourceId, index) => ({
    runId: `run-${index}`,
    source: { id: sourceId },
  }))
  const shards = createDiverseSynthesisShards(analyses, 3)
  assert.deepEqual(
    shards.flat().map((analysis) => analysis.runId).sort(),
    analyses.map((analysis) => analysis.runId).sort(),
  )
  assert.ok(shards.every((shard) => shard.length <= 3))
})

test('sentez shardları aynı üretim soyunu kaynak çeşitliliği gibi çoğaltmaz', () => {
  const analyses = [
    { runId: 'ollama-a', independenceGroupId: 'ollama:qwen14b', source: { id: 'lesson-a' } },
    { runId: 'ollama-b', independenceGroupId: 'ollama:qwen14b', source: { id: 'lesson-b' } },
    { runId: 'human-a', independenceGroupId: 'human:book-a', source: { id: 'book-a' } },
    { runId: 'human-b', independenceGroupId: 'human:book-b', source: { id: 'book-b' } },
  ]
  const shards = createDiverseSynthesisShards(analyses, 3)

  assert.deepEqual(
    shards.flat().map((analysis) => analysis.runId).sort(),
    analyses.map((analysis) => analysis.runId).sort(),
  )
  assert.equal(shards[0].filter((analysis) => analysis.independenceGroupId === 'ollama:qwen14b').length, 1)
})
