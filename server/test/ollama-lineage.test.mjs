import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultConfig } from '../config.mjs'
import { canonicalOllamaIndependenceGroupId } from '../ollama-lineage.mjs'
import { generateTeacherSources } from '../pipeline.mjs'
import { createTrainingStore } from '../store.mjs'
import { sourceFromLesson } from '../teacher-feed.mjs'
import { temporaryDirectory } from './helpers.mjs'

test('canonical Ollama lineage is conservative across metadata and incomplete identity', () => {
  const named = canonicalOllamaIndependenceGroupId({
    resolvedModelName: ' QWEN2.5-CODER:14B ',
    weightsDigest: `sha256:${'a'.repeat(64)}`,
    campaignId: 'campaign-a',
    catalogFingerprint: 'catalog-a',
    promptHash: 'prompt-a',
  })
  const sameNamed = canonicalOllamaIndependenceGroupId({
    resolvedModelName: 'qwen2.5-coder:14b',
    weightsDigest: `sha256:${'b'.repeat(64)}`,
    campaignId: 'campaign-b',
    catalogFingerprint: 'catalog-b',
    promptHash: 'prompt-b',
  })
  assert.equal(named, sameNamed, 'catalog, campaign, prompt and a path-local digest must not manufacture independence')
  assert.match(named, /^ollama-lineage:[a-f0-9]{24}$/u)

  const digestOnly = canonicalOllamaIndependenceGroupId({ weightsDigest: `sha256:${'C'.repeat(64)}` })
  assert.equal(digestOnly, canonicalOllamaIndependenceGroupId({ weightsDigest: 'c'.repeat(64) }))
  assert.notEqual(digestOnly, named)

  assert.equal(
    canonicalOllamaIndependenceGroupId(),
    canonicalOllamaIndependenceGroupId({ resolvedModelName: 'unknown', campaignId: 'untrusted-campaign' }),
    'unknown Ollama observations must collapse to one provider group',
  )
})

test('generic generation and teacher feed assign the same resolved Ollama model lineage', async (t) => {
  const dataDirectory = await temporaryDirectory(t)
  const store = await createTrainingStore({ dataDirectory, forceJson: true })
  t.after(() => store.close())
  store.updateConfig({
    ollama: { model: 'qwen2.5-coder:14b' },
    budgets: { maxDailyTeacherRequests: 10, maxDailyTeacherTokens: 100_000 },
  })
  const ollama = {
    async generateStructured() {
      return {
        value: { lessons: [{ text: 'Ayşe kitabı dikkatlice açtı.', kind: 'example', rationale: 'temel eylem' }] },
        model: 'qwen2.5-coder:14b',
        promptTokens: 8,
        completionTokens: 6,
      }
    },
  }
  await generateTeacherSources({ topic: 'temel eylem', count: 1 }, {
    store,
    ollama,
    signal: new AbortController().signal,
  })
  const genericSource = store.listSources()[0]

  const feedSource = sourceFromLesson({
    lesson: {
      text: 'Ali kapıyı yavaşça açtı.',
      resolvedModel: 'QWEN2.5-CODER:14B',
      attemptSeed: 42,
      skill: 'canonical-transitive',
      form: 'affirmative',
      wordCount: 4,
      qualityScore: 0.95,
    },
    campaign: {
      id: 'unrelated-campaign',
      seed: 7,
      curriculum: 'unrelated-catalog-version',
      protocolVersion: 999,
      protocolFingerprint: 'unrelated-prompt-protocol',
      protocol: { spellingPolicyId: 'test-spelling' },
      modelIdentity: {
        name: 'qwen2.5-coder:14b',
        digest: `sha256:${'d'.repeat(64)}`,
        fingerprint: 'path-specific-model-fingerprint',
      },
    },
    plan: {
      batchIndex: 91,
      batchId: 'different-batch',
      unit: { title: 'Başka katalog', instruction: 'Başka istem' },
      axes: { domain: 'other', tense: 'past', length: 'short', wordOrder: 'sov' },
    },
    config: defaultConfig(dataDirectory),
  })

  assert.equal(genericSource.provenance.independenceGroupId, feedSource.provenance.independenceGroupId)
  assert.equal(
    genericSource.provenance.independenceGroupId,
    canonicalOllamaIndependenceGroupId({ resolvedModelName: 'qwen2.5-coder:14b' }),
  )
})
