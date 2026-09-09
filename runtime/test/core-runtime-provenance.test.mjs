import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer as createViteServer } from 'vite'

test('runtime preserves the current teacher generator model with legacy-compatible fallbacks', async (t) => {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const { resolveGeneratorModel, resolveIndependenceGroupId } = await vite.ssrLoadModule('/runtime/core-runtime.mjs')

  assert.equal(resolveGeneratorModel({
    generationModel: 'qwen2.5-coder:14b',
    modelIdentity: 'model-fingerprint',
    modelDigest: 'sha256:model-digest',
    model: 'legacy-model',
  }), 'qwen2.5-coder:14b')

  assert.equal(resolveGeneratorModel({ modelIdentity: 'model-fingerprint', modelDigest: 'sha256:model-digest' }), 'model-fingerprint')
  assert.equal(resolveGeneratorModel({ modelDigest: 'sha256:model-digest' }), 'sha256:model-digest')
  assert.equal(resolveGeneratorModel({ model: 'legacy-teacher:14b' }), 'legacy-teacher:14b')
  assert.equal(resolveGeneratorModel({ generationModel: '   ', model: ' legacy-teacher:14b ' }), 'legacy-teacher:14b')
  assert.equal(resolveGeneratorModel(null), undefined)

  const canonicalDeclared = resolveIndependenceGroupId({
    id: 'lesson-a',
    sourceType: 'ollama-generated',
    provenance: {
      independenceGroupId: ' ollama-catalog:frozen ',
      generationModel: 'qwen2.5-coder:14b',
    },
  })
  const legacyA = resolveIndependenceGroupId({
    id: 'lesson-a',
    sourceType: 'ollama-generated',
    provenance: { generationModel: 'qwen2.5-coder:14b' },
  })
  const legacyB = resolveIndependenceGroupId({
    id: 'lesson-b',
    sourceType: 'ollama-generated',
    provenance: { generationModel: 'qwen2.5-coder:14b' },
  })
  assert.equal(canonicalDeclared, legacyA)
  assert.equal(legacyA, legacyB)
  assert.match(legacyA, /^ollama-lineage:/u)
  assert.equal(resolveIndependenceGroupId({
    id: 'lesson-c',
    sourceType: 'ollama-generated',
    provenance: {
      independenceGroupId: 'ollama-catalog:another-fake-witness',
      generationModel: ' QWEN2.5-CODER:14B ',
      campaignId: 'another-campaign',
    },
  }), legacyA)
  const humanBook = resolveIndependenceGroupId({
    id: 'human-book',
    contentHash: 'sha256:physical-book',
    title: 'İlk başlık',
    sourceType: 'manual-text',
    provenance: { independenceGroupId: 'forged-witness-a' },
  })
  assert.match(humanBook, /^source-lineage:/u)
  assert.equal(resolveIndependenceGroupId({
    id: 'renamed-copy',
    contentHash: 'SHA256:PHYSICAL-BOOK',
    title: 'Yeniden adlandırılmış aynı fiziksel içerik',
    sourceType: 'manual-text',
    provenance: { independenceGroupId: 'forged-witness-b' },
  }), humanBook, 'caller-provided group labels and titles cannot split one content lineage')
  assert.notEqual(resolveIndependenceGroupId({
    id: 'different-book',
    contentHash: 'sha256:different-book',
    sourceType: 'manual-text',
    provenance: { independenceGroupId: 'forged-witness-c' },
  }), humanBook)

  const fileLineage = resolveIndependenceGroupId({
    id: 'file-version-a',
    contentHash: 'sha256:file-v1',
    sourceType: 'local-file',
    locator: '/approved/books/a.txt',
    provenance: { canonicalPath: '/approved/books/a.txt', independenceGroupId: 'fake-file-a' },
  })
  assert.equal(resolveIndependenceGroupId({
    id: 'file-version-b',
    contentHash: 'sha256:file-v2',
    sourceType: 'local-file',
    locator: '/approved/books/a.txt',
    provenance: { canonicalPath: '/approved/books/a.txt', independenceGroupId: 'fake-file-b' },
  }), fileLineage, 'versions of one server-resolved physical file remain one witness')

  assert.match(resolveIndependenceGroupId({
    id: 'legacy-human-book', sourceType: 'manual-text', provenance: { independenceGroupId: 'ignored' },
  }), /^source-lineage:/u)
})
