import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer as createViteServer } from 'vite'
import { defaultConfig } from '../../server/config.mjs'
import { createTrainingStore } from '../../server/store.mjs'
import { temporaryDirectory } from '../../server/test/helpers.mjs'

test('core runtime keeps legacy selection unchanged and requires explicit canonical plastic selection', async (t) => {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const runtime = await vite.ssrLoadModule('/runtime/core-runtime.mjs')
  const textLab = await vite.ssrLoadModule('/src/text-lab/engine/index.ts')

  const legacyConfig = defaultConfig('/tmp/core-runtime-plastic-selection')
  const legacy = runtime.resolveRuntimeTextLabSelection([], legacyConfig)
  assert.equal(textLab.resolveTextLabRepresentationProfileId(legacy.config), 'legacy-quarter-id-v2')
  assert.equal(legacy.plasticRegionRequest.requested, false)
  assert.equal(legacy.reencodePlan, null)

  const adaptiveOffConfig = defaultConfig('/tmp/core-runtime-plastic-selection')
  adaptiveOffConfig.plasticRegions = {
    ...adaptiveOffConfig.plasticRegions,
    mode: 'off',
    representationProfile: 'adaptive-eighth-id-v1',
  }
  const adaptiveOff = runtime.resolveRuntimeTextLabSelection([{
    id: 'adaptive-off-source', contentHash: 'adaptive-off-hash', provenance: {},
  }], adaptiveOffConfig)
  assert.equal(adaptiveOff.plasticRegionRequest.requested, false)
  assert.equal(textLab.resolveTextLabRepresentationProfileId(adaptiveOff.config), 'adaptive-eighth-id-v1')
  assert.equal(adaptiveOff.reencodePlan.rows[0].status, 'encode-required')

  const adaptiveConfig = defaultConfig('/tmp/core-runtime-plastic-selection')
  adaptiveConfig.plasticRegions = {
    ...adaptiveConfig.plasticRegions,
    mode: 'shadow',
    representationProfile: 'adaptive-eighth-id-v1',
  }
  const fresh = runtime.resolveRuntimeTextLabSelection([{
    id: 'plain-source', contentHash: 'plain-hash', provenance: {},
  }], adaptiveConfig)
  assert.equal(textLab.resolveTextLabRepresentationProfileId(fresh.config), 'adaptive-eighth-id-v1')
  assert.equal(fresh.config.identityRatio, 0.125)
  assert.equal(fresh.reencodePlan.rows[0].status, 'encode-required')
  assert.equal(fresh.reencodePlan.willMutate, false)

  const manifestConfig = textLab.createTextLabConfigForProfile('adaptive-eighth-id-v1', {
    minimumLeafGraphemes: 7,
  })
  const imported = runtime.resolveRuntimeTextLabSelection([{
    id: 'imported-source',
    contentHash: 'imported-hash',
    provenance: {
      textLabManifest: {
        format: 'text-lab-recompute-proof-v1',
        config: manifestConfig,
      },
    },
  }], adaptiveConfig)
  assert.deepEqual(imported.config, manifestConfig)
  assert.equal(imported.reencodePlan.rows[0].status, 'reuse-verified-manifest')

  assert.throws(
    () => runtime.resolveRuntimeTextLabSelection([{
      id: 'legacy-import',
      contentHash: 'legacy-hash',
      provenance: {
        textLabManifest: {
          format: 'text-lab-recompute-proof-v1',
          config: { ...textLab.DEFAULT_TEXT_LAB_CONFIG },
        },
      },
    }], adaptiveConfig),
    (error) => error?.code === 'REPRESENTATION_SPACE_MISMATCH',
  )
})

test('plastic thought source selection excludes evaluation data and stratifies repeated Ollama lineage samples', async (t) => {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const { selectPlasticRegionThoughtSources } = await vite.ssrLoadModule('/runtime/core-runtime.mjs')
  const sources = [
    { sourceId: 'ollama-a', sourceDigest: 'a', independenceGroupId: 'ollama-lineage:qwen', split: 'train', text: 'A', provenance: { kind: 'ollama' } },
    { sourceId: 'ollama-b', sourceDigest: 'b', independenceGroupId: 'ollama-lineage:qwen', split: 'train', text: 'B', provenance: { kind: 'ollama' } },
    { sourceId: 'human-a', sourceDigest: 'c', independenceGroupId: 'book-a', split: 'train', text: 'C', provenance: { kind: 'human' } },
    { sourceId: 'validation-a', sourceDigest: 'd', independenceGroupId: 'book-v', split: 'validation', text: 'D', provenance: { kind: 'human' } },
    { sourceId: 'test-a', sourceDigest: 'e', independenceGroupId: 'book-t', split: 'test', text: 'E', provenance: { kind: 'human' } },
  ]
  const selected = selectPlasticRegionThoughtSources({ sources }, 10)
  assert.equal(selected.length, 3)
  assert.deepEqual(new Set(selected.map((source) => source.independenceGroupId)), new Set([
    'ollama-lineage:qwen', 'book-a',
  ]))
  assert.ok(selected.every((source) => source.split === 'train'))
  assert.equal(selected.filter((source) => source.independenceGroupId === 'ollama-lineage:qwen').length, 2)
  assert.equal(selectPlasticRegionThoughtSources({ sources: [...sources].reverse() }, 10)
    .map((source) => source.sourceId).join(','), selected.map((source) => source.sourceId).join(','))
  assert.deepEqual(selectPlasticRegionThoughtSources({ sources }, 0), [])
})

test('real opt-in pipeline stage derives immutable observations from adaptive thought ticks without activation', async (t) => {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const runtime = await vite.ssrLoadModule('/runtime/core-runtime.mjs')
  const textLab = await vite.ssrLoadModule('/src/text-lab/engine/index.ts')
  const config = defaultConfig('/tmp/core-runtime-plastic-stage')
  config.plasticRegions = {
    ...config.plasticRegions,
    mode: 'shadow',
    representationProfile: 'adaptive-eighth-id-v1',
    thoughtSamplesPerCycle: 1,
    thoughtTicksPerSample: 2,
  }
  const textLabConfig = textLab.createTextLabConfigForProfile('adaptive-eighth-id-v1')
  const analysis = await textLab.analyzeText({
    id: 'adaptive-stage-source',
    title: 'Adaptive stage source',
    rawText: 'Ali ekmeği kesti. Ayşe kapıyı dikkatlice açtı.',
  }, textLabConfig)
  const nodes = analysis.nodes.slice(0, Math.min(12, analysis.nodes.length))
  assert.ok(nodes.length >= 2)
  const resolved = nodes.map((node) => ({
    memberKey: `member:${node.id}`,
    kind: 'raw',
    text: node.text,
    identity: new Float32Array(node.identity),
    relation: new Float32Array(node.relation),
    identityDigest: node.identityDigest,
    sourceDigest: analysis.sourceDigest,
    analysisRunId: analysis.runId,
    neuronId: node.id,
  }))
  const familyRun = {
    format: 'noron-aileleri-run',
    version: 1,
    id: 'family-run-adaptive-stage',
    fingerprint: 'family-run-adaptive-stage-fingerprint',
    representationSpaceId: analysis.representationSpaceId,
    families: [{ id: 'family-content', region: 'content' }],
    profiles: resolved.map((entry) => ({ memberKey: entry.memberKey, kind: 'raw' })),
    memberships: resolved.map((entry) => ({
      memberKey: entry.memberKey, familyId: 'family-content', weight: 1, rank: 1,
    })),
    neuronTransitions: resolved.map((entry, index) => ({
      fromMemberKey: entry.memberKey,
      toMemberKey: resolved[(index + 1) % resolved.length].memberKey,
      probability: 1,
    })),
    familyTransitions: [],
    routingPreview: null,
    coalition: { tick: 0, activeFamilyCount: 0, coalitions: [], unmatchedReason: null },
  }
  const languageState = {
    format: 'language-experience-state',
    version: 1,
    fingerprint: 'language-state-adaptive-stage',
    familyRunFingerprint: familyRun.fingerprint,
    representationSpaceId: familyRun.representationSpaceId,
    neuronTransitions: [],
    familyTransitions: [],
  }
  const store = await createTrainingStore({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => store.close())
  const controller = new AbortController()
  const state = {
    runtimeStartedAt: Date.now(),
    artifactIds: [],
    inputManifestHash: 'adaptive-stage-manifest',
    representationSpaceId: analysis.representationSpaceId,
    textLabConfig,
    plasticRegionRequest: { requested: true, mode: 'shadow', profile: 'adaptive-eighth-id-v1' },
    plasticRegionReencodePlan: { fingerprint: 'reencode-plan', counts: { 'encode-required': 1 } },
    familyRun,
    trainingResult: { state: languageState },
    thoughtObservationNeurons: resolved,
    dataset: {
      sources: [{
        sourceId: 'adaptive-stage-source',
        sourceDigest: analysis.sourceDigest,
        independenceGroupId: 'human-book-stage',
        split: 'train',
        text: analysis.source.text,
        provenance: { kind: 'human' },
      }],
    },
  }
  const result = await runtime.REAL_STAGE_HANDLERS.plasticRegions({}, {
    store,
    config,
    jobId: null,
    signal: controller.signal,
    checkpoint: async () => {},
    progress: async () => {},
    emit: async () => {},
  }, state)
  assert.equal(result.verified, true)
  assert.equal(result.mode, 'shadow')
  assert.equal(result.safetyMode, 'trace-only-no-active-base')
  assert.equal(result.atlasActivated, false)
  assert.ok(result.observationCount >= 56)
  assert.ok(result.observationCount % 56 === 0)
  assert.ok(state.artifactIds.length >= 3)
  assert.ok(!state.artifactIds.some((id) => store.getArtifactMetadata(id).kind === 'plastic-region-atlas-head'))
  assert.equal(store.getActiveSnapshot(), undefined)
})
