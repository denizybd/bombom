import { describe, expect, it } from 'vitest'
import {
  DEFAULT_TEXT_LAB_CONFIG,
  analyzeText,
} from '../text-lab/engine'
// @ts-expect-error The production runtime is an ESM JavaScript boundary bundled by Vite.
import { REAL_STAGE_HANDLERS } from '../../runtime/core-runtime.mjs'

function memoryRuntimeStore() {
  const artifacts: Array<Record<string, any>> = []
  return {
    artifacts,
    getConfig: () => ({ budgets: { maxRunMinutes: 5, diskMb: 512 } }),
    getJob: () => ({ state: 'running' }),
    findArtifactByContentHash: (kind: string, contentHash: string) =>
      artifacts.find((artifact) => artifact.kind === kind && artifact.contentHash === contentHash),
    createArtifact: (input: Record<string, any>) => {
      const artifact = { id: `artifact-${artifacts.length + 1}`, createdAt: new Date().toISOString(), ...input }
      artifacts.push(artifact)
      return structuredClone(artifact)
    },
  }
}

async function fixture() {
  const config = {
    ...DEFAULT_TEXT_LAB_CONFIG,
    matrixSide: 8,
    minimumLeafGraphemes: 6,
  }
  const original = await analyzeText({
    id: 'browser-text-source',
    title: 'Tarayıcı arşivi',
    rawText: '  Birinci düşünce gerçekleşti.\r\n\r\nİkinci düşünce bağlam kurdu.  ',
  }, config)
  const manifest = {
    format: 'text-lab-recompute-proof-v1',
    originalSourceId: original.source.id,
    sourceDigest: original.sourceDigest,
    representationSpaceId: original.representationSpaceId,
    config: { ...original.config },
    identityProofs: original.nodes.map(({ start, end, identityDigest }) => ({ start, end, identityDigest })),
  }
  return { original, manifest }
}

async function runTextLab(manifest: Record<string, any>) {
  const { original } = await fixture()
  const store = memoryRuntimeStore()
  const state: Record<string, any> = {
    sources: [{
      id: 'service-source',
      title: 'İçe aktarılan arşiv',
      sourceType: 'manual-text',
      locator: null,
      content: original.source.text,
      contentHash: 'content-hash',
      bytes: original.source.text.length,
      provenance: { kind: 'text-lab-archive', textLabManifest: manifest },
    }],
    artifactIds: [],
    runtimeStartedAt: Date.now(),
  }
  const context = {
    signal: new AbortController().signal,
    store,
    jobId: 'manifest-test-job',
    checkpoint: async () => undefined,
    progress: async () => undefined,
  }
  const result = await REAL_STAGE_HANDLERS.textLab({}, context, state)
  return { result, state, store }
}

describe('TextLab archive recomputation proof', () => {
  it('recomputes the canonical text and verifies every frozen identity cell digest', async () => {
    const { manifest } = await fixture()
    const { result, state, store } = await runTextLab(manifest)

    expect(result.verified).toBe(true)
    expect(result.sourceCount).toBe(1)
    expect(result.chunkCount).toBe(1)
    expect(state.analyses[0].source.text).toBe('  Birinci düşünce gerçekleşti.\n\nİkinci düşünce bağlam kurdu.  ')
    expect(store.artifacts[0].payload.metadata.identityProofVerified).toBe(true)
  })

  it('rejects a syntactically valid but altered identity digest before synthesis', async () => {
    const { manifest } = await fixture()
    manifest.identityProofs[0] = {
      ...manifest.identityProofs[0],
      identityDigest: manifest.identityProofs[0].identityDigest === '0000000000000000'
        ? '1000000000000000'
        : '0000000000000000',
    }

    await expect(runTextLab(manifest)).rejects.toMatchObject({ code: 'TEXT_LAB_IDENTITY_MISMATCH' })
  })
})
