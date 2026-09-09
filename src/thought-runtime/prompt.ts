import { DEFAULT_TEXT_LAB_CONFIG, createMatrixLayout } from '../text-lab/domain/config'
import { encodeIdentity, projectIdentityToRelation } from '../text-lab/engine/encode'
import { analyzeText } from '../text-lab/engine/pipeline'
import { createContextSpaceId, createIdentitySpaceId, createRepresentationSpaceId } from '../text-lab/engine/representation'
import { createThoughtNeuron } from './model'
import { thoughtDigest } from './deterministic'
import { projectThoughtVector } from './math'
import {
  THOUGHT_MATRIX_CELLS,
  THOUGHT_MATRIX_SIDE,
  type PromptEncodingOptions,
  type ThoughtFamilyMembership,
  type ThoughtNeuron,
  type ThoughtPromptEncoding,
} from './types'

function promptConfig(options: PromptEncodingOptions = {}) {
  return {
    ...DEFAULT_TEXT_LAB_CONFIG,
    ...(options.representationProfile
      ? { representationProfile: options.representationProfile }
      : {}),
    matrixSide: THOUGHT_MATRIX_SIDE,
    identityRatio: options.identityRatio ?? DEFAULT_TEXT_LAB_CONFIG.identityRatio,
    minimumLeafGraphemes: options.minimumLeafGraphemes ?? DEFAULT_TEXT_LAB_CONFIG.minimumLeafGraphemes,
    hashSeed: options.hashSeed ?? DEFAULT_TEXT_LAB_CONFIG.hashSeed,
  }
}

export function canonicalizeThoughtPrompt(value: string): string {
  return String(value ?? '').replace(/\r\n?/g, '\n').normalize('NFC')
}

function flatFallback(text: string, options: PromptEncodingOptions, reason: string): ThoughtPromptEncoding {
  const config = promptConfig(options)
  const layout = createMatrixLayout(config)
  const identity = encodeIdentity(text, layout, config)
  const relation = projectIdentityToRelation(identity, layout, config)
  const representationSpaceId = createRepresentationSpaceId(
    createIdentitySpaceId(config, layout),
    createContextSpaceId(config, layout),
  )
  const signal = new Float32Array(THOUGHT_MATRIX_CELLS)
  signal.set(identity, 0)
  signal.set(relation, layout.relationStart)
  return {
    signal,
    temporaryNeurons: [{
      id: 'prompt-flat-fallback',
      parentId: null,
      depth: 0,
      text,
      weight: 1,
      identityDigest: thoughtDigest('prompt-flat-identity', identity),
    }],
    mode: 'flat-fallback',
    fallbackReason: reason,
    identityLength: layout.identityLength,
    representationSpaceId,
    analysisFingerprint: thoughtDigest('prompt-flat', text, config, reason),
  }
}

/**
 * Builds temporary hierarchical TextLab neurons and deterministically pools their
 * I/R representations into S0. A flat encoder is used only when explicitly
 * allowed and its reason is retained in the returned diagnostics.
 */
export async function encodeThoughtPrompt(
  value: string,
  options: PromptEncodingOptions = {},
): Promise<ThoughtPromptEncoding> {
  const text = canonicalizeThoughtPrompt(value)
  if (!text.trim()) throw new RangeError('Düşünce sinyali için boş olmayan bir metin gerekir.')
  const config = promptConfig(options)
  try {
    const analysis = await analyzeText({ id: 'thought-prompt', title: 'Thought prompt', rawText: text }, config)
    const rootIds = new Set(analysis.rootIds)
    const rawWeights = analysis.nodes.map((node) =>
      (rootIds.has(node.id) ? 2 : 1) * Math.sqrt(Math.max(1, node.graphemeCount)) / (1 + node.depth))
    const totalWeight = rawWeights.reduce((sum, weight) => sum + weight, 0)
    const pooled = new Float32Array(THOUGHT_MATRIX_CELLS)
    const temporaryNeurons = analysis.nodes.map((node, index) => {
      const weight = rawWeights[index] / totalWeight
      for (let coordinate = 0; coordinate < node.identity.length; coordinate += 1) {
        pooled[coordinate] += node.identity[coordinate] * weight
      }
      for (let coordinate = 0; coordinate < node.relation.length; coordinate += 1) {
        pooled[analysis.layout.relationStart + coordinate] += node.relation[coordinate] * weight
      }
      return {
        id: node.id,
        parentId: node.parentId,
        depth: node.depth,
        text: node.text,
        weight,
        identityDigest: node.identityDigest,
      }
    })
    return {
      signal: projectThoughtVector(pooled, 1),
      temporaryNeurons,
      mode: 'textlab-hierarchy',
      fallbackReason: null,
      identityLength: analysis.layout.identityLength,
      representationSpaceId: analysis.representationSpaceId,
      analysisFingerprint: thoughtDigest(
        'prompt-hierarchy',
        analysis.representationSpaceId,
        analysis.sourceDigest,
        temporaryNeurons.map((node) => [node.id, node.parentId, node.weight, node.identityDigest]),
      ),
    }
  } catch (error) {
    if (!options.allowFlatFallback) throw error
    const reason = error instanceof Error ? `${error.name}: ${error.message}` : String(error)
    return flatFallback(text, options, reason)
  }
}

export async function createPromptThoughtNeuron(input: {
  id: string
  text: string
  sourceId?: string
  familyMemberships?: readonly ThoughtFamilyMembership[]
  encoding?: PromptEncodingOptions
}): Promise<ThoughtNeuron> {
  const encoded = await encodeThoughtPrompt(input.text, input.encoding)
  const layout = createMatrixLayout(promptConfig(input.encoding))
  return createThoughtNeuron({
    id: input.id,
    identity: encoded.signal.slice(0, layout.identityLength),
    relation: encoded.signal.slice(layout.relationStart),
    familyMemberships: input.familyMemberships,
    provenance: {
      kind: 'textlab-surface',
      sourceId: input.sourceId,
      surface: canonicalizeThoughtPrompt(input.text),
    },
  }, {
    side: THOUGHT_MATRIX_SIDE,
    cellCount: THOUGHT_MATRIX_CELLS,
    identityLength: layout.identityLength,
    relationStart: layout.relationStart,
    relationLength: layout.relationLength,
  })
}
