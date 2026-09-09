import { createHash } from 'node:crypto'

const UNKNOWN_MODEL_SENTINELS = new Set(['', 'unknown', 'bilinmiyor', 'ollama', 'local'])

function normalizedModelName(value) {
  if (typeof value !== 'string') return ''
  const normalized = value
    .normalize('NFKC')
    .trim()
    .replace(/\s+/gu, ' ')
    .toLocaleLowerCase('en-US')
  return UNKNOWN_MODEL_SENTINELS.has(normalized) ? '' : normalized
}

function normalizedWeightsDigest(value) {
  if (typeof value !== 'string') return ''
  const normalized = value.normalize('NFKC').trim().toLocaleLowerCase('en-US')
  if (!normalized) return ''
  const sha256 = normalized.match(/^sha256:([a-f0-9]{64})$/u)?.[1]
  return sha256 ?? normalized
}

function lineageHash(identity) {
  return createHash('sha256')
    .update('bombom-ollama-independence-lineage-v2\0')
    .update(identity)
    .digest('hex')
    .slice(0, 24)
}

/**
 * Return one conservative evidence-independence group for an Ollama lineage.
 *
 * A resolved model name is the primary identity because both the generic
 * generator and the closed-choice teacher feed observe it. A weights digest is
 * used only when no resolved name exists; otherwise a digest that is missing on
 * one ingestion path must not split evidence from the same named model.
 * Campaign, catalog, seed and prompt metadata are intentionally not inputs.
 */
export function canonicalOllamaIndependenceGroupId({ resolvedModelName, weightsDigest } = {}) {
  const model = normalizedModelName(resolvedModelName)
  if (model) return `ollama-lineage:${lineageHash(`model\0${model}`)}`

  const digest = normalizedWeightsDigest(weightsDigest)
  if (digest) return `ollama-lineage:${lineageHash(`weights\0${digest}`)}`

  // Missing Ollama identity is uncertainty, not evidence of independence.
  return `ollama-lineage:${lineageHash('provider\0unknown')}`
}
