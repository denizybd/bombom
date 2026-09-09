import { createHash } from 'node:crypto'

const FLOAT32_TAG = 'bombom/float32-base64-v1'

export function stableRuntimeHash(...parts) {
  const hash = createHash('sha256')
  for (const part of parts) {
    const value = typeof part === 'string' ? part : JSON.stringify(part)
    hash.update(String(Buffer.byteLength(value)))
    hash.update(':')
    hash.update(value)
    hash.update('\n')
  }
  return hash.digest('hex')
}

/**
 * SQLite stores artifacts as JSON. Typed arrays are explicitly encoded so a
 * checkpoint can be restored byte-for-byte instead of becoming an object with
 * numeric property names after JSON.stringify.
 */
export function encodeArtifactValue(value, seen = new WeakSet()) {
  if (value instanceof Float32Array) {
    const bytes = Buffer.from(value.buffer, value.byteOffset, value.byteLength)
    return { $type: FLOAT32_TAG, length: value.length, data: bytes.toString('base64') }
  }
  if (Array.isArray(value)) return value.map((item) => encodeArtifactValue(item, seen))
  if (value && typeof value === 'object') {
    if (seen.has(value)) throw new TypeError('Döngüsel nesne kalıcı eğitim eserine yazılamaz.')
    seen.add(value)
    const output = {}
    for (const [key, item] of Object.entries(value)) {
      if (item !== undefined) output[key] = encodeArtifactValue(item, seen)
    }
    seen.delete(value)
    return output
  }
  return value
}

export function decodeArtifactValue(value) {
  if (Array.isArray(value)) return value.map(decodeArtifactValue)
  if (value && typeof value === 'object') {
    if (value.$type === FLOAT32_TAG) {
      if (!Number.isInteger(value.length) || value.length < 0 || typeof value.data !== 'string') {
        throw new TypeError('Float32 kontrol noktası bozuk.')
      }
      const bytes = Buffer.from(value.data, 'base64')
      if (bytes.byteLength !== value.length * Float32Array.BYTES_PER_ELEMENT) {
        throw new TypeError('Float32 kontrol noktası uzunluğu doğrulanamadı.')
      }
      // Copy to an exactly aligned ArrayBuffer before constructing the view.
      const copy = Uint8Array.from(bytes)
      return new Float32Array(copy.buffer, copy.byteOffset, value.length).slice()
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, decodeArtifactValue(item)]))
  }
  return value
}

export function encodedByteLength(value) {
  return Buffer.byteLength(JSON.stringify(value), 'utf8')
}

export function findCachedArtifact(store, kind, contentHash, { repairIndex = true } = {}) {
  if (!contentHash) return undefined
  if (typeof store.findArtifactByContentHash === 'function') {
    return store.findArtifactByContentHash(kind, contentHash, { repairIndex })
  }
  return undefined
}

export function persistArtifact(context, { kind, contentHash, value, metadata = {} }) {
  context.checkResources?.(`artifact-write:${kind}`)
  if (context.jobClaim) {
    const assertClaim = context.store.assertRunnableJobClaim ?? context.store.assertJobClaim
    if (typeof assertClaim !== 'function') {
      const error = new Error('Claim korumalı runtime store doğrulaması eksik.')
      error.code = 'STORE_CLAIM_CONTRACT_MISSING'
      throw error
    }
    assertClaim.call(context.store, context.jobId, context.jobClaim.owner, context.jobClaim.attempt)
  }
  const cached = findCachedArtifact(context.store, kind, contentHash, {
    repairIndex: !context.jobClaim,
  })
  if (cached) return { artifact: cached, value: decodeArtifactValue(cached.payload?.value) }

  const encoded = encodeArtifactValue(value)
  const bytes = encodedByteLength(encoded)
  const diskBudget = Number(context.store.getConfig().budgets.diskMb) * 1024 * 1024
  if (bytes > diskBudget) {
    const error = new Error(`${kind} kontrol noktası tek başına disk bütçesini aşıyor.`)
    error.code = 'DISK_BUDGET_EXCEEDED'
    throw error
  }
  const input = {
    jobId: context.jobId,
    kind,
    contentHash,
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      bytes,
      metadata,
      value: encoded,
    },
  }
  let artifact
  if (context.jobClaim) {
    if (typeof context.store.createClaimedArtifact !== 'function') {
      const error = new Error('Claim korumalı runtime artifact primitive’i eksik.')
      error.code = 'STORE_CLAIM_CONTRACT_MISSING'
      throw error
    }
    artifact = context.store.createClaimedArtifact(
      context.jobId,
      context.jobClaim.owner,
      context.jobClaim.attempt,
      input,
    )
  } else {
    artifact = context.store.createArtifact(input)
  }
  return { artifact, value }
}

export function restoreArtifact(artifact, expectedKind) {
  if (!artifact || artifact.kind !== expectedKind || artifact.payload?.format !== 'bombom-runtime-artifact') {
    throw new TypeError(`${expectedKind} kontrol noktası eksik veya uyumsuz.`)
  }
  return decodeArtifactValue(artifact.payload.value)
}
