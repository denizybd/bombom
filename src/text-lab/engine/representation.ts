import type { MatrixLayout, TextLabConfig } from '../domain/types'
import { hashInteger, hashString } from './hash'

export const IDENTITY_ENCODER_VERSION = 'surface-ngram-v2'
export const CONTEXT_ENCODER_VERSION = 'structural-context-v2'
export const CANONICALIZATION_VERSION = 'unicode-nfc-lf-v1'

function hex(value: number): string {
  return value.toString(16).padStart(8, '0')
}

export function digestText(value: string): string {
  return `${hex(hashString(value, 0x243f6a88))}${hex(hashString(value, 0x9e3779b9))}`
}

export function digestVector(vector: ArrayLike<number>): string {
  const buffer = new ArrayBuffer(4)
  const view = new DataView(buffer)
  let first = 0x811c9dc5
  let second = 0x9e3779b9
  for (let index = 0; index < vector.length; index += 1) {
    view.setFloat32(0, vector[index], true)
    const bits = view.getUint32(0, true)
    first = hashInteger(bits ^ index, first)
    second = hashInteger(bits ^ Math.imul(index + 1, 0x85ebca6b), second)
  }
  return `${hex(first)}${hex(second)}`
}

function fingerprint(label: string, payload: unknown): string {
  return `${label}-${digestText(JSON.stringify(payload)).toUpperCase()}`
}

export function createIdentitySpaceId(config: TextLabConfig, layout: MatrixLayout): string {
  return fingerprint('IS', {
    canonicalization: CANONICALIZATION_VERSION,
    encoder: IDENTITY_ENCODER_VERSION,
    side: layout.side,
    identityLength: layout.identityLength,
    hashSeed: config.hashSeed,
  })
}

export function createContextSpaceId(config: TextLabConfig, layout: MatrixLayout): string {
  return fingerprint('CS', {
    encoder: CONTEXT_ENCODER_VERSION,
    relationLength: layout.relationLength,
    contextRetention: config.contextRetention,
    parentTraceWeight: config.parentTraceWeight,
    siblingTraceWeight: config.siblingTraceWeight,
    minimumTrace: config.minimumTrace,
  })
}

export function createRepresentationSpaceId(
  identitySpaceId: string,
  contextSpaceId: string,
): string {
  return fingerprint('RS', { identitySpaceId, contextSpaceId })
}
