import assert from 'node:assert/strict'
import test from 'node:test'
import {
  decodeArtifactValue,
  encodeArtifactValue,
  stableRuntimeHash,
} from '../artifacts.mjs'

test('Float32Array kontrol noktası bit düzeyinde geri yüklenir', () => {
  const input = {
    identity: new Float32Array([0, -0, 0.125, -0.75, Number.MIN_VALUE]),
    nested: [new Float32Array([1, -1])],
  }
  const restored = decodeArtifactValue(JSON.parse(JSON.stringify(encodeArtifactValue(input))))
  assert.ok(restored.identity instanceof Float32Array)
  assert.deepEqual(
    [...new Uint8Array(restored.identity.buffer)],
    [...new Uint8Array(input.identity.buffer)],
  )
  assert.deepEqual([...restored.nested[0]], [1, -1])
})

test('çalışma parmak izi deterministik ve alan duyarlıdır', () => {
  assert.equal(stableRuntimeHash('a', { b: 1 }), stableRuntimeHash('a', { b: 1 }))
  assert.notEqual(stableRuntimeHash('ab', 'c'), stableRuntimeHash('a', 'bc'))
})
