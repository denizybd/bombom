import assert from 'node:assert/strict'
import test from 'node:test'
import { aggregateWorkerMemoryUsage } from '../worker-resources.mjs'

test('isolated worker meters parent and child RSS under one service envelope', () => {
  const sample = aggregateWorkerMemoryUsage({ rss: 900, heapUsed: 400, external: 50 }, 300)
  assert.deepEqual(sample, { rss: 1_200, heapUsed: 400, external: 50 })
  assert.equal(Object.isFrozen(sample), true)
})

test('invalid parent samples cannot subtract from worker RSS', () => {
  assert.equal(aggregateWorkerMemoryUsage({ rss: 900 }, -500).rss, 900)
  assert.equal(aggregateWorkerMemoryUsage({ rss: 900 }, Number.NaN).rss, 900)
})
