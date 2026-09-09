import assert from 'node:assert/strict'
import test from 'node:test'
import { CONFIG_VERSION, defaultConfig, mergeAndValidateConfig } from '../config.mjs'
import { MEBIBYTE, ResourceGovernor, evaluateResourceState } from '../resource-policy.mjs'

function sample({ rssMb = 1_000, hostFreeMb = 8_000, diskFreeMb = 40_000 } = {}) {
  return {
    rssBytes: rssMb * MEBIBYTE,
    heapUsedBytes: 0,
    externalBytes: 0,
    hostTotalMemoryBytes: 16_384 * MEBIBYTE,
    hostFreeMemoryBytes: hostFreeMb * MEBIBYTE,
    diskTotalBytes: 256_000 * MEBIBYTE,
    diskFreeBytes: diskFreeMb * MEBIBYTE,
  }
}

test('resource policy exposes deterministic 70/85/100 RAM tiers', () => {
  const config = defaultConfig('/tmp/no-write')
  assert.equal(evaluateResourceState(config, sample({ rssMb: 2_000 })).memoryTier, 'normal')
  assert.equal(evaluateResourceState(config, sample({ rssMb: 3_000 })).memoryTier, 'soft')
  assert.equal(evaluateResourceState(config, sample({ rssMb: 3_600 })).memoryTier, 'spill')
  const hard = evaluateResourceState(config, sample({ rssMb: 4_096 }))
  assert.equal(hard.memoryTier, 'hard')
  assert.equal(hard.mustPause, true)
})

test('start admission preserves physical SSD reserve but running recovery remains possible', () => {
  const config = defaultConfig('/tmp/no-write')
  const constrained = sample({ diskFreeMb: 15_360 + config.resourcePolicy.walReserveMb - 1 })
  const admission = evaluateResourceState(config, constrained, { admission: true })
  const running = evaluateResourceState(config, constrained, { admission: false })
  assert.equal(admission.canStart, false)
  assert.equal(admission.allowed, false)
  assert.equal(running.allowed, true)
  assert.equal(running.mustPause, false)
})

test('logical SSD hard limit pauses without scanning artifacts', () => {
  const config = defaultConfig('/tmp/no-write')
  const state = evaluateResourceState(config, sample(), {
    logicalStorageBytes: config.budgets.diskMb * MEBIBYTE,
  })
  assert.equal(state.diskTier, 'hard')
  assert.equal(state.allowed, false)
  assert.equal(state.reasonCodes.includes('LOGICAL_DISK_HARD_LIMIT'), true)
})

test('governor caches host sampling for the configured interval', () => {
  const config = defaultConfig('/tmp/no-write')
  let calls = 0
  let now = 1_000
  const store = { getConfig: () => config, approximateStorageBytes: () => 0 }
  const governor = new ResourceGovernor({
    store,
    clock: () => now,
    sampler: () => { calls += 1; return sample() },
  })
  governor.assertSafe()
  governor.assertSafe()
  assert.equal(calls, 1)
  now += config.resourcePolicy.sampleIntervalMs
  governor.assertSafe()
  assert.equal(calls, 2)
})

test('v5 config keeps bounded RAM and validates concept lifecycle thresholds', () => {
  const config = defaultConfig('/tmp/no-write')
  assert.equal(config.version, CONFIG_VERSION)
  assert.equal(config.budgets.ramMb, 4_096)
  assert.equal(config.budgets.diskMb, 4_096)
  assert.equal(mergeAndValidateConfig(config, { budgets: { ramMb: 6_144 } }).budgets.ramMb, 6_144)
  assert.throws(() => mergeAndValidateConfig(config, { budgets: { ramMb: 8_193 } }), /8192/u)
  assert.throws(() => mergeAndValidateConfig(config, {
    resourcePolicy: { ramSoftRatio: 0.9, ramSpillRatio: 0.8 },
  }), /soft < spill < hard/u)
  assert.equal(config.conceptMacros.mode, 'proposal-shadow')
  assert.throws(() => mergeAndValidateConfig(config, {
    conceptMacros: { activationMinimumOccurrences: 2, proposalMinimumOccurrences: 3 },
  }), /Aktif makro gözlem eşiği/u)
  assert.throws(() => mergeAndValidateConfig(config, {
    conceptMacros: { minimumLength: 7, maximumLength: 3 },
  }), /minimumLength/u)
  assert.throws(() => mergeAndValidateConfig(config, {
    conceptMacros: { maximumNestedDepth: 4 },
  }), /3/u)
})
