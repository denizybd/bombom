import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultConfig, mergeAndValidateConfig } from '../config.mjs'
import { fromPublicConfig, toPublicConfig } from '../public-api.mjs'

test('plastic-region configuration is enabled but behavior-off on the legacy representation by default', () => {
  const config = defaultConfig('/tmp/plastic-region-config-test')
  assert.equal(config.plasticRegions.enabled, true)
  assert.equal(config.plasticRegions.mode, 'off')
  assert.equal(config.plasticRegions.representationProfile, 'legacy')
  assert.equal(config.plasticRegions.maximumRegions, 56)
  assert.equal(config.plasticRegions.maximumMembershipsPerPage, 3)
  assert.equal(config.plasticRegions.compatibilityBlend, 0.25)
  assert.equal(config.plasticRegions.minimumTrace, 0.02)
  assert.equal(config.plasticRegions.semanticProbesEnabled, false)
  assert.equal(toPublicConfig(config).plasticRegions.mode, 'off')
})

test('plastic-region configuration validates modes/caps and blocks any legacy coordinate reinterpretation', () => {
  const config = defaultConfig('/tmp/plastic-region-config-test')
  assert.throws(
    () => mergeAndValidateConfig(config, { plasticRegions: { mode: 'active' } }),
    /Legacy temsil profili/,
  )
  const adaptive = mergeAndValidateConfig(config, {
    plasticRegions: {
      representationProfile: 'adaptive-eighth-id-v1',
      mode: 'active',
      maximumRegions: 40,
      maximumActiveGates: 4,
      semanticProbesEnabled: true,
    },
  })
  assert.equal(adaptive.plasticRegions.mode, 'active')
  assert.equal(adaptive.plasticRegions.maximumRegions, 40)
  assert.equal(adaptive.plasticRegions.thoughtSamplesPerCycle, 4)
  assert.equal(adaptive.plasticRegions.thoughtTicksPerSample, 16)
  assert.equal(adaptive.plasticRegions.semanticProbesEnabled, true)
  assert.deepEqual(fromPublicConfig({
    ollamaBaseUrl: config.ollama.baseUrl,
    automationLevel: config.automation.level,
    autoRun: config.automation.autoRun,
    batchSize: config.automation.batchSize,
    budgets: config.budgets,
    plasticRegions: adaptive.plasticRegions,
  }).plasticRegions, adaptive.plasticRegions)
  assert.throws(
    () => mergeAndValidateConfig(config, { plasticRegions: { maximumMembershipsPerPage: 4 } }),
    /1 ile 3/,
  )
  assert.throws(
    () => mergeAndValidateConfig(config, { plasticRegions: { compatibilityBlend: 0 } }),
    /compatibilityBlend/,
  )
  assert.throws(
    () => mergeAndValidateConfig(adaptive, { plasticRegions: { minimumTrace: -0.01 } }),
    /minimumTrace/,
  )
  assert.throws(
    () => mergeAndValidateConfig(adaptive, { plasticRegions: { minimumTrace: 0 } }),
    /Active plastik bölge/,
  )
})
