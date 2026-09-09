import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultConfig } from '../config.mjs'
import { BASE_PIPELINE_STAGES, resolvePipelineStages } from '../pipeline.mjs'

test('default/off pipeline stage order remains byte-for-byte legacy compatible', () => {
  const config = defaultConfig('/tmp/plastic-region-stage-test')
  assert.deepEqual(resolvePipelineStages(config), [...BASE_PIPELINE_STAGES])
  assert.deepEqual(resolvePipelineStages({}), [...BASE_PIPELINE_STAGES])
})

test('canonical opt-in inserts observation stage before concept mining only', () => {
  const config = defaultConfig('/tmp/plastic-region-stage-test')
  config.plasticRegions = {
    ...config.plasticRegions,
    mode: 'shadow',
    representationProfile: 'adaptive-eighth-id-v1',
  }
  const stages = resolvePipelineStages(config)
  assert.deepEqual(stages, [
    'collect',
    'textLab',
    'synthesis',
    'families',
    'languageExperience',
    'plasticRegions',
    'conceptMacros',
    'evaluate',
  ])
  assert.deepEqual(BASE_PIPELINE_STAGES, [
    'collect',
    'textLab',
    'synthesis',
    'families',
    'languageExperience',
    'conceptMacros',
    'evaluate',
  ])
})
