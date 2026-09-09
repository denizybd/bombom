import { createInitialPlasticRegionAtlas } from './atlas.ts'
import type {
  PlasticRegionAtlasV1,
  PlasticRegionPrototypeV1,
  RegionLifecycleEvidenceV1,
} from './types.ts'

export function testAtlas(maximumMembershipsPerPage = 3): PlasticRegionAtlasV1 {
  return createInitialPlasticRegionAtlas({
    representationSpaceId: 'test-adaptive-eighth-id-v1',
    behaviorDimension: 2,
    maximumMembershipsPerPage,
  })
}

export function testPrototype(first = 1, second = 0): PlasticRegionPrototypeV1 {
  return { dimension: 2, values: [first, second] }
}

export function strongEvidence(
  evidenceId = 'evidence-1',
  codes: RegionLifecycleEvidenceV1['codes'] = ['counterfactual-utility'],
): RegionLifecycleEvidenceV1 {
  return {
    evidenceId,
    validationAuthority: 'independent-holdout',
    observationCount: 6,
    sourceGroupIds: ['source-a', 'source-b', 'source-c'],
    confidenceMass: 1.5,
    cohesion: 0.9,
    holdoutGain: 0.08,
    shamGain: 0.01,
    codes,
  }
}
