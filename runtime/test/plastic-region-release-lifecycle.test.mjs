import assert from 'node:assert/strict'
import test from 'node:test'
import {
  applyRegionLifecycleProposal,
  createPlasticRegionAtlas,
  proposeRegionBirth,
  proposeRegionGrow,
  proposeRegionMerge,
  proposeRegionRetire,
  proposeRegionShrink,
  proposeRegionSplit,
} from '../../src/adaptive-regions/index.ts'
import {
  createPlasticRegionReleaseEvaluationCandidate,
  createPlasticRegionReleaseSham,
  rebuildPlasticRegionReleaseProposal,
} from '../plastic-region-release-runner.mjs'
import { plasticRegionReleasePolicy } from '../plastic-region-release.mjs'

const REPRESENTATION_SPACE_ID = 'RS-RELEASE-LIFECYCLE-TEST'

function prototype(first = 1, second = 0) {
  return { dimension: 8, values: [first, second, 0, 0, 0, 0, 0, 0] }
}

function evidence(id, codes, authority = 'train-only') {
  return {
    evidenceId: id,
    validationAuthority: authority,
    observationCount: 40,
    sourceGroupIds: ['source-a', 'source-b', 'source-c'],
    confidenceMass: 3,
    cohesion: 0.9,
    holdoutGain: authority === 'independent-holdout' ? 0.1 : 0,
    shamGain: authority === 'independent-holdout' ? 0.01 : 0,
    codes,
  }
}

function activeBirth(base, id, allocations, shape = prototype()) {
  return applyRegionLifecycleProposal(base, proposeRegionBirth(base, {
    regionId: id,
    allocations,
    prototype: shape,
    evidence: evidence(`${id}-base`, ['counterfactual-utility', 'unexplained-residual'], 'independent-holdout'),
  }))
}

function releaseEvidence(proposal) {
  return { ...proposal.evidence, validationAuthority: 'independent-holdout', holdoutGain: 0.1, shamGain: 0.01 }
}

const cases = [
  {
    action: 'birth',
    make() {
      const base = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
      return [base, proposeRegionBirth(base, {
        regionId: 'born',
        allocations: [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })),
        prototype: prototype(),
        evidence: evidence('birth', ['counterfactual-utility', 'unexplained-residual']),
      })]
    },
  },
  {
    action: 'grow',
    make() {
      let base = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
      base = activeBirth(base, 'grown', [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })))
      base = activeBirth(base, 'grow-donor', [20, 21, 22, 23].map((pageIndex) => ({ pageIndex, weight: 0.25 })), prototype(0, 1))
      return [base, proposeRegionGrow(base, {
        regionId: 'grown', allocations: [{ pageIndex: 4, weight: 0.25 }],
        evidence: evidence('grow', ['capacity-pressure']),
      })]
    },
  },
  {
    action: 'shrink',
    make() {
      let base = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
      base = activeBirth(base, 'shrunk', [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.5 })))
      base = activeBirth(base, 'shrink-donor', [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.5 })), prototype(0, 1))
      return [base, proposeRegionShrink(base, {
        regionId: 'shrunk', allocations: [{ pageIndex: 0, weight: 0.25 }],
        evidence: evidence('shrink', ['low-unique-utility']),
      })]
    },
  },
  {
    action: 'split',
    make() {
      let base = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
      base = activeBirth(base, 'parent', [{ pageIndex: 0, weight: 1 }, { pageIndex: 1, weight: 1 }])
      return [base, proposeRegionSplit(base, {
        sourceRegionId: 'parent', firstChildId: 'child-a', secondChildId: 'child-b',
        firstChildPrototype: prototype(1, 0), secondChildPrototype: prototype(0, 1),
        fractions: [
          { pageIndex: 0, firstChildFraction: 0.5 },
          { pageIndex: 1, firstChildFraction: 0.5 },
        ],
        evidence: evidence('split', ['bimodal-behavior']),
      })]
    },
  },
  {
    action: 'merge',
    make() {
      let base = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
      base = activeBirth(base, 'merge-a', [{ pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 }])
      base = activeBirth(base, 'merge-b', [{ pageIndex: 2, weight: 0.5 }, { pageIndex: 3, weight: 0.5 }], prototype(0, 1))
      base = activeBirth(base, 'merge-donor-a', [{ pageIndex: 0, weight: 0.5 }, { pageIndex: 1, weight: 0.5 }], prototype(0, -1))
      base = activeBirth(base, 'merge-donor-b', [{ pageIndex: 2, weight: 0.5 }, { pageIndex: 3, weight: 0.5 }], prototype(-1, 0))
      return [base, proposeRegionMerge(base, {
        sourceRegionIds: ['merge-b', 'merge-a'], mergedRegionId: 'merged', prototype: prototype(0.5, 0.5),
        evidence: evidence('merge', ['redundant-behavior']),
      })]
    },
  },
  {
    action: 'retire',
    make() {
      let base = createPlasticRegionAtlas({ representationSpaceId: REPRESENTATION_SPACE_ID, behaviorDimension: 8 })
      base = activeBirth(base, 'retired', [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })))
      base = activeBirth(base, 'retire-donor', [0, 1, 2, 3].map((pageIndex) => ({ pageIndex, weight: 0.25 })), prototype(0, 1))
      return [base, proposeRegionRetire(base, {
        regionId: 'retired', evidence: evidence('retire', ['low-unique-utility']),
      })]
    },
  },
]

for (const scenario of cases) {
  test(`release bridge reconstructs ${scenario.action} exactly and finalizes it only with independent evidence`, () => {
    const [base, proposal] = scenario.make()
    assert.equal(proposal.action, scenario.action)
    assert.deepEqual(rebuildPlasticRegionReleaseProposal(base, proposal, proposal.evidence), proposal)
    const finalized = rebuildPlasticRegionReleaseProposal(base, proposal, releaseEvidence(proposal))
    assert.equal(finalized.eligible, true)
    assert.deepEqual(finalized.rejectionReasons, [])
    assert.equal(finalized.action, scenario.action)
    assert.equal(finalized.baseAtlasFingerprint, base.fingerprint)
    const policy = plasticRegionReleasePolicy({})
    const candidate = createPlasticRegionReleaseEvaluationCandidate(base, proposal, policy)
    const sham = createPlasticRegionReleaseSham(base, proposal, candidate, policy)
    assert.equal(sham.eligible, true)
    assert.equal(sham.action, scenario.action)
    assert.notEqual(sham.proposedAtlas.fingerprint, candidate.proposedAtlas.fingerprint)
  })
}
