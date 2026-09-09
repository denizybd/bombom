export const ADAPTIVE_REGION_FORMAT = 'plastic-region-atlas-head' as const
export const ADAPTIVE_REGION_VERSION = 1 as const
export const ADAPTIVE_REGION_PROPOSAL_FORMAT = 'plastic-region-atlas-proposal' as const

export type PlasticRegionLifecycle =
  | 'provisional'
  | 'active'
  | 'retiring'
  | 'retired'

export type PlasticRegionOrigin = 'discovery' | 'split' | 'merge'

export type RegionLifecycleAction =
  | 'birth'
  | 'grow'
  | 'shrink'
  | 'split'
  | 'merge'
  | 'retire'

export type RegionEvidenceCode =
  | 'unexplained-residual'
  | 'stable-coactivation'
  | 'capacity-pressure'
  | 'counterfactual-utility'
  | 'bimodal-behavior'
  | 'redundant-behavior'
  | 'low-unique-utility'

export type RegionEvidenceValidationAuthority = 'train-only' | 'independent-holdout'

export interface AdaptiveRegionGeometryV1 {
  readonly side: 64
  readonly cellCount: 4096
  readonly identityRows: 8
  readonly identityLength: 512
  readonly relationStartRow: 8
  readonly relationRows: 56
  readonly relationLength: 3584
  readonly pageSide: 8
  readonly pageRows: 7
  readonly pageColumns: 8
  readonly pageCount: 56
  readonly cellsPerPage: 64
  readonly fingerprint: string
}

export interface MatrixCoordinate {
  readonly row: number
  readonly column: number
}

export interface RelationPageCoordinate {
  readonly pageIndex: number
  readonly cellIndex: number
}

export interface PlasticRegionMembershipV1 {
  readonly regionId: string
  readonly weight: number
}

export interface PlasticPageAssignmentV1 {
  readonly pageIndex: number
  readonly memberships: readonly PlasticRegionMembershipV1[]
  readonly freeMass: number
}

export interface PlasticRegionPrototypeV1 {
  readonly dimension: number
  readonly values: readonly number[]
}

/**
 * Structural evidence only. Deliberately contains no human semantic label.
 * Source ids identify independence groups, not individual repeated episodes.
 */
export interface PlasticRegionEvidenceV1 {
  readonly observationCount: number
  readonly sourceGroupIds: readonly string[]
  readonly confidenceMass: number
  readonly cohesion: number
  readonly holdoutUtility: number
  readonly shamUtility: number
}

export interface PlasticRegionV1 {
  readonly id: string
  readonly lifecycle: PlasticRegionLifecycle
  readonly origin: PlasticRegionOrigin
  readonly bornRevision: number
  readonly retiredRevision: number | null
  readonly parentRegionIds: readonly string[]
  readonly prototype: PlasticRegionPrototypeV1
  readonly evidence: PlasticRegionEvidenceV1
}

export interface PlasticRegionAtlasV1 {
  readonly format: typeof ADAPTIVE_REGION_FORMAT
  readonly version: typeof ADAPTIVE_REGION_VERSION
  readonly representationSpaceId: string
  readonly geometryFingerprint: string
  readonly revision: number
  readonly parentFingerprint: string | null
  readonly behaviorDimension: number
  readonly maximumMembershipsPerPage: number
  readonly regions: readonly PlasticRegionV1[]
  readonly pages: readonly PlasticPageAssignmentV1[]
  readonly fingerprint: string
}

export interface PlasticPageObservationV1 {
  readonly observationId: string
  readonly episodeId: string
  readonly sourceGroupId: string
  readonly pageIndex: number
  readonly confidence: number
  readonly behavior: readonly number[]
  readonly outcomeCredit: number
  readonly ablationUtility: number
}

export interface PlasticEvidenceReferenceV1 {
  readonly observationId: string
  readonly episodeId: string
  readonly sourceGroupId: string
  /** Unexplained residual support for this page, not an independence vote. */
  readonly support: number
}

export interface PlasticPageSketchV1 {
  readonly pageIndex: number
  readonly behaviorDimension: number
  readonly behaviorPrototype: readonly number[]
  readonly outcomeCredit: number
  readonly ablationUtility: number
  readonly cohesion: number
  readonly observationCount: number
  readonly sourceGroupIds: readonly string[]
  readonly confidenceMass: number
  readonly evidence: readonly PlasticEvidenceReferenceV1[]
  readonly fingerprint: string
}

export interface RegionGateV1 {
  readonly regionId: string
  readonly compatibility: number
  readonly prototypeCompatibility: number
  readonly contextScore: number
  readonly confidence: number
  readonly capacityPenalty: number
  readonly logit: number
  readonly probability: number
}

export interface RegionGateDistributionV1 {
  readonly gates: readonly RegionGateV1[]
  readonly unmatchedProbability: number
  readonly omittedRegionProbability: number
}

export interface RegionLifecycleEvidenceV1 {
  readonly evidenceId: string
  readonly validationAuthority: RegionEvidenceValidationAuthority
  readonly observationCount: number
  readonly sourceGroupIds: readonly string[]
  readonly confidenceMass: number
  readonly cohesion: number
  readonly holdoutGain: number
  readonly shamGain: number
  readonly codes: readonly RegionEvidenceCode[]
}

export interface RegionLifecyclePolicyV1 {
  readonly minimumObservationCount: number
  readonly minimumIndependentSources: number
  readonly minimumConfidenceMass: number
  readonly minimumCohesion: number
  readonly minimumConstructiveHoldoutGain: number
  readonly maximumShamGain: number
  readonly maximumNonRegressionLoss: number
}

export type RegionProposalRejection =
  | 'MISSING_INDEPENDENT_EVALUATION'
  | 'INSUFFICIENT_REGION_CAPACITY'
  | 'INSUFFICIENT_OBSERVATIONS'
  | 'INSUFFICIENT_INDEPENDENT_SOURCES'
  | 'INSUFFICIENT_CONFIDENCE_MASS'
  | 'INSUFFICIENT_COHESION'
  | 'INSUFFICIENT_HOLDOUT_GAIN'
  | 'SHAM_EFFECT_TOO_HIGH'
  | 'HOLDOUT_REGRESSION_TOO_HIGH'

export interface RegionPageDeltaV1 {
  readonly pageIndex: number
  readonly before: PlasticPageAssignmentV1
  readonly after: PlasticPageAssignmentV1
}

export interface RegionLifecycleProposalV1 {
  readonly format: typeof ADAPTIVE_REGION_PROPOSAL_FORMAT
  readonly version: 1
  readonly action: RegionLifecycleAction
  readonly proposalId: string
  readonly baseAtlasFingerprint: string
  readonly targetRegionIds: readonly string[]
  readonly proposedRegionIds: readonly string[]
  readonly evidence: RegionLifecycleEvidenceV1
  readonly policy: RegionLifecyclePolicyV1
  readonly eligible: boolean
  readonly rejectionReasons: readonly RegionProposalRejection[]
  readonly pageDeltas: readonly RegionPageDeltaV1[]
  readonly proposedAtlas: PlasticRegionAtlasV1
  readonly fingerprint: string
}

export interface RegionAllocationChangeV1 {
  readonly pageIndex: number
  readonly weight: number
}

export interface RegionSplitFractionV1 {
  readonly pageIndex: number
  /** Share of the source membership transferred to the first child. */
  readonly firstChildFraction: number
}
