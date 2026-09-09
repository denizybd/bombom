export type PlasticRegionMode = 'off' | 'shadow' | 'proposal' | 'active'

export interface RegionMembershipView {
  regionId: string
  weight: number
}

export interface RegionPageView {
  pageIndex: number
  memberships: RegionMembershipView[]
  freeMass: number
}

export interface PlasticRegionView {
  id: string
  lifecycle: string
  origin: string
  bornRevision: number
  retiredRevision: number | null
  parentRegionIds: string[]
  prototype: { dimension: number; values: number[]; truncated: boolean }
  evidence: {
    observationCount: number
    sourceGroupCount: number
    confidenceMass: number
    cohesion: number
    holdoutUtility: number
    shamUtility: number
  }
}

export interface PlasticRegionAtlasView {
  format: 'plastic-region-atlas-head'
  version: 1
  representationSpaceId: string
  geometryFingerprint: string | null
  revision: number
  parentFingerprint: string | null
  behaviorDimension: number
  maximumMembershipsPerPage: number
  fingerprint: string
  regions: PlasticRegionView[]
  pages: RegionPageView[]
  stats: {
    regionCount: number
    pageCount: number
    lifecycleCounts: Record<string, number>
    assignedMass: number
    freeMass: number
  }
  truncated: boolean
}

export interface PlasticRegionStatus {
  enabled: boolean
  configuredMode: PlasticRegionMode
  effectiveMode: PlasticRegionMode
  effectiveModeReason: string
  runtimeWired: boolean
  representationProfile: string
  semanticProbesEnabled: boolean
  authority: 'none' | 'active' | 'candidate-research'
  routingEligible: boolean
  activationCandidateSnapshotId: string | null
  available: boolean
  reason: string | null
  snapshot: { id: string; status: string; createdAt: string } | null
  head: {
    artifactId: string
    contentHash: string | null
    fingerprint: string
    revision: number
    representationSpaceId: string
    geometryFingerprint: string | null
    regionCount: number
    pageCount: number
    createdAt: string | null
  } | null
  release: {
    artifactId: string | null
    releaseId: string | null
    fingerprint: string | null
    proposalId: string | null
    reason: string | null
  }
  readiness: {
    shadowEvidenceAvailable: boolean
    reason: string | null
    evidenceArtifactId: string | null
    evidenceFingerprint: string | null
  }
  counts: { totalArtifacts: number; byKind: Record<string, number> }
  limits: {
    maximumObservationsPerBatch: number
    maximumObservationBatchesPerCycle: number
    maximumRegions: number
    maximumMembershipsPerPage: number
    maximumActiveGates: number
    releaseMinimumCases: number
    releaseMinimumSourceGroups: number
    releaseMinimumHoldoutGain: number
    releaseMaximumShamGain: number
    releaseMaximumRegressionLoss: number
    maximumArtifactBytes: number
  }
}

export interface PlasticRegionArtifactView {
  artifactId: string
  kind: string
  contentHash: string | null
  createdAt: string | null
  jobId: string | null
  payloadBytes: number
}

export type SemanticProbeStatus = 'unknown' | 'provisional' | 'calibrated'

export interface SemanticProbeHypothesisView {
  hypothesis: {
    hypothesisId: string
    namespace: string
    label: string
  }
  status: SemanticProbeStatus
  atlasFingerprint: string
  observationCount: number
  independentSourceCount: number
  target: {
    featureId: string
    responseKind: string
    footprint: Array<{ pageIndex: number; weight: number }>
  } | null
  evidence: {
    evaluationSplit: 'validation' | 'test'
    targetBrier: number
    targetEce: number
    gainVsBaseline: { point: number; low: number; high: number; confidenceLevel: number }
    gainVsSham: { point: number; low: number; high: number; confidenceLevel: number }
    reasons: string[]
  } | null
}

export interface SemanticProbeInspectionView {
  available: boolean
  interpretationOnly: true
  trainingWritesAllowed: false
  routingMutationAllowed: false
  releaseAuthority: false
  atlasFingerprint: string | null
  bytesRead: number
  legacySkeletonCount: number
  items: SemanticProbeHypothesisView[]
  omittedCount: number
}

export interface PlasticRegionDashboard {
  status: PlasticRegionStatus
  atlas: PlasticRegionAtlasView | null
  artifacts: PlasticRegionArtifactView[]
  probes: SemanticProbeInspectionView
}

export interface PlasticRegionAtlasDetail {
  artifact: PlasticRegionArtifactView | null
  proposal: {
    proposalId: string | null
    action: string | null
    eligible: boolean
    rejectionReasons: string[]
    fingerprint: string | null
    baseAtlasFingerprint: string | null
  } | null
  atlas: PlasticRegionAtlasView
}

export type PlasticRegionReleaseJobState =
  | 'waiting'
  | 'active'
  | 'complete'
  | 'failed'
  | 'cancelled'

export interface PlasticRegionReleaseJobView {
  id: string
  state: PlasticRegionReleaseJobState
  progress: number
  stage: string | null
  message: string | null
  duplicate: boolean
  releaseResult: {
    released: boolean
    awaitingApproval: boolean
    automaticActivation: false
    evaluationId: string | null
    snapshotId: string | null
    releaseFingerprint: string | null
    auditArtifactId: string | null
    benchmarkReportArtifactId: string | null
    reasonCodes: string[]
    metrics: {
      baseMean: number | null
      candidateMean: number | null
      shamMean: number | null
      conservativeHoldoutGain: number | null
      conservativeShamGain: number | null
      holdoutRegressionHigh: number | null
    }
  } | null
}
