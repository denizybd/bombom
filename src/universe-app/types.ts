import type { StructuralStopGateStatus } from '../training-app/types'

export type MembershipTuple = [
  neuron: number,
  family: number,
  score: number,
  weight: number,
  rank: number,
  core: 0 | 1,
]

export type FamilyTransitionTuple = [
  from: number,
  to: number,
  probability: number,
  prior: number,
  experience: number,
  inhibition: number,
  sourceCount: number,
  supportCount: number,
]

export type NeuronTransitionTuple = [from: number, to: number, probability: number]
export type MacroExpansionTuple = [
  macroNeuron: number,
  memberNeuron: number,
  ordinal: number,
  nestedComposite: 0 | 1,
]

export interface AtlasConceptMacro {
  macroKind: 'sequence-macro' | 'equivalence-concept'
  lifecycle: string
  runtimeStatus: 'active' | 'inactive-lifecycle' | 'research-snapshot' | 'rejected-snapshot' | string
  runtimeActive: boolean
  expansionLength: number
  depth: number
  fidelity: number
  compressionGain: number
  compressionRatio: number
  sourceGroupCount: number
  observationCount: number
}

export interface AtlasFamily {
  id: string
  index: number
  /** Provenance of the family; older atlas payloads intentionally omit it. */
  origin?: string
  /** Visual-only placement class. It never changes routing or membership evidence. */
  layoutZone?: 'core' | 'language-surface' | string
  /** Pinned language seeds keep their deterministic atlas anchor as members grow. */
  layoutPinned?: boolean
  /** Stable, manually assigned position in the language-surface workflow arc. */
  layoutSlot?: number
  /** Optional parent seed used to place autonomously grown language satellites. */
  parentFamilyId?: string | null
  label?: string
  /** Normalized local [tangent-x, tangent-y, outward] hint inside the language zone. */
  atlasAnchor?: [number, number, number]
  region: string
  regionVerified: boolean
  lifecycle: string
  supportCount: number
  sourceCount: number
  memberCount: number
  coreMemberCount: number
  cohesion: number
  separation: number
  stability: number
  utility: number | null
  utilityStatus: string
  predecessors: number[]
}

export interface AtlasNeuron {
  id: string
  index: number
  text: string
  kind: 'raw' | 'synthesis' | 'language-output' | 'language-nursery' | 'concept-macro'
  origin?: 'manual-language-seed' | 'autonomous-language' | string
  status?: string
  /** Visual placement only; concept overlays do not enter the family force graph. */
  layoutZone?: 'concept-surface' | string
  /** Bounded metadata summary. Signal/identity vectors are intentionally absent. */
  conceptMacro?: AtlasConceptMacro
  sourceCount: number
  evidenceCount: number
  membershipCount: number
  primaryFamily: number
  diagnostics: number[]
  affinity: { region: string; score: number; verified: boolean } | null
}

export interface AtlasPlasticRegionSummary {
  id: string
  lifecycle: string
  origin: string
  effectiveCapacity: number
  assignedPageCount: number
  dominantPageCount: number
  meanMembershipWeight: number
  globalShare: number
  cohesion: number
  observationCount: number
  sourceGroupCount: number
}

export interface AtlasPlasticRegionPageSummary {
  pageIndex: number
  dominantRegionId: string | null
  dominantWeight: number
  freeMass: number
  membershipCount: number
}

export interface AtlasPlasticRegionProjection {
  artifact: { id: string; kind: string; contentHash: string | null; createdAt: string; jobId: string | null }
  headFingerprint: string | null
  headRevision: number
  geometryFingerprint: string | null
  representationSpaceId: string
  configuredMode: 'off' | 'shadow' | 'proposal' | 'active' | string
  /** Effective opt-in mode after the global enabled switch is applied. */
  effectiveMode: 'off' | 'shadow' | 'proposal' | 'active' | string
  effectiveModeReason: string
  runtimeWired: boolean
  /** True only when this exact selected head is the runtime-resolved head. */
  exactRuntimeBinding: boolean
  /** True only for an active mode backed by a verified release chain. */
  behaviorApplied: boolean
  release: {
    eligible: boolean
    artifactId: string | null
    releaseId: string | null
    fingerprint: string | null
    proposalId: string | null
    reason: string | null
  }
  compatible: boolean
  mismatch: string | null
  regionCount: number
  activeRegionCount: number
  pageCount: number
  assignedMass: number
  freeMass: number
  dominantRegionId: string | null
  /** Bounded evidence/allocation summaries only; prototype vectors are absent. */
  regions: AtlasPlasticRegionSummary[]
  pages: AtlasPlasticRegionPageSummary[]
}

export interface NeuronAtlas {
  format: 'bombom-neuron-atlas'
  version: 1
  /** Includes the exact selected snapshot's language and concept overlay lineage. */
  fingerprint?: string
  available: boolean
  reason?: string
  /** A safe reduced projection is being shown while the dense core is busy. */
  partial?: boolean
  busy?: boolean
  busyStage?: string
  selection: {
    selectedBy: string
    snapshot: { id: string; name: string; status: string; createdAt: string; active: boolean } | null
    artifact: { id: string; kind: string; contentHash: string | null; createdAt: string; jobId: string | null } | null
  }
  run: {
    id: string
    fingerprint: string
    createdAt: string
    representationSpaceId: string
    behaviorSpaceId: string
    corpusFingerprint: string
  } | null
  stats: {
    familyCount: number
    neuronCount: number
    membershipCount: number
    familyTransitionCount: number
    neuronTransitionCount: number
    distinctSourceCount?: number
    validatedUtilityFamilyCount?: number
    mergeProposedFamilyCount?: number
    priorEdgeCount?: number
    experienceEdgeCount?: number
    inhibitionEdgeCount?: number
    learnedFamilyCount?: number
    languageOutputFamilyCount?: number
    languageOutputSeedNeuronCount?: number
    languageOutputNurseryCount?: number
    languageOutputProvisionalNeuronCount?: number
    conceptMacroCount?: number
    conceptMacroTotalCount?: number
    conceptMacroActiveCount?: number
    conceptMacroExpansionLinkCount?: number
    plasticRegionCount?: number
    plasticRegionActiveCount?: number
    plasticRegionPageCount?: number
  }
  projection?: {
    semanticAxes: boolean
    layout: string
    note: string
    languageOutput?: {
      scaffoldFingerprint: string
      headFingerprint: string | null
      headRevision: number
      persisted: boolean
      compatible: boolean
      mismatch: boolean
      familyCount: number
      seedNeuronCount: number
      nurseryCount: number
      provisionalNeuronCount: number
      membershipEvidenceCount: number
    }
    conceptMacros?: {
      artifact: { id: string; kind: string; contentHash: string | null; createdAt: string; jobId: string | null } | null
      headFingerprint: string | null
      headRevision: number
      compatible: boolean
      mismatch: string | null
      snapshotStatus: string | null
      runtimeEligible: boolean
      totalMacroCount: number
      nodeCount: number
      linkCount: number
      activeCount: number
      truncated: boolean
      byLifecycle: Array<{ lifecycle: string; count: number }>
    }
    plasticRegions?: AtlasPlasticRegionProjection
  }
  families: AtlasFamily[]
  neurons: AtlasNeuron[]
  links: {
    memberships: MembershipTuple[]
    familyTransitions: FamilyTransitionTuple[]
    neuronTransitions: NeuronTransitionTuple[]
    macroExpansions?: MacroExpansionTuple[]
  }
  warnings?: string[]
}

export interface UniverseLayout {
  familyPositions: Float32Array
  neuronPositions: Float32Array
  radius: number
}

export interface LiveTrainingState {
  serviceState: string
  activeStage: string | null
  progress: number
  campaignStatus: string | null
  campaignAccepted: number
  campaignGenerated: number
  completedBlocks: number
  gateStatus: StructuralStopGateStatus | null
  gatePassed: boolean
  updatedAt: string | null
}

export type UniverseSelection =
  | { kind: 'family'; index: number }
  | { kind: 'neuron'; index: number }
  | null

export type ColorMode = 'family' | 'lifecycle' | 'utility' | 'region'
