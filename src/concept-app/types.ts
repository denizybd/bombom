export type MacroLifecycle =
  | 'observing'
  | 'eligible-proposal'
  | 'virtual-provisional'
  | 'shadow-validated'
  | 'active'
  | 'deprecated'

export interface MacroExpansionEntry {
  position: number
  memberKey: string | null
  familyId: string | null
  macroId: string | null
  label?: string | null
}

export interface ConceptMacroSummary {
  id: string
  fingerprint: string
  kind: 'sequence-macro' | 'equivalence-concept' | string
  lifecycle: MacroLifecycle | string
  label: string
  expansionLength: number
  orderedExpansion: MacroExpansionEntry[]
  occurrenceCount: number
  sourceGroupCount: number
  sourceGroupIds: string[]
  completionProbability: number | null
  pathCohesion: number | null
  effectCohesion: number | null
  compressionGain: number | null
  predictiveUtility: number | null
  shadowFidelity: number | null
  falseActivationRate: number | null
  structuralEquivalencePassed: boolean
  structuralEquivalenceAuditId: string | null
  logicalDuration: number
  nestedDepth: number
  expansionDigest: string | null
  fallbackSafe: boolean
  physicalIdentityAllocated: boolean
  updatedAt: string | null
}

export interface ConceptEnvelope {
  ok: true
  available: boolean
  mode: string
  head: null | {
    id: string
    fingerprint: string | null
    revision: number
    representationSpaceId: string | null
    familyRunFingerprint: string | null
    createdAt: string | null
    updatedAt: string | null
    snapshotId: string | null
    snapshotStatus: string | null
  }
  stats: {
    totalMacros: number
    lifecycleCounts: Record<string, number>
    observationCount: number
    sourceGroupCount: number
    shadowEvaluationCount: number
    activeCount: number
    structuralValidatedCount: number
    virtualCount: number
    fallbackProtectedCount: number
  }
  total: number
  offset: number
  limit: number
  macros: ConceptMacroSummary[]
}
