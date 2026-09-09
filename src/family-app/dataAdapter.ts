import type {
  FamilyAnalysisInput,
  FamilyRunInput,
  FamilySynthesisDecisionInput,
} from '../families/types'
import type { SynthesisArchiveRecord } from '../synthesis-app/persistence'
import { loadApprovedSynthesisArchive } from '../synthesis-app/persistence'
import type { PersistedProject, TextLabAnalysis } from '../text-lab/engine'
import { loadAllProjects } from '../text-lab/persistence/indexedDb'

export interface FamilySourceBundle {
  projects: PersistedProject[]
  decisions: SynthesisArchiveRecord[]
  warnings: string[]
}

/**
 * Loads both upstream archives without ever opening a write transaction.
 * A failure in one archive does not hide usable data from the other archive.
 */
export async function loadFamilySourceBundle(): Promise<FamilySourceBundle> {
  const [projectResult, archiveResult] = await Promise.allSettled([
    loadAllProjects(),
    loadApprovedSynthesisArchive(),
  ])
  const warnings: string[] = []
  if (projectResult.status === 'rejected') {
    warnings.push(projectResult.reason instanceof Error
      ? `Metin Laboratuvarı arşivi açılamadı: ${projectResult.reason.message}`
      : 'Metin Laboratuvarı arşivi açılamadı.')
  }
  if (archiveResult.status === 'rejected') {
    warnings.push(archiveResult.reason instanceof Error
      ? `Nöron Sentezi arşivi açılamadı: ${archiveResult.reason.message}`
      : 'Nöron Sentezi arşivi açılamadı.')
  }
  return {
    projects: projectResult.status === 'fulfilled' ? projectResult.value : [],
    decisions: archiveResult.status === 'fulfilled' ? archiveResult.value : [],
    warnings,
  }
}

/** Dense K is deliberately omitted. H⁰ is retained because behavior profiling uses it read-only. */
export function toFamilyAnalysisInput(analysis: TextLabAnalysis): FamilyAnalysisInput {
  return {
    format: analysis.format,
    version: analysis.version,
    runId: analysis.runId,
    createdAt: analysis.createdAt,
    independenceGroupId: analysis.independenceGroupId,
    sourceDigest: analysis.sourceDigest,
    identitySpaceId: analysis.identitySpaceId,
    contextSpaceId: analysis.contextSpaceId,
    representationSpaceId: analysis.representationSpaceId,
    source: analysis.source,
    config: analysis.config,
    layout: analysis.layout,
    rootIds: [...analysis.rootIds],
    nodes: analysis.nodes.map((node) => ({
      id: node.id,
      sourceId: node.sourceId,
      paragraphIndex: node.paragraphIndex,
      start: node.start,
      end: node.end,
      text: node.text,
      graphemeCount: node.graphemeCount,
      depth: node.depth,
      branch: node.branch,
      parentId: node.parentId,
      childIds: [...node.childIds],
      splitBoundary: node.splitBoundary,
      // These copies can be transferred to the Worker without detaching the
      // live TextLab project objects held by the UI.
      identity: node.identity.slice(),
      relation: node.relation.slice(),
      identityDigest: node.identityDigest,
      identityNorm: node.identityNorm,
      relationNorm: node.relationNorm,
      contextSupport: node.contextSupport,
      contextRevision: node.contextRevision,
      trace: { ...node.trace },
    })),
    habitProbabilities: analysis.habitProbabilities.slice(),
  }
}

function toFamilySynthesisDecisionInput(record: SynthesisArchiveRecord): FamilySynthesisDecisionInput {
  return {
    ...record,
    analysisRunIds: [...record.analysisRunIds],
    sourceDigests: [...record.sourceDigests],
    config: { ...record.config },
    candidate: {
      ...record.candidate,
      members: record.candidate.members.map((member) => ({ ...member })),
      independentSupports: record.candidate.independentSupports.map((support) => ({ ...support })),
      medoid: { ...record.candidate.medoid },
      identity: record.candidate.identity.slice(),
      relation: record.candidate.relation.slice(),
    },
  }
}

export function createFamilyRunInput(
  analyses: readonly TextLabAnalysis[],
  decisions: readonly SynthesisArchiveRecord[],
  previousRun: FamilyRunInput['previousRun'] = null,
  activations: FamilyRunInput['activations'] = undefined,
  dataSplit: Pick<FamilyRunInput, 'discoveryIndependenceGroupIds' | 'evaluationIndependenceGroupIds'> = {},
): FamilyRunInput {
  return {
    analyses: analyses.map(toFamilyAnalysisInput),
    ...dataSplit,
    synthesisDecisions: decisions
      .filter((record) => record.decision === 'approved')
      .map(toFamilySynthesisDecisionInput),
    previousRun,
    activations,
  }
}

/** Every listed buffer belongs to the disposable Worker input snapshot. */
export function familyRunTransferables(input: FamilyRunInput): Transferable[] {
  const buffers = new Set<ArrayBuffer>()
  const retain = (buffer: ArrayBufferLike) => {
    if (buffer instanceof ArrayBuffer) buffers.add(buffer)
  }
  for (const analysis of input.analyses) {
    retain(analysis.habitProbabilities.buffer)
    for (const node of analysis.nodes) {
      retain(node.identity.buffer)
      retain(node.relation.buffer)
    }
  }
  for (const decision of input.synthesisDecisions ?? []) {
    retain(decision.candidate.identity.buffer)
    retain(decision.candidate.relation.buffer)
  }
  return [...buffers]
}
