import type { FamilyMembership, FamilyRun } from '../families/types'
import type {
  LanguageCorpusSource,
  LanguageExperienceConfig,
  SurfaceBinding,
  SurfacePiece,
} from './types'
import { clamp, compareText, normalizeSurface } from './deterministic'

interface EvidenceCandidate {
  memberKey: string
  identityDigest: string
  sourceDigest: string
  start: number
  end: number
  text: string
}

function membershipIndex(memberships: readonly FamilyMembership[]): Map<string, FamilyMembership[]> {
  const index = new Map<string, FamilyMembership[]>()
  for (const membership of memberships) {
    const values = index.get(membership.memberKey) ?? []
    values.push(membership)
    index.set(membership.memberKey, values)
  }
  for (const values of index.values()) {
    values.sort((left, right) => left.rank - right.rank || compareText(left.familyId, right.familyId))
  }
  return index
}

function alignmentScore(piece: SurfacePiece, evidence: EvidenceCandidate): number {
  const intersection = Math.max(0, Math.min(piece.end, evidence.end) - Math.max(piece.start, evidence.start))
  if (!intersection) return 0
  if (piece.start === evidence.start && piece.end === evidence.end) return 1
  const pieceCoverage = intersection / Math.max(1, piece.end - piece.start)
  const evidenceCoverage = intersection / Math.max(1, evidence.end - evidence.start)
  const sameSurface = piece.normalizedText && piece.normalizedText === normalizeSurface(evidence.text) ? 1 : 0
  return clamp(0.58 * pieceCoverage + 0.32 * evidenceCoverage + 0.1 * sameSurface)
}

function normalizedFamilyWeights(
  memberships: readonly FamilyMembership[],
): Array<{ familyId: string; weight: number }> {
  const accumulated = new Map<string, number>()
  for (const membership of memberships) {
    if (!Number.isFinite(membership.weight) || membership.weight <= 0) continue
    accumulated.set(membership.familyId, (accumulated.get(membership.familyId) ?? 0) + membership.weight)
  }
  const total = [...accumulated.values()].reduce((sum, value) => sum + value, 0)
  if (total <= 1e-12) return []
  return [...accumulated]
    .map(([familyId, weight]) => ({ familyId, weight: weight / total }))
    .sort((left, right) => right.weight - left.weight || compareText(left.familyId, right.familyId))
}

function validateRunCompatibility(familyRun: FamilyRun, sources: readonly LanguageCorpusSource[]): void {
  if (
    familyRun?.format !== 'noron-aileleri-run' || familyRun.version !== 1 ||
    !familyRun.representationSpaceId || !familyRun.behaviorSpaceId || !familyRun.fingerprint
  ) throw new TypeError('Dil deneyimi için geçerli bir Nöron Aileleri v1 modeli gerekir.')
  const sourceDigests = new Set(sources.map((source) => source.sourceDigest))
  if (sourceDigests.size !== sources.length) {
    // Exact-content duplicates are handled later; a digest collision with two
    // different source records is ambiguous at the coordinate-binding layer.
    const byDigest = new Map<string, string>()
    for (const source of sources) {
      const existing = byDigest.get(source.sourceDigest)
      if (existing !== undefined && existing !== source.text) {
        throw new TypeError('Aynı kaynak özeti farklı metinlerle kullanılamaz.')
      }
      byDigest.set(source.sourceDigest, source.text)
    }
  }
}

/**
 * Aligns observations with frozen family members by source coordinates. No I,
 * R, membership, profile or family object is ever modified.
 */
export function bindSurfacePieces(
  pieces: readonly SurfacePiece[],
  familyRun: FamilyRun,
  config: LanguageExperienceConfig,
): SurfacePiece[] {
  validateRunCompatibility(familyRun, [])
  const memberships = membershipIndex(familyRun.memberships)
  const evidenceBySource = new Map<string, EvidenceCandidate[]>()
  for (const profile of familyRun.profiles) {
    for (const evidence of profile.evidence) {
      if (
        !evidence.sourceDigest || !Number.isInteger(evidence.start) || !Number.isInteger(evidence.end) ||
        evidence.start < 0 || evidence.end <= evidence.start
      ) continue
      const candidates = evidenceBySource.get(evidence.sourceDigest) ?? []
      candidates.push({
        memberKey: profile.memberKey,
        identityDigest: profile.identityDigest,
        sourceDigest: evidence.sourceDigest,
        start: evidence.start,
        end: evidence.end,
        text: evidence.text,
      })
      evidenceBySource.set(evidence.sourceDigest, candidates)
    }
  }
  for (const candidates of evidenceBySource.values()) {
    candidates.sort((left, right) =>
      left.start - right.start || left.end - right.end || compareText(left.memberKey, right.memberKey),
    )
  }

  return pieces.map((piece): SurfacePiece => {
    const bestByMember = new Map<string, SurfaceBinding>()
    for (const evidence of evidenceBySource.get(piece.sourceDigest) ?? []) {
      if (evidence.end <= piece.start) continue
      if (evidence.start >= piece.end) break
      const score = alignmentScore(piece, evidence)
      if (score < config.minimumAlignmentScore) continue
      const existing = bestByMember.get(evidence.memberKey)
      if (!existing || score > existing.score) {
        bestByMember.set(evidence.memberKey, {
          memberKey: evidence.memberKey,
          identityDigest: evidence.identityDigest,
          score,
          familyWeights: normalizedFamilyWeights(memberships.get(evidence.memberKey) ?? []),
        })
      }
    }
    const bindings = [...bestByMember.values()]
      .sort((left, right) => right.score - left.score || compareText(left.memberKey, right.memberKey))
      .slice(0, config.maximumPieceBindings)
    return {
      ...piece,
      childIds: [...piece.childIds],
      bindings: bindings.map((binding) => ({
        ...binding,
        familyWeights: binding.familyWeights.map((weight) => ({ ...weight })),
      })),
    }
  })
}

export function assertFamilyIdentitySnapshot(
  familyRun: FamilyRun,
  before: ReadonlyMap<string, string>,
): void {
  if (familyRun.profiles.length !== before.size) {
    throw new Error('Dil deneyimi sırasında dondurulmuş profil kümesi değişti.')
  }
  for (const profile of familyRun.profiles) {
    if (before.get(profile.memberKey) !== profile.identityDigest) {
      throw new Error('Dil deneyimi sırasında dondurulmuş nöron kimliği değişti.')
    }
  }
}

export function snapshotFamilyIdentities(familyRun: FamilyRun): Map<string, string> {
  return new Map(familyRun.profiles.map((profile) => [profile.memberKey, profile.identityDigest]))
}

export function validateLanguageAdapterInput(
  familyRun: FamilyRun,
  sources: readonly LanguageCorpusSource[],
): void {
  validateRunCompatibility(familyRun, sources)
  if (!sources.length) throw new TypeError('Dil deneyimi için en az bir kaynak gerekir.')
}
