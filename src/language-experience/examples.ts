import {
  LANGUAGE_BOS_FAMILY,
  LANGUAGE_BOS_MEMBER,
  LANGUAGE_EOS_FAMILY,
  LANGUAGE_EOS_MEMBER,
} from './types'
import type {
  LanguageDataset,
  LanguageExample,
  SurfaceBinding,
  SurfacePiece,
} from './types'
import { compareText, normalizePositive } from './deterministic'

export interface WeightedTarget {
  key: string
  weight: number
}

export type LanguagePieceIndex = ReadonlyMap<string, SurfacePiece>

export function prepareLanguagePieceIndex(dataset: LanguageDataset): LanguagePieceIndex {
  return new Map(dataset.pieces.map((piece) => [piece.id, piece]))
}

function bindingWeights(bindings: readonly SurfaceBinding[]): WeightedTarget[] {
  return normalizePositive(bindings.map((binding) => ({
    key: binding.memberKey,
    score: binding.score,
  }))).map(({ key, probability }) => ({ key, weight: probability }))
}

export function targetMemberWeights(example: LanguageExample): WeightedTarget[] {
  return example.eos
    ? [{ key: LANGUAGE_EOS_MEMBER, weight: 1 }]
    : bindingWeights(example.targetBindings)
}

export function targetFamilyWeights(example: LanguageExample): WeightedTarget[] {
  if (example.eos) return [{ key: LANGUAGE_EOS_FAMILY, weight: 1 }]
  const accumulated = new Map<string, number>()
  for (const binding of example.targetBindings) {
    for (const family of binding.familyWeights) {
      accumulated.set(
        family.familyId,
        (accumulated.get(family.familyId) ?? 0) + binding.score * family.weight,
      )
    }
  }
  return normalizePositive([...accumulated].map(([key, score]) => ({ key, score })))
    .map(({ key, probability }) => ({ key, weight: probability }))
}

export function teacherForcedRecentMembers(
  example: LanguageExample,
  dataset: LanguageDataset,
  preparedIndex?: LanguagePieceIndex,
): string[] {
  const pieceById = preparedIndex ?? prepareLanguagePieceIndex(dataset)
  return example.prefixPieceIds
    .map((id) => pieceById.get(id)?.bindings
      .slice()
      .sort((left, right) => right.score - left.score || compareText(left.memberKey, right.memberKey))[0]?.memberKey,
    )
    .filter((memberKey): memberKey is string => Boolean(memberKey))
}

export function memberOriginForExample(
  example: LanguageExample,
  dataset: LanguageDataset,
  preparedIndex?: LanguagePieceIndex,
): string {
  return teacherForcedRecentMembers(example, dataset, preparedIndex).at(-1) ?? LANGUAGE_BOS_MEMBER
}

export function familyOriginsForExample(
  example: LanguageExample,
  dataset: LanguageDataset,
  preparedIndex?: LanguagePieceIndex,
): WeightedTarget[] {
  const pieceById = preparedIndex ?? prepareLanguagePieceIndex(dataset)
  const previousPiece = pieceById.get(example.prefixPieceIds.at(-1) ?? '')
  if (!previousPiece?.bindings.length) return [{ key: LANGUAGE_BOS_FAMILY, weight: 1 }]
  const accumulated = new Map<string, number>()
  for (const binding of previousPiece.bindings) {
    for (const family of binding.familyWeights) {
      accumulated.set(
        family.familyId,
        (accumulated.get(family.familyId) ?? 0) + binding.score * family.weight,
      )
    }
  }
  const normalized = normalizePositive([...accumulated].map(([key, score]) => ({ key, score })))
    .map(({ key, probability }) => ({ key, weight: probability }))
  return normalized.length ? normalized : [{ key: LANGUAGE_BOS_FAMILY, weight: 1 }]
}
