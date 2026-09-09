import type { FamilyRun } from '../families/types'
import { bindSurfacePieces, snapshotFamilyIdentities, assertFamilyIdentitySnapshot, validateLanguageAdapterInput } from './adapter'
import {
  canonicalizeLanguageText,
  compareText,
  deterministicFirst,
  deterministicOrder,
  languageDigest,
  normalizeSurface,
} from './deterministic'
import { buildSurfaceHierarchy } from './hierarchy'
import type {
  DatasetSplit,
  DuplicateSourceRecord,
  LanguageCorpusSource,
  LanguageDataset,
  LanguageExample,
  LanguageExperienceConfig,
  SurfaceBinding,
  SurfacePiece,
  SourceGroupSplitAssignment,
} from './types'

interface RetainedSource extends LanguageCorpusSource {
  independenceGroupId: string
  canonicalFingerprint: string
  split: DatasetSplit
}

function allocateSplitCounts(count: number, config: LanguageExperienceConfig): Record<DatasetSplit, number> {
  if (count === 0) return { train: 0, validation: 0, test: 0 }
  if (count === 1) return { train: 1, validation: 0, test: 0 }
  if (count === 2) return { train: 1, validation: 0, test: 1 }
  const splits: DatasetSplit[] = ['train', 'validation', 'test']
  const ratios = [config.trainRatio, config.validationRatio, config.testRatio]
  const raw = ratios.map((ratio) => ratio * count)
  const allocated = raw.map(Math.floor)
  let remaining = count - allocated.reduce((sum, value) => sum + value, 0)
  const byRemainder = splits.map((split, index) => ({ split, index, remainder: raw[index] - allocated[index] }))
    .sort((left, right) => right.remainder - left.remainder || left.index - right.index)
  for (let index = 0; index < remaining; index += 1) allocated[byRemainder[index].index] += 1

  if (count >= 3) {
    for (let index = 0; index < allocated.length; index += 1) {
      if (allocated[index] > 0) continue
      const donor = allocated
        .map((value, donorIndex) => ({ value, donorIndex }))
        .filter((entry) => entry.value > 1)
        .sort((left, right) => right.value - left.value || left.donorIndex - right.donorIndex)[0]
      if (donor) {
        allocated[donor.donorIndex] -= 1
        allocated[index] += 1
      }
    }
  }
  remaining = allocated.reduce((sum, value) => sum + value, 0)
  if (remaining !== count) throw new Error('Dil korpusu bölüm sayıları dengelenemedi.')
  return { train: allocated[0], validation: allocated[1], test: allocated[2] }
}

function deduplicateSources(
  input: readonly LanguageCorpusSource[],
): { retained: Omit<RetainedSource, 'split'>[]; duplicates: DuplicateSourceRecord[] } {
  const ordered = [...input]
    .map((source) => ({
      ...source,
      independenceGroupId: source.independenceGroupId?.trim() || source.sourceId,
      text: canonicalizeLanguageText(source.text),
    }))
    .sort((left, right) =>
      compareText(left.independenceGroupId, right.independenceGroupId) ||
      compareText(left.sourceId, right.sourceId) ||
      compareText(left.sourceDigest, right.sourceDigest),
    )
  const retained: Omit<RetainedSource, 'split'>[] = []
  const duplicates: DuplicateSourceRecord[] = []
  const byChunkIdentity = new Map<string, Omit<RetainedSource, 'split'>>()
  const groupBySourceId = new Map<string, string>()
  const byContent = new Map<string, Omit<RetainedSource, 'split'>>()
  const byNormalizedContent = new Map<string, Omit<RetainedSource, 'split'>>()
  const byDigest = new Map<string, Omit<RetainedSource, 'split'>>()

  for (const source of ordered) {
    if (!source.sourceId || !source.sourceDigest || !source.text.trim()) {
      throw new TypeError('Her dil kaynağının kimliği, özeti ve boş olmayan metni olmalıdır.')
    }
    const canonicalFingerprint = languageDigest('canonical-source', source.text)
    const normalizedFingerprint = languageDigest('normalized-source', normalizeSurface(source.text))
    const knownGroup = groupBySourceId.get(source.sourceId)
    if (knownGroup !== undefined && knownGroup !== source.independenceGroupId) {
      const sameId = retained.find((candidate) => candidate.sourceId === source.sourceId)!
      duplicates.push({
        canonicalSourceDigest: canonicalFingerprint,
        keptSourceId: sameId.sourceId,
        rejectedSourceId: source.sourceId,
        reason: 'duplicate-source-id',
      })
      continue
    }
    const chunkIdentity = `${source.independenceGroupId}\u0000${source.sourceId}\u0000${source.sourceDigest}`
    const sameChunk = byChunkIdentity.get(chunkIdentity)
    if (sameChunk) {
      duplicates.push({
        canonicalSourceDigest: canonicalFingerprint,
        keptSourceId: sameChunk.sourceId,
        rejectedSourceId: source.sourceId,
        reason: 'duplicate-source-id',
      })
      continue
    }
    const sameDigest = byDigest.get(source.sourceDigest)
    if (sameDigest && sameDigest.canonicalFingerprint !== canonicalFingerprint) {
      duplicates.push({
        canonicalSourceDigest: canonicalFingerprint,
        keptSourceId: sameDigest.sourceId,
        rejectedSourceId: source.sourceId,
        reason: 'digest-content-conflict',
      })
      continue
    }
    const sameContent = byContent.get(canonicalFingerprint)
    if (sameContent) {
      duplicates.push({
        canonicalSourceDigest: canonicalFingerprint,
        keptSourceId: sameContent.sourceId,
        rejectedSourceId: source.sourceId,
        reason: 'exact-canonical-duplicate',
      })
      continue
    }
    const sameNormalizedContent = byNormalizedContent.get(normalizedFingerprint)
    if (sameNormalizedContent) {
      duplicates.push({
        canonicalSourceDigest: canonicalFingerprint,
        keptSourceId: sameNormalizedContent.sourceId,
        rejectedSourceId: source.sourceId,
        reason: 'normalized-surface-duplicate',
      })
      continue
    }
    const value = { ...source, canonicalFingerprint }
    retained.push(value)
    byChunkIdentity.set(chunkIdentity, value)
    groupBySourceId.set(source.sourceId, source.independenceGroupId)
    byDigest.set(source.sourceDigest, value)
    byContent.set(canonicalFingerprint, value)
    byNormalizedContent.set(normalizedFingerprint, value)
  }
  return { retained, duplicates }
}

export function splitLanguageSources(
  sources: readonly LanguageCorpusSource[],
  config: LanguageExperienceConfig,
  previousAssignments: readonly SourceGroupSplitAssignment[] = [],
): { sources: RetainedSource[]; duplicates: DuplicateSourceRecord[]; warnings: string[] } {
  const { retained, duplicates } = deduplicateSources(sources)
  const groups = new Map<string, Omit<RetainedSource, 'split'>[]>()
  for (const source of retained) {
    const chunks = groups.get(source.independenceGroupId) ?? []
    chunks.push(source)
    groups.set(source.independenceGroupId, chunks)
  }
  const orderedGroups = deterministicOrder([...groups], config.seed, ([groupId, chunks]) =>
    languageDigest('independence-group', groupId, chunks.map((chunk) => chunk.canonicalFingerprint).sort(compareText)),
  )
  // Teacher-generated material may teach transitions, but it must never grade the
  // system that learned from it. If a physical source group contains any generated
  // chunk, keep the complete group in train so source-atomicity cannot leak a sibling
  // chunk into validation or test.
  const syntheticGroupIds = new Set(
    orderedGroups
      .filter(([, chunks]) => chunks.some((chunk) => ['ollama', 'generated'].includes(chunk.provenance?.kind ?? 'unknown')))
      .map(([groupId]) => groupId),
  )
  const holdoutEligibleGroups = orderedGroups.filter(([groupId]) => !syntheticGroupIds.has(groupId))
  const counts = allocateSplitCounts(holdoutEligibleGroups.length, config)
  const assignedBefore = new Map<string, DatasetSplit>()
  for (const assignment of previousAssignments) {
    const existing = assignedBefore.get(assignment.independenceGroupId)
    if (
      !assignment.independenceGroupId || !['train', 'validation', 'test'].includes(assignment.split) ||
      (existing && existing !== assignment.split)
    ) throw new TypeError('Önceki kaynak grubu holdout atamaları bozuk veya çelişkili.')
    assignedBefore.set(assignment.independenceGroupId, assignment.split)
  }
  const groupAssignments = new Map<string, DatasetSplit>()
  const current: Record<DatasetSplit, number> = { train: 0, validation: 0, test: 0 }
  for (const groupId of syntheticGroupIds) groupAssignments.set(groupId, 'train')
  for (const [groupId] of holdoutEligibleGroups) {
    const fixed = assignedBefore.get(groupId)
    if (!fixed) continue
    groupAssignments.set(groupId, fixed)
    current[fixed] += 1
  }
  const splitOrder: DatasetSplit[] = ['train', 'validation', 'test']
  for (const [groupId] of holdoutEligibleGroups) {
    if (groupAssignments.has(groupId)) continue
    const selected = [...splitOrder].sort((left, right) => {
      const leftDeficit = counts[left] - current[left]
      const rightDeficit = counts[right] - current[right]
      return rightDeficit - leftDeficit || splitOrder.indexOf(left) - splitOrder.indexOf(right)
    })[0]
    groupAssignments.set(groupId, selected)
    current[selected] += 1
  }
  const assigned = orderedGroups.flatMap(([_, chunks]) => chunks.map((source): RetainedSource => ({
    ...source,
    split: groupAssignments.get(source.independenceGroupId) ?? 'train',
  })))
    .sort((left, right) => compareText(left.sourceId, right.sourceId) || compareText(left.sourceDigest, right.sourceDigest))
  const warnings: string[] = []
  if (holdoutEligibleGroups.length < 3) {
    warnings.push('Üçten az bağımsız gerçek kaynak grubunda eğitim/doğrulama/test bölümlerinin tamamı doldurulamaz.')
  }
  if (syntheticGroupIds.size) {
    const overridden = [...syntheticGroupIds].filter((groupId) => {
      const previous = assignedBefore.get(groupId)
      return previous !== undefined && previous !== 'train'
    }).length
    warnings.push(`${syntheticGroupIds.size} sentetik kaynak grubu yalnız eğitim bölümünde tutuldu; doğrulama ve teste alınmadı.`)
    if (overridden) warnings.push(`${overridden} eski sentetik holdout ataması güvenli eğitim atamasıyla değiştirildi.`)
  }
  if (duplicates.length) {
    warnings.push(`${duplicates.length} yinelenen veya çakışan kaynak sızıntı korumasıyla dışlandı.`)
  }
  return { sources: assigned, duplicates, warnings }
}

function cloneBindings(bindings: readonly SurfaceBinding[]): SurfaceBinding[] {
  return bindings.map((binding) => ({
    ...binding,
    familyWeights: binding.familyWeights.map((entry) => ({ ...entry })),
  }))
}

function examplesForSource(
  pieces: readonly SurfacePiece[],
  source: RetainedSource,
  config: LanguageExperienceConfig,
): LanguageExample[] {
  const byId = new Map(pieces.map((piece) => [piece.id, piece]))
  const examples: LanguageExample[] = []
  for (const parent of pieces) {
    if (!parent.childIds.length) continue
    const children = parent.childIds.map((id) => byId.get(id)).filter((piece): piece is SurfacePiece => Boolean(piece))
    if (!children.length || !config.includeHierarchyLevels.includes(children[0].level)) continue
    for (let targetIndex = 0; targetIndex < children.length; targetIndex += 1) {
      const target = children[targetIndex]
      const prefix = children.slice(Math.max(0, targetIndex - config.maximumPrefixPieces), targetIndex)
      examples.push({
        id: languageDigest('example', source.sourceDigest, parent.id, target.id, targetIndex),
        sourceId: source.sourceId,
        sourceDigest: source.sourceDigest,
        split: source.split,
        granularity: target.level,
        parentPieceId: parent.id,
        prefixPieceIds: prefix.map((piece) => piece.id),
        targetPieceId: target.id,
        targetText: target.text,
        targetNormalizedText: target.normalizedText,
        targetBindings: cloneBindings(target.bindings),
        eos: false,
        corruptionBindings: [],
      })
    }
    const prefix = children.slice(Math.max(0, children.length - config.maximumPrefixPieces))
    examples.push({
      id: languageDigest('example-eos', source.sourceDigest, parent.id, children.length),
      sourceId: source.sourceId,
      sourceDigest: source.sourceDigest,
      split: source.split,
      granularity: children[0].level,
      parentPieceId: parent.id,
      prefixPieceIds: prefix.map((piece) => piece.id),
      targetPieceId: null,
      targetText: '',
      targetNormalizedText: '',
      targetBindings: [],
      eos: true,
      corruptionBindings: [],
    })
  }
  const capped = examples.length <= config.maximumExamplesPerSource
    ? examples
    : deterministicOrder(examples, config.seed, (example) => example.id).slice(0, config.maximumExamplesPerSource)
  return capped.sort((left, right) => compareText(left.id, right.id))
}

export function attachCorruptions(examples: readonly LanguageExample[], config: LanguageExperienceConfig): LanguageExample[] {
  const pools = new Map<string, LanguageExample[]>()
  for (const example of examples) {
    if (example.eos || !example.targetBindings.length) continue
    const key = `${example.split}\u0000${example.granularity}`
    const values = pools.get(key) ?? []
    values.push(example)
    pools.set(key, values)
  }
  for (const values of pools.values()) values.sort((left, right) => compareText(left.id, right.id))

  return examples.map((example) => {
    if (example.eos) return { ...example, corruptionBindings: [] }
    const pool = pools.get(`${example.split}\u0000${example.granularity}`) ?? []
    const alternative = deterministicFirst(
      pool,
      config.seed,
      (candidate) => `${example.id}\u0000${candidate.id}`,
      (candidate) => (
        candidate.id !== example.id &&
        candidate.targetNormalizedText !== example.targetNormalizedText &&
        candidate.targetBindings.some((binding) => !example.targetBindings.some((own) => own.memberKey === binding.memberKey))
      ),
    )
    return { ...example, corruptionBindings: cloneBindings(alternative?.targetBindings ?? []) }
  })
}

export function buildLanguageDataset(
  familyRun: FamilyRun,
  sourceInputs: readonly LanguageCorpusSource[],
  config: LanguageExperienceConfig,
  previousAssignments: readonly SourceGroupSplitAssignment[] = [],
): LanguageDataset {
  validateLanguageAdapterInput(familyRun, sourceInputs)
  const identitySnapshot = snapshotFamilyIdentities(familyRun)
  const split = splitLanguageSources(sourceInputs, config, previousAssignments)
  const allPieces: SurfacePiece[] = []
  const allExamples: LanguageExample[] = []
  for (const source of split.sources) {
    const hierarchy = buildSurfaceHierarchy(source, config)
    const pieces = bindSurfacePieces(hierarchy, familyRun, config)
    allPieces.push(...pieces)
    allExamples.push(...examplesForSource(pieces, source, config))
  }
  const examples = attachCorruptions(allExamples, config)
    .sort((left, right) => compareText(left.sourceDigest, right.sourceDigest) || compareText(left.id, right.id))
  const configFingerprint = languageDigest('config', config)
  const fingerprint = languageDigest('dataset',
    familyRun.fingerprint,
    configFingerprint,
    split.sources.map((source) => [source.canonicalFingerprint, source.independenceGroupId, source.split]),
    examples.map((example) => example.id),
  )
  const warnings = [...split.warnings]
  const unbound = examples.filter((example) => !example.eos && !example.targetBindings.length).length
  if (unbound) warnings.push(`${unbound} hedef yüzey mevcut dondurulmuş nöronlara güvenle bağlanamadı; yalnız doğum gözlemi olabilir.`)
  const stats = Object.fromEntries((['train', 'validation', 'test'] as const).map((datasetSplit) => {
    const sources = split.sources.filter((source) => source.split === datasetSplit)
    return [datasetSplit, {
      sourceCount: sources.length,
      chunkCount: sources.length,
      independenceGroupCount: new Set(sources.map((source) => source.independenceGroupId)).size,
      exampleCount: examples.filter((example) => example.split === datasetSplit).length,
      characterCount: sources.reduce((sum, source) => sum + source.text.length, 0),
    }]
  })) as LanguageDataset['stats']
  assertFamilyIdentitySnapshot(familyRun, identitySnapshot)
  return {
    format: 'language-experience-dataset',
    version: 1,
    id: `LD-${fingerprint.slice(-20)}`,
    fingerprint,
    representationSpaceId: familyRun.representationSpaceId,
    behaviorSpaceId: familyRun.behaviorSpaceId,
    familyRunFingerprint: familyRun.fingerprint,
    configFingerprint,
    sources: split.sources,
    pieces: allPieces,
    examples,
    duplicates: split.duplicates,
    stats,
    warnings,
  }
}
