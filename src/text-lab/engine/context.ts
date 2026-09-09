import type {
  ContextStepExplanation,
  MatrixLayout,
  RawTextNeuron,
  TextLabConfig,
} from '../domain/types'
import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  resolveTextLabRepresentationProfileId,
} from '../domain/config'
import { coprimeMultiplier, hashInteger } from './hash'
import { clamp, cosineRegion, vectorNorm } from './matrix'
import { projectIdentityToRelation } from './encode'
import { digestVector } from './representation'
import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_RELATION_LENGTH,
  relationIndexFromPage,
} from './relationPages'

type TraceOperator = 'parent' | 'left' | 'right'

const OPERATOR_SEEDS: Record<TraceOperator, number> = {
  parent: 0x243f6a88,
  left: 0x13198a2e,
  right: 0xa4093822,
}

export function transformRelation(
  relation: ArrayLike<number>,
  operator: TraceOperator,
  seed: number,
  pageAware = false,
): Float32Array {
  if (pageAware) return transformAdaptiveRelationPages(relation, operator, seed)
  const output = new Float32Array(relation.length)
  if (!relation.length) return output
  const operatorSeed = seed ^ OPERATOR_SEEDS[operator]
  const multiplier = coprimeMultiplier(relation.length, operatorSeed)
  const offset = hashInteger(relation.length, operatorSeed ^ 0x9e3779b9) % relation.length

  for (let index = 0; index < relation.length; index += 1) {
    const target = (Math.imul(index, multiplier) + offset) % relation.length
    const sign = (hashInteger(index, operatorSeed ^ 0x7f4a7c15) & 1) === 0 ? 1 : -1
    output[target] = relation[index] * sign
  }
  return output
}

/**
 * Adaptive-profile transform: permutes complete physical pages, then applies a
 * separate signed permutation inside each page. No cell crosses a page without
 * its complete source page, so 8×8 atoms remain meaningful to the later atlas.
 */
function transformAdaptiveRelationPages(
  relation: ArrayLike<number>,
  operator: TraceOperator,
  seed: number,
): Float32Array {
  if (relation.length !== ADAPTIVE_RELATION_LENGTH) {
    throw new RangeError(`Sayfa-duyarlı ilişki dönüşümü ${ADAPTIVE_RELATION_LENGTH} hücre gerektirir.`)
  }
  const output = new Float32Array(relation.length)
  const operatorSeed = seed ^ OPERATOR_SEEDS[operator]
  const pageMultiplier = coprimeMultiplier(ADAPTIVE_PAGE_COUNT, operatorSeed)
  const pageOffset = hashInteger(
    ADAPTIVE_PAGE_COUNT,
    operatorSeed ^ 0x9e3779b9,
  ) % ADAPTIVE_PAGE_COUNT

  for (let sourcePage = 0; sourcePage < ADAPTIVE_PAGE_COUNT; sourcePage += 1) {
    const targetPage = (
      Math.imul(sourcePage, pageMultiplier) + pageOffset
    ) % ADAPTIVE_PAGE_COUNT
    const pageSeed = operatorSeed ^ hashInteger(sourcePage, 0x6a09e667)
    const cellMultiplier = coprimeMultiplier(ADAPTIVE_CELLS_PER_PAGE, pageSeed)
    const cellOffset = hashInteger(
      ADAPTIVE_CELLS_PER_PAGE,
      pageSeed ^ 0xbb67ae85,
    ) % ADAPTIVE_CELLS_PER_PAGE

    for (let sourceCell = 0; sourceCell < ADAPTIVE_CELLS_PER_PAGE; sourceCell += 1) {
      const targetCell = (
        Math.imul(sourceCell, cellMultiplier) + cellOffset
      ) % ADAPTIVE_CELLS_PER_PAGE
      const sign = (hashInteger(sourceCell, pageSeed ^ 0x3c6ef372) & 1) === 0 ? 1 : -1
      output[relationIndexFromPage(targetPage, targetCell)] =
        relation[relationIndexFromPage(sourcePage, sourceCell)] * sign
    }
  }
  return output
}

function usesAdaptivePages(config: TextLabConfig): boolean {
  return resolveTextLabRepresentationProfileId(config) === ADAPTIVE_TEXT_LAB_PROFILE_ID
}

function traceGain(compatibility: number, minimumTrace: number): number {
  return minimumTrace + (1 - minimumTrace) * Math.max(0, compatibility)
}

function zeroTrace() {
  return {
    inheritedNorm: 0,
    parentIdentityCompatibility: 0,
    parentGain: 0,
    siblingId: null,
    siblingIdentityCompatibility: null,
    siblingGain: 0,
  }
}

export function propagateContext(
  nodes: RawTextNeuron[],
  rootIds: readonly string[],
  layout: MatrixLayout,
  config: TextLabConfig,
): RawTextNeuron[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))

  for (const rootId of rootIds) {
    const root = byId.get(rootId)
    if (!root) continue
    root.relation.fill(0)
    root.relationNorm = 0
    root.contextSupport = 0
    root.contextRevision = 0
    root.trace = zeroTrace()

    const queue = [...root.childIds]
    let queueIndex = 0
    while (queueIndex < queue.length) {
      const childId = queue[queueIndex]
      queueIndex += 1
      if (!childId) continue
      const child = byId.get(childId)
      if (!child || !child.parentId) continue
      const parent = byId.get(child.parentId)
      if (!parent) continue
      const siblingId = parent.childIds.find((id) => id !== child.id) ?? null
      const sibling = siblingId ? byId.get(siblingId) ?? null : null

      const parentIdentity = parent.identity
      const childIdentity = child.identity
      const parentCompatibility = cosineRegion(
        childIdentity,
        parentIdentity,
        0,
        layout.identityLength,
        child.identityNorm,
        parent.identityNorm,
      )
      const parentGain = traceGain(parentCompatibility, config.minimumTrace)
      const parentSignature = transformRelation(
        projectIdentityToRelation(parentIdentity, layout, config),
        'parent',
        config.hashSeed,
        usesAdaptivePages(config),
      )

      const parentRelation = parent.relation
      const nextRelation = new Float32Array(layout.relationLength)
      let siblingCompatibility: number | null = null
      let siblingGain = 0
      let siblingSignature: Float32Array | null = null

      if (sibling) {
        const siblingIdentity = sibling.identity
        siblingCompatibility = cosineRegion(
          childIdentity,
          siblingIdentity,
          0,
          layout.identityLength,
          child.identityNorm,
          sibling.identityNorm,
        )
        siblingGain = traceGain(siblingCompatibility, config.minimumTrace)
        const projectedSibling = projectIdentityToRelation(siblingIdentity, layout, config)
        const direction = child.branch === 'left' ? 'right' : 'left'
        siblingSignature = transformRelation(
          projectedSibling,
          direction,
          config.hashSeed,
          usesAdaptivePages(config),
        )
      }

      for (let index = 0; index < nextRelation.length; index += 1) {
        const inherited = config.contextRetention * parentRelation[index]
        const parentMark = config.parentTraceWeight * parentGain * parentSignature[index]
        const siblingMark = siblingSignature
          ? config.siblingTraceWeight * siblingGain * siblingSignature[index]
          : 0
        nextRelation[index] = clamp(inherited + parentMark + siblingMark)
      }

      child.relation = nextRelation
      child.relationNorm = vectorNorm(nextRelation)
      child.contextSupport = 1
      child.contextRevision = 1
      child.trace = {
        inheritedNorm: config.contextRetention * parent.relationNorm,
        parentIdentityCompatibility: parentCompatibility,
        parentGain,
        siblingId,
        siblingIdentityCompatibility: siblingCompatibility,
        siblingGain,
      }
      queue.push(...child.childIds)
    }
  }

  return nodes
}

export function explainContextStep(
  node: RawTextNeuron,
  nodes: readonly RawTextNeuron[],
  layout: MatrixLayout,
  config: TextLabConfig,
): ContextStepExplanation {
  const empty = new Float32Array(layout.relationLength)
  if (!node.parentId) {
    return { before: empty, trace: empty.slice(), after: empty.slice(), parentTrace: empty.slice(), siblingTrace: empty.slice() }
  }

  const byId = new Map(nodes.map((candidate) => [candidate.id, candidate]))
  const parent = byId.get(node.parentId)
  if (!parent) {
    return { before: empty, trace: empty.slice(), after: empty.slice(), parentTrace: empty.slice(), siblingTrace: empty.slice() }
  }

  const before = Float32Array.from(
    parent.relation,
    (value) => value * config.contextRetention,
  )
  const parentSignature = transformRelation(
    projectIdentityToRelation(parent.identity, layout, config),
    'parent',
    config.hashSeed,
    usesAdaptivePages(config),
  )
  const parentTrace = Float32Array.from(
    parentSignature,
    (value) => value * config.parentTraceWeight * node.trace.parentGain,
  )
  const siblingTrace = new Float32Array(layout.relationLength)
  if (node.trace.siblingId) {
    const sibling = byId.get(node.trace.siblingId)
    if (sibling) {
      const operator = node.branch === 'left' ? 'right' : 'left'
      const signature = transformRelation(
        projectIdentityToRelation(sibling.identity, layout, config),
        operator,
        config.hashSeed,
        usesAdaptivePages(config),
      )
      const gain = config.siblingTraceWeight * node.trace.siblingGain
      for (let index = 0; index < signature.length; index += 1) {
        siblingTrace[index] = signature[index] * gain
      }
    }
  }
  const trace = Float32Array.from(parentTrace, (value, index) => value + siblingTrace[index])
  return {
    before,
    trace,
    after: Float32Array.from(node.relation),
    parentTrace,
    siblingTrace,
  }
}

/**
 * Applies one experience to the plastic context without ever exposing identity
 * to mutation. Repeated evidence gains less influence as support accumulates.
 */
export function applyContextObservation(
  node: RawTextNeuron,
  observation: ArrayLike<number>,
  identityCompatibility: number,
  maximumAlpha = 0.5,
): number {
  if (observation.length !== node.relation.length) {
    throw new RangeError('Bağlam gözlemi nöronun ilişki alanıyla uyuşmuyor.')
  }
  const identityBefore = node.identityDigest
  if (digestVector(node.identity) !== identityBefore) {
    throw new TypeError('Nöronun sabit kimlik alanı güncellemeden önce bozulmuş.')
  }
  const evidence = clamp(identityCompatibility, 0, 1)
  if (evidence === 0) return 0
  const alpha = Math.min(maximumAlpha, evidence / Math.max(1, node.contextSupport + evidence))
  for (let index = 0; index < node.relation.length; index += 1) {
    node.relation[index] = clamp(node.relation[index] + alpha * (observation[index] - node.relation[index]))
  }
  node.contextSupport += evidence
  node.contextRevision += 1
  node.relationNorm = vectorNorm(node.relation)
  if (digestVector(node.identity) !== identityBefore) {
    throw new TypeError('Bağlam güncellemesi sabit kimlik alanını değiştirdi.')
  }
  return alpha
}
