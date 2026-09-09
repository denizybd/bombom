import { canonicalExpansion, compareMacroText, computeExpansionDigest } from './deterministic.ts'
import type { CompositeNeuronV1, MacroMemberReferenceV1 } from './types.ts'

interface WalkState {
  visiting: Set<string>
  memo: Map<string, number>
}

function allExpansions(composite: CompositeNeuronV1): readonly (readonly MacroMemberReferenceV1[])[] {
  return [composite.orderedExpansion, ...composite.variants.map((variant) => variant.orderedExpansion)]
}

function depthOf(
  id: string,
  catalog: ReadonlyMap<string, CompositeNeuronV1>,
  state: WalkState,
): number {
  const memoized = state.memo.get(id)
  if (memoized !== undefined) return memoized
  if (state.visiting.has(id)) throw new Error(`Makro genişletme döngüsü bulundu: ${id}`)
  const composite = catalog.get(id)
  if (!composite) throw new Error(`Makro genişletmesi bilinmeyen bileşik üyeye başvuruyor: ${id}`)
  state.visiting.add(id)
  let childDepth = 0
  for (const expansion of allExpansions(composite)) {
    for (const member of expansion) {
      if (member.kind !== 'composite-neuron') continue
      childDepth = Math.max(childDepth, depthOf(member.id, catalog, state))
    }
  }
  state.visiting.delete(id)
  const depth = 1 + childDepth
  state.memo.set(id, depth)
  return depth
}

export function computeCompositeDepth(
  compositeId: string,
  composites: readonly CompositeNeuronV1[],
): number {
  const catalog = new Map(composites.map((composite) => [composite.id, composite]))
  if (catalog.size !== composites.length) throw new Error('Bileşik nöron kimlikleri benzersiz olmalıdır.')
  return depthOf(compositeId, catalog, { visiting: new Set(), memo: new Map() })
}

/** Validates reference existence, acyclicity and the hard V1 depth ceiling. */
export function validateCompositeDag(
  composites: readonly CompositeNeuronV1[],
  maximumDepth = 3,
): void {
  if (!Number.isInteger(maximumDepth) || maximumDepth < 1 || maximumDepth > 3) {
    throw new RangeError('Makro DAG derinliği 1 ile 3 arasında olmalıdır.')
  }
  const catalog = new Map<string, CompositeNeuronV1>()
  for (const composite of composites) {
    if (!composite.id || catalog.has(composite.id)) {
      throw new Error('Bileşik nöron kimlikleri boş olamaz ve benzersiz olmalıdır.')
    }
    for (const expansion of allExpansions(composite)) {
      if (expansion.length < 2 || expansion.length > 8) {
        throw new RangeError(`Makro ${composite.id} varyantları 2 ile 8 doğrudan üye taşımalıdır.`)
      }
      canonicalExpansion(expansion)
    }
    if (computeExpansionDigest(canonicalExpansion(composite.orderedExpansion)) !== composite.expansionDigest) {
      throw new Error(`Makro ${composite.id} genişletme digest'i ile uyuşmuyor.`)
    }
    if (!composite.rawConstituentsRetained || composite.physicalIdentityAllocated) {
      throw new Error(`Makro ${composite.id} V1 sanal-overlay değişmezlerini ihlal ediyor.`)
    }
    catalog.set(composite.id, composite)
  }
  for (const composite of composites) {
    for (const expansion of allExpansions(composite)) {
      for (const member of expansion) {
        if (member.kind !== 'composite-neuron') continue
        const child = catalog.get(member.id)
        if (!child) throw new Error(`Makro ${composite.id} bilinmeyen bileşik üyeye başvuruyor: ${member.id}`)
      }
    }
  }
  const state: WalkState = { visiting: new Set(), memo: new Map() }
  for (const composite of [...composites].sort((left, right) => compareMacroText(left.id, right.id))) {
    const depth = depthOf(composite.id, catalog, state)
    if (depth !== composite.depth) {
      throw new Error(`Makro ${composite.id} kayıtlı derinliği ${composite.depth}, hesaplanan ${depth}.`)
    }
    if (depth > maximumDepth) {
      throw new RangeError(`Makro ${composite.id} izin verilen ${maximumDepth} DAG derinliğini aşıyor.`)
    }
  }
  for (const composite of composites) {
    if (composite.lifecycle === 'deprecated') continue
    for (const expansion of allExpansions(composite)) {
      for (const member of expansion) {
        if (member.kind !== 'composite-neuron') continue
        const child = catalog.get(member.id)!
        if (member.identityDigest !== child.expansionDigest) {
          throw new Error(`Makro ${composite.id} nested üye digest'i güncel açılımla uyuşmuyor: ${member.id}`)
        }
      }
    }
  }
}

export function expandCompositeNeuron(
  compositeId: string,
  composites: readonly CompositeNeuronV1[],
  maximumDepth = 3,
  maximumExpandedConstituents = 512,
): MacroMemberReferenceV1[] {
  validateCompositeDag(composites, maximumDepth)
  const catalog = new Map(composites.map((composite) => [composite.id, composite]))
  const visiting = new Set<string>()
  const expand = (id: string, depth: number): MacroMemberReferenceV1[] => {
    if (depth > maximumDepth) throw new RangeError('Makro genişletme derinlik sınırını aştı.')
    if (visiting.has(id)) throw new Error(`Makro genişletme döngüsü bulundu: ${id}`)
    const composite = catalog.get(id)
    if (!composite) throw new Error(`Bilinmeyen bileşik nöron: ${id}`)
    if (composite.lifecycle === 'deprecated') throw new Error(`Emekli bileşik nöron genişletilemez: ${id}`)
    visiting.add(id)
    const result: MacroMemberReferenceV1[] = []
    for (const member of composite.orderedExpansion) {
      if (member.kind === 'raw-neuron') result.push({ ...member, familyIds: [...member.familyIds] })
      else {
        const child = catalog.get(member.id)
        if (!child || member.identityDigest !== child.expansionDigest) {
          throw new Error(`Nested makro açılım digest'i doğrulanamadı: ${member.id}`)
        }
        result.push(...expand(member.id, depth + 1))
      }
      if (result.length > maximumExpandedConstituents) {
        throw new RangeError('Makro genişletmesi yapılandırılmış ham üye sınırını aştı.')
      }
    }
    visiting.delete(id)
    return result
  }
  return expand(compositeId, 1)
}
