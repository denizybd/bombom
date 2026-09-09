import { compareText, stableDigest } from './sparse'

export interface SourceIdentityRef {
  /**
   * Explicit provenance lineage. Different documents produced by the same
   * synthetic campaign/model lineage are one independent witness even when
   * their source ids and content digests differ.
   */
  independenceGroupId?: string
  sourceId?: string
  sourceDigest?: string
}

/**
 * Conservative evidence-lineage equivalence index.
 *
 * Declared provenance groups are authoritative. Chunks are also the same
 * source when they share a source id, while exact duplicate chunks are joined
 * when they share a digest even if an importer gave them different ids.
 * Taking the transitive closure prevents correlated output or copied chunks
 * from becoming multiple independent witnesses.
 */
export interface IndependentSourceIndex {
  groupKey(ref: SourceIdentityRef): string
  representativeSourceId(ref: SourceIdentityRef): string
  groupCount: number
}

class DisjointSet {
  private readonly parent = new Map<string, string>()

  add(key: string): void {
    if (!this.parent.has(key)) this.parent.set(key, key)
  }

  root(key: string): string {
    this.add(key)
    const parent = this.parent.get(key)!
    if (parent === key) return key
    const root = this.root(parent)
    this.parent.set(key, root)
    return root
  }

  union(left: string, right: string): void {
    const leftRoot = this.root(left)
    const rightRoot = this.root(right)
    if (leftRoot === rightRoot) return
    const [root, child] = compareText(leftRoot, rightRoot) <= 0
      ? [leftRoot, rightRoot]
      : [rightRoot, leftRoot]
    this.parent.set(child, root)
  }

  keys(): string[] {
    return [...this.parent.keys()].sort(compareText)
  }
}

function sourceToken(value: string | undefined): string | null {
  const normalized = value?.trim()
  return normalized ? `source:${normalized}` : null
}

function digestToken(value: string | undefined): string | null {
  const normalized = value?.trim()
  return normalized ? `digest:${normalized}` : null
}

function declaredGroupToken(value: string | undefined): string | null {
  const normalized = value?.trim()
  return normalized ? `declared:${normalized}` : null
}

function pairKey(ref: SourceIdentityRef): string {
  return [
    ref.independenceGroupId?.trim() ?? '',
    ref.sourceId?.trim() ?? '',
    ref.sourceDigest?.trim() ?? '',
  ].join('\u0000')
}

export function createIndependentSourceIndex(
  input: readonly SourceIdentityRef[],
): IndependentSourceIndex {
  const records = [...input]
    .map((ref) => ({
      independenceGroupId: ref.independenceGroupId?.trim() ?? '',
      sourceId: ref.sourceId?.trim() ?? '',
      sourceDigest: ref.sourceDigest?.trim() ?? '',
    }))
    .filter((ref) => ref.independenceGroupId || ref.sourceId || ref.sourceDigest)
    .sort((left, right) =>
      compareText(left.independenceGroupId, right.independenceGroupId) ||
      compareText(left.sourceId, right.sourceId) ||
      compareText(left.sourceDigest, right.sourceDigest),
    )
  const disjoint = new DisjointSet()
  for (const record of records) {
    const declared = declaredGroupToken(record.independenceGroupId)
    const source = sourceToken(record.sourceId)
    const digest = digestToken(record.sourceDigest)
    if (declared) disjoint.add(declared)
    if (source) disjoint.add(source)
    if (digest) disjoint.add(digest)
    const tokens = [declared, source, digest].filter((token): token is string => Boolean(token))
    for (let index = 1; index < tokens.length; index += 1) disjoint.union(tokens[0], tokens[index])
  }

  const tokensByRoot = new Map<string, string[]>()
  for (const token of disjoint.keys()) {
    const root = disjoint.root(token)
    const tokens = tokensByRoot.get(root) ?? []
    tokens.push(token)
    tokensByRoot.set(root, tokens)
  }
  const groupByToken = new Map<string, string>()
  const representativeByGroup = new Map<string, string>()
  for (const tokens of tokensByRoot.values()) {
    tokens.sort(compareText)
    const declaredGroups = tokens.filter((token) => token.startsWith('declared:'))
    // An explicit provenance group must retain the same key when more outputs
    // from that lineage are added later. If exact duplicate evidence bridges
    // two declared groups, both declarations enter the conservative key.
    const identityTokens = declaredGroups.length ? declaredGroups : tokens
    const group = `ISG-${stableDigest([
      'independent-source-v2',
      declaredGroups.length ? 'declared' : 'inferred',
      ...identityTokens,
    ]).toUpperCase()}`
    const sourceIds = tokens
      .filter((token) => token.startsWith('source:'))
      .map((token) => token.slice('source:'.length))
      .sort(compareText)
    representativeByGroup.set(group, sourceIds[0] ?? tokens[0]?.slice(tokens[0].indexOf(':') + 1) ?? group)
    tokens.forEach((token) => groupByToken.set(token, group))
  }

  const resolve = (ref: SourceIdentityRef): string => {
    const declared = declaredGroupToken(ref.independenceGroupId)
    const source = sourceToken(ref.sourceId)
    const digest = digestToken(ref.sourceDigest)
    const known =
      (declared && groupByToken.get(declared)) ||
      (source && groupByToken.get(source)) ||
      (digest && groupByToken.get(digest))
    if (known) return known
    return `ISG-${stableDigest(['independent-source-v2', 'unseen', pairKey(ref)]).toUpperCase()}`
  }

  return {
    groupKey: resolve,
    representativeSourceId(ref) {
      const group = resolve(ref)
      const representative = representativeByGroup.get(group)
      if (representative) return representative
      const sourceId = ref.sourceId?.trim()
      if (sourceId) return sourceId
      return ref.sourceDigest?.trim() || group
    },
    groupCount: tokensByRoot.size,
  }
}
