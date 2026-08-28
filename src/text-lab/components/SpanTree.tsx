import { ChevronDown, ChevronRight, FileText, GitBranch, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { RawTextNeuron } from '../domain/types'

interface SpanTreeProps {
  nodes: readonly RawTextNeuron[]
  rootIds: readonly string[]
  selectedId: string
  onSelect: (id: string) => void
}

function previewText(value: string): string {
  const visible = value.replace(/\n/g, ' ↵ ').replace(/\t/g, ' ⇥ ').replace(/ /g, '·')
  return visible.length > 34 ? `${visible.slice(0, 34)}…` : visible || '∅'
}

export function SpanTree({ nodes, rootIds, selectedId, onSelect }: SpanTreeProps) {
  const byId = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes])
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(rootIds))
  const [query, setQuery] = useState('')

  useEffect(() => {
    const lineage: string[] = []
    let current = byId.get(selectedId)
    while (current) {
      lineage.push(current.id)
      current = current.parentId ? byId.get(current.parentId) : undefined
    }
    setExpanded((previous) => new Set([...previous, ...lineage]))
  }, [byId, selectedId])

  const toggle = (id: string) => {
    setExpanded((previous) => {
      const next = new Set(previous)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const normalizedQuery = query.trim().toLocaleLowerCase('tr-TR')
  const matches = normalizedQuery
    ? nodes.filter((node) => node.text.toLocaleLowerCase('tr-TR').includes(normalizedQuery)).slice(0, 200)
    : null

  const renderNode = (id: string): React.ReactNode => {
    const node = byId.get(id)
    if (!node) return null
    const hasChildren = node.childIds.length > 0
    const isExpanded = expanded.has(id)
    return (
      <li key={id} role="treeitem" aria-expanded={hasChildren ? isExpanded : undefined}>
        <div className={`tl-tree-row ${selectedId === id ? 'is-selected' : ''}`}>
          <button
            className="tl-tree-toggle"
            onClick={() => hasChildren && toggle(id)}
            disabled={!hasChildren}
            aria-label={hasChildren ? `${id} dalını ${isExpanded ? 'daralt' : 'genişlet'}` : `${id} yaprak`}
          >
            {hasChildren ? isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} /> : <span />}
          </button>
          <button className="tl-tree-select" onClick={() => onSelect(id)} title={node.text}>
            {node.depth === 0 ? <FileText size={13} /> : <GitBranch size={13} />}
            <span>
              <b>{id.replace('HN-', '')}</b>
              <small>{previewText(node.text)}</small>
            </span>
            <em>{node.graphemeCount}</em>
          </button>
        </div>
        {hasChildren && isExpanded && (
          <ul role="group">{node.childIds.map(renderNode)}</ul>
        )}
      </li>
    )
  }

  return (
    <div className="tl-tree">
      <label className="tl-tree-search">
        <Search size={14} />
        <input aria-label="Ham nöron metninde ara" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Metinde ara" />
      </label>
      {matches ? (
        <div className="tl-tree-results">
          <span>{matches.length}{nodes.length > 200 ? '+' : ''} eşleşme</span>
          {matches.length ? matches.map((node) => (
            <button key={node.id} className={selectedId === node.id ? 'is-selected' : ''} onClick={() => onSelect(node.id)}>
              <b>{node.id}</b><small>{previewText(node.text)}</small>
            </button>
          )) : <p>Eşleşen ham nöron yok.</p>}
        </div>
      ) : (
        <ul className="tl-tree-root" role="tree" aria-label="Ham nöron bölünme ağacı">
          {rootIds.map(renderNode)}
        </ul>
      )}
    </div>
  )
}
