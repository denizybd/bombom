import { useMemo } from 'react'

export interface FamilyGraphNode {
  id: string
  label: string
  region: string
  activation: number
  size: number
  status?: string
}

export interface FamilyGraphEdge {
  id: string
  from: string
  to: string
  weight: number
  kind?: 'transition' | 'temporary' | 'coactivation'
  label?: string
}

interface FamilyGraphProps {
  nodes: readonly FamilyGraphNode[]
  edges: readonly FamilyGraphEdge[]
  selectedId: string | null
  onSelect: (id: string) => void
  label: string
}

const REGION_ORDER = ['content', 'event', 'role', 'operator', 'unknown', 'goal', 'language', 'control', 'emergent']

function normalizeRegion(value: string): string {
  const key = value.toLocaleLowerCase('en-US')
  if (key.includes('içer') || key.includes('content')) return 'content'
  if (key.includes('olay') || key.includes('eylem') || key.includes('event')) return 'event'
  if (key.includes('rol') || key.includes('iliş') || key.includes('role')) return 'role'
  if (key.includes('operat') || key.includes('operator')) return 'operator'
  if (key.includes('belirsiz') || key.includes('unknown')) return 'unknown'
  if (key.includes('amaç') || key.includes('değer') || key.includes('goal')) return 'goal'
  if (key.includes('dil') || key.includes('language')) return 'language'
  if (key.includes('kontrol') || key.includes('control')) return 'control'
  return 'emergent'
}

function regionColor(region: string): string {
  switch (normalizeRegion(region)) {
    case 'content': return '#303a79'
    case 'event': return '#c72d3e'
    case 'role': return '#087d70'
    case 'operator': return '#b9780b'
    case 'unknown': return '#77736c'
    case 'goal': return '#7a478c'
    case 'language': return '#3b6b91'
    case 'control': return '#55514a'
    default: return '#77736c'
  }
}

export function FamilyGraph({ nodes, edges, selectedId, onSelect, label }: FamilyGraphProps) {
  const positions = useMemo(() => {
    const groups = new Map<string, FamilyGraphNode[]>()
    for (const node of nodes) {
      const region = normalizeRegion(node.region)
      const group = groups.get(region) ?? []
      group.push(node)
      groups.set(region, group)
    }
    const usedRegions = REGION_ORDER.filter((region) => groups.has(region))
    const coordinates = new Map<string, { x: number; y: number }>()
    usedRegions.forEach((region, regionIndex) => {
      const group = groups.get(region) ?? []
      const x = usedRegions.length <= 1 ? 500 : 105 + (regionIndex / (usedRegions.length - 1)) * 790
      group.forEach((node, nodeIndex) => {
        const y = group.length <= 1 ? 245 : 90 + (nodeIndex / (group.length - 1)) * 310
        coordinates.set(node.id, { x, y })
      })
    })
    return coordinates
  }, [nodes])

  return (
    <div className="fa-family-graph" aria-label={label}>
      <div className="fa-family-graph__stage">
        <svg viewBox="0 0 1000 500" aria-hidden="true">
          <defs>
            <marker id="fa-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
            </marker>
          </defs>
          {edges.map((edge) => {
            const from = positions.get(edge.from)
            const to = positions.get(edge.to)
            if (!from || !to || edge.weight <= 0) return null
            const dx = to.x - from.x
            const dy = to.y - from.y
            const length = Math.max(1, Math.hypot(dx, dy))
            const startX = from.x + (dx / length) * 54
            const startY = from.y + (dy / length) * 34
            const endX = to.x - (dx / length) * 58
            const endY = to.y - (dy / length) * 38
            const normalX = -dy / length
            const normalY = dx / length
            const curve = edge.from.localeCompare(edge.to) < 0 ? 24 : -24
            const middleX = (startX + endX) / 2 + normalX * curve
            const middleY = (startY + endY) / 2 + normalY * curve
            return (
              <path
                key={edge.id}
                d={`M ${startX} ${startY} Q ${middleX} ${middleY} ${endX} ${endY}`}
                className={`fa-family-graph__edge is-${edge.kind ?? 'transition'}`}
                style={{ '--fa-edge-weight': Math.min(1, Math.max(0, edge.weight)) } as React.CSSProperties}
                markerEnd="url(#fa-arrow)"
              />
            )
          })}
        </svg>
        {nodes.map((node) => {
          const position = positions.get(node.id) ?? { x: 500, y: 250 }
          const active = node.activation > 0.01
          return (
            <button
              key={node.id}
              className={`fa-family-node ${selectedId === node.id ? 'is-selected' : ''} ${active ? 'is-active' : ''}`}
              style={{
                left: `${position.x / 10}%`,
                top: `${position.y / 5}%`,
                '--fa-family-color': regionColor(node.region),
                '--fa-family-activation': Math.min(1, Math.max(0, node.activation)),
              } as React.CSSProperties}
              onClick={() => onSelect(node.id)}
              aria-label={`${node.label}, ${node.region}, etkinlik ${node.activation.toFixed(3)}`}
            >
              <span>{node.region}</span>
              <b>{node.label}</b>
              <small>{node.size} üye{active ? ` · A ${node.activation.toFixed(2)}` : ''}</small>
            </button>
          )
        })}
        {nodes.length === 0 && <div className="fa-family-graph__empty">Görüntülenecek aile yok.</div>}
      </div>
      <div className="fa-graph-legend" aria-label="Ağ çizgileri açıklaması">
        <span><i className="transition" /> Kalıcı aile geçişi G</span>
        <span><i className="temporary" /> Geçici olay/rol bağı</span>
        <span><i className="coactivation" /> Eşzamanlı etkinlik</span>
      </div>
      <details className="fa-accessible-data">
        <summary>Ağı erişilebilir tablolarda göster</summary>
        <div>
          <table>
            <thead><tr><th>Aile</th><th>İşlev bölgesi</th><th>Üye</th><th>Etkinlik</th><th>Durum</th></tr></thead>
            <tbody>{nodes.map((node) => <tr key={node.id}><th>{node.label}</th><td>{node.region}</td><td>{node.size}</td><td>{node.activation.toFixed(4)}</td><td>{node.status ?? '—'}</td></tr>)}</tbody>
          </table>
          <table>
            <thead><tr><th>Kaynak</th><th>Hedef</th><th>Tür</th><th>Ağırlık</th></tr></thead>
            <tbody>{edges.map((edge) => <tr key={edge.id}><th>{nodes.find((node) => node.id === edge.from)?.label ?? edge.from}</th><td>{nodes.find((node) => node.id === edge.to)?.label ?? edge.to}</td><td>{edge.kind ?? 'transition'}</td><td>{edge.weight.toFixed(4)}</td></tr>)}</tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
