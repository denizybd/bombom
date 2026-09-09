import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import type { PlasticRegionAtlasView, RegionPageView } from './types'

interface Props {
  atlas: PlasticRegionAtlasView | null
  selectedRegionId: string | null
  onSelectRegion: (regionId: string | null) => void
}

export function regionHue(value: string): number {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0) % 360
}

function pageMap(atlas: PlasticRegionAtlasView | null) {
  return new Map((atlas?.pages ?? []).map((page) => [page.pageIndex, page]))
}

function strongest(page: RegionPageView | null | undefined) {
  return [...(page?.memberships ?? [])].sort((left, right) => right.weight - left.weight || left.regionId.localeCompare(right.regionId))[0] ?? null
}

export function RegionAtlasCanvas({ atlas, selectedRegionId, onSelectRegion }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredPage, setHoveredPage] = useState<number | null>(null)
  const pages = useMemo(() => pageMap(atlas), [atlas])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return
    const size = canvas.width
    const cell = size / 64
    context.clearRect(0, 0, size, size)
    context.fillStyle = '#101827'
    context.fillRect(0, 0, size, cell * 8)

    for (let pageIndex = 0; pageIndex < 56; pageIndex += 1) {
      const page = pages.get(pageIndex)
      const dominant = strongest(page)
      const pageRow = Math.floor(pageIndex / 8)
      const pageColumn = pageIndex % 8
      const x = pageColumn * cell * 8
      const y = (pageRow + 1) * cell * 8
      const selectedWeight = page?.memberships.find((entry) => entry.regionId === selectedRegionId)?.weight ?? 0
      const intensity = selectedRegionId ? selectedWeight : dominant?.weight ?? 0
      context.fillStyle = dominant
        ? `hsl(${regionHue(dominant.regionId)} 62% ${94 - intensity * 47}%)`
        : '#f4f1ea'
      context.fillRect(x, y, cell * 8, cell * 8)
      const freeMass = page?.freeMass ?? 1
      if (freeMass > .01) {
        context.strokeStyle = `rgba(42, 47, 58, ${Math.min(.28, freeMass * .22)})`
        context.lineWidth = Math.max(1, cell * .18)
        for (let diagonal = -8; diagonal < 16; diagonal += 2) {
          context.beginPath()
          context.moveTo(x + Math.max(0, diagonal) * cell, y + Math.max(0, -diagonal) * cell)
          context.lineTo(x + Math.min(8, diagonal + 8) * cell, y + Math.min(8, 8 - diagonal) * cell)
          context.stroke()
        }
      }
      context.strokeStyle = selectedWeight > 0 ? '#ff5d57' : hoveredPage === pageIndex ? '#272d3d' : 'rgba(35,42,58,.24)'
      context.lineWidth = selectedWeight > 0 ? 4 : hoveredPage === pageIndex ? 3 : 1.5
      context.strokeRect(x + 1, y + 1, cell * 8 - 2, cell * 8 - 2)
      context.fillStyle = intensity > .58 ? 'rgba(255,255,255,.88)' : 'rgba(25,31,44,.68)'
      context.font = `600 ${Math.max(10, Math.round(cell * 1.35))}px ui-monospace, monospace`
      context.fillText(String(pageIndex).padStart(2, '0'), x + cell * .75, y + cell * 1.8)
    }

    context.strokeStyle = '#7080ff'
    context.lineWidth = 5
    context.beginPath()
    context.moveTo(0, cell * 8)
    context.lineTo(size, cell * 8)
    context.stroke()
    context.fillStyle = 'rgba(255,255,255,.78)'
    context.font = `700 ${Math.max(12, Math.round(cell * 1.6))}px system-ui`
    context.fillText('I · DEĞİŞMEZ KİMLİK', cell * 1.2, cell * 4.8)
  }, [atlas, hoveredPage, pages, selectedRegionId])

  const point = (event: ReactPointerEvent<HTMLCanvasElement> | ReactMouseEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const column = Math.max(0, Math.min(63, Math.floor((event.clientX - bounds.left) / bounds.width * 64)))
    const row = Math.max(0, Math.min(63, Math.floor((event.clientY - bounds.top) / bounds.height * 64)))
    if (row < 8) return null
    return Math.floor((row - 8) / 8) * 8 + Math.floor(column / 8)
  }
  const hovered = hoveredPage === null ? null : pages.get(hoveredPage)
  const dominant = strongest(hovered)
  const selectPage = (pageIndex: number | null) => {
    const page = pageIndex === null ? undefined : pages.get(pageIndex)
    onSelectRegion(strongest(page)?.regionId ?? null)
  }
  const moveKeyboardCursor = (event: ReactKeyboardEvent<HTMLCanvasElement>) => {
    const current = hoveredPage ?? 0
    let next = current
    if (event.key === 'ArrowLeft') next = Math.max(0, current - 1)
    else if (event.key === 'ArrowRight') next = Math.min(55, current + 1)
    else if (event.key === 'ArrowUp') next = Math.max(0, current - 8)
    else if (event.key === 'ArrowDown') next = Math.min(55, current + 8)
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = 55
    else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      selectPage(current)
      return
    } else if (event.key === 'Escape') {
      event.preventDefault()
      onSelectRegion(null)
      return
    } else return
    event.preventDefault()
    setHoveredPage(next)
  }

  return (
    <section className="pra-map-card">
      <div className="pra-map-copy">
        <div><span>FİZİKSEL HARİTA</span><h2>8 kimlik satırı · 56 plastik mikro-sayfa</h2></div>
        <small>Taralı alan henüz sahiplenilmemiş kapasiteyi gösterir.</small>
      </div>
      <div className="pra-map-frame">
        <canvas
          ref={canvasRef}
          width={768}
          height={768}
          tabIndex={0}
          aria-describedby="pra-map-help"
          aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Home End Enter Space Escape"
          aria-label={`Plastik bölge atlasının 64 çarpı 64 fiziksel haritası. ${atlas ? `${atlas.regions.length} bölge yüklü.` : 'Henüz canlı atlas yok.'}`}
          onFocus={() => setHoveredPage((current) => current ?? 0)}
          onBlur={() => setHoveredPage(null)}
          onKeyDown={moveKeyboardCursor}
          onPointerMove={(event) => setHoveredPage(point(event))}
          onPointerLeave={() => setHoveredPage(null)}
          onClick={(event) => selectPage(point(event))}
        />
        {hoveredPage !== null && (
          <div className="pra-map-tooltip">
            <b>Mikro-sayfa {hoveredPage}</b>
            <span>{dominant ? `${dominant.regionId} · %${Math.round(dominant.weight * 100)}` : 'Henüz bölge yok'}</span>
            <small>Serbest kapasite %{Math.round((hovered?.freeMass ?? 1) * 100)}</small>
          </div>
        )}
      </div>
      <p className="pra-map-help" id="pra-map-help">Klavye: ok tuşlarıyla mikro-sayfalar arasında ilerle, Enter ile baskın bölgeyi seç, Escape ile seçimi temizle.</p>
    </section>
  )
}
