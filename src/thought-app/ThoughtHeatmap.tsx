import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import type { ThoughtMatrix } from './types'

interface ThoughtHeatmapProps {
  matrix: ThoughtMatrix
  title: string
  caption: string
}

function mix(start: readonly number[], end: readonly number[], amount: number): string {
  const ratio = Math.max(0, Math.min(1, amount))
  const channels = start.map((value, index) => Math.round(value + ((end[index] ?? value) - value) * ratio))
  return `rgb(${channels.join(' ')})`
}

function heatColor(value: number, domain: number): string {
  if (!Number.isFinite(value) || domain <= 0) return 'rgb(249 247 241)'
  const intensity = Math.min(1, Math.abs(value) / domain) ** 0.72
  return value < 0
    ? mix([249, 247, 241], [48, 82, 142], intensity)
    : mix([249, 247, 241], [190, 49, 56], intensity)
}

function matrixStats(matrix: ThoughtMatrix) {
  let active = 0
  let strongestIndex = 0
  let strongestMagnitude = -1
  matrix.values.forEach((value, index) => {
    const magnitude = Math.abs(value)
    if (magnitude > 1e-9) active += 1
    if (magnitude > strongestMagnitude) {
      strongestMagnitude = magnitude
      strongestIndex = index
    }
  })
  return {
    active,
    strongestX: strongestIndex % matrix.side,
    strongestY: Math.floor(strongestIndex / matrix.side),
    domain: Math.max(Math.abs(matrix.min), Math.abs(matrix.max), 1e-9),
  }
}

function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return '0'
  if (Math.abs(value) < .00005) return '0'
  return value.toLocaleString('tr-TR', { maximumFractionDigits: 4 })
}

function exactCellNumber(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : '0.0000'
}

export function ThoughtHeatmap({ matrix, title, caption }: ThoughtHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hovered, setHovered] = useState<{ x: number; y: number; value: number } | null>(null)
  const stats = useMemo(() => matrixStats(matrix), [matrix])

  const inspectCell = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const x = Math.min(matrix.side - 1, Math.max(0, Math.floor((event.clientX - bounds.left) / bounds.width * matrix.side)))
    const y = Math.min(matrix.side - 1, Math.max(0, Math.floor((event.clientY - bounds.top) / bounds.height * matrix.side)))
    setHovered({ x, y, value: matrix.values[y * matrix.side + x] ?? 0 })
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return

    const pixelSize = canvas.width / matrix.side
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.imageSmoothingEnabled = false
    matrix.values.forEach((value, index) => {
      const x = index % matrix.side
      const y = Math.floor(index / matrix.side)
      context.fillStyle = heatColor(value, stats.domain)
      context.fillRect(x * pixelSize, y * pixelSize, Math.ceil(pixelSize), Math.ceil(pixelSize))
    })

    context.strokeStyle = 'rgba(55, 51, 46, .12)'
    context.lineWidth = 1
    for (let index = 8; index < matrix.side; index += 8) {
      const offset = Math.round(index * pixelSize) + .5
      context.beginPath()
      context.moveTo(offset, 0)
      context.lineTo(offset, canvas.height)
      context.stroke()
      context.beginPath()
      context.moveTo(0, offset)
      context.lineTo(canvas.width, offset)
      context.stroke()
    }
  }, [matrix, stats.domain])

  return (
    <article className="tha-heatmap-card">
      <header>
        <div><span>{title}</span><strong>{caption}</strong></div>
        <small>{matrix.side} × {matrix.side}</small>
      </header>
      <div
        className="tha-heatmap-frame"
        role="img"
        aria-label={`${title}: ${matrix.side} çarpı ${matrix.side} yoğunluk haritası. ${stats.active} etkin kare.`}
      >
        <canvas
          ref={canvasRef}
          width={640}
          height={640}
          onPointerMove={inspectCell}
          onPointerLeave={() => setHovered(null)}
        />
        {stats.active === 0 && <span className="tha-heatmap-zero">Sinyal sakin</span>}
      </div>
      <footer>
        <span><b>Etkin kare</b>{stats.active.toLocaleString('tr-TR')}</span>
        <span><b>{hovered ? 'Seçili kare' : 'En güçlü nokta'}</b>{hovered?.x ?? stats.strongestX}, {hovered?.y ?? stats.strongestY} · {exactCellNumber(hovered?.value ?? matrix.values[stats.strongestY * matrix.side + stats.strongestX] ?? 0)}</span>
        <span><b>Ortalama</b>{compactNumber(matrix.mean)}</span>
      </footer>
      <div className="tha-heat-legend" aria-hidden="true">
        <small>− yoğunluk</small><i /><small>sakin</small><i /><small>+ yoğunluk</small>
      </div>
    </article>
  )
}
