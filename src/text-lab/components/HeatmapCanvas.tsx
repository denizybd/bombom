import { useEffect, useId, useRef, useState } from 'react'
import type { MatrixLayout } from '../domain/types'

export type HeatmapZone = 'all' | 'identity' | 'relation'

interface HoverCell {
  row: number
  column: number
  index: number
  value: number
  zone: 'Kimlik' | 'İlişki'
  x: number
  y: number
}

interface HeatmapCanvasProps {
  matrix: ArrayLike<number>
  layout: MatrixLayout
  zone?: HeatmapZone
  label: string
  compact?: boolean
}

function cellColor(value: number, active: boolean): string {
  if (!active) return '#f0eee8'
  const intensity = Math.min(1, Math.abs(value))
  if (intensity < 0.002) return '#fffefa'
  if (value >= 0) {
    const lightness = 98 - intensity * 52
    return `hsl(355 72% ${lightness}%)`
  }
  const lightness = 97 - intensity * 58
  return `hsl(233 35% ${lightness}%)`
}

export function HeatmapCanvas({
  matrix,
  layout,
  zone = 'all',
  label,
  compact = false,
}: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<HoverCell | null>(null)
  const [keyboardIndex, setKeyboardIndex] = useState(0)
  const [keyboardActive, setKeyboardActive] = useState(false)
  const descriptionId = useId()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return

    const draw = () => {
      const box = canvas.getBoundingClientRect()
      const logicalSize = Math.max(120, box.width)
      const ratio = Math.min(2, window.devicePixelRatio || 1)
      canvas.width = Math.round(logicalSize * ratio)
      canvas.height = Math.round(logicalSize * ratio)
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      context.clearRect(0, 0, logicalSize, logicalSize)

      const cell = logicalSize / layout.side
      for (let row = 0; row < layout.side; row += 1) {
        for (let column = 0; column < layout.side; column += 1) {
          const index = row * layout.side + column
          const isIdentity = index < layout.identityLength
          const active = zone === 'all' || (zone === 'identity' ? isIdentity : !isIdentity)
          context.fillStyle = cellColor(matrix[index] ?? 0, active)
          context.fillRect(column * cell, row * cell, Math.ceil(cell + 0.2), Math.ceil(cell + 0.2))

          if (active && (matrix[index] ?? 0) < -0.04 && cell >= 4) {
            context.strokeStyle = `rgb(255 255 255 / ${0.22 + Math.abs(matrix[index] ?? 0) * 0.35})`
            context.lineWidth = Math.max(0.45, cell * 0.06)
            context.beginPath()
            context.moveTo(column * cell, (row + 1) * cell)
            context.lineTo((column + 1) * cell, row * cell)
            context.stroke()
          }
        }
      }

      if (cell >= 5) {
        context.strokeStyle = 'rgb(57 52 45 / 10%)'
        context.lineWidth = 0.5
        for (let index = 0; index <= layout.side; index += 1) {
          const position = Math.round(index * cell) + 0.5
          context.beginPath()
          context.moveTo(position, 0)
          context.lineTo(position, logicalSize)
          context.stroke()
          context.beginPath()
          context.moveTo(0, position)
          context.lineTo(logicalSize, position)
          context.stroke()
        }
      }

      const boundary = layout.identityRows * cell
      context.strokeStyle = '#303a79'
      context.lineWidth = compact ? 1.5 : 2
      context.setLineDash([5, 4])
      context.beginPath()
      context.moveTo(0, boundary)
      context.lineTo(logicalSize, boundary)
      context.stroke()
      context.setLineDash([])
      context.strokeStyle = '#b9b4aa'
      context.lineWidth = 1
      context.strokeRect(0.5, 0.5, logicalSize - 1, logicalSize - 1)
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [compact, layout, matrix, zone])

  const handlePointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const box = canvas.getBoundingClientRect()
    const column = Math.min(layout.side - 1, Math.max(0, Math.floor(((event.clientX - box.left) / box.width) * layout.side)))
    const row = Math.min(layout.side - 1, Math.max(0, Math.floor(((event.clientY - box.top) / box.height) * layout.side)))
    const index = row * layout.side + column
    setHover({
      row,
      column,
      index,
      value: matrix[index] ?? 0,
      zone: index < layout.identityLength ? 'Kimlik' : 'İlişki',
      x: event.clientX - box.left,
      y: event.clientY - box.top,
    })
  }

  const keyboardCell: HoverCell | null = keyboardActive
    ? (() => {
        const index = Math.min(layout.cellCount - 1, Math.max(0, keyboardIndex))
        const row = Math.floor(index / layout.side)
        const column = index % layout.side
        const width = canvasRef.current?.getBoundingClientRect().width ?? 240
        return {
          row,
          column,
          index,
          value: matrix[index] ?? 0,
          zone: index < layout.identityLength ? 'Kimlik' : 'İlişki',
          x: ((column + 0.5) / layout.side) * width,
          y: ((row + 0.5) / layout.side) * width,
        }
      })()
    : null
  const activeCell = hover ?? keyboardCell

  return (
    <div className={`tl-heatmap ${compact ? 'tl-heatmap--compact' : ''}`}>
      <canvas
        ref={canvasRef}
        tabIndex={0}
        aria-label={`${label}. Ok tuşlarıyla hücre değerlerini inceleyin.`}
        aria-describedby={descriptionId}
        role="img"
        onPointerMove={handlePointer}
        onPointerLeave={() => setHover(null)}
        onFocus={() => setKeyboardActive(true)}
        onBlur={() => setKeyboardActive(false)}
        onKeyDown={(event) => {
          let next = keyboardIndex
          const row = Math.floor(keyboardIndex / layout.side)
          const column = keyboardIndex % layout.side
          if (event.key === 'ArrowUp') next = Math.max(0, row - 1) * layout.side + column
          else if (event.key === 'ArrowDown') next = Math.min(layout.side - 1, row + 1) * layout.side + column
          else if (event.key === 'ArrowLeft') next = row * layout.side + Math.max(0, column - 1)
          else if (event.key === 'ArrowRight') next = row * layout.side + Math.min(layout.side - 1, column + 1)
          else if (event.key === 'Home') next = 0
          else if (event.key === 'End') next = layout.cellCount - 1
          else return
          event.preventDefault()
          setKeyboardIndex(Math.min(layout.cellCount - 1, Math.max(0, next)))
        }}
      />
      <div className="tl-heatmap__zones" aria-hidden="true">
        <span>Ⅰ · KİMLİK</span>
        <span>R · İLİŞKİ</span>
      </div>
      {activeCell && (
        <div
          className="tl-heatmap__tooltip"
          style={{ left: Math.min(activeCell.x + 12, 190), top: Math.max(8, activeCell.y - 45) }}
        >
          <b>{activeCell.zone}</b>
          <span>[{activeCell.row}, {activeCell.column}]</span>
          <strong>{activeCell.value.toFixed(4)}</strong>
        </div>
      )}
      <span id={descriptionId} className="tl-sr-only" aria-live="polite">
        {activeCell
          ? `${activeCell.zone} alanı, satır ${activeCell.row}, sütun ${activeCell.column}, değer ${activeCell.value.toFixed(4)}`
          : label}
      </span>
    </div>
  )
}
