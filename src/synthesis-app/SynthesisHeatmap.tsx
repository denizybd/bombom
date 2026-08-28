import { useEffect, useId, useMemo, useRef, useState } from 'react'
import type { MatrixLayout } from '../text-lab/engine'

type HeatmapFocus = 'identity' | 'relation'

interface HoveredCell {
  row: number
  column: number
  value: number
  x: number
  y: number
}

interface SynthesisHeatmapProps {
  identity: ArrayLike<number>
  relation: ArrayLike<number>
  layout: MatrixLayout
  focus: HeatmapFocus
  label: string
}

function colorFor(value: number, active: boolean): string {
  if (!active) return '#efede6'
  const intensity = Math.min(1, Math.abs(value))
  if (intensity < 0.002) return '#fffefa'
  if (value >= 0) return `hsl(355 72% ${98 - intensity * 52}%)`
  return `hsl(233 35% ${97 - intensity * 58}%)`
}

export function SynthesisHeatmap({
  identity,
  relation,
  layout,
  focus,
  label,
}: SynthesisHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const descriptionId = useId()
  const [hover, setHover] = useState<HoveredCell | null>(null)
  const values = useMemo(() => {
    const matrix = new Float32Array(layout.cellCount)
    matrix.set(Array.from(identity).slice(0, layout.identityLength), 0)
    matrix.set(Array.from(relation).slice(0, layout.relationLength), layout.relationStart)
    return matrix
  }, [identity, layout, relation])

  const summary = useMemo(() => {
    const vector = focus === 'identity' ? identity : relation
    let minimum = Number.POSITIVE_INFINITY
    let maximum = Number.NEGATIVE_INFINITY
    let squared = 0
    for (let index = 0; index < vector.length; index += 1) {
      const value = Number(vector[index] ?? 0)
      minimum = Math.min(minimum, value)
      maximum = Math.max(maximum, value)
      squared += value * value
    }
    return {
      minimum: Number.isFinite(minimum) ? minimum : 0,
      maximum: Number.isFinite(maximum) ? maximum : 0,
      norm: Math.sqrt(squared),
    }
  }, [focus, identity, relation])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return

    const draw = () => {
      const box = canvas.getBoundingClientRect()
      const size = Math.max(180, box.width)
      const pixelRatio = Math.min(2, window.devicePixelRatio || 1)
      canvas.width = Math.round(size * pixelRatio)
      canvas.height = Math.round(size * pixelRatio)
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
      context.clearRect(0, 0, size, size)
      const cell = size / layout.side

      for (let row = 0; row < layout.side; row += 1) {
        for (let column = 0; column < layout.side; column += 1) {
          const index = row * layout.side + column
          const isIdentity = index < layout.identityLength
          const active = focus === 'identity' ? isIdentity : !isIdentity
          context.fillStyle = colorFor(values[index] ?? 0, active)
          context.fillRect(column * cell, row * cell, Math.ceil(cell + 0.2), Math.ceil(cell + 0.2))
        }
      }

      if (cell >= 5) {
        context.strokeStyle = 'rgb(57 52 45 / 10%)'
        context.lineWidth = 0.5
        for (let index = 0; index <= layout.side; index += 1) {
          const position = Math.round(index * cell) + 0.5
          context.beginPath()
          context.moveTo(position, 0)
          context.lineTo(position, size)
          context.stroke()
          context.beginPath()
          context.moveTo(0, position)
          context.lineTo(size, position)
          context.stroke()
        }
      }

      const boundary = layout.identityRows * cell
      context.strokeStyle = '#303a79'
      context.lineWidth = 2
      context.setLineDash([5, 4])
      context.beginPath()
      context.moveTo(0, boundary)
      context.lineTo(size, boundary)
      context.stroke()
      context.setLineDash([])
      context.strokeStyle = '#bbb7ad'
      context.lineWidth = 1
      context.strokeRect(0.5, 0.5, size - 1, size - 1)
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [focus, layout, values])

  const handlePointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const box = canvas.getBoundingClientRect()
    const column = Math.min(
      layout.side - 1,
      Math.max(0, Math.floor(((event.clientX - box.left) / box.width) * layout.side)),
    )
    const row = Math.min(
      layout.side - 1,
      Math.max(0, Math.floor(((event.clientY - box.top) / box.height) * layout.side)),
    )
    const index = row * layout.side + column
    setHover({
      row,
      column,
      value: values[index] ?? 0,
      x: event.clientX - box.left,
      y: event.clientY - box.top,
    })
  }

  return (
    <div className="ns-heatmap">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={label}
        aria-describedby={descriptionId}
        onPointerMove={handlePointer}
        onPointerLeave={() => setHover(null)}
      />
      <div className="ns-heatmap__zones" aria-hidden="true">
        <span className={focus === 'identity' ? 'is-active' : ''}>Ⅰ · SABİT KİMLİK</span>
        <span className={focus === 'relation' ? 'is-active' : ''}>R · PLASTİK BAĞLAM</span>
      </div>
      {hover && (
        <div
          className="ns-heatmap__tooltip"
          style={{ left: Math.min(hover.x + 12, 182), top: Math.max(8, hover.y - 46) }}
        >
          <span>[{hover.row}, {hover.column}]</span>
          <b>{hover.value.toFixed(4)}</b>
        </div>
      )}
      <span id={descriptionId} className="ns-sr-only">
        En düşük {summary.minimum.toFixed(4)}, en yüksek {summary.maximum.toFixed(4)}, norm {summary.norm.toFixed(4)}.
      </span>
    </div>
  )
}
