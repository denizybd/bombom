import { useEffect, useRef, useState } from 'react'

interface SimilarityCanvasProps {
  similarities: Float32Array
  count: number
  selectedRow: number
  selectedColumn: number | null
  labels: readonly string[]
  onSelect: (row: number, column: number) => void
}

interface HoverPair {
  row: number
  column: number
  value: number
  x: number
  y: number
}

function similarityColor(value: number, diagonal: boolean): [number, number, number] {
  if (diagonal) return [45, 50, 82]
  const intensity = Math.min(1, Math.abs(value))
  if (value >= 0) {
    return [250 - Math.round(165 * intensity), 247 - Math.round(210 * intensity), 243 - Math.round(195 * intensity)]
  }
  return [247 - Math.round(190 * intensity), 247 - Math.round(185 * intensity), 250 - Math.round(125 * intensity)]
}

export function SimilarityCanvas({
  similarities,
  count,
  selectedRow,
  selectedColumn,
  labels,
  onSelect,
}: SimilarityCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<HoverPair | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || count === 0) return
    const context = canvas.getContext('2d')
    if (!context) return
    const pixels = new Uint8ClampedArray(count * count * 4)
    for (let row = 0; row < count; row += 1) {
      for (let column = 0; column < count; column += 1) {
        const pixel = (row * count + column) * 4
        const [red, green, blue] = similarityColor(
          similarities[row * count + column],
          row === column,
        )
        pixels[pixel] = red
        pixels[pixel + 1] = green
        pixels[pixel + 2] = blue
        pixels[pixel + 3] = 255
      }
    }
    canvas.width = count
    canvas.height = count
    context.putImageData(new ImageData(pixels, count, count), 0, 0)

    context.strokeStyle = '#f0a119'
    context.lineWidth = Math.max(1, count / 220)
    context.strokeRect(0, selectedRow, count, 1)
    context.strokeRect(selectedRow, 0, 1, count)
    if (selectedColumn !== null) {
      context.strokeStyle = '#ffffff'
      context.lineWidth = Math.max(1, count / 180)
      context.strokeRect(selectedColumn, selectedRow, 1, 1)
    }
  }, [count, selectedColumn, selectedRow, similarities])

  const locate = (event: { clientX: number; clientY: number }) => {
    const canvas = canvasRef.current
    if (!canvas || count === 0) return null
    const box = canvas.getBoundingClientRect()
    const column = Math.min(count - 1, Math.max(0, Math.floor(((event.clientX - box.left) / box.width) * count)))
    const row = Math.min(count - 1, Math.max(0, Math.floor(((event.clientY - box.top) / box.height) * count)))
    return { row, column, box }
  }

  return (
    <div className="tl-similarity-canvas">
      <canvas
        ref={canvasRef}
        role="img"
        tabIndex={0}
        aria-label={`${count} ham nöronun tam benzerlik matrisi. Ok tuşlarıyla seçili çifti değiştirin.`}
        onPointerMove={(event) => {
          const located = locate(event)
          if (!located) return
          setHover({
            row: located.row,
            column: located.column,
            value: similarities[located.row * count + located.column],
            x: event.clientX - located.box.left,
            y: event.clientY - located.box.top,
          })
        }}
        onPointerLeave={() => setHover(null)}
        onClick={(event) => {
          const located = locate(event)
          if (located) onSelect(located.row, located.column)
        }}
        onKeyDown={(event) => {
          const column = selectedColumn ?? selectedRow
          if (event.key === 'ArrowUp') {
            event.preventDefault()
            onSelect(Math.max(0, selectedRow - 1), column)
          } else if (event.key === 'ArrowDown') {
            event.preventDefault()
            onSelect(Math.min(count - 1, selectedRow + 1), column)
          } else if (event.key === 'ArrowLeft') {
            event.preventDefault()
            onSelect(selectedRow, Math.max(0, column - 1))
          } else if (event.key === 'ArrowRight') {
            event.preventDefault()
            onSelect(selectedRow, Math.min(count - 1, column + 1))
          }
        }}
      />
      {hover && (
        <div className="tl-similarity-tooltip" style={{ left: Math.min(hover.x + 12, 360), top: Math.max(8, hover.y - 58) }}>
          <b>{labels[hover.row]} × {labels[hover.column]}</b>
          <span>K = {hover.value.toFixed(4)}</span>
        </div>
      )}
    </div>
  )
}
