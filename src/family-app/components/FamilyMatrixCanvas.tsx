import { useEffect, useId, useRef, useState } from 'react'

interface FamilyMatrixCanvasProps {
  values: ArrayLike<number>
  rowLabels: readonly string[]
  columnLabels: readonly string[]
  selectedRow: number
  selectedColumn: number
  onSelect: (row: number, column: number) => void
  label: string
}

interface MatrixCell {
  row: number
  column: number
  value: number
  x: number
  y: number
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0))

function colorFor(value: number): string {
  const intensity = clamp01(value)
  return `hsl(${238 - intensity * 12} ${20 + intensity * 37}% ${97 - intensity * 58}%)`
}

export function FamilyMatrixCanvas({
  values,
  rowLabels,
  columnLabels,
  selectedRow,
  selectedColumn,
  onSelect,
  label,
}: FamilyMatrixCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<MatrixCell | null>(null)
  const descriptionId = useId()
  const rows = rowLabels.length
  const columns = columnLabels.length

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || rows === 0 || columns === 0) return
    const context = canvas.getContext('2d')
    if (!context) return

    const draw = () => {
      const box = canvas.getBoundingClientRect()
      const width = Math.max(280, box.width)
      const height = Math.max(210, box.height)
      const ratio = Math.min(2, window.devicePixelRatio || 1)
      canvas.width = Math.round(width * ratio)
      canvas.height = Math.round(height * ratio)
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      context.clearRect(0, 0, width, height)

      const cellWidth = width / columns
      const cellHeight = height / rows
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          const value = Number(values[row * columns + column] ?? 0)
          context.fillStyle = colorFor(value)
          context.fillRect(
            column * cellWidth,
            row * cellHeight,
            Math.ceil(cellWidth + 0.2),
            Math.ceil(cellHeight + 0.2),
          )
        }
      }

      if (cellWidth >= 5 && cellHeight >= 4) {
        context.strokeStyle = 'rgb(52 49 42 / 11%)'
        context.lineWidth = 0.5
        for (let column = 0; column <= columns; column += 1) {
          const x = Math.round(column * cellWidth) + 0.5
          context.beginPath()
          context.moveTo(x, 0)
          context.lineTo(x, height)
          context.stroke()
        }
        for (let row = 0; row <= rows; row += 1) {
          const y = Math.round(row * cellHeight) + 0.5
          context.beginPath()
          context.moveTo(0, y)
          context.lineTo(width, y)
          context.stroke()
        }
      }

      if (selectedRow >= 0 && selectedRow < rows && selectedColumn >= 0 && selectedColumn < columns) {
        context.strokeStyle = '#f0a119'
        context.lineWidth = 2
        context.strokeRect(
          selectedColumn * cellWidth + 1,
          selectedRow * cellHeight + 1,
          Math.max(1, cellWidth - 2),
          Math.max(1, cellHeight - 2),
        )
      }
      context.strokeStyle = '#bbb7ad'
      context.lineWidth = 1
      context.strokeRect(0.5, 0.5, width - 1, height - 1)
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [columns, rows, selectedColumn, selectedRow, values])

  const locate = (clientX: number, clientY: number): MatrixCell | null => {
    const canvas = canvasRef.current
    if (!canvas || rows === 0 || columns === 0) return null
    const box = canvas.getBoundingClientRect()
    const column = Math.min(columns - 1, Math.max(0, Math.floor(((clientX - box.left) / box.width) * columns)))
    const row = Math.min(rows - 1, Math.max(0, Math.floor(((clientY - box.top) / box.height) * rows)))
    return {
      row,
      column,
      value: Number(values[row * columns + column] ?? 0),
      x: clientX - box.left,
      y: clientY - box.top,
    }
  }

  const selectedValue = rows && columns && selectedRow >= 0 && selectedColumn >= 0
    ? Number(values[selectedRow * columns + selectedColumn] ?? 0)
    : 0

  return (
    <div className="fa-membership-matrix">
      <div
        className="fa-membership-matrix__columns"
        aria-hidden="true"
        style={{ '--fa-matrix-columns': Math.max(1, columns) } as React.CSSProperties}
      >
        {columnLabels.map((column, index) => <span key={`${index}-${column}`} title={column}>{column}</span>)}
      </div>
      <div className="fa-membership-matrix__canvas">
        <canvas
          ref={canvasRef}
          role="img"
          tabIndex={0}
          aria-label={`${label}. Ok tuşlarıyla üyelik hücrelerini inceleyin.`}
          aria-describedby={descriptionId}
          onPointerMove={(event) => setHover(locate(event.clientX, event.clientY))}
          onPointerLeave={() => setHover(null)}
          onClick={(event) => {
            const cell = locate(event.clientX, event.clientY)
            if (cell) onSelect(cell.row, cell.column)
          }}
          onKeyDown={(event) => {
            let row = selectedRow
            let column = selectedColumn
            if (event.key === 'ArrowUp') row = Math.max(0, row - 1)
            else if (event.key === 'ArrowDown') row = Math.min(rows - 1, row + 1)
            else if (event.key === 'ArrowLeft') column = Math.max(0, column - 1)
            else if (event.key === 'ArrowRight') column = Math.min(columns - 1, column + 1)
            else if (event.key === 'Home') { row = 0; column = 0 }
            else if (event.key === 'End') { row = Math.max(0, rows - 1); column = Math.max(0, columns - 1) }
            else return
            event.preventDefault()
            onSelect(row, column)
          }}
        />
        {hover && (
          <div
            className="fa-canvas-tooltip"
            style={{ left: Math.min(hover.x + 12, 420), top: Math.max(8, hover.y - 52) }}
          >
            <b>{rowLabels[hover.row]}</b>
            <span>{columnLabels[hover.column]}</span>
            <strong>{hover.value.toFixed(4)}</strong>
          </div>
        )}
      </div>
      <span id={descriptionId} className="fa-sr-only" aria-live="polite">
        {rows && columns
          ? `${rowLabels[selectedRow] ?? 'Nöron'}, ${columnLabels[selectedColumn] ?? 'aile'}, üyelik ${selectedValue.toFixed(4)}`
          : 'Üyelik matrisi boş.'}
      </span>
      <details className="fa-accessible-data">
        <summary>Üyelikleri erişilebilir tabloda göster</summary>
        <div>
          <table>
            <thead><tr><th>Nöron</th>{columnLabels.map((column, index) => <th key={`${index}-${column}`}>{column}</th>)}</tr></thead>
            <tbody>
              {rowLabels.map((rowLabel, row) => (
                <tr key={`${row}-${rowLabel}`}>
                  <th>{rowLabel}</th>
                  {columnLabels.map((columnLabel, column) => (
                    <td key={`${column}-${columnLabel}`}>{Number(values[row * columns + column] ?? 0).toFixed(4)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
