import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";

import MatrixGrid, {
  MATRIX_SIZE,
  clampMatrixValue,
  formatMatrixValue,
  readMatrixValue,
  toNormalizedMatrix,
  type Matrix,
  type MatrixCellContext,
  type MatrixDisplayMode,
  type MutableMatrix,
} from "./MatrixGrid";

export type MatrixPaintTool = "positive" | "negative" | "erase";

export interface MatrixEditorProps {
  matrix: Matrix;
  onChange: (matrix: MutableMatrix) => void;
  tool: MatrixPaintTool;
  /** Called by the built-in tool buttons. Leave unset when tools are controlled elsewhere. */
  onToolChange?: (tool: MatrixPaintTool) => void;
  /** Controlled paint magnitude in the [0, 1] range. */
  intensity: number;
  /** Called by the built-in controlled range input. */
  onIntensityChange?: (intensity: number) => void;
  readonly?: boolean;
  label?: string;
  mode?: MatrixDisplayMode;
  showValues?: boolean;
  showControls?: boolean;
  className?: string;
}

const toolOptions: ReadonlyArray<{
  tool: MatrixPaintTool;
  label: string;
  symbol: string;
}> = [
  { tool: "positive", label: "Pozitif boya", symbol: "+" },
  { tool: "negative", label: "Negatif boya", symbol: "−" },
  { tool: "erase", label: "Sil", symbol: "○" },
];

const joinClassNames = (...classNames: Array<string | false | null | undefined>) =>
  classNames.filter(Boolean).join(" ");

const cellKey = (row: number, column: number) => `${row}:${column}`;

const getPaintValue = (tool: MatrixPaintTool, intensity: number): number => {
  if (tool === "erase") return 0;
  const magnitude = Math.abs(clampMatrixValue(intensity));
  return tool === "negative" ? -magnitude : magnitude;
};

/** A controlled 8x8 matrix painter with mouse, pen, touch and keyboard support. */
export function MatrixEditor({
  matrix,
  onChange,
  tool,
  onToolChange,
  intensity,
  onIntensityChange,
  readonly = false,
  label = "Matris düzenleyici",
  mode = "large",
  showValues = mode === "large",
  showControls = true,
  className,
}: MatrixEditorProps) {
  const editorRef = useRef<HTMLDivElement>(null);
  const activePointerRef = useRef<number | null>(null);
  const lastPaintedCellRef = useRef<string | null>(null);
  const matrixRef = useRef<Matrix>(matrix);
  const [activeCell, setActiveCell] = useState<Pick<MatrixCellContext, "row" | "column"> | null>(null);

  useEffect(() => {
    matrixRef.current = matrix;
  }, [matrix]);

  const finishPainting = useCallback((event?: PointerEvent) => {
    if (
      event &&
      activePointerRef.current !== null &&
      event.pointerId !== activePointerRef.current
    ) {
      return;
    }
    activePointerRef.current = null;
    lastPaintedCellRef.current = null;
  }, []);

  useEffect(() => {
    window.addEventListener("pointerup", finishPainting);
    window.addEventListener("pointercancel", finishPainting);
    return () => {
      window.removeEventListener("pointerup", finishPainting);
      window.removeEventListener("pointercancel", finishPainting);
    };
  }, [finishPainting]);

  const paintCell = useCallback(
    (row: number, column: number, selectedTool = tool) => {
      if (readonly) return;

      const key = cellKey(row, column);
      if (lastPaintedCellRef.current === key) return;

      const nextValue = getPaintValue(selectedTool, intensity);
      if (readMatrixValue(matrixRef.current, row, column) === nextValue) {
        lastPaintedCellRef.current = key;
        return;
      }

      const nextMatrix = toNormalizedMatrix(matrixRef.current);
      nextMatrix[row][column] = nextValue;
      matrixRef.current = nextMatrix;
      lastPaintedCellRef.current = key;
      onChange(nextMatrix);
    },
    [intensity, onChange, readonly, tool],
  );

  const handlePointerDown = useCallback(
    (
      cell: MatrixCellContext,
      event: ReactPointerEvent<HTMLButtonElement>,
    ) => {
      if (readonly || (event.pointerType === "mouse" && event.button !== 0)) return;
      event.preventDefault();
      setActiveCell({ row: cell.row, column: cell.column });
      activePointerRef.current = event.pointerId;
      lastPaintedCellRef.current = null;
      paintCell(cell.row, cell.column);

      // Touch browsers implicitly capture the pointer. Releasing it lets sibling
      // cells receive enter events; pointer-move hit testing below is a fallback.
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [paintCell, readonly],
  );

  const handlePointerEnter = useCallback(
    (
      cell: MatrixCellContext,
      event: ReactPointerEvent<HTMLButtonElement>,
    ) => {
      setActiveCell({ row: cell.row, column: cell.column });
      if (activePointerRef.current !== event.pointerId) return;
      if (event.pointerType === "mouse" && (event.buttons & 1) === 0) {
        finishPainting();
        return;
      }
      paintCell(cell.row, cell.column);
    },
    [finishPainting, paintCell],
  );

  const handlePointerMove = useCallback(
    (
      cell: MatrixCellContext,
      event: ReactPointerEvent<HTMLButtonElement>,
    ) => {
      if (activePointerRef.current !== event.pointerId) return;
      setActiveCell({ row: cell.row, column: cell.column });
      if (event.pointerType === "mouse" && (event.buttons & 1) === 0) {
        finishPainting();
        return;
      }

      const hit = document
        .elementFromPoint(event.clientX, event.clientY)
        ?.closest<HTMLElement>("[data-row][data-column]");
      if (hit && editorRef.current?.contains(hit)) {
        const row = Number(hit.dataset.row);
        const column = Number(hit.dataset.column);
        if (
          Number.isInteger(row) &&
          Number.isInteger(column) &&
          row >= 0 &&
          row < MATRIX_SIZE &&
          column >= 0 &&
          column < MATRIX_SIZE
        ) {
          paintCell(row, column);
          return;
        }
      }

      paintCell(cell.row, cell.column);
    },
    [finishPainting, paintCell],
  );

  const focusCell = useCallback((row: number, column: number) => {
    const selector = `button[data-row="${row}"][data-column="${column}"]`;
    editorRef.current?.querySelector<HTMLButtonElement>(selector)?.focus();
  }, []);

  const handleCellKeyDown = useCallback(
    (
      cell: MatrixCellContext,
      event: ReactKeyboardEvent<HTMLButtonElement>,
    ) => {
      setActiveCell({ row: cell.row, column: cell.column });
      let nextRow = cell.row;
      let nextColumn = cell.column;

      switch (event.key) {
        case "ArrowUp":
          nextRow = Math.max(0, cell.row - 1);
          break;
        case "ArrowDown":
          nextRow = Math.min(MATRIX_SIZE - 1, cell.row + 1);
          break;
        case "ArrowLeft":
          nextColumn = Math.max(0, cell.column - 1);
          break;
        case "ArrowRight":
          nextColumn = Math.min(MATRIX_SIZE - 1, cell.column + 1);
          break;
        case "Home":
          nextColumn = 0;
          break;
        case "End":
          nextColumn = MATRIX_SIZE - 1;
          break;
        case "Enter":
        case " ":
          event.preventDefault();
          lastPaintedCellRef.current = null;
          paintCell(cell.row, cell.column);
          return;
        case "Backspace":
        case "Delete":
          event.preventDefault();
          lastPaintedCellRef.current = null;
          paintCell(cell.row, cell.column, "erase");
          return;
        default:
          return;
      }

      event.preventDefault();
      setActiveCell({ row: nextRow, column: nextColumn });
      focusCell(nextRow, nextColumn);
    },
    [focusCell, paintCell],
  );

  const handleIntensityChange = (event: ChangeEvent<HTMLInputElement>) => {
    onIntensityChange?.(clampMatrixValue(Number(event.currentTarget.value)));
  };

  const controlledIntensity = Math.abs(clampMatrixValue(intensity));

  return (
    <div
      className={joinClassNames(
        "nn-matrix-editor",
        readonly && "nn-matrix-editor--readonly",
        className,
      )}
      ref={editorRef}
    >
      {showControls ? (
        <div className="nn-matrix-editor__controls">
          <div
            aria-label="Boyama aracı"
            className="nn-matrix-editor__tools"
            role="group"
          >
            {toolOptions.map((option) => (
              <button
                aria-label={option.label}
                aria-pressed={tool === option.tool}
                className={joinClassNames(
                  "nn-matrix-editor__tool",
                  `nn-matrix-editor__tool--${option.tool}`,
                  tool === option.tool && "is-active",
                )}
                disabled={readonly || !onToolChange}
                key={option.tool}
                onClick={() => onToolChange?.(option.tool)}
                title={option.label}
                type="button"
              >
                <span aria-hidden="true" className="nn-matrix-editor__tool-symbol">
                  {option.symbol}
                </span>
                <span className="nn-matrix-editor__tool-label">{option.label}</span>
              </button>
            ))}
          </div>

          <label className="nn-matrix-editor__intensity">
            <span className="nn-matrix-editor__intensity-label">Yoğunluk</span>
            <input
              aria-valuetext={`Değer ${controlledIntensity.toFixed(4)}`}
              disabled={readonly || tool === "erase" || !onIntensityChange}
              max={1}
              min={0}
              onChange={handleIntensityChange}
              step={0.0001}
              type="range"
              value={controlledIntensity}
            />
            <output className="nn-matrix-editor__intensity-value">
              {controlledIntensity.toFixed(4)}
            </output>
          </label>
        </div>
      ) : null}

      <div className="nn-matrix-editor__readout" aria-live="polite">
        <span>{activeCell ? `SATIR ${activeCell.row + 1} · SÜTUN ${activeCell.column + 1}` : "HÜCRE DEĞERİ"}</span>
        <output>
          {activeCell
            ? formatMatrixValue(readMatrixValue(matrix, activeCell.row, activeCell.column))
            : "0.0000"}
        </output>
      </div>

      <MatrixGrid
        disabled={readonly}
        getCellLabel={(cell) => {
          const value = readMatrixValue(matrix, cell.row, cell.column);
          const formatted = formatMatrixValue(value);
          return `Satır ${cell.row + 1}, sütun ${cell.column + 1}, değer ${formatted}. ${
            readonly ? "Salt okunur" : "Boyamak için Enter veya Boşluk tuşuna basın"
          }`;
        }}
        interactive={!readonly}
        label={label}
        matrix={matrix}
        mode={mode}
        onCellKeyDown={handleCellKeyDown}
        onCellPointerDown={handlePointerDown}
        onCellPointerEnter={handlePointerEnter}
        onCellPointerMove={handlePointerMove}
        showValues={showValues}
      />
    </div>
  );
}

export default MatrixEditor;
