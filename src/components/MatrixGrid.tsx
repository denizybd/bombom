import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from "react";
import { MATRIX_SIZE } from "../engine/types";

export { MATRIX_SIZE };

export type Matrix = ReadonlyArray<ReadonlyArray<number>>;
export type MutableMatrix = number[][];
export type MatrixDisplayMode = "compact" | "large";

export interface MatrixCellContext {
  row: number;
  column: number;
  value: number;
}

export interface MatrixGridProps {
  matrix: Matrix;
  /** A short, descriptive name read by assistive technology. */
  label?: string;
  mode?: MatrixDisplayMode;
  /** Defaults to true in large mode and false in compact mode. */
  showValues?: boolean;
  className?: string;
  cellClassName?: string | ((cell: MatrixCellContext) => string | undefined);
  formatValue?: (value: number) => string;
  getCellLabel?: (cell: MatrixCellContext) => string;
  interactive?: boolean;
  disabled?: boolean;
  onCellPointerDown?: (
    cell: MatrixCellContext,
    event: ReactPointerEvent<HTMLButtonElement>,
  ) => void;
  onCellPointerEnter?: (
    cell: MatrixCellContext,
    event: ReactPointerEvent<HTMLButtonElement>,
  ) => void;
  onCellPointerMove?: (
    cell: MatrixCellContext,
    event: ReactPointerEvent<HTMLButtonElement>,
  ) => void;
  onCellKeyDown?: (
    cell: MatrixCellContext,
    event: ReactKeyboardEvent<HTMLButtonElement>,
  ) => void;
}

const joinClassNames = (...classNames: Array<string | false | null | undefined>) =>
  classNames.filter(Boolean).join(" ");

export const clampMatrixValue = (value: number): number => {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-1, Math.min(1, value));
};

export const readMatrixValue = (matrix: Matrix, row: number, column: number): number =>
  clampMatrixValue(matrix[row]?.[column] ?? 0);

export const toNormalizedMatrix = (matrix: Matrix): MutableMatrix =>
  Array.from({ length: MATRIX_SIZE }, (_, row) =>
    Array.from({ length: MATRIX_SIZE }, (_, column) =>
      readMatrixValue(matrix, row, column),
    ),
  );

export const formatMatrixValue = (value: number): string => {
  if (Math.abs(value) < 0.00005) return "0.0000";
  return `${value > 0 ? "+" : "\u2212"}${Math.abs(value).toFixed(4)}`;
};

const getHeatmapStyle = (value: number): CSSProperties => {
  const magnitude = Math.abs(value);
  // White at zero, a saturated red at either signed extreme.
  const red = Math.round(255 - 40 * magnitude);
  const green = Math.round(255 - 217 * magnitude);
  const blue = Math.round(255 - 205 * magnitude);
  const style: CSSProperties = {
    // Full magnitude resolves to the interface's core red: #d72632.
    backgroundColor: `rgb(${red}, ${green}, ${blue})`,
  };

  if (value < 0) {
    style.backgroundImage =
      "repeating-linear-gradient(135deg, rgba(91, 0, 0, 0.34) 0, rgba(91, 0, 0, 0.34) 2px, transparent 2px, transparent 6px)";
  }

  return style;
};

const defaultCellLabel = (
  { row, column }: MatrixCellContext,
  formattedValue: string,
): string => `Satır ${row + 1}, sütun ${column + 1}, değer ${formattedValue}`;

/**
 * Fixed 32x32 signed heatmap. Missing or non-finite cells are safely displayed as zero.
 * Magnitude controls the white-to-red fill; negative values also receive a hatch.
 */
export function MatrixGrid({
  matrix,
  label = "32 çarpı 32 matris",
  mode = "large",
  showValues = mode === "large",
  className,
  cellClassName,
  formatValue = formatMatrixValue,
  getCellLabel,
  interactive = false,
  disabled = false,
  onCellPointerDown,
  onCellPointerEnter,
  onCellPointerMove,
  onCellKeyDown,
}: MatrixGridProps) {
  const cells = Array.from({ length: MATRIX_SIZE * MATRIX_SIZE }, (_, index) => {
    const row = Math.floor(index / MATRIX_SIZE);
    const column = index % MATRIX_SIZE;
    const value = readMatrixValue(matrix, row, column);
    return { row, column, value } satisfies MatrixCellContext;
  });

  return (
    <div
      aria-colcount={MATRIX_SIZE}
      aria-label={label}
      aria-readonly={!interactive || undefined}
      aria-rowcount={MATRIX_SIZE}
      className={joinClassNames(
        "nn-matrix-grid",
        `nn-matrix-grid--${mode}`,
        interactive && "nn-matrix-grid--interactive",
        disabled && "nn-matrix-grid--disabled",
        className,
      )}
      role="grid"
      style={{ "--matrix-size": MATRIX_SIZE } as CSSProperties}
    >
      {Array.from({ length: MATRIX_SIZE }, (_, row) => (
        <div className="nn-matrix-grid__row" key={row} role="row">
          {cells
            .slice(row * MATRIX_SIZE, (row + 1) * MATRIX_SIZE)
            .map((cell) => {
              const formattedValue = formatValue(cell.value);
              const accessibleLabel =
                getCellLabel?.(cell) ?? defaultCellLabel(cell, formattedValue);
              const resolvedCellClassName =
                typeof cellClassName === "function"
                  ? cellClassName(cell)
                  : cellClassName;
              const commonProps = {
                "aria-label": accessibleLabel,
                className: joinClassNames(
                  "nn-matrix-cell",
                  `nn-matrix-cell--${mode}`,
                  cell.value < 0 && "nn-matrix-cell--negative",
                  cell.value > 0 && "nn-matrix-cell--positive",
                  cell.value === 0 && "nn-matrix-cell--zero",
                  resolvedCellClassName,
                ),
                "data-column": cell.column,
                "data-row": cell.row,
                "data-sign": cell.value < 0 ? "negative" : cell.value > 0 ? "positive" : "zero",
                style: getHeatmapStyle(cell.value),
                title: accessibleLabel,
              } as const;

              const contents = (
                <>
                  {showValues ? (
                    <span aria-hidden="true" className="nn-matrix-cell__value">
                      {formattedValue}
                    </span>
                  ) : null}
                  {!showValues && cell.value < 0 ? (
                    <span aria-hidden="true" className="nn-matrix-cell__sign">
                      −
                    </span>
                  ) : null}
                </>
              );

              if (interactive) {
                return (
                  <button
                    {...commonProps}
                    aria-disabled={disabled || undefined}
                    disabled={disabled}
                    key={`${cell.row}-${cell.column}`}
                    onKeyDown={(event) => onCellKeyDown?.(cell, event)}
                    onPointerDown={(event) => onCellPointerDown?.(cell, event)}
                    onPointerEnter={(event) => onCellPointerEnter?.(cell, event)}
                    onPointerMove={(event) => onCellPointerMove?.(cell, event)}
                    role="gridcell"
                    tabIndex={cell.row === 0 && cell.column === 0 ? 0 : -1}
                    type="button"
                  >
                    {contents}
                  </button>
                );
              }

              return (
                <div
                  {...commonProps}
                  key={`${cell.row}-${cell.column}`}
                  role="gridcell"
                >
                  {contents}
                </div>
              );
            })}
        </div>
      ))}
    </div>
  );
}

export default MatrixGrid;
