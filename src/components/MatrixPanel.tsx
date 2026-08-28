import { useId, type ReactNode } from "react";

import MatrixEditor, { type MatrixEditorProps } from "./MatrixEditor";
import MatrixGrid, {
  type Matrix,
  type MatrixDisplayMode,
} from "./MatrixGrid";

export interface MatrixPanelEditorConfig
  extends Omit<
    MatrixEditorProps,
    "matrix" | "label" | "mode" | "showValues" | "className"
  > {
  className?: string;
}

export interface MatrixPanelProps {
  title: ReactNode;
  subtitle?: ReactNode;
  eyebrow?: ReactNode;
  matrix: Matrix;
  /** Accessible matrix name. Falls back to a string title or "Matris". */
  label?: string;
  mode?: MatrixDisplayMode;
  showValues?: boolean;
  /** Presence switches the panel from a display grid to a controlled editor. */
  editor?: MatrixPanelEditorConfig;
  actions?: ReactNode;
  status?: ReactNode;
  footer?: ReactNode;
  showLegend?: boolean;
  className?: string;
  matrixClassName?: string;
}

const joinClassNames = (...classNames: Array<string | false | null | undefined>) =>
  classNames.filter(Boolean).join(" ");

/** A semantic card shell shared by matrix previews, inspectors and editors. */
export function MatrixPanel({
  title,
  subtitle,
  eyebrow,
  matrix,
  label,
  mode = "large",
  showValues = mode === "large",
  editor,
  actions,
  status,
  footer,
  showLegend = false,
  className,
  matrixClassName,
}: MatrixPanelProps) {
  const reactId = useId();
  const titleId = `nn-matrix-panel-${reactId.replace(/:/g, "")}`;
  const matrixLabel = label ?? (typeof title === "string" ? title : "Matris");

  return (
    <section
      aria-labelledby={titleId}
      className={joinClassNames(
        "nn-matrix-panel",
        `nn-matrix-panel--${mode}`,
        editor && "nn-matrix-panel--editable",
        className,
      )}
    >
      <header className="nn-matrix-panel__header">
        <div className="nn-matrix-panel__heading">
          {eyebrow ? (
            <div className="nn-matrix-panel__eyebrow">{eyebrow}</div>
          ) : null}
          <div className="nn-matrix-panel__title-line">
            <h3 className="nn-matrix-panel__title" id={titleId}>
              {title}
            </h3>
            {status ? <div className="nn-matrix-panel__status">{status}</div> : null}
          </div>
          {subtitle ? (
            <div className="nn-matrix-panel__subtitle">{subtitle}</div>
          ) : null}
        </div>
        {actions ? <div className="nn-matrix-panel__actions">{actions}</div> : null}
      </header>

      <div className="nn-matrix-panel__body">
        {editor ? (
          <MatrixEditor
            {...editor}
            className={joinClassNames(
              "nn-matrix-panel__matrix",
              matrixClassName,
              editor.className,
            )}
            label={matrixLabel}
            matrix={matrix}
            mode={mode}
            showValues={showValues}
          />
        ) : (
          <MatrixGrid
            className={joinClassNames(
              "nn-matrix-panel__matrix",
              matrixClassName,
            )}
            label={matrixLabel}
            matrix={matrix}
            mode={mode}
            showValues={showValues}
          />
        )}
      </div>

      {showLegend ? (
        <div aria-label="Matris renk açıklaması" className="nn-matrix-panel__legend">
          <span className="nn-matrix-panel__legend-item">
            <span
              aria-hidden="true"
              className="nn-matrix-panel__legend-swatch nn-matrix-panel__legend-swatch--positive"
            />
            Pozitif
          </span>
          <span className="nn-matrix-panel__legend-item">
            <span
              aria-hidden="true"
              className="nn-matrix-panel__legend-swatch nn-matrix-panel__legend-swatch--negative"
            />
            Negatif / taralı
          </span>
          <span className="nn-matrix-panel__legend-note">
            Renk yoğunluğu mutlak büyüklüğü gösterir.
          </span>
        </div>
      ) : null}

      {footer ? <footer className="nn-matrix-panel__footer">{footer}</footer> : null}
    </section>
  );
}

export default MatrixPanel;
