import type { MatrixLayout } from '../domain/types'
import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_IDENTITY_LENGTH,
  ADAPTIVE_IDENTITY_ROWS,
  ADAPTIVE_MATRIX_CELLS,
  ADAPTIVE_MATRIX_SIDE,
  ADAPTIVE_PAGE_COLUMNS,
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_PAGE_ROWS,
  ADAPTIVE_PAGE_SIDE,
  ADAPTIVE_REGION_GEOMETRY,
  ADAPTIVE_RELATION_LENGTH,
  ADAPTIVE_RELATION_ROWS,
  relationIndexFromPage,
  relationPageFromRelationIndex,
} from '../../adaptive-regions/geometry.ts'

export {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_IDENTITY_LENGTH,
  ADAPTIVE_IDENTITY_ROWS,
  ADAPTIVE_MATRIX_CELLS,
  ADAPTIVE_MATRIX_SIDE,
  ADAPTIVE_PAGE_COLUMNS,
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_PAGE_ROWS,
  ADAPTIVE_PAGE_SIDE,
  ADAPTIVE_REGION_GEOMETRY,
  ADAPTIVE_RELATION_LENGTH,
  ADAPTIVE_RELATION_ROWS,
  relationIndexFromPage,
  relationPageFromRelationIndex,
}

/**
 * TextLab integration guard for the canonical adaptive-region geometry. It
 * contains no learned atlas state and therefore cannot drift with revisions.
 */
export function assertAdaptiveRelationPageLayout(layout: MatrixLayout): void {
  if (
    layout.side !== ADAPTIVE_MATRIX_SIDE ||
    layout.cellCount !== ADAPTIVE_MATRIX_CELLS ||
    layout.identityRows !== ADAPTIVE_IDENTITY_ROWS ||
    layout.identityLength !== ADAPTIVE_IDENTITY_LENGTH ||
    layout.relationStart !== ADAPTIVE_IDENTITY_LENGTH ||
    layout.relationLength !== ADAPTIVE_RELATION_LENGTH
  ) {
    throw new RangeError(
      'Uyarlanabilir sayfa geometrisi 64×64, I=8×64 ve R=56×64 düzeni gerektirir.',
    )
  }
}

/** Maps an 8×8 page-local cell to the row-major R-array index. */
export function relationPageCellIndex(pageIndex: number, localCellIndex: number): number {
  return relationIndexFromPage(pageIndex, localCellIndex)
}

/** Maps an 8×8 page-local cell to the row-major 8×64 identity-array index. */
export function identityPageCellIndex(pageIndex: number, localCellIndex: number): number {
  if (!Number.isInteger(pageIndex) || pageIndex < 0 || pageIndex >= ADAPTIVE_PAGE_COLUMNS) {
    throw new RangeError('Kimlik mikro-sayfası 0 ile 7 arasında olmalıdır.')
  }
  if (
    !Number.isInteger(localCellIndex) ||
    localCellIndex < 0 ||
    localCellIndex >= ADAPTIVE_CELLS_PER_PAGE
  ) {
    throw new RangeError('Mikro-sayfa hücresi 0 ile 63 arasında olmalıdır.')
  }
  const localRow = Math.floor(localCellIndex / ADAPTIVE_PAGE_SIDE)
  const localColumn = localCellIndex % ADAPTIVE_PAGE_SIDE
  return localRow * ADAPTIVE_MATRIX_SIDE + pageIndex * ADAPTIVE_PAGE_SIDE + localColumn
}
