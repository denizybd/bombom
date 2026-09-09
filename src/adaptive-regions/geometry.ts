import { adaptiveRegionDigest } from './deterministic.ts'
import type {
  AdaptiveRegionGeometryV1,
  MatrixCoordinate,
  RelationPageCoordinate,
} from './types.ts'

export const ADAPTIVE_MATRIX_SIDE = 64 as const
export const ADAPTIVE_MATRIX_CELLS = 4096 as const
export const ADAPTIVE_IDENTITY_ROWS = 8 as const
export const ADAPTIVE_IDENTITY_LENGTH = 512 as const
export const ADAPTIVE_RELATION_START_ROW = 8 as const
export const ADAPTIVE_RELATION_ROWS = 56 as const
export const ADAPTIVE_RELATION_LENGTH = 3584 as const
export const ADAPTIVE_PAGE_SIDE = 8 as const
export const ADAPTIVE_PAGE_ROWS = 7 as const
export const ADAPTIVE_PAGE_COLUMNS = 8 as const
export const ADAPTIVE_PAGE_COUNT = 56 as const
export const ADAPTIVE_CELLS_PER_PAGE = 64 as const

const geometryPayload = {
  side: ADAPTIVE_MATRIX_SIDE,
  cellCount: ADAPTIVE_MATRIX_CELLS,
  identityRows: ADAPTIVE_IDENTITY_ROWS,
  identityLength: ADAPTIVE_IDENTITY_LENGTH,
  relationStartRow: ADAPTIVE_RELATION_START_ROW,
  relationRows: ADAPTIVE_RELATION_ROWS,
  relationLength: ADAPTIVE_RELATION_LENGTH,
  pageSide: ADAPTIVE_PAGE_SIDE,
  pageRows: ADAPTIVE_PAGE_ROWS,
  pageColumns: ADAPTIVE_PAGE_COLUMNS,
  pageCount: ADAPTIVE_PAGE_COUNT,
  cellsPerPage: ADAPTIVE_CELLS_PER_PAGE,
} as const

export const ADAPTIVE_REGION_GEOMETRY: AdaptiveRegionGeometryV1 = Object.freeze({
  ...geometryPayload,
  fingerprint: adaptiveRegionDigest('PRGEOM', geometryPayload),
})

function assertIntegerInRange(value: number, minimum: number, maximum: number, label: string): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} ${minimum}..${maximum} aralığında bir tam sayı olmalıdır.`)
  }
}

export function matrixIndex(row: number, column: number): number {
  assertIntegerInRange(row, 0, ADAPTIVE_MATRIX_SIDE - 1, 'Matris satırı')
  assertIntegerInRange(column, 0, ADAPTIVE_MATRIX_SIDE - 1, 'Matris sütunu')
  return row * ADAPTIVE_MATRIX_SIDE + column
}

export function matrixCoordinate(index: number): MatrixCoordinate {
  assertIntegerInRange(index, 0, ADAPTIVE_MATRIX_CELLS - 1, 'Matris indeksi')
  return {
    row: Math.floor(index / ADAPTIVE_MATRIX_SIDE),
    column: index % ADAPTIVE_MATRIX_SIDE,
  }
}

export function isIdentityMatrixIndex(index: number): boolean {
  assertIntegerInRange(index, 0, ADAPTIVE_MATRIX_CELLS - 1, 'Matris indeksi')
  return index < ADAPTIVE_IDENTITY_LENGTH
}

export function relationIndexFromMatrixIndex(index: number): number {
  assertIntegerInRange(index, ADAPTIVE_IDENTITY_LENGTH, ADAPTIVE_MATRIX_CELLS - 1, 'R matris indeksi')
  return index - ADAPTIVE_IDENTITY_LENGTH
}

export function matrixIndexFromRelationIndex(relationIndex: number): number {
  assertIntegerInRange(relationIndex, 0, ADAPTIVE_RELATION_LENGTH - 1, 'R indeksi')
  return relationIndex + ADAPTIVE_IDENTITY_LENGTH
}

export function relationPageFromMatrixCoordinate(row: number, column: number): RelationPageCoordinate {
  assertIntegerInRange(row, ADAPTIVE_RELATION_START_ROW, ADAPTIVE_MATRIX_SIDE - 1, 'R satırı')
  assertIntegerInRange(column, 0, ADAPTIVE_MATRIX_SIDE - 1, 'R sütunu')
  const localRow = row - ADAPTIVE_RELATION_START_ROW
  const pageRow = Math.floor(localRow / ADAPTIVE_PAGE_SIDE)
  const pageColumn = Math.floor(column / ADAPTIVE_PAGE_SIDE)
  return {
    pageIndex: pageRow * ADAPTIVE_PAGE_COLUMNS + pageColumn,
    cellIndex: (localRow % ADAPTIVE_PAGE_SIDE) * ADAPTIVE_PAGE_SIDE + (column % ADAPTIVE_PAGE_SIDE),
  }
}

export function relationPageFromMatrixIndex(index: number): RelationPageCoordinate {
  const coordinate = matrixCoordinate(index)
  return relationPageFromMatrixCoordinate(coordinate.row, coordinate.column)
}

export function relationPageFromRelationIndex(relationIndex: number): RelationPageCoordinate {
  return relationPageFromMatrixIndex(matrixIndexFromRelationIndex(relationIndex))
}

export function matrixCoordinateFromRelationPage(pageIndex: number, cellIndex: number): MatrixCoordinate {
  assertIntegerInRange(pageIndex, 0, ADAPTIVE_PAGE_COUNT - 1, 'Mikro-sayfa indeksi')
  assertIntegerInRange(cellIndex, 0, ADAPTIVE_CELLS_PER_PAGE - 1, 'Mikro-sayfa hücresi')
  const pageRow = Math.floor(pageIndex / ADAPTIVE_PAGE_COLUMNS)
  const pageColumn = pageIndex % ADAPTIVE_PAGE_COLUMNS
  return {
    row: ADAPTIVE_RELATION_START_ROW + pageRow * ADAPTIVE_PAGE_SIDE + Math.floor(cellIndex / ADAPTIVE_PAGE_SIDE),
    column: pageColumn * ADAPTIVE_PAGE_SIDE + cellIndex % ADAPTIVE_PAGE_SIDE,
  }
}

export function matrixIndexFromRelationPage(pageIndex: number, cellIndex: number): number {
  const coordinate = matrixCoordinateFromRelationPage(pageIndex, cellIndex)
  return matrixIndex(coordinate.row, coordinate.column)
}

export function relationIndexFromPage(pageIndex: number, cellIndex: number): number {
  return relationIndexFromMatrixIndex(matrixIndexFromRelationPage(pageIndex, cellIndex))
}

export function extractRelationPage(
  relation: ArrayLike<number>,
  pageIndex: number,
): Float32Array {
  if (relation.length !== ADAPTIVE_RELATION_LENGTH) {
    throw new RangeError(`R vektörü ${ADAPTIVE_RELATION_LENGTH} hücre içermelidir.`)
  }
  const page = new Float32Array(ADAPTIVE_CELLS_PER_PAGE)
  for (let cellIndex = 0; cellIndex < ADAPTIVE_CELLS_PER_PAGE; cellIndex += 1) {
    page[cellIndex] = relation[relationIndexFromPage(pageIndex, cellIndex)]
  }
  return page
}
