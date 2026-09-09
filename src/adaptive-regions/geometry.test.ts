import { describe, expect, it } from 'vitest'
import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_IDENTITY_LENGTH,
  ADAPTIVE_MATRIX_CELLS,
  ADAPTIVE_PAGE_COUNT,
  ADAPTIVE_REGION_GEOMETRY,
  ADAPTIVE_RELATION_LENGTH,
  extractRelationPage,
  isIdentityMatrixIndex,
  matrixCoordinate,
  matrixCoordinateFromRelationPage,
  matrixIndex,
  matrixIndexFromRelationPage,
  matrixIndexFromRelationIndex,
  relationIndexFromMatrixIndex,
  relationIndexFromPage,
  relationPageFromMatrixIndex,
  relationPageFromRelationIndex,
} from './geometry.ts'

describe('8/56 hedef geometri', () => {
  it('64x64 matrisi 8 I satırı ve 56 R satırına eksiksiz böler', () => {
    expect(ADAPTIVE_REGION_GEOMETRY).toMatchObject({
      side: 64,
      cellCount: 4096,
      identityRows: 8,
      identityLength: 512,
      relationRows: 56,
      relationLength: 3584,
      pageSide: 8,
      pageRows: 7,
      pageColumns: 8,
      pageCount: 56,
      cellsPerPage: 64,
    })
    expect(ADAPTIVE_IDENTITY_LENGTH + ADAPTIVE_RELATION_LENGTH).toBe(ADAPTIVE_MATRIX_CELLS)
    expect(ADAPTIVE_PAGE_COUNT * ADAPTIVE_CELLS_PER_PAGE).toBe(ADAPTIVE_RELATION_LENGTH)
  })

  it('bütün R hücrelerinde ilişki indeksi ↔ sayfa/hücre eşlemesini bijektif yapar', () => {
    const matrixIndices = new Set<number>()
    const pairs = new Set<string>()
    for (let relationIndex = 0; relationIndex < ADAPTIVE_RELATION_LENGTH; relationIndex += 1) {
      const pair = relationPageFromRelationIndex(relationIndex)
      const matrix = matrixIndexFromRelationIndex(relationIndex)
      expect(relationIndexFromPage(pair.pageIndex, pair.cellIndex)).toBe(relationIndex)
      expect(matrixIndexFromRelationPage(pair.pageIndex, pair.cellIndex)).toBe(matrix)
      expect(relationPageFromMatrixIndex(matrix)).toEqual(pair)
      matrixIndices.add(matrix)
      pairs.add(`${pair.pageIndex}:${pair.cellIndex}`)
    }
    expect(matrixIndices.size).toBe(ADAPTIVE_RELATION_LENGTH)
    expect(pairs.size).toBe(ADAPTIVE_RELATION_LENGTH)
    expect(Math.min(...matrixIndices)).toBe(ADAPTIVE_IDENTITY_LENGTH)
    expect(Math.max(...matrixIndices)).toBe(ADAPTIVE_MATRIX_CELLS - 1)
  })

  it('matris koordinatlarını tersinir yapar ve I alanını R sayfasına sokmaz', () => {
    for (const index of [0, 511, 512, 1023, 4095]) {
      const coordinate = matrixCoordinate(index)
      expect(matrixIndex(coordinate.row, coordinate.column)).toBe(index)
    }
    expect(isIdentityMatrixIndex(511)).toBe(true)
    expect(isIdentityMatrixIndex(512)).toBe(false)
    expect(() => relationIndexFromMatrixIndex(511)).toThrow(/R matris indeksi/)
    expect(() => relationPageFromMatrixIndex(511)).toThrow(/R satırı/)
    expect(matrixCoordinateFromRelationPage(55, 63)).toEqual({ row: 63, column: 63 })
  })

  it('tek bir 8x8 sayfayı fiziksel koordinat sırasıyla çıkarır', () => {
    const relation = new Float32Array(ADAPTIVE_RELATION_LENGTH)
    for (let cellIndex = 0; cellIndex < 64; cellIndex += 1) {
      relation[relationIndexFromPage(9, cellIndex)] = cellIndex / 64
    }
    expect(Array.from(extractRelationPage(relation, 9))).toEqual(
      Array.from({ length: 64 }, (_, index) => Math.fround(index / 64)),
    )
  })
})
