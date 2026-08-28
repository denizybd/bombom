import { describe, expect, it } from 'vitest'
import { DEFAULT_TEXT_LAB_CONFIG, createMatrixLayout } from '../domain/config'
import { transformRelation } from './context'
import { encodeIdentity, encodeSurfaceMatrix, projectIdentityToRelation } from './encode'
import { vectorNorm } from './matrix'

describe('surface and relation encoding', () => {
  const config = { ...DEFAULT_TEXT_LAB_CONFIG, matrixSide: 16 }
  const layout = createMatrixLayout(config)

  it('produces the same identity for the same exact surface', () => {
    const first = encodeIdentity('finans', layout, config)
    const second = encodeIdentity('finans', layout, config)
    expect(Array.from(first)).toEqual(Array.from(second))
    expect(vectorNorm(first)).toBeGreaterThan(0)
  })

  it('does not write surface features into the relation mask', () => {
    const matrix = encodeSurfaceMatrix('boşluk ve noktalama!', layout, config)
    expect(Array.from(matrix.slice(layout.relationStart))).toEqual(
      Array(layout.relationLength).fill(0),
    )
  })

  it('uses different physical transforms for left and right traces', () => {
    const identity = encodeIdentity('aynı iz', layout, config)
    const projected = projectIdentityToRelation(identity, layout, config)
    const left = transformRelation(projected, 'left', config.hashSeed)
    const right = transformRelation(projected, 'right', config.hashSeed)

    expect(Array.from(left)).not.toEqual(Array.from(right))
    expect(vectorNorm(left)).toBeCloseTo(vectorNorm(projected), 6)
    expect(vectorNorm(right)).toBeCloseTo(vectorNorm(projected), 6)
  })
})

