import { describe, expect, it } from 'vitest'
import {
  ADAPTIVE_TEXT_LAB_PROFILE_ID,
  DEFAULT_TEXT_LAB_CONFIG,
  LEGACY_TEXT_LAB_PROFILE_ID,
  createMatrixLayout,
  createTextLabConfigForProfile,
} from '../domain/config'
import { transformRelation } from './context'
import { encodeIdentity, encodeSurfaceMatrix, projectIdentityToRelation } from './encode'
import { vectorNorm } from './matrix'
import {
  createContextSpaceId,
  createIdentitySpaceId,
  createRepresentationSpaceId,
  digestVector,
} from './representation'
import {
  ADAPTIVE_CELLS_PER_PAGE,
  ADAPTIVE_PAGE_COUNT,
  relationIndexFromPage,
  relationPageFromRelationIndex,
} from './relationPages'

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

  it('keeps the historical default encoder and space fingerprints byte-stable', () => {
    const legacyLayout = createMatrixLayout(DEFAULT_TEXT_LAB_CONFIG)
    const identity = encodeIdentity('finans', legacyLayout, DEFAULT_TEXT_LAB_CONFIG)
    const relation = projectIdentityToRelation(identity, legacyLayout, DEFAULT_TEXT_LAB_CONFIG)
    const identitySpaceId = createIdentitySpaceId(DEFAULT_TEXT_LAB_CONFIG, legacyLayout)
    const contextSpaceId = createContextSpaceId(DEFAULT_TEXT_LAB_CONFIG, legacyLayout)
    const namedLegacy = createTextLabConfigForProfile(LEGACY_TEXT_LAB_PROFILE_ID)

    expect(Array.from(encodeIdentity('finans', legacyLayout, namedLegacy))).toEqual(
      Array.from(identity),
    )
    expect(createIdentitySpaceId(namedLegacy, legacyLayout)).toBe(identitySpaceId)
    expect(createContextSpaceId(namedLegacy, legacyLayout)).toBe(contextSpaceId)

    expect({
      identityDigest: digestVector(identity),
      relationDigest: digestVector(relation),
      identitySpaceId,
      contextSpaceId,
      representationSpaceId: createRepresentationSpaceId(identitySpaceId, contextSpaceId),
    }).toEqual({
      identityDigest: '9d42e8eca0dea6ea',
      relationDigest: '444f30e41a4693ec',
      identitySpaceId: 'IS-BA205A73D8449AB0',
      contextSpaceId: 'CS-B2269D48FDF53ABC',
      representationSpaceId: 'RS-9BCC37EC5C9F297D',
    })
  })

  it('encodes the opt-in adaptive profile deterministically in a distinct space', () => {
    const adaptive = createTextLabConfigForProfile(ADAPTIVE_TEXT_LAB_PROFILE_ID)
    const adaptiveLayout = createMatrixLayout(adaptive)
    const legacyLayout = createMatrixLayout(DEFAULT_TEXT_LAB_CONFIG)
    const first = encodeIdentity('finans ve gelecek', adaptiveLayout, adaptive)
    const second = encodeIdentity('finans ve gelecek', adaptiveLayout, adaptive)
    const firstRelation = projectIdentityToRelation(first, adaptiveLayout, adaptive)
    const secondRelation = projectIdentityToRelation(second, adaptiveLayout, adaptive)
    const adaptiveIdentitySpace = createIdentitySpaceId(adaptive, adaptiveLayout)
    const adaptiveContextSpace = createContextSpaceId(adaptive, adaptiveLayout)
    const configWithUnrelatedAtlasMetadata = { ...adaptive, atlasRevision: 99 }

    expect(first).toHaveLength(512)
    expect(firstRelation).toHaveLength(3_584)
    expect(Array.from(first)).toEqual(Array.from(second))
    expect(Array.from(firstRelation)).toEqual(Array.from(secondRelation))
    expect(createIdentitySpaceId(configWithUnrelatedAtlasMetadata, adaptiveLayout)).toBe(
      adaptiveIdentitySpace,
    )
    expect(createContextSpaceId(configWithUnrelatedAtlasMetadata, adaptiveLayout)).toBe(
      adaptiveContextSpace,
    )
    expect(adaptiveIdentitySpace).not.toBe(
      createIdentitySpaceId(DEFAULT_TEXT_LAB_CONFIG, legacyLayout),
    )
    expect(adaptiveContextSpace).not.toBe(
      createContextSpaceId(DEFAULT_TEXT_LAB_CONFIG, legacyLayout),
    )
    expect(createRepresentationSpaceId(adaptiveIdentitySpace, adaptiveContextSpace)).not.toBe(
      createRepresentationSpaceId(
        createIdentitySpaceId(DEFAULT_TEXT_LAB_CONFIG, legacyLayout),
        createContextSpaceId(DEFAULT_TEXT_LAB_CONFIG, legacyLayout),
      ),
    )
  })

  it('covers all 3584 relation coordinates with 56 disjoint 8x8 pages', () => {
    const coordinates = new Set<number>()
    for (let pageIndex = 0; pageIndex < ADAPTIVE_PAGE_COUNT; pageIndex += 1) {
      for (let cellIndex = 0; cellIndex < ADAPTIVE_CELLS_PER_PAGE; cellIndex += 1) {
        const relationIndex = relationIndexFromPage(pageIndex, cellIndex)
        coordinates.add(relationIndex)
        expect(relationPageFromRelationIndex(relationIndex)).toEqual({ pageIndex, cellIndex })
      }
    }

    expect(coordinates.size).toBe(3_584)
    expect(Math.min(...coordinates)).toBe(0)
    expect(Math.max(...coordinates)).toBe(3_583)
  })

  it('breaks relation-page symmetry with deterministic page-local transforms', () => {
    const adaptive = createTextLabConfigForProfile(ADAPTIVE_TEXT_LAB_PROFILE_ID)
    const adaptiveLayout = createMatrixLayout(adaptive)
    const identity = Float32Array.from(
      { length: adaptiveLayout.identityLength },
      (_unused, index) => ((index * 37) % 257 - 128) / 128,
    )
    const relation = projectIdentityToRelation(identity, adaptiveLayout, adaptive)
    const pageDigests = Array.from({ length: ADAPTIVE_PAGE_COUNT }, (_unused, pageIndex) => {
      const page = Float32Array.from(
        { length: ADAPTIVE_CELLS_PER_PAGE },
        (_cell, cellIndex) => relation[relationIndexFromPage(pageIndex, cellIndex)],
      )
      expect(vectorNorm(page)).toBeGreaterThan(0)
      return digestVector(page)
    })

    expect(new Set(pageDigests).size).toBe(ADAPTIVE_PAGE_COUNT)
  })

  it('keeps each adaptive context trace as a complete page-level permutation', () => {
    const relation = new Float32Array(3_584)
    for (let pageIndex = 0; pageIndex < ADAPTIVE_PAGE_COUNT; pageIndex += 1) {
      for (let cellIndex = 0; cellIndex < ADAPTIVE_CELLS_PER_PAGE; cellIndex += 1) {
        relation[relationIndexFromPage(pageIndex, cellIndex)] = (pageIndex + 1) / 56
      }
    }

    const transformed = transformRelation(
      relation,
      'parent',
      DEFAULT_TEXT_LAB_CONFIG.hashSeed,
      true,
    )
    const sourceMagnitudes = new Set<number>()
    for (let pageIndex = 0; pageIndex < ADAPTIVE_PAGE_COUNT; pageIndex += 1) {
      const magnitudes = new Set<number>()
      for (let cellIndex = 0; cellIndex < ADAPTIVE_CELLS_PER_PAGE; cellIndex += 1) {
        magnitudes.add(Math.abs(transformed[relationIndexFromPage(pageIndex, cellIndex)]))
      }
      expect(magnitudes.size).toBe(1)
      sourceMagnitudes.add([...magnitudes][0])
    }

    expect(sourceMagnitudes.size).toBe(ADAPTIVE_PAGE_COUNT)
    expect(vectorNorm(transformed)).toBeCloseTo(vectorNorm(relation), 6)
  })
})
