import { MATRIX_SIZE, type Matrix, type SeededRandomState } from "./types";

const UINT32_RANGE = 0x1_0000_0000;
const MULBERRY_INCREMENT = 0x6d2b79f5;

export function normalizeSeed(seed: number): number {
  if (!Number.isFinite(seed)) return 0;
  return Math.trunc(seed) >>> 0;
}

export function createSeededRandom(seed: number): SeededRandomState {
  const normalized = normalizeSeed(seed);
  return { seed: normalized, state: normalized, draws: 0 };
}

/** A pure, state-carrying Mulberry32 sample in [0, 1). */
export function nextRandom(rng: SeededRandomState): {
  value: number;
  rng: SeededRandomState;
} {
  const state = (normalizeSeed(rng.state) + MULBERRY_INCREMENT) >>> 0;
  let value = state;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  value = (value ^ (value >>> 14)) >>> 0;

  return {
    value: value / UINT32_RANGE,
    rng: {
      seed: normalizeSeed(rng.seed),
      state,
      draws: Math.max(0, Math.trunc(rng.draws)) + 1,
    },
  };
}

export function createRandomSignedMatrix(
  rng: SeededRandomState,
  size = MATRIX_SIZE,
): { matrix: Matrix; rng: SeededRandomState } {
  const matrix: Matrix = [];
  let nextState = rng;

  for (let row = 0; row < size; row += 1) {
    const values: number[] = [];
    for (let column = 0; column < size; column += 1) {
      const draw = nextRandom(nextState);
      nextState = draw.rng;
      values.push(draw.value * 2 - 1);
    }
    matrix.push(values);
  }

  return { matrix, rng: nextState };
}
