import { describe, expect, it } from "vitest";
import { MATRIX_SIZE, createRandomSignedMatrix, createSeededRandom, nextRandom } from "./index";

describe("seeded random state", () => {
  it("replays the same values and state for the same seed", () => {
    let first = createSeededRandom(12345);
    let second = createSeededRandom(12345);
    const firstValues: number[] = [];
    const secondValues: number[] = [];

    for (let index = 0; index < 8; index += 1) {
      const firstDraw = nextRandom(first);
      const secondDraw = nextRandom(second);
      first = firstDraw.rng;
      second = secondDraw.rng;
      firstValues.push(firstDraw.value);
      secondValues.push(secondDraw.value);
    }

    expect(firstValues).toEqual(secondValues);
    expect(first).toEqual(second);
    expect(first.draws).toBe(8);
    expect(firstValues.every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it("creates a deterministic 32 x 32 signed matrix", () => {
    const first = createRandomSignedMatrix(createSeededRandom(7));
    const second = createRandomSignedMatrix(createSeededRandom(7));
    expect(first).toEqual(second);
    expect(first.matrix).toHaveLength(MATRIX_SIZE);
    expect(first.matrix.every((row) => row.length === MATRIX_SIZE)).toBe(true);
    expect(first.matrix.flat().every((value) => value >= -1 && value <= 1)).toBe(true);
    expect(first.rng.draws).toBe(MATRIX_SIZE * MATRIX_SIZE);
  });
});
