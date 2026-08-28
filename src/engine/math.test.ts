import { describe, expect, it } from "vitest";
import {
  computeHabitProbabilities,
  computeNeurogenesisPressure,
  computeNeuronSimilarityMatrix,
  computeSequentialSelectionDistribution,
  cosineSimilarity,
  createMatrix,
  createSeededRandom,
  MATRIX_SIZE,
  orderCandidatesByHabit,
  processSignalAtNeuron,
  updateHabitEvidence,
} from "./index";

function patternedMatrix(): number[][] {
  return Array.from({ length: MATRIX_SIZE }, (_, row) =>
    Array.from({ length: MATRIX_SIZE }, (_, column) => ((row + column) % 3 - 1) / 2),
  );
}

describe("cosineSimilarity", () => {
  it("stays in [-1, 1] and represents opposite matrices", () => {
    const matrix = patternedMatrix();
    const opposite = matrix.map((row) => row.map((value) => -value));

    expect(cosineSimilarity(matrix, matrix)).toBeGreaterThan(0.999999);
    expect(cosineSimilarity(matrix, opposite)).toBeLessThan(-0.999999);
    expect(cosineSimilarity(matrix, matrix)).toBeLessThanOrEqual(1);
    expect(cosineSimilarity(matrix, opposite)).toBeGreaterThanOrEqual(-1);
  });

  it("returns zero rather than NaN for zero matrices", () => {
    const zero = createMatrix(0);
    expect(cosineSimilarity(zero, zero)).toBe(0);
    expect(Number.isNaN(cosineSimilarity(zero, zero))).toBe(false);
  });
});

describe("processSignalAtNeuron", () => {
  it("keeps every transformed cell within [-1, 1]", () => {
    const result = processSignalAtNeuron(patternedMatrix(), createMatrix(1));
    expect(result.outputSignal.flat().every((value) => value >= -1 && value <= 1)).toBe(
      true,
    );
  });

  it("uses alpha=0 for complete incompatibility and leaves the signal unchanged", () => {
    const signal = createMatrix(1);
    const result = processSignalAtNeuron(signal, createMatrix(-1));
    expect(result.alpha).toBe(0);
    expect(result.outputSignal).toEqual(signal);
    expect(result.changeMagnitude).toBe(0);
  });

  it("leaves a fully matching signal equal to the neuron prototype", () => {
    const matrix = patternedMatrix();
    const result = processSignalAtNeuron(matrix, matrix);
    expect(result.alpha).toBeGreaterThan(0.999999);
    expect(result.outputSignal).toEqual(matrix);
    expect(result.delta.flat().every((value) => value === 0)).toBe(true);
  });
});

describe("habit and similarity matrices", () => {
  it("normalizes non-empty H rows, leaves empty rows at zero and disables self edges", () => {
    const probabilities = computeHabitProbabilities([
      [99, 1, 3],
      [0, 42, 0],
      [2, 2, 9],
    ]);

    expect(probabilities[0]).toEqual([0, 0.25, 0.75]);
    expect(probabilities[1]).toEqual([0, 0, 0]);
    expect(probabilities[2]).toEqual([0.5, 0.5, 0]);
    expect(probabilities[0].reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
    expect(probabilities[2].reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
  });

  it("computes symmetric K independently from habit evidence", () => {
    const first = patternedMatrix();
    const second = first.map((row) => row.map((value) => -value));
    const similarities = computeNeuronSimilarityMatrix([
      { id: "N1", matrix: first },
      { id: "N2", matrix: second },
    ]);

    expect(similarities).toHaveLength(2);
    expect(similarities[0][1]).toBeCloseTo(similarities[1][0]);
    expect(similarities[0][1]).toBeLessThan(-0.999999);
  });

  it("increments only the realized W_ij edge", () => {
    const evidence = [
      [0, 1, 2],
      [3, 0, 4],
      [5, 6, 0],
    ];
    const updated = updateHabitEvidence(evidence, 0, 2);

    expect(updated[0][2]).toBe(3);
    expect(updated.flat().filter((value, index) => value !== evidence.flat()[index])).toEqual([
      3,
    ]);
    expect(evidence[0][2]).toBe(2);
  });
});

describe("candidate ordering and sequential selection", () => {
  it("excludes self and breaks ties reproducibly from the seed", () => {
    const probabilities = [
      [0, 0.5, 0.5, 0],
      [0.3, 0, 0.4, 0.3],
      [0.3, 0.3, 0, 0.4],
      [0.3, 0.3, 0.4, 0],
    ];
    const first = orderCandidatesByHabit(probabilities, 0, createSeededRandom(2026));
    const second = orderCandidatesByHabit(probabilities, 0, createSeededRandom(2026));

    expect(first.order).toEqual(second.order);
    expect(first.order).not.toContain(0);
    expect(new Set(first.order)).toEqual(new Set([1, 2]));
    expect(first.order.slice().sort()).toEqual([1, 2]);
  });

  it("including the unmatched outcome sums to one", () => {
    const distribution = computeSequentialSelectionDistribution(
      [2, 0, 1],
      [0.4, 0.8, 0.2],
    );

    expect(distribution.entries[0].selectionProbability).toBeCloseTo(0.2);
    expect(distribution.entries[1].selectionProbability).toBeCloseTo(0.32);
    expect(distribution.entries[2].selectionProbability).toBeCloseTo(0.384);
    expect(distribution.unmatchedProbability).toBeCloseTo(0.096);
    expect(distribution.totalProbability).toBeCloseTo(1, 12);
  });
});

describe("computeNeurogenesisPressure", () => {
  it("is high for an unfamiliar path with high representation error", () => {
    const pressure = computeNeurogenesisPressure({
      selectedHabitProbability: 0,
      candidateAcceptances: [0],
      alpha: 0,
      previousSmoothed: 0,
      lambda: 0.8,
    });
    expect(pressure.habitDistance).toBe(1);
    expect(pressure.representationError).toBe(1);
    expect(pressure.instantaneous).toBe(1);
    expect(pressure.smoothed).toBeCloseTo(0.2);
  });

  it("stays low when a neuron explains the signal even on an unfamiliar path", () => {
    const pressure = computeNeurogenesisPressure({
      selectedHabitProbability: 0,
      candidateAcceptances: [0.99, 0.2],
      alpha: 0,
      previousSmoothed: 0,
      lambda: 0.8,
    });
    expect(pressure.habitDistance).toBe(1);
    expect(pressure.representationError).toBeCloseTo(0.01);
    expect(pressure.instantaneous).toBeCloseTo(0.01);
  });

  it("uses 1-alpha as representation error when there are no candidates", () => {
    const pressure = computeNeurogenesisPressure({
      candidateAcceptances: [],
      alpha: 0.25,
      previousSmoothed: 0,
      lambda: 0,
    });
    expect(pressure.habitDistance).toBe(1);
    expect(pressure.representationError).toBe(0.75);
    expect(pressure.instantaneous).toBe(0.75);
  });
});
