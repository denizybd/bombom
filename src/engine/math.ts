import { nextRandom } from "./random";
import {
  MATRIX_SIZE,
  DEFAULT_PRESSURE_SMOOTHING,
  type CandidateOrderResult,
  type Matrix,
  type NeurogenesisPressure,
  type NeurogenesisPressureInput,
  type Neuron,
  type SeededRandomState,
  type SequentialSelectionDistribution,
  type SignalProcessingResult,
} from "./types";

export const COSINE_EPSILON = 1e-8;

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function createMatrix(value = 0, size = MATRIX_SIZE): Matrix {
  return Array.from({ length: size }, () => Array(size).fill(value));
}

export function cloneMatrix(matrix: readonly (readonly number[])[]): Matrix {
  return matrix.map((row) => [...row]);
}

export function isFiniteMatrix(
  matrix: readonly (readonly number[])[],
  size?: number,
): boolean {
  if (!Array.isArray(matrix)) return false;
  if (size !== undefined && matrix.length !== size) return false;
  if (matrix.length === 0) return false;

  const width = size ?? matrix[0]?.length;
  if (!width) return false;
  return matrix.every(
    (row) =>
      Array.isArray(row) &&
      row.length === width &&
      row.every((value) => Number.isFinite(value)),
  );
}

export function isSignedUnitMatrix(
  matrix: readonly (readonly number[])[],
  size = MATRIX_SIZE,
): boolean {
  return (
    isFiniteMatrix(matrix, size) &&
    matrix.every((row) => row.every((value) => value >= -1 && value <= 1))
  );
}

function assertCompatibleMatrices(
  first: readonly (readonly number[])[],
  second: readonly (readonly number[])[],
): void {
  if (!isFiniteMatrix(first) || !isFiniteMatrix(second)) {
    throw new RangeError("Matrices must be finite, rectangular and non-empty.");
  }
  if (
    first.length !== second.length ||
    first.some((row, rowIndex) => row.length !== second[rowIndex].length)
  ) {
    throw new RangeError("Matrices must have matching dimensions.");
  }
}

export function cosineSimilarity(
  first: readonly (readonly number[])[],
  second: readonly (readonly number[])[],
  epsilon = COSINE_EPSILON,
): number {
  assertCompatibleMatrices(first, second);

  let dotProduct = 0;
  let firstSquared = 0;
  let secondSquared = 0;

  for (let row = 0; row < first.length; row += 1) {
    for (let column = 0; column < first[row].length; column += 1) {
      const left = first[row][column];
      const right = second[row][column];
      dotProduct += left * right;
      firstSquared += left * left;
      secondSquared += right * right;
    }
  }

  const denominator = Math.sqrt(firstSquared) * Math.sqrt(secondSquared) + epsilon;
  const similarity = dotProduct / denominator;
  return Number.isFinite(similarity) ? clamp(similarity, -1, 1) : 0;
}

export function processSignalAtNeuron(
  signal: readonly (readonly number[])[],
  neuronMatrix: readonly (readonly number[])[],
): SignalProcessingResult {
  assertCompatibleMatrices(signal, neuronMatrix);

  const compatibility = cosineSimilarity(signal, neuronMatrix);
  const alpha = clamp(compatibility, 0, 1);
  let squaredChange = 0;

  const outputSignal = signal.map((row, rowIndex) =>
    row.map((value, columnIndex) =>
      clamp(
        (1 - alpha) * value + alpha * neuronMatrix[rowIndex][columnIndex],
        -1,
        1,
      ),
    ),
  );

  const delta = outputSignal.map((row, rowIndex) =>
    row.map((value, columnIndex) => {
      const change = Math.abs(value - signal[rowIndex][columnIndex]);
      squaredChange += change * change;
      return change;
    }),
  );

  return {
    compatibility,
    alpha,
    outputSignal,
    delta,
    changeMagnitude: Math.sqrt(squaredChange),
  };
}

export function computeHabitProbabilities(
  habitEvidence: readonly (readonly number[])[],
): number[][] {
  const size = habitEvidence.length;

  return Array.from({ length: size }, (_, rowIndex) => {
    const sanitized = Array.from({ length: size }, (_, columnIndex) => {
      if (rowIndex === columnIndex) return 0;
      const value = habitEvidence[rowIndex]?.[columnIndex] ?? 0;
      return Number.isFinite(value) && value > 0 ? value : 0;
    });
    const total = sanitized.reduce((sum, value) => sum + value, 0);
    return total > 0 ? sanitized.map((value) => value / total) : sanitized;
  });
}

function matrixFromNeuron(value: Neuron | Matrix): Matrix {
  return Array.isArray(value) ? value : value.matrix;
}

export function computeNeuronSimilarityMatrix(
  neurons: readonly (Neuron | Matrix)[],
): number[][] {
  const matrices = neurons.map(matrixFromNeuron);
  return matrices.map((matrix) =>
    matrices.map((candidate) => cosineSimilarity(matrix, candidate)),
  );
}

function sameHabitValue(first: number, second: number): boolean {
  return Math.abs(first - second) <= 1e-12;
}

/**
 * Sorts non-self neurons with positive H_ij by descending probability and
 * shuffles exact tie groups with the state-carrying seeded generator.
 */
export function orderCandidatesByHabit(
  habitProbabilities: readonly (readonly number[])[],
  currentNeuronIndex: number,
  rng: SeededRandomState,
): CandidateOrderResult {
  const size = habitProbabilities.length;
  if (
    !Number.isInteger(currentNeuronIndex) ||
    currentNeuronIndex < 0 ||
    currentNeuronIndex >= size
  ) {
    throw new RangeError("Current neuron index is outside the habit matrix.");
  }

  const row = habitProbabilities[currentNeuronIndex] ?? [];
  const candidates = Array.from({ length: size }, (_, index) => index)
    .filter(
      (index) =>
        index !== currentNeuronIndex &&
        Number.isFinite(row[index]) &&
        (row[index] ?? 0) > 0,
    )
    .sort((left, right) => {
      const difference = (row[right] ?? 0) - (row[left] ?? 0);
      return sameHabitValue(difference, 0) ? left - right : difference;
    });

  let nextState = rng;
  let groupStart = 0;
  while (groupStart < candidates.length) {
    let groupEnd = groupStart + 1;
    while (
      groupEnd < candidates.length &&
      sameHabitValue(
        row[candidates[groupStart]] ?? 0,
        row[candidates[groupEnd]] ?? 0,
      )
    ) {
      groupEnd += 1;
    }

    for (let index = groupEnd - 1; index > groupStart; index -= 1) {
      const draw = nextRandom(nextState);
      nextState = draw.rng;
      const swapIndex = groupStart + Math.floor(draw.value * (index - groupStart + 1));
      [candidates[index], candidates[swapIndex]] = [
        candidates[swapIndex],
        candidates[index],
      ];
    }
    groupStart = groupEnd;
  }

  return { order: candidates, rng: nextState };
}

export function computeSequentialSelectionDistribution(
  order: readonly number[],
  candidateAcceptances: readonly number[],
): SequentialSelectionDistribution {
  let reachProbability = 1;
  const entries = order.map((candidateIndex) => {
    const acceptance = clamp(candidateAcceptances[candidateIndex] ?? 0, 0, 1);
    const selectionProbability = reachProbability * acceptance;
    const entry = {
      candidateIndex,
      acceptance,
      reachProbability,
      selectionProbability,
    };
    reachProbability *= 1 - acceptance;
    return entry;
  });

  const unmatchedProbability = clamp(reachProbability, 0, 1);
  const totalProbability =
    entries.reduce((sum, entry) => sum + entry.selectionProbability, 0) +
    unmatchedProbability;

  return { entries, unmatchedProbability, totalProbability };
}

export function updateHabitEvidence(
  habitEvidence: readonly (readonly number[])[],
  fromNeuronIndex: number,
  toNeuronIndex: number,
  amount = 1,
): number[][] {
  const size = habitEvidence.length;
  if (
    !Number.isInteger(fromNeuronIndex) ||
    !Number.isInteger(toNeuronIndex) ||
    fromNeuronIndex < 0 ||
    toNeuronIndex < 0 ||
    fromNeuronIndex >= size ||
    toNeuronIndex >= size ||
    fromNeuronIndex === toNeuronIndex
  ) {
    throw new RangeError("Habit update requires two distinct, valid neurons.");
  }
  if (!Number.isFinite(amount) || amount < 0) {
    throw new RangeError("Habit update amount must be a finite non-negative number.");
  }

  return Array.from({ length: size }, (_, rowIndex) =>
    Array.from({ length: size }, (_, columnIndex) => {
      if (rowIndex === columnIndex) return 0;
      const previous = habitEvidence[rowIndex]?.[columnIndex] ?? 0;
      return rowIndex === fromNeuronIndex && columnIndex === toNeuronIndex
        ? previous + amount
        : previous;
    }),
  );
}

export function computeNeurogenesisPressure({
  selectedHabitProbability,
  candidateAcceptances,
  alpha,
  previousSmoothed,
  lambda = DEFAULT_PRESSURE_SMOOTHING,
}: NeurogenesisPressureInput): NeurogenesisPressure {
  const smoothing = clamp(lambda, 0, 1);
  const habitDistance =
    selectedHabitProbability === undefined
      ? 1
      : 1 - clamp(selectedHabitProbability, 0, 1);
  const representationError =
    candidateAcceptances.length > 0
      ? 1 - Math.max(...candidateAcceptances.map((value) => clamp(value, 0, 1)))
      : 1 - clamp(alpha, 0, 1);
  const instantaneous = clamp(habitDistance * representationError, 0, 1);
  const smoothed = clamp(
    smoothing * clamp(previousSmoothed, 0, 1) + (1 - smoothing) * instantaneous,
    0,
    1,
  );

  return { habitDistance, representationError, instantaneous, smoothed };
}
