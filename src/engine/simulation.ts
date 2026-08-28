import {
  cloneMatrix,
  computeHabitProbabilities,
  computeNeurogenesisPressure,
  computeNeuronSimilarityMatrix,
  computeSequentialSelectionDistribution,
  isSignedUnitMatrix,
  orderCandidatesByHabit,
  processSignalAtNeuron,
  updateHabitEvidence,
} from "./math";
import { createSeededRandom, nextRandom } from "./random";
import {
  DEFAULT_MAX_STEPS,
  DEFAULT_PRESSURE_SMOOTHING,
  MATRIX_SIZE,
  type CandidateEvaluation,
  type LoopReport,
  type Matrix,
  type NetworkState,
  type Neuron,
  type NeurogenesisPressure,
  type SequentialSelectionDistribution,
  type SimulationOptions,
  type SimulationState,
  type SimulationStopReason,
  type StepSimulationOptions,
} from "./types";

const EMPTY_PRESSURE: NeurogenesisPressure = {
  habitDistance: 0,
  representationError: 0,
  instantaneous: 0,
  smoothed: 0,
};

const NO_LOOP: LoopReport = { detected: false, cycle: [] };

function cloneNeuron(neuron: Neuron): Neuron {
  return { ...neuron, matrix: cloneMatrix(neuron.matrix) };
}

function safeEvidence(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

export function recomputeNetworkDerivedState(network: NetworkState): NetworkState {
  const neurons = network.neurons.map(cloneNeuron);
  const size = neurons.length;
  const habitEvidence = Array.from({ length: size }, (_, rowIndex) =>
    Array.from({ length: size }, (_, columnIndex) =>
      rowIndex === columnIndex
        ? 0
        : safeEvidence(network.habitEvidence[rowIndex]?.[columnIndex] ?? 0),
    ),
  );

  return {
    neurons,
    habitEvidence,
    habitProbabilities: computeHabitProbabilities(habitEvidence),
    neuronSimilarities: computeNeuronSimilarityMatrix(neurons),
  };
}

export function createInitialNetwork(neuron: Neuron): NetworkState {
  if (!neuron.id.trim()) throw new RangeError("A neuron id cannot be empty.");
  if (!isSignedUnitMatrix(neuron.matrix, MATRIX_SIZE)) {
    throw new RangeError(`Neuron matrices must be finite ${MATRIX_SIZE} x ${MATRIX_SIZE} values in [-1, 1].`);
  }

  const neurons = [cloneNeuron(neuron)];
  const habitEvidence = [[0]];
  return {
    neurons,
    habitEvidence,
    habitProbabilities: [[0]],
    neuronSimilarities: computeNeuronSimilarityMatrix(neurons),
  };
}

export function createInitialSimulationState(
  network: NetworkState,
  initialSignal: Matrix,
  options: SimulationOptions = {},
): SimulationState {
  const normalizedNetwork = recomputeNetworkDerivedState(network);
  if (normalizedNetwork.neurons.length === 0) {
    throw new RangeError("A simulation needs at least one neuron.");
  }
  if (!isSignedUnitMatrix(initialSignal, MATRIX_SIZE)) {
    throw new RangeError(`Signals must be finite ${MATRIX_SIZE} x ${MATRIX_SIZE} values in [-1, 1].`);
  }

  const currentNeuronId =
    options.currentNeuronId ?? normalizedNetwork.neurons[0].id;
  if (!normalizedNetwork.neurons.some((neuron) => neuron.id === currentNeuronId)) {
    throw new RangeError("The initial neuron does not exist in the network.");
  }

  const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
  if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
    throw new RangeError("Maximum steps must be a positive integer.");
  }

  return {
    network: normalizedNetwork,
    initialSignal: cloneMatrix(initialSignal),
    signal: cloneMatrix(initialSignal),
    currentNeuronId,
    step: 0,
    maxSteps,
    rng: createSeededRandom(options.seed ?? 1),
    pressure: { ...EMPTY_PRESSURE },
    timeline: [],
    path: [currentNeuronId],
    status: "idle",
    stopReason: null,
    loop: { ...NO_LOOP, cycle: [] },
    error: null,
  };
}

export function expandNetworkForNewNeuron(
  network: NetworkState,
  neuron: Neuron,
  initialEvidence = 1,
): NetworkState {
  if (!neuron.id.trim()) throw new RangeError("A neuron id cannot be empty.");
  if (network.neurons.some((candidate) => candidate.id === neuron.id)) {
    throw new RangeError(`Neuron id '${neuron.id}' already exists.`);
  }
  if (!isSignedUnitMatrix(neuron.matrix, MATRIX_SIZE)) {
    throw new RangeError(`Neuron matrices must be finite ${MATRIX_SIZE} x ${MATRIX_SIZE} values in [-1, 1].`);
  }
  if (!Number.isFinite(initialEvidence) || initialEvidence < 0) {
    throw new RangeError("Initial habit evidence must be finite and non-negative.");
  }

  const previousSize = network.neurons.length;
  const nextSize = previousSize + 1;
  const neurons = [...network.neurons.map(cloneNeuron), cloneNeuron(neuron)];
  const habitEvidence = Array.from({ length: nextSize }, (_, rowIndex) =>
    Array.from({ length: nextSize }, (_, columnIndex) => {
      if (rowIndex === columnIndex) return 0;
      if (rowIndex < previousSize && columnIndex < previousSize) {
        return safeEvidence(network.habitEvidence[rowIndex]?.[columnIndex] ?? 0);
      }
      return initialEvidence;
    }),
  );

  return {
    neurons,
    habitEvidence,
    habitProbabilities: computeHabitProbabilities(habitEvidence),
    neuronSimilarities: computeNeuronSimilarityMatrix(neurons),
  };
}

function validateSimulationState(state: SimulationState): string | null {
  const { network } = state;
  if (network.neurons.length === 0) return "The network has no neurons.";
  if (!Number.isInteger(state.step) || state.step < 0) return "Step is invalid.";
  if (!Number.isInteger(state.maxSteps) || state.maxSteps <= 0) {
    return "Maximum steps is invalid.";
  }
  if (!isSignedUnitMatrix(state.signal, MATRIX_SIZE)) return "Signal matrix is invalid.";
  if (!isSignedUnitMatrix(state.initialSignal, MATRIX_SIZE)) {
    return "Initial signal matrix is invalid.";
  }

  const ids = new Set<string>();
  for (const neuron of network.neurons) {
    if (!neuron.id.trim() || ids.has(neuron.id)) return "Neuron ids are invalid.";
    ids.add(neuron.id);
    if (!isSignedUnitMatrix(neuron.matrix, MATRIX_SIZE)) {
      return `Neuron '${neuron.id}' has an invalid matrix.`;
    }
  }
  if (!ids.has(state.currentNeuronId)) return "Current neuron does not exist.";

  const size = network.neurons.length;
  if (network.habitEvidence.length !== size) return "Habit evidence size is invalid.";
  for (let row = 0; row < size; row += 1) {
    if (network.habitEvidence[row]?.length !== size) {
      return "Habit evidence size is invalid.";
    }
    for (let column = 0; column < size; column += 1) {
      const value = network.habitEvidence[row][column];
      if (!Number.isFinite(value) || value < 0) return "Habit evidence is invalid.";
      if (row === column && value !== 0) return "Self-connections must remain zero.";
    }
  }

  if (
    !Number.isFinite(state.rng.seed) ||
    !Number.isFinite(state.rng.state) ||
    !Number.isInteger(state.rng.draws) ||
    state.rng.draws < 0 ||
    !Number.isFinite(state.pressure.smoothed)
  ) {
    return "Random or pressure state is numerically invalid.";
  }
  return null;
}

function stopWithError(state: SimulationState, message: string): SimulationState {
  return {
    ...state,
    status: "stopped",
    stopReason: "invalid-state",
    error: message,
  };
}

function sampleDistribution(
  distribution: SequentialSelectionDistribution,
  rng: SimulationState["rng"],
): { selectedIndex: number | null; sample: number; rng: SimulationState["rng"] } {
  const draw = nextRandom(rng);
  let cumulative = 0;
  for (const entry of distribution.entries) {
    cumulative += entry.selectionProbability;
    if (draw.value < cumulative) {
      return { selectedIndex: entry.candidateIndex, sample: draw.value, rng: draw.rng };
    }
  }
  return { selectedIndex: null, sample: draw.value, rng: draw.rng };
}

/** Detects an immediately repeated path fragment of two to four edges. */
export function detectShortLoop(path: readonly string[]): LoopReport {
  const maximumCycleLength = Math.min(4, Math.floor(path.length / 2));
  for (let length = 2; length <= maximumCycleLength; length += 1) {
    const previous = path.slice(path.length - length * 2, path.length - length);
    const latest = path.slice(path.length - length);
    if (latest.every((id, index) => id === previous[index])) {
      return { detected: true, cycle: latest };
    }
  }
  return { detected: false, cycle: [] };
}

function resolveStopReason(
  candidateCount: number,
  selectedIndex: number | null,
  nextStep: number,
  maxSteps: number,
): SimulationStopReason | null {
  if (candidateCount === 0) return "no-candidates";
  if (selectedIndex === null) return "no-match";
  if (nextStep >= maxSteps) return "max-steps";
  return null;
}

export function stepSimulation(
  state: SimulationState,
  options: StepSimulationOptions = {},
): SimulationState {
  if (state.status === "stopped" && state.stopReason !== null) return state;
  if (state.step >= state.maxSteps) {
    return { ...state, status: "stopped", stopReason: "max-steps", error: null };
  }

  const validationError = validateSimulationState(state);
  if (validationError) return stopWithError(state, validationError);

  try {
    const baseNetwork = recomputeNetworkDerivedState(state.network);
    const fromNeuronIndex = baseNetwork.neurons.findIndex(
      (neuron) => neuron.id === state.currentNeuronId,
    );
    const fromNeuron = baseNetwork.neurons[fromNeuronIndex];
    const processed = processSignalAtNeuron(state.signal, fromNeuron.matrix);

    const ordered = orderCandidatesByHabit(
      baseNetwork.habitProbabilities,
      fromNeuronIndex,
      state.rng,
    );
    const acceptances = Array(baseNetwork.neurons.length).fill(0) as number[];
    for (const candidateIndex of ordered.order) {
      acceptances[candidateIndex] = Math.max(
        0,
        processSignalAtNeuron(
          processed.outputSignal,
          baseNetwork.neurons[candidateIndex].matrix,
        ).compatibility,
      );
    }

    const distribution = computeSequentialSelectionDistribution(
      ordered.order,
      acceptances,
    );
    const sampled =
      ordered.order.length > 0
        ? sampleDistribution(distribution, ordered.rng)
        : { selectedIndex: null, sample: null, rng: ordered.rng };
    const selectedIndex = sampled.selectedIndex;
    const selectedHabitProbability =
      selectedIndex === null
        ? undefined
        : baseNetwork.habitProbabilities[fromNeuronIndex][selectedIndex];
    const pressure = computeNeurogenesisPressure({
      selectedHabitProbability,
      candidateAcceptances: ordered.order.map((index) => acceptances[index]),
      alpha: processed.alpha,
      previousSmoothed: state.pressure.smoothed,
      lambda: options.pressureSmoothing ?? DEFAULT_PRESSURE_SMOOTHING,
    });

    let network = baseNetwork;
    let habitBefore: number | null = null;
    let habitAfter: number | null = null;
    if (selectedIndex !== null) {
      habitBefore = baseNetwork.habitProbabilities[fromNeuronIndex][selectedIndex];
      const habitEvidence = updateHabitEvidence(
        baseNetwork.habitEvidence,
        fromNeuronIndex,
        selectedIndex,
      );
      const habitProbabilities = computeHabitProbabilities(habitEvidence);
      habitAfter = habitProbabilities[fromNeuronIndex][selectedIndex];
      network = { ...baseNetwork, habitEvidence, habitProbabilities };
    }

    const selectedRank = distribution.entries.findIndex(
      (entry) => entry.candidateIndex === selectedIndex,
    );
    const candidates: CandidateEvaluation[] = distribution.entries.map(
      (entry, index) => ({
        rank: index + 1,
        neuronId: baseNetwork.neurons[entry.candidateIndex].id,
        neuronIndex: entry.candidateIndex,
        habitProbability:
          baseNetwork.habitProbabilities[fromNeuronIndex][entry.candidateIndex],
        signalCompatibility: entry.acceptance,
        reachProbability: entry.reachProbability,
        selectionProbability: entry.selectionProbability,
        outcome:
          selectedIndex === null || index < selectedRank
            ? "rejected"
            : index === selectedRank
              ? "accepted"
              : "not-evaluated",
      }),
    );

    const toNeuronId =
      selectedIndex === null ? null : baseNetwork.neurons[selectedIndex].id;
    const path =
      toNeuronId === null
        ? [...state.path]
        : [...(state.path.length > 0 ? state.path : [fromNeuron.id]), toNeuronId];
    const loop = detectShortLoop(path);
    const nextStep = state.step + 1;
    const stopReason = resolveStopReason(
      ordered.order.length,
      selectedIndex,
      nextStep,
      state.maxSteps,
    );

    const log = {
      step: nextStep,
      fromNeuronId: fromNeuron.id,
      toNeuronId,
      incomingSignal: cloneMatrix(state.signal),
      neuronMatrix: cloneMatrix(fromNeuron.matrix),
      transformedSignal: cloneMatrix(processed.outputSignal),
      delta: cloneMatrix(processed.delta),
      compatibility: processed.compatibility,
      alpha: processed.alpha,
      changeMagnitude: processed.changeMagnitude,
      candidates,
      unmatched: selectedIndex === null,
      habitBefore,
      habitAfter,
      pressure,
      rngSample: sampled.sample,
      loop,
      stopReason,
    };

    return {
      ...state,
      network,
      signal: cloneMatrix(processed.outputSignal),
      currentNeuronId: toNeuronId ?? fromNeuron.id,
      step: nextStep,
      rng: sampled.rng,
      pressure,
      timeline: [...state.timeline, log],
      path,
      status:
        stopReason !== null
          ? "stopped"
          : state.status === "running"
            ? "running"
            : "paused",
      stopReason,
      loop,
      error: null,
    };
  } catch (error) {
    return stopWithError(
      state,
      error instanceof Error ? error.message : "Unknown numerical simulation error.",
    );
  }
}

export function stopSimulationByUser(state: SimulationState): SimulationState {
  return {
    ...state,
    status: "stopped",
    stopReason: "user-stopped",
    error: null,
  };
}
