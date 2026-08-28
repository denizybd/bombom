import { describe, expect, it } from "vitest";
import {
  createInitialNetwork,
  createInitialSimulationState,
  createMatrix,
  expandNetworkForNewNeuron,
  stepSimulation,
  stopSimulationByUser,
  type SimulationState,
} from "./index";

function threeNeuronNetwork() {
  const initial = createInitialNetwork({ id: "N1", matrix: createMatrix(0.5) });
  const two = expandNetworkForNewNeuron(initial, {
    id: "N2",
    matrix: createMatrix(0.4),
  });
  return expandNetworkForNewNeuron(two, { id: "N3", matrix: createMatrix(0.3) });
}

function runUntilStopped(state: SimulationState): SimulationState {
  let current: SimulationState = { ...state, status: "running" };
  while (current.status !== "stopped") current = stepSimulation(current);
  return current;
}

describe("network expansion", () => {
  it("safely grows W, H and K with one-valued new off-diagonal evidence", () => {
    const initial = createInitialNetwork({ id: "N1", matrix: createMatrix(0) });
    const expanded = expandNetworkForNewNeuron(initial, {
      id: "N2",
      matrix: createMatrix(0.5),
    });

    expect(expanded.neurons.map((neuron) => neuron.id)).toEqual(["N1", "N2"]);
    expect(expanded.habitEvidence).toEqual([
      [0, 1],
      [1, 0],
    ]);
    expect(expanded.habitProbabilities).toEqual([
      [0, 1],
      [1, 0],
    ]);
    expect(expanded.neuronSimilarities).toHaveLength(2);
    expect(expanded.neuronSimilarities.every((row) => row.length === 2)).toBe(true);
  });
});

describe("stepSimulation", () => {
  it("stops a single-neuron start with a no-candidates timeline entry", () => {
    const network = createInitialNetwork({ id: "N1", matrix: createMatrix(1) });
    const initial = createInitialSimulationState(network, createMatrix(0.5), { seed: 3 });
    const result = stepSimulation(initial);

    expect(result.status).toBe("stopped");
    expect(result.stopReason).toBe("no-candidates");
    expect(result.step).toBe(1);
    expect(result.timeline).toHaveLength(1);
    expect(result.timeline[0].candidates).toEqual([]);
    expect(result.timeline[0].rngSample).toBeNull();
    expect(result.timeline[0].pressure.representationError).toBeCloseTo(
      1 - result.timeline[0].alpha,
    );
  });

  it("stops with no-candidates when the current row is all zero in a multi-neuron network", () => {
    const network = threeNeuronNetwork();
    network.habitEvidence[0] = [0, 0, 0];
    const initial = createInitialSimulationState(network, createMatrix(0.5), { seed: 7 });
    const result = stepSimulation(initial);

    expect(result.status).toBe("stopped");
    expect(result.stopReason).toBe("no-candidates");
    expect(result.timeline[0].candidates).toEqual([]);
    expect(result.timeline[0].rngSample).toBeNull();
    expect(result.network.habitProbabilities[0]).toEqual([0, 0, 0]);
  });

  it("stops with no-match when all candidates have zero acceptance", () => {
    const initial = createInitialNetwork({ id: "N1", matrix: createMatrix(1) });
    const network = expandNetworkForNewNeuron(initial, {
      id: "N2",
      matrix: createMatrix(-1),
    });
    const state = createInitialSimulationState(network, createMatrix(1), { seed: 19 });
    const result = stepSimulation(state);

    expect(result.stopReason).toBe("no-match");
    expect(result.timeline[0].unmatched).toBe(true);
    expect(result.timeline[0].candidates[0].outcome).toBe("rejected");
    expect(result.timeline[0].candidates[0].signalCompatibility).toBe(0);
    expect(result.network.habitEvidence).toEqual(network.habitEvidence);
  });

  it("learns only an accepted transition and reports H before and after", () => {
    const network = threeNeuronNetwork();
    const state = createInitialSimulationState(network, createMatrix(0.5), {
      seed: 42,
      maxSteps: 5,
    });
    const result = stepSimulation(state);
    const log = result.timeline[0];

    expect(log.toNeuronId).not.toBeNull();
    expect(log.habitBefore).toBeCloseTo(0.5);
    expect(log.habitAfter).toBeCloseTo(2 / 3);
    expect(result.network.habitEvidence[0].filter((value) => value === 2)).toHaveLength(1);
    expect(network.habitEvidence[0]).toEqual([0, 1, 1]);
    expect(log.candidates.some((candidate) => candidate.outcome === "accepted")).toBe(true);
  });

  it("stops exactly at the configured maximum step after a valid transition", () => {
    const network = expandNetworkForNewNeuron(
      createInitialNetwork({ id: "N1", matrix: createMatrix(0.5) }),
      { id: "N2", matrix: createMatrix(0.5) },
    );
    const state = createInitialSimulationState(network, createMatrix(0.5), {
      seed: 1,
      maxSteps: 1,
    });
    const result = stepSimulation(state);
    expect(result.step).toBe(1);
    expect(result.stopReason).toBe("max-steps");
    expect(result.timeline[0].toNeuronId).toBe("N2");
  });

  it("produces exactly the same path, logs and learned W from the same seed", () => {
    const network = threeNeuronNetwork();
    const makeState = () =>
      createInitialSimulationState(network, createMatrix(0.5), {
        seed: 90210,
        maxSteps: 8,
      });
    const first = runUntilStopped(makeState());
    const second = runUntilStopped(makeState());

    expect(first.path).toEqual(second.path);
    expect(first.timeline).toEqual(second.timeline);
    expect(first.network.habitEvidence).toEqual(second.network.habitEvidence);
    expect(first.rng).toEqual(second.rng);
    expect(first.stopReason).toBe("max-steps");
  });

  it("allows loops, detects a repeated short path, and does not stop for the loop", () => {
    const network = expandNetworkForNewNeuron(
      createInitialNetwork({ id: "N1", matrix: createMatrix(0.5) }),
      { id: "N2", matrix: createMatrix(0.5) },
    );
    const final = runUntilStopped(
      createInitialSimulationState(network, createMatrix(0.5), {
        seed: 5,
        maxSteps: 5,
      }),
    );
    expect(final.path).toEqual(["N1", "N2", "N1", "N2", "N1", "N2"]);
    expect(final.timeline.some((entry) => entry.loop.detected)).toBe(true);
    expect(final.stopReason).toBe("max-steps");
  });

  it("turns malformed numerical state into an explicit invalid-state stop", () => {
    const network = createInitialNetwork({ id: "N1", matrix: createMatrix(0) });
    const state = createInitialSimulationState(network, createMatrix(0));
    state.signal[0][0] = Number.NaN;
    const result = stepSimulation(state);
    expect(result.status).toBe("stopped");
    expect(result.stopReason).toBe("invalid-state");
    expect(result.error).toMatch(/signal/i);
  });

  it("records an explicit user stop without changing learned state", () => {
    const state = createInitialSimulationState(threeNeuronNetwork(), createMatrix(0.5));
    const stopped = stopSimulationByUser(state);
    expect(stopped.stopReason).toBe("user-stopped");
    expect(stopped.network).toBe(state.network);
    expect(stopped.timeline).toEqual([]);
  });
});
