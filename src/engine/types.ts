export const MATRIX_SIZE = 32 as const;
export const DEFAULT_MAX_STEPS = 25;
export const DEFAULT_PRESSURE_SMOOTHING = 0.8;

/**
 * Runtime matrices are validated as 32 x 32 by the simulation boundary. Keeping
 * the type structural makes matrix editing in the UI straightforward.
 */
export type Matrix = number[][];

export interface Neuron {
  id: string;
  matrix: Matrix;
}

export interface SeededRandomState {
  /** The normalized, original uint32 seed shown to the user. */
  seed: number;
  /** Internal uint32 state after `draws` samples. */
  state: number;
  draws: number;
}

export interface NetworkState {
  neurons: Neuron[];
  /** Raw, persistent muscle-memory evidence W. */
  habitEvidence: number[][];
  /** Row-normalized muscle-memory probabilities H. */
  habitProbabilities: number[][];
  /** Cosine similarity matrix K for the neuron prototypes. */
  neuronSimilarities: number[][];
}

export type CandidateOutcome = "rejected" | "accepted" | "not-evaluated";

export interface CandidateEvaluation {
  rank: number;
  neuronId: string;
  neuronIndex: number;
  habitProbability: number;
  signalCompatibility: number;
  /** Probability that evaluation reaches this candidate. */
  reachProbability: number;
  /** Unconditional probability that this candidate is selected. */
  selectionProbability: number;
  outcome: CandidateOutcome;
}

export interface SequentialSelectionEntry {
  candidateIndex: number;
  acceptance: number;
  reachProbability: number;
  selectionProbability: number;
}

export interface SequentialSelectionDistribution {
  entries: SequentialSelectionEntry[];
  unmatchedProbability: number;
  totalProbability: number;
}

export interface SignalProcessingResult {
  compatibility: number;
  alpha: number;
  outputSignal: Matrix;
  delta: Matrix;
  /** Frobenius norm of the absolute cell-change matrix. */
  changeMagnitude: number;
}

export interface NeurogenesisPressure {
  habitDistance: number;
  representationError: number;
  instantaneous: number;
  smoothed: number;
}

export interface NeurogenesisPressureInput {
  /** Omit when no candidate was accepted. */
  selectedHabitProbability?: number;
  candidateAcceptances: readonly number[];
  /** Used for representation error only when there are no candidates. */
  alpha: number;
  previousSmoothed: number;
  lambda?: number;
}

export interface LoopReport {
  detected: boolean;
  /** Repeating node sequence, empty when no short loop was found. */
  cycle: string[];
}

export type SimulationStopReason =
  | "max-steps"
  | "no-candidates"
  | "no-match"
  | "user-stopped"
  | "invalid-state";

export type SimulationStatus = "idle" | "running" | "paused" | "stopped";

export interface SimulationStepLog {
  /** One-based step number. */
  step: number;
  fromNeuronId: string;
  toNeuronId: string | null;
  incomingSignal: Matrix;
  neuronMatrix: Matrix;
  transformedSignal: Matrix;
  delta: Matrix;
  compatibility: number;
  alpha: number;
  changeMagnitude: number;
  candidates: CandidateEvaluation[];
  unmatched: boolean;
  /** H_ij immediately before learning, or null without a transition. */
  habitBefore: number | null;
  /** H_ij immediately after learning, or null without a transition. */
  habitAfter: number | null;
  pressure: NeurogenesisPressure;
  /** Categorical draw used for selection, null if no candidate exists. */
  rngSample: number | null;
  loop: LoopReport;
  stopReason: SimulationStopReason | null;
}

export interface SimulationState {
  network: NetworkState;
  initialSignal: Matrix;
  signal: Matrix;
  currentNeuronId: string;
  step: number;
  maxSteps: number;
  rng: SeededRandomState;
  pressure: NeurogenesisPressure;
  timeline: SimulationStepLog[];
  /** Starts with the initial neuron and appends each accepted destination. */
  path: string[];
  status: SimulationStatus;
  stopReason: SimulationStopReason | null;
  loop: LoopReport;
  error: string | null;
}

export interface CandidateOrderResult {
  order: number[];
  rng: SeededRandomState;
}

export interface SimulationOptions {
  seed?: number;
  maxSteps?: number;
  currentNeuronId?: string;
}

export interface StepSimulationOptions {
  pressureSmoothing?: number;
}
