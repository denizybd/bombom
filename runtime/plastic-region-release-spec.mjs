import { stableRuntimeHash } from './artifacts.mjs'

export const PLASTIC_REGION_RELEASE_RUNNER_VERSION = 1
export const PLASTIC_REGION_RELEASE_SOURCE_POLICY = 'controlled-benchmark-only-no-ollama-v1'
export const PLASTIC_REGION_RELEASE_SEEDS = Object.freeze([0x51a7, 0x9e37, 0xc0de, 0xd1ce, 0xf17e])
export const PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT = 30
export const PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT = 96
export const PLASTIC_REGION_RELEASE_DEFAULT_MAXIMUM_TICKS = 16

const ACTORS = Object.freeze(['Ada', 'Bora', 'Cem', 'Duru'])
const ACTIONS = Object.freeze(['taşıdı', 'çekti', 'yerleştirdi'])
const PATIENTS = Object.freeze(['kırmızı kutu', 'mavi kutu', 'sarı kutu', 'yeşil kutu'])
const INSTRUMENTS = Object.freeze(['kanca', 'halat', 'robot kol'])

function accusative(value) {
  return ({
    'kırmızı kutu': 'kırmızı kutuyu',
    'mavi kutu': 'mavi kutuyu',
    'sarı kutu': 'sarı kutuyu',
    'yeşil kutu': 'yeşil kutuyu',
  })[value] ?? value
}

function instrumental(value) {
  return ({ kanca: 'kancayla', halat: 'halatla', 'robot kol': 'robot kolla' })[value] ?? `${value} ile`
}

export function plasticRegionReleaseCases(caseCount = PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT) {
  if (!Number.isSafeInteger(caseCount) || caseCount < PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT ||
      caseCount > PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT) {
    throw new RangeError(
      `Release holdout vaka sayısı ${PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT}..${PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT} olmalıdır.`,
    )
  }
  return ACTIONS.flatMap((action) => ACTORS.flatMap((actor) =>
    PATIENTS.flatMap((patient) => INSTRUMENTS.map((instrument) => {
      const combinationId = stableRuntimeHash(
        'plastic-region-release-combination-v1', actor, action, patient, instrument,
      ).slice(0, 20)
      const instrumentSurface = instrumental(instrument)
      return Object.freeze({
        id: `PRB-${combinationId}`,
        sourceGroupId: `PRBG-${combinationId}`,
        split: 'frozen-holdout',
        prompt: `${actor} ${accusative(patient)} ${instrumentSurface} ${action}.`,
        expected: Object.freeze({
          format: 'event-graph-v1', actor, action, patient, instrument,
          operators: Object.freeze([]), negated: false, goals: Object.freeze([]),
        }),
      })
    }))),
  ).sort((left, right) =>
    stableRuntimeHash('plastic-region-release-case-order-v1', left.id)
      .localeCompare(stableRuntimeHash('plastic-region-release-case-order-v1', right.id)) ||
    left.id.localeCompare(right.id)).slice(0, caseCount)
}

function normalizedGraphValue(value) {
  if (value === null || value === undefined) return null
  const result = String(value).normalize('NFC').toLocaleLowerCase('tr-TR')
    .replace(/[\p{P}\p{S}]+/gu, ' ').replace(/\s+/gu, ' ').trim()
  return result || null
}

function graphAtoms(value) {
  const atoms = new Set()
  for (const key of ['actor', 'action', 'patient', 'instrument']) {
    const entry = normalizedGraphValue(value?.[key])
    if (entry) atoms.add(`${key}:${entry}`)
  }
  for (const entry of value?.operators ?? []) {
    const normalized = normalizedGraphValue(entry)
    if (normalized) atoms.add(`operator:${normalized}`)
  }
  for (const goal of value?.goals ?? []) {
    const relation = normalizedGraphValue(goal?.relation)
    const target = normalizedGraphValue(goal?.target)
    if (relation && target) atoms.add(`goal:${relation}:${target}`)
  }
  atoms.add(`negated:${value?.negated ? '1' : '0'}`)
  return atoms
}

export function scorePlasticRegionReleaseGraph(predicted, expected) {
  const predictedAtoms = graphAtoms(predicted)
  const expectedAtoms = graphAtoms(expected)
  let matched = 0
  for (const atom of predictedAtoms) if (expectedAtoms.has(atom)) matched += 1
  const precision = predictedAtoms.size ? matched / predictedAtoms.size : expectedAtoms.size ? 0 : 1
  const recall = expectedAtoms.size ? matched / expectedAtoms.size : predictedAtoms.size ? 0 : 1
  return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
}

export function plasticRegionReleaseCorpusFingerprint(caseCount) {
  return stableRuntimeHash('plastic-region-release-corpus-v1', plasticRegionReleaseCases(caseCount))
}

export const PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT = stableRuntimeHash(
  'plastic-region-release-scorer-contract-v1',
  String(normalizedGraphValue),
  String(graphAtoms),
  String(scorePlasticRegionReleaseGraph),
)

export const PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT = stableRuntimeHash(
  'plastic-region-release-engine-contract-v1',
  'thought-runtime-engine-request-v1',
  'seeded-stochastic',
  'learning-off',
  'teacher-off',
  'event-graph-v1',
)

const RUNNER_SPEC = Object.freeze({
  version: PLASTIC_REGION_RELEASE_RUNNER_VERSION,
  sourcePolicy: PLASTIC_REGION_RELEASE_SOURCE_POLICY,
  seeds: PLASTIC_REGION_RELEASE_SEEDS,
  defaultCaseCount: PLASTIC_REGION_RELEASE_DEFAULT_CASE_COUNT,
  maximumCaseCount: PLASTIC_REGION_RELEASE_MAXIMUM_CASE_COUNT,
  maximumTicks: PLASTIC_REGION_RELEASE_DEFAULT_MAXIMUM_TICKS,
  comparison: 'paired-current-base-candidate-footprint-sham-v2',
  confidence: 'source-group-bootstrap-95-conservative-bound-v1',
  automaticActivation: false,
  engineContractFingerprint: PLASTIC_REGION_RELEASE_ENGINE_CONTRACT_FINGERPRINT,
  scorerFingerprint: PLASTIC_REGION_RELEASE_SCORER_FINGERPRINT,
})

export const PLASTIC_REGION_RELEASE_RUNNER_FINGERPRINT = stableRuntimeHash(
  'plastic-region-release-runner-v1',
  RUNNER_SPEC,
)
