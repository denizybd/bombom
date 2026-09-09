import { stableDigest } from '../families/sparse.ts'
import { compareText, normalizeSurface } from '../language-experience/deterministic.ts'
import { LANGUAGE_EOS_FAMILY, type SurfacePieceLevel } from '../language-experience/types.ts'
import type {
  LanguageOutputFamilyDefinition,
  LanguageOutputFamilyId,
  LanguageOutputManualTransition,
  LanguageOutputScaffoldV1,
  LanguageOutputSeedClassification,
  LanguageOutputSeedClassifierInput,
  LanguageOutputSeedNeuron,
  LanguageRoleGatewayV1,
} from './types.ts'

export const LANGUAGE_ROLE_GATEWAY_ID = '@language-output/gateway/role-to-word' as const

function outputDigest(label: string, ...parts: readonly unknown[]): string {
  return `${label}-${stableDigest(['language-output-v1', label, ...parts]).toUpperCase()}`
}

/** Runtime immutability for the exported manual contract, including nested arrays. */
function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const nested of Object.values(value as Record<string, unknown>)) deepFreeze(nested)
    Object.freeze(value)
  }
  return value
}

export const LANGUAGE_OUTPUT_FAMILY_IDS = Object.freeze({
  wordOrder: '@language-output/family/word-order',
  verbTense: '@language-output/family/verb-tense',
  caseMarking: '@language-output/family/case-marking',
  connectiveNegation: '@language-output/family/connective-negation',
  punctuation: '@language-output/family/punctuation',
  eos: LANGUAGE_EOS_FAMILY,
} satisfies Record<string, LanguageOutputFamilyId>)

const ALL_ROLES = ['actor', 'action', 'patient', 'instrument', 'operator', 'goal'] as const
const TOKEN_LEVELS: SurfacePieceLevel[] = ['token', 'subword', 'grapheme']

export const LANGUAGE_OUTPUT_FAMILY_DEFINITIONS: readonly LanguageOutputFamilyDefinition[] = deepFreeze([
  {
    id: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder,
    label: 'Sözcük sırası',
    description: 'Geçici düşünce rollerini Türkçe yüzey sırasına koyan, yüzey basmayan kontrol ailesi.',
    kind: 'control', order: 1, emitsSurface: false, frozen: true,
    acceptedSpanLevels: ['sentence', 'clause', 'token'],
    atlas: { layer: 'language-output', placement: 'peripheral-island', islandId: 'manual-language-scaffold', ordinal: 1, radialScale: 1.2 },
  },
  {
    id: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense,
    label: 'Fiil ve zaman',
    description: 'Eylem lemmasını zaman, kişi ve olumluluk biçimbirimleriyle yüzeye çıkaran aile.',
    kind: 'morphology', order: 2, emitsSurface: true, frozen: true,
    acceptedSpanLevels: [...TOKEN_LEVELS],
    atlas: { layer: 'language-output', placement: 'peripheral-island', islandId: 'manual-language-scaffold', ordinal: 2, radialScale: 1.22 },
  },
  {
    id: LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking,
    label: 'Nesne ve yönelme ekleri',
    description: 'Hasta, hedef, yer ve araç rollerini Türkçe hâl ekleriyle işaretleyen aile.',
    kind: 'morphology', order: 3, emitsSurface: true, frozen: true,
    acceptedSpanLevels: [...TOKEN_LEVELS],
    atlas: { layer: 'language-output', placement: 'peripheral-island', islandId: 'manual-language-scaffold', ordinal: 3, radialScale: 1.24 },
  },
  {
    id: LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation,
    label: 'Bağlaç ve olumsuzluk',
    description: 'Cümleler arası bağı ve olayın olumluluk kutbunu yüzeye taşıyan aile.',
    kind: 'surface', order: 4, emitsSurface: true, frozen: true,
    acceptedSpanLevels: ['clause', 'token', 'subword'],
    atlas: { layer: 'language-output', placement: 'peripheral-island', islandId: 'manual-language-scaffold', ordinal: 4, radialScale: 1.26 },
  },
  {
    id: LANGUAGE_OUTPUT_FAMILY_IDS.punctuation,
    label: 'Noktalama',
    description: 'Cümle türüne uygun tek terminal noktalama işaretini seçen yüzey ailesi.',
    kind: 'surface', order: 5, emitsSurface: true, frozen: true,
    acceptedSpanLevels: ['sentence', 'clause', 'token', 'grapheme'],
    atlas: { layer: 'language-output', placement: 'peripheral-island', islandId: 'manual-language-scaffold', ordinal: 5, radialScale: 1.28 },
  },
  {
    id: LANGUAGE_OUTPUT_FAMILY_IDS.eos,
    label: 'Cümleyi bitirme / EOS',
    description: 'Zorunlu roller ve terminal noktalama tamamlandıktan sonra üretimi durduran kontrol ailesi.',
    kind: 'terminal', order: 6, emitsSurface: false, frozen: true,
    acceptedSpanLevels: ['sentence', 'clause', 'token'],
    atlas: { layer: 'language-output', placement: 'peripheral-island', islandId: 'manual-language-scaffold', ordinal: 6, radialScale: 1.3 },
  },
])

function seed(
  id: string,
  familyId: LanguageOutputFamilyId,
  kind: LanguageOutputSeedNeuron['kind'],
  surface: string | null,
  features: LanguageOutputSeedNeuron['features'],
  allowedRoles: LanguageOutputSeedNeuron['allowedRoles'] = [],
  priorWeight = 1,
): LanguageOutputSeedNeuron {
  return {
    id: `@language-output/seed/${id}`,
    familyId,
    kind,
    surface,
    normalizedSurface: surface ? normalizeSurface(surface) : '',
    features: { ...features },
    allowedRoles: [...allowedRoles],
    priorWeight,
    frozen: true,
  }
}

const LANGUAGE_OUTPUT_SEED_NEURONS: readonly LanguageOutputSeedNeuron[] = deepFreeze([
  seed('order/sov', LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, 'control', null, { partOfSpeech: 'control' }, ['actor', 'patient', 'instrument', 'action']),
  seed('order/operator-sov', LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, 'control', null, { partOfSpeech: 'control' }, ['operator', 'actor', 'patient', 'action'], 0.8),
  seed('tense/past', LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, 'morpheme', '-DI', { partOfSpeech: 'verb', tense: 'past' }, ['action']),
  seed('tense/evidential-past', LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, 'morpheme', '-mIş', { partOfSpeech: 'verb', tense: 'evidential-past' }, ['action'], 0.75),
  seed('tense/present-continuous', LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, 'morpheme', '-(I)yor', { partOfSpeech: 'verb', tense: 'present-continuous' }, ['action']),
  seed('tense/future', LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, 'morpheme', '-(y)AcAk', { partOfSpeech: 'verb', tense: 'future' }, ['action'], 0.85),
  seed('case/accusative', LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, 'morpheme', '-(y)I', { partOfSpeech: 'noun', case: 'accusative' }, ['patient']),
  seed('case/dative', LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, 'morpheme', '-(y)A', { partOfSpeech: 'noun', case: 'dative' }, ['patient', 'goal']),
  seed('case/locative', LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, 'morpheme', '-DA', { partOfSpeech: 'noun', case: 'locative' }, ['goal'], 0.8),
  seed('case/ablative', LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, 'morpheme', '-DAn', { partOfSpeech: 'noun', case: 'ablative' }, ['goal'], 0.8),
  seed('case/instrumental', LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, 'morpheme', '-(y)lA', { partOfSpeech: 'noun', case: 'instrumental' }, ['instrument']),
  seed('connector/and', LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, 'lexeme', 've', { partOfSpeech: 'connector' }),
  seed('connector/but', LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, 'lexeme', 'ama', { partOfSpeech: 'connector' }),
  seed('connector/because', LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, 'lexeme', 'çünkü', { partOfSpeech: 'connector' }),
  seed('negation/copular', LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, 'lexeme', 'değil', { partOfSpeech: 'particle', polarity: 'negative' }, ['operator']),
  seed('negation/verbal', LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, 'morpheme', '-mA', { partOfSpeech: 'verb', polarity: 'negative' }, ['action']),
  seed('punctuation/period', LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, 'punctuation', '.', { partOfSpeech: 'punctuation' }),
  seed('punctuation/question', LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, 'punctuation', '?', { partOfSpeech: 'punctuation' }),
  seed('punctuation/exclamation', LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, 'punctuation', '!', { partOfSpeech: 'punctuation' }, [], 0.7),
  seed('punctuation/comma', LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, 'punctuation', ',', { partOfSpeech: 'punctuation' }, [], 0.65),
  seed('eos', LANGUAGE_OUTPUT_FAMILY_IDS.eos, 'terminal', null, { partOfSpeech: 'control' }),
])

export const LANGUAGE_OUTPUT_MANUAL_TRANSITIONS: readonly LanguageOutputManualTransition[] = deepFreeze([
  { fromKey: LANGUAGE_ROLE_GATEWAY_ID, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, priorEvidence: 1, explanation: 'Canlı düşünce rolleri önce sözcük sırası planına girer.' },
  { fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, priorEvidence: 0.7, explanation: 'Rol sırası, gerekli ad öbeği hâl işaretlerini açabilir.' },
  { fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, priorEvidence: 0.35, explanation: 'Operatör ve bağlaçlar yüzey planına kontrollü biçimde eklenebilir.' },
  { fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, priorEvidence: 0.9, explanation: 'Türkçe temel planda yüklem ve zaman işaretleme sona yakın gelir.' },
  { fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, priorEvidence: 0.9, explanation: 'Ad öbeği işaretlerinden sonra yüklem tamamlanır.' },
  { fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, priorEvidence: 0.75, explanation: 'Olumsuzluk ve operatörler yüklem biçimine bağlanır.' },
  { fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, priorEvidence: 1, explanation: 'Tamamlanan yüklem terminal noktalama adayını açar.' },
  { fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, toKey: LANGUAGE_OUTPUT_FAMILY_IDS.eos, priorEvidence: 1, explanation: 'Terminal noktalama sonrasında EOS seçilebilir.' },
])

const ROLE_GATEWAY: LanguageRoleGatewayV1 = deepFreeze({
  id: LANGUAGE_ROLE_GATEWAY_ID,
  virtual: true,
  acceptedRoles: [...ALL_ROLES],
  explanation: 'Geçici düşünce rollerini kalıcı nöron etiketi yapmadan yüzey adaylarına bağlayan sanal kapı.',
})

function fixedStructureDigest(input: Pick<
  LanguageOutputScaffoldV1,
  'roleGateway' | 'families' | 'seedNeurons' | 'manualTransitions'
>): string {
  return outputDigest(
    'language-output-fixed-structure',
    input.roleGateway,
    input.families,
    input.seedNeurons,
    input.manualTransitions,
  )
}

const FIXED_STRUCTURE_FINGERPRINT = fixedStructureDigest({
  roleGateway: ROLE_GATEWAY,
  families: [...LANGUAGE_OUTPUT_FAMILY_DEFINITIONS],
  seedNeurons: [...LANGUAGE_OUTPUT_SEED_NEURONS],
  manualTransitions: [...LANGUAGE_OUTPUT_MANUAL_TRANSITIONS],
})

function cloneFamily(family: LanguageOutputFamilyDefinition): LanguageOutputFamilyDefinition {
  return { ...family, acceptedSpanLevels: [...family.acceptedSpanLevels], atlas: { ...family.atlas } }
}

function cloneSeedNeuron(neuron: LanguageOutputSeedNeuron): LanguageOutputSeedNeuron {
  return { ...neuron, features: { ...neuron.features }, allowedRoles: [...neuron.allowedRoles] }
}

function scaffoldPayload(scaffold: Omit<LanguageOutputScaffoldV1, 'fingerprint'>): unknown {
  return scaffold
}

export function computeLanguageOutputScaffoldFingerprint(
  scaffold: Omit<LanguageOutputScaffoldV1, 'fingerprint'>,
): string {
  return outputDigest('language-output-scaffold', scaffoldPayload(scaffold))
}

export function createLanguageOutputScaffold(): LanguageOutputScaffoldV1 {
  const base: Omit<LanguageOutputScaffoldV1, 'fingerprint'> = {
    format: 'language-output-scaffold',
    version: 1,
    id: 'language-output-scaffold-v1',
    seedRevision: 1,
    roleGateway: { ...ROLE_GATEWAY, acceptedRoles: [...ROLE_GATEWAY.acceptedRoles] },
    families: LANGUAGE_OUTPUT_FAMILY_DEFINITIONS.map(cloneFamily),
    seedNeurons: LANGUAGE_OUTPUT_SEED_NEURONS.map(cloneSeedNeuron),
    manualTransitions: LANGUAGE_OUTPUT_MANUAL_TRANSITIONS.map((edge) => ({ ...edge })),
  }
  const scaffold = { ...base, fingerprint: computeLanguageOutputScaffoldFingerprint(base) }
  validateLanguageOutputScaffold(scaffold)
  return scaffold
}

export function validateLanguageOutputScaffold(scaffold: LanguageOutputScaffoldV1): void {
  if (
    scaffold?.format !== 'language-output-scaffold' || scaffold.version !== 1 ||
    scaffold.id !== 'language-output-scaffold-v1' || scaffold.seedRevision !== 1 ||
    scaffold.roleGateway?.id !== LANGUAGE_ROLE_GATEWAY_ID ||
    scaffold.roleGateway.virtual !== true
  ) throw new TypeError('Dil çıktı scaffold kaydı eksik veya uyumsuz.')
  const expectedIds = Object.values(LANGUAGE_OUTPUT_FAMILY_IDS).sort()
  const actualIds = scaffold.families.map((family) => family.id).sort()
  if (actualIds.length !== 6 || new Set(actualIds).size !== 6 || actualIds.join('\0') !== expectedIds.join('\0')) {
    throw new TypeError('Dil çıktı scaffold tam olarak altı sabit aile taşımalıdır.')
  }
  const familyIds = new Set(actualIds)
  for (const family of scaffold.families) {
    if (
      family.frozen !== true || !Number.isInteger(family.order) || family.order < 1 || family.order > 6 ||
      !family.label || !family.description || !family.acceptedSpanLevels.length ||
      family.atlas?.layer !== 'language-output' || family.atlas.placement !== 'peripheral-island' ||
      !Number.isFinite(family.atlas.radialScale) || family.atlas.radialScale <= 1
    ) throw new TypeError('Dil çıktı scaffold aile tanımı bozuk.')
  }
  if (new Set(scaffold.families.map((family) => family.order)).size !== 6) {
    throw new TypeError('Dil çıktı scaffold aile sırası benzersiz olmalıdır.')
  }
  const neuronIds = new Set<string>()
  for (const neuron of scaffold.seedNeurons) {
    if (
      !neuron.id || neuronIds.has(neuron.id) || !familyIds.has(neuron.familyId) || neuron.frozen !== true ||
      !Number.isFinite(neuron.priorWeight) || neuron.priorWeight <= 0 || neuron.priorWeight > 1
    ) throw new TypeError('Dil çıktı seed nöronu bozuk veya yineleniyor.')
    neuronIds.add(neuron.id)
  }
  const validTransitionKeys = new Set<string>([LANGUAGE_ROLE_GATEWAY_ID, ...familyIds, ...neuronIds])
  const edgeIds = new Set<string>()
  for (const edge of scaffold.manualTransitions) {
    const key = `${edge.fromKey}\0${edge.toKey}`
    if (
      edgeIds.has(key) || !validTransitionKeys.has(edge.fromKey) || !validTransitionKeys.has(edge.toKey) ||
      !Number.isFinite(edge.priorEvidence) || edge.priorEvidence <= 0 || !edge.explanation
    ) throw new TypeError('Dil çıktı scaffold geçişi bozuk veya yineleniyor.')
    edgeIds.add(key)
  }
  if (fixedStructureDigest(scaffold) !== FIXED_STRUCTURE_FINGERPRINT) {
    throw new TypeError('Dil çıktı scaffoldı altı sabit aile, seed veya elle kurulan geçiş sözleşmesini değiştiremez.')
  }
  const { fingerprint: _fingerprint, ...base } = scaffold
  if (computeLanguageOutputScaffoldFingerprint(base) !== scaffold.fingerprint) {
    throw new TypeError('Dil çıktı scaffold fingerprint doğrulaması başarısız.')
  }
}

const CONNECTIVE_OR_NEGATION = new Set([
  've', 'ama', 'fakat', 'ancak', 'çünkü', 'veya', 'yahut', 'sonra', 'önce',
  'değil', 'yok', 'hayır', 'mı', 'mi', 'mu', 'mü', 'ma', 'me',
])
const EXACT_TENSE_MORPHEME = /^(?:dı|di|du|dü|tı|ti|tu|tü|mış|miş|muş|müş|yor|acak|ecek|ar|er|ır|ir|ur|ür)$/u
const TOKEN_TENSE_ENDING = /(?:dı|di|du|dü|tı|ti|tu|tü|mış|miş|muş|müş|ıyor|iyor|uyor|üyor|acak|ecek)(?:m|n|k|nız|niz|nuz|nüz|lar|ler)?$/u
const EXACT_CASE_MORPHEME = /^(?:ı|i|u|ü|a|e|yı|yi|yu|yü|ya|ye|da|de|ta|te|dan|den|tan|ten|la|le|yla|yle)$/u
const TOKEN_CASE_ENDING = /(?:yı|yi|yu|yü|ya|ye|da|de|ta|te|dan|den|tan|ten|yla|yle)$/u

function familyAccepts(familyId: LanguageOutputFamilyId, level: SurfacePieceLevel): boolean {
  return LANGUAGE_OUTPUT_FAMILY_DEFINITIONS
    .find((family) => family.id === familyId)?.acceptedSpanLevels.includes(level) ?? false
}

/**
 * Conservative, deterministic bootstrap classifier. It supplies seed evidence,
 * not a permanent linguistic truth; learned evidence remains in a separate head.
 */
export function classifyLanguageOutputSeed(
  input: LanguageOutputSeedClassifierInput,
): LanguageOutputSeedClassification[] {
  const surface = input.normalizedSurface === undefined
    ? normalizeSurface(input.surface)
    : normalizeSurface(input.normalizedSurface)
  if (input.eos) {
    return [{ familyId: LANGUAGE_OUTPUT_FAMILY_IDS.eos, confidence: 1, membershipEligible: false, reason: 'Açık EOS eğitim hedefi.' }]
  }
  if (!surface) return []
  const output: LanguageOutputSeedClassification[] = []
  const add = (
    familyId: LanguageOutputFamilyId,
    confidence: number,
    membershipEligible: boolean,
    reason: string,
  ) => {
    if (familyAccepts(familyId, input.granularity)) output.push({ familyId, confidence, membershipEligible, reason })
  }
  if (/^[.!?;,:…]$/u.test(surface)) {
    add(LANGUAGE_OUTPUT_FAMILY_IDS.punctuation, 1, true, 'Tek noktalama yüzeyi sabit seed ailesiyle eşleşti.')
  } else if (CONNECTIVE_OR_NEGATION.has(surface)) {
    add(LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation, 0.98, true, 'Denetimli bağlaç, soru veya olumsuzluk seed yüzeyi eşleşti.')
  } else if (
    (input.granularity !== 'token' && EXACT_TENSE_MORPHEME.test(surface)) ||
    (input.granularity === 'token' && surface.length >= 4 && TOKEN_TENSE_ENDING.test(surface))
  ) {
    add(LANGUAGE_OUTPUT_FAMILY_IDS.verbTense, input.granularity === 'token' ? 0.78 : 0.96, true, 'Türkçe zaman biçimbirimi için temkinli yüzey eşleşmesi.')
  } else if (
    (input.granularity !== 'token' && EXACT_CASE_MORPHEME.test(surface)) ||
    (input.granularity === 'token' && surface.length >= 4 && TOKEN_CASE_ENDING.test(surface))
  ) {
    add(LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking, input.granularity === 'token' ? 0.7 : 0.94, true, 'Türkçe hâl biçimbirimi için temkinli yüzey eşleşmesi.')
  }
  if (!output.length && /[\p{L}\p{N}]/u.test(surface)) {
    add(LANGUAGE_OUTPUT_FAMILY_IDS.wordOrder, 0.6, false, 'Sözlüksel yüzey yalnız sıra deneyimine katılır; aile üyeliği etiketi almaz.')
  }
  return output.sort((left, right) =>
    right.confidence - left.confidence || compareText(left.familyId, right.familyId),
  )
}
