import type {
  LanguageIntentSlot,
  LanguageIntentSlotCandidate,
  LanguageIntentV1,
  LanguageMorphFeatures,
  LanguageOutputFamilyId,
  LanguageOutputGenerationResult,
  LanguageOutputGenerationStep,
  LanguageOutputHeadV1,
  LanguageOutputRole,
  LanguageOutputScaffoldV1,
  LanguageOutputToken,
  SentenceAssemblyState,
} from './types.ts'
import {
  normalizeTurkishSurface,
  realizeTurkishNoun,
  realizeTurkishVerb,
  turkishQuestionParticle,
} from './morphology.ts'
import { createLanguageOutputScaffold } from './scaffold.ts'

const WORD_ORDER = '@language-output/family/word-order' as const
const VERB_TENSE = '@language-output/family/verb-tense' as const
const CASE_MARKING = '@language-output/family/case-marking' as const
const CONNECTIVE_NEGATION = '@language-output/family/connective-negation' as const
const PUNCTUATION = '@language-output/family/punctuation' as const
const EOS = '@language/EOS/family' as const

const GENERATION_CHAIN: readonly LanguageOutputFamilyId[] = [
  WORD_ORDER,
  VERB_TENSE,
  CASE_MARKING,
  CONNECTIVE_NEGATION,
  PUNCTUATION,
  EOS,
]

const QUESTION_PARTICLES = new Set(['mı', 'mi', 'mu', 'mü'])
const NEGATION_OPERATORS = new Set(['değil', 'yok', 'hayır'])
const CONNECTIVES = new Set(['ama', 'ancak', 'fakat', 've', 'çünkü', 'oysa'])
const MAX_OUTPUT_TOKENS = 128

interface SelectedSlot {
  role: LanguageOutputRole
  slotIndex: number
  candidate: LanguageIntentSlotCandidate
}

interface NominalPlan {
  tokenId: string
  sourceMemberKey: string
  morphemes: string[]
  features: LanguageMorphFeatures
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function normalizedKey(value: string): string {
  return normalizeTurkishSurface(value).toLocaleLowerCase('tr-TR')
}

function familyForRole(role: LanguageOutputRole): LanguageOutputFamilyId {
  if (role === 'action') return VERB_TENSE
  if (role === 'patient' || role === 'instrument' || role === 'goal') return CASE_MARKING
  if (role === 'operator') return CONNECTIVE_NEGATION
  return WORD_ORDER
}

function learnedMembershipScore(
  head: LanguageOutputHeadV1 | undefined,
  candidate: LanguageIntentSlotCandidate,
  familyId: LanguageOutputFamilyId,
): number {
  if (!head) return 0
  return head.membershipEvidence
    .filter((item) => item.memberKey === candidate.memberKey && item.familyId === familyId)
    .reduce((sum, item) => sum + finite(item.score), 0)
}

function chooseCandidate(
  slot: LanguageIntentSlot,
  head: LanguageOutputHeadV1 | undefined,
): LanguageIntentSlotCandidate | null {
  const familyId = familyForRole(slot.role)
  return [...slot.candidates]
    .filter((candidate) => Boolean(normalizeTurkishSurface(candidate?.surface ?? '')))
    .sort((left, right) => (
      finite(right.score) + learnedMembershipScore(head, right, familyId) -
      finite(left.score) - learnedMembershipScore(head, left, familyId)
    ) || String(left.memberKey).localeCompare(String(right.memberKey)) || left.surface.localeCompare(right.surface, 'tr-TR'))[0] ?? null
}

function isProperNoun(surface: string): boolean {
  const words = normalizeTurkishSurface(surface).split(/\s+/u).filter(Boolean)
  if (!words.length) return false
  // A single capitalized token is useful evidence for names such as Ada or
  // Ankara. For phrases, require every word to be title-cased so sentence
  // capitalization alone cannot turn "Kırmızı kutu" into a proper noun.
  return words.every((word) => {
    const first = [...word][0]
    return Boolean(first && /\p{Lu}/u.test(first))
  })
}

function roleOrder(item: SelectedSlot): number {
  if (item.role === 'operator' && CONNECTIVES.has(normalizedKey(item.candidate.surface))) return 0
  const order: Record<LanguageOutputRole, number> = {
    actor: 1,
    patient: 2,
    goal: 3,
    instrument: 4,
    operator: 5,
    action: 6,
  }
  return order[item.role]
}

function grammaticalCase(role: LanguageOutputRole): NonNullable<LanguageMorphFeatures['case']> {
  if (role === 'patient') return 'accusative'
  if (role === 'goal') return 'dative'
  if (role === 'instrument') return 'instrumental'
  return 'none'
}

function capitalizeTurkish(value: string): string {
  const characters = [...value]
  if (!characters.length) return value
  return `${characters[0].toLocaleUpperCase('tr-TR')}${characters.slice(1).join('')}`
}

function cloneToken(token: LanguageOutputToken): LanguageOutputToken {
  return { ...token, features: { ...token.features } }
}

function cloneState(state: SentenceAssemblyState): SentenceAssemblyState {
  return {
    ...state,
    emittedRoles: [...state.emittedRoles],
    tokens: state.tokens.map(cloneToken),
    nextFamilyIds: [...state.nextFamilyIds],
  }
}

function assembleText(tokens: readonly LanguageOutputToken[]): string {
  let text = ''
  for (const token of tokens) {
    if (!token.surface || token.kind === 'control' || token.kind === 'terminal') continue
    if (!text || token.attached) text += token.surface
    else text += ` ${token.surface}`
  }
  return text.trim()
}

function insertAfter(tokens: readonly LanguageOutputToken[], tokenId: string, additions: readonly LanguageOutputToken[]) {
  const result: LanguageOutputToken[] = []
  for (const token of tokens) {
    result.push(token)
    if (token.id === tokenId) result.push(...additions)
  }
  return result
}

function stateFor(
  intent: LanguageIntentV1,
  emittedRoles: LanguageOutputRole[],
  tokens: LanguageOutputToken[],
  stageIndex: number,
  complete = false,
): SentenceAssemblyState {
  return {
    intentFingerprint: intent.fingerprint,
    emittedRoles: [...emittedRoles],
    tokens: tokens.map(cloneToken),
    nextFamilyIds: complete ? [] : [...GENERATION_CHAIN.slice(stageIndex + 1)],
    punctuationEmitted: tokens.some((token) => token.kind === 'punctuation'),
    complete,
  }
}

function unmatchedResult(
  intent: LanguageIntentV1,
  scaffold: LanguageOutputScaffoldV1 | undefined,
  head: LanguageOutputHeadV1 | undefined,
  tokens: LanguageOutputToken[] = [],
  steps: LanguageOutputGenerationStep[] = [],
  stoppedBy: LanguageOutputGenerationResult['stoppedBy'] = 'unmatched',
): LanguageOutputGenerationResult {
  return {
    text: assembleText(tokens),
    tokens: tokens.map(cloneToken),
    steps,
    stoppedBy,
    intentFingerprint: intent.fingerprint,
    scaffoldFingerprint: scaffold?.fingerprint ?? 'manual-language-output-scaffold-v1',
    headFingerprint: head?.fingerprint ?? 'none',
  }
}

/**
 * Runs the frozen control chain without changing the intent, scaffold, learned
 * head or any upstream thought/family record.
 */
export function generateTurkishSentence(
  intent: LanguageIntentV1,
  scaffold?: LanguageOutputScaffoldV1,
  head?: LanguageOutputHeadV1,
): LanguageOutputGenerationResult {
  if (!intent || intent.format !== 'language-output-intent' || intent.version !== 1) {
    throw new TypeError('LanguageIntentV1 gerekli.')
  }
  const effectiveScaffold = scaffold ?? createLanguageOutputScaffold()
  if (head && head.scaffoldFingerprint !== effectiveScaffold.fingerprint) {
    throw new TypeError('Dil çıkış başlığı farklı bir scaffold sürümüne ait.')
  }

  const selected = intent.slots.flatMap((slot, slotIndex) => {
    const candidate = chooseCandidate(slot, head)
    return candidate ? [{ role: slot.role, slotIndex, candidate }] : []
  })
  const action = selected.find((item) => item.role === 'action')
  if (!action) return unmatchedResult(intent, effectiveScaffold, head)

  const ordered = [...selected]
    .filter((item) => {
      if (item.role !== 'operator') return true
      const key = normalizedKey(item.candidate.surface)
      return !QUESTION_PARTICLES.has(key) && !NEGATION_OPERATORS.has(key)
    })
    .sort((left, right) => roleOrder(left) - roleOrder(right) || left.slotIndex - right.slotIndex ||
      left.candidate.memberKey.localeCompare(right.candidate.memberKey))

  let tokenSequence = 0
  const token = (
    kind: LanguageOutputToken['kind'],
    familyId: LanguageOutputFamilyId,
    surface: string,
    attached: boolean,
    features: LanguageMorphFeatures,
    sourceMemberKey: string | null,
  ): LanguageOutputToken => ({
    id: `${intent.fingerprint.slice(0, 16)}:${String(tokenSequence++).padStart(3, '0')}`,
    kind,
    familyId,
    surface,
    attached,
    features: { ...features },
    sourceMemberKey,
  })

  const nominalPlans: NominalPlan[] = []
  let actionTokenId = ''
  const verb = realizeTurkishVerb(action.candidate.surface, {
    lemma: action.candidate.surface,
    tense: intent.tense,
    polarity: intent.polarity,
    person: 3,
    number: 'singular',
  })

  let tokens = ordered.map((item) => {
    if (item.role === 'action') {
      const emitted = token('lexeme', VERB_TENSE, verb.stem, false, verb.features, item.candidate.memberKey)
      actionTokenId = emitted.id
      return emitted
    }
    if (item.role === 'operator') {
      const key = normalizedKey(item.candidate.surface)
      return token(
        CONNECTIVES.has(key) ? 'lexeme' : 'lexeme',
        CONNECTIVE_NEGATION,
        item.candidate.surface,
        false,
        { lemma: key, partOfSpeech: CONNECTIVES.has(key) ? 'connector' : 'particle' },
        item.candidate.memberKey,
      )
    }
    const features: LanguageMorphFeatures = {
      lemma: item.candidate.surface,
      partOfSpeech: 'noun',
      case: grammaticalCase(item.role),
      number: 'singular',
      properNoun: isProperNoun(item.candidate.surface),
    }
    const nominal = realizeTurkishNoun(item.candidate.surface, features)
    const emitted = token('lexeme', familyForRole(item.role), nominal.stem, false, nominal.features, item.candidate.memberKey)
    nominalPlans.push({ tokenId: emitted.id, sourceMemberKey: item.candidate.memberKey, morphemes: nominal.morphemes, features: nominal.features })
    return emitted
  })

  const firstSurface = tokens.find((item) => item.surface && !item.attached)
  if (firstSurface) firstSurface.surface = capitalizeTurkish(firstSurface.surface)
  const emittedRoles = [...new Set(ordered.map((item) => item.role))]
  const steps: LanguageOutputGenerationStep[] = []
  const pushStep = (stageIndex: number, emittedToken: LanguageOutputToken | null, explanation: string, complete = false) => {
    const state = stateFor(intent, emittedRoles, tokens, stageIndex, complete)
    steps.push({
      index: steps.length,
      familyId: GENERATION_CHAIN[stageIndex],
      emittedToken: emittedToken ? cloneToken(emittedToken) : null,
      state: cloneState(state),
      explanation,
    })
  }

  pushStep(0, tokens[0] ?? null, 'Rol adayları sabit Türkçe sırasına yerleştirildi.')
  if (tokens.length > MAX_OUTPUT_TOKENS) {
    tokens = tokens.slice(0, MAX_OUTPUT_TOKENS)
    return unmatchedResult(intent, effectiveScaffold, head, tokens, steps, 'maximum-steps')
  }

  const verbSuffixes = [
    verb.tenseMorpheme
      ? token('morpheme', VERB_TENSE, verb.tenseMorpheme, true, verb.features, action.candidate.memberKey)
      : null,
    verb.personMorpheme
      ? token('morpheme', VERB_TENSE, verb.personMorpheme, true, verb.features, action.candidate.memberKey)
      : null,
  ].filter((item): item is LanguageOutputToken => Boolean(item))
  tokens = insertAfter(tokens, actionTokenId, verbSuffixes)
  pushStep(1, verbSuffixes[0] ?? null, 'Fiil köküne temel zaman ve kişi ekleri bağlandı.')

  const caseTokens: LanguageOutputToken[] = []
  for (const plan of nominalPlans) {
    const suffixes = plan.morphemes.map((surface) => token(
      'morpheme', CASE_MARKING, surface, true, plan.features, plan.sourceMemberKey,
    ))
    caseTokens.push(...suffixes)
    tokens = insertAfter(tokens, plan.tokenId, suffixes)
  }
  pushStep(2, caseTokens[0] ?? null, 'Nesne, hedef ve araç rolleri ünlü uyumlu durum ekleriyle işaretlendi.')

  const connectiveTokens: LanguageOutputToken[] = []
  if (verb.negativeMorpheme) {
    const negative = token(
      'morpheme', CONNECTIVE_NEGATION, verb.negativeMorpheme, true,
      { ...verb.features, polarity: 'negative' }, action.candidate.memberKey,
    )
    connectiveTokens.push(negative)
    tokens = insertAfter(tokens, actionTokenId, [negative])
  }
  if (intent.sentenceKind === 'question') {
    const finiteVerb = verb.surface
    const question = token(
      'morpheme', CONNECTIVE_NEGATION, turkishQuestionParticle(finiteVerb), false,
      { partOfSpeech: 'particle' }, null,
    )
    connectiveTokens.push(question)
    tokens.push(question)
  }
  const visibleConnector = tokens.find((item) => item.familyId === CONNECTIVE_NEGATION && item.kind === 'lexeme')
  pushStep(3, connectiveTokens[0] ?? visibleConnector ?? null, 'Bağlaç, soru parçacığı ve olumsuzluk kanıtı yüzeye uygulandı.')

  const punctuationSurface = intent.sentenceKind === 'question' ? '?' : intent.sentenceKind === 'exclamation' ? '!' : '.'
  const punctuation = token(
    'punctuation', PUNCTUATION, punctuationSurface, true,
    { partOfSpeech: 'punctuation' }, null,
  )
  tokens.push(punctuation)
  pushStep(4, punctuation, 'Cümle türüne uygun tek kapanış işareti eklendi.')

  const eos = token('terminal', EOS, '', true, { partOfSpeech: 'control' }, null)
  tokens.push(eos)
  pushStep(5, eos, 'EOS terminali üretimi tek ve denetlenebilir biçimde kapattı.', true)

  if (tokens.length > MAX_OUTPUT_TOKENS) {
    return unmatchedResult(intent, effectiveScaffold, head, tokens.slice(0, MAX_OUTPUT_TOKENS), steps, 'maximum-steps')
  }
  return unmatchedResult(intent, effectiveScaffold, head, tokens, steps, 'eos')
}
