import type { EventGraphV1 } from '../thought-runtime/types.ts'
import type {
  LanguageIntentSlot,
  LanguageIntentSlotCandidate,
  LanguageIntentV1,
  LanguageMorphFeatures,
  LanguageOutputRole,
} from './types.ts'
import { normalizeTurkishSurface } from './morphology.ts'

type LanguageTense = NonNullable<LanguageMorphFeatures['tense']>

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonicalize(item)]),
    )
  }
  return typeof value === 'number' && !Number.isFinite(value) ? String(value) : value
}

/** Small deterministic digest; identity only, never a cryptographic trust proof. */
export function languageIntentDigest(label: string, value: unknown): string {
  const input = JSON.stringify([label, canonicalize(value)])
  const states = [0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35]
  const primes = [0x01000193, 0x27d4eb2d, 0x165667b1, 0x9e3779b1]
  for (let index = 0; index < input.length; index += 1) {
    const code = input.charCodeAt(index)
    for (let state = 0; state < states.length; state += 1) {
      states[state] = Math.imul(
        (states[state] ^ code ^ Math.imul(index + 1, state + 17)) >>> 0,
        primes[state],
      ) >>> 0
      states[state] ^= states[state] >>> 13
    }
  }
  return states.map((state) => (state >>> 0).toString(16).padStart(8, '0')).join('')
}

function cleanSurface(value: string | null | undefined): string {
  return normalizeTurkishSurface(value ?? '').replace(/[.!?,;:]+$/u, '')
}

function normalizedOperator(value: string): string {
  return cleanSurface(value).toLocaleLowerCase('tr-TR')
}

interface ActionAnalysis {
  lemma: string
  tense: LanguageTense
  negative: boolean
}

function analyzeAction(value: string): ActionAnalysis {
  const action = cleanSurface(value).toLocaleLowerCase('tr-TR')
  const patterns: Array<{ expression: RegExp; tense: LanguageTense; negative: boolean }> = [
    { expression: /^(.*?)(?:ma|me)(?:dı|di|du|dü|tı|ti|tu|tü)$/u, tense: 'past', negative: true },
    { expression: /^(.*?)m(?:ıyor|iyor|uyor|üyor)$/u, tense: 'present-continuous', negative: true },
    { expression: /^(.*?)(?:ma|me)y(?:acak|ecek)$/u, tense: 'future', negative: true },
    { expression: /^(.*?)(?:ma|me)m(?:ış|miş|muş|müş)$/u, tense: 'evidential-past', negative: true },
    { expression: /^(.*?)(?:ma|me)z$/u, tense: 'aorist', negative: true },
    { expression: /^(.*?)(?:dı|di|du|dü|tı|ti|tu|tü)$/u, tense: 'past', negative: false },
    { expression: /^(.*?)(?:mış|miş|muş|müş)$/u, tense: 'evidential-past', negative: false },
    { expression: /^(.*?)(?:ıyor|iyor|uyor|üyor)$/u, tense: 'present-continuous', negative: false },
    { expression: /^(.*?)(?:yacak|yecek|acak|ecek)$/u, tense: 'future', negative: false },
  ]
  for (const pattern of patterns) {
    const match = pattern.expression.exec(action)
    if (match?.[1]) return { lemma: match[1], tense: pattern.tense, negative: pattern.negative }
  }
  return { lemma: action, tense: 'past', negative: false }
}

function tenseFromOperators(operators: readonly string[], fallback: LanguageTense): LanguageTense {
  const text = operators.map(normalizedOperator).join(' ')
  if (/\b(?:yarın|sonra|gelecekte)\b/u.test(text)) return 'future'
  if (/\b(?:şimdi|halen|hâlen|şu an)\b/u.test(text)) return 'present-continuous'
  if (/\b(?:dün|az önce|geçmişte)\b/u.test(text)) return 'past'
  if (/\b(?:genellikle|daima|her zaman|sık sık)\b/u.test(text)) return 'aorist'
  return fallback
}

function sentenceKind(operators: readonly string[]): LanguageIntentV1['sentenceKind'] {
  const normalized = operators.map((value) => normalizeTurkishSurface(value).toLocaleLowerCase('tr-TR'))
  if (normalized.some((value) => /(?:^|\s)(?:mı|mi|mu|mü)(?:\s|$)|\?$/u.test(value))) return 'question'
  if (normalized.some((value) => /!$|^(?:haydi|hemen|sakın)$/u.test(value))) return 'exclamation'
  return 'statement'
}

function slotCandidate(role: LanguageOutputRole, surface: string, ordinal: number, discriminator = ''): LanguageIntentSlotCandidate {
  const key = languageIntentDigest('event-graph-role-v1', { role, surface, ordinal, discriminator })
  return { memberKey: `event-graph/${role}/${key.slice(0, 24)}`, surface, score: 1 }
}

function singleSlot(role: LanguageOutputRole, surface: string | null | undefined, ordinal = 0): LanguageIntentSlot | null {
  const cleaned = cleanSurface(surface)
  return cleaned ? { role, candidates: [slotCandidate(role, cleaned, ordinal)] } : null
}

/**
 * Bridges a frozen EventGraph into a transient generation intent. No member,
 * family, graph array or role hypothesis is ever mutated by this conversion.
 */
export function eventGraphToLanguageIntent(graph: EventGraphV1): LanguageIntentV1 {
  if (!graph || graph.format !== 'event-graph-v1') throw new TypeError('EventGraphV1 gerekli.')
  const rawOperators = Array.isArray(graph.operators) ? [...graph.operators] : []
  const operators = rawOperators.map(cleanSurface).filter(Boolean)
  const action = analyzeAction(graph.action ?? '')
  const slots: LanguageIntentSlot[] = []
  const append = (slot: LanguageIntentSlot | null) => { if (slot) slots.push(slot) }

  append(singleSlot('actor', graph.actor))
  if (action.lemma) append({ role: 'action', candidates: [slotCandidate('action', action.lemma, 0)] })
  append(singleSlot('patient', graph.patient))
  append(singleSlot('instrument', graph.instrument))
  operators.forEach((surface, index) => append({
    role: 'operator',
    candidates: [slotCandidate('operator', surface, index)],
  }))
  ;(Array.isArray(graph.goals) ? graph.goals : []).forEach((goal, index) => {
    const target = cleanSurface(goal?.target)
    if (!target) return
    slots.push({
      role: 'goal',
      candidates: [slotCandidate('goal', target, index, cleanSurface(goal?.relation))],
    })
  })

  const semantic = {
    slots,
    tense: tenseFromOperators(operators, action.tense),
    polarity: graph.negated || action.negative || operators.some((value) => (
      /^(?:değil|yok|hayır)$/u.test(normalizedOperator(value))
    )) ? 'negative' as const : 'positive' as const,
    sentenceKind: sentenceKind(rawOperators),
  }
  const fingerprint = languageIntentDigest('language-output-intent-v1', semantic)
  return {
    format: 'language-output-intent',
    version: 1,
    id: `LI-${fingerprint.slice(0, 24)}`,
    fingerprint,
    ...semantic,
  }
}
