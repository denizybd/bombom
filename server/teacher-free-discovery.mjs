import { createHash } from 'node:crypto'
import { OLLAMA_DIVERSE_MAX_NUM_PREDICT } from './ollama.mjs'

const sha256 = (value) => createHash('sha256').update(String(value)).digest('hex')

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]))
  }
  return value
}

const stableHash = (value) => sha256(JSON.stringify(stableValue(value)))

export const TEACHER_DISCOVERY_PROTOCOL = Object.freeze({
  version: 1,
  id: 'turkish-free-surface-discovery-v1',
  outputSchemaPolicyId: 'bounded-unique-sentence-array-v1',
  validationPolicyId: 'server-owned-teacher-text-gate-v1',
  schedulingPolicyId: 'deterministic-alternating-controlled-first-v1',
  authority: 'training-only-proposal-no-evaluation',
})

export function createTeacherDiscoverySchema(count) {
  if (!Number.isInteger(count) || count < 1 || count > 2) {
    throw new TypeError('Keşif cümlesi sayısı 1–2 arasında olmalı.')
  }
  return {
    type: 'object',
    additionalProperties: false,
    required: ['sentences'],
    properties: {
      sentences: {
        type: 'array',
        minItems: count,
        maxItems: count,
        uniqueItems: true,
        items: { type: 'string', minLength: 12, maxLength: 280 },
      },
    },
  }
}

export function buildTeacherDiscoveryMessages({ plan, count, attempt }) {
  return [
    {
      role: 'system',
      content: [
        `JSON şemasındaki sentences alanında tam ${count} farklı Türkçe cümle döndür.`,
        'Cümleleri kendin yaz; açıklama, etiket, numara, alıntı, bağlantı, kişisel veri veya kod ekleme.',
        'Her öğe tek, doğal ve bağımsız bir cümle olsun. Aynı özne, eylem veya kalıbı tekrar etme.',
        'Bu çıktı yalnız eğitim adayıdır; sunucu dil, biçim, yazım, uzunluk, tekrar ve güvenlik denetimlerini bağımsız yapacaktır.',
      ].join(' '),
    },
    {
      role: 'user',
      content: [
        `Beceri hedefi: ${plan.unit.title}. ${plan.unit.instruction}`,
        `Biçim: ${plan.form}. Alan: ${plan.axes.domain}.`,
        `Zaman: ${plan.axes.tense}. Uzunluk: ${plan.axes.length}. Sıra: ${plan.axes.wordOrder}.`,
        `Parti=${plan.batchIndex + 1}; deneme=${attempt + 1}.`,
        'Önceden hazırlanmış bir cümleyi seçme; bu koşulları karşılayan yeni yüzeyler keşfet.',
      ].join('\n'),
    },
  ]
}

export function teacherDiscoveryCompletionTokenLimit(count) {
  // Keep the discovery protocol inside the same server-owned bound enforced
  // by OllamaClient.generateDiverseStructured. A two-sentence batch used to
  // request 320 tokens and therefore failed before any model call could be
  // accepted.
  return Math.min(OLLAMA_DIVERSE_MAX_NUM_PREDICT, 96 + (Math.max(1, count) * 112))
}

export const TEACHER_DISCOVERY_PROTOCOL_FINGERPRINT = stableHash({
  protocol: TEACHER_DISCOVERY_PROTOCOL,
  deterministicLogic: [
    createTeacherDiscoverySchema,
    buildTeacherDiscoveryMessages,
    teacherDiscoveryCompletionTokenLimit,
  ].map(String),
})

export function createTeacherDiscoveryAttempt({ plan, count, attempt, seed }) {
  const schema = createTeacherDiscoverySchema(count)
  const messages = buildTeacherDiscoveryMessages({ plan, count, attempt })
  const fingerprint = stableHash({
    protocol: TEACHER_DISCOVERY_PROTOCOL_FINGERPRINT,
    plan: {
      batchIndex: plan.batchIndex,
      batchSeed: plan.batchSeed,
      unitId: plan.unit.id,
      form: plan.form,
      axes: plan.axes,
    },
    count,
    attempt,
    seed,
    schema,
    messages,
  })
  return {
    id: `tfd-${fingerprint.slice(0, 24)}`,
    fingerprint,
    schema,
    messages,
    maxCompletionTokens: teacherDiscoveryCompletionTokenLimit(count),
  }
}
