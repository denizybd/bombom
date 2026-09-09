import { createHash } from 'node:crypto'

export const TEACHER_FEED_CATALOG_ID = 'turkish-foundation-event-cards-v2'
export const TEACHER_FEED_RENDERER_ID = 'deterministic-turkish-surface-v2'
export const TEACHER_FEED_MORPHOLOGY_ID = 'curated-3sg-paradigms-v1'

const sha256 = (value) => createHash('sha256').update(String(value)).digest('hex')

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]))
  }
  return value
}

const stableHash = (value) => sha256(JSON.stringify(stableValue(value)))

const ACTORS = Object.freeze([
  { id: 'ayse', surface: 'Ayşe', possessive: "Ayşe'nin arkadaşı" },
  { id: 'mert', surface: 'Mert', possessive: "Mert'in arkadaşı" },
  { id: 'deniz', surface: 'Deniz', possessive: "Deniz'in arkadaşı" },
  { id: 'ece', surface: 'Ece', possessive: "Ece'nin arkadaşı" },
  { id: 'kerem', surface: 'Kerem', possessive: "Kerem'in arkadaşı" },
  { id: 'selin', surface: 'Selin', possessive: "Selin'in arkadaşı" },
  { id: 'bora', surface: 'Bora', possessive: "Bora'nın arkadaşı" },
  { id: 'derya', surface: 'Derya', possessive: "Derya'nın arkadaşı" },
  { id: 'can', surface: 'Can', possessive: "Can'ın arkadaşı" },
  { id: 'elif', surface: 'Elif', possessive: "Elif'in arkadaşı" },
  { id: 'ozan', surface: 'Ozan', possessive: "Ozan'ın arkadaşı" },
  { id: 'ipek', surface: 'İpek', possessive: "İpek'in arkadaşı" },
])

const VERBS = Object.freeze({
  karistir: { positive: ['karıştırdı', 'karıştırıyor', 'karıştıracak', 'karıştırır'], negative: ['karıştırmadı', 'karıştırmıyor', 'karıştırmayacak', 'karıştırmaz'] },
  duzenle: { positive: ['düzenledi', 'düzenliyor', 'düzenleyecek', 'düzenler'], negative: ['düzenlemedi', 'düzenlemiyor', 'düzenlemeyecek', 'düzenlemez'] },
  sik: { positive: ['sıktı', 'sıkıyor', 'sıkacak', 'sıkar'], negative: ['sıkmadı', 'sıkmıyor', 'sıkmayacak', 'sıkmaz'] },
  olc: { positive: ['ölçtü', 'ölçüyor', 'ölçecek', 'ölçer'], negative: ['ölçmedi', 'ölçmüyor', 'ölçmeyecek', 'ölçmez'] },
  isaretle: { positive: ['işaretledi', 'işaretliyor', 'işaretleyecek', 'işaretler'], negative: ['işaretlemedi', 'işaretlemiyor', 'işaretlemeyecek', 'işaretlemez'] },
  incele: { positive: ['inceledi', 'inceliyor', 'inceleyecek', 'inceler'], negative: ['incelemedi', 'incelemiyor', 'incelemeyecek', 'incelemez'] },
  tasi: { positive: ['taşıdı', 'taşıyor', 'taşıyacak', 'taşır'], negative: ['taşımadı', 'taşımıyor', 'taşımayacak', 'taşımaz'] },
  say: { positive: ['saydı', 'sayıyor', 'sayacak', 'sayar'], negative: ['saymadı', 'saymıyor', 'saymayacak', 'saymaz'] },
  hazirla: { positive: ['hazırladı', 'hazırlıyor', 'hazırlayacak', 'hazırlar'], negative: ['hazırlamadı', 'hazırlamıyor', 'hazırlamayacak', 'hazırlamaz'] },
  sula: { positive: ['suladı', 'suluyor', 'sulayacak', 'sular'], negative: ['sulamadı', 'sulamıyor', 'sulamayacak', 'sulamaz'] },
  denetle: { positive: ['denetledi', 'denetliyor', 'denetleyecek', 'denetler'], negative: ['denetlemedi', 'denetlemiyor', 'denetlemeyecek', 'denetlemez'] },
  paketle: { positive: ['paketledi', 'paketliyor', 'paketleyecek', 'paketler'], negative: ['paketlemedi', 'paketlemiyor', 'paketlemeyecek', 'paketlemez'] },
  boya: { positive: ['boyadı', 'boyuyor', 'boyayacak', 'boyar'], negative: ['boyamadı', 'boyamıyor', 'boyamayacak', 'boyamaz'] },
  kes: { positive: ['kesti', 'kesiyor', 'kesecek', 'keser'], negative: ['kesmedi', 'kesmiyor', 'kesmeyecek', 'kesmez'] },
  yuvarla: { positive: ['yuvarladı', 'yuvarlıyor', 'yuvarlayacak', 'yuvarlar'], negative: ['yuvarlamadı', 'yuvarlamıyor', 'yuvarlamayacak', 'yuvarlamaz'] },
  temizle: { positive: ['temizledi', 'temizliyor', 'temizleyecek', 'temizler'], negative: ['temizlemedi', 'temizlemiyor', 'temizlemeyecek', 'temizlemez'] },
  gotur: { positive: ['götürdü', 'götürüyor', 'götürecek', 'götürür'], negative: ['götürmedi', 'götürmüyor', 'götürmeyecek', 'götürmez'] },
  al: { positive: ['aldı', 'alıyor', 'alacak', 'alır'], negative: ['almadı', 'almıyor', 'almayacak', 'almaz'] },
  bekle: { positive: ['bekledi', 'bekliyor', 'bekleyecek', 'bekler'], negative: ['beklemedi', 'beklemiyor', 'beklemeyecek', 'beklemez'] },
})

const escapeRegExp = (value) => String(value).replace(/[.*+?^${}()|[\]\\]/gu, '\\$&')
const finiteVerbAlternatives = Object.values(VERBS)
  .flatMap((paradigm) => [...paradigm.positive, ...paradigm.negative])
  .sort((left, right) => right.length - left.length)
  .map(escapeRegExp)

/** Exact, reviewed finite forms; unlike suffix guessing this cannot call a noun a tense witness. */
export const TEACHER_FEED_FINITE_VERB_SURFACE = new RegExp(
  `(?:^|[^\\p{L}\\p{M}])(?:${finiteVerbAlternatives.join('|')})(?=$|[^\\p{L}\\p{M}])`,
  'iu',
)

/**
 * Every noun phrase and every finite verb below is a reviewed surface form.
 * The runtime never guesses Turkish suffixes while producing training data.
 */
const EVENT_CARDS = Object.freeze([
  { id: 'home-soup', title: 'çorbayı hazırlama', domain: 'ev ve gündelik yaşam', objectShort: 'çorbayı', objectLong: 'sıcak çorbayı', location: 'mutfakta', source: 'ocaktan', goal: 'masaya', tool: 'tahta kaşık ile', toolCompact: 'kaşıkla', actionVerb: 'karistir', toolVerb: 'karistir' },
  { id: 'home-laundry', title: 'çamaşırları düzenleme', domain: 'ev ve gündelik yaşam', objectShort: 'çamaşırları', objectLong: 'temiz çamaşırları', location: 'salonda', source: 'sepetten', goal: 'dolaba', tool: 'iki elini kullanarak', toolCompact: 'elleriyle', actionVerb: 'duzenle', toolVerb: 'duzenle' },
  { id: 'workshop-screw', title: 'vidayı sıkma', domain: 'atölye ve üretim', objectShort: 'vidayı', objectLong: 'gevşek vidayı', location: 'atölyede', source: 'tezgâhtan', goal: 'kutuya', tool: 'küçük pense ile', toolCompact: 'penseyle', actionVerb: 'sik', toolVerb: 'sik' },
  { id: 'workshop-panel', title: 'panoyu ölçme', domain: 'atölye ve üretim', objectShort: 'panoyu', objectLong: 'mavi panoyu', location: 'atölyede', source: 'raftan', goal: 'duvara', tool: 'tahta cetvel ile', toolCompact: 'cetvelle', actionVerb: 'olc', toolVerb: 'olc' },
  { id: 'school-book', title: 'kitabı işaretleme', domain: 'okul ve öğrenme', objectShort: 'kitabı', objectLong: 'kalın kitabı', location: 'sınıfta', source: 'raftan', goal: 'öğretmene', tool: 'renkli kalem ile', toolCompact: 'kalemle', actionVerb: 'incele', toolVerb: 'isaretle' },
  { id: 'school-map', title: 'haritayı inceleme', domain: 'okul ve öğrenme', objectShort: 'haritayı', objectLong: 'yeni haritayı', location: 'okulda', source: 'dolaptan', goal: 'panoya', tool: 'tahta cetvel ile', toolCompact: 'cetvelle', actionVerb: 'incele', toolVerb: 'olc' },
  { id: 'transport-package', title: 'paketi taşıma', domain: 'ulaşım ve lojistik', objectShort: 'paketi', objectLong: 'ağır paketi', location: 'depoda', source: 'kamyondan', goal: 'araca', tool: 'taşıma arabası ile', toolCompact: 'arabayla', actionVerb: 'tasi', toolVerb: 'tasi' },
  { id: 'transport-crate', title: 'kasayı sayma', domain: 'ulaşım ve lojistik', objectShort: 'kasayı', objectLong: 'tahta kasayı', location: 'garajda', source: 'depodan', goal: 'kamyona', tool: 'sağlam ip ile', toolCompact: 'iple', actionVerb: 'say', toolVerb: 'tasi' },
  { id: 'garden-soil', title: 'toprağı hazırlama', domain: 'doğa ve bahçe', objectShort: 'toprağı', objectLong: 'kuru toprağı', location: 'bahçede', source: 'çuvaldan', goal: 'saksıya', tool: 'küçük kürek ile', toolCompact: 'kürekle', actionVerb: 'hazirla', toolVerb: 'tasi' },
  { id: 'garden-seedling', title: 'fideleri sulama', domain: 'doğa ve bahçe', objectShort: 'fideleri', objectLong: 'genç fideleri', location: 'bahçede', source: 'seradan', goal: 'toprağa', tool: 'bahçe hortumu ile', toolCompact: 'hortumla', actionVerb: 'sula', toolVerb: 'sula' },
  { id: 'shop-order', title: 'siparişi denetleme', domain: 'alışveriş ve hizmet', objectShort: 'siparişi', objectLong: 'yeni siparişi', location: 'dükkânda', source: 'masadan', goal: 'müşteriye', tool: 'kalem kullanarak', toolCompact: 'kalemle', actionVerb: 'denetle', toolVerb: 'denetle' },
  { id: 'shop-parcel', title: 'paketi paketleme', domain: 'alışveriş ve hizmet', objectShort: 'paketi', objectLong: 'kahverengi paketi', location: 'mağazada', source: 'raftan', goal: 'kasaya', tool: 'sağlam bant ile', toolCompact: 'bantla', actionVerb: 'paketle', toolVerb: 'paketle' },
  { id: 'art-canvas', title: 'tuvali boyama', domain: 'sanat ve hobi', objectShort: 'tuvali', objectLong: 'beyaz tuvali', location: 'atölyede', source: 'duvardan', goal: 'şövaleye', tool: 'ince fırça ile', toolCompact: 'fırçayla', actionVerb: 'incele', toolVerb: 'boya' },
  { id: 'art-fabric', title: 'kumaşı kesme', domain: 'sanat ve hobi', objectShort: 'kumaşı', objectLong: 'kırmızı kumaşı', location: 'odada', source: 'dolaptan', goal: 'masaya', tool: 'keskin makas ile', toolCompact: 'makasla', actionVerb: 'duzenle', toolVerb: 'kes' },
  { id: 'sport-ball', title: 'topu yuvarlama', domain: 'spor ve hareket', objectShort: 'topu', objectLong: 'kırmızı topu', location: 'sahada', source: 'sepetten', goal: 'oyuncuya', tool: 'uzun sopa ile', toolCompact: 'sopayla', actionVerb: 'yuvarla', toolVerb: 'yuvarla' },
  { id: 'sport-mat', title: 'minderi temizleme', domain: 'spor ve hareket', objectShort: 'minderi', objectLong: 'mavi minderi', location: 'salonda', source: 'dolaptan', goal: 'sporcuya', tool: 'temiz bez ile', toolCompact: 'bezle', actionVerb: 'duzenle', toolVerb: 'temizle' },
])

const TIME = Object.freeze([
  { id: 'past', cue: 'dün', compactCue: 'dün' },
  { id: 'progressive', cue: 'şimdi', compactCue: 'şimdi' },
  { id: 'future', cue: 'yarın', compactCue: 'yarın' },
  { id: 'aorist', cue: 'her gün', compactCue: 'daima' },
])

const OPTIONAL_ADJUNCTS = Object.freeze([
  { id: 'location', cardField: 'location' },
  { id: 'schedule', surface: 'günlük çalışma planına göre' },
  { id: 'care', surface: 'büyük bir dikkatle' },
  { id: 'repeat', surface: 'yeniden' },
  { id: 'quiet', surface: 'çevresindekileri rahatsız etmeden' },
])

const CONNECTORS = Object.freeze({
  contrast: ['ama', 'fakat', 'ancak', 'oysa'],
  continuation: ['sonra', 'ardından', 'böylece'],
  coordination: ['ve', 'sonra'],
  coreference: ['sonra', 'ardından'],
})

export const TEACHER_FEED_CATALOG_FINGERPRINT = stableHash({
  id: TEACHER_FEED_CATALOG_ID,
  renderer: TEACHER_FEED_RENDERER_ID,
  morphology: TEACHER_FEED_MORPHOLOGY_ID,
  actors: ACTORS,
  verbs: VERBS,
  eventCards: EVENT_CARDS,
  time: TIME,
  adjuncts: OPTIONAL_ADJUNCTS,
  connectors: CONNECTORS,
  rendererLogic: [
    words,
    requestedWordRange,
    tenseIndex,
    cardFor,
    actorFor,
    verbSurface,
    relationKind,
    relationSpec,
    adjunctSurface,
    patientPronoun,
    polarityQuestionParticle,
    orderedClause,
    subset,
    renderCandidate,
    createTeacherChoiceBoard,
    materializeTeacherChoice,
    teacherChoiceSchema,
    teacherChoicePromptBoard,
  ].map(String),
})

export const TEACHER_FEED_ACTOR_IDS = Object.freeze(ACTORS.map((item) => item.id))
export const TEACHER_FEED_EVENT_CARD_IDS = Object.freeze(EVENT_CARDS.map((item) => item.id))

function words(text) {
  return String(text ?? '').match(/[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)?/gu) ?? []
}

function requestedWordRange(axes) {
  const match = typeof axes?.length === 'string' ? axes.length.match(/(\d+)[–-](\d+)\s+kelime/iu) : null
  return match ? { minimum: Number(match[1]), maximum: Number(match[2]) } : { minimum: 4, maximum: 24 }
}

function tenseIndex(axes) {
  const value = String(axes?.tense ?? '')
  if (value.includes('şimdiki')) return 1
  if (value.includes('gelecek')) return 2
  if (value.includes('geniş')) return 3
  return 0
}

function cardFor(id) {
  return EVENT_CARDS.find((item) => item.id === id)
}

function actorFor(id) {
  return ACTORS.find((item) => item.id === id)
}

function verbSurface(verbId, tense, negative = false) {
  const paradigm = VERBS[verbId]
  return paradigm?.[negative ? 'negative' : 'positive']?.[tense]
}

function relationKind(plan) {
  if (plan.form === 'contrast') return 'contrast'
  if (plan.form === 'continuation') return 'continuation'
  if (plan.unit.id === 'coordination') return 'coordination'
  if (plan.unit.id === 'pronoun-reference') return 'coreference'
  return 'none'
}

export function isTeacherFeedCompositePlan(plan) {
  return relationKind(plan) !== 'none'
}

function relationSpec(unitId, card, { compact = false } = {}) {
  const patient = compact ? card.objectShort : card.objectLong
  if (unitId === 'dative-goal') return { parts: [patient, card.goal], visibleRoles: { patient, goal: card.goal }, verbId: 'gotur', witness: { role: 'goal', surface: card.goal } }
  if (unitId === 'ablative-source') return { parts: [card.source, patient], visibleRoles: { source: card.source, patient }, verbId: 'al', witness: { role: 'source', surface: card.source } }
  if (unitId === 'instrumental-tool') {
    const instrument = compact ? card.toolCompact : card.tool
    return { parts: [instrument, patient], visibleRoles: { instrument, patient }, verbId: card.toolVerb, witness: { role: 'instrument', surface: instrument } }
  }
  return { parts: [patient], visibleRoles: { patient }, verbId: card.actionVerb, witness: { role: 'patient', surface: patient } }
}

function adjunctSurface(adjunct, card) {
  return adjunct.cardField ? card[adjunct.cardField] : adjunct.surface
}

function patientPronoun(card) {
  return /(?:ları|leri)$/iu.test(card.objectShort) ? 'onları' : 'onu'
}

function polarityQuestionParticle(verb) {
  const last = [...String(verb)].filter((letter) => /[aeıioöuü]/iu.test(letter)).at(-1)?.toLocaleLowerCase('tr-TR')
  if (last === 'a' || last === 'ı') return 'mı'
  if (last === 'o' || last === 'u') return 'mu'
  if (last === 'ö' || last === 'ü') return 'mü'
  return 'mi'
}

function orderedClause({ subject, time, mandatoryContext = [], relation, adjuncts = [], verb, wordOrder, questionParticle = null, capitalize = true }) {
  const parts = wordOrder?.startsWith('zaman veya yer')
    ? [time, ...mandatoryContext, subject, ...adjuncts, ...relation, verb, questionParticle]
    : wordOrder?.includes('odaklı')
      ? [subject, time, ...mandatoryContext, ...relation, ...adjuncts, verb, questionParticle]
      : [subject, time, ...mandatoryContext, ...adjuncts, ...relation, verb, questionParticle]
  const text = parts.filter(Boolean).join(' ').replace(/\s+/gu, ' ').trim()
  return (capitalize ? text.charAt(0).toLocaleUpperCase('tr-TR') : text.charAt(0).toLocaleLowerCase('tr-TR')) + text.slice(1)
}

function subset(mask, values) {
  return values.filter((_, index) => (mask & (1 << index)) !== 0)
}

function renderCandidate(plan, selection, variant) {
  const primary = cardFor(selection.primaryCardId)
  const primaryActor = actorFor(selection.primaryActorId)
  if (!primary || !primaryActor) return null
  let secondaryActor = actorFor(selection.secondaryActorId) ?? ACTORS[(ACTORS.indexOf(primaryActor) + 1) % ACTORS.length]
  if (secondaryActor.id === primaryActor.id) secondaryActor = ACTORS[(ACTORS.indexOf(primaryActor) + 1) % ACTORS.length]

  const range = requestedWordRange(plan.axes)
  const compact = range.maximum <= 8
  const tense = tenseIndex(plan.axes)
  // Short composites need a one-token habitual cue to preserve both the
  // requested time-first order and the strict 4–8 word surface contract.
  const time = compact ? TIME[tense].compactCue : TIME[tense].cue
  const relation = relationKind(plan)
  const secondary = relation === 'contrast' ? primary : (cardFor(selection.secondaryCardId) ?? primary)
  const primarySpec = relationSpec(plan.unit.id, primary, { compact })
  const negative = plan.form === 'negative' || plan.unit.id === 'negation'
  const question = plan.form === 'question' || plan.unit.id === 'interrogative'
  const subject = plan.unit.id === 'modifier-possessive' ? primaryActor.possessive : primaryActor.surface
  const mandatoryContext = plan.unit.id === 'locative-context' ? [primary.location] : []
  const optional = OPTIONAL_ADJUNCTS.filter((item) => item.id !== 'location' || !mandatoryContext.length)
  const terminal = question && relation === 'none' ? '?' : '.'
  const candidates = []

  for (let mask = 0; mask < (1 << optional.length); mask += 1) {
    const selectedAdjuncts = subset(mask, optional).map((item) => ({ id: item.id, surface: adjunctSurface(item, primary) }))
    const adjuncts = selectedAdjuncts.map((item) => item.surface)
    let firstSpec = primarySpec
    let firstTime = time
    let firstSubject = subject
    let secondClause = null
    let secondState = null
    let connector = null

    if (relation === 'continuation' || relation === 'coreference') {
      const patient = compact ? primary.objectShort : primary.objectLong
      const pronoun = patientPronoun(primary)
      if (plan.unit.id === 'dative-goal') {
        firstSpec = { parts: [patient], visibleRoles: { patient }, verbId: primary.actionVerb, witness: { role: 'patient', surface: patient } }
        const spec = { parts: [pronoun, primary.goal], visibleRoles: { patient: pronoun, goal: primary.goal }, verbId: 'gotur' }
        const predicateSurface = verbSurface(spec.verbId, tense)
        secondClause = orderedClause({ subject: null, time: null, relation: spec.parts, verb: predicateSurface, wordOrder: 'doğal özne-nesne-yüklem', capitalize: false })
        secondState = { card: primary, actor: primaryActor, actorSurface: null, actorExplicit: false, spec, predicateSurface, polarity: 'positive', patientCoreference: true }
      } else if (plan.unit.id === 'ablative-source') {
        firstSpec = { parts: [primary.source, patient], visibleRoles: { source: primary.source, patient }, verbId: 'al', witness: { role: 'source', surface: primary.source } }
        const spec = { parts: [pronoun], visibleRoles: { patient: pronoun }, verbId: primary.actionVerb }
        const predicateSurface = verbSurface(spec.verbId, tense)
        secondClause = orderedClause({ subject: null, time: null, relation: spec.parts, verb: predicateSurface, wordOrder: 'doğal özne-nesne-yüklem', capitalize: false })
        secondState = { card: primary, actor: primaryActor, actorSurface: null, actorExplicit: false, spec, predicateSurface, polarity: 'positive', patientCoreference: true }
      } else {
        firstSpec = compact
          ? { parts: [patient], visibleRoles: { patient }, verbId: 'al', witness: { role: 'patient', surface: patient } }
          : { parts: [primary.source, patient], visibleRoles: { source: primary.source, patient }, verbId: 'al', witness: { role: 'patient', surface: patient } }
        const spec = { parts: [pronoun, primary.goal], visibleRoles: { patient: pronoun, goal: primary.goal }, verbId: 'gotur' }
        const predicateSurface = verbSurface(spec.verbId, tense)
        secondClause = orderedClause({ subject: null, time: null, relation: spec.parts, verb: predicateSurface, wordOrder: 'doğal özne-nesne-yüklem', capitalize: false })
        secondState = { card: primary, actor: primaryActor, actorSurface: null, actorExplicit: false, spec, predicateSurface, polarity: 'positive', patientCoreference: true }
      }
      connector = CONNECTORS[relation][variant % CONNECTORS[relation].length]
      // Only the ablative short form needs subject ellipsis to keep the source,
      // a time-first cue and both events inside the eight-word contract.
      if (compact) {
        if (plan.axes.wordOrder?.startsWith('zaman veya yer')) {
          if (plan.unit.id === 'ablative-source') firstSubject = null
        } else firstTime = null
      }
    } else if (relation === 'contrast') {
      const spec = compact
        ? { parts: [], visibleRoles: {}, verbId: primarySpec.verbId }
        : { parts: [primary.objectLong], visibleRoles: { patient: primary.objectLong }, verbId: primarySpec.verbId }
      const predicateSurface = verbSurface(spec.verbId, tense, true)
      secondClause = orderedClause({ subject: secondaryActor.surface, time: null, relation: spec.parts, verb: predicateSurface, wordOrder: 'doğal özne-nesne-yüklem' })
      secondState = { card: primary, actor: secondaryActor, actorSurface: secondaryActor.surface, actorExplicit: true, spec, predicateSurface, polarity: 'negative', ellipsis: compact ? { patientFromEvent: 0 } : null }
      connector = CONNECTORS.contrast[variant % CONNECTORS.contrast.length]
    } else if (relation === 'coordination') {
      const spec = relationSpec('canonical-transitive', secondary, { compact })
      const predicateSurface = verbSurface(spec.verbId, tense)
      secondClause = orderedClause({ subject: secondaryActor.surface, time: null, relation: spec.parts, verb: predicateSurface, wordOrder: 'doğal özne-nesne-yüklem' })
      secondState = { card: secondary, actor: secondaryActor, actorSurface: secondaryActor.surface, actorExplicit: true, spec, predicateSurface, polarity: 'positive' }
      connector = CONNECTORS.coordination[variant % CONNECTORS.coordination.length]
    }

    const firstPredicateSurface = verbSurface(firstSpec.verbId, tense, negative)
    const questionParticle = question && relation === 'none' ? polarityQuestionParticle(firstPredicateSurface) : null
    const firstClause = orderedClause({
      subject: firstSubject,
      time: firstTime,
      mandatoryContext,
      relation: firstSpec.parts,
      adjuncts,
      verb: firstPredicateSurface,
      wordOrder: plan.axes.wordOrder,
      questionParticle,
    })
    const text = secondClause ? `${firstClause}, ${connector} ${secondClause}${terminal}` : `${firstClause}${terminal}`
    const wordCount = words(text).length
    if (wordCount < range.minimum || wordCount > range.maximum) continue
    candidates.push({
      text,
      wordCount,
      adjunctIds: selectedAdjuncts.map((item) => item.id),
      selectedAdjuncts,
      connector,
      firstState: {
        card: primary,
        actor: primaryActor,
        actorSurface: firstSubject,
        actorExplicit: Boolean(firstSubject),
        actorPossessive: plan.unit.id === 'modifier-possessive',
        spec: firstSpec,
        predicateSurface: firstPredicateSurface,
        polarity: negative ? 'negative' : 'positive',
        timeSurface: firstTime,
        locationSurface: mandatoryContext.includes(primary.location) || selectedAdjuncts.some((item) => item.id === 'location') ? primary.location : null,
        modifiers: selectedAdjuncts
          .filter((item) => item.id !== 'location')
          .map((item) => ({
            id: item.id,
            kind: item.id === 'schedule' ? 'basis' : item.id === 'repeat' ? 'iteration' : item.id === 'quiet' ? 'manner-constraint' : 'manner',
            surface: item.surface,
            explicit: true,
          })),
      },
      secondState,
      questionParticle,
    })
  }
  if (!candidates.length) return null
  candidates.sort((left, right) => left.wordCount - right.wordCount || left.text.localeCompare(right.text, 'tr'))
  const chosen = candidates[variant % candidates.length]

  const semanticEvent = (state, eventIndex) => {
    const roles = Object.fromEntries(Object.entries(state.spec.visibleRoles).map(([role, surface]) => [role, {
      surface,
      explicit: true,
      ...(state.patientCoreference && role === 'patient' ? { refersTo: { event: 0, role: 'patient' } } : {}),
    }]))
    if (state.actorExplicit) {
      roles.actor = state.actorPossessive
        ? { entityId: `friend-of-${state.actor.id}`, surface: state.actorSurface, explicit: true, possessor: { entityId: state.actor.id, surface: state.actor.surface } }
        : { entityId: state.actor.id, surface: state.actorSurface, explicit: true }
    }
    if (state.locationSurface && !roles.location) roles.location = { surface: state.locationSurface, explicit: true }
    if (state.timeSurface) roles.time = { surface: state.timeSurface, explicit: true }
    return {
      event: eventIndex,
      cardId: state.card.id,
      predicate: { id: state.spec.verbId, surface: state.predicateSurface, explicit: true },
      tense: TIME[tense].id,
      polarity: state.polarity,
      epistemicStatus: question && eventIndex === 0 ? 'queried' : 'asserted',
      roles,
      modifiers: state.modifiers ?? [],
      ...(state.actorExplicit ? {} : {
        implicitActor: eventIndex === 0
          ? { entityId: 'implicit-agent', recoverableFromSurface: false }
          : { refersTo: { event: 0, role: 'actor' } },
      }),
      ...(state.ellipsis ? { ellipsis: state.ellipsis } : {}),
    }
  }
  const events = [semanticEvent(chosen.firstState, 0), ...(chosen.secondState ? [semanticEvent(chosen.secondState, 1)] : [])]
  const witnessEvent = plan.unit.id === 'dative-goal' && chosen.secondState ? 1 : 0
  const unitWitness = plan.unit.id === 'negation'
    ? { unitId: plan.unit.id, event: 0, feature: 'polarity', value: 'negative' }
    : plan.unit.id === 'interrogative'
      ? { unitId: plan.unit.id, event: 0, feature: 'question', value: 'polarity' }
      : plan.unit.id === 'tense-aspect'
        ? { unitId: plan.unit.id, event: 0, feature: 'tense', value: TIME[tense].id }
        : plan.unit.id === 'modifier-possessive'
          ? { unitId: plan.unit.id, event: 0, role: 'actor', feature: 'possessor', surface: chosen.firstState.actorSurface }
          : plan.unit.id === 'locative-context'
            ? { unitId: plan.unit.id, event: 0, role: 'location', surface: chosen.firstState.locationSurface }
          : plan.unit.id === 'coordination'
            ? { unitId: plan.unit.id, feature: 'relation', value: relation }
            : plan.unit.id === 'pronoun-reference'
              ? { unitId: plan.unit.id, event: 1, role: 'patient', feature: 'coreference', surface: chosen.secondState?.spec.visibleRoles.patient }
              : { unitId: plan.unit.id, event: witnessEvent, ...(witnessEvent === 1 ? { role: 'goal', surface: primary.goal } : chosen.firstState.spec.witness) }
  const eventCard = {
    version: 2,
    events,
    relation: { kind: relation, ...(relation === 'contrast' ? { contrastDimension: 'polarity', sharedPredicate: true, sharedPatient: true } : {}) },
    question: question ? { event: 0, focus: 'polarity', particle: chosen.questionParticle, answerStatus: 'unknown' } : null,
    coreference: ['continuation', 'coreference'].includes(relation) ? {
      antecedent: { event: 0, role: 'patient' },
      mention: { event: 1, role: 'patient', case: 'acc', surface: chosen.secondState?.spec.visibleRoles.patient },
    } : null,
    unitWitness,
  }
  const semanticIdentity = {
    version: 1,
    unitId: plan.unit.id,
    form: plan.form,
    tense: TIME[tense].id,
    relation,
    primary: {
      cardId: primary.id,
      actorId: chosen.firstState.actorExplicit ? primaryActor.id : 'implicit-agent',
      predicateId: chosen.firstState.spec.verbId,
      polarity: chosen.firstState.polarity,
    },
    secondary: chosen.secondState ? {
      cardId: chosen.secondState.card.id,
      actorId: chosen.secondState.actorExplicit ? chosen.secondState.actor.id : 'event-0-actor',
      predicateId: chosen.secondState.spec.verbId,
      polarity: chosen.secondState.polarity,
    } : null,
    questionFocus: question ? 'polarity' : null,
  }
  return {
    text: chosen.text,
    wordCount: chosen.wordCount,
    resolvedSelection: {
      primaryCardId: primary.id,
      primaryActorId: chosen.firstState.actorExplicit ? primaryActor.id : null,
      secondaryCardId: chosen.secondState?.card.id ?? null,
      secondaryActorId: chosen.secondState?.actorExplicit ? chosen.secondState.actor.id : null,
    },
    semanticCard: eventCard,
    semanticCardDigest: stableHash(eventCard),
    semanticIdentityDigest: stableHash(semanticIdentity),
    renderPlan: {
      rendererId: TEACHER_FEED_RENDERER_ID,
      morphologyId: TEACHER_FEED_MORPHOLOGY_ID,
      requestedLength: plan.axes.length,
      requestedWordOrder: plan.axes.wordOrder,
      adjunctIds: chosen.adjunctIds,
      connector: chosen.connector,
      terminal,
    },
  }
}

/**
 * Builds a deterministic board of already-rendered and structurally distinct
 * alternatives. Ollama only ranks these integer IDs; it cannot author text.
 */
export function createTeacherChoiceBoard(plan, { seed, size = 16, evaluate } = {}) {
  const domainCards = EVENT_CARDS.filter((item) => item.domain === plan.axes.domain)
  if (!domainCards.length) throw Object.assign(new Error(`Müfredat alanı için olay kartı yok: ${plan.axes.domain}`), { code: 'NO_REALIZABLE_EVENT_CARD' })
  const candidates = []
  const seenText = new Set()
  const seenSemantic = new Set()
  const attempts = Math.max(256, size * 24)
  for (let offset = 0; offset < attempts && candidates.length < size; offset += 1) {
    const digest = createHash('sha256').update(`${seed}\0${plan.batchId}\0${offset}`).digest()
    const primary = domainCards[digest[0] % domainCards.length]
    const secondary = domainCards[digest[1] % domainCards.length]
    const actor = ACTORS[digest[2] % ACTORS.length]
    const secondaryActor = ACTORS[digest[3] % ACTORS.length]
    const variant = digest.readUInt16BE(4)
    const selection = {
      primaryCardId: primary.id,
      primaryActorId: actor.id,
      secondaryCardId: secondary.id,
      secondaryActorId: secondaryActor.id,
    }
    const rendered = renderCandidate(plan, selection, variant)
    if (!rendered || seenText.has(rendered.text) || seenSemantic.has(rendered.semanticIdentityDigest)) continue
    const quality = typeof evaluate === 'function' ? evaluate(rendered.text) : { accepted: true, score: 1, reasons: [] }
    if (!quality.accepted) continue
    seenText.add(rendered.text)
    seenSemantic.add(rendered.semanticIdentityDigest)
    const id = candidates.length
    candidates.push({
      id,
      selection: rendered.resolvedSelection,
      ...rendered,
      qualityScore: Number(quality.score ?? 1),
      gloss: [
        `${rendered.resolvedSelection.primaryActorId ?? 'implicit-agent'}/${rendered.resolvedSelection.primaryCardId}`,
        rendered.resolvedSelection.secondaryActorId ? `${rendered.resolvedSelection.secondaryActorId}/${rendered.resolvedSelection.secondaryCardId}` : null,
        TIME[tenseIndex(plan.axes)].id,
        relationKind(plan),
      ].filter(Boolean).join(' · '),
    })
  }
  if (!candidates.length) {
    const error = new Error('İstenen plan için kalite kapısından geçen kapalı olay kartı üretilemedi.')
    error.code = 'NO_REALIZABLE_EVENT_CARD'
    throw error
  }
  const fingerprintPayload = {
    catalogId: TEACHER_FEED_CATALOG_ID,
    catalogFingerprint: TEACHER_FEED_CATALOG_FINGERPRINT,
    plan: { batchId: plan.batchId, unitId: plan.unit.id, form: plan.form, axes: plan.axes },
    seed,
    choices: candidates.map((item) => ({ id: item.id, semanticCardDigest: item.semanticCardDigest, semanticIdentityDigest: item.semanticIdentityDigest, textDigest: sha256(item.text), renderPlan: item.renderPlan })),
  }
  return {
    version: 1,
    id: `choice-board-${stableHash(fingerprintPayload).slice(0, 24)}`,
    fingerprint: stableHash(fingerprintPayload),
    catalogId: TEACHER_FEED_CATALOG_ID,
    catalogFingerprint: TEACHER_FEED_CATALOG_FINGERPRINT,
    choices: candidates,
  }
}

export function materializeTeacherChoice(board, choiceId) {
  if (!Number.isInteger(choiceId)) return { text: '', surfaceReasons: ['invalid-choice-id'] }
  const choice = board?.choices?.find((item) => item.id === choiceId)
  if (!choice) return { text: '', surfaceReasons: ['unknown-choice-id'] }
  const surfaceAssembly = {
    policyId: TEACHER_FEED_RENDERER_ID,
    catalogId: TEACHER_FEED_CATALOG_ID,
    catalogFingerprint: TEACHER_FEED_CATALOG_FINGERPRINT,
    boardId: board.id,
    boardFingerprint: board.fingerprint,
    choiceId,
    rawModelSelection: choiceId,
    resolvedSelection: choice.selection,
    semanticCardDigest: choice.semanticCardDigest,
    semanticIdentityDigest: choice.semanticIdentityDigest,
    semanticCard: choice.semanticCard,
    renderPlan: choice.renderPlan,
    finalTextDigest: sha256(choice.text),
    surfaceOrigin: 'server-vetted-catalog',
    selectionOrigin: 'ollama-closed-choice',
  }
  return {
    text: choice.text,
    surfaceReasons: [],
    surfaceAssembly,
    semanticCardDigest: choice.semanticCardDigest,
    semanticIdentityDigest: choice.semanticIdentityDigest,
    choiceSignature: stableHash({ catalog: TEACHER_FEED_CATALOG_FINGERPRINT, semanticIdentityDigest: choice.semanticIdentityDigest }),
  }
}

export function teacherChoiceSchema(board, count) {
  return {
    type: 'object',
    additionalProperties: false,
    required: ['choices'],
    properties: {
      choices: {
        type: 'array',
        minItems: 1,
        maxItems: count,
        uniqueItems: true,
        items: { type: 'integer', enum: board.choices.map((item) => item.id) },
      },
    },
  }
}

export function teacherChoicePromptBoard(board) {
  return board.choices.map((item) => `${item.id}: ${item.gloss}`).join('\n')
}
