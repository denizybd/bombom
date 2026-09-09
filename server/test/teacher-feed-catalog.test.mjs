import assert from 'node:assert/strict'
import test from 'node:test'
import {
  createTeacherFeedLessonSchema,
  evaluateTeacherText,
  materializeTeacherLesson,
  planTeacherFeedBatch,
  TURKISH_FOUNDATIONS_V1,
} from '../teacher-feed.mjs'
import {
  createTeacherChoiceBoard,
  TEACHER_FEED_ACTOR_IDS,
  TEACHER_FEED_CATALOG_FINGERPRINT,
  TEACHER_FEED_CATALOG_ID,
  TEACHER_FEED_RENDERER_ID,
} from '../teacher-feed-catalog.mjs'

const boardFor = (plan, seed = plan.batchSeed) => createTeacherChoiceBoard(plan, {
  seed,
  size: 16,
  evaluate: (text) => evaluateTeacherText(text, {
    unit: plan.unit,
    form: plan.form,
    axes: plan.axes,
    qualityThreshold: 0.78,
  }),
})

let curriculumMatrixCache

function curriculumMatrix() {
  curriculumMatrixCache ??= Array.from({ length: 240 }, (_, batchIndex) => {
    const plan = planTeacherFeedBatch({ seed: 42, batchIndex, campaignId: 'catalog-matrix' })
    return { plan, board: boardFor(plan) }
  })
  return curriculumMatrixCache
}

function normalizedSurface(value) {
  return String(value ?? '').normalize('NFC').toLocaleLowerCase('tr-TR')
}

function assertSurfaceContains(text, fragment, message) {
  assert.ok(
    normalizedSurface(text).includes(normalizedSurface(fragment)),
    `${message}: ${JSON.stringify(fragment)} not found in ${JSON.stringify(text)}`,
  )
}

function requestedWordRange(length) {
  const match = String(length ?? '').match(/(\d+)[–-](\d+)\s+kelime/iu)
  assert.ok(match, `missing word range in ${JSON.stringify(length)}`)
  return { minimum: Number(match[1]), maximum: Number(match[2]) }
}

function collectExactStringValues(value, values = new Set()) {
  if (typeof value === 'string') values.add(value)
  else if (Array.isArray(value)) value.forEach((item) => collectExactStringValues(item, values))
  else if (value && typeof value === 'object') Object.values(value).forEach((item) => collectExactStringValues(item, values))
  return values
}

function containsStringSurfaceField(schema) {
  if (!schema || typeof schema !== 'object') return false
  if (schema.type === 'string') return true
  return Object.values(schema).some(containsStringSurfaceField)
}

test('closed-choice catalog realizes a broad seeded curriculum matrix without model-authored text', () => {
  const coverage = { units: new Set(), forms: new Set(), lengths: new Set(), domains: new Set(), tenses: new Set(), wordOrders: new Set() }
  for (const { plan, board } of curriculumMatrix()) {
    assert.ok(board.choices.length >= 2, `${plan.unit.id}/${plan.form}/${plan.axes.length}`)
    assert.equal(board.catalogId, TEACHER_FEED_CATALOG_ID)
    assert.equal(board.catalogFingerprint, TEACHER_FEED_CATALOG_FINGERPRINT)
    assert.match(board.fingerprint, /^[a-f0-9]{64}$/u)
    const schema = createTeacherFeedLessonSchema(plan.unit, 2, { board })
    assert.deepEqual(Object.keys(schema.properties), ['choices'])
    assert.equal(schema.properties.choices.items.type, 'integer')
    assert.equal(containsStringSurfaceField(schema), false)
    const lesson = materializeTeacherLesson(board.choices[0].id, { board })
    const quality = evaluateTeacherText(lesson.text, { unit: plan.unit, form: plan.form, axes: plan.axes, qualityThreshold: 0.78 })
    assert.equal(quality.accepted, true, `${lesson.text}: ${quality.reasons.join(', ')}`)
    assert.equal(lesson.surfaceAssembly.surfaceOrigin, 'server-vetted-catalog')
    assert.equal(lesson.surfaceAssembly.selectionOrigin, 'ollama-closed-choice')
    assert.equal(lesson.surfaceAssembly.policyId, TEACHER_FEED_RENDERER_ID)
    coverage.units.add(plan.unit.id)
    coverage.forms.add(plan.form)
    coverage.lengths.add(plan.axes.length)
    coverage.domains.add(plan.axes.domain)
    coverage.tenses.add(plan.axes.tense)
    coverage.wordOrders.add(plan.axes.wordOrder)
  }
  assert.equal(coverage.units.size, TURKISH_FOUNDATIONS_V1.length)
  assert.equal(coverage.forms.size, 5)
  assert.equal(coverage.lengths.size, 3)
  assert.equal(coverage.domains.size, 8)
  assert.equal(coverage.tenses.size, 4)
  assert.equal(coverage.wordOrders.size, 3)
})

test('every catalog semantic card tells the same truth as its compact or normal surface', () => {
  const observedLengths = new Set()
  const allowedRoleNames = new Set(['actor', 'patient', 'goal', 'source', 'instrument', 'location', 'time'])
  let pluralCoreferenceCount = 0

  for (const { plan, board } of curriculumMatrix()) {
    const range = requestedWordRange(plan.axes.length)
    observedLengths.add(plan.axes.length.split(':')[0])
    for (const choice of board.choices) {
      assert.ok(choice.wordCount >= range.minimum && choice.wordCount <= range.maximum, `${choice.text}: ${choice.wordCount}`)
      assert.match(choice.semanticIdentityDigest, /^[a-f0-9]{64}$/u)
      assert.equal(choice.renderPlan.requestedLength, plan.axes.length)
      for (const event of choice.semanticCard.events) {
        assert.ok(event.cardId)
        assert.equal(event.predicate.explicit, true)
        assertSurfaceContains(choice.text, event.predicate.surface, `event ${event.event} predicate`)
        for (const [roleName, role] of Object.entries(event.roles)) {
          assert.ok(allowedRoleNames.has(roleName), `unexpected role ${roleName}`)
          if (!role.explicit) continue
          assert.ok(role.surface, `event ${event.event} explicit ${roleName} must have a surface`)
          assertSurfaceContains(choice.text, role.surface, `event ${event.event} explicit ${roleName}`)
        }
        for (const modifier of event.modifiers ?? []) {
          assert.equal(modifier.explicit, true)
          assert.ok(['basis', 'iteration', 'manner', 'manner-constraint'].includes(modifier.kind), modifier.kind)
          assertSurfaceContains(choice.text, modifier.surface, `event ${event.event} explicit modifier`)
        }
      }
      if (choice.semanticCard.coreference) {
        const antecedent = choice.semanticCard.events[0].roles.patient?.surface
        const mention = choice.semanticCard.coreference.mention.surface
        const expectedPronoun = /(?:ları|leri)$/iu.test(String(antecedent)) ? 'onları' : 'onu'
        assert.equal(mention, expectedPronoun, choice.text)
        if (expectedPronoun === 'onları') pluralCoreferenceCount += 1
        assert.equal(choice.semanticCard.events[1].roles.patient?.surface, expectedPronoun, choice.text)
        assertSurfaceContains(choice.text, expectedPronoun, 'coreference number agreement')
      }
    }
  }

  assert.ok(observedLengths.has('kısa'), 'compact surfaces were not exercised')
  assert.ok(observedLengths.has('orta'), 'normal surfaces were not exercised')
  assert.ok(observedLengths.has('uzun'), 'expanded normal surfaces were not exercised')
  assert.ok(pluralCoreferenceCount > 0, 'plural antecedent/pronoun agreement was not exercised')
})

test('contrast cards keep one patient and predicate while changing polarity', () => {
  let checked = 0
  for (const { plan, board } of curriculumMatrix()) {
    if (plan.form !== 'contrast') continue
    for (const choice of board.choices) {
      const { events, relation } = choice.semanticCard
      assert.equal(events.length, 2, choice.text)
      const [first, second] = events
      assert.equal(relation.kind, 'contrast')
      assert.equal(relation.contrastDimension, 'polarity')
      assert.equal(relation.sharedPredicate, true)
      assert.equal(relation.sharedPatient, true)
      assert.equal(first.cardId, second.cardId, choice.text)
      assert.equal(first.predicate.id, second.predicate.id, choice.text)
      assert.notEqual(first.polarity, second.polarity, choice.text)
      if (second.ellipsis) assert.equal(Object.hasOwn(second.ellipsis, 'predicateFromEvent'), false, choice.text)
      const secondPatient = second.roles.patient?.surface
        ?? (second.ellipsis?.patientFromEvent === 0 ? first.roles.patient?.surface : null)
      assert.equal(secondPatient, first.roles.patient?.surface, choice.text)
      checked += 1
    }
  }
  assert.ok(checked > 0)
})

test('possessive modifiers name the friend as actor and retain the actual possessor', () => {
  let checked = 0
  for (const { plan, board } of curriculumMatrix()) {
    if (plan.unit.id !== 'modifier-possessive') continue
    for (const choice of board.choices) {
      const actor = choice.semanticCard.events[0].roles.actor
      assert.ok(actor, choice.text)
      assert.equal(actor.explicit, true)
      assert.ok(actor.possessor, choice.text)
      assert.equal(actor.entityId, `friend-of-${actor.possessor.entityId}`)
      assertSurfaceContains(actor.surface, actor.possessor.surface, 'possessive actor')
      assertSurfaceContains(choice.text, actor.surface, 'possessive actor surface')
      assert.deepEqual(choice.semanticCard.unitWitness, {
        unitId: 'modifier-possessive',
        event: 0,
        role: 'actor',
        feature: 'possessor',
        surface: actor.surface,
      })
      checked += 1
    }
  }
  assert.ok(checked > 0)
})

test('interrogatives are explicit polarity questions rather than unsupported causal questions', () => {
  let checked = 0
  for (const { plan, board } of curriculumMatrix()) {
    if (plan.form !== 'question' && plan.unit.id !== 'interrogative') continue
    for (const choice of board.choices) {
      const question = choice.semanticCard.question
      const event = choice.semanticCard.events[question?.event ?? -1]
      assert.ok(question, choice.text)
      assert.equal(question.focus, 'polarity')
      assert.equal(question.answerStatus, 'unknown')
      assert.equal(Object.hasOwn(question, 'expectedAnswer'), false)
      assert.match(question.particle, /^(?:mı|mi|mu|mü)$/u)
      assert.equal(choice.text.endsWith('?'), true)
      assert.equal(normalizedSurface(choice.text).includes('neden'), false)
      assert.ok(event)
      assert.equal(event.epistemicStatus, 'queried')
      assert.ok(normalizedSurface(choice.text).endsWith(`${normalizedSurface(event.predicate.surface)} ${question.particle}?`), choice.text)
      assert.deepEqual(choice.semanticCard.unitWitness, {
        unitId: plan.unit.id,
        event: 0,
        feature: 'question',
        value: 'polarity',
      })
      checked += 1
    }
  }
  assert.ok(checked > 0)
})

test('time-or-place-first render plans really begin with an explicit time or place', () => {
  let checked = 0
  for (const { plan, board } of curriculumMatrix()) {
    if (!plan.axes.wordOrder.startsWith('zaman veya yer')) continue
    for (const choice of board.choices) {
      assert.equal(choice.renderPlan.requestedWordOrder, plan.axes.wordOrder)
      const roles = choice.semanticCard.events[0].roles
      const leadingSurfaces = [roles.time?.surface, roles.location?.surface].filter(Boolean)
      assert.ok(leadingSurfaces.length > 0, choice.text)
      assert.ok(
        leadingSurfaces.some((surface) => normalizedSurface(choice.text).startsWith(normalizedSurface(surface))),
        `${choice.text} must start with ${leadingSurfaces.join(' or ')}`,
      )
      checked += 1
    }
  }
  assert.ok(checked > 0)
})

test('compact time-first continuations never turn an omitted subject into a hidden named actor', () => {
  let omittedActors = 0
  let explicitActors = 0
  const basePlan = planTeacherFeedBatch({ seed: 91, batchIndex: 0, campaignId: 'compact-time-first-actor' })
  const cases = ['ablative-source', 'dative-goal', 'pronoun-reference'].map((unitId, index) => {
    const plan = {
      ...basePlan,
      batchId: `compact-time-first-${unitId}`,
      batchSeed: 91_000 + index,
      unit: TURKISH_FOUNDATIONS_V1.find((unit) => unit.id === unitId),
      form: 'continuation',
      axes: {
        domain: 'ulaşım ve lojistik',
        tense: 'belirgin geçmiş zaman',
        length: 'kısa: 4–8 kelime',
        wordOrder: 'zaman veya yer ifadesiyle başlayan sıra',
      },
    }
    return { plan, board: boardFor(plan) }
  })

  for (const { plan, board } of cases) {
    for (const choice of board.choices) {
      assert.ok(['continuation', 'coreference'].includes(choice.semanticCard.relation.kind))
      const [first, second] = choice.semanticCard.events
      const materialized = materializeTeacherLesson(choice.id, { board })
      if (!first.roles.actor) {
        assert.equal(plan.unit.id, 'ablative-source', 'only the overfull source-bearing short clause may omit its subject')
        assert.deepEqual(first.implicitActor, { entityId: 'implicit-agent', recoverableFromSurface: false }, choice.text)
        assert.deepEqual(second.implicitActor, { refersTo: { event: 0, role: 'actor' } }, choice.text)
        assert.equal(choice.selection.primaryActorId, null)
        assert.equal(choice.resolvedSelection.primaryActorId, null)
        assert.equal(choice.selection.secondaryActorId, null)
        assert.equal(choice.resolvedSelection.secondaryActorId, null)
        assert.equal(materialized.surfaceAssembly.resolvedSelection.primaryActorId, null)
        assert.equal(materialized.surfaceAssembly.resolvedSelection.secondaryActorId, null)
        assert.equal(Object.hasOwn(choice, 'semanticIdentity'), false, 'semantic identity must stay opaque and digest-only')
        assert.match(choice.semanticIdentityDigest, /^[a-f0-9]{64}$/u)

        const exposedIdentityValues = collectExactStringValues({
          selection: choice.selection,
          resolvedSelection: choice.resolvedSelection,
          semanticCard: choice.semanticCard,
          glossTokens: choice.gloss.split(/[^\p{L}\p{N}-]+/u),
        })
        for (const actorId of TEACHER_FEED_ACTOR_IDS) {
          assert.equal(exposedIdentityValues.has(actorId), false, `${choice.text} leaked hidden actor id ${actorId}`)
        }
        omittedActors += 1
      } else {
        assert.notEqual(plan.unit.id, 'ablative-source', 'the forced ellipsis case unexpectedly retained a selected actor')
        assert.equal(first.roles.actor.explicit, true)
        assert.equal(Object.hasOwn(first, 'implicitActor'), false)
        assert.equal(choice.selection.primaryActorId, first.roles.actor.entityId)
        assert.equal(choice.resolvedSelection.primaryActorId, first.roles.actor.entityId)
        assert.equal(materialized.surfaceAssembly.resolvedSelection.primaryActorId, first.roles.actor.entityId)
        assertSurfaceContains(choice.text, first.roles.actor.surface, 'compact explicit actor')
        explicitActors += 1
      }
    }
  }
  assert.ok(omittedActors > 0, 'subject-ellipsis path was not exercised')
  assert.ok(explicitActors > 0, 'subject-fitting path was not exercised')
})

test('same plan and seed replay the exact board while unknown choices fail closed', () => {
  const plan = planTeacherFeedBatch({ seed: 73, batchIndex: 17, campaignId: 'catalog-replay' })
  const first = boardFor(plan)
  const replay = boardFor(plan)
  assert.equal(first.id, replay.id)
  assert.equal(first.fingerprint, replay.fingerprint)
  assert.deepEqual(first.choices, replay.choices)
  assert.deepEqual(materializeTeacherLesson(0, { board: first }), materializeTeacherLesson(0, { board: replay }))
  assert.deepEqual(materializeTeacherLesson(99_999, { board: first }).surfaceReasons, ['unknown-choice-id'])
  assert.deepEqual(materializeTeacherLesson('0', { board: first }).surfaceReasons, ['invalid-choice-id'])
})
