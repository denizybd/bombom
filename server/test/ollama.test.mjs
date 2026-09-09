import test from 'node:test'
import assert from 'node:assert/strict'
import { OllamaTeacher } from '../ollama.mjs'
import { teacherLessonSchema } from '../schema.mjs'
import { defaultConfig } from '../config.mjs'
import { TEACHER_CONTEXT_TOKEN_LIMIT, TEACHER_FEED_CONTEXT_TOKEN_LIMIT } from '../teacher-usage.mjs'
import { createTeacherChoiceBoard } from '../teacher-feed-catalog.mjs'
import { createTeacherFeedLessonSchema, evaluateTeacherText, filterTeacherLessons, materializeTeacherLesson, TURKISH_FOUNDATIONS_V1 } from '../teacher-feed.mjs'

test('Ollama adapter forces deterministic structured output and validates JSON', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:latest'
  let requestBody
  const teacher = new OllamaTeacher(() => config, async (_url, init) => {
    requestBody = JSON.parse(init.body)
    return Response.json({ model: 'teacher:latest', message: { content: JSON.stringify({ lessons: [{ text: 'Ayşe kağıdı kesti.', kind: 'contrast', rationale: 'Araç değişimi' }] }) }, prompt_eval_count: 10, eval_count: 8 })
  })
  const result = await teacher.generateStructured({ messages: [{ role: 'user', content: 'test' }], schema: teacherLessonSchema, options: { temperature: 0.9 } })
  assert.equal(result.value.lessons.length, 1)
  assert.equal(requestBody.stream, false)
  assert.deepEqual(requestBody.format, teacherLessonSchema)
  assert.equal(requestBody.options.temperature, 0)
  assert.equal(requestBody.options.num_ctx, TEACHER_CONTEXT_TOKEN_LIMIT)
})

test('Ollama adapter rejects schema-invalid teacher output', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:latest'
  const teacher = new OllamaTeacher(() => config, async () => Response.json({ message: { content: '{"lessons":[]}' } }))
  await assert.rejects(teacher.generateStructured({ messages: [], schema: teacherLessonSchema }), { code: 'SCHEMA_VALIDATION_FAILED' })
})

test('campaign-only diverse generation uses bounded sampling while critic generation stays deterministic', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:14b'
  const bodies = []
  const teacher = new OllamaTeacher(() => config, async (_url, init) => {
    bodies.push(JSON.parse(init.body))
    return Response.json({
      model: 'teacher:14b',
      message: { content: JSON.stringify({ lessons: [{ text: 'Çocuk sabah çantasını sessizce hazırladı.', kind: 'paraphrase', rationale: 'Çeşitli yüzey' }] }) },
    })
  })
  await teacher.generateStructured({ messages: [], schema: teacherLessonSchema, options: { seed: 99, temperature: 0.9 } })
  await teacher.generateDiverseStructured({ messages: [], schema: teacherLessonSchema, seed: 1234, maxCompletionTokens: 200 })
  assert.equal(bodies[0].options.temperature, 0)
  assert.equal(bodies[0].options.seed, 99)
  assert.deepEqual(bodies[1].options, {
    seed: 1234,
    temperature: 0.65,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.12,
    num_predict: 200,
    num_ctx: TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
  })
})

test('campaign closed integer choices cross the real adapter boundary before server materialization and quality gates', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:14b'
  const unit = TURKISH_FOUNDATIONS_V1.find((item) => item.id === 'canonical-transitive')
  const plan = {
    batchIndex: 0,
    batchId: 'tfb-real-adapter-v7',
    batchSeed: 41,
    unit,
    form: 'statement',
    axes: {
      domain: 'ev ve gündelik yaşam',
      tense: 'belirgin geçmiş zaman',
      length: 'orta: 9–15 kelime',
      wordOrder: 'doğal özne-nesne-yüklem',
    },
  }
  const boardOptions = {
    seed: 41,
    size: 4,
    evaluate: (text) => evaluateTeacherText(text, { unit, form: plan.form, axes: plan.axes }),
  }
  const board = createTeacherChoiceBoard(plan, boardOptions)
  assert.deepEqual(createTeacherChoiceBoard(plan, boardOptions), board)

  const schema = createTeacherFeedLessonSchema(unit, 1, { board })
  assert.deepEqual(schema.required, ['choices'])
  assert.equal(schema.properties.choices.items.type, 'integer')
  assert.deepEqual(schema.properties.choices.items.enum, board.choices.map((choice) => choice.id))

  const selectedChoiceId = board.choices[0].id
  let requestBody
  const teacher = new OllamaTeacher(() => config, async (_url, init) => {
    requestBody = JSON.parse(init.body)
    return Response.json({
      model: 'teacher:14b',
      message: { content: JSON.stringify({ choices: [selectedChoiceId] }) },
      prompt_eval_count: 11,
      eval_count: 19,
    })
  })
  const generated = await teacher.generateDiverseStructured({ messages: [], schema, seed: 41, maxCompletionTokens: 128 })
  assert.deepEqual(requestBody.format, schema)
  assert.deepEqual(generated.value, { choices: [selectedChoiceId] })

  const lesson = {
    ...materializeTeacherLesson(generated.value.choices[0], { board }),
    skill: unit.id,
    form: plan.form,
  }
  assert.equal(lesson.text, board.choices[0].text)
  assert.equal(lesson.surfaceAssembly.boardFingerprint, board.fingerprint)
  assert.equal(lesson.surfaceAssembly.selectionOrigin, 'ollama-closed-choice')
  assert.equal(lesson.surfaceAssembly.surfaceOrigin, 'server-vetted-catalog')
  const filtered = filterTeacherLessons([lesson], { unit, form: plan.form, axes: plan.axes })
  assert.equal(filtered.accepted.length, 1)
})

test('campaign timeout is classified once and conservatively accounts the bounded output', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:14b'
  config.ollama.timeoutMs = 300_000
  let calls = 0
  const teacher = new OllamaTeacher(() => config, async () => {
    calls += 1
    return Response.json({ error: 'timed out waiting for llama runner to respond' }, { status: 500 })
  })
  await assert.rejects(
    teacher.generateDiverseStructured({ messages: [], schema: teacherLessonSchema, seed: 123 }),
    (error) => {
      assert.equal(error.code, 'OLLAMA_TIMEOUT')
      assert.equal(error.retryable, false)
      assert.equal(error.timeoutMs, 300_000)
      assert.equal(error.completionTokens, 128)
      assert.equal(error.usageEstimated, true)
      assert.equal(error.usage.usageEstimated, true)
      return true
    },
  )
  assert.equal(calls, 1)
})

test('diverse campaign generation makes one HTTP call and exposes invalid-output token usage', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:14b'
  config.ollama.retries = 2
  let calls = 0
  const teacher = new OllamaTeacher(() => config, async () => {
    calls += 1
    return Response.json({
      model: 'teacher:14b',
      message: { content: '{geçersiz-json' },
      prompt_eval_count: 13,
      eval_count: 7,
      total_duration: 1234,
    })
  })
  await assert.rejects(
    teacher.generateDiverseStructured({ messages: [], schema: teacherLessonSchema, seed: 81 }),
    (error) => {
      assert.equal(error.code, 'OLLAMA_INVALID_JSON')
      assert.equal(error.promptTokens, 13)
      assert.equal(error.completionTokens, 7)
      assert.equal(error.totalDurationNs, 1234)
      assert.deepEqual(error.usage, { promptTokens: 13, completionTokens: 7, totalDurationNs: 1234, model: 'teacher:14b' })
      return true
    },
  )
  assert.equal(calls, 1, 'campaign retry ownership belongs to generateBatch')
})

test('Ollama adapter preserves whether each token counter was actually reported', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:14b'
  const teacher = new OllamaTeacher(() => config, async () => Response.json({
    model: 'teacher:14b',
    message: { content: JSON.stringify({ lessons: [{ text: 'Ayşe kağıdı kesti.', kind: 'contrast', rationale: 'Sayaçsız geçerli yanıt' }] }) },
    prompt_eval_count: 0,
  }))

  const result = await teacher.generateDiverseStructured({ messages: [], schema: teacherLessonSchema, seed: 82 })
  assert.equal(result.promptTokens, 0)
  assert.equal(result.completionTokens, 0)
  assert.equal(result.promptTokensReported, true)
  assert.equal(result.completionTokensReported, false)
})

test('Ollama adapter rejects coercible non-numeric token counters as unreported', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:14b'
  for (const invalidCounter of [null, '', false]) {
    const teacher = new OllamaTeacher(() => config, async () => Response.json({
      model: 'teacher:14b',
      message: { content: JSON.stringify({ lessons: [{ text: 'Ayşe kağıdı kesti.', kind: 'contrast', rationale: 'Geçersiz sayaç türü' }] }) },
      prompt_eval_count: invalidCounter,
      eval_count: invalidCounter,
    }))

    const result = await teacher.generateDiverseStructured({ messages: [], schema: teacherLessonSchema, seed: 83 })
    assert.equal(result.promptTokens, 0)
    assert.equal(result.completionTokens, 0)
    assert.equal(result.promptTokensReported, false)
    assert.equal(result.completionTokensReported, false)
  }
})

test('deterministic structured generation keeps retries and aggregates retry token usage', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'teacher:latest'
  config.ollama.retries = 2
  let calls = 0
  const teacher = new OllamaTeacher(() => config, async () => {
    calls += 1
    if (calls === 1) {
      return Response.json({ message: { content: '{invalid' }, prompt_eval_count: 2, eval_count: 3, total_duration: 10 })
    }
    return Response.json({
      model: 'teacher:latest',
      message: { content: JSON.stringify({ lessons: [{ text: 'Ayşe kağıdı kesti.', kind: 'contrast', rationale: 'Geçerli tekrar' }] }) },
      prompt_eval_count: 5,
      eval_count: 7,
      total_duration: 20,
    })
  })
  const result = await teacher.generateStructured({ messages: [], schema: teacherLessonSchema })
  assert.equal(calls, 2)
  assert.equal(result.promptTokens, 7)
  assert.equal(result.completionTokens, 10)
  assert.equal(result.totalDurationNs, 30)
})

test('Ollama lifecycle unload uses keep_alive zero without teacher inference', async () => {
  const config = defaultConfig('/tmp/no-write')
  config.ollama.model = 'qwen2.5-coder:14b'
  let pathname
  let body
  const teacher = new OllamaTeacher(() => config, async (url, init) => {
    pathname = new URL(url).pathname
    body = JSON.parse(init.body)
    return Response.json({ model: 'qwen2.5-coder:14b', done: true, done_reason: 'unload' })
  })
  const result = await teacher.unload()
  assert.equal(pathname, '/api/generate')
  assert.deepEqual(body, {
    model: 'qwen2.5-coder:14b',
    prompt: '',
    stream: false,
    keep_alive: 0,
  })
  assert.deepEqual(result, { model: 'qwen2.5-coder:14b', unloaded: true, doneReason: 'unload' })
})
