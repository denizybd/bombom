import assert from 'node:assert/strict'
import test from 'node:test'
import {
  reserveBoundedTeacherCall,
  teacherPromptTokenUpperBound,
  TEACHER_CONTEXT_TOKEN_LIMIT,
  TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
} from '../teacher-usage.mjs'

test('prompt bounds keep the general default and accept an explicit enforced context', () => {
  assert.equal(teacherPromptTokenUpperBound([], {}), TEACHER_CONTEXT_TOKEN_LIMIT)
  assert.equal(
    teacherPromptTokenUpperBound([], {}, { contextTokenLimit: TEACHER_FEED_CONTEXT_TOKEN_LIMIT }),
    TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
  )
  assert.throws(
    () => teacherPromptTokenUpperBound([], {}, { contextTokenLimit: 0 }),
    /pozitif ve güvenli bir tam sayı/u,
  )
})

test('bounded reservation uses the explicit context plus completion while default stays conservative', () => {
  const reservations = []
  const store = {
    reserveTeacherCallUsage(input) {
      reservations.push(structuredClone(input))
      return { id: `reservation-${reservations.length}`, ...structuredClone(input) }
    },
  }
  reserveBoundedTeacherCall(store, {
    messages: [],
    schema: {},
    maxCompletionTokens: 128,
    model: 'teacher:14b',
  })
  reserveBoundedTeacherCall(store, {
    messages: [],
    schema: {},
    maxCompletionTokens: 128,
    model: 'teacher:14b',
    contextTokenLimit: TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
  })

  assert.equal(reservations[0].maxPromptTokens, TEACHER_CONTEXT_TOKEN_LIMIT)
  assert.equal(reservations[0].reservedTokens, TEACHER_CONTEXT_TOKEN_LIMIT + 128)
  assert.equal(reservations[1].maxPromptTokens, TEACHER_FEED_CONTEXT_TOKEN_LIMIT)
  assert.equal(reservations[1].reservedTokens, TEACHER_FEED_CONTEXT_TOKEN_LIMIT + 128)
})
