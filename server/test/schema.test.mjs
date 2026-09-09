import test from 'node:test'
import assert from 'node:assert/strict'
import { assertSchema, teacherLessonSchema, validateJsonSchema } from '../schema.mjs'

test('structured teacher schema accepts only bounded exact objects', () => {
  const valid = { lessons: [{ text: 'Ali ekmeği kesti.', kind: 'continuation', rationale: 'Rol ilişkisi' }] }
  assert.equal(validateJsonSchema(valid, teacherLessonSchema).valid, true)
  assert.equal(validateJsonSchema({ lessons: [{ text: 'x', kind: 'unknown', rationale: '' }], hidden: true }, teacherLessonSchema).valid, false)
  assert.throws(() => assertSchema({ lessons: 'nope' }, teacherLessonSchema), { code: 'SCHEMA_VALIDATION_FAILED' })
})

test('array uniqueItems rejects duplicate structured choices', () => {
  const schema = {
    type: 'object', additionalProperties: false, required: ['choices'],
    properties: { choices: { type: 'array', minItems: 1, uniqueItems: true, items: { type: 'integer', enum: [0, 1, 2] } } },
  }
  assert.equal(validateJsonSchema({ choices: [0, 2] }, schema).valid, true)
  assert.equal(validateJsonSchema({ choices: [1, 1] }, schema).valid, false)
})
