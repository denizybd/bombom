const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value)

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (isRecord(value)) return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`
  return JSON.stringify(value)
}

export function validateJsonSchema(value, schema, path = '$') {
  const errors = []
  visit(value, schema, path, errors)
  return { valid: errors.length === 0, errors }
}

function visit(value, schema, path, errors) {
  if (!schema || typeof schema !== 'object') {
    errors.push(`${path}: geçersiz şema`)
    return
  }
  if (schema.enum && !schema.enum.some((item) => Object.is(item, value))) {
    errors.push(`${path}: izin verilen değerlerden biri değil`)
    return
  }
  const type = schema.type
  if (type === 'object') {
    if (!isRecord(value)) {
      errors.push(`${path}: nesne olmalı`)
      return
    }
    for (const key of schema.required ?? []) {
      if (!(key in value)) errors.push(`${path}.${key}: zorunlu alan eksik`)
    }
    for (const [key, child] of Object.entries(schema.properties ?? {})) {
      if (key in value) visit(value[key], child, `${path}.${key}`, errors)
    }
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!(key in (schema.properties ?? {}))) errors.push(`${path}.${key}: bilinmeyen alan`)
      }
    }
  } else if (type === 'array') {
    if (!Array.isArray(value)) {
      errors.push(`${path}: dizi olmalı`)
      return
    }
    if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${path}: en az ${schema.minItems} öğe olmalı`)
    if (schema.maxItems !== undefined && value.length > schema.maxItems) errors.push(`${path}: en fazla ${schema.maxItems} öğe olmalı`)
    if (schema.uniqueItems === true && new Set(value.map(canonicalJson)).size !== value.length) errors.push(`${path}: öğeler benzersiz olmalı`)
    value.forEach((item, index) => visit(item, schema.items ?? {}, `${path}[${index}]`, errors))
  } else if (type === 'string') {
    if (typeof value !== 'string') errors.push(`${path}: metin olmalı`)
    else {
      if (schema.minLength !== undefined && value.length < schema.minLength) errors.push(`${path}: çok kısa`)
      if (schema.maxLength !== undefined && value.length > schema.maxLength) errors.push(`${path}: çok uzun`)
      if (schema.pattern && !(new RegExp(schema.pattern).test(value))) errors.push(`${path}: beklenen biçime uymuyor`)
    }
  } else if (type === 'number' || type === 'integer') {
    if (typeof value !== 'number' || !Number.isFinite(value) || (type === 'integer' && !Number.isInteger(value))) errors.push(`${path}: ${type === 'integer' ? 'tam sayı' : 'sayı'} olmalı`)
    else {
      if (schema.minimum !== undefined && value < schema.minimum) errors.push(`${path}: alt sınırın altında`)
      if (schema.maximum !== undefined && value > schema.maximum) errors.push(`${path}: üst sınırın üzerinde`)
    }
  } else if (type === 'boolean' && typeof value !== 'boolean') {
    errors.push(`${path}: boolean olmalı`)
  }
}

export function assertSchema(value, schema, label = 'Yanıt') {
  const result = validateJsonSchema(value, schema)
  if (!result.valid) {
    const error = new Error(`${label} şema doğrulamasından geçemedi: ${result.errors.slice(0, 8).join('; ')}`)
    error.code = 'SCHEMA_VALIDATION_FAILED'
    error.details = result.errors
    throw error
  }
  return value
}

export const teacherLessonSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['lessons'],
  properties: {
    lessons: {
      type: 'array',
      minItems: 1,
      maxItems: 24,
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['text', 'kind', 'rationale'],
        properties: {
          text: { type: 'string', minLength: 3, maxLength: 4_000 },
          kind: { type: 'string', enum: ['paraphrase', 'contrast', 'continuation', 'role_swap', 'correction'] },
          rationale: { type: 'string', minLength: 1, maxLength: 500 },
        },
      },
    },
  },
}
