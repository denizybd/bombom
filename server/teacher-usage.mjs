export const TEACHER_CONTEXT_TOKEN_LIMIT = 32_768
export const TEACHER_FEED_CONTEXT_TOKEN_LIMIT = 4_096

function positiveSafeInteger(value) {
  return Number.isSafeInteger(value) && value > 0 ? value : 0
}

function explicitContextTokenLimit(value) {
  if (value === undefined) return TEACHER_CONTEXT_TOKEN_LIMIT
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError('Öğretmen bağlam token sınırı pozitif ve güvenli bir tam sayı olmalı.')
  }
  return value
}

function strictReportedTokenComponent(source, tokenKey, reportedKey) {
  const nested = source?.usage
  const directHasValue = Object.prototype.hasOwnProperty.call(source ?? {}, tokenKey)
  const nestedHasValue = Object.prototype.hasOwnProperty.call(nested ?? {}, tokenKey)
  const directFlag = typeof source?.[reportedKey] === 'boolean' ? source[reportedKey] : undefined
  const nestedFlag = typeof nested?.[reportedKey] === 'boolean' ? nested[reportedKey] : undefined
  const explicitFlag = directFlag ?? nestedFlag
  const raw = directHasValue ? source[tokenKey] : nestedHasValue ? nested[tokenKey] : undefined
  const valid = typeof raw === 'number' && Number.isSafeInteger(raw) && raw >= 0
  const reported = explicitFlag === false
    ? false
    : explicitFlag === true
      ? valid
      : (directHasValue || nestedHasValue) && valid
  return { reported, value: reported ? raw : 0 }
}

export function teacherPromptTokenUpperBound(_messages, _schema, _extra = {}) {
  // The caller must pass the same explicit limit it enforces as Ollama
  // `num_ctx`. General structured/thought calls deliberately retain the
  // conservative 32K default; the closed-choice feed owns a smaller, separately
  // versioned 4K contract.
  return explicitContextTokenLimit(_extra?.contextTokenLimit)
}

/**
 * Recover the conservative prompt debt for an already persisted attempt.
 * `ollama-num-ctx-v1` is proof that `contextTokenLimit` was enforced at the
 * adapter boundary. Attempts predating that proof retain the historical 32K
 * floor even if they happen to contain an untrusted similarly named field.
 */
export function teacherAttemptPromptTokenUpperBound(attempt) {
  const persistedPromptTokens = positiveSafeInteger(attempt?.maxPromptTokens)
  const declaredContextLimit = positiveSafeInteger(attempt?.contextTokenLimit)
  const hasEnforcedContextProof = attempt?.promptBoundPolicy === 'ollama-num-ctx-v1' && declaredContextLimit > 0
  return Math.max(
    persistedPromptTokens,
    hasEnforcedContextProof ? declaredContextLimit : TEACHER_CONTEXT_TOKEN_LIMIT,
  )
}

/**
 * Reserve one bounded Ollama call before network I/O. A completed call is
 * settled to strict reported usage below; a crash before settlement leaves
 * the conservative maximum charged, so uncertainty can over-count but cannot
 * silently reopen the daily budget.
 */
export function reserveBoundedTeacherCall(store, {
  messages,
  schema,
  maxCompletionTokens,
  model,
  signal,
  thoughtEpisodeId,
  contextTokenLimit = TEACHER_CONTEXT_TOKEN_LIMIT,
  usageDay = new Date().toISOString().slice(0, 10),
}) {
  signal?.throwIfAborted()
  if (typeof store?.reserveTeacherCallUsage !== 'function') {
    const error = new Error('Atomik öğretmen kullanım rezervasyonu store primitive’i eksik.')
    error.code = 'STORE_USAGE_CONTRACT_MISSING'
    throw error
  }
  if (!Number.isSafeInteger(maxCompletionTokens) || maxCompletionTokens <= 0) {
    throw new TypeError('Öğretmen çıktı token üst sınırı pozitif bir tam sayı olmalı.')
  }
  const selectedModel = typeof model === 'string' ? model.trim() : ''
  if (!selectedModel) {
    const error = new Error('Ollama modeli seçilmedi.')
    error.code = 'OLLAMA_MODEL_REQUIRED'
    throw error
  }
  const maxPromptTokens = teacherPromptTokenUpperBound(messages, schema, {
    model,
    contextTokenLimit,
  })
  const reservedTokens = maxPromptTokens + maxCompletionTokens
  return store.reserveTeacherCallUsage({
    usageDay,
    model: selectedModel,
    maxPromptTokens,
    maxCompletionTokens,
    reservedTokens,
    ...(thoughtEpisodeId ? { thoughtEpisodeId } : {}),
  })
}

/**
 * Replace the crash-safe pre-I/O maximum with strict reported usage. Missing
 * or malformed components retain their corresponding reserved maximum. The
 * store settlement is idempotent and may record an actual count above the
 * reservation so an untrusted/misbehaving adapter cannot hide spent tokens.
 */
export function settleBoundedTeacherCall(store, reservation, source) {
  if (typeof store?.settleTeacherCallUsage !== 'function') {
    const error = new Error('Atomik öğretmen kullanım kapatma primitive’i eksik.')
    error.code = 'STORE_USAGE_CONTRACT_MISSING'
    throw error
  }
  if (!reservation || typeof reservation.id !== 'string' || !reservation.id) {
    throw new TypeError('Öğretmen kullanım rezervasyonu kimliği eksik.')
  }
  const prompt = strictReportedTokenComponent(source, 'promptTokens', 'promptTokensReported')
  const completion = strictReportedTokenComponent(source, 'completionTokens', 'completionTokensReported')
  const promptTokens = prompt.reported ? prompt.value : reservation.maxPromptTokens
  const completionTokens = completion.reported ? completion.value : reservation.maxCompletionTokens
  const usageEstimated = source?.usageEstimated === true || source?.usage?.usageEstimated === true || !prompt.reported || !completion.reported
  return store.settleTeacherCallUsage(reservation.id, {
    promptTokens,
    completionTokens,
    usageEstimated,
  })
}
