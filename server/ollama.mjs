import { assertSchema } from './schema.mjs'
import { TEACHER_CONTEXT_TOKEN_LIMIT, TEACHER_FEED_CONTEXT_TOKEN_LIMIT } from './teacher-usage.mjs'

export const OLLAMA_DIVERSE_DEFAULT_NUM_PREDICT = 128
export const OLLAMA_DIVERSE_MAX_NUM_PREDICT = 256

const wait = (ms, signal) => new Promise((resolve, reject) => {
  const timer = setTimeout(resolve, ms)
  signal?.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason ?? new DOMException('Aborted', 'AbortError')) }, { once: true })
})

function combinedSignal(signal, timeoutMs) {
  const timeout = AbortSignal.timeout(timeoutMs)
  return signal ? AbortSignal.any([signal, timeout]) : timeout
}

function responseUsage(result) {
  const promptRaw = result?.prompt_eval_count
  const completionRaw = result?.eval_count
  const promptTokensReported = Object.prototype.hasOwnProperty.call(result ?? {}, 'prompt_eval_count') &&
    typeof promptRaw === 'number' && Number.isSafeInteger(promptRaw) && promptRaw >= 0
  const completionTokensReported = Object.prototype.hasOwnProperty.call(result ?? {}, 'eval_count') &&
    typeof completionRaw === 'number' && Number.isSafeInteger(completionRaw) && completionRaw >= 0
  return {
    promptTokens: promptTokensReported ? promptRaw : 0,
    completionTokens: completionTokensReported ? completionRaw : 0,
    promptTokensReported,
    completionTokensReported,
    totalDurationNs: Math.max(0, Number(result?.total_duration ?? 0) || 0),
  }
}

function addUsage(target, addition) {
  target.promptTokens += addition.promptTokens
  target.completionTokens += addition.completionTokens
  target.promptTokensReported ||= addition.promptTokensReported
  target.completionTokensReported ||= addition.completionTokensReported
  target.totalDurationNs += addition.totalDurationNs
}

function attachUsage(error, usage, model) {
  if (!error || typeof error !== 'object') return error
  error.promptTokens = usage.promptTokens
  error.completionTokens = usage.completionTokens
  error.promptTokensReported = usage.promptTokensReported
  error.completionTokensReported = usage.completionTokensReported
  error.totalDurationNs = usage.totalDurationNs
  error.model = model
  error.usage = {
    promptTokens: usage.promptTokens,
    completionTokens: usage.completionTokens,
    totalDurationNs: usage.totalDurationNs,
    model,
  }
  return error
}

function ollamaTimeoutError(timeoutMs, cause, detail = '') {
  const error = new Error(`Ollama ${Math.round(timeoutMs / 1_000)} saniyelik yanıt sınırını aştı.${detail ? ` ${detail}` : ''}`)
  error.code = 'OLLAMA_TIMEOUT'
  error.retryable = false
  error.timeoutMs = timeoutMs
  error.cause = cause
  return error
}

export class OllamaTeacher {
  constructor(getConfig, fetchImpl = fetch) {
    this.getConfig = getConfig
    this.fetchImpl = fetchImpl
  }

  async request(pathname, init = {}, { signal, retries } = {}) {
    const config = this.getConfig().ollama
    const attempts = (retries ?? config.retries) + 1
    let lastError
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      try {
        const response = await this.fetchImpl(`${config.baseUrl}${pathname}`, {
          ...init,
          headers: { 'Content-Type': 'application/json', Accept: 'application/json', ...init.headers },
          signal: combinedSignal(signal, config.timeoutMs),
        })
        if (!response.ok) {
          const raw = await response.text().catch(() => '')
          let detail = raw
          try { detail = JSON.parse(raw)?.error ?? raw } catch { /* bounded plain-text detail below */ }
          detail = String(detail ?? '').replace(/\s+/gu, ' ').trim().slice(0, 240)
          if (/timed?\s*out|timeout|deadline/iu.test(detail)) {
            throw ollamaTimeoutError(config.timeoutMs, undefined, detail)
          }
          const error = new Error(`Ollama HTTP ${response.status}${detail ? `: ${detail}` : ''}`)
          error.code = response.status >= 500 ? 'OLLAMA_TEMPORARY_ERROR' : 'OLLAMA_REQUEST_REJECTED'
          error.retryable = response.status >= 500 || response.status === 429
          error.status = response.status
          throw error
        }
        return await response.json()
      } catch (error) {
        if (!signal?.aborted && error?.name === 'TimeoutError') throw ollamaTimeoutError(config.timeoutMs, error)
        lastError = error
        if (signal?.aborted || error.name === 'AbortError' || error.code === 'OLLAMA_TIMEOUT') throw error
        if (attempt + 1 >= attempts || error.retryable === false || error.code === 'OLLAMA_REQUEST_REJECTED') throw error
        await wait(Math.min(2_000, 200 * 2 ** attempt), signal)
      }
    }
    throw lastError
  }

  async listModels(options = {}) {
    const result = await this.request('/api/tags', { method: 'GET', headers: {} }, options)
    const models = Array.isArray(result.models) ? result.models : []
    return models.map((model) => ({
      name: String(model.name ?? model.model ?? ''),
      digest: typeof model.digest === 'string' ? model.digest : null,
      size: Number(model.size ?? 0),
      modifiedAt: model.modified_at ?? null,
      details: model.details ?? null,
    })).filter((model) => model.name)
  }

  async check(options = {}) {
    const started = Date.now()
    try {
      const models = await this.listModels({ ...options, retries: 0 })
      return { available: true, latencyMs: Date.now() - started, models }
    } catch (error) {
      return { available: false, latencyMs: Date.now() - started, models: [], error: error.message }
    }
  }

  /**
   * Release the local model allocation before RAM-heavy deterministic stages.
   * This is a lifecycle operation, not a teacher inference, so it deliberately
   * consumes neither request nor token quota.
   */
  async unload(model, options = {}) {
    const config = this.getConfig().ollama
    const selectedModel = String(model || config.model || '').trim()
    if (!selectedModel) {
      const error = new Error('Boşaltılacak Ollama modeli seçilmedi.')
      error.code = 'OLLAMA_MODEL_REQUIRED'
      throw error
    }
    const result = await this.request('/api/generate', {
      method: 'POST',
      body: JSON.stringify({
        model: selectedModel,
        prompt: '',
        stream: false,
        keep_alive: 0,
      }),
    }, { ...options, retries: 0 })
    return {
      model: String(result?.model ?? selectedModel),
      unloaded: result?.done !== false,
      doneReason: typeof result?.done_reason === 'string' ? result.done_reason : null,
    }
  }

  async generateStructured({ model, messages, schema, signal, options = {}, retries }) {
    return this.#generateStructured({
      model,
      messages,
      schema,
      signal,
      sampling: { seed: 0, ...options, temperature: 0, num_ctx: TEACHER_CONTEXT_TOKEN_LIMIT },
      retries,
    })
  }

  /**
   * Server-internal diverse generation path. The browser cannot pass raw Ollama
   * options: campaigns expose only a seed and the service owns these bounds.
   * Deterministic critic/cross-check calls keep using generateStructured above.
   */
  async generateDiverseStructured({ model, messages, schema, signal, seed, maxCompletionTokens = OLLAMA_DIVERSE_DEFAULT_NUM_PREDICT }) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffff_ffff) {
      const error = new TypeError('Çeşitli üretim seed değeri 0–4294967295 arasında bir tam sayı olmalı.')
      error.code = 'INVALID_TEACHER_SEED'
      throw error
    }
    if (!Number.isInteger(maxCompletionTokens) || maxCompletionTokens < 64 || maxCompletionTokens > OLLAMA_DIVERSE_MAX_NUM_PREDICT) {
      const error = new TypeError(`Çeşitli üretim çıktı sınırı 64–${OLLAMA_DIVERSE_MAX_NUM_PREDICT} arasında bir tam sayı olmalı.`)
      error.code = 'INVALID_TEACHER_OUTPUT_LIMIT'
      throw error
    }
    return this.#generateStructured({
      model,
      messages,
      schema,
      signal,
      sampling: {
        seed,
        temperature: 0.65,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.12,
        num_predict: maxCompletionTokens,
        num_ctx: TEACHER_FEED_CONTEXT_TOKEN_LIMIT,
      },
      retries: 0,
    })
  }

  async #generateStructured({ model, messages, schema, signal, sampling, retries }) {
    const config = this.getConfig().ollama
    const selectedModel = String(model || config.model || '').trim()
    if (!selectedModel) {
      const error = new Error('Ollama modeli seçilmedi.')
      error.code = 'OLLAMA_MODEL_REQUIRED'
      throw error
    }
    const retryLimit = retries ?? config.retries
    const usage = {
      promptTokens: 0,
      completionTokens: 0,
      promptTokensReported: false,
      completionTokensReported: false,
      totalDurationNs: 0,
    }
    let resolvedModel = selectedModel
    let lastError
    for (let attempt = 0; attempt <= retryLimit; attempt += 1) {
      try {
        const result = await this.request('/api/chat', {
          method: 'POST',
          body: JSON.stringify({
            model: selectedModel,
            stream: false,
            messages: attempt === 0 ? messages : [...messages, { role: 'system', content: 'Önceki yanıt şemaya uymadı. Bu denemede yalnızca eksiksiz ve geçerli JSON döndür.' }],
            format: schema,
            options: sampling,
          }),
        }, { signal, retries: 0 })
        addUsage(usage, responseUsage(result))
        resolvedModel = result?.model ?? resolvedModel
        const raw = result?.message?.content
        if (typeof raw !== 'string') {
          const error = new Error('Ollama yapılandırılmış bir metin yanıtı döndürmedi.')
          error.code = 'OLLAMA_INVALID_RESPONSE'
          throw error
        }
        let value
        try { value = JSON.parse(raw) } catch {
          const error = new Error('Ollama yanıtı geçerli JSON değil.')
          error.code = 'OLLAMA_INVALID_JSON'
          throw error
        }
        assertSchema(value, schema, 'Ollama yanıtı')
        return {
          value,
          model: resolvedModel,
          ...usage,
        }
      } catch (error) {
        if (error?.code === 'OLLAMA_TIMEOUT' && usage.completionTokens === 0 && Number.isInteger(sampling?.num_predict)) {
          usage.completionTokens = sampling.num_predict
          error.usageEstimated = true
        }
        lastError = attachUsage(error, usage, resolvedModel)
        if (error?.usageEstimated === true) lastError.usage = { ...lastError.usage, usageEstimated: true }
        if (signal?.aborted || attempt >= retryLimit) throw lastError
      }
    }
    throw lastError
  }
}
