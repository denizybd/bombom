import { mkdtemp, rm } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'

export async function temporaryDirectory(t, prefix = 'bombom-training-') {
  const directory = await mkdtemp(path.join(os.tmpdir(), prefix))
  t.after(() => rm(directory, { recursive: true, force: true, maxRetries: 5, retryDelay: 25 }))
  return directory
}

export async function listenService(t, service) {
  const address = await service.listen({ port: 0, host: '127.0.0.1' })
  t.after(() => service.close())
  return `http://127.0.0.1:${address.port}`
}

export async function jsonFetch(base, pathname, options = {}) {
  const response = await fetch(`${base}${pathname}`, {
    ...options,
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  })
  const body = await response.json()
  return { response, body }
}

export async function waitFor(predicate, { timeoutMs = 4_000, intervalMs = 20 } = {}) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const value = await predicate()
    if (value) return value
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
  throw new Error('Koşul zamanında gerçekleşmedi.')
}
