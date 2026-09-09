import { opendir } from 'node:fs/promises'
import path from 'node:path'
import { readAllowedFile } from './source-policy.mjs'

async function collectFiles(directory, extensions, output, maxFiles, depth = 0) {
  if (depth > 12 || output.length >= maxFiles) return
  const handle = await opendir(directory)
  for await (const entry of handle) {
    if (output.length >= maxFiles) break
    const entryPath = path.join(directory, entry.name)
    if (entry.isSymbolicLink()) continue
    if (entry.isDirectory()) await collectFiles(entryPath, extensions, output, maxFiles, depth + 1)
    else if (entry.isFile() && extensions.includes(path.extname(entry.name).toLowerCase())) output.push(entryPath)
  }
}

export async function scanWatchedSources(store, { signal } = {}) {
  const config = store.getConfig()
  const files = []
  for (const directory of config.sourcePolicy.watchedDirectories) {
    if (signal?.aborted) throw signal.reason
    try { await collectFiles(directory, config.sourcePolicy.allowedExtensions, files, config.sourcePolicy.maxScanFiles) } catch (error) {
      error.code ??= 'WATCH_DIRECTORY_FAILED'
      throw error
    }
  }
  const results = []
  for (const file of files) {
    if (signal?.aborted) throw signal.reason
    try { results.push(store.createSource(await readAllowedFile(file, config))) } catch (error) {
      results.push({ file, error: { code: error.code ?? 'SOURCE_SCAN_FAILED', message: error.message } })
    }
  }
  return {
    scanned: files.length,
    inserted: results.filter((item) => item.source && !item.duplicate).length,
    duplicates: results.filter((item) => item.duplicate).length,
    failed: results.filter((item) => item.error).length,
    errors: results.filter((item) => item.error).slice(0, 50),
  }
}

export class SourceWatcher {
  constructor(store, onResult = () => {}) {
    this.store = store
    this.onResult = onResult
    this.timer = null
    this.running = false
    this.scanning = false
    this.activePromise = null
  }
  start() {
    if (this.running) return
    this.running = true
    this.schedule(0)
  }
  async stop() { this.running = false; clearTimeout(this.timer); this.timer = null; await this.activePromise }
  schedule(delay) {
    if (!this.running || this.timer) return
    this.timer = setTimeout(() => { this.timer = null; void this.tick() }, delay)
    this.timer.unref?.()
  }
  async tick() {
    const config = this.store.getConfig()
    if (config.sourcePolicy.watchEnabled && config.sourcePolicy.watchedDirectories.length && !this.scanning) {
      this.scanning = true
      this.activePromise = scanWatchedSources(this.store)
      try { this.onResult(await this.activePromise) } catch (error) { this.onResult({ error: { code: error.code ?? 'WATCH_SCAN_FAILED', message: error.message } }) } finally { this.scanning = false; this.activePromise = null }
    }
    this.schedule(this.store.getConfig().sourcePolicy.scanIntervalMs)
  }
}
