import os from 'node:os'
import { mkdirSync, statfsSync } from 'node:fs'

export const MEBIBYTE = 1024 * 1024

function finiteNonNegative(value, fallback = 0) {
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? number : fallback
}

export function readPhysicalDiskBytes(dataDirectory) {
  mkdirSync(dataDirectory, { recursive: true, mode: 0o700 })
  const stats = statfsSync(dataDirectory)
  const blockSize = finiteNonNegative(stats.bsize || stats.frsize, 1)
  return {
    totalBytes: finiteNonNegative(stats.blocks) * blockSize,
    freeBytes: finiteNonNegative(stats.bavail) * blockSize,
  }
}

/**
 * A small, payload-free resource sample. This function deliberately never
 * walks the artifact store: the status page and the queue may call it often.
 */
export function sampleHostResources({ dataDirectory, memoryUsage = process.memoryUsage(), diskReader = readPhysicalDiskBytes } = {}) {
  const disk = diskReader(dataDirectory)
  return Object.freeze({
    sampledAt: new Date().toISOString(),
    rssBytes: finiteNonNegative(memoryUsage.rss),
    heapUsedBytes: finiteNonNegative(memoryUsage.heapUsed),
    externalBytes: finiteNonNegative(memoryUsage.external),
    hostTotalMemoryBytes: finiteNonNegative(os.totalmem()),
    hostFreeMemoryBytes: finiteNonNegative(os.freemem()),
    diskTotalBytes: disk.totalBytes,
    diskFreeBytes: disk.freeBytes,
  })
}

export function evaluateResourceState(config, sample, { logicalStorageBytes = 0, admission = false } = {}) {
  const policy = config.resourcePolicy
  const ramLimitBytes = finiteNonNegative(config.budgets.ramMb) * MEBIBYTE
  const logicalDiskLimitBytes = finiteNonNegative(config.budgets.diskMb) * MEBIBYTE
  const ramRatio = ramLimitBytes > 0 ? sample.rssBytes / ramLimitBytes : Number.POSITIVE_INFINITY
  const logicalDiskRatio = logicalDiskLimitBytes > 0
    ? finiteNonNegative(logicalStorageBytes) / logicalDiskLimitBytes
    : Number.POSITIVE_INFINITY
  const usableDiskFreeBytes = Math.max(0, sample.diskFreeBytes - policy.walReserveMb * MEBIBYTE)
  const belowHostReserve = sample.hostFreeMemoryBytes < policy.minHostFreeMb * MEBIBYTE
  const belowDiskStart = usableDiskFreeBytes < policy.minDiskFreeStartMb * MEBIBYTE
  const belowDiskWarning = usableDiskFreeBytes < policy.minDiskFreeWarnMb * MEBIBYTE
  const hardRam = ramRatio >= policy.ramHardRatio
  const spillRam = ramRatio >= policy.ramSpillRatio
  const softRam = ramRatio >= policy.ramSoftRatio
  // On macOS, os.freemem() excludes readily reclaimable file cache. Treat a
  // low host-free sample as pressure only when this process is already above
  // its own soft envelope; RSS remains the deterministic hard authority.
  const hostPressure = belowHostReserve && softRam
  const logicalDiskHard = logicalDiskRatio >= 1

  let memoryTier = 'normal'
  if (hardRam) memoryTier = 'hard'
  else if (spillRam || hostPressure) memoryTier = 'spill'
  else if (softRam) memoryTier = 'soft'

  let diskTier = 'normal'
  if (logicalDiskHard || usableDiskFreeBytes <= 0) diskTier = 'hard'
  else if (belowDiskStart) diskTier = 'admission-blocked'
  else if (belowDiskWarning || logicalDiskRatio >= 0.85) diskTier = 'warning'

  const reasonCodes = []
  if (hardRam) reasonCodes.push('PROCESS_RAM_HARD_LIMIT')
  if (hostPressure) reasonCodes.push('HOST_MEMORY_RESERVE_LOW')
  if (logicalDiskHard) reasonCodes.push('LOGICAL_DISK_HARD_LIMIT')
  if (belowDiskStart) reasonCodes.push('PHYSICAL_DISK_START_RESERVE_LOW')
  else if (belowDiskWarning) reasonCodes.push('PHYSICAL_DISK_WARNING_RESERVE_LOW')

  return Object.freeze({
    memoryTier,
    diskTier,
    ramRatio,
    logicalDiskRatio,
    usableDiskFreeBytes,
    shouldThrottle: softRam || belowDiskWarning || hostPressure,
    shouldSpill: spillRam || hostPressure,
    // Running work may still write its recovery checkpoint while the start
    // reserve is low. Only a true hard limit pauses an existing job.
    mustPause: hardRam || logicalDiskHard || usableDiskFreeBytes <= 0,
    canStart: !hardRam && !hostPressure && !logicalDiskHard && !belowDiskStart,
    allowed: admission
      ? !hardRam && !hostPressure && !logicalDiskHard && !belowDiskStart
      : !hardRam && !logicalDiskHard && usableDiskFreeBytes > 0,
    reasonCodes,
  })
}

export function resourcePolicyError(state, { admission = false } = {}) {
  const error = new Error(admission
    ? 'Yeni eğitim bloğu güvenli RAM/SSD rezervi sağlanana kadar başlatılmadı.'
    : 'Eğitim bloğu güvenli RAM/SSD sınırında geri alınabilir biçimde duraklatıldı.')
  error.code = state.reasonCodes.includes('PROCESS_RAM_HARD_LIMIT') ||
    state.reasonCodes.includes('HOST_MEMORY_RESERVE_LOW')
    ? 'RAM_BUDGET_EXCEEDED'
    : 'DISK_RESOURCE_RESERVE_EXCEEDED'
  error.resourceState = state
  error.retryable = true
  return error
}

export class ResourceGovernor {
  constructor({ store, getConfig = () => store.getConfig(), sampler = sampleHostResources, clock = () => Date.now() }) {
    this.store = store
    this.getConfig = getConfig
    this.sampler = sampler
    this.clock = clock
    this.lastSampleAt = -Infinity
    this.last = null
  }

  inspect({ admission = false, force = false } = {}) {
    const config = this.getConfig()
    if (!config.resourcePolicy?.enabled) {
      return { sample: null, state: Object.freeze({
        memoryTier: 'disabled', diskTier: 'disabled', ramRatio: 0, logicalDiskRatio: 0,
        usableDiskFreeBytes: Number.POSITIVE_INFINITY, shouldThrottle: false,
        shouldSpill: false, mustPause: false, canStart: true, allowed: true,
        reasonCodes: [],
      }) }
    }
    const currentTime = this.clock()
    if (force || !this.last || currentTime - this.lastSampleAt >= config.resourcePolicy.sampleIntervalMs) {
      const sample = this.sampler({ dataDirectory: config.dataDirectory })
      this.last = {
        sample,
        state: evaluateResourceState(config, sample, {
          logicalStorageBytes: this.store.approximateStorageBytes(),
          admission,
        }),
      }
      this.lastSampleAt = currentTime
    } else if (admission) {
      this.last = {
        sample: this.last.sample,
        state: evaluateResourceState(config, this.last.sample, {
          logicalStorageBytes: this.store.approximateStorageBytes(),
          admission: true,
        }),
      }
    }
    return this.last
  }

  assertSafe(options = {}) {
    const result = this.inspect(options)
    if (!result.state.allowed) throw resourcePolicyError(result.state, options)
    return result
  }
}
