/**
 * ResourcePolicy owns one service-wide RAM envelope. Process isolation must
 * therefore meter parent + worker RSS, rather than accidentally granting each
 * process the full allowance.
 */
export function aggregateWorkerMemoryUsage(workerMemoryUsage, parentRssBytes) {
  const ownRssBytes = finiteBytes(workerMemoryUsage?.rss)
  return Object.freeze({
    ...workerMemoryUsage,
    rss: ownRssBytes + finiteBytes(parentRssBytes),
  })
}

function finiteBytes(value) {
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? number : 0
}
