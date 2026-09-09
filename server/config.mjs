import path from 'node:path'

export const CONFIG_VERSION = 6

export function defaultConfig(dataDirectory = path.resolve('server-data')) {
  return {
    version: CONFIG_VERSION,
    dataDirectory,
    ollama: {
      enabled: true,
      baseUrl: 'http://127.0.0.1:11434',
      model: '',
      timeoutMs: 60_000,
      retries: 2,
    },
    automation: {
      autoRun: false,
      level: 'proposal',
      batchSize: 50,
      concurrency: 1,
      leaseMs: 30_000,
      pollMs: 350,
    },
    thought: {
      enabled: true,
      maximumInputCharacters: 16_000,
      maximumTicks: 512,
      traceChunkSize: 64,
      maximumStoredTicks: 512,
      maximumEpisodeBytes: 32 * 1024 * 1024,
      maximumHeadBytes: 64 * 1024 * 1024,
      ollamaCrossCheck: false,
      // Ollama is an auxiliary critic, never the authority for a model write.
      // This value is deliberately fixed by validation below.
      ollamaTrustWeight: 0.25,
    },
    conceptMacros: {
      enabled: true,
      // New concepts are observational until independent shadow evidence
      // proves that the expanded path and the shortcut are equivalent.
      mode: 'proposal-shadow',
      minimumLength: 2,
      maximumLength: 8,
      maximumObservationsPerCycle: 2_048,
      maximumCandidates: 2_048,
      maximumActiveMacros: 512,
      proposalMinimumOccurrences: 5,
      proposalMinimumSourceGroups: 2,
      activationMinimumOccurrences: 10,
      activationMinimumSourceGroups: 3,
      minimumCompletionProbability: 0.80,
      minimumCohesion: 0.82,
      minimumShadowFidelity: 0.95,
      maximumFalseActivationRate: 0.05,
      maximumNestedDepth: 3,
      thoughtSamplesPerCycle: 4,
      thoughtTicksPerSample: 16,
    },
    // Adaptive plastic regions are a separate, opt-in overlay. The legacy
    // 16x64 I / 48x64 R representation remains authoritative until a new
    // representation lineage has been rebuilt and independently validated.
    plasticRegions: {
      enabled: true,
      mode: 'off',
      representationProfile: 'legacy',
      thoughtSamplesPerCycle: 4,
      thoughtTicksPerSample: 16,
      maximumObservationsPerBatch: 2_048,
      maximumObservationBatchesPerCycle: 64,
      maximumCumulativeObservations: 8_192,
      maximumEvidenceHeadsPerCycle: 64,
      maximumRegions: 56,
      maximumMembershipsPerPage: 3,
      maximumActiveGates: 4,
      compatibilityBlend: 0.25,
      minimumTrace: 0.02,
      releaseMinimumCases: 30,
      releaseMinimumSourceGroups: 3,
      releaseMinimumHoldoutGain: 0.01,
      releaseMaximumShamGain: 0.03,
      releaseMaximumRegressionLoss: 0.01,
      maximumArtifactBytes: 4 * 1024 * 1024,
      // NEW interpretation companion. Disabled by default because the frozen
      // 288-case pack is intentionally heavier than ordinary train tracing.
      semanticProbesEnabled: false,
    },
    budgets: {
      cpuPercent: 75,
      ramMb: 4_096,
      maxRunMinutes: 120,
      // This is the logical training-store ceiling. Physical free-space
      // admission is enforced independently by resourcePolicy so a corrupt or
      // unexpectedly large run cannot fill the host volume.
      diskMb: 4_096,
      maxRequestBytes: 2_000_000,
      maxSourceBytes: 2_000_000,
      maxBatchBytes: 20_000_000,
      maxDailyJobs: 50,
      maxDailyTeacherRequests: 100,
      maxDailyTeacherTokens: 200_000,
      maxEvents: 2_000,
      maxJobAttempts: 2,
    },
    resourcePolicy: {
      enabled: true,
      sampleIntervalMs: 1_000,
      ramSoftRatio: 0.70,
      ramSpillRatio: 0.85,
      ramHardRatio: 1.00,
      minHostFreeMb: 256,
      minDiskFreeWarnMb: 20_480,
      minDiskFreeStartMb: 15_360,
      walReserveMb: 512,
      artifactCompressionMinBytes: 64 * 1024,
      progressFlushMs: 2_000,
      neuronCacheMb: 384,
      neuronPageBytes: 2 * 1024 * 1024,
    },
    sourcePolicy: {
      watchEnabled: false,
      scanIntervalMs: 60_000,
      maxScanFiles: 5_000,
      watchedDirectories: [],
      allowedHttpOrigins: [],
      allowedExtensions: ['.txt', '.md', '.json', '.jsonl'],
    },
    evaluation: {
      minOverallScore: 0.55,
      minSyntaxContrastAccuracy: 0.5,
      maxRegressionRate: 0.03,
      maxMemorizationGap: 0.35,
      baselineTolerance: 0.015,
    },
  }
}

const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value)
const finite = (value) => typeof value === 'number' && Number.isFinite(value)

function boundedNumber(value, name, min, max, integer = false) {
  if (!finite(value) || value < min || value > max || (integer && !Number.isInteger(value))) {
    throw new TypeError(`${name}, ${min} ile ${max} arasında${integer ? ' bir tam sayı' : ''} olmalı.`)
  }
  return value
}

function stringArray(value, name, mapper = (item) => item) {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new TypeError(`${name} bir metin dizisi olmalı.`)
  }
  return [...new Set(value.map((item) => mapper(item.trim())).filter(Boolean))]
}

export function mergeAndValidateConfig(current, patch) {
  if (!isRecord(patch)) throw new TypeError('Yapılandırma gövdesi bir nesne olmalı.')
  const next = structuredClone(current)

  if (patch.ollama !== undefined) {
    if (!isRecord(patch.ollama)) throw new TypeError('ollama bir nesne olmalı.')
    if (patch.ollama.enabled !== undefined) next.ollama.enabled = Boolean(patch.ollama.enabled)
    if (patch.ollama.model !== undefined) {
      if (typeof patch.ollama.model !== 'string' || patch.ollama.model.length > 160) throw new TypeError('Geçersiz Ollama model adı.')
      next.ollama.model = patch.ollama.model.trim()
    }
    if (patch.ollama.baseUrl !== undefined) {
      const url = new URL(String(patch.ollama.baseUrl))
      const loopback = ['127.0.0.1', 'localhost', '::1', '[::1]'].includes(url.hostname)
      if (url.protocol !== 'http:' || !loopback || url.username || url.password) {
        throw new TypeError('Ollama yalnızca yerel, kimlik bilgisiz bir HTTP adresi kullanabilir.')
      }
      next.ollama.baseUrl = url.origin
    }
    if (patch.ollama.timeoutMs !== undefined) next.ollama.timeoutMs = boundedNumber(patch.ollama.timeoutMs, 'ollama.timeoutMs', 1_000, 300_000, true)
    if (patch.ollama.retries !== undefined) next.ollama.retries = boundedNumber(patch.ollama.retries, 'ollama.retries', 0, 5, true)
  }

  if (patch.automation !== undefined) {
    if (!isRecord(patch.automation)) throw new TypeError('automation bir nesne olmalı.')
    if (patch.automation.autoRun !== undefined) next.automation.autoRun = Boolean(patch.automation.autoRun)
    if (patch.automation.level !== undefined) {
      if (!['proposal', 'controlled'].includes(patch.automation.level)) throw new TypeError('automation.level proposal veya controlled olmalı.')
      next.automation.level = patch.automation.level
    }
    if (patch.automation.batchSize !== undefined) next.automation.batchSize = boundedNumber(patch.automation.batchSize, 'automation.batchSize', 1, 10_000, true)
    if (patch.automation.concurrency !== undefined) next.automation.concurrency = boundedNumber(patch.automation.concurrency, 'automation.concurrency', 1, 1, true)
    if (patch.automation.leaseMs !== undefined) next.automation.leaseMs = boundedNumber(patch.automation.leaseMs, 'automation.leaseMs', 5_000, 300_000, true)
    if (patch.automation.pollMs !== undefined) next.automation.pollMs = boundedNumber(patch.automation.pollMs, 'automation.pollMs', 100, 10_000, true)
  }

  if (patch.thought !== undefined) {
    if (!isRecord(patch.thought)) throw new TypeError('thought bir nesne olmalı.')
    if (patch.thought.enabled !== undefined) next.thought.enabled = Boolean(patch.thought.enabled)
    if (patch.thought.ollamaCrossCheck !== undefined) next.thought.ollamaCrossCheck = Boolean(patch.thought.ollamaCrossCheck)
    if (patch.thought.maximumInputCharacters !== undefined) {
      next.thought.maximumInputCharacters = boundedNumber(
        patch.thought.maximumInputCharacters,
        'thought.maximumInputCharacters',
        1,
        100_000,
        true,
      )
    }
    if (patch.thought.maximumTicks !== undefined) {
      next.thought.maximumTicks = boundedNumber(patch.thought.maximumTicks, 'thought.maximumTicks', 1, 512, true)
    }
    if (patch.thought.traceChunkSize !== undefined) {
      if (patch.thought.traceChunkSize !== 64) {
        throw new TypeError('thought.traceChunkSize v1 replay sözleşmesi gereği 64 olarak sabittir.')
      }
      next.thought.traceChunkSize = 64
    }
    if (patch.thought.maximumStoredTicks !== undefined) {
      next.thought.maximumStoredTicks = boundedNumber(patch.thought.maximumStoredTicks, 'thought.maximumStoredTicks', 64, 512, true)
    }
    if (patch.thought.maximumEpisodeBytes !== undefined) {
      next.thought.maximumEpisodeBytes = boundedNumber(
        patch.thought.maximumEpisodeBytes,
        'thought.maximumEpisodeBytes',
        64 * 1024,
        64 * 1024 * 1024,
        true,
      )
    }
    if (patch.thought.maximumHeadBytes !== undefined) {
      next.thought.maximumHeadBytes = boundedNumber(
        patch.thought.maximumHeadBytes,
        'thought.maximumHeadBytes',
        256 * 1024,
        256 * 1024 * 1024,
        true,
      )
    }
    if (patch.thought.ollamaTrustWeight !== undefined && patch.thought.ollamaTrustWeight !== 0.25) {
      throw new TypeError('thought.ollamaTrustWeight güvenlik gereği 0.25 olarak sabittir.')
    }
    next.thought.ollamaTrustWeight = 0.25
    if (next.thought.maximumStoredTicks < 64) {
      throw new TypeError('thought.maximumStoredTicks v1 iz parçası boyutu nedeniyle en az 64 olmalı.')
    }
  }

  if (patch.conceptMacros !== undefined) {
    if (!isRecord(patch.conceptMacros)) throw new TypeError('conceptMacros bir nesne olmalı.')
    if (patch.conceptMacros.enabled !== undefined) next.conceptMacros.enabled = Boolean(patch.conceptMacros.enabled)
    if (patch.conceptMacros.mode !== undefined) {
      if (!['off', 'observe', 'proposal-shadow', 'controlled'].includes(patch.conceptMacros.mode)) {
        throw new TypeError('conceptMacros.mode off, observe, proposal-shadow veya controlled olmalı.')
      }
      next.conceptMacros.mode = patch.conceptMacros.mode
    }
    const integerLimits = {
      minimumLength: [2, 8],
      maximumLength: [2, 8],
      maximumObservationsPerCycle: [1, 100_000],
      maximumCandidates: [16, 100_000],
      maximumActiveMacros: [1, 10_000],
      proposalMinimumOccurrences: [2, 10_000],
      proposalMinimumSourceGroups: [1, 1_000],
      activationMinimumOccurrences: [2, 100_000],
      activationMinimumSourceGroups: [2, 1_000],
      // Concept-macro V1 validates and expands a DAG of at most three layers.
      // Reject misleading values instead of silently clamping them later.
      maximumNestedDepth: [1, 3],
      thoughtSamplesPerCycle: [0, 64],
      thoughtTicksPerSample: [1, 64],
    }
    for (const [key, [minimum, maximum]] of Object.entries(integerLimits)) {
      if (patch.conceptMacros[key] !== undefined) {
        next.conceptMacros[key] = boundedNumber(
          patch.conceptMacros[key],
          `conceptMacros.${key}`,
          minimum,
          maximum,
          true,
        )
      }
    }
    for (const key of [
      'minimumCompletionProbability', 'minimumCohesion', 'minimumShadowFidelity', 'maximumFalseActivationRate',
    ]) {
      if (patch.conceptMacros[key] !== undefined) {
        next.conceptMacros[key] = boundedNumber(patch.conceptMacros[key], `conceptMacros.${key}`, 0, 1)
      }
    }
    if (next.conceptMacros.minimumLength > next.conceptMacros.maximumLength) {
      throw new TypeError('conceptMacros minimumLength, maximumLength değerini aşamaz.')
    }
    if (next.conceptMacros.activationMinimumOccurrences < next.conceptMacros.proposalMinimumOccurrences) {
      throw new TypeError('Aktif makro gözlem eşiği öneri eşiğinden düşük olamaz.')
    }
    if (next.conceptMacros.activationMinimumSourceGroups < next.conceptMacros.proposalMinimumSourceGroups) {
      throw new TypeError('Aktif makro kaynak eşiği öneri eşiğinden düşük olamaz.')
    }
  }

  if (patch.plasticRegions !== undefined) {
    if (!isRecord(patch.plasticRegions)) throw new TypeError('plasticRegions bir nesne olmalı.')
    if (patch.plasticRegions.enabled !== undefined) next.plasticRegions.enabled = Boolean(patch.plasticRegions.enabled)
    if (patch.plasticRegions.semanticProbesEnabled !== undefined) {
      next.plasticRegions.semanticProbesEnabled = Boolean(patch.plasticRegions.semanticProbesEnabled)
    }
    if (patch.plasticRegions.mode !== undefined) {
      if (!['off', 'shadow', 'proposal', 'active'].includes(patch.plasticRegions.mode)) {
        throw new TypeError('plasticRegions.mode off, shadow, proposal veya active olmalı.')
      }
      next.plasticRegions.mode = patch.plasticRegions.mode
    }
    if (patch.plasticRegions.representationProfile !== undefined) {
      if (!['legacy', 'adaptive-eighth-id-v1'].includes(patch.plasticRegions.representationProfile)) {
        throw new TypeError('plasticRegions.representationProfile legacy veya adaptive-eighth-id-v1 olmalı.')
      }
      next.plasticRegions.representationProfile = patch.plasticRegions.representationProfile
    }
    const integerLimits = {
      thoughtSamplesPerCycle: [1, 64],
      thoughtTicksPerSample: [1, 512],
      maximumObservationsPerBatch: [56, 100_000],
      maximumObservationBatchesPerCycle: [1, 1_024],
      maximumCumulativeObservations: [56, 65_536],
      maximumEvidenceHeadsPerCycle: [1, 256],
      maximumRegions: [1, 56],
      maximumMembershipsPerPage: [1, 3],
      maximumActiveGates: [1, 8],
      releaseMinimumCases: [1, 1_000_000],
      releaseMinimumSourceGroups: [2, 100_000],
      maximumArtifactBytes: [64 * 1024, 64 * 1024 * 1024],
    }
    for (const [key, [minimum, maximum]] of Object.entries(integerLimits)) {
      if (patch.plasticRegions[key] !== undefined) {
        next.plasticRegions[key] = boundedNumber(
          patch.plasticRegions[key],
          `plasticRegions.${key}`,
          minimum,
          maximum,
          true,
        )
      }
    }
    if (next.plasticRegions.maximumActiveGates > next.plasticRegions.maximumRegions) {
      throw new TypeError('plasticRegions.maximumActiveGates, maximumRegions değerini aşamaz.')
    }
    if (patch.plasticRegions.compatibilityBlend !== undefined) {
      next.plasticRegions.compatibilityBlend = boundedNumber(
        patch.plasticRegions.compatibilityBlend,
        'plasticRegions.compatibilityBlend',
        Number.EPSILON,
        0.5,
      )
    }
    if (patch.plasticRegions.minimumTrace !== undefined) {
      next.plasticRegions.minimumTrace = boundedNumber(
        patch.plasticRegions.minimumTrace,
        'plasticRegions.minimumTrace',
        0,
        1,
      )
    }
    for (const key of [
      'releaseMinimumHoldoutGain',
      'releaseMaximumShamGain',
      'releaseMaximumRegressionLoss',
    ]) {
      if (patch.plasticRegions[key] !== undefined) {
        next.plasticRegions[key] = boundedNumber(patch.plasticRegions[key], `plasticRegions.${key}`, 0, 1)
      }
    }
    if (next.plasticRegions.mode === 'active' && next.plasticRegions.minimumTrace <= 0) {
      throw new TypeError('Active plastik bölge kipinde minimumTrace sıfırdan büyük olmalı.')
    }
    // The adaptive runtime is strictly opt-in. No non-off mode may silently
    // reinterpret legacy coordinates as the canonical 8/56 representation.
    if (
      next.plasticRegions.representationProfile === 'legacy' &&
      next.plasticRegions.mode !== 'off'
    ) {
      throw new TypeError('Legacy temsil profili yalnız off plastik bölge kipinde kullanılabilir.')
    }
  }

  if (patch.budgets !== undefined) {
    if (!isRecord(patch.budgets)) throw new TypeError('budgets bir nesne olmalı.')
    const limits = {
      cpuPercent: [10, 95],
      // The default remains 4 GiB. A 16 GiB workstation may explicitly grant
      // an isolated Ollama-free benchmark up to 8 GiB after observing its
      // checkpointed peak; the service-wide governor still enforces the
      // selected 70/85/100% tiers.
      ramMb: [512, 8_192],
      maxRunMinutes: [5, 10_080],
      diskMb: [512, 4_096],
      maxRequestBytes: [1_024, 20_000_000],
      maxSourceBytes: [1_024, 50_000_000],
      maxBatchBytes: [1_024, 500_000_000],
      maxDailyJobs: [1, 10_000],
      maxDailyTeacherRequests: [0, 100_000],
      maxDailyTeacherTokens: [0, 100_000_000],
      maxEvents: [100, 100_000],
      maxJobAttempts: [1, 10],
    }
    for (const [key, range] of Object.entries(limits)) {
      if (patch.budgets[key] !== undefined) next.budgets[key] = boundedNumber(patch.budgets[key], `budgets.${key}`, range[0], range[1], true)
    }
  }

  if (patch.resourcePolicy !== undefined) {
    if (!isRecord(patch.resourcePolicy)) throw new TypeError('resourcePolicy bir nesne olmalı.')
    if (patch.resourcePolicy.enabled !== undefined) next.resourcePolicy.enabled = Boolean(patch.resourcePolicy.enabled)
    const limits = {
      sampleIntervalMs: [250, 60_000, true],
      ramSoftRatio: [0.25, 0.90, false],
      ramSpillRatio: [0.50, 0.98, false],
      ramHardRatio: [0.75, 1.00, false],
      minHostFreeMb: [256, 131_072, true],
      minDiskFreeWarnMb: [1_024, 1_048_576, true],
      minDiskFreeStartMb: [1_024, 1_048_576, true],
      walReserveMb: [64, 16_384, true],
      artifactCompressionMinBytes: [4_096, 16 * 1024 * 1024, true],
      progressFlushMs: [250, 60_000, true],
      neuronCacheMb: [32, 16_384, true],
      neuronPageBytes: [64 * 1024, 16 * 1024 * 1024, true],
    }
    for (const [key, [minimum, maximum, integer]] of Object.entries(limits)) {
      if (patch.resourcePolicy[key] !== undefined) {
        next.resourcePolicy[key] = boundedNumber(
          patch.resourcePolicy[key],
          `resourcePolicy.${key}`,
          minimum,
          maximum,
          integer,
        )
      }
    }
    if (!(next.resourcePolicy.ramSoftRatio < next.resourcePolicy.ramSpillRatio &&
      next.resourcePolicy.ramSpillRatio < next.resourcePolicy.ramHardRatio)) {
      throw new TypeError('RAM eşikleri soft < spill < hard sırasında olmalı.')
    }
    if (next.resourcePolicy.minDiskFreeWarnMb < next.resourcePolicy.minDiskFreeStartMb) {
      throw new TypeError('Disk uyarı eşiği, yeni iş başlatma eşiğinden düşük olamaz.')
    }
  }

  if (patch.sourcePolicy !== undefined) {
    if (!isRecord(patch.sourcePolicy)) throw new TypeError('sourcePolicy bir nesne olmalı.')
    if (patch.sourcePolicy.watchEnabled !== undefined) next.sourcePolicy.watchEnabled = Boolean(patch.sourcePolicy.watchEnabled)
    if (patch.sourcePolicy.scanIntervalMs !== undefined) next.sourcePolicy.scanIntervalMs = boundedNumber(patch.sourcePolicy.scanIntervalMs, 'sourcePolicy.scanIntervalMs', 5_000, 3_600_000, true)
    if (patch.sourcePolicy.maxScanFiles !== undefined) next.sourcePolicy.maxScanFiles = boundedNumber(patch.sourcePolicy.maxScanFiles, 'sourcePolicy.maxScanFiles', 1, 100_000, true)
    if (patch.sourcePolicy.watchedDirectories !== undefined) {
      next.sourcePolicy.watchedDirectories = stringArray(patch.sourcePolicy.watchedDirectories, 'sourcePolicy.watchedDirectories', (item) => path.resolve(item))
    }
    if (patch.sourcePolicy.allowedHttpOrigins !== undefined) {
      next.sourcePolicy.allowedHttpOrigins = stringArray(patch.sourcePolicy.allowedHttpOrigins, 'sourcePolicy.allowedHttpOrigins', (item) => {
        const url = new URL(item)
        if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
          throw new TypeError('HTTP izin listesine yalnızca origin biçiminde adres eklenebilir.')
        }
        return url.origin
      })
    }
    if (patch.sourcePolicy.allowedExtensions !== undefined) {
      next.sourcePolicy.allowedExtensions = stringArray(patch.sourcePolicy.allowedExtensions, 'sourcePolicy.allowedExtensions', (item) => item.startsWith('.') ? item.toLowerCase() : `.${item.toLowerCase()}`)
    }
  }

  if (patch.evaluation !== undefined) {
    if (!isRecord(patch.evaluation)) throw new TypeError('evaluation bir nesne olmalı.')
    for (const key of ['minOverallScore', 'minSyntaxContrastAccuracy', 'maxRegressionRate', 'maxMemorizationGap', 'baselineTolerance']) {
      if (patch.evaluation[key] !== undefined) next.evaluation[key] = boundedNumber(patch.evaluation[key], `evaluation.${key}`, 0, 1)
    }
  }

  next.version = CONFIG_VERSION
  next.dataDirectory = current.dataDirectory
  return next
}

export function publicConfig(config) {
  return structuredClone(config)
}
