import {
  ADAPTIVE_REGION_GEOMETRY,
  PAGE_BEHAVIOR_DIMENSION,
  createPlasticRegionAtlas,
  validateAdaptiveRegionAtlas,
} from '../src/adaptive-regions/index.ts'
import {
  normalizePlasticRegionMode,
  readPlasticRegionRelease,
  readPlasticRegionSnapshot,
} from '../runtime/plastic-region-cache.mjs'

export const LEGACY_REGION_PROFILE = 'legacy-quarter-id-v2'
export const ADAPTIVE_REGION_PROFILE = 'adaptive-eighth-id-v1'

const MODE_RANK = Object.freeze({ off: 0, shadow: 1, proposal: 2, active: 3 })
const MODES = Object.freeze(['off', 'shadow', 'proposal', 'active'])

function runtimeError(code, message, details = undefined) {
  const error = new Error(message)
  error.code = code
  error.status = 409
  if (details !== undefined) error.details = details
  return error
}

function finiteUnit(value, fallback, label, { strictlyPositive = false, maximum = 1 } = {}) {
  const resolved = value ?? fallback
  const validLowerBound = strictlyPositive ? resolved > 0 : resolved >= 0
  if (!Number.isFinite(resolved) || !validLowerBound || resolved > maximum) {
    throw runtimeError(
      'INVALID_PLASTIC_REGION_POLICY',
      `${label} ${strictlyPositive ? '(0' : '[0'},${maximum}${strictlyPositive ? ']' : ']'} aralığında olmalı.`,
    )
  }
  return resolved
}

function boundedInteger(value, fallback, label, minimum, maximum) {
  const resolved = value ?? fallback
  if (!Number.isSafeInteger(resolved) || resolved < minimum || resolved > maximum) {
    throw runtimeError('INVALID_PLASTIC_REGION_POLICY', `${label} ${minimum}..${maximum} aralığında tam sayı olmalı.`)
  }
  return resolved
}

/**
 * Configuration aliases are deliberately kept at the service boundary. The
 * physical encoder and every immutable artifact use the canonical TextLab
 * profile ids below.
 */
export function canonicalPlasticRegionProfile(value) {
  if (value === 'legacy' || value === LEGACY_REGION_PROFILE || value === undefined || value === null) {
    return LEGACY_REGION_PROFILE
  }
  if (value === 'adaptive-8x56' || value === ADAPTIVE_REGION_PROFILE) return ADAPTIVE_REGION_PROFILE
  throw runtimeError('INVALID_PLASTIC_REGION_POLICY', 'Bilinmeyen plastik bölge temsil profili.', { value })
}

export function plasticRegionProfileForLayout(layout) {
  if (
    layout?.side === 64 && layout?.cellCount === 4096 &&
    layout?.identityLength === 512 && layout?.relationStart === 512 && layout?.relationLength === 3584
  ) return ADAPTIVE_REGION_PROFILE
  if (
    layout?.side === 64 && layout?.cellCount === 4096 &&
    layout?.identityLength === 1024 && layout?.relationStart === 1024 && layout?.relationLength === 3072
  ) return LEGACY_REGION_PROFILE
  return null
}

function restrictedMode(requested, configured) {
  return MODES[Math.min(MODE_RANK[requested], MODE_RANK[configured])]
}

function offResult(requestedMode, configuredMode, reason) {
  return Object.freeze({
    requestedMode,
    configuredMode,
    mode: 'off',
    reason,
    atlas: null,
    atlasArtifactId: null,
    atlasContentHash: null,
    ephemeralAtlas: false,
    adaptiveRegion: undefined,
  })
}

/**
 * Resolves the most restrictive request/config/snapshot mode. This is the
 * only server adapter allowed to turn the new companion mechanism on. It does
 * not mutate a snapshot, reinterpret a legacy vector, or activate a proposal.
 */
export function resolvePlasticRegionRuntime({
  store,
  base,
  requestedMode,
  policy = undefined,
  verificationOnly = false,
}) {
  const config = policy ?? store.getConfig().plasticRegions
  const requested = normalizePlasticRegionMode(requestedMode ?? config?.mode ?? 'off')
  const configured = normalizePlasticRegionMode(config?.mode ?? 'off')
  if (config?.enabled === false) return offResult(requested, configured, 'disabled-by-policy')

  const mode = restrictedMode(requested, configured)
  if (mode === 'off') {
    return offResult(requested, configured, requested === 'off' ? 'request-off' : 'policy-off')
  }

  const configuredProfile = canonicalPlasticRegionProfile(config?.representationProfile)
  const physicalProfile = plasticRegionProfileForLayout(base?.layout)
  const baseProfile = base?.representationProfile ?? base?.reference?.representationProfile ?? null
  if (
    configuredProfile !== ADAPTIVE_REGION_PROFILE ||
    physicalProfile !== ADAPTIVE_REGION_PROFILE ||
    baseProfile !== ADAPTIVE_REGION_PROFILE
  ) {
    throw runtimeError(
      'PLASTIC_REGION_REENCODE_REQUIRED',
      'Plastik bölge çalışma kipi için kaynaklar açıkça 8/56 temsil uzayında yeniden kodlanmalıdır; eski I/R kayıtları dönüştürülmedi.',
      { configuredProfile, physicalProfile, baseProfile, requestedMode: requested, configuredMode: configured },
    )
  }
  const representationSpaceId = base?.familyRun?.representationSpaceId
  if (typeof representationSpaceId !== 'string' || !representationSpaceId) {
    throw runtimeError('PLASTIC_REGION_BASE_INVALID', 'Plastik bölge çalışma tabanı temsil uzayı kimliği taşımıyor.')
  }

  const restored = readPlasticRegionSnapshot(store, base?.snapshot ?? null)
  if (restored.metadata && !restored.compatible) {
    throw runtimeError('PLASTIC_REGION_ATLAS_INVALID', 'Snapshot üzerindeki plastik bölge atlası doğrulanamadı.', {
      artifactId: restored.metadata.id,
      reason: restored.reason,
    })
  }
  if ((mode === 'proposal' || mode === 'active') && !restored.head) {
    throw runtimeError(
      'PLASTIC_REGION_ATLAS_REQUIRED',
      `${mode} kipi snapshot'a bağlı, doğrulanmış bir atlas başlığı gerektirir.`,
      { snapshotId: base?.reference?.snapshotId ?? base?.snapshot?.id ?? null },
    )
  }
  if (
    mode === 'active' && !verificationOnly &&
    (!base?.reference?.activeAtStart || !base?.reference?.commitEligible)
  ) {
    throw runtimeError(
      'PLASTIC_REGION_ACTIVE_BASE_REQUIRED',
      'Etkin plastik bölge yönlendirmesi yalnız doğrulanmış etkin snapshot üzerinde çalışabilir.',
    )
  }

  const atlas = restored.head ?? createPlasticRegionAtlas({
    representationSpaceId,
    behaviorDimension: PAGE_BEHAVIOR_DIMENSION,
    maximumMembershipsPerPage: Math.min(3, Math.max(1, Number(config?.maximumMembershipsPerPage ?? 3))),
  })
  try {
    validateAdaptiveRegionAtlas(atlas)
  } catch (cause) {
    throw runtimeError('PLASTIC_REGION_ATLAS_INVALID', 'Plastik bölge atlasının geometri veya fingerprint kanıtı geçersiz.', {
      cause: String(cause?.message ?? cause),
    })
  }
  if (
    atlas.representationSpaceId !== representationSpaceId ||
    atlas.geometryFingerprint !== ADAPTIVE_REGION_GEOMETRY.fingerprint ||
    atlas.behaviorDimension !== PAGE_BEHAVIOR_DIMENSION
  ) {
    throw runtimeError('PLASTIC_REGION_ATLAS_MISMATCH', 'Atlas, seçilen FamilyRun fiziksel temsil uzayıyla uyuşmuyor.', {
      expectedRepresentationSpaceId: representationSpaceId,
      actualRepresentationSpaceId: atlas.representationSpaceId,
      expectedGeometryFingerprint: ADAPTIVE_REGION_GEOMETRY.fingerprint,
      actualGeometryFingerprint: atlas.geometryFingerprint,
      expectedBehaviorDimension: PAGE_BEHAVIOR_DIMENSION,
      actualBehaviorDimension: atlas.behaviorDimension,
    })
  }
  let release = null
  if (mode === 'active') {
    release = readPlasticRegionRelease(store, base?.snapshot ?? null, { releasePolicy: config })
    if (!release.compatible) {
      throw runtimeError(
        release.reason === 'release-manifest-absent'
          ? 'PLASTIC_REGION_RELEASE_REQUIRED'
          : 'PLASTIC_REGION_RELEASE_INVALID',
        'Etkin plastik bölge yönlendirmesi bağımsız holdout/sham kanıtına bağlı doğrulanmış release zinciri gerektirir.',
        { reason: release.reason, snapshotId: base?.snapshot?.id ?? null, cause: release.error?.details ?? null },
      )
    }
  }

  const compatibilityBlend = mode === 'active'
    ? finiteUnit(config?.compatibilityBlend, 0.25, 'plasticRegions.compatibilityBlend', {
      strictlyPositive: true,
      maximum: 0.5,
    })
    : 0
  const minimumTrace = finiteUnit(config?.minimumTrace, 0.02, 'plasticRegions.minimumTrace', {
    strictlyPositive: mode === 'active',
  })
  const maximumActiveGates = mode === 'active'
    ? boundedInteger(config?.maximumActiveGates, 4, 'plasticRegions.maximumActiveGates', 1, 56)
    : undefined
  const reason = mode === requested
    ? (restored.head ? 'snapshot-atlas-bound' : 'canonical-empty-shadow-atlas')
    : `restricted-by-policy:${configured}`
  return Object.freeze({
    requestedMode: requested,
    configuredMode: configured,
    mode,
    reason,
    atlas,
    atlasArtifactId: restored.metadata?.id ?? null,
    atlasContentHash: restored.metadata?.contentHash ?? null,
    releaseManifestArtifactId: release?.manifestMetadata?.id ?? null,
    releaseFingerprint: release?.manifest?.fingerprint ?? null,
    ephemeralAtlas: !restored.head,
    adaptiveRegion: Object.freeze({
      mode,
      atlasFingerprint: atlas.fingerprint,
      compatibilityBlend,
      minimumTrace,
      ...(mode === 'active' ? { maximumActiveGates } : {}),
    }),
  })
}
