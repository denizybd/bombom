import type {
  CoalitionState,
  FamilyRun,
  FamilyTransition,
  LearnedFamily,
  SparseVector,
} from '../families/types'

const DATABASE_NAME = 'noron-aile-laboratuvari'
const DATABASE_VERSION = 1
const PROJECT_STORE = 'family-projects'
const COALITION_STORE = 'coalition-logs'
const WRITE_LOCK_NAME = 'noron-aileleri:indexeddb-write'
const MAX_COALITION_LOGS = 100

export interface FamilyProject {
  format: 'noron-aileleri-project'
  version: 1
  id: string
  name: string
  storageRevision: number
  savedAt: string
  run: FamilyRun
}

export interface CoalitionLogRecord {
  format: 'noron-aileleri-coalition-log'
  version: 1
  id: string
  runId: string
  recordedAt: string
  coalition: CoalitionState
}

export class FamilyProjectWriteConflictError extends Error {
  readonly code = 'FAMILY_PROJECT_WRITE_CONFLICT'

  constructor() {
    super('Bu aile modeli başka bir sekmede değiştirildi. Eski görünüm arşivin üzerine yazılmadı; arşivi yenileyin.')
    this.name = 'FamilyProjectWriteConflictError'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object'
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function unit(value: unknown): value is number {
  return finite(value) && value >= 0 && value <= 1
}

function nonNegative(value: unknown): value is number {
  return finite(value) && value >= 0
}

function assertSparseVector(value: SparseVector, label: string): void {
  if (
    !value ||
    !Number.isInteger(value.dimension) ||
    value.dimension < 1 ||
    !Array.isArray(value.entries) ||
    !nonNegative(value.norm)
  ) {
    throw new TypeError(`${label} geçerli bir seyrek vektör değil.`)
  }
  let previous = -1
  let normSquared = 0
  for (const entry of value.entries) {
    if (
      !entry ||
      !Number.isInteger(entry.index) ||
      entry.index <= previous ||
      entry.index >= value.dimension ||
      !finite(entry.value) ||
      entry.value === 0
    ) {
      throw new TypeError(`${label} sıralı, benzersiz ve sonlu koordinatlar taşımalıdır.`)
    }
    previous = entry.index
    normSquared += entry.value * entry.value
  }
  if (Math.abs(Math.sqrt(normSquared) - value.norm) > 1e-4) {
    throw new TypeError(`${label} normu koordinatlarıyla uyuşmuyor.`)
  }
}

function assertFamily(family: LearnedFamily, behaviorSpaceId: string): void {
  if (
    !family ||
    typeof family.id !== 'string' ||
    !family.id ||
    typeof family.fingerprint !== 'string' ||
    !family.fingerprint ||
    family.behaviorSpaceId !== behaviorSpaceId ||
    !Array.isArray(family.memberKeys) ||
    !Array.isArray(family.coreMemberKeys) ||
    new Set(family.memberKeys).size !== family.memberKeys.length ||
    new Set(family.coreMemberKeys).size !== family.coreMemberKeys.length ||
    family.coreMemberKeys.some((key) => !family.memberKeys.includes(key)) ||
    !Number.isInteger(family.supportCount) ||
    family.supportCount < 0 ||
    !Number.isInteger(family.sourceCount) ||
    family.sourceCount < 0 ||
    !unit(family.cohesion) ||
    !finite(family.separation) ||
    !unit(family.stability) ||
    (family.utility !== null && !finite(family.utility))
  ) {
    throw new TypeError('Aile arşivindeki bir öğrenilmiş aile geçersiz.')
  }
  assertSparseVector(family.prototype, `“${family.id}” aile prototipi`)
}

function assertTransition(
  transition: FamilyTransition,
  familyIds: ReadonlySet<string>,
): void {
  if (
    !transition ||
    typeof transition.id !== 'string' ||
    !transition.id ||
    !familyIds.has(transition.fromFamilyId) ||
    !familyIds.has(transition.toFamilyId) ||
    !nonNegative(transition.priorEvidence) ||
    !nonNegative(transition.experienceEvidence) ||
    !nonNegative(transition.inhibitionEvidence) ||
    !Number.isInteger(transition.sourceCount) ||
    transition.sourceCount < 0 ||
    !Number.isInteger(transition.supportCount) ||
    transition.supportCount < 0 ||
    !unit(transition.probability)
  ) {
    throw new TypeError('Aile arşivindeki bir geçiş geçersiz.')
  }
}

export function validateFamilyRun(value: unknown): FamilyRun {
  if (!isRecord(value)) throw new TypeError('Dosya geçerli bir Nöron Aileleri koşusu değil.')
  const run = value as unknown as FamilyRun
  if (
    run.format !== 'noron-aileleri-run' ||
    run.version !== 1 ||
    typeof run.id !== 'string' ||
    !run.id ||
    typeof run.fingerprint !== 'string' ||
    !run.fingerprint ||
    typeof run.createdAt !== 'string' ||
    !Number.isFinite(Date.parse(run.createdAt)) ||
    typeof run.representationSpaceId !== 'string' ||
    !run.representationSpaceId ||
    typeof run.behaviorSpaceId !== 'string' ||
    !run.behaviorSpaceId ||
    !isRecord(run.config) ||
    !run.inputManifest ||
    !Array.isArray(run.inputManifest.entries) ||
    typeof run.inputManifest.corpusFingerprint !== 'string' ||
    !Array.isArray(run.profiles) ||
    !Array.isArray(run.families) ||
    !Array.isArray(run.memberships) ||
    !Array.isArray(run.neuronTransitions) ||
    !Array.isArray(run.familyTransitions) ||
    !Array.isArray(run.utilityReports) ||
    !run.coalition ||
    !Array.isArray(run.warnings) ||
    !run.stats
  ) {
    throw new TypeError('Nöron Aileleri koşusu eksik veya tanınmayan bir biçimde.')
  }

  const manifestKeys = run.inputManifest.entries.map((entry) => `${entry.kind}\0${entry.id}`)
  if (
    new Set(manifestKeys).size !== manifestKeys.length ||
    run.inputManifest.entries.some((entry) => (
      !entry ||
      typeof entry.id !== 'string' ||
      !entry.id ||
      entry.representationSpaceId !== run.representationSpaceId ||
      typeof entry.revision !== 'string' ||
      typeof entry.fingerprint !== 'string'
    ))
  ) {
    throw new TypeError('Koşunun girdi manifesti geçersiz veya yinelenen kayıt içeriyor.')
  }

  const profileKeys = new Set<string>()
  for (const profile of run.profiles) {
    if (
      !profile ||
      typeof profile.memberKey !== 'string' ||
      !profile.memberKey ||
      profileKeys.has(profile.memberKey) ||
      profile.representationSpaceId !== run.representationSpaceId ||
      profile.behaviorSpaceId !== run.behaviorSpaceId ||
      !Array.isArray(profile.evidence) ||
      !Array.isArray(profile.regionAffinities) ||
      profile.regionAffinities.some((affinity) => !unit(affinity.score)) ||
      Object.values(profile.diagnostics).some((metric) => !finite(metric))
    ) {
      throw new TypeError('Koşudaki bir davranış profili geçersiz.')
    }
    assertSparseVector(profile.vector, `“${profile.memberKey}” davranış profili`)
    profileKeys.add(profile.memberKey)
  }

  const familyIds = new Set<string>()
  for (const family of run.families) {
    assertFamily(family, run.behaviorSpaceId)
    if (familyIds.has(family.id)) throw new TypeError('Koşuda yinelenen aile kimliği var.')
    if (family.memberKeys.some((key) => !profileKeys.has(key))) {
      throw new TypeError('Bir aile var olmayan davranış profiline başvuruyor.')
    }
    familyIds.add(family.id)
  }

  const membershipKeys = new Set<string>()
  const membershipCounts = new Map<string, number>()
  for (const membership of run.memberships) {
    const key = `${membership.memberKey}\0${membership.familyId}`
    if (
      membershipKeys.has(key) ||
      !profileKeys.has(membership.memberKey) ||
      !familyIds.has(membership.familyId) ||
      !Number.isInteger(membership.rank) ||
      membership.rank < 1 ||
      !unit(membership.score) ||
      !unit(membership.weight) ||
      !unit(membership.breakdown.behaviorSimilarity) ||
      !unit(membership.breakdown.regionCompatibility) ||
      !unit(membership.breakdown.evidenceConfidence) ||
      !unit(membership.breakdown.combined)
    ) {
      throw new TypeError('Koşudaki bir yumuşak aile üyeliği geçersiz.')
    }
    membershipKeys.add(key)
    membershipCounts.set(
      membership.memberKey,
      (membershipCounts.get(membership.memberKey) ?? 0) + 1,
    )
  }
  if ([...membershipCounts.values()].some((count) => count > run.config.maximumMembershipsPerNeuron)) {
    throw new TypeError('Bir nöron ayarlanan seyrek üyelik sınırını aşıyor.')
  }

  const neuronTransitionKeys = new Set<string>()
  const neuronOutgoing = new Map<string, { probability: number; count: number }>()
  for (const transition of run.neuronTransitions) {
    const key = `${transition.fromMemberKey}\0${transition.toMemberKey}`
    if (
      !transition ||
      neuronTransitionKeys.has(key) ||
      transition.fromMemberKey === transition.toMemberKey ||
      !profileKeys.has(transition.fromMemberKey) ||
      !profileKeys.has(transition.toMemberKey) ||
      !unit(transition.probability) ||
      transition.probability <= 0
    ) {
      throw new TypeError('Koşudaki bir seyrek nöron geçişi geçersiz.')
    }
    neuronTransitionKeys.add(key)
    const row = neuronOutgoing.get(transition.fromMemberKey) ?? { probability: 0, count: 0 }
    row.probability += transition.probability
    row.count += 1
    neuronOutgoing.set(transition.fromMemberKey, row)
  }
  if ([...neuronOutgoing.values()].some((row) => (
    Math.abs(row.probability - 1) > 1e-4 ||
    row.count > run.config.maximumTransitionNeighbors
  ))) {
    throw new TypeError('Seyrek nöron geçiş satırı normalize değil veya top-k sınırını aşıyor.')
  }

  const transitionIds = new Set<string>()
  const outgoing = new Map<string, number>()
  for (const transition of run.familyTransitions) {
    assertTransition(transition, familyIds)
    if (transitionIds.has(transition.id)) throw new TypeError('Koşuda yinelenen aile geçişi var.')
    transitionIds.add(transition.id)
    outgoing.set(
      transition.fromFamilyId,
      (outgoing.get(transition.fromFamilyId) ?? 0) + transition.probability,
    )
  }
  if ([...outgoing.values()].some((sum) => Math.abs(sum - 1) > 1e-4)) {
    throw new TypeError('Aile geçiş olasılıkları satır bazında normalize değil.')
  }

  if (run.utilityReports.some((report) => !familyIds.has(report.familyId) || !finite(report.deltaUtility))) {
    throw new TypeError('Koşudaki bir susturma testi var olmayan aileye başvuruyor.')
  }
  if (
    run.coalition.coalitions.some((coalition) => (
      !familyIds.has(coalition.eventFamilyId) ||
      !profileKeys.has(coalition.eventMemberKey) ||
      coalition.memberKeys.some((key) => !profileKeys.has(key)) ||
      coalition.bindings.some((binding) => !profileKeys.has(binding.memberKey))
    ))
  ) {
    throw new TypeError('Geçici koalisyon var olmayan aile veya nörona başvuruyor.')
  }
  if (
    run.routingPreview && (
      !profileKeys.has(run.routingPreview.currentMemberKey) ||
      run.routingPreview.entries.some((entry) => !profileKeys.has(entry.memberKey)) ||
      Math.abs(run.routingPreview.totalProbability - 1) > 1e-4
    )
  ) {
    throw new TypeError('İki seviyeli yönlendirme dağılımı geçersiz.')
  }

  return run
}

export function validateFamilyProject(value: unknown): FamilyProject {
  if (!isRecord(value)) throw new TypeError('Dosya geçerli bir Nöron Aileleri projesi değil.')
  const project = value as unknown as FamilyProject
  if (
    project.format !== 'noron-aileleri-project' ||
    project.version !== 1 ||
    typeof project.id !== 'string' ||
    !project.id ||
    typeof project.name !== 'string' ||
    !Number.isInteger(project.storageRevision) ||
    project.storageRevision < 1 ||
    typeof project.savedAt !== 'string' ||
    !Number.isFinite(Date.parse(project.savedAt))
  ) {
    throw new TypeError('Nöron Aileleri proje zarfı geçersiz.')
  }
  validateFamilyRun(project.run)
  return project
}

export function resolveNextFamilyStorageRevision(
  currentRevision: number | null,
  expectedRevision: number,
): number {
  if (!Number.isInteger(expectedRevision) || expectedRevision < 0) {
    throw new RangeError('Beklenen aile arşivi sürümü geçersiz.')
  }
  const actual = currentRevision ?? 0
  if (expectedRevision !== actual) throw new FamilyProjectWriteConflictError()
  return actual + 1
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    request.onupgradeneeded = () => {
      const database = request.result
      if (!database.objectStoreNames.contains(PROJECT_STORE)) {
        const store = database.createObjectStore(PROJECT_STORE, { keyPath: 'id' })
        store.createIndex('by-saved-at', 'savedAt')
        store.createIndex('by-run-fingerprint', 'run.fingerprint', { unique: true })
      }
      if (!database.objectStoreNames.contains(COALITION_STORE)) {
        const store = database.createObjectStore(COALITION_STORE, { keyPath: 'id' })
        store.createIndex('by-recorded-at', 'recordedAt')
        store.createIndex('by-run-id', 'runId')
      }
    }
    request.onsuccess = () => {
      request.result.onversionchange = () => request.result.close()
      resolve(request.result)
    }
    request.onerror = () => reject(request.error ?? new Error('Nöron Aileleri arşivi açılamadı.'))
    request.onblocked = () => reject(new Error('Aile arşivi yükseltmesi başka bir açık sekme tarafından engelleniyor.'))
  })
}

function awaitRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Aile arşivi işlemi tamamlanamadı.'))
  })
}

function awaitTransaction(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onabort = () => reject(transaction.error ?? new Error('Aile arşivi işlemi geri alındı.'))
    transaction.onerror = () => reject(transaction.error ?? new Error('Aile arşivi işlemi tamamlanamadı.'))
  })
}

function withWriteLock<T>(operation: () => Promise<T>): Promise<T> {
  const locks = typeof navigator !== 'undefined' ? navigator.locks : undefined
  return locks ? locks.request(WRITE_LOCK_NAME, operation) : operation()
}

export async function loadFamilyProjects(): Promise<FamilyProject[]> {
  const database = await openDatabase()
  try {
    const transaction = database.transaction(PROJECT_STORE, 'readonly')
    const records = await awaitRequest(transaction.objectStore(PROJECT_STORE).getAll()) as unknown[]
    await awaitTransaction(transaction)
    return records.map(validateFamilyProject).sort((left, right) => right.savedAt.localeCompare(left.savedAt))
  } finally {
    database.close()
  }
}

export async function saveFamilyRun(
  runInput: FamilyRun,
  name = 'Aile modeli',
): Promise<FamilyProject> {
  const run = validateFamilyRun(runInput)
  return withWriteLock(async () => {
    const database = await openDatabase()
    try {
      const transaction = database.transaction(PROJECT_STORE, 'readwrite')
      const store = transaction.objectStore(PROJECT_STORE)
      const id = `family-project:${run.fingerprint}`
      const current = await awaitRequest(store.get(id)) as unknown
      const currentProject = current === undefined ? null : validateFamilyProject(current)
      const project: FamilyProject = {
        format: 'noron-aileleri-project',
        version: 1,
        id,
        name: name.trim() || 'Aile modeli',
        storageRevision: resolveNextFamilyStorageRevision(
          currentProject?.storageRevision ?? null,
          currentProject?.storageRevision ?? 0,
        ),
        savedAt: new Date().toISOString(),
        run,
      }
      store.put(project)
      await awaitTransaction(transaction)
      return project
    } finally {
      database.close()
    }
  })
}

export async function saveCoalitionLog(
  runId: string,
  coalition: CoalitionState,
): Promise<CoalitionLogRecord> {
  return withWriteLock(async () => {
    const database = await openDatabase()
    try {
      const transaction = database.transaction(COALITION_STORE, 'readwrite')
      const store = transaction.objectStore(COALITION_STORE)
      const recordedAt = new Date().toISOString()
      const record: CoalitionLogRecord = {
        format: 'noron-aileleri-coalition-log',
        version: 1,
        id: `coalition:${runId}:${recordedAt}:${coalition.tick}`,
        runId,
        recordedAt,
        coalition,
      }
      store.put(record)
      const all = await awaitRequest(store.index('by-recorded-at').getAll()) as CoalitionLogRecord[]
      const overflow = all
        .sort((left, right) => left.recordedAt.localeCompare(right.recordedAt))
        .slice(0, Math.max(0, all.length - MAX_COALITION_LOGS))
      overflow.forEach((entry) => store.delete(entry.id))
      await awaitTransaction(transaction)
      return record
    } finally {
      database.close()
    }
  })
}

export function stringifyFamilyProject(project: FamilyProject): string {
  return JSON.stringify(validateFamilyProject(project), null, 2)
}

export function parseFamilyProject(value: string): FamilyProject {
  return validateFamilyProject(JSON.parse(value))
}
