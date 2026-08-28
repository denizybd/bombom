import type { PersistedProject, TextSource } from '../domain/types'
import { DEFAULT_TEXT_LAB_CONFIG, normalizeTextLabConfig } from '../domain/config'
import { analysisMatchesProject, validateProject } from './projectSchema'

const DATABASE_NAME = 'metin-laboratuvari'
// v4 fences out old open bundles that wrote projects without revision CAS.
const DATABASE_VERSION = 4
const PROJECT_STORE_NAME = 'projects'
const SOURCE_STORE_NAME = 'sources'
const META_STORE_NAME = 'metadata'
const CURRENT_PROJECT_KEY = 'current'
const ACTIVE_SOURCE_KEY = 'active-source-id'
const SUMMARY_PREFIX = 'summary:'
// New key intentionally forces one final clear-and-rebuild after the physical
// frozen-identity/plastic-relation split and storage-CAS model landed.
const IDENTITY_V2_JOURNAL_KEY = 'migration:identity-v2:fixed-fields-v1'
const PENDING_REBUILD_KEY = 'migration:identity-v2:pending-source-ids'
const WRITE_LOCK_NAME = 'metin-laboratuvari:indexeddb-write'

type ProjectAnalysis = PersistedProject['analysis']
type ProjectConfig = PersistedProject['config']

interface StoredProjectEnvelope {
  format: 'metin-laboratuvari-project'
  version: 1 | 2
  storageRevision?: number
  savedAt: string
  source: TextSource
  config: ProjectConfig
  analysis?: ProjectAnalysis
}

interface ProjectInventoryEntry {
  key: IDBValidKey
  source: TextSource
  config: ProjectConfig
  storageRevision: number
  savedAt: string
}

interface IdentityV2MigrationJournal {
  migration: 'identity-v2'
  databaseVersion: 4
  projectVersion: 2
  status: 'preparing' | 'complete'
  startedAt: string
  completedAt: string | null
  projectCount: number
  sourceCount: number
}

export interface IdentityV2MigrationPreparation {
  /** True while one or more non-empty sources still need identity-v2 analysis. */
  required: boolean
  /** Only the projects that still need rebuilding; their analysis is always null. */
  projects: PersistedProject[]
  sourceCount: number
}

export interface ProjectSummary {
  sourceId: string
  title: string
  updatedAt: string
  characterCount: number
  neuronCount: number
}

let fallbackWriteTail: Promise<void> = Promise.resolve()

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object'
}

function isTextSource(value: unknown): value is TextSource {
  if (!isRecord(value)) return false
  return (
    typeof value.id === 'string' &&
    value.id.length > 0 &&
    typeof value.title === 'string' &&
    typeof value.rawText === 'string'
  )
}

function isStoredProjectEnvelope(value: unknown): value is StoredProjectEnvelope {
  if (!isRecord(value)) return false
  return (
    value.format === 'metin-laboratuvari-project' &&
    (value.version === 1 || value.version === 2) &&
    typeof value.savedAt === 'string' &&
    isTextSource(value.source) &&
    isRecord(value.config)
  )
}

function isVersion2Project(value: unknown): value is StoredProjectEnvelope & { version: 2 } {
  return isStoredProjectEnvelope(value) && value.version === 2
}

function storedRevision(value: { storageRevision?: unknown }): number {
  return Number.isInteger(value.storageRevision) && (value.storageRevision as number) >= 1
    ? value.storageRevision as number
    : 1
}

function normalizeStoredProject(value: unknown): PersistedProject {
  if (!isVersion2Project(value)) {
    throw new Error('Yerel proje kimlik-v2 biçiminde değil.')
  }
  const config = normalizeTextLabConfig(value.config)
  const candidateAnalysis = value.analysis ?? null
  const analysis = candidateAnalysis && isRecord(candidateAnalysis) && analysisMatchesProject(
    value.source,
    config,
    candidateAnalysis as ProjectAnalysis,
  )
    ? candidateAnalysis
    : null
  return validateProject({
    format: 'metin-laboratuvari-project',
    version: 2,
    storageRevision: storedRevision(value),
    savedAt: value.savedAt,
    source: { ...value.source },
    config,
    analysis,
  })
}

function isCompleteMigrationJournal(value: unknown): value is IdentityV2MigrationJournal {
  if (!isRecord(value)) return false
  return (
    value.migration === 'identity-v2' &&
    value.databaseVersion === 4 &&
    value.projectVersion === 2 &&
    value.status === 'complete' &&
    typeof value.startedAt === 'string' &&
    typeof value.completedAt === 'string'
  )
}

function sourceEquals(left: TextSource, right: TextSource): boolean {
  return left.id === right.id && left.title === right.title && left.rawText === right.rawText
}

function configEquals(left: ProjectConfig, right: ProjectConfig): boolean {
  const leftRecord = left as unknown as Record<string, unknown>
  const rightRecord = right as unknown as Record<string, unknown>
  const leftKeys = Object.keys(leftRecord).sort()
  const rightKeys = Object.keys(rightRecord).sort()
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key, index) => key === rightKeys[index] && leftRecord[key] === rightRecord[key])
  )
}

function summaryFromProject(project: PersistedProject): ProjectSummary {
  return {
    sourceId: project.source.id,
    title: project.source.title,
    updatedAt: project.savedAt,
    characterCount: project.source.rawText.length,
    neuronCount: project.analysis?.stats.neuronCount ?? 0,
  }
}

function hasRebuildableText(project: Pick<PersistedProject, 'source'>): boolean {
  return project.source.rawText.trim().length > 0
}

function sortProjects(projects: PersistedProject[]): PersistedProject[] {
  return projects.sort((left, right) => right.savedAt.localeCompare(left.savedAt))
}

function normalizePendingIds(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return Array.from(new Set(value.filter((entry): entry is string => typeof entry === 'string')))
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    let blocked = false

    request.onupgradeneeded = () => {
      const database = request.result
      if (!database.objectStoreNames.contains(PROJECT_STORE_NAME)) {
        database.createObjectStore(PROJECT_STORE_NAME)
      }
      if (!database.objectStoreNames.contains(SOURCE_STORE_NAME)) {
        database.createObjectStore(SOURCE_STORE_NAME)
      }
      if (!database.objectStoreNames.contains(META_STORE_NAME)) {
        database.createObjectStore(META_STORE_NAME)
      }
    }
    request.onsuccess = () => {
      if (blocked) {
        request.result.close()
        return
      }
      request.result.onversionchange = () => request.result.close()
      resolve(request.result)
    }
    request.onerror = () => reject(request.error ?? new Error('Yerel veritabanı açılamadı.'))
    request.onblocked = () => {
      blocked = true
      reject(new Error('Yerel veri yükseltmesi başka bir açık sekme tarafından engelleniyor. Diğer sekmeyi kapatıp yeniden deneyin.'))
    }
  })
}

function awaitRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Yerel kayıt işlemi tamamlanamadı.'))
  })
}

function awaitTransaction(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onabort = () => reject(
      transaction.error ?? new Error('Yerel kayıt işlemi tamamlanmadan geri alındı.'),
    )
  })
}

async function runTransaction<T>(
  database: IDBDatabase,
  stores: string | string[],
  mode: IDBTransactionMode,
  operation: (transaction: IDBTransaction) => Promise<T>,
): Promise<T> {
  const transaction = database.transaction(stores, mode)
  const completion = awaitTransaction(transaction)
  try {
    const result = await operation(transaction)
    await completion
    return result
  } catch (error) {
    try {
      transaction.abort()
    } catch {
      // It may already have aborted because of a failed request.
    }
    try {
      await completion
    } catch {
      // Preserve the more specific operation error.
    }
    throw error
  }
}

async function getAllEntries(store: IDBObjectStore): Promise<Array<{ key: IDBValidKey; value: unknown }>> {
  const [keys, values] = await Promise.all([
    awaitRequest(store.getAllKeys()),
    awaitRequest(store.getAll()),
  ])
  if (keys.length !== values.length) {
    throw new Error('Yerel kayıt envanteri tutarsız döndü; hiçbir veri değiştirilmedi.')
  }
  return keys.map((key, index) => ({ key, value: values[index] }))
}

async function withFallbackWriteQueue<T>(operation: () => Promise<T>): Promise<T> {
  const previous = fallbackWriteTail
  let release!: () => void
  fallbackWriteTail = new Promise<void>((resolve) => {
    release = resolve
  })
  await previous
  try {
    return await operation()
  } finally {
    release()
  }
}

/** Serializes multi-transaction writes across tabs when Web Locks are available. */
export function withTextLabStorageWriteLock<T>(operation: () => Promise<T>): Promise<T> {
  if (typeof navigator !== 'undefined' && navigator.locks) {
    return navigator.locks.request(WRITE_LOCK_NAME, { mode: 'exclusive' }, operation)
  }
  return withFallbackWriteQueue(operation)
}

function inventoryProject(entry: { key: IDBValidKey; value: unknown }): ProjectInventoryEntry {
  if (!isStoredProjectEnvelope(entry.value)) {
    throw new Error(`“${String(entry.key)}” yerel kaydının ham metni veya ayarları okunamadı; hiçbir veri silinmedi.`)
  }
  return {
    key: entry.key,
    source: { ...entry.value.source },
    config: normalizeTextLabConfig(entry.value.config),
    storageRevision: entry.value.version === 2 ? storedRevision(entry.value) : 0,
    savedAt: entry.value.savedAt,
  }
}

function shortStableHash(value: string): string {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0').toUpperCase()
}

function allocateMigratedSourceId(source: TextSource, discriminator: string, usedIds: Set<string>): string {
  const hash = shortStableHash(`${source.id}\u0000${discriminator}\u0000${source.title}\u0000${source.rawText}`)
  const base = `${source.id}-MIG-${hash}`
  let candidate = base
  let suffix = 2
  while (usedIds.has(candidate)) {
    candidate = `${base}-${suffix}`
    suffix += 1
  }
  usedIds.add(candidate)
  return candidate
}

/** Collapses exact legacy aliases but preserves conflicting historical ID collisions. */
function normalizeInventory(entries: ProjectInventoryEntry[]): ProjectInventoryEntry[] {
  const groups = new Map<string, ProjectInventoryEntry[]>()
  for (const entry of entries) {
    const group = groups.get(entry.source.id) ?? []
    group.push(entry)
    groups.set(entry.source.id, group)
  }

  const usedIds = new Set(groups.keys())
  const normalized: ProjectInventoryEntry[] = []
  for (const [sourceId, group] of groups) {
    group.sort((left, right) => {
      const leftIsCanonical = String(left.key) === sourceId ? 1 : 0
      const rightIsCanonical = String(right.key) === sourceId ? 1 : 0
      return rightIsCanonical - leftIsCanonical || right.savedAt.localeCompare(left.savedAt)
    })

    const canonical = group[0]
    normalized.push(canonical)
    for (const duplicate of group.slice(1)) {
      if (sourceEquals(canonical.source, duplicate.source) && configEquals(canonical.config, duplicate.config)) {
        continue
      }
      const nextId = allocateMigratedSourceId(duplicate.source, String(duplicate.key), usedIds)
      normalized.push({ ...duplicate, source: { ...duplicate.source, id: nextId } })
    }
  }
  return normalized
}

async function deleteSummaryEntries(metadata: IDBObjectStore): Promise<void> {
  const keys = await awaitRequest(metadata.getAllKeys())
  await Promise.all(
    keys
      .filter((key) => String(key).startsWith(SUMMARY_PREFIX))
      .map((key) => awaitRequest(metadata.delete(key))),
  )
}

async function writeAndVerifySources(sourceStore: IDBObjectStore, sources: TextSource[]): Promise<void> {
  await Promise.all(sources.map((source) => awaitRequest(sourceStore.put(source, source.id))))
  const restored = await Promise.all(sources.map((source) => awaitRequest(sourceStore.get(source.id))))
  restored.forEach((value, index) => {
    if (!isTextSource(value) || !sourceEquals(value, sources[index])) {
      throw new Error('Ham metin doğrulaması başarısız oldu; geçiş bütünüyle geri alındı.')
    }
  })
}

function projectsNeedingRebuild(projects: PersistedProject[]): PersistedProject[] {
  return sortProjects(projects.filter((project) => project.analysis === null && hasRebuildableText(project)))
}

async function readVersion2Projects(store: IDBObjectStore): Promise<PersistedProject[]> {
  const values = await awaitRequest(store.getAll()) as unknown[]
  if (values.some((value) => !isVersion2Project(value))) {
    throw new Error('Yerel projeler henüz kimlik-v2 biçimine hazırlanmamış.')
  }
  return sortProjects(values.map(normalizeStoredProject))
}

export class ProjectWriteConflictError extends Error {
  readonly code = 'TEXT_LAB_WRITE_CONFLICT'

  constructor() {
    super('Bu metin başka bir sekmede değiştirildi. Eski görünüm yerel kaydın üzerine yazılmadı; arşivi yeniden açın.')
    this.name = 'ProjectWriteConflictError'
  }
}

export function resolveNextStorageRevision(
  currentRevision: number | null,
  expectedRevision: number,
): number {
  if (!Number.isInteger(expectedRevision) || expectedRevision < 0) {
    throw new RangeError('Beklenen yerel kayıt sürümü geçersiz.')
  }
  if (currentRevision !== null && (!Number.isInteger(currentRevision) || currentRevision < 1)) {
    throw new RangeError('Arşivdeki yerel kayıt sürümü geçersiz.')
  }
  const actualRevision = currentRevision ?? 0
  if (expectedRevision !== actualRevision) throw new ProjectWriteConflictError()
  return actualRevision + 1
}

export async function saveProject(project: PersistedProject): Promise<PersistedProject> {
  const normalizedConfig = normalizeTextLabConfig(project.config)
  const prepared = validateProject({
    ...project,
    source: { ...project.source },
    config: normalizedConfig,
    analysis: analysisMatchesProject(project.source, normalizedConfig, project.analysis)
      ? project.analysis
      : null,
  })
  return withTextLabStorageWriteLock(async () => {
    const database = await openDatabase()
    try {
      return await runTransaction(database, [PROJECT_STORE_NAME, SOURCE_STORE_NAME, META_STORE_NAME], 'readwrite', async (transaction) => {
        const projects = transaction.objectStore(PROJECT_STORE_NAME)
        const sources = transaction.objectStore(SOURCE_STORE_NAME)
        const metadata = transaction.objectStore(META_STORE_NAME)
        const [rawPending, currentValue] = await Promise.all([
          awaitRequest(metadata.get(PENDING_REBUILD_KEY)),
          awaitRequest(projects.get(prepared.source.id)),
        ])
        const pendingSet = new Set(normalizePendingIds(rawPending))
        const currentRevision = currentValue === undefined
          ? null
          : normalizeStoredProject(currentValue).storageRevision
        const savedProject: PersistedProject = {
          ...prepared,
          storageRevision: resolveNextStorageRevision(currentRevision, prepared.storageRevision),
          savedAt: new Date().toISOString(),
        }

        if (savedProject.analysis) pendingSet.delete(savedProject.source.id)
        else if (hasRebuildableText(savedProject)) pendingSet.add(savedProject.source.id)
        else pendingSet.delete(savedProject.source.id)

        await Promise.all([
          awaitRequest(projects.put(savedProject, savedProject.source.id)),
          awaitRequest(sources.put(savedProject.source, savedProject.source.id)),
          awaitRequest(metadata.put(savedProject.source.id, ACTIVE_SOURCE_KEY)),
          awaitRequest(metadata.put(summaryFromProject(savedProject), `${SUMMARY_PREFIX}${savedProject.source.id}`)),
          awaitRequest(metadata.put(Array.from(pendingSet), PENDING_REBUILD_KEY)),
        ])
        return savedProject
      })
    } finally {
      database.close()
    }
  })
}

export async function loadProject(sourceId?: string): Promise<PersistedProject | null> {
  const database = await openDatabase()
  try {
    return await runTransaction(database, [PROJECT_STORE_NAME, META_STORE_NAME], 'readonly', async (transaction) => {
      const activeId = sourceId ?? await awaitRequest(
        transaction.objectStore(META_STORE_NAME).get(ACTIVE_SOURCE_KEY),
      ) as string | undefined
      const store = transaction.objectStore(PROJECT_STORE_NAME)
      const value = activeId
        ? await awaitRequest(store.get(activeId))
        : await awaitRequest(store.get(CURRENT_PROJECT_KEY))
      return value === undefined ? null : normalizeStoredProject(value)
    })
  } finally {
    database.close()
  }
}

export async function loadAllProjects(): Promise<PersistedProject[]> {
  const database = await openDatabase()
  try {
    return await runTransaction(database, PROJECT_STORE_NAME, 'readonly', async (transaction) => (
      readVersion2Projects(transaction.objectStore(PROJECT_STORE_NAME))
    ))
  } finally {
    database.close()
  }
}

export async function loadAllSources(): Promise<TextSource[]> {
  const database = await openDatabase()
  try {
    return await runTransaction(database, SOURCE_STORE_NAME, 'readonly', async (transaction) => {
      const values = await awaitRequest(transaction.objectStore(SOURCE_STORE_NAME).getAll()) as unknown[]
      if (values.some((value) => !isTextSource(value))) {
        throw new Error('Ham metin arşivindeki kayıtlardan biri bozuk.')
      }
      return (values as TextSource[]).sort((left, right) => left.title.localeCompare(right.title, 'tr'))
    })
  } finally {
    database.close()
  }
}

export async function listProjects(): Promise<ProjectSummary[]> {
  const database = await openDatabase()
  try {
    return await runTransaction(database, META_STORE_NAME, 'readonly', async (transaction) => {
      const entries = await getAllEntries(transaction.objectStore(META_STORE_NAME))
      return entries
        .filter((entry) => String(entry.key).startsWith(SUMMARY_PREFIX))
        .map((entry) => entry.value)
        .filter((value): value is ProjectSummary => (
          isRecord(value) &&
          typeof value.sourceId === 'string' &&
          typeof value.title === 'string' &&
          typeof value.updatedAt === 'string' &&
          typeof value.characterCount === 'number' &&
          typeof value.neuronCount === 'number'
        ))
        .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
    })
  } finally {
    database.close()
  }
}

/** Crash-safe one-time preparation for the fixed identity-field model. */
export async function prepareIdentityV2Migration(): Promise<IdentityV2MigrationPreparation> {
  return withTextLabStorageWriteLock(async () => {
    const database = await openDatabase()
    try {
      const existing = await runTransaction(database, [PROJECT_STORE_NAME, SOURCE_STORE_NAME, META_STORE_NAME], 'readwrite', async (transaction) => {
        const projectStore = transaction.objectStore(PROJECT_STORE_NAME)
        const sourceStore = transaction.objectStore(SOURCE_STORE_NAME)
        const metadata = transaction.objectStore(META_STORE_NAME)
        const [journal, rawProjects, rawSources] = await Promise.all([
          awaitRequest(metadata.get(IDENTITY_V2_JOURNAL_KEY)),
          awaitRequest(projectStore.getAll()),
          awaitRequest(sourceStore.getAll()),
        ])
        const sourceCount = rawSources.length

        if (!isCompleteMigrationJournal(journal) || rawProjects.some((value) => !isVersion2Project(value))) {
          const now = new Date().toISOString()
          const preparing: IdentityV2MigrationJournal = {
            migration: 'identity-v2',
            databaseVersion: 4,
            projectVersion: 2,
            status: 'preparing',
            startedAt: isRecord(journal) && typeof journal.startedAt === 'string' ? journal.startedAt : now,
            completedAt: null,
            projectCount: rawProjects.length,
            sourceCount,
          }
          await awaitRequest(metadata.put(preparing, IDENTITY_V2_JOURNAL_KEY))
          return null
        }

        const projects = rawProjects.map((rawProject) => {
          const normalized = normalizeStoredProject(rawProject)
          const envelope = rawProject as StoredProjectEnvelope
          const hasRevision = Number.isInteger(envelope.storageRevision) && (envelope.storageRevision ?? 0) >= 1
          const contentChanged = !configEquals(envelope.config, normalized.config) ||
            (envelope.analysis ?? null) !== normalized.analysis
          return hasRevision && contentChanged
            ? { ...normalized, storageRevision: normalized.storageRevision + 1 }
            : normalized
        })
        const projectSourceMap = new Map(projects.map((project) => [project.source.id, project.source]))
        const usedIds = new Set(projectSourceMap.keys())
        const recoveredSources: TextSource[] = []
        const recoveredAt = new Date().toISOString()
        for (const rawSource of rawSources) {
          if (!isTextSource(rawSource)) {
            throw new Error('Ham metin arşivindeki kayıtlardan biri bozuk; hiçbir veri değiştirilmedi.')
          }
          const projectSource = projectSourceMap.get(rawSource.id)
          if (projectSource && sourceEquals(projectSource, rawSource)) continue
          const source = projectSource
            ? { ...rawSource, id: allocateMigratedSourceId(rawSource, 'completed-journal', usedIds) }
            : rawSource
          projects.push({
            format: 'metin-laboratuvari-project',
            version: 2,
            storageRevision: 1,
            savedAt: recoveredAt,
            source,
            config: { ...DEFAULT_TEXT_LAB_CONFIG },
            analysis: null,
          })
          projectSourceMap.set(source.id, source)
          usedIds.add(source.id)
          recoveredSources.push(source)
        }
        for (const project of projects) {
          if (!rawSources.some((source) => isTextSource(source) && sourceEquals(source, project.source))) {
            recoveredSources.push(project.source)
          }
        }
        sortProjects(projects)
        const pendingProjects = projectsNeedingRebuild(projects)
        await Promise.all([
          ...projects.map((project) => awaitRequest(projectStore.put(project, project.source.id))),
          ...recoveredSources.map((source) => awaitRequest(sourceStore.put(source, source.id))),
          ...projects.map((project) => awaitRequest(
            metadata.put(summaryFromProject(project), `${SUMMARY_PREFIX}${project.source.id}`),
          )),
          awaitRequest(metadata.put(pendingProjects.map((project) => project.source.id), PENDING_REBUILD_KEY)),
        ])
        return {
          required: pendingProjects.length > 0,
          projects: pendingProjects,
          sourceCount: new Set([
            ...rawSources.filter(isTextSource).map((source) => source.id),
            ...recoveredSources.map((source) => source.id),
          ]).size,
        } satisfies IdentityV2MigrationPreparation
      })
      if (existing) return existing

      return await runTransaction(database, [PROJECT_STORE_NAME, SOURCE_STORE_NAME, META_STORE_NAME], 'readwrite', async (transaction) => {
        const projectStore = transaction.objectStore(PROJECT_STORE_NAME)
        const sourceStore = transaction.objectStore(SOURCE_STORE_NAME)
        const metadata = transaction.objectStore(META_STORE_NAME)
        const [projectEntries, sourceEntries, activeId, journal] = await Promise.all([
          getAllEntries(projectStore),
          getAllEntries(sourceStore),
          awaitRequest(metadata.get(ACTIVE_SOURCE_KEY)),
          awaitRequest(metadata.get(IDENTITY_V2_JOURNAL_KEY)),
        ])

        const inventory = normalizeInventory(projectEntries.map(inventoryProject))
        const migratedProjects: PersistedProject[] = inventory.map((entry) => ({
          format: 'metin-laboratuvari-project',
          version: 2,
          storageRevision: Math.max(1, entry.storageRevision + 1),
          savedAt: entry.savedAt,
          source: entry.source,
          config: entry.config,
          analysis: null,
        }))

        const sourceMap = new Map<string, TextSource>()
        const usedIds = new Set(inventory.map((entry) => entry.source.id))
        for (const project of migratedProjects) sourceMap.set(project.source.id, project.source)
        for (const entry of sourceEntries) {
          if (!isTextSource(entry.value)) {
            throw new Error(`“${String(entry.key)}” ham metin kaydı bozuk; hiçbir veri değiştirilmedi.`)
          }
          const stored = sourceMap.get(entry.value.id)
          if (!stored) {
            sourceMap.set(entry.value.id, entry.value)
            usedIds.add(entry.value.id)
          } else if (!sourceEquals(stored, entry.value)) {
            const nextId = allocateMigratedSourceId(entry.value, String(entry.key), usedIds)
            sourceMap.set(nextId, { ...entry.value, id: nextId })
          }
        }
        const sources = Array.from(sourceMap.values())
        const migratedProjectIds = new Set(migratedProjects.map((project) => project.source.id))
        const recoveredAt = new Date().toISOString()
        for (const source of sources) {
          if (migratedProjectIds.has(source.id)) continue
          migratedProjects.push({
            format: 'metin-laboratuvari-project',
            version: 2,
            storageRevision: 1,
            savedAt: recoveredAt,
            source,
            config: { ...DEFAULT_TEXT_LAB_CONFIG },
            analysis: null,
          })
          migratedProjectIds.add(source.id)
        }

        await Promise.all([
          awaitRequest(projectStore.clear()),
          awaitRequest(sourceStore.clear()),
          deleteSummaryEntries(metadata),
        ])
        await writeAndVerifySources(sourceStore, sources)
        await Promise.all(migratedProjects.flatMap((project) => [
          awaitRequest(projectStore.put(project, project.source.id)),
          awaitRequest(metadata.put(summaryFromProject(project), `${SUMMARY_PREFIX}${project.source.id}`)),
        ]))

        const projectIds = new Set(migratedProjects.map((project) => project.source.id))
        const preservedActiveId = typeof activeId === 'string' && projectIds.has(activeId)
          ? activeId
          : migratedProjects[0]?.source.id
        if (preservedActiveId) await awaitRequest(metadata.put(preservedActiveId, ACTIVE_SOURCE_KEY))
        else await awaitRequest(metadata.delete(ACTIVE_SOURCE_KEY))

        const pendingProjects = projectsNeedingRebuild(migratedProjects)
        const completedAt = new Date().toISOString()
        const completed: IdentityV2MigrationJournal = {
          migration: 'identity-v2',
          databaseVersion: 4,
          projectVersion: 2,
          status: 'complete',
          startedAt: isRecord(journal) && typeof journal.startedAt === 'string' ? journal.startedAt : completedAt,
          completedAt,
          projectCount: migratedProjects.length,
          sourceCount: sources.length,
        }
        await Promise.all([
          awaitRequest(metadata.put(completed, IDENTITY_V2_JOURNAL_KEY)),
          awaitRequest(metadata.put(pendingProjects.map((project) => project.source.id), PENDING_REBUILD_KEY)),
        ])

        return {
          required: pendingProjects.length > 0,
          projects: pendingProjects,
          sourceCount: sources.length,
        }
      })
    } finally {
      database.close()
    }
  })
}

/** Clears analyses atomically while retaining every source and config. */
export async function clearDerivedDataKeepingSources(): Promise<PersistedProject[]> {
  return withTextLabStorageWriteLock(async () => {
    const database = await openDatabase()
    try {
      return await runTransaction(database, [PROJECT_STORE_NAME, SOURCE_STORE_NAME, META_STORE_NAME], 'readwrite', async (transaction) => {
        const projectStore = transaction.objectStore(PROJECT_STORE_NAME)
        const sourceStore = transaction.objectStore(SOURCE_STORE_NAME)
        const metadata = transaction.objectStore(META_STORE_NAME)
        const rawProjects = await readVersion2Projects(projectStore)
        const projects = rawProjects.map((project): PersistedProject => ({
          ...project,
          storageRevision: project.storageRevision + 1,
          savedAt: new Date().toISOString(),
          analysis: null,
        }))

        await writeAndVerifySources(sourceStore, projects.map((project) => project.source))
        await deleteSummaryEntries(metadata)
        await Promise.all(projects.flatMap((project) => [
          awaitRequest(projectStore.put(project, project.source.id)),
          awaitRequest(metadata.put(summaryFromProject(project), `${SUMMARY_PREFIX}${project.source.id}`)),
        ]))
        await awaitRequest(metadata.put(
          projects.filter(hasRebuildableText).map((project) => project.source.id),
          PENDING_REBUILD_KEY,
        ))
        return sortProjects(projects)
      })
    } finally {
      database.close()
    }
  })
}

/** Replaces one project's derived analysis and updates its rebuild journal entry. */
export async function replaceProjectAnalysis(sourceId: string, analysis: ProjectAnalysis): Promise<PersistedProject> {
  return withTextLabStorageWriteLock(async () => {
    const database = await openDatabase()
    try {
      return await runTransaction(database, [PROJECT_STORE_NAME, SOURCE_STORE_NAME, META_STORE_NAME], 'readwrite', async (transaction) => {
        const projectStore = transaction.objectStore(PROJECT_STORE_NAME)
        const sourceStore = transaction.objectStore(SOURCE_STORE_NAME)
        const metadata = transaction.objectStore(META_STORE_NAME)
        const [value, rawPending] = await Promise.all([
          awaitRequest(projectStore.get(sourceId)),
          awaitRequest(metadata.get(PENDING_REBUILD_KEY)),
        ])
        if (!isVersion2Project(value)) {
          throw new Error('Analizi değiştirilecek kaynak yerel arşivde bulunamadı.')
        }
        const currentProject = normalizeStoredProject(value)
        if (
          analysis &&
          !analysisMatchesProject(currentProject.source, currentProject.config, analysis)
        ) {
          throw new Error('Analiz başka bir kaynak veya ayar sürümüne ait; yerel kayıt değiştirilmedi.')
        }

        const project = validateProject({
          ...currentProject,
          storageRevision: currentProject.storageRevision + 1,
          savedAt: new Date().toISOString(),
          analysis,
        })
        const pending = new Set(normalizePendingIds(rawPending))
        if (analysis) pending.delete(sourceId)
        else if (hasRebuildableText(project)) pending.add(sourceId)
        else pending.delete(sourceId)

        await Promise.all([
          awaitRequest(projectStore.put(project, sourceId)),
          awaitRequest(sourceStore.put(project.source, sourceId)),
          awaitRequest(metadata.put(summaryFromProject(project), `${SUMMARY_PREFIX}${sourceId}`)),
          awaitRequest(metadata.put(Array.from(pending), PENDING_REBUILD_KEY)),
        ])
        return project
      })
    } finally {
      database.close()
    }
  })
}

export async function clearProject(): Promise<void> {
  return withTextLabStorageWriteLock(async () => {
    const database = await openDatabase()
    try {
      await runTransaction(database, PROJECT_STORE_NAME, 'readwrite', async (transaction) => {
        await awaitRequest(transaction.objectStore(PROJECT_STORE_NAME).delete(CURRENT_PROJECT_KEY))
      })
    } finally {
      database.close()
    }
  })
}
