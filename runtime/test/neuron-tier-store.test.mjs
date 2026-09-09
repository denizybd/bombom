import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import test from 'node:test'
import { temporaryDirectory } from '../../server/test/helpers.mjs'
import {
  normalizeNeuronTierPolicy,
  NeuronRoutingSketchIndex,
  openNeuronTierRepository,
  rankNeuronRoutingSketches,
  scoreNeuronUtility,
  writeNeuronTierSnapshot,
} from '../neuron-tier-store.mjs'

const LAYOUT = { identityLength: 4, relationLength: 4, cellCount: 8 }
const PHYSICAL_LAYOUT = { identityLength: 512, relationLength: 3_584, cellCount: 4_096 }

function digestIdentity(identity) {
  return createHash('sha256')
    .update(Buffer.from(identity.buffer, identity.byteOffset, identity.byteLength))
    .digest('hex')
}

function neuron(id, offset = 0) {
  const identity = new Float32Array([1 + offset, -0, 0.125 + offset, -0.75])
  return {
    id,
    identity,
    relation: new Float32Array([offset, -offset, 0.5, -0.5]),
    identityDigest: digestIdentity(identity),
    familyMemberships: [{ familyId: `family-${offset % 2}`, weight: 1 }],
    provenance: { kind: 'seed', surface: `nöron ${id}` },
  }
}

function physicalNeuron(id, firstPageValue) {
  const identity = Float32Array.from({ length: PHYSICAL_LAYOUT.identityLength }, (_, index) => index % 2 ? -1 : 1)
  const relation = new Float32Array(PHYSICAL_LAYOUT.relationLength).fill(-1)
  relation.fill(firstPageValue, 0, 64)
  return {
    id,
    identity,
    relation,
    identityDigest: digestIdentity(identity),
    familyMemberships: [],
    provenance: { kind: 'test' },
  }
}

function policy(patch = {}) {
  return normalizeNeuronTierPolicy({
    hotBudgetBytes: 512,
    warmBudgetBytes: 2_048,
    pageTargetBytes: 420,
    maximumPageBytes: 800,
    maximumWorkingSetNeurons: 3,
    hotPromotionHits: 2,
    idleHalfLifeAccesses: 4,
    ...patch,
  })
}

async function snapshot(t, neurons, policyValue = policy()) {
  const directory = await temporaryDirectory(t)
  const result = await writeNeuronTierSnapshot({
    directory,
    modelFingerprint: 'model-a:head-a',
    representationSpaceId: 'space-a',
    layout: LAYOUT,
    neurons,
    policy: policyValue,
    identityDigestOf: digestIdentity,
  })
  return { directory, ...result, policy: policyValue }
}

function exactFloatBytes(vector) {
  return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength)
}

function canonicalManifestValue(value) {
  if (value === null || typeof value !== 'object') return value
  if (Array.isArray(value)) return value.map(canonicalManifestValue)
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalManifestValue(value[key])]))
}

test('manifest and SSD pages are content-addressed and independent of input ordering', async (t) => {
  const values = [neuron('z', 3), neuron('a', 1), neuron('m', 2)]
  const left = await snapshot(t, values)
  const right = await snapshot(t, [...values].reverse())

  assert.equal(left.manifest.fingerprint, right.manifest.fingerprint)
  assert.deepEqual(left.manifest.pages.map((page) => page.contentHash), right.manifest.pages.map((page) => page.contentHash))
  assert.deepEqual(left.manifest.index.map((entry) => entry.id), ['a', 'm', 'z'])
  const leftManifest = await readFile(left.manifestPath)
  const rightManifest = await readFile(right.manifestPath)
  assert.deepEqual(leftManifest, rightManifest)
  assert.equal(left.manifest.version, 2)
  assert.equal(left.manifest.physicalRepresentationSpaceId, 'space-a')
  assert.equal(left.manifest.thoughtModelFingerprint, 'model-a:head-a')
  assert.equal(left.manifest.representationProfile, undefined)
  assert.equal(Object.hasOwn(left.manifest, 'representationSpaceId'), false)
  assert.equal(left.manifest.relationPageSketches, undefined, 'non-512/3584 layouts must not claim physical page sketches')
})

test('legacy V1 manifests remain readable but their ambiguous representation binding is never reinterpreted', async (t) => {
  const created = await snapshot(t, [neuron('legacy', 1)])
  const {
    fingerprint: _fingerprint,
    physicalRepresentationSpaceId: _physical,
    thoughtModelFingerprint: _thought,
    ...v2Body
  } = created.manifest
  const legacyBody = {
    ...v2Body,
    version: 1,
    representationSpaceId: 'historical-ambiguous-value',
  }
  const canonical = JSON.stringify(canonicalManifestValue(legacyBody))
  const legacyFingerprint = createHash('sha256')
    .update(Buffer.from(`bombom-neuron-tier-manifest-v1\0${canonical}`))
    .digest('hex')
  const legacyManifest = { ...legacyBody, fingerprint: legacyFingerprint }
  const legacyPath = `${created.directory}/manifests/${legacyFingerprint}.json`
  await writeFile(legacyPath, JSON.stringify(canonicalManifestValue(legacyManifest)))

  const repository = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: legacyPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  assert.equal(repository.manifest.version, 1)
  assert.equal(repository.manifest.physicalRepresentationSpaceId, null)
  assert.equal(repository.manifest.thoughtModelFingerprint, null)
  assert.equal(repository.manifest.legacyRepresentationSpaceId, 'historical-ambiguous-value')
  assert.equal(repository.manifest.representationBinding, 'legacy-v1-unverified')
  repository.close()

  const ambiguousBody = {
    ...legacyBody,
    relationPageSketches: {
      format: 'relation-page-sign-projection-v1',
      pageCount: 56,
    },
  }
  const ambiguousCanonical = JSON.stringify(canonicalManifestValue(ambiguousBody))
  const ambiguousFingerprint = createHash('sha256')
    .update(Buffer.from(`bombom-neuron-tier-manifest-v1\0${ambiguousCanonical}`))
    .digest('hex')
  const ambiguousPath = `${created.directory}/manifests/${ambiguousFingerprint}.json`
  await writeFile(ambiguousPath, JSON.stringify(canonicalManifestValue({ ...ambiguousBody, fingerprint: ambiguousFingerprint })))
  await assert.rejects(
    openNeuronTierRepository({
      directory: created.directory,
      manifestPath: ambiguousPath,
      policy: created.policy,
      identityDigestOf: digestIdentity,
    }),
    (error) => error.code === 'NEURON_MANIFEST_INCOMPATIBLE',
  )
})

test('distance monotonically lowers fixed-point utility', () => {
  const evidence = { routeProbability: 0.4, familyProbability: 0.3, observedUseProbability: 0.2 }
  const near = scoreNeuronUtility({ ...evidence, distance: 0.1 })
  const middle = scoreNeuronUtility({ ...evidence, distance: 0.5 })
  const far = scoreNeuronUtility({ ...evidence, distance: 0.9 })
  assert.ok(near.fixed > middle.fixed)
  assert.ok(middle.fixed > far.fixed)
  assert.equal(near.probability, near.fixed / 1_000_000)
})

test('compact sign sketches rank signal-near neurons without reading dense SSD records', async (t) => {
  const near = neuron('near', 0)
  const far = {
    ...neuron('far', 0),
    identity: Float32Array.from(near.identity, (value) => value === 0 ? -1 : -value),
    relation: Float32Array.from(near.relation, (value) => value === 0 ? -1 : -value),
  }
  far.identityDigest = digestIdentity(far.identity)
  const created = await snapshot(t, [far, near])
  const repository = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  const signal = new Float32Array([...near.identity, ...near.relation])
  const before = repository.getStats()
  const ranked = repository.rankBySignal(signal, 2)
  const after = repository.getStats()

  assert.equal(ranked[0].id, 'near')
  assert.equal(ranked[0].distance, 0)
  assert.ok(ranked[1].distance > ranked[0].distance)
  assert.equal(after.io.coldReads, before.io.coldReads)
  assert.equal(after.resident.totalBytes, 0)
  assert.throws(
    () => repository.queryByRelationPages(signal, [0], 2),
    (error) => error.code === 'NEURON_RELATION_PAGE_SKETCH_UNAVAILABLE',
  )
  repository.close()
})

test('512/3584 manifests carry atlas-revision-free physical page sketches and rerank a bounded candidate set', async (t) => {
  const tierPolicy = policy({
    hotBudgetBytes: 80_000,
    warmBudgetBytes: 80_000,
    pageTargetBytes: 50_000,
    maximumPageBytes: 80_000,
    maximumWorkingSetNeurons: 8,
  })
  const firstDirectory = await temporaryDirectory(t)
  const secondDirectory = await temporaryDirectory(t)
  const values = [physicalNeuron('far', -1), physicalNeuron('near', 1)]
  const first = await writeNeuronTierSnapshot({
    directory: firstDirectory,
    modelFingerprint: 'thought-head:atlas-revision-1',
    physicalRepresentationSpaceId: 'RS-PHYSICAL-8X56',
    thoughtModelFingerprint: 'thought-model-a',
    representationProfile: 'adaptive-eighth-id-v1',
    layout: PHYSICAL_LAYOUT,
    neurons: values,
    policy: tierPolicy,
    identityDigestOf: digestIdentity,
  })
  const second = await writeNeuronTierSnapshot({
    directory: secondDirectory,
    modelFingerprint: 'thought-head:atlas-revision-99',
    physicalRepresentationSpaceId: 'RS-PHYSICAL-8X56',
    thoughtModelFingerprint: 'thought-model-a',
    representationProfile: 'adaptive-eighth-id-v1',
    layout: PHYSICAL_LAYOUT,
    neurons: values,
    policy: tierPolicy,
    identityDigestOf: digestIdentity,
  })

  assert.equal(first.manifest.relationPageSketches.format, 'relation-page-sign-projection-v1')
  assert.equal(first.manifest.representationProfile, 'adaptive-eighth-id-v1')
  assert.equal(first.manifest.relationPageSketches.pageCount, 56)
  assert.equal(first.manifest.relationPageSketches.pageLength, 64)
  assert.equal(first.manifest.relationPageSketches.physicalFingerprint, second.manifest.relationPageSketches.physicalFingerprint)
  assert.notEqual(first.manifest.fingerprint, second.manifest.fingerprint, 'thought-head lineage still belongs to the complete manifest CAS')
  assert.equal(first.manifest.index[0].relationPageSketches.length, 56 * 4)

  const repository = await openNeuronTierRepository({
    directory: firstDirectory,
    manifestPath: first.manifestPath,
    policy: tierPolicy,
    identityDigestOf: digestIdentity,
  })
  const signal = new Float32Array([...values[1].identity, ...values[1].relation])
  const before = repository.getStats()
  const result = repository.queryByRelationPages(signal, [0], 2)
  const after = repository.getStats()
  assert.equal(result.candidates[0].id, 'near')
  assert.equal(result.candidates[0].relationPageDistance, 0)
  assert.equal(result.candidates[1].relationPageDistance, 1)
  assert.equal(result.diagnostics.method, 'lsh-multiprobe-plus-relation-pages-v1')
  assert.deepEqual(result.diagnostics.selectedPageIndexes, [0])
  assert.equal(result.diagnostics.denseRecordReads, 0)
  assert.equal(after.io.coldReads, before.io.coldReads)
  assert.equal(after.resident.totalBytes, 0)
  assert.throws(
    () => repository.queryByRelationPages(signal, [0, 1, 2, 3, 4], 2),
    /en fazla 4/u,
  )
  repository.close()
})

test('profileless 512/3584 manifests never acquire adaptive relation-page semantics from dimensions alone', async (t) => {
  const tierPolicy = policy({
    hotBudgetBytes: 80_000,
    warmBudgetBytes: 80_000,
    pageTargetBytes: 50_000,
    maximumPageBytes: 80_000,
    maximumWorkingSetNeurons: 8,
  })
  const directory = await temporaryDirectory(t)
  const created = await writeNeuronTierSnapshot({
    directory,
    modelFingerprint: 'profileless-512-3584',
    physicalRepresentationSpaceId: 'RS-PROFILELESS-SAME-DIMENSIONS',
    thoughtModelFingerprint: 'thought-model-profileless',
    layout: PHYSICAL_LAYOUT,
    neurons: [physicalNeuron('profileless', 1)],
    policy: tierPolicy,
    identityDigestOf: digestIdentity,
  })
  assert.equal(created.manifest.representationProfile, undefined)
  assert.equal(created.manifest.relationPageSketches, undefined)
  assert.equal(Object.hasOwn(created.manifest.index[0], 'relationPageSketches'), false)

  const repository = await openNeuronTierRepository({
    directory,
    manifestPath: created.manifestPath,
    policy: tierPolicy,
    identityDigestOf: digestIdentity,
  })
  assert.equal(repository.manifest.representationProfile, null)
  assert.equal(repository.manifest.relationPageSketches, null)
  assert.throws(
    () => repository.queryByRelationPages(new Float32Array(4_096), [0], 1),
    (error) => error.code === 'NEURON_RELATION_PAGE_SKETCH_UNAVAILABLE',
  )
  repository.close()
})

test('historical profileless V2 page sketches stay readable for audit but cannot be queried as adaptive semantics', async (t) => {
  const tierPolicy = policy({
    hotBudgetBytes: 80_000,
    warmBudgetBytes: 80_000,
    pageTargetBytes: 50_000,
    maximumPageBytes: 80_000,
    maximumWorkingSetNeurons: 8,
  })
  const directory = await temporaryDirectory(t)
  const explicit = await writeNeuronTierSnapshot({
    directory,
    modelFingerprint: 'historical-ambiguous-v2',
    physicalRepresentationSpaceId: 'RS-HISTORICAL-AMBIGUOUS',
    thoughtModelFingerprint: 'thought-model-historical',
    representationProfile: 'adaptive-eighth-id-v1',
    layout: PHYSICAL_LAYOUT,
    neurons: [physicalNeuron('historical', 1)],
    policy: tierPolicy,
    identityDigestOf: digestIdentity,
  })
  const {
    fingerprint: _fingerprint,
    representationProfile: _representationProfile,
    ...ambiguousBody
  } = explicit.manifest
  const ambiguousFingerprint = createHash('sha256')
    .update(Buffer.from(`bombom-neuron-tier-manifest-v2\0${JSON.stringify(canonicalManifestValue(ambiguousBody))}`))
    .digest('hex')
  const ambiguousManifest = { ...ambiguousBody, fingerprint: ambiguousFingerprint }
  const ambiguousPath = `${directory}/manifests/${ambiguousFingerprint}.json`
  await writeFile(ambiguousPath, JSON.stringify(canonicalManifestValue(ambiguousManifest)))

  const repository = await openNeuronTierRepository({
    directory,
    manifestPath: ambiguousPath,
    policy: tierPolicy,
    identityDigestOf: digestIdentity,
  })
  assert.equal(repository.manifest.representationProfile, null)
  assert.equal(repository.manifest.relationPageProfileBinding, 'unverified-or-not-adaptive')
  assert.equal(repository.manifest.relationPageSketches, null)
  assert.throws(
    () => repository.queryByRelationPages(new Float32Array(4_096), [0], 1),
    (error) => error.code === 'NEURON_RELATION_PAGE_SKETCH_UNAVAILABLE',
  )
  repository.close()
})

test('10K/100K compact routing benchmark keeps top-128 selection bounded', () => {
  const query = '5a'.repeat(16)
  const entries = Array.from({ length: 100_000 }, (_, index) => ({
    id: `neuron-${String(index).padStart(6, '0')}`,
    routingSketch: createHash('sha256').update(`routing-${index}`).digest('hex').slice(0, 32),
  }))
  const brute = entries.slice(0, 10_000).map((entry) => ({
    id: entry.id,
    distance: [...entry.routingSketch].reduce((sum, nibble, index) => {
      const xor = Number.parseInt(nibble, 16) ^ Number.parseInt(query[index], 16)
      return sum + xor.toString(2).replaceAll('0', '').length
    }, 0) / 128,
  })).sort((left, right) => left.distance - right.distance || left.id.localeCompare(right.id)).slice(0, 128)

  const start10k = performance.now()
  const ranked10k = rankNeuronRoutingSketches(entries.slice(0, 10_000), query, 128, 128)
  const elapsed10k = performance.now() - start10k
  const start100k = performance.now()
  const ranked100k = rankNeuronRoutingSketches(entries, query, 128, 128)
  const elapsed100k = performance.now() - start100k

  assert.deepEqual(ranked10k, brute)
  assert.equal(ranked100k.length, 128)
  assert.deepEqual(ranked100k, [...ranked100k].sort((left, right) => left.distance - right.distance || left.id.localeCompare(right.id)))
  // Broad CI guardrails: algorithmic regressions to global N log N sorting are
  // caught without pretending a unit test is a production p95 benchmark.
  assert.ok(elapsed10k < 250, `10K sketch rank took ${elapsed10k.toFixed(1)}ms`)
  assert.ok(elapsed100k < 1_000, `100K sketch rank took ${elapsed100k.toFixed(1)}ms`)
})

test('100K production LSH query inspects a fixed bounded index slice under the tick target', () => {
  const query = '5a'.repeat(16)
  const entries = new Map(Array.from({ length: 100_000 }, (_, index) => {
    const entry = {
      id: `neuron-${String(index).padStart(6, '0')}`,
      routingSketch: createHash('sha256').update(`bounded-routing-${index}`).digest('hex').slice(0, 32),
    }
    return [entry.id, entry]
  }))
  const index = new NeuronRoutingSketchIndex(entries)
  // Warm JIT before timing the production query itself; index construction is
  // a one-time manifest-open cost, never a tick cost.
  index.query(query, 128, 128)
  const startedAt = performance.now()
  const result = index.query(query, 128, 128)
  const elapsed = performance.now() - startedAt

  assert.ok(result.candidates.length > 0 && result.candidates.length <= 128)
  assert.equal(result.diagnostics.method, 'lsh-multiprobe-v1')
  assert.ok(result.diagnostics.bucketProbes <= 68)
  assert.ok(result.diagnostics.inspectedIndexEntries <= 512)
  assert.ok(elapsed < 100, `100K bounded tick query took ${elapsed.toFixed(1)}ms`)
})

test('far/low-probability neuron uses one bounded record read and remains cold', async (t) => {
  const created = await snapshot(t, [neuron('a', 1), neuron('b', 2), neuron('c', 3)])
  const repository = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  const before = repository.getStats()
  const restored = await repository.get('b', {
    distance: 1,
    routeProbability: 0,
    familyProbability: 0,
    observedUseProbability: 0,
  })
  const after = repository.getStats()

  assert.equal(repository.tierOf('b'), 'cold')
  assert.equal(after.io.coldReads - before.io.coldReads, 1)
  assert.equal(after.io.fullPageReads - before.io.fullPageReads, 0)
  assert.ok(after.io.bytesRead - before.io.bytesRead < created.policy.maximumPageBytes)
  assert.deepEqual(exactFloatBytes(restored.identity), exactFloatBytes(neuron('b', 2).identity))
  assert.deepEqual(exactFloatBytes(restored.relation), exactFloatBytes(neuron('b', 2).relation))
  repository.close()
})

test('probable neurons promote cold->warm->hot while returned mutation cannot alter the cache', async (t) => {
  const source = neuron('likely', 1)
  const created = await snapshot(t, [source])
  const repository = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  const evidence = { distance: 0, routeProbability: 1, familyProbability: 1, observedUseProbability: 1 }
  const first = await repository.get('likely', evidence)
  assert.equal(repository.tierOf('likely'), 'warm')
  first.identity[0] = 99
  const second = await repository.get('likely', evidence)
  assert.equal(repository.tierOf('likely'), 'hot')
  assert.equal(second.identity[0], source.identity[0])
  second.identity[0] = -99
  assert.equal((await repository.get('likely', evidence)).identity[0], source.identity[0])
  assert.equal(repository.getStats().io.coldReads, 1)
  assert.equal(repository.getStats().io.fullPageReads, 0)
  repository.close()
})

test('hot budget applies deterministic utility-aware LRU demotion', async (t) => {
  const created = await snapshot(t, [neuron('a', 1), neuron('b', 2)], policy({
    hotBudgetBytes: 350,
    hotPromotionHits: 1,
    pageTargetBytes: 300,
    maximumPageBytes: 500,
  }))
  const repository = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  const likely = { distance: 0, routeProbability: 1, familyProbability: 1, observedUseProbability: 1 }
  await repository.get('a', likely)
  assert.equal(repository.tierOf('b'), 'cold', 'same SSD page must not make an unused neuron resident')
  await repository.get('b', likely)

  assert.equal(repository.getStats().tiers.hotNeurons, 1)
  assert.equal(repository.tierOf('a'), 'warm', 'oldest equal-utility record is demoted first')
  assert.equal(repository.tierOf('b'), 'hot')
  assert.ok(repository.getStats().resident.hotBytes <= created.policy.hotBudgetBytes)
  repository.close()
})

test('warm byte budget and logical-idle decay return unlikely records to SSD', async (t) => {
  const created = await snapshot(t, [neuron('a', 1), neuron('b', 2)], policy({
    warmBudgetBytes: 350,
    hotPromotionHits: 99,
    pageTargetBytes: 300,
    maximumPageBytes: 500,
  }))
  const repository = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  const likely = { distance: 0, routeProbability: 1, familyProbability: 1, observedUseProbability: 1 }
  await repository.get('a', likely)
  await repository.get('b', likely)
  assert.equal(repository.getStats().tiers.warmNeurons, 1)
  assert.equal(repository.tierOf('a'), 'cold')
  assert.equal(repository.tierOf('b'), 'warm')

  await repository.advanceLogicalClock(32)
  assert.equal(repository.tierOf('b'), 'cold')
  assert.equal(repository.getStats().resident.totalBytes, 0)
  repository.close()
})

test('working set is bounded, pinned for callback, and policy state replay is deterministic', async (t) => {
  const created = await snapshot(t, [neuron('a', 1), neuron('b', 2), neuron('c', 3), neuron('d', 4)])
  const repository = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  const requests = ['a', 'b', 'c', 'd'].map((id) => ({
    id,
    evidence: { distance: 0.2, routeProbability: 0.8, familyProbability: 0.5, observedUseProbability: 0.7 },
  }))
  const ids = await repository.withWorkingSet(requests, (lookup) => [...lookup.keys()])
  assert.deepEqual(ids, ['a', 'b', 'c'])
  const state = repository.exportPolicyState()
  repository.close()

  const restored = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
    restoredPolicyState: state,
  })
  assert.equal(restored.exportPolicyState().fingerprint, state.fingerprint)
  assert.equal(restored.getStats().logicalClock, 3)
  restored.close()
})

test('record/page corruption and stale identity digests fail closed', async (t) => {
  const created = await snapshot(t, [neuron('a', 1)])
  const page = created.manifest.pages[0]
  const pagePath = `${created.directory}/${page.relativePath}`
  const bytes = await readFile(pagePath)
  bytes[bytes.length - 1] ^= 0xff
  await writeFile(pagePath, bytes)

  const cold = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  await assert.rejects(
    cold.get('a', { distance: 1 }),
    (error) => error.code === 'NEURON_RECORD_CORRUPT',
  )
  const warm = await openNeuronTierRepository({
    directory: created.directory,
    manifestPath: created.manifestPath,
    policy: created.policy,
    identityDigestOf: digestIdentity,
  })
  await assert.rejects(
    warm.verifyPageForNeuron('a'),
    (error) => error.code === 'NEURON_PAGE_CORRUPT',
  )
  cold.close()
  warm.close()

  const invalid = neuron('bad', 4)
  invalid.identityDigest = 'not-the-vector-digest'
  await assert.rejects(
    snapshot(t, [invalid]),
    (error) => error.code === 'NEURON_IDENTITY_MISMATCH',
  )
})
