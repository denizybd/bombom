import { readdir, stat } from 'node:fs/promises'
import path from 'node:path'
import {
  openNeuronTierRepository,
  writeNeuronTierSnapshot,
} from './neuron-tier-store.mjs'

async function directoryBytes(directory) {
  const entries = await readdir(directory, { withFileTypes: true }).catch((error) => {
    if (error?.code === 'ENOENT') return []
    throw error
  })
  let total = 0
  for (const entry of entries) {
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) total += await directoryBytes(target)
    else if (entry.isFile()) total += (await stat(target)).size
  }
  return total
}

export function thoughtNeuronTierDirectory(dataDirectory) {
  return path.join(path.resolve(dataDirectory), 'neuron-tiers')
}

export function measureThoughtNeuronResidencyBytes(dataDirectory) {
  return directoryBytes(thoughtNeuronTierDirectory(dataDirectory))
}

function allPreparedNeurons(model, head) {
  const byId = new Map(model.neurons.map((neuron) => [neuron.id, neuron]))
  for (const birth of head.births ?? []) {
    if (byId.has(birth.id)) continue
    byId.set(birth.id, {
      id: birth.id,
      identity: Float32Array.from(birth.identity),
      relation: Float32Array.from(birth.relation),
      identityDigest: birth.identityDigest,
      familyMemberships: (birth.familyMemberships ?? []).map((membership) => ({ ...membership })),
      provenance: { kind: 'nursery-promotion', nurseryId: birth.nurseryId },
    })
  }
  return [...byId.values()].sort((left, right) => left.id.localeCompare(right.id))
}

function boundedInsert(rows, value, maximum, compare) {
  let position = 0
  while (position < rows.length && compare(rows[position], value) <= 0) position += 1
  if (position >= maximum) return
  rows.splice(position, 0, value)
  if (rows.length > maximum) rows.pop()
}

function membershipOrder(left, right) {
  return right.weight - left.weight || left.familyId.localeCompare(right.familyId) || left.id.localeCompare(right.id)
}

function edgeOrder(left, right) {
  return right.strength - left.strength || left.toId.localeCompare(right.toId)
}

/**
 * Payload-free topology used by the SSD candidate gate. All N-sized scans are
 * paid once during bootstrap/open; tick methods inspect fixed-size slices.
 */
export function createThoughtNeuronRoutingTopology({ model, head, routingPreview = [] }) {
  const validIds = new Set(model.neurons.map((neuron) => neuron.id))
  for (const birth of head.births ?? []) validIds.add(birth.id)
  const memberFamilies = new Map()
  const familyMembers = new Map()
  const recordMemberships = (id, memberships = []) => {
    const rows = memberships
      .map((membership) => ({
        id,
        familyId: String(membership.familyId),
        weight: Math.max(0, Math.min(1, Number(membership.weight) || 0)),
      }))
      .filter((membership) => membership.familyId && membership.weight > 0)
      .sort(membershipOrder)
      .slice(0, 5)
    memberFamilies.set(id, rows)
    for (const row of rows) {
      const members = familyMembers.get(row.familyId) ?? []
      boundedInsert(members, row, 32, membershipOrder)
      familyMembers.set(row.familyId, members)
    }
  }
  for (const neuron of model.neurons) recordMemberships(neuron.id, neuron.familyMemberships)
  for (const birth of head.births ?? []) recordMemberships(birth.id, birth.familyMemberships)

  const edgesByOrigin = new Map()
  const recordEdge = (fromId, toId, strength) => {
    if (!validIds.has(fromId) || !validIds.has(toId) || fromId === toId || !(strength > 0)) return
    const rows = edgesByOrigin.get(fromId) ?? []
    const previous = rows.find((row) => row.toId === toId)
    if (previous) {
      if (strength > previous.strength) {
        previous.strength = strength
        rows.sort(edgeOrder)
      }
    } else boundedInsert(rows, { toId, strength }, 96, edgeOrder)
    edgesByOrigin.set(fromId, rows)
  }
  for (const row of model.habitRows ?? []) {
    for (const edge of row.edges ?? []) {
      recordEdge(
        row.fromNeuronId,
        edge.toNeuronId,
        Math.max(0, (Number(edge.priorEvidence) || 0) + (Number(edge.excitationEvidence) || 0)) /
          (1 + Math.max(0, Number(edge.inhibitionEvidence) || 0)),
      )
    }
  }
  for (const edge of head.neuronEdges ?? []) {
    recordEdge(
      edge.fromMemberKey,
      edge.toMemberKey,
      Math.max(0, Number(edge.excitation) || 0) / (1 + Math.max(0, Number(edge.inhibition) || 0)),
    )
  }
  // Nursery promotion keeps the temporary neuron's validated entry/exit
  // topology. The SSD gate must see the same shortcuts as the in-memory
  // Thought Runtime index or the promoted neuron becomes practically cold.
  for (const birth of head.births ?? []) {
    for (const entry of birth.entryEvidence ?? []) {
      recordEdge(entry.memberKey, birth.id, Math.max(0, Number(entry.weight) || 0))
    }
    for (const entry of birth.outgoingEvidence ?? []) {
      recordEdge(birth.id, entry.memberKey, Math.max(0, Number(entry.weight) || 0))
    }
  }
  for (const rows of edgesByOrigin.values()) {
    const maximum = rows[0]?.strength ?? 1
    for (const row of rows) row.probability = Math.max(0, Math.min(1, row.strength / maximum))
  }
  const previewIds = Object.freeze([...new Set(routingPreview.map((entry) =>
    typeof entry === 'string' ? entry : entry?.memberKey))]
    .filter((id) => validIds.has(id)).slice(0, 32))

  return Object.freeze({
    previewIds,
    membershipsFor(ids, maximumIds = 128) {
      const retained = [...new Set(ids)].slice(0, maximumIds)
      return {
        rows: new Map(retained.map((id) => [id, (memberFamilies.get(id) ?? []).map((row) => ({ ...row }))])),
        inspectedIndexEntries: retained.length,
      }
    },
    topEdgesFrom(ids, { maximumOrigins = 32, maximumPerOrigin = 8, maximumEntries = 256 } = {}) {
      const origins = [...new Set(ids)].slice(0, maximumOrigins)
      const rows = []
      let inspectedIndexEntries = 0
      for (const fromId of origins) {
        for (const edge of (edgesByOrigin.get(fromId) ?? []).slice(0, maximumPerOrigin)) {
          inspectedIndexEntries += 1
          rows.push({ fromId, ...edge })
          if (rows.length >= maximumEntries) break
        }
        if (rows.length >= maximumEntries) break
      }
      return { rows, inspectedIndexEntries, originLookups: origins.length }
    },
    topMembersForFamilies(familyIds, { maximumFamilies = 16, maximumPerFamily = 4 } = {}) {
      const retained = [...new Set(familyIds)].slice(0, maximumFamilies)
      const rows = []
      let inspectedIndexEntries = 0
      for (const familyId of retained) {
        for (const member of (familyMembers.get(familyId) ?? []).slice(0, maximumPerFamily)) {
          inspectedIndexEntries += 1
          rows.push({ familyId, id: member.id, weight: member.weight })
        }
      }
      return { rows, inspectedIndexEntries, familyLookups: retained.length }
    },
    diagnostics: Object.freeze({
      memberCount: validIds.size,
      familyCount: familyMembers.size,
      transitionOriginCount: edgesByOrigin.size,
      previewCount: previewIds.length,
      maximumEdgesPerOrigin: 96,
      maximumMembersPerFamily: 32,
    }),
  })
}

/**
 * Opt-in, non-destructive bridge for an existing ThoughtModel/ThoughtHead.
 * It writes a content-addressed SSD view and leaves the source snapshot intact.
 */
export async function createThoughtNeuronResidencySnapshot({
  dataDirectory,
  model,
  head,
  representationSpaceId,
  representationProfile = null,
  policy,
  maximumSnapshotBytes,
  residencyModelFingerprint = `${model.fingerprint}:${head.fingerprint}`,
}) {
  const directory = thoughtNeuronTierDirectory(dataDirectory)
  const patches = new Map(head.contextPatches.map((patch) => [patch.neuronId, patch.delta]))
  const neurons = allPreparedNeurons(model, head).map((neuron) => {
    const routingRelationDelta = patches.get(neuron.id)
    return routingRelationDelta === undefined ? neuron : { ...neuron, routingRelationDelta }
  })
  return writeNeuronTierSnapshot({
    directory,
    modelFingerprint: residencyModelFingerprint,
    physicalRepresentationSpaceId: representationSpaceId,
    thoughtModelFingerprint: model.fingerprint,
    representationProfile,
    layout: model.layout,
    neurons,
    policy,
    maximumSnapshotBytes,
  })
}

export async function openThoughtNeuronResidency({
  dataDirectory,
  manifestFingerprint,
  manifestPath,
  expectedModelFingerprint,
  expectedRepresentationSpaceId,
  expectedRepresentationProfile,
  expectedThoughtModelFingerprint,
  policy,
  restoredPolicyState,
  routingTopology = null,
}) {
  const repository = await openNeuronTierRepository({
    directory: thoughtNeuronTierDirectory(dataDirectory),
    manifestFingerprint,
    manifestPath,
    policy,
    restoredPolicyState,
  })
  if (expectedModelFingerprint && repository.manifest.modelFingerprint !== expectedModelFingerprint) {
    repository.close()
    const error = new TypeError('Nöron residency manifesti farklı ThoughtModel/ThoughtHead tabanına ait.')
    error.code = 'NEURON_TIER_STALE_BASE'
    throw error
  }
  if (repository.manifest.version >= 2) {
    if (
      expectedRepresentationSpaceId &&
      repository.manifest.physicalRepresentationSpaceId !== expectedRepresentationSpaceId
    ) {
      repository.close()
      const error = new TypeError('Nöron residency manifesti farklı fiziksel temsil uzayına ait.')
      error.code = 'NEURON_TIER_REPRESENTATION_MISMATCH'
      throw error
    }
    if (
      expectedRepresentationProfile !== undefined &&
      (repository.manifest.representationProfile ?? null) !== (expectedRepresentationProfile ?? null)
    ) {
      repository.close()
      const error = new TypeError('Nöron residency manifesti farklı açık temsil profiline ait.')
      error.code = 'NEURON_TIER_PROFILE_MISMATCH'
      throw error
    }
    if (
      expectedThoughtModelFingerprint &&
      repository.manifest.thoughtModelFingerprint !== expectedThoughtModelFingerprint
    ) {
      repository.close()
      const error = new TypeError('Nöron residency manifesti farklı ThoughtModel gövdesine ait.')
      error.code = 'NEURON_TIER_THOUGHT_MODEL_MISMATCH'
      throw error
    }
  }
  return {
    manifest: repository.manifest,
    repository,
    /**
     * Integration seam: sparse H/LSH/family indexes first produce <=128 ids;
     * only those vectors are materialized before the existing synchronous tick.
     */
    withCandidateLookup(requests, callback, options) {
      return repository.withWorkingSet(requests, callback, options)
    },
    rankBySignal(signal, maximum) {
      return repository.rankBySignal(signal, maximum)
    },
    queryBySignal(signal, maximum) {
      return repository.queryBySignal(signal, maximum)
    },
    queryByRelationPages(signal, pageIndexes, maximum) {
      return repository.queryByRelationPages(signal, pageIndexes, maximum)
    },
    sampleIds(seed, maximum) {
      return repository.sampleIds(seed, maximum)
    },
    routingTopology,
  }
}
