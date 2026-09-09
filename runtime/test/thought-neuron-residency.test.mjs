import assert from 'node:assert/strict'
import test from 'node:test'
import { createThoughtNeuronRoutingTopology } from '../thought-neuron-residency.mjs'

test('100K routing topology pays N only at bootstrap and keeps tick lookups bounded', () => {
  const neuronCount = 100_000
  const neurons = Array.from({ length: neuronCount }, (_, index) => ({
    id: `n-${String(index).padStart(6, '0')}`,
    familyMemberships: [{ familyId: `f-${index % 64}`, weight: 1 }],
  }))
  const habitRows = Array.from({ length: neuronCount }, (_, index) => ({
    fromNeuronId: neurons[index].id,
    edges: [{
      toNeuronId: neurons[(index + 1) % neuronCount].id,
      priorEvidence: 1,
      excitationEvidence: 0,
      inhibitionEvidence: 0,
    }],
  }))
  const topology = createThoughtNeuronRoutingTopology({
    model: { neurons, habitRows },
    head: { births: [], neuronEdges: [] },
    routingPreview: neurons.slice(0, 32).map((neuron) => ({ memberKey: neuron.id })),
  })
  const seeds = neurons.slice(20_000, 20_128).map((neuron) => neuron.id)

  // Warm the tiny query paths before enforcing the per-tick target.
  topology.membershipsFor(seeds, 128)
  const startedAt = performance.now()
  const memberships = topology.membershipsFor(seeds, 128)
  const edges = topology.topEdgesFrom(seeds, {
    maximumOrigins: 32,
    maximumPerOrigin: 8,
    maximumEntries: 256,
  })
  const families = [...memberships.rows.values()].flat().map((row) => row.familyId)
  const members = topology.topMembersForFamilies(families, {
    maximumFamilies: 16,
    maximumPerFamily: 4,
  })
  const elapsed = performance.now() - startedAt

  assert.equal(memberships.inspectedIndexEntries, 128)
  assert.ok(edges.originLookups <= 32)
  assert.ok(edges.inspectedIndexEntries <= 256)
  assert.ok(members.familyLookups <= 16)
  assert.ok(members.inspectedIndexEntries <= 64)
  assert.ok(elapsed < 100, `100K topology tick lookup took ${elapsed.toFixed(1)}ms`)
})

test('SSD candidate topology keeps promoted birth entry and outgoing shortcuts reachable', () => {
  const topology = createThoughtNeuronRoutingTopology({
    model: {
      neurons: [
        { id: 'origin', familyMemberships: [{ familyId: 'family-a', weight: 1 }] },
        { id: 'target', familyMemberships: [{ familyId: 'family-b', weight: 1 }] },
      ],
      habitRows: [],
    },
    head: {
      neuronEdges: [],
      births: [{
        id: 'born-shortcut',
        familyMemberships: [{ familyId: 'family-a', weight: 1 }],
        entryEvidence: [{ memberKey: 'origin', weight: 0.8 }],
        outgoingEvidence: [{ memberKey: 'target', weight: 0.6 }],
      }],
    },
  })
  assert.deepEqual(topology.topEdgesFrom(['origin']).rows, [{
    fromId: 'origin', toId: 'born-shortcut', strength: 0.8, probability: 1,
  }])
  assert.deepEqual(topology.topEdgesFrom(['born-shortcut']).rows, [{
    fromId: 'born-shortcut', toId: 'target', strength: 0.6, probability: 1,
  }])
})
