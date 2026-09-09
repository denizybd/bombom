import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer as createViteServer } from 'vite'

const SPACE = 'text-lab-64x64-v1'
const FAMILY = { representationSpaceId: SPACE, fingerprint: 'FAMILY-RUN-TRACE' }
const SOURCE = {
  sourceId: 'source-a',
  sourceDigest: 'source-digest-a',
  independenceGroupId: 'book-a',
  split: 'train',
}

function neuron(id) {
  return { id, identityDigest: `ID-${id}`, familyMemberships: [{ familyId: 'F-1', weight: 1 }] }
}

function tick(tickNumber, branch) {
  return { tick: tickNumber, branches: [branch] }
}

test('unmatched tick ends macro adjacency instead of joining transitions across the gap', async (t) => {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const { macroTraceFromThoughtResult } = await vite.ssrLoadModule('/runtime/core-runtime.mjs')
  const transitions = [
    { tick: 0, fromNeuronId: 'a', toNeuronId: 'b' },
    { tick: 2, fromNeuronId: 'c', toNeuronId: 'd' },
  ]
  const result = {
    __catalogNeurons: ['a', 'b', 'c', 'd'].map(neuron),
    episode: { runKey: 'run-gap', episodeId: 'episode-gap', temporaryBirths: [], branches: [{ id: 'b2', mass: 1, transitions }] },
    ticks: [
      tick(0, { branchId: 'b0', parentBranchId: null, beforeSignal: [1, 0], afterSignal: [.8, .2], macroExecution: null }),
      tick(1, { branchId: 'b1', parentBranchId: 'b0', beforeSignal: [.8, .2], afterSignal: [.8, .2], macroExecution: null }),
      tick(2, { branchId: 'b2', parentBranchId: 'b1', beforeSignal: [.8, .2], afterSignal: [.6, .4], macroExecution: null }),
    ],
  }
  assert.equal(macroTraceFromThoughtResult(result, SOURCE, FAMILY, 'semantic-gap'), null)
})

test('snapshot-bound compiled tick becomes one composite member for nested motif mining', async (t) => {
  const vite = await createViteServer({ appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })
  t.after(() => vite.close())
  const { macroTraceFromThoughtResult } = await vite.ssrLoadModule('/runtime/core-runtime.mjs')
  const macroHead = {
    fingerprint: 'MACRO-HEAD-1',
    macros: [{
      id: 'CM-a-b-c',
      lifecycle: 'active',
      expansionDigest: 'EXP-a-b-c',
      orderedExpansion: ['a', 'b', 'c'].map((id) => ({
        kind: 'raw-neuron', id, identityDigest: `ID-${id}`, familyIds: ['F-1'],
      })),
    }],
  }
  const transitions = [
    { tick: 0, fromNeuronId: 'a', toNeuronId: 'b' },
    { tick: 0, fromNeuronId: 'b', toNeuronId: 'c' },
    { tick: 1, fromNeuronId: 'c', toNeuronId: 'd' },
  ]
  const result = {
    __catalogNeurons: ['a', 'b', 'c', 'd'].map(neuron),
    episode: { runKey: 'run-nested', episodeId: 'episode-nested', temporaryBirths: [], branches: [{ id: 'b1', mass: 1, transitions }] },
    ticks: [
      tick(0, {
        branchId: 'b0', parentBranchId: null, beforeSignal: [1, 0], afterSignal: [.5, .5],
        macroExecution: {
          macroId: 'CM-a-b-c', macroHeadFingerprint: 'MACRO-HEAD-1', compiled: true,
          expandedConstituentIds: ['a', 'b', 'c'],
        },
      }),
      tick(1, { branchId: 'b1', parentBranchId: 'b0', beforeSignal: [.5, .5], afterSignal: [0, 1], macroExecution: null }),
    ],
  }
  const trace = macroTraceFromThoughtResult(result, SOURCE, FAMILY, 'semantic-nested', macroHead)
  assert.equal(trace.steps.length, 2)
  assert.deepEqual(trace.steps[0].member, {
    kind: 'composite-neuron', id: 'CM-a-b-c', identityDigest: 'EXP-a-b-c', familyIds: ['F-1'],
  })
  assert.equal(trace.steps[1].member.id, 'c')
})
