import test from 'node:test'
import assert from 'node:assert/strict'
import { createTrainingService } from '../http-server.mjs'
import { createNeuronAtlas } from '../neuron-atlas.mjs'
import { normalizeTextSource } from '../source-policy.mjs'
import { encodeArtifactValue } from '../../runtime/artifacts.mjs'
import { PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND } from '../../runtime/plastic-region-cache.mjs'
import { createLanguageOutputScaffold, LANGUAGE_OUTPUT_FAMILY_IDS } from '../../src/language-output/scaffold.ts'
import { createPlasticRegionAtlas } from '../../src/adaptive-regions/index.ts'
import { temporaryDirectory, listenService, jsonFetch } from './helpers.mjs'

function familyRun(sourceIds) {
  const [firstSourceId, secondSourceId] = sourceIds
  const profile = (memberKey, text, sourceId, kind = 'raw') => ({
    memberKey,
    kind,
    representationSpaceId: 'RS-test',
    behaviorSpaceId: 'BS-test',
    vector: { dimension: 4, entries: [{ index: 0, value: 1 }], norm: 1 },
    diagnostics: {},
    regionAffinities: [],
    evidence: [{ memberKey, analysisRunId: `analysis-${sourceId}`, sourceId, sourceDigest: `digest-${sourceId}`, neuronId: `neuron-${memberKey}`, paragraphIndex: 0, start: 0, end: text.length, text }],
    identityDigest: `identity-${memberKey}`,
    text,
    derivedFromMemberKeys: [],
  })
  const family = (id, region, memberKeys, fields = {}) => ({
    id,
    fingerprint: `fingerprint-${id}`,
    behaviorSpaceId: 'BS-test',
    region,
    regionVerified: false,
    lifecycle: 'stable',
    lifecycleReason: 'İki bağımsız kaynaktan gelen benzer davranışlar kararlı kaldı.',
    prototype: { dimension: 4, entries: [{ index: 1, value: 0.75 }], norm: 0.75 },
    prototypeDigest: `prototype-${id}`,
    memberKeys,
    coreMemberKeys: memberKeys.slice(0, 1),
    independentEvidence: [],
    supportCount: memberKeys.length,
    sourceCount: 2,
    cohesion: 0.82,
    separation: 0.71,
    stability: 0.93,
    utility: 0.12,
    predecessorFamilyIds: [],
    ...fields,
  })
  const longText = `yüksek üyelik ${'uzun bağlam '.repeat(40)}`
  return {
    format: 'noron-aileleri-run',
    version: 1,
    id: 'family-run-inspection',
    fingerprint: 'family-run-fingerprint',
    createdAt: '2026-08-29T00:00:00.000Z',
    representationSpaceId: 'RS-test',
    behaviorSpaceId: 'BS-test',
    config: {},
    inputManifest: { entries: [], corpusFingerprint: 'corpus-test' },
    profiles: [
      profile('member-high', longText, firstSourceId),
      profile('member-low', 'düşük üyelik', secondSourceId, 'synthesis'),
      profile('member-event', 'kesti', secondSourceId),
    ],
    families: [
      family('family-content', 'content', ['member-high', 'member-low']),
      family('family-event', 'event', ['member-event'], { lifecycle: 'strengthening', supportCount: 1, utility: null }),
    ],
    memberships: [
      { memberKey: 'member-high', familyId: 'family-content', rank: 1, score: 0.95, weight: 0.7, core: true, breakdown: {} },
      { memberKey: 'member-low', familyId: 'family-content', rank: 2, score: 0.62, weight: 0.3, core: false, breakdown: {} },
      { memberKey: 'member-event', familyId: 'family-event', rank: 1, score: 0.88, weight: 1, core: true, breakdown: {} },
    ],
    neuronTransitions: [{ fromMemberKey: 'member-high', toMemberKey: 'member-event', probability: 1 }],
    familyTransitions: [
      { id: 'transition-low', fromFamilyId: 'family-content', toFamilyId: 'family-content', priorEvidence: 1, experienceEvidence: 0, inhibitionEvidence: 0, sourceCount: 1, supportCount: 1, probability: 0.2 },
      { id: 'transition-high', fromFamilyId: 'family-content', toFamilyId: 'family-event', priorEvidence: 3, experienceEvidence: 2, inhibitionEvidence: 0.1, sourceCount: 2, supportCount: 5, probability: 0.8 },
    ],
    utilityReports: [{
      familyId: 'family-content', mode: 'source-holdout-diagnostic', status: 'validated', caseCount: 16, sourceFoldCount: 2,
      fullLoss: 0.4, ablatedLoss: 0.52, deltaUtility: 0.12, confidenceLow: 0.05, confidenceHigh: 0.19,
      folds: [{ heldOutSourceId: firstSourceId, heldOutSourceDigest: 'digest-a', caseCount: 8, fullLoss: 0.4, ablatedLoss: 0.52, deltaUtility: 0.12 }],
      explanation: 'Kaynak dışı sınamada yararlı.',
    }],
    coalition: { tick: 0, activeFamilyCount: 0, coalitions: [], unmatchedReason: null },
    routingPreview: null,
    warnings: [],
    stats: { distinctSourceCount: 2 },
    denseProbe: new Float32Array([0.25, -0.5, 0.75]),
  }
}

function forbiddenProjectionKeys(value, found = []) {
  if (Array.isArray(value)) {
    for (const item of value) forbiddenProjectionKeys(item, found)
    return found
  }
  if (!value || typeof value !== 'object') return found
  for (const [key, item] of Object.entries(value)) {
    if (['payload', 'prototype', 'prototypeDigest', 'profiles', 'memberships', 'vector', 'entries', 'identity', 'relation', '$type', 'denseProbe'].includes(key)) found.push(key)
    forbiddenProjectionKeys(item, found)
  }
  return found
}

test('family inspection exposes a bounded, vector-free projection of the latest snapshot artifact', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const first = service.store.createSource(normalizeTextSource({ text: 'Ali ekmeği kesti.', title: 'Birinci gerçek kaynak' }, service.store.getConfig())).source
  const second = service.store.createSource(normalizeTextSource({ text: 'Ayşe kağıdı kesti.', title: 'İkinci gerçek kaynak' }, service.store.getConfig())).source
  const run = familyRun([first.id, second.id])
  const artifact = service.store.createArtifact({
    id: 'family-artifact',
    jobId: 'job-family',
    kind: 'family-run',
    contentHash: 'family-artifact-hash',
    payload: {
      format: 'bombom-runtime-artifact',
      version: 1,
      bytes: 1234,
      metadata: { fingerprint: run.fingerprint },
      value: encodeArtifactValue(run),
    },
  })
  const runtimeFallback = service.store.createArtifact({
    id: 'large-runtime-fallback',
    kind: 'runtime-model-state',
    payload: { format: 'bombom-runtime-artifact', version: 1, value: { familyRun: encodeArtifactValue(run), padding: 'x'.repeat(512 * 1024) } },
  })
  const unrelated = service.store.createArtifact({
    id: 'large-unrelated-artifact',
    kind: 'text-lab-analysis',
    payload: { padding: 'y'.repeat(512 * 1024) },
  })
  const snapshot = service.store.createSnapshot({ id: 'family-snapshot', label: '14B ilk aile deneyi', status: 'rejected', artifactIds: [runtimeFallback.id, unrelated.id, artifact.id], sourceIds: [first.id, second.id] })
  const base = await listenService(t, service)

  const openedArtifactIds = []
  const getArtifact = service.store.getArtifact.bind(service.store)
  service.store.getArtifact = (artifactId) => {
    openedArtifactIds.push(artifactId)
    return getArtifact(artifactId)
  }

  const { response, body } = await jsonFetch(base, '/api/families/inspection?limit=1&memberLimit=1&transitionLimit=1')
  assert.equal(response.status, 200)
  assert.equal(body.available, true)
  assert.equal(body.selection.selectedBy, 'latest-snapshot')
  assert.equal(body.selection.snapshot.id, snapshot.id)
  assert.equal(body.selection.artifact.id, artifact.id)
  assert.equal(body.run.fingerprint, run.fingerprint)
  assert.deepEqual(body.summary.byLifecycle, [{ key: 'stable', count: 1 }, { key: 'strengthening', count: 1 }])
  assert.equal(body.summary.familyCount, 2)
  assert.equal(body.summary.validatedUtilityFamilyCount, 1)
  assert.deepEqual(body.pagination, { offset: 0, limit: 1, total: 2, returned: 1, hasMore: true, nextOffset: 1 })
  assert.equal(body.families.length, 1)
  assert.equal(body.families[0].topMembers.length, 1)
  assert.equal(body.families[0].topMembers[0].memberKey, 'member-high')
  assert.equal(body.families[0].topMembers[0].softScore, 0.95)
  assert.equal(body.families[0].topMembers[0].sources[0].name, 'Birinci gerçek kaynak')
  assert.equal(body.families[0].topMembers[0].textTruncated, true)
  assert.ok(body.families[0].topMembers[0].text.length <= 240)
  assert.deepEqual(body.families[0].outgoingTransitions.map((item) => item.id), ['transition-high'])
  assert.equal(body.families[0].outgoingTransitions[0].toRegion, 'event')
  assert.equal(body.families[0].utilityEvidence.status, 'validated')
  assert.deepEqual(forbiddenProjectionKeys(body), [])
  assert.doesNotMatch(JSON.stringify(body), /bombom\/float32-base64-v1/u)
  assert.deepEqual(openedArtifactIds, [artifact.id], 'selection must open only the authoritative family payload')

  openedArtifactIds.length = 0
  const atlasResult = await jsonFetch(base, `/api/neuron-atlas?snapshotId=${encodeURIComponent(snapshot.id)}`)
  assert.equal(atlasResult.response.status, 200)
  assert.equal(atlasResult.body.format, 'bombom-neuron-atlas')
  assert.equal(atlasResult.body.available, true)
  assert.equal(atlasResult.body.selection.snapshot.id, snapshot.id)
  assert.equal(atlasResult.body.run.fingerprint, run.fingerprint)
  assert.deepEqual(atlasResult.body.stats, {
    familyCount: 8,
    neuronCount: 24,
    membershipCount: 24,
    familyTransitionCount: 9,
    neuronTransitionCount: 1,
    learnedFamilyCount: 2,
    languageOutputFamilyCount: 6,
    languageOutputSeedNeuronCount: 21,
    languageOutputNurseryCount: 0,
    languageOutputProvisionalNeuronCount: 0,
    distinctSourceCount: 2,
    validatedUtilityFamilyCount: 1,
    mergeProposedFamilyCount: 0,
    priorEdgeCount: 9,
    experienceEdgeCount: 1,
    inhibitionEdgeCount: 1,
  })
  assert.equal(atlasResult.body.families.length, 8)
  assert.equal(atlasResult.body.neurons.length, 24)
  assert.equal(atlasResult.body.links.memberships.length, 24)
  assert.equal(atlasResult.body.links.familyTransitions.length, 9)
  assert.equal(atlasResult.body.links.neuronTransitions.length, 1)
  assert.equal(atlasResult.body.neurons[0].primaryFamily, 0)
  assert.deepEqual(atlasResult.body.links.memberships[0], [0, 0, 0.95, 0.7, 1, 1])
  assert.equal(atlasResult.body.families.filter((family) => family.layoutZone === 'language-surface').length, 6)
  assert.equal(atlasResult.body.projection.languageOutput.persisted, false)
  assert.doesNotMatch(JSON.stringify(atlasResult.body), /"(?:vector|prototype|identity|relation|evidence)"\s*:/u)
  assert.deepEqual(openedArtifactIds, [artifact.id], 'atlas must also open exactly one authoritative family payload')

  const repeatedAtlas = await jsonFetch(base, `/api/neuron-atlas?snapshotId=${encodeURIComponent(snapshot.id)}`)
  assert.equal(repeatedAtlas.response.status, 200)
  assert.equal(repeatedAtlas.body.run.fingerprint, run.fingerprint)
  assert.deepEqual(openedArtifactIds, [artifact.id], 'an immutable atlas cache hit must not inflate the FamilyRun again')
})

test('family inspection returns an explicit empty result before the first family run', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)
  const { response, body } = await jsonFetch(base, '/api/families/inspection')
  assert.equal(response.status, 200)
  assert.equal(body.available, false)
  assert.equal(body.selection.selectedBy, 'none')
  assert.equal(body.summary.familyCount, 0)
  assert.deepEqual(body.families, [])
})

test('3B atlas projects the persisted plastic output head without moving a core neuron into the language island', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  t.after(() => service.close())
  const run = familyRun(['source-a', 'source-b'])
  const scaffold = createLanguageOutputScaffold()
  const familyArtifact = service.store.createArtifact({
    id: 'language-atlas-family', kind: 'family-run', contentHash: 'language-atlas-family-hash',
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(run) },
  })
  const scaffoldArtifact = service.store.createArtifact({
    id: 'language-atlas-scaffold', kind: 'language-output-scaffold', contentHash: scaffold.fingerprint,
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(scaffold) },
  })
  const outputNeuron = {
    id: '@language-output/provisional/test', fingerprint: 'provisional-test', status: 'provisional', kind: 'surface-controller',
    normalizedSurface: 'değil', representativeSurface: 'değil', familyId: LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation,
    familyEvidence: 2.7, meanFamilyConfidence: .9, observationCount: 3,
    sourceDigests: ['digest-a', 'digest-b'], sourceGroupIds: ['group-a', 'group-b'], exampleIds: ['e1', 'e2', 'e3'],
    sourceProposalIds: ['p1'], promotedAtRevision: 1, reversible: true, physicalIdentityAllocated: false,
  }
  const outputHead = {
    format: 'language-output-head', version: 1, id: 'output-head-test', fingerprint: 'output-head-test', revision: 1,
    scaffoldFingerprint: scaffold.fingerprint, representationSpaceId: run.representationSpaceId,
    familyRunFingerprint: run.fingerprint, membershipEvidence: [{
      memberKey: 'member-high', familyId: LANGUAGE_OUTPUT_FAMILY_IDS.connectiveNegation,
      excitatoryEvidence: 2, inhibitoryEvidence: 0, score: Math.log(3), updateCount: 2,
      sourceDigests: ['digest-a'], sourceGroupIds: ['group-a'], classifierConfidence: .95,
    }], transitionEvidence: [{
      kind: 'family-transition',
      fromKey: LANGUAGE_OUTPUT_FAMILY_IDS.verbTense,
      toKey: LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking,
      excitatoryEvidence: 2,
      inhibitoryEvidence: 0,
      score: Math.log(3),
      updateCount: 4,
      sourceDigests: ['digest-a', 'digest-b'],
      sourceGroupIds: ['group-a', 'group-b'],
    }], nursery: [], outputNeurons: [outputNeuron],
  }
  const headArtifact = service.store.createArtifact({
    id: 'language-atlas-head', kind: 'language-output-head', contentHash: outputHead.fingerprint,
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(outputHead) },
  })
  const snapshot = service.store.createSnapshot({
    id: 'language-atlas-snapshot', status: 'candidate',
    artifactIds: [familyArtifact.id, scaffoldArtifact.id, headArtifact.id],
  })

  const atlas = createNeuronAtlas(service.store, { snapshotId: snapshot.id })
  const core = atlas.neurons.find((neuron) => neuron.id === 'member-high')
  const autonomous = atlas.neurons.find((neuron) => neuron.id === outputNeuron.id)
  const languageFamily = atlas.families.find((family) => family.id === outputNeuron.familyId)
  assert.equal(atlas.projection.languageOutput.persisted, true)
  assert.equal(atlas.projection.languageOutput.compatible, true)
  assert.equal(atlas.stats.languageOutputProvisionalNeuronCount, 1)
  const learnedLanguageEdge = atlas.links.familyTransitions.find((edge) => (
    atlas.families[edge[0]]?.id === LANGUAGE_OUTPUT_FAMILY_IDS.verbTense &&
    atlas.families[edge[1]]?.id === LANGUAGE_OUTPUT_FAMILY_IDS.caseMarking
  ))
  assert.ok(learnedLanguageEdge)
  assert.equal(learnedLanguageEdge[3], 0)
  assert.equal(learnedLanguageEdge[4], 2)
  assert.equal(core.primaryFamily, 0, 'plastic language membership must not replace the core primary family')
  assert.equal(autonomous.kind, 'language-output')
  assert.equal(autonomous.primaryFamily, languageFamily.index)
  assert.ok(atlas.links.memberships.some((membership) => membership[0] === core.index && membership[1] === languageFamily.index))
})

test('3B atlas binds the vector-free concept surface to the exact selected snapshot lineage', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const run = familyRun(['source-a', 'source-b'])
  const familyArtifact = service.store.createArtifact({
    id: 'concept-atlas-family', kind: 'family-run', contentHash: 'concept-atlas-family-hash',
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(run) },
  })
  const macro = (id, lifecycle, expansion, fields = {}) => ({
    id, kind: 'sequence-macro', lifecycle, orderedExpansion: expansion,
    depth: 1, observationCount: 8, independentGroupCount: 3, sourceCount: 3,
    metrics: { fidelity: .97, shadowCompressionGain: 4 },
    mdl: { compressionGain: 6, compressionRatio: .42 },
    entryPrototype: { dimension: 4_096, vector: new Array(4_096).fill(.123456789), digest: `${id}-entry` },
    effectPrototype: new Array(4_096).fill(.234567891),
    ...fields,
  })
  const child = macro('CMACRO-CHILD', 'shadow-validated', [
    { kind: 'raw-neuron', id: 'member-high' },
    { kind: 'raw-neuron', id: 'member-event' },
  ])
  const parent = macro('CMACRO-PARENT', 'active', [
    { kind: 'composite-neuron', id: child.id },
    { kind: 'raw-neuron', id: 'member-low' },
  ], { depth: 2, observationCount: 11 })
  const selectedHead = {
    format: 'concept-macro-head', version: 1, id: 'selected-concept-head', revision: 4,
    fingerprint: 'selected-concept-fingerprint', representationSpaceId: run.representationSpaceId,
    familyRunFingerprint: run.fingerprint, macros: [parent, child],
  }
  const selectedMacroArtifact = service.store.createArtifact({
    id: 'selected-concept-artifact', kind: 'concept-macro-head', contentHash: selectedHead.fingerprint,
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(selectedHead) },
  })
  const snapshot = service.store.createSnapshot({
    id: 'concept-atlas-snapshot', status: 'rejected', artifactIds: [familyArtifact.id, selectedMacroArtifact.id],
  })
  const unrelatedHead = {
    ...selectedHead, id: 'global-newer-head', fingerprint: 'global-newer-fingerprint',
    macros: [macro('CMACRO-GLOBAL-ONLY', 'active', [{ kind: 'raw-neuron', id: 'member-high' }])],
  }
  service.store.createArtifact({
    id: 'global-newer-concept-artifact', kind: 'concept-macro-head', contentHash: unrelatedHead.fingerprint,
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(unrelatedHead) },
  })

  const opened = []
  const getArtifact = service.store.getArtifact.bind(service.store)
  service.store.getArtifact = (id) => { opened.push(id); return getArtifact(id) }
  const base = await listenService(t, service)
  const first = await jsonFetch(base, `/api/neuron-atlas?snapshotId=${encodeURIComponent(snapshot.id)}`)
  assert.equal(first.response.status, 200)
  assert.equal(first.body.projection.conceptMacros.headFingerprint, selectedHead.fingerprint)
  assert.equal(first.body.projection.conceptMacros.snapshotStatus, 'rejected')
  assert.equal(first.body.projection.conceptMacros.runtimeEligible, false)
  assert.equal(first.body.projection.conceptMacros.activeCount, 0)
  assert.equal(first.body.stats.conceptMacroCount, 2)
  assert.equal(first.body.stats.conceptMacroExpansionLinkCount, 4)
  assert.ok(first.body.neurons.some((neuron) => neuron.id === parent.id && neuron.layoutZone === 'concept-surface'))
  assert.ok(!first.body.neurons.some((neuron) => neuron.id === 'CMACRO-GLOBAL-ONLY'))
  const parentNode = first.body.neurons.find((neuron) => neuron.id === parent.id)
  assert.equal(parentNode.conceptMacro.runtimeStatus, 'rejected-snapshot')
  assert.equal(parentNode.conceptMacro.runtimeActive, false)
  assert.equal(parentNode.conceptMacro.fidelity, .97)
  assert.equal(parentNode.conceptMacro.compressionGain, 6)
  assert.equal(parentNode.conceptMacro.sourceGroupCount, 3)
  assert.ok(first.body.links.macroExpansions.some((edge) => {
    const [from, to, , nested] = edge
    return first.body.neurons[from]?.id === parent.id && first.body.neurons[to]?.id === child.id && nested === 1
  }))
  assert.doesNotMatch(JSON.stringify(first.body), /entryPrototype|effectPrototype|"vector"|0\.123456789/u)
  assert.equal(first.response.headers.get('etag'), `"${first.body.fingerprint}"`)
  assert.notEqual(first.body.fingerprint, run.fingerprint)
  assert.deepEqual(opened, [familyArtifact.id, selectedMacroArtifact.id], 'atlas may open only the family and macro artifacts named by the selected snapshot')

  const repeated = await jsonFetch(base, `/api/neuron-atlas?snapshotId=${encodeURIComponent(snapshot.id)}`)
  assert.equal(repeated.body.fingerprint, first.body.fingerprint)
  assert.deepEqual(opened, [familyArtifact.id, selectedMacroArtifact.id], 'macro-aware immutable cache must avoid reopening either payload')

  const activeSnapshot = service.store.createSnapshot({
    id: 'concept-atlas-active-snapshot', status: 'candidate', artifactIds: [familyArtifact.id, selectedMacroArtifact.id],
  })
  service.store.activateSnapshot(activeSnapshot.id)
  const active = createNeuronAtlas(service.store, { snapshotId: activeSnapshot.id })
  const activeParent = active.neurons.find((neuron) => neuron.id === parent.id)
  assert.equal(active.projection.conceptMacros.runtimeEligible, true)
  assert.equal(active.projection.conceptMacros.activeCount, 1)
  assert.equal(activeParent.conceptMacro.runtimeStatus, 'active')
  assert.equal(activeParent.conceptMacro.runtimeActive, true)
  assert.notEqual(active.fingerprint, first.body.fingerprint, 'active/rejected selection state must participate in the atlas cache and ETag fingerprint')
})

test('3B atlas exposes only bounded plastic-region dominance summaries from the exact snapshot', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const run = familyRun(['source-a', 'source-b'])
  const familyArtifact = service.store.createArtifact({
    id: 'plastic-summary-family', kind: 'family-run', contentHash: 'plastic-summary-family-hash',
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(run) },
  })
  const regionHead = createPlasticRegionAtlas({
    representationSpaceId: run.representationSpaceId,
    revision: 3, parentFingerprint: 'region-parent', behaviorDimension: 8, maximumMembershipsPerPage: 3,
    regions: [
      {
        id: 'latent-a', lifecycle: 'active', origin: 'discovery', bornRevision: 1, retiredRevision: null,
        parentRegionIds: [], prototype: { dimension: 8, values: [.9, -.5, .2, 0, 0, 0, 0, 0] },
        evidence: { observationCount: 12, sourceGroupIds: ['g1', 'g2'], confidenceMass: 2, cohesion: .9, holdoutUtility: .2, shamUtility: .01 },
      },
      {
        id: 'latent-b', lifecycle: 'provisional', origin: 'split', bornRevision: 2, retiredRevision: null,
        parentRegionIds: ['latent-a'], prototype: { dimension: 8, values: [-.2, .8, .1, 0, 0, 0, 0, 0] },
        evidence: { observationCount: 6, sourceGroupIds: ['g2'], confidenceMass: 1, cohesion: .8, holdoutUtility: .1, shamUtility: .02 },
      },
    ],
    pages: Array.from({ length: 56 }, (_, pageIndex) => pageIndex === 0
      ? { pageIndex, memberships: [{ regionId: 'latent-a', weight: .7 }, { regionId: 'latent-b', weight: .2 }], freeMass: .1 }
      : pageIndex === 1
        ? { pageIndex, memberships: [{ regionId: 'latent-a', weight: .3 }], freeMass: .7 }
        : { pageIndex, memberships: [], freeMass: 1 }),
  })
  const regionArtifact = service.store.createArtifact({
    id: 'plastic-summary-head', kind: PLASTIC_REGION_ATLAS_HEAD_ARTIFACT_KIND, contentHash: 'plastic-summary-head-hash',
    payload: {
      format: 'bombom-runtime-artifact', version: 1,
      metadata: {
        representationSpaceId: regionHead.representationSpaceId,
        fingerprint: regionHead.fingerprint,
        geometryFingerprint: regionHead.geometryFingerprint,
        revision: regionHead.revision,
        behaviorDimension: regionHead.behaviorDimension,
        maximumMembershipsPerPage: regionHead.maximumMembershipsPerPage,
      },
      value: encodeArtifactValue(regionHead),
    },
  })
  const snapshot = service.store.createSnapshot({
    id: 'plastic-summary-snapshot', status: 'candidate', artifactIds: [familyArtifact.id, regionArtifact.id],
    evaluation: { passed: true, runtimeVerified: true },
  })
  const opened = []
  const getArtifact = service.store.getArtifact.bind(service.store)
  service.store.getArtifact = (id) => { opened.push(id); return getArtifact(id) }

  const atlas = createNeuronAtlas(service.store, { snapshotId: snapshot.id })
  const projection = atlas.projection.plasticRegions
  assert.equal(projection.headFingerprint, regionHead.fingerprint)
  assert.equal(projection.compatible, true)
  assert.equal(projection.configuredMode, 'off')
  assert.equal(projection.effectiveMode, 'off')
  assert.equal(projection.effectiveModeReason, 'policy-off')
  assert.equal(projection.behaviorApplied, false)
  assert.equal(projection.exactRuntimeBinding, true)
  assert.equal(projection.release.eligible, false)
  assert.equal(projection.regionCount, 2)
  assert.equal(projection.activeRegionCount, 1)
  assert.equal(projection.assignedMass, 1.2)
  assert.equal(projection.freeMass, 54.8)
  assert.equal(projection.dominantRegionId, 'latent-a')
  assert.deepEqual(projection.regions[0], {
    id: 'latent-a', lifecycle: 'active', origin: 'discovery', effectiveCapacity: 1,
    assignedPageCount: 2, dominantPageCount: 2, meanMembershipWeight: .5, globalShare: .833333,
    cohesion: .9, observationCount: 12, sourceGroupCount: 2,
  })
  assert.doesNotMatch(JSON.stringify(projection), /prototype|values|sourceGroupIds|0\.9,-0\.5/u)
  assert.deepEqual(opened, [familyArtifact.id, regionArtifact.id, regionArtifact.id, regionArtifact.id])

  const repeated = createNeuronAtlas(service.store, { snapshotId: snapshot.id })
  assert.equal(repeated.fingerprint, atlas.fingerprint)
  assert.deepEqual(opened, [familyArtifact.id, regionArtifact.id, regionArtifact.id, regionArtifact.id], 'region-aware atlas cache must not reopen immutable payloads')

  service.store.updateConfig({ plasticRegions: {
    representationProfile: 'adaptive-eighth-id-v1', mode: 'shadow',
  } })
  const shadow = createNeuronAtlas(service.store, { snapshotId: snapshot.id })
  assert.equal(shadow.projection.plasticRegions.configuredMode, 'shadow')
  assert.equal(shadow.projection.plasticRegions.effectiveMode, 'shadow')
  assert.notEqual(shadow.fingerprint, atlas.fingerprint, 'configured/effective mode must participate in the 3B ETag fingerprint')
})

test('3B atlas does not hydrate a FamilyRun while a memory-heavy training job is active', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  await service.queue.stop({ abort: true })
  const first = service.store.createSource(normalizeTextSource({ text: 'Ali ekmeği kesti.', title: 'Kaynak A' }, service.store.getConfig())).source
  const second = service.store.createSource(normalizeTextSource({ text: 'Ayşe kağıdı kesti.', title: 'Kaynak B' }, service.store.getConfig())).source
  const run = familyRun([first.id, second.id])
  const artifact = service.store.createArtifact({
    id: 'busy-family-artifact', kind: 'family-run', contentHash: 'busy-family-hash',
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(run) },
  })
  service.store.createSnapshot({ id: 'busy-family-snapshot', status: 'rejected', artifactIds: [artifact.id] })
  const activeJob = service.store.createJob({ type: 'full-cycle', payload: {} }).job
  service.store.updateJob(activeJob.id, { state: 'running', stage: 'families' })
  const opened = []
  const getArtifact = service.store.getArtifact.bind(service.store)
  service.store.getArtifact = (id) => { opened.push(id); return getArtifact(id) }
  const result = createNeuronAtlas(service.store, { snapshotId: 'busy-family-snapshot' })
  assert.equal(result.available, true)
  assert.equal(result.partial, true)
  assert.equal(result.busy, true)
  assert.equal(result.busyStage, 'families')
  assert.equal(result.stats.languageOutputFamilyCount, 6)
  assert.equal(result.stats.languageOutputSeedNeuronCount, 21)
  assert.deepEqual(opened, [], 'busy atlas must use metadata only and never parse the family payload')
})

test('3B atlas reserves RAM for a queued campaign before its worker claim', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  await service.queue.stop({ abort: true })
  const run = familyRun(['source-a', 'source-b'])
  const artifact = service.store.createArtifact({
    id: 'queued-family-artifact', kind: 'family-run', contentHash: 'queued-family-hash',
    payload: { format: 'bombom-runtime-artifact', version: 1, value: encodeArtifactValue(run) },
  })
  service.store.createSnapshot({ id: 'queued-family-snapshot', status: 'rejected', artifactIds: [artifact.id] })
  service.store.createJob({ type: 'teacher-feed-campaign', payload: {} })
  const opened = []
  const getArtifact = service.store.getArtifact.bind(service.store)
  service.store.getArtifact = (id) => { opened.push(id); return getArtifact(id) }

  const result = createNeuronAtlas(service.store, { snapshotId: 'queued-family-snapshot' })
  assert.equal(result.available, true)
  assert.equal(result.partial, true)
  assert.equal(result.busy, true)
  assert.deepEqual(opened, [], 'queued campaign must reserve RAM before the worker is claimed')
})

test('family inspection rejects oversized pagination and unknown explicit snapshots', async (t) => {
  const service = await createTrainingService({ dataDirectory: await temporaryDirectory(t), forceJson: true })
  const base = await listenService(t, service)

  let result = await jsonFetch(base, '/api/families/inspection?memberLimit=13')
  assert.equal(result.response.status, 400)
  assert.equal(result.body.code, 'INVALID_FAMILY_INSPECTION_QUERY')

  result = await jsonFetch(base, '/api/families/inspection?snapshotId=missing-snapshot')
  assert.equal(result.response.status, 404)
  assert.equal(result.body.code, 'SNAPSHOT_NOT_FOUND')
})
