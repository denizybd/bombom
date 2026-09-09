import type { AtlasFamily, NeuronAtlas, UniverseLayout } from './types'

function hash32(value: string) {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

function signedHash(value: string, salt: number) {
  return ((hash32(`${salt}:${value}`) / 0xffffffff) * 2) - 1
}

function finite(value: number | undefined, fallback = 0) {
  return Number.isFinite(value) ? Number(value) : fallback
}

/**
 * Original v1 family projection. Keeping this routine intact makes an atlas
 * without a language overlay bit-for-bit compatible with the old renderer.
 */
function legacyFamilyLayout(atlas: NeuronAtlas) {
  const familyCount = atlas.families.length
  const positions = new Float64Array(familyCount * 3)
  if (familyCount === 0) return positions
  const sorted = [...atlas.families].sort((left, right) => left.id.localeCompare(right.id))
  const rank = new Map(sorted.map((family, index) => [family.index, index]))
  const goldenAngle = Math.PI * (3 - Math.sqrt(5))
  const initialRadius = Math.max(24, Math.sqrt(familyCount) * 5.2)

  for (const family of atlas.families) {
    const itemRank = rank.get(family.index) ?? family.index
    const y = 1 - ((itemRank + 0.5) / familyCount) * 2
    const radial = Math.sqrt(Math.max(0, 1 - y * y))
    const angle = itemRank * goldenAngle + signedHash(atlas.run?.fingerprint ?? '', 11) * Math.PI
    const offset = family.index * 3
    positions[offset] = Math.cos(angle) * radial * initialRadius
    positions[offset + 1] = y * initialRadius
    positions[offset + 2] = Math.sin(angle) * radial * initialRadius
  }

  const forces = new Float64Array(positions.length)
  const iterations = familyCount > 512 ? 55 : 130
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    forces.fill(0)
    for (let left = 0; left < familyCount; left += 1) {
      const lo = left * 3
      for (let right = left + 1; right < familyCount; right += 1) {
        const ro = right * 3
        let dx = positions[lo] - positions[ro]
        let dy = positions[lo + 1] - positions[ro + 1]
        let dz = positions[lo + 2] - positions[ro + 2]
        let distanceSquared = dx * dx + dy * dy + dz * dz
        if (distanceSquared < 0.0001) {
          dx = signedHash(`${left}:${right}`, 3) * 0.1
          dy = signedHash(`${left}:${right}`, 5) * 0.1
          dz = signedHash(`${left}:${right}`, 7) * 0.1
          distanceSquared = dx * dx + dy * dy + dz * dz
        }
        const scale = 50 / (distanceSquared + 18)
        forces[lo] += dx * scale
        forces[lo + 1] += dy * scale
        forces[lo + 2] += dz * scale
        forces[ro] -= dx * scale
        forces[ro + 1] -= dy * scale
        forces[ro + 2] -= dz * scale
      }
    }

    for (const edge of atlas.links.familyTransitions) {
      const [from, to, probability] = edge
      if (from === to || from < 0 || to < 0 || from >= familyCount || to >= familyCount) continue
      const fo = from * 3
      const toOffset = to * 3
      const dx = positions[toOffset] - positions[fo]
      const dy = positions[toOffset + 1] - positions[fo + 1]
      const dz = positions[toOffset + 2] - positions[fo + 2]
      const distance = Math.max(0.001, Math.hypot(dx, dy, dz))
      const ideal = 13 + 11 * (1 - Math.min(1, probability * 6))
      const scale = ((distance - ideal) / distance) * (0.005 + probability * 0.024)
      forces[fo] += dx * scale
      forces[fo + 1] += dy * scale
      forces[fo + 2] += dz * scale
      forces[toOffset] -= dx * scale
      forces[toOffset + 1] -= dy * scale
      forces[toOffset + 2] -= dz * scale
    }

    const temperature = 0.72 * (1 - iteration / iterations) + 0.08
    for (let index = 0; index < familyCount; index += 1) {
      const offset = index * 3
      forces[offset] -= positions[offset] * 0.006
      forces[offset + 1] -= positions[offset + 1] * 0.006
      forces[offset + 2] -= positions[offset + 2] * 0.006
      const forceLength = Math.max(1, Math.hypot(forces[offset], forces[offset + 1], forces[offset + 2]))
      const limited = Math.min(1.4, forceLength) * temperature / forceLength
      positions[offset] += forces[offset] * limited
      positions[offset + 1] += forces[offset + 1] * limited
      positions[offset + 2] += forces[offset + 2] * limited
    }
  }
  return positions
}

type Vector3 = [number, number, number]

function languageSurfaceFamily(family: AtlasFamily | undefined): boolean {
  return family?.layoutZone === 'language-surface'
}

function conceptSurfaceNeuron(neuron: NeuronAtlas['neurons'][number] | undefined): boolean {
  return neuron?.layoutZone === 'concept-surface' && neuron.kind === 'concept-macro'
}

function vectorLength(vector: Vector3): number {
  return Math.hypot(vector[0], vector[1], vector[2])
}

function normalizeVector(vector: Vector3, fallback: Vector3): Vector3 {
  const length = vectorLength(vector)
  if (length < 0.000001) return fallback
  return [vector[0] / length, vector[1] / length, vector[2] / length]
}

function cross(left: Vector3, right: Vector3): Vector3 {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ]
}

function addScaled(target: Vector3, vector: Vector3, scale: number): Vector3 {
  return [
    target[0] + vector[0] * scale,
    target[1] + vector[1] * scale,
    target[2] + vector[2] * scale,
  ]
}

function deterministicZoneFrame(atlas: NeuronAtlas): { outward: Vector3; tangent: Vector3; vertical: Vector3 } {
  const key = atlas.run?.representationSpaceId ?? atlas.run?.fingerprint ?? 'empty-representation-space'
  const outward = normalizeVector([
    signedHash(key, 101),
    signedHash(key, 103),
    signedHash(key, 107),
  ], [1, 0, 0])
  const reference: Vector3 = Math.abs(outward[1]) < 0.88 ? [0, 1, 0] : [1, 0, 0]
  const tangent = normalizeVector(cross(reference, outward), [0, 0, 1])
  const vertical = normalizeVector(cross(outward, tangent), [0, 1, 0])
  return { outward, tangent, vertical }
}

function deterministicConceptFrame(atlas: NeuronAtlas): { outward: Vector3; tangent: Vector3; vertical: Vector3 } {
  const key = atlas.projection?.conceptMacros?.headFingerprint ?? atlas.run?.representationSpaceId ?? 'empty-concept-space'
  const outward = normalizeVector([
    signedHash(key, 211),
    signedHash(key, 223),
    signedHash(key, 227),
  ], [-1, 0, 0])
  const reference: Vector3 = Math.abs(outward[1]) < 0.88 ? [0, 1, 0] : [1, 0, 0]
  const tangent = normalizeVector(cross(reference, outward), [0, 0, 1])
  const vertical = normalizeVector(cross(outward, tangent), [0, 1, 0])
  return { outward, tangent, vertical }
}

function positionAt(positions: Float64Array, familyIndex: number): Vector3 {
  const offset = familyIndex * 3
  return [positions[offset] ?? 0, positions[offset + 1] ?? 0, positions[offset + 2] ?? 0]
}

function writePosition(positions: Float64Array, familyIndex: number, value: Vector3): void {
  const offset = familyIndex * 3
  positions[offset] = value[0]
  positions[offset + 1] = value[1]
  positions[offset + 2] = value[2]
}

function maximumFamilyRadius(positions: Float64Array, families: readonly AtlasFamily[]): number {
  let radius = 0
  for (const family of families) radius = Math.max(radius, vectorLength(positionAt(positions, family.index)))
  return radius
}

function finiteAnchor(value: AtlasFamily['atlasAnchor']): Vector3 | null {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) return null
  return [Number(value[0]), Number(value[1]), Number(value[2])]
}

function bounded(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
}

/**
 * Builds a core-only view with dense local indexes. Language and cross-zone
 * edges are deliberately absent so their display links cannot exert force on
 * the learned family cloud.
 */
function coreAtlas(atlas: NeuronAtlas, coreFamilies: readonly AtlasFamily[]): {
  atlas: NeuronAtlas
  globalIndexByLocal: number[]
} {
  const localIndexByGlobal = new Map<number, number>()
  const globalIndexByLocal: number[] = []
  const families = coreFamilies.map((family, localIndex) => {
    localIndexByGlobal.set(family.index, localIndex)
    globalIndexByLocal.push(family.index)
    return { ...family, index: localIndex }
  })
  const familyTransitions = atlas.links.familyTransitions.flatMap((edge) => {
    const from = localIndexByGlobal.get(edge[0])
    const to = localIndexByGlobal.get(edge[1])
    return from === undefined || to === undefined ? [] : [[from, to, ...edge.slice(2)] as typeof edge]
  })
  return {
    atlas: {
      ...atlas,
      families,
      stats: { ...atlas.stats, familyCount: families.length, familyTransitionCount: familyTransitions.length },
      links: { ...atlas.links, familyTransitions },
    },
    globalIndexByLocal,
  }
}

function languageSeedPosition(
  atlas: NeuronAtlas,
  family: AtlasFamily,
  coreRadius: number,
  arcRadius: number,
  frame: ReturnType<typeof deterministicZoneFrame>,
): Vector3 {
  const centerDistance = coreRadius + arcRadius + 12
  let position: Vector3 = [
    frame.outward[0] * centerDistance,
    frame.outward[1] * centerDistance,
    frame.outward[2] * centerDistance,
  ]
  const explicitAnchor = finiteAnchor(family.atlasAnchor)
  if (explicitAnchor) {
    position = addScaled(position, frame.tangent, bounded(explicitAnchor[0], -1, 1) * arcRadius)
    position = addScaled(position, frame.vertical, bounded(explicitAnchor[1], -1, 1) * arcRadius)
    return addScaled(position, frame.outward, bounded(explicitAnchor[2], -1, 1) * arcRadius * 0.25)
  }

  const explicitSlot = Number.isSafeInteger(family.layoutSlot) && Number(family.layoutSlot) >= 0
  const angle = explicitSlot
    ? (Number(family.layoutSlot) - 3) * 0.28
    : ((hash32(`language-surface:${family.id}`) / 0xffffffff) * 2 - 1) * Math.PI
  position = addScaled(position, frame.tangent, Math.sin(angle) * arcRadius)
  position = addScaled(position, frame.vertical, Math.cos(angle) * arcRadius * 0.62)
  return position
}

function languageFamilyLayout(
  atlas: NeuronAtlas,
  positions: Float64Array,
  languageFamilies: readonly AtlasFamily[],
  coreRadius: number,
): void {
  const arcRadius = Math.max(7, Math.min(15, coreRadius * 0.28))
  const frame = deterministicZoneFrame(atlas)
  const familyById = new Map(atlas.families.map((family) => [family.id, family]))
  const placed = new Set<number>()
  const active = new Set<number>()

  const place = (family: AtlasFamily): Vector3 => {
    if (placed.has(family.index)) return positionAt(positions, family.index)
    if (active.has(family.index)) {
      const fallback = languageSeedPosition(atlas, family, coreRadius, arcRadius, frame)
      writePosition(positions, family.index, fallback)
      placed.add(family.index)
      return fallback
    }
    active.add(family.index)
    const parent = family.parentFamilyId ? familyById.get(family.parentFamilyId) : undefined
    let position: Vector3
    if (parent && languageSurfaceFamily(parent) && family.layoutPinned !== true) {
      const parentPosition = place(parent)
      const angle = (hash32(`language-satellite:${family.id}`) / 0xffffffff) * Math.PI * 2
      const satelliteRadius = 3.6 + (hash32(`language-satellite-radius:${family.id}`) % 3_000) / 1_000
      position = addScaled(parentPosition, frame.tangent, Math.cos(angle) * satelliteRadius)
      position = addScaled(position, frame.vertical, Math.sin(angle) * satelliteRadius)
      position = addScaled(position, frame.outward, 1.2)
    } else {
      position = languageSeedPosition(atlas, family, coreRadius, arcRadius, frame)
    }
    writePosition(positions, family.index, position)
    active.delete(family.index)
    placed.add(family.index)
    return position
  }

  for (const family of [...languageFamilies].sort((left, right) => left.id.localeCompare(right.id))) place(family)
}

function familyLayout(atlas: NeuronAtlas) {
  const languageFamilies = atlas.families.filter(languageSurfaceFamily)
  if (!languageFamilies.length) return legacyFamilyLayout(atlas)

  const coreFamilies = atlas.families.filter((family) => !languageSurfaceFamily(family))
  const positions = new Float64Array(atlas.families.length * 3)
  if (coreFamilies.length) {
    const projection = coreAtlas(atlas, coreFamilies)
    const corePositions = legacyFamilyLayout(projection.atlas)
    projection.globalIndexByLocal.forEach((globalIndex, localIndex) => {
      writePosition(positions, globalIndex, positionAt(corePositions, localIndex))
    })
  }
  const coreRadius = maximumFamilyRadius(positions, coreFamilies)
  languageFamilyLayout(atlas, positions, languageFamilies, coreRadius)
  return positions
}

function neuronLayout(atlas: NeuronAtlas, familyPositions: Float64Array) {
  const positions = new Float32Array(atlas.neurons.length * 3)
  const memberships = Array.from({ length: atlas.neurons.length }, () => [] as typeof atlas.links.memberships)
  for (const membership of atlas.links.memberships) {
    if (membership[0] >= 0 && membership[0] < memberships.length) memberships[membership[0]]?.push(membership)
  }

  const conceptNeurons = atlas.neurons.filter(conceptSurfaceNeuron).sort((left, right) => left.id.localeCompare(right.id))
  const conceptRank = new Map(conceptNeurons.map((neuron, index) => [neuron.index, index]))
  const coreFamilies = atlas.families.filter((family) => !languageSurfaceFamily(family))
  const coreRadius = Math.max(8, maximumFamilyRadius(familyPositions, coreFamilies))
  const conceptFrame = deterministicConceptFrame(atlas)
  const conceptRingRadius = Math.max(7, Math.min(20, 5 + Math.sqrt(conceptNeurons.length) * 0.72))
  const conceptCenterDistance = coreRadius + conceptRingRadius + 13

  for (const neuron of atlas.neurons) {
    const offset = neuron.index * 3
    if (conceptSurfaceNeuron(neuron)) {
      const rank = conceptRank.get(neuron.index) ?? neuron.index
      const angle = rank * Math.PI * (3 - Math.sqrt(5)) + signedHash(atlas.projection?.conceptMacros?.headFingerprint ?? '', 229) * Math.PI
      const depth = Math.max(0, Math.min(3, finite(neuron.conceptMacro?.depth)))
      let position: Vector3 = [
        conceptFrame.outward[0] * (conceptCenterDistance + depth * 1.35),
        conceptFrame.outward[1] * (conceptCenterDistance + depth * 1.35),
        conceptFrame.outward[2] * (conceptCenterDistance + depth * 1.35),
      ]
      const radial = conceptRingRadius * (0.58 + ((hash32(`concept-radius:${neuron.id}`) % 4_000) / 10_000))
      position = addScaled(position, conceptFrame.tangent, Math.cos(angle) * radial)
      position = addScaled(position, conceptFrame.vertical, Math.sin(angle) * radial * 0.76)
      positions[offset] = position[0]
      positions[offset + 1] = position[1]
      positions[offset + 2] = position[2]
      continue
    }
    const ownMemberships = memberships[neuron.index] ?? []
    const primary = neuron.primaryFamily >= 0 ? neuron.primaryFamily : ownMemberships[0]?.[1] ?? -1
    if (primary < 0 || primary >= atlas.families.length) {
      positions[offset] = signedHash(neuron.id, 13) * 7
      positions[offset + 1] = signedHash(neuron.id, 17) * 7
      positions[offset + 2] = signedHash(neuron.id, 19) * 7
      continue
    }
    const primaryOffset = primary * 3
    const primaryIsLanguageSurface = languageSurfaceFamily(atlas.families[primary])
    const layoutMemberships = ownMemberships.filter((membership) =>
      languageSurfaceFamily(atlas.families[membership[1]]) === primaryIsLanguageSurface,
    )
    let bx = 0
    let by = 0
    let bz = 0
    let totalWeight = 0
    for (const membership of layoutMemberships) {
      const family = membership[1]
      if (family < 0 || family >= atlas.families.length) continue
      const weight = Math.max(0.01, membership[2] * (0.35 + membership[3]))
      const familyOffset = family * 3
      bx += familyPositions[familyOffset] * weight
      by += familyPositions[familyOffset + 1] * weight
      bz += familyPositions[familyOffset + 2] * weight
      totalWeight += weight
    }
    if (totalWeight > 0) {
      bx /= totalWeight
      by /= totalWeight
      bz /= totalWeight
    } else {
      bx = familyPositions[primaryOffset]
      by = familyPositions[primaryOffset + 1]
      bz = familyPositions[primaryOffset + 2]
    }

    const family = atlas.families[primary]
    const localRadius = 2.4 + Math.sqrt(Math.max(1, family?.memberCount ?? 1)) * 0.42
    const diagnostics = neuron.diagnostics
    const diagnosticX = (finite(diagnostics[1], 0.5) * 2 - 1) * 0.62 + signedHash(neuron.id, 23) * 0.38
    const diagnosticY = (finite(diagnostics[2], 0.5) * 2 - 1) * 0.62 + signedHash(neuron.id, 29) * 0.38
    const diagnosticZ = (finite(diagnostics[5]) - finite(diagnostics[4])) * 0.7 + signedHash(neuron.id, 31) * 0.3
    const length = Math.max(0.001, Math.hypot(diagnosticX, diagnosticY, diagnosticZ))
    const radial = localRadius * (0.28 + (hash32(neuron.id) % 10_000) / 14_000)
    positions[offset] = familyPositions[primaryOffset] * 0.78 + bx * 0.22 + diagnosticX / length * radial
    positions[offset + 1] = familyPositions[primaryOffset + 1] * 0.78 + by * 0.22 + diagnosticY / length * radial
    positions[offset + 2] = familyPositions[primaryOffset + 2] * 0.78 + bz * 0.22 + diagnosticZ / length * radial
  }
  return positions
}

export function computeUniverseLayout(atlas: NeuronAtlas): UniverseLayout {
  const doubleFamilyPositions = familyLayout(atlas)
  const familyPositions = Float32Array.from(doubleFamilyPositions)
  const neuronPositions = neuronLayout(atlas, doubleFamilyPositions)
  let radius = 1
  for (let index = 0; index < familyPositions.length; index += 3) {
    radius = Math.max(radius, Math.hypot(familyPositions[index] ?? 0, familyPositions[index + 1] ?? 0, familyPositions[index + 2] ?? 0))
  }
  for (const neuron of atlas.neurons) {
    if (!conceptSurfaceNeuron(neuron)) continue
    const index = neuron.index * 3
    radius = Math.max(radius, Math.hypot(neuronPositions[index] ?? 0, neuronPositions[index + 1] ?? 0, neuronPositions[index + 2] ?? 0))
  }
  return { familyPositions, neuronPositions, radius: radius + 14 }
}
