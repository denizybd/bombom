import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import {
  buildFamilyLineData,
  buildMacroExpansionLineData,
  buildNeuronLineData,
  familyColor,
  familyForSelection,
  hash32,
  neuronColor,
} from './scene-data'
import type { ColorMode, NeuronAtlas, UniverseLayout, UniverseSelection } from './types'

interface UniverseSceneProps {
  atlas: NeuronAtlas
  layout: UniverseLayout
  selection: UniverseSelection
  colorMode: ColorMode
  showNeurons: boolean
  showFamilyPaths: boolean
  showNeuronPaths: boolean
  autoRotate: boolean
  onSelect: (selection: UniverseSelection) => void
  onHover: (selection: UniverseSelection) => void
  onWebglState: (state: 'ready' | 'lost' | 'unsupported') => void
}

interface SceneRuntime {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  familyMesh: THREE.InstancedMesh<THREE.IcosahedronGeometry, THREE.MeshBasicMaterial>
  familyPoints: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>
  neuronPoints: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>
  familyLines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
  neuronLines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
  macroLines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
  membershipLines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
  halo: THREE.Mesh<THREE.IcosahedronGeometry, THREE.MeshBasicMaterial>
  renderOnce: () => void
}

function replaceLineData(
  line: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>,
  positions: Float32Array,
  colors: Float32Array,
) {
  line.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  line.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  line.geometry.computeBoundingSphere()
}

function membershipLineData(atlas: NeuronAtlas, layout: UniverseLayout, selection: UniverseSelection) {
  const positions: number[] = []
  const colors: number[] = []
  let memberships = atlas.links.memberships
  if (selection?.kind === 'neuron') memberships = memberships.filter((item) => item[0] === selection.index)
  else if (selection?.kind === 'family') {
    memberships = memberships
      .filter((item) => item[1] === selection.index)
      .sort((left, right) => right[3] - left[3] || right[2] - left[2])
      .slice(0, 600)
  } else memberships = []

  for (const [neuron, family, score, weight] of memberships) {
    const no = neuron * 3
    const fo = family * 3
    if (no < 0 || fo < 0 || no + 2 >= layout.neuronPositions.length || fo + 2 >= layout.familyPositions.length) continue
    positions.push(
      layout.neuronPositions[no] ?? 0,
      layout.neuronPositions[no + 1] ?? 0,
      layout.neuronPositions[no + 2] ?? 0,
      layout.familyPositions[fo] ?? 0,
      layout.familyPositions[fo + 1] ?? 0,
      layout.familyPositions[fo + 2] ?? 0,
    )
    const strength = Math.max(0.16, Math.min(0.86, score * (0.4 + weight)))
    colors.push(0.45 * strength, 0.42 * strength, strength, 0.45 * strength, 0.42 * strength, strength)
  }
  return { positions: Float32Array.from(positions), colors: Float32Array.from(colors) }
}

function seededRandom(seedValue: string) {
  let state = hash32(seedValue) || 0x9e3779b9
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 0xffffffff
  }
}

function makeStars(seed: string, radius: number) {
  const random = seededRandom(seed)
  const positions = new Float32Array(1_800 * 3)
  for (let index = 0; index < positions.length; index += 3) {
    const theta = random() * Math.PI * 2
    const cosine = random() * 2 - 1
    const radial = radius * (1.8 + random() * 2.8)
    const planar = Math.sqrt(1 - cosine * cosine)
    positions[index] = Math.cos(theta) * planar * radial
    positions[index + 1] = cosine * radial
    positions[index + 2] = Math.sin(theta) * planar * radial
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  return new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ color: 0x7895c7, size: 0.13, transparent: true, opacity: 0.48, depthWrite: false }),
  )
}

function disposeScene(scene: THREE.Scene) {
  scene.traverse((object) => {
    const mesh = object as THREE.Mesh
    mesh.geometry?.dispose?.()
    if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose())
    else mesh.material?.dispose?.()
  })
}

export function UniverseScene({
  atlas,
  layout,
  selection,
  colorMode,
  showNeurons,
  showFamilyPaths,
  showNeuronPaths,
  autoRotate,
  onSelect,
  onHover,
  onWebglState,
}: UniverseSceneProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const runtimeRef = useRef<SceneRuntime | null>(null)
  const callbacksRef = useRef({ onSelect, onHover, onWebglState })
  const optionsRef = useRef({ autoRotate, showNeurons })
  callbacksRef.current = { onSelect, onHover, onWebglState }
  optionsRef.current = { autoRotate, showNeurons }

  useEffect(() => {
    const host = hostRef.current
    if (!host) return undefined
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' })
    } catch {
      callbacksRef.current.onWebglState('unsupported')
      return undefined
    }

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.setClearColor(0x030713, 1)
    renderer.domElement.className = 'nu-canvas'
    renderer.domElement.tabIndex = 0
    renderer.domElement.setAttribute('aria-label', 'Nöron ve ailelerin etkileşimli üç boyutlu atlası')
    renderer.domElement.setAttribute('aria-describedby', 'nu-projection-note')
    host.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x030713, 0.006)
    const camera = new THREE.PerspectiveCamera(47, 1, 0.08, Math.max(1_500, layout.radius * 12))
    camera.position.set(layout.radius * 0.18, layout.radius * 0.12, layout.radius * 1.72)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = !reducedMotion
    controls.dampingFactor = 0.065
    controls.rotateSpeed = 0.55
    controls.zoomSpeed = 0.72
    controls.panSpeed = 0.55
    controls.minDistance = 3
    controls.maxDistance = layout.radius * 5
    controls.autoRotateSpeed = 0.28

    scene.add(new THREE.HemisphereLight(0x9bbcff, 0x15122c, 1.8))
    const keyLight = new THREE.DirectionalLight(0xdde8ff, 2.3)
    keyLight.position.set(28, 40, 35)
    scene.add(keyLight)
    const rimLight = new THREE.PointLight(0x865cff, 110, layout.radius * 3.4)
    rimLight.position.set(-layout.radius, -layout.radius * 0.45, -layout.radius * 0.8)
    scene.add(rimLight)
    scene.add(makeStars(atlas.run?.fingerprint ?? 'empty-atlas', layout.radius))

    const familyGeometry = new THREE.IcosahedronGeometry(1, 1)
    // Instance colors must stay readable even when a family sits behind the
    // dense path layer; unlit color is intentional, while geometry/fog keeps
    // the 3B depth cue.
    const familyMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.96, fog: true })
    const familyMesh = new THREE.InstancedMesh(familyGeometry, familyMaterial, atlas.families.length)
    familyMesh.name = 'families'
    familyMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    const dummy = new THREE.Object3D()
    for (const family of atlas.families) {
      const offset = family.index * 3
      const size = Math.min(2.8, 0.72 + Math.sqrt(Math.max(1, family.memberCount)) * 0.115)
      dummy.position.set(layout.familyPositions[offset] ?? 0, layout.familyPositions[offset + 1] ?? 0, layout.familyPositions[offset + 2] ?? 0)
      dummy.scale.setScalar(size)
      dummy.updateMatrix()
      familyMesh.setMatrixAt(family.index, dummy.matrix)
      familyMesh.setColorAt(family.index, new THREE.Color(...familyColor(atlas, family.index, 'family')))
    }
    familyMesh.instanceMatrix.needsUpdate = true
    familyMesh.instanceColor!.needsUpdate = true
    scene.add(familyMesh)

    const familyPointGeometry = new THREE.BufferGeometry()
    familyPointGeometry.setAttribute('position', new THREE.BufferAttribute(layout.familyPositions.slice(), 3))
    const familyPointColors = new Float32Array(atlas.families.length * 3)
    for (const family of atlas.families) familyPointColors.set(familyColor(atlas, family.index, 'family'), family.index * 3)
    familyPointGeometry.setAttribute('color', new THREE.BufferAttribute(familyPointColors, 3))
    const familyPoints = new THREE.Points(
      familyPointGeometry,
      new THREE.PointsMaterial({ size: 6.2, vertexColors: true, sizeAttenuation: false, transparent: true, opacity: 0.9, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending }),
    )
    familyPoints.name = 'family-glow'
    familyPoints.renderOrder = 5
    scene.add(familyPoints)

    const neuronGeometry = new THREE.BufferGeometry()
    neuronGeometry.setAttribute('position', new THREE.BufferAttribute(layout.neuronPositions.slice(), 3))
    neuronGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(atlas.neurons.length * 3), 3))
    const neuronMaterial = new THREE.PointsMaterial({ size: 0.54, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.9, depthWrite: false })
    const neuronPoints = new THREE.Points(neuronGeometry, neuronMaterial)
    neuronPoints.name = 'neurons'
    scene.add(neuronPoints)

    const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.52, depthWrite: false, blending: THREE.AdditiveBlending })
    const familyLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial)
    familyLines.name = 'family-paths'
    scene.add(familyLines)
    const neuronLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial.clone())
    neuronLines.material.opacity = 0.58
    neuronLines.name = 'neuron-paths'
    scene.add(neuronLines)
    const macroLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial.clone())
    macroLines.material.opacity = 0.42
    macroLines.name = 'concept-macro-expansions'
    scene.add(macroLines)
    const membershipLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial.clone())
    membershipLines.material.opacity = 0.36
    membershipLines.name = 'memberships'
    scene.add(membershipLines)

    const halo = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1, 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, wireframe: true, depthWrite: false }),
    )
    halo.visible = false
    scene.add(halo)

    const raycaster = new THREE.Raycaster()
    raycaster.params.Points!.threshold = 1.15
    const pointer = new THREE.Vector2()
    let hovered: UniverseSelection = null
    let pointerFrame = 0
    let downPosition: [number, number] | null = null
    let disposed = false
    let animationFrame = 0

    const selectionKey = (value: UniverseSelection) => value ? `${value.kind}:${value.index}` : ''
    const hitTest = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1
      raycaster.setFromCamera(pointer, camera)
      const familyHit = raycaster.intersectObject(familyMesh, false)[0]
      let next: UniverseSelection = familyHit?.instanceId !== undefined
        ? { kind: 'family', index: familyHit.instanceId }
        : null
      if (!next && optionsRef.current.showNeurons) {
        const neuronHit = raycaster.intersectObject(neuronPoints, false)[0]
        if (neuronHit?.index !== undefined) next = { kind: 'neuron', index: neuronHit.index }
      }
      if (selectionKey(next) !== selectionKey(hovered)) {
        hovered = next
        renderer.domElement.classList.toggle('is-pointing', Boolean(next))
        callbacksRef.current.onHover(next)
      }
    }

    const handleMove = (event: PointerEvent) => {
      if (pointerFrame) return
      pointerFrame = window.requestAnimationFrame(() => {
        pointerFrame = 0
        hitTest(event)
      })
    }
    const handleLeave = () => {
      hovered = null
      renderer.domElement.classList.remove('is-pointing')
      callbacksRef.current.onHover(null)
    }
    const handleDown = (event: PointerEvent) => { downPosition = [event.clientX, event.clientY] }
    const handleUp = (event: PointerEvent) => {
      if (!downPosition || Math.hypot(event.clientX - downPosition[0], event.clientY - downPosition[1]) > 5) return
      callbacksRef.current.onSelect(hovered)
    }
    const handleContextLost = (event: Event) => {
      event.preventDefault()
      callbacksRef.current.onWebglState('lost')
    }
    const handleContextRestored = () => callbacksRef.current.onWebglState('ready')
    renderer.domElement.addEventListener('pointermove', handleMove)
    renderer.domElement.addEventListener('pointerleave', handleLeave)
    renderer.domElement.addEventListener('pointerdown', handleDown)
    renderer.domElement.addEventListener('pointerup', handleUp)
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost)
    renderer.domElement.addEventListener('webglcontextrestored', handleContextRestored)

    const resize = () => {
      const width = Math.max(1, host.clientWidth)
      const height = Math.max(1, host.clientHeight)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(host)
    resize()

    const renderOnce = () => renderer.render(scene, camera)
    const animate = () => {
      if (disposed) return
      animationFrame = window.requestAnimationFrame(animate)
      if (document.hidden) return
      controls.autoRotate = optionsRef.current.autoRotate && !reducedMotion
      controls.update()
      renderOnce()
    }
    runtimeRef.current = { renderer, scene, camera, controls, familyMesh, familyPoints, neuronPoints, familyLines, neuronLines, macroLines, membershipLines, halo, renderOnce }
    callbacksRef.current.onWebglState('ready')
    animate()

    return () => {
      disposed = true
      window.cancelAnimationFrame(animationFrame)
      if (pointerFrame) window.cancelAnimationFrame(pointerFrame)
      resizeObserver.disconnect()
      controls.dispose()
      renderer.domElement.removeEventListener('pointermove', handleMove)
      renderer.domElement.removeEventListener('pointerleave', handleLeave)
      renderer.domElement.removeEventListener('pointerdown', handleDown)
      renderer.domElement.removeEventListener('pointerup', handleUp)
      renderer.domElement.removeEventListener('webglcontextlost', handleContextLost)
      renderer.domElement.removeEventListener('webglcontextrestored', handleContextRestored)
      disposeScene(scene)
      renderer.dispose()
      renderer.domElement.remove()
      runtimeRef.current = null
    }
  }, [atlas, layout])

  useEffect(() => {
    const runtime = runtimeRef.current
    if (!runtime) return
    runtime.neuronPoints.visible = showNeurons
    runtime.familyLines.visible = showFamilyPaths
    runtime.neuronLines.visible = showNeuronPaths && Boolean(selection)
    runtime.macroLines.visible = showNeurons && Boolean(atlas.links.macroExpansions?.length)
    runtime.membershipLines.visible = Boolean(selection)

    const selectedFamily = familyForSelection(atlas, selection)
    const dummy = new THREE.Object3D()
    for (const family of atlas.families) {
      const offset = family.index * 3
      const selected = selection?.kind === 'family' && selection.index === family.index
      const size = Math.min(2.8, 0.72 + Math.sqrt(Math.max(1, family.memberCount)) * 0.115) * (selected ? 1.28 : 1)
      dummy.position.set(layout.familyPositions[offset] ?? 0, layout.familyPositions[offset + 1] ?? 0, layout.familyPositions[offset + 2] ?? 0)
      dummy.scale.setScalar(size)
      dummy.updateMatrix()
      runtime.familyMesh.setMatrixAt(family.index, dummy.matrix)
      const color = familyColor(atlas, family.index, colorMode)
      runtime.familyMesh.setColorAt(family.index, new THREE.Color(...color).multiplyScalar(selectedFamily === null || selectedFamily === family.index ? 1 : 0.32))
    }
    runtime.familyMesh.instanceMatrix.needsUpdate = true
    runtime.familyMesh.instanceColor!.needsUpdate = true

    const familyPointColors = runtime.familyPoints.geometry.getAttribute('color') as THREE.BufferAttribute
    for (const family of atlas.families) {
      const base = familyColor(atlas, family.index, colorMode)
      const selected = selectedFamily === null || selectedFamily === family.index
      const brightness = selected ? 1 : 0.22
      familyPointColors.setXYZ(family.index, base[0] * brightness, base[1] * brightness, base[2] * brightness)
    }
    familyPointColors.needsUpdate = true

    const neuronColors = runtime.neuronPoints.geometry.getAttribute('color') as THREE.BufferAttribute
    for (const neuron of atlas.neurons) {
      const family = neuron.primaryFamily
      const base = neuronColor(atlas, neuron.index, colorMode)
      const selected = selection?.kind === 'neuron' && selection.index === neuron.index
      const inFamily = selectedFamily !== null && family === selectedFamily
      const isConcept = neuron.kind === 'concept-macro'
      const brightness = selected ? 1.65 : selectedFamily === null ? (isConcept ? 1.18 : 0.78) : inFamily ? 1.05 : isConcept ? 0.34 : 0.12
      neuronColors.setXYZ(neuron.index, Math.min(1, base[0] * brightness), Math.min(1, base[1] * brightness), Math.min(1, base[2] * brightness))
    }
    neuronColors.needsUpdate = true

    const familyData = buildFamilyLineData(atlas, layout.familyPositions, selectedFamily)
    replaceLineData(runtime.familyLines, familyData.positions, familyData.colors)
    const neuronData = buildNeuronLineData(atlas, layout.neuronPositions, selection)
    replaceLineData(runtime.neuronLines, neuronData.positions, neuronData.colors)
    const macroData = buildMacroExpansionLineData(atlas, layout.neuronPositions, selection)
    replaceLineData(runtime.macroLines, macroData.positions, macroData.colors)
    const membershipData = membershipLineData(atlas, layout, selection)
    replaceLineData(runtime.membershipLines, membershipData.positions, membershipData.colors)

    runtime.halo.visible = Boolean(selection)
    if (selection) {
      const positions = selection.kind === 'family' ? layout.familyPositions : layout.neuronPositions
      const offset = selection.index * 3
      runtime.halo.position.set(positions[offset] ?? 0, positions[offset + 1] ?? 0, positions[offset + 2] ?? 0)
      const family = selection.kind === 'family' ? atlas.families[selection.index] : null
      runtime.halo.scale.setScalar(family ? Math.min(4, 1.15 + Math.sqrt(Math.max(1, family.memberCount)) * 0.15) : 0.75)
    }
    runtime.renderOnce()
  }, [atlas, colorMode, layout, selection, showFamilyPaths, showNeuronPaths, showNeurons])

  return <div className="nu-scene" ref={hostRef} />
}
