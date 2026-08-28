import {
  Activity,
  AlertTriangle,
  Box,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleGauge,
  Download,
  FileJson,
  FileText,
  Focus,
  Gauge,
  History,
  Info,
  Maximize2,
  Network,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Save,
  Settings2,
  Shuffle,
  SkipForward,
  Sparkles,
  Trash2,
  Upload,
  X,
  Zap,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import {
  type ChangeEvent,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { MatrixEditor } from './components/MatrixEditor'
import { MatrixGrid } from './components/MatrixGrid'
import {
  type Matrix,
  MATRIX_SIZE,
  type NetworkState,
  type Neuron,
  type SimulationState,
  computeHabitProbabilities,
  computeNeuronSimilarityMatrix,
  cosineSimilarity,
  createInitialNetwork,
  createInitialSimulationState,
  expandNetworkForNewNeuron,
  stepSimulation,
} from './engine'

type PaintTool = 'positive' | 'negative' | 'erase'
type DockTab = 'timeline' | 'habit' | 'similarity' | 'pressure' | 'log'
type InspectorMode = 'step' | 'edit'

const STORAGE_KEY = 'sinaps-sandbox:v2'
const DEFAULT_SEED = 1847

const zeroMatrix = (): Matrix =>
  Array.from({ length: MATRIX_SIZE }, () => Array(MATRIX_SIZE).fill(0))

const cloneMatrix = (matrix: Matrix): Matrix => matrix.map((row) => [...row])

function starterNeuronMatrix(): Matrix {
  return Array.from({ length: MATRIX_SIZE }, (_, row) =>
    Array.from({ length: MATRIX_SIZE }, (_, column) => {
      const diagonal = Math.abs(row - column)
      if (diagonal === 0) return 0.9
      if (diagonal === 1) return 0.42
      if (row + column === MATRIX_SIZE - 1) return -0.58
      return 0
    }),
  )
}

function starterSignalMatrix(): Matrix {
  const center = (MATRIX_SIZE - 1) / 2
  return Array.from({ length: MATRIX_SIZE }, (_, row) =>
    Array.from({ length: MATRIX_SIZE }, (_, column) => {
      const distance = Math.hypot(row - center, column - center)
      if (distance < MATRIX_SIZE * 0.2) return 0.82
      if (distance < MATRIX_SIZE * 0.35) return 0.28
      if ((row + column) % 17 === 0) return -0.36
      return 0
    }),
  )
}

function seededMatrix(seed: number): Matrix {
  let state = (Math.trunc(seed) || 1) >>> 0
  const next = () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
  return Array.from({ length: MATRIX_SIZE }, () =>
    Array.from({ length: MATRIX_SIZE }, () => {
      const value = next() * 2 - 1
      return Math.abs(value) < 0.14 ? 0 : Number(value.toFixed(4))
    }),
  )
}

function defaultNetwork(): NetworkState {
  return createInitialNetwork({ id: 'N1', matrix: starterNeuronMatrix() })
}

function freshSimulation(
  network = defaultNetwork(),
  signal = starterSignalMatrix(),
  seed = DEFAULT_SEED,
  maxSteps = 25,
): SimulationState {
  return createInitialSimulationState(network, signal, {
    seed,
    maxSteps,
    currentNeuronId: network.neurons[0]?.id,
  })
}

function isSignedEightByEight(value: unknown): value is Matrix {
  return (
    Array.isArray(value) &&
    value.length === MATRIX_SIZE &&
    value.every(
      (row) =>
        Array.isArray(row) &&
        row.length === MATRIX_SIZE &&
        row.every(
          (cell) => typeof cell === 'number' && Number.isFinite(cell) && cell >= -1 && cell <= 1,
        ),
    )
  )
}

function isValidNetworkPayload(
  neurons: unknown,
  evidence: unknown,
  signal: unknown,
): neurons is Neuron[] {
  if (!Array.isArray(neurons) || neurons.length === 0 || !isSignedEightByEight(signal)) return false
  const ids = new Set<string>()
  for (const candidate of neurons) {
    if (
      !candidate ||
      typeof candidate !== 'object' ||
      typeof (candidate as Neuron).id !== 'string' ||
      !(candidate as Neuron).id.trim() ||
      ids.has((candidate as Neuron).id) ||
      !isSignedEightByEight((candidate as Neuron).matrix)
    ) return false
    ids.add((candidate as Neuron).id)
  }
  if (!Array.isArray(evidence) || evidence.length !== neurons.length) return false
  return evidence.every(
    (row, rowIndex) =>
      Array.isArray(row) &&
      row.length === neurons.length &&
      row.every(
        (cell, columnIndex) =>
          typeof cell === 'number' &&
          Number.isFinite(cell) &&
          cell >= 0 &&
          (rowIndex !== columnIndex || cell === 0),
      ),
  )
}

function restoreSession(): {
  simulation: SimulationState
  seed: number
  speed: number
} {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) throw new Error('empty')
    const parsed = JSON.parse(raw)
    const neurons = parsed.network?.neurons as Neuron[]
    const evidence = parsed.network?.habitEvidence as number[][]
    const signal = parsed.initialSignal as Matrix
    if (!isValidNetworkPayload(neurons, evidence, signal)) {
      throw new Error('invalid network')
    }
    const network: NetworkState = {
      neurons,
      habitEvidence: evidence,
      habitProbabilities: computeHabitProbabilities(evidence),
      neuronSimilarities: computeNeuronSimilarityMatrix(neurons),
    }
    const seed = Number.isFinite(parsed.seed) ? parsed.seed : DEFAULT_SEED
    const maxSteps = Number.isFinite(parsed.maxSteps) ? parsed.maxSteps : 25
    return {
      simulation: freshSimulation(network, signal, seed, maxSteps),
      seed,
      speed: Number.isFinite(parsed.speed) ? parsed.speed : 3,
    }
  } catch {
    return { simulation: freshSimulation(), seed: DEFAULT_SEED, speed: 3 }
  }
}

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value))
const percent = (value: number | undefined | null) =>
  typeof value === 'number' && Number.isFinite(value) ? `%${Math.round(value * 100)}` : '—'
const decimal = (value: number | undefined | null, digits = 4) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'

function statusCopy(simulation: SimulationState) {
  if (simulation.status === 'running') return { label: 'Akışta', tone: 'live' }
  if (simulation.status === 'paused') return { label: 'Duraklatıldı', tone: 'warn' }
  if (simulation.status === 'stopped') return { label: 'Tamamlandı', tone: 'stop' }
  return { label: 'Hazır', tone: 'idle' }
}

function stopReasonCopy(reason: SimulationState['stopReason']) {
  switch (reason) {
    case 'max-steps':
      return 'Adım sınırına ulaşıldı. Kas hafızası korunuyor.'
    case 'no-candidates':
      return 'Mevcut nöronun çıkış adayı yok. Yeni bir nöron ekleyebilirsiniz.'
    case 'no-match':
      return 'Hiçbir aday bu sinyali kabul etmedi.'
    case 'user-stopped':
      return 'Akış kullanıcı tarafından duraklatıldı.'
    case 'invalid-state':
      return 'Sayısal kararsızlık algılandı; güvenli biçimde durduruldu.'
    default:
      return null
  }
}

function Button({
  children,
  icon,
  tone = 'default',
  compact = false,
  className = '',
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode
  tone?: 'default' | 'primary' | 'danger' | 'ghost'
  compact?: boolean
}) {
  return (
    <button
      className={`button button--${tone} ${compact ? 'button--compact' : ''} ${className}`}
      {...props}
    >
      {icon}
      {children}
    </button>
  )
}

function PanelHeading({
  eyebrow,
  title,
  trailing,
}: {
  eyebrow: string
  title: string
  trailing?: ReactNode
}) {
  return (
    <div className="panel-heading">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
      </div>
      {trailing}
    </div>
  )
}

function NetworkStage({
  simulation,
  selectedId,
  onSelect,
  onAdd,
  canEdit,
}: {
  simulation: SimulationState
  selectedId: string
  onSelect: (id: string) => void
  onAdd: () => void
  canEdit: boolean
}) {
  const [zoom, setZoom] = useState(1)
  const { network } = simulation
  const last = simulation.timeline.at(-1) as any
  const count = network.neurons.length
  const positions = useMemo(
    () =>
      network.neurons.map((neuron, index) => {
        if (count === 1) return { id: neuron.id, x: 500, y: 260 }
        const angle = -Math.PI / 2 + (Math.PI * 2 * index) / count
        const radiusX = count === 2 ? 245 : 330
        const radiusY = count <= 3 ? 135 : 175
        return {
          id: neuron.id,
          x: 500 + Math.cos(angle) * radiusX,
          y: 260 + Math.sin(angle) * radiusY,
        }
      }),
    [network.neurons, count],
  )
  const positionById = new Map(positions.map((position) => [position.id, position]))
  const candidates = new Map(
    (last?.candidates ?? []).map((candidate: any) => [candidate.neuronId, candidate]),
  )

  return (
    <section className="panel network-panel">
      <PanelHeading
        eyebrow="CANLI SAHNE"
        title="Nöron ağı"
        trailing={
          <div className="stage-actions">
            <button
              className="icon-button"
              onClick={() => setZoom((value) => clamp(value - 0.1, 0.7, 1.3))}
              aria-label="Uzaklaştır"
              title="Uzaklaştır"
            >
              <ZoomOut size={16} />
            </button>
            <button
              className="icon-button"
              onClick={() => setZoom(1)}
              aria-label="Ağa sığdır"
              title="Ağa sığdır"
            >
              <Maximize2 size={16} />
            </button>
            <button
              className="icon-button"
              onClick={() => setZoom((value) => clamp(value + 0.1, 0.7, 1.3))}
              aria-label="Yakınlaştır"
              title="Yakınlaştır"
            >
              <ZoomIn size={16} />
            </button>
            <Button compact icon={<Plus size={15} />} onClick={onAdd} disabled={!canEdit} title={!canEdit ? 'Nöron eklemek için akışı duraklatın' : undefined}>
              Yeni nöron
            </Button>
          </div>
        }
      />

      <div className="network-stage">
        <div className="stage-legend" aria-label="Ağ göstergeleri">
          <span><i className="legend-dot legend-dot--active" /> Aktif</span>
          <span><i className="legend-line legend-line--habit" /> H alışkanlığı</span>
          <span><i className="legend-line legend-line--accepted" /> Kabul</span>
        </div>

        <div
          className="network-world"
          style={{ '--network-zoom': zoom } as CSSProperties}
        >
          <svg className="edge-layer" viewBox="0 0 1000 520" aria-hidden="true">
            <defs>
              <marker id="arrow-default" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
              </marker>
              <marker id="arrow-accepted" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
              </marker>
            </defs>
            {network.neurons.flatMap((from, fromIndex) =>
              network.neurons.map((to, toIndex) => {
                if (fromIndex === toIndex) return null
                const weight = network.habitProbabilities[fromIndex]?.[toIndex] ?? 0
                if (weight <= 0) return null
                const source = positionById.get(from.id)!
                const target = positionById.get(to.id)!
                const dx = target.x - source.x
                const dy = target.y - source.y
                const length = Math.max(Math.hypot(dx, dy), 1)
                const normalX = -dy / length
                const normalY = dx / length
                const curve = fromIndex < toIndex ? 28 : -28
                const startX = source.x + (dx / length) * 56
                const startY = source.y + (dy / length) * 56
                const endX = target.x - (dx / length) * 62
                const endY = target.y - (dy / length) * 62
                const middleX = (startX + endX) / 2 + normalX * curve
                const middleY = (startY + endY) / 2 + normalY * curve
                const candidate = last?.fromNeuronId === from.id ? candidates.get(to.id) as any : null
                const accepted = candidate?.outcome === 'accepted'
                const rejected = candidate?.outcome === 'rejected'
                return (
                  <path
                    key={`${from.id}-${to.id}`}
                    className={`network-edge ${accepted ? 'network-edge--accepted' : ''} ${rejected ? 'network-edge--rejected' : ''}`}
                    d={`M ${startX} ${startY} Q ${middleX} ${middleY} ${endX} ${endY}`}
                    style={{ '--edge-weight': weight } as CSSProperties}
                    markerEnd={`url(#${accepted ? 'arrow-accepted' : 'arrow-default'})`}
                  />
                )
              }),
            )}
          </svg>

          {network.neurons.map((neuron, index) => {
            const position = positions[index]
            const isActive = neuron.id === simulation.currentNeuronId
            const isSelected = neuron.id === selectedId
            const candidate = candidates.get(neuron.id) as any
            return (
              <button
                key={neuron.id}
                className={`neuron-node ${isActive ? 'neuron-node--active' : ''} ${isSelected ? 'neuron-node--selected' : ''} ${candidate?.outcome === 'accepted' ? 'neuron-node--accepted' : ''}`}
                style={{ left: `${position.x / 10}%`, top: `${position.y / 5.2}%` }}
                onClick={() => onSelect(neuron.id)}
                aria-label={`${neuron.id} nöronunu seç${isActive ? ', şu anda aktif' : ''}`}
              >
                <span className="neuron-node__topline">
                  <strong>{neuron.id}</strong>
                  {isActive && <span className="micro-badge">AKTİF</span>}
                  {candidate && <span className="candidate-rank">{candidate.rank}</span>}
                </span>
                <MatrixGrid matrix={neuron.matrix} mode="compact" label={`${neuron.id} matrisi`} />
                {candidate && (
                  <span className={`candidate-result candidate-result--${candidate.outcome}`}>
                    {candidate.outcome === 'accepted' ? <Check size={12} /> : candidate.outcome === 'rejected' ? <X size={12} /> : null}
                    {candidate.outcome === 'accepted' ? 'kabul' : candidate.outcome === 'rejected' ? 'ret' : percent(candidate.signalCompatibility)}
                  </span>
                )}
              </button>
            )
          })}
        </div>

        {count === 1 && simulation.step === 0 && (
          <div className="stage-empty-card">
            <span className="stage-empty-card__icon"><Network size={20} /></span>
            <div>
              <strong>Henüz çıkış adayı yok</strong>
              <p>İlk sinyali N1 işleyecek. Dolaşım için ikinci bir nöron ekleyin.</p>
            </div>
            <Button compact icon={<Plus size={14} />} onClick={onAdd} disabled={!canEdit}>Ekle</Button>
          </div>
        )}

        {simulation.stopReason && (
          <div className="stage-status-card" role="status">
            <AlertTriangle size={17} />
            <span>{stopReasonCopy(simulation.stopReason)}</span>
          </div>
        )}

        {(last?.candidates?.length ?? 0) > 0 && (
          <div className="candidate-strip" aria-label="Aday değerlendirme sırası">
            <span className="candidate-strip__label">Kontrol sırası</span>
            {last.candidates.map((candidate: any) => (
              <span key={candidate.neuronId} className={`candidate-chip candidate-chip--${candidate.outcome}`}>
                <b>{candidate.rank}</b> {candidate.neuronId}
                {candidate.outcome === 'accepted' ? <Check size={12} /> : candidate.outcome === 'rejected' ? <X size={12} /> : null}
              </span>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

function RelationshipMatrix({
  kind,
  network,
  editable,
  onEvidenceChange,
}: {
  kind: 'habit' | 'similarity'
  network: NetworkState
  editable?: boolean
  onEvidenceChange?: (row: number, column: number, value: number) => void
}) {
  const values = kind === 'habit' ? network.habitProbabilities : network.neuronSimilarities
  return (
    <div className="relationship-wrap">
      <div
        className="relationship-grid"
        role="grid"
        style={{ '--matrix-count': network.neurons.length } as CSSProperties}
      >
        <span className="relationship-corner" />
        {network.neurons.map((neuron) => <strong key={`head-${neuron.id}`}>{neuron.id}</strong>)}
        {network.neurons.map((rowNeuron, rowIndex) => (
          <div className="relationship-row" key={rowNeuron.id} role="row">
            <strong>{rowNeuron.id}</strong>
            {network.neurons.map((columnNeuron, columnIndex) => {
              const value = values[rowIndex]?.[columnIndex] ?? 0
              const self = rowIndex === columnIndex
              const red = Math.round(249 - Math.abs(value) * 42)
              const green = Math.round(247 - Math.abs(value) * 211)
              const blue = Math.round(242 - Math.abs(value) * 196)
              return (
                <div
                  key={`${rowNeuron.id}-${columnNeuron.id}`}
                  className={`relationship-cell ${self ? 'relationship-cell--self' : ''} ${value < 0 ? 'relationship-cell--negative' : ''}`}
                  role="gridcell"
                  title={`${rowNeuron.id} → ${columnNeuron.id}: ${value.toFixed(4)}`}
                  style={{ backgroundColor: self && kind === 'habit' ? undefined : `rgb(${red} ${green} ${blue})` }}
                >
                  {kind === 'habit' && editable && !self ? (
                    <label>
                      <span>{percent(value)}</span>
                      <input
                        type="number"
                        min="0"
                        step="0.0001"
                        value={network.habitEvidence[rowIndex][columnIndex].toFixed(4)}
                        onChange={(event) => onEvidenceChange?.(rowIndex, columnIndex, Number(event.target.value))}
                        aria-label={`${rowNeuron.id} ile ${columnNeuron.id} ham bağlantı kanıtı`}
                      />
                    </label>
                  ) : (
                    <>
                      <span>{kind === 'habit' ? percent(value) : value.toFixed(4)}</span>
                      {kind === 'habit' && !self && <small>W {network.habitEvidence[rowIndex][columnIndex].toFixed(1)}</small>}
                    </>
                  )}
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

function App() {
  const restored = useMemo(restoreSession, [])
  const [simulation, setSimulation] = useState(restored.simulation)
  const [seed, setSeed] = useState(restored.seed)
  const [speed, setSpeed] = useState(restored.speed)
  const [selectedId, setSelectedId] = useState(simulation.network.neurons[0].id)
  const [paintTool, setPaintTool] = useState<PaintTool>('positive')
  const [intensity, setIntensity] = useState(0.7)
  const [dockTab, setDockTab] = useState<DockTab>('timeline')
  const [inspectorMode, setInspectorMode] = useState<InspectorMode>('step')
  const [inspectedStep, setInspectedStep] = useState<number | null>(null)
  const [showResetDialog, setShowResetDialog] = useState(false)
  const [notice, setNotice] = useState<{ message: string; tone: 'success' | 'error' } | null>(null)
  const importRef = useRef<HTMLInputElement>(null)
  const resetButtonRef = useRef<HTMLButtonElement>(null)
  const resetDialogRef = useRef<HTMLDivElement>(null)

  const last = simulation.timeline.at(-1) as any
  const inspectedLog =
    inspectedStep === null
      ? last
      : (simulation.timeline.find((entry) => entry.step === inspectedStep) as any) ?? last
  const selectedNeuron =
    simulation.network.neurons.find((neuron) => neuron.id === selectedId) ??
    simulation.network.neurons[0]
  const currentNeuron =
    simulation.network.neurons.find((neuron) => neuron.id === simulation.currentNeuronId) ??
    simulation.network.neurons[0]

  const preview = useMemo(() => {
    if (inspectedLog) {
      return {
        incoming: inspectedLog.incomingSignal as Matrix,
        neuron: inspectedLog.neuronMatrix as Matrix,
        transformed: inspectedLog.transformedSignal as Matrix,
        delta: inspectedLog.delta as Matrix,
        compatibility: inspectedLog.compatibility as number,
        alpha: inspectedLog.alpha as number,
        changeMagnitude: inspectedLog.changeMagnitude as number,
      }
    }
    const compatibility = cosineSimilarity(simulation.signal, currentNeuron.matrix)
    const alpha = clamp(compatibility)
    const transformed = simulation.signal.map((row, rowIndex) =>
      row.map((value, columnIndex) =>
        (1 - alpha) * value + alpha * currentNeuron.matrix[rowIndex][columnIndex],
      ),
    )
    const delta = transformed.map((row, rowIndex) =>
      row.map((value, columnIndex) => Math.abs(value - simulation.signal[rowIndex][columnIndex])),
    )
    const changeMagnitude = Math.sqrt(delta.flat().reduce((sum, value) => sum + value * value, 0))
    return { incoming: simulation.signal, neuron: currentNeuron.matrix, transformed, delta, compatibility, alpha, changeMagnitude }
  }, [inspectedLog, simulation.signal, currentNeuron])

  const pressureObject = simulation.pressure as unknown as any
  const accumulatedPressure =
    typeof pressureObject === 'number'
      ? pressureObject
      : pressureObject?.accumulated ?? pressureObject?.smoothed ?? pressureObject?.value ?? 0
  const instantPressure =
    last?.pressure?.instantaneous ?? last?.pressure?.instant ?? last?.pressure?.growth ?? last?.pressure ?? 0
  const status = statusCopy(simulation)
  const canEditTopology = simulation.status !== 'running'
  const canEditEvidence = canEditTopology && simulation.step === 0

  useEffect(() => {
    const payload = {
      network: simulation.network,
      initialSignal: simulation.initialSignal,
      seed,
      speed,
      maxSteps: simulation.maxSteps,
    }
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  }, [simulation.network, simulation.initialSignal, simulation.maxSteps, seed, speed])

  useEffect(() => {
    if (simulation.status !== 'running') return
    const delay = 1850 - speed * 300
    const timer = window.setInterval(() => {
      setSimulation((current) => stepSimulation(current))
    }, Math.max(180, delay))
    return () => window.clearInterval(timer)
  }, [simulation.status, speed])

  useEffect(() => {
    if (!notice) return
    const timer = window.setTimeout(() => setNotice(null), 2600)
    return () => window.clearTimeout(timer)
  }, [notice])

  useEffect(() => {
    if (!showResetDialog) return
    resetDialogRef.current?.focus()
  }, [showResetDialog])

  function notify(message: string, tone: 'success' | 'error' = 'success') {
    setNotice({ message, tone })
  }

  function closeResetDialog() {
    setShowResetDialog(false)
    window.setTimeout(() => resetButtonRef.current?.focus(), 0)
  }

  function handleResetDialogKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      event.preventDefault()
      closeResetDialog()
      return
    }
    if (event.key !== 'Tab') return
    const controls = resetDialogRef.current?.querySelectorAll<HTMLElement>(
      'button:not(:disabled), [href], input:not(:disabled), [tabindex]:not([tabindex="-1"])',
    )
    if (!controls?.length) return
    const first = controls[0]
    const lastControl = controls[controls.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      lastControl.focus()
    } else if (!event.shiftKey && document.activeElement === lastControl) {
      event.preventDefault()
      first.focus()
    }
  }

  function startSimulation() {
    setInspectorMode('step')
    setInspectedStep(null)
    setSimulation((current) => {
      if (current.status === 'stopped') {
        const fresh = freshSimulation(current.network, current.initialSignal, seed, current.maxSteps)
        return { ...fresh, status: 'running' }
      }
      return { ...current, status: 'running', stopReason: null }
    })
  }

  function pauseSimulation() {
    setSimulation((current) =>
      current.status === 'running'
        ? { ...current, status: 'paused', stopReason: 'user-stopped' }
        : current,
    )
  }

  function singleStep() {
    setInspectorMode('step')
    setInspectedStep(null)
    setSimulation((current) => {
      const base = current.status === 'stopped'
        ? freshSimulation(current.network, current.initialSignal, seed, current.maxSteps)
        : current
      const stepped = stepSimulation({ ...base, status: 'running', stopReason: null })
      return stepped.status === 'running' ? { ...stepped, status: 'paused' } : stepped
    })
  }

  function resetExperiment() {
    setSimulation((current) => freshSimulation(current.network, current.initialSignal, seed, current.maxSteps))
    setInspectorMode('step')
    setInspectedStep(null)
    notify('Deney sıfırlandı; kas hafızası korundu.')
  }

  function confirmNetworkReset() {
    const next = freshSimulation(defaultNetwork(), starterSignalMatrix(), DEFAULT_SEED, 25)
    setSimulation(next)
    setSeed(DEFAULT_SEED)
    setSelectedId('N1')
    setInspectorMode('edit')
    setInspectedStep(null)
    setShowResetDialog(false)
    window.localStorage.removeItem(STORAGE_KEY)
    notify('Ağ ve kas hafızası başlangıca döndü.')
  }

  function updateInitialSignal(matrix: Matrix) {
    setSimulation((current) => ({
      ...current,
      initialSignal: cloneMatrix(matrix),
      signal: current.step === 0 ? cloneMatrix(matrix) : current.signal,
    }))
  }

  function randomizeSignal() {
    updateInitialSignal(seededMatrix(seed))
  }

  function addNeuron() {
    if (!canEditTopology) return
    const nextNumber =
      Math.max(0, ...simulation.network.neurons.map((neuron) => Number(neuron.id.replace(/\D/g, '')) || 0)) + 1
    const neuron: Neuron = {
      id: `N${nextNumber}`,
      matrix: seededMatrix(seed + nextNumber * 977),
    }
    setSimulation((current) => ({
      ...current,
      network: expandNetworkForNewNeuron(current.network, neuron, 1),
      status: current.status === 'idle' ? 'idle' : 'paused',
      stopReason: null,
    }))
    setSelectedId(neuron.id)
    setInspectorMode('edit')
    notify(`${neuron.id} eklendi; yeni bağlantı kanıtları W = 1.`)
  }

  function updateNeuronMatrix(matrix: Matrix) {
    setSimulation((current) => {
      const neurons = current.network.neurons.map((neuron) =>
        neuron.id === selectedNeuron.id ? { ...neuron, matrix: cloneMatrix(matrix) } : neuron,
      )
      return {
        ...current,
        network: {
          ...current.network,
          neurons,
          neuronSimilarities: computeNeuronSimilarityMatrix(neurons),
        },
      }
    })
  }

  function updateEvidence(row: number, column: number, value: number) {
    if (!canEditEvidence || row === column || !Number.isFinite(value)) return
    setSimulation((current) => {
      const habitEvidence = current.network.habitEvidence.map((values) => [...values])
      habitEvidence[row][column] = Math.max(0, value)
      return {
        ...current,
        network: {
          ...current.network,
          habitEvidence,
          habitProbabilities: computeHabitProbabilities(habitEvidence),
        },
      }
    })
  }

  function changeSeed(value: number) {
    if (!Number.isFinite(value)) return
    setSeed(Math.trunc(value))
    if (simulation.step === 0) {
      setSimulation((current) => freshSimulation(current.network, current.initialSignal, Math.trunc(value), current.maxSteps))
    }
  }

  function changeMaxSteps(value: number) {
    const next = Math.max(1, Math.min(250, Math.trunc(value) || 1))
    setSimulation((current) => ({ ...current, maxSteps: next }))
  }

  function exportNetwork() {
    const data = JSON.stringify(
      {
        format: 'sinaps-sandbox',
        version: 1,
        exportedAt: new Date().toISOString(),
        network: simulation.network,
        initialSignal: simulation.initialSignal,
        seed,
        maxSteps: simulation.maxSteps,
      },
      null,
      2,
    )
    const url = URL.createObjectURL(new Blob([data], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `sinaps-ag-${new Date().toISOString().slice(0, 10)}.json`
    anchor.click()
    URL.revokeObjectURL(url)
    notify('Ağ durumu JSON olarak dışa aktarıldı.')
  }

  async function importNetwork(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    try {
      const parsed = JSON.parse(await file.text())
      if (parsed.format !== 'sinaps-sandbox') {
        throw new Error('format')
      }
      const evidence = parsed.network.habitEvidence as number[][]
      const signal = parsed.initialSignal as Matrix
      if (!isValidNetworkPayload(parsed.network.neurons, evidence, signal)) {
        throw new Error('network')
      }
      const network: NetworkState = {
        neurons: parsed.network.neurons,
        habitEvidence: evidence,
        habitProbabilities: computeHabitProbabilities(evidence),
        neuronSimilarities: computeNeuronSimilarityMatrix(parsed.network.neurons),
      }
      const nextSeed = Number.isFinite(parsed.seed) ? parsed.seed : seed
      setSimulation(freshSimulation(network, signal, nextSeed, parsed.maxSteps ?? 25))
      setSeed(nextSeed)
      setSelectedId(network.neurons[0].id)
      setInspectedStep(null)
      notify('Ağ doğrulandı ve güvenle içe aktarıldı.')
    } catch {
      notify('Bu dosya geçerli bir Sinaps Sandbox ağı değil.', 'error')
    } finally {
      event.target.value = ''
    }
  }

  const dockTabs: Array<{ id: DockTab; label: string; icon: ReactNode }> = [
    { id: 'timeline', label: 'Zaman çizgisi', icon: <History size={15} /> },
    { id: 'habit', label: 'Kas Hafızası H', icon: <BrainCircuit size={15} /> },
    { id: 'similarity', label: 'Gerçek Benzerlik K', icon: <Focus size={15} /> },
    { id: 'pressure', label: 'Oluşma Baskısı', icon: <Gauge size={15} /> },
    { id: 'log', label: 'Adım Günlüğü', icon: <FileJson size={15} /> },
  ]

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><BrainCircuit size={24} /></span>
          <div>
            <strong>SİNAPS</strong>
            <span>GÖRSEL SANDBOX · v0.1</span>
          </div>
        </div>

        <div className="transport" aria-label="Simülasyon kontrolleri">
          {simulation.status === 'running' ? (
            <Button icon={<Pause size={16} />} onClick={pauseSimulation}>Duraklat</Button>
          ) : (
            <Button tone="primary" icon={<Play size={16} fill="currentColor" />} onClick={startSimulation}>Başlat</Button>
          )}
          <Button icon={<SkipForward size={16} />} onClick={singleStep} disabled={simulation.status === 'running'}>
            Tek adım
          </Button>
          <div className="speed-control">
            <span>HIZ</span>
            <input
              type="range"
              min="1"
              max="5"
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
              aria-label="Simülasyon hızı"
            />
            <b>{speed}×</b>
          </div>
        </div>

        <div className="session-controls">
          <a className="icon-button suite-link" href="/metin-laboratuvari/" title="Metin Laboratuvarını aç" aria-label="Metin Laboratuvarını aç">
            <FileText size={16} />
          </a>
          <a className="icon-button suite-link" href="/noron-sentezi/" title="Nöron Sentez Laboratuvarını aç" aria-label="Nöron Sentez Laboratuvarını aç">
            <Sparkles size={17} />
          </a>
          <label className="number-control">
            <span>SEED</span>
            <input
              type="number"
              value={seed}
              onChange={(event) => changeSeed(Number(event.target.value))}
              disabled={simulation.status === 'running' || simulation.step > 0}
              title={simulation.step > 0 ? 'Seed değiştirmek için deneyi sıfırlayın' : undefined}
            />
          </label>
          <label className="number-control number-control--short">
            <span>ÜST SINIR</span>
            <input
              type="number"
              value={simulation.maxSteps}
              min="1"
              max="250"
              onChange={(event) => changeMaxSteps(Number(event.target.value))}
            />
          </label>
          <button className="icon-button" onClick={exportNetwork} title="JSON dışa aktar" aria-label="JSON dışa aktar">
            <Download size={16} />
          </button>
          <button className="icon-button" onClick={() => importRef.current?.click()} title="JSON içe aktar" aria-label="JSON içe aktar">
            <Upload size={16} />
          </button>
          <input ref={importRef} type="file" accept="application/json" hidden onChange={importNetwork} />
          <button ref={resetButtonRef} className="icon-button icon-button--danger" onClick={() => setShowResetDialog(true)} title="Bütün ağı sıfırla" aria-label="Bütün ağı sıfırla">
            <Trash2 size={16} />
          </button>
        </div>
      </header>

      <div className="status-strip">
        <span className={`status-pill status-pill--${status.tone}`}><i /> {status.label}</span>
        <span><b>ADIM</b> {simulation.step} / {simulation.maxSteps}</span>
        <span><b>AKTİF</b> {simulation.currentNeuronId}</span>
        <span><b>UYUM</b> {decimal(last?.compatibility ?? preview.compatibility)}</span>
        <span><b>α</b> {decimal(last?.alpha ?? preview.alpha)}</span>
        <span><b>‖Δ‖</b> {decimal(last?.changeMagnitude ?? preview.changeMagnitude)}</span>
        {simulation.loop.detected && <span className="loop-badge"><RotateCcw size={12} /> Döngü algılandı</span>}
        <span className="save-state"><Save size={12} /> Yerelde kayıtlı</span>
      </div>
      <span className="sr-only" aria-live="polite">
        {simulation.stopReason ? stopReasonCopy(simulation.stopReason) : `Simülasyon ${status.label.toLocaleLowerCase('tr-TR')}.`}
      </span>

      <main className="workspace-grid">
        <section className="panel signal-panel">
          <PanelHeading eyebrow="BAŞLANGIÇ" title="Sinyal editörü" trailing={<span className="matrix-size">32 × 32</span>} />
          <p className="panel-intro">Ham sinyali boyayın. Pozitif ve negatif değerler yol boyunca korunur.</p>
          <MatrixEditor
            matrix={simulation.initialSignal}
            onChange={updateInitialSignal}
            tool={paintTool}
            onToolChange={setPaintTool}
            intensity={intensity}
            onIntensityChange={setIntensity}
            readonly={simulation.status === 'running'}
            label="Başlangıç sinyali"
            mode="large"
            showValues={false}
          />
          <div className="signal-actions">
            <Button compact icon={<Shuffle size={14} />} onClick={randomizeSignal} disabled={!canEditTopology}>Seed ile üret</Button>
            <Button compact tone="ghost" icon={<X size={14} />} onClick={() => updateInitialSignal(zeroMatrix())} disabled={!canEditTopology}>Sıfırla</Button>
          </div>
          <div className="sidebar-divider" />
          <div className="experiment-card">
            <span className="experiment-card__icon"><RotateCcw size={17} /></span>
            <div>
              <strong>Yeni deney</strong>
              <p>Sinyali ve geçici yolu sıfırlar; W kas hafızasını korur.</p>
            </div>
            <button className="text-button" onClick={resetExperiment}>Sıfırla</button>
          </div>
          <div className="memory-note">
            <BrainCircuit size={16} />
            <span><b>Kalıcı kas hafızası</b> Deneyler arasında {simulation.timeline.length ? 'güncellendi' : 'hazır'}.</span>
          </div>
        </section>

        <NetworkStage
          simulation={simulation}
          selectedId={selectedId}
          onSelect={(id) => { setSelectedId(id); setInspectorMode('step'); setInspectedStep(null) }}
          onAdd={addNeuron}
          canEdit={canEditTopology}
        />

        <section className="panel inspector-panel">
          <PanelHeading
            eyebrow="İNCELEYİCİ"
            title={inspectorMode === 'edit' ? `${selectedNeuron.id} matrisi` : `Adım ${inspectedLog?.step ?? 'önizleme'}`}
            trailing={
              <div className="segmented-control">
                <button className={inspectorMode === 'step' ? 'active' : ''} onClick={() => setInspectorMode('step')}>Akış</button>
                <button className={inspectorMode === 'edit' ? 'active' : ''} onClick={() => setInspectorMode('edit')} disabled={!canEditTopology}>Düzenle</button>
              </div>
            }
          />

          {inspectorMode === 'edit' ? (
            <>
              <p className="panel-intro">{selectedNeuron.id} prototipini boyayın. Bu işlem gerçek benzerlik K'yı yeniden hesaplar.</p>
              <MatrixEditor
                matrix={selectedNeuron.matrix}
                onChange={updateNeuronMatrix}
                tool={paintTool}
                onToolChange={setPaintTool}
                intensity={intensity}
                onIntensityChange={setIntensity}
                readonly={!canEditTopology}
                label={`${selectedNeuron.id} prototip matrisi`}
                mode="large"
                showValues={false}
              />
              <Button compact icon={<Shuffle size={14} />} onClick={() => updateNeuronMatrix(seededMatrix(seed + Number(selectedNeuron.id.slice(1)) * 977))}>
                Seed ile yeniden üret
              </Button>
            </>
          ) : (
            <>
              {inspectedStep !== null && inspectedStep !== last?.step && (
                <div className="history-inspection-note">
                  <History size={14} />
                  <span>Geçmişteki {inspectedStep}. adımı inceliyorsunuz.</span>
                  <button onClick={() => setInspectedStep(null)}>Son adıma dön</button>
                </div>
              )}
              <div className="matrix-triptych">
                <div><span>GELEN S<sub>t</sub></span><MatrixGrid matrix={preview.incoming} mode="compact" label="Gelen sinyal" /></div>
                <ChevronRight className="triptych-arrow" size={14} />
                <div><span>NÖRON N<sub>i</sub></span><MatrixGrid matrix={preview.neuron} mode="compact" label="Mevcut nöron" /></div>
                <ChevronRight className="triptych-arrow" size={14} />
                <div><span>DÖNÜŞEN S<sub>t+1</sub></span><MatrixGrid matrix={preview.transformed} mode="compact" label="Dönüşmüş sinyal" /></div>
              </div>

              <div className="metric-pair">
                <div className="metric-card metric-card--accent">
                  <span>UYUM c<sub>i</sub></span>
                  <strong>{decimal(preview.compatibility)}</strong>
                  <small>kosinüs benzerliği</small>
                </div>
                <div className="metric-card">
                  <span>DEĞİŞİM ORANI α<sub>t</sub></span>
                  <strong>{decimal(preview.alpha)}</strong>
                  <small>uyumdan türetilir</small>
                </div>
              </div>

              <div className="delta-card">
                <div>
                  <span className="eyebrow">DEĞİŞİM HARİTASI |Δ<sub>t</sub>|</span>
                  <strong>{decimal(preview.changeMagnitude)} toplam büyüklük</strong>
                </div>
                <MatrixGrid matrix={preview.delta} mode="compact" label="Mutlak değişim haritası" />
              </div>

              <div className="formula-card">
                <Sparkles size={15} />
                <span>S<sub>t+1</sub> = (1 − α<sub>t</sub>) S<sub>t</sub> + α<sub>t</sub> N<sub>i</sub></span>
              </div>
            </>
          )}
        </section>
      </main>

      <section className="bottom-dock">
        <div className="dock-summary">
          <div className="dock-tabs" role="tablist" aria-label="Alt bilgi görünümleri">
            {dockTabs.map((tab) => (
              <button
                key={tab.id}
                role="tab"
                aria-selected={dockTab === tab.id}
                className={dockTab === tab.id ? 'active' : ''}
                onClick={() => setDockTab(tab.id)}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>
          <div className="pressure-mini">
            <span>NÖRON OLUŞMA BASKISI</span>
            <div className="pressure-bar"><i style={{ width: `${clamp(accumulatedPressure) * 100}%` }} /></div>
            <strong>{percent(accumulatedPressure)}</strong>
          </div>
        </div>

        <div className="dock-content">
          {dockTab === 'timeline' && (
            <div className="timeline-view">
              <div className="timeline-path">
                <span className="timeline-start"><Zap size={14} /> HAM SİNYAL</span>
                <span className="timeline-node timeline-node--initial">
                  <ChevronRight size={14} />
                  <b>{simulation.path[0] ?? simulation.currentNeuronId}</b>
                  <small>giriş</small>
                </span>
                {simulation.timeline.map((entry) => (
                  <button
                    className={`timeline-node ${inspectedLog?.step === entry.step ? 'timeline-node--active' : ''}`}
                    key={entry.step}
                    onClick={() => { setInspectedStep(entry.step); setInspectorMode('step') }}
                    aria-label={`${entry.step}. adımı incele: ${entry.fromNeuronId} nöronundan ${entry.toNeuronId ?? 'eşleşmeyen sonuca'}`}
                  >
                    <ChevronRight size={14} />
                    <b>{entry.toNeuronId ?? '∅'}</b>
                    <small>{entry.step}. adım</small>
                  </button>
                ))}
              </div>
              <div className="timeline-stats">
                <div><span>TOPLAM ADIM</span><strong>{simulation.step}</strong></div>
                <div><span>İNCELENEN GEÇİŞ</span><strong>{inspectedLog?.toNeuronId ? `${inspectedLog.fromNeuronId} → ${inspectedLog.toNeuronId}` : '—'}</strong></div>
                <div><span>H ÖNCE → SONRA</span><strong>{inspectedLog?.toNeuronId ? `${percent(inspectedLog.habitBefore)} → ${percent(inspectedLog.habitAfter)}` : '—'}</strong></div>
                <div><span>SONUÇ</span><strong>{inspectedLog?.unmatched ? 'Eşleşmedi' : inspectedLog?.toNeuronId ? 'Kabul edildi' : 'Bekliyor'}</strong></div>
              </div>
            </div>
          )}

          {dockTab === 'habit' && (
            <div className="matrix-dock-view">
              <div className="dock-explanation">
                <span className="dock-icon"><BrainCircuit size={19} /></span>
                <div>
                  <h3>Öğrenilmiş kas hafızası H</h3>
                  <p>Satırlar %100'e normalize edilir. Büyük değer, adayın daha önce değerlendirilmesi demektir; kabul garantisi değildir.</p>
                  <span className="range-label">ARALIK 0 → 1 · HAM KANIT W HÜCRE İÇİNDE</span>
                </div>
                {!canEditEvidence && <span className="locked-note"><Settings2 size={13} /> W yalnızca adım 0'da düzenlenir</span>}
              </div>
              <RelationshipMatrix kind="habit" network={simulation.network} editable={canEditEvidence} onEvidenceChange={updateEvidence} />
            </div>
          )}

          {dockTab === 'similarity' && (
            <div className="matrix-dock-view">
              <div className="dock-explanation">
                <span className="dock-icon dock-icon--neutral"><Focus size={19} /></span>
                <div>
                  <h3>Gerçek matris benzerliği K</h3>
                  <p>Nöron prototiplerinin kosinüs benzerliği. Yollar kullanıldıkça değişmez; yalnızca prototip düzenlenince yeniden hesaplanır.</p>
                  <span className="range-label">ARALIK −1 → +1 · NEGATİF DEĞERLER TARALI</span>
                </div>
              </div>
              <RelationshipMatrix kind="similarity" network={simulation.network} />
            </div>
          )}

          {dockTab === 'pressure' && (
            <div className="pressure-view">
              <div className="pressure-gauge" style={{ '--pressure': clamp(accumulatedPressure) } as CSSProperties}>
                <div><strong>{percent(accumulatedPressure)}</strong><span>BİRİKMİŞ P<sub>t</sub></span></div>
              </div>
              <div className="pressure-copy">
                <span className="eyebrow">YALNIZCA GÖSTERGE</span>
                <h3>Yeni bir temsil ne kadar gerekli?</h3>
                <p>Alışılmadık yol ile mevcut nöronların açıklayamadığı sinyal aynı anda yükseldiğinde baskı artar. Sistem bu sürümde otomatik nöron oluşturmaz.</p>
              </div>
              <div className="pressure-equation">
                <div><span>ANLIK G<sub>t</sub></span><strong>{decimal(typeof instantPressure === 'number' ? instantPressure : 0)}</strong></div>
                <div><span>YUMUŞATMA λ</span><strong>0.800</strong></div>
                <div><span>BİRİKMİŞ P<sub>t</sub></span><strong>{decimal(accumulatedPressure)}</strong></div>
                <code>G<sub>t</sub> = D<sub>t</sub> × E<sub>t</sub></code>
              </div>
            </div>
          )}

          {dockTab === 'log' && (
            <div className="log-view">
              {simulation.timeline.length ? (
                <table>
                  <thead><tr><th>Adım</th><th>Geçiş</th><th>Sıra</th><th>Aday</th><th>Kas hafızası H</th><th>Sinyal uyumu</th><th>Seçilme olasılığı</th><th>Sonuç</th></tr></thead>
                  <tbody>
                    {[...simulation.timeline].reverse().flatMap((entry) =>
                      entry.candidates.length
                        ? entry.candidates.map((candidate) => (
                            <tr key={`${entry.step}-${candidate.neuronId}`}>
                              <td><b>{entry.step}</b></td>
                              <td>{entry.fromNeuronId} → {entry.toNeuronId ?? '∅'}</td>
                              <td><span className="rank-box">{candidate.rank}</span></td>
                              <td><b>{candidate.neuronId}</b></td>
                              <td>{percent(candidate.habitProbability)}</td>
                              <td>{percent(candidate.signalCompatibility)}</td>
                              <td>{percent(candidate.selectionProbability)}</td>
                              <td><span className={`outcome outcome--${candidate.outcome}`}>
                                {candidate.outcome === 'accepted' ? <Check size={12} /> : candidate.outcome === 'rejected' ? <X size={12} /> : <CircleGauge size={12} />}
                                {candidate.outcome === 'accepted' ? 'Kabul edildi' : candidate.outcome === 'rejected' ? 'Reddedildi' : 'Değerlendirilmedi'}
                              </span></td>
                            </tr>
                          ))
                        : [(
                            <tr key={`${entry.step}-empty`}>
                              <td><b>{entry.step}</b></td>
                              <td>{entry.fromNeuronId} → ∅</td>
                              <td>—</td>
                              <td colSpan={4}>Çıkış adayı bulunamadı</td>
                              <td><span className="outcome outcome--not-evaluated"><AlertTriangle size={12} /> Durduruldu</span></td>
                            </tr>
                          )],
                    )}
                  </tbody>
                </table>
              ) : (
                <div className="log-empty"><Activity size={24} /><strong>Henüz aday değerlendirmesi yok</strong><p>Tek adım ilerletildiğinde sıralı kabul günlüğü burada görünecek.</p></div>
              )}
            </div>
          )}
        </div>
      </section>

      {notice && (
        <div className={`toast toast--${notice.tone}`} role={notice.tone === 'error' ? 'alert' : 'status'}>
          {notice.tone === 'error' ? <AlertTriangle size={15} /> : <Check size={15} />}
          {notice.message}
        </div>
      )}

      {showResetDialog && (
        <div className="modal-backdrop" role="presentation" onMouseDown={closeResetDialog}>
          <div
            ref={resetDialogRef}
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="reset-title"
            tabIndex={-1}
            onKeyDown={handleResetDialogKeyDown}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <span className="modal-icon"><AlertTriangle size={23} /></span>
            <span className="eyebrow">GERİ ALINAMAZ SIFIRLAMA</span>
            <h2 id="reset-title">Bütün ağı başlangıca döndür?</h2>
            <p>Nöronlar, W ve H kas hafızası, bütün geçmiş ve oluşma baskısı silinecek. Yalnızca düzenlenebilir N1 kalacak.</p>
            <div className="modal-actions">
              <Button onClick={closeResetDialog}>Vazgeç</Button>
              <Button tone="danger" icon={<Trash2 size={15} />} onClick={confirmNetworkReset}>Ağı sıfırla</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
