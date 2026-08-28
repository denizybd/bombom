import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  Braces,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Clock3,
  Database,
  Download,
  FileText,
  FlaskConical,
  FolderOpen,
  GitBranch,
  Grid3X3,
  Info,
  Network,
  Plus,
  Pause,
  Play,
  RotateCcw,
  Save,
  ScanSearch,
  Settings2,
  Sparkles,
  Split,
  Upload,
  Waypoints,
  X,
} from 'lucide-react'
import {
  type ChangeEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { HeatmapCanvas, type HeatmapZone } from './components/HeatmapCanvas'
import { SimilarityCanvas } from './components/SimilarityCanvas'
import { SpanTree } from './components/SpanTree'
import {
  DEFAULT_TEXT_LAB_CONFIG,
  MAX_ANALYSIS_NODES,
  countGraphemes,
  composeMatrix,
  createMatrixLayout,
  detailFromAnalysisPair,
  explainContextStep,
  findParagraphRanges,
  relationToFullMatrix,
  type PersistedProject,
  type PipelineProgress,
  type RawTextNeuron,
  type TextLabAnalysis,
  type TextLabConfig,
  type TextSource,
} from './engine'
import {
  listProjects,
  loadProject,
  prepareIdentityV2Migration,
  replaceProjectAnalysis,
  saveProject,
  type ProjectSummary,
} from './persistence/indexedDb'
import {
  analysisMatchesProject,
  createPersistedProject,
  parseProject,
  stringifyProject,
  validateProject,
} from './persistence/projectSchema'
import type { WorkerRequest, WorkerResponse } from './workers/protocol'

type LabView = 'source' | 'tree' | 'flow' | 'similarity'
type RunState = 'idle' | 'running' | 'rebuilding' | 'cancelling'

const SAMPLE_TEXT = `Finans piyasaları sabah sakin açıldı. Yatırımcılar yeni verileri beklerken işlem hacmi düşük kaldı.

Öğleden sonra finans hisseleri yükseldi. Piyasanın yönü değişse de yatırımcıların temkinli davranışı sürdü.`

const INITIAL_SOURCE: TextSource = {
  id: 'source-1',
  title: 'İlk ham metin deneyi',
  rawText: SAMPLE_TEXT,
}

function createSourceId(): string {
  const token = typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
    : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
  return `SRC-${token.toUpperCase()}`
}

function createBlankSource(): TextSource {
  return { id: createSourceId(), title: 'Yeni ham metin', rawText: '' }
}

function analyzeWithFreshWorker(
  source: TextSource,
  config: TextLabConfig,
  signal: AbortSignal,
  onProgress: (progress: PipelineProgress) => void,
): Promise<TextLabAnalysis> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./workers/pipeline.worker.ts', import.meta.url), { type: 'module' })
    const requestId = `rebuild-${source.id}-${Date.now()}`
    const cleanup = () => {
      signal.removeEventListener('abort', abort)
      worker.terminate()
    }
    const abort = () => {
      cleanup()
      reject(new DOMException('Yeniden üretim durduruldu.', 'AbortError'))
    }
    if (signal.aborted) return abort()
    signal.addEventListener('abort', abort, { once: true })
    worker.onerror = () => {
      cleanup()
      reject(new Error('Yeniden üretim işçisi beklenmeyen biçimde durdu.'))
    }
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data
      if (response.requestId !== requestId) return
      if (response.type === 'progress') onProgress(response.progress)
      else if (response.type === 'result') {
        cleanup()
        resolve(response.analysis)
      } else if (response.type === 'cancelled') {
        abort()
      } else if (response.type === 'error') {
        cleanup()
        reject(new Error(response.message))
      }
    }
    const request: WorkerRequest = { type: 'analyze', requestId, source, config }
    worker.postMessage(request)
  })
}

function withIdentityRebuildLock<T>(operation: () => Promise<T>): Promise<T> {
  if (typeof navigator !== 'undefined' && navigator.locks) {
    return navigator.locks.request(
      'metin-laboratuvari:identity-v2-rebuild',
      { mode: 'exclusive' },
      operation,
    )
  }
  return operation()
}

const STAGES: Array<{ id: LabView; label: string; short: string; icon: ReactNode }> = [
  { id: 'source', label: 'Kaynak', short: 'Metni koru', icon: <FileText size={15} /> },
  { id: 'tree', label: 'Bölünme', short: 'Ham nöronlar', icon: <Split size={15} /> },
  { id: 'flow', label: 'Bağlam', short: 'Mikro izler', icon: <Waypoints size={15} /> },
  { id: 'similarity', label: 'Benzerlik', short: 'Tam K matrisi', icon: <Grid3X3 size={15} /> },
]

function Button({
  children,
  icon,
  tone = 'default',
  compact = false,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode
  tone?: 'default' | 'primary' | 'ghost' | 'danger'
  compact?: boolean
}) {
  return (
    <button className={`tl-button tl-button--${tone} ${compact ? 'tl-button--compact' : ''}`} {...props}>
      {icon}{children}
    </button>
  )
}

function formatNumber(value: number): string {
  return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 })
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`
  return `${(milliseconds / 1000).toFixed(2)} sn`
}

function visibleText(value: string, showMarks: boolean): string {
  if (!showMarks) return value
  return value.replace(/ /g, '·').replace(/\t/g, '⇥').replace(/\n/g, '↵\n')
}

function nodeKind(node: RawTextNeuron): string {
  if (node.depth === 0) return 'Paragraf kökü'
  if (node.childIds.length === 0) return 'Ham yaprak'
  return 'Ara parça'
}

function lineageFor(node: RawTextNeuron, nodes: readonly RawTextNeuron[]): RawTextNeuron[] {
  const byId = new Map(nodes.map((candidate) => [candidate.id, candidate]))
  const lineage: RawTextNeuron[] = []
  let current: RawTextNeuron | undefined = node
  while (current) {
    lineage.push(current)
    current = current.parentId ? byId.get(current.parentId) : undefined
  }
  return lineage.reverse()
}

interface ScaleEstimate {
  graphemes: number
  paragraphs: number
  neurons: number
  pairs: number
  bytes: number
  upperNeurons: number
  upperPairs: number
  level: 'safe' | 'warn' | 'large'
}

function estimateScale(source: TextSource, config: TextLabConfig): ScaleEstimate {
  const canonical = source.rawText.replace(/\r\n?/g, '\n').normalize('NFC')
  const graphemes = countGraphemes(canonical)
  const paragraphs = Math.max(1, findParagraphRanges(canonical).length)
  const leaves = Math.max(paragraphs, Math.ceil(graphemes / config.minimumLeafGraphemes))
  const neurons = Math.max(paragraphs, leaves * 2 - paragraphs)
  const pairs = (neurons * (neurons - 1)) / 2
  const upperNeurons = Math.max(paragraphs, graphemes * 2 - paragraphs)
  const upperPairs = (upperNeurons * (upperNeurons - 1)) / 2
  const layout = createMatrixLayout(config)
  const boundedNeurons = Math.min(upperNeurons, MAX_ANALYSIS_NODES)
  // Matrices + K/H + RGBA generation/backing buffers, with an object/clone margin.
  const bytes = Math.round(
    (boundedNeurons * layout.cellCount * 4 + boundedNeurons * boundedNeurons * 16) * 1.3,
  )
  return {
    graphemes,
    paragraphs,
    neurons,
    pairs,
    upperNeurons,
    upperPairs,
    bytes,
    level:
      upperNeurons > MAX_ANALYSIS_NODES || upperPairs > 5_000_000
        ? 'large'
        : upperPairs > 750_000
          ? 'warn'
          : 'safe',
  }
}

function StageRail({
  view,
  hasAnalysis,
  stale,
  locked,
  onChange,
}: {
  view: LabView
  hasAnalysis: boolean
  stale: boolean
  locked: boolean
  onChange: (view: LabView) => void
}) {
  return (
    <nav className="tl-stage-rail" aria-label="Metin işleme aşamaları">
      {STAGES.map((stage, index) => {
        const disabled = locked || (stage.id !== 'source' && !hasAnalysis)
        return (
          <button
            key={stage.id}
            className={`${view === stage.id ? 'is-active' : ''} ${disabled ? 'is-disabled' : ''}`}
            disabled={disabled}
            onClick={() => onChange(stage.id)}
          >
            <span className="tl-stage-index">{index + 1}</span>
            {stage.icon}
            <span><b>{stage.label}</b><small>{stage.short}</small></span>
            {stale && stage.id !== 'source' && <i title="Kaynak değişti" />}
          </button>
        )
      })}
    </nav>
  )
}

function SourceView({
  source,
  config,
  estimate,
  runState,
  progress,
  stale,
  error,
  onSourceChange,
  onConfigChange,
  onTextFile,
  onStart,
  onCancel,
  projects,
  persistenceAvailable,
  onNewSource,
  onOpenProject,
}: {
  source: TextSource
  config: TextLabConfig
  estimate: ScaleEstimate
  runState: RunState
  progress: PipelineProgress | null
  stale: boolean
  error: string | null
  onSourceChange: (source: TextSource) => void
  onConfigChange: (config: TextLabConfig) => void
  onTextFile: (event: ChangeEvent<HTMLInputElement>) => void
  onStart: () => void
  onCancel: () => void
  projects: ProjectSummary[]
  persistenceAvailable: boolean
  onNewSource: () => void
  onOpenProject: (sourceId: string) => void
}) {
  const [acceptedScale, setAcceptedScale] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const requiresConfirmation = estimate.level !== 'safe'
  const canStart = source.rawText.trim().length > 0 && (!requiresConfirmation || acceptedScale)

  useEffect(() => setAcceptedScale(false), [estimate.level, estimate.pairs])

  return (
    <main className="tl-source-layout" aria-busy={runState !== 'idle'}>
      <section className="tl-panel tl-source-card">
        <div className="tl-panel-heading">
          <div><span>KAYNAK 01</span><h1>Ham metni içeri al</h1></div>
          <span className="tl-source-assurance"><Check size={13} /> Kanonik metin korunur</span>
        </div>
        <p className="tl-panel-copy">
          Bu aşamada hiçbir anlam etiketi, hazır dil modeli veya dikkat kapısı kullanılmaz. Metin yalnızca kendi yüzey izleriyle işlenir.
        </p>
        <label className="tl-field">
          <span>KAYNAK ADI</span>
          <input disabled={runState !== 'idle'} value={source.title} onChange={(event) => onSourceChange({ ...source, title: event.target.value })} />
        </label>
        <label className="tl-field tl-field--grow">
          <span>HAM METİN</span>
          <textarea
            value={source.rawText}
            onChange={(event) => onSourceChange({ ...source, rawText: event.target.value })}
            spellCheck={false}
            disabled={runState !== 'idle'}
            placeholder="Paragrafları veya kitap metnini buraya yapıştır…"
          />
        </label>
        <div className="tl-source-actions">
          <Button compact icon={<FolderOpen size={14} />} onClick={() => fileRef.current?.click()} disabled={runState !== 'idle'}>.txt dosyası aç</Button>
          <Button
            compact
            tone="ghost"
            icon={<RotateCcw size={14} />}
            onClick={() => onSourceChange({
              ...source,
              title: INITIAL_SOURCE.title,
              rawText: INITIAL_SOURCE.rawText,
            })}
            disabled={runState !== 'idle'}
          >Örnek metin</Button>
          <input ref={fileRef} type="file" accept="text/plain,.txt" hidden onChange={onTextFile} />
          <span>{formatNumber(source.rawText.length)} karakter · {formatNumber(estimate.graphemes)} grapheme</span>
        </div>

        <div className="tl-rule-strip">
          <div><b>NFC</b><span>Unicode düzeni</span></div>
          <div><b>↵ · ? !</b><span>Boşluk ve noktalama</span></div>
          <div><b>TR</b><span>Türkçe karakterler</span></div>
          <div><b>I 🔒</b><span>Kimlik byte-sabit</span></div>
          <div><b>R ↻</b><span>Bağlam plastik</span></div>
        </div>
      </section>

      <aside className="tl-source-side">
        <section className="tl-panel tl-archive-card">
          <div className="tl-panel-heading tl-panel-heading--compact">
            <div><span>YEREL ARŞİV</span><h2>{persistenceAvailable ? `${projects.length} ayrı metin` : 'Geçici mod'}</h2></div>
            <Button compact icon={<Plus size={14} />} onClick={onNewSource} disabled={runState !== 'idle'}>Yeni metin</Button>
          </div>
          <div className="tl-project-list">
            {projects.map((project) => (
              <button
                key={project.sourceId}
                className={project.sourceId === source.id ? 'is-active' : ''}
                onClick={() => onOpenProject(project.sourceId)}
                disabled={!persistenceAvailable || runState !== 'idle' || project.sourceId === source.id}
              >
                <span><b>{project.title || 'Adsız kaynak'}</b><small>{project.characterCount.toLocaleString('tr-TR')} karakter</small></span>
                <strong>{project.neuronCount ? `${project.neuronCount.toLocaleString('tr-TR')} HN` : 'Taslak'}</strong>
              </button>
            ))}
          </div>
        </section>

        <section className={`tl-panel tl-scale-card tl-scale-card--${estimate.level}`}>
          <div className="tl-panel-heading tl-panel-heading--compact">
            <div><span>İŞ YÜKÜ ÖNİZLEMESİ</span><h2>{estimate.level === 'safe' ? 'Uygun deney' : estimate.level === 'warn' ? 'Cihazı zorlayabilir' : 'Çok büyük deney'}</h2></div>
            {estimate.level === 'safe' ? <Check size={19} /> : <AlertTriangle size={19} />}
          </div>
          <div className="tl-scale-metrics">
            <div><span>TAHMİNİ HAM NÖRON</span><strong>≈ {formatNumber(estimate.neurons)}</strong></div>
            <div><span>BENZERSİZ ÇİFT</span><strong>≈ {formatNumber(estimate.pairs)}</strong></div>
            <div><span>DENEY SINIRINDA TEPE BELLEK</span><strong>≤ {formatBytes(estimate.bytes)}</strong></div>
            <div><span>OLASI N ÜST SINIRI</span><strong>≤ {formatNumber(estimate.upperNeurons)}</strong></div>
          </div>
          <div className="tl-equation">N × (N − 1) / 2</div>
          <p>Tüm çiftler hesaplanır. Sistem gizlice örnekleme, birleştirme veya nöron silme yapmaz.</p>
          <p className="tl-safety-boundary">Bu sürümün tam çift deney sınırı <b>{formatNumber(MAX_ANALYSIS_NODES)} gerçek ham nörondur.</b> Kesin sayı bölünme sırasında bunu aşarsa işlem K ayrılmadan durur.</p>
          {requiresConfirmation && (
            <label className="tl-confirm-scale">
              <input type="checkbox" checked={acceptedScale} disabled={runState !== 'idle'} onChange={(event) => setAcceptedScale(event.target.checked)} />
              <span>Bu iş yükünü anladım; tam deneyi başlat.</span>
            </label>
          )}
        </section>

        <section className="tl-panel tl-settings-card">
          <div className="tl-panel-heading tl-panel-heading--compact">
            <div><span>BÖLÜNME DENEYİ</span><h2>Yaprak eşiği</h2></div><Settings2 size={17} />
          </div>
          <label className="tl-range-field">
            <span>En küçük parça <b>{config.minimumLeafGraphemes} grapheme</b></span>
            <input
              type="range"
              min="2"
              max="12"
              value={config.minimumLeafGraphemes}
              disabled={runState !== 'idle'}
              onChange={(event) => onConfigChange({ ...config, minimumLeafGraphemes: Number(event.target.value) })}
            />
          </label>
          <p>Nokta → boşluk → grapheme sırası kullanılır. Sol ve sağ çocuk, üst parçayı eksiksiz yeniden oluşturur.</p>
        </section>

        {stale && (
          <div className="tl-inline-warning"><AlertTriangle size={16} /><span>Kaynak değişti. Görünen eski sonuç yeniden üretildiğinde değiştirilecek.</span></div>
        )}
        {error && <div className="tl-inline-error" role="alert"><X size={16} /><span>{error}</span></div>}

        {runState !== 'idle' && progress ? (
          <section className="tl-panel tl-progress-card" aria-live="polite">
            <div className="tl-progress-heading"><span>{progress.message}</span><b>%{Math.round(progress.ratio * 100)}</b></div>
            <div className="tl-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress.ratio * 100)}><i style={{ width: `${progress.ratio * 100}%` }} /></div>
            <div className="tl-progress-meta"><span>{progress.stage.toLocaleUpperCase('tr-TR')}</span><span>{formatNumber(progress.completed)} / {formatNumber(progress.total)}</span></div>
            <Button tone="danger" icon={<CircleStop size={15} />} onClick={onCancel} disabled={runState === 'cancelling'}>
              {runState === 'cancelling' ? 'Güvenle durduruluyor…' : 'İşlemi durdur'}
            </Button>
          </section>
        ) : (
          <Button tone="primary" icon={<Sparkles size={17} />} onClick={onStart} disabled={!canStart}>
            Ham nöronları üret
          </Button>
        )}
      </aside>
    </main>
  )
}

function BranchView({
  analysis,
  selected,
  showMarks,
  onShowMarks,
  onSelect,
}: {
  analysis: TextLabAnalysis
  selected: RawTextNeuron
  showMarks: boolean
  onShowMarks: (value: boolean) => void
  onSelect: (id: string) => void
}) {
  const byId = new Map(analysis.nodes.map((node) => [node.id, node]))
  const parent = selected.parentId ? byId.get(selected.parentId) ?? null : null
  const children = selected.childIds.map((id) => byId.get(id)).filter(Boolean) as RawTextNeuron[]
  const root = byId.get(analysis.rootIds[selected.paragraphIndex]) ?? selected
  const before = analysis.source.text.slice(root.start, selected.start)
  const after = analysis.source.text.slice(selected.end, root.end)
  const lineage = lineageFor(selected, analysis.nodes)

  return (
    <div className="tl-observation-view">
      <div className="tl-view-heading">
        <div><span>SEÇİLİ DAL</span><h2>Kanonik metnin kayıpsız bölünmesi</h2></div>
        <label className="tl-switch"><input type="checkbox" checked={showMarks} onChange={(event) => onShowMarks(event.target.checked)} /><span>Biçim işaretleri</span></label>
      </div>
      <div className="tl-breadcrumbs">
        {lineage.map((node, index) => (
          <button key={node.id} onClick={() => onSelect(node.id)}><span>{index === 0 ? 'P' : index}</span>{node.id}</button>
        ))}
      </div>
      <div className="tl-source-highlight" aria-label="Seçili ham nöronun paragraf içindeki yeri">
        <span>{visibleText(before, showMarks)}</span>
        <mark>{visibleText(selected.text, showMarks)}</mark>
        <span>{visibleText(after, showMarks)}</span>
      </div>
      <div className="tl-branch-diagram">
        {parent ? (
          <button className="tl-span-card tl-span-card--parent" onClick={() => onSelect(parent.id)}>
            <span>ÜST PARÇA · {parent.id}</span><p>{visibleText(parent.text, showMarks)}</p><small>{parent.graphemeCount} grapheme</small>
          </button>
        ) : <div className="tl-root-note"><FileText size={17} /><span>Bu düğüm paragrafın başlangıç köküdür.</span></div>}
        <ChevronRight className="tl-branch-arrow tl-branch-arrow--down" size={18} />
        <button className="tl-span-card tl-span-card--selected">
          <span>SEÇİLİ OLUŞ · {selected.id}</span><p>{visibleText(selected.text, showMarks)}</p><small>{nodeKind(selected)} · derinlik {selected.depth}</small>
        </button>
        {children.length > 0 ? (
          <>
            <div className="tl-split-line"><i /><Split size={17} /><i /></div>
            <div className="tl-child-grid">
              {children.map((child, index) => (
                <button key={child.id} className="tl-span-card" onClick={() => onSelect(child.id)}>
                  <span>{index === 0 ? 'SOL' : 'SAĞ'} ÇOCUK · {child.id}</span><p>{visibleText(child.text, showMarks)}</p><small>{child.graphemeCount} grapheme</small>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="tl-leaf-note"><Check size={16} /><span>Yaprak eşiğine ulaşıldı; bu ham oluş daha fazla bölünmedi.</span></div>
        )}
      </div>
      <div className="tl-honesty-note"><Info size={16} /><p><b>Bu bir kavram değildir.</b> Metindeki tek bir oluşumun, kaynağı ve konumu korunmuş ham kaydıdır.</p></div>
    </div>
  )
}

function FlowView({
  analysis,
  selected,
  onSelect,
}: {
  analysis: TextLabAnalysis
  selected: RawTextNeuron
  onSelect: (id: string) => void
}) {
  const lineage = useMemo(() => lineageFor(selected, analysis.nodes), [analysis.nodes, selected])
  const [cursor, setCursor] = useState(lineage.length - 1)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    setCursor(lineage.length - 1)
    setPlaying(false)
  }, [lineage.length, selected.id])

  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => {
      setCursor((current) => {
        if (current >= lineage.length - 1) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 900)
    return () => window.clearInterval(timer)
  }, [lineage.length, playing])

  const current = lineage[Math.max(0, cursor)] ?? selected
  const explanation = explainContextStep(current, analysis.nodes, analysis.layout, analysis.config)
  const before = relationToFullMatrix(explanation.before, analysis.layout)
  const trace = relationToFullMatrix(explanation.trace, analysis.layout)
  const after = relationToFullMatrix(explanation.after, analysis.layout)

  return (
    <div className="tl-flow-view">
      <div className="tl-view-heading">
        <div><span>BAĞLAM AKIŞI</span><h2>Duvar üzerinde kalan mikro izler</h2></div>
        <div className="tl-flow-controls">
          <button onClick={() => setCursor(0)} aria-label="Başa dön"><RotateCcw size={14} /></button>
          <button onClick={() => setCursor((value) => Math.max(0, value - 1))} aria-label="Önceki iz"><ChevronLeft size={15} /></button>
          <button className="is-primary" onClick={() => { if (cursor >= lineage.length - 1) setCursor(0); setPlaying((value) => !value) }} aria-label={playing ? 'Akışı duraklat' : 'Akışı oynat'}>{playing ? <Pause size={15} /> : <Play size={15} fill="currentColor" />}</button>
          <button onClick={() => setCursor((value) => Math.min(lineage.length - 1, value + 1))} aria-label="Sonraki iz"><ChevronRight size={15} /></button>
        </div>
      </div>
      <div className="tl-flow-path">
        {lineage.map((node, index) => (
          <button key={node.id} className={`${index === cursor ? 'is-active' : ''} ${index < cursor ? 'is-past' : ''}`} onClick={() => setCursor(index)}>
            <span>{index === 0 ? <FileText size={13} /> : <GitBranch size={13} />}</span><b>{node.id}</b><small>{index === 0 ? 'kök' : `${index}. iz`}</small>
          </button>
        ))}
      </div>

      {current.parentId ? (
        <>
          <div className="tl-flow-triptych">
            <div><span>ÖNCE · ρR<sub>p</sub></span><HeatmapCanvas matrix={before} layout={analysis.layout} zone="relation" label="Üstten taşınan ilişki alanı" compact /></div>
            <ChevronRight size={18} />
            <div><span>BIRAKILAN İZ</span><HeatmapCanvas matrix={trace} layout={analysis.layout} zone="relation" label="Ebeveyn ve kardeş mikro izi" compact /></div>
            <ChevronRight size={18} />
            <div><span>SONRA · R<sub>c</sub></span><HeatmapCanvas matrix={after} layout={analysis.layout} zone="relation" label="Çocuğun ilişki alanı" compact /></div>
          </div>
          <div className="tl-flow-metrics">
            <div><span>ÜST UYUMU</span><strong>{current.trace.parentIdentityCompatibility.toFixed(4)}</strong><small>izin büyüklüğünü etkiler</small></div>
            <div><span>ÜST KAZANCI</span><strong>{current.trace.parentGain.toFixed(4)}</strong><small>g(c,p)</small></div>
            <div><span>KARDEŞ KAZANCI</span><strong>{current.trace.siblingGain.toFixed(4)}</strong><small>yönlü mikro iz</small></div>
            <div><span>İLİŞKİ NORMU</span><strong>{current.relationNorm.toFixed(4)}</strong><small>‖R‖</small></div>
          </div>
          <details className="tl-science-detail">
            <summary><Braces size={14} /> Bilimsel ayrıntı</summary>
            <code>R<sub>c</sub> = clip(ρR<sub>p</sub> + a·g(c,p)·T<sub>P</sub>q<sub>p</sub> + b·g(c,s)·T<sub>dir</sub>q<sub>s</sub>)</code>
            <p>Koordinatların semantik adı yoktur. Sol ve sağ sıra, aynı ilişki alanına uygulanan farklı fiziksel dönüşümlerle iz bırakır.</p>
          </details>
        </>
      ) : (
        <div className="tl-root-flow-empty"><Waypoints size={25} /><h3>Paragraf kökünde dış bağlam yok</h3><p>R alanı sıfırdan başlar. İlk çocuk, paragraf kimliğinin ve kardeşinin mikro izini alır.</p></div>
      )}
      <button className="tl-flow-target" onClick={() => onSelect(current.id)}>İncelenen oluşu sağ panelde aç · {current.id}</button>
    </div>
  )
}

function SimilarityView({
  analysis,
  selectedIndex,
  comparisonIndex,
  onPairSelect,
}: {
  analysis: TextLabAnalysis
  selectedIndex: number
  comparisonIndex: number | null
  onPairSelect: (row: number, column: number) => void
}) {
  const first = analysis.nodes[selectedIndex]
  const fallback = selectedIndex === 0 ? 1 : 0
  const secondIndex = comparisonIndex ?? (analysis.nodes.length > 1 ? fallback : selectedIndex)
  const second = analysis.nodes[secondIndex]
  const detail = detailFromAnalysisPair(
    first,
    second,
    analysis.layout,
    analysis.config,
    analysis.stats.medianRelationNorm,
  )

  return (
    <div className="tl-similarity-view">
      <div className="tl-view-heading">
        <div><span>TAM BENZERLİK K</span><h2>{formatNumber(analysis.stats.uniquePairCount)} çift gerçekten hesaplandı</h2></div>
        <div className="tl-similarity-legend"><span><i className="positive" /> pozitif K</span><span><i className="negative" /> negatif K</span><span><i className="diagonal" /> öz</span><span><i className="selected" /> seçili</span></div>
      </div>
      <div className="tl-similarity-main">
        <SimilarityCanvas
          similarities={analysis.similarities}
          count={analysis.nodes.length}
          selectedRow={selectedIndex}
          selectedColumn={secondIndex}
          labels={analysis.nodes.map((node) => node.id)}
          onSelect={onPairSelect}
        />
        <div className="tl-pair-card">
          <span>SEÇİLİ ÇİFT</span>
          <div className="tl-pair-text"><b>{first.id}</b><p>{first.text}</p></div>
          <X size={14} />
          <div className="tl-pair-text"><b>{second.id}</b><p>{second.text}</p></div>
          <div className="tl-pair-score"><span>BİRLEŞİK K</span><strong>{detail.combined.toFixed(4)}</strong></div>
          <div className="tl-pair-diagnostics">
            <div><span>KİMLİK</span><b>{detail.identity.toFixed(4)}</b></div>
            <div><span>İLİŞKİ</span><b>{detail.relation.toFixed(4)}</b></div>
            <div><span>R KANITI</span><b>{detail.relationConfidence.toFixed(4)}</b></div>
            <div><span>BAŞLANGIÇ H⁰</span><b>{analysis.habitProbabilities[selectedIndex * analysis.nodes.length + secondIndex].toFixed(4)}</b></div>
          </div>
        </div>
      </div>
      <div className="tl-matrix-note"><ScanSearch size={16} /><p>K, işaretli koordinat uzayındaki kosinüstür; semantik anlam veya zıtlık kanıtı değildir. Tanısal I/R değerleri yeni bir ilişki vektörü oluşturmaz.</p></div>
    </div>
  )
}

function Inspector({
  analysis,
  selected,
  selectedIndex,
  zone,
  onZoneChange,
  onSelect,
  onCompare,
}: {
  analysis: TextLabAnalysis
  selected: RawTextNeuron
  selectedIndex: number
  zone: HeatmapZone
  onZoneChange: (zone: HeatmapZone) => void
  onSelect: (id: string) => void
  onCompare: (id: string) => void
}) {
  const neighbors = useMemo(() => {
    const count = analysis.nodes.length
    return analysis.nodes
      .map((node, index) => ({
        node,
        index,
        similarity: analysis.similarities[selectedIndex * count + index],
        habit: analysis.habitProbabilities[selectedIndex * count + index],
      }))
      .filter(({ index }) => index !== selectedIndex)
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, 6)
  }, [analysis, selectedIndex])

  return (
    <aside className="tl-panel tl-inspector">
      <div className="tl-inspector-heading">
        <div><span>HAM NÖRON</span><h2>{selected.id}</h2></div><span className="tl-raw-badge">BİRLEŞTİRİLMEDİ</span>
      </div>
      <div className="tl-node-text"><span>TAM METİN PARÇASI</span><p>{selected.text}</p></div>
      <div className="tl-node-meta">
        <div><span>TÜR</span><b>{nodeKind(selected)}</b></div>
        <div><span>KONUM</span><b>{selected.start}—{selected.end}</b></div>
        <div><span>DERİNLİK</span><b>{selected.depth}</b></div>
        <div><span>GRAPHEME</span><b>{selected.graphemeCount}</b></div>
      </div>
      <div className="tl-zone-tabs" role="tablist" aria-label="Isı haritası bölgesi">
        {(['all', 'identity', 'relation'] as HeatmapZone[]).map((candidate) => (
          <button key={candidate} role="tab" aria-selected={zone === candidate} className={zone === candidate ? 'is-active' : ''} onClick={() => onZoneChange(candidate)}>
            {candidate === 'all' ? 'Tüm harita' : candidate === 'identity' ? 'Kimlik I' : 'İlişki R'}
          </button>
        ))}
      </div>
      <HeatmapCanvas matrix={composeMatrix(selected.identity, selected.relation, analysis.layout)} layout={analysis.layout} zone={zone} label={`${selected.id} ısı haritası`} />
      <div className="tl-norm-row"><span><i className="identity" /> ‖I‖ <b>{selected.identityNorm.toFixed(4)}</b></span><span><i className="relation" /> ‖R‖ <b>{selected.relationNorm.toFixed(4)}</b></span></div>
      <div className="tl-vector-contract">
        <span><b>I KİLİTLİ</b><code>{selected.identityDigest.slice(0, 10)}</code></span>
        <span><b>R SÜRÜMÜ</b><code>r{selected.contextRevision} · {selected.contextSupport.toFixed(1)} kanıt</code></span>
      </div>

      <div className="tl-neighbor-heading"><span>EN YAKIN HAM OLUŞLAR</span><small>K azalan sıra</small></div>
      <div className="tl-neighbor-list">
        {neighbors.length ? neighbors.map(({ node, similarity, habit }) => (
          <div key={node.id} className="tl-neighbor-row">
            <button onClick={() => onSelect(node.id)}><b>{node.id}</b><span>{node.text || '∅'}</span></button>
            <button className="tl-neighbor-score" onClick={() => onCompare(node.id)} title="Çifti benzerlik ekranında karşılaştır"><strong>{similarity.toFixed(4)}</strong><small>H⁰ {habit.toFixed(4)}</small></button>
          </div>
        )) : <p className="tl-empty-copy">Karşılaştırılacak ikinci bir oluş yok.</p>}
      </div>
    </aside>
  )
}

function Workspace({
  analysis,
  view,
  selectedId,
  onSelectedId,
  onView,
}: {
  analysis: TextLabAnalysis
  view: Exclude<LabView, 'source'>
  selectedId: string
  onSelectedId: (id: string) => void
  onView: (view: LabView) => void
}) {
  const [zone, setZone] = useState<HeatmapZone>('all')
  const [showMarks, setShowMarks] = useState(false)
  const [comparisonId, setComparisonId] = useState<string | null>(null)
  const selectedIndex = Math.max(0, analysis.nodes.findIndex((node) => node.id === selectedId))
  const selected = analysis.nodes[selectedIndex]
  const comparisonIndex = comparisonId
    ? analysis.nodes.findIndex((node) => node.id === comparisonId)
    : null

  const choosePair = (row: number, column: number) => {
    onSelectedId(analysis.nodes[row].id)
    setComparisonId(analysis.nodes[column].id)
  }

  return (
    <main className="tl-workspace">
      <section className="tl-panel tl-tree-panel">
        <div className="tl-panel-heading tl-panel-heading--compact">
          <div><span>KAYNAK AĞACI</span><h2>{formatNumber(analysis.nodes.length)} ham oluş</h2></div><GitBranch size={18} />
        </div>
        <p className="tl-tree-intro">Her tekrar ayrı kalır. Bir düğümü seçerek kaynak, iz ve benzerlik görünümünü eşzamanlı değiştir.</p>
        <SpanTree nodes={analysis.nodes} rootIds={analysis.rootIds} selectedId={selected.id} onSelect={onSelectedId} />
      </section>

      <section className="tl-panel tl-observation-panel">
        {view === 'tree' && <BranchView analysis={analysis} selected={selected} showMarks={showMarks} onShowMarks={setShowMarks} onSelect={onSelectedId} />}
        {view === 'flow' && <FlowView analysis={analysis} selected={selected} onSelect={onSelectedId} />}
        {view === 'similarity' && (
          <SimilarityView
            analysis={analysis}
            selectedIndex={selectedIndex}
            comparisonIndex={comparisonIndex !== null && comparisonIndex >= 0 ? comparisonIndex : null}
            onPairSelect={choosePair}
          />
        )}
      </section>

      <Inspector
        analysis={analysis}
        selected={selected}
        selectedIndex={selectedIndex}
        zone={zone}
        onZoneChange={setZone}
        onSelect={onSelectedId}
        onCompare={(id) => { setComparisonId(id); onView('similarity') }}
      />
    </main>
  )
}

export default function TextLabApp() {
  const [source, setSource] = useState<TextSource>(() => ({
    ...INITIAL_SOURCE,
    id: createSourceId(),
  }))
  const [config, setConfig] = useState<TextLabConfig>({ ...DEFAULT_TEXT_LAB_CONFIG })
  const [analysis, setAnalysis] = useState<TextLabAnalysis | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [view, setView] = useState<LabView>('source')
  const [runState, setRunState] = useState<RunState>('idle')
  const [progress, setProgress] = useState<PipelineProgress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stale, setStale] = useState(false)
  const [hydrated, setHydrated] = useState(false)
  const [initializationState, setInitializationState] = useState<'loading' | 'ready' | 'failed'>('loading')
  const [initializationAttempt, setInitializationAttempt] = useState(0)
  const [saveState, setSaveState] = useState<'loading' | 'saved' | 'saving' | 'unavailable'>('loading')
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const workerRef = useRef<Worker | null>(null)
  const requestIdRef = useRef<string | null>(null)
  const rebuildAbortRef = useRef<AbortController | null>(null)
  const storageRevisionsRef = useRef(new Map<string, number>())
  const saveQueueRef = useRef<Promise<void>>(Promise.resolve())
  const sourceRef = useRef(source)
  const configRef = useRef(config)
  const importRef = useRef<HTMLInputElement>(null)
  const estimate = useMemo(() => estimateScale(source, config), [config, source])
  sourceRef.current = source
  configRef.current = config
  const canUseWorkspace = hydrated || initializationState === 'failed'

  const persistSnapshot = useCallback((
    snapshotSource: TextSource,
    snapshotConfig: TextLabConfig,
    snapshotAnalysis: TextLabAnalysis | null,
  ): Promise<PersistedProject> => {
    const operation = saveQueueRef.current
      .catch(() => undefined)
      .then(async () => {
        const expectedRevision = storageRevisionsRef.current.get(snapshotSource.id) ?? 0
        const saved = await saveProject(createPersistedProject(
          snapshotSource,
          snapshotConfig,
          snapshotAnalysis,
          expectedRevision,
        ))
        storageRevisionsRef.current.set(saved.source.id, saved.storageRevision)
        return saved
      })
    saveQueueRef.current = operation.then(() => undefined, () => undefined)
    return operation
  }, [])

  useEffect(() => {
    const worker = new Worker(new URL('./workers/pipeline.worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data
      if (response.requestId !== requestIdRef.current) return
      if (response.type === 'progress') {
        setProgress(response.progress)
      } else if (response.type === 'result') {
        const matchesCurrentProject = analysisMatchesProject(
          sourceRef.current,
          configRef.current,
          response.analysis,
        )
        setAnalysis(response.analysis)
        setSelectedId(response.analysis.rootIds[0] ?? response.analysis.nodes[0]?.id ?? '')
        setStale(!matchesCurrentProject)
        setRunState('idle')
        setProgress(null)
        setView(matchesCurrentProject ? 'tree' : 'source')
        setError(null)
      } else if (response.type === 'cancelled') {
        setRunState('idle')
        setProgress(null)
      } else if (response.type === 'error') {
        setRunState('idle')
        setProgress(null)
        setError(response.message)
      }
    }
    return () => worker.terminate()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    rebuildAbortRef.current = controller
    setHydrated(false)
    setInitializationState('loading')
    setSaveState('loading')

    const initialize = async () => {
      await withIdentityRebuildLock(async () => {
        const migration = await prepareIdentityV2Migration()
        if (!migration.required) return

        setRunState('rebuilding')
        setSaveState('saving')
        const failures: string[] = []
        const total = migration.projects.length
        for (let index = 0; index < migration.projects.length; index += 1) {
          if (controller.signal.aborted) throw new DOMException('Yeniden üretim durduruldu.', 'AbortError')
          const project = migration.projects[index]
          try {
            const nextAnalysis = await analyzeWithFreshWorker(
              project.source,
              project.config,
              controller.signal,
              (local) => setProgress({
                ...local,
                ratio: (index + local.ratio) / Math.max(1, total),
                message: `${index + 1}/${total} · ${project.source.title || 'Adsız kaynak'} · ${local.message}`,
              }),
            )
            await replaceProjectAnalysis(project.source.id, nextAnalysis)
          } catch (caught) {
            if (caught instanceof DOMException && caught.name === 'AbortError') throw caught
            failures.push(`${project.source.title || project.source.id}: ${caught instanceof Error ? caught.message : 'yeniden üretilemedi'}`)
          }
        }
        if (failures.length) {
          setError(`Bazı kaynaklar yeniden üretilemedi: ${failures.join(' · ')}`)
        }
      })

      if (controller.signal.aborted) throw new DOMException('Yeniden üretim durduruldu.', 'AbortError')
      const project = await loadProject()
      const validProject = project ? validateProject(project) : null
      const projectSummaries = await listProjects()
      if (controller.signal.aborted || rebuildAbortRef.current !== controller) {
        throw new DOMException('Yeniden üretim durduruldu.', 'AbortError')
      }
      if (validProject) {
        storageRevisionsRef.current.set(validProject.source.id, validProject.storageRevision)
        setSource(validProject.source)
        setConfig(validProject.config)
        setAnalysis(validProject.analysis)
        setStale(!analysisMatchesProject(validProject.source, validProject.config, validProject.analysis))
        setSelectedId(validProject.analysis?.rootIds[0] ?? validProject.analysis?.nodes[0]?.id ?? '')
        setView(validProject.analysis ? 'tree' : 'source')
      }
      setProjects(projectSummaries)
      setSaveState('saved')
      setHydrated(true)
      setInitializationState('ready')
    }

    void initialize()
      .catch((caught) => {
        if (rebuildAbortRef.current !== controller) return
        if (caught instanceof DOMException && caught.name === 'AbortError') {
          setError('Kimlik-v2 yeniden üretimi durduruldu. Sayfa yenilendiğinde kaldığı kaynaklardan devam eder.')
        } else {
          setError(caught instanceof Error ? caught.message : 'Yerel arşiv yeni kimlik modeline hazırlanamadı.')
        }
        setSaveState('unavailable')
        setInitializationState('failed')
      })
      .finally(() => {
        if (rebuildAbortRef.current !== controller) return
        setRunState('idle')
        setProgress(null)
        rebuildAbortRef.current = null
      })

    return () => controller.abort()
  }, [initializationAttempt])

  useEffect(() => {
    if (!hydrated || runState !== 'idle') return
    setSaveState((current) => current === 'unavailable' ? current : 'saving')
    const timer = window.setTimeout(() => {
      void persistSnapshot(source, config, analysis)
        .then(async () => {
          setSaveState('saved')
          setProjects(await listProjects())
        })
        .catch(() => setSaveState('unavailable'))
    }, 700)
    return () => window.clearTimeout(timer)
  }, [analysis, config, hydrated, persistSnapshot, runState, source])

  const updateSource = (next: TextSource) => {
    if (!canUseWorkspace || runState !== 'idle') return
    setSource(next)
    if (analysis && !analysisMatchesProject(next, config, analysis)) {
      setAnalysis(null)
      setSelectedId('')
      setStale(false)
      setView('source')
    }
  }

  const updateConfig = (next: TextLabConfig) => {
    if (!canUseWorkspace || runState !== 'idle') return
    setConfig(next)
    if (analysis && !analysisMatchesProject(source, next, analysis)) {
      setAnalysis(null)
      setSelectedId('')
      setStale(false)
      setView('source')
    }
  }

  const openProject = async (sourceId: string) => {
    if (!hydrated || runState !== 'idle' || sourceId === source.id) return
    try {
      await persistSnapshot(source, config, analysis)
      const project = await loadProject(sourceId)
      if (!project) throw new Error('Seçilen metin yerel arşivde bulunamadı.')
      const validProject = validateProject(project)
      storageRevisionsRef.current.set(validProject.source.id, validProject.storageRevision)
      setSource(validProject.source)
      setConfig(validProject.config)
      setAnalysis(validProject.analysis)
      setSelectedId(validProject.analysis?.rootIds[0] ?? validProject.analysis?.nodes[0]?.id ?? '')
      setStale(!analysisMatchesProject(validProject.source, validProject.config, validProject.analysis))
      setView(validProject.analysis ? 'tree' : 'source')
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Metin arşivden açılamadı.')
    }
  }

  const newSource = async () => {
    if (!canUseWorkspace || runState !== 'idle') return
    if (!hydrated) {
      const next = createBlankSource()
      setSource(next)
      setAnalysis(null)
      setSelectedId('')
      setStale(false)
      setView('source')
      setError('Geçici çalışma modu: yeni metin yerel arşive kaydedilmeyecek.')
      return
    }
    try {
      await persistSnapshot(source, config, analysis)
      const next = createBlankSource()
      storageRevisionsRef.current.delete(next.id)
      setSource(next)
      setAnalysis(null)
      setSelectedId('')
      setStale(false)
      setView('source')
      setError(null)
    } catch {
      setError('Mevcut metin kaydedilemediği için yeni metin açılamadı.')
    }
  }

  const startAnalysis = () => {
    if (!canUseWorkspace || !workerRef.current || runState !== 'idle') return
    const requestId = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `request-${Date.now()}-${Math.random().toString(16).slice(2)}`
    requestIdRef.current = requestId
    setRunState('running')
    setError(null)
    setProgress({ stage: 'preparing', completed: 0, total: 1, ratio: 0, message: 'Deney hazırlanıyor…' })
    const request: WorkerRequest = { type: 'analyze', requestId, source, config }
    workerRef.current.postMessage(request)
  }

  const cancelAnalysis = () => {
    if (runState === 'rebuilding' && rebuildAbortRef.current) {
      setRunState('cancelling')
      rebuildAbortRef.current.abort()
      return
    }
    if (!workerRef.current || !requestIdRef.current) return
    setRunState('cancelling')
    const request: WorkerRequest = { type: 'cancel', requestId: requestIdRef.current }
    workerRef.current.postMessage(request)
  }

  const importTextFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file || !canUseWorkspace || runState !== 'idle') return
    let persistenceStarted = false
    try {
      const nextSource: TextSource = {
        id: createSourceId(),
        title: file.name.replace(/\.txt$/i, ''),
        rawText: await file.text(),
      }
      if (hydrated) {
        persistenceStarted = true
        setSaveState('saving')
        await persistSnapshot(source, config, analysis)
        const saved = await persistSnapshot(nextSource, config, null)
        setSource(saved.source)
        setConfig(saved.config)
        setAnalysis(saved.analysis)
        setProjects(await listProjects())
        setSaveState('saved')
      } else {
        setSource(nextSource)
        setAnalysis(null)
      }
      setSelectedId('')
      setStale(false)
      setView('source')
      setError(hydrated ? null : 'Geçici çalışma modu: içe aktarılan metin yerel arşive kaydedilmedi.')
    } catch (caught) {
      if (persistenceStarted) setSaveState('unavailable')
      setError(caught instanceof Error ? caught.message : 'Metin dosyası okunamadı veya kaydedilemedi.')
    } finally {
      event.target.value = ''
    }
  }

  const exportProject = () => {
    if (!canUseWorkspace) return
    const project = createPersistedProject(source, config, analysis)
    const url = URL.createObjectURL(new Blob([stringifyProject(project)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `metin-laboratuvari-${new Date().toISOString().slice(0, 10)}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const importProject = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file || !canUseWorkspace || runState !== 'idle') return
    let persistenceStarted = false
    try {
      const project: PersistedProject = parseProject(await file.text())
      if (!hydrated) {
        setSource({ ...project.source, id: createSourceId() })
        setConfig(project.config)
        setAnalysis(null)
        setSelectedId('')
        setStale(false)
        setView('source')
        setError('Geçici çalışma modu: proje metni yeni kimlikle açıldı; analiz ve yerel kayıt yeniden üretilmeli.')
        return
      }
      persistenceStarted = true
      setSaveState('saving')
      await persistSnapshot(source, config, analysis)
      const existing = await loadProject(project.source.id)
      const importedSource = existing
        ? { ...project.source, id: createSourceId() }
        : project.source
      const importedAnalysis = existing ? null : project.analysis
      storageRevisionsRef.current.set(importedSource.id, 0)
      const saved = await persistSnapshot(importedSource, project.config, importedAnalysis)
      setSource(saved.source)
      setConfig(saved.config)
      setAnalysis(saved.analysis)
      setSelectedId(saved.analysis?.rootIds[0] ?? saved.analysis?.nodes[0]?.id ?? '')
      setStale(false)
      setView(saved.analysis ? 'tree' : 'source')
      setProjects(await listProjects())
      setSaveState('saved')
      setError(existing
        ? 'İçe aktarılan proje kimliği yerel bir metinle çakıştı. Mevcut kayıt korunarak yeni kimlik ayrıldı; analiz yeniden üretilmeli.'
        : null)
    } catch (caught) {
      if (persistenceStarted) setSaveState('unavailable')
      setError(caught instanceof Error ? caught.message : 'Proje dosyası açılamadı.')
      setView('source')
    } finally {
      event.target.value = ''
    }
  }

  const activeAnalysisView = view === 'source' ? 'tree' : view
  const interactionLocked = !canUseWorkspace || runState !== 'idle'
  const visibleRunState: RunState = initializationState === 'loading' ? 'rebuilding' : runState
  const navigationLocked = interactionLocked

  const retryInitialization = () => {
    if (initializationState !== 'failed' || runState !== 'idle') return
    if (
      source.rawText.trim().length > 0 &&
      !window.confirm('Arşiv yeniden açılırken geçici çalışma ekrandan kalkabilir. Devam etmeden önce projeyi dışa aktardın mı?')
    ) return
    setError(null)
    setInitializationAttempt((current) => current + 1)
  }

  return (
    <div className="tl-app-shell">
      <header className="tl-topbar">
        <div className="tl-brand">
          <span className="tl-brand-mark"><Network size={23} /></span>
          <div><strong>METİN LABORATUVARI</strong><span>HAM TEMSİL OLUŞTURUCU · v0.2</span></div>
        </div>
        <div className="tl-topbar-center">
          {visibleRunState !== 'idle' ? <span className="tl-run-pill"><i /> {initializationState === 'loading' ? 'Arşiv açılıyor' : 'İşleniyor'}</span> : analysis ? <span className="tl-ready-pill"><Check size={13} /> Deney hazır</span> : <span className="tl-idle-pill"><FlaskConical size={13} /> Kaynak bekleniyor</span>}
          {initializationState === 'failed' && <span className="tl-stale-pill"><AlertTriangle size={13} /> Geçici çalışma</span>}
          {stale && <span className="tl-stale-pill"><AlertTriangle size={13} /> Sonuç eski kaynakta</span>}
        </div>
        <div className="tl-topbar-actions">
          <a className={`tl-back-link ${navigationLocked ? 'is-disabled' : ''}`} aria-disabled={navigationLocked} tabIndex={navigationLocked ? -1 : undefined} onClick={(event) => navigationLocked && event.preventDefault()} href="/"><ArrowLeft size={14} /> Sinaps</a>
          <a className={`tl-back-link ${navigationLocked ? 'is-disabled' : ''}`} aria-disabled={navigationLocked} tabIndex={navigationLocked ? -1 : undefined} onClick={(event) => navigationLocked && event.preventDefault()} href="/noron-sentezi/"><Sparkles size={14} /> Nöron Sentezi</a>
          <button disabled={!canUseWorkspace} onClick={exportProject} title="Projeyi dışa aktar" aria-label="Projeyi dışa aktar"><Download size={16} /></button>
          <button disabled={interactionLocked} onClick={() => importRef.current?.click()} title="Proje aç" aria-label="Proje aç"><Upload size={16} /></button>
          <input ref={importRef} type="file" accept="application/json" hidden onChange={importProject} />
        </div>
      </header>

      <StageRail view={view} hasAnalysis={Boolean(analysis)} stale={stale} locked={interactionLocked} onChange={setView} />

      <div className="tl-status-strip">
        <span><Database size={12} /><b>KAYNAK</b> {source.title || 'Adsız kaynak'}</span>
        <span><FileText size={12} /><b>PARAGRAF</b> {analysis ? formatNumber(analysis.stats.paragraphCount) : `≈ ${formatNumber(estimate.paragraphs)}`}</span>
        <span><GitBranch size={12} /><b>HAM NÖRON</b> {analysis ? formatNumber(analysis.stats.neuronCount) : `≈ ${formatNumber(estimate.neurons)}`}</span>
        <span><BarChart3 size={12} /><b>ÇİFT</b> {analysis ? formatNumber(analysis.stats.uniquePairCount) : `≈ ${formatNumber(estimate.pairs)}`}</span>
        {analysis && <span><Clock3 size={12} /><b>SÜRE</b> {formatDuration(analysis.stats.durationMs)}</span>}
        <span className="tl-save-state"><Save size={12} /> {saveState === 'saved' ? 'Yerelde kayıtlı' : saveState === 'saving' ? 'Kaydediliyor…' : saveState === 'loading' ? 'Kayıt açılıyor…' : 'Yerel kayıt kullanılamıyor'}</span>
      </div>

      {stale && <div className="tl-global-stale"><AlertTriangle size={14} /><span>Bu analiz önceki kaynak veya ayarlara ait. Yeniden üretmeden güncel deney olarak kullanma.</span></div>}
      {initializationState === 'failed' ? (
        <div className="tl-global-stale tl-init-failure" role="alert">
          <AlertTriangle size={14} />
          <span>Yerel arşiv açılamadı. Geçici modda metin ve analiz kullanabilirsin; hiçbir değişiklik tarayıcıya kaydedilmez. Projeyi dışa aktararak yedekle.</span>
          <button type="button" onClick={retryInitialization}>Arşivi yeniden dene</button>
        </div>
      ) : saveState === 'unavailable' && <div className="tl-global-stale" role="alert"><AlertTriangle size={14} /><span>Yerel arşive yazılamıyor. Çalışmanı dışa aktararak yedekle ve tarayıcı depolama iznini kontrol et.</span></div>}

      {view === 'source' || !analysis ? (
        <SourceView
          source={source}
          config={config}
          estimate={estimate}
          runState={visibleRunState}
          progress={progress}
          stale={stale}
          error={error}
          onSourceChange={updateSource}
          onConfigChange={updateConfig}
          onTextFile={importTextFile}
          onStart={startAnalysis}
          onCancel={cancelAnalysis}
          projects={projects}
          persistenceAvailable={hydrated}
          onNewSource={newSource}
          onOpenProject={openProject}
        />
      ) : (
        <Workspace
          analysis={analysis}
          view={activeAnalysisView}
          selectedId={selectedId || analysis.rootIds[0] || analysis.nodes[0].id}
          onSelectedId={(id) => { if (!interactionLocked) setSelectedId(id) }}
          onView={(next) => { if (!interactionLocked) setView(next) }}
        />
      )}

      {analysis && view !== 'source' && (
        <footer className="tl-analysis-footer">
          <span><Database size={13} /><b>MATRİSLER</b> {formatBytes(analysis.stats.matrixBytes)}</span>
          <span><Grid3X3 size={13} /><b>K + H</b> {formatBytes(analysis.stats.similarityBytes)}</span>
          <span><Braces size={13} /><b>KURAL</b> I fiziksel olarak kilitli · R plastik · K aynı uzay</span>
          <span><Info size={13} /> Sadeleştirme ve kavram oluşumu bu deneyin dışında</span>
        </footer>
      )}
    </div>
  )
}
