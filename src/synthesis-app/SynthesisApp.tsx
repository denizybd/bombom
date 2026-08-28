import {
  Activity,
  Archive,
  ArrowLeft,
  Ban,
  Check,
  CheckCircle2,
  CircleStop,
  Database,
  FileText,
  FlaskConical,
  Info,
  Layers3,
  LockKeyhole,
  Network,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  UsersRound,
  X,
} from 'lucide-react'
import {
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import {
  createSynthesisConfig,
  DEFAULT_SYNTHESIS_CONFIG,
  SynthesisCancelledError,
  type SynthesisCandidate,
  type SynthesisAnalysisInput,
  type SynthesisConfig,
  type SynthesisProgress,
  type SynthesisRun,
} from '../synthesis'
import type { SynthesisWorkerRequest, SynthesisWorkerResponse } from '../synthesis/workerProtocol'
import type { PersistedProject, TextLabAnalysis } from '../text-lab/engine'
import { loadAllProjects } from '../text-lab/persistence/indexedDb'
import { validateProject } from '../text-lab/persistence/projectSchema'
import { SynthesisHeatmap } from './SynthesisHeatmap'
import {
  loadSynthesisArchive,
  saveCandidateDecision,
  type CandidateDecision,
  type SynthesisArchiveRecord,
} from './persistence'

type LabView = 'corpus' | 'candidates' | 'archive'
type RunState = 'idle' | 'running' | 'cancelling'
type DecisionState = 'idle' | 'saving'
type CandidateFilter = 'all' | 'pending' | 'approved' | 'rejected'

const STAGES: Array<{
  id: LabView
  label: string
  short: string
  icon: ReactNode
}> = [
  { id: 'corpus', label: 'Korpus', short: 'Kaynakları seç', icon: <Database size={15} /> },
  { id: 'candidates', label: 'Adaylar', short: 'Kanıtı incele', icon: <Network size={15} /> },
  { id: 'archive', label: 'Sentez arşivi', short: 'Kararları koru', icon: <Archive size={15} /> },
]

function formatInteger(value: number): string {
  return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 })
}

function formatDecimal(value: number, digits = 3): string {
  return value.toLocaleString('tr-TR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

function formatPercent(value: number): string {
  return `%${Math.round(Math.max(-1, Math.min(1, value)) * 100)}`
}

function shortFingerprint(value: string): string {
  const segments = value.split('-')
  const digest = segments.at(-1) ?? value
  return `${segments.length > 1 ? `${segments[0]}-` : ''}${digest.slice(0, 12)}…`
}

function sourceName(projects: readonly PersistedProject[], sourceId: string): string {
  return projects.find((project) => project.source.id === sourceId)?.source.title || 'Adsız kaynak'
}

function Button({
  children,
  icon,
  tone = 'default',
  wide = false,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode
  tone?: 'default' | 'primary' | 'success' | 'danger' | 'ghost'
  wide?: boolean
}) {
  return (
    <button
      className={`ns-button ns-button--${tone} ${wide ? 'ns-button--wide' : ''}`}
      {...props}
    >
      {icon}{children}
    </button>
  )
}

function eligibleNodes(analysis: TextLabAnalysis, config: SynthesisConfig): number {
  return analysis.nodes.reduce((count, node) => {
    const leaf = node.childIds.length === 0
    const short = node.graphemeCount <= config.maximumSpanGraphemes
    const eligible =
      config.nodeSelection === 'all' ||
      (config.nodeSelection === 'leaves' && leaf) ||
      (config.nodeSelection === 'short-spans' && short) ||
      (config.nodeSelection === 'leaves-or-short-spans' && (leaf || short))
    return count + (eligible ? 1 : 0)
  }, 0)
}

function crossSourcePairs(analyses: readonly TextLabAnalysis[], config: SynthesisConfig): number {
  const counts = new Map<string, number>()
  for (const analysis of analyses) {
    counts.set(
      analysis.sourceDigest,
      (counts.get(analysis.sourceDigest) ?? 0) + eligibleNodes(analysis, config),
    )
  }
  let prior = 0
  let total = 0
  for (const count of counts.values()) {
    total += prior * count
    prior += count
  }
  return total
}

function createWorkerAnalysisInput(analysis: TextLabAnalysis): SynthesisAnalysisInput {
  return {
    format: analysis.format,
    version: analysis.version,
    runId: analysis.runId,
    sourceDigest: analysis.sourceDigest,
    identitySpaceId: analysis.identitySpaceId,
    contextSpaceId: analysis.contextSpaceId,
    representationSpaceId: analysis.representationSpaceId,
    source: analysis.source,
    config: analysis.config,
    layout: analysis.layout,
    nodes: analysis.nodes.map((node) => ({
      id: node.id,
      sourceId: node.sourceId,
      paragraphIndex: node.paragraphIndex,
      start: node.start,
      end: node.end,
      text: node.text,
      graphemeCount: node.graphemeCount,
      depth: node.depth,
      parentId: node.parentId,
      childIds: node.childIds,
      splitBoundary: node.splitBoundary,
      identity: node.identity,
      relation: node.relation,
      identityDigest: node.identityDigest,
    })),
  }
}

function decisionFor(
  archive: readonly SynthesisArchiveRecord[],
  candidate: SynthesisCandidate,
): CandidateDecision | null {
  return archive.find((record) => record.candidateFingerprint === candidate.fingerprint)?.decision ?? null
}

function SettingsField({
  label,
  value,
  minimum,
  maximum,
  step,
  disabled,
  onChange,
  format = formatDecimal,
}: {
  label: string
  value: number
  minimum: number
  maximum: number
  step: number
  disabled: boolean
  onChange: (value: number) => void
  format?: (value: number) => string
}) {
  return (
    <div className="ns-field ns-field--wide">
      <div className="ns-field__line"><span>{label}</span><b>{format(value)}</b></div>
      <input
        type="range"
        min={minimum}
        max={maximum}
        step={step}
        value={value}
        disabled={disabled}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  )
}

function CorpusView({
  projects,
  activeSpaceId,
  selectedRunIds,
  config,
  runState,
  progress,
  error,
  onActiveSpace,
  onToggleRun,
  onSelectAll,
  onConfig,
  onRefresh,
  onRun,
  onCancel,
}: {
  projects: PersistedProject[]
  activeSpaceId: string
  selectedRunIds: Set<string>
  config: SynthesisConfig
  runState: RunState
  progress: SynthesisProgress | null
  error: string | null
  onActiveSpace: (id: string) => void
  onToggleRun: (analysis: TextLabAnalysis) => void
  onSelectAll: (analyses: readonly TextLabAnalysis[]) => void
  onConfig: (config: SynthesisConfig) => void
  onRefresh: () => void
  onRun: () => void
  onCancel: () => void
}) {
  const [query, setQuery] = useState('')
  const [acceptedLargeRun, setAcceptedLargeRun] = useState(false)
  const [acceptedIncompleteCorpus, setAcceptedIncompleteCorpus] = useState(false)
  const analyses = projects.flatMap((project) => project.analysis ? [project.analysis] : [])
  const pendingProjects = projects.filter(
    (project) => project.source.rawText.trim().length > 0 && !project.analysis,
  )
  const spaces = useMemo(() => {
    const groups = new Map<string, TextLabAnalysis[]>()
    for (const analysis of analyses) {
      const group = groups.get(analysis.representationSpaceId) ?? []
      group.push(analysis)
      groups.set(analysis.representationSpaceId, group)
    }
    return [...groups.entries()].sort((left, right) => right[1].length - left[1].length)
  }, [analyses])
  const compatible = analyses.filter((analysis) => analysis.representationSpaceId === activeSpaceId)
  const selected = compatible.filter((analysis) => selectedRunIds.has(analysis.runId))
  const selectedSources = new Set(selected.map((analysis) => analysis.sourceDigest)).size
  const selectedNeurons = selected.reduce((sum, analysis) => sum + eligibleNodes(analysis, config), 0)
  const comparisons = crossSourcePairs(selected, config)
  const largeRun = comparisons > 5_000_000
  const canRun =
    runState === 'idle' &&
    (pendingProjects.length === 0 || acceptedIncompleteCorpus) &&
    selectedSources >= config.minimumSources &&
    comparisons > 0 &&
    (!largeRun || acceptedLargeRun)
  const visibleProjects = projects.filter((project) => {
    const search = query.trim().toLocaleLowerCase('tr-TR')
    return !search || project.source.title.toLocaleLowerCase('tr-TR').includes(search)
  })

  useEffect(() => setAcceptedLargeRun(false), [comparisons])
  useEffect(() => setAcceptedIncompleteCorpus(false), [pendingProjects.length])

  return (
    <main className="ns-main ns-corpus-layout">
      <section className="ns-panel ns-source-panel">
        <div className="ns-panel__heading">
          <div>
            <span className="ns-eyebrow">KORPUS 01</span>
            <h1>Ham nöron arşivini seç</h1>
          </div>
          <span className="ns-chip ns-chip--good"><LockKeyhole size={11} /> Ham veri salt okunur</span>
        </div>
        <p className="ns-copy">
          Yalnız aynı temsil parmak izine sahip deneyler karşılaştırılır. Metin Laboratuvarı kayıtları bu program tarafından değiştirilmez.
        </p>
        {pendingProjects.length > 0 && (
          <div className="ns-pending-corpus" role="status">
            <div className="ns-error">
              <RefreshCw size={15} />
              <span>{pendingProjects.length} metin yeniden üretiliyor, analiz bekliyor veya daha önce sınırı aştı. Varsayılan olarak eksik korpusla sentez başlatılmaz.</span>
            </div>
            {analyses.length >= 2 && (
              <label className="ns-large-confirm">
                <input
                  type="checkbox"
                  checked={acceptedIncompleteCorpus}
                  disabled={runState !== 'idle'}
                  onChange={(event) => setAcceptedIncompleteCorpus(event.target.checked)}
                />
                <span>Hazır olmayan {pendingProjects.length} metni yalnız bu koşudan hariç tutup seçili hazır deneylerle devam edeceğimi anladım.</span>
              </label>
            )}
          </div>
        )}

        {analyses.length === 0 ? (
          <div className="ns-empty-state">
            <div>
              <FileText size={28} />
              <h2>Henüz senteze hazır deney yok</h2>
              <p>Metin Laboratuvarı’nda en az iki farklı metni yeni sabit kimlik alanıyla yeniden işle. Kaynak metinlerin korunmaya devam eder.</p>
              <a className="ns-back-link" href="/metin-laboratuvari/"><FlaskConical size={13} /><span>Metin Laboratuvarı’na git</span></a>
            </div>
          </div>
        ) : (
          <>
            <div className="ns-source-toolbar">
              <label>
                <Search size={14} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Kaynaklarda ara…" />
              </label>
              <button className="ns-quiet-button" onClick={() => onSelectAll(compatible)} disabled={runState !== 'idle'}>
                <Check size={13} /> Tüm uyumluları seç
              </button>
              <button className="ns-icon-button" onClick={onRefresh} disabled={runState !== 'idle'} aria-label="Arşivi yenile" title="Arşivi yenile">
                <RefreshCw size={14} />
              </button>
            </div>

            {spaces.length > 1 && (
              <label className="ns-field">
                <span>TEMSİL PARMAK İZİ GRUBU</span>
                <select value={activeSpaceId} disabled={runState !== 'idle'} onChange={(event) => onActiveSpace(event.target.value)}>
                  {spaces.map(([spaceId, group], index) => (
                    <option key={spaceId} value={spaceId}>Grup {index + 1} · {group.length} deney · {shortFingerprint(spaceId)}</option>
                  ))}
                </select>
              </label>
            )}

            <div className="ns-selection-summary" aria-live="polite">
              <div><span>SEÇİLİ KAYNAK</span><strong>{selectedSources}</strong></div>
              <div><span>UYGUN NÖRON</span><strong>{formatInteger(selectedNeurons)}</strong></div>
              <div><span>PARMAK İZİ GRUBU</span><strong>{spaces.findIndex(([id]) => id === activeSpaceId) + 1}</strong></div>
            </div>

            <div className="ns-source-list">
              {visibleProjects.map((project) => {
                const analysis = project.analysis
                const incompatible = Boolean(analysis && analysis.representationSpaceId !== activeSpaceId)
                const selectedCard = Boolean(analysis && selectedRunIds.has(analysis.runId))
                return (
                  <label
                    key={project.source.id}
                    className={`ns-source-card ${selectedCard ? 'is-selected' : ''} ${incompatible ? 'is-incompatible' : ''}`}
                    title={incompatible ? 'Bu kayıt seçili temsil parmak iziyle uyuşmuyor.' : undefined}
                  >
                    <input
                      type="checkbox"
                      checked={selectedCard}
                      disabled={!analysis || incompatible || runState !== 'idle'}
                      onChange={() => analysis && onToggleRun(analysis)}
                    />
                    <span className="ns-source-card__body">
                      <b>{project.source.title || 'Adsız kaynak'}</b>
                      <span className="ns-source-card__meta">
                        <span>{formatInteger(project.source.rawText.length)} karakter</span>
                        {analysis && <code title={analysis.representationSpaceId}>{shortFingerprint(analysis.representationSpaceId)}</code>}
                      </span>
                    </span>
                    <span className="ns-source-card__state">
                      {analysis ? (
                        <>
                          <strong>{formatInteger(analysis.stats.neuronCount)} HN</strong>
                          <span className={`ns-chip ${incompatible ? 'ns-chip--warn' : 'ns-chip--good'}`}>
                            {incompatible ? 'Farklı uzay' : 'Hazır'}
                          </span>
                        </>
                      ) : (
                        <span className="ns-chip ns-chip--warn">Yeniden işle</span>
                      )}
                    </span>
                  </label>
                )
              })}
            </div>
          </>
        )}
      </section>

      <aside className="ns-control-column">
        <section className={`ns-panel ns-workload-panel ${largeRun ? 'ns-workload-panel--warn' : ''}`}>
          <div className="ns-panel__heading">
            <div><span className="ns-eyebrow">İŞ YÜKÜ</span><h2>{comparisons > 5_000_000 ? 'Büyük çapraz deney' : 'Seyrek çapraz deney'}</h2></div>
            <Activity size={18} />
          </div>
          <div className="ns-workload-grid">
            <div><span>UYGUN ÜYE</span><strong>{formatInteger(selectedNeurons)}</strong></div>
            <div><span>ÇAPRAZ KAYNAK ÇİFTİ</span><strong>{formatInteger(comparisons)}</strong></div>
            <div><span>KAYNAK ÇEŞİTLİLİĞİ</span><strong>{selectedSources}</strong></div>
            <div><span>SAKLANAN KENAR SINIRI</span><strong>{formatInteger(config.maximumRetainedEdges)}</strong></div>
          </div>
          <p className="ns-copy">Aynı kaynağın kendi içindeki çiftleri doğum kanıtı sayılmaz. Yoğun N×N matris bellekte tutulmaz.</p>
          {largeRun && (
            <label className="ns-large-confirm">
              <input type="checkbox" checked={acceptedLargeRun} disabled={runState !== 'idle'} onChange={(event) => setAcceptedLargeRun(event.target.checked)} />
              <span>{formatInteger(comparisons)} çapraz karşılaştırmanın cihazı uzun süre meşgul edebileceğini anladım.</span>
            </label>
          )}
        </section>

        <section className="ns-panel ns-settings-panel">
          <div className="ns-panel__heading">
            <div><span className="ns-eyebrow">EŞİKLER</span><h2>Doğum filtresi</h2></div>
            <SlidersHorizontal size={17} />
          </div>
          <div className="ns-form-grid">
            <label className="ns-field">
              <span>ÜYE SEÇİMİ</span>
              <select
                value={config.nodeSelection}
                disabled={runState !== 'idle'}
                onChange={(event) => onConfig(createSynthesisConfig({ ...config, nodeSelection: event.target.value as SynthesisConfig['nodeSelection'] }))}
              >
                <option value="leaves-or-short-spans">Yaprak + kısa parça</option>
                <option value="leaves">Yalnız yaprak</option>
                <option value="short-spans">Yalnız kısa parça</option>
                <option value="all">Tüm ham nöronlar</option>
              </select>
            </label>
            <label className="ns-field">
              <span>EN UZUN PARÇA</span>
              <input
                type="number"
                min="2"
                max="256"
                value={config.maximumSpanGraphemes}
                disabled={runState !== 'idle'}
                onChange={(event) => onConfig({ ...config, maximumSpanGraphemes: Math.max(2, Number(event.target.value)) })}
              />
            </label>
            <SettingsField label="Kimlik eşiği Kᵢ" value={config.minimumIdentitySimilarity} minimum={0} maximum={1} step={0.01} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumIdentitySimilarity: value, identityOverrideSimilarity: Math.max(value, config.identityOverrideSimilarity) })} />
            <SettingsField label="Birleşik eşik K" value={config.minimumCombinedSimilarity} minimum={-1} maximum={1} step={0.01} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumCombinedSimilarity: value })} />
            <SettingsField label="Bağlam eşiği Kᵣ" value={config.minimumRelationSimilarity} minimum={-1} maximum={1} step={0.01} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumRelationSimilarity: value })} />
            <SettingsField label="Kimlik ağırlığı" value={config.identityWeight} minimum={0.05} maximum={1} step={0.05} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, identityWeight: value })} />
            <label className="ns-field">
              <span>ASGARİ DESTEK</span>
              <input
                type="number"
                min={config.minimumSources}
                max="64"
                value={config.minimumSupport}
                disabled={runState !== 'idle'}
                onChange={(event) => onConfig({ ...config, minimumSupport: Math.max(config.minimumSources, Number(event.target.value)) })}
              />
            </label>
            <label className="ns-field">
              <span>ASGARİ KAYNAK</span>
              <input
                type="number"
                min="2"
                max="32"
                value={config.minimumSources}
                disabled={runState !== 'idle'}
                onChange={(event) => {
                  const value = Math.max(2, Number(event.target.value))
                  onConfig({ ...config, minimumSources: value, minimumSupport: Math.max(value, config.minimumSupport) })
                }}
              />
            </label>
          </div>
          <div className="ns-setting-note"><Info size={13} /><span>Kᵣ = −1 bağlam alt eşiğini kapatır. Bağlam yine birleşik puanı etkiler; güveni düşükse etkisi kendiliğinden azalır.</span></div>
        </section>

        {error && <div className="ns-error" role="alert"><X size={15} /><span>{error}</span></div>}

        {runState !== 'idle' && progress ? (
          <section className="ns-panel ns-progress-panel" aria-live="polite">
            <div className="ns-progress-heading"><span>{progress.message}</span><b>%{Math.round(progress.ratio * 100)}</b></div>
            <div
              className="ns-progress-track"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress.ratio * 100)}
            ><i style={{ width: `${progress.ratio * 100}%` }} /></div>
            <div className="ns-progress-meta"><span>{progress.stage.toLocaleUpperCase('tr-TR')}</span><span>{formatInteger(progress.completed)} / {formatInteger(progress.total)}</span></div>
            <Button tone="danger" wide icon={<CircleStop size={15} />} onClick={onCancel} disabled={runState === 'cancelling'}>
              {runState === 'cancelling' ? 'Güvenle durduruluyor…' : 'Sentezi durdur'}
            </Button>
          </section>
        ) : (
          <Button tone="primary" wide icon={<Play size={16} />} onClick={onRun} disabled={!canRun}>
            Sentez adaylarını üret
          </Button>
        )}
      </aside>
    </main>
  )
}

function ResultsView({
  run,
  projects,
  archive,
  selectedCandidateId,
  filter,
  onFilter,
  onSelect,
  onDecision,
  onBack,
  decisionState,
}: {
  run: SynthesisRun | null
  projects: PersistedProject[]
  archive: SynthesisArchiveRecord[]
  selectedCandidateId: string | null
  filter: CandidateFilter
  onFilter: (filter: CandidateFilter) => void
  onSelect: (id: string) => void
  onDecision: (decision: CandidateDecision) => void
  onBack: () => void
  decisionState: DecisionState
}) {
  if (!run) {
    return (
      <main className="ns-main">
        <section className="ns-panel ns-empty-state">
          <div><Network size={30} /><h2>Henüz aday üretilmedi</h2><p>Korpus aşamasında en az iki uyumlu kaynak seçerek çapraz benzerlik deneyini başlat.</p><Button icon={<ArrowLeft size={14} />} onClick={onBack}>Korpusa dön</Button></div>
        </section>
      </main>
    )
  }

  const filtered = run.candidates.filter((candidate) => {
    const decision = decisionFor(archive, candidate)
    return filter === 'all' || (filter === 'pending' && !decision) || decision === filter
  })
  const selected = filtered.find((candidate) => candidate.id === selectedCandidateId) ?? filtered[0] ?? null
  const layout = projects.find((project) => project.analysis?.runId === selected?.medoid.analysisRunId)?.analysis?.layout
  const relevantEdges = selected ? (() => {
    const members = new Set(selected.members.map((member) => member.key))
    return run.edges.filter((edge) => members.has(edge.left.key) && members.has(edge.right.key))
  })() : []
  const combinedCohesion = relevantEdges.length
    ? relevantEdges.reduce((sum, edge) => sum + edge.combinedSimilarity, 0) / relevantEdges.length
    : selected ? configWeightedCohesion(selected, run.config.identityWeight) : 0
  const selectedDecision = selected ? decisionFor(archive, selected) : null

  return (
    <main className="ns-main ns-results-layout">
      <aside className="ns-panel ns-candidate-sidebar">
        <div className="ns-candidate-sidebar__head">
          <div className="ns-panel__heading"><div><span className="ns-eyebrow">SONUÇLAR 02</span><h2>{run.candidates.length} doğum adayı</h2></div><span className="ns-chip ns-chip--indigo">{formatInteger(run.stats.retainedEdgeCount)} bağ</span></div>
          <div className="ns-candidate-sidebar__filters" role="group" aria-label="Aday filtresi">
            {([
              ['all', 'Tümü'],
              ['pending', 'Bekleyen'],
              ['approved', 'Onaylı'],
              ['rejected', 'Reddedilen'],
            ] as Array<[CandidateFilter, string]>).map(([id, label]) => (
              <button key={id} aria-pressed={filter === id} className={`ns-filter-chip ${filter === id ? 'is-active' : ''}`} onClick={() => onFilter(id)}>{label}</button>
            ))}
          </div>
        </div>
        <div className="ns-candidate-list">
          {filtered.map((candidate) => {
            const decision = decisionFor(archive, candidate)
            return (
              <button key={candidate.id} aria-current={selected?.id === candidate.id ? 'true' : undefined} className={`ns-candidate-row ${selected?.id === candidate.id ? 'is-active' : ''}`} onClick={() => onSelect(candidate.id)}>
                <span className="ns-candidate-row__rank">{String(run.candidates.indexOf(candidate) + 1).padStart(2, '0')}</span>
                <span className="ns-candidate-row__body"><b>{candidate.medoid.text.replace(/\s+/g, ' ') || 'Boşluk örüntüsü'}</b><span>{candidate.sourceCount} kaynak · {candidate.supportCount} bağımsız destek</span></span>
                <span className="ns-candidate-row__score"><strong>{formatDecimal(candidate.birthScore, 2)}</strong><small>{decision === 'approved' ? 'ONAYLI' : decision === 'rejected' ? 'RED' : 'DOĞUM'}</small></span>
              </button>
            )
          })}
          {filtered.length === 0 && <div className="ns-empty-state"><div><Search size={24} /><h3>Bu filtrede aday yok</h3><p>Başka bir karar filtresi seçebilirsin.</p></div></div>}
        </div>
      </aside>

      {selected && layout ? (
        <section className="ns-candidate-detail">
          <article className="ns-panel ns-detail-hero">
            <div className="ns-detail-hero__top">
              <div className="ns-detail-title">
                <span className="ns-eyebrow">{selected.id}</span>
                <h1>Sentez nöronu adayı</h1>
                <p>Parmak izi <code title={selected.fingerprint}>{shortFingerprint(selected.fingerprint)}</code> · kimlik doğum anında dondurulur</p>
              </div>
              <div className="ns-score-orbit" aria-label={`Doğum puanı ${formatDecimal(selected.birthScore, 2)}`}><span><b>{formatDecimal(selected.birthScore, 2)}</b><small>DOĞUM</small></span></div>
            </div>
            <div className="ns-medoid-block">
              <span>SABİT KİMLİĞİ TAŞIYAN MEDOID METİN</span>
              <blockquote>“{selected.medoid.text.replace(/\n/g, ' ↵ ') || 'boşluk'}”</blockquote>
              <small>{sourceName(projects, selected.medoid.sourceId)} · paragraf {selected.medoid.paragraphIndex + 1} · {selected.medoid.graphemeCount} grapheme</small>
            </div>
            <div className="ns-candidate-metrics">
              <div className="ns-candidate-metric"><span>KİMLİK Kᵢ</span><strong>{formatPercent(selected.identityCohesion)}</strong><small>kimlik kohezyonu</small></div>
              <div className="ns-candidate-metric"><span>BAĞLAM Kᵣ</span><strong>{formatPercent(selected.relationCohesion)}</strong><small>bağlam kohezyonu</small></div>
              <div className="ns-candidate-metric"><span>BİRLEŞİK K</span><strong>{formatPercent(combinedCohesion)}</strong><small>{relevantEdges.length} kanıt bağı</small></div>
              <div className="ns-candidate-metric"><span>KAYNAK</span><strong>{selected.sourceCount}</strong><small>bağımsız metin</small></div>
              <div className="ns-candidate-metric"><span>DESTEK</span><strong>{selected.supportCount}</strong><small>{selected.members.length} küme üyesi</small></div>
            </div>
          </article>

          <article className="ns-panel ns-map-panel">
            <div className="ns-panel__heading"><div><span className="ns-eyebrow">MATRİS AYRIMI</span><h2>Kimlik sabit, bağlam plastiktir</h2></div><Layers3 size={17} /></div>
            <div className="ns-map-grid">
              <div className="ns-map-card">
                <div className="ns-map-card__head"><b>Ⅰ · Sabit kimlik alanı</b><span>{selected.identity.length} hücre</span></div>
                <SynthesisHeatmap identity={selected.identity} relation={selected.relation} layout={layout} focus="identity" label={`${selected.id} sabit kimlik ısı haritası`} />
                <p className="ns-map-card__note"><LockKeyhole size={12} /> Medoid ham nöronundan bayt düzeyinde kopyalanır. Onaydan sonra öğrenme bu alanı değiştiremez.</p>
              </div>
              <div className="ns-map-card">
                <div className="ns-map-card__head"><b>R · Başlangıç bağlam alanı</b><span>{selected.relation.length} hücre</span></div>
                <SynthesisHeatmap identity={selected.identity} relation={selected.relation} layout={layout} focus="relation" label={`${selected.id} plastik bağlam ısı haritası`} />
                <p className="ns-map-card__note"><Activity size={12} /> Bağımsız desteklerin ortalamasıdır. İlerideki sinyal geçişleri yalnız bu alan üzerinde mikro iz bırakabilir.</p>
              </div>
            </div>
          </article>

          <article className="ns-panel ns-member-panel">
            <div className="ns-panel__heading"><div><span className="ns-eyebrow">KANIT ZİNCİRİ</span><h2>Yeni nöronu doğuran oluşlar</h2></div><span className="ns-chip ns-chip--good"><ShieldCheck size={11} /> Geri izlenebilir</span></div>
            <table className="ns-member-table">
              <thead><tr><th className="ns-member-table__text">METİN</th><th>KAYNAK</th><th>HAM NÖRON</th><th>KONUM</th></tr></thead>
              <tbody>
                {selected.independentSupports.slice(0, 20).map((member) => (
                  <tr key={member.key}>
                    <td title={member.text}>{member.text.replace(/\s+/g, ' ') || 'Boşluk örüntüsü'}</td>
                    <td title={sourceName(projects, member.sourceId)}>{sourceName(projects, member.sourceId)}</td>
                    <td><code>{member.neuronId}</code></td>
                    <td>P{member.paragraphIndex + 1} · {member.start}–{member.end}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>

          <div className="ns-panel ns-decision-bar">
            <div>{selectedDecision === 'approved' ? <CheckCircle2 size={19} /> : selectedDecision === 'rejected' ? <Ban size={19} /> : <FlaskConical size={19} />}<div><b>{selectedDecision === 'approved' ? 'Bu aday sentez arşivinde onaylı' : selectedDecision === 'rejected' ? 'Bu aday reddedildi' : 'İnsan kararı bekleniyor'}</b><span>Karar yalnız ayrı sentez arşivine yazılır; ham nöronlara dokunulmaz.</span></div></div>
            <div className="ns-decision-bar__actions">
              <Button disabled={decisionState === 'saving'} tone="danger" icon={<Ban size={14} />} onClick={() => onDecision('rejected')}>Adayı reddet</Button>
              <Button disabled={decisionState === 'saving'} tone="success" icon={<CheckCircle2 size={14} />} onClick={() => onDecision('approved')}>{decisionState === 'saving' ? 'Arşive yazılıyor…' : 'Yeni nöronu onayla'}</Button>
            </div>
          </div>
        </section>
      ) : (
        <section className="ns-panel ns-empty-state"><div><Network size={28} /><h2>İncelenecek aday yok</h2><p>Bu deney seçili eşiklerle yeni bir doğum adayı üretmedi.</p><Button icon={<ArrowLeft size={14} />} onClick={onBack}>Eşikleri düzenle</Button></div></section>
      )}
    </main>
  )
}

function configWeightedCohesion(candidate: SynthesisCandidate, identityWeight: number): number {
  return candidate.identityCohesion * identityWeight + candidate.relationCohesion * (1 - identityWeight)
}

function ArchiveView({ records }: { records: SynthesisArchiveRecord[] }) {
  const approved = records.filter((record) => record.decision === 'approved')
  const rejected = records.filter((record) => record.decision === 'rejected')
  const sources = new Set(records.flatMap((record) => record.sourceDigests)).size
  return (
    <main className="ns-main ns-archive-layout">
      <section className="ns-panel ns-archive-panel">
        <div className="ns-archive-panel__head"><div className="ns-panel__heading"><div><span className="ns-eyebrow">ARŞİV 03</span><h1>Sentez kararları</h1></div><span className="ns-chip ns-chip--good"><Database size={11} /> Ayrı yerel veritabanı</span></div><p className="ns-copy">Onaylanan kimlik vektörü ve başlangıç bağlamı, doğum kanıtlarıyla birlikte saklanır. Reddedilen adaylar da aynı deneyde yeniden önerilmemesi için iz bırakır.</p></div>
        {records.length ? (
          <div className="ns-archive-list">
            {records.map((record) => (
              <article key={record.id} className={`ns-archive-record ${record.decision === 'rejected' ? 'is-rejected' : ''}`}>
                <span className="ns-archive-record__icon">{record.decision === 'approved' ? <CheckCircle2 size={18} /> : <Ban size={18} />}</span>
                <span className="ns-archive-record__body"><b>{record.candidate.medoid.text.replace(/\s+/g, ' ') || 'Boşluk örüntüsü'}</b><span>{record.candidate.id} · {record.candidate.sourceCount} kaynak · {record.candidate.supportCount} destek</span></span>
                <span className="ns-archive-record__meta"><strong>{record.decision === 'approved' ? 'ONAYLI' : 'RED'}</strong><time dateTime={record.reviewedAt}>{new Date(record.reviewedAt).toLocaleString('tr-TR')}</time></span>
              </article>
            ))}
          </div>
        ) : (
          <div className="ns-empty-state"><div><Archive size={28} /><h2>Sentez arşivi boş</h2><p>Adaylar ekranında verdiğin onay ve ret kararları burada, ham metin arşivinden bağımsız biçimde saklanacak.</p></div></div>
        )}
      </section>
      <aside className="ns-archive-aside">
        <section className="ns-panel"><span className="ns-eyebrow">ARŞİV ÖZETİ</span><div className="ns-archive-stat"><span>Onaylanmış yeni nöron</span><strong>{approved.length}</strong></div><div className="ns-archive-stat"><span>Reddedilmiş aday</span><strong>{rejected.length}</strong></div><div className="ns-archive-stat"><span>Kanıt sağlayan kaynak</span><strong>{sources}</strong></div></section>
        <section className="ns-panel"><span className="ns-eyebrow">DEĞİŞMEZLİK SÖZÜ</span><p className="ns-copy"><b>I alanı</b> doğum anında dondurulur. <b>R alanı</b> sonraki programlarda yeni bağlamlarla değişebilir. Bu arşiv ham Metin Laboratuvarı verisini hiçbir zaman güncellemez.</p></section>
      </aside>
    </main>
  )
}

export function SynthesisApp() {
  const [view, setView] = useState<LabView>('corpus')
  const [projects, setProjects] = useState<PersistedProject[]>([])
  const [archive, setArchive] = useState<SynthesisArchiveRecord[]>([])
  const [activeSpaceId, setActiveSpaceId] = useState('')
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set())
  const [config, setConfig] = useState<SynthesisConfig>(() => createSynthesisConfig())
  const [run, setRun] = useState<SynthesisRun | null>(null)
  const [progress, setProgress] = useState<SynthesisProgress | null>(null)
  const [runState, setRunState] = useState<RunState>('idle')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null)
  const [candidateFilter, setCandidateFilter] = useState<CandidateFilter>('all')
  const [decisionState, setDecisionState] = useState<DecisionState>('idle')
  const workerRef = useRef<Worker | null>(null)
  const cancelWorkerRef = useRef<(() => void) | null>(null)

  const refresh = async () => {
    setError(null)
    try {
      const [storedProjects, loadedArchive] = await Promise.all([
        loadAllProjects(),
        loadSynthesisArchive(),
      ])
      const loadedProjects: PersistedProject[] = []
      let rejectedProjectCount = 0
      for (const project of storedProjects) {
        try {
          loadedProjects.push(validateProject(project))
        } catch {
          rejectedProjectCount += 1
        }
      }
      setProjects(loadedProjects)
      setArchive(loadedArchive)
      if (rejectedProjectCount > 0) {
        setError(`${rejectedProjectCount} bozuk veya kaynakla uyuşmayan deney sentez korpusundan çıkarıldı.`)
      }
      const analyses = loadedProjects.flatMap((project) => project.analysis ? [project.analysis] : [])
      const groups = new Map<string, TextLabAnalysis[]>()
      for (const analysis of analyses) {
        const group = groups.get(analysis.representationSpaceId) ?? []
        group.push(analysis)
        groups.set(analysis.representationSpaceId, group)
      }
      const bestGroup = [...groups.entries()].sort((left, right) => right[1].length - left[1].length)[0]
      const nextSpaceId = groups.has(activeSpaceId) ? activeSpaceId : bestGroup?.[0] ?? ''
      const compatible = groups.get(nextSpaceId) ?? []
      setActiveSpaceId(nextSpaceId)
      setSelectedRunIds((current) => {
        const valid = new Set(compatible.map((analysis) => analysis.runId))
        const retained = new Set([...current].filter((id) => valid.has(id)))
        return retained.size ? retained : valid
      })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Yerel arşiv okunamadı.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])
  useEffect(() => () => {
    cancelWorkerRef.current?.()
    workerRef.current?.terminate()
  }, [])

  const analyses = projects.flatMap((project) => project.analysis ? [project.analysis] : [])
  const selectedAnalyses = analyses.filter(
    (analysis) => analysis.representationSpaceId === activeSpaceId && selectedRunIds.has(analysis.runId),
  )

  const handleSpace = (spaceId: string) => {
    const compatible = analyses.filter((analysis) => analysis.representationSpaceId === spaceId)
    setActiveSpaceId(spaceId)
    setSelectedRunIds(new Set(compatible.map((analysis) => analysis.runId)))
  }

  const handleRun = async () => {
    setError(null)
    setRunState('running')
    setProgress({
      stage: 'validating',
      completed: 0,
      total: 1,
      ratio: 0,
      message: 'Korpus doğrulanıyor…',
      eligibleMemberCount: 0,
      retainedEdgeCount: 0,
      candidateCount: 0,
    })
    try {
      const validated = createSynthesisConfig(config)
      const nextRun = await new Promise<SynthesisRun>((resolve, reject) => {
        const worker = new Worker(
          new URL('../synthesis/synthesis.worker.ts', import.meta.url),
          { type: 'module', name: 'noron-sentezi' },
        )
        workerRef.current = worker
        let settled = false
        const finish = (callback: () => void) => {
          if (settled) return
          settled = true
          worker.terminate()
          if (workerRef.current === worker) workerRef.current = null
          cancelWorkerRef.current = null
          callback()
        }
        cancelWorkerRef.current = () => finish(() => reject(new SynthesisCancelledError()))
        worker.onmessage = (event: MessageEvent<SynthesisWorkerResponse>) => {
          const response = event.data
          if (response.type === 'progress') {
            setProgress(response.progress)
            return
          }
          if (response.type === 'result') {
            finish(() => resolve(response.run))
            return
          }
          const error = new Error(response.message)
          error.name = response.name
          finish(() => reject(error))
        }
        worker.onerror = (event) => {
          finish(() => reject(new Error(event.message || 'Sentez Worker işlemi durdu.')))
        }
        worker.postMessage({
          type: 'run',
          analyses: selectedAnalyses.map(createWorkerAnalysisInput),
          config: validated,
        } satisfies SynthesisWorkerRequest)
      })
      setRun(nextRun)
      setSelectedCandidateId(nextRun.candidates[0]?.id ?? null)
      setCandidateFilter('all')
      setView('candidates')
    } catch (cause) {
      if (!(cause instanceof SynthesisCancelledError)) {
        setError(cause instanceof Error ? cause.message : 'Sentez deneyi tamamlanamadı.')
      }
    } finally {
      setRunState('idle')
      setProgress(null)
    }
  }

  const handleCancel = () => {
    if (runState === 'idle') return
    setRunState('cancelling')
    cancelWorkerRef.current?.()
  }

  const handleDecision = async (decision: CandidateDecision) => {
    if (!run || !selectedCandidateId || decisionState === 'saving') return
    const candidate = run.candidates.find((item) => item.id === selectedCandidateId)
    if (!candidate) return
    const previousDecision = decisionFor(archive, candidate)
    if (previousDecision === decision) return
    const action = decision === 'approved' ? 'yeni nöron olarak onaylamak' : 'reddetmek'
    const changeNotice = previousDecision
      ? ` Önceki “${previousDecision === 'approved' ? 'onay' : 'ret'}” kararı değiştirilecek.`
      : ''
    if (!window.confirm(
      `“${candidate.medoid.text.replace(/\s+/g, ' ') || 'Boşluk örüntüsü'}” adayını ${action} istiyor musun? ${candidate.sourceCount} kaynak ve ${candidate.supportCount} bağımsız destek arşivlenecek.${changeNotice}`,
    )) return
    setError(null)
    setDecisionState('saving')
    try {
      const record = await saveCandidateDecision(run, candidate, decision)
      const nextArchive = [record, ...archive.filter((item) => item.id !== record.id)]
      setArchive(nextArchive)
      if (candidateFilter === 'pending') {
        const nextCandidate = run.candidates.find(
          (item) => item.id !== candidate.id && !decisionFor(nextArchive, item),
        )
        setSelectedCandidateId(nextCandidate?.id ?? null)
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Aday kararı arşivlenemedi.')
    } finally {
      setDecisionState('idle')
    }
  }

  const approvedCount = archive.filter((record) => record.decision === 'approved').length
  const analysisCount = analyses.length
  const pendingAnalysisCount = projects.filter(
    (project) => project.source.rawText.trim().length > 0 && !project.analysis,
  ).length
  const statusLabel = runState !== 'idle'
    ? 'Sentez hesaplanıyor'
    : pendingAnalysisCount > 0
      ? `${pendingAnalysisCount} metin hazırlanıyor`
      : analysisCount >= 2
      ? 'Senteze hazır'
      : 'Kaynak bekleniyor'

  if (loading) {
    return <div className="ns-app"><div className="ns-loading"><div><span className="ns-spinner" /><span>Yerel nöron arşivleri hazırlanıyor…</span></div></div></div>
  }

  return (
    <div className="ns-app">
      <header className="ns-topbar">
        <div className="ns-brand"><span className="ns-brand__mark"><Network size={21} /></span><div><strong>NÖRON SENTEZİ</strong><small>YÜZEY KALIBI DOĞUM LABORATUVARI · v0.1</small></div></div>
        <span className={`ns-status-pill ${runState !== 'idle' ? 'ns-status-pill--running' : pendingAnalysisCount === 0 && analysisCount >= 2 ? 'ns-status-pill--ready' : ''}`}><i />{statusLabel}</span>
        <div className="ns-topbar__actions"><a href="/metin-laboratuvari/" className="ns-back-link"><ArrowLeft size={13} /><span>Metin Laboratuvarı</span></a><a href="/noron-aileleri/" className="ns-back-link"><UsersRound size={13} /><span>Nöron Aileleri</span></a></div>
      </header>

      <nav className="ns-stage-rail" aria-label="Nöron sentezi aşamaları">
        {STAGES.map((stage, index) => (
          <button key={stage.id} aria-current={view === stage.id ? 'step' : undefined} className={view === stage.id ? 'is-active' : ''} onClick={() => setView(stage.id)} disabled={runState !== 'idle'}>
            <span className="ns-stage-rail__index">{index + 1}</span>{stage.icon}<span><b>{stage.label}</b><small>{stage.short}</small></span>
          </button>
        ))}
      </nav>

      <div className="ns-context-strip">
        <span><b>HAM DENEY</b>{analysisCount}</span>
        <span><b>AKTİF UZAY</b>{activeSpaceId ? shortFingerprint(activeSpaceId) : '—'}</span>
        <span><b>SON KOŞU</b>{run?.id ?? '—'}</span>
        <span><b>ADAY</b>{run?.candidates.length ?? 0}</span>
        <span className="ns-context-strip__last"><b>ONAYLI SENTEZ</b>{approvedCount}</span>
      </div>

      {error && view !== 'corpus' && <div className="ns-global-error ns-error" role="alert"><X size={15} /><span>{error}</span></div>}

      {view === 'corpus' && (
        <CorpusView
          projects={projects}
          activeSpaceId={activeSpaceId}
          selectedRunIds={selectedRunIds}
          config={config}
          runState={runState}
          progress={progress}
          error={error}
          onActiveSpace={handleSpace}
          onToggleRun={(analysis) => setSelectedRunIds((current) => {
            const next = new Set(current)
            if (next.has(analysis.runId)) next.delete(analysis.runId)
            else next.add(analysis.runId)
            return next
          })}
          onSelectAll={(compatible) => setSelectedRunIds(new Set(compatible.map((analysis) => analysis.runId)))}
          onConfig={(next) => setConfig({ ...DEFAULT_SYNTHESIS_CONFIG, ...next })}
          onRefresh={() => void refresh()}
          onRun={() => void handleRun()}
          onCancel={handleCancel}
        />
      )}
      {view === 'candidates' && (
        <ResultsView
          run={run}
          projects={projects}
          archive={archive}
          selectedCandidateId={selectedCandidateId}
          filter={candidateFilter}
          onFilter={(next) => {
            setCandidateFilter(next)
            const candidate = run?.candidates.find((item) => {
              const decision = decisionFor(archive, item)
              return next === 'all' || (next === 'pending' && !decision) || decision === next
            })
            setSelectedCandidateId(candidate?.id ?? null)
          }}
          onSelect={setSelectedCandidateId}
          onDecision={(decision) => void handleDecision(decision)}
          onBack={() => setView('corpus')}
          decisionState={decisionState}
        />
      )}
      {view === 'archive' && <ArchiveView records={archive} />}
    </div>
  )
}
