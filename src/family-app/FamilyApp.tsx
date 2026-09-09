import {
  Activity,
  AlertTriangle,
  Archive,
  BarChart3,
  BrainCircuit,
  Check,
  CheckCircle2,
  CircleStop,
  Database,
  Download,
  GitBranch,
  Info,
  Layers3,
  Network,
  Pause,
  Play,
  RefreshCw,
  Route,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Split,
  TestTube2,
  UsersRound,
  Waypoints,
  X,
  Zap,
} from 'lucide-react'
import {
  type ChangeEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { SuiteMenu } from '../components/SuiteMenu'
import {
  DEFAULT_FAMILY_CONFIG,
} from '../families'
import {
  createFamilyInputManifest,
} from '../families'
import {
  FUNCTIONAL_REGIONS,
  type BehaviorDiagnostics,
  type BehaviorProfile,
  type EventCoalition,
  type EventRoleBinding,
  type FamilyConfig,
  type FamilyLifecycle,
  type FamilyProgress,
  type FamilyRun,
  type FamilyUtilityReport,
  type FunctionalRegionId,
  type LearnedFamily,
  type NeuronActivation,
} from '../families/types'
import type { FamilyWorkerRequest, FamilyWorkerResponse } from '../families/workerProtocol'
import type { SynthesisArchiveRecord } from '../synthesis-app/persistence'
import type { PersistedProject, TextLabAnalysis } from '../text-lab/engine'
import {
  createFamilyRunInput,
  familyRunTransferables,
  loadFamilySourceBundle,
} from './dataAdapter'
import {
  loadFamilyProjects,
  saveCoalitionLog,
  saveFamilyRun,
  type FamilyProject,
} from './persistence'
import { FamilyGraph, type FamilyGraphEdge, type FamilyGraphNode } from './components/FamilyGraph'
import { FamilyMatrixCanvas } from './components/FamilyMatrixCanvas'
import { ProfileBars, type ProfileBarItem } from './components/ProfileBars'

type LabView = 'inputs' | 'profiles' | 'families' | 'flow' | 'validation'
type RunState = 'idle' | 'running' | 'cancelling'
type FamilyViewMode = 'matrix' | 'graph'
type ProfileFilter = 'all' | FunctionalRegionId
type FamilyFilter = 'all' | 'provisional' | 'strengthening' | 'stable' | 'attention'

type ControlledConfig = Pick<
  FamilyConfig,
  | 'minimumFormationSimilarity'
  | 'minimumMembershipScore'
  | 'minimumCoreMembership'
  | 'minimumSupport'
  | 'minimumSources'
  | 'maximumFamilies'
  | 'maximumMembershipsPerNeuron'
  | 'runtimeActivationThreshold'
>

const INITIAL_CONFIG: ControlledConfig = {
  minimumFormationSimilarity: DEFAULT_FAMILY_CONFIG.minimumFormationSimilarity,
  minimumMembershipScore: DEFAULT_FAMILY_CONFIG.minimumMembershipScore,
  minimumCoreMembership: DEFAULT_FAMILY_CONFIG.minimumCoreMembership,
  minimumSupport: DEFAULT_FAMILY_CONFIG.minimumSupport,
  minimumSources: DEFAULT_FAMILY_CONFIG.minimumSources,
  maximumFamilies: DEFAULT_FAMILY_CONFIG.maximumFamilies,
  maximumMembershipsPerNeuron: DEFAULT_FAMILY_CONFIG.maximumMembershipsPerNeuron,
  runtimeActivationThreshold: DEFAULT_FAMILY_CONFIG.runtimeActivationThreshold,
}

const MAX_MATRIX_ROWS = 600
const MAX_MATRIX_COLUMNS = 64

const STAGES: Array<{ id: LabView; label: string; short: string; icon: ReactNode }> = [
  { id: 'inputs', label: 'Girdiler', short: 'Nöronları seç', icon: <Database size={15} /> },
  { id: 'profiles', label: 'Profiller', short: 'Davranışı gör', icon: <BarChart3 size={15} /> },
  { id: 'families', label: 'Aileler', short: 'Yumuşak üyelik', icon: <UsersRound size={15} /> },
  { id: 'flow', label: 'Düşünce Akışı', short: 'Koalisyon ve rota', icon: <Waypoints size={15} /> },
  { id: 'validation', label: 'Doğrulama', short: 'Fayda ve yaşam', icon: <ShieldCheck size={15} /> },
]

function Button({
  children,
  icon,
  tone = 'default',
  compact = false,
  wide = false,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode
  tone?: 'default' | 'primary' | 'danger' | 'success' | 'ghost'
  compact?: boolean
  wide?: boolean
}) {
  return (
    <button
      className={`fa-button fa-button--${tone} ${compact ? 'fa-button--compact' : ''} ${wide ? 'fa-button--wide' : ''}`}
      {...props}
    >
      {icon}{children}
    </button>
  )
}

function formatInteger(value: number): string {
  return Number.isFinite(value) ? value.toLocaleString('tr-TR', { maximumFractionDigits: 0 }) : '—'
}

function formatDecimal(value: number | null | undefined, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('tr-TR', { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : '—'
}

function formatPercent(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `%${Math.round(Math.min(1, Math.max(-1, value)) * 100)}`
    : '—'
}

function shortId(value: string): string {
  const visible = value.replaceAll('\u0000', ' · ')
  return visible.length > 22 ? `${visible.slice(0, 19)}…` : visible
}

function previewText(value: string, maximum = 58): string {
  const visible = value.replace(/\s+/g, ' ').trim() || 'Boşluk örüntüsü'
  return visible.length > maximum ? `${visible.slice(0, maximum - 1)}…` : visible
}

function regionDefinition(region: FunctionalRegionId) {
  return FUNCTIONAL_REGIONS.find((definition) => definition.id === region) ?? FUNCTIONAL_REGIONS[0]
}

function lifecycleLabel(lifecycle: FamilyLifecycle): string {
  switch (lifecycle) {
    case 'provisional': return 'Geçici'
    case 'strengthening': return 'Güçleniyor'
    case 'stable': return 'Kararlı'
    case 'split-proposed': return 'Bölünme önerisi'
    case 'merge-proposed': return 'Birleşme önerisi'
    case 'retired': return 'Emekli'
  }
}

function lifecycleTone(lifecycle: FamilyLifecycle): string {
  if (lifecycle === 'stable') return 'fa-chip--good'
  if (lifecycle === 'split-proposed' || lifecycle === 'merge-proposed') return 'fa-chip--warn'
  if (lifecycle === 'retired') return 'fa-chip--red'
  return 'fa-chip--indigo'
}

function profileDominantRegion(profile: BehaviorProfile): FunctionalRegionId {
  return [...profile.regionAffinities]
    .sort((left, right) => right.score - left.score || left.region.localeCompare(right.region))[0]?.region ?? 'content'
}

function familyLabel(family: LearnedFamily, profiles: ReadonlyMap<string, BehaviorProfile>): string {
  const representative = family.coreMemberKeys.map((key) => profiles.get(key)).find(Boolean)
    ?? family.memberKeys.map((key) => profiles.get(key)).find(Boolean)
  return representative ? previewText(representative.text, 34) : `${regionDefinition(family.region).label} ${family.id.slice(-6)}`
}

function profileBars(profile: BehaviorProfile, comparison?: BehaviorProfile): ProfileBarItem[] {
  const dimensions: Array<{ key: keyof BehaviorDiagnostics; label: string; group: string }> = [
    { key: 'spanScale', label: 'Parça ölçeği', group: 'Yapısal iz' },
    { key: 'relativePosition', label: 'Göreli konum', group: 'Yapısal iz' },
    { key: 'depthScale', label: 'Bölünme derinliği', group: 'Yapısal iz' },
    { key: 'relationStrength', label: 'İlişki gücü', group: 'Bağlam davranışı' },
    { key: 'localContinuity', label: 'Yerel süreklilik', group: 'Bağlam davranışı' },
    { key: 'bridgeImpact', label: 'Bağlam köprüsü', group: 'Bağlam davranışı' },
    { key: 'leftRightAsymmetry', label: 'Yön asimetrisi', group: 'Bağlam davranışı' },
    { key: 'transitionEntropy', label: 'Geçiş çeşitliliği', group: 'Geçiş davranışı' },
    { key: 'transitionConcentration', label: 'Geçiş yoğunlaşması', group: 'Geçiş davranışı' },
    { key: 'shortScopePotential', label: 'Kısa kapsam etkisi', group: 'Geçiş davranışı' },
  ]
  return dimensions.map(({ key, label, group }) => ({
    id: key,
    label,
    group,
    value: profile.diagnostics[key],
    comparison: comparison?.diagnostics[key],
  }))
}

function selectedAnalysesFor(
  projects: readonly PersistedProject[],
  selectedRunIds: ReadonlySet<string>,
): TextLabAnalysis[] {
  return projects.flatMap((project) => (
    project.analysis && selectedRunIds.has(project.analysis.runId) ? [project.analysis] : []
  ))
}

function downloadRun(run: FamilyRun): void {
  const json = JSON.stringify(run, (_key, value) => (
    value instanceof Float32Array ? Array.from(value) : value
  ), 2)
  const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `noron-aileleri-${new Date().toISOString().slice(0, 10)}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}

function InputView({
  projects,
  decisions,
  activeSpaceId,
  selectedRunIds,
  selectedDecisionIds,
  config,
  runState,
  progress,
  error,
  onSpace,
  onToggleAnalysis,
  onToggleDecision,
  onSelectAll,
  onConfig,
  onRefresh,
  onRun,
  onCancel,
}: {
  projects: PersistedProject[]
  decisions: SynthesisArchiveRecord[]
  activeSpaceId: string
  selectedRunIds: Set<string>
  selectedDecisionIds: Set<string>
  config: ControlledConfig
  runState: RunState
  progress: FamilyProgress | null
  error: string | null
  onSpace: (space: string) => void
  onToggleAnalysis: (runId: string) => void
  onToggleDecision: (id: string) => void
  onSelectAll: () => void
  onConfig: (config: ControlledConfig) => void
  onRefresh: () => void
  onRun: () => void
  onCancel: () => void
}) {
  const [query, setQuery] = useState('')
  const analyses = projects.flatMap((project) => project.analysis ? [project.analysis] : [])
  const spaces = Array.from(new Set([
    ...analyses.map((analysis) => analysis.representationSpaceId),
    ...decisions.map((decision) => decision.representationSpaceId),
  ])).sort()
  const compatibleAnalyses = analyses.filter((analysis) => analysis.representationSpaceId === activeSpaceId)
  const compatibleDecisions = decisions.filter((decision) => decision.representationSpaceId === activeSpaceId)
  const normalizedQuery = query.trim().toLocaleLowerCase('tr-TR')
  const visibleAnalyses = compatibleAnalyses.filter((analysis) => (
    !normalizedQuery || analysis.source.title.toLocaleLowerCase('tr-TR').includes(normalizedQuery)
  ))
  const selectedAnalyses = compatibleAnalyses.filter((analysis) => selectedRunIds.has(analysis.runId))
  const selectedDecisions = compatibleDecisions.filter((decision) => selectedDecisionIds.has(decision.id))
  const distinctSources = new Set(selectedAnalyses.map((analysis) => analysis.sourceDigest)).size
  const rawNeuronCount = selectedAnalyses.reduce((sum, analysis) => sum + analysis.nodes.length, 0)
  const canRun = runState === 'idle' && selectedAnalyses.length > 0

  return (
    <main className="fa-main fa-input-layout" aria-busy={runState !== 'idle'}>
      <section className="fa-panel fa-panel--padded">
        <div className="fa-panel__heading">
          <div><span className="fa-eyebrow">GİRDİLER 01</span><h1>Davranış kanıtını seç</h1></div>
          <span className="fa-chip fa-chip--good"><ShieldCheck size={11} /> Salt okunur</span>
        </div>
        <p className="fa-copy">Metin Laboratuvarı analizleri davranış izini, yalnız güvenlik denetiminden geçmiş onaylı sentez kayıtları ise yeni nöronları sağlar. Yukarı akış arşivleri değiştirilmez.</p>

        <div className="fa-source-toolbar">
          <label className="fa-search"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Kaynaklarda ara" aria-label="Kaynaklarda ara" /></label>
          <Button compact icon={<RefreshCw size={13} />} onClick={onRefresh} disabled={runState !== 'idle'}>Yenile</Button>
          <Button compact tone="ghost" icon={<Check size={13} />} onClick={onSelectAll} disabled={runState !== 'idle' || compatibleAnalyses.length === 0}>Tümünü seç</Button>
        </div>

        <label className="fa-field" style={{ marginBottom: 10 }}>
          <span>TEMSİL UZAYI</span>
          <select value={activeSpaceId} disabled={runState !== 'idle'} onChange={(event) => onSpace(event.target.value)}>
            {spaces.length ? spaces.map((space) => <option key={space} value={space}>{shortId(space)}</option>) : <option value="">Uyumlu uzay yok</option>}
          </select>
        </label>

        <div className="fa-source-summary">
          <div><span>SEÇİLİ ANALİZ</span><strong>{selectedAnalyses.length}</strong><small>{compatibleAnalyses.length} uyumlu</small></div>
          <div><span>BAĞIMSIZ KAYNAK</span><strong>{distinctSources}</strong><small>digest ile sayılır</small></div>
          <div><span>HAM NÖRON</span><strong>{formatInteger(rawNeuronCount)}</strong><small>dense K taşınmaz</small></div>
          <div><span>ONAYLI SENTEZ</span><strong>{selectedDecisions.length}</strong><small>{compatibleDecisions.length} güvenli kayıt</small></div>
        </div>

        <div className="fa-source-list">
          {visibleAnalyses.map((analysis) => {
            const selected = selectedRunIds.has(analysis.runId)
            return (
              <label key={analysis.runId} className={`fa-source-row ${selected ? 'is-selected' : ''}`}>
                <input type="checkbox" checked={selected} disabled={runState !== 'idle'} onChange={() => onToggleAnalysis(analysis.runId)} />
                <span className="fa-source-row__body"><b>{analysis.source.title || 'Adsız kaynak'}</b><span>{shortId(analysis.runId)} · {analysis.nodes.length.toLocaleString('tr-TR')} ham nöron</span></span>
                <span className="fa-source-row__state"><strong>{analysis.stats.paragraphCount} P</strong><small>{new Date(analysis.createdAt).toLocaleDateString('tr-TR')}</small></span>
              </label>
            )
          })}
          {visibleAnalyses.length === 0 && <div className="fa-empty"><div><Database size={25} /><h3>Uyumlu analiz yok</h3><p>Önce Metin Laboratuvarı’nda bu temsil uzayı için ham nöron analizi üret.</p></div></div>}
        </div>

        {compatibleDecisions.length > 0 && (
          <details className="fa-accessible-data" open>
            <summary>Onaylı sentez nöronları · {selectedDecisions.length}/{compatibleDecisions.length}</summary>
            <div className="fa-evidence-list">
              {compatibleDecisions.map((decision) => (
                <label key={decision.id} className="fa-source-row">
                  <input type="checkbox" checked={selectedDecisionIds.has(decision.id)} disabled={runState !== 'idle'} onChange={() => onToggleDecision(decision.id)} />
                  <span className="fa-source-row__body"><b>{previewText(decision.candidate.medoid.text)}</b><span>{decision.candidate.sourceCount} kaynak · {decision.candidate.supportCount} destek</span></span>
                  <span className="fa-source-row__state"><strong>{formatDecimal(decision.candidate.birthScore, 2)}</strong><small>ONAYLI</small></span>
                </label>
              ))}
            </div>
          </details>
        )}
      </section>

      <aside className="fa-control-column">
        <section className="fa-panel fa-panel--padded">
          <div className="fa-panel__heading"><div><span className="fa-eyebrow">OLUŞUM EŞİKLERİ</span><h2>Aile doğumu</h2></div><SlidersHorizontal size={17} /></div>
          <div className="fa-form-grid">
            <RangeField label="Aile oluşum benzerliği" value={config.minimumFormationSimilarity} min={0.2} max={0.98} step={0.01} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumFormationSimilarity: value })} />
            <RangeField label="Asgari üyelik" value={config.minimumMembershipScore} min={0.1} max={0.95} step={0.01} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumMembershipScore: value })} />
            <RangeField label="Çekirdek üyelik" value={config.minimumCoreMembership} min={0.2} max={0.99} step={0.01} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumCoreMembership: value })} />
            <RangeField label="Çalışma anı ateşleme" value={config.runtimeActivationThreshold} min={0.05} max={0.95} step={0.01} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, runtimeActivationThreshold: value })} />
            <NumberField label="Asgari destek" value={config.minimumSupport} min={2} max={1000} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumSupport: Math.max(value, config.minimumSources) })} />
            <NumberField label="Asgari kaynak" value={config.minimumSources} min={2} max={100} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, minimumSources: value, minimumSupport: Math.max(value, config.minimumSupport) })} />
            <NumberField label="En çok aile" value={config.maximumFamilies} min={2} max={512} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, maximumFamilies: value })} />
            <NumberField label="Nöron başına üyelik" value={config.maximumMembershipsPerNeuron} min={1} max={12} disabled={runState !== 'idle'} onChange={(value) => onConfig({ ...config, maximumMembershipsPerNeuron: value })} />
          </div>
          <div className="fa-setting-note"><Info size={13} /><span>Aile oluşum eşiği kalıcı kümeyi, çalışma anı eşiği ise mevcut bir ailenin ateşlenmesini yönetir. Bunlar aynı karar değildir.</span></div>
        </section>

        <section className="fa-panel fa-panel--padded">
          <div className="fa-panel__heading"><div><span className="fa-eyebrow">İŞ YÜKÜ</span><h2>Seyrek deney</h2></div><Activity size={17} /></div>
          <div className="fa-workload">
            <div><span>PROFİL ADAYI</span><strong>{formatInteger(rawNeuronCount + selectedDecisions.length)}</strong></div>
            <div><span>KAYNAK KATLAMASI</span><strong>{distinctSources}</strong></div>
          </div>
          <p className="fa-copy">Motor tam bir N×N aile tablosu saklamaz. Seyrek davranış profilleri, sınırlı aday komşular ve nöron başına ilk üyelikler korunur.</p>
        </section>

        <div className="fa-boundary-note"><AlertTriangle size={15} /><span><b>Semantik sınır:</b> “İçerik”, “olay”, “rol” ve “operatör” burada doğrulanmış dilbilgisi etiketleri değil, davranıştan çıkarılan işlev hipotezleridir. Amaç, dil ve kontrol bölgeleri çalışma zamanı deneyimi olmadan doğrulanmaz.</span></div>

        {error && <div className="fa-error" role="alert"><X size={15} /><span>{error}</span></div>}
        {distinctSources > 0 && distinctSources < config.minimumSources && <div className="fa-warning"><AlertTriangle size={15} /><span>Seçili korpus asgari kaynak eşiğinin altında. Profiller üretilebilir; güvenilir aile oluşmayabilir.</span></div>}

        {runState !== 'idle' && progress ? (
          <section className="fa-panel fa-panel--padded" aria-live="polite">
            <div className="fa-progress-heading"><span>{progress.message}</span><b>%{Math.round(progress.ratio * 100)}</b></div>
            <div className="fa-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress.ratio * 100)}><i style={{ width: `${progress.ratio * 100}%` }} /></div>
            <div className="fa-progress-meta"><span>{progress.stage.toLocaleUpperCase('tr-TR')}</span><span>{formatInteger(progress.completed)} / {formatInteger(progress.total)}</span></div>
            <Button wide tone="danger" icon={<CircleStop size={15} />} onClick={onCancel} disabled={runState === 'cancelling'}>{runState === 'cancelling' ? 'Güvenle durduruluyor…' : 'Aile deneyini durdur'}</Button>
          </section>
        ) : (
          <Button wide tone="primary" icon={<Play size={16} />} onClick={onRun} disabled={!canRun}>Davranış ailelerini üret</Button>
        )}
      </aside>
    </main>
  )
}

function RangeField({ label, value, min, max, step, disabled, onChange }: { label: string; value: number; min: number; max: number; step: number; disabled: boolean; onChange: (value: number) => void }) {
  return <label className="fa-field fa-field--wide"><span className="fa-field__line"><span>{label}</span><b>{value.toFixed(2)}</b></span><input type="range" value={value} min={min} max={max} step={step} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} /></label>
}

function NumberField({ label, value, min, max, disabled, onChange }: { label: string; value: number; min: number; max: number; disabled: boolean; onChange: (value: number) => void }) {
  return <label className="fa-field"><span>{label}</span><input type="number" value={value} min={min} max={max} disabled={disabled} onChange={(event) => onChange(Math.max(min, Math.min(max, Math.trunc(Number(event.target.value)) || min)))} /></label>
}

function ProfilesView({
  run,
  selectedKey,
  comparisonKey,
  filter,
  onFilter,
  onSelect,
  onComparison,
}: {
  run: FamilyRun
  selectedKey: string | null
  comparisonKey: string | null
  filter: ProfileFilter
  onFilter: (filter: ProfileFilter) => void
  onSelect: (key: string) => void
  onComparison: (key: string | null) => void
}) {
  const [query, setQuery] = useState('')
  const normalizedQuery = query.trim().toLocaleLowerCase('tr-TR')
  const filtered = run.profiles.filter((profile) => {
    const region = profileDominantRegion(profile)
    return (filter === 'all' || region === filter) && (!normalizedQuery || profile.text.toLocaleLowerCase('tr-TR').includes(normalizedQuery))
  })
  const selected = filtered.find((profile) => profile.memberKey === selectedKey)
    ?? run.profiles.find((profile) => profile.memberKey === selectedKey)
    ?? filtered[0]
    ?? null
  const comparison = run.profiles.find((profile) => profile.memberKey === comparisonKey) ?? null
  const topEvidence = selected?.evidence.slice(0, 8) ?? []

  return (
    <main className="fa-main fa-profile-layout">
      <aside className="fa-panel fa-sidebar-panel">
        <div className="fa-sidebar-head">
          <div className="fa-panel__heading"><div><span className="fa-eyebrow">PROFİLLER 02</span><h2>{formatInteger(run.profiles.length)} davranış profili</h2></div><span className="fa-chip fa-chip--indigo">Q seyrek</span></div>
          <label className="fa-search" style={{ marginTop: 10 }}><Search size={13} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Nöron izinde ara" /></label>
          <div className="fa-filter-row">
            <button className={filter === 'all' ? 'is-active' : ''} onClick={() => onFilter('all')}>Tümü</button>
            {FUNCTIONAL_REGIONS.slice(0, 4).map((region) => <button key={region.id} className={filter === region.id ? 'is-active' : ''} onClick={() => onFilter(region.id)}>{region.label}</button>)}
          </div>
        </div>
        <div className="fa-neuron-list">
          {filtered.map((profile, index) => {
            const affinity = [...profile.regionAffinities].sort((left, right) => right.score - left.score)[0]
            return (
              <button key={profile.memberKey} className={`fa-neuron-row ${selected?.memberKey === profile.memberKey ? 'is-active' : ''}`} onClick={() => onSelect(profile.memberKey)}>
                <span className="fa-list-rank">{String(index + 1).padStart(2, '0')}</span>
                <span className="fa-list-body"><b>{previewText(profile.text)}</b><span>{profile.kind === 'raw' ? 'Ham oluş' : 'Sentez nöronu'} · {profile.evidence.length} kanıt</span></span>
                <span className="fa-list-score"><strong>{formatDecimal(affinity?.score, 2)}</strong><small>{regionDefinition(affinity?.region ?? 'content').label.toLocaleUpperCase('tr-TR')}</small></span>
              </button>
            )
          })}
          {filtered.length === 0 && <div className="fa-empty"><div><Search size={23} /><h3>Bu filtrede profil yok</h3></div></div>}
        </div>
      </aside>

      {selected ? (
        <section className="fa-profile-main">
          <article className="fa-panel fa-profile-hero">
            <div className="fa-panel__heading"><div><span className="fa-eyebrow">{shortId(selected.memberKey)}</span><h1>Davranış profili</h1></div><span className="fa-chip fa-chip--indigo">{selected.vector.entries.length}/{selected.vector.dimension} özellik</span></div>
            <div className="fa-profile-text"><span>TEMSİL EDİLEN METİN İZİ</span><blockquote>“{selected.text.replace(/\n/g, ' ↵ ') || 'boşluk'}”</blockquote><small>{selected.kind === 'raw' ? 'Ham metin nöronu' : 'Onaylı sentez nöronu'} · kimlik {selected.identityDigest.slice(0, 12)}…</small></div>
            <div className="fa-metric-grid">
              <div><span>Q NORMU</span><strong>{formatDecimal(selected.vector.norm)}</strong><small>davranış büyüklüğü</small></div>
              <div><span>Q ÖZELLİĞİ</span><strong>{selected.vector.entries.length}</strong><small>seyrek koordinat</small></div>
              <div><span>KANIT</span><strong>{selected.evidence.length}</strong><small>bağımsız oluş</small></div>
              <div><span>BAĞLAM GÜCÜ</span><strong>{formatPercent(selected.diagnostics.relationStrength)}</strong><small>R davranışı</small></div>
            </div>
          </article>
          <article className="fa-panel fa-profile-panel">
            <div className="fa-panel__heading">
              <div><span className="fa-eyebrow">Q TANISI</span><h2>Bu nöron ağda ne yapıyor?</h2></div>
              <label className="fa-field" style={{ minWidth: 185 }}><span>KARŞILAŞTIR</span><select value={comparison?.memberKey ?? ''} onChange={(event) => onComparison(event.target.value || null)}><option value="">Karşılaştırma yok</option>{run.profiles.filter((profile) => profile.memberKey !== selected.memberKey).slice(0, 500).map((profile) => <option key={profile.memberKey} value={profile.memberKey}>{previewText(profile.text, 35)}</option>)}</select></label>
            </div>
            <ProfileBars items={profileBars(selected, comparison ?? undefined)} label={`${previewText(selected.text)} davranış tanıları`} />
            <div className="fa-setting-note"><Info size={13} /><span>Çubuklar kelimenin sözlük anlamını değil; konum, bağlam, yön ve geçiş davranışlarını gösterir. Turuncu işaret seçilen ikinci nörondur.</span></div>
          </article>
        </section>
      ) : <section className="fa-panel fa-empty"><div><BarChart3 size={28} /><h2>İncelenecek profil yok</h2></div></section>}

      <aside className="fa-profile-aside">
        <section className="fa-panel fa-panel--padded">
          <div className="fa-panel__heading"><div><span className="fa-eyebrow">İŞLEV HİPOTEZLERİ</span><h2>Üst bölge yakınlığı</h2></div><BrainCircuit size={16} /></div>
          <div className="fa-evidence-list">
            {selected?.regionAffinities.map((affinity) => {
              const definition = regionDefinition(affinity.region)
              const stateLabel = affinity.verified
                ? 'Çalışma zamanında doğrulandı'
                : definition.verification === 'runtime-required'
                  ? 'Runtime deneyimi gerekli'
                  : 'Statik hipotez'
              return (
                <div key={affinity.region} className="fa-evidence-row">
                  <b>{definition.label} · {formatDecimal(affinity.score, 3)}</b>
                  <span>{affinity.explanation}</span>
                  <span className={`fa-chip ${affinity.verified ? 'fa-chip--good' : 'fa-chip--warn'}`}>{stateLabel}</span>
                </div>
              )
            })}
          </div>
        </section>
        <section className="fa-panel fa-panel--padded">
          <div className="fa-panel__heading"><div><span className="fa-eyebrow">BAĞIMSIZ KANIT</span><h2>Geri izlenebilir oluşlar</h2></div><GitBranch size={16} /></div>
          <div className="fa-evidence-list">
            {topEvidence.map((evidence) => <div key={evidence.memberKey} className="fa-evidence-row"><b>{previewText(evidence.text)}</b><span>{shortId(evidence.neuronId)} · P{evidence.paragraphIndex + 1} · {evidence.start}–{evidence.end}</span></div>)}
            {topEvidence.length === 0 && <p className="fa-copy">Bu profil için gösterilebilir bağımsız kaynak kanıtı yok.</p>}
          </div>
        </section>
      </aside>
    </main>
  )
}

function FamiliesView({
  run,
  selectedFamilyId,
  filter,
  viewMode,
  onFilter,
  onViewMode,
  onSelectFamily,
  onSelectProfile,
}: {
  run: FamilyRun
  selectedFamilyId: string | null
  filter: FamilyFilter
  viewMode: FamilyViewMode
  onFilter: (filter: FamilyFilter) => void
  onViewMode: (view: FamilyViewMode) => void
  onSelectFamily: (id: string) => void
  onSelectProfile: (key: string) => void
}) {
  const profiles = useMemo(() => new Map(run.profiles.map((profile) => [profile.memberKey, profile])), [run.profiles])
  const visibleFamilies = run.families.filter((family) => {
    if (filter === 'all') return true
    if (filter === 'attention') return family.lifecycle === 'split-proposed' || family.lifecycle === 'merge-proposed' || family.lifecycle === 'retired'
    return family.lifecycle === filter
  })
  const selected = run.families.find((family) => family.id === selectedFamilyId) ?? visibleFamilies[0] ?? run.families[0] ?? null
  const familyMemberships = selected ? run.memberships.filter((membership) => membership.familyId === selected.id).sort((left, right) => right.weight - left.weight) : []
  const membershipByPair = useMemo(() => new Map(run.memberships.map((membership) => [`${membership.memberKey}\u0001${membership.familyId}`, membership])), [run.memberships])
  const matrixFamilies = run.families.slice(0, MAX_MATRIX_COLUMNS)
  const matrixProfiles = useMemo(() => {
    if (!selected) return run.profiles.slice(0, MAX_MATRIX_ROWS)
    const selectedKeys = new Set(familyMemberships.map((membership) => membership.memberKey))
    return [...run.profiles].sort((left, right) => Number(selectedKeys.has(right.memberKey)) - Number(selectedKeys.has(left.memberKey)) || left.memberKey.localeCompare(right.memberKey)).slice(0, MAX_MATRIX_ROWS)
  }, [familyMemberships, run.profiles, selected])
  const matrixValues = useMemo(() => {
    const values = new Float32Array(matrixProfiles.length * matrixFamilies.length)
    matrixProfiles.forEach((profile, row) => matrixFamilies.forEach((family, column) => {
      values[row * matrixFamilies.length + column] = membershipByPair.get(`${profile.memberKey}\u0001${family.id}`)?.weight ?? 0
    }))
    return values
  }, [matrixFamilies, matrixProfiles, membershipByPair])
  const selectedColumn = Math.max(0, matrixFamilies.findIndex((family) => family.id === selected?.id))
  const selectedRow = Math.max(0, matrixProfiles.findIndex((profile) => profile.memberKey === familyMemberships[0]?.memberKey))
  const activationByFamily = new Map(run.coalition.coalitions.flatMap((coalition) => coalition.familyActivations).map((activation) => [activation.familyId, activation.activation]))
  const graphNodes: FamilyGraphNode[] = run.families.map((family) => ({ id: family.id, label: familyLabel(family, profiles), region: regionDefinition(family.region).label, activation: activationByFamily.get(family.id) ?? 0, size: family.memberKeys.length, status: lifecycleLabel(family.lifecycle) }))
  const graphEdges: FamilyGraphEdge[] = run.familyTransitions.map((transition) => ({ id: transition.id, from: transition.fromFamilyId, to: transition.toFamilyId, weight: transition.probability, kind: 'transition', label: `G ${transition.probability.toFixed(4)}` }))

  return (
    <main className="fa-main fa-family-layout">
      <aside className="fa-panel fa-sidebar-panel">
        <div className="fa-sidebar-head">
          <div className="fa-panel__heading"><div><span className="fa-eyebrow">AİLELER 03</span><h2>{run.families.length} davranış ailesi</h2></div><span className="fa-chip fa-chip--indigo">{run.memberships.length} M</span></div>
          <div className="fa-filter-row">{([['all', 'Tümü'], ['provisional', 'Geçici'], ['strengthening', 'Güçlenen'], ['stable', 'Kararlı'], ['attention', 'İncele']] as Array<[FamilyFilter, string]>).map(([id, label]) => <button key={id} className={filter === id ? 'is-active' : ''} onClick={() => onFilter(id)}>{label}</button>)}</div>
        </div>
        <div className="fa-family-list">
          {visibleFamilies.map((family, index) => (
            <button key={family.id} className={`fa-family-row ${selected?.id === family.id ? 'is-active' : ''}`} onClick={() => onSelectFamily(family.id)}>
              <span className="fa-list-rank">{String(index + 1).padStart(2, '0')}</span>
              <span className="fa-list-body"><b>{familyLabel(family, profiles)}</b><span>{regionDefinition(family.region).label} · {family.sourceCount} kaynak · {family.supportCount} destek</span></span>
              <span className="fa-list-score"><strong>{formatDecimal(family.cohesion, 2)}</strong><small>{lifecycleLabel(family.lifecycle).toLocaleUpperCase('tr-TR')}</small></span>
            </button>
          ))}
          {visibleFamilies.length === 0 && <div className="fa-empty"><div><UsersRound size={25} /><h3>Bu durumda aile yok</h3></div></div>}
        </div>
      </aside>

      {selected ? (
        <section className="fa-family-content">
          <article className="fa-panel fa-family-hero">
            <div className="fa-panel__heading">
              <div><span className="fa-eyebrow">{selected.id}</span><h1>{familyLabel(selected, profiles)}</h1></div>
              <span className={`fa-chip ${lifecycleTone(selected.lifecycle)}`}>{lifecycleLabel(selected.lifecycle)}</span>
            </div>
            <p className="fa-copy">Üst işlev bölgesi: <b>{regionDefinition(selected.region).label}</b>. {selected.regionVerified ? 'Bu yakınlık çalışma zamanı kanıtıyla doğrulandı.' : 'Bu yakınlık statik davranıştan çıkarılan bir hipotezdir.'}</p>
            <div className="fa-family-metric-grid">
              <div><span>ETKİN ÜYE</span><strong>{formatDecimal(familyMemberships.reduce((sum, membership) => sum + membership.weight, 0), 1)}</strong><small>{selected.memberKeys.length} kayıt</small></div>
              <div><span>ÇEKİRDEK</span><strong>{selected.coreMemberKeys.length}</strong><small>yüksek M</small></div>
              <div><span>KOHEZYON</span><strong>{formatPercent(selected.cohesion)}</strong><small>Q iç uyumu</small></div>
              <div><span>AYRIŞMA</span><strong>{formatPercent(selected.separation)}</strong><small>en yakın aile farkı</small></div>
              <div><span>KARARLILIK</span><strong>{formatPercent(selected.stability)}</strong><small>koşular arası</small></div>
              <div><span>FAYDA ΔU</span><strong>{formatDecimal(selected.utility, 4)}</strong><small>ablasyon</small></div>
            </div>
            <div className="fa-setting-note"><Info size={13} /><span>{selected.lifecycleReason}</span></div>
          </article>

          <article className="fa-panel fa-membership-panel">
            <div className="fa-panel__heading">
              <div><span className="fa-eyebrow">YUMUŞAK ÜYELİK M</span><h2>Nöronlar tek bir kutuya kapatılmaz</h2></div>
              <div className="fa-tabs"><button className={viewMode === 'matrix' ? 'is-active' : ''} onClick={() => onViewMode('matrix')}><Layers3 size={11} /> Matris</button><button className={viewMode === 'graph' ? 'is-active' : ''} onClick={() => onViewMode('graph')}><Network size={11} /> Ağ</button></div>
            </div>
            {viewMode === 'matrix' ? (
              <>
                {(run.profiles.length > MAX_MATRIX_ROWS || run.families.length > MAX_MATRIX_COLUMNS) && <div className="fa-warning" style={{ marginTop: 12 }}><Info size={14} /><span>Görsel pencere ilk {MAX_MATRIX_ROWS} profil ve {MAX_MATRIX_COLUMNS} aileyle sınırlıdır; motorun seyrek üyelik arşivi eksiksizdir.</span></div>}
                <FamilyMatrixCanvas
                  values={matrixValues}
                  rowLabels={matrixProfiles.map((profile) => previewText(profile.text, 22))}
                  columnLabels={matrixFamilies.map((family) => `${regionDefinition(family.region).label} ${family.id.slice(-4)}`)}
                  selectedRow={selectedRow}
                  selectedColumn={selectedColumn}
                  onSelect={(row, column) => {
                    const profile = matrixProfiles[row]
                    const family = matrixFamilies[column]
                    if (profile) onSelectProfile(profile.memberKey)
                    if (family) onSelectFamily(family.id)
                  }}
                  label="Nöronların ailelere yumuşak üyelik matrisi"
                />
              </>
            ) : (
              <FamilyGraph nodes={graphNodes} edges={graphEdges} selectedId={selected.id} onSelect={onSelectFamily} label="Aileler arası seyrek geçiş ağı" />
            )}
          </article>

          <article className="fa-panel fa-member-panel">
            <div className="fa-panel__heading"><div><span className="fa-eyebrow">ÜYELİK AYRINTISI</span><h2>Davranışça en yakın nöronlar</h2></div><span className="fa-chip fa-chip--good">Kaynak bağımsız</span></div>
            <div style={{ overflow: 'auto', marginTop: 12 }}>
              <table className="fa-data-table">
                <thead><tr><th>Nöron izi</th><th>M ağırlığı</th><th>Q uyumu</th><th>Bölge uyumu</th><th>Kanıt güveni</th><th>Durum</th></tr></thead>
                <tbody>{familyMemberships.map((membership) => {
                  const profile = profiles.get(membership.memberKey)
                  return <tr key={membership.memberKey}><th><button className="fa-button fa-button--ghost fa-button--compact" onClick={() => onSelectProfile(membership.memberKey)}>{previewText(profile?.text ?? membership.memberKey, 42)}</button></th><td>{membership.weight.toFixed(4)}</td><td>{membership.breakdown.behaviorSimilarity.toFixed(4)}</td><td>{membership.breakdown.regionCompatibility.toFixed(4)}</td><td>{membership.breakdown.evidenceConfidence.toFixed(4)}</td><td>{membership.core ? 'Çekirdek' : `${membership.rank}. üyelik`}</td></tr>
                })}</tbody>
              </table>
            </div>
          </article>
        </section>
      ) : <section className="fa-panel fa-empty"><div><UsersRound size={29} /><h2>Seçili eşiklerle aile oluşmadı</h2><p>Girdi sayısını artırabilir veya oluşum eşiğini dikkatle düşürebilirsin.</p></div></section>}
    </main>
  )
}

function FlowView({
  run,
  prompt,
  runState,
  progress,
  error,
  onPrompt,
  onRun,
  onCancel,
  onFamily,
}: {
  run: FamilyRun
  prompt: string
  runState: RunState
  progress: FamilyProgress | null
  error: string | null
  onPrompt: (value: string) => void
  onRun: (activations: NeuronActivation[]) => void
  onCancel: () => void
  onFamily: (id: string) => void
}) {
  const profileByKey = useMemo(() => new Map(run.profiles.map((profile) => [profile.memberKey, profile])), [run.profiles])
  const normalizedPrompt = prompt.normalize('NFC').toLocaleLowerCase('tr-TR')
  const matchedProfiles = useMemo(() => {
    if (!normalizedPrompt.trim()) return []
    return run.profiles
      .filter((profile) => {
        const text = profile.text.normalize('NFC').toLocaleLowerCase('tr-TR').trim()
        return text.length >= 2 && text.length <= 80 && normalizedPrompt.includes(text)
      })
      .sort((left, right) => right.text.length - left.text.length || left.memberKey.localeCompare(right.memberKey))
      .slice(0, run.config.maximumActiveNeurons)
  }, [normalizedPrompt, run.config.maximumActiveNeurons, run.profiles])
  const coalition = [...run.coalition.coalitions].sort((left, right) => right.dominance - left.dominance)[0] ?? null
  const eventProfile = coalition ? profileByKey.get(coalition.eventMemberKey) ?? null : null
  const bindingFor = (role: EventRoleBinding['role']) => coalition?.bindings.find((binding) => binding.role === role) ?? null
  const agent = bindingFor('agent-hypothesis')
  const patient = bindingFor('patient-hypothesis')
  const instrument = bindingFor('instrument-hypothesis')
  const operator = bindingFor('operator')
  const bindingText = (binding: EventRoleBinding | null) => binding ? profileByKey.get(binding.memberKey)?.text ?? shortId(binding.memberKey) : '—'
  const familyById = new Map(run.families.map((family) => [family.id, family]))
  const routing = run.routingPreview

  return (
    <main className="fa-main fa-flow-layout">
      <section className="fa-panel fa-panel--padded">
        <div className="fa-panel__heading"><div><span className="fa-eyebrow">DÜŞÜNCE AKIŞI 04</span><h1>Geçici koalisyonu yaşat</h1></div><span className="fa-chip fa-chip--warn">İç ses üretimi değil</span></div>
        <p className="fa-copy">Bu deney hazır bir dil modeli kullanmaz. Yazdığın metinde yüzey olarak bulunan mevcut nöron izleri etkinleştirilir; aile ateşlemesi, geçici roller ve iki seviyeli yönlendirme yeniden hesaplanır.</p>
        <div className="fa-flow-toolbar" style={{ marginTop: 14 }}>
          <label className="fa-flow-prompt"><span>DENEY SİNYALİ</span><textarea value={prompt} onChange={(event) => onPrompt(event.target.value)} placeholder="Örnek: Ali ekmeği bıçakla kesebilir." /></label>
          <div className="fa-flow-controls">
            {runState !== 'idle' ? (
              <Button tone="danger" icon={runState === 'cancelling' ? <Pause size={15} /> : <CircleStop size={15} />} onClick={onCancel} disabled={runState === 'cancelling'}>Durdur</Button>
            ) : (
              <Button
                tone="primary"
                icon={<Zap size={15} />}
                disabled={matchedProfiles.length === 0}
                onClick={() => onRun(matchedProfiles.map((profile, index) => ({
                  memberKey: profile.memberKey,
                  activation: Math.max(.55, 1 - index * .035),
                  goalRelevance: 0,
                  conflict: 0,
                  repetition: 0,
                })))}
              >Koalisyonu çalıştır</Button>
            )}
          </div>
        </div>
        <div className={matchedProfiles.length ? 'fa-good-note' : 'fa-warning'} style={{ marginTop: 10 }}><Search size={14} /><span>{matchedProfiles.length ? `${matchedProfiles.length} mevcut nöron izi yüzey eşleşmesiyle etkinleştirilecek: ${matchedProfiles.slice(0, 5).map((profile) => `“${previewText(profile.text, 20)}”`).join(', ')}${matchedProfiles.length > 5 ? '…' : ''}` : 'Henüz mevcut profillerle birebir yüzey eşleşmesi bulunmadı. Bu alan bilinmeyen metne sahte anlam atamaz.'}</span></div>
        {runState !== 'idle' && progress && <div aria-live="polite"><div className="fa-progress-heading" style={{ marginTop: 12 }}><span>{progress.message}</span><b>%{Math.round(progress.ratio * 100)}</b></div><div className="fa-progress-track"><i style={{ width: `${progress.ratio * 100}%` }} /></div></div>}
        {error && <div className="fa-error" style={{ marginTop: 10 }}><X size={14} /><span>{error}</span></div>}
      </section>

      <div className="fa-flow-grid">
        <section className="fa-panel fa-event-frame">
          <div className="fa-panel__heading"><div><span className="fa-eyebrow">GEÇİCİ OLAY ÇERÇEVESİ</span><h2>{coalition ? `Baskınlık ${formatDecimal(coalition.dominance, 3)}` : 'Baskın koalisyon yok'}</h2></div><span className="fa-chip fa-chip--indigo">tick {run.coalition.tick}</span></div>
          {coalition ? (
            <>
              <div className="fa-event-stage">
                <svg className="fa-event-lines" viewBox="0 0 800 430" aria-hidden="true"><line x1="400" y1="215" x2="155" y2="95" /><line x1="400" y1="215" x2="650" y2="95" /><line x1="400" y1="215" x2="160" y2="345" /><line className="operator" x1="400" y1="215" x2="645" y2="345" /></svg>
                <div className="fa-event-core"><span>OLAY MERKEZİ</span><b>{previewText(eventProfile?.text ?? coalition.eventMemberKey, 34)}</b><small>{familyById.get(coalition.eventFamilyId) ? regionDefinition(familyById.get(coalition.eventFamilyId)!.region).label : 'Olay ailesi'}</small></div>
                <RoleSlot className="is-agent" label="YAPAN HİPOTEZİ" binding={agent} text={bindingText(agent)} />
                <RoleSlot className="is-patient" label="ETKİLENEN HİPOTEZİ" binding={patient} text={bindingText(patient)} />
                <RoleSlot className="is-instrument" label="ARAÇ HİPOTEZİ" binding={instrument} text={bindingText(instrument)} />
                <RoleSlot className="is-operator" label="OPERATÖR" binding={operator} text={bindingText(operator)} />
              </div>
              <div className="fa-boundary-note" style={{ marginTop: 10 }}><Info size={14} /><span>Bu rol bağları <b>geçicidir</b>. Örneğin “bıçak” kalıcı olarak araç ailesine taşınmaz; yalnız bu olayda araç hipotezi olarak bağlanabilir.</span></div>
              <details className="fa-accessible-data"><summary>Koalisyonu erişilebilir tabloda göster</summary><div><table><thead><tr><th>Rol</th><th>Nöron</th><th>Puan</th><th>Açıklama</th></tr></thead><tbody>{coalition.bindings.map((binding) => <tr key={`${binding.role}-${binding.memberKey}`}><th>{binding.role}</th><td>{bindingText(binding)}</td><td>{binding.score.toFixed(4)}</td><td>{binding.explanation}</td></tr>)}</tbody></table></div></details>
            </>
          ) : <div className="fa-empty"><div><Waypoints size={29} /><h2>Koalisyon oluşmadı</h2><p>{run.coalition.unmatchedReason ?? 'Etkin aileler geçici bir olay çerçevesi kuracak kadar uyumlu değildi.'}</p></div></div>}
        </section>

        <aside className="fa-router-panel">
          <section className="fa-panel fa-panel--padded">
            <div className="fa-panel__heading"><div><span className="fa-eyebrow">AİLE ETKİNLİĞİ A</span><h2>Aynı anda çalışan bölgeler</h2></div><Activity size={16} /></div>
            <div className="fa-evidence-list">{coalition?.familyActivations.slice(0, 12).map((activation) => {
              const family = familyById.get(activation.familyId)
              return <button key={activation.familyId} className="fa-router-step" onClick={() => onFamily(activation.familyId)}><span>A</span><div><b>{family ? familyLabel(family, profileByKey) : activation.familyId}</b><small>{family ? regionDefinition(family.region).label : 'Aile'}</small></div><strong>{activation.activation.toFixed(4)}</strong></button>
            })}{!coalition && <p className="fa-copy">Etkin aile bulunmuyor.</p>}</div>
          </section>

          <section className="fa-panel fa-panel--padded">
            <div className="fa-panel__heading"><div><span className="fa-eyebrow">İKİ SEVİYELİ YÖNLENDİRME</span><h2>Önce aile, sonra nöron</h2></div><Route size={16} /></div>
            {routing ? (
              <>
                <div className="fa-evidence-list">{routing.entries.slice(0, 8).map((entry, index) => <div key={entry.memberKey} className="fa-router-step"><span>{index + 1}</span><div><b>{previewText(profileByKey.get(entry.memberKey)?.text ?? entry.memberKey, 34)}</b><small>G aile {entry.targetFamilyProbability.toFixed(3)} · H {entry.breakdown.habit.toFixed(3)} · sinyal {entry.breakdown.signal.toFixed(3)}</small></div><strong>{entry.probability.toFixed(4)}</strong></div>)}</div>
                <div className="fa-equation" style={{ marginTop: 11 }}>P(j | i,S) = P(Fⱼ | Fᵢ,S) × P(j | Fⱼ,S)</div>
                <p className="fa-copy">{routing.explanation}</p>
              </>
            ) : <div className="fa-empty"><div><Route size={25} /><h3>Rota önizlemesi yok</h3><p>Bir çalışma zamanı etkinliği olmadan yönlendirme dağılımı kurulmadı.</p></div></div>}
          </section>
        </aside>
      </div>
    </main>
  )
}

function RoleSlot({ className, label, binding, text }: { className: string; label: string; binding: EventRoleBinding | null; text: string }) {
  return <div className={`fa-role-slot ${className}`}><span>{label}</span><b>{previewText(text, 28)}</b><small>{binding ? `geçici · ${binding.score.toFixed(3)}` : 'eşleşmedi'}</small></div>
}

function ValidationView({ run, selectedFamilyId, onFamily }: { run: FamilyRun; selectedFamilyId: string | null; onFamily: (id: string) => void }) {
  const profiles = useMemo(() => new Map(run.profiles.map((profile) => [profile.memberKey, profile])), [run.profiles])
  const selected = run.families.find((family) => family.id === selectedFamilyId) ?? run.families[0] ?? null
  const utility = run.utilityReports.find((report) => report.familyId === selected?.id) ?? null
  const validated = run.utilityReports.filter((report) => report.status === 'validated').length
  const stable = run.families.filter((family) => family.lifecycle === 'stable').length
  const attention = run.families.filter((family) => family.lifecycle === 'split-proposed' || family.lifecycle === 'merge-proposed').length
  const meanUtility = run.utilityReports.length ? run.utilityReports.reduce((sum, report) => sum + report.deltaUtility, 0) / run.utilityReports.length : 0

  return (
    <main className="fa-main fa-validation-layout">
      <div className="fa-validation-summary">
        <article><span>AİLE</span><strong>{run.families.length}</strong><small>öğrenilmiş küme</small></article>
        <article><span>KARARLI</span><strong>{stable}</strong><small>yaşam eşiğini geçti</small></article>
        <article><span>FAYDASI DOĞRULANAN</span><strong>{validated}</strong><small>kaynak holdout</small></article>
        <article><span>İNCELEME GEREKEN</span><strong>{attention}</strong><small>bölünme / birleşme</small></article>
        <article><span>ORTALAMA ΔU</span><strong>{formatDecimal(meanUtility, 4)}</strong><small>aile açık − susturulmuş</small></article>
      </div>

      <div className="fa-validation-grid">
        <section className="fa-panel fa-panel--padded">
          <div className="fa-panel__heading">
            <div><span className="fa-eyebrow">DOĞRULAMA 05</span><h1>Aile susturulunca ne kaybediliyor?</h1></div>
            <label className="fa-field" style={{ minWidth: 210 }}><span>AİLE</span><select value={selected?.id ?? ''} onChange={(event) => onFamily(event.target.value)}>{run.families.map((family) => <option key={family.id} value={family.id}>{familyLabel(family, profiles)}</option>)}</select></label>
          </div>
          {selected && utility ? <UtilityReportView report={utility} /> : <div className="fa-empty"><div><TestTube2 size={28} /><h2>Yeterli ablasyon kanıtı yok</h2><p>Kaynak katlamalı karşılaştırma için daha fazla bağımsız bağlam gerekebilir.</p></div></div>}
        </section>

        <aside style={{ display: 'grid', gap: 12 }}>
          <section className="fa-panel fa-panel--padded">
            <div className="fa-panel__heading"><div><span className="fa-eyebrow">YAŞAM DÖNGÜSÜ</span><h2>{selected ? lifecycleLabel(selected.lifecycle) : 'Aile yok'}</h2></div><Split size={16} /></div>
            {selected && <>
              <div className="fa-lifecycle">
                <LifecycleStep index={1} label="Geçici" detail="Davranış benzerliği bulundu" state={lifecycleState(selected.lifecycle, 0)} />
                <LifecycleStep index={2} label="Güçleniyor" detail="Bağımsız destek birikiyor" state={lifecycleState(selected.lifecycle, 1)} />
                <LifecycleStep index={3} label="Kararlı" detail="Uyum, ayrışma ve fayda korundu" state={lifecycleState(selected.lifecycle, 2)} />
                {(selected.lifecycle === 'split-proposed' || selected.lifecycle === 'merge-proposed' || selected.lifecycle === 'retired') && <LifecycleStep index={4} label={lifecycleLabel(selected.lifecycle)} detail={selected.lifecycleReason} state="current" />}
              </div>
              <div className="fa-setting-note"><Info size={13} /><span>{selected.lifecycleReason}</span></div>
            </>}
          </section>

          <section className="fa-panel fa-panel--padded">
            <div className="fa-panel__heading"><div><span className="fa-eyebrow">KOŞU ÇIKTISI</span><h2>Deneysel arşiv sınırı</h2></div><Archive size={16} /></div>
            <p className="fa-copy">Bu aşama yukarı akış arşivlerine yazmaz. Kalıcı aile arşivi bağlanana kadar koşuyu taşınabilir JSON olarak dışa aktarabilirsin.</p>
            <Button wide icon={<Download size={14} />} onClick={() => downloadRun(run)} style={{ marginTop: 12 }}>Koşuyu dışa aktar</Button>
            <Button wide tone="ghost" icon={<Archive size={14} />} disabled title="Ayrı aile arşivi bağlantısı sonraki güvenli depolama adımında açılacak" style={{ marginTop: 6 }}>Aile arşivine kaydet</Button>
          </section>
        </aside>
      </div>

      {run.warnings.length > 0 && <section className="fa-panel fa-panel--padded"><div className="fa-panel__heading"><div><span className="fa-eyebrow">MOTOR UYARILARI</span><h2>Bu koşunun yorum sınırları</h2></div><AlertTriangle size={17} /></div><div className="fa-evidence-list">{run.warnings.map((warning, index) => <div key={`${index}-${warning}`} className="fa-warning"><AlertTriangle size={13} /><span>{warning}</span></div>)}</div></section>}
    </main>
  )
}

function UtilityReportView({ report }: { report: FamilyUtilityReport }) {
  const scale = Math.max(.000001, ...report.folds.flatMap((fold) => [fold.fullLoss, fold.ablatedLoss]))
  return <>
    <div className="fa-ablation-grid"><div><span>TAM KAYIP</span><strong>{formatDecimal(report.fullLoss, 4)}</strong><small>aile açık</small></div><div><span>SUSTURULMUŞ KAYIP</span><strong>{formatDecimal(report.ablatedLoss, 4)}</strong><small>aile kapalı</small></div><div><span>FAYDA ΔU</span><strong className={report.deltaUtility > 0 ? 'is-positive' : ''}>{formatDecimal(report.deltaUtility, 4)}</strong><small>{report.confidenceLow.toFixed(4)}…{report.confidenceHigh.toFixed(4)}</small></div></div>
    <div className={`fa-${report.status === 'validated' ? 'good-note' : report.status === 'neutral-or-harmful' ? 'warning' : 'setting-note'}`} style={{ marginTop: 12 }}><TestTube2 size={14} /><span>{report.explanation}</span></div>
    <div className="fa-ablation-bars">{report.folds.slice(0, 20).map((fold) => { const sourceKey = fold.heldOutSourceId ?? fold.heldOutSourceDigest; return <div className="fa-ablation-row" key={sourceKey}><span title={`${sourceKey} · ${fold.heldOutSourceDigest}`}>{shortId(sourceKey)}</span><div className="fa-ablation-track" aria-label={`Tam kayıp ${fold.fullLoss.toFixed(4)}, susturulmuş kayıp ${fold.ablatedLoss.toFixed(4)}`}><i style={{ width: `${Math.max(1, fold.fullLoss / scale * 100)}%` }} /><em style={{ width: `${Math.max(1, fold.ablatedLoss / scale * 100)}%` }} /></div><b>{fold.deltaUtility.toFixed(4)}</b></div> })}</div>
    <details className="fa-accessible-data"><summary>Kaynak katlarını erişilebilir tabloda göster</summary><div><table><thead><tr><th>Tutulan kaynak</th><th>Vaka</th><th>Tam kayıp</th><th>Susturulmuş</th><th>ΔU</th></tr></thead><tbody>{report.folds.map((fold) => { const sourceKey = fold.heldOutSourceId ?? fold.heldOutSourceDigest; return <tr key={sourceKey}><th>{sourceKey}</th><td>{fold.caseCount}</td><td>{fold.fullLoss.toFixed(6)}</td><td>{fold.ablatedLoss.toFixed(6)}</td><td>{fold.deltaUtility.toFixed(6)}</td></tr> })}</tbody></table></div></details>
  </>
}

function lifecycleState(lifecycle: FamilyLifecycle, index: number): 'pending' | 'done' | 'current' {
  const level = lifecycle === 'provisional' ? 0 : lifecycle === 'strengthening' ? 1 : 2
  if (index < level) return 'done'
  if (index === level && lifecycle !== 'split-proposed' && lifecycle !== 'merge-proposed' && lifecycle !== 'retired') return 'current'
  if (index <= level) return 'done'
  return 'pending'
}

function LifecycleStep({ index, label, detail, state }: { index: number; label: string; detail: string; state: 'pending' | 'done' | 'current' }) {
  return <div className={`fa-lifecycle-step ${state === 'done' ? 'is-done' : state === 'current' ? 'is-current' : ''}`}><span>{state === 'done' ? <Check size={12} /> : index}</span><div><b>{label}</b><small>{detail}</small></div><strong>{state === 'done' ? 'GEÇTİ' : state === 'current' ? 'ŞİMDİ' : 'BEKLİYOR'}</strong></div>
}

export function FamilyApp() {
  const [loading, setLoading] = useState(true)
  const [projects, setProjects] = useState<PersistedProject[]>([])
  const [decisions, setDecisions] = useState<SynthesisArchiveRecord[]>([])
  const [sourceWarnings, setSourceWarnings] = useState<string[]>([])
  const [familyProjects, setFamilyProjects] = useState<FamilyProject[]>([])
  const [archiveNotice, setArchiveNotice] = useState<string | null>(null)
  const [view, setView] = useState<LabView>('inputs')
  const [activeSpaceId, setActiveSpaceId] = useState('')
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set())
  const [selectedDecisionIds, setSelectedDecisionIds] = useState<Set<string>>(new Set())
  const [config, setConfig] = useState<ControlledConfig>(INITIAL_CONFIG)
  const [runState, setRunState] = useState<RunState>('idle')
  const [progress, setProgress] = useState<FamilyProgress | null>(null)
  const [run, setRun] = useState<FamilyRun | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedProfileKey, setSelectedProfileKey] = useState<string | null>(null)
  const [comparisonProfileKey, setComparisonProfileKey] = useState<string | null>(null)
  const [profileFilter, setProfileFilter] = useState<ProfileFilter>('all')
  const [selectedFamilyId, setSelectedFamilyId] = useState<string | null>(null)
  const [familyFilter, setFamilyFilter] = useState<FamilyFilter>('all')
  const [familyViewMode, setFamilyViewMode] = useState<FamilyViewMode>('matrix')
  const [flowPrompt, setFlowPrompt] = useState('')
  const workerRef = useRef<Worker | null>(null)
  const runTokenRef = useRef<string | null>(null)
  const destinationRef = useRef<LabView>('profiles')

  const analyses = useMemo(() => projects.flatMap((project) => project.analysis ? [project.analysis] : []), [projects])
  const compatibleAnalyses = useMemo(() => analyses.filter((analysis) => analysis.representationSpaceId === activeSpaceId), [activeSpaceId, analyses])
  const compatibleDecisions = useMemo(() => decisions.filter((decision) => decision.representationSpaceId === activeSpaceId), [activeSpaceId, decisions])

  const applyBundle = (
    nextProjects: PersistedProject[],
    nextDecisions: SynthesisArchiveRecord[],
    warnings: string[],
    archived: readonly FamilyProject[],
  ) => {
    setProjects(nextProjects)
    setDecisions(nextDecisions)
    setSourceWarnings(warnings)
    const spaceCounts = new Map<string, number>()
    for (const project of nextProjects) if (project.analysis) spaceCounts.set(project.analysis.representationSpaceId, (spaceCounts.get(project.analysis.representationSpaceId) ?? 0) + 1)
    for (const decision of nextDecisions) spaceCounts.set(decision.representationSpaceId, (spaceCounts.get(decision.representationSpaceId) ?? 0) + .1)
    const nextSpace = [...spaceCounts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] ?? ''
    setActiveSpaceId(nextSpace)
    setSelectedRunIds(new Set(nextProjects.flatMap((project) => project.analysis?.representationSpaceId === nextSpace ? [project.analysis.runId] : [])))
    setSelectedDecisionIds(new Set(nextDecisions.filter((decision) => decision.representationSpaceId === nextSpace).map((decision) => decision.id)))
    const restored = archived.find((project) => project.run.representationSpaceId === nextSpace)?.run ?? null
    setRun(restored)
    setSelectedProfileKey(restored?.profiles[0]?.memberKey ?? null)
    setComparisonProfileKey(null)
    setSelectedFamilyId(restored?.families[0]?.id ?? null)
    if (restored) {
      setArchiveNotice('Yerel arşivdeki tarihsel aile modeli geri yüklendi; güncel kaynak kanıtı ancak yeni bir koşu ve manifest kontrolüyle doğrulanır.')
    }
  }

  const refresh = async () => {
    setLoading(true)
    setError(null)
    try {
      const [bundleResult, archiveResult] = await Promise.allSettled([
        loadFamilySourceBundle(),
        loadFamilyProjects(),
      ])
      if (bundleResult.status === 'rejected') throw bundleResult.reason
      const archived = archiveResult.status === 'fulfilled' ? archiveResult.value : []
      setFamilyProjects(archived)
      setArchiveNotice(archiveResult.status === 'rejected'
        ? 'Aile arşivi açılamadı; deney yapılabilir fakat sonuçlar bu oturumda kalır.'
        : null)
      applyBundle(
        bundleResult.value.projects,
        bundleResult.value.decisions,
        bundleResult.value.warnings,
        archived,
      )
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Yukarı akış arşivleri açılamadı.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let active = true
    void Promise.allSettled([loadFamilySourceBundle(), loadFamilyProjects()])
      .then(([bundleResult, archiveResult]) => {
        if (!active) return
        if (bundleResult.status === 'rejected') throw bundleResult.reason
        const archived = archiveResult.status === 'fulfilled' ? archiveResult.value : []
        setFamilyProjects(archived)
        setArchiveNotice(archiveResult.status === 'rejected'
          ? 'Aile arşivi açılamadı; deney yapılabilir fakat sonuçlar bu oturumda kalır.'
          : null)
        applyBundle(
          bundleResult.value.projects,
          bundleResult.value.decisions,
          bundleResult.value.warnings,
          archived,
        )
      })
      .catch((caught) => {
        if (active) setError(caught instanceof Error ? caught.message : 'Yukarı akış arşivleri açılamadı.')
      })
      .finally(() => { if (active) setLoading(false) })
    return () => {
      active = false
      workerRef.current?.terminate()
      workerRef.current = null
    }
  }, [])

  const selectSpace = (space: string) => {
    setActiveSpaceId(space)
    setSelectedRunIds(new Set(analyses.filter((analysis) => analysis.representationSpaceId === space).map((analysis) => analysis.runId)))
    setSelectedDecisionIds(new Set(decisions.filter((decision) => decision.representationSpaceId === space).map((decision) => decision.id)))
    const restored = familyProjects.find((project) => project.run.representationSpaceId === space)?.run ?? null
    setRun(restored)
    setSelectedProfileKey(restored?.profiles[0]?.memberKey ?? null)
    setComparisonProfileKey(null)
    setSelectedFamilyId(restored?.families[0]?.id ?? null)
    setArchiveNotice(restored
      ? 'Bu temsil uzayının tarihsel aile modeli yerel arşivden açıldı; güncel korpus için yeniden çalıştırabilirsin.'
      : null)
    setView('inputs')
  }

  const archiveRunIfInputsAreCurrent = async (
    completedRun: FamilyRun,
    analysisRunIds: readonly string[],
    decisionIds: readonly string[],
    mode: 'model' | 'coalition',
  ) => {
    setArchiveNotice(mode === 'model'
      ? 'Girdi manifesti yeniden doğrulanıyor; aile modeli henüz arşivlenmedi…'
      : 'Girdi manifesti yeniden doğrulanıyor; geçici koalisyon günlüğü henüz yazılmadı…')
    try {
      const freshBundle = await loadFamilySourceBundle()
      const freshAnalyses = freshBundle.projects
        .flatMap((project) => project.analysis ? [project.analysis] : [])
        .filter((analysis) => analysisRunIds.includes(analysis.runId))
      const freshDecisions = freshBundle.decisions.filter((decision) => decisionIds.includes(decision.id))
      if (
        freshAnalyses.length !== analysisRunIds.length ||
        freshDecisions.length !== decisionIds.length
      ) {
        setArchiveNotice('Hesap sırasında bir kaynak veya sentez kararı değişti; sonuç gösteriliyor fakat eski kanıt arşive yazılmadı.')
        return
      }
      const freshInput = createFamilyRunInput(freshAnalyses, freshDecisions)
      const freshManifest = createFamilyInputManifest(
        freshInput.analyses,
        freshInput.synthesisDecisions ?? [],
      )
      if (freshManifest.corpusFingerprint !== completedRun.inputManifest.corpusFingerprint) {
        setArchiveNotice('Hesap sırasında yukarı akış verisi değişti; sonuç gösteriliyor fakat manifest uyuşmadığı için arşive yazılmadı.')
        return
      }
      if (mode === 'coalition') {
        await saveCoalitionLog(completedRun.id, completedRun.coalition)
        setArchiveNotice('Geçici koalisyon özeti bounded deney günlüğüne kaydedildi; rol bağları kalıcı aile üyeliğine yazılmadı.')
        return
      }
      const saved = await saveFamilyRun(
        completedRun,
        `Aile modeli · ${new Date(completedRun.createdAt).toLocaleString('tr-TR')}`,
      )
      setFamilyProjects((current) => [
        saved,
        ...current.filter((project) => project.id !== saved.id),
      ].sort((left, right) => right.savedAt.localeCompare(left.savedAt)))
      setArchiveNotice(`Aile modeli yerel arşive kaydedildi · revizyon ${saved.storageRevision}`)
    } catch (caught) {
      setArchiveNotice(caught instanceof Error
        ? `Aile modeli hesaplandı fakat arşivlenemedi: ${caught.message}`
        : 'Aile modeli hesaplandı fakat yerel arşive yazılamadı.')
    }
  }

  const runExperiment = (activations?: NeuronActivation[], destination: LabView = 'profiles') => {
    const selectedAnalyses = selectedAnalysesFor(projects, selectedRunIds)
    const selectedDecisions = decisions.filter((decision) => selectedDecisionIds.has(decision.id))
    const selectedAnalysisRunIds = selectedAnalyses.map((analysis) => analysis.runId)
    const selectedSynthesisDecisionIds = selectedDecisions.map((decision) => decision.id)
    if (selectedAnalyses.length === 0) {
      setError('En az bir uyumlu Metin Laboratuvarı analizi seçmelisin.')
      return
    }
    workerRef.current?.terminate()
    const worker = new Worker(new URL('../families/families.worker.ts', import.meta.url), { type: 'module', name: 'noron-aileleri' })
    const runToken = `family-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
    workerRef.current = worker
    runTokenRef.current = runToken
    destinationRef.current = destination
    setRunState('running')
    setProgress({ stage: 'validating', completed: 0, total: 1, ratio: 0, message: 'Girdiler doğrulanıyor…', profileCount: 0, familyCount: 0, membershipCount: 0, transitionCount: 0 })
    setError(null)
    worker.onerror = () => {
      if (runTokenRef.current !== runToken) return
      setError('Nöron ailesi Worker işlemi beklenmeyen biçimde durdu.')
      setRunState('idle')
      setProgress(null)
      worker.terminate()
      workerRef.current = null
      runTokenRef.current = null
    }
    worker.onmessage = (event: MessageEvent<FamilyWorkerResponse>) => {
      const response = event.data
      if (response.runToken !== runToken || runTokenRef.current !== runToken) return
      if (response.type === 'progress') {
        setProgress(response.progress)
      } else if (response.type === 'result') {
        setRun(response.run)
        setSelectedProfileKey(response.run.profiles[0]?.memberKey ?? null)
        setComparisonProfileKey(null)
        setSelectedFamilyId(response.run.families[0]?.id ?? null)
        setRunState('idle')
        setProgress(null)
        setView(destinationRef.current)
        worker.terminate()
        workerRef.current = null
        runTokenRef.current = null
        void archiveRunIfInputsAreCurrent(
          response.run,
          selectedAnalysisRunIds,
          selectedSynthesisDecisionIds,
          destination === 'flow' ? 'coalition' : 'model',
        )
      } else if (response.type === 'cancelled') {
        setRunState('idle')
        setProgress(null)
        worker.terminate()
        workerRef.current = null
        runTokenRef.current = null
      } else if (response.type === 'error') {
        setError(response.message)
        setRunState('idle')
        setProgress(null)
        worker.terminate()
        workerRef.current = null
        runTokenRef.current = null
      }
    }
    const input = createFamilyRunInput(selectedAnalyses, selectedDecisions, run, activations)
    const request: FamilyWorkerRequest = {
      type: 'run',
      runToken,
      input,
      config,
    }
    worker.postMessage(request, familyRunTransferables(input))
  }

  const cancelRun = () => {
    const worker = workerRef.current
    const runToken = runTokenRef.current
    if (!worker || !runToken) return
    setRunState('cancelling')
    runTokenRef.current = null
    worker.terminate()
    workerRef.current = null
    setProgress(null)
    setRunState('idle')
  }

  const navigate = (next: LabView) => {
    if (runState !== 'idle') return
    if (next !== 'inputs' && !run) return
    setView(next)
  }

  const status = runState !== 'idle'
    ? { label: 'Aileler hesaplanıyor', tone: 'is-running' }
    : run
      ? { label: `${run.families.length} aile hazır`, tone: 'is-ready' }
      : sourceWarnings.length
        ? { label: 'Eksik arşiv bağlantısı', tone: 'is-warn' }
        : { label: 'Korpus bekleniyor', tone: '' }

  if (loading) return <div className="fa-app"><div className="fa-empty" style={{ minHeight: 520 }}><div><span className="fa-status-pill is-running"><i /> Salt-okur arşivler hazırlanıyor…</span></div></div></div>

  return (
    <div className="fa-app">
      <header className="fa-topbar">
        <SuiteMenu currentApp="families" navigationLocked={runState !== 'idle'}><div className="fa-brand"><span className="fa-brand__mark"><UsersRound size={22} /></span><div><strong>NÖRON AİLELERİ</strong><small>DAVRANIŞ, ÜYELİK VE KOALİSYON LABORATUVARI · v0.1</small></div></div></SuiteMenu>
        <span className={`fa-status-pill ${status.tone}`}><i />{status.label}</span>
        <div className="fa-topbar__actions">{run && <button className="fa-icon-button" aria-label="Koşuyu dışa aktar" title="Koşuyu dışa aktar" onClick={() => downloadRun(run)}><Download size={15} /></button>}</div>
      </header>

      <nav className="fa-stage-rail" aria-label="Nöron aileleri aşamaları">
        {STAGES.map((stage, index) => {
          const disabled = runState !== 'idle' || (stage.id !== 'inputs' && !run)
          return <button key={stage.id} className={view === stage.id ? 'is-active' : ''} aria-current={view === stage.id ? 'step' : undefined} disabled={disabled} onClick={() => navigate(stage.id)}><span className="fa-stage-rail__index">{index + 1}</span>{stage.icon}<span><b>{stage.label}</b><small>{stage.short}</small></span></button>
        })}
      </nav>

      <div className="fa-context-strip">
        <span><b>TEMSİL UZAYI</b>{activeSpaceId ? shortId(activeSpaceId) : '—'}</span>
        <span><b>ANALİZ</b>{selectedRunIds.size}</span>
        <span><b>ONAYLI SENTEZ</b>{selectedDecisionIds.size}</span>
        <span><b>PROFİL</b>{run?.profiles.length ?? 0}</span>
        <span><b>AİLE</b>{run?.families.length ?? 0}</span>
        <span><b>ÜYELİK</b>{run?.memberships.length ?? 0}</span>
        <span><b>YEREL MODEL</b>{familyProjects.length}</span>
        <span className="fa-context-strip__last"><b>YAZMA SINIRI</b>Yukarı akış salt okunur</span>
      </div>

      {sourceWarnings.map((warning) => <div className="fa-global-message" key={warning}><div className="fa-warning"><AlertTriangle size={14} /><span>{warning}</span></div></div>)}
      {archiveNotice && <div className="fa-global-message"><div className={archiveNotice.includes('kaydedildi') ? 'fa-good-note' : 'fa-warning'}><Archive size={14} /><span>{archiveNotice}</span></div></div>}
      {run?.warnings.map((warning) => view !== 'validation' && <div className="fa-global-message" key={warning}><div className="fa-warning"><Info size={14} /><span>{warning}</span></div></div>)}

      {view === 'inputs' && <InputView projects={projects} decisions={decisions} activeSpaceId={activeSpaceId} selectedRunIds={selectedRunIds} selectedDecisionIds={selectedDecisionIds} config={config} runState={runState} progress={progress} error={error} onSpace={selectSpace} onToggleAnalysis={(runId) => setSelectedRunIds((current) => { const next = new Set(current); if (next.has(runId)) next.delete(runId); else next.add(runId); return next })} onToggleDecision={(id) => setSelectedDecisionIds((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next })} onSelectAll={() => { setSelectedRunIds(new Set(compatibleAnalyses.map((analysis) => analysis.runId))); setSelectedDecisionIds(new Set(compatibleDecisions.map((decision) => decision.id))) }} onConfig={setConfig} onRefresh={() => void refresh()} onRun={() => runExperiment(undefined, 'profiles')} onCancel={cancelRun} />}
      {view === 'profiles' && run && <ProfilesView run={run} selectedKey={selectedProfileKey} comparisonKey={comparisonProfileKey} filter={profileFilter} onFilter={setProfileFilter} onSelect={setSelectedProfileKey} onComparison={setComparisonProfileKey} />}
      {view === 'families' && run && <FamiliesView run={run} selectedFamilyId={selectedFamilyId} filter={familyFilter} viewMode={familyViewMode} onFilter={setFamilyFilter} onViewMode={setFamilyViewMode} onSelectFamily={setSelectedFamilyId} onSelectProfile={(key) => { setSelectedProfileKey(key); setView('profiles') }} />}
      {view === 'flow' && run && <FlowView run={run} prompt={flowPrompt} runState={runState} progress={progress} error={error} onPrompt={setFlowPrompt} onRun={(activations) => runExperiment(activations, 'flow')} onCancel={cancelRun} onFamily={(id) => { setSelectedFamilyId(id); setView('families') }} />}
      {view === 'validation' && run && <ValidationView run={run} selectedFamilyId={selectedFamilyId} onFamily={setSelectedFamilyId} />}
    </div>
  )
}
