import {
  Activity,
  AlertTriangle,
  Archive,
  BarChart3,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Clock,
  Database,
  FileText,
  Gauge,
  Info,
  ListChecks,
  Loader2,
  Layers3,
  Network,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Server,
  Settings2,
  ShieldCheck,
  Search,
  Sparkles,
  Upload,
  Wifi,
  WifiOff,
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
import { SuiteMenu } from '../components/SuiteMenu'
import { loadAllProjects } from '../text-lab/persistence/indexedDb'
import { DEFAULT_TRAINING_SERVICE_URL, TrainingApi, TrainingApiError } from './api'
import { canStartRun, clampRatio, formatBytes, formatDuration, isRunBusy, metricTrend, PIPELINE_ORDER } from './state'
import { isStructuralGateEarlyStop, isStructuralGatePassed, structuralGateStatusLabel } from './structural-gate'
import type {
  DashboardData,
  MetricValue,
  ModelSnapshot,
  PipelineStage,
  PipelineStageId,
  QuarantineItem,
  QueueItem,
  RunState,
  TeacherGenerateRequest,
  TeacherFeedCampaign,
  TeacherFeedCampaignRequest,
  TrainingConfig,
  TrainingEvent,
  FamilyInspection,
  InspectedFamily,
  TrainingMetrics,
  TrainingSource,
} from './types'

type ViewId = 'operation' | 'quarantine' | 'activity' | 'families' | 'models'
type BusyAction = 'refresh' | 'start' | 'pause' | 'resume' | 'cancel' | 'config' | 'source' | 'teacher' | 'campaign' | 'sync' | 'review' | 'activate' | 'rollback' | null

const EMPTY_CONFIG: TrainingConfig = {
  ollamaBaseUrl: 'http://127.0.0.1:11434',
  model: '',
  automationLevel: 'proposal',
  autoRun: false,
  batchSize: 64,
  budgets: { cpuPercent: 60, ramMb: 4096, maxRunMinutes: 120, diskMb: 4096 },
}

const DEFAULT_FEED_CAMPAIGN: TeacherFeedCampaignRequest = {
  model: '',
  seed: 20260829,
  targetAccepted: 128,
  generationBatchSize: 2,
  trainEveryAccepted: 32,
  curriculum: 'turkish-foundations-v1',
  mode: 'research-only',
  generationMode: 'hybrid',
  continuousPlan: {
    blockAccepted: 128,
    maxBlocks: 32,
    minimumDiskHeadroomMb: 512,
    maximumSyntheticSources: 100_000,
  },
}

const STAGE_LABELS: Record<PipelineStageId, { title: string; description: string }> = {
  collect: { title: 'Topla', description: 'Seçili gerçek ve öğretmen kaynaklarını al' },
  quarantine: { title: 'Karantina', description: 'Ham veriyi üretim sisteminden ayır' },
  validate: { title: 'Doğrula', description: 'Şema, tekrar, kaynak ve boyut denetimi' },
  encode: { title: 'Kodla', description: 'Metni sürümlü nöron girdisine dönüştür' },
  synthesize: { title: 'Sentezle', description: 'Yeni nöron adaylarını güvenli alanda üret' },
  families: { title: 'Aileleştir', description: 'Davranış ve yumuşak aidiyetleri güncelle' },
  language_experience: { title: 'Dil deneyimi', description: 'Sonraki parçayı tahmin edip gerçekle karşılaştır' },
  plastic_regions: { title: 'Bölge gözlemi', description: '8/56 çalışma alanı için sınırlı atlas kanıtı topla' },
  concept_macros: { title: 'Kavramlaştır', description: 'Tekrarlanan yolları geri açılabilir makro adaylarına dönüştür' },
  evaluate: { title: 'Değerlendir', description: 'Görülmemiş test verisinde adayı ölç' },
  commit: { title: 'Kabul et', description: 'Yalnız daha iyi adayı yeni sürüm yap' },
  rollback: { title: 'Geri al', description: 'İstenirse doğrulanmış eski bir sürümden yeni geri dönüş kaydı oluştur' },
}

const VIEW_LABELS: Array<{ id: ViewId; label: string; icon: ReactNode }> = [
  { id: 'operation', label: 'Eğitim merkezi', icon: <BrainCircuit size={15} /> },
  { id: 'quarantine', label: 'Karantina', icon: <ShieldCheck size={15} /> },
  { id: 'activity', label: 'Kuyruk ve olaylar', icon: <Activity size={15} /> },
  { id: 'families', label: 'Aile denetimi', icon: <Network size={15} /> },
  { id: 'models', label: 'Ölçümler ve sürümler', icon: <BarChart3 size={15} /> },
]

function Button({
  children,
  icon,
  tone = 'default',
  compact = false,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode
  tone?: 'default' | 'primary' | 'danger' | 'success' | 'ghost'
  compact?: boolean
}) {
  return <button className={`tr-button tr-button--${tone} ${compact ? 'tr-button--compact' : ''}`} {...props}>{icon}{children}</button>
}

function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? '—' : date.toLocaleString('tr-TR', { dateStyle: 'short', timeStyle: 'short' })
}

function formatCount(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('tr-TR') : '—'
}

function formatMetric(metric: MetricValue): string {
  if (metric.current === null || !Number.isFinite(metric.current)) return 'Henüz ölçülmedi'
  if (metric.unit === 'ratio') return `%${Math.round(metric.current * 100)}`
  if (metric.unit === 'milliseconds') return `${Math.round(metric.current)} ms`
  return formatCount(metric.current)
}

function runStateLabel(state: RunState): string {
  return ({
    offline: 'Servis çevrimdışı', idle: 'Çalışmaya hazır', starting: 'Başlatılıyor', running: 'Eğitim çalışıyor',
    pausing: 'Duraklatılıyor', paused: 'Eğitim duraklatıldı', cancelling: 'İptal ediliyor', failed: 'İşlem hatası',
  } satisfies Record<RunState, string>)[state]
}

function sourceKindLabel(kind: TrainingSource['kind']): string {
  return kind === 'pasted_text' ? 'Gerçek metin' : kind === 'file' ? 'Yerel dosya' : 'Öğretmen üretimi'
}

function eventLevelLabel(level: TrainingEvent['level']): string {
  return ({ debug: 'AYRINTI', info: 'BİLGİ', success: 'BAŞARILI', warning: 'UYARI', error: 'HATA' })[level]
}

function stageStateLabel(state: PipelineStage['state']): string {
  return ({ pending: 'Bekliyor', active: 'Çalışıyor', complete: 'Tamamlandı', failed: 'Hata', skipped: 'Atlandı', rolled_back: 'Geri alındı' })[state]
}

function campaignStatusLabel(status: TeacherFeedCampaign['status']): string {
  return ({
    queued: 'Kuyrukta',
    running: 'Üretiyor',
    pausing: 'Duraklatılıyor',
    paused: 'Duraklatıldı',
    completed: 'Tamamlandı',
    cancelling: 'İptal ediliyor',
    cancelled: 'İptal edildi',
    failed: 'Hata',
  } satisfies Record<TeacherFeedCampaign['status'], string>)[status]
}

function isFeedCampaignActive(status: TeacherFeedCampaign['status']): boolean {
  return status === 'queued' || status === 'running' || status === 'pausing' || status === 'paused' || status === 'cancelling'
}

function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return <div className="tr-empty">{icon}<strong>{title}</strong><p>{children}</p></div>
}

function LoadingState({ label = 'Yerel servis okunuyor…' }: { label?: string }) {
  return <div className="tr-loading" role="status"><Loader2 size={19} /><span>{label}</span></div>
}

function Pipeline({ stages }: { stages: PipelineStage[] }) {
  const byId = new Map(stages.map((stage) => [stage.id, stage]))
  return (
    <section className="tr-pipeline" aria-labelledby="pipeline-title">
      <div className="tr-section-heading">
        <div><span className="tr-eyebrow">GÜVENLİ EĞİTİM HATTI</span><h2 id="pipeline-title">Bir parti nasıl ilerliyor?</h2></div>
        <small>Her aşama kaydedilir · kabul ve geri alma ayrıdır</small>
      </div>
      <ol>
        {PIPELINE_ORDER.map((id, index) => {
          const stage = byId.get(id) ?? { id, state: 'pending' as const, progress: 0 }
          const definition = STAGE_LABELS[id]
          return (
            <li key={id} className={`is-${stage.state}`} aria-current={stage.state === 'active' ? 'step' : undefined}>
              <div className="tr-pipeline__number">{stage.state === 'complete' ? <Check size={13} /> : index + 1}</div>
              <div><strong>{definition.title}</strong><span>{definition.description}</span></div>
              <small>{stageStateLabel(stage.state)}{stage.state === 'active' ? ` · %${Math.round(clampRatio(stage.progress) * 100)}` : ''}</small>
              {index < PIPELINE_ORDER.length - 1 && <ChevronRight className="tr-pipeline__arrow" size={15} aria-hidden="true" />}
            </li>
          )
        })}
      </ol>
    </section>
  )
}

function PlasticRegionReadOnlyCard({ config }: { config?: TrainingConfig['plasticRegions'] }) {
  const mode = config?.enabled === false ? 'Politikayla kapalı' : ({
    off: 'Kapalı',
    shadow: 'Gölge gözlem ayarlı',
    proposal: 'Öneri ayarlı',
    active: 'Etkin kip ayarlı',
  } as const)[config?.mode ?? 'off']
  const adaptive = config?.representationProfile === 'adaptive-eighth-id-v1'
  return (
    <section className="tr-region-status" aria-labelledby="region-status-title">
      <Layers3 size={20} />
      <div>
        <span className="tr-eyebrow">YENİ EŞLİKÇİ MEKANİZMA · SALT OKUNUR</span>
        <h2 id="region-status-title">Uyarlanabilir 8/56 bölge atlası</h2>
        <p>Bu merkez yalnız ayarı gösterir; buradan çalışma kipi değiştirilemez. Gerçek uygunluk, doğrulanmış release zinciri ve etkili kapanma nedeni atlas ekranında denetlenir.</p>
      </div>
      <dl>
        <div><dt>Ayarlı kip</dt><dd>{mode}</dd></div>
        <div><dt>Temsil</dt><dd>{adaptive ? '8 hücre satırı kimlik · 56 sayfa ilişki' : 'Eski temsil · yeniden kodlama gerekir'}</dd></div>
      </dl>
      <a href="/bolge-atlasi/">Bölge atlası denetimini aç <ChevronRight size={14} /></a>
    </section>
  )
}

function MetricCard({ label, metric, beneficial = 'up', help }: { label: string; metric: MetricValue; beneficial?: 'up' | 'down'; help: string }) {
  const trend = metricTrend(metric)
  const good = trend !== 'unknown' && trend !== 'same' && trend === beneficial
  return (
    <article className="tr-metric-card" title={help}>
      <span>{label}</span>
      <strong>{formatMetric(metric)}</strong>
      <small className={trend === 'unknown' || trend === 'same' ? '' : good ? 'is-good' : 'is-bad'}>
        {trend === 'unknown' ? 'İlk ölçüm bekleniyor' : trend === 'same' ? 'Önceki sürümle aynı' : `${good ? 'İyileşti' : 'Geriledi'} · önce ${metric.previous === null ? '—' : formatMetric({ ...metric, current: metric.previous })}`}
      </small>
    </article>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="tr-field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>
}

export function TrainingApp() {
  const api = useMemo(() => new TrainingApi(DEFAULT_TRAINING_SERVICE_URL), [])
  const [view, setView] = useState<ViewId>('operation')
  const [data, setData] = useState<DashboardData | null>(null)
  const [config, setConfig] = useState<TrainingConfig>(EMPTY_CONFIG)
  const [initialLoading, setInitialLoading] = useState(true)
  const [busyAction, setBusyAction] = useState<BusyAction>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [notice, setNotice] = useState<{ tone: 'success' | 'warning' | 'error'; message: string } | null>(null)
  const [events, setEvents] = useState<TrainingEvent[]>([])
  const [eventConnected, setEventConnected] = useState(false)
  const [textName, setTextName] = useState('Yeni eğitim metni')
  const [textProvenance, setTextProvenance] = useState('Kullanıcı tarafından sağlandı')
  const [textValue, setTextValue] = useState('')
  const [teacher, setTeacher] = useState<TeacherGenerateRequest>({ topic: '', objective: '', count: 40, language: 'tr', diversity: 'balanced', model: '' })
  const [feedDraft, setFeedDraft] = useState<TeacherFeedCampaignRequest>(DEFAULT_FEED_CAMPAIGN)
  const [feedCampaigns, setFeedCampaigns] = useState<TeacherFeedCampaign[]>([])
  const [feedCampaignLoading, setFeedCampaignLoading] = useState(true)
  const [feedCampaignError, setFeedCampaignError] = useState<string | null>(null)
  const [reviewReasons, setReviewReasons] = useState<Record<string, string>>({})
  const [rollbackTarget, setRollbackTarget] = useState<ModelSnapshot | null>(null)
  const [rollbackReason, setRollbackReason] = useState('Son sağlam sürüme kontrollü dönüş')
  const [cancelConfirm, setCancelConfirm] = useState(false)
  const [configDirty, setConfigDirty] = useState(false)
  const [familyInspection, setFamilyInspection] = useState<FamilyInspection | null>(null)
  const [familyLoading, setFamilyLoading] = useState(false)
  const [familyError, setFamilyError] = useState<string | null>(null)
  const [familyPageOffset, setFamilyPageOffset] = useState(0)
  const [familySnapshotId, setFamilySnapshotId] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mountedRef = useRef(true)
  const actionLockRef = useRef<Exclude<BusyAction, null> | null>(null)

  const refresh = useCallback(async (quiet = false, signal?: AbortSignal) => {
    if (!quiet) setBusyAction('refresh')
    try {
      const dashboard = await api.getDashboard(signal)
      if (!mountedRef.current) return
      setData(dashboard)
      setConnectionError(null)
      if (!configDirty) {
        setConfig(dashboard.status.config)
        setTeacher((current) => ({ ...current, model: current.model || dashboard.status.config.model }))
        setFeedDraft((current) => ({ ...current, model: current.model || dashboard.status.config.model }))
      }
    } catch (error) {
      if (error instanceof TrainingApiError && error.code === 'ABORTED') return
      if (!mountedRef.current) return
      setConnectionError(error instanceof Error ? error.message : 'Yerel eğitim servisine bağlanılamadı.')
    } finally {
      if (mountedRef.current) {
        setInitialLoading(false)
        if (!quiet) setBusyAction(null)
      }
    }
  }, [api, configDirty])

  const refreshFeedCampaigns = useCallback(async (quiet = false, signal?: AbortSignal) => {
    if (!quiet) setFeedCampaignLoading(true)
    try {
      const campaigns = await api.getTeacherFeedCampaigns(signal)
      if (!mountedRef.current) return
      setFeedCampaigns(campaigns)
      setFeedCampaignError(null)
    } catch (error) {
      if (error instanceof TrainingApiError && error.code === 'ABORTED') return
      if (!mountedRef.current) return
      setFeedCampaignError(error instanceof Error ? error.message : 'Besleme kampanyaları okunamadı.')
    } finally {
      if (mountedRef.current && !quiet) setFeedCampaignLoading(false)
    }
  }, [api])

  const refreshFamilies = useCallback(async (offset: number, quiet = false, signal?: AbortSignal) => {
    if (!quiet) setFamilyLoading(true)
    try {
      const inspection = await api.getFamilyInspection({ snapshotId: familySnapshotId || undefined, offset, limit: 50, memberLimit: 8, transitionLimit: 8 }, signal)
      if (!mountedRef.current) return
      setFamilyInspection(inspection)
      setFamilyError(null)
    } catch (error) {
      if (error instanceof TrainingApiError && error.code === 'ABORTED') return
      if (!mountedRef.current) return
      setFamilyError(error instanceof Error ? error.message : 'Aile denetimi okunamadı.')
    } finally {
      if (mountedRef.current && !quiet) setFamilyLoading(false)
    }
  }, [api, familySnapshotId])

  const liveRefreshRef = useRef({ refresh, refreshFeedCampaigns, refreshFamilies, view, familyPageOffset })
  useEffect(() => {
    liveRefreshRef.current = { refresh, refreshFeedCampaigns, refreshFamilies, view, familyPageOffset }
  }, [familyPageOffset, refresh, refreshFamilies, refreshFeedCampaigns, view])

  useEffect(() => {
    mountedRef.current = true
    const controller = new AbortController()
    void refresh(false, controller.signal)
    // SSE drives normal updates. This slow fallback covers a suspended browser,
    // an event-stream reconnect gap, or a legacy service that emits no events.
    const poll = window.setInterval(() => void refresh(true), 60_000)
    return () => {
      mountedRef.current = false
      controller.abort()
      window.clearInterval(poll)
    }
  }, [refresh])

  useEffect(() => {
    const controller = new AbortController()
    void refreshFeedCampaigns(false, controller.signal)
    const poll = window.setInterval(() => void refreshFeedCampaigns(true), 30_000)
    return () => {
      controller.abort()
      window.clearInterval(poll)
    }
  }, [refreshFeedCampaigns])

  useEffect(() => {
    if (view !== 'families') return
    const controller = new AbortController()
    void refreshFamilies(familyPageOffset, false, controller.signal)
    const poll = window.setInterval(() => void refreshFamilies(familyPageOffset, true), 60_000)
    return () => {
      controller.abort()
      window.clearInterval(poll)
    }
  }, [familyPageOffset, refreshFamilies, view])

  useEffect(() => {
    const controller = new AbortController()
    let dashboardTimer: number | null = null
    let campaignTimer: number | null = null
    let familyTimer: number | null = null
    const debounce = (timer: number | null, callback: () => void, delay: number) => {
      if (timer !== null) window.clearTimeout(timer)
      return window.setTimeout(callback, delay)
    }
    const stop = api.subscribeToEvents({
      onEvent: (event) => {
        setEvents((current) => [event, ...current.filter((item) => item.id !== event.id)].slice(0, 400))
        dashboardTimer = debounce(dashboardTimer, () => {
          dashboardTimer = null
          void liveRefreshRef.current.refresh(true)
        }, 400)
        const type = event.type ?? ''
        if (!type || type.startsWith('teacher-feed-') || type.startsWith('job-')) {
          campaignTimer = debounce(campaignTimer, () => {
            campaignTimer = null
            void liveRefreshRef.current.refreshFeedCampaigns(true)
          }, 400)
        }
        if (
          liveRefreshRef.current.view === 'families' &&
          (!type || ['job-succeeded', 'snapshot-activated', 'snapshot-rolled-back', 'candidate-quarantined', 'teacher-feed-training-cycle-completed'].includes(type))
        ) {
          familyTimer = debounce(familyTimer, () => {
            familyTimer = null
            void liveRefreshRef.current.refreshFamilies(liveRefreshRef.current.familyPageOffset, true)
          }, 750)
        }
      },
      onConnectionChange: (connected) => {
        setEventConnected(connected)
        if (connected) {
          dashboardTimer = debounce(dashboardTimer, () => {
            dashboardTimer = null
            void liveRefreshRef.current.refresh(true)
          }, 100)
        }
      },
    }, controller.signal)
    return () => {
      controller.abort()
      stop()
      if (dashboardTimer !== null) window.clearTimeout(dashboardTimer)
      if (campaignTimer !== null) window.clearTimeout(campaignTimer)
      if (familyTimer !== null) window.clearTimeout(familyTimer)
    }
  }, [api])

  useEffect(() => {
    const refreshVisible = () => {
      if (document.visibilityState !== 'visible') return
      void liveRefreshRef.current.refresh(true)
      void liveRefreshRef.current.refreshFeedCampaigns(true)
      if (liveRefreshRef.current.view === 'families') {
        void liveRefreshRef.current.refreshFamilies(liveRefreshRef.current.familyPageOffset, true)
      }
    }
    document.addEventListener('visibilitychange', refreshVisible)
    return () => document.removeEventListener('visibilitychange', refreshVisible)
  }, [])

  useEffect(() => {
    const closeModal = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setCancelConfirm(false)
      setRollbackTarget(null)
    }
    document.addEventListener('keydown', closeModal)
    return () => document.removeEventListener('keydown', closeModal)
  }, [])

  const status = data?.status
  const runState = status?.state ?? 'offline'
  const selectedSources = data?.sources.filter((source) => source.selected && source.state !== 'disabled') ?? []
  const selectedRealSources = selectedSources.filter((source) => source.kind !== 'ollama_generated').length
  const selectedSyntheticSources = selectedSources.filter((source) => source.kind === 'ollama_generated').length
  const selectedModelAvailable = Boolean(config.model && data?.models.some((model) => model.name === config.model))
  const selectedFeedModelAvailable = Boolean(feedDraft.model && data?.models.some((model) => model.name === feedDraft.model))
  const activeFeedCampaign = feedCampaigns.find((campaign) => isFeedCampaignActive(campaign.status)) ?? null
  const displayedFeedCampaign = activeFeedCampaign ?? feedCampaigns[0] ?? null
  const displayedStopGate = displayedFeedCampaign?.continuous?.stopGate ?? null
  const displayedStopGateLabel = displayedStopGate ? structuralGateStatusLabel(displayedStopGate) : null
  const displayedStopGatePassed = displayedStopGate ? isStructuralGatePassed(displayedStopGate) : false
  const displayedStopGateEarlyStopped = displayedStopGate ? isStructuralGateEarlyStop(displayedStopGate.status) : false
  const running = isRunBusy(runState)
  const startEnabled = !connectionError && canStartRun(runState, selectedSources.length) && busyAction === null
  const feedDraftValid = Number.isInteger(feedDraft.seed) && feedDraft.seed >= 0 && feedDraft.seed <= 0xffff_ffff
    && Number.isInteger(feedDraft.targetAccepted) && feedDraft.targetAccepted >= 16 && feedDraft.targetAccepted <= 10_000
    && Number.isInteger(feedDraft.generationBatchSize) && feedDraft.generationBatchSize >= 1 && feedDraft.generationBatchSize <= 2
    && Number.isInteger(feedDraft.trainEveryAccepted) && feedDraft.trainEveryAccepted >= 8 && feedDraft.trainEveryAccepted <= 512
    && feedDraft.trainEveryAccepted <= feedDraft.targetAccepted
    && Boolean(feedDraft.continuousPlan)
    && feedDraft.continuousPlan!.blockAccepted === feedDraft.targetAccepted
    && Number.isInteger(feedDraft.continuousPlan!.maxBlocks) && feedDraft.continuousPlan!.maxBlocks >= 1 && feedDraft.continuousPlan!.maxBlocks <= 100
    && feedDraft.targetAccepted * feedDraft.continuousPlan!.maxBlocks <= 100_000

  const performAction = async (kind: Exclude<BusyAction, null>, operation: () => Promise<{ message?: string }>, successMessage: string) => {
    if (actionLockRef.current) return
    actionLockRef.current = kind
    setBusyAction(kind)
    setNotice(null)
    try {
      const result = await operation()
      setNotice({ tone: 'success', message: result.message || successMessage })
      await refresh(true)
    } catch (error) {
      setNotice({ tone: 'error', message: error instanceof Error ? error.message : 'İşlem tamamlanamadı.' })
    } finally {
      actionLockRef.current = null
      setBusyAction(null)
    }
  }

  const startRun = () => performAction('start', () => api.startRun({
    automationLevel: config.automationLevel,
    model: config.model,
    sourceIds: selectedSources.map((source) => source.id),
    config,
  }), 'Eğitim partisi güvenli hatta başlatıldı.')

  const saveConfig = () => performAction('config', async () => {
    await api.updateConfig(config)
    setConfigDirty(false)
    return { message: 'Model, otomasyon ve kaynak bütçeleri kaydedildi.' }
  }, 'Ayarlar kaydedildi.')

  const addTextSource = () => {
    if (!textValue.trim()) return
    void performAction('source', async () => {
      await api.addTextSource({ name: textName.trim() || 'Adsız eğitim metni', text: textValue, provenance: textProvenance.trim() || 'Kullanıcı tarafından sağlandı' })
      setTextValue('')
      return { message: 'Metin doğrudan eğitime değil, karantinaya gönderildi.' }
    }, 'Metin karantinaya alındı.')
  }

  const uploadFiles = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])
    event.target.value = ''
    if (!files.length) return
    void performAction('source', async () => {
      let uploadedCount = 0
      for (let offset = 0; offset < files.length; offset += 8) {
        try {
          const uploaded = await api.addFileSources(files.slice(offset, offset + 8), 'Kullanıcının yerel dosya seçicisi')
          uploadedCount += uploaded.length
        } catch (error) {
          const reason = error instanceof Error ? error.message : 'Bilinmeyen aktarım hatası.'
          await refresh(true)
          throw new Error(`${uploadedCount.toLocaleString('tr-TR')} dosya aktarıldı; sıradaki grup reddedildi. ${reason}`)
        }
      }
      return { message: `${uploadedCount.toLocaleString('tr-TR')} dosya karantinaya gönderildi.` }
    }, 'Dosyalar karantinaya alındı.')
  }

  const syncTextLab = () => void performAction('sync', async () => {
    const projects = (await loadAllProjects()).filter((project) => project.analysis && project.source.rawText.trim())
    if (!projects.length) return { message: 'Aktarılabilecek analizli Metin Laboratuvarı kaydı bulunamadı.' }
    const uniqueProjects = Array.from(new Map(projects.map((project) => [
      JSON.stringify([
        project.source.rawText.replace(/\r\n?/g, '\n').normalize('NFC'),
        Object.fromEntries(Object.entries(project.analysis!.config).sort(([left], [right]) => left.localeCompare(right))),
      ]),
      project,
    ])).values())
    const skippedExactDuplicates = projects.length - uniqueProjects.length
    const results: PromiseSettledResult<TrainingSource>[] = []
    for (let offset = 0; offset < uniqueProjects.length; offset += 8) {
      const batch = uniqueProjects.slice(offset, offset + 8)
      results.push(...await Promise.allSettled(batch.map((project) => api.addTextSource({
        name: project.source.title || 'Adsız Metin Laboratuvarı kaydı',
        text: project.source.rawText,
        provenance: `Metin Laboratuvarı · kaynak ${project.source.id}`,
        analysisManifest: {
          format: 'text-lab-recompute-proof-v1',
          originalSourceId: project.source.id,
          sourceDigest: project.analysis!.sourceDigest,
          representationSpaceId: project.analysis!.representationSpaceId,
          config: { ...project.analysis!.config },
          identityProofs: project.analysis!.nodes.map((node) => ({
            start: node.start,
            end: node.end,
            identityDigest: node.identityDigest,
          })),
        },
      }))))
    }
    const fulfilled = results.filter((result): result is PromiseFulfilledResult<TrainingSource> => result.status === 'fulfilled')
    const succeeded = fulfilled.length
    const quarantined = fulfilled.filter((result) => result.value.state === 'quarantined').length
    const alreadyReviewed = succeeded - quarantined
    const failures = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected')
    const failed = failures.length
    const duplicateNote = skippedExactDuplicates ? ` ${skippedExactDuplicates} birebir aynı yerel metin tekrar gönderilmedi.` : ''
    const firstFailure = failures[0]?.reason instanceof Error ? ` İlk ret: ${failures[0].reason.message}` : ''
    if (failed) {
      await refresh(true)
      throw new Error(`${succeeded} kayıt aktarıldı; ${failed} kayıt güvenlik veya tutarlılık denetiminden geçemedi.${firstFailure}${duplicateNote}`)
    }
    const reviewNote = quarantined
      ? ` ${quarantined} kayıt karantinada yeniden insan onayı bekliyor.`
      : ''
    const unchangedNote = alreadyReviewed
      ? ` ${alreadyReviewed} uyumlu kayıt daha önce karara bağlandığı için durumu değiştirilmedi.`
      : ''
    return { message: `${succeeded} analizli kaynak servisle eşleştirildi.${reviewNote}${unchangedNote} I/R matrisleri taşınmadı.${duplicateNote}` }
  }, 'Metin Laboratuvarı arşivi aktarıldı.')

  const generateTeacherData = () => {
    if (!teacher.topic.trim() || !teacher.objective.trim() || !config.model) return
    void performAction('teacher', async () => {
      const result = await api.generateTeacherData({ ...teacher, model: config.model, language: 'tr' })
      return { message: result.message || `${result.generatedCount} öğretmen örneği doğrudan karantinaya gönderildi.` }
    }, 'Öğretmen örnekleri üretildi.')
  }

  const performFeedCampaignAction = async (
    operation: () => Promise<TeacherFeedCampaign>,
    successMessage: string,
  ) => {
    if (actionLockRef.current) return
    actionLockRef.current = 'campaign'
    setBusyAction('campaign')
    setNotice(null)
    try {
      const campaign = await operation()
      setFeedCampaigns((current) => [campaign, ...current.filter((item) => item.id !== campaign.id)])
      setNotice({ tone: 'success', message: successMessage })
      await refreshFeedCampaigns(true)
    } catch (error) {
      setNotice({ tone: 'error', message: error instanceof Error ? error.message : 'Besleme kampanyası işlemi tamamlanamadı.' })
    } finally {
      actionLockRef.current = null
      setBusyAction(null)
    }
  }

  const startFeedCampaign = () => {
    if (!feedDraft.model || !feedDraftValid || activeFeedCampaign) return
    void performFeedCampaignAction(
      () => api.createTeacherFeedCampaign(feedDraft),
      '14B eğitimi güvenli, yeniden başlatılabilir bloklar halinde araştırma kipinde başlatıldı.',
    )
  }

  const reviewItem = (item: QuarantineItem, decision: 'approve' | 'reject') => void performAction('review', async () => {
    await api.reviewQuarantine(item.id, decision, { reason: reviewReasons[item.id]?.trim() || undefined })
    return { message: decision === 'approve' ? 'Örnek doğrulanmış havuza alındı.' : 'Örnek reddedildi; nöron sistemine ulaşmadı.' }
  }, 'Karantina kararı kaydedildi.')

  const toggleSource = (source: TrainingSource) => void performAction('source', async () => {
    await api.setSourceSelected(source.id, !source.selected)
    return { message: `${source.name} ${source.selected ? 'eğitim seçiminden çıkarıldı' : 'eğitim için seçildi'}.` }
  }, 'Kaynak seçimi güncellendi.')

  const confirmRollback = () => {
    if (!rollbackTarget?.rollbackToken || !rollbackReason.trim()) return
    void performAction('rollback', () => api.rollbackSnapshot(rollbackTarget.id, {
      confirmToken: rollbackTarget.rollbackToken!,
      reason: rollbackReason.trim(),
    }), 'Model güvenli sürüme geri alındı.')
    setRollbackTarget(null)
  }

  const activateCandidate = (snapshot: ModelSnapshot) => void performAction(
    'activate',
    () => api.activateSnapshot(snapshot.id),
    'Doğrulanmış aday etkin model olarak kabul edildi.',
  )

  return (
    <div className="tr-app">
      <header className="tr-topbar">
        <SuiteMenu currentApp="training" navigationLocked={running}>
          <div className="tr-brand">
            <span className="tr-brand__mark"><Sparkles size={20} /></span>
            <span><strong>OTONOM EĞİTİM</strong><small>YEREL ÖĞRETMEN · GÜVENLİ ÖĞRENME</small></span>
          </div>
        </SuiteMenu>
        <div className={`tr-run-status is-${runState}`} role="status"><i />{runStateLabel(runState)}</div>
        <div className="tr-topbar__actions">
          <Button compact aria-label="Verileri yenile" title="Verileri yenile" disabled={busyAction !== null} icon={<RefreshCw size={14} className={busyAction === 'refresh' || familyLoading ? 'is-spinning' : ''} />} onClick={() => { void refresh(); if (view === 'families') void refreshFamilies(familyPageOffset) }} />
          {runState === 'paused' ? (
            <Button tone="primary" icon={<Play size={14} />} disabled={busyAction !== null} onClick={() => void performAction('resume', () => api.resumeRun(), 'Eğitim kaldığı güvenli noktadan devam ediyor.')}>Devam et</Button>
          ) : runState === 'running' || runState === 'starting' ? (
            <Button icon={<Pause size={14} />} disabled={busyAction !== null || runState !== 'running'} onClick={() => void performAction('pause', () => api.pauseRun(), 'Duraklatma güvenli parti sınırında uygulanacak.')}>Duraklat</Button>
          ) : (
            <Button tone="primary" icon={<Play size={14} />} disabled={!startEnabled} onClick={startRun}>Eğitimi başlat</Button>
          )}
          {running && <Button tone="danger" icon={<CircleStop size={14} />} disabled={busyAction !== null} onClick={() => setCancelConfirm(true)}>İptal</Button>}
        </div>
      </header>

      <nav className="tr-view-tabs" aria-label="Otonom eğitim bölümleri">
        {VIEW_LABELS.map((item) => <button key={item.id} className={view === item.id ? 'is-active' : ''} aria-current={view === item.id ? 'page' : undefined} onClick={() => setView(item.id)}>{item.icon}<span>{item.label}</span>{item.id === 'quarantine' && Boolean(data?.quarantine.filter((entry) => entry.state === 'pending').length) && <b>{data!.quarantine.filter((entry) => entry.state === 'pending').length}</b>}</button>)}
      </nav>

      <div className="tr-context-strip">
        <span>{connectionError ? <WifiOff size={12} /> : <Wifi size={12} />}<b>SERVİS</b> {DEFAULT_TRAINING_SERVICE_URL}</span>
        <span><Server size={12} /><b>OLLAMA</b> {status?.ollama.connected ? `${status.ollama.version ?? 'bağlı'} · ${config.model || 'model seçilmedi'}` : 'bağlantı yok'}</span>
        <span><Database size={12} /><b>KAYNAK</b> {selectedSources.length.toLocaleString('tr-TR')} seçili</span>
        <span className="tr-context-strip__last"><Clock size={12} /><b>SON GÜNCELLEME</b> {formatDate(status?.updatedAt)}</span>
      </div>

      <main className="tr-main">
        {connectionError && (
          <section className="tr-connection-error" role="alert">
            <WifiOff size={25} /><div><strong>Otonom eğitim servisi çalışmıyor</strong><p>{connectionError}</p><small>Bu ekran sahte veri üretmez. Yerel servis bağlanana kadar eğitim, kuyruk ve model kayıtları gösterilemez.</small></div>
            <Button icon={<RefreshCw size={14} />} disabled={busyAction !== null} onClick={() => void refresh()}>Yeniden dene</Button>
          </section>
        )}

        {notice && <div className={`tr-notice is-${notice.tone}`} role="status">{notice.tone === 'success' ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}<span>{notice.message}</span><button aria-label="Bildirimi kapat" onClick={() => setNotice(null)}><X size={14} /></button></div>}

        {status?.warnings.map((warning) => <div key={warning.id} className={`tr-warning is-${warning.severity}`} role={warning.severity === 'critical' ? 'alert' : 'status'}><AlertTriangle size={17} /><div><strong>{warning.title}</strong><p>{warning.message}</p></div><time>{formatDate(warning.createdAt)}</time></div>)}

        {initialLoading && !data ? <LoadingState /> : view === 'operation' ? (
          <>
            <section className="tr-hero-grid">
              <article className="tr-panel tr-panel--hero">
                <div className="tr-eyebrow">KONTROLLÜ OTONOMİ</div>
                <h1>Model deney üretir; sistemimiz yaşayarak öğrenir.</h1>
                <p>Ollama yalnızca yerel öğretmen ve veri hazırlayıcıdır. Ürettiği hiçbir metin, etiket veya karar nöronlara doğrudan yazılamaz; önce karantina, doğrulama ve görülmemiş veri sınavından geçer.</p>
                <div className="tr-safety-row"><span><ShieldCheck size={15} /> Doğrudan nöron yazımı kapalı</span><span><Archive size={15} /> Her kabul öncesi sürüm kaydı</span><span><RotateCcw size={15} /> Başarısız aday etkin sürüme dokunmaz</span></div>
              </article>
              <article className="tr-panel tr-run-card">
                <div className="tr-section-heading"><div><span className="tr-eyebrow">AKTİF OTURUM</span><h2>{status?.runId ? `Çalışma ${status.runId.slice(0, 12)}` : 'Henüz eğitim başlamadı'}</h2></div><span className={`tr-state-badge is-${runState}`}>{runStateLabel(runState)}</span></div>
                <div className="tr-big-progress"><div><strong>%{Math.round(clampRatio(status?.progress ?? 0) * 100)}</strong><span>{status?.activeStage ? STAGE_LABELS[status.activeStage].title : 'Bekliyor'}</span></div><progress max="1" value={clampRatio(status?.progress ?? 0)} aria-label="Genel eğitim ilerlemesi" /></div>
                <dl><div><dt>İşlenen</dt><dd>{formatCount(status?.processedItems)}</dd></div><div><dt>Kabul</dt><dd>{formatCount(status?.acceptedItems)}</dd></div><div><dt>Red</dt><dd>{formatCount(status?.rejectedItems)}</dd></div><div><dt>Kuyruk</dt><dd>{formatCount(status?.queuedItems)}</dd></div></dl>
                <Button tone="primary" icon={<Play size={15} />} disabled={!startEnabled} onClick={startRun}>{selectedSources.length ? `${selectedSources.length} kaynakla eğitimi başlat` : 'Önce kaynak seç'}</Button>
              </article>
            </section>

            <Pipeline stages={status?.stages ?? []} />

            <PlasticRegionReadOnlyCard config={status?.config.plasticRegions} />

            <section className="tr-panel tr-panel--padded tr-feed-campaign" aria-labelledby="feed-campaign-title">
              <div className="tr-section-heading">
                <div>
                  <span className="tr-eyebrow">BLOKLU · YENİDEN OYNATILABİLİR · SENTETİK</span>
                  <h2 id="feed-campaign-title">Güvenli Sürekli 14B Eğitimi</h2>
                  <p>Her {feedDraft.targetAccepted.toLocaleString('tr-TR')} kabulde blok kapanır, aday snapshot üretilir ve bağımsız yapısal sınav çalışır. Sınav geçmezse en fazla {feedDraft.continuousPlan?.maxBlocks ?? 0} sınırlı blok ilerlenir.</p>
                </div>
                <span className="tr-feed-mode"><ShieldCheck size={13} /> Araştırma kipi</span>
              </div>

              <div className="tr-feed-risk" role="note">
                <AlertTriangle size={18} />
                <div>
                  <strong>Bu kampanya doğruluk kanıtı değildir.</strong>
                  <p>{feedDraft.generationMode === 'hybrid' ? 'Hibrit kip kontrollü seçim ile serbest keşfi ayrı partilerde dönüşümlü çalıştırır. Keşif cümleleri sunucu denetiminden geçmeden eğitime girmez.' : 'Kontrollü kipte 14B serbest cümle yazmaz; yalnızca denetlenmiş olay seçeneklerini sıralar.'} Sonuçlar sentetiktir, gerçek kaynak holdout’una giremez, etkin modeli otomatik değiştiremez ve tek başına “düşünce” kanıtlamaz.</p>
                </div>
              </div>

              <div className="tr-feed-layout">
                <div className="tr-feed-settings">
                  <div className="tr-form-grid">
                    <Field label="Öğretmen modeli" hint="Yerel Ollama modeli">
                      <select value={feedDraft.model} disabled={Boolean(activeFeedCampaign)} onChange={(event) => setFeedDraft({ ...feedDraft, model: event.target.value })}>
                        <option value="">Model seçin</option>
                        {feedDraft.model && !selectedFeedModelAvailable && <option value={feedDraft.model} disabled>{feedDraft.model} · bulunamıyor</option>}
                        {data?.models.map((model) => <option key={model.name} value={model.name}>{model.name}{model.parameterSize ? ` · ${model.parameterSize}` : ''}</option>)}
                      </select>
                    </Field>
                    <Field label="Tekrarlanabilir seed" hint="Aynı ayarlar aynı kampanya sırasını üretir">
                      <input type="number" step="1" value={feedDraft.seed} disabled={Boolean(activeFeedCampaign)} onChange={(event) => setFeedDraft({ ...feedDraft, seed: Number(event.target.value) })} />
                    </Field>
                    <Field label="Üretim düzeni" hint="Mevcut kontrollü hat veya ayrı keşif hattıyla hibrit">
                      <select value={feedDraft.generationMode} disabled={Boolean(activeFeedCampaign)} onChange={(event) => setFeedDraft({ ...feedDraft, generationMode: event.target.value as TeacherFeedCampaignRequest['generationMode'] })}>
                        <option value="hybrid">Hibrit · kontrollü + keşif</option>
                        <option value="controlled">Yalnız kontrollü · eski davranış</option>
                      </select>
                    </Field>
                    <Field label="Blok kabul hedefi" hint="Her yapısal sınavdan önce 16–10.000 cümle">
                      <input type="number" min="16" max="10000" step="16" value={feedDraft.targetAccepted} disabled={Boolean(activeFeedCampaign)} onChange={(event) => {
                        const targetAccepted = Number(event.target.value)
                        setFeedDraft({
                          ...feedDraft,
                          targetAccepted,
                          continuousPlan: feedDraft.continuousPlan ? { ...feedDraft.continuousPlan, blockAccepted: targetAccepted } : undefined,
                        })
                      }} />
                    </Field>
                    <Field label="Üretim partisi" hint="14B / 16 GB için canlı ölçümle 2 önerilir">
                      <input type="number" min="1" max="2" step="1" value={feedDraft.generationBatchSize} disabled={Boolean(activeFeedCampaign)} onChange={(event) => setFeedDraft({ ...feedDraft, generationBatchSize: Number(event.target.value) })} />
                    </Field>
                    <Field label="Eğitim çevrimi eşiği" hint="Bu kadar yeni kabulden sonra">
                      <input type="number" min="8" max="512" step="8" value={feedDraft.trainEveryAccepted} disabled={Boolean(activeFeedCampaign)} onChange={(event) => setFeedDraft({ ...feedDraft, trainEveryAccepted: Number(event.target.value) })} />
                    </Field>
                    <Field label="En çok güvenli blok" hint="Kapı geçmezse başarı saymadan duraklatır">
                      <input type="number" min="1" max="100" step="1" value={feedDraft.continuousPlan?.maxBlocks ?? 32} disabled={Boolean(activeFeedCampaign)} onChange={(event) => setFeedDraft({
                        ...feedDraft,
                        continuousPlan: feedDraft.continuousPlan ? { ...feedDraft.continuousPlan, maxBlocks: Number(event.target.value) } : undefined,
                      })} />
                    </Field>
                  </div>
                  <div className="tr-feed-contract">
                    <span><b>Müfredat</b> Türkçe temeller v1</span>
                    <span><b>Üretim hattı</b> {feedDraft.generationMode === 'hybrid' ? 'Kontrollü %50 · keşif %50' : 'Yalnız kontrollü'}</span>
                    <span><b>Yüzey kökeni</b> {feedDraft.generationMode === 'hybrid' ? 'Katalog ve ayrı Ollama keşfi' : 'Doğrulanmış yerel katalog'}</span>
                    <span><b>Kalıcı yetki</b> Yok · yalnız aday</span>
                    <span><b>Durdurma kapısı</b> 144 olay · 5 seed · bağımsız</span>
                    <span><b>Disk rezervi</b> {feedDraft.continuousPlan?.minimumDiskHeadroomMb ?? 0} MiB</span>
                  </div>
                  {feedCampaignError && <div className="tr-inline-warning"><AlertTriangle size={15} /><span>{feedCampaignError} Son sağlam kampanya görünümü korunuyor.</span></div>}
                  <Button
                    tone="primary"
                    icon={busyAction === 'campaign' ? <Loader2 size={14} className="is-spinning" /> : <Play size={14} />}
                    disabled={Boolean(activeFeedCampaign) || busyAction !== null || !status?.ollama.connected || !selectedFeedModelAvailable || !feedDraftValid}
                    onClick={startFeedCampaign}
                  >
                    {activeFeedCampaign ? 'Aktif kampanya tamamlanmayı bekliyor' : 'Güvenli sürekli eğitimi başlat'}
                  </Button>
                  {!feedDraftValid && <small className="tr-feed-validation">Blok, parti, eğitim eşiği, blok sayısı ve seed geçerli olmalı; toplam plan 100.000 kabulü aşamaz.</small>}
                </div>

                <div className="tr-feed-live" aria-live="polite">
                  {feedCampaignLoading && !displayedFeedCampaign ? <LoadingState label="Besleme kampanyaları okunuyor…" /> : !displayedFeedCampaign ? (
                    <div className="tr-feed-empty"><BrainCircuit size={28} /><strong>Henüz besleme kampanyası yok</strong><p>Soldaki güvenli varsayılanlarla bloklu ve durdurulabilir 14B kampanyasını başlatabilirsiniz.</p></div>
                  ) : (
                    <>
                      <header>
                        <div><span className="tr-eyebrow">CANLI KAMPANYA</span><strong>{displayedFeedCampaign.model}</strong><small>{displayedFeedCampaign.id} · seed {displayedFeedCampaign.seed}</small></div>
                        <span className={`tr-feed-status is-${displayedFeedCampaign.status}`}><i />{campaignStatusLabel(displayedFeedCampaign.status)}</span>
                      </header>
                      <div className="tr-feed-progress">
                        <div><strong>{formatCount(displayedFeedCampaign.accepted)} / {formatCount(displayedFeedCampaign.targetAccepted)}</strong><span>{displayedFeedCampaign.continuous ? 'azami plan içinde kabul' : 'kabul edilen cümle'}</span></div>
                        <progress max={Math.max(1, displayedFeedCampaign.targetAccepted)} value={Math.min(displayedFeedCampaign.accepted, displayedFeedCampaign.targetAccepted)} aria-label="Besleme kampanyası ilerlemesi" />
                      </div>
                      <dl className="tr-feed-counters">
                        <div><dt>Üretildi</dt><dd>{formatCount(displayedFeedCampaign.generated)}</dd></div>
                        <div><dt>Kabul</dt><dd>{formatCount(displayedFeedCampaign.accepted)}</dd></div>
                        <div><dt>Tekrar</dt><dd>{formatCount(displayedFeedCampaign.duplicates)}</dd></div>
                        <div><dt>Kalite reddi</dt><dd>{formatCount(displayedFeedCampaign.rejectedQuality)}</dd></div>
                        <div><dt>Model isteği</dt><dd>{formatCount(displayedFeedCampaign.teacherRequests)}</dd></div>
                        <div><dt>Model tokenı</dt><dd>{formatCount(displayedFeedCampaign.teacherTokens)}</dd></div>
                        <div><dt>Eğitim çevrimi</dt><dd>{formatCount(displayedFeedCampaign.completedTrainingCycles)}</dd></div>
                        <div><dt>Sıradaki parti</dt><dd>{formatCount(displayedFeedCampaign.nextBatchIndex)}</dd></div>
                        {displayedFeedCampaign.continuous && <div><dt>Başlangıç korpusu</dt><dd>{formatCount(displayedFeedCampaign.continuous.bootstrapSourceCount)}</dd></div>}
                        {displayedFeedCampaign.continuous && <div><dt>Tamamlanan blok</dt><dd>{formatCount(displayedFeedCampaign.continuous.completedBlocks)} / {formatCount(displayedFeedCampaign.continuous.maximumBlocks)}</dd></div>}
                      </dl>
                      {displayedFeedCampaign.lanes && <div className="tr-feed-contract" aria-label="Hibrit üretim hatları">
                        <span><b>Kontrollü hat</b>{formatCount(displayedFeedCampaign.lanes.controlled.accepted)} kabul · {formatCount(displayedFeedCampaign.lanes.controlled.rejectedQuality)} red</span>
                        <span><b>Keşif hattı</b>{formatCount(displayedFeedCampaign.lanes.discovery.accepted)} kabul · {formatCount(displayedFeedCampaign.lanes.discovery.rejectedQuality)} red</span>
                        <span><b>Keşif yetkisi</b> Eğitim adayı · değerlendirme dışı</span>
                      </div>}
                      <div className="tr-feed-meta">
                        <span><b>Son snapshot</b>{displayedFeedCampaign.lastSnapshotId ?? 'Henüz yok'}</span>
                        <span><b>Son güncelleme</b>{formatDate(displayedFeedCampaign.updatedAt)}</span>
                        {displayedStopGate && <span><b>Yapısal durdurma kapısı</b>{displayedStopGateLabel}</span>}
                      </div>
                      {displayedStopGatePassed && <div className="tr-inline-warning"><CheckCircle2 size={15} /><span>Bağımsız deterministik yapısal benchmark bütün kapıları geçti; kampanya yeni blok satın almadan sona erdi. Aday model yine otomatik etkinleştirilmedi.</span></div>}
                      {displayedStopGateEarlyStopped && <div className="tr-inline-warning"><AlertTriangle size={15} /><span>{displayedStopGateLabel}. Bu kayıt bir benchmark geçişi veya tamamlanmış tam ölçüm değildir.</span></div>}
                      {displayedFeedCampaign.stopReason && <div className="tr-inline-warning"><Info size={15} /><span>{displayedFeedCampaign.stopReason}</span></div>}
                      {isFeedCampaignActive(displayedFeedCampaign.status) && (
                        <div className="tr-inline-actions">
                          {displayedFeedCampaign.status === 'paused' ? (
                            <Button tone="primary" icon={<Play size={14} />} disabled={busyAction !== null} onClick={() => void performFeedCampaignAction(() => api.resumeTeacherFeedCampaign(displayedFeedCampaign.id), 'Besleme kampanyası kaldığı checkpoint’ten devam ediyor.')}>Devam et</Button>
                          ) : (
                            <Button icon={<Pause size={14} />} disabled={busyAction !== null || !['queued', 'running'].includes(displayedFeedCampaign.status)} onClick={() => void performFeedCampaignAction(() => api.pauseTeacherFeedCampaign(displayedFeedCampaign.id), 'Kampanya güvenli parti sınırında duraklatılacak.')}>Duraklat</Button>
                          )}
                          <Button tone="danger" icon={<CircleStop size={14} />} disabled={busyAction !== null || displayedFeedCampaign.status === 'cancelling'} onClick={() => void performFeedCampaignAction(() => api.cancelTeacherFeedCampaign(displayedFeedCampaign.id), 'Kampanya iptali güvenli parti sınırında uygulanacak.')}>İptal et</Button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </section>

            <section className="tr-operation-grid">
              <article className="tr-panel tr-panel--padded">
                <div className="tr-section-heading"><div><span className="tr-eyebrow">YEREL ÖĞRETMEN</span><h2>Ollama bağlantısı</h2></div><span className={`tr-connect-pill ${status?.ollama.connected ? 'is-online' : ''}`}><i />{status?.ollama.connected ? 'Bağlı' : 'Bağlantı yok'}</span></div>
                <div className="tr-form-grid">
                  <Field label="Ollama adresi" hint="Ollama yalnızca eğitim örneği ve karşı örnek üretir."><input value={config.ollamaBaseUrl} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, ollamaBaseUrl: event.target.value }) }} /></Field>
                  <Field label="Öğretmen modeli"><select value={config.model} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, model: event.target.value }); setTeacher({ ...teacher, model: event.target.value }); setFeedDraft({ ...feedDraft, model: event.target.value }) }}><option value="">Model seçin</option>{config.model && !selectedModelAvailable && <option value={config.model} disabled>{config.model} · artık bulunamıyor</option>}{data?.models.map((model) => <option key={model.name} value={model.name}>{model.name}{model.parameterSize ? ` · ${model.parameterSize}` : ''}</option>)}</select></Field>
                </div>
                {!status?.ollama.connected && <div className="tr-inline-warning"><Info size={15} /><span>Ollama kapalıyken yalnız öğretmen üretimi kullanılamaz; gerçek metin ekleme ve gerçek korpus eğitimi çalışmaya devam eder.</span></div>}
                {status?.ollama.connected && config.model && !selectedModelAvailable && <div className="tr-inline-warning"><AlertTriangle size={15} /><span>Kayıtlı öğretmen modeli artık Ollama listesinde yok. Yeni öğretmen örneği üretmek için kullanılabilir bir model seçin; gerçek korpus eğitimi bundan bağımsız çalışır.</span></div>}
              </article>

              <article className="tr-panel tr-panel--padded">
                <div className="tr-section-heading"><div><span className="tr-eyebrow">OTOMASYON SINIRI</span><h2>Karar yetkisi</h2></div><ShieldCheck size={21} /></div>
                <div className="tr-choice-group" role="radiogroup" aria-label="Otomasyon seviyesi">
                  <label className={config.automationLevel === 'proposal' ? 'is-selected' : ''}><input type="radio" name="automation" value="proposal" checked={config.automationLevel === 'proposal'} onChange={() => { setConfigDirty(true); setConfig({ ...config, automationLevel: 'proposal' }) }} /><span><b>Öneri modu</b><small>Sistem aday üretir; insan onayı olmadan kalıcı sürüm oluşturmaz.</small></span></label>
                  <label className={config.automationLevel === 'controlled' ? 'is-selected' : ''}><input type="radio" name="automation" value="controlled" checked={config.automationLevel === 'controlled'} onChange={() => { setConfigDirty(true); setConfig({ ...config, automationLevel: 'controlled' }) }} /><span><b>Kontrollü otomasyon</b><small>Yalnız bütçe ve kalite eşiklerini geçen adayları otomatik kabul eder.</small></span></label>
                </div>
                <label className={`tr-auto-run ${config.autoRun ? 'is-selected' : ''}`}><input type="checkbox" checked={config.autoRun} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, autoRun: event.target.checked }) }} /><span><b>Onaylı parti birikince otomatik sıraya al</b><small>Yeni kaynak sayısı parti büyüklüğüne ulaştığında sınırlı bir eğitim çevrimi başlatır. Karar yetkisi yukarıdaki moddan gelir.</small></span><i aria-hidden="true" /></label>
                <div className="tr-inline-warning"><AlertTriangle size={15} /><span>“Sınırsız” çalışma modu yoktur. Zaman, bellek, disk ve kalite kapıları zorunludur; işlemci yüzdesi bu sürümde işletim sistemi kotası değil, izlenen çalışma politikasıdır.</span></div>
              </article>

              <article className="tr-panel tr-panel--padded tr-budget-panel">
                <div className="tr-section-heading"><div><span className="tr-eyebrow">KAYNAK BÜTÇELERİ</span><h2>Bir çalışma ne kadar büyüyebilir?</h2></div><Gauge size={21} /></div>
                <div className="tr-budget-grid">
                  <Field label="Otomatik başlatma eşiği" hint="Bu kadar yeni onaylı kaynak birikince"><input type="number" min="8" max="4096" step="8" value={config.batchSize} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, batchSize: Number(event.target.value) }) }} /></Field>
                  <Field label="İşlemci politika hedefi" hint={`%${config.budgets.cpuPercent} · sert kota değil`}><input type="range" min="10" max="95" step="5" value={config.budgets.cpuPercent} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, budgets: { ...config.budgets, cpuPercent: Number(event.target.value) } }) }} /></Field>
                  <Field label="Bellek" hint={`${formatCount(config.budgets.ramMb)} MB · varsayılan 4 GB, açık üst sınır 8 GB`}><input type="number" min="512" max="8192" step="256" value={config.budgets.ramMb} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, budgets: { ...config.budgets, ramMb: Number(event.target.value) } }) }} /></Field>
                  <Field label="Zaman" hint={formatDuration(config.budgets.maxRunMinutes)}><input type="number" min="5" max="10080" step="5" value={config.budgets.maxRunMinutes} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, budgets: { ...config.budgets, maxRunMinutes: Number(event.target.value) } }) }} /></Field>
                  <Field label="Disk" hint={`${formatCount(config.budgets.diskMb)} MB · yeni eğitim verisi`}><input type="number" min="512" max="4096" step="512" value={config.budgets.diskMb} onChange={(event) => { setConfigDirty(true); setConfig({ ...config, budgets: { ...config.budgets, diskMb: Number(event.target.value) } }) }} /></Field>
                </div>
                {status?.resources && <div className="tr-feed-contract" aria-label="Canlı kaynak politikası">
                  <span><b>Servis RAM</b>{formatCount(status.resources.processRssMb)} / {formatCount(status.resources.ramLimitMb)} MB · {status.resources.memoryTier}</span>
                  <span><b>SSD boş</b>{formatCount(status.resources.diskFreeMb)} MB</span>
                  <span><b>Eğitim deposu</b>{formatCount(status.resources.logicalStorageMb)} / {formatCount(status.resources.logicalDiskLimitMb)} MB</span>
                  <span><b>Yeni blok</b>{status.resources.canStart ? 'Güvenli' : 'Rezerv bekliyor'}</span>
                </div>}
                <Button tone={configDirty ? 'primary' : 'default'} icon={<Settings2 size={14} />} disabled={!configDirty || busyAction !== null || running} onClick={saveConfig}>{busyAction === 'config' ? 'Kaydediliyor…' : 'Sınırları kaydet'}</Button>
              </article>
            </section>

            <section className="tr-source-grid">
              <article className="tr-panel tr-panel--padded">
                <div className="tr-section-heading"><div><span className="tr-eyebrow">GERÇEK VERİ</span><h2>Metin veya yerel dosya ekle</h2></div><FileText size={21} /></div>
                <div className="tr-form-grid"><Field label="Kaynak adı"><input value={textName} onChange={(event) => setTextName(event.target.value)} /></Field><Field label="Kaynak geçmişi"><input value={textProvenance} onChange={(event) => setTextProvenance(event.target.value)} /></Field></div>
                <Field label="Metni buraya yapıştırın" hint={`${textValue.length.toLocaleString('tr-TR')} karakter`}><textarea rows={7} value={textValue} placeholder="Kitap, makale, diyalog veya izinli metin…" onChange={(event) => setTextValue(event.target.value)} /></Field>
                <div className="tr-inline-actions"><Button tone="primary" icon={<ShieldCheck size={14} />} disabled={!textValue.trim() || busyAction !== null} onClick={addTextSource}>Karantinaya gönder</Button><Button icon={<Upload size={14} />} disabled={busyAction !== null} onClick={() => fileInputRef.current?.click()}>Dosya seç</Button><input ref={fileInputRef} className="tr-sr-only" type="file" accept=".txt,.md,.json,.jsonl,text/plain,text/markdown,application/json" multiple onChange={uploadFiles} /></div>
                <button className="tr-archive-transfer" disabled={busyAction !== null} onClick={syncTextLab}><Database size={17} /><span><b>Mevcut Metin Laboratuvarı arşivini aktar</b><small>Ham metin, özgün matris ayarı ve sabit kimlik parmak izleri taşınır; servis aynı I alanını yeniden üretip doğrular.</small></span><ChevronRight size={16} /></button>
              </article>

              <article className="tr-panel tr-panel--padded tr-teacher-panel">
                <div className="tr-section-heading"><div><span className="tr-eyebrow">SENTETİK AZINLIK</span><h2>Öğretmen örneği üret</h2></div><Sparkles size={21} /></div>
                <div className="tr-inline-warning is-important"><AlertTriangle size={15} /><span>Gerçek ve kaynak geçmişi bilinen metinler çoğunlukta kalmalıdır. Ollama üretimleri ayrı etiketlenir; gerçek kaynak holdout’unun yerini tutmaz ve toplam payı etkinleştirme kapısında %25 ile sınırlandırılır.</span></div>
                <div className="tr-form-grid"><Field label="Konu"><input value={teacher.topic} placeholder="Örn. araç–eylem ilişkisi" onChange={(event) => setTeacher({ ...teacher, topic: event.target.value })} /></Field><Field label="Üretim amacı"><input value={teacher.objective} placeholder="Örn. görülmemiş özne/araç birleşimleri" onChange={(event) => setTeacher({ ...teacher, objective: event.target.value })} /></Field><Field label="Örnek adedi"><input type="number" min="4" max="1000" step="4" value={teacher.count} onChange={(event) => setTeacher({ ...teacher, count: Number(event.target.value) })} /></Field><Field label="Çeşitlilik"><select value={teacher.diversity} onChange={(event) => setTeacher({ ...teacher, diversity: event.target.value as TeacherGenerateRequest['diversity'] })}><option value="focused">Odaklı</option><option value="balanced">Dengeli</option><option value="broad">Geniş</option></select></Field></div>
                <div className="tr-teacher-facts"><span><b>Dil</b> Türkçe</span><span><b>Model</b> {config.model || 'Seçilmedi'}</span><span><b>Hedef</b> Karantina</span></div>
                <Button tone="primary" icon={<Sparkles size={14} />} disabled={!status?.ollama.connected || !selectedModelAvailable || !teacher.topic.trim() || !teacher.objective.trim() || !Number.isInteger(teacher.count) || teacher.count < 4 || teacher.count > 1000 || busyAction !== null} onClick={generateTeacherData}>{busyAction === 'teacher' ? 'Üretiliyor…' : 'Örnekleri üret ve karantinaya al'}</Button>
              </article>
            </section>

            <section className="tr-panel tr-panel--padded">
              <div className="tr-section-heading"><div><span className="tr-eyebrow">SEÇİLİ KAYNAKLAR</span><h2>Eğitime girecek veri havuzu</h2></div><span className="tr-data-balance"><b>{selectedRealSources}</b> gerçek · <b>{selectedSyntheticSources}</b> sentetik</span></div>
              {selectedSources.length > 0 && selectedSources.length < 3 && <div className="tr-inline-warning tr-balance-warning" role="alert"><AlertTriangle size={15} /><span>Bu korpus deney için çalışır; ancak train, doğrulama ve test bölümlerini ayrı fiziksel kaynaklarla doldurup bir modeli etkinleştirebilmek için en az 3 bağımsız metin gerekir.</span></div>}
              {selectedSyntheticSources > 0 && (selectedRealSources === 0 || selectedSyntheticSources / Math.max(1, selectedSources.length) > .25) && <div className="tr-inline-warning tr-balance-warning" role="alert"><AlertTriangle size={15} /><span>Sentetik örnek oranı yüksek. Öğretmenin kendi anlatım kalıplarını çoğaltmamak için gerçek kaynakları çoğunlukta tutun; önerilen sentetik üst sınır %25’tir.</span></div>}
              {!data?.sources.length ? <EmptyState icon={<Database size={24} />} title="Henüz kaynak yok">Yukarıdan gerçek metin ekleyin veya öğretmen örneği üretin. Tüm yeni içerik önce karantinaya gider.</EmptyState> : <div className="tr-source-list">{data.sources.map((source) => <label key={source.id} className={source.selected ? 'is-selected' : ''}><input type="checkbox" checked={source.selected} disabled={running || busyAction !== null || source.state !== 'ready'} onChange={() => toggleSource(source)} /><span><b>{source.name}</b><small>{sourceKindLabel(source.kind)} · {formatBytes(source.bytes)} · {source.provenance}</small></span><em className={`is-${source.state}`}>{source.state === 'ready' ? 'Hazır' : source.state === 'quarantined' ? 'Karantinada' : source.state === 'processing' ? 'İşleniyor' : source.state === 'disabled' ? 'Kapalı' : 'Hata'}</em></label>)}</div>}
            </section>
          </>
        ) : view === 'quarantine' ? (
          <QuarantineView items={data?.quarantine ?? []} reasons={reviewReasons} setReasons={setReviewReasons} busy={busyAction === 'review'} onReview={reviewItem} />
        ) : view === 'activity' ? (
          <ActivityView queue={data?.queue ?? []} events={events} eventConnected={eventConnected} />
        ) : view === 'families' ? (
          <FamilyAuditView inspection={familyInspection} snapshots={data?.snapshots ?? []} selectedSnapshotId={familySnapshotId} loading={familyLoading} error={familyError} onRetry={() => void refreshFamilies(familyPageOffset)} onPageChange={setFamilyPageOffset} onSnapshotChange={(snapshotId) => { setFamilySnapshotId(snapshotId); setFamilyPageOffset(0) }} />
        ) : (
          <ModelsView metrics={data?.metrics ?? null} snapshots={data?.snapshots ?? []} activeSnapshotId={status?.activeSnapshotId ?? null} onActivate={activateCandidate} onRollback={setRollbackTarget} actionBusy={busyAction === 'activate' || busyAction === 'rollback'} />
        )}
      </main>

      {cancelConfirm && <div className="tr-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setCancelConfirm(false) }}><section className="tr-modal" role="dialog" aria-modal="true" aria-labelledby="cancel-title"><span className="tr-modal__icon is-danger"><CircleStop size={24} /></span><h2 id="cancel-title">Aktif eğitimi iptal et?</h2><p>Çalışan parti güvenli biçimde durdurulur. Henüz kabul edilmemiş aday model kalıcı sürüme yazılmaz.</p><div className="tr-modal__actions"><Button onClick={() => setCancelConfirm(false)}>Vazgeç</Button><Button tone="danger" onClick={() => { setCancelConfirm(false); void performAction('cancel', () => api.cancelRun(), 'İptal istendi; sistem güvenli sınırda duracak.') }}>Eğitimi iptal et</Button></div></section></div>}

      {rollbackTarget && <div className="tr-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setRollbackTarget(null) }}><section className="tr-modal" role="dialog" aria-modal="true" aria-labelledby="rollback-title"><span className="tr-modal__icon"><RotateCcw size={24} /></span><h2 id="rollback-title">{rollbackTarget.name} sürümüne dön?</h2><p>Aktif modelin yerini bu kayıt alacak. Mevcut sürüm silinmez; denetim geçmişinde geri alınmış olarak saklanır.</p><Field label="Geri alma gerekçesi"><textarea rows={3} value={rollbackReason} onChange={(event) => setRollbackReason(event.target.value)} /></Field>{!rollbackTarget.rollbackToken && <div className="tr-inline-warning"><AlertTriangle size={14} />Bu sürüm için sunucu onay anahtarı vermedi; geri alma yapılamaz.</div>}<div className="tr-modal__actions"><Button onClick={() => setRollbackTarget(null)}>Vazgeç</Button><Button tone="danger" disabled={!rollbackTarget.rollbackToken || !rollbackReason.trim()} icon={<RotateCcw size={14} />} onClick={confirmRollback}>Onayla ve geri al</Button></div></section></div>}
    </div>
  )
}

function QuarantineView({ items, reasons, setReasons, busy, onReview }: { items: QuarantineItem[]; reasons: Record<string, string>; setReasons: React.Dispatch<React.SetStateAction<Record<string, string>>>; busy: boolean; onReview: (item: QuarantineItem, decision: 'approve' | 'reject') => void }) {
  const pending = items.filter((item) => item.state === 'pending')
  return <section className="tr-panel tr-panel--padded"><div className="tr-section-heading"><div><span className="tr-eyebrow">İNSAN KONTROL KAPISI</span><h1>Karantina incelemesi</h1><p>Model üretimi ve dış kaynaklar onaylanana kadar öğrenme hattının dışında tutulur.</p></div><span className="tr-count-badge">{pending.length} bekliyor</span></div>{!items.length ? <EmptyState icon={<ShieldCheck size={27} />} title="Karantina boş">Yeni bir metin veya öğretmen üretimi geldiğinde kalite kontrolleriyle burada görünür.</EmptyState> : <div className="tr-quarantine-list">{items.map((item) => <article key={item.id} className={`is-${item.state}`}><header><div><span className="tr-eyebrow">{item.sourceName}</span><h2>{item.preview || 'Önizleme üretilemedi'}</h2></div><span className={`tr-review-state is-${item.state}`}>{item.state === 'pending' ? 'Karar bekliyor' : item.state === 'approved' ? 'Onaylandı' : 'Reddedildi'}</span></header><div className="tr-quarantine-meta"><span>{formatCount(item.characterCount)} karakter</span><span>{item.language || 'Dil belirsiz'}</span><span>{item.duplicateScore === undefined ? 'Tekrar ölçülmedi' : `%${Math.round(item.duplicateScore * 100)} tekrar benzerliği`}</span><span>{item.provenance}</span></div><div className="tr-checks">{item.checks.map((check) => <span key={check.id} className={`is-${check.state}`} title={check.message}><i>{check.state === 'pass' ? <Check size={11} /> : check.state === 'fail' ? <X size={11} /> : <AlertTriangle size={11} />}</i>{check.label}</span>)}</div>{item.state === 'pending' && <div className="tr-review-actions"><input aria-label="İnceleme notu" placeholder="İsteğe bağlı karar notu" value={reasons[item.id] ?? ''} onChange={(event) => setReasons((current) => ({ ...current, [item.id]: event.target.value }))} /><Button tone="danger" compact disabled={busy} icon={<X size={13} />} onClick={() => onReview(item, 'reject')}>Reddet</Button><Button tone="success" compact disabled={busy || item.checks.some((check) => check.state === 'fail')} icon={<Check size={13} />} onClick={() => onReview(item, 'approve')}>Onayla</Button></div>}</article>)}</div>}</section>
}

function ActivityView({ queue, events, eventConnected }: { queue: QueueItem[]; events: TrainingEvent[]; eventConnected: boolean }) {
  return <div className="tr-activity-grid"><section className="tr-panel tr-panel--padded"><div className="tr-section-heading"><div><span className="tr-eyebrow">KALICI İŞ KUYRUĞU</span><h1>Bekleyen ve çalışan işler</h1></div><ListChecks size={21} /></div>{!queue.length ? <EmptyState icon={<ListChecks size={26} />} title="Kuyruk boş">Eğitim başladığında her aşama ayrı, izlenebilir bir iş olarak burada görünür.</EmptyState> : <div className="tr-table-wrap"><table><thead><tr><th>İş</th><th>Aşama</th><th>Durum</th><th>Deneme</th><th>İlerleme</th><th>Güncelleme</th></tr></thead><tbody>{queue.map((item: QueueItem) => <tr key={item.id}><td><b>{item.label}</b><small>{item.sourceName || item.id.slice(0, 12)}</small></td><td>{STAGE_LABELS[item.stage].title}</td><td><span className={`tr-queue-state is-${item.state}`}>{item.state === 'waiting' ? 'Bekliyor' : item.state === 'active' ? 'Çalışıyor' : item.state === 'complete' ? 'Tamamlandı' : item.state === 'failed' ? 'Hata' : 'İptal'}</span></td><td>{item.attempts}</td><td><progress max="1" value={clampRatio(item.progress)} /><small>%{Math.round(clampRatio(item.progress) * 100)}</small></td><td>{formatDate(item.updatedAt)}</td></tr>)}</tbody></table></div>}</section><section className="tr-panel tr-panel--padded tr-event-panel"><div className="tr-section-heading"><div><span className="tr-eyebrow">CANLI OLAY AKIŞI</span><h2>Servis günlüğü</h2></div><span className={`tr-connect-pill ${eventConnected ? 'is-online' : ''}`}><i />{eventConnected ? 'Canlı' : 'Yeniden bağlanıyor'}</span></div>{!events.length ? <EmptyState icon={<Activity size={24} />} title="Henüz canlı olay yok">Servis bağlantısı kurulduğunda yeni olaylar burada akacak. Geçmiş uydurulmaz.</EmptyState> : <ol className="tr-event-list" aria-live="polite">{events.map((event) => <li key={event.id} className={`is-${event.level}`}><time>{formatDate(event.at)}</time><b>{eventLevelLabel(event.level)}</b><span>{event.stage ? `${STAGE_LABELS[event.stage].title} · ` : ''}{event.message}</span></li>)}</ol>}</section></div>
}

const FAMILY_LIFECYCLE_LABELS: Record<string, string> = {
  provisional: 'İlk aday',
  strengthening: 'Güçleniyor',
  stable: 'Kararlı',
  'split-proposed': 'Bölünme önerisi',
  'merge-proposed': 'Birleşme önerisi',
  retired: 'Emekli',
}

const FAMILY_REGION_LABELS: Record<string, string> = {
  content: 'İçerik',
  event: 'Olay / eylem',
  role: 'Rol',
  operator: 'İşleç',
}

function familyLifecycleLabel(value: string): string {
  return FAMILY_LIFECYCLE_LABELS[value] ?? value
}

function familyRegionLabel(value: string): string {
  return FAMILY_REGION_LABELS[value] ?? value
}

function formatScore(value: number | null): string {
  return typeof value === 'number' && Number.isFinite(value) ? `%${Math.round(clampRatio(value) * 100)}` : '—'
}

function formatSigned(value: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return `${value > 0 ? '+' : ''}${value.toLocaleString('tr-TR', { minimumFractionDigits: 4, maximumFractionDigits: 4 })}`
}

function FamilyAuditView({
  inspection,
  snapshots,
  selectedSnapshotId,
  loading,
  error,
  onRetry,
  onPageChange,
  onSnapshotChange,
}: {
  inspection: FamilyInspection | null
  snapshots: ModelSnapshot[]
  selectedSnapshotId: string
  loading: boolean
  error: string | null
  onRetry: () => void
  onPageChange: (offset: number) => void
  onSnapshotChange: (snapshotId: string) => void
}) {
  const [query, setQuery] = useState('')
  const [lifecycle, setLifecycle] = useState('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const families = inspection?.families ?? []
  const visibleFamilies = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase('tr-TR')
    return families.filter((family) => {
      if (lifecycle !== 'all' && family.lifecycle !== lifecycle) return false
      if (!normalized) return true
      return [family.id, family.region, family.lifecycle, ...family.topMembers.map((member) => member.text)]
        .some((value) => value.toLocaleLowerCase('tr-TR').includes(normalized))
    })
  }, [families, lifecycle, query])
  const selected = visibleFamilies.find((family) => family.id === selectedId) ?? visibleFamilies[0] ?? null

  useEffect(() => {
    if (selected && selected.id !== selectedId) setSelectedId(selected.id)
    if (!selected && selectedId !== null) setSelectedId(null)
  }, [selected, selectedId])

  if (loading && !inspection) return <LoadingState label="Aile kanıtları okunuyor…" />
  if (error && !inspection) {
    return <section className="tr-panel tr-panel--padded"><EmptyState icon={<AlertTriangle size={27} />} title="Aile denetimi okunamadı"><>{error} <button className="tr-inline-link" onClick={onRetry}>Yeniden dene</button></></EmptyState></section>
  }
  if (!inspection?.available) {
    return <section className="tr-panel tr-panel--padded"><div className="tr-section-heading"><div><span className="tr-eyebrow">DAVRANIŞTAN DOĞAN KÜMELER</span><h1>Aile denetimi</h1><p>Aile adları önceden verilmez; benzer davranış ve bağımsız kaynak desteğiyle oluşur.</p></div><Network size={23} /></div><EmptyState icon={<Network size={30} />} title="Henüz denetlenebilir aile çalışması yok">{inspection?.reason || 'İlk gerçek aileleştirme aşaması tamamlandığında kanıtlar burada görünür.'} Bir ailenin doğabilmesi için en az 3 destek ve 2 bağımsız fiziksel kaynak gerekir.</EmptyState></section>
  }

  const summary = inspection.summary
  const page = inspection.pagination
  const selectionLabel = inspection.selection.snapshot
    ? `${inspection.selection.snapshot.name || `Sürüm ${inspection.selection.snapshot.id.slice(0, 10)}`}${inspection.selection.snapshot.active ? ' · etkin' : ''}`
    : inspection.selection.artifact
      ? `Son aile çalışması · ${inspection.selection.artifact.id.slice(0, 10)}`
      : 'Kayıt seçilemedi'

  return (
    <div className="tr-family-stack">
      <section className="tr-panel tr-panel--padded tr-family-overview">
        <div className="tr-section-heading">
          <div><span className="tr-eyebrow">DAVRANIŞTAN DOĞAN KÜMELER</span><h1>Aile denetimi</h1><p>Etiket değil kanıt gösterilir. “Bölge”, sistemin davranıştan çıkardığı işlev hipotezidir; kelimenin kesin anlamı değildir.</p></div>
          <div className="tr-family-selection"><label><span>Denetlenen sürüm</span><select value={selectedSnapshotId} onChange={(event) => onSnapshotChange(event.target.value)} disabled={loading}><option value="">En güncel aile çalışması</option>{snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>{snapshot.name} · {formatDate(snapshot.createdAt)}</option>)}</select></label><b>{selectionLabel}</b><small>{formatDate(inspection.run?.createdAt ?? inspection.selection.artifact?.createdAt)}</small></div>
        </div>
        {error && <div className="tr-inline-warning"><AlertTriangle size={15} /><span>Son yenileme başarısız: {error}. Ekranda son sağlam okuma tutuluyor.</span></div>}
        <div className="tr-family-summary">
          <article><span>Toplam aile</span><strong>{formatCount(summary.familyCount)}</strong><small>{summary.byLifecycle.map((item) => `${familyLifecycleLabel(item.key)} ${item.count}`).join(' · ') || 'Yaşam evresi yok'}</small></article>
          <article><span>Bağımsız kaynaklı</span><strong>{formatCount(summary.multiSourceFamilyCount)}</strong><small>En az iki fiziksel kaynaktan kanıt alan aile</small></article>
          <article><span>Faydası doğrulanan</span><strong>{formatCount(summary.validatedUtilityFamilyCount)}</strong><small>{formatCount(summary.utilityMeasuredFamilyCount)} aile için susturma sınavı ölçüldü</small></article>
          <article><span>Kaynak çeşitliliği</span><strong>{formatCount(summary.distinctSourceCount)}</strong><small>Aynı kitabın parçaları tek kaynak sayılır</small></article>
          <article><span>Yumuşak aidiyet</span><strong>{formatCount(summary.membershipCount)}</strong><small>Bir nöron birden fazla aileye ağırlıklı bağlanabilir</small></article>
          <article><span>Aile geçişi</span><strong>{formatCount(summary.familyTransitionCount)}</strong><small>Sabit yol değil, gözlenen sonraki-aile olasılığı</small></article>
        </div>
      </section>

      {!summary.familyCount ? (
        <section className="tr-panel tr-panel--padded"><EmptyState icon={<Network size={28} />} title="Bu çalışmada aile doğmadı">Bu bir hata olmak zorunda değildir. En az 3 benzer davranış desteği ve 2 bağımsız kaynak aynı kümede buluşmadan sistem aile üretmez.</EmptyState></section>
      ) : (
        <section className="tr-family-browser">
          <aside className="tr-panel tr-family-list-panel">
            <div className="tr-family-tools">
              <label><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Aile veya üye ara" aria-label="Aile veya üye ara" /></label>
              <select value={lifecycle} onChange={(event) => setLifecycle(event.target.value)} aria-label="Yaşam evresine göre süz"><option value="all">Tüm yaşam evreleri</option>{summary.byLifecycle.map((item) => <option key={item.key} value={item.key}>{familyLifecycleLabel(item.key)} · {item.count}</option>)}</select>
            </div>
            <div className="tr-family-list" role="listbox" aria-label="Öğrenilmiş aileler">
              {!visibleFamilies.length ? <EmptyState icon={<Search size={23} />} title="Bu sayfada eşleşme yok">Arama veya yaşam evresi süzgecini değiştirin.</EmptyState> : visibleFamilies.map((family) => <button key={family.id} className={selected?.id === family.id ? 'is-selected' : ''} onClick={() => setSelectedId(family.id)} role="option" aria-selected={selected?.id === family.id}><span className={`tr-family-life is-${family.lifecycle}`}>{familyLifecycleLabel(family.lifecycle)}</span><strong>{familyRegionLabel(family.region)} ailesi</strong><small>{family.topMembers.slice(0, 3).map((member) => member.text).filter(Boolean).join(' · ') || family.id.slice(0, 18)}</small><div><span>{family.sourceCount} kaynak</span><span>{family.supportCount} destek</span><span>{formatScore(family.cohesion)} uyum</span></div></button>)}
            </div>
            <footer className="tr-family-pagination"><Button compact icon={<ChevronLeft size={13} />} disabled={page.offset <= 0 || loading} onClick={() => onPageChange(Math.max(0, page.offset - page.limit))}>Önceki</Button><span>{page.returned ? page.offset + 1 : 0}–{page.offset + page.returned} / {page.total}</span><Button compact disabled={!page.hasMore || page.nextOffset === null || loading} onClick={() => page.nextOffset !== null && onPageChange(page.nextOffset)}>Sonraki<ChevronRight size={13} /></Button></footer>
          </aside>

          <FamilyDetail family={selected} />
        </section>
      )}
    </div>
  )
}

function FamilyDetail({ family }: { family: InspectedFamily | null }) {
  if (!family) return <section className="tr-panel tr-panel--padded"><EmptyState icon={<Network size={26} />} title="Aile seçin">Kanıtlarını görmek için soldaki listeden bir aile seçin.</EmptyState></section>
  const utility = family.utilityEvidence
  const utilityLabel = utility?.status === 'validated'
    ? 'Faydası doğrulandı'
    : utility?.status === 'neutral-or-harmful'
      ? 'Nötr veya zararlı'
      : 'Kanıt yetersiz'
  return (
    <article className="tr-panel tr-panel--padded tr-family-detail">
      <header>
        <div><span className="tr-eyebrow">{family.regionVerified ? 'BÖLGE KANITI' : 'BÖLGE HİPOTEZİ'}</span><h2>{familyRegionLabel(family.region)} ailesi</h2><code>{family.id}</code></div>
        <span className={`tr-family-life is-${family.lifecycle}`}>{familyLifecycleLabel(family.lifecycle)}</span>
      </header>
      <p className="tr-family-reason">{family.lifecycleReason || 'Yaşam evresi gerekçesi kaydedilmedi.'}</p>
      <dl className="tr-family-scores">
        <div><dt>Kaynak</dt><dd>{family.sourceCount}</dd><small>bağımsız fiziksel kayıt</small></div>
        <div><dt>Destek</dt><dd>{family.supportCount}</dd><small>{family.memberCount} üye · {family.coreMemberCount} çekirdek</small></div>
        <div><dt>İç uyum</dt><dd>{formatScore(family.cohesion)}</dd><progress max="1" value={clampRatio(family.cohesion ?? 0)} /></div>
        <div><dt>Ayrışma</dt><dd>{formatScore(family.separation)}</dd><progress max="1" value={clampRatio(family.separation ?? 0)} /></div>
        <div><dt>Kararlılık</dt><dd>{formatScore(family.stability)}</dd><progress max="1" value={clampRatio(family.stability ?? 0)} /></div>
        <div><dt>Fayda</dt><dd>{formatSigned(family.utility)}</dd><small>{utilityLabel}</small></div>
      </dl>

      <section className={`tr-utility-evidence is-${utility?.status ?? 'missing'}`}>
        <div><strong>Susturma sınavı · {utilityLabel}</strong><span>{utility?.explanation || 'Bu ailenin tahmine katkısını aile kapalıyken karşılaştırmak için henüz yeterli bağımsız vaka yok.'}</span></div>
        {utility && <dl><div><dt>Vaka</dt><dd>{utility.caseCount}</dd></div><div><dt>Kaynak katı</dt><dd>{utility.sourceFoldCount}</dd></div><div><dt>Fayda farkı</dt><dd>{formatSigned(utility.deltaUtility)}</dd></div><div><dt>Güven aralığı</dt><dd>{formatSigned(utility.confidenceLow)}…{formatSigned(utility.confidenceHigh)}</dd></div></dl>}
      </section>

      <section className="tr-family-section">
        <div className="tr-section-heading"><div><span className="tr-eyebrow">YUMUŞAK AİTLİK</span><h2>En güçlü üyeler</h2></div><small>İlk {family.topMembers.length} üye · ham matris taşınmadı</small></div>
        {!family.topMembers.length ? <EmptyState icon={<Database size={23} />} title="Üye özeti yok">Bu kayıt için gösterilebilir üye kanıtı bulunamadı.</EmptyState> : <div className="tr-family-members">{family.topMembers.map((member) => <article key={member.memberKey}><div><strong title={member.textTruncated ? member.memberKey : undefined}>{member.text || 'Metinsiz üye'}</strong><span>{member.core ? 'Çekirdek' : `Sıra ${member.rank}`}</span></div><small>{member.kind} · {member.sourceCount} kaynak · ağırlık {formatScore(member.weight)}</small><progress max="1" value={clampRatio(member.softScore ?? 0)} title={`Yumuşak aidiyet ${formatScore(member.softScore)}`} /><p>{member.sources.map((source) => source.name).join(' · ') || 'Kaynak adı saklanmamış'}</p></article>)}</div>}
      </section>

      <section className="tr-family-section">
        <div className="tr-section-heading"><div><span className="tr-eyebrow">OLASILIKLI YOL</span><h2>Sonraki aile adayları</h2></div><small>En güçlü {family.outgoingTransitions.length} geçiş</small></div>
        {!family.outgoingTransitions.length ? <EmptyState icon={<ChevronRight size={24} />} title="Geçiş kanıtı yok">Bu aileden sonra tekrar eden, kaynak destekli bir aile geçişi henüz oluşmadı.</EmptyState> : <div className="tr-family-transitions">{family.outgoingTransitions.map((transition) => <article key={transition.id}><span><b>{familyRegionLabel(transition.toRegion)}</b><small>{familyLifecycleLabel(transition.toLifecycle)} · {transition.sourceCount} kaynak · {transition.supportCount} destek</small></span><div><progress max="1" value={clampRatio(transition.probability ?? 0)} /><strong>{formatScore(transition.probability)}</strong></div><p>alışkanlık {formatScore(transition.priorEvidence)} · deneyim {formatScore(transition.experienceEvidence)} · baskılama {formatScore(transition.inhibitionEvidence)}</p></article>)}</div>}
      </section>
      {family.predecessorFamilyIds.length > 0 && <footer className="tr-family-lineage"><b>Önceki aile kayıtları</b><span>{family.predecessorFamilyIds.join(' · ')}</span></footer>}
    </article>
  )
}

function ModelsView({ metrics, snapshots, activeSnapshotId, onActivate, onRollback, actionBusy }: { metrics: TrainingMetrics | null; snapshots: ModelSnapshot[]; activeSnapshotId: string | null; onActivate: (snapshot: ModelSnapshot) => void; onRollback: (snapshot: ModelSnapshot) => void; actionBusy: boolean }) {
  const metricCards = metrics ? [
    ['Görülmemiş ilk seçim', metrics.holdoutTop1, 'up' as const, 'Sistemin eğitimde görmediği devamı ilk adayda bulma oranı.'],
    ['Bağımsız test ilk seçim', metrics.unseenCombinationAccuracy, 'up' as const, 'Eğitim ve doğrulamaya ayrılmayan fiziksel kaynak gruplarında top-1 sonraki parça doğruluğu.'],
    ['Dilbilgisel tercih', metrics.grammaticalPreference, 'up' as const, 'Düzgün devamı bozuk karşı örneğe tercih etme oranı.'],
    ['Aile faydası', metrics.familyUtility, 'up' as const, 'Aile yönlendirmesinin aileler kapalıyken oluşan tabana katkısı.'],
    ['Öğrenilmiş aile', metrics.familyCount, 'up' as const, 'Davranış benzerliği, en az üç destek ve iki bağımsız kaynakla doğan temel nöron ailesi sayısı. Artış tek başına kalite değildir.'],
    ['Uygun doğum önerisi', metrics.eligibleBirthProposalCount, 'up' as const, `Toplam ${formatMetric(metrics.birthProposalCount)} sürpriz/yenilik kümesi içinde bağımsız kaynak eşiğini geçen, fakat henüz sabit kimlik tahsis edilmemiş nöron önerileri.`],
    ['Uygun dil ailesi', metrics.eligibleEmergentLanguageFamilyCount, 'up' as const, `Toplam ${formatMetric(metrics.emergentLanguageFamilyCount)} geçiş-imzası kümesi içinde holdout susturma faydası kanıtlanan kendiliğinden dil aileleri.`],
    ['Doğrulanmış kavram makrosu', metrics.conceptMacroShadowValidatedCount, 'up' as const, `Toplam ${formatMetric(metrics.conceptMacroCount)} tekrar örüntüsü içinde ayrılmış veride genişletilmiş yoluna sadık kalan, ham nöronları silmeyen sanal kavramlar. Etkin: ${formatMetric(metrics.conceptMacroActiveCount)}.`],
    ['Nöron sayısı', metrics.neuronCount, 'up' as const, 'Kimliği korunmuş toplam nöron. Artış tek başına başarı değildir.'],
    ['Geri alma oranı', metrics.rollbackRate, 'down' as const, 'Tüm sürüm geçmişi içinde kullanıcı tarafından açıkça başlatılan geri dönüşlerin oranı.'],
  ] as const : []
  return <div className="tr-model-stack"><section className="tr-panel tr-panel--padded"><div className="tr-section-heading"><div><span className="tr-eyebrow">GÖRÜLMEMİŞ VERİ SINAVI</span><h1>Model ölçümleri</h1><p>Eğitim verisindeki ezber değil, ayrılmış doğrulama verisindeki değişim gösterilir.</p></div>{metrics && <span className="tr-count-badge">{formatCount(metrics.evaluationSampleCount)} test örneği</span>}</div>{!metrics ? <EmptyState icon={<BarChart3 size={26} />} title="Henüz karşılaştırmalı ölçüm yok">İlk aday model değerlendirme aşamasına ulaştığında ölçümleri burada görünür.</EmptyState> : <div className="tr-metric-grid">{metricCards.map(([label, metric, beneficial, help]) => <MetricCard key={label} label={label} metric={metric} beneficial={beneficial} help={help} />)}</div>}</section><section className="tr-panel tr-panel--padded"><div className="tr-section-heading"><div><span className="tr-eyebrow">GERİ DÖNÜLEBİLİR GEÇMİŞ</span><h2>Model sürümleri</h2></div><Archive size={21} /></div>{!snapshots.length ? <EmptyState icon={<Archive size={25} />} title="Henüz sürüm kaydı yok">İlk eğitim değerlendirmesi tamamlandığında aday sürüm kaydı burada oluşur.</EmptyState> : <div className="tr-snapshot-list">{snapshots.map((snapshot) => <article key={snapshot.id} className={snapshot.id === activeSnapshotId ? 'is-active' : ''}><div className="tr-snapshot-icon"><Archive size={17} /></div><div><div className="tr-snapshot-title"><strong>{snapshot.name}</strong>{snapshot.id === activeSnapshotId && <span>Etkin</span>}<em className={`is-${snapshot.state}`}>{snapshot.state === 'candidate' ? 'Aday' : snapshot.state === 'committed' ? 'Kabul' : snapshot.state === 'rejected' ? 'Reddedildi' : 'Geri alındı'}</em></div><small>{formatDate(snapshot.createdAt)} · {formatBytes(snapshot.bytes)} · {snapshot.note || snapshot.corpusFingerprint.slice(0, 16)}</small></div><div className="tr-snapshot-metrics"><span>İlk seçim <b>{formatMetric(snapshot.metrics.holdoutTop1)}</b></span><span>Dil <b>{formatMetric(snapshot.metrics.grammaticalPreference)}</b></span><span>Aile <b>{formatMetric(snapshot.metrics.familyUtility)}</b></span></div><div className="tr-snapshot-actions">{snapshot.state === 'candidate' && <Button compact tone="success" icon={<CheckCircle2 size={13} />} disabled={actionBusy} onClick={() => onActivate(snapshot)}>Adayı kabul et</Button>}<Button compact icon={<RotateCcw size={13} />} disabled={actionBusy || snapshot.id === activeSnapshotId || snapshot.state !== 'committed'} onClick={() => onRollback(snapshot)}>Bu sürüme dön</Button></div></article>)}</div>}</section></div>
}
