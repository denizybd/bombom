import {
  Activity,
  AlertCircle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Clock3,
  GitBranch,
  Gauge,
  Loader2,
  Layers3,
  MessageSquareText,
  Network,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Target,
  Waves,
  X,
} from 'lucide-react'
import {
  type FormEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { SuiteMenu } from '../components/SuiteMenu'
import { DEFAULT_THOUGHT_SERVICE_URL, ThoughtApi, ThoughtApiError } from './api'
import {
  canRequestLearningProposal,
  clamp01,
  formatRatio,
  scoreBar,
  statusLabel,
  teacherModeLabel,
} from './model'
import { ThoughtHeatmap } from './ThoughtHeatmap'
import type {
  ExpectedStructuralOutcome,
  NewbornPhase,
  ReplayVerification,
  StartThoughtEpisodeRequest,
  TeacherMode,
  ThoughtAdaptiveGate,
  ThoughtBranch,
  ThoughtCandidate,
  ThoughtCoalition,
  ThoughtEpisode,
  ThoughtEpisodeSummary,
  ThoughtEpisodeState,
  ThoughtGraph,
  ThoughtGraphLevel,
  ThoughtLearningMode,
  ThoughtNewborn,
  ThoughtServiceStatus,
  ThoughtSnapshotOption,
  ThoughtTick,
} from './types'

type BusyAction = 'refresh' | 'start' | 'pause' | 'resume' | 'cancel' | 'replay' | 'feedback-approve' | 'feedback-reject' | null

interface ExpectedOutcomeDraft {
  actor: string
  action: string
  patient: string
  instrument: string
  operators: string
  negated: boolean
  goals: string
}

const EMPTY_EXPECTATION: ExpectedOutcomeDraft = {
  actor: '',
  action: '',
  patient: '',
  instrument: '',
  operators: '',
  negated: false,
  goals: '',
}

function Button({
  children,
  icon,
  tone = 'default',
  compact = false,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode
  tone?: 'default' | 'primary' | 'danger' | 'ghost' | 'success'
  compact?: boolean
}) {
  return <button className={`tha-button tha-button--${tone} ${compact ? 'tha-button--compact' : ''}`} {...props}>{icon}{children}</button>
}

function formatDate(value?: string): string {
  if (!value) return 'Zaman bekleniyor'
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) return 'Zaman bekleniyor'
  return date.toLocaleString('tr-TR', { dateStyle: 'short', timeStyle: 'short' })
}

function shortPrompt(value: string, limit = 86): string {
  const compact = value.replace(/\s+/g, ' ').trim()
  return compact.length > limit ? `${compact.slice(0, limit - 1)}…` : compact
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Beklenmeyen bir sorun oluştu.'
}

function stateDescription(state: ThoughtEpisodeState): string {
  return ({
    queued: 'Başlamak için yerel sırada bekliyor.',
    running: 'Sinyal dalları şu anda ilerliyor.',
    paused: 'Kaldığı tik ve aynı rastgelelik tohumu korundu.',
    completed: 'Bölüm güvenli biçimde sonlandı.',
    failed: 'Akış tamamlanamadı; kayıt inceleme için korundu.',
    cancelled: 'Akış kullanıcı tarafından durduruldu.',
    unknown: 'Servisten güncel durum bekleniyor.',
  } satisfies Record<ThoughtEpisodeState, string>)[state]
}

function newbornLabel(phase: NewbornPhase): string {
  return ({
    none: 'Doğum yok',
    watching: 'Birikim izleniyor',
    candidate: 'Yeni nöron adayı',
    eligible: 'Doğuma uygun',
    born: 'Yeni nöron doğdu',
    quarantined: 'Güvenli alanda bekliyor',
    rejected: 'Aday elendi',
    unknown: 'Durum açıklanmadı',
  } satisfies Record<NewbornPhase, string>)[phase]
}

function regionModeLabel(mode: string): string {
  return ({ off: 'Kapalı', shadow: 'Gölge gözlem', proposal: 'Öneri', active: 'Etkin' } as Record<string, string>)[mode] ?? mode
}

function regionReasonLabel(reason?: string | null): string {
  return ({
    'request-off': 'Bu bölüm bölge mekanizmasını istemedi.',
    'policy-off': 'Sistem politikası davranış etkisini kapalı tutuyor.',
    'disabled-by-policy': 'Bölge mekanizması sistem politikasıyla kapalı.',
    'PLASTIC_REGION_REENCODE_REQUIRED': 'Bu tabanın önce 8/56 temsilinde yeniden kodlanması gerekiyor.',
    'adaptive-representation-required': 'Bu tabanın önce 8/56 temsilinde yeniden kodlanması gerekiyor.',
    'verified-atlas-required': 'Bu snapshot’a bağlı doğrulanmış atlas bekleniyor.',
    'active-snapshot-required': 'Active yalnız etkin model snapshot’ında çalışabilir.',
    'verified-region-release-required': 'Active için bağımsız holdout ve sham release kanıtı bekleniyor.',
  } as Record<string, string>)[reason ?? ''] ?? (reason ? `Güvenli kapanma nedeni: ${reason}` : 'Çalışma nedeni bildirilmedi.')
}

function compactFingerprint(value?: string | null): string {
  if (!value) return '—'
  return value.length > 22 ? `${value.slice(0, 10)}…${value.slice(-8)}` : value
}

function AdaptiveRegionPanel({ episode, tick, branch }: {
  episode: ThoughtEpisodeSummary
  tick: ThoughtTick | null
  branch: ThoughtBranch | null
}) {
  const runtime = episode.regionRuntime
  const branchTelemetry = branch?.adaptiveRegion ?? null
  const tickTelemetry = tick?.adaptiveRegion ?? null
  if (!runtime && !branchTelemetry && !tickTelemetry) return null
  const mode = branchTelemetry?.mode ?? tickTelemetry?.mode ?? episode.regionMode
  const activeApplied = branchTelemetry?.activeApplied ?? tickTelemetry?.activeApplied ?? runtime?.behaviorApplied ?? false
  const strongestGate = branchTelemetry?.gates.gates.reduce<ThoughtAdaptiveGate | null>((strongest, gate) =>
    (gate.probability ?? -1) > (strongest?.probability ?? -1) ? gate : strongest, null) ?? null
  const unmatched = branchTelemetry?.gates.unmatchedProbability ?? null
  const writeMask = branchTelemetry?.pageWriteMask ?? []
  const writtenPages = writeMask.filter((value) => value > 0).length
  const maskMean = writeMask.length ? writeMask.reduce((sum, value) => sum + value, 0) / writeMask.length : null
  const maskPeak = writeMask.length ? Math.max(...writeMask) : null
  const maskPeakPage = maskPeak === null ? null : writeMask.indexOf(maskPeak)
  const atlasFingerprint = branchTelemetry?.atlasFingerprint ?? tickTelemetry?.atlasFingerprint ?? runtime?.atlasFingerprint ?? null
  const bindingFingerprint = branchTelemetry?.bindingFingerprint ?? tickTelemetry?.bindingFingerprint ?? runtime?.bindingFingerprint ?? null
  return (
    <section className={`tha-region-telemetry is-${mode}`} aria-labelledby="adaptive-region-title">
      <header>
        <div><Layers3 size={18} /><span><small>8/56 EŞLİKÇİ TELEMETRİ · SALT OKUNUR</small><h3 id="adaptive-region-title">Uyarlanabilir bölge çalışma izi</h3></span></div>
        <b>{regionModeLabel(mode)} · {activeApplied ? 'davranış uygulandı' : 'davranış uygulanmadı'}</b>
      </header>
      <p>{activeApplied ? 'Bu tik doğrulanmış active atlas yolunu kullandı.' : mode === 'shadow' ? 'Bu tik yalnız gözlem topladı; nöron seçimini değiştirmedi.' : mode === 'proposal' ? 'Bu tik geri alınabilir öneri hesabı yaptı; etkin modeli değiştirmedi.' : regionReasonLabel(runtime?.reason)}</p>
      <dl>
        <div><dt>Atlas</dt><dd>{runtime?.atlasRevision === null || runtime?.atlasRevision === undefined ? 'revizyon —' : `revizyon ${runtime.atlasRevision}`}</dd><code title={atlasFingerprint ?? undefined}>{compactFingerprint(atlasFingerprint)}</code></div>
        <div><dt>En güçlü kapı</dt><dd>{strongestGate?.regionId ?? 'Kapı yok'}</dd><small>kapı {formatRatio(strongestGate?.compatibility)} · prototip {formatRatio(strongestGate?.prototypeCompatibility)}</small></div>
        <div><dt>Eşleşmeyen kütle</dt><dd>{formatRatio(unmatched)}</dd><small>atlanmış bölge {formatRatio(branchTelemetry?.gates.omittedRegionProbability)}</small></div>
        <div><dt>Yazma maskesi</dt><dd>{writeMask.length ? `${writtenPages} / 56 sayfa` : 'Kayıt yok'}</dd><small>ortalama {formatRatio(maskMean)} · tepe {formatRatio(maskPeak)}{maskPeakPage === null ? '' : ` (sayfa ${maskPeakPage})`}</small></div>
        <div><dt>Uyum karışımı</dt><dd>{formatRatio(branchTelemetry?.effectiveCompatibility)}</dd><small>eski {formatRatio(branchTelemetry?.legacyCompatibility)} · bölge {formatRatio(branchTelemetry?.regionCompatibility)}</small></div>
        <div><dt>Replay bağı</dt><dd>{bindingFingerprint ? 'Bağlı' : 'Bağ yok'}</dd><code title={bindingFingerprint ?? undefined}>{compactFingerprint(bindingFingerprint)}</code></div>
      </dl>
      <footer>Atlas {compactFingerprint(atlasFingerprint)} · release {compactFingerprint(runtime?.releaseFingerprint)} · {tickTelemetry ? `${tickTelemetry.observedBranchCount} dal gözlendi` : 'tik atlas özeti bekleniyor'}</footer>
    </section>
  )
}

function parseExpectation(draft: ExpectedOutcomeDraft): { value: ExpectedStructuralOutcome | null; invalidGoals: string[] } {
  const operators = draft.operators.split(',').map((item) => item.trim()).filter(Boolean)
  const goals: ExpectedStructuralOutcome['goals'] = []
  const invalidGoals: string[] = []
  draft.goals.split('\n').map((item) => item.trim()).filter(Boolean).forEach((line) => {
    const separator = line.indexOf(':')
    const relation = separator >= 0 ? line.slice(0, separator).trim() : ''
    const target = separator >= 0 ? line.slice(separator + 1).trim() : ''
    if (!relation || !target || !/^[\p{L}\p{N}][\p{L}\p{N}_-]{0,63}$/u.test(relation)) invalidGoals.push(line)
    else goals.push({ relation, target })
  })
  const actor = draft.actor.trim() || null
  const action = draft.action.trim() || null
  const patient = draft.patient.trim() || null
  const instrument = draft.instrument.trim() || null
  const hasEvidence = Boolean(actor || action || patient || instrument || operators.length || goals.length || draft.negated)
  return {
    value: hasEvidence ? {
      format: 'event-graph-v1',
      actor,
      action,
      patient,
      instrument,
      operators,
      negated: draft.negated,
      goals,
    } : null,
    invalidGoals,
  }
}

function MetricGauge({ label, value, help, tone = 'blue' }: {
  label: string
  value: number | null
  help: string
  tone?: 'blue' | 'red' | 'green' | 'amber'
}) {
  return (
    <article className={`tha-gauge is-${tone}`} title={help}>
      <div><span>{label}</span><strong>{formatRatio(value)}</strong></div>
      <div className="tha-gauge__track" aria-hidden="true"><i style={{ width: `${clamp01(value) * 100}%` }} /></div>
      <small>{help}</small>
    </article>
  )
}

function CandidateBreakdown({ candidates }: { candidates: ThoughtCandidate[] }) {
  if (!candidates.length) {
    return (
      <div className="tha-mini-empty">
        <GitBranch size={20} />
        <strong>Aday ayrıntısı bekleniyor</strong>
        <span>Bu tik henüz karşılaştırılan nöron puanlarını göndermedi.</span>
      </div>
    )
  }
  return (
    <div className="tha-candidates">
      <div className="tha-candidates__legend">
        <span>Birleşik</span>
        <span title="Başlangıçtaki kas hafızası olasılığı">H⁰ önseli</span>
        <span title="Bölümler arasında korunan yavaş deneyim izi">Kalıcı deneyim E</span>
        <span title="Yalnız bu bölümde hızla oluşan dal izi">Hızlı dal izi e</span>
        <span title="Nöron ailesinin verdiği yumuşak destek">Aile G</span>
        <span title="Anlık sinyal ile adayın uyumu">Sinyal q</span>
        <span title="Aynı döngüye saplanmayı azaltan eksi katkı">Tekrar cezası</span>
      </div>
      {candidates.slice(0, 12).map((candidate) => (
        <article key={candidate.id} className={candidate.selected ? 'is-selected' : ''}>
          <div className="tha-candidate__identity">
            <b>{candidate.rank}</b>
            <span><strong>{candidate.label}</strong><small>{candidate.familyId ? `Aile ${candidate.familyId}` : candidate.neuronId ? `Nöron ${candidate.neuronId}` : 'Kimlik bilgisi bekleniyor'}</small></span>
            {candidate.selected && <CheckCircle2 size={15} aria-label="Seçilen aday" />}
          </div>
          <div className="tha-candidate__total">
            <div><i style={{ width: `${scoreBar(candidate.total ?? candidate.probability) * 100}%` }} /></div>
            <strong>{formatRatio(candidate.total ?? candidate.probability)}</strong>
          </div>
          <div className="tha-candidate__parts">
            <span title="H⁰: Başlangıçtaki kas hafızası olasılığı"><small>H⁰</small><b>{formatRatio(candidate.h0Prior)}</b><i style={{ width: `${scoreBar(candidate.h0Prior) * 100}%` }} /></span>
            <span title="E: Bölümler arasında korunan yavaş deneyim"><small>E</small><b>{formatRatio(candidate.persistentExperience)}</b><i style={{ width: `${scoreBar(candidate.persistentExperience) * 100}%` }} /></span>
            <span title="e: Yalnız bu bölümde hızla oluşan dal izi"><small>e</small><b>{formatRatio(candidate.branchTrace)}</b><i style={{ width: `${scoreBar(candidate.branchTrace) * 100}%` }} /></span>
            <span title="G: Nöron ailesinin yumuşak desteği"><small>G</small><b>{formatRatio(candidate.family)}</b><i style={{ width: `${scoreBar(candidate.family) * 100}%` }} /></span>
            <span title="q: Anlık sinyal ile adayın uyumu"><small>q</small><b>{formatRatio(candidate.signalQ)}</b><i style={{ width: `${scoreBar(candidate.signalQ) * 100}%` }} /></span>
            <span title="Aynı döngüye saplanmayı azaltan eksi katkı"><small>Ceza</small><b>{formatRatio(candidate.repetitionPenalty)}</b><i style={{ width: `${scoreBar(candidate.repetitionPenalty) * 100}%` }} /></span>
          </div>
        </article>
      ))}
    </div>
  )
}

function CoalitionCard({ coalition }: { coalition: ThoughtCoalition | null }) {
  const roleLabel = (role: string) => ({
    actor: 'Yapan', agent: 'Yapan', patient: 'Etkilenen', affected: 'Etkilenen',
    instrument: 'Araç', operator: 'Operatör', goal: 'Hedef',
  }[role.toLowerCase()] ?? role)
  return (
    <article className="tha-detail-card tha-coalition-card">
      <header><span><Network size={16} /> KOALİSYON</span><strong>{coalition?.label ?? 'Henüz birleşme yok'}</strong></header>
      {coalition ? (
        <>
          <div className="tha-detail-metrics">
            <span><b>Güç</b>{formatRatio(coalition.score)}</span>
            <span><b>Uyum</b>{formatRatio(coalition.agreement)}</span>
            <span><b>Üye</b>{coalition.memberIds.length}</span>
          </div>
          <div className="tha-member-cloud">
            {(coalition.memberLabels.length ? coalition.memberLabels : coalition.memberIds).map((member, index) => <span key={`${member}-${index}`}>{member}</span>)}
            {!coalition.memberIds.length && <small>Üyeler henüz raporlanmadı.</small>}
          </div>
          <div className="tha-role-hypotheses">
            <div><b>Geçici rol bağları</b><small>Bunlar kalıcı nöron/aile etiketi değildir.</small></div>
            {coalition.roleHypotheses.length ? coalition.roleHypotheses.map((hypothesis, index) => (
              <span key={`${hypothesis.role}-${hypothesis.target}-${index}`}><b>{roleLabel(hypothesis.role)}</b><strong>{hypothesis.target}</strong><small>{formatRatio(hypothesis.score)}</small></span>
            )) : <p>Yapan, etkilenen, araç veya operatör henüz çıkarılmadı.</p>}
          </div>
        </>
      ) : <p>Birden fazla düşünce parçası aynı yönü desteklediğinde burada birlikte görünür.</p>}
    </article>
  )
}

function NewbornCard({ newborn, pressure }: { newborn: ThoughtNewborn | null; pressure: number | null }) {
  const phase = newborn?.phase ?? 'none'
  return (
    <article className={`tha-detail-card tha-newborn-card is-${phase}`}>
      <header><span><Sparkles size={16} /> YENİ NÖRON</span><strong>{newbornLabel(phase)}</strong></header>
      <div className="tha-birth-orbit" aria-hidden="true"><i /><span>{formatRatio(newborn?.confidence ?? pressure)}</span></div>
      <p>{newborn?.reason ?? (phase === 'none'
        ? 'Mevcut yollar sinyali yeterince karşılıyor; doğum baskısı eşik altında.'
        : 'Tekrarlanan uyumsuzluk güvenli alanda birikiyor.')}</p>
      {(newborn?.label || newborn?.id) && <small>{newborn.label ?? 'Adsız aday'}{newborn.id ? ` · ${newborn.id}` : ''}</small>}
    </article>
  )
}

function StructuralOutcomeCard({ graph, trust }: { graph: ThoughtGraph | null; trust: number | null }) {
  const outcome = graph?.structuralOutcome ?? null
  const evaluation = graph?.structuralEvaluation ?? null
  const fields = outcome ? [
    ['Yapan', outcome.actor],
    ['Eylem', outcome.action],
    ['Etkilenen', outcome.patient],
    ['Araç', outcome.instrument],
  ] as const : []
  return (
    <article className="tha-detail-card tha-coalition-card">
      <header><span><Target size={16} /> YAPISAL SONUÇ</span><strong>{outcome ? 'Olay grafiği kuruldu' : 'Henüz sonuç yok'}</strong></header>
      {outcome ? (
        <>
          <div className="tha-detail-metrics">
            <span><b>Graph-F1</b>{formatRatio(evaluation?.graphF1)}</span>
            <span><b>Kısıt</b>{formatRatio(evaluation?.constraintAccuracy)}</span>
            <span><b>Öğrenme güveni</b>{formatRatio(trust)}</span>
          </div>
          <div className="tha-role-hypotheses">
            {fields.map(([label, value]) => <span key={label}><b>{label}</b><strong>{value ?? 'çıkarılmadı'}</strong></span>)}
            <span><b>Olumsuzluk</b><strong>{outcome.negated ? 'var' : 'yok'}</strong></span>
            {outcome.operators.map((operator) => <span key={`operator-${operator}`}><b>Operatör</b><strong>{operator}</strong></span>)}
            {outcome.goals.map((goal) => <span key={`${goal.relation}-${goal.target}`}><b>{goal.relation}</b><strong>{goal.target}</strong></span>)}
          </div>
        </>
      ) : <p>Bölüm tamamlandığında sistemin kendi kurduğu yapan–eylem–etkilenen–araç yapısı burada görünür.</p>}
    </article>
  )
}

function LanguageOutputCard({ graph }: { graph: ThoughtGraph | null }) {
  const output = graph?.languageOutput ?? null
  return (
    <article className="tha-detail-card tha-coalition-card">
      <header><span><MessageSquareText size={16} /> CÜMLE ÇIKIŞ KATMANI</span><strong>{output?.stoppedBy === 'eos' ? 'EOS ile tamamlandı' : 'Henüz tamamlanmadı'}</strong></header>
      {output ? (
        <>
          <p className="tha-language-output">{output.text || output.error || 'Yapısal sonuç henüz yüzey cümlesine dönüşmedi.'}</p>
          <div className="tha-role-hypotheses">
            <div><b>Elle kurulan omurga</b><small>Plastik üyelikler eğitimle gelişir</small></div>
            {output.steps.map((step, index) => (
              <span key={`${step.familyId}-${index}`}>
                <b>{index + 1}. kapı</b>
                <strong>{step.emittedSurface || step.explanation}</strong>
                <small>{step.complete ? 'EOS' : 'geçti'}</small>
              </span>
            ))}
          </div>
        </>
      ) : <p>Baskın düşünce olay grafiğine dönüştüğünde altı dil ailesi burada denetlenebilir bir cümle kuracak.</p>}
    </article>
  )
}

export function ThoughtApp() {
  const api = useMemo(() => new ThoughtApi(DEFAULT_THOUGHT_SERVICE_URL), [])
  const [status, setStatus] = useState<ThoughtServiceStatus | null>(null)
  const [snapshots, setSnapshots] = useState<ThoughtSnapshotOption[]>([])
  const [episodes, setEpisodes] = useState<ThoughtEpisode[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [episode, setEpisode] = useState<ThoughtEpisode | null>(null)
  const [initialLoading, setInitialLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [busyAction, setBusyAction] = useState<BusyAction>(null)
  const [notice, setNotice] = useState<{ tone: 'success' | 'error' | 'info'; text: string } | null>(null)
  const [replay, setReplay] = useState<ReplayVerification | null>(null)
  const [prompt, setPrompt] = useState('')
  const [snapshotId, setSnapshotId] = useState('')
  const [teacherMode, setTeacherMode] = useState<TeacherMode>('off')
  const [learningMode, setLearningMode] = useState<ThoughtLearningMode>('off')
  const [seed, setSeed] = useState(42)
  const [maxTicks, setMaxTicks] = useState(64)
  const [expectation, setExpectation] = useState<ExpectedOutcomeDraft>(EMPTY_EXPECTATION)
  const [tickCursor, setTickCursor] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [followLive, setFollowLive] = useState(true)
  const [selectedBranchId, setSelectedBranchId] = useState<string | null>(null)
  const [graphLevel, setGraphLevel] = useState<ThoughtGraphLevel>('family')
  const [graph, setGraph] = useState<ThoughtGraph | null>(null)
  const [graphLoading, setGraphLoading] = useState(false)
  const [graphError, setGraphError] = useState<string | null>(null)
  const [reviewReason, setReviewReason] = useState('')
  const mountedRef = useRef(true)
  const actionLockRef = useRef<Exclude<BusyAction, null> | null>(null)
  const proposalEligible = canRequestLearningProposal(status, snapshotId)
  const tickLimit = Math.max(1, status?.maximumTicks ?? 512)

  const refreshOverview = useCallback(async (quiet = false, signal?: AbortSignal) => {
    if (!quiet) setBusyAction('refresh')
    const [statusResult, episodesResult, snapshotsResult] = await Promise.allSettled([
      api.getStatus(signal),
      api.listEpisodes(signal),
      api.listSnapshots(signal),
    ])
    if (!mountedRef.current) return

    if (statusResult.status === 'fulfilled') setStatus(statusResult.value)
    else setStatus(null)
    if (episodesResult.status === 'fulfilled') {
      const next = episodesResult.value
      setEpisodes(next.map((item) => ({ ...item, ticks: [], warnings: [] })))
      setSelectedId((current) => current && next.some((item) => item.id === current) ? current : next[0]?.id ?? null)
    }
    if (snapshotsResult.status === 'fulfilled') {
      const activeSnapshotId = statusResult.status === 'fulfilled' ? statusResult.value.activeSnapshotId : null
      const mapped = snapshotsResult.value.map((item) => ({
        ...item,
        active: item.id === activeSnapshotId,
        commitEligible: item.id === activeSnapshotId,
      }))
      setSnapshots(mapped)
      setSnapshotId((current) => current && mapped.some((item) => item.id === current)
        ? current
        : mapped.find((item) => item.active)?.id ?? '')
    }

    if (statusResult.status === 'rejected' && episodesResult.status === 'rejected') {
      const error = episodesResult.reason instanceof ThoughtApiError ? episodesResult.reason : statusResult.reason
      if (!(error instanceof ThoughtApiError && error.code === 'ABORTED')) setConnectionError(errorMessage(error))
    } else {
      setConnectionError(null)
    }
    setInitialLoading(false)
    if (!quiet) setBusyAction(null)
  }, [api])

  const loadEpisode = useCallback(async (id: string, quiet = false, signal?: AbortSignal) => {
    if (!quiet) setDetailLoading(true)
    try {
      const next = await api.getEpisodeWithTicks(id, signal)
      if (!mountedRef.current) return
      setEpisode(next)
      setEpisodes((current) => {
        const without = current.filter((item) => item.id !== next.id)
        return [next, ...without].sort((left, right) => (right.createdAt ?? '').localeCompare(left.createdAt ?? ''))
      })
      setTickCursor((current) => followLive ? Math.max(0, next.ticks.length - 1) : Math.min(current, Math.max(0, next.ticks.length - 1)))
      setDetailError(null)
    } catch (error) {
      if (error instanceof ThoughtApiError && error.code === 'ABORTED') return
      if (mountedRef.current) setDetailError(errorMessage(error))
    } finally {
      if (mountedRef.current && !quiet) setDetailLoading(false)
    }
  }, [api, followLive])

  const loadGraph = useCallback(async (level: ThoughtGraphLevel, signal?: AbortSignal) => {
    setGraphLoading(true)
    try {
      const next = await api.getGraph({ level, limit: 60, ...(selectedId ? { episodeId: selectedId } : {}) }, signal)
      if (!mountedRef.current) return
      setGraph(next)
      setGraphError(null)
    } catch (error) {
      if (error instanceof ThoughtApiError && error.code === 'ABORTED') return
      if (mountedRef.current) {
        setGraph(null)
        setGraphError(errorMessage(error))
      }
    } finally {
      if (mountedRef.current) setGraphLoading(false)
    }
  }, [api, selectedId])

  useEffect(() => {
    mountedRef.current = true
    const controller = new AbortController()
    void refreshOverview(false, controller.signal)
    const poll = window.setInterval(() => void refreshOverview(true), 4_000)
    return () => {
      mountedRef.current = false
      controller.abort()
      window.clearInterval(poll)
    }
  }, [refreshOverview])

  useEffect(() => {
    if (!selectedId) {
      setEpisode(null)
      setDetailError(null)
      return
    }
    const controller = new AbortController()
    void loadEpisode(selectedId, false, controller.signal)
    const poll = window.setInterval(() => void loadEpisode(selectedId, true), 2_000)
    return () => {
      controller.abort()
      window.clearInterval(poll)
    }
  }, [loadEpisode, selectedId])

  useEffect(() => {
    const controller = new AbortController()
    void loadGraph(graphLevel, controller.signal)
    return () => controller.abort()
  }, [episode?.state, graphLevel, loadGraph])

  useEffect(() => {
    if (!proposalEligible) setLearningMode('off')
  }, [proposalEligible])

  useEffect(() => {
    setMaxTicks((current) => Math.min(current, tickLimit))
  }, [tickLimit])

  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => {
      setFollowLive(false)
      setTickCursor((current) => {
        const last = Math.max(0, (episode?.ticks.length ?? 1) - 1)
        if (current >= last) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 780)
    return () => window.clearInterval(timer)
  }, [episode?.ticks.length, playing])

  const tick = episode?.ticks[tickCursor] ?? null

  useEffect(() => {
    setSelectedBranchId(tick?.selectedBranchId ?? tick?.branches[0]?.id ?? null)
  }, [tick?.index, tick?.selectedBranchId])

  const branch = tick?.branches.find((item) => item.id === selectedBranchId) ?? tick?.branches[0] ?? null
  const candidates = branch?.candidates.length ? branch.candidates : tick?.candidates ?? []
  const coalition = branch?.coalition ?? tick?.coalition ?? null
  const unmatched = branch?.unmatched ?? tick?.unmatched ?? null
  const edge = branch?.edge ?? tick?.edge ?? null
  const birthPressure = branch?.birthPressure ?? tick?.birthPressure ?? null
  const newborn = branch?.newborn ?? tick?.newborn ?? null
  const afterMatrix = branch?.signalAfter ?? tick?.signalAfter
  const selectedSummary = episodes.find((item) => item.id === selectedId)
  const shownEpisode = episode ?? selectedSummary ?? null
  const runningCount = episodes.filter((item) => item.state === 'running').length
  const graphNodeNames = new Map(graph?.nodes.map((node) => [node.id, node.label]) ?? [])
  const reviewable = shownEpisode?.state === 'completed' && shownEpisode.commitStatus === 'pending-review' && shownEpisode.commitEligible

  const chooseEpisode = (id: string) => {
    setSelectedId(id)
    setEpisode(null)
    setReplay(null)
    setPlaying(false)
    setFollowLive(true)
    setTickCursor(0)
  }

  const runAction = async (
    kind: Exclude<BusyAction, null>,
    operation: () => Promise<ThoughtEpisode>,
    success: string,
  ) => {
    if (actionLockRef.current) return
    actionLockRef.current = kind
    setBusyAction(kind)
    setNotice(null)
    try {
      const next = await operation()
      if (!mountedRef.current) return
      setEpisode(next)
      setNotice({ tone: 'success', text: success })
      await refreshOverview(true)
    } catch (error) {
      if (mountedRef.current) setNotice({ tone: 'error', text: errorMessage(error) })
    } finally {
      actionLockRef.current = null
      if (mountedRef.current) setBusyAction(null)
    }
  }

  const startEpisode = async (event: FormEvent) => {
    event.preventDefault()
    if (!prompt.trim() || actionLockRef.current) return
    const parsedExpectation = parseExpectation(expectation)
    if (parsedExpectation.invalidGoals.length) {
      setNotice({ tone: 'error', text: `Hedef ilişkisini “ilişki: hedef” biçiminde yaz: ${parsedExpectation.invalidGoals[0]}` })
      return
    }
    actionLockRef.current = 'start'
    setBusyAction('start')
    setNotice(null)
    const payload: StartThoughtEpisodeRequest = {
      prompt: prompt.trim(),
      ...(snapshotId ? { snapshotId } : {}),
      teacherMode,
      learningMode: proposalEligible ? learningMode : 'off',
      seed: Math.round(seed),
      maxTicks: Math.max(1, Math.min(tickLimit, Math.round(maxTicks))),
      expectedOutcome: parsedExpectation.value,
    }
    try {
      const created = await api.startEpisode(payload)
      if (!mountedRef.current) return
      setEpisodes((current) => [created, ...current.filter((item) => item.id !== created.id)])
      setSelectedId(created.id)
      setEpisode(created)
      setTickCursor(Math.max(0, created.ticks.length - 1))
      setFollowLive(true)
      setPrompt('')
      setNotice({ tone: 'success', text: 'Yeni düşünce bölümü sıraya alındı. Tikler geldikçe ekran kendiliğinden güncellenecek.' })
      await refreshOverview(true)
    } catch (error) {
      if (mountedRef.current) setNotice({ tone: 'error', text: errorMessage(error) })
    } finally {
      actionLockRef.current = null
      if (mountedRef.current) setBusyAction(null)
    }
  }

  const replayEpisode = async () => {
    if (!shownEpisode || actionLockRef.current) return
    actionLockRef.current = 'replay'
    setBusyAction('replay')
    setReplay(null)
    setNotice(null)
    try {
      const verification = await api.replayEpisode(shownEpisode.id)
      if (!mountedRef.current) return
      setReplay(verification)
      setNotice({
        tone: verification.matches === false ? 'error' : 'success',
        text: verification.message,
      })
    } catch (error) {
      if (mountedRef.current) setNotice({ tone: 'error', text: errorMessage(error) })
    } finally {
      actionLockRef.current = null
      if (mountedRef.current) setBusyAction(null)
    }
  }

  const cancelEpisode = () => {
    if (!shownEpisode) return
    if (!window.confirm('Bu düşünce bölümünü durdurmak istediğine emin misin? Tamamlanan tikler kayıtta kalır.')) return
    void runAction('cancel', () => api.cancelEpisode(shownEpisode.id), 'Bölüm durduruldu; tamamlanan tikler korundu.')
  }

  const reviewProposal = async (decision: 'approve' | 'reject') => {
    if (!shownEpisode || !reviewable || actionLockRef.current) return
    const kind = decision === 'approve' ? 'feedback-approve' : 'feedback-reject'
    actionLockRef.current = kind
    setBusyAction(kind)
    setNotice(null)
    try {
      const result = await api.feedbackEpisode(shownEpisode.id, decision, reviewReason.trim())
      if (!mountedRef.current) return
      setEpisode(result.episode)
      setEpisodes((current) => [result.episode, ...current.filter((item) => item.id !== result.episode.id)])
      setReviewReason('')
      setNotice({
        tone: 'success',
        text: decision === 'approve'
          ? `Öneri etkinleştirilmedi; bağımsız değerlendirme için aday sürüm${result.snapshot?.id ? ` ${result.snapshot.id}` : ''} oluşturuldu.`
          : 'Öğrenme önerisi reddedildi. Bölümün değişmez düşünce izi korunuyor.',
      })
      await refreshOverview(true)
    } catch (error) {
      if (mountedRef.current) setNotice({ tone: 'error', text: errorMessage(error) })
    } finally {
      actionLockRef.current = null
      if (mountedRef.current) setBusyAction(null)
    }
  }

  return (
    <div className="tha-app">
      <header className="tha-topbar">
        <SuiteMenu currentApp="thought" navigationLocked={busyAction === 'start'}>
          <span className="tha-brand">
            <span className="tha-brand__mark"><Waves size={22} /></span>
            <span><strong>DÜŞÜNCE AKIŞI</strong><small>SİNYALİN CANLI YOLCULUĞU</small></span>
          </span>
        </SuiteMenu>
        <div className={`tha-service-pill ${connectionError ? 'is-offline' : status?.available === false ? 'is-waiting' : 'is-online'}`}>
          <i />
          <span>{connectionError ? 'Servis bekleniyor' : status?.available === false ? 'Hazırlanıyor' : runningCount ? `${runningCount} deney çalışıyor` : 'Akış hazır'}</span>
        </div>
        <div className="tha-topbar__actions">
          <span className="tha-topbar__count"><BrainCircuit size={14} /> {episodes.length} bölüm</span>
          <Button
            compact
            tone="ghost"
            aria-label="Ekranı yenile"
            title="Ekranı yenile"
            disabled={busyAction !== null}
            icon={<RefreshCw size={15} className={busyAction === 'refresh' ? 'is-spinning' : ''} />}
            onClick={() => void refreshOverview(false)}
          >Yenile</Button>
        </div>
      </header>

      <div className="tha-context-strip">
        <span><b>SERVİS</b>{DEFAULT_THOUGHT_SERVICE_URL}</span>
        <span><b>AKTİF</b>{status?.activeEpisodes ?? runningCount}</span>
        <span><b>SIRADA</b>{status?.queuedEpisodes ?? episodes.filter((item) => item.state === 'queued').length}</span>
        <span><b>HARİTA</b>64 × 64</span>
        <span className="tha-context-strip__last"><b>SÜRÜM</b>{status?.version ?? 'bekleniyor'}</span>
      </div>

      <main className="tha-main">
        <section className="tha-intro-grid">
          <article className="tha-hero">
            <span className="tha-eyebrow">DÜŞÜNCE DİNAMİĞİ DENEYİ</span>
            <h1>Düşünceyi sonuçtan önce, <em>hareket ederken</em> gör.</h1>
            <p>Her tikte sinyalin nasıl değiştiğini, hangi yolların yarıştığını, hangi düşüncelerin birleştiğini ve yeni bir nöron ihtiyacının nerede doğduğunu izle.</p>
            <div className="tha-hero__facts">
              <span><GitBranch size={14} /><b>1–8 dal</b>aynı anda karşılaştırılır</span>
              <span><ShieldCheck size={14} /><b>Tekrar oynat</b>aynı tohumla doğrulanır</span>
              <span><Target size={14} /><b>Dış sonuç</b>yalnız değerlendirme kanıtıdır</span>
            </div>
          </article>

          <form className="tha-start-card" onSubmit={startEpisode}>
            <div className="tha-section-heading">
            <div><span className="tha-eyebrow">YENİ BÖLÜM</span><h2>Bir dinamik deney başlat</h2></div>
              <Sparkles size={20} />
            </div>
            <label className="tha-field tha-field--prompt">
              <span>Başlangıç girdisi</span>
              <textarea value={prompt} rows={3} maxLength={12_000} placeholder="Örn. Yağmur başladığında açık pencereleri kapatmalıyım…" onChange={(event) => setPrompt(event.target.value)} />
              <small>{prompt.length.toLocaleString('tr-TR')} / 12.000 karakter</small>
            </label>
            <div className="tha-start-grid">
              <label className="tha-field">
                <span>Başlangıç sürümü</span>
                <select value={snapshotId} onChange={(event) => {
                  const next = event.target.value
                  setSnapshotId(next)
                  if (!status?.activeSnapshotId || next !== status.activeSnapshotId) setLearningMode('off')
                }}>
                  <option value="">Son araştırma snapshotını kullansın</option>
                  {snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>{snapshot.name}{snapshot.active ? ' · etkin' : ''}</option>)}
                </select>
              </label>
              <label className="tha-field">
                <span>Öğretmen denetimi</span>
                <select value={teacherMode} onChange={(event) => setTeacherMode(event.target.value as TeacherMode)}>
                  <option value="off">Kapalı</option>
                  <option value="ollama-cross-check">Ollama ile sonradan karşılaştır</option>
                </select>
              </label>
              <label className="tha-field">
                <span>Tekrar tohumu</span>
                <input type="number" min={0} max={4_294_967_295} step={1} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
              </label>
              <label className="tha-field">
                <span>En fazla tik</span>
                <input type="number" min={1} max={tickLimit} step={1} value={maxTicks} onChange={(event) => setMaxTicks(Number(event.target.value))} />
              </label>
              <label className="tha-field tha-field--wide">
                <span>Kalıcı değişiklik davranışı</span>
                <select value={learningMode} onChange={(event) => setLearningMode(event.target.value as ThoughtLearningMode)}>
                  <option value="off">Salt iz — hiçbir değişiklik önerme</option>
                  <option value="proposal" disabled={!proposalEligible}>Öğrenme önerisi hazırla — inceleme zorunlu</option>
                </select>
                <small>{proposalEligible
                  ? 'Öneri modu yalnız bir değişiklik adayı hazırlayabilir. Otomatik kabul veya etkinleştirme yapmaz.'
                  : 'Öneri modu yalnız etkin ve bağımsız doğrulanmış snapshot seçildiğinde açılır; bu seçim araştırma izi üretir.'}</small>
              </label>
            </div>
            <details className="tha-expectation">
              <summary><Target size={15} /><span><b>Beklenen dış yapı</b><small>İsteğe bağlı · yalnız ölçme kanıtı</small></span><ChevronRight size={15} /></summary>
              <div className="tha-expectation__notice">
                Bu alanlar nöronlara isim ya da aile etiketi vermez. Bölümün kendi başına kurduğu yapıyı dışarıdan verilen bir sonuçla karşılaştırır.
              </div>
              <div className="tha-start-grid">
                <label className="tha-field"><span>Yapan</span><input value={expectation.actor} placeholder="örn. çocuk" onChange={(event) => setExpectation((current) => ({ ...current, actor: event.target.value }))} /></label>
                <label className="tha-field"><span>Eylem</span><input value={expectation.action} placeholder="örn. kapattı" onChange={(event) => setExpectation((current) => ({ ...current, action: event.target.value }))} /></label>
                <label className="tha-field"><span>Etkilenen</span><input value={expectation.patient} placeholder="örn. pencere" onChange={(event) => setExpectation((current) => ({ ...current, patient: event.target.value }))} /></label>
                <label className="tha-field"><span>Araç</span><input value={expectation.instrument} placeholder="örn. uzaktan kumanda" onChange={(event) => setExpectation((current) => ({ ...current, instrument: event.target.value }))} /></label>
                <label className="tha-field tha-field--wide"><span>Operatörler</span><input value={expectation.operators} placeholder="olasılık, zorunluluk (virgülle ayır)" onChange={(event) => setExpectation((current) => ({ ...current, operators: event.target.value }))} /></label>
                <label className="tha-field tha-field--wide"><span>Hedef ilişkileri</span><textarea rows={2} value={expectation.goals} placeholder={'amaç: odayı kuru tutmak\nönkoşul: yağmurun başlaması'} onChange={(event) => setExpectation((current) => ({ ...current, goals: event.target.value }))} /><small>Her satıra “ilişki: hedef” yaz.</small></label>
              </div>
              <label className={`tha-negation ${expectation.negated ? 'is-selected' : ''}`}>
                <input type="checkbox" checked={expectation.negated} onChange={(event) => setExpectation((current) => ({ ...current, negated: event.target.checked }))} />
                <span><b>Olumsuzluk var</b><small>Beklenen yapıda “değil / yapmadı” benzeri olumsuzlama bulunuyor.</small></span>
                <i />
              </label>
            </details>
            <div className="tha-start-card__footer">
              <small>{learningMode === 'proposal'
                ? 'Dış yapısal kanıt yeterliyse bölüm sonunda insan kararı bekleyen bir öneri oluşabilir.'
                : teacherMode === 'off' ? 'Akış yalnız kendi nöronlarıyla ilerler.' : 'Ollama akışı yönetmez; tamamlanan yapıyı çapraz denetler.'}</small>
              <Button type="submit" tone="primary" disabled={!prompt.trim() || busyAction !== null || status?.available === false || status?.enabled === false || status?.researchBaseAvailable === false} icon={busyAction === 'start' ? <Loader2 className="is-spinning" size={16} /> : <Play size={16} />}>Bölümü başlat</Button>
            </div>
          </form>
        </section>

        {notice && (
          <div className={`tha-notice is-${notice.tone}`} role="status">
            {notice.tone === 'error' ? <AlertCircle size={17} /> : <CheckCircle2 size={17} />}
            <span>{notice.text}</span>
            <button type="button" aria-label="Bildirimi kapat" onClick={() => setNotice(null)}><X size={15} /></button>
          </div>
        )}

        {connectionError && (
          <section className="tha-offline">
            <AlertCircle size={20} />
            <div><strong>Düşünce servisi henüz yanıt vermiyor</strong><p>{connectionError}</p><small>Sayfa açık kalabilir; servis geldiğinde liste kendiliğinden yenilenir.</small></div>
            <Button compact onClick={() => void refreshOverview(false)} icon={<RefreshCw size={14} />}>Tekrar dene</Button>
          </section>
        )}

        {!connectionError && status && (status.available === false || status.enabled === false || status.researchBaseAvailable === false) && (
          <section className="tha-readiness-warning">
            <AlertCircle size={18} />
            <div>
              <strong>{status.enabled === false ? 'Düşünce çalışma katmanı kapalı' : status.available === false ? 'Düşünce motoru bağlanmayı bekliyor' : 'Araştırma tabanı henüz oluşmadı'}</strong>
              <p>{status.message ?? (status.researchBaseAvailable === false
                ? 'Önce aile çalışması içeren bir snapshot üretildiğinde yeni bölüm başlatılabilir.'
                : 'Mevcut bölümler okunabilir; yeni bölüm için yerel motorun hazır olması gerekir.')}</p>
            </div>
          </section>
        )}

        <section className="tha-workspace">
          <aside className="tha-episodes-panel">
            <header>
              <div><span className="tha-eyebrow">BÖLÜMLER</span><h2>Dinamik deneyleri</h2></div>
              <span>{episodes.length}</span>
            </header>
            {initialLoading ? (
              <div className="tha-side-loading"><Loader2 className="is-spinning" size={18} /> Bölümler okunuyor…</div>
            ) : episodes.length ? (
              <div className="tha-episode-list">
                {episodes.map((item) => {
                  const progress = item.maxTicks > 0 ? clamp01(item.tickCount / item.maxTicks) : 0
                  return (
                    <button key={item.id} type="button" className={`${item.id === selectedId ? 'is-selected' : ''} is-${item.state}`} onClick={() => chooseEpisode(item.id)}>
                      <div className="tha-episode-list__top"><span className={`tha-state is-${item.state}`}><i />{statusLabel(item.state)}</span><time>{formatDate(item.createdAt)}</time></div>
                      <strong>{shortPrompt(item.prompt)}</strong>
                      <div className="tha-episode-list__meta"><span>Tohum {item.seed}</span><span>{item.tickCount} / {item.maxTicks} tik</span><span>{teacherModeLabel(item.teacherMode)}</span><span>{item.learningMode === 'proposal' ? 'Öneri açık' : 'Salt iz'}</span></div>
                      <div className="tha-episode-list__track"><i style={{ width: `${progress * 100}%` }} /></div>
                    </button>
                  )
                })}
              </div>
            ) : (
              <div className="tha-side-empty"><Waves size={25} /><strong>Henüz bölüm yok</strong><p>Yukarıdaki kısa formdan ilk düşünceyi başlat.</p></div>
            )}
          </aside>

          <section className="tha-inspector">
            {!shownEpisode ? (
              <div className="tha-inspector-empty">
                <div className="tha-empty-orbit"><BrainCircuit size={28} /></div>
                <span className="tha-eyebrow">İNCELEME ALANI</span>
                <h2>Bir düşünce bölümü seç</h2>
                <p>Sinyalin önceki ve sonraki halini, yarışan dalları, koalisyonları ve doğum baskısını burada tik tik göreceksin.</p>
              </div>
            ) : (
              <>
                <header className="tha-episode-header">
                  <div className="tha-episode-header__copy">
                    <div><span className={`tha-state is-${shownEpisode.state}`}><i />{statusLabel(shownEpisode.state)}</span><small>{stateDescription(shownEpisode.state)}</small></div>
                    <h2>{shownEpisode.prompt}</h2>
                    <div className="tha-episode-header__meta">
                      <span><b>Tohum</b>{shownEpisode.seed}</span>
                      <span><b>Sürüm</b>{shownEpisode.snapshotName ?? shownEpisode.snapshotId ?? 'Etkin sürüm'}</span>
                      <span><b>Denetim</b>{teacherModeLabel(shownEpisode.teacherMode)}</span>
                      <span><b>Değişiklik</b>{shownEpisode.learningMode === 'proposal' ? 'Öneri modu' : 'Salt iz'}</span>
                      <span><b>Başlangıç</b>{formatDate(shownEpisode.createdAt)}</span>
                    </div>
                  </div>
                  <div className="tha-episode-actions">
                    {shownEpisode.state === 'running' && <Button compact disabled={busyAction !== null} icon={<Pause size={14} />} onClick={() => void runAction('pause', () => api.pauseEpisode(shownEpisode.id), 'Düşünce bölümü duraklatıldı.')}>Duraklat</Button>}
                    {shownEpisode.state === 'paused' && <Button compact tone="success" disabled={busyAction !== null} icon={<Play size={14} />} onClick={() => void runAction('resume', () => api.resumeEpisode(shownEpisode.id), 'Düşünce bölümü kaldığı yerden sürüyor.')}>Sürdür</Button>}
                    {['queued', 'running', 'paused'].includes(shownEpisode.state) && <Button compact tone="danger" disabled={busyAction !== null} icon={<CircleStop size={14} />} onClick={cancelEpisode}>Durdur</Button>}
                    <Button compact disabled={busyAction !== null || ['queued', 'running'].includes(shownEpisode.state)} icon={busyAction === 'replay' ? <Loader2 className="is-spinning" size={14} /> : <RotateCcw size={14} />} onClick={() => void replayEpisode()}>Aynı tohumla oynat</Button>
                  </div>
                </header>

                {replay && (
                  <div className={`tha-replay-result ${replay.matches === false ? 'is-mismatch' : 'is-match'}`}>
                    <ShieldCheck size={17} />
                    <div><strong>{replay.matches === true ? 'Akış birebir doğrulandı' : replay.matches === false ? 'Akışta ayrışma bulundu' : 'Tekrar deneyi tamamlandı'}</strong><span>{replay.message}</span></div>
                    {replay.firstMismatchTick !== null && <small>İlk fark: tik {replay.firstMismatchTick}</small>}
                  </div>
                )}

                {reviewable && (
                  <section className="tha-review-panel">
                    <div className="tha-review-panel__icon"><ShieldCheck size={21} /></div>
                    <div className="tha-review-panel__copy">
                      <span className="tha-eyebrow">İNSAN KARARI BEKLEYEN ÖNERİ</span>
                      <h3>Bu bölüm denetlenebilir bir değişiklik adayı hazırladı</h3>
                      <p>Onay, değişikliği etkin modele yazmaz. Ayrı bir aday snapshot oluşturur; adayın ayrıca bağımsız değerlendirme ve etkinleştirme kapısından geçmesi gerekir.</p>
                      <input maxLength={1_000} value={reviewReason} placeholder="Karar notu (isteğe bağlı)" onChange={(event) => setReviewReason(event.target.value)} />
                    </div>
                    <div className="tha-review-panel__actions">
                      <Button tone="danger" disabled={busyAction !== null} icon={busyAction === 'feedback-reject' ? <Loader2 className="is-spinning" size={15} /> : <X size={15} />} onClick={() => void reviewProposal('reject')}>Öneriyi reddet</Button>
                      <Button tone="primary" disabled={busyAction !== null} icon={busyAction === 'feedback-approve' ? <Loader2 className="is-spinning" size={15} /> : <ShieldCheck size={15} />} onClick={() => void reviewProposal('approve')}>Değerlendirmeye gönder</Button>
                    </div>
                  </section>
                )}

                {shownEpisode.state === 'completed' && ['candidate', 'rejected'].includes(shownEpisode.commitStatus) && (
                  <div className={`tha-review-state is-${shownEpisode.commitStatus}`}>
                    {shownEpisode.commitStatus === 'candidate' ? <ShieldCheck size={16} /> : <X size={16} />}
                    <div>
                      <strong>{shownEpisode.commitStatus === 'candidate' ? 'Bağımsız değerlendirme adayı oluşturuldu' : 'Öğrenme önerisi reddedildi'}</strong>
                      <span>{shownEpisode.commitStatus === 'candidate'
                        ? `Bu aday henüz etkin değil${shownEpisode.candidateSnapshotId ? ` · ${shownEpisode.candidateSnapshotId}` : ''}.`
                        : 'Bölüm izi korundu; önerilen kalıcı değişiklik uygulanmadı.'}</span>
                    </div>
                  </div>
                )}

                {detailError && <div className="tha-detail-error"><AlertCircle size={16} /><span>{detailError}</span><Button compact onClick={() => selectedId && void loadEpisode(selectedId)}>Tekrar oku</Button></div>}
                {detailLoading && !episode ? <div className="tha-inspector-loading"><Loader2 className="is-spinning" size={20} /> Bölümün tikleri okunuyor…</div> : (
                  <>
                    {shownEpisode.resultText && <blockquote className="tha-result"><span>BASKIN DÜŞÜNCEDEN ÇIKAN İFADE</span><p>{shownEpisode.resultText}</p>{shownEpisode.stopReason && <small>{shownEpisode.stopReason}</small>}</blockquote>}

                    <section className="tha-timeline">
                      <header>
                        <div><span className="tha-eyebrow">TİK OYNATICI</span><h3>{tick ? `Tik ${tick.index}` : 'Tik bekleniyor'}</h3></div>
                        <div className="tha-playback-controls">
                          <Button compact tone="ghost" aria-label="Önceki tik" disabled={!tick || tickCursor <= 0} icon={<ChevronLeft size={15} />} onClick={() => { setPlaying(false); setFollowLive(false); setTickCursor((value) => Math.max(0, value - 1)) }}>Önceki</Button>
                          <button type="button" className="tha-play-button" aria-label={playing ? 'Oynatmayı duraklat' : 'Tikleri oynat'} disabled={!episode?.ticks.length} onClick={() => { if (tickCursor >= (episode?.ticks.length ?? 1) - 1) setTickCursor(0); setFollowLive(false); setPlaying((value) => !value) }}>{playing ? <Pause size={17} /> : <Play size={17} />}</button>
                          <Button compact tone="ghost" aria-label="Sonraki tik" disabled={!tick || tickCursor >= (episode?.ticks.length ?? 1) - 1} icon={<ChevronRight size={15} />} onClick={() => { setPlaying(false); setFollowLive(false); setTickCursor((value) => Math.min((episode?.ticks.length ?? 1) - 1, value + 1)) }}>Sonraki</Button>
                          <button type="button" className={`tha-live-toggle ${followLive ? 'is-live' : ''}`} disabled={shownEpisode.state !== 'running'} onClick={() => { setFollowLive(true); setTickCursor(Math.max(0, (episode?.ticks.length ?? 1) - 1)) }}><i />Canlı</button>
                        </div>
                      </header>
                      <div className="tha-timeline__rail">
                        <input
                          type="range"
                          min={0}
                          max={Math.max(0, (episode?.ticks.length ?? 1) - 1)}
                          value={Math.min(tickCursor, Math.max(0, (episode?.ticks.length ?? 1) - 1))}
                          disabled={!episode?.ticks.length}
                          aria-label="Gösterilen tik"
                          onChange={(event) => { setPlaying(false); setFollowLive(false); setTickCursor(Number(event.target.value)) }}
                        />
                        <div><span>Başlangıç</span><strong>{episode?.ticks.length ? `${tickCursor + 1} / ${episode.ticks.length}` : '0 / 0'}</strong><span>{shownEpisode.maxTicks} tik sınırı</span></div>
                      </div>
                      {tick?.note && <p className="tha-tick-note"><Activity size={14} />{tick.note}</p>}
                    </section>

                    <AdaptiveRegionPanel episode={shownEpisode} tick={tick} branch={branch} />

                    {tick && afterMatrix ? (
                      <>
                        <section className="tha-signal-section">
                          <div className="tha-section-heading">
                            <div><span className="tha-eyebrow">SİNYAL DEĞİŞİMİ</span><h3>Vuruştan önce ve sonra</h3></div>
                            <small>{tick.activeNeuronLabel ?? tick.activeNeuronId ?? 'Etkin nöron açıklanmadı'}</small>
                          </div>
                          <div className="tha-heatmap-grid">
                            <ThoughtHeatmap matrix={tick.signalBefore} title="ÖNCE" caption="Nörona giren sinyal" />
                            <div className="tha-signal-arrow"><ArrowRight size={20} /><span>Nöron izi</span></div>
                            <ThoughtHeatmap matrix={afterMatrix} title="SONRA" caption={branch ? `${branch.label} dalından çıkan sinyal` : 'Nörondan çıkan sinyal'} />
                          </div>
                        </section>

                        <section className="tha-branch-section">
                          <div className="tha-section-heading">
                            <div><span className="tha-eyebrow">OLASI YOLLAR</span><h3>{tick.branches.length ? `${tick.branches.length} dal yarıştı` : 'Dal bilgisi bekleniyor'}</h3></div>
                            <small>En fazla sekiz olasılık birlikte görülebilir</small>
                          </div>
                          {tick.branches.length ? (
                            <div className="tha-branch-tabs" role="tablist" aria-label="Düşünce dalları">
                              {tick.branches.map((item) => (
                                <button
                                  key={item.id}
                                  type="button"
                                  role="tab"
                                  aria-selected={item.id === branch?.id}
                                  className={`${item.id === branch?.id ? 'is-active' : ''} ${item.selected ? 'is-winner' : ''} ${item.pruned ? 'is-pruned' : ''}`}
                                  onClick={() => setSelectedBranchId(item.id)}
                                >
                                  <span><b>{item.rank}</b>{item.label}</span>
                                  <strong>{formatRatio(item.probability)}</strong>
                                  {item.selected && <small>Seçilen yol</small>}
                                </button>
                              ))}
                            </div>
                          ) : <div className="tha-inline-empty"><GitBranch size={18} />Bu tik için dallanma kaydı gelmedi.</div>}
                        </section>

                        <section className="tha-analysis-grid">
                          <article className="tha-candidate-panel">
                            <div className="tha-section-heading"><div><span className="tha-eyebrow">ADAY HESABI</span><h3>Neden bu yol öne çıktı?</h3></div><small>{branch?.label ?? 'Tik geneli'}</small></div>
                            <CandidateBreakdown candidates={candidates} />
                          </article>
                          <aside className="tha-side-analysis">
                            <CoalitionCard coalition={coalition} />
                            <NewbornCard newborn={newborn} pressure={birthPressure} />
                          </aside>
                        </section>

                        <section className="tha-gauge-grid">
                          <MetricGauge label="Eşleşmeyen kısım" value={unmatched} tone="red" help="Mevcut nöronların açıklayamadığı sinyal payı." />
                          <MetricGauge label="Kenar kanıtı" value={edge} tone="blue" help="Bu geçiş yolunun geçmiş deneyim desteği." />
                          <MetricGauge label="Doğum baskısı" value={birthPressure} tone="amber" help="Yeni bir nöron ihtiyacının biriken gücü." />
                          <MetricGauge label="Yapısal puan" value={tick.structuralScore ?? shownEpisode.structuralScore ?? graph?.structuralEvaluation?.resultScore ?? null} tone="green" help="Kurulan olay grafiğinin dış yapısal sonuçla eşleşme puanı." />
                          <MetricGauge label="Güven" value={tick.trust ?? shownEpisode.trust} tone="green" help="Sonucun gözlenen kanıtlarla desteklenme düzeyi." />
                        </section>
                        <section className="tha-analysis-grid">
                          <StructuralOutcomeCard graph={graph} trust={shownEpisode.trust ?? null} />
                          <LanguageOutputCard graph={graph} />
                        </section>
                      </>
                    ) : (
                      <div className="tha-waiting-tick">
                        <div className="tha-empty-orbit"><Clock3 size={25} /></div>
                        <strong>İlk tik bekleniyor</strong>
                        <p>Bölüm sıraya alındıysa sinyal haritası ve dallar işlem başladığında burada belirecek.</p>
                      </div>
                    )}
                  </>
                )}
              </>
            )}
          </section>
        </section>

        <section className="tha-graph-panel">
          <header>
            <div className="tha-section-heading">
              <div><span className="tha-eyebrow">SEYREK BAĞLANTI HARİTASI</span><h3>{selectedId ? 'Seçili bölümün izlediği yollar' : 'Kalıcı güçlü yollar'}</h3></div>
              <small>Yalnız güçlü bağlantılar gösterilir; bütün çiftler çizilmez.</small>
            </div>
            <div className="tha-graph-controls">
              <div role="group" aria-label="Bağlantı haritası seviyesi">
                <button type="button" className={graphLevel === 'family' ? 'is-active' : ''} onClick={() => setGraphLevel('family')}>Aileler</button>
                <button type="button" className={graphLevel === 'neuron' ? 'is-active' : ''} onClick={() => setGraphLevel('neuron')}>Nöronlar</button>
              </div>
              <Button compact tone="ghost" disabled={graphLoading} icon={<RefreshCw size={14} className={graphLoading ? 'is-spinning' : ''} />} onClick={() => void loadGraph(graphLevel)}>Yenile</Button>
            </div>
          </header>
          {graphLoading && !graph ? <div className="tha-graph-empty"><Loader2 className="is-spinning" size={19} /> Bağlantılar okunuyor…</div> : graph ? (
            <div className="tha-graph-content">
              <div className="tha-graph-summary">
                <span><b>Toplam düğüm</b><strong>{graph.totalNodeCount.toLocaleString('tr-TR')}</strong><small>{graph.level === 'family' ? 'aile' : 'nöron'}</small></span>
                <span><b>Toplam kenar</b><strong>{graph.totalEdgeCount.toLocaleString('tr-TR')}</strong><small>öğrenilmiş geçiş</small></span>
                <span><b>Bu görünüm</b><strong>{graph.edges.length.toLocaleString('tr-TR')}</strong><small>{graph.truncated ? 'en güçlü kenar' : 'görünen kenar'}</small></span>
              </div>
              <div className="tha-edge-list">
                <div><span>Kaynak</span><span>Hedef</span><span>Bağ gücü</span></div>
                {graph.edges.slice(0, 12).map((item) => (
                  <article key={item.id}>
                    <span><i />{graphNodeNames.get(item.from) ?? item.from}</span>
                    <ArrowRight size={13} />
                    <span>{graphNodeNames.get(item.to) ?? item.to}</span>
                    <div><i style={{ width: `${scoreBar(item.weight) * 100}%` }} /></div>
                    <strong>{formatRatio(item.weight)}</strong>
                  </article>
                ))}
                {!graph.edges.length && <p>Bu seviyede güçlü bir bağlantı henüz oluşmadı.</p>}
              </div>
            </div>
          ) : (
            <div className="tha-graph-empty"><Network size={20} /><div><strong>Bağlantı özeti henüz kullanılamıyor</strong><span>{graphError ?? 'Servis ilk nöron ve aile yollarını oluşturduğunda burada görünecek.'}</span></div></div>
          )}
        </section>
      </main>
    </div>
  )
}
