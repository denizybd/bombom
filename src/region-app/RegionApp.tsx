import { Activity, Database, Layers3, RefreshCw, ScanSearch, ShieldCheck } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { SuiteMenu } from '../components/SuiteMenu'
import {
  activateReleasedRegionSnapshot,
  configureSemanticRegionProbes,
  enableAdaptiveRegionProposals,
  enableAdaptiveRegionShadow,
  enableReleasedAdaptiveRegions,
  loadRegionAtlasArtifact,
  loadRegionDashboard,
  loadRegionReleaseJob,
  queueRegionReleaseEvaluation,
} from './api'
import { RegionAtlasCanvas, regionHue } from './RegionAtlasCanvas'
import type {
  PlasticRegionAtlasDetail,
  PlasticRegionDashboard,
  PlasticRegionReleaseJobView,
  PlasticRegionView,
} from './types'
import './region-app.css'

const percent = (value: number) => `%${Math.round(Math.max(0, Math.min(1, value)) * 100)}`
const compact = (value: number) => value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })
const bytes = (value: number) => value < 1024 ? `${value} B` : value < 1024 ** 2
  ? `${(value / 1024).toFixed(1)} KiB`
  : `${(value / 1024 ** 2).toFixed(1)} MiB`

function modeLabel(mode: string) {
  return ({ off: 'Kapalı', shadow: 'Gölge gözlem', proposal: 'Öneri', active: 'Etkin' } as Record<string, string>)[mode] ?? mode
}

function reasonLabel(reason?: string | null) {
  const labels: Record<string, string> = {
    'policy-off': 'Yapılandırma davranış etkisini kapalı tutuyor.',
    'request-off': 'Bu çalışma bölge mekanizmasını istemedi.',
    'disabled-by-policy': 'Bölge mekanizması sistem politikasıyla kapalı.',
    'adaptive-representation-required': 'Önce verilerin 8/56 temsilinde yeniden kodlanması gerekiyor.',
    'verified-atlas-required': 'Doğrulanmış ve bu snapshot’a bağlı atlas başlığı bekleniyor.',
    'active-snapshot-required': 'Active için atlasın etkin model snapshot’ına ait olması gerekiyor.',
    'verified-region-release-required': 'Active için bağımsız holdout ve sham release kanıtı bekleniyor.',
    'release-manifest-absent': 'Henüz doğrulanmış release manifesti yok.',
  }
  if (!reason) return 'Çalışma uygunluğu nedeni henüz bildirilmedi.'
  if (labels[reason]) return labels[reason]
  if (reason.startsWith('release-') || reason.includes('chain')) return 'Release kanıt zinciri doğrulanamadı; davranış güvenli biçimde kapalı.'
  return `Güvenli kapanma nedeni: ${reason}`
}

function RegionCard({ region, capacity, selected, onSelect }: {
  region: PlasticRegionView
  capacity: number
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button type="button" className={`pra-region-card ${selected ? 'is-selected' : ''}`} onClick={onSelect} aria-pressed={selected}>
      <span className="pra-region-dot" style={{ backgroundColor: `hsl(${regionHue(region.id)} 62% 55%)` }} />
      <span><b>{region.id}</b><small>{region.lifecycle} · {compact(capacity)} sayfa</small></span>
      <em>{percent(region.evidence.cohesion)}</em>
    </button>
  )
}

export function RegionApp() {
  const [dashboard, setDashboard] = useState<PlasticRegionDashboard | null>(null)
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null)
  const [inspected, setInspected] = useState<PlasticRegionAtlasDetail | null>(null)
  const [inspectionLoadingId, setInspectionLoadingId] = useState<string | null>(null)
  const [setupOpen, setSetupOpen] = useState(false)
  const [mutationBusy, setMutationBusy] = useState<'shadow' | 'proposal' | 'active' | 'release' | 'snapshot' | 'probes' | null>(null)
  const [actionNotice, setActionNotice] = useState<string | null>(null)
  const [releaseJob, setReleaseJob] = useState<PlasticRegionReleaseJobView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const requestRef = useRef(0)
  const abortRef = useRef<AbortController | null>(null)
  const inspectionAbortRef = useRef<AbortController | null>(null)
  const mutationAbortRef = useRef<AbortController | null>(null)

  const refresh = useCallback(async () => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const requestId = requestRef.current + 1
    requestRef.current = requestId
    setLoading(true)
    try {
      const next = await loadRegionDashboard(undefined, controller.signal)
      if (requestRef.current !== requestId || controller.signal.aborted) return
      setDashboard(next)
      setInspected(null)
      setSelectedRegionId((current) => current && next.atlas?.regions.some((region) => region.id === current)
        ? current
        : next.atlas?.regions[0]?.id ?? null)
      setError(null)
    } catch (cause) {
      if (!controller.signal.aborted && requestRef.current === requestId) {
        setError(cause instanceof Error ? cause.message : 'Bölge atlası okunamadı.')
      }
    } finally {
      if (requestRef.current === requestId) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    return () => {
      abortRef.current?.abort()
      inspectionAbortRef.current?.abort()
      mutationAbortRef.current?.abort()
    }
  }, [refresh])

  const inspectArtifact = useCallback(async (artifactId: string) => {
    inspectionAbortRef.current?.abort()
    const controller = new AbortController()
    inspectionAbortRef.current = controller
    setInspectionLoadingId(artifactId)
    try {
      const detail = await loadRegionAtlasArtifact(artifactId, undefined, controller.signal)
      if (controller.signal.aborted) return
      setInspected(detail)
      setSelectedRegionId(detail.atlas.regions[0]?.id ?? null)
      setError(null)
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'Atlas geçmişi okunamadı.')
      }
    } finally {
      if (!controller.signal.aborted) setInspectionLoadingId(null)
    }
  }, [])

  const showLiveAtlas = useCallback(() => {
    inspectionAbortRef.current?.abort()
    setInspectionLoadingId(null)
    setInspected(null)
    setSelectedRegionId(dashboard?.atlas?.regions[0]?.id ?? null)
  }, [dashboard])

  const changeMode = useCallback(async (next: 'shadow' | 'proposal' | 'active') => {
    mutationAbortRef.current?.abort()
    const controller = new AbortController()
    mutationAbortRef.current = controller
    setMutationBusy(next)
    setActionNotice(null)
    try {
      if (next === 'shadow') await enableAdaptiveRegionShadow(undefined, controller.signal)
      else if (next === 'proposal') await enableAdaptiveRegionProposals(undefined, controller.signal)
      else await enableReleasedAdaptiveRegions(undefined, controller.signal)
      if (controller.signal.aborted) return
      setSetupOpen(false)
      setActionNotice(next === 'shadow'
        ? 'Ayrı 8/56 soyu gölge gözleme hazırlandı. Eski snapshot ve nöronlar değiştirilmedi.'
        : next === 'proposal'
          ? 'Öneri üretimi açıldı. Yeni atlaslar yalnız aday olarak kalacak; kendiliğinden etkinleşmeyecek.'
          : 'Bağımsız release taşıyan etkin snapshot için bölgesel yönlendirme açıkça etkinleştirildi.')
      await refresh()
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'Bölge çalışma kipi değiştirilemedi.')
      }
    } finally {
      if (!controller.signal.aborted) setMutationBusy(null)
    }
  }, [refresh])

  const queueRelease = useCallback(async () => {
    const artifactId = inspected?.artifact?.artifactId
    if (!artifactId || !inspected.proposal) return
    mutationAbortRef.current?.abort()
    const controller = new AbortController()
    mutationAbortRef.current = controller
    setMutationBusy('release')
    setActionNotice(null)
    try {
      const job = await queueRegionReleaseEvaluation(artifactId, undefined, controller.signal)
      if (controller.signal.aborted) return
      setReleaseJob(job)
      setActionNotice(job.duplicate
        ? 'Aynı önerinin mevcut bağımsız sınaması gösteriliyor.'
        : 'Bağımsız sınama güvenli iş kuyruğuna alındı. Bu işlem Ollama kullanmaz ve otomatik etkinleştirme yapmaz.')
      setError(null)
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'Bağımsız sınama başlatılamadı.')
      }
    } finally {
      if (!controller.signal.aborted) setMutationBusy(null)
    }
  }, [inspected])

  const approveReleasedSnapshot = useCallback(async (requestedSnapshotId?: string) => {
    const snapshotId = requestedSnapshotId ?? (releaseJob?.releaseResult?.released
      ? releaseJob.releaseResult.snapshotId
      : null)
    if (!snapshotId) return
    mutationAbortRef.current?.abort()
    const controller = new AbortController()
    mutationAbortRef.current = controller
    setMutationBusy('snapshot')
    setActionNotice(null)
    try {
      await activateReleasedRegionSnapshot(snapshotId, undefined, controller.signal)
      if (controller.signal.aborted) return
      setActionNotice('Kanıtlanmış aday snapshot insan onayıyla etkin model yapıldı. Bölge yönlendirmesi hâlâ kapalıdır; son aşamayı ayrıca açabilirsiniz.')
      setError(null)
      await refresh()
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'Kanıtlanmış aday snapshot kabul edilemedi.')
      }
    } finally {
      if (!controller.signal.aborted) setMutationBusy(null)
    }
  }, [refresh, releaseJob])

  const changeSemanticProbes = useCallback(async (enabled: boolean) => {
    mutationAbortRef.current?.abort()
    const controller = new AbortController()
    mutationAbortRef.current = controller
    setMutationBusy('probes')
    setActionNotice(null)
    try {
      await configureSemanticRegionProbes(enabled, undefined, controller.signal)
      if (controller.signal.aborted) return
      setActionNotice(enabled
        ? 'Salt-okunur anlam denetimi açıldı. Sonraki uygun eğitim çevrimi dondurulmuş 288 vakayı gölge kipte ölçecek; canlı rota değişmeyecek.'
        : 'Yeni anlam denetimi çevrimleri kapatıldı. Eski immutable ölçüm kayıtları denetim için korunuyor.')
      setError(null)
      await refresh()
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Anlam denetimi ayarlanamadı.')
    } finally {
      if (!controller.signal.aborted) setMutationBusy(null)
    }
  }, [refresh])

  useEffect(() => {
    if (!releaseJob || ['complete', 'failed', 'cancelled'].includes(releaseJob.state)) return
    let alive = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const next = await loadRegionReleaseJob(releaseJob.id)
        if (!alive) return
        if (next) {
          setReleaseJob(next)
          if (['complete', 'failed', 'cancelled'].includes(next.state)) {
            setActionNotice(next.state === 'complete' && next.releaseResult?.released
              ? 'Bağımsız sınama geçti. Yeni atlas yalnız aday snapshotta; etkinleşmek için insan onayı bekliyor.'
              : next.state === 'complete'
                ? `Bağımsız sınama tamamlandı fakat öneri yayınlanmadı${next.releaseResult?.reasonCodes.length ? `: ${next.releaseResult.reasonCodes.join(', ')}` : '.'}`
              : next.message ?? 'Bağımsız sınama güvenli biçimde sonlandı.')
            await refresh()
            return
          }
        } else {
          setReleaseJob((current) => current ? {
            ...current,
            state: 'cancelled',
            message: 'Release işi artık bulunamıyor; otomatik yeniden deneme durduruldu.',
          } : null)
          return
        }
      } catch (cause) {
        if (alive) setError(cause instanceof Error ? cause.message : 'Release işi izlenemedi.')
      }
      if (alive) timer = window.setTimeout(() => void poll(), 2_000)
    }
    timer = window.setTimeout(() => void poll(), 1_000)
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [refresh, releaseJob])

  const capacities = useMemo(() => {
    const result = new Map<string, number>()
    for (const page of (inspected?.atlas ?? dashboard?.atlas)?.pages ?? []) {
      for (const membership of page.memberships) {
        result.set(membership.regionId, (result.get(membership.regionId) ?? 0) + membership.weight)
      }
    }
    return result
  }, [dashboard, inspected])
  const atlas = inspected?.atlas ?? dashboard?.atlas ?? null
  const selected = atlas?.regions.find((region) => region.id === selectedRegionId) ?? null
  const status = dashboard?.status
  const releaseEligible = Boolean(status?.release.artifactId && status.release.fingerprint)
  const behaviorApplied = status?.effectiveMode === 'active' && status.authority === 'active' && status.routingEligible
  const adaptiveConfigured = status?.representationProfile === 'adaptive-eighth-id-v1'
  const hasShadowEvidence = status?.readiness.shadowEvidenceAvailable === true
  const selectedProbeItems = useMemo(() => dashboard?.probes.items.filter((item) =>
    selectedRegionId ? item.target?.featureId === selectedRegionId : true) ?? [], [dashboard, selectedRegionId])

  return (
    <main className="pra-shell">
      <header className="pra-header">
        <SuiteMenu currentApp="regions">
          <span className="pra-mark"><Layers3 size={22} /></span>
        </SuiteMenu>
        <div><span>UYARLANABİLİR PLASTİK BÖLGE ATLASI</span><h1>Nöronun iç çalışma alanları</h1></div>
        <button type="button" className="pra-refresh" onClick={() => void refresh()} disabled={loading} aria-busy={loading}>
          <RefreshCw size={17} className={loading ? 'is-spinning' : ''} /> Yenile
        </button>
      </header>

      {error && <div className="pra-alert" role="alert"><b>Bağlantı kurulamadı</b><span>{error}</span></div>}

      <section className="pra-status-grid">
        <article><Activity /><span><small>Etkili çalışma kipi</small><b>{modeLabel(status?.effectiveMode ?? 'off')}</b></span><em title={status?.effectiveModeReason}>Ayarlı: {modeLabel(status?.configuredMode ?? 'off')} · {behaviorApplied ? 'davranışa uygulanıyor' : status?.effectiveMode === 'shadow' ? 'yalnız gözlem' : status?.effectiveMode === 'proposal' ? 'yalnız öneri' : 'davranış kapalı'}</em></article>
        <article><Layers3 /><span><small>Öğrenilmiş bölge</small><b>{atlas?.stats.regionCount ?? 0}</b></span><em>{atlas ? `${status?.authority === 'active' ? 'etkin model' : status?.authority === 'candidate-research' ? 'araştırma adayı' : 'inceleme'} · revizyon ${atlas.revision}` : 'Atlas bekleniyor'}</em></article>
        <article><ScanSearch /><span><small>Serbest kapasite</small><b>{compact(atlas?.stats.freeMass ?? 56)}</b></span><em>56 sayfa eşdeğeri</em></article>
        <article><Database /><span><small>Kanıt artifacti</small><b>{status?.counts.totalArtifacts ?? 0}</b></span><em>{status?.representationProfile ?? 'legacy'}</em></article>
        <article><ShieldCheck /><span><small>Active release uygunluğu</small><b>{releaseEligible ? 'Doğrulandı' : 'Bekleniyor'}</b></span><em title={status?.release.reason ?? undefined}>{releaseEligible ? `${status?.limits.releaseMinimumCases ?? 0}+ holdout vaka · ${status?.limits.releaseMinimumSourceGroups ?? 0}+ kaynak` : reasonLabel(status?.release.reason)}</em></article>
      </section>

      <div className={`pra-runtime-note ${behaviorApplied ? 'is-active' : ''}`} role="status">
        <ShieldCheck size={18} />
        <div>
          <b>{behaviorApplied ? 'Doğrulanmış atlas davranışa uygulanıyor' : status?.authority === 'candidate-research' ? 'Gösterilen atlas bir araştırma adayı; canlı model değil' : status?.effectiveMode === 'shadow' ? 'Atlas yalnız gözlem topluyor' : status?.effectiveMode === 'proposal' ? 'Atlas yalnız geri alınabilir öneri üretiyor' : 'Atlas davranışı değiştirmiyor'}</b>
          <span>{behaviorApplied ? `Release ${status?.release.releaseId ?? status?.release.fingerprint?.slice(0, 18)} etkin snapshot ile eşleşti.` : status?.authority === 'candidate-research' ? 'Bu harita ancak doğrulanmış release ve ayrı insan onayından sonra etkin snapshot olabilir.' : reasonLabel(status?.effectiveModeReason)}</span>
        </div>
      </div>

      {actionNotice && <div className="pra-action-notice" role="status">{actionNotice}</div>}

      {!behaviorApplied && (!adaptiveConfigured || status?.configuredMode === 'off') && (
        <section className="pra-optin" aria-labelledby="pra-optin-title">
          <div>
            <span>YENİ VE İSTEĞE BAĞLI KATMAN</span>
            <h2 id="pra-optin-title">8/56 gölge gözlem soyunu hazırla</h2>
            <p>Bu seçim mevcut 16/48 nöronlarını, ailelerini, makrolarını veya snapshotlarını dönüştürmez. Ham kaynaklar sonraki uygun eğitim çevriminde ayrı bir temsil soyunda yeniden kodlanır; gölge kip canlı düşünce rotasını değiştirmez.</p>
          </div>
          {!setupOpen ? (
            <button type="button" onClick={() => setSetupOpen(true)}>Hazırlığı incele</button>
          ) : (
            <div className="pra-optin-confirm">
              <ul>
                <li>Eski kayıtlar yerinde ve replay edilebilir kalır.</li>
                <li>Otomatik eğitim açıksa sonraki uygun çevrim 8/56 kanıtı toplamaya başlayabilir.</li>
                <li>Etkin yönlendirme için daha sonra bağımsız release kanıtı gerekir.</li>
              </ul>
              <div><button type="button" onClick={() => void changeMode('shadow')} disabled={mutationBusy !== null}>{mutationBusy === 'shadow' ? 'Hazırlanıyor…' : 'Gölge hazırlığını aç'}</button><button type="button" className="is-quiet" onClick={() => setSetupOpen(false)} disabled={mutationBusy !== null}>Vazgeç</button></div>
            </div>
          )}
        </section>
      )}

      {adaptiveConfigured && status?.configuredMode === 'shadow' && (
        <section className="pra-optin is-proposal" aria-labelledby="pra-proposal-title">
          <div>
            <span>İKİNCİ GÜVENLİ AŞAMA</span>
            <h2 id="pra-proposal-title">Kanıttan geri alınabilir atlas önerileri üret</h2>
            <p>{hasShadowEvidence ? 'Gölge kanıtı bulundu. Öneri kipi yalnız değişmez adaylar üretir; canlı yönlendirmeyi ve kalıcı nöronları değiştirmez.' : 'Önce Otonom Eğitim ekranında en az bir gölge gözlem çevrimi tamamlanmalı. Böylece öneri, boş bir haritadan değil gerçek birikimli kanıttan doğar.'}</p>
          </div>
          {hasShadowEvidence
            ? <button type="button" onClick={() => void changeMode('proposal')} disabled={mutationBusy !== null}>{mutationBusy === 'proposal' ? 'Açılıyor…' : 'Öneri üretimini aç'}</button>
            : <a href="/otonom-egitim/">Eğitim merkezine git</a>}
        </section>
      )}

      {releaseEligible && status?.authority === 'candidate-research' && status.activationCandidateSnapshotId && (
        <section className="pra-optin is-active-ready" aria-labelledby="pra-candidate-title">
          <div>
            <span>AYRI ADAY MODEL · İNSAN ONAYI</span>
            <h2 id="pra-candidate-title">Release taşıyan aday snapshotı model olarak kabul et</h2>
            <p>Haritadaki atlas henüz canlı model değildir; araştırma adayıdır. Bu onay snapshotı etkin model yapar, fakat bölge yönlendirmesini açmaz. Eski snapshot denetim geçmişinde korunur; 8/56 geri dönüşü exact release yenilemesi ister.</p>
          </div>
          <button type="button" onClick={() => void approveReleasedSnapshot(status.activationCandidateSnapshotId ?? undefined)} disabled={mutationBusy !== null}>{mutationBusy === 'snapshot' ? 'Doğrulanıyor…' : 'Aday snapshotı kabul et'}</button>
        </section>
      )}

      {releaseEligible && status?.authority === 'active' && status?.configuredMode !== 'active' && (
        <section className="pra-optin is-active-ready" aria-labelledby="pra-active-title">
          <div>
            <span>SON İNSAN ONAYI</span>
            <h2 id="pra-active-title">Doğrulanmış atlası canlı yönlendirmede kullan</h2>
            <p>Şu an etkin model snapshotı, bağımsız holdout ve sham zincirini taşıyor. Bu düğme yalnız bölge katkısını açar; eski snapshotlar yerinde kalır, fakat 8/56 geri dönüşü yeni bir release yenilemesi olmadan etkinleştirilmez.</p>
          </div>
          <button type="button" onClick={() => void changeMode('active')} disabled={mutationBusy !== null}>{mutationBusy === 'active' ? 'Doğrulanıyor…' : 'Doğrulanmış atlası etkin kullan'}</button>
        </section>
      )}

      {adaptiveConfigured && status?.available && (
        <section className="pra-optin is-probes" aria-labelledby="pra-probe-optin-title">
          <div>
            <span>YENİ · SALT-OKUNUR YORUM KATMANI</span>
            <h2 id="pra-probe-optin-title">Dış anlam hipotezlerini {status.semanticProbesEnabled ? 'ölçmeye hazır' : 'ölç'}</h2>
            <p>Bu ayrı tanı paketi 144 olay birleşiminin iki yüzeyini train/validation/test olarak sınar. Ollama kullanmaz; nöron, atlas, aile, release veya yönlendirme ağırlığı yazmaz. Çalışması ek süre gerektirir.</p>
          </div>
          <button type="button" onClick={() => void changeSemanticProbes(!status.semanticProbesEnabled)} disabled={mutationBusy !== null}>{mutationBusy === 'probes' ? 'Kaydediliyor…' : status.semanticProbesEnabled ? 'Yeni ölçümleri durdur' : 'Sonraki çevrimde ölç'}</button>
        </section>
      )}

      {inspected && (
        <div className="pra-inspection-note" role="status">
          <ScanSearch size={18} />
          <div>
            <b>Geçici/geçmiş atlas yalnız inceleme için açık</b>
            <span>
              {inspected.proposal
                ? `${inspected.proposal.action ?? 'değişiklik'} önerisi · ${inspected.proposal.eligible ? 'kanıt kapısına aday' : 'henüz uygun değil'}`
                : 'Değişmez atlas başlığı'}; bu seçim canlı yönlendirmeye uygulanmıyor.
            </span>
          </div>
          <button type="button" onClick={showLiveAtlas}>{status?.authority === 'active' ? 'Canlı atlasa dön' : 'Aday atlas görünümüne dön'}</button>
        </div>
      )}

      {inspected?.proposal && inspected.artifact && (
        <section className="pra-release-control" aria-labelledby="pra-release-title">
          <ShieldCheck size={20} />
          <div>
            <span>YENİ BAĞIMSIZ KANIT KAPISI</span>
            <h2 id="pra-release-title">Bu öneriyi gerçek motorla karşılaştır</h2>
            <p>Taban, aday ve aynı büyüklükte sahte atlas; dondurulmuş holdout üzerinde aynı seedlerle ölçülür. Varsayılan kapı yaklaşık 450 kısa düşünce koşusudur. Ollama ve istemciden gelen puanlar kullanılmaz; başarı bile otomatik etkinleştirme yapmaz.</p>
            {releaseJob && (
              <div className={`pra-release-job is-${releaseJob.state}`}>
                <b>{releaseJob.state === 'waiting'
                  ? 'Sırada'
                  : releaseJob.state === 'active'
                    ? 'Sınanıyor'
                    : releaseJob.state === 'complete' && releaseJob.releaseResult?.released
                      ? 'Geçti · insan onayı bekliyor'
                      : releaseJob.state === 'complete'
                        ? 'Tamamlandı · yayınlanmadı'
                        : 'Güvenli biçimde durdu'}</b>
                <span>%{Math.round(releaseJob.progress * 100)} · {releaseJob.id.slice(0, 18)}…</span>
                {releaseJob.message && <small>{releaseJob.message}</small>}
                {releaseJob.releaseResult && (
                  <div className={`pra-release-result ${releaseJob.releaseResult.released ? 'is-passed' : 'is-rejected'}`}>
                    <strong>{releaseJob.releaseResult.released ? 'Bağımsız kapılar geçti' : 'Öneri release kapısını geçmedi'}</strong>
                    {releaseJob.releaseResult.reasonCodes.length > 0 && <small>{releaseJob.releaseResult.reasonCodes.join(' · ')}</small>}
                    {releaseJob.releaseResult.metrics.conservativeHoldoutGain !== null && (
                      <small>Holdout kazancı {compact(releaseJob.releaseResult.metrics.conservativeHoldoutGain)} · sham {compact(releaseJob.releaseResult.metrics.conservativeShamGain ?? 0)}</small>
                    )}
                    {releaseJob.releaseResult.auditArtifactId && <small>Denetim: {releaseJob.releaseResult.auditArtifactId.slice(0, 22)}…</small>}
                  </div>
                )}
              </div>
            )}
          </div>
          {releaseJob?.state === 'complete' && releaseJob.releaseResult?.released && releaseJob.releaseResult.snapshotId
            ? <button type="button" onClick={() => void approveReleasedSnapshot()} disabled={mutationBusy !== null}>{mutationBusy === 'snapshot' ? 'Doğrulanıyor…' : 'Aday snapshotı kabul et'}</button>
            : <button type="button" onClick={() => void queueRelease()} disabled={mutationBusy !== null || releaseJob?.state === 'active' || releaseJob?.state === 'waiting' || (releaseJob?.state === 'complete' && releaseJob.releaseResult?.released === false)}>{mutationBusy === 'release' ? 'Sıraya alınıyor…' : releaseJob?.state === 'complete' ? 'Sınama sonucu kaydedildi' : releaseJob ? 'Aynı sınamayı göster' : 'Bağımsız sınamayı başlat'}</button>}
        </section>
      )}

      <section className="pra-grid">
        <RegionAtlasCanvas atlas={atlas} selectedRegionId={selectedRegionId} onSelectRegion={setSelectedRegionId} />
        <aside className="pra-side">
          <section className="pra-panel">
            <header><span>GİZİL BÖLGELER</span><small>{atlas?.regions.length ?? 0} kayıt</small></header>
            <div className="pra-region-list">
              {(atlas?.regions ?? []).map((region) => (
                <RegionCard key={region.id} region={region} capacity={capacities.get(region.id) ?? 0} selected={region.id === selectedRegionId} onSelect={() => setSelectedRegionId(region.id)} />
              ))}
              {!atlas?.regions.length && <div className="pra-empty"><Layers3 /><b>Henüz öğrenilmiş bölge yok</b><span>Geçici öneriler üretildiğinde aşağıdaki kanıt geçmişinden ayrı bir inceleme görünümü olarak açılabilir.</span></div>}
            </div>
          </section>

          <section className="pra-panel pra-detail">
            <header><span>SEÇİLİ BÖLGE</span><ShieldCheck size={17} /></header>
            {selected ? (
              <>
                <h3>{selected.id}</h3>
                <dl>
                  <div><dt>Yaşam döngüsü</dt><dd>{selected.lifecycle}</dd></div>
                  <div><dt>Efektif kapasite</dt><dd>{compact(capacities.get(selected.id) ?? 0)}</dd></div>
                  <div><dt>Bağımsız kaynak</dt><dd>{selected.evidence.sourceGroupCount}</dd></div>
                  <div><dt>Gözlem</dt><dd>{selected.evidence.observationCount.toLocaleString('tr-TR')}</dd></div>
                  <div><dt>Tutarlılık</dt><dd>{percent(selected.evidence.cohesion)}</dd></div>
                  <div><dt>Holdout katkısı</dt><dd>{compact(selected.evidence.holdoutUtility)}</dd></div>
                  <div><dt>Sham katkısı</dt><dd>{compact(selected.evidence.shamUtility)}</dd></div>
                </dl>
                <p>Bu kimlik semantik bir isim değildir. “Kip”, “kişi” veya “matematik” gibi açıklamalar yalnız ayrı probe kanıtıyla gösterilir.</p>
              </>
            ) : <p>İncelemek için haritadan veya listeden bir bölge seçin.</p>}
          </section>
        </aside>
      </section>

      <section className="pra-panel pra-history">
        <header><span>DEĞİŞMEZ KANIT GEÇMİŞİ</span><small>Son {dashboard?.artifacts.length ?? 0} artifact</small></header>
        <div className="pra-history-table">
          {(dashboard?.artifacts ?? []).map((artifact) => {
            const inspectable = artifact.kind === 'plastic-region-atlas-head' || artifact.kind === 'plastic-region-atlas-proposal'
            return inspectable ? (
            <button
              type="button"
              key={artifact.artifactId}
              className={inspected?.artifact?.artifactId === artifact.artifactId ? 'is-selected' : ''}
              onClick={() => void inspectArtifact(artifact.artifactId)}
              disabled={inspectionLoadingId === artifact.artifactId}
              aria-pressed={inspected?.artifact?.artifactId === artifact.artifactId}
            >
              <span><b>{artifact.kind}</b><small>{artifact.artifactId.slice(0, 18)}…</small></span>
              <em>{bytes(artifact.payloadBytes)}</em>
              <time>{inspectionLoadingId === artifact.artifactId ? 'Açılıyor…' : artifact.createdAt ? new Date(artifact.createdAt).toLocaleString('tr-TR') : '—'}</time>
            </button>
            ) : (
              <div key={artifact.artifactId}>
                <span><b>{artifact.kind}</b><small>{artifact.artifactId.slice(0, 18)}…</small></span>
                <em>{bytes(artifact.payloadBytes)}</em>
                <time>{artifact.createdAt ? new Date(artifact.createdAt).toLocaleString('tr-TR') : '—'}</time>
              </div>
            )
          })}
          {!dashboard?.artifacts.length && <div className="pra-history-empty">Henüz atlas gözlemi veya önerisi kaydedilmedi.</div>}
        </div>
      </section>

      <section className="pra-panel pra-probes" aria-labelledby="pra-probes-title">
        <header>
          <span id="pra-probes-title">DIŞ ANLAM HİPOTEZLERİ</span>
          <small>{dashboard?.probes.available ? `${dashboard.probes.items.length} hipotez` : 'Henüz ölçüm yok'}</small>
        </header>
        <div className="pra-probe-explainer">
          <ScanSearch size={18} />
          <p>Bu etiketler bölgenin adı değildir. Dondurulmuş olay-grafiği testlerinin, canlı rotaya dokunmadan belirli bir bölge tepkisiyle korelasyonunu ölçer; eğitim, release veya yönlendirme yetkisi taşımaz.</p>
        </div>
        <div className="pra-probe-grid">
          {selectedProbeItems.map((item) => (
            <article key={item.hypothesis.hypothesisId} className={`is-${item.status}`}>
              <div><span>{item.hypothesis.namespace}</span><em>{item.status === 'calibrated' ? 'kalibre' : item.status === 'provisional' ? 'geçici' : 'bilinmiyor'}</em></div>
              <h3>{item.hypothesis.label}</h3>
              <p>{item.target ? `${item.target.featureId} tepkisiyle sınanıyor` : 'Henüz güvenilir hedef bölge seçilemedi'}.</p>
              <dl>
                <div><dt>Gözlem</dt><dd>{item.observationCount.toLocaleString('tr-TR')}</dd></div>
                <div><dt>Bağımsız kaynak</dt><dd>{item.independentSourceCount}</dd></div>
                {item.evidence && <>
                  <div><dt>Brier ↓</dt><dd>{item.evidence.targetBrier.toFixed(3)}</dd></div>
                  <div><dt>ECE ↓</dt><dd>{item.evidence.targetEce.toFixed(3)}</dd></div>
                  <div><dt>Tabana karşı %95 alt sınır</dt><dd>{item.evidence.gainVsBaseline.low.toFixed(3)}</dd></div>
                  <div><dt>Sham’a karşı %95 alt sınır</dt><dd>{item.evidence.gainVsSham.low.toFixed(3)}</dd></div>
                </>}
              </dl>
              {item.evidence?.reasons.length ? <small>{item.evidence.reasons.join(' · ')}</small> : null}
            </article>
          ))}
          {!selectedProbeItems.length && (
            <div className="pra-probe-empty">
              <b>{selectedRegionId ? 'Seçili bölge için kanıtlı anlam hipotezi yok' : 'Semantik probe henüz çalıştırılmadı'}</b>
              <span>Bu boşluk bir anlam atanmadığını dürüstçe gösterir; sistem varsayımsal etiket üretmez.</span>
            </div>
          )}
        </div>
        {(dashboard?.probes.legacySkeletonCount ?? 0) > 0 && <small className="pra-legacy-probe">{dashboard?.probes.legacySkeletonCount} eski boş probe kaydı yalnız tarihsel denetim için tutuluyor; yeni kanıta çevrilmedi.</small>}
      </section>
    </main>
  )
}
