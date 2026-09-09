import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Database,
  GitBranch,
  RefreshCw,
  Route,
  Search,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { SuiteMenu } from '../components/SuiteMenu'
import { loadConceptMacros } from './api'
import type { ConceptEnvelope, ConceptMacroSummary } from './types'

const LIFECYCLES = [
  ['all', 'Tümü'],
  ['observing', 'Gözleniyor'],
  ['eligible-proposal', 'Önerilebilir'],
  ['virtual-provisional', 'Sanal makro'],
  ['shadow-validated', 'Shadow doğrulandı'],
  ['active', 'Etkin'],
  ['deprecated', 'Emekli'],
] as const

function percent(value: number | null): string {
  return value === null ? '—' : `%${Math.round(Math.max(0, Math.min(1, value)) * 100)}`
}

function short(value: string | null | undefined, size = 15): string {
  if (!value) return '—'
  return value.length > size ? `${value.slice(0, size)}…` : value
}

function Metric({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return <div className={`concept-metric ${good ? 'is-good' : ''}`}><span>{label}</span><strong>{value}</strong></div>
}

function MacroPath({ macro }: { macro: ConceptMacroSummary }) {
  return (
    <div className="macro-path" aria-label={`${macro.expansionLength} adımlık açık yol`}>
      {macro.orderedExpansion.map((entry, index) => (
        <span className="macro-path__pair" key={`${entry.position}-${entry.memberKey ?? entry.familyId ?? entry.macroId}`}>
          <span className="macro-path__node" title={entry.memberKey ?? entry.familyId ?? entry.macroId ?? undefined}>
            <small>{entry.macroId ? 'MAKRO' : entry.familyId ? 'AİLE' : 'NÖRON'}</small>
            <b>{entry.label || short(entry.familyId ?? entry.memberKey ?? entry.macroId, 13)}</b>
          </span>
          {index < macro.orderedExpansion.length - 1 && <ArrowRight size={14} aria-hidden="true" />}
        </span>
      ))}
    </div>
  )
}

export function ConceptApp() {
  const [data, setData] = useState<ConceptEnvelope | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [lifecycle, setLifecycle] = useState('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const requestRef = useRef<AbortController | null>(null)

  const refresh = useCallback((background = false) => {
    requestRef.current?.abort()
    const controller = new AbortController()
    requestRef.current = controller
    if (!background) setLoading(true)
    loadConceptMacros(controller.signal)
      .then((value) => {
        setData(value)
        setError(null)
        setSelectedId((current) => current && value.macros.some((macro) => macro.id === current) ? current : value.macros[0]?.id ?? null)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Kavram verileri alınamadı.')
      })
      .finally(() => {
        if (requestRef.current === controller) {
          requestRef.current = null
          setLoading(false)
        }
      })
  }, [])

  useEffect(() => {
    refresh()
    const timer = window.setInterval(() => refresh(true), 5_000)
    return () => { requestRef.current?.abort(); window.clearInterval(timer) }
  }, [refresh])

  const filtered = useMemo(() => (data?.macros ?? []).filter((macro) => {
    if (lifecycle !== 'all' && macro.lifecycle !== lifecycle) return false
    if (!query.trim()) return true
    const key = `${macro.label} ${macro.id} ${macro.orderedExpansion.map((entry) => `${entry.memberKey} ${entry.familyId}`).join(' ')}`.toLocaleLowerCase('tr-TR')
    return key.includes(query.trim().toLocaleLowerCase('tr-TR'))
  }), [data, lifecycle, query])
  const selected = filtered.find((macro) => macro.id === selectedId) ?? filtered[0] ?? null

  return (
    <main className="concept-shell">
      <header className="concept-topbar">
        <SuiteMenu currentApp="concepts">
          <span className="concept-brand"><Route size={21} /><span><b>BOMBOM</b><small>KAVRAM FİDANLIĞI</small></span></span>
        </SuiteMenu>
        <div className="concept-mode"><span className="concept-live" />{data?.mode ?? 'proposal-shadow'} · ham yollar korunuyor</div>
        <button className="concept-refresh" type="button" onClick={() => refresh()} disabled={loading}><RefreshCw size={16} className={loading ? 'is-spinning' : ''} /> Yenile</button>
      </header>

      <section className="concept-hero">
        <div><p className="eyebrow">GERİ ALINABİLİR KAVRAM DOĞUMU</p><h1>Tekrarlanan yolları önce gözle, sonra kısalt.</h1><p>Bir aday hiçbir ham nöronu silmez. Hem çıktı hem de hızlı iz, baskı ve sonraki rota yapısı açık yolla eşdeğer bulunmadan yönlendirmeye katılmaz.</p></div>
        <div className="concept-safety"><ShieldCheck size={24} /><div><b>Güvenli sözleşme</b><span>Bağımsız kaynak · sıra duyarlılığı · açılabilir replay · holdout kapısı</span></div></div>
      </section>

      <section className="concept-stats">
        <Metric label="Yol gözlemi" value={(data?.stats.observationCount ?? 0).toLocaleString('tr-TR')} />
        <Metric label="Makro adayı" value={(data?.stats.totalMacros ?? 0).toLocaleString('tr-TR')} />
        <Metric label="Sanal / shadow" value={(data?.stats.virtualCount ?? 0).toLocaleString('tr-TR')} />
        <Metric label="Etkin kısayol" value={(data?.stats.activeCount ?? 0).toLocaleString('tr-TR')} good={(data?.stats.activeCount ?? 0) > 0} />
        <Metric label="Yapısal eşdeğer" value={(data?.stats.structuralValidatedCount ?? 0).toLocaleString('tr-TR')} good={(data?.stats.structuralValidatedCount ?? 0) > 0} />
        <Metric label="Shadow karşılaştırması" value={(data?.stats.shadowEvaluationCount ?? 0).toLocaleString('tr-TR')} />
        <Metric label="Geri dönüş korumalı" value={(data?.stats.fallbackProtectedCount ?? 0).toLocaleString('tr-TR')} good />
      </section>

      {error && <section className="concept-empty is-error"><Database size={26} /><div><b>Yerel kavram servisine ulaşılamadı</b><p>{error}</p></div></section>}
      {!error && !data?.available && <section className="concept-empty"><Sparkles size={26} /><div><b>Fidanlık henüz boş</b><p>İlk yeni eğitim çevrimi yol gözlemlerini ve proposal-only makroları burada oluşturacak.</p></div></section>}

      <section className="concept-workspace">
        <aside className="concept-list-panel">
          <div className="concept-toolbar">
            <label><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Yol veya aile ara" /></label>
            <div className="concept-tabs">{LIFECYCLES.map(([id, label]) => <button type="button" key={id} className={lifecycle === id ? 'is-active' : ''} onClick={() => setLifecycle(id)}>{label}</button>)}</div>
          </div>
          <div className="concept-list">
            {filtered.map((macro) => (
              <button className={`concept-card ${selected?.id === macro.id ? 'is-selected' : ''}`} key={macro.id} type="button" onClick={() => setSelectedId(macro.id)}>
                <div className="concept-card__head"><span className={`lifecycle lifecycle--${macro.lifecycle}`}>{macro.lifecycle}</span><small>{macro.kind === 'equivalence-concept' ? 'Eşdeğerlik' : `${macro.expansionLength} adım`}</small></div>
                <b>{macro.label}</b>
                <MacroPath macro={macro} />
                <div className="concept-card__metrics"><span>{macro.occurrenceCount} tekrar</span><span>{macro.sourceGroupCount} kaynak grubu</span><span>kazanç {macro.compressionGain?.toFixed(2) ?? '—'}</span></div>
              </button>
            ))}
            {!filtered.length && data?.available && <p className="concept-no-match">Bu filtrede bir makro bulunamadı.</p>}
          </div>
        </aside>

        <article className="concept-detail">
          {selected ? <>
            <div className="concept-detail__head"><div><p className="eyebrow">SEÇİLİ KOMPOZİT NÖRON</p><h2>{selected.label}</h2><code>{short(selected.id, 38)}</code></div><span className={`lifecycle lifecycle--${selected.lifecycle}`}>{selected.lifecycle}</span></div>
            <section className="detail-block"><h3><GitBranch size={17} /> Geri açılabilir yol</h3><MacroPath macro={selected} /><p>Makro başarısız olduğunda bu sıralı açılım aynen çalıştırılır. Ham nöronlar silinmez.</p></section>
            <section className="detail-grid">
              <Metric label="Tamamlama güveni" value={percent(selected.completionProbability)} good={(selected.completionProbability ?? 0) >= .8} />
              <Metric label="Yol uyumu" value={percent(selected.pathCohesion)} good={(selected.pathCohesion ?? 0) >= .82} />
              <Metric label="Shadow sadakati" value={percent(selected.shadowFidelity)} good={(selected.shadowFidelity ?? 0) >= .95} />
              <Metric label="Yanlış tetiklenme" value={percent(selected.falseActivationRate)} good={(selected.falseActivationRate ?? 1) <= .05} />
              <Metric label="Yapısal eşdeğerlik" value={selected.structuralEquivalencePassed ? 'Geçti' : 'Bekliyor'} good={selected.structuralEquivalencePassed} />
              <Metric label="Sıkıştırma kazancı" value={selected.compressionGain?.toFixed(3) ?? '—'} good={(selected.compressionGain ?? 0) > 0} />
              <Metric label="Holdout faydası" value={selected.predictiveUtility?.toFixed(3) ?? '—'} good={(selected.predictiveUtility ?? 0) > 0} />
            </section>
            <section className="detail-block detail-proof"><h3><CheckCircle2 size={17} /> Terfi kanıtı</h3><div className="proof-row"><Activity size={15} /><span>{selected.occurrenceCount} çakışmayan gerçekleşme</span></div><div className="proof-row"><Database size={15} /><span>{selected.sourceGroupCount} bağımsız kaynak grubu</span></div><div className="proof-row"><ShieldCheck size={15} /><span>{selected.fallbackSafe ? 'Açık yola geri dönüş hazır' : 'Geri dönüş kanıtı eksik'}</span></div><div className="proof-row"><ShieldCheck size={15} /><span>{selected.structuralEquivalencePassed ? 'Hızlı iz ve sonraki rota eşdeğerliği doğrulandı' : 'Yapısal eşdeğerlik kapısı bekliyor; runtime kısayolu kapalı'}</span></div><div className="proof-row"><GitBranch size={15} /><span>İç içe derinlik {selected.nestedDepth}</span></div></section>
            <section className="detail-block"><h3>Kimlik ve soy</h3><dl><dt>Açılım özeti</dt><dd>{short(selected.expansionDigest, 34)}</dd><dt>Fiziksel I/R</dt><dd>{selected.physicalIdentityAllocated ? 'Tahsis edildi' : 'Sanal denetleyici; henüz tahsis edilmedi'}</dd><dt>Son güncelleme</dt><dd>{selected.updatedAt ? new Date(selected.updatedAt).toLocaleString('tr-TR') : '—'}</dd></dl></section>
          </> : <div className="concept-detail__placeholder"><Route size={32} /><b>İncelenecek bir makro seç</b><p>Yol açılımı ve terfi kanıtları burada görünecek.</p></div>}
        </article>
      </section>
    </main>
  )
}
