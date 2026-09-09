import {
  Activity,
  Atom,
  BrainCircuit,
  Database,
  GitBranch,
  Layers3,
  LocateFixed,
  RefreshCw,
  Search,
  Sparkles,
  X,
} from 'lucide-react'
import { useCallback, useDeferredValue, useEffect, useMemo, useState } from 'react'
import { SuiteMenu } from '../components/SuiteMenu'
import { loadLiveTrainingState, loadNeuronAtlas } from './api'
import { computeUniverseLayout } from './layout'
import { UniverseScene } from './UniverseScene'
import type { ColorMode, LiveTrainingState, NeuronAtlas, UniverseLayout, UniverseSelection } from './types'
import { structuralGateStatusLabel } from '../training-app/structural-gate'

const number = new Intl.NumberFormat('tr-TR')
const percent = new Intl.NumberFormat('tr-TR', { style: 'percent', maximumFractionDigits: 1 })

function shortId(value: string, length = 13) {
  return value.length <= length ? value : `${value.slice(0, length)}…`
}

function dateLabel(value?: string | null) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString('tr-TR', { dateStyle: 'short', timeStyle: 'short' })
}

function lifecycleLabel(value: string) {
  return ({
    stable: 'Kararlı', provisional: 'Geçici', 'merge-proposed': 'Birleşme adayı', retired: 'Emekli',
    observing: 'Gözleniyor', 'eligible-proposal': 'Öneriye uygun', 'virtual-provisional': 'Sanal geçici',
    'shadow-validated': 'Gölge doğrulandı', active: 'Etkin', deprecated: 'Kullanımdan kaldırıldı',
  } as Record<string, string>)[value] ?? value
}

function macroRuntimeLabel(value: string) {
  return ({
    active: 'seçili etkin snapshot’ta çalışabilir',
    'inactive-lifecycle': 'snapshot etkin; yaşam evresi henüz çalıştırılmıyor',
    'research-snapshot': 'araştırma snapshot’ı; salt okunur',
    'rejected-snapshot': 'reddedilmiş snapshot; çalıştırılmıyor',
  } as Record<string, string>)[value] ?? value
}

function plasticRegionModeLabel(value: string) {
  return ({ off: 'kapalı', shadow: 'gölge', proposal: 'öneri', active: 'etkin' } as Record<string, string>)[value] ?? value
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return <div className={accent ? 'nu-stat is-accent' : 'nu-stat'}><strong>{value}</strong><span>{label}</span></div>
}

function Toggle({ checked, onChange, children }: { checked: boolean; onChange: (value: boolean) => void; children: React.ReactNode }) {
  return (
    <label className="nu-toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span aria-hidden="true"><i /></span>
      <b>{children}</b>
    </label>
  )
}

function UniverseFallback({ atlas, state }: { atlas: NeuronAtlas; state: 'lost' | 'unsupported' }) {
  return (
    <section className="nu-fallback" role="status">
      <Atom size={34} />
      <h2>{state === 'lost' ? '3B bağlam geçici olarak kayboldu' : 'Bu tarayıcı 3B görünümü açamadı'}</h2>
      <p>Veri güvende. Aşağıdaki salt-okunur aile özeti kullanılabilir; sayfayı yenileyerek 3B görünümü tekrar deneyebilirsin.</p>
      <div className="nu-fallback__table" role="table" aria-label="En büyük nöron aileleri">
        {atlas.families.slice().sort((a, b) => b.memberCount - a.memberCount).slice(0, 12).map((family) => (
          <div role="row" key={family.id}><code>{family.label || shortId(family.id, 18)}</code><span>{family.region}</span><strong>{number.format(family.memberCount)} nöron</strong></div>
        ))}
      </div>
    </section>
  )
}

function SelectionPanel({ atlas, selection, onClose, onSelect }: {
  atlas: NeuronAtlas
  selection: UniverseSelection
  onClose: () => void
  onSelect: (selection: UniverseSelection) => void
}) {
  const membershipsByFamily = useMemo(() => {
    if (selection?.kind !== 'family') return []
    return atlas.links.memberships
      .filter((membership) => membership[1] === selection.index)
      .sort((left, right) => right[3] - left[3] || right[2] - left[2])
      .slice(0, 10)
  }, [atlas, selection])
  const membershipsByNeuron = useMemo(() => {
    if (selection?.kind !== 'neuron') return []
    return atlas.links.memberships
      .filter((membership) => membership[0] === selection.index)
      .sort((left, right) => right[3] - left[3] || right[2] - left[2])
  }, [atlas, selection])

  if (!selection) return (
    <aside className="nu-panel nu-detail nu-detail--empty">
      <LocateFixed size={23} />
      <h2>Evreni incele</h2>
      <p>Bir aile küresine veya nöron noktasına dokun. Aitlikleri, yapısal ölçüleri ve yakın yolları burada açılır.</p>
      <small>Fare: döndür · tekerlek: yakınlaştır · sağ tuş: kaydır</small>
    </aside>
  )

  if (selection.kind === 'family') {
    const family = atlas.families[selection.index]
    if (!family) return null
    const outgoing = atlas.links.familyTransitions.filter((edge) => edge[0] === family.index)
    const incoming = atlas.links.familyTransitions.filter((edge) => edge[1] === family.index)
    return (
      <aside className="nu-panel nu-detail" aria-live="polite">
        <button className="nu-icon-button nu-detail__close" aria-label="Seçimi kapat" onClick={onClose}><X size={15} /></button>
        <span className="nu-eyebrow">AİLE #{family.index + 1}</span>
        <h2>{family.label || shortId(family.id, 24)}</h2>
        {family.label && <p className="nu-code">{family.id}</p>}
        <div className="nu-pills"><span>{family.region}</span><span>{lifecycleLabel(family.lifecycle)}</span>{family.origin === 'manual-language-seed' && <span>elle kurulan dil çekirdeği</span>}{family.regionVerified && <span>doğrulanmış rol</span>}</div>
        <dl className="nu-metrics">
          <div><dt>Nöron</dt><dd>{number.format(family.memberCount)}</dd></div>
          <div><dt>Çekirdek</dt><dd>{number.format(family.coreMemberCount)}</dd></div>
          <div><dt>Kaynak</dt><dd>{number.format(family.sourceCount)}</dd></div>
          <div><dt>Destek</dt><dd>{number.format(family.supportCount)}</dd></div>
          <div><dt>İç uyum</dt><dd>{percent.format(family.cohesion)}</dd></div>
          <div><dt>Ayrışma</dt><dd>{percent.format(family.separation)}</dd></div>
          <div><dt>Kararlılık</dt><dd>{percent.format(family.stability)}</dd></div>
          <div><dt>Yol</dt><dd>{number.format(outgoing.length)} ↗ · {number.format(incoming.length)} ↙</dd></div>
        </dl>
        <div className="nu-detail__section">
          <h3>En güçlü üyeler</h3>
          {membershipsByFamily.map(([neuronIndex, , score, weight, , core]) => {
            const neuron = atlas.neurons[neuronIndex]
            return neuron ? <button key={neuron.id} onClick={() => onSelect({ kind: 'neuron', index: neuronIndex })}><span>{neuron.text || shortId(neuron.id, 25)}</span><small>{core ? 'çekirdek · ' : ''}{percent.format(weight || score)}</small></button> : null
          })}
        </div>
      </aside>
    )
  }

  const neuron = atlas.neurons[selection.index]
  if (!neuron) return null
  const outgoing = atlas.links.neuronTransitions.filter((edge) => edge[0] === neuron.index)
  const incoming = atlas.links.neuronTransitions.filter((edge) => edge[1] === neuron.index)
  if (neuron.kind === 'concept-macro' && neuron.conceptMacro) {
    const expansion = (atlas.links.macroExpansions ?? [])
      .filter((edge) => edge[0] === neuron.index)
      .sort((left, right) => left[2] - right[2])
    return (
      <aside className="nu-panel nu-detail" aria-live="polite">
        <button className="nu-icon-button nu-detail__close" aria-label="Seçimi kapat" onClick={onClose}><X size={15} /></button>
        <span className="nu-eyebrow">KAVRAM YÜZEYİ · {neuron.conceptMacro.macroKind === 'equivalence-concept' ? 'EŞDEĞERLİK' : 'SIRA MAKROSU'}</span>
        <h2>{neuron.text}</h2>
        <p className="nu-code">{neuron.id}</p>
        <div className="nu-pills"><span>Head evresi: {lifecycleLabel(neuron.conceptMacro.lifecycle)}</span><span>Çalışma durumu: {macroRuntimeLabel(neuron.conceptMacro.runtimeStatus)}</span></div>
        <dl className="nu-metrics">
          <div><dt>Açılım</dt><dd>{number.format(neuron.conceptMacro.expansionLength)} adım</dd></div>
          <div><dt>Derinlik</dt><dd>{number.format(neuron.conceptMacro.depth)}</dd></div>
          <div><dt>Gölge sadakati</dt><dd>{percent.format(neuron.conceptMacro.fidelity)}</dd></div>
          <div><dt>Sıkıştırma</dt><dd>{number.format(neuron.conceptMacro.compressionGain)}</dd></div>
          <div><dt>Kaynak grubu</dt><dd>{number.format(neuron.conceptMacro.sourceGroupCount)}</dd></div>
          <div><dt>Gözlem</dt><dd>{number.format(neuron.conceptMacro.observationCount)}</dd></div>
        </dl>
        <div className="nu-detail__section">
          <h3>Korunan açılım yolu</h3>
          {expansion.length ? expansion.map(([, memberIndex, ordinal, nested]) => {
            const member = atlas.neurons[memberIndex]
            return member ? <button key={`${ordinal}:${member.id}`} onClick={() => onSelect({ kind: 'neuron', index: memberIndex })}><span>{ordinal + 1}. {member.text || shortId(member.id, 23)}</span><small>{nested ? 'iç içe makro' : member.kind === 'synthesis' ? 'sentez nöronu' : 'ham nöron'}</small></button> : null
          }) : <p>Bu görünüm sınırında çözülebilen açılım üyesi yok.</p>}
        </div>
      </aside>
    )
  }
  return (
    <aside className="nu-panel nu-detail" aria-live="polite">
      <button className="nu-icon-button nu-detail__close" aria-label="Seçimi kapat" onClick={onClose}><X size={15} /></button>
      <span className="nu-eyebrow">NÖRON #{neuron.index + 1}</span>
      <h2>{neuron.text || shortId(neuron.id, 30)}</h2>
      <p className="nu-code">{neuron.id}</p>
      <div className="nu-pills"><span>{neuron.kind === 'synthesis' ? 'sentez' : neuron.kind === 'language-output' ? 'elle kurulan çıktı nöronu' : neuron.kind === 'language-nursery' ? 'otonom nursery adayı' : 'ham'}</span>{neuron.affinity && <span>{neuron.affinity.region} {percent.format(neuron.affinity.score)}</span>}</div>
      <dl className="nu-metrics">
        <div><dt>Kaynak</dt><dd>{number.format(neuron.sourceCount)}</dd></div>
        <div><dt>Kanıt</dt><dd>{number.format(neuron.evidenceCount)}</dd></div>
        <div><dt>Aitlik</dt><dd>{number.format(neuron.membershipCount)}</dd></div>
        <div><dt>Yol</dt><dd>{number.format(outgoing.length)} ↗ · {number.format(incoming.length)} ↙</dd></div>
      </dl>
      <div className="nu-detail__section">
        <h3>Ait olduğu aileler</h3>
        {membershipsByNeuron.map(([, familyIndex, score, weight, rank, core]) => {
          const family = atlas.families[familyIndex]
          return family ? <button key={family.id} onClick={() => onSelect({ kind: 'family', index: familyIndex })}><span>{shortId(family.id, 24)}</span><small>#{rank + 1} · {core ? 'çekirdek · ' : ''}{percent.format(weight || score)}</small></button> : null
        })}
      </div>
    </aside>
  )
}

export function UniverseApp() {
  const [atlas, setAtlas] = useState<NeuronAtlas | null>(null)
  const [layout, setLayout] = useState<UniverseLayout | null>(null)
  const [live, setLive] = useState<LiveTrainingState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [selection, setSelection] = useState<UniverseSelection>(null)
  const [hover, setHover] = useState<UniverseSelection>(null)
  const [query, setQuery] = useState('')
  const deferredQuery = useDeferredValue(query)
  const [colorMode, setColorMode] = useState<ColorMode>('family')
  const [showNeurons, setShowNeurons] = useState(true)
  const [showFamilyPaths, setShowFamilyPaths] = useState(true)
  const [showNeuronPaths, setShowNeuronPaths] = useState(false)
  const [autoRotate, setAutoRotate] = useState(true)
  const [webglState, setWebglState] = useState<'loading' | 'ready' | 'lost' | 'unsupported'>('loading')
  const [plasticRegionId, setPlasticRegionId] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setError(null)
    loadNeuronAtlas(controller.signal).then((value) => {
      if (!value.available) throw new Error(value.reason || 'Görüntülenecek aile çalışması bulunamadı.')
      window.setTimeout(() => {
        if (!controller.signal.aborted) {
          setAtlas(value)
          setLayout(computeUniverseLayout(value))
        }
      }, 0)
    }).catch((reason) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason))
    })
    return () => controller.abort()
  }, [refreshKey])

  useEffect(() => {
    let controller = new AbortController()
    const refresh = () => {
      controller.abort()
      controller = new AbortController()
      loadLiveTrainingState(controller.signal).then(setLive).catch(() => {})
    }
    refresh()
    const interval = window.setInterval(refresh, 5_000)
    return () => { window.clearInterval(interval); controller.abort() }
  }, [])

  useEffect(() => {
    if (!atlas?.partial) return undefined
    const interval = window.setInterval(() => setRefreshKey((value) => value + 1), 15_000)
    return () => window.clearInterval(interval)
  }, [atlas?.partial])

  const plasticRegions = atlas?.projection?.plasticRegions
  useEffect(() => {
    if (!plasticRegions?.compatible) {
      setPlasticRegionId('')
      return
    }
    setPlasticRegionId((current) => plasticRegions.regions.some((region) => region.id === current)
      ? current
      : plasticRegions.dominantRegionId ?? plasticRegions.regions[0]?.id ?? '')
  }, [plasticRegions?.compatible, plasticRegions?.headFingerprint])
  const selectedPlasticRegion = plasticRegions?.regions.find((region) => region.id === plasticRegionId) ?? null

  const searchResults = useMemo(() => {
    if (!atlas || deferredQuery.trim().length < 2) return []
    const needle = deferredQuery.trim().toLocaleLowerCase('tr-TR')
    const familyResults = atlas.families
      .filter((family) => family.id.toLocaleLowerCase('tr-TR').includes(needle) || family.label?.toLocaleLowerCase('tr-TR').includes(needle) || family.region.toLocaleLowerCase('tr-TR').includes(needle))
      .slice(0, 4)
      .map((family) => ({ key: `family:${family.index}`, label: family.label || shortId(family.id, 35), sub: `${number.format(family.memberCount)} nöron · ${family.region}`, selection: { kind: 'family', index: family.index } as UniverseSelection }))
    const neuronResults = atlas.neurons
      .filter((neuron) => neuron.text.toLocaleLowerCase('tr-TR').includes(needle) || neuron.id.toLocaleLowerCase('tr-TR').includes(needle))
      .slice(0, 6)
      .map((neuron) => ({ key: `neuron:${neuron.index}`, label: neuron.text || shortId(neuron.id, 35), sub: neuron.kind === 'concept-macro' ? `${neuron.conceptMacro?.expansionLength ?? 0} adım · kavram yüzeyi` : `${neuron.membershipCount} aile · ${neuron.kind}`, selection: { kind: 'neuron', index: neuron.index } as UniverseSelection }))
    return [...familyResults, ...neuronResults].slice(0, 8)
  }, [atlas, deferredQuery])

  const choose = useCallback((next: UniverseSelection) => {
    setSelection(next)
    setQuery('')
  }, [])

  const hoveredLabel = useMemo(() => {
    if (!atlas || !hover) return null
    if (hover.kind === 'family') {
      const family = atlas.families[hover.index]
      return family ? `Aile · ${family.label || shortId(family.id, 26)} · ${number.format(family.memberCount)} nöron` : null
    }
    const neuron = atlas.neurons[hover.index]
    return neuron ? `${neuron.kind === 'concept-macro' ? 'Kavram makrosu' : 'Nöron'} · ${neuron.text || shortId(neuron.id, 30)}` : null
  }, [atlas, hover])

  const loading = !error && (!atlas || !layout)
  return (
    <div className="nu-app">
      <header className="nu-topbar">
        <SuiteMenu currentApp="universe">
          <div className="nu-brand"><span><Atom size={21} /></span><div><strong>NÖRON EVRENİ 3B</strong><small>CANLI SİSTEM ATLASI · SALT OKUNUR</small></div></div>
        </SuiteMenu>
        <div className="nu-topbar__center">
          <span className={`nu-live ${live?.campaignStatus === 'running' ? 'is-running' : ''}`}><i />{live?.campaignStatus === 'running' ? 'Eğitim canlı' : live?.campaignStatus ? `Eğitim ${live.campaignStatus}` : 'Servis bekleniyor'}</span>
          {live && <span>{live.activeStage ? live.activeStage.replaceAll('teacher-feed:', '') : live.serviceState} · %{Math.round(live.progress * 100)}</span>}
        </div>
        <button className="nu-refresh" onClick={() => setRefreshKey((value) => value + 1)} disabled={loading}><RefreshCw size={15} className={loading ? 'is-spinning' : ''} /> Atlası yenile</button>
      </header>

      {error ? (
        <main className="nu-error" role="alert"><Atom size={42} /><h1>Evren verisi açılamadı</h1><p>{error}</p><button onClick={() => setRefreshKey((value) => value + 1)}><RefreshCw size={15} /> Yeniden dene</button></main>
      ) : loading ? (
        <main className="nu-loading" role="status"><div className="nu-loader"><i /><i /><i /></div><h1>Nöron evreni kuruluyor</h1><p>{atlas ? `${number.format(atlas.stats.neuronCount)} nöronun deterministik 3B yerleşimi hesaplanıyor…` : 'Salt-okunur aile ve geçiş katmanları getiriliyor…'}</p></main>
      ) : atlas && layout ? (
        <main className="nu-workspace">
          {atlas.partial && (
            <div className="nu-partial-notice" role="status">
              <BrainCircuit size={15} />
              <div><strong>Güvenli hafif görünüm</strong><small>Yoğun aile hesabı sürüyor; dil çekirdeği açık. Tam nöron evreni işlem bitince otomatik gelecek.</small></div>
            </div>
          )}
          <UniverseScene
            atlas={atlas}
            layout={layout}
            selection={selection}
            colorMode={colorMode}
            showNeurons={showNeurons}
            showFamilyPaths={showFamilyPaths}
            showNeuronPaths={showNeuronPaths}
            autoRotate={autoRotate}
            onSelect={choose}
            onHover={setHover}
            onWebglState={setWebglState}
          />

          <section className="nu-overview" aria-label="Evren özeti">
            <Stat label="NÖRON" value={number.format(atlas.stats.neuronCount)} accent />
            <Stat label="AİLE · DİL" value={`${number.format(atlas.stats.familyCount)} · ${number.format(atlas.stats.languageOutputFamilyCount ?? 0)}`} />
            <Stat label="YUMUŞAK AİTLİK" value={number.format(atlas.stats.membershipCount)} />
            <Stat label="AİLE YOLU" value={number.format(atlas.stats.familyTransitionCount)} />
            <Stat label="NÖRON YOLU" value={number.format(atlas.stats.neuronTransitionCount)} />
            <Stat label="KAVRAM MAKROSU" value={`${number.format(atlas.stats.conceptMacroCount ?? 0)} · ${number.format(atlas.stats.conceptMacroActiveCount ?? 0)} etkin`} />
            <Stat label="KAYNAK" value={number.format(atlas.stats.distinctSourceCount ?? 0)} />
          </section>

          <aside className="nu-panel nu-controls">
            <span className="nu-eyebrow">GÖRÜNÜM KATMANLARI</span>
            <div className="nu-search">
              <Search size={15} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Nöron metni veya aile ara" aria-label="Evren içinde ara" />
              {query && <button aria-label="Aramayı temizle" onClick={() => setQuery('')}><X size={13} /></button>}
              {searchResults.length > 0 && <div className="nu-search__results">{searchResults.map((result) => <button key={result.key} onClick={() => choose(result.selection)}><span>{result.label}</span><small>{result.sub}</small></button>)}</div>}
            </div>
            <div className="nu-control-group">
              <Toggle checked={showNeurons} onChange={setShowNeurons}>Nöron noktaları</Toggle>
              <Toggle checked={showFamilyPaths} onChange={setShowFamilyPaths}>Aile geçiş yolları</Toggle>
              <Toggle checked={showNeuronPaths} onChange={setShowNeuronPaths}>Seçili nöron yolları</Toggle>
              <Toggle checked={autoRotate} onChange={setAutoRotate}>Yavaş otomatik dönüş</Toggle>
            </div>
            <label className="nu-select"><span>RENK ANLAMI</span><select value={colorMode} onChange={(event) => setColorMode(event.target.value as ColorMode)}><option value="family">Aile kimliği</option><option value="lifecycle">Yaşam evresi</option><option value="utility">Ölçülen fayda</option><option value="region">İşlev bölgesi</option></select></label>
            {plasticRegions && (
              <section className="nu-plastic-regions" aria-label="Plastik bölge atlası özeti">
                <div className="nu-plastic-regions__head">
                  <span>PLASTİK BÖLGE ATLASI</span>
                  <b className={plasticRegions.compatible ? 'is-compatible' : ''}>{plasticRegions.compatible ? `r${plasticRegions.headRevision}` : 'uyumsuz'}</b>
                </div>
                <code title={plasticRegions.headFingerprint ?? undefined}>{shortId(plasticRegions.headFingerprint ?? 'fingerprint yok', 25)}</code>
                <small title={plasticRegions.effectiveModeReason}>Ayarlı {plasticRegionModeLabel(plasticRegions.configuredMode)} · etkili {plasticRegionModeLabel(plasticRegions.effectiveMode)} · {plasticRegions.behaviorApplied ? 'doğrulanmış rota etkisi' : plasticRegions.effectiveMode === 'proposal' ? 'öneri önizlemesi' : plasticRegions.effectiveMode === 'shadow' ? 'yalnız gözlem' : 'davranış kapalı'}</small>
                <small title={plasticRegions.release.reason ?? undefined}>{plasticRegions.release.eligible ? `Release doğrulandı · ${shortId(plasticRegions.release.fingerprint ?? plasticRegions.release.artifactId ?? '—', 18)}` : 'Active release kanıtı yok'}</small>
                {plasticRegions.compatible ? (
                  <>
                    <label>
                      <span>GİZİL BÖLGE İNCELEME SEÇİMİ</span>
                      <select value={plasticRegionId} onChange={(event) => setPlasticRegionId(event.target.value)}>
                        {plasticRegions.regions.map((region) => <option key={region.id} value={region.id}>{shortId(region.id, 24)}</option>)}
                      </select>
                    </label>
                    {selectedPlasticRegion && (
                      <dl>
                        <div><dt>Kapasite</dt><dd>{number.format(selectedPlasticRegion.effectiveCapacity)}</dd></div>
                        <div><dt>Baskın sayfa</dt><dd>{number.format(selectedPlasticRegion.dominantPageCount)} / {number.format(plasticRegions.pageCount)}</dd></div>
                        <div><dt>Atlas payı</dt><dd>{percent.format(selectedPlasticRegion.globalShare)}</dd></div>
                        <div><dt>Tutarlılık</dt><dd>{percent.format(selectedPlasticRegion.cohesion)}</dd></div>
                      </dl>
                    )}
                  </>
                ) : <p>Bu head seçili aile çalışmasının temsil uzayına ait değil; 3B veriye karıştırılmadı.</p>}
                <a href="/bolge-atlasi/">Fiziksel atlası aç</a>
              </section>
            )}
            <div className="nu-legend"><span><i className="is-family" /> aile</span><span><i className="is-language" /> dil çıkış adası</span><span><i className="is-concept" /> kavram yüzeyi</span><span><i className="is-neuron" /> nöron</span><span><i className="is-experience" /> deneyim yolu</span><span><i className="is-inhibition" /> baskılayıcı yol</span><span><i className="is-prior" /> başlangıç yolu</span></div>
          </aside>

          <SelectionPanel atlas={atlas} selection={selection} onClose={() => setSelection(null)} onSelect={choose} />

          <footer className="nu-footer">
            <span id="nu-projection-note"><Layers3 size={13} /> 3B eksenler semantik değildir; mesafe etkileşim ve aidiyet izdüşümüdür.</span>
            <span><Database size={13} /> {atlas.partial ? 'Yoğun eğitim · hafif dil görünümü' : atlas.selection.snapshot?.status === 'rejected' ? 'Reddedilmiş araştırma snapshot’ı; etkin model değildir' : atlas.selection.snapshot?.active ? 'Etkin snapshot' : 'Araştırma snapshot’ı'} · {dateLabel(atlas.selection.snapshot?.createdAt)}</span>
            <span><GitBranch size={13} /> {shortId(atlas.run?.fingerprint ?? '—', 12)}</span>
            {plasticRegions && <span><Layers3 size={13} /> bölge {shortId(plasticRegions.headFingerprint ?? '—', 12)} · {plasticRegions.compatible ? 'uyumlu' : 'karantinada'}</span>}
          </footer>

          {hoveredLabel && <div className="nu-hover-label"><Sparkles size={12} />{hoveredLabel}</div>}
          {webglState !== 'ready' && webglState !== 'loading' && <UniverseFallback atlas={atlas} state={webglState} />}

          <div className="nu-live-card">
            <Activity size={15} />
            <div><strong>{live?.campaignAccepted ? `${number.format(live.campaignAccepted)} kabul` : 'Canlı ölçüm'}</strong><small title={live ? structuralGateStatusLabel({ status: live.gateStatus, passed: live.gatePassed }) : undefined}>{live ? `${number.format(live.campaignGenerated)} üretim · ${number.format(live.completedBlocks)} blok · kapı ${structuralGateStatusLabel({ status: live.gateStatus, passed: live.gatePassed })}` : 'Servis bilgisi bekleniyor'}</small></div>
          </div>
        </main>
      ) : null}
    </div>
  )
}
