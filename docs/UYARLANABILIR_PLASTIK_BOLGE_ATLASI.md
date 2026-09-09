# Uyarlanabilir Plastik Bölge Atlası

Bu belge, tek bir nöronun plastik `R` alanında işlevsel bölgelerin veriyle
keşfedilmesi için fiziksel, matematiksel ve kalıcılık sözleşmesidir. Bu yapı
nöron ailelerinin veya kavram makrolarının yeni adı değildir.

## 1. Mevcut sistem ile yeni mekanizmanın sınırı

| Yapı | Görevi | Bu çalışmadaki durumu |
| --- | --- | --- |
| TextLab 16/48 uzayı | Mevcut metin temsili | Değişmeden, okunabilir ve replay edilebilir kalır |
| Nöron ailesi | Farklı nöronları davranışlarına göre gruplar | Yerinde korunur |
| Kavram makrosu | Tekrarlanan sıralı nöron yollarını sıkıştırır | Yerinde korunur |
| Plastik bölge atlası | Bir nöronun `R` koordinatlarının birlikte nasıl kullanıldığını öğrenir | Yeni companion mekanizmadır |
| 8/56 uzayı | Atlas için hedef fiziksel temsil | Yeni ve ayrı bir representation lineage'dır |

Eski 16/48 vektörler kesilmez, genişletilmez veya yeni koordinatlarda yeniden
yorumlanmaz. Yalnız ham kaynak ve dış provenance yeni uzaya taşınabilir; bütün
öğrenilmiş vektör, kenar, aile ve makro kanıtı yeni uzayda yeniden üretilir.

Uygulama düzeyindeki ayrım da aynıdır:

- **Mevcut sisteme yapılan uyumluluk geliştirmeleri:** TextLab kayıtlarına isteğe bağlı
  açık temsil-profili kanıtı eklenmesi, düşünce motorunun bu kanıtı taşıması,
  SSD sayfa deposunun yeni fiziksel geometriyi imzalı olarak okuyabilmesi ve 3B
  evrenin varsa atlas özetini gösterebilmesi.
- **Yeni eklenen mekanizmalar:** atlas matematiği, bölge gözlemleri, birikimli kanıt
  başlığı, yaşam döngüsü önerileri, shadow/release kapıları, Bölge Atlası ekranı ve
  bölgeye özgü replay bağı.
- **Değiştirilmeyen yapılar:** profilsiz tarihsel kayıtların anlamı, mevcut nöron
  aileleri, kavram makroları, dil çıktı katmanı ve eski 16/48 vektörlerin içeriği.

Temsil profili ile atlas çalışma kipi birbirinden bağımsızdır. Açıkça 8/56
profilinde üretilmiş bir snapshot `off` kipinde de 8/56 kodlayıcısını kullanır;
yalnız atlas rota/yazım kararına katılmaz. Buna karşılık, sadece matris ölçülerine
bakarak profilsiz bir snapshot'a 8/56 anlamı atanmaz.

## 2. Hedef fiziksel geometri

Hedef nöron gösterimi:

\[
Z_i=[I_i\mid R_i],\qquad
I_i\in[-1,1]^{512},\quad R_i\in[-1,1]^{3584}
\]

- Satır `0..7`: değişmez kimlik `I`.
- Satır `8..63`: plastik ilişki alanı `R`.
- `R`, `7×8=56` adet değişmez `8×8` mikro-sayfadır.
- Her mikro-sayfa 64 hücre taşır.

Mikro-sayfa koordinatı:

\[
p=8\left\lfloor\frac{row-8}{8}\right\rfloor+
\left\lfloor\frac{column}{8}\right\rfloor
\]

Yerel `(u,v)` hücresinin global koordinatı:

\[
row=8+8\left\lfloor\frac p8\right\rfloor+u,
\qquad column=8(p\bmod8)+v
\]

Fiziksel mikro-sayfalar taşınmaz. Mantıksal bölgelerin büyüklüğü ve topolojisi
yalnız yumuşak üyelik metadata'sı ile değişir.

## 3. Öğrenilmiş bölge atlası

Mikro-sayfa `p` ile gizil bölge `k` arasındaki üyelik:

\[
A_{pk}\in[0,1],\qquad \sum_k A_{pk}+f_p=1
\]

`f_p`, sayfanın henüz atanmadığı serbest kapasitedir. Bir bölgenin efektif
büyüklüğü:

\[
C_k=\sum_p A_{pk}
\]

Bir sayfa en fazla üç canlı bölge üyeliği taşır. Kararlı bölgeler en az bir
sayfa eşdeğeri kapasite taşıdığı için bölge sayısı keyfi bir sunucu sınırına
değil 56 sayfalık fiziksel bütçeye bağlıdır.

Bölge kimlikleri opaktır. `kişi`, `kip`, `matematik` gibi açıklamalar atlasın
kimliğine, fingerprint'ine veya routing matematiğine girmez. Bunlar yalnız
sonradan çalışan probe katmanının doğrulanabilir korelasyon açıklamalarıdır.

## 4. Gözlem ve veri sızıntısı sınırı

Her tickte bir mikro-sayfa için en az şu bounded özetler çıkarılır:

- sinyal ve aday enerjisi,
- sayfa-içi cosine uyumu,
- residual ve yazım büyüklüğü,
- rota katkısı,
- seyrek eş-etkinleşme sketch'i,
- aile/koalisyon association sketch'i,
- sonuç kredisi ve örneklenmiş ablation etkisi,
- bağımsız kaynak grubu desteği.

Her gözlem `pre` ve `post` diye ayrılır. Canlı gating yalnız sonuçtan önce
bilinen `pre` özelliklerini kullanabilir. Ödül, sonuç grafiği ve ablation gibi
`post` özellikleri yalnız episode sonrası konsolidasyona girebilir.

Kaynak prototipleri önce kendi içinde normalize edilir, sonra kaynak grupları
eşit ağırlıkla birleştirilir. Aynı Ollama model/weights soyu kaç örnek üretirse
üretsin tek bağımsız kaynak grubu olarak kalır.

## 5. Canlı gating ve mikro iz

Bölge olasılıkları, bölgeler ile açıklanamayan kanal üzerinde normalize edilir:

\[
(g_{ti1},\ldots,g_{tiK},g_{ti\varnothing})=softmax(\ell)
\]

\[
\sum_k g_{tik}+g_{ti\varnothing}=1
\]

Sayfa yazım kapısı:

\[
m_{tip}=m_{min}+(1-m_{min})
\left(\sum_k g_{tik}A_{pk}+g_{ti\varnothing}f_p\right)
\]

`m_min>0`, her temasın mikro iz bırakmasını sağlar. Dal-içi hızlı yazım:

\[
\Delta R^{fast}_{i,p}\leftarrow clip(
\Delta R^{fast}_{i,p}+\eta\,eligibility_t\,m_{tip}(S^R_{t,p}-Z^R_{i,p}),-1,1)
\]

`I` hiçbir atlas işleminde yazılabilir değildir. Paralel dalların hızlı bölge
durumları birbirinden tamamen ayrıdır.

İlk devreye almada sinyal dönüşümü ve rota kararı değişmez. Daha sonraki
candidate snapshotlarda bölgesel uyum bounded bir karışım olarak eklenebilir:

\[
c_t=(1-\lambda)c_{legacy}+\lambda c_{region}
\]

`lambda=0`, legacy davranışın tam karşılığıdır.

## 6. Yaşam döngüsü

Atlas `K=0`, bütün `f_p=1` ile başlar. Önce yalnız gözlem toplar.

- **Doğum:** kalıcı açıklanamayan residual, en az üç bağımsız kaynak, null
  dağılımını aşan cohesion ve pozitif holdout faydası gerekir.
- **Büyüme:** bir sayfanın benzerlik, residual basıncı ve marjinal ablation
  faydası birlikte pozitifse üyelik küçük bir adımla artar.
- **Küçülme:** üyelik kaldırıldığında holdout sonucu non-inferior ise kapasite
  serbest havuza döner.
- **Bölünme:** iki kararlı davranış modu, MDL kazancı, iki tarafta bağımsız
  kaynak ve sham split üstünlüğü gerekir. Çocuk üyeliklerinin toplamı ebeveyn
  üyeliğine eşittir.
- **Birleşme:** prototip ve etkiler benzer, benzersiz koşullu bilgi düşük ve
  merge ablation non-inferior ise yeni bir kimlik doğar.
- **Emeklilik:** birkaç atlas döneminde düşük benzersiz fayda gösteren bölge
  tombstone ve predecessor zinciri korunarak emekliye ayrılır.

Hiçbir topoloji değişikliği episode ortasında yapılmaz. Akış her zaman:

`immutable proposal -> shadow validation -> candidate atlas -> snapshot gate`

şeklindedir. Serbest kapasite yoksa eski veri ezilmez; `capacity-blocked`
önerisi üretilir.

## 7. Kalıcı veri ve replay bağı

Yeni artifact aileleri:

- `plastic-region-observation-batch`
- `plastic-region-atlas-head`
- `plastic-region-atlas-proposal`
- `plastic-region-shadow-report`
- `plastic-region-evidence-head`
- `plastic-region-release-benchmark-report`
- `plastic-region-release-evaluation-audit`
- `plastic-region-release-evidence`
- `plastic-region-release-manifest`
- `plastic-region-release-renewal-evidence`
- `plastic-region-release-renewal-manifest`
- `plastic-region-release-renewal-audit`
- `plastic-region-semantic-probe-spec`
- `plastic-region-semantic-probe-observations`
- `plastic-region-semantic-probe-calibration`
- `plastic-region-semantic-probe-evaluation`

Eski `plastic-region-probe-cache` boş skeleton kayıtları yalnız tarihsel audit
için okunur; yeni çevrimler bunları üretmez, kopyalamaz veya başka bir formata
terfi ettirmez. Ölçülebilir probe kanıtı yoksa doğru sonuç `unknown`dur ve boş
bir artifact yazılmaz.

`relationPageSketches`, ayrı bir atlas artifact'i değildir. Yalnız açık
`adaptive-eighth-id-v1` profilinde üretilen SSD nöron manifestinin içinde yer
alan, vektör taşımayan fiziksel aday-seçim özetidir. Aynı ölçülere sahip
profilsiz veya tarihsel bir manifest bu anlamla yorumlanmaz.

Fiziksel encoder sözleşmesi `representationSpaceId` ile, öğrenilmiş atlas ise
ayrı `regionAtlasFingerprint` ile tanımlanır. Atlas her öğrendiğinde fiziksel
temsil uzayı değişmez.

Atlas routing veya yazımı etkilediğinde episode ve checkpoint en az şu bağı
taşır:

- `regionGeometryFingerprint`
- `regionAtlasRevision`
- `regionAtlasFingerprint`
- `regionRoutingMode`
- `regionReleaseFingerprint` (etkin kipte)
- fiziksel SSD nöron yerleşiminin `residencyManifestFingerprint` değeri

Replay en yeni atlası kullanmaz; kayıtlı tam atlas bulunamıyorsa fail-closed
olur. Commit işlemi snapshot ve atlas base fingerprintlerini CAS ile doğrular;
yarışı kaybeden öneri `STALE_REGION_ATLAS_BASE` olur.

Tarihsel snapshotlar silinmez; ancak genel rollback yordamı hedefi yeni bir
wrapper snapshot olarak etkinleştirdiği için, snapshot'a imzalı bir 8/56 release
manifesti bu wrapper'a kopyalanamaz. Adaptive geri dönüş bu nedenle exact hedef
için sunucunun yeniden ölçtüğü release/renewal bağı bulunana kadar
`PLASTIC_REGION_ROLLBACK_RENEWAL_REQUIRED` ile fail-closed durur. Legacy rollback
davranışı değişmez.

## 8. Çalışma kipleri

- `off`: seçili ve kanıtlanmış fiziksel temsil profili korunur; atlas rota,
  sinyal veya yazım kararına katılmaz. Profilsiz tarihsel profilde eski davranış,
  açık 8/56 profilinde ise 8/56 kodlayıcı ve sıfır atlas müdahalesi vardır.
- `shadow`: gözlem ve alternatif skor üretir; rota, sinyal, R patch ve kalıcı
  head değişmez.
- `proposal`: doğrulanmış candidate üzerinde bölgesel sonuçları gösterir fakat
  otomatik etkinleştirme yapmaz.
- `active`: yalnız snapshot'a bağlı, holdout/sham kapısını geçmiş atlas routing
  ve/veya R yazımına katılır.

Kip değişimi request ile serbestçe yapılamaz. Etkili kip, seçili snapshot ve
sunucu politikasının daha kısıtlayıcı olanıdır.

### Kullanıcıya açık güvenli geçiş

Bölge Atlası ekranı mevcut modeli yerinde dönüştüren tek bir aç/kapat düğmesi
sunmaz. Geçiş üç ayrı kullanıcı kararıdır:

1. `legacy/off -> adaptive-eighth-id-v1/shadow`: ham kaynaklardan ayrı temsil
   soyu hazırlanır; eski vektör, aile, makro ve snapshotlar değiştirilmez.
2. `shadow -> proposal`: ancak gölge gözlem/kanıt artifacti görüldükten sonra
   geri alınabilir topoloji önerileri açılır.
3. `proposal -> active`: öneri önce sunucuya ait, Ollama'sız, dondurulmuş
   taban–aday–aynı ayak izli sham sınamasına gönderilir. Başarılı sonuç bile
   otomatik etkinleştirme yapmaz. Release aday snapshotı insan tarafından
   kabul edildikten sonra canlı kip ayrıca ve açıkça istenir.

Release işi normal HTTP sürecinde çalışmaz; ayrı worker sürecinde yürür. İş
kuyruğu duraklatma/iptal ve kaynak politikalarına uyar. İstemci holdout skoru,
sham sonucu veya başarı kararı gönderemez; yalnız immutable proposal artifact
kimliğini seçebilir. Başarılı release adayının candidate snapshotı ve üretici
iş sonucu aynı transaction içinde yazılır; hiçbir aşama adayı otomatik olarak
etkinleştirmez.

## 9. Kabul kapıları

- `4096=512+3584` ve 56 sayfa bijeksiyonu eksiksizdir.
- `sum(A[p,*])+f[p]=1±1e-6`.
- Bütün olasılık dağılımları `1±1e-6` toplam verir.
- `I` digest hiçbir atlas testinde değişmez.
- `R` ve bütün yamalar sonlu ve `[-1,1]` içindedir.
- `off` eski tick/replay digestlerini değiştirmez.
- `shadow` mevcut rota ve kalıcı delta üzerinde sıfır etkilidir.
- Aynı base, girdi, seed ve atlas aynı hash zincirini üretir.
- Target ablation anlamlı kayıp, eşlenmiş sham ablation düşük kayıp gösterir.
- Shuffled veride yanlış doğum oranı en fazla `%5` olur.
- Atlas tick p95 süresine en fazla `%15` ek yük getirir.
- Tick sırasında tüm nöronlar, bütün SSD sayfaları veya `N²` tablo taranmaz.

## 10. İnsan tarafından okunabilir yorum

Probe katmanı atlası değiştirmez. Bir bölge için yalnız şu tip bir ifade
üretebilir:

> `latent-region-12`, kip karşıtlıklarında 0.71 ablation katkısı gösterdi;
> güven orta, üç bağımsız kaynak grubu.

Kalibre edilmemiş hücre sigmoid'i olasılık diye adlandırılmaz. Arayüzde ham
imzalı değer, kanıt opaklığı, sayfa aidiyeti ve canlı bölge kapısı ayrı
katmanlar olarak gösterilir.

## 11. Yeni, isteğe bağlı anlamsal probe katmanı

Bu katman atlasın, nöron ailesinin veya kavram makrosunun devamı değildir;
tamamen yeni ve salt-okunur bir yorum companion'ıdır. `morphology`,
`event-role` veya ileride `mathematics` gibi etiketler dış hipotez olarak ayrı
artifactlerde tutulur. Bir etiket hiçbir zaman bölge kimliğine, prototipine,
sayfa özelliğine, routing skoruna veya eğitim deltasına yazılmaz. Bütün probe
artifactleri şu kapalı yetki alanlarını taşır:

- `trainingWritesAllowed: false`
- `routingMutationAllowed: false`
- `atlasMutationAllowed: false`
- evaluation için `releaseAuthority: false`

Katman `plasticRegions.semanticProbesEnabled` ile ayrıca ve açıkça etkinleşir;
varsayılanı kapalıdır. Kapatıldığında mevcut atlas çevriminin, ailelerin,
kavram makrolarının, nöron rotasının ve legacy temsilin tek bir digestini dahi
değiştirmez.

İlk gerçek üretici, Ollama kullanmayan dondurulmuş `144 kombinasyon × 2 yüzey`
olay-grafiği tanı paketini açık bir snapshot ve doğrulanmış atlas üzerinde
çalıştırır. Düşünce motorundan yalnız `shadow`, `activeApplied:false`
telemetrisi okunur. Etiket otoritesi istemciden alınmaz; immutable deterministik
probe-spec artifactine bağlanır. `trusted-deterministic-pack` kipinde her etiket
bağının otoritesi `deterministic`, `contentHash` değeri de producer'ın exact
`probeSpecFingerprint` değeri olmak zorundadır; aksi artifact persistence öncesi
reddedilir. Train bölümü hedef korelasyonunu ve source-
balanced isotonic kalibrasyonu kurar. Validation/test hiçbir modeli tekrar
ayarlamaz; Brier skoru, expected calibration error (ECE) ve kaynak-grubu
bootstrap `%95` güven aralığını ölçer.

Kontrol özelliği etikete bakılarak seçilmez. Hedef bölgeyle mikro-sayfa aidiyet
ayak izi en yakın olan başka opak bölge deterministik sham olur. Bir hipotez
ancak hedefin hem sabit tabandan hem de bu footprint-matched sham'den üstünlüğü
güven aralığının alt sınırında da korunduğunda ve ECE sınırını geçtiğinde
`calibrated` olabilir. Daha az destek `provisional`; sınıf veya bağımsız kaynak
desteği yetersizse `unknown`dur.

Kaynak bağımsızlığı istemci `independenceGroupId` beyanından alınmaz. Sunucunun
fiziksel locator, içerik özeti, örnek özeti ve varsa Ollama model/weights
soyundan türetilir; ortak herhangi bir soy izi aynı tanığı birleştirir. Train
ile validation/test arasında tek bir fiziksel/örnek izi çakışması bile bütün
evaluation'ı fail-closed durdurur. Ollama üretimi veya Ollama etiketli kayıtlar
validation/teste hiçbir koşulda giremez.

Probe kabul kriterleri:

- train/evaluation soy izi kesişimi tam `0`;
- replay'de atlas, input manifest, probe spec, partition, kalibrasyon ve gözlem
  fingerprintleri birebir aynı;
- hedef Brier kazancı ve hedef–sham farkı `%95` alt sınırda politika eşiğini
  geçer;
- ECE politika tavanını aşmaz;
- pozitif ve negatif sınıfların her birinde yeterli bağımsız kaynak bulunur;
- kayıt/özellik/soy izi/artifact byte sınırları persistence öncesi uygulanır;
- hiçbir probe sonucu atlas release veya canlı routing yetkisi vermez.

## 12. Yeni mekanizma: değişmeyen atlas için model-release yenilemesi

Bir atlasın daha önce release edilmiş olması, daha sonra üretilen farklı bir
model snapshotında otomatik olarak canlı kullanım yetkisi vermez. Bu boşluk
mevcut yaşam döngüsü önerisinin bir uzantısı gibi gösterilmez; ayrı
`unchanged-atlas-model-renewal` mekanizmasıdır.

Renewal yalnız güncel etkin snapshotın bağımsız genel değerlendirmeyi geçmiş
doğrudan adayında çalışır. Aday, etkin atlasın aynı artifact kimliğini, içerik
özetini, fingerprintini ve revizyonunu byte düzeyinde taşımak zorundadır. Sunucu
yeni model adayını aynı dondurulmuş vaka ve beş seed üzerinde üç eşlenmiş yolla
ölçer: atlas kapalı taban, değişmeyen atlas etkin aday ve aynı sayfa/kütle ayak
izine sahip deterministik prototip sham'i. Ortak eşleme anahtarı üç yolun RNG
akışını aynı tutar; Ollama çağrılmaz ve istemci metriği kabul edilmez.

Başarıda ham satırları taşıyan immutable renewal evidence, ardından exact
base/source/atlas/deploy politikasına bağlı renewal manifest ve en son yeni bir
candidate snapshot yazılır. Candidate yayını ile üretici işin başarılı sonucu
tek transaction içinde kalıcılaşır; süreç ikisinin arasında kapanırsa yarım,
yeniden kullanılamaz bir aday kalmaz. Mevcut etkin snapshot değişmez. Kullanıcı bu adayı
ayrıca etkinleştirmeden `active` davranış açılmaz. Başarısızlıkta yalnız
self-contained audit kalır. Rollback wrapperı renewal sayılmaz; rollback hedefi
aynı ayrı ölçüm ve CAS zincirinden geçmeden canlı atlas yetkisi kazanamaz.
Ardışık değişmeyen-atlas yenilemeleri eski release zincirini yeniden doğrular;
bu doğrulama kötü/bozuk soylarda sınırsız çalışmaması için 64 hopta fail-closed
durur.

Sunucu bu işlemi `POST /api/regions/snapshots/:candidateId/renew-release`
üzerinden başlatır. İstek yalnız isteğe bağlı idempotency anahtarı taşıyabilir;
puan, ham vaka, benchmark sonucu veya atlas içeriği kabul edilmez. İş sonucu
tarayıcıya sınırlı bir özet olarak verilir; eşlenmiş ham satırlar immutable
artifact içinde kalır.

## 13. Kasıtlı geçici ölçek sınırı

Canlı tick yönlendirmesi seyrek aday kapılarıyla 100K nöron ölçeğinde sınanır;
ancak atlas release ve değişmeyen-atlas renewal değerlendirmesi bugün doğrulanmış
tabanı dense olarak açmaktadır. Bu ayrı değerlendirme yolu en fazla 16.384 nöron
ve 256 MiB dense vektör bütçesi kabul eder. Sınır aşılırsa payload RAM'e
açılmadan iş `PLASTIC_REGION_RELEASE_DENSE_BASE_UNSUPPORTED` ile beklemeye alınır
ve aday etkinleştirilemez. Bu, atlas veya aile sayısına konmuş yapay bir öğrenme
sınırı değildir; SSD-sayfalı bağımsız release değerlendiricisi tamamlanana kadar
uygulanan fail-closed bellek güvenliğidir.
