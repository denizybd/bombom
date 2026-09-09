# Qwen 14B canlı otomatik eğitim kanıtı — 29 Ağustos 2026

Bu tutanak, `qwen2.5-coder:14b` modelinin kalıcı yerel eğitim servisini gerçekten
beslediğini, çekirdek motorların gerçek artifact ürettiğini ve yetersiz kanıtın etkin modele
geçmediğini aynı zincir üzerinde kaydeder. Saatler servis kayıtlarında UTC, arayüzde yerel
saat (UTC+03:00) olarak gösterilir.

## Son durum

- Servis: `http://127.0.0.1:8787`, SQLite, sağlıklı.
- Öğretmen: `qwen2.5-coder:14b`, 14.8B, Q4_K_M, yerel Ollama bağlantısı açık.
- Otomasyon: `controlled`, `autoRun=true`, yeni/değişmiş kaynak parti eşiği `8`,
  eşzamanlı iş sayısı `1`.
- Etkin snapshot: yok. Bu bilinçli bir güvenlik sonucudur; ilk korpus `%100` sentetiktir.

## 1. Öğretmen üretimi ve insan karantinası

İki gerçek Ollama çağrısı toplam 12 kayıt üretti. Şema hatası veya içerik parmak izi tekrarı
oluşmadı. İnsan denetiminden sonra 8 kayıt kabul, 4 kayıt ret aldı.

| Kanıt | Değer |
|---|---:|
| `teacher-sources-quarantined` olayı | 2 |
| Üretilen / eklenen | 12 / 12 |
| Onaylanan | 8 |
| Reddedilen | 4 |
| Tekrar | 0 |
| Öğretmen token kullanımı | 1.400 |

Reddedilen bozuk veya güvenilmez örnekler arasında `Ali, kitabı okuyor değil.`, `Kitabı
okuyor değil Ali.`, `Nasıllar sen? İyi misin?` ve doğal olmayan `sinirli bir gün geçirdi`
kuruluşu bulunur. Bu kayıtlar `selected=false` ve `disabled` kaldı; nöron hattına girmedi.

## 2. Otomatik tetikleme kanıtı

Sekizinci onaydan sonra servis, elle `/api/run/start` çağrısı olmadan aşağıdaki işi üretti:

- İş: `13997ff2-a895-47cd-8ac0-ef902a22674e`
- Tetik: `approved-source-batch`
- Idempotency karması:
  `auto:bd8e39e682dab5beead166018a6578e94586c43abd3338c57be7a4ec07ed9593`
- Başlangıç: `2026-08-29T07:05:30.274Z`
- Bitiş: `2026-08-29T07:05:31.757Z`
- İş durumu: `succeeded`; etkinleştirme kararı: `committed=false`.

Bu iş, otomatik eşik ve kuyruk sisteminin gerçekten çalıştığının doğrudan kalıcı kaydıdır.

## 3. Sentetik holdout korumalı temiz kanıt çevrimi

Canlı denetimde sentetik öğretmen örneklerinin ön veri manifestinde holdout'a yazılabildiği
görüldü. Kural düzeltildi: `ollama` ve diğer sentetik fiziksel kaynak grupları artık hem ön
manifestte hem dil veri kümesinde yalnız `train` bölümünde tutulur. Eski sentetik holdout
ataması varsa `train` olarak geçersiz kılınır ve uyarı kaydedilir.

Düzeltmeden sonra aynı sabit korpusla ikinci, açık etiketli kanıt çevrimi çalıştırıldı:

- İş: `3cb33f0b-d72c-4476-9b1a-8e69d66f1a0b`
- Snapshot: `d8f48b15-2067-4fcb-82a7-2e800dd19322`
- Korpus parmak izi: `dataset-3A0999B82A8C9495AB1081865F3F6B67`
- Girdi manifest karması:
  `664a4cd2e67688d5cb9eebf399d3de45f52df980f4466ccc3f850ff1dcae2512`
- Temsil uzayı: `RS-01BA3FAEB3ECE358`
- Başlangıç/bitiş: `2026-08-29T07:09:07.503Z` — `2026-08-29T07:09:08.954Z`

Gerçek aşama sonuçları:

| Aşama | Doğrulandı | Ölçülebilir çıktı |
|---|---|---|
| Collect | Evet | 8 kaynak, 308 byte; tamamı train, validation/test boş |
| TextLab | Evet | 8 kayıpsız analiz, 170 ham nöron |
| Synthesis | Evet | 10.717 karşılaştırma, 12/12 eşik geçen sentez adayı |
| Families | Evet | 182 profil, 11 aile, 910 üyelik |
| Language experience | Evet | 272 train deneyimi, 5.973 öğrenilmiş geçiş, 8 dil kümesi |
| Evaluate | Hayır | Gerçek holdout yok ve sentetik oranı 1.00 |

Artifact zinciri:

| Tür | Artifact kimliği | İçerik karması |
|---|---|---|
| Sentez | `3ca1ead2-e66f-41f2-9d3f-2994f05d6656` | `3e97a729b850c7d3a28afbb417fcafad4c98d34a72763d3397c8554f8c799c04` |
| Aile | `72edd110-d765-4df8-b39e-23ba191dfc00` | `fca4c40aac6fb17dc771ad49325153b2afded4ed9fc3ec1a930fd105ae7c8701` |
| Runtime | `7c5ef526-287c-4d59-ac90-8ca37f4ce96d` | `780bcafbb91eb718fdc62cf004aa3fba7a93c67a12c142e0d925fa3fe434e8ca` |

Sekiz TextLab artifacti de aynı işin `artifactIds` listesinde içerik karmalarıyla saklanır.
Sentez artifacti içerik-adresli cache'den değişmeden tekrar kullanılmıştır.

## 4. Aile kanıtı

Son aile çalışması `FR-9f9c27842164cbe9`; tam parmak izi
`9f9c27842164cbe972f2a16e0112fcbb` değeridir.

- 11 temel ailenin tamamı çok kaynaklıdır.
- 121 aileler arası geçiş vardır.
- 11 ailenin ablation katkısı ölçülmüş, 1 aile yerel katkı güven aralığını geçmiştir.
- 10 aile `operator`, 1 aile `content` bölge hipotezi taşır.
- Ailelerin tamamı `provisional`; bölge etiketlerinin hiçbiri semantik gerçek olarak
  doğrulanmış değildir.
- 8 emergent dil kümesi gözlenmiş, gerçek holdout olmadığı için `eligible` küme sayısı 0'dır.

Örneğin `AF-8a3e2e3131462f9b` ailesi 5 kaynakta 6 destek, `0.6815` cohesion,
`0.2157` separation ve `0.01028` pozitif yerel utility göstermiştir. Bu yalnız yerel aile
katkısıdır; ailenin kararlı veya anlamının doğru olduğu iddiası değildir.

## 5. Güvenlik kapısının doğru reddi

Değerlendirme kanıtı:

- Sentetik oran: `1.00`; izin verilen üst sınır: `0.25`.
- Gerçek validation örneği: `0`.
- Gerçek test örneği: `0`.
- `runtimeVerified=false`, `evaluation.passed=false`, `committed=false`.
- Snapshot durumu: `rejected`; boyut: 7.851.128 byte.
- `activeSnapshotId=null`.

Dolayısıyla kanıtlanan iddia şudur: 14B veri üretti; veri insan karantinasından geçti; sabit
kimlikli nöronlar, sentez adayları, aileler ve geçişler oluştu; fakat yalnız sentetik veri
model doğruluğu kanıtı sayılmadı ve etkin modele geçmedi.

## 6. Sürekli otomasyon durumu

Tamamlanan kaynakların temsil sürümleri iş kaydına yazıldı. Başarısız ama teknik olarak
tamamlanan bir çevrimden sonra aynı sekiz kaynak tekrar tekrar yeni parti sayılmaz. Sistem
şimdi 8 yeni veya temsil olarak değişmiş, insan-onaylı kaynak bekliyor. Eşik dolunca bütün
seçili korpusla tek bir yeni çevrim otomatik kuyruğa alınacaktır.

Önemli sınır: 14B öğretmen üretimi otomatik insan onayı değildir. Üretimler karantinada
bekler. Geçerli bir model değerlendirmesi için mevcut 8 sentetik kaynağın yanına en az 24
bağımsız gerçek kaynak gerekir (`8 / 32 = %25`); daha güvenli başlangıç 32 gerçek kaynakla
sentetik oranını `%20` yapmaktır.

## 7. Regresyon doğrulaması

- Vitest: 30 dosya, 155/155 test başarılı.
- Servis/runtime: 45/45 test başarılı.
- TypeScript proje denetimi: başarılı.
- Eğitim servisi üretim paketi: başarılı.

Kanıtlar canlı olarak şu salt-okunur uç noktalarda tekrar görülebilir:

- `/api/status`
- `/api/jobs/3cb33f0b-d72c-4476-9b1a-8e69d66f1a0b`
- `/api/pipeline/events?after=0`
- `/api/families/inspection?limit=100&memberLimit=8&transitionLimit=8`
- `/api/artifacts`
- `/api/snapshots`
- `/api/budgets`

## 8. v7 kapalı seçimli 68-cümle araştırma koşusu

Bu bölüm, yukarıdaki v6 kayıtlarından sonra yürütülen ayrı bir v7 deneyidir; önceki
kanıtları değiştirmez. Qwen serbest metin üretmedi. Her istekte, sunucunun kurup kalite
kapısından geçirdiği kapalı olay kartı panosundan benzersiz tamsayı kimlikleri seçti.
Nihai Türkçe cümle ve yapısal olay kartı deterministik sunucu renderer'ı tarafından
üretildi. Buradaki rastgelelik seed'li müfredat, kart panosu ve model seçimi düzeyindedir;
kontrolsüz serbest cümle üretimi değildir.

- Kampanya: `70cc512c-f52f-45cd-9b82-60c81a4febad`
- Model: `qwen2.5-coder:14b`; dondurulmuş digest:
  `9ec8897f747e246e970bc5cfdda85d22f1123dc2e3d34978a010a75968716849`
- Model parmak izi:
  `583e8013a1c22e9ae1c4f9502eafbfbb20ccfc481bdfdaa29ccb073ba5f0730e`
- Seed: `2026082907`
- Kip: `research-only`, `proposalOnly=true`
- Hedef: 68; model seçim partisi: 2; eğitim eşikleri: 24 / 48 / 68
- Protokol: v7,
  `b4761183324a24c840f29547fdea6d61dd14a959b1f8c1cbedab56756b17b064`
- Katalog: `turkish-foundation-event-cards-v2`,
  `9cbfa40ecf26d9506450f470f376345ddfeecc98401a49b3b5df523274d911ad`
- Başlangıç/bitiş: `2026-08-29T15:24:04.083Z` — `2026-08-29T16:39:13.020Z`
- Son iş durumu: `succeeded`; ilerleme: 1; hata ve duruş nedeni: yok

| Son kampanya sayacı | Değer |
|---|---:|
| Üretilen aday | 74 |
| Benzersiz kabul edilen | 68 |
| Yakın tekrar | 6 |
| Yapısal kalite reddi | 0 |
| Ollama isteği | 39 |
| Prompt / completion token | 16.741 / 477 |
| Toplam öğretmen tokenı | 17.218 |
| Tamamlanan eğitim çevrimi | 3 |
| Son planlanan parti | 34 |

34 değişmez `teacher-feed-batch` artifacti, 0–33 arası kesintisiz parti dizisini taşır.
Artifactlerdeki toplam `acceptedChoiceEvidence` sayısı 68'dir. Tüm seçimler tamsayıdır;
board, semantik kart, semantik kimlik ve nihai metin digest alanlarında eksik kayıt yoktur.
68 semantik kart digest'i, 68 semantik kimlik digest'i ve 68 nihai metin digest'i de
benzersizdir. 37 benzersiz doğrulanmış kart panosu kullanılmıştır. Bütün partilerde
`evaluationEligible=false`, `proposalOnly=true` ve aynı v7 protokol parmak izi bulunur.

| Eşik | Kaynak | Snapshot | Korpus parmak izi | Boyut | Son durum |
|---:|---:|---|---|---:|---|
| 24 | 24 | `teacher-feed-70cc512c-f52f-45cd-9b82-60c81a4febad-000001` | `dataset-417B78A4958FBD3AAD0EB24CEC35174A` | 91.518.754 byte | rejected |
| 48 | 48 | `teacher-feed-70cc512c-f52f-45cd-9b82-60c81a4febad-000002` | `dataset-D8D2D0F8CC1EF0FBB579493E8ED60389` | 171.726.041 byte | rejected |
| 68 | 68 | `teacher-feed-70cc512c-f52f-45cd-9b82-60c81a4febad-000003` | `dataset-57D5476B2E9158C6B75C8F6FD92C8990` | 251.779.438 byte | rejected |

Final 68-kaynak çevriminin ölçülebilir çıktıları:

| Aşama | Doğrulandı | Ölçülebilir çıktı |
|---|---|---|
| Collect / TextLab | Evet | 68 kaynak, 68 analiz, 3.902 sabit kimlikli profil |
| Synthesis | Evet | 9 sınırlı shard, 637.283 karşılaştırma, 374 benzersiz öneri adayı |
| Families | Evet | 77 aile, 19.510 üyelik, 924 aile geçişi, 68 farklı kaynak |
| Language experience | Evet | 5.995 train örneği, 137.192 öğrenilmiş geçiş, 64 emergent dil kümesi |
| Evaluate | Hayır | validation/test 0/0, sentetik oran 1.00, değerlendirme örneği 0 |

Final aile çalışması `FR-675b029f3b69fbe3`; tam parmak izi
`675b029f3b69fbe3af1ba4e1ad0e7e37` değeridir. 77 ailenin tamamı çok kaynaklıdır;
75'i `provisional`, 2'si `merge-proposed` durumundadır. 77 ailenin utility değeri
ölçülmüş, yalnız 5'i yerel doğrulama eşiğini geçmiştir. Bütün bölge hipotezleri
`operator` olarak kalır. Bunlar öğrenilmiş yapı hipotezleridir; insan anlamı, doğruluk
veya düşünce kanıtı değildir. Doğum önerisi ve otomatik doğum için uygun öneri sayısı 0'dır.

### 8.1 Kaynak sınırı ve dayanıklı kurtarma

İlk canlı ayarlar 4.096 MiB RAM ve 60 dakika çalışma sınırıydı. Final 68-kaynak çevrimi
bu sınırların yetersiz olduğunu canlı olarak gösterdi. İş checkpoint'te kontrollü
duraklatıldı; süre 180 dakikaya çıkarıldı. İkinci deneme `RAM_BUDGET_EXCEEDED` ile
güvenli biçimde durdu. İşletim sistemi örneklemesinde tepe fiziksel bellek yaklaşık
5,2 GiB ölçüldü.

Yalnız şu koşullarda açık devam komutuna izin veren kurtarma kapısı eklendi:

- iş bir `teacher-feed-campaign` olmalı;
- kip `research-only` ve `proposalOnly=true` olmalı;
- kalıcı ve yeniden oynatılabilir bir `pendingTrainingMilestone` bulunmalı;
- yarım kalmış `pendingGenerationBatch` veya `pendingBatchAccounting` bulunmamalı;
- hata yalnız `RAM_BUDGET_EXCEEDED`, `RUN_TIME_BUDGET_EXCEEDED` veya
  `LEASE_EXPIRED` olmalı.

Servis temiz biçimde yeniden başlatıldı, RAM tavanı 8.192 MiB yapıldı ve aynı kalıcı iş
üçüncü denemede devam etti. Devam öncesi ve sonrası öğretmen sayaçları değişmedi:
39 istek ve 17.218 token. Kaynak sayısı 68, kaynak kimlikleri ve deterministik üçüncü
snapshot kimliği korundu; 14B üretimi yeniden oynatılmadı. Final durumda üç pending alanı
da `null`, `lastTrainedAccepted=68` ve kaynak kimliklerinin 68'i de benzersizdir.

Canlı kaynak denetiminde 68/68 kayıt `approved`, `ready` ve seçili durumdadır. Günlük
kullanım 98 öğretmen isteği ve 42.823 token; kampanya öncesi taban çıkarıldığında kampanya
sayaçlarıyla tam olarak 39 istek ve 17.218 token eşleşir. Etkin snapshot koşudan önce ve
sonra `null` kaldı; kampanyaya ait `snapshot-activated` olayı yoktur.

### 8.2 Kanıtın sınırı

Bütün 68 kaynak aynı sentetik bağımsızlık grubunda ve yalnız `train` bölümündedir.
Validation/test kaynağı ve gerçek holdout yoktur. Bu nedenle final değerlendirmede
`overallScore=0`, `runtimeVerified=false`, `evaluation.passed=false` ve
`committed=false`; üç snapshot da reddedildi.

Kanıtlanan sonuç şudur: v7 kapalı seçim, deterministik Türkçe yüzey, yapısal semantik iz,
tekrar kapısı, tam-sayılı istek/token muhasebesi, dayanıklı checkpoint ve 24/48/68 nöron
çevrimleri canlı 14B modeliyle uçtan uca çalıştı. Yalnız sentetik veri dil doğruluğu,
genelleme veya düşünce yeteneği kanıtı sayılmadı.

### 8.3 Canlı darboğaz sonrası sertleştirme

Canlı profil, 5,2 GiB tepenin doğrudan 64×64 matrislerden gelmediğini gösterdi. 3.902
nöronun ham `I+R` Float32 toplamı yaklaşık 60,97 MiB'dir. Baskın maliyet,
`attachCorruptions` içinde 5.995 örneğin her biri için aynı granularity havuzunun tamamını
sıralayıp yalnız ilk adayı kullanmaktı. Eğitim öncesi hazırlık 493,27 saniye, gerçek
teacher-forcing döngüsü 24,81 saniye sürdü. Örnekleme yaklaşık 230 milyon tekrarlı digest
hesabı ve yoğun geçici allocation gösterdi.

Düzeltme, tam sıralama yerine aynı rank, metin anahtarı ve kararlı girdi sırası kurallarıyla
tek geçişli deterministik minimum seçer. Genel deterministik sıralama da anahtar ve rank'i
eleman başına bir kez hesaplayan decorate-sort-undecorate biçimine çevrildi. Golden
testlerde yeni seçimler eski comparator ve tam-sıralama referansıyla derin eşittir; tie,
boş havuz, aynı yüzey, aynı kimlik ve ortak üye dışlamaları ayrıca sınanır. Bu korpusta
beklenen digest işi yaklaşık 230 milyondan uygun çift sayısı olan yaklaşık 9,8 milyona
iner. Aile aşamasından sonra dil eğitiminin kullanmadığı büyük analiz ve sentez çalışma
kümeleri de serbest bırakılır.

Lease katmanında kısa heartbeat ve sabit execution lease ayrıldı. Recovery ancak ikisi de
bittiğinde takeover yapar. Checkpoint, progress ve bütün terminal job yazıları
`leaseOwner + attempt` ile fence edilir; eski worker yeni denemenin kaydını ezemez.
`LEASE_EXPIRED` açık kurtarması da yalnız eksiksiz ve üretim-pending alanları boş olan
research-only/proposal-only öğretmen kilometre taşında mümkündür. Ödünleşim şudur: yanlış
eşzamanlı takeover önlenir; gerçek worker çöküşünde recovery sabit azami çalışma süresi ve
grace sonuna kadar bekleyebilir.

Kuyruğun hem ana polling çevrimi hem lease yenileme timer'ı geçici `SQLITE_BUSY`
çakışmalarını süreç-düşüren istisnaya çevirmeden kaydeder ve yeniden dener. Kalp atımı
yenilemesi ayrıca tek `BEGIN IMMEDIATE` transaction'ında aynı owner, attempt ve çalışan
durumunu doğrular. Veri yazan bütün pipeline işlemleri yalnız `running` claim kabul eder;
`pause_requested`, `cancel_requested`, süresi geçmiş veya başka worker'a geçmiş denemeler
artifact, snapshot, kaynak, payload ya da olay kaydı yazamaz.

Controlled kipte snapshot aktivasyonu, kaynakların işlendi işareti ve üreten işin
`succeeded` terminal kaydı tek transaction'da tamamlanır. Bu terminal yazı başarısızsa üçü
birlikte geri alınır. Elle aktivasyonda da snapshot'ın üreten iş kimliği/attempt'i ve
başarılı sonuç sertifikası eşleşmek zorundadır. İptal, timeout veya worker kaybı sırasında
Ollama çağrıları tekrar oynatılmaz: gerçek token bilgisi geldiyse aynen, yanıt belirsizse
önceden kalıcılaştırılmış üst sınır tam bir kez muhasebeleştirilir.

İş kontrolü de aynı transaction sınırına alındı. Pause/cancel isteği güncel satırı kilit
altında okur; worker'ın başarı, retry veya hata kapatması aynı claim'i yeniden doğrular.
Böylece araya giren bir iptal eski worker tarafından `succeeded` ile ezilemez. Kuyrukta ya
da duraklatılmış öğretmen işinde kalmış istek/token borcu terminal duruma geçerken tam bir
kez kapatılır.

Öğretmen kampanyası artık her 14B ağ çağrısından **önce** istekle birlikte kalıcı prompt
ve completion üst sınırını atomik olarak ayırır. Kota yetmiyorsa model hiç çağrılmaz ve
deterministik seed değişmeden kalır. Yanıt ya da sınıflandırılmış hata geldiğinde bu üst
sınır, checkpoint ile aynı transaction içinde gerçek veya muhafazakâr kullanıma
uzlaştırılır. Eski tek-seferlik öğretmen ve düşünce çapraz-kontrol yolları da ağ çağrısı
öncesi aynı sınırlı rezervasyon kuralını uygular ve her çağrıyı kalıcı, tek-kapatımlık bir
rezervasyon kimliğiyle izler. Bütün öğretmen yolları Ollama sınırında `num_ctx=32768`
uygular; böylece model şablonundan bağımsız gerçek bir prompt tavanı vardır. Geçersiz
`null`, boş metin veya boolean sayaçlar raporlanmış sıfır kabul edilmez. Sağlayıcı yine de
rezervasyondan yüksek bir sayı bildirirse gerçek sayı kaydedilir ve sonraki çağrı ağ
erişiminden önce bütçede kesilir. Böylece süreç çökmesi, eşzamanlı iş veya eksik/bozuk
Ollama sayacı bütçeyi sessizce aşamaz; belirsizlikte eksik saymak yerine kayıtlı üst sınır
kullanılır.

Model seçimi ve önceden iptal edilmiş çağrı, rezervasyondan önce doğrulanır; ağ erişimi
olmayan bu iki durumda istek veya token yazılmaz. Kalıcı `disagreed`/`unavailable`
düşünce çapraz-kontrolü pause/resume sonrasında yeniden satın alınmaz. Yükseltme sırasında
v7'den kalmış belirsiz bir çağrının küçük byte tahmini de güvenilir sayılmaz: prompt
hesabı 32.768 tabanına çıkarılır, daha önce ayrılmış miktarın yalnız farkı atomik ve tek
seferlik eklenir.

Tarayıcı yanıt bağlantısı öğretmen çağrısı bitmeden kapanırsa aktif istek iptal edilir;
belirsiz kalan kullanım yine kayıtlı üst sınırla kapatılır ve sıradaki parça başlatılmaz.
Düşünce bölümü pause/cancel/completion geçişleri ile son artifact yazımı da atomik durum
kapıları kullanır. Böylece eski bir worker daha yeni kontrol kararını veya terminal durumu
ezemez; paralel dalların ve kalıcı başlığın yarım sonuçla kirlenmesi engellenir.
Düşünce çapraz-kontrolünde bölümün hâlâ `running` olduğu doğrulaması ile token
rezervasyonu da aynı transaction içindedir; başka bir servis örneğinden gelen kontrol
kararı önce yazıldıysa yeni Ollama geçişi hiç satın alınmaz. Sahipsiz `queued/paused`
veya terminal legacy v7 çağrı borçları servis açılışında ve claim/control sınırlarında 32.768 context
tabanına idempotent uzlaştırılır; canlı worker claim'i yerinde bırakılır. Eski çok parçalı
öğretmen akışı da parçalar arasında event loop'a dönerek bağlantı kapanışını yeni bir
rezervasyondan önce gözlemler.

Bu sertleştirme yeni kampanyalar için öğretmen sözleşmesini v8'e yükseltti. Yukarıdaki
tamamlanmış canlı koşu tarihsel ve değişmez v7 kanıtıdır; yeniden yorumlanmadı veya
yeniden yazılmadı.

Son regresyon sonucu:

- Vitest: 38 dosya, 204/204 test başarılı.
- Servis/runtime: 158/158 test başarılı.
- Store/pipeline/queue/HTTP/öğretmen odaklı muhasebe, yarış ve yeniden başlatma yolları
  ayrıca iki bağımsız denetim turundan geçirildi.
- TypeScript proje denetimi: başarılı.
- Arayüz ve eğitim servisi üretim paketleri: başarılı.
