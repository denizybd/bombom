# Derin Operasyon — 30 Ağustos 2026

Bu tutanak, Qwen 14B ile sürekli araştırma beslemesinin ve RAM/SSD katmanlama
politikasının canlı makinede devreye alınmasını kaydeder. Buradaki “anlam/mantık” ifadesi
bir model iddiası değildir: yalnız Ollama'dan bağımsız yapısal benchmarkın bütün kapıları
geçmesiyle oluşabilecek ölçüm sonucudur.

## Canlı kampanya

- Kampanya: `d9ad78b4-6346-4379-9adf-4522ad496872`
- Başlangıç: `2026-08-29T22:27:25.910Z` (`2026-08-30 01:27`, UTC+03:00)
- Model: `qwen2.5-coder:14b`, 14.8B, Q4_K_M
- Dondurulmuş model digest'i:
  `9ec8897f747e246e970bc5cfdda85d22f1123dc2e3d34978a010a75968716849`
- Seed: `2026083001`
- Protokol: kapalı seçimli Türkçe temel müfredat v9, `num_ctx=4096`
- Kip: `research-only`; etkin modele doğrudan yazma yok
- Blok: 64 yeni kabul; azami 32 blok; normalleştirilmiş üst hedef 2.048 kabul
- Eğitim kilometre taşı: her 64 yeni kabul
- Başlangıç korpusu: 68 seçili öğretmen kaynağı
- Başlangıç korpus parmak izi:
  `e82a92237a94234f5df8fc1ece70626974798ded618b7246f53683548545dd89`

İlk canlı kanıt anında kampanya `running / teacher-feed:generating` durumundaydı. İlk
parti tek çağrıda 2/2, ilk iki parti toplamında 4/4 örnek kalıcı kalite kapısından geçti;
tekrar ve yapısal kalite reddi sıfırdı. İkinci parti eksik seçimi tamamlamak için bir ek
çağrı kullandığından bu noktadaki sayaç 3 istek ve 811 token'dı. Kayıtların provenance
alanları kampanya kimliği, seed, deterministik batch kimliği ve batch sırasını taşıyor.

`2026-08-29T22:32:28.841Z` kalıcı durum kesitinde sayaç 7 tamamlanmış batch,
14 üretilen, 14 kabul, 0 tekrar, 0 kalite reddi, 8 Ollama isteği ve 2.825 token'dı.
SQLite kaynak metadata indeksinde aynı kampanya kimliğiyle 14 kayıt, artifact arşivinde
aynı işe bağlı 7 değişmez `teacher-feed-batch` kaydı bağımsız olarak sayıldı.

Kampanya açık bırakılmıştır. Her 64 kabulden sonra model RAM'den çıkarılır, gerçek
TextLab → Sentez → Aile → Dil Deneyimi zinciri çalışır ve sonuç bağımsız yapısal kapıya
girer. Kapı geçmezse bir sonraki 64'lük bloğa yalnız kaynak rezervleri uygunsa devam edilir.
Azami blok, kaynak veya disk sınırı başarı sayılmaz; kampanya geri alınabilir checkpoint'te
`paused` olur.

### Mac yeniden başlatma kurtarması

Makine kapandığında kampanya ilk 64 kabulü, 33 isteği, 14.294 token muhasebesini ve
`teacher-feed-d9ad78b4-6346-4379-9adf-4522ad496872-000001` snapshot'ını korudu. Eski
worker'ın heartbeat'i bitmişti fakat uzun CPU işini yanlışlıkla çalmamak için ayrılan sabit
execution lease hâlâ canlıydı. Açık kullanıcı `resume` isteği için dar bir kurtarma kapısı
eklendi: yalnız `JOB_WORKER_EXITED` kanıtlı, heartbeat'i bitmiş, research-only sürekli
kampanya; açık öğretmen çağrısı, batch muhasebesi veya eğitim kilometre taşı yoksa ve
`accepted == lastTrainedAccepted` olan snapshot/benchmark sınırında yeniden kuyruğa
alınabilir. Canlı heartbeat veya belirsiz ücretli çağrı taşıyan iş bu kapıdan geçemez.

Kampanya aynı kimlik ve sayaçlarla attempt 3'e alındı. Ollama çağrısı tekrarlanmadı;
worker doğrudan 7.366 nöronlu bağımsız yapısal benchmarkı başlattı. Benchmarkın bildirilen
dense vektör gövdesi 120.684.544 byte, öngörülen geçici tepesi 362.053.632 byte olup
16.384 nöron / 768 MiB güvenlik sınırının içindedir.

## 32K darboğazı ve 4K düzeltmesi

İlk profil kampanyası `a158bf92-52cf-4b5f-bbcb-ea5262868771` ile 32K bağlamda yapıldı.
Model görünümü 15.808.405.174 byte, Ollama'nın bildirdiği VRAM görünümü
11.583.220.612 byte ve runner RSS'i yaklaşık 11,2 GiB oldu. Swap yaklaşık 7,8 GiB'ye,
fiziksel boş disk yaklaşık 12 GiB'ye kadar geriledi. İlk çağrı 300 saniyelik sınırda
sonuçlanmadı. Kampanya veri üretmeden kontrollü duraklatılıp iptal edildi; kalıcı sayaçları
0 kabul, 1 istek ve muhafazakâr 32.896 token'dır.

Aynı model ve aynı kapalı seçim işi `num_ctx=4096` ile doğrudan profillendi:

| Ölçüm | 32K | 4K |
|---|---:|---:|
| Model çalışma görünümü | 15.808.405.174 byte | 9.459.613.039 byte |
| İlk geçerli sonuç | 300 sn timeout | 12,7229 sn |
| Sonuç | yok | `{"choices":[3,9]}` |
| Prompt / çıktı token | belirsiz borç | 190 / 13 |

4K çalışma görünümü yaklaşık 6,35 GB küçüldü. Canlı üretim runner'ı `-c 4096` ile
başladı ve Ollama API'si de `context_length=4096` bildirdi. Bu profil çağrısı eğitim
verisine veya kotasına yazılmadı; yalnız kapasite ölçümüdür.

## RAM politikası

- Servis + eğitim worker'ı için tek ortak sert RSS zarfı 4.096 MiB'dir.
- `%70 / %85 / %100` eşikleri sırasıyla soft, spill ve hard katmanlarıdır.
- Worker ayrı süreçte çalışır; ebeveyn servis RSS'i worker ölçümüne eklenir. Böylece her
  süreç yanlışlıkla ayrı bir 4 GiB hakkı almaz.
- Aynı anda yalnız bir eğitim işi ve en fazla iki deneme vardır.
- 14B model yalnız seçim üretimi sırasında yüklüdür; eğitim aşaması ve bütün worker çıkış
  yolları `keep_alive=0` temizliği uygular.
- Canlı 4K koşuda servis yaklaşık 45 MiB, worker ilk üretim bölümünde yaklaşık 359 MiB
  RSS kullanırken model ağırlıkları Ollama'nın ayrı sürecinde tutuldu. İşletim sistemi
  baskısı başarı değildir ve işin matematiğini değiştirmez.

## SSD ve nöron yerleşimi

- Eğitim arşivinin mantıksal sınırı 4.096 MiB, WAL/checkpoint rezervi 512 MiB'dir.
- Kullanılabilir fiziksel disk 20 GiB altında uyarı verir; 15 GiB altında yeni blok
  başlamaz. Çalışan blok son kurtarma checkpoint'ini yazabilsin diye yalnız gerçek hard
  sınırda kesilir.
- Canlı başlangıçta WAL sonrası kullanılabilir alan yaklaşık 20.106 MiB ve `canStart=true`
  idi. Model yüklenip swap yerleşirken yaklaşık 17.587 MiB ölçüldü; sistem uyarı katmanında
  kaldı ve bir sonraki blok için 15 GiB giriş kapısı korunuyor.
- Büyük immutable artifact'ler gzip-6 ile içerik adresli saklanır. Kaynak ve artifact
  durum listeleri payload açmadan SQLite metadata indekslerinden okunur.
- Nöron sayfaları hedef 2 MiB, sert üst sınır 4 MiB'dir. Her kayıt SHA-256, kesin
  `offset/length` ve değişmez `I` digest'i ile doğrulanır.

Nöronun RAM'de kalma skoru:

\[
u=0.35(1-d)+0.30p_{rota}+0.20p_{aile}+0.15p_{kullanım}
\]

`u≥0.62` hot, `0.18≤u<0.62` warm, altı cold'dur. Uzaklık arttıkça diğer kanıtlar
sabitken skor artamaz. Cold nöronun bütün sayfası RAM'e alınmaz; yalnız manifestteki
kesin byte dilimi SSD'den okunur. Tick başına en fazla 128 nöronluk çalışma kümesi,
68 LSH bucket, 512 LSH girdisi ve toplam 1.104 metadata incelemesi vardır. Hot/warm
cache toplamı 384 MiB ile sınırlıdır.

## Yapısal durdurma kapısı

“Yeterince eğitim oldu” kararı kabul sayısından veya Ollama görüşünden çıkmaz. Gereken
bağımsız kapı 144 kontrollü olay birleşimi, en az `%30` gerçek holdout, 5 seed,
kaynak-grubu bootstrap `%95` güven aralığı, recurrence/baseline karşılaştırmaları,
hedef ve sham aile/kenar ablation'ları, bozuntu sonrası geri dönüş ve doğum
hassasiyeti/yanlış-doğum ölçümlerini birlikte ister.

Ollama üretimi veya Ollama etiketi validation/test'e giremez. İlk 64 kabulden oluşan
snapshot'ın beş-seed `full` ölçümü 440 gözlemde ortalama `0,269076`, `%95` güven aralığında
`0,267018–0,271050` verdi. Alt sınır `0,75` kapısının altında olduğu için sonuç
`failed-early` olarak kesinleştirildi; kalan 3.720 tanısal episode çalıştırılmadı. Bu sonuç
öğrenmenin yapısal başarı kapısını henüz geçmediğini gösterir; anlam, mantık, bilinç veya
doğal dil çıkışı oluştuğu iddia edilmemektedir. Blok kalıcı kapandıktan sonra aynı kampanya
64 kabulden 128 hedefine geçti; 14B model aynı doğrulanmış digest ve 4K bağlamla yeniden
yüklendi. Canlı doğrulama sonunda sayaç 69 üretim, 68 kabul, 35 istek ve 15.206 tokena
ilerledi; ikinci blok hedefi 128 kabul olarak kaldı.
Dense yapısal benchmark 16.384 nöron ya da
768 MiB tahmini tepeyi aşarsa yanlış başarı vermek yerine
`STRUCTURAL_BENCHMARK_DENSE_BASE_UNSUPPORTED` ile duraklar.

### Aşamalı benchmark ve kesin kısa devre

Tam benchmarkın 4.160 episode'luk maliyetini öğrenmenin çok erken döneminde tekrar tekrar
ödememek için değerlendirme üç aşamaya ayrılmıştır:

1. Dondurulmuş holdout'un tamamı önce tek bağımsız seed ile `full` modelde ölçülür. Ortalama
   graph-F1 `0,60` altında kalırsa ayrı bir
   `independent-structural-benchmark-screen-v1` kanıtı yazılır. Bu yalnız çalışma
   zamanlaması için bir kabul ön ölçümüdür; sonuç `screened-out`, `passed=false` ve
   `conclusive=false` olur. Yapısal kapıyı hiçbir koşulda geçiremez.
2. Ön ölçüm eşiği aşılırsa `full` varyantın beş seed'i tamamlanır. Kaynak-grubu bootstrap
   `%95` güven aralığının alt sınırı `0,75` altında kalıyorsa ana başarı koşulunun artık
   geçemeyeceği matematiksel olarak kesindir. Ayrı
   `independent-structural-benchmark-early-fail-v1` kanıtı; gerçek intervali, tam
   440 gözlemi ve atlanan varyantları kaydeder. Sonuç `failed-early`, `passed=false` ve
   `conclusive=true` olur.
3. Yalnız beş-seed ana ölçümün güven alt sınırı `0,75` veya üstündeyse baseline,
   recurrence, hedef/sham ablation, sinyal bozuntusu ve doğum ölçümlerinin tamamı çalışır.

Ölçülmeyen metriklere `0` veya `null` yer tutucu yazılmaz; erken karar kayıtları tam
benchmark artifact'i gibi sunulmaz. Tam benchmark, erken başarısızlık ve ön ölçüm ayrı
format ve artifact türleridir. Her kayıt exact snapshot/input-manifest bağı, Ollama'sız
kaynak politikası, observation chunk SHA-256 karmaları ve karar-politikası parmak iziyle
yeniden doğrulanır. Geçersiz veya eski manifestli bir kayıt önbelleği kilitleyemez.

Bu değişiklik mevcut beş-seed observation parçalarını korumak için ana runner ve benchmark
parmak izlerini değiştirmez. Duraklatma veya makine kapanması sonrasında tamamlanmış seed
parçaları yeniden hesaplanmadan okunur. Başarısız blok kalıcı olarak kapandıktan sonra aynı
14B kampanyası bir sonraki 64 kabul hedefine otomatik döner.

## Dayanıklılık ve doğrulama

- Canlı veri tabanı önce şu dosyaya kopyalandı:
  `server-data/backups/autonomous-training-pre-deep-operation-2026-08-30.sqlite`
- Yedek SHA-256:
  `0fa49428542497d463daf2d227990045164ae99156e8f749da2ebf6aba9f4647`
- Checkpoint, token rezervasyonu, kaynak yazımı ve kampanya ilerlemesi atomiktir.
- Pause/resume, iptal, stale worker, timeout, çift commit ve model drift kapıları testlidir.
- Tam servis/runtime testi: 246/246 geçti.
- İstemci testi: 217/217 geçti.
- TypeScript denetimi, istemci üretim paketi ve eğitim servisi üretim paketi geçti.

Canlı ekran: `http://127.0.0.1:5173/otonom-egitim/`
