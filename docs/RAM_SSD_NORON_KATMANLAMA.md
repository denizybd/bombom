# RAM/SSD Nöron Katmanlama Sözleşmesi

Bu katmanlama, nöron matematiğini değiştirmeden yalnız nöron vektörlerinin nerede tutulduğunu belirler. Aynı model, giriş ve seed için alınan nöronların `Float32` byte'ları ve sabit `I` digest'leri aynıdır; dolayısıyla düşünce rotasının matematiksel sonucu cache durumundan etkilenmez.

## Yerleşim politikası

Her aday için yakın gelecekte kullanılma olasılığı sabit noktalı (`10⁶`) bir skorla hesaplanır:

\[
u=0.35(1-d)+0.30p_{rota}+0.20p_{aile}+0.15p_{kullanım}
\]

Burada `d∈[0,1]` sinyal–nöron uzaklığıdır. Diğer kanıtlar sabitken uzaklık arttığında skor kesinlikle artamaz.

- **Hot:** Tam çözülmüş `I/R` vektörü RAM'deki sınırlı LRU cache'tedir.
- **Warm:** Yalnız istenmiş nöronun hash'li binary kayıt dilimi sınırlı RAM cache'indedir; aynı sayfadaki diğer nöronlar örtük biçimde yüklenmez.
- **Cold:** RAM'de vektör veya sayfa gövdesi yoktur; yalnız küçük manifest/index kaydı bulunur.

Varsayılan eşikler `hot≥0.62`, `warm≥0.18` değerleridir. Hot terfisi iki ardışık erişim ister. Skor erişim sıralı deterministik EMA ile güncellenir; wall-clock zamanı kullanılmaz. Kullanılmayan kayıtların efektif skoru mantıksal erişim sayısıyla azalır. Önce eşiğin altındaki veriler düşürülür, sonra byte bütçesi aşılırsa düşük fayda öncelikli deterministik LRU uygulanır.

Uzak/düşük olasılıklı bir cold nöron istendiğinde tüm sayfa açılmaz. RAM'e yalnız manifestin gösterdiği kesin `offset/length` aralığı okunur, kayıt hash'i doğrulanır ve çağrıya kopyası verilir. Skor warm eşiğinin altındaysa depo bu vektörü çağrı sonrasında RAM'de tutmaz. Tüm sayfa hash'i `verifyPageForNeuron(...)` ile başlangıçta veya düşük öncelikli arka plan denetiminde ayrıca doğrulanabilir.

## SSD biçimi ve güvenlik

- Nöronlar kimliğe göre sıralanıp hedefi 2 MiB, sert üst sınırı 4 MiB olan değişmez sayfalara ayrılır.
- Sayfa ve kayıtlar SHA-256 ile içerik adreslidir.
- Manifest; model/head fingerprintini, temsil uzayını, I/R uzunluğunu, her sayfanın hash/boyutunu ve her nöronun kesin byte aralığını taşır.
- `I` ve `R`, platformdan bağımsız `IEEE754 Float32 little-endian` olarak yazılır.
- Okumada kayıt hash'i ve `I` digest'i yeniden doğrulanır. Warm sayfa yüklemesinde ayrıca tüm sayfa hash'i doğrulanır.
- Yazma yeni bir içerik-adresli görünüm oluşturur. Kaynak snapshot, `FamilyRun` veya eski artifact yerinde değiştirilmez ve silinmez.
- Aynı girdi ve sayfa politikası aynı manifest fingerprintini üretir; manifestte zaman damgası yoktur.

## Düşünce runtime entegrasyon kapısı

`runtime/thought-neuron-residency.mjs` var olan `ThoughtModel/ThoughtHead` için opt-in bir SSD görünümü oluşturur. Mevcut tick motoru senkron olduğu için entegrasyon iki aşamalıdır:

1. Seyrek `H`, aile üyeliği ve LSH metadata indeksleri en fazla 128 aday kimliği üretir.
2. `withCandidateLookup(...)` yalnız bu kimliklerin vektörlerini RAM çalışma kümesine getirir ve mevcut senkron rota/tick fonksiyonu callback içinde çalışır.

Bu sınır, tick sırasında bütün nöronları ya da bütün SSD sayfalarını taramayı yasaklar. Çalışma kümesi üst sınırı ayrıca policy tarafından zorlanır. Cache politika durumu `exportPolicyState()` ile checkpoint edilebilir; durum farklı manifest veya politikada geri yüklenemez.

Üretim sinyal kapısı manifest açılırken bir kez kurulan dört adet 16-bit sign-LSH bandı kullanır. Her tick'te exact band ve tek-bit komşuları multi-probe edilir: en fazla 68 bucket ve 512 kompakt indeks girdisi incelenir. Üyelik, çıkış kenarı ve aile güçlü-üye indeksleri de önceden hazırlanır; toplam tick metadata incelemesi en fazla 1.104 girdidir. Keşif, bütün kimlikleri SHA ile sıralamak yerine seed'den türetilen eş-asal stride ile sabit sayıda kimlik örnekler. Katalog 128 veya daha küçükse tüm nöronlar çalışma kümesine girer ve eski motorla tick digest eşdeğerliği korunur. Daha büyük katalogda sonuç bilinçli olarak `dynamic-bounded-approximation` tanısı taşır; global exact arama iddiası yoktur.

İlk residency kurulumu için artifact metadata boyutları vektörler decode edilmeden önce topluca RAM/SSD kaynak kapısından geçirilir. Güvenli rezerv yoksa `NEURON_RESIDENCY_RESOURCE_WAIT` ile hiçbir artifact açılmadan beklenir. Kabul edilen ilk bootstrap, kaynak artifact biçimi nedeniyle bir kez dense katalog doğrulaması yapmak zorundadır; SSD sayfaları page-bounded akışla yazılır ve tüm katalog için ikinci bir record-buffer veya routing-vector kopyası oluşturulmaz.

Bağımsız yapısal benchmark henüz bu aday kapısına bağlanmış değildir. Canlı yaklaşık 3.9K katalog mevcut bounded-dense benchmark zarfında exact çalışır. Benchmark tarafında 16.384 nöron veya 768 MiB tahmini geçici dense tepe sınırı aşılırsa iş başarı sayılmaz; `JOB_PAUSED / STRUCTURAL_BENCHMARK_DENSE_BASE_UNSUPPORTED` ile durur. Bu nedenle 100K düşünce runtime paging desteği vardır, fakat 100K full-dense yapısal benchmark desteği varmış gibi raporlanmaz.

`repository.manifest` yalnız payload içermeyen manifest özetini döndürür; büyük nöron indeksini ikinci kez kopyalamaz. Tek bir nöronun katmanı, digest'i, SSD sayfa kapısı ve kullanılabilirlik skoru `repository.describe(id)` ile O(1) okunur.

Sunucu kaynak politikasına önerilen dar eşleme:

- `hotBudgetBytes = neuronCacheMb × 0.75 × MiB`
- `warmBudgetBytes = neuronCacheMb × 0.25 × MiB`
- `pageTargetBytes = resourcePolicy.neuronPageBytes`
- `maximumPageBytes = 2 × resourcePolicy.neuronPageBytes`

Katmanlama modülü bu genel ayarlara bağımlı değildir; bağımsız bütçelerle de kullanılabilir. Böylece migration veya rollout önce shadow/trace-only kipinde doğrulanabilir.
