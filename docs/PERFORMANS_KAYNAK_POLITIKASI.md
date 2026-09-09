# Eğitim Performans ve Kaynak Politikası

Bu politika, model matematiğini değiştirmeden 16 GiB birleşik belleğe ve sınırlı SSD boşluğuna sahip geliştirme makinesinde uzun eğitimlerin güvenli ve tekrar oynatılabilir kalmasını sağlar.

## RAM zarfı

- Eğitim servisi için varsayılan sert RSS sınırı **4.096 MiB**'dir. 16 GiB geliştirme
  makinesinde Ollama kapalı bağımsız benchmark, gözlenmiş ve checkpoint'lenmiş tepe
  4 GiB'yi aştığında açık kullanıcı kararıyla en fazla **8.192 MiB** zarfa alınabilir;
  canlı derin operasyon koşusu **6.144 MiB** kullanır.
- `%70` üzerinde yeni büyük çalışma kümeleri ertelenebilir (`soft`).
- `%85` üzerinde uzak nöronlar RAM'den düşürülür ve yalnız SSD kapısından çağrılır (`spill`).
- `%100` üzerinde çalışan bölüm geri alınabilir checkpoint sınırında duraklatılır (`hard`).
- 14B öğretmen yalnız cümle üretirken bellekte tutulur. Matris, aile ve dil aşamasından hemen önce Ollama `keep_alive=0` ile serbest bırakılır.
- Nöron cache bütçesi 384 MiB'dir: yaklaşık `%75` çözülmüş hot vektörlere, `%25` yalnız istenen warm kayıt dilimlerine ayrılır.

## Ollama bağlam politikası

Ollama çağrıları tek bir bağlam boyutunu paylaşmaz. Bellek maliyeti, çağrının gerçekten
gerektirdiği protokole göre sınırlandırılır:

- Kapalı seçimli 14B besleme protokolü `num_ctx=4096` kullanır. Bu çağrıda model yalnız
  sunucunun doğruladığı kart panosundan kimlik seçer; uzun metin veya bütün korpus prompta
  konmaz.
- Genel amaçlı yapılandırılmış öğretmen ve Düşünce Akışı çapraz denetimleri için üst sınır
  `num_ctx=32768` olarak kalır. Bu iki iş türü besleme worker'ının 4K sınırını miras almaz.
- Token kotası, besleme çağrısında gerçekten zorlanan `4096 + completion üst sınırı`
  üzerinden çağrıdan önce kalıcı rezerve edilir. Yanıt veya süreç belirsiz kalırsa aynı
  borç ikinci kez oynatılmadan muhafazakâr biçimde kapatılır.
- Besleme protokolü bağlam boyutunu ve model digest'ini checkpoint'e yazar. Eski veya
  farklı protokollü yarım çağrı yeni ayarla sessizce devam etmez.
- Öğretmen worker'ı başarı, hata, duraklatma, iptal ve timeout çıkışlarının tamamında
  bağımsız dört saniyelik kapanış isteğiyle modeli boşaltmayı dener. Eğitim kilometre taşı
  başlamadan önce de model serbest bırakılır.

30 Ağustos 2026 canlı A/B profilinde aynı Qwen 14B modeli için 32K görünüm
15.808.405.174 byte'a çıkmış ve ilk çağrı 300 saniyede tamamlanamamıştır. Aynı kapalı
seçim protokolü 4K bağlamda 9.459.613.039 byte model görünümüyle 12,72 saniyede geçerli
yanıt vermiştir. Bu yaklaşık 6,35 GB daha küçük çalışma görünümüdür; eğitim kalitesini
azaltmak için değil, kısa ve şeması kapalı besleme çağrısının gerçek ihtiyacına göre
ayrılmıştır.

## SSD zarfı

- Yeni eğitim kayıtlarının mantıksal/kalıcı sınırı **4.096 MiB**'dir.
- WAL ve son kurtarma checkpoint'i için **512 MiB** fiziksel alan ayrılır.
- Kullanılabilir boşluk 20 GiB altındaysa arayüz uyarır.
- WAL rezervinden sonra 15 GiB kalmıyorsa yeni blok başlatılmaz; mevcut bölümün checkpoint yazmasına izin verilir.
- Büyük yeni artifact payload'ları gzip seviye 6 ile kayıpsız sıkıştırılır. Açılırken byte uzunluğu ve SHA-256 doğrulanır. Eski inline artifact'ler dönüştürülmeden okunmaya devam eder.
- Tek bir artifact'in açılmış JSON gövdesi en fazla **256 MiB** olabilir. Yazma ve okuma aynı sınırı uygular; daha büyük sonuçlar iz/checkpoint manifestleri gibi parçalara ayrılmadan kabul edilmez.
- Runtime model, aynı snapshot'taki büyük `FamilyRun` gövdesini ikinci kez kopyalamaz; değişmez artifact kimliği ve fingerprint ile başvurur.
- Durum, snapshot ve aile ekranları payload açmaz; SQLite metadata yan tablosunu kullanır.
- Sık progress/event yazıları 2 saniyelik pencerede birleştirilir. Checkpoint, başlangıç, bitiş ve hata kayıtları anında yazılır.

## Nöron kullanılabilirlik olasılığı

Nöronun RAM'de kalma kararı, eğitim sonucunu değiştirmeyen sabit noktalı bir tahmindir:

\[
u=0.35(1-d)+0.30p_{rota}+0.20p_{aile}+0.15p_{kullanım}
\]

Burada `d` sinyal uzaklığıdır. Diğer kanıtlar aynıyken uzaklık arttığında `u` artamaz. `u≥0.62` hot, `0.18≤u<0.62` warm, daha düşük değer cold olur. Cold nöron için tüm sayfa değil, manifestteki kesin `offset/length` aralığı okunur. Her çalışma kümesi en fazla 128 nörondur.

## Durdurma ve doğruluk

- Kaynak baskısı hiçbir zaman başarı olarak yorumlanmaz; çalışma `paused` olur.
- 14B çıktısı validation/test verisi veya yapısal başarı hükmü olamaz.
- “Anlam/mantık oluştu” sonucu ancak Ollama kullanmayan, 144 olay birleşimli, 5 seed'li bağımsız yapısal benchmarkın bütün kapıları geçmesiyle yazılır.
- Aynı snapshot, giriş ve seed için cache katmanı değişse de nöron byte'ları ve tick digest zinciri aynı kalmalıdır.

## Ölçüm hedefleri

- 10K nöronda p95 `≤50 ms/tick`, 100K nöronda `≤100 ms/tick` hedefi.
- Eğitim worker'ı RSS sert sınırı `≤4 GiB`; servis metadata sorguları bu bütçeye büyük payload eklememeli.
- Durum üretimi `≤100 ms`, snapshot listesi `≤200 ms` hedefi.
- Uzun koşu sonunda worker kapanmalı; RAM işletim sistemine geri dönmeli.
