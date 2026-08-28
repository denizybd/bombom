# Sinaps Sandbox

Katman sırası olmayan yönlü bir nöron grafında, işaretli `32 × 32` sinyal matrisinin adım adım nasıl dönüştüğünü gösteren görsel bir sandbox.

Uygulama iki farklı ilişkiyi özellikle ayrı tutar:

- **Gerçek benzerlik K:** Nöron prototiplerinin kosinüs benzerliği.
- **Kas hafızası H:** Kullanım kanıtı `W` üzerinden satır bazında normalize edilen geçiş alışkanlığı.

Sinyal ham haliyle ilk nörona girer. Her nöronda uyuma göre dönüşür, adaylar kas hafızası sırasıyla değerlendirilir ve gerçekleşen geçişin `W` kanıtı güçlenir. Nöron oluşma baskısı yalnızca gösterilir; otomatik nöron üretmez.

## Çalıştırma

Gereksinimler: güncel Node.js ve pnpm.

```bash
pnpm install
pnpm dev
```

Vite'ın gösterdiği yerel adresi tarayıcıda açın. Varsayılan adres genellikle `http://localhost:5173` olur.

## Doğrulama

```bash
pnpm test
pnpm build
```

## Temel kullanım

1. Soldaki editörde başlangıç sinyalini pozitif, negatif veya silme aracıyla boyayın.
2. Ortadaki **Yeni nöron** düğmesiyle ikinci bir nöron ekleyin.
3. Sağdaki **Düzenle** görünümünde seçili nöronun prototipini boyayın.
4. **Tek adım** ile matematiği adım adım izleyin veya **Başlat** ile otomatik oynatın.
5. Alt alanda zaman çizgisi, `H`, `K`, oluşma baskısı ve aday günlüğü arasında geçiş yapın.

Uygulama ağı tarayıcıda yerel olarak saklar. Üstteki içe/dışa aktarma kontrolleriyle durum JSON dosyası olarak taşınabilir. **Deneyi sıfırla** kas hafızasını korur; çöp kutusu ile açılan **Ağı sıfırla** akışı bütün kalıcı ağı açık onaydan sonra siler.

Matematiksel tanımlar için [docs/MATEMATIKSEL_MODEL.md](docs/MATEMATIKSEL_MODEL.md) dosyasına bakın.

## Yapı

- `src/engine/`: Saf TypeScript matematik motoru, seeded RNG ve testler.
- `src/components/`: Erişilebilir 32 × 32 matris görünümleri ve boyama editörü.
- `src/App.tsx`: Simülasyon denetleyicisi, SVG ağ sahnesi, kalıcılık ve bütün kullanıcı akışları.
- `src/styles.css`: Masaüstü odaklı görsel sistem ve responsive düzen.

Bu ilk sürümde backend, metin/LLM entegrasyonu, otomatik nöron doğumu, prototiplerin otomatik öğrenmesi, nöron silme/birleştirme ve ödül tabanlı pekiştirme yoktur.

## Ayrı program: Metin Laboratuvarı

Ham metni paragraf köklerinden başlayarak kayıpsız bir oluş ağacına dönüştüren ikinci uygulama aynı projede, fakat Sinaps'tan bağımsız çalışır:

```text
http://localhost:5173/metin-laboratuvari/
```

Bu program yüzey kimliği ile bağlamsal ilişki alanını ayrı maskelerde tutar; ebeveyn ve kardeş mikro-izlerini üretir, küçük deneylerde bütün nöron çiftlerinin `K` benzerliğini hesaplar ve başlangıç `H` olasılıklarını hazırlar. Kendi Worker işlem hattı, IndexedDB kaydı, proje dosyası ve testleri vardır. Sinaps'ın ağına veya yerel kayıt anahtarına dokunmaz.

Matematiksel tanım için [docs/METIN_LABORATUVARI_MODEL.md](docs/METIN_LABORATUVARI_MODEL.md) dosyasına bakın.

## Ayrı program: Nöron Sentezi

Farklı ham metinlerde tekrarlanan yüzey ve bağlam örüntülerinden, insan onayına sunulan yeni nöron adayları çıkarır:

```text
http://localhost:5173/noron-sentezi/
```

Sentez programı yoğun yeni bir çift tablosu saklamaz; çapraz kaynak çiftlerini Worker içinde akıtır, her üyenin yalnız en güçlü komşularını tutar ve çakışan üst/alt metin parçalarını bağımsız kanıt saymaz. Yeni nöronun sabit kimliği gerçek bir medoid üyeden byte düzeyinde kopyalanır; plastik ilişki alanı bağımsız desteklerden başlatılır. Aday ancak kullanıcı onayından sonra ayrı sentez arşivine girer.

Matematiksel tanım için [docs/NORON_SENTEZ_MODEL.md](docs/NORON_SENTEZ_MODEL.md) dosyasına bakın.

## Ayrı program: Nöron Aileleri

Tamamlanmış Metin Laboratuvarı analizlerini ve onaylı sentez nöronlarını salt-okunur kullanarak davranış profilleri, yumuşak üyelikler, öğrenilmiş aileler ve geçici olay koalisyonları üretir:

```text
http://localhost:5173/noron-aileleri/
```

Bu program yeni bir yoğun `N×F` veya `F×F` tablo oluşturmaz. Davranış profili `Q`, üyelik `M` ve aile geçişi `G` seyrek ve sınırlandırılmıştır. İşlev bölgeleri mimari sözleşmedir; hangi nöronların bu bölgelere yaklaştığı veriden hipotez olarak öğrenilir. `araç`, `yapan` ve `etkilenen` gibi roller kalıcı kelime etiketleri değil, yalnız o anki olay koalisyonunda yaşayan geçici bağlardır. Ailelerin faydası kaynak bazında ayrılmış susturma deneyiyle ölçülür ve bütün sonuçlar üçüncü bir yerel arşive yazılır.

Matematiksel tanım için [docs/NORON_AILELERI_MODEL.md](docs/NORON_AILELERI_MODEL.md) dosyasına bakın.
