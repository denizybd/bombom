# Bombom

An experimental, local research environment for neuron graphs, text representations,
learned families, and signal propagation. Bombom combines visual laboratories with
a TypeScript simulation engine and a local training service. The interface and
detailed research notes are currently in Turkish.

This is a research prototype. Its visualizations and experimental metrics do not
establish general language understanding, consciousness, or competitive model quality.
Ollama can optionally supply training examples; it is not required for the basic sandbox.
No trained model or local training archive is included in this repository.

## Quick start

Install Node.js 24+ and pnpm, then run:

```bash
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173` for the signal sandbox, or
`http://127.0.0.1:5173/otonom-egitim/` for the training laboratory.
For a first experiment, draw a signal in the left editor, add a neuron with
**Yeni nöron**, then use **Tek adım** to inspect individual transitions.
The application stores browser state locally; the training service stores its
archive in the ignored `server-data/` directory.

Run `pnpm test`, `pnpm test:service`, `pnpm build`, and
`pnpm build:training-service` to check the project.

## Sinaps Sandbox

Katman sırası olmayan yönlü bir nöron grafında, işaretli `32 × 32` sinyal matrisinin adım adım nasıl dönüştüğünü gösteren görsel bir sandbox.

Uygulama iki farklı ilişkiyi özellikle ayrı tutar:

- **Gerçek benzerlik K:** Nöron prototiplerinin kosinüs benzerliği.
- **Kas hafızası H:** Kullanım kanıtı `W` üzerinden satır bazında normalize edilen geçiş alışkanlığı.

Sinyal ham haliyle ilk nörona girer. Her nöronda uyuma göre dönüşür, adaylar kas hafızası sırasıyla değerlendirilir ve gerçekleşen geçişin `W` kanıtı güçlenir. Nöron oluşma baskısı yalnızca gösterilir; otomatik nöron üretmez.

## Çalıştırma

macOS'ta en kolay yol proje kökündeki `BOMBOMU_BASLAT.command` dosyasına çift
tıklamaktır. Bu başlatıcı hem görsel uygulamaları hem de yalnız yerelde çalışan
otonom eğitim servisini açar; Codex'in paketli Node.js çalışma ortamını da
bulabilir ve Otonom Eğitim ekranını tarayıcıda gösterir. İkinci kez çalıştırılırsa
mevcut servisleri çoğaltmak yerine açık oturumu kullanır.

İlk kullanımdan önce terminalde bağımlılıkları kurun. Gereksinimler Node.js 24+ ve pnpm'dir:

```bash
pnpm install
pnpm dev
```

Vite'ın gösterdiği yerel adresi tarayıcıda açın. Bu proje varsayılan olarak `http://127.0.0.1:5173` adresini kullanır.

## Doğrulama

```bash
pnpm test
pnpm test:service
pnpm build
pnpm build:training-service
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

Bu sınırlama yalnız ilk görsel Sinaps sandbox'ı için geçerlidir. Metin, aile ve
otonom eğitim deneyleri aşağıdaki ayrı programlarda yürütülür; bu programlar
sabit kimlikleri geriye dönük değiştirmez.

## Ayrı program: Metin Laboratuvarı

Ham metni paragraf köklerinden başlayarak kayıpsız bir oluş ağacına dönüştüren ikinci uygulama aynı projede, fakat Sinaps'tan bağımsız çalışır:

```text
http://127.0.0.1:5173/metin-laboratuvari/
```

Bu program yüzey kimliği ile bağlamsal ilişki alanını ayrı maskelerde tutar; ebeveyn ve kardeş mikro-izlerini üretir, küçük deneylerde bütün nöron çiftlerinin `K` benzerliğini hesaplar ve başlangıç `H` olasılıklarını hazırlar. Kendi Worker işlem hattı, IndexedDB kaydı, proje dosyası ve testleri vardır. Sinaps'ın ağına veya yerel kayıt anahtarına dokunmaz.

Matematiksel tanım için [docs/METIN_LABORATUVARI_MODEL.md](docs/METIN_LABORATUVARI_MODEL.md) dosyasına bakın.

## Ayrı program: Nöron Sentezi

Farklı ham metinlerde tekrarlanan yüzey ve bağlam örüntülerinden, insan onayına sunulan yeni nöron adayları çıkarır:

```text
http://127.0.0.1:5173/noron-sentezi/
```

Sentez programı yoğun yeni bir çift tablosu saklamaz; çapraz kaynak çiftlerini Worker içinde akıtır, her üyenin yalnız en güçlü komşularını tutar ve çakışan üst/alt metin parçalarını bağımsız kanıt saymaz. Yeni nöronun sabit kimliği gerçek bir medoid üyeden byte düzeyinde kopyalanır; plastik ilişki alanı bağımsız desteklerden başlatılır. Aday ancak kullanıcı onayından sonra ayrı sentez arşivine girer.

Matematiksel tanım için [docs/NORON_SENTEZ_MODEL.md](docs/NORON_SENTEZ_MODEL.md) dosyasına bakın.

## Ayrı program: Nöron Aileleri

Tamamlanmış Metin Laboratuvarı analizlerini ve onaylı sentez nöronlarını salt-okunur kullanarak davranış profilleri, yumuşak üyelikler, öğrenilmiş aileler ve geçici olay koalisyonları üretir:

```text
http://127.0.0.1:5173/noron-aileleri/
```

Bu program yeni bir yoğun `N×F` veya `F×F` tablo oluşturmaz. Davranış profili `Q`, üyelik `M` ve aile geçişi `G` seyrek ve sınırlandırılmıştır. İşlev bölgeleri mimari sözleşmedir; hangi nöronların bu bölgelere yaklaştığı veriden hipotez olarak öğrenilir. `araç`, `yapan` ve `etkilenen` gibi roller kalıcı kelime etiketleri değil, yalnız o anki olay koalisyonunda yaşayan geçici bağlardır. Ailelerin faydası kaynak bazında ayrılmış susturma deneyiyle ölçülür ve bütün sonuçlar üçüncü bir yerel arşive yazılır.

Matematiksel tanım için [docs/NORON_AILELERI_MODEL.md](docs/NORON_AILELERI_MODEL.md) dosyasına bakın.

## Ayrı program: Otonom Eğitim Laboratuvarı

Metin Laboratuvarı → Nöron Sentezi → Nöron Aileleri → sonraki-parça dil
deneyimi → kavram makroları → kaynak-holdout değerlendirmesi zincirini kalıcı ve geri alınabilir
bir yerel servis üzerinde çalıştırır:

```text
http://127.0.0.1:5173/otonom-egitim/
```

Yerel servis yalnız `127.0.0.1:8787` üzerinde dinler ve kayıtlarını varsayılan
olarak `server-data/` altında SQLite'a yazar; çalışılan Node.js ortamında
`node:sqlite` yoksa küçük/acil deneyler için 64 MiB ile sınırlı atomik bir JSON dosyasına
geri düşer. Ollama bu mimaride
modelin kendisi değil, isteğe bağlı veri öğretmenidir: yapılandırılmış örnek
üretir, her örnek kaynak geçmişiyle karantinaya girer ve doğrudan nöron ya da
aktif model yazamaz. Gerçek korpus eğitimi Ollama kapalıyken de çalışır.

Program; metin/dosya aktarımı, Metin Laboratuvarı kimlik kanıtı, Ollama model
seçimi, karantina, otomatik parti kuyruğu, duraklatma/iptal, CPU-RAM-zaman-disk
bütçe ayarları, canlı olaylar, fiziksel kaynak grubu düzeyinde
train/validation/test ayrımı, sürümleme ve onay metinli geri alma sunar. RAM,
çalışma süresi ve kalıcı disk sınırları uygulanır; CPU yüzdesi bu sürümde
gözlenen/ayarlanan bir politika değeridir, işletim sistemi düzeyinde sert bir
kota değildir. Öneri modu geçen adayı insan onayında tutar; kontrollü mod bile
normal eğitim akışında yalnız gerçek motorların doğruladığı ve kalite kapısını
geçen adayı etkinleştirebilir. Dil doğrulama/test örnekleri geçiş öğrenmesine
yazılmaz; mevcut holdout metriğinin dondurulmuş aile uzayı üzerindeki dil
başlığını ölçtüğünü ayrıntılı belgede özellikle açıklıyoruz.

**Aile denetimi** sekmesi en güncel veya seçilen model sürümündeki öğrenilmiş
aileleri açar. Yaşam evresi, bölge hipotezi, bağımsız kaynak/destek sayısı,
iç uyum, ayrışma, kararlılık, susturma faydası, yumuşak üyeler ve olasılıklı
sonraki-aile geçişleri görülebilir. Arayüz ham I/R/Q matrislerini veya aile
prototiplerini HTTP üzerinden taşımaz; sonuçlar sayfalı ve sert limitlidir.

Dil çekirdeği elle “dil ailesi” etiketi vermez. Nöronların öğrenilmiş
sonraki-parça geçiş imzalarını kümeler ve her kümeyi görülmemiş kaynaklarda
ablation ile sınar. Surprise + novelty yüksek olduğunda yeni nöron yalnız bir
doğum önerisi olur; sabit kimlik tahsis etmek ayrıca onay gerektirir.

Servisi tek başına çalıştırmak için önce bundle üretin:

```bash
pnpm build:training-service
pnpm training:service
```

Kurulu `qwen2.5-coder:14b` modelinin öğretmen → karantina → gerçek öğrenme hattını
kalıcı arşive dokunmadan sınamak için önce servis paketini üretip şu yalıtılmış
smoke deneyini çalıştırabilirsiniz:

```bash
pnpm build:training-service
pnpm smoke:ollama-14b
```

Deney geçici bir veri dizini kullanır. Tamamı Ollama üretimi olan deneme
otomatik olarak etkin modele dönüşmemelidir; buna rağmen TextLab profilleri ve
aile artifact'i oluşup oluşmadığı raporlanır.

Ayrıntılı mimari ve denklemler için
[docs/OTONOM_EGITIM_MODEL.md](docs/OTONOM_EGITIM_MODEL.md) dosyasına bakın.
Uzun 14B koşularındaki RAM/SSD eşikleri, artifact sıkıştırması, ayrı worker
süreci ve nöron hot/warm/cold yerleşimi için ayrıca
[performans kaynak politikası](docs/PERFORMANS_KAYNAK_POLITIKASI.md) ile
[nöron katmanlama sözleşmesine](docs/RAM_SSD_NORON_KATMANLAMA.md) bakın. 30 Ağustos
2026 tarihli 14B sürekli koşu, 32K→4K bellek profili ve canlı kabul kanıtı
[Derin Operasyon tutanağında](docs/DERIN_OPERASYON_2026-08-30.md) bulunur.

## Ayrı program: Düşünce Akışı

Doğrulanmış bir `64 × 64` aile snapshot'ı üzerinde sinyalin nöron ve aile
katmanlarında nasıl dallandığını, hangi kas-hafızası ve bağlam katkılarının
seçimi etkilediğini ve yeni nöron baskısının nasıl oluştuğunu izleyen deney
ekranıdır:

```text
http://127.0.0.1:5173/dusunce-akisi/
```

Bu ilk kilometre taşı doğal dil cevabı üretmez ve bilinç iddiasında bulunmaz.
Sabit `I` kimliği değişmeden kalırken plastik `R` alanı yalnız bölüm içinde hızlı
iz taşır; kalıcı `E⁺/E⁻` ve `R` deltaları ancak dışarıdan verilmiş yapısal sonuç
kanıtıyla öneri haline gelir. Etkin ve bağımsız değerlendirmeden geçmiş bir
snapshot yoksa bütün bölümler araştırma/iz kipinde çalışır ve modele yazmaz.

Matematik, seyrek sınırlar, replay/checkpoint sözleşmesi ve yapısal benchmark
kapıları için [docs/DUSUNCE_AKISI_MODEL.md](docs/DUSUNCE_AKISI_MODEL.md)
dosyasına bakın.

## Ayrı program: Kavram Fidanlığı

Düşünce izlerinde tekrar eden sıralı nöron yollarını ve aynı aile dizisini
farklı yüzeylerle gerçekleştiren eşdeğerlik adaylarını inceler:

```text
http://127.0.0.1:5173/kavram-fidanligi/
```

Bir tekrar doğrudan kavram sayılmaz. Aday; bağımsız kaynak, MDL sıkıştırma,
etki tutarlılığı, pozitif tetikleme başarısı, yanlış tetiklenme ve ayrılmış
veride expanded-yol sadakati kapılarından geçer. Ham nöronlar hiçbir zaman
silinmez; makro her zaman aynı sıraya geri açılabilir. 14B kaynakları aynı model
ve katalog soyundaysa tek bağımsız kanıt grubu sayılır ve kendi başına makroyu
etkinleştiremez.

Saf çekirdek, RAM sınırı, replay defteri, yaşam döngüsü ve iç içe DAG kuralları
[Kavram Makroları V1 belgesinde](docs/KAVRAM_MAKROLARI_V1.md) tanımlanmıştır.

## Yeni program: Plastik Bölge Atlası

`64 × 64` nöronun değişmez üst `8 × 64` kimlik alanını koruyup kalan
`56 × 64` ilişki alanını 56 fiziksel `8 × 8` mikro-sayfa üzerinden inceleyen
atlas ve güvenli devreye alma ekranıdır. Harita ve geçmiş kayıtlar salt okunur;
ayar değişiklikleri ayrı, açık onaylı denetimlerdir:

```text
http://127.0.0.1:5173/bolge-atlasi/
```

Bu yapı nöron ailesinin veya kavram makrosunun yerine geçmez. Aileler farklı
nöronları, makrolar tekrarlanan yolları gruplar; plastik atlas ise tek bir
nöronun ilişki koordinatlarının hangi gizil işlevlerde birlikte kullanıldığını
öğrenir. Bölge adı ve büyüklüğü elle verilmez. Fiziksel sayfalar sabit kalır;
veriyle oluşan yumuşak üyelikler doğum, büyüme, küçülme, bölünme, birleşme ve
emeklilik önerileriyle değişir.

Mevcut `16/48` kayıtlar yeniden yorumlanmaz. Yeni `8/56` profil açıkça
seçilmedikçe çalışma yolu `off` kalır; `shadow` yalnız gözlem toplar,
`proposal` geri alınabilir aday üretir, `active` ise ancak doğrulanmış ve tam
fingerprint ile bağlanmış snapshot üzerinde davranışı etkileyebilir. Matematik,
kalıcılık ve geçiş kapıları
[Uyarlanabilir Plastik Bölge Atlası belgesinde](docs/UYARLANABILIR_PLASTIK_BOLGE_ATLASI.md)
tanımlanmıştır.

Ekrandaki hazırlık akışı sıralıdır: önce ayrı 8/56 soyunda davranışsız `shadow`,
sonra birikimli kanıt oluşunca `proposal`, ardından dondurulmuş holdout üzerinde
Ollama kullanmayan taban–aday–sahte atlas sınaması. Geçen sınama yeni bir aday
snapshot üretir fakat kendiliğinden etkinleştirme yapmaz. Kullanıcı önce Otonom
Eğitim ekranında o adayı kabul eder, sonra release zinciri hâlâ tam ise Bölge
Atlası ekranından canlı bölgesel yönlendirmeyi ayrıca açar.

Etkin atlas aynı kalırken model snapshotı değişirse eski release yetkisi yeniden
kullanılmaz; sunucunun bağımsız değişmeyen-atlas yenileme sınaması yeni ve ayrı
bir aday üretir. Genel rollback de bu zinciri atlayamaz. Canlı seyrek yönlendirme
100K nöron testine sahipken release/yenileme değerlendiricisinin doğrulanmış dense
taban sınırı şimdilik 16.384 nörondur; daha büyük adaylar SSD-sayfalı evaluator
gelene kadar güvenli biçimde bekletilir.

## Ayrı program: Nöron Evreni 3B

Seçilen araştırma snapshot’ındaki nöronları, öğrenilmiş aileleri ve aralarındaki
geçişleri tek bir 3B haritada inceleyen salt-okunur gözlem programıdır. Seçilen
snapshot etkin değilse bu durum görünümde açıkça belirtilir:

```text
http://127.0.0.1:5173/noron-evreni/
```

Program yalnız vektör içermeyen yerel atlas görünümünü okur; eğitim başlatmaz,
modeli değiştirmez ve arşivlere yazmaz. Büyük ağlarda güçlü bağlantılar ile
yakın çevre ayrıntıları kademeli gösterilir. Kapsam ve güvenlik sınırı
[Nöron Evreni 3B notunda](docs/NORON_EVRENI_3B.md) özetlenmiştir.
