# Otonom Eğitim Laboratuvarı — Mimari ve Matematiksel Sözleşme

## 1. Amaç ve güven sınırı

Otonom Eğitim Laboratuvarı, yerel bir Ollama modelini öğrenen ağın kendisi olarak değil,
**deney hazırlayan öğretmen** olarak kullanır. Öğrenilen I/R/Q/M/H/G durumları bu projenin
motorları tarafından üretilir. Ollama cevabı hiçbir koşulda doğrudan nöron kimliğine,
geçiş tablosuna, aile üyeliğine veya etkin modele yazılmaz.

Sistem iki özneyi kesin olarak ayırır:

- **Öğretmen `T`:** Yeni örnek, karşı örnek, yeniden anlatım ve yapılandırılmış açıklama önerir.
- **Öğrenen `L`:** Örneği kendi temsil uzayında yaşar, sonraki parçayı tahmin eder ve yalnız
  ölçülen hata sinyaline göre plastik durumunu değiştirir.

Bu sınır, ortaya çıkan ağın Ollama'nın gizli vektörlerini kopyalamasını engeller. Mevcut
runtime Ollama embeddingi istemez. Gelecekte böyle bir yardımcı eklenirse yalnız tekrar
bulma veya karantina sıralama amacıyla kullanılabilir; sabit yüzey kimliği `I` yerine
geçemez.

## 2. Kalıcı varlıklar

Kaynak içeriği ve tekrar kullanılabilir çalışma eserleri içerik karmalarıyla izlenir. İş,
snapshot ve olay kayıtları ise kendi kimlikleriyle saklanır; bütün kayıtların içerik-adresli
olduğu varsayılmaz.

| Varlık | Saklanan temel alanlar | Gerçek davranış |
|---|---|---|
| Kaynak | `id`, `contentHash`, başlık, içerik, tür, köken, durum, zamanlar | Yeni kaynak `quarantine` durumunda doğar. Aynı içerik karması yeni kayıt üretmez; daha sonra gelen Metin Laboratuvarı kanıtı mevcut kaydı güçlendirebilir. |
| İş | tür, payload, durum, lease, deneme, aşama, ilerleme, sonuç/hata | İsteğe bağlı idempotency anahtarı aynı anahtarla ikinci işi engeller. |
| Artifact | tür, isteğe bağlı içerik karması, payload, iş kimliği | Aynı `tür + içerik karması` bulunduğunda eser yeniden kullanılır. Float32 matrisler byte-korumalı base64 biçiminde saklanır. |
| Snapshot | ebeveyn, metrikler, artifact/kaynak referansları, değerlendirme, durum | Etkinleşme ve sonradan `superseded` olma durumu güncellenir; geri alma hedefi kopyalanarak yeni snapshot oluşturulur. |
| Olay | artan sıra, iş, aşama, seviye, ileti | Günlük `maxEvents` ile sınırlıdır; sınır aşılınca önem ayrımı yapılmadan en eski olaylar silinir. |

Tercih edilen backend `server-data/autonomous-training.sqlite` dosyasındaki SQLite'tır.
Çalışılan Node.js ortamında `node:sqlite` bulunmazsa servis
`server-data/autonomous-training.json` atomik dosyasına geri düşer. Bu yol yalnız küçük/acil
deneyler içindir ve bütün dosyayı atomik yenilediği için 64 MiB mantıksal kayıt sınırıyla
kapatılmıştır; büyük korpusta SQLite destekli Node.js zorunludur. JSON commit yazımı hata
verirse bellek durumu ve işlem derinliği geri alınır, sonraki yazıların sessizce yalnız
bellekte kalmasına izin verilmez.

Uygulamadaki kaynak türleri `manual-text`, `local-file`, `local-upload`,
`allowlisted-url` ve `ollama-generated` değerleridir. Metin Laboratuvarı aktarımı ayrıca
`text-lab-archive` kökeni ve yeniden üretim kanıtı taşır. Öğrenenin serbest üretimini
otomatik olarak kaynak tablosuna geri yazan bir yol yoktur.

### Metin Laboratuvarı kimlik kanıtı

Arayüz analizli bir Metin Laboratuvarı kaydını aktarırken hazır I/R matrislerini servis
veritabanına kopyalamaz. Bunun yerine ham metinle birlikte şu yeniden üretim manifestini
gönderir:

- özgün Metin Laboratuvarı kaynak kimliği,
- `sourceDigest` ve `representationSpaceId`,
- analizin tam temsil yapılandırması,
- her nöron için `start`, `end` ve `identityDigest` üçlüsü.

HTTP sınırı bir manifestte 1–1600 kimlik kanıtına izin verir. Metin başı/sonu kırpılmadan,
yalnız satır sonu ve NFC kanonikleştirmesiyle saklanır. Gerçek runtime manifestli kaynağı
normal 2400 karakterlik parçalara bölmez; özgün yapılandırmayla bütünü yeniden analiz eder ve
kaynak karmasını, temsil uzayını ve sıralanmış bütün kimlik üçlülerini karşılaştırır. Bir
uyuşmazlık `TEXT_LAB_IDENTITY_MISMATCH` ile işi durdurur. Önbellekten bulunan analiz de aynı
kanıttan yeniden geçirilir.

Daha önce onaylanmış ham bir kayıt sonradan bu daha güçlü manifestle yükseltilirse eski insan
kararı yeni temsil uzayına taşınmaz: kaynak yeniden karantinaya alınır, seçimden çıkarılır ve
yeniden onay bekler. Bu kaynağı eski temsil revizyonuyla kullanan bekleyen adaylar da
geçersizleştirilir. Eski bir sürüme geri dönmek model artifact'ini geri getirebilir, fakat
değişmiş veya karantinadaki kaynağı yeniden “işlenmiş” sayamaz. Aynı manifestin uyumlu bir
kopyası ise mevcut kaydı ve kararını değiştirmez.

Aynı çevrimde birden fazla manifestli kaynak varsa temsil yapılandırmaları birebir aynı
olmalıdır. En az bir manifest varsa o yapılandırma çevrimdeki manifest taşımayan metinlere de
uygulanır; farklı Metin Laboratuvarı temsil uzayları tek çevrimde karıştırılmaz.

Manifest taşımayan metinler kayıpsız biçimde yaklaşık 2400 UTF-16 kod birimi hedefli kanonik
parçalara ayrılır; kesim surrogate çiftini veya birleşik işareti bölmemek için birkaç kod
birimi ileri taşınabilir. Parçalar birleştirildiğinde kaynak metni aynen geri vermelidir.
Sentez işi en fazla sekiz analizlik sınırlı shard'larda çalışır ve her shard'ın en az iki
fiziksel kaynak içermesini ister. Bağımsızlık kanıtı parça kimliğiyle değil fiziksel kaynak
kimliğiyle hesaplanır.

## 3. Dayanıklı iş akışı

Karantina ve insan kararı iş başlamadan önceki kaynak kabul kapısıdır. Onaylı bir tam eğitim
işinin gerçek yürütme sırası şöyledir:

```text
collect → textLab → synthesis → families → languageExperience → conceptMacros → evaluate
        → evaluation-gate → candidate-snapshot → activate | reject
```

Kuyruk her aşama sınırında iptal, duraklatma ve RAM bütçesini denetler. Uzun gerçek-motor
döngüleri ayrıca iptal/duraklatma sinyalini ve çalışma süresini denetler. Tamamlanan büyük
çıktılar içerik karmalı artifact olarak kalıcılaştırılır. Bir iş duraklatılıp yeniden
başlatıldığında boru `collect` aşamasından yeniden kurulur, fakat aynı girdiye ait kalıcı
TextLab, sentez ve aile eserleri önbellekten kullanılabilir. Sistem tam olarak son iç döngü
indisinden devam ettiğini iddia etmez.

Aile artifact önbelleği yalnız korpus karmasına bağlı değildir. Tam önceki `FamilyRun`,
kontrollü sentez kararlarının bütün kanıtı ve etkin snapshot/model soy bilgisi de anahtara
girer. Rollback yeni bir snapshot kimliği oluşturduğu için terk edilmiş daldaki gelişim
artifact'i geri dönülen dala sessizce taşınamaz.

Bir aday yalnız hâlâ `candidate` durumundaysa ve `parentId` değeri etkin snapshot ile aynıysa
etkinleştirilebilir. Etkin snapshot, aday durumu ve kaynakların `processedSnapshotId`
işaretleri tek transaction içinde değişir. Böylece eski bir dalda değerlendirilmiş kardeş
aday daha yeni modeli ezemez ve çökme anında yarım etkinleştirme oluşmaz. Geri alma da yeni,
değişmez bir geri-dönüş snapshot'ı üretir ve hedef sürümde bulunmayan kaynakları sonraki
eğitim için yeniden işlenebilir duruma getirir.

Çalışan işin lease'i servis başlangıcında veya boş kuyruğun dönemsel kurtarma taramasında
sona ermişse ve deneme hakkı varsa iş yeniden `queued` yapılır; hak kalmamışsa `failed` olur.
`pause` güvenli denetim noktasında işi
`paused` yapar. `cancel` adayın etkinleşmesini engeller ve etkin snapshot'a dokunmaz;
iptalden önce tamamlanmış değişmez artifactler önbellek amacıyla kalabilir.

### Kabul koşulları

1. JSON istek alanları veya dosya yüklemesinin multipart alanları geçerli olmalıdır.
2. Metin satır sonları tek biçime getirilmeli, NFC normalize edilmeli ve metin boş olmamalıdır.
3. Aynı içerik karması ikinci bir kaynak doğurmamalıdır; çağrı mevcut kaydı `duplicate`
   olarak döndürür.
4. Kaynak, istek, parti, günlük kullanım ve kalıcı disk sınırları aşılmamalıdır.
5. `collect` yalnız onaylanmış kaynakları kabul eder. Örtük/otomatik toplama yalnız seçili
   kayıtları kullanır; düşük seviyeli iş API'sinde açık `sourceIds` verilirse bu kimlikler
   seçili bayrağından bağımsız olarak onaylı kaynaklar arasından çözülür.
6. `collect` sonunda çözülmüş fiziksel kaynak kimlikleri iş kaydına, o anda kullanılan
   temsil revizyonları snapshot girdisine dondurulur. Bir retry aynı korpusu yeniden kurar;
   sonradan değişen TextLab kanıtı eski adayın üzerine yeni revizyon damgası vuramaz.
7. Dil veri kümesinde aynı fiziksel belgenin bütün parçaları aynı
   `independenceGroupId` ile tek train/validation/test bölümünde kalmalıdır.
8. Önceki dil durumunda kaydedilmiş bir kaynak-grubu bölümü sonraki revizyonda
   değiştirilemez.
9. Yalnız `train` dil örnekleri dil geçiş kanıtlarına ve doğum gözlemlerine yazılır;
   validation/test örnekleri ölçüm içindir.
10. Otomatik etkinleştirme doğrulaması için Ollama kaynaklarının seçili fiziksel kaynaklara
   oranı en fazla `%25` olmalıdır. Bu eşik mevcut runtime'da sabittir.
11. `conceptMacros` yalnız sınırlı sayıda research düşünce izi üretir. Train izleri motif
    gözlemine, validation/test izleri yalnız expanded-vs-shortcut shadow sınamasına gider.
12. Aynı 14B model+katalog soyundan gelen bütün kaynaklar tek bağımsız kanıt grubudur;
    kampanya büyüklüğü kavram terfi eşiğini yapay biçimde çoğaltamaz.
13. Bir makro yalnız kontrollü mod, genel model kabul kapısı ve kendi bağımsız kaynak,
    tetikleme, fidelity ve yanlış-aktivasyon kapıları birlikte geçerse `active` olur.
    Ham yol bütün yaşam evrelerinde korunur.

Önemli kapsam notu: sentez ve temel aile öğrenimi, dil bölümü ayrılmadan önce seçili korpusun
tamamını kullanır. Bu nedenle mevcut validation/test sonucu **dondurulmuş aile uzayı
üzerindeki dil başlığı holdout'udur**; bütün TextLab → sentez → aile zincirinin uçtan uca
görülmemiş-kaynak testi değildir.

### Aile denetimi okuma sınırı

Yerel servis `GET /api/families/inspection` ile bir `FamilyRun` kaydının ince ve salt-okunur
bir izdüşümünü verir. `snapshotId` gönderilirse o sürüm, gönderilmezse aile artifact'i taşıyan
en yeni snapshot seçilir; snapshot yoksa son bağımsız aile artifact'i kullanılabilir. İlk
çalışmadan önce hata uydurmak yerine `available:false` döner.

Yanıt; yaşam evresi ve bölge hipotezi, destek/kaynak sayısı, cohesion, separation,
stability, utility, kaynak-holdout susturma raporu, en güçlü yumuşak üyelikler ve sınırlı
çıkış geçişlerini içerir. Ham profil/prototip vektörleri, `Float32` alanları ve artifact
payload'u bu API sınırını geçmez. Bir sayfada en fazla 100 aile, aile başına 12 üye ve 12
geçiş dönebilir. Kaynak başına yalnız sınırlı kimlik/ad özeti verilir. Bu ekranın “bölge”
ifadesi semantik doğruluk etiketi değil, davranıştan öğrenilmiş işlev hipotezidir.

## 4. Dil deneyimi

Bir kaynak, yüzey parçalarının sıralı gözlemleri olarak yazılsın:

\[
X = (x_1, x_2, \ldots, x_T, \mathrm{EOS})
\]

`x_t` paragraf, cümle, yan cümle, token veya alt-parça seviyesinde bir yüzey parçası
olabilir; grafem seviyesi motor tarafından desteklenir fakat varsayılan deney listesinde
kapalıdır. Eğitim olayı, `x_{<t}` geçmişi verildiğinde gerçek devam `x_t`'yi tahmin etmektir.
Öğretmen zorlamasında her sonraki bağlam modelin örneklediği çıktıdan değil, gözlenen gerçek
geçmişten kurulur. Serbest üretim ayrı bir çıkarım yordamıdır ve greedy veya seed'li örnekleme
ile en fazla 512 adım çalışabilir.

Dil yolu aile ve aile içi nöron olmak üzere gerçekten iki seviyelidir. Bir öğrenilmiş
kenarın uyarıcı kanıtı `E^+`, baskılayıcı kanıtı `E^-` ise kenar puanı:

\[
h(E^+,E^-)=\operatorname{clip}\left(
\log\frac{E^++0.25}{E^-+0.25},-12,12\right)
\]

Bir aile adayı için uygulanan puan:

\[
s_F(f)=w_{LF}L_f+w_{FP}P_f+w_C A_f+w_H V_f
\]

Burada `L_f` öğrenilmiş aile geçiş kanıtı, `P_f` dondurulmuş aile geçiş önseli,
`A_f` geçici koalisyon aktivasyonu ve `V_f` son dört üyeden gelen yakın-geçmiş kanıtıdır.
Aile softmax'ına sıfır puanlı bir `unmatched` sonucu da katılır. Son üyenin aile üyelikleri
başlangıç kökenlerini belirler; geçmiş yoksa özel BOS ailesi kullanılır.

Bir `f` ailesindeki çıktı nöronu `y` için:

\[
s_N(y\mid f)=w_{LN}D_y+w_M M_{y,f}+w_H V_y-w_R R_y
\]

`D_y` son üyeden öğrenilmiş doğrudan geçiş, `M_{y,f}` yumuşak aile üyeliği, `V_y`
öğrenilmiş yakın geçmiş ile dondurulmuş nöron önselinin toplamı, `R_y` ise son sekiz
çıktıdaki tekrar cezasıdır. Her aile içindeki softmax olasılığı aile olasılığıyla çarpılır;
aynı nörona birden fazla aileden gelen kütle toplanır. EOS özel bir aile/üye yoludur.

Varsayılan olarak rota en fazla 16 aile ve 32 çıktı nöronu gösterir; kalıcı seyrek geçiş
listeleri köken başına sırasıyla 48 ve 96 kenarla sınırlandırılır.

Öğrenme kaybı gerçek devamın negatif log olasılığıdır:

\[
\mathcal{L}_t = -\log(P(x_t\mid C_t,x_{<t})+\varepsilon)
\]

Doğru kenarın uyarıcı kanıtı artırılır; yüksek olasılıklı yanlış aday için ayrı baskılayıcı
kanıt artırılır. Kanıt sayaçları negatif olamaz, üst tavana ve köken başına seyrek kenar
sınırına tabidir. Varsayılan `allowSourceReplay=false` olduğu için daha önce eğitilmiş
kanonik train kaynağı yeni gelişim kanıtı olarak tekrar yazılmaz.

### Sürpriz ve doğum önerisi

Hiçbir mevcut aday gerçek devamı yeterince açıklamıyorsa:

\[
\nu_t = 1 - \max_{y\in\mathcal{K}_t}\operatorname{sim}(x_t,y)
\]

değeri yükselir. Uygulama bunu, hedef parçanın mevcut bağlanma skorundan türetilen yenilik ve
gerçek hedef olasılığından türetilen sürpriz ile izler. Varsayılan durumda en az üç gözlem ve
iki bağımsız kaynak grubu gerektiğinde öneri `eligible` olur. Bir **doğum önerisi** doğrudan
nöron değildir; `requiresExternalApproval=true` taşır. Bu çevrim doğum önerisine sabit kimlik
tahsis etmez.

## 5. Dil ailesinin kendiliğinden oluşması

`language` bölgesi hazır bir kelime listesi değildir. Motor, öğrenilmiş nöron geçişlerinin
hedef aile dağılımlarını imza olarak çıkarır ve bu imzaları complete-link kosinüs eşiğiyle
kümeler. Her küme, validation ve test örneklerinde o kümeden çıkan nöron geçişleri
susturularak yeniden ölçülür.

Varsayılan uygunluk koşulları en az iki üye, en az iki bağımsız eğitim kaynak grubu, en az
bir holdout vakası ve en az `0.005` pozitif holdout kayıp farkıdır. Eşiği geçen kayıt
`eligible`, diğerleri `observing` durumundadır. Bu keşif dondurulmuş `FamilyRun` nesnesini
değiştirmez; sonucu runtime model artifactinde ayrı kanıt olarak saklar.

Motor ayrıca öğretmen-zorlamalı ölçümde EOS doğruluğu, alternatif yanlış hedefe karşı
dilbilgisel tercih ve son iki token yer değiştirdiğinde dağılım duyarlılığını ölçer. Serbest
üretimde EOS'a erişme ve tekrar döngüsü şu an snapshot etkinleştirme kapısının ayrı bir metriği
değildir.

## 6. Öğretmen paketleri

Ollama'dan serbest metin yerine JSON Schema'ya uyan paket istenir. Gerçek şema biçimi:

```json
{
  "lessons": [
    {
      "text": "Ali bıçakla ekmeği kesti.",
      "kind": "role_swap",
      "rationale": "Özne, araç ve nesne rollerini çeşitlendirir."
    },
    {
      "text": "Mert makasla kumaşı kesti.",
      "kind": "contrast",
      "rationale": "Aynı eylemin farklı katılımcılarla karşı örneğidir."
    }
  ]
}
```

`kind` yalnız `paraphrase`, `contrast`, `continuation`, `role_swap` veya `correction`
olabilir. Servis bir isteği en fazla 24 derslik Ollama çağrılarına böler; her dersin metni
3–4000, gerekçesi 1–500 karakterdir.

`runtime/smoke-ollama-teacher.mjs`, kurulu 14B öğretmenin bu sözleşmeye uyup uymadığını
yalıtılmış geçici depoda uçtan uca sınar. Üretimler önce karantinaya girer, test tarafından
açıkça onaylanır ve gerçek TextLab → sentez → aile → dil deneyimi hattına gönderilir. Smoke
testi profil ve aile oluşumunu bekler; korpusun `%100` sentetik olması nedeniyle snapshot'ın
etkinleşmemesini de ayrıca doğrular. Böylece “öğretmen sistemi besleyebiliyor” ile “öğretmen
tek başına güvenilir modeli etkinleştirebiliyor” iddiaları birbirine karıştırılmaz.

Tek seferlik, insan incelemeli öğretmen üretiminde sıcaklık ve seed sıfırdır. Zaman aşımı,
kullanıcı iptali ve yapılandırılmış sınırlı tekrar uygulanır. Şemaya uymayan yanıtta sonraki denemeye yalnız geçerli JSON talimatı eklenir;
son deneme de geçmezse hiçbir kaynak yazılmaz. Başarılı her ders ayrı
`ollama-generated` kaynak olarak karantinaya girer ve model, konu, amaç, çeşitlilik,
ders türü, gerekçe ve şema karmasını kökeninde taşır.

Onaylanan Ollama ve diğer sentetik kaynaklar yalnız `train` bölümüne atanır; aynı fiziksel
kaynak grubundaki hiçbir parça validation/test tarafına sızamaz. Eski bir snapshot sentetik
grubu holdout'a atamışsa bu atama güvenli biçimde `train` olarak düzeltilir ve uyarı üretilir.
Buna ek olarak otomatik etkinleştirme tüm seçili korpustaki sentetik payını `%25` ile
sınırlar. Güvenilir değerlendirme için en az üç bağımsız gerçek kaynak grubu yine gereklidir.

### 6.1. Seed'li 14B araştırma besleme kampanyası

Uzun besleme, tek seferlik HTTP üretiminin art arda çağrılması değildir. Kalıcı
`teacher-feed-campaign` işi; dondurulmuş model adı/digest'i, kampanya seed'ini, kabul
hedefini, üretim partisi boyutunu, eğitim kilometre taşını, müfredat sürümünü ve tam ilerleme
sayaçlarını taşır. Her cümle partisinden sonra kaynaklar ve bir sonraki cursor aynı store
transaction'ında yazılır. Servis kapanırsa kuyruk lease kurtarması son tamamlanan parti veya
eğitim kilometre taşından devam eder. Model digest'i değişmişse kampanya yeni modeli aynı soy
saymak yerine güvenli biçimde duraklar.

"Rastgele" burada ölçülemeyen gürültü anlamına gelmez. `turkish-foundations-v1` müfredatı;
yapan–eylem–etkilenen, durum ekleri, araç, olumsuzluk, soru, zaman, niteleme, bağlama ve
gönderim hücrelerini seed'li sırayla dolaşır. Aynı kullanıcı seed'i farklı kampanya
kimliklerinde de aynı müfredat ve seçim panosu sırasını kurar; kampanya kimliği yalnız kalıcı
kayıt ad alanıdır.

14B artık eğitim cümlesini serbestçe yazmaz. Sunucu, çekimleri tek tek gözden geçirilmiş olay
kartlarından 16 seçeneklik değişmez bir pano kurar; model JSON Schema içinde yalnız bir veya
iki farklı tam sayı kimliği seçer. Türkçe yüzey ve olay kartı sunucudaki dondurulmuş katalog
tarafından birlikte üretilir. Böylece modelin rol, ek, soru cevabı veya cümlede görünmeyen kişi
kimliği uydurması veri etiketine dönüşemez. Model seçimi yine seed'li sınırlı örnekleme kullanır
(`temperature=0.65`, `top_p=0.9`); ham örnekleme seçenekleri tarayıcıdan kabul edilmez.

Her çıktı şu kapılardan geçer:

- yalnız panodaki benzersiz tam sayı kimliklerini kabul eden JSON Schema,
- NFC/boşluk kanonikleştirmesi, tek cümle, uzunluk, noktalama ve kelime sınırı,
- bağlantı, işaretleme, öğretmen üst-dili ve düşük sözcük çeşitliliği reddi,
- yüzey exact hash'i, semantik kimlik digest'i ve karakter n-gram Jaccard ile tekrar reddi,
- açık rol/modifier yüzeyleri, tekil–çoğul gönderim, soru belirsizliği ve iyelik sahibinin
  semantik kartla aynı gerçeği söylemesi,
- kampanya içi ve sınırlı mevcut-korpus tekrar hafızası.

Kullanıcının kampanyayı başlatması yalnız bu deterministik kapıdan geçen sentetik kaynaklar
için sınırlı otomatik kabul yetkisidir. Kayıtlar `ollama-generated`,
`evaluationEligible:false`, `managedByTeacherCampaign:true` ve `proposalOnly:true` kalır.
Aynı model digest'inden gelen bütün cümleler ortak `independenceGroupId` taşır; yüz ayrı
cümle yüz bağımsız tanık sayılmaz.

Kabul sayısı tanımlı kilometre taşına ulaştığında TextLab → sentez → aile → dil deneyimi
borusu kampanyanın o ana kadarki tam kaynak sınırıyla yeniden çalışır. Snapshot kimliği
kampanya+kilometre taşı üzerinden deterministiktir; çökme sonrası aynı girdi ikinci bir aday
doğurmaz. Çevrim zorunlu olarak `proposal` kipindedir. Tamamı sentetik veri validation/test,
etkin model veya ThoughtRuntimeHead kredisi üretemez.

Model isteği, token kullanımı, ilerleme checkpoint'i ve kaynak partisi kendi ilgili store
transaction'larında atomik yazılır. Yeniden başlatma, aynı pahalı seed'i ikinci kez çağırmaz.
Kaynağın seçimi sonradan kaldırılırsa bağlı aday snapshot atomik olarak geçersiz olur; eski
kilometre taşı güncel seçili kaynak digest'iyle yeniden planlanır. Ollama yanıtındaki model adı
ve kampanya başındaki model digest'i değişirse hiçbir cümle kabul edilmeden kampanya güvenli
biçimde durur.

## 7. Değerlendirme ve etkinleştirme

Aday runtime artifactinde en az şu dil ölçümleri hesaplanır:

- top-1 ve top-k sonraki parça doğruluğu,
- mean reciprocal rank,
- ortalama negatif log olasılığı ve perplexity-benzeri `exp(NLL)`,
- alternatif yanlış hedefe karşı dilbilgisel tercih ve fark,
- yakın token sırası değişimine dağılım duyarlılığı,
- öğretmen-zorlamalı EOS başarısı,
- kaynak-holdout aile susturma faydası,
- doğum önerisi ve uygun doğum önerisi sayısı.

Snapshot'ın özet fayda puanı validation ölçümlerinden şu ağırlıklarla üretilir:

\[
U=0.30\,Top1+0.18\,Top5+0.18\,MRR+0.16\,Grammar+0.08\,EOS+0.10\,FamilyUtility
\]

Runtime doğrulamasının doğru olması için en az üç bağımsız kaynak grubu, boş olmayan ve
bağlanmış hedef içeren validation/test bölümleri, en az bir öğrenilmiş temel aile ve en fazla
`%25` sentetik kaynak oranı gerekir.

Geçerli varsayılan kalite kapısı `U ≥ 0.55`, dilbilgisel tercih `≥ 0.50`, etkin puana göre
gerileme `≤ 0.03` ve train top-1 ile test top-1 arasındaki ezberleme farkı `≤ 0.35`
koşullarını uygular. Etkin baseline varsa aday puanı baseline puanının `0.015` altına
düşemez. Bu eşikler servis yapılandırması üzerinden değiştirilebilir.

Normal eğitim borusunda kalite kapısını veya runtime doğrulamasını geçemeyen snapshot
`rejected` olur; gerekçesi ve tamamlanmış artifact referansları saklanır, etkin model
değişmez. Geçen aday öneri modunda insan etkinleştirmesi bekler, kontrollü modda otomatik
etkinleşir. Geri alma, hedef etkin/superseded snapshot'ın artifactlerini referanslayan yeni
bir snapshot üretir ve `ROLLBACK:<snapshot-id>` onay metni ister.

## 8. Otomasyon seviyeleri

Normal kaynak karantinası iki seviyede de insan kararını gerektirir. Tek istisna, kullanıcının
açıkça başlattığı ve yukarıdaki değişmez araştırma/kalite sözleşmesine bağlı sentetik besleme
kampanyasıdır; bu istisna etkinleştirme yetkisi vermez.

- **Öneri:** Sentez adayları öneri olarak artifactte kalır; doğrulanmış ve kalite kapısını
  geçen snapshot da insan etkinleştirmesi bekler.
- **Kontrollü:** Sentez motorunun kendi eşiklerinden geçen adaylar o çevrimin aile girdisine
  otomatik katılır; bütün runtime doğrulaması ve snapshot kalite kapısı da geçerse snapshot
  otomatik etkinleşir.

`autoRun`, otomasyon seviyesinden ayrı bir anahtardır. Açıldığında, önceki tamamlanmış bir
çevrimde aynı temsil sürümüyle denenmemiş yeni/değişmiş onaylı kaynak sayısı parti eşiğine
ulaşırsa, başka aktif/bekleyen iş yokken bütün seçili onaylı korpusla tam çevrim kuyruğa
alınır. Tamamlanan iş her kaynak için içerik + Metin Laboratuvarı temsil sürümünü kaydeder;
reddedilen bir snapshot yüzünden eski korpus her tek yeni onayda yeniden parti sayılmaz.
Korpusun tamamını yeniden kullanmak, yeni veri geldiğinde eski seçili kaynakları unutmayı
önler.

Sınırsız ve denetimsiz kip yoktur. Eşzamanlılık yapılandırmada ve doğrulamada tam olarak
birdir. Kaynak/istek/parti boyutu, günlük iş ve öğretmen kullanımı, RAM, süre, kalıcı disk ve
olay sayısı sınırları uygulanır. `cpuPercent` ayarı arayüzde ve iş bağlamında taşınır, ancak
bu sürümde işletim sistemi düzeyinde CPU throttling uygulamaz. Ollama isteğinde `keep_alive`
alanı gönderilmez.

## 9. Yerel servis ve güvenlik

Servis yalnız `127.0.0.1` üzerinde dinler. Kalıcı kaynak gerçekliği tarayıcı IndexedDB yerine
tercihen SQLite'tadır; arayüz kontrol ve gözlem istemcisidir. Yerel servis:

- yalnız `http://localhost`, `http://127.0.0.1` veya `http://[::1]` kökenlerine CORS verir,
- veri uçlarında boyutu sınırlı JSON, dosya aktarımında boyutu sınırlı multipart kabul eder;
  canlı olaylar salt-okunur SSE ile akar,
- yol ile dosya alma uç noktasını açık kök izin listesiyle sınırlar ve gerçek yol üzerinden
  sembolik bağ/path traversal kontrolü yapar,
- URL kaynaklarında protokol, host, yönlendirme, özel IP ve IPv4'ü IPv6 içine gömen
  mapped/translated/NAT64 adres kontrolleri uygular,
- Ollama'ya yalnız loopback adresinden erişir,
- iş lease'lerini ve snapshot etkinleştirmesini seçili kalıcı backend'in transaction'ı
  içinde değiştirir,
- kaynak, iş, olay, snapshot, artifact, kullanım ve yapılandırma kayıtlarının tamamında
  mantıksal kalıcı-disk bütçesini yazmadan önce uygular; olay sınırında en eski kayıtları
  transaction içinde budar; bayt toplamını her yazıda bütün artifact'leri okumak yerine
  işlemsel bir sayaçla günceller ve servis açılışında gerçek kayıt boyutlarıyla uzlaştırır,
- tarayıcı istemcisinin snapshot veya değerlendirme sonucu oluşturmasına izin vermez;
  snapshot yalnız gerçek eğitim borusu içinde doğar,
- iş oluştururken isteğe bağlı idempotency anahtarını uygular.

Besleme kontrol yüzeyi `POST/GET /api/teacher/feed/campaigns` ve
`POST /api/teacher/feed/campaigns/:id/pause|resume|cancel` uçlarından oluşur. Aynı anda yalnız
bir etkin veya duraklatılmış kampanya bulunabilir. Genel iş oluşturma ucu doğrulanmamış bir
`teacher-feed-campaign` payload'ı kabul etmez.

Öğretmen yanıt şeması araç adı veya araç argümanı içermez; ek alanlar şema doğrulamasında
reddedilir. Serviste kullanıcı hesabı, oturum veya API anahtarı yoktur. Loopback ve CORS
sınırı bir kimlik doğrulama mekanizması değildir; aynı makinedeki başka süreçler HTTP API'ye
ulaşabileceği için servis güvenilen yerel kullanıcı/host sınırında çalıştırılmalıdır. Geri
alma onay metni yanlış tıklamaya karşı korumadır, gizli bir yetkilendirme anahtarı değildir.

## 10. Bilinen sınırlar

Bu sistemin bir cümleyi tahmin etmesi bilinç veya insan anlamında düşünce kanıtı değildir.
"Düşünce benzeri" değerlendirme; girdiden uzun yaşayan iç durum, aileler arası rekabet,
bağlama duyarlı rol değişimi, görülmemiş fiziksel kaynakta genelleme ve karşı-olgusal fayda gibi
ölçülebilir davranışlarla sınırlıdır. Büyük veri, bu mekanizmaların yerine geçmez; yalnız
doğru mekanizmanın daha fazla bağımsız deney yaşamasını sağlar.

Mevcut sürümün başlıca teknik sınırları şunlardır:

- Holdout, dondurulmuş aile uzayındaki dil başlığını ölçer; sentez ve temel aile öğrenimi
  seçili korpusun tamamını daha önce görmüştür.
- Onaylı Ollama ve diğer sentetik örnekler dil validation/test bölümlerinden zorunlu olarak
  dışlanır; güvenlik kapısı ayrıca seçili korpustaki toplam sentetik oranını sınırlar.
- Serbest üretim mevcut olsa da serbest-koşu cümle kalitesi, EOS erişimi ve tekrar döngüsü
  snapshot kabul puanına henüz bağlanmamıştır.
- Uygun doğum önerileri artifactte kalır; sabit kimlik tahsis eden insan-onaylı doğum
  iş akışı bu çevrimin parçası değildir.
- CPU yüzdesi sert throttle değildir ve servis kimlik doğrulaması içermez.
