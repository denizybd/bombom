# Nöron Aileleri — Matematiksel Model v1

Bu belge dördüncü programın sınırını tanımlar. Program, Metin Laboratuvarı'nın sabit kimlik (`I`), plastik ilişki (`R`) ve başlangıç geçiş önselini (`H⁰`) değiştirmeden okur; onaylanmış Nöron Sentezi kayıtlarını aynı kanıtların türevi olarak kullanır; davranış profilleri, yumuşak aile üyelikleri, seyrek aile geçişleri, geçici olay koalisyonları ve kaynak-ayrımlı susturma deneyleri üretir.

Programın amacı ham metinden kesin dilbilgisel veya anlamsal gerçek ilan etmek değildir. `içerik`, `olay`, `rol` ve `operatör` sonuçları açıklanabilir işlev hipotezleridir. `amaç`, `dil` ve `kontrol` bölgeleri yalnız statik metinle doğrulanamaz; çalışma zamanı deneyimi veya beklenen çıktı gelene kadar uykuda ve doğrulanmamış kalır.

## 1. Üç ayrı yapı

Sistem şu kavramları birbirine karıştırmaz:

1. **İşlev bölgesi:** Mimaride bulunan geniş çalışma sözleşmesi (`content`, `event`, `role`, `operator`, `goal`, `language`, `control`).
2. **Öğrenilmiş aile:** Benzer davranış profillerinin veri içinden oluşturduğu kalıcı ama gözden geçirilebilir alt küme.
3. **Geçici koalisyon:** Yalnız bir düşünce anında kurulan olay ve rol bağları.

Bu ayrım nedeniyle `bıçak` kalıcı olarak “araç ailesine” yazılmaz. Bir içerik nöronu olarak kalır; yalnız belirli bir `kesme` koalisyonunda `instrument-hypothesis` rolüne geçici olarak bağlanabilir.

## 2. Değişmez ham kayıtlar

Bir ham nöronun mevcut alanları:

```text
N_i = (I_i, R_i, H⁰_i)
```

- `I_i`: yüzey kimliği; byte düzeyinde sabit,
- `R_i`: Metin Laboratuvarı'nın ürettiği plastik bağlam,
- `H⁰_i`: benzerlikten türetilmiş başlangıç geçiş önseli.

`H⁰`, yaşanmış kas hafızası değildir. Nöron Aileleri onu `priorEvidence` olarak taşır. Daha sonra yaşanan koalisyon geçişleri `experienceEvidence`, engelleme kanıtları ise `inhibitionEvidence` alanında ayrı tutulur.

Aile üyeliği `M`, davranış profili `Q` ve aile geçişi `G`, `I` veya `R` içine yazılmaz. Ayrı, sürümlü ve seyrek kayıtlardır.

## 3. Davranış uzayı

Her nöron için yüzey kimliğinden ayrı, deterministik bir davranış profili çıkarılır:

```text
Q_i = profile(N_i, yerel bağlam, ağaç konumu, H⁰ özeti, işlev sondaları)
```

Profil kanalları şunları özetler:

- önceki ve sonraki çakışmayan komşular,
- ebeveyn, kardeş ve sol/sağ dal izleri,
- paragraf içindeki bağıl konum,
- derinlik ve parça ölçeği,
- ilişki gücü,
- `H⁰` entropisi ve yoğunlaşması,
- yerel devamlılık ve köprü etkisi,
- kısa kapsamlı operatör olma potansiyeli,
- zayıf Türkçe biçimbilimsel ipuçları.

`Q` yeni bir yoğun 64×64 matris değildir. Sabit boyutlu bir hash uzayında en güçlü `k` koordinatı tutan seyrek vektördür:

```text
Q_i = {(d_1,v_1), ..., (d_k,v_k)},  k <= maximumProfileFeatures
```

Davranış uzayı kimliği, yalnız temsil uzayını değil profil kodlayıcı sürümünü ve davranışı etkileyen bütün ayarları da kapsar:

```text
behaviorSpaceId = hash(
  profileEncoderVersion,
  representationSpaceId,
  fullFamilyConfig,
  identitySimilarityWeight,
  habitExponent
)
```

Farklı davranış uzayları sessizce birleştirilmez.

## 4. Bağımsız kanıt

Metin Laboratuvarı aynı paragrafı üst ve alt düğümlere böler. İç içe veya çakışan bu düğümler ayrı gerçeklik kanıtı sayılmaz. Kanıt birimi:

```text
(sourceDigest, paragraphIndex, nonOverlappingSpan)
```

Bir aile için destekler deterministik olarak seçilir; aynı kaynak ve paragraftaki çakışan aralıkların yalnız biri bağımsız oy verir. Aynı ham metnin farklı proje kimliğiyle yeniden içe aktarılması da yeni kaynak sayılmaz.

Onaylı sentez nöronu yeni bağımsız kanıt değildir. Profili, adayın `independentSupports` kayıtlarının birleşiminden oluşur; destek ve kaynak sayaçlarına aday diye ilave bir oy eklenmez.

## 5. İşlev bölgesi yakınlığı

Statik korpusun destekleyebildiği dört bölge için güven değerleri üretilir:

```text
A_i = {
  content:  a_ic,
  event:    a_ie,
  role:     a_ir,
  operator: a_io
}
```

Bu değerler sözlük etiketi değil; yapısal ve davranışsal ipuçlarının birleşimidir. Örneğin olay merkezi olma, rol bağlama kapasitesi, kısa kapsamlı değiştirici olma ve köprü davranışı ayrı kanıtlar üretir. Sonuç arayüzde daima “hipotez” olarak gösterilir.

`goal`, `language` ve `control` statik koşuda sıfır veya uykuda tutulur. Bu bölgelerin doğrulanması için çalışma zamanı hedefi, beklenen çıktı, düzeltme veya durma deneyimi gerekir.

## 6. Aile keşfi ve yumuşak üyelik

Davranış benzerliği seyrek kosinüs ile ölçülür:

```text
K_Q(i,j) = cos(Q_i,Q_j)
```

Bütün `N²` çiftleri oluşturulmaz. Seyrek koordinatlar ters indekse alınır; yalnız ortak güçlü özellik taşıyan sınırlı adaylar karşılaştırılır. Nöron başına en güçlü komşular ve aile başına sınırlı üye tutulur.

Bir nöronun aile uyumu:

```text
s_ik =
  α * behaviorSimilarity(Q_i,C_k)
  + β * regionCompatibility(i,k)
  + γ * evidenceConfidence(i)
```

`C_k`, ailenin bağımsız çekirdek üyelerinden deterministik olarak yeniden hesaplanan seyrek prototiptir. Sıraya bağlı, gizli bir EMA kullanılmaz.

Bir nöron tek aileye kilitlenmez. En güçlü birkaç üyelik normalize edilir:

```text
M_ik >= 0
sum(k) M_ik = 1
count(k where M_ik > 0) <= maximumMembershipsPerNeuron
```

Birlikte görünmek aynı aile olma kanıtı değildir. Birlikte görünme aileler arası geçiş veya geçici rol bağı üretir; aile içi üyelik davranışsal ikame edilebilirliğe dayanır.

## 7. Aile oluşumu ile ateşlenmesi farklıdır

Bir ailenin doğması için çoklu bağımsız destek, kaynak çeşitliliği, iç uyum ve dış ayrışma gerekir. Buna karşılık mevcut bir aile, çalışma zamanında tek ve çok güçlü bir operatör nöronuyla dahi ateşlenebilir.

Bu nedenle iki ayrı eşik vardır:

- `minimumFormationSimilarity`: kalıcı aile oluşumu,
- `runtimeActivationThreshold`: o anki çalışma alanında ateşlenme.

Aile aktivasyonu üye sayısı yanlılığını azaltacak biçimde normalize edilir:

```text
A_k =
  sum(i) a_i M_ik
  / sqrt(sum(i) M_ik² + ε)
  * coherence(activeMembers_k)
```

Çok üyeli bir aile yalnız büyük olduğu için küçük ama uyumlu aileyi bastıramaz.

## 8. Yaşam döngüsü

Öğrenilmiş aileler şu durumları taşır:

```text
provisional → strengthening → stable
                         ↘ split-proposed
                         ↘ merge-proposed
                         ↘ retired
```

İlk korpus koşusunda bir aile doğrudan `stable` olamaz. Terfi için sonraki korpus revizyonunda prototip benzerliği, üyelik Jaccard kararlılığı, dış ayrışma ve kaynak-ayrımlı fayda gerekir. Bölme veya birleştirme, eski kimliği sessizce değiştirmez; yeni aile ve selef kimliğiyle öneri üretir.

## 9. Seyrek aile geçişleri

Nöron düzeyi önsel geçişleri aile düzeyinde toplanır:

```text
B_ab = sourceNormalize(
  sum(i,j) M_ia * H⁰_ij * M_jb
)
```

Kalıcı kanıt:

```text
W_ab = priorEvidence_ab
     + experienceEvidence_ab
     - inhibitionEvidence_ab
```

Çalışma zamanı olasılığı:

```text
G_ab = max(0,W_ab) / sum(z) max(0,W_az)
```

Her aile yalnız en güçlü birkaç çıkışı saklar. `G`, kalıcı gerçek diye ayrıca yazılmaz; kanıtlardan yeniden türetilebilen normalize edilmiş görünümdür.

## 10. Kaynak-ayrımlı susturma deneyi

Bir ailenin yalnız kümelenmiş olması faydalı olduğunu göstermez. İlk vekil görev, gizli komşu/aile tahminidir. Kaynaklar `sourceDigest` düzeyinde ayrılır; aynı kaynak hem eğitim hem sınama tarafında bulunamaz.

Tam model kaybı ile aile susturulmuş model kaybı karşılaştırılır:

```text
DeltaU_k = Loss(ablated k) - Loss(full)
```

Pozitif `DeltaU`, ailenin gizli veride tahmine katkı sağladığını gösterir. Ailenin üyelikleri ve ilgili `G` kenarları susturulur, kalan dağılım yeniden normalize edilir ve iki koşul aynı aday evreninde değerlendirilir.

Bu deney nedensel anlam veya doğruluk kanıtı değildir. Çıktı `source-holdout-diagnostic` adıyla, kaynak fold sayısı, örnek sayısı ve güven aralığıyla saklanır. Yetersiz kaynakta açıkça `insufficient-evidence` döner.

## 11. Geçici olay koalisyonu

Koalisyon, aktif nöron ve ailelerin bounded bir çalışma alanında birleşmesidir. Olay ailesine güçlü üyeliği olan aktif bir nöron olay tohumu olabilir. Çevresindeki içerik, rol ve operatör hipotezleri sınırlı bir beam aramasıyla bağlanır.

Koalisyon puanı açıklanabilir parçalardan oluşur:

```text
score = activation
      + coherence
      + persistence
      + goalRelevance
      - conflict
      - repetition
      - cooldown
```

Rol bağları `temporary: true` taşır. Kalıcı aile üyeliklerine geri yazılmaz. Olay tohumu yoksa sistem sahte bir çerçeve üretmek yerine açık `unmatchedReason` döndürür.

## 12. İki seviyeli yönlendirme

Bir sonraki nöron seçimi, tek bir katı aile etiketine göre yapılmaz. Yumuşak üyeliklerin tamamı katkı verir:

```text
score(j) =
  λ * H⁰_ij
  + μ * sum(a,b) M_ia G_ab M_jb
  + ν * signalCompatibility(S,j)
  - ρ * repetition(j)
```

Skorlar sıcaklıklı softmax ile olasılığa çevrilir. Aday yoksa veya bütün kanıt sıfırsa `NaN` üretmek yerine açık bir `unmatchedProbability` döner.

Bu, sırasız nöron ağını iki aşamada daraltır:

```text
önce aday aileler → sonra o ailelerin aday nöronları
```

## 13. Bellek ve işlem sınırları

Yeni katmanda yoğun `N×F` üyelik veya `F×F` geçiş matrisi bulunmaz. Varsayılan güvenlik sınırları:

- davranış profili başına sınırlı özellik,
- nöron başına sınırlı aile üyeliği,
- aile başına sınırlı çıkış,
- sınırlı aktif nöron ve aile,
- sınırlı koalisyon ve beam genişliği.

Ana maliyet yaklaşık olarak:

```text
O(N * profileTopK + retainedBehaviorEdges + retainedMemberships + retainedG)
```

Hesap ayrı Worker içinde yürür; ilerleme aşamalıdır ve kullanıcı iptalini düzenli olarak denetler. Dense `K` Worker'a gönderilmez.

## 14. Depolama sınırı

Program şu kaynakları salt-okunur kullanır:

- `metin-laboratuvari` v4,
- `noron-sentez-laboratuvari` v1.

Kendi çıktısını ayrı depoya yazar:

```text
noron-aile-laboratuvari v1
```

Proje zarfı `storageRevision` taşır. Aile koşusu; tam girdi manifesti, korpus fingerprinti, davranış uzayı, profiller, aileler, üyelikler, geçiş kanıtları, susturma raporları ve bounded koalisyon özetini saklar. Dense `K/H`, ham metin ve canlı çalışma durumu kopyalanmaz.

## 15. Bu sürümün bilinçli sınırı

Bu sürüm, statik metindeki yapısal davranıştan aile hipotezi çıkarır ve mekanik olarak sınar. Aşağıdakileri henüz iddia etmez:

- gerçek dünya doğruluğunu bilmek,
- kelimelerin tek ve kesin anlamını çözmek,
- amaç veya değeri metinden kendiliğinden edinmek,
- doğal dilde kusursuz çıktı üretmek,
- aileleri otomatik meta-nöronlara sıkıştırmak.

Beklenen çıktı, insan tercihi ve yaşanan nöron geçişleri daha sonra aynı kanıt modeline `experienceEvidence`, hedef yakınlığı ve çalışma zamanı bölge doğrulaması olarak eklenebilir. Kalıcı meta-nöron doğumu ise ancak aileler birkaç korpus revizyonunda kararlı ve faydalı kaldıktan sonra yapılmalıdır.
