# Kavram Makroları V1 — Saf Çekirdek Sözleşmesi

## Amaç ve sınır

`src/concept-macros`, tekrar eden nöron yollarını geri açılabilir sanal bileşikler olarak inceleyen saf ve deterministik çekirdektir. Bu sürüm:

- server, eğitim pipeline'ı ve UI ile doğrudan bağlantı kurmaz;
- yeni bir `64×64` fiziksel kimlik üretmez;
- ham nöronları, yolları veya kanıtı silmez;
- sık tekrarı tek başına “anlam” ya da “kavram kanıtı” saymaz;
- aynı girdi, head ve yapılandırma için aynı fingerprint zincirini üretir.

Bu nedenle modül bir **kavramsal sıkıştırma adayı üreticisidir**. Semantik iddia ancak bağımsız kaynak, MDL kazancı, etki tutarlılığı ve expanded-vs-macro holdout doğrulaması birlikte geçilirse güçlenir.

## İki ayrı bileşik türü

### `sequence-macro`

Tam olarak aynı sıralı üyelerin tekrarını temsil eder. Exact imza; üye türü, üye kimliği ve donmuş identity digest'inin sırasından oluşur. Örneğin:

```text
[N-yapan, N-eylem, N-nesne] -> M-olay-kısayolu
```

Bu, yolun sık kullanılmasını sıkıştırır; üyelerin birbirine semantik olarak eşdeğer olduğunu iddia etmez.

### `equivalence-concept`

Aynı aile sırasını ve benzer giriş→çıkış etkisini üreten birden fazla exact yüzeyi toplar. Örneğin üç farklı somut `[yapan, eylem, nesne]` sırası aynı family signature altında bir aday oluşturabilir. Bu türün öneri olabilmesi için varsayılan olarak en az iki farklı exact varyant gerekir.

## Temel artifactler

- `concept-macro-observation-batch`: bounded motif miner çıktısı.
- `concept-macro-head`: yaşayan sanal makrolar, kanıt ve replay ledger'ları.
- `concept-macro-shadow-report`: expanded yol ile sanal makronun holdout karşılaştırması.

`MacroHeadV1.macros` içindeki her `CompositeNeuronV1` şu değişmezleri taşır:

```text
physicalIdentityAllocated = false
rawConstituentsRetained = true
orderedExpansion + expansionDigest
```

`expansionDigest`, aile atamasından bağımsızdır; sıralı exact üye/digest bilgisini doğrular. `familySignature` ayrı tutulur. Böylece FamilyRun değiştiğinde fiziksel kimliği değişmeyen sıralar korunabilir, aileye bağımlı hipotezler yeniden doğrulanabilir.

## Girdi sözleşmesi

Madenci `MacroTraceV1[]` alır. Bir iz:

```ts
{
  format: 'macro-trace-v1',
  version: 1,
  id: string,
  representationSpaceId: string,
  familyRunFingerprint: string | null,
  sourceId: string,
  sourceDigest?: string | null,
  independenceGroupId: string,
  episodeId: string,
  runKey: string,
  semanticInputDigest: string,
  split: 'train' | 'validation' | 'test',
  steps: Array<{
    member: {
      kind: 'raw-neuron' | 'composite-neuron',
      id: string,
      identityDigest: string | null,
      familyIds: string[],
    },
    signalBefore: number[],
    signalAfter: number[],
  }>,
}
```

`semanticInputDigest`, cycle/run kimliklerini içermeyen gerçek kaynak oluşumunu tanımlamalıdır. Aynı kitapta iki ayrı gerçek oluş sayılacaksa paragraf/offset/path bu digest'e katılmalıdır. Aynı oluş başka cycle, episode veya run adıyla replay edilirse digest aynı kalmalıdır.

## Replay ve bağımsızlık

İki ayrı idempotency katmanı vardır:

1. `runKey + episodeId`: aynı transaction replay'ini engeller.
2. `(sourceDigest || sourceId) + semanticInputDigest`: aynı semantik girdinin başka cycle, episode veya independence-group etiketiyle tekrar sayılmasını engeller.

Her observation ayrıca `signature + semantic trace + span` tabanlı `semanticOccurrenceKey` taşır. Kanıt kaydı source, source digest, independence group, episode, run, semantic input ve tick aralığını korur.

Head içindeki iki exact anahtar listesi varsayılan en fazla `4096` kayıt tutar. Canonical sıralamada taşan eski anahtarlar, sabit boyutlu iki ayrı Bloom özetine aktarılır (`262.144` bit, `7` hash). Bloom özeti false negative üretmez; doygunlukta unseen bir girdiyi replay sanan güvenli false-positive üretebilir. Böylece replay geçmişi RAM/JSON boyutunda sınırsız büyümez.

Bağımsızlık eşiği `independenceGroupId` üzerinden ölçülür. Her retained kaynak grubundaki **bütün** gözlemler önce kendi online ortalamasını günceller. Son prototip, grup ortalamalarını gözlem sayılarına bakmadan eşit ağırlıkla birleştirir. Böylece aynı kitabın çok sayıdaki oluşu tek başına başka kitapları bastıramaz. Grup sayısı aday başına bounded'dır; varsayılan canonical ilk `4` grup dense prototipe katılır, diğer gruplar bağımsızlık/MDL kanıtında kalır.

## Bounded 2–8 motif madenciliği

Madenci tüm olası motifleri kalıcı bellekte tutmaz. İzleri canonical sıraya koyup bounded Space-Saving heavy-hitter tablosunda işler.

- motif uzunluğu: `2..8`;
- varsayılan ağır-vurucu kapasitesi: `2048`;
- aday başına saklanan kanıt: `64`;
- aday başına exact varyant: `16`;
- aynı imzanın aynı izde üst üste binen pencereleri yalnız bir kez sayılır;
- exact ve family imzaları ayrı aday akışlarıdır;
- eviction sayısı diagnostics içinde görünür;
- Space-Saving tahmini ile gerçek, saklandığı sürede gözlenen destek ayrı alanlardır; öneri kapısı tahmini değil gerçek desteği kullanır.

Kanıt kapasitesi dolduğunda farklı independence group'ların en az birer örneğini koruyan kaynak-çeşitli bounded seçim uygulanır.

### RAM sınırı

Madenci, aday başına 64 adet tam observation vektörü saklamaz. Her retained kaynak grubu için yalnız:

- dört çevrimiçi `Float32` dense prototip özeti (`entry`, `exit`, `trajectory`, `effect`),
- bir yönsel cohesion özeti,
- bounded kanıt ve varyant metadata'sı

tutar. Grup aggregate'leri artifact içinde little-endian Float32 + canonical base64 biçiminde JSON-uyumlu saklanır. Üst seviye public `entry/exit/trajectory/effect` alanları geriye uyumluluk için `number[]` kalır.

Global dense prototip bütçesi `4.194.304` hücre, yani yaklaşık `16 MiB` Float32 payload'dur. Etkin kapasite:

\[
K_{effective}=\min\left(K_{config},\left\lfloor\frac{4{,}194{,}304}{5gd}\right\rfloor\right)
\]

olur; `g` retained kaynak grubu sınırıdır. `d=4096`, `g=4` için varsayılan `maximumHeavyHitters=2048` isteği gerçek worst-case hesabıyla `51` dense adaya daralır. Çıktının canonical JS dizilerine materialize edilmesi ayrıca geçici alan kullanır, fakat eski observation-retention çarpanı ortadan kalkmıştır.

## MDL sıkıştırma kapısı

Uzunluğu `L`, gözlem sayısı `n`, tanım ek maliyeti `o` olan bir motif için:

\[
C_{raw}=nL
\]

\[
C_{definition}=L+o
\]

\[
C_{macro}=C_{definition}+n
\]

\[
gain=C_{raw}-C_{macro}=n(L-1)-(L+o)
\]

Varsayılan öneri kapısı:

- en az `3` gerçek gözlem;
- en az `2` bağımsız grup;
- `compressionGain >= 1`;
- etki cohesion `>= 0.82`;
- `equivalence-concept` için en az `2` exact varyant.

Yalnız frekans yeterli değildir. Kazanç negatifse motif tekrar etse bile makro önerisine yükselmez.

## Giriş, çıkış, trajectory ve effect

Bir pencere için:

- `entry`: ilk adımın `signalBefore` değeri;
- `exit`: terminal-anchor sözleşmesi gereği son üyenin `signalBefore` değeri;
- `trajectory`: terminal üye hariç yürütülen adımların `signalAfter - signalBefore` farklarının ortalaması;
- `effect`: son çıkış ile ilk girişin farkı.

Prototipler independence-group dengeli ortalamadır. `effectCohesion`, gözlenen effect'lerin prototiple cosine uyumunun ortalamasıdır. Bu alanlar ileride bir makronun yalnız benzer başlangıçta ve benzer beklenen etkide etkinleşmesi için gerekli imzaları sağlar.

## Yaşam döngüsü

Geçişler tek yönlüdür:

```text
observing
  -> eligible-proposal
  -> virtual-provisional
  -> shadow-validated
  -> active
  -> deprecated
```

Her canlı aşamadan `deprecated` durumuna geçilebilir; geriye veya aşama atlayarak geçilemez.

- `observing`: kanıt birikiyor.
- `eligible-proposal`: bağımsızlık, MDL ve cohesion kapıları geçti.
- `virtual-provisional`: sanal yürütmeye açık, kalıcı route yerine geçmez.
- `shadow-validated`: expanded yol ile holdout karşılaştırmasını geçti.
- `active`: ayrı ve açık bir lifecycle kararıyla etkinleştirildi.
- `deprecated`: kayıt ve expansion korunur, seçim için kullanılmamalıdır.

İlk sürümde “active” olmak bile ham constituents'in silinmesine izin vermez.

`shadow-validated -> active` geçişinde ayrıca `structuralEquivalencePassed=true` gerekir. Bu boolean yalnız fingerprint'li `MacroStructuralEquivalenceAuditV1` kaydıyla değiştirilebilir ve audit exact composite execution fingerprint'ine bağlanır. Endpoint benzerliği tek başına yeterli değildir; hızlı R izi, decay ve next-route davranışını inceleyen ayrı yapısal doğrulayıcı başarısızsa makro active olamaz. Yürütülebilir tanım değişirse bu kapı ve eski audit sıfırlanır.

## Shadow doğrulaması

`evaluateMacroShadow`, validation/test vakalarında değişmeyen expanded yolu ve sanal makro sonucunu karşılaştırır:

- `fidelity`: çıktı vektörlerinin normalize edilmiş yakınlığı;
- `falseActivationRate`: etkinleşmemesi gereken negatif vakalardaki yanlış etkinleşme;
- `activationRecall`: etkinleşmesi gereken pozitif vakaların tamamlanma oranı;
- `holdoutUtility`: beklenen çıktı varsa expanded hata ile macro hata arasındaki fark;
- `observedCompressionGain`: expanded maliyet eksi macro maliyeti.

Varsayılan kapılar:

```text
caseCount >= 3
independentGroupCount >= 2
positiveCaseCount >= 1
negativeCaseCount >= 1
fidelity >= 0.98
falseActivationRate <= 0.05
activationRecall >= 0.95
holdoutUtility >= 0
observedCompressionGain >= 0
```

Pozitif vaka yoksa `activationRecall = 0` olur ve rapor geçmez. Bu, yalnız false-positive ölçerek hiç çalışmayan bir makronun doğrulanmasını engeller. Train vakaları shadow değerlendirmesine alınmaz.

Birden fazla ayrık shadow raporu `applyMacroShadowEvaluation` ile güvenli biçimde birikir. Fidelity vaka sayısıyla, utility beklenen-çıktılı vaka sayısıyla ağırlıklandırılır; positive/missed ve negative/false sayaçları toplanır; shadow compression kazancı toplanır. Aynı report id ikinci kez sayılmaz. Daha önce sayılmış bir case id'yi başka raporda yeniden kullanmak hata verir. Böylece son küçük batch önceki doğrulama kanıtını ezemez.

## Nested DAG

Bir makro başka bir `composite-neuron`a sıralı üye olarak başvurabilir. V1 sınırları:

- doğrudan expansion uzunluğu yine `2..8`;
- en fazla `3` bileşik katmanı;
- cycle kesin hata;
- bilinmeyen nested başvuru kesin hata;
- geri açılan ham üye sayısı varsayılan en fazla `512`;
- expansion sırası birebir korunur.

Derinliği 3'ü aşan yeni aday head'e alınmaz. `expandCompositeNeuron` her aktif/sanal makroyu ham sıraya deterministik biçimde geri açar.

## FamilyRun sürekliliği

`rebaseMacroHeadReference` yalnız aynı `representationSpaceId` içinde çalışır. Farklı temsil uzayına otomatik migration yasaktır.

Yeni katalogda raw üye `id + identityDigest` birebir eşleşirse:

- sequence-macro ve expansion digest korunur;
- family id'leri ve family signature güncellenir.

Bir exact üye kayıpsa veya identity digest değişmişse kayıt silinmez; `deprecated` olur. `equivalence-concept` family signature'ı FamilyRun değişimiyle değişirse de eski aile hipotezi otomatik taşınmaz ve deprecated olur. Yeni aile düzeninde yeniden gözlenmesi gerekir.

## Public API

`src/concept-macros/index.ts` şunları dışa açar:

- `createMacroConfig`, `validateMacroConfig`;
- `createMacroHead`, `validateMacroHead`;
- `mineMacroMotifs`, `ingestMacroTraces`;
- `transitionCompositeLifecycle`;
- `evaluateMacroShadow`, `applyMacroShadowEvaluation`;
- `validateCompositeDag`, `computeCompositeDepth`, `expandCompositeNeuron`;
- `rebaseMacroHeadReference`;
- packed kaynak-grup prototipleri ve bounded replay Bloom yardımcıları;
- `createMacroStructuralEquivalenceAudit`, `applyMacroStructuralEquivalenceAudit`;
- canonical signature, digest, vektör ve MDL yardımcıları;
- bütün V1 tipleri.

Örnek saf akış:

```ts
const head0 = createMacroHead({
  representationSpaceId,
  familyRunFingerprint,
})

const { head: head1, mining } = ingestMacroTraces(head0, traces)
const eligible = head1.macros.find((macro) => macro.lifecycle === 'eligible-proposal')

const head2 = transitionCompositeLifecycle(head1, eligible.id, 'virtual-provisional')
const report = evaluateMacroShadow(
  head2.macros.find((macro) => macro.id === eligible.id),
  holdoutCases,
  head2.config,
)
const head3 = applyMacroShadowEvaluation(head2, report)
```

Pipeline, servis ve UI entegrasyonu bu çekirdeğin dışındadır. Bu ayrım, madencilik ve doğrulama matematiğinin yan etkisiz test edilmesini sağlar.
