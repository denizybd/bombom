# Düşünce Akışı Motoru — Matematiksel Model ve Çalışma Sözleşmesi

Bu belge, Bombom’un sıra numarasıyla dizilmemiş olasılıksal nöron ağı için uygulanan saf 64×64 düşünce motorunu tanımlar. Motor bir metin üreticisi değildir. Bir başlangıç sinyalini, kas hafızası niteliğindeki seyrek yolları, anlık bağlamı, nöron ailelerini ve yenilik baskısını birlikte kullanarak yaşatır. Öğrenme doğrudan taban modele yazılmaz; doğrulanabilir bir `headDelta` önerisi olarak çıkar.

## 1. Temel nesneler

Her sinyal tam 4096 hücrelidir:

\[
S_t\in[-1,1]^{4096}.
\]

Her nöronun gösterimi iki ayrı bölümden oluşur:

\[
Z_i=[I_i\mid R_i],
\]

burada:

- \(I_i\), nöronun değişmez kimlik alanıdır. Oluşturulduğu anda digest’i alınır ve erişildiği her tick öncesi/sonrası doğrulanır.
- \(R_i\), bağlam/ilişki alanıdır. Taban \(R_i\) doğrudan değiştirilmez; kalıcı head yaması ve dal-içi hızlı yama sanal olarak üstüne eklenir.

Bir dalda kullanılan etkin gösterim:

\[
Z_i^{(b,t)}=
\left[I_i\mid
\operatorname{clip}\left(R_i+\Delta R_i^{head}+\Delta R_i^{fast,b,t},-1,1\right)
\right].
\]

Kimlik ve ilişki sınırı snapshot’taki fiziksel TextLab yerleşiminden gelir. “4096 hücrenin tamamı kimliktir” gibi bir varsayım yoktur.

## 2. Başlangıç sinyali \(S_0\)

Prompt tek bir surface hash ile kodlanmaz. Mevcut TextLab kodlayıcısı metni hiyerarşik düğümlere ayırır. Kök ve alt düğümlerin fiziksel \(I/R\) gösterimleri deterministik ağırlıklarla birleştirilir:

\[
w_n=\frac{(2\;\text{if root else}\;1)\sqrt{\max(1,g_n)}}{1+d_n},
\qquad
\bar w_n=\frac{w_n}{\sum_m w_m},
\]

\[
S_0=\operatorname{project}\left(\sum_n\bar w_n[I_n\mid R_n]\right).
\]

Burada \(g_n\) grapheme sayısı, \(d_n\) düğüm derinliğidir. Hiyerarşik analiz başarısız olursa flat kodlama ancak açık `flat-fallback` tanısı ile kullanılabilir. Prompt’un `representationSpaceId` ve I/R sınırı, FamilyRun fiziksel uzayıyla eşleşmek zorundadır.

## 3. Sinyalin nörondan geçişi

Uyum:

\[
c_t=\max(0,\cos(S_t,Z_i^{(b,t)})).
\]

Sinyalin değişim oranı elle seçilmez; tam olarak uyumdur:

\[
\alpha_t=c_t.
\]

Geçiş:

\[
S_{t+1}=\operatorname{project}\left((1-c_t)S_t+c_tZ_i^{(b,t)}\right).
\]

Böylece sinyal uyumlu nöronda daha çok iz alır; uyumsuz nöron onu zorla kendine benzemez.

## 4. Seyrek kas hafızası \(H\)

Taban kanıt \(H^0_{ij}\), doğrulanmış kalıcı uyarım/baskılama \(E^+_{ij},E^-_{ij}\) ve dal-içi hızlı iz \(e^+_{ij},e^-_{ij}\) birlikte kullanılır:

\[
\ell_{ij}=\log(H^0_{ij}+E^+_{ij}+e^+_{ij}+\varepsilon)
-\log(1+E^-_{ij}+e^-_{ij}).
\]

Her origin satırında:

\[
H_{ij}=\operatorname{softmax}_{j}(\ell_{ij}).
\]

Bu bir oran veya sonradan normalize edilmiş “edge strength” kestirimi değildir. Formül doğrudan log-kanıt farkı ve satır softmax’ıdır. Her nöronda en fazla 96 çıkış, tüm başlıkta en fazla 250.000 kenar tutulur.

## 5. Aile üyeliği ve türetilmiş aile grafı

\(M_{ia}\), nöron \(i\)’nin aile \(a\)’ya yumuşak üyeliğidir; bir nöron en fazla beş aile taşır. Aile grafı ayrıca öğrenilmez:

\[
\widetilde G_{ab}=\sum_{i,j}M_{ia}H_{ij}M_{jb},
\qquad
G_{ab}=\frac{\widetilde G_{ab}}{\sum_v\widetilde G_{av}}.
\]

Boyut düzeltmesi veya \(\sqrt{|a||b|}\) cezası yoktur; kanıt kütlesi korunur. \(G\), model/head fingerprintine bağlı runtime-only bir cache’tir. Head revizyonu değişirse yeniden türetilir. Dal-içi hızlı H kanıtı, yalnız geçerli origin satırının aile projeksiyonuna bounded yerel delta olarak eklenir; her tick küresel \(G\) hesaplanmaz.

Runtime satırı için yalnız top-128 aile geçişi indekslenir; geri kalan evidence tek bir `tail` kütlesi olarak korunur. Bu tail ve birleşik logit sonrasında top-8 dışında kalan aile ağırlıkları, \(P(\varnothing)\)’a gider. Bounded çalışma uğruna olasılık kütlesi sessizce yeniden normalize edilip kaybedilmez.

## 6. Gerçek iki aşamalı yönlendirme

### 6.1 Aile aşaması

Bounded aile önerileri şu kaynakların birleşimidir:

- geçerli ailelerin güçlü \(G\) çıkışları,
- sinyal yakınlık indeksinden gelen nöronların aileleri,
- geçici doğumların aileleri,
- branch/path üzerinden o tick yeniden türetilen koalisyon,
- geçici hedef hipotezleri,
- dış adaylar ve doğrulanmış dil deneyimi satırı.

Her öneri için birleşik logit hesaplandıktan sonra en güçlü sekiz aile tutulur:

\[
\sigma_f=-\log(1-0.99c_f),
\qquad
L_f=\log((1-\rho)G_f+\rho/|F|+\varepsilon)
+\sigma_f+k_f+g_f-r_f,
\]

\[
P(f\mid b,t)=\operatorname{softmax}_f(L_f),
\qquad \rho=0.02.
\]

Burada \(c_f\) bounded sinyal-aile cosine uyumu, \(\sigma_f\) onun log-kanıt dönüşümü, \(k_f\) canlı koalisyon desteği, \(g_f\) hedef desteği, \(r_f\) tekrar cezasıdır. Bu dönüşüm, kas yolunda olmayan fakat sinyalle neredeyse tam uyuşan bir aileye gerçekten yarışma şansı verir. FamilyRun içindeki eski koalisyon yalnız başlangıç önselidir; canlı koalisyon her tick branch/path aktivasyonundan yeniden kurulur. Rol hipotezleri kalıcı nöron/aile etiketi değildir.

### 6.2 Seçilen aile içindeki nöron aşaması

Aile \(f\) için kas yolu:

\[
r_j=H_{ij}M_{jf}.
\]

Adaylar \(r_j\)’ye göre sıralanır:

\[
h_j=\frac{r_j}{\max_k r_k}.
\]

Temsil uyumu yalnız cosine’dır; tekrar cezası içine karıştırılmaz:

\[
q_j=\max(0,\cos(S',Z_j)).
\]

Sıralı kabul ve erişim:

\[
a_j=1-(1-q_j)^{1+h_j},
\]

\[
\operatorname{reach}_j=\prod_{k<j}(1-a_k),
\]

\[
P(j,f)=P(f)\operatorname{reach}_j a_j.
\]

Eşleşmeme kütlesi:

\[
P(\varnothing)=\sum_f P(f)\prod_{j\in f}(1-a_j)
+\text{değerlendirilemeyen aile kütlesi}.
\]

Her routing sonucunda:

\[
\sum_{f,j}P(j,f)+P(\varnothing)=1\pm10^{-6}.
\]

## 7. Bounded aday üretimi ve dallanma

Tick sırasında bütün nöronlar veya bütün kenarlar taranmaz. Episode başında şu indeksler hazırlanır:

- origin → en güçlü H satırı,
- aile → bounded üyeler,
- aile → üye-ağırlık lookup tablosu,
- deterministik çok tablalı sign-LSH sinyal yakınlık kovaları,
- dil deneyimi origin → doğrulanmış hedef satırı.

Her tick aday havuzu en fazla 128, exact cosine değerlendirmesi en fazla 32’dir. Kas yolu, LSH, geçici doğum, öğretmen satırı, dış aday ve yüzde 2 seed’li keşif bounded biçimde birleşir. Bir branch yalnız kendi origin dil satırını görür.

Dağılım entropisi:

\[
\mathcal H=-\sum_x p_x\log p_x.
\]

Dal sayısı normalize entropy ile değil, ham Shannon entropy’nin perplexity’siyle belirlenir:

\[
K=\operatorname{clip}_{1..8}\left(\left\lceil e^{\mathcal H}\right\rceil\right).
\]

Deterministik kip en güçlü \(K\) dalı tutar. Seeded-stochastic kip en güçlü \(K-1\) dalı korur; anlamlı artık kütle varsa yalnız bir seed’li keşif dalı ayırır.

## 8. Dal-içi hızlı izler

Hızlı edge izleri 0.90 eligibility ile söner. Varsayılan pozitif ve negatif adımlar:

\[
\eta^+_{fast}=0.05,\qquad \eta^-_{fast}=0.025.
\]

Kanıt tavanı 50’dir. Reddedilen aday yalnız \(e^-\) edge kanıtı alabilir; o nöronun R alanı kirletilmez. Bir tickte yalnız sinyalin gerçekten içinden geçtiği `currentNeuron` kendi \(\Delta R^{fast}\) izini alır. Henüz seçilmiş “next” nöron, ancak sonraki tickte current olduğunda R izi alabilir.

## 9. Yenilik baskısı ve nöron doğumu

Seçilen yolun kas uzaklığı:

\[
d_H=
\begin{cases}
\operatorname{clip}\left(\frac{-\log(H_{ij^*}+\varepsilon)}
{\log(\max(2,|C_t|))}\right), & j^*\text{ seçildiyse},\\
1, & \varnothing\text{ seçildiyse}.
\end{cases}
\]

Temsil hatası:

\[
d_R=1-\max_{j\in C_t}q_j.
\]

İki baskı ayrılır:

\[
p_{edge}=d_H(1-d_R),
\]

\[
p_{birth}=d_Hd_RP(\varnothing).
\]

Yani sinyal iyi açıklanıyor ama alışılmış yol zayıfsa mevcut edge güçlenir; hem yol hem temsil yetersiz ve eşleşmeme kütlesi yüksekse doğum baskısı oluşur. Doğum EMA’sı:

\[
\bar p_t=0.8\bar p_{t-1}+0.2p_{birth,t}.
\]

\(\bar p_t\ge0.65\) koşulu üç ardışık tick sürmeli. Eşleşmeme terminal değildir; aynı origin üzerinde yaşayan aktif self-hold dalıdır. Böylece baskı birikebilir.

Residual, bounded adaylar içindeki maksimum-q gösterimin açıkladığı pozitif bileşen çıkarılarak hesaplanır. Seçilmiş kas adayı başka olsa veya \(\varnothing\) seçilse bile aynı maksimum-q açıklayıcı kullanılır.

### Residual cohesion

Her residual L2-normalize edilerek \(u_t\) yapılır. Bütün örnekleri saklamak yerine sabit bellekli özet tutulur:

\[
U_n=\sum_{t=1}^{n}u_t.
\]

Yeni örneğin önceki bütün örneklerle dot toplamı \(u_{n+1}\cdot U_n\)’dir. Dal içi ve dallar arası bütün çiftler hesaba katılır. Ortalama cohesion en az 0.82 olmalıdır. Böylece ayrı dallarda kendi içinde tutarlı fakat birbirine zıt residual’lar yanlış doğum üretemez.

Üç veya daha çok uyumlu residual, bölüm-içi ortalama prototype oluşturur. Mevcut nöronlarla bounded duplicate kontrolü yapılır. Yeni nöron residual prototype’a en yakın mevcut ailelerden en fazla beş yumuşak üyelik alır; uygun aile yoksa `nursery-provisional:*` ailesine bağlanır. Geçici nöron aynı episode’un sonraki tick aday havuzuna hemen girer.

## 10. Nursery ve küresel terfi

Geçici nöron doğrudan küresel kataloğa yazılmaz. Doğrulanmış sonuç skoru en az 0.80 ise nursery önerisi üretilebilir.

Terfi kapıları:

- en az 3 gözlem,
- en az 2 bağımsız fiziksel kaynak grubu,
- toplam trust mass en az 1,
- ortalama sonuç skoru en az 0.80,
- residual cohesion en az 0.82,
- en az 32 global tick cooldown.

Nursery I alanı ilk bölümde contributing branch’lerin kendi residual ortalamalarının eşit ağırlıklı ortalamasından oluşturulur; branch probability bu kimliği ele geçiremez. Digest ilk doğumda alınır ve daha sonra asla değiştirilmez. Sonraki bağımsız kaynaklar yalnız destek/trust/cohesion ve plastik R prototype’ını günceller. Her kaynak grubu kendi R ortalamasını taşır; nursery R’si kaynak prototiplerinin eşit ağırlıklı ortalamasıdır.

`nurseryUpserts` yalnız adayları, `births` yalnız bütün kapıları geçmiş `born-neuron-v1` kayıtlarını taşır. Terfi etmemiş `temp-*` kimlikleri kalıcı edge veya context patch içinde bulunamaz.

## 11. Beklenen sonuç ve kredi

Beklenen yapı:

```ts
{
  format: 'event-graph-v1',
  actor: string | null,
  action: string | null,
  patient: string | null,
  instrument: string | null,
  operators: string[],
  negated: boolean,
  goals: Array<{ relation: string, target: string }>
}
```

Her retained branch kendi EventGraph’ını ve kendi sonucunu alır. Graph-F1; dolu slotlar, operator kümesi, goal relation-target çiftleri ve negation atomu üstünden hesaplanır. Gold’da null olan bir slota fazladan tahmin false-positive’dir. Constraint accuracy yedi ayrı kısıtın eşit ortalamasıdır: actor, action, patient, instrument, operators, goals, negation.

Branch sonucu:

\[
score=0.7\,F1+0.3\,constraintAccuracy,
\qquad r=2\,score-1.
\]

Sondan geriye eligibility:

\[
\lambda_k=0.90^{T-1-k}.
\]

Kalıcı edge önerileri:

\[
\Delta E^+_k=0.05\max(r,0)\,trust\,\lambda_k,
\]

\[
\Delta E^-_k=0.025\max(-r,0)\,trust\,\lambda_k.
\]

Trust katsayısı model confidence’tan öğrenilmez:

| Kanıt türü | Trust |
|---|---:|
| human | 1.00 |
| deterministic | 1.00 |
| observed | 0.60 |
| ollama-cross-checked | 0.25 |
| self | 0.00 |

Beklenen graph tek başına kalıcı kredi vermez; bağımsız `outcomeEvidence` zorunludur. Aynı episode içinde bir edge’in `trustMass` değeri dış kanıt trust’ını aşamaz; `updateCount` ayrı tutulur.

## 12. Determinizm, checkpoint ve replay

İdari `episodeId`, matematiksel rastgeleliğe veya tick hashine girmez. Semantik `runKey` şunlardan türetilir:

- \(S_0\) digest’i,
- seed ve seçim stratejisi,
- maxTicks ve matematiği etkileyen config,
- model/base/head fingerprint ve revision,
- language-state fingerprint,
- bounded dış aday listesi.

UI metadata’sı (`selectedBy`, `activeAtStart`, `commitEligible`) runKey’e girmez. Branch ve geçici doğum kimlikleri sabit uzunluklu runKey tabanlı digest’lerdir.

Replay bir incremental hash zinciridir; her tick tüm geçmiş yeniden hashlenmez. Checkpoint her 64 tickte ve pause/cancel güvenli sınırında şu durumu taşır:

- tam branch sinyalleri,
- yol/geçiş kayıtları,
- hızlı edge ve R izleri,
- residual/pressure özetleri,
- geçici doğumlar,
- replay zinciri ve deterministic seed/runKey.

Hash tabanlı RNG için runKey+seed+tick+branch kimliği tam yeniden kurulum bilgisidir. `replay.expectedTickCount`, orijinal maxTicks’i değiştirmeden exact prefixte durur; 0 tick de geçerlidir. Pause→resume izi kesintisiz çalışmayla birebir aynı olmalıdır.

## 13. Varsayılan sabitler

| Parametre | Değer |
|---|---:|
| Varsayılan / azami tick | 64 / 512 |
| Nöron başına H edge | 96 |
| Küresel H edge | 250.000 |
| Aday havuzu / exact eval | 128 / 32 |
| Aile adayı | 8 |
| Dal sayısı | 1–8 |
| Keşif rezervi | 0.02 |
| Fast eligibility | 0.90 |
| Fast \(\eta^+ / \eta^-\) | 0.05 / 0.025 |
| Kanıt tavanı | 50 |
| Birth EMA / threshold / ardışık | 0.8 / 0.65 / 3 |
| Cohesion | 0.82 |
| Nursery sonuç skoru | 0.80 |
| Terfi gözlem / kaynak / trust / cooldown | 3 / 2 / 1 / 32 |
| Nöron başına aile üyeliği | 5 |

## 14. Öğrenilen ve sabit kalanlar

Öğrenilen veya veriden türetilenler:

- doğrulanmış \(E^+,E^-\) kanıtı,
- kalıcı R patch önerileri,
- nursery destek/trust/cohesion ve kaynak-dengeli R,
- terfi etmiş yeni nöronlar,
- M üzerinden her head revisionunda yeniden türetilen G,
- branch/path üzerinden canlı koalisyon ve geçici rol hipotezleri.

Sabit sözleşmeler:

- 64×64 sinyal boyutu ve snapshot I/R sınırı,
- \(\alpha=c\), H logit’i, \(G=M^THM\), hazard ve pressure denklemleri,
- seyrek kapasite sınırları,
- trust tablosu,
- nöron I digest’i,
- expected graph’ın nöron/aile özelliğine sızmaması.

## 15. Ölçek sınırı ve dürüst kapasite hesabı

Mevcut runtime katalogu eager Float32 I/R vektörleri taşır. Yalnız ham vektör hesabı:

\[
100{,}000\times4096\times4
=1{,}638{,}400{,}000\text{ byte}
\approx1.53\text{ GiB}.
\]

Dolayısıyla 100K nöronu mevcut eager tasarımla 512 MiB altında tuttuğumuzu iddia etmek matematiksel olarak yanlıştır. Tick yönlendirmesi N veya N² taramaz ve 128/32 sınırında kalır; fakat katalog belleği ayrı bir sorundur. 100K fiziksel hedef için sonraki altyapı adımı lazy/paged `vectorProvider`, quantized depolama veya disk/mmap tabanlı ANN olmalıdır. Mevcut arayüzün lookup/index ayrımı bu geçişe hazırlanmıştır, fakat henüz eager vektör maliyetini ortadan kaldırmaz.

## 16. Bilinen failure mode’lar ve deney kapıları

- LSH yalnız yaklaşık aday bulur; ilgili nöron kovaya düşmezse kas yolu/aile/öğretmen/keşif kaynaklarından gelmelidir.
- Yanlış veya tek kaynaktan tekrarlanan metin bağımsız kanıt sayılmaz; source-group ayrımı korunmalıdır.
- FamilyRun etiketi canlı düşünce kanıtı değildir. Koalisyonun tick sırasında yeniden oluşması ve ablation altında davranışın bozulması gerekir.
- Bir EventGraph skorunun yüksek olması tek başına “düşünce var” kanıtı değildir; ezber/surface leakage olabilir.
- Eager 100K katalog bellek hedefini geçer; tick bounded testi bu fiziksel gerçeği gizlemez.

Minimum nedensel kabul deneyi şunları birlikte istemelidir:

1. bağımsız fiziksel kaynak gruplarıyla en az yüzde 30 group holdout,
2. beş seed ve source-group bootstrap güven aralığı,
3. H0-only, recurrence-off ve shuffled-edge baseline’larına anlamlı üstünlük,
4. hedef aile/edge ablation’ında büyük, sham ablation’da küçük düşüş,
5. pertürbasyondan sonra attractor/persistence geri kazanımı,
6. bilinen yapıda düşük sahte doğum, tekrarlanan gerçek yenilikte yüksek doğum precision’ı,
7. eğitimde görülmemiş actor/action/patient birleşimlerinde yeni kompozisyon.

Bu kapılar geçilmeden sistemin “yapısal düşünce ürettiği” değil, yalnızca denetlenebilir bir dinamik ağ çalıştırdığı söylenmelidir.

## 17. Saf API yüzeyi

Ana adaptör:

```ts
runEpisode(request, context)
// -> { episode, ticks, headDelta?, graph }
```

ve yapılandırılabilir factory:

```ts
createThoughtRuntimeEngine(overrides)
```

Sunucu biçimindeki `{memberKey, identity, relation, identityDigest, text, ...}` kayıtları, FamilyRun üyelikleri ve H0 geçişleriyle açıkça adapte edilir. `sourceIdentityDigest` upstream TextLab digest’ini tanısal olarak korur; runtime I bütünlüğü kendi canonical digest’iyle doğrular. Taban model veya head sessizce overwrite edilmez. `headDelta`, ana modeli mutate etmeyen tek kalıcı öğrenme çıktısıdır.
