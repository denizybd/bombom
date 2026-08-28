# Metin Laboratuvarı — Matematiksel Model v0.2

Bu belge, ham metnin nöron oluşlarına ve bağlamsal ısı haritalarına nasıl dönüştürüldüğünü tanımlar. Bu aşamada üretilen her düğüm bir **ham gözlemdir**; kavram değildir. Aynı yazı farklı konumlarda tekrar ederse düğümler ayrı tutulur.

## 1. Sabit hücre uzayı

Her ham nöron iki fiziksel olarak ayrı dizide saklanır; birleşik matris yalnız görünüm sırasında oluşturulur:

\[
N_v = [I_v \mid R_v]
\]

- `I`: Kaynak yüzeyinin değişmeyen kimlik çapası.
- `R`: Üst bağlam ve kardeş geçişlerinin yazabildiği ilişki alanı.

`I` ve `R` aynı `Float32Array` belleğini paylaşmaz. Her `I` için byte düzeyindeki değerlerden `identityDigest` üretilir. Kayıt açılırken hem bu özet hem de kaynak parçasından yeniden kodlanan kimlik doğrulanır. Böylece bir bağlam güncellemesinin kimliği sessizce değiştirmesi veri sözleşmesi düzeyinde engellenir.

Koordinatlar yalnız fiziksel kapasitedir. `R` içinde “önce”, “sonra”, “finans” veya başka bir insan anlamıyla adlandırılmış hücre bulunmaz.

## 2. Kayıpsız bölünme

Her paragraf bir kök düğümdür. Düğüm, kaynak metinde `[başlangıç, bitiş)` aralığı taşır. Bir parça durma eşiğinden büyükse orta noktaya en yakın sınır şu sırayla seçilir:

1. cümle sonu,
2. boşluk sınırı,
3. Unicode grapheme sınırı.

Eşit uzaklıktaki iki adayda soldaki seçilir. Her iç düğüm için şu değişmez korunur:

\[
text(sol) + text(sağ) = text(ebeveyn)
\]

Trim, kelime düzeltme, oluş birleştirme veya sessiz örnekleme yapılmaz.

## 3. Yüzey kimliği

Kaynak yalnız satır sonları ve Unicode NFC açısından kanonikleştirilir. Büyük/küçük harf, Türkçe karakterler, boşluk ve noktalama korunur.

Bir parçadan 1–4 dereceli grapheme n-gramları, başlangıç/bitiş izleri ve kaba göreli konum kopyaları çıkarılır. Her özellik iki kararlı işaretli hash ile `I` maskesine yazılır:

\[
z[h_k(f)] \mathrel{+}= sign_k(f)\,w_f/\sqrt{2}
\]

Sonuç yalnız hücre sınırını korumak için maksimum mutlak değere göre ölçeklenir. Kaynak konumu `I` içine yazılmaz; konum metadata'da, bağlam ise `R` içindedir.

## 4. Yazma imzası ve fiziksel dönüşümler

Kimliğin ilişki alanına bırakacağı yazma imzası, sabit tohumlu seyrek bir işaretli izdüşümdür:

\[
q_v = normalize(A\,normalize(I_v))
\]

`A` yoğun ve öğrenilmiş bir anlam tablosu değildir. Her `I` koordinatı iki `R` koordinatına gider.

İlişki alanında üç kararlı işaretli permütasyon kullanılır:

- `T_P`: üst düğümün izi,
- `T_L`: soldan gelen kardeş izi,
- `T_R`: sağdan gelen kardeş izi.

Bunlar ayrı anlam bölgeleri değil, aynı koordinat alanını farklı şekilde büken fiziksel işlemlerdir. Böylece sıra bilgisi hazır “önce/sonra” etiketi olmadan sonuçta farklı iz bırakır.

## 5. Bağlamın yukarıdan aşağı aktarılması

Paragraf kökü dış bağlam taşımadığı için:

\[
R_{root}=0
\]

Ebeveyn `p`, çocuk `c`, diğer çocuk `s` olsun. Temas kazancı:

\[
g(a,b)=g_0+(1-g_0)\max(0,\cos(I_a,I_b))
\]

Buradaki küçük `g₀`, her temasın mikroskobik de olsa iz bırakmasını; kosinüs bölümü ise uyumlu temasın daha güçlü yazmasını sağlar.

Çocuğun ilişki alanı:

\[
R_c=clip\left(
\rho R_p+
a\,g(c,p)T_Pq_p+
b\,g(c,s)T_{dir}q_s,
-1,1
\right)
\]

İlk deneyin kararlı katsayıları:

- `ρ = 0.80` üst bağlamı koruma,
- `a = 0.15` ebeveyn izi,
- `b = 0.05` kardeş mikro izi,
- `g₀ = 0.10` asgari temas izi.

Katsayıların toplamı birdir. `I` bu geçişte değişmez; yalnız `R` yazılır.

Sonraki bağlam gözlemleri için uyuma bağlı plastik güncelleme kullanılır:

\[
\alpha_t=\min\left(\alpha_{max},\frac{e_t}{\max(1,s_t+e_t)}\right)
\]

\[
R_{t+1}=clip\left(R_t+\alpha_t(C_t-R_t),-1,1\right),
\qquad I_{t+1}=I_t
\]

Burada `eₜ` kimlik uyumu, `sₜ` birikmiş bağlam desteğidir. Destek büyüdükçe tek bir yeni gözlemin etkisi azalır. Her güncellemeden önce ve sonra `identityDigest` doğrulanır.

## 6. Benzerlik K

Tanısal bölge benzerlikleri aynı koordinatı aynı koordinatla karşılaştırır:

\[
K_I(i,j)=\cos(I_i,I_j), \qquad K_R(i,j)=\cos(R_i,R_j)
\]

Sıfır ilişki alanı, yapay bir bağlam benzerliği üretmez. Deneydeki sıfır olmayan ilişki normlarının medyanı `m_R` ile güven katsayısı:

\[
u_i=\min(1,\|R_i\|/m_R), \qquad c_{ij}=u_i u_j
\]

Birleşik benzerlik:

\[
K_{ij}=\frac{\lambda K_I+(1-\lambda)c_{ij}K_R}
{\lambda+(1-\lambda)c_{ij}}
\]

İlk sürümde `λ = 0.5` kullanılır. `R` kanıtı yoksa sonuç doğrudan `K_I` olur; ilişki izi güçlendikçe bağlam ölçüme doğal biçimde katılır.

Küçük deneyde bütün benzersiz çiftler hesaplanır:

\[
\frac{N(N-1)}{2}
\]

Matris simetriktir ve köşegen `Kᵢᵢ = 1` değerini taşır.

Bu ilk tarayıcı sürümü, kesin bölünme sayısı 1.600 ham nöronu aşarsa `K` için büyük tabloları ayırmadan açık hatayla durur. Veri örneklenmez, kesilmez ve birleştirilmez; kullanıcı metni küçültür veya yaprak eşiğini büyütür.

## 7. Başlangıç bağlantıları

Negatif benzerlik geçiş kanıtına çevrilmez. Başlangıç kanıtı:

\[
B_{ij}=\max(0,K_{ij})^p, \qquad p=2
\]

İlk kas-hafızası olasılıkları satır bazında normalize edilir:

\[
H^0_{ij}=\frac{B_{ij}}{\sum_k B_{ik}}
\]

Bir satır tamamen sıfırsa yalnız o düğüm için gözlenen ağaç komşusuna deterministik güvenlik bağı verilir. İleride gerçekleşen gerçek geçişler ayrı `M` kanıtına yazılacaktır:

\[
W=B+M
\]

Böylece metinden gelen başlangıç yakınlığı ile sonradan öğrenilen kas hafızası aynı şeymiş gibi gösterilmez. Kenarlarda ilişki vektörü yoktur; ilişki nöronun `R` alanında, kenar ise yalnız skaler kanıt/olasılıktır.

## 8. Sabit işlem sırası

1. Kaynağı kanonikleştir ve kaydet.
2. Paragraf köklerini ve kayıpsız aralık ağacını kur.
3. Bütün `I` alanlarını üret.
4. Kökten yapraklara tek geçişte `R` alanlarını üret.
5. Bütün çiftler için `K` hesapla.
6. `B` ve başlangıç `H` değerlerini üret.

Aynı kaynak, ayar ve seed aynı işlemi tekrarlar. İlk sürüm özellikle otomatik birleştirme, sadeleştirme, küme, kavram nöronu, canlı nörodoğum, hazır embedding/LLM ve metin üretimi içermez.

## 9. Temsil uzayı sözleşmesi

Farklı metinlerdeki aynı koordinatın gerçekten aynı fiziksel kurala ait olduğunu kanıtlamak için her analiz şu kimlikleri taşır:

- `identitySpaceId`: kanonikleştirme sürümü, yüzey kodlayıcı sürümü, matris maskesi ve hash tohumu,
- `contextSpaceId`: bağlam kodlayıcı sürümü, ilişki alanı ve aktarım katsayıları,
- `representationSpaceId`: bu iki uzayın birleşik özeti,
- `sourceDigest`: aynı içeriğin farklı kaynak kimlikleriyle tekrar sayılmasını önleyen metin özeti,
- `runId`: kaynak, temsil uzayı ve bölünme ayarına bağlı deney kimliği.

Kaynaklar arası sentez yalnız aynı `representationSpaceId` taşıyan tamamlanmış analizleri birlikte kullanır. Yapı uyuşmazlığı sessiz dönüşümle gizlenmez.
