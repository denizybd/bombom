# Nöron Sentezi — Matematiksel Model v1

Bu belge üçüncü programın görevini sınırlar: Metin Laboratuvarı'nın ürettiği ham nöronları değiştirmeden okumak, farklı metinlerde tekrarlanan örüntülerden yeni nöron adayları çıkarmak ve doğumu insan onayına sunmak.

## 1. Girdi sözleşmesi

Her ham nöron iki fiziksel vektör taşır:

- `I`: metin parçasının sabit kimliği,
- `R`: o parçanın öğrenilebilir bağlamı.

Yalnız aynı `representationSpaceId` değerine sahip kayıtlar karşılaştırılır. Kaynak bağımsızlığı proje kimliğiyle değil, ham metnin `sourceDigest` parmak iziyle ölçülür; aynı metnin iki kez içe aktarılması iki ayrı kanıt sayılmaz.

## 2. Aday üye seçimi

Varsayılan seçim, yaprakları ve en fazla 24 grafemlik kısa parçaları alır. Böylece bütün uzun paragraf düğümlerini her koşuda karşılaştırmak yerine anlamlı olma ihtimali yüksek yüzey örüntülerine odaklanılır.

Bir kaynaktaki uygun nöron sayısı `n_a`, diğerindeki `n_b` ise incelenecek çapraz çift sayısı:

```text
P = toplam(a < b) n_a * n_b
```

Aynı kaynağın iki nöronu doğum kanıtı oluşturmak üzere eşleştirilmez.

## 3. İki kanallı benzerlik

Bir `x, y` çifti için kimlik ve ilişki benzerlikleri ayrı hesaplanır:

```text
K_I(x,y) = cos(I_x, I_y)
K_R(x,y) = cos(R_x, R_y)
```

İlişki vektörü henüz yeterince gelişmemişse sıfır bağlamın sonucu bozmasını önlemek için bir güven değeri kullanılır:

```text
g(x,y) = min(1, ||R_x|| / m_R) * min(1, ||R_y|| / m_R)
```

Burada `m_R`, korpustaki pozitif ilişki normlarının medyanıdır. Birleşik değer:

```text
w_R = (1 - w_I) * g(x,y)
K(x,y) = (w_I*K_I + w_R*K_R) / (w_I + w_R)
```

Varsayılan `w_I = 0.8` değeridir. Çift önce asgari `K_I` eşiğini geçmelidir. Çok güçlü kimlik eşleşmeleri `identityOverrideSimilarity` eşiğini aşarsa zayıf veya henüz oluşmamış bağlama rağmen kanıt olarak tutulabilir.

## 4. Seyrek bağ belleği

Program tam bir `N x N` tablo saklamaz. Çiftleri akış halinde inceler; her üye için yalnız en güçlü `k` bağı ve koşu genelinde sınırlı sayıda bağı tutar. Bellek üst sınırı yaklaşık olarak:

```text
O(N*k + E_max)
```

Karşılaştırma süresi hâlâ seçilen farklı-kaynak çiftlerinin sayısıyla orantılıdır. Büyük koşular bu nedenle arayüzde ayrıca açık onay ister.

Hesap ana arayüz iş parçacığında değil ayrı bir Web Worker içinde yürür. Worker'a yalnız kaynak/fingerprint bilgisi ile gerekli `I`/`R` nöron alanları gönderilir; yoğun `K` ve `H` matrisleri IndexedDB tarafında kalır ve ikinci kez belleğe kopyalanmaz. Kullanıcı “Durdur” dediğinde Worker sonlandırıldığı için büyük bir kümenin iç hesabı devam ediyor olsa bile iptal ana arayüzde anında uygulanır.

## 5. Küme ve bağımsız destek

Güçlü bağlar deterministik, boyutu sınırlı tam-bağ uyumlu kümelere birleştirilir. `A-B` ve `B-C` eşikleri geçse bile `A-C` geçmiyorsa üçü aynı kümeye alınmaz; böylece tek-bağ zincirlemesi birbirine benzemeyen uçlardan sahte bir aday doğuramaz. Aynı metindeki üst düğüm ile onun içindeki çakışan alt düğümler ayrı gözlem sayılmaz. Bir adayın doğabilmesi için:

```text
bağımsız destek >= minimumSupport
farklı sourceDigest sayısı >= minimumSources
```

Varsayılan olarak en az üç bağımsız parça ve iki farklı ham metin gerekir.

## 6. Yeni nöronun doğumu

Kümenin sabit kimliği ortalama alınarak bulanıklaştırılmaz. Kümedeki diğer kimliklere toplam kosinüs yakınlığı en yüksek gerçek üye medoid seçilir:

```text
m = argmax(z) toplam(x) cos(I_z, I_x)
I_yeni = copy(I_m)
```

Bu kopya doğumdan sonra kilitlidir ve `identityDigest` ile doğrulanır. İlk ilişki alanı ise bağımsız desteklerin ortalamasıdır:

```text
R_yeni = (1/s) * toplam(x) R_x
```

Doğum puanı sıralama amacıyla hesaplanır:

```text
C_I = ortalama çift kimlik benzerliği
C_R = ortalama çift ilişki benzerliği
B = max(0,C_I) * clamp((C_R+1)/2,0,1) * log(1+s) * d
```

Burada `s` bağımsız destek, `d` farklı kaynak sayısıdır. Puan tek başına nöronu kalıcılaştırmaz; aday kullanıcı tarafından onaylanır veya reddedilir.

## 7. Değişmezler

1. Ham Metin Laboratuvarı kayıtları üçüncü program tarafından değiştirilmez.
2. `I_yeni`, seçilen medoidin `I` vektörünün byte düzeyinde kopyasıdır.
3. Öğrenme yalnız `R` üzerinde devam eder; `I` hiçbir bağlam güncellemesinden etkilenmez.
4. Aynı girdi, ayar ve temsil uzayı deterministik aday parmak izleri üretir.
5. Onaylanan adaylar ayrı `noron-sentez-laboratuvari` deposunda tutulur.

## 8. Bu sürümün bilinçli sınırı

Bu ilk sentez aşaması semantik kavramı doğrudan bildiğini iddia etmez. Tekrarlanan metinsel ve bağlamsal yüzey örüntülerini adaylaştırır. Daha sonra konuşacağımız otomatik sadeleştirme, nöron birleştirme, küme hiyerarşisi ve kas hafızasından sapma sonucu yeni nöron doğurma mekanizması bu güvenli çekirdeğin üzerine kurulacaktır.
