# Matematiksel model

Bu belge Sinaps Sandbox'ın tek-adım işlemini özetler. Bütün sinyal ve nöron prototipleri aynı koordinat sisteminde, `32 × 32` ve `[-1, 1]` aralığındadır. Hesaplamalar ara adımlarda yuvarlanmaz; arayüz kesin değerleri dört ondalık basamakla (`0.0000`) gösterir.

## 1. Nöronda sinyal dönüşümü

Sinyal `Sₜ`, mevcut nöronun prototipi `Nᵢ` ile Frobenius iç çarpımından türetilen kosinüs benzerliğiyle karşılaştırılır:

```text
cᵢ = ⟨Sₜ, Nᵢ⟩F / (‖Sₜ‖F ‖Nᵢ‖F + 10⁻⁸)
αₜ = clamp(cᵢ, 0, 1)
Sₜ₊₁ = (1 − αₜ)Sₜ + αₜNᵢ
Δₜ = |Sₜ₊₁ − Sₜ|
```

Negatif veya sıfır uyumda `αₜ = 0` olur ve sinyal değişmez. Güncelleme konveks birleşim olduğu için değerler `[-1, 1]` aralığında kalır.

## 2. K ve H farklı ilişkilerdir

Nöronların gerçek matris benzerliği:

```text
Kᵢⱼ = ⟨Nᵢ, Nⱼ⟩F / (‖Nᵢ‖F ‖Nⱼ‖F + 10⁻⁸)
```

Her yönlü bağlantının kalıcı, negatif olmayan ham kullanım kanıtı `Wᵢⱼ` değeridir. Öz bağlantılar kapalıdır. Kas hafızası olasılığı her kaynak nöronun satırında normalize edilir:

```text
Hᵢⱼ = Wᵢⱼ / Σₖ≠ᵢ Wᵢₖ
```

Bir yolun kullanılması `K` değerini değiştirmez. Gerçek bir `i → j` geçişinde yalnızca `Wᵢⱼ ← Wᵢⱼ + 1` uygulanır ve ilgili `H` satırı yeniden hesaplanır.

## 3. Sıralı aday kabulü

Öz bağlantı dışındaki adaylar `Hᵢⱼ` değerine göre büyükten küçüğe sıralanır. Eşitlikler seeded RNG ile deterministik biçimde çözülür. Dönüşmüş sinyal ile aday prototipi arasındaki pozitif uyum adayın kabul oranıdır:

```text
aⱼ = clamp(cosine(Sₜ₊₁, Nⱼ), 0, 1)
```

Kas hafızası hangi adaya önce bakılacağını, sinyal uyumu ise adayın kabul edilme ihtimalini belirler:

```text
P(J = πᵣ) = aπᵣ ∏ₘ<ᵣ(1 − aπₘ)
P(J = ∅)  = ∏ₘ(1 − aπₘ)
```

Dağılım eşleşmeme olasılığıyla birlikte `1` eder. Sonuç simülasyon durumunda taşınan Mulberry32 RNG ile örneklenir; aynı ilk durum ve seed aynı yolu üretir.

## 4. Nöron oluşma baskısı

Seçilen yolun alışkanlıktan uzaklığı `Dₜ = 1 − Hᵢⱼ`, mevcut adayların temsil hatası `Eₜ = 1 − max(aⱼ)` olarak alınır. Hiçbir aday kabul edilmezse `Dₜ = 1`; hiç aday yoksa `Eₜ = 1 − αₜ` kullanılır.

```text
Gₜ = DₜEₜ
Pₜ = 0.8Pₜ₋₁ + 0.2Gₜ
```

`Gₜ` anlık, `Pₜ` görsel olarak yumuşatılmış baskıdır. Bu sürümde ikisi de yalnızca açıklayıcı göstergedir; otomatik topoloji değişikliği yapmaz.

## 5. Durma ve kalıcılık

Bir adım; aday olmaması, eşleşme olmaması, üst sınıra ulaşılması veya geçersiz sayısal durum nedeniyle akışı durdurabilir. Döngüler serbesttir; iki ile dört nöronluk yakın tekrarlar yalnızca işaretlenir.

Deney sıfırlama sinyali, adımı, yolu ve geçici baskıyı sıfırlar; nöronları ve `W/H` kas hafızasını korur. Ağ sıfırlama bütün kalıcı yapıyı başlangıçtaki tek nöron durumuna döndürür.
