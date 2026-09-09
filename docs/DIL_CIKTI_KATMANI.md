# Dil Çıktı Katmanı: Elle Kurulan Çekirdek, Plastik Çevre

## Amaç

Bu katman düşünce ağının yerine geçmez. Düşünce akışında baskın hâle gelen geçici
rol hipotezlerini yüzey metnine dönüştüren, sınırları açık bir çıkış denetleyicisidir.
Rol bilgisi hiçbir düşünce nöronuna kalıcı etiket olarak yazılmaz.

Akış:

```text
baskın düşünce koalisyonu
  -> geçici olay grafiği
  -> rolü sözcüğe çevirme kapısı
  -> sözcük sırası
  -> nesne / yönelme / araç ekleri
  -> fiil, zaman ve olumsuzluk
  -> bağlaç
  -> noktalama
  -> cümleyi bitirme (EOS)
```

## İki ayrı kalıcı katman

### `LanguageOutputScaffoldV1`

Elle yerleştirilen altı ailenin kimliği, kabul ettiği yüzey aralıkları, başlangıç
geçiş önselleri ve 3B atlas yuvaları değişmezdir. Bu kayıt bir `FamilyRun` değildir
ve ampirik aile modelini değiştirmez.

### `LanguageOutputHeadV1`

Eğitimle değişebilen bölüm şunları taşır:

- yüzey nöronu -> çıktı ailesi yumuşak üyelik kanıtı,
- aileler arası uyarıcı ve baskılayıcı geçiş kanıtı,
- henüz güven eşiğini geçmemiş sözcük/ek adayları için nursery,
- kaynak grubu sayıları ve deterministik revizyon parmak izi.

Elle kurulan çekirdeğin tanımı burada yeniden yazılmaz. Öğrenme yalnız plastik
kanıtı artırır veya baskılar.

## Altı çekirdek aile

| Sıra | Aile | Görev | Tipik yüzey aralığı |
|---:|---|---|---|
| 1 | Sözcük sırası | Olay rollerini doğrusal bir plana dizer | sözcük, kısa öbek, yan tümce |
| 2 | Fiil ve zaman | Eylemi çekimli yükleme dönüştürür | fiil kökü, alt-sözcük, sözcük |
| 3 | Hâl ve yönelme | Belirtme, yönelme, bulunma, ayrılma ve araç ilişkisini kurar | alt-sözcük, sözcük |
| 4 | Bağlaç ve olumsuzluk | Önerme bağlantısı ile kutupluluğu denetler | alt-sözcük, sözcük, kısa öbek |
| 5 | Noktalama | Tamamlanan yüzey parçalarını ayırır | grafem, noktalama tokenı |
| 6 | EOS | Yalnız tamamlanmış planı sonlandırır | kontrol işareti |

`Rolü sözcüğe çevirme` bu altı aileden önce duran sanal bir kapıdır; yedinci bir
öğrenilmiş aile değildir. Görevi olay grafiğindeki actor/action/patient/instrument/
goal alanlarını mevcut yüzey adaylarıyla eşlemektir.

## Yönlendirme

Bir sonraki çıktı ailesinin skoru, değişmez önsel ile plastik deneyimin birlikte
hesaplanmasıdır:

\[
L_{ab}=\log(P^0_{ab}+E^+_{ab}+\epsilon)-\log(1+E^-_{ab})
\]

\[
P(b\mid a,s)=\operatorname{softmax}_b\left(L_{ab}+C_b(s)-K_b(s)\right)
\]

Burada `C`, mevcut cümle durumunun aile gereksinimi; `K` ise sıra ihlali ve erken
EOS gibi kısıt cezalarıdır. Elle kurulan önsel öğrenmeyi başlatır fakat tek başına
çıktının doğru olduğuna dair kanıt sayılmaz.

Bir yüzey nöronunun aile üyeliği de aynı biçimde ayrı kanıt taşır:

\[
M_{nf}=\operatorname{softmax}_f\left(B_{nf}+U^+_{nf}-U^-_{nf}\right)
\]

`B`, Türkçe biçim örüntülerinden gelen başlangıç önerisidir. `U` yalnız train
bölümündeki teacher-forcing yaşantısından güncellenir. Validation ve test üyelik
öğrenimine yazılamaz.

## Doğum ve otonom büyüme

Tanımsız fakat tekrar eden bir yüzey doğrudan çekirdek aileye kesin etiketle
yazılmaz. Nursery kaydı açılır ve olası aile ağırlıkları saklanır. Terfi için:

- birden fazla gözlem,
- en az iki bağımsız kaynak grubu,
- tutarlı yüzey/işlev örüntüsü,
- yeterli pozitif kanıt,
- temsil uzayı ve kimlik özeti uyumu

gerekir. Terfi, yeni bir **çıktı nöronu** üretir; temel I/R nöronunu veya dondurulmuş
`FamilyRun` kaydını değiştirmez. Böylece yanlış bir morfoloji tahmini geri alınabilir.

## Türkçe yüzey gerçekleştirme değişmezleri

- Ekler ayrı sözcük olarak basılmaz.
- Ünlü uyumu ve temel ünsüz yumuşaması deterministik uygulanır.
- Özel ad ekleri kesme işaretiyle ayrılır.
- Belirtme ve yönelme aynı anda aynı role uygulanmaz.
- Olumsuzluk bir kez uygulanır; hem bağımsız `değil` hem fiil olumsuzluğu çiftlenmez.
- Noktalama tekilleştirilir.
- Gerekli roller veya yüklem tamamlanmadan EOS kabul edilmez.

## 3B atlas sözleşmesi

Altı aile `language-surface` adlı ayrı bir bölgede, ana aile yoğunluğunun dışında
sabit bir yay üzerinde bulunur. Çekirdek aile koordinatları yalnız çekirdek aile ve
çekirdek geçişlerinden hesaplanır. Dil bölgesine giden bağlantılar çizilir fakat
çekirdek yerleşimine kuvvet uygulamaz. Sonradan doğan çıktı aileleri ilgili manuel
ailenin çevresinde uydu kümeler olarak konumlanır.

Bu görünüm bir bilinç kanıtı değildir; düşünce dinamiği ile yüzey cümlesi arasındaki
denetlenebilir geçişleri gösterir.
