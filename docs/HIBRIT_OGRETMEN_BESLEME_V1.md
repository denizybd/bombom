# Hibrit Öğretmen Besleme V1

Bu özellik mevcut kapalı seçim hattını değiştirmeden yanına serbest yüzey keşfi ekler.

## Korunan mevcut davranış

- `generationMode` verilmezse kampanya `controlled` çalışır.
- Ollama yalnız sunucunun hazırladığı en çok 16 olay kartından kimlik seçer.
- Sunucu cümleyi kendi dondurulmuş kataloğundan üretir.
- Mevcut protokol fingerprint'i, checkpoint biçimi ve kontrollü istek idempotency hash'i değişmez.
- Eski kampanyalar `generationMode` alanı taşımadan devam edebilir.

## Yeni eklenen davranış

- `generationMode: "hybrid"` ayrı bir hibrit protokol fingerprint'i oluşturur.
- Çizelge deterministiktir: çift indeksli partiler kontrollü, tek indeksli partiler keşiftir. İlk parti her zaman kontrol hattıdır.
- Keşif partisinde Ollama yalnız `sentences: string[]` biçiminde 1–2 yeni Türkçe cümle yazar.
- Modelin iddia ettiği anlam veya etiket kabul edilmez. Beceri ve biçim hedefleri sunucu planından gelir.
- Sunucu cümleyi bağımsız olarak tek-cümle, uzunluk, Türkçe yazım, biçim yüzeyi, müfredat yüzeyi, tekrar ve yakın-tekrar kapılarından geçirir.
- Keşif kaynağı `ollama-authored-discovery`, kontrollü kaynak `ollama-selected-server-rendered` kökeniyle saklanır.
- Her iki kaynak da `evaluationEligible: false` ve `proposalOnly: true` kalır; validation/test verisine dönüşemez ve etkin modeli doğrudan değiştiremez.

## Kalıcılık ve yeniden oynatma

- Parti checkpoint'i `lane` alanıyla hangi hattın çalıştığını sabitler.
- Her pahalı çağrıdan önce istek ve azami token bütçesi kalıcı olarak ayrılır.
- Prompt/şema fingerprint'i aynı seed ve planla yeniden üretilip doğrulanır.
- Kesinti sırasında sonucu belirsiz kalan çağrı yeniden gönderilmez; mevcut muhafazakâr kullanım muhasebesi uygulanır.
- Parti artifact'i kontrollü kanıtı `acceptedChoiceEvidence`, keşif kanıtını `acceptedDiscoveryEvidence` altında ayrı tutar.

## Ölçüm

Hibrit kampanya durumu iki bağımsız sayaç grubu taşır:

- `lanes.controlled`: üretilen, kabul, tekrar, kalite reddi, istek ve token.
- `lanes.discovery`: aynı metriklerin serbest keşif karşılığı.

Toplam sayaçlar mevcut ekran ve otomasyonlarla uyumluluk için korunur. Hat sayaçları keşfin marjinal katkısını kontrollü temelden ayırmak içindir.
