# Nöron Evreni 3B

Nöron Evreni 3B, seçilen araştırma snapshot’ındaki nöron, aile, üyelik ve
geçiş katmanlarını uzamsal bir haritada incelemek için tasarlanmış salt-okunur
bir gözlem programıdır. Snapshot etkin değilse veya değerlendirme kapısından
reddedilmişse bu durum ekranda açıkça belirtilir. Programın adresi:

```text
http://127.0.0.1:5173/noron-evreni/
```

## Salt-okunur sınır

Görünüm yalnız yerel servisin `GET` uçlarını kullanır. Eğitim çalıştırmaz,
snapshot etkinleştirmez, geri alma yapmaz ve tarayıcı arşivlerine model verisi
yazmaz. Servis yoksa örnek veya sahte ağ üretmek yerine çevrimdışı durumunu
gösterir.

## Görsel kapsam

Nöronlar ve aileler ayrı katmanlardır; üyelikler ile geçişler gerektiğinde
açılıp kapanabilir. Güvenli atlas sınırına sığan seçili FamilyRun içindeki
bütün nöron, yumuşak üyelik ve saklanan temel geçişler gösterilir. Yoğun
`64×64` kimlik/ilişki vektörleri tarayıcıya gönderilmez.

Binlerce öğede arayüzü ve eğitim işlerini yormamak için düğümler toplu GPU
çizimiyle, kenarlar birleşik geometriyle gösterilir. Görünmeyen sekmede çizim
durur; düşük hareket tercihi otomatik dönüşü ve yumuşak kamera hareketlerini
kapatır.
