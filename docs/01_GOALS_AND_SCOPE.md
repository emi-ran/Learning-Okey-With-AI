# Goals and Scope

## 1. Uzun vadeli ürün hedefi

Aynı engine ve öğrenilmiş model üç farklı kullanım biçimini desteklemelidir.

### A. Self-play training

Dört AI oyuncu birbirlerine karşı oynar.

```text
AI-A
AI-B
AI-C
AI-D
```

Başlangıçta aynı policy kullanılabilir. Eğitim ilerledikçe geçmiş checkpoint'lerden rakipler kullanılarak modelin yalnızca kendi son sürümüne aşırı uyum sağlaması engellenebilir.

### B. Autonomous player

Model gerçek bir 101 Okey oyun durumunda kendi hamlesini seçebilir.

### C. Human hand advisor

Kullanıcı kendi elini ve görünür masa durumunu verir.

Model örneğin:

- "Soldan gelen taşı al."
- "Ortadan çek."
- "Kırmızı 8'i at."
- "Şu üç peri aç."
- "Henüz açma."
- "Bu taşı rakibe verme riski yüksek."
- "Çift stratejisine devam etmek daha iyi."

gibi öneriler üretebilir.

Bu nedenle eğitim sırasında kullanılan observation formatı ileride insan el analizi için de yeniden kullanılabilir olmalıdır.

---

# 2. Öğrenilecek ve kodlanacak şeylerin ayrımı

## Engine tarafından KESİN olarak bilinen şeyler

- taş seti,
- gösterge,
- Okey,
- sahte Okey,
- geçerli seri/per,
- çift,
- açılış şartları,
- katlamalı limitler,
- legal işleme,
- Okey geri alma,
- legal yandan taş alma,
- ceza kuralları,
- bitiş koşulları,
- puanlama,
- destenin bitmesi,
- sıranın ilerlemesi.

Bunlar neural network'e bırakılmaz.

## AI tarafından öğrenilecek şeyler

- hangi taşların gelecekte değerli olduğu,
- hangi taşı tutmanın daha iyi olduğu,
- ne zaman açmanın avantajlı olduğu,
- çift mi seri mi hedeflenmeli,
- Okeyin ne kadar süre tutulması gerektiği,
- soldan gelen taşın alınmaya değer olup olmadığı,
- rakiplerin attığı/aldığı taşlardan çıkarım,
- tehlikeli discard,
- skor durumuna göre risk alma,
- açıldıktan sonra en iyi işleme sırası,
- bitmeye yaklaşan rakibe karşı savunma,
- kısa vadeli ve uzun vadeli ödül dengesi.

---

# 3. Bilgi gizliliği

Engine bütün elleri bilir.

Bir oyuncunun policy'si ise **yalnızca gerçek bir oyuncunun görebileceği bilgileri** almalıdır.

Bir oyuncunun observation'ında rakiplerin gizli elleri bulunamaz.

İzin verilen bilgiler örneğin:

- kendi eli,
- gösterge/Okey bilgisi,
- masadaki açık perler,
- açık çift alanı,
- discard geçmişi,
- soldaki son discard,
- oyuncuların açıp açmadığı,
- kimin çift açtığı,
- katlamalı eşikler,
- ortada kaç taş kaldığı,
- o ana kadar görünür olmuş taşlar,
- skor/round durumu,
- sıra bilgisi.

Bu kural test edilmelidir. Hidden information leakage kritik bug sayılır.

---

# 4. Performans hedefi

Grafik arayüz eğitim sırasında kullanılmamalıdır.

Environment tamamen headless olmalıdır.

Amaç:

- çok sayıda environment'ı paralel çalıştırmak,
- mümkün olduğunca az Python object allocation yapmak,
- legal move üretimini optimize etmek,
- state representation'ı kompakt tutmak,
- neural inference'ı batch olarak yapmak.

İlk benchmark'larda CPU performansı ölçülmelidir.

Örnek metrikler:

```text
games / second
turns / second
legal-action-generation time
meld-solver time
environment-step time
neural-inference time
```

---

# 5. Kapsam dışı ilk aşama

İlk sürümde gerekli değildir:

- mobil uygulama,
- görüntü tanıma,
- OCR,
- ekran görüntüsünden taş algılama,
- animasyon,
- gerçek 101 Okey Plus istemcisine bağlanma,
- insan benzeri taş dizme arayüzü.

Bunlar engine ve model olgunlaştıktan sonra ayrı entegrasyon katmanlarıdır.
