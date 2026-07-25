# 101 Okey Plus–Style Rules Specification

> Bu belge proje için authoritative rule specification olarak kullanılmalıdır.
>
> Kod bu belgeye uymalıdır. Bir kural belirsizse agent kendi başına sessizce karar vermemeli; `09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md` içindeki durum kontrol edilmelidir.

---

# 1. Taş seti

Toplam **106 fiziksel taş** bulunur.

- 4 renk
- her renkte 1–13
- her normal taştan 2 fiziksel kopya
- 2 sahte Okey

Hesap:

```text
4 × 13 × 2 = 104 normal taş
104 + 2 sahte Okey = 106
```

Her fiziksel taş engine içinde benzersiz kimlik taşımalıdır.

Örneğin aynı `Kırmızı 7` taşının iki kopyası oyun mantığında aynı değere sahip olsa da debug/invariant kontrolleri için farklı physical IDs bulunmalıdır.

---

# 2. Gösterge ve Okey

Açılan gösterge taşının aynı renkte bir sonraki sayısı Okey olur.

Örnek:

```text
Gösterge: Sarı 11
Okey:     Sarı 12
```

13'ten sonra 1'e dönülür:

```text
Gösterge: Kırmızı 13
Okey:     Kırmızı 1
```

Gerçek Okey olan iki fiziksel normal taş wildcard/joker gibi kullanılabilir.

---

# 3. Sahte Okey

Sahte Okey wildcard değildir.

Sahte Okey, o el için belirlenmiş **gerçek Okey taşının normal değeri** gibi davranır.

Örnek:

```text
Gösterge = Sarı 11
Gerçek Okey = Sarı 12

Sahte Okey => normal Sarı 12 taşı gibi değerlendirilir.
```

---

# 4. Seri

Seri:

- aynı renk,
- ardışık sayı,
- en az 3 taş

olmalıdır.

Geçerli:

```text
Kırmızı 3-4-5
Kırmızı 8-9-10-11
Mavi 1-2-3
Sarı 11-12-13
```

Geçersiz:

```text
12-13-1
13-1-2
```

Okey belirlerken 13→1 dönüşü vardır; seri oluştururken yoktur.

---

# 5. Aynı sayı peri

En az 3, en fazla 4 farklı renkte aynı sayı.

Geçerli:

```text
Kırmızı 8
Mavi 8
Siyah 8
```

Geçerli:

```text
Kırmızı 8
Mavi 8
Siyah 8
Sarı 8
```

Aynı renkten iki aynı sayı aynı per içerisinde kullanılamaz.

Geçersiz:

```text
Kırmızı 8
Kırmızı 8
Mavi 8
```

---

# 6. Okeyin perlerde kullanımı

Gerçek Okey herhangi bir eksik legal taşın yerine geçebilir.

Örnek:

```text
Kırmızı 5
Kırmızı 6
OKEY
```

Okey burada `Kırmızı 7` olabilir.

Bir per içerisinde birden fazla gerçek Okey kullanılmasına izin verilir.

Örnek:

```text
Kırmızı 5
OKEY
OKEY
Kırmızı 8
```

Okeyler 6 ve 7 olarak yorumlanabilir.

Engine, kullanılan her Okeyin o meld içerisindeki **temsil ettiği gerçek tile value** bilgisini saklamalıdır.

Bu bilgi daha sonra masadaki Okeyin gerçek taşıyla değiştirilerek alınabilmesi için gereklidir.

---

# 7. Normal açılış

Seri/per açılışında oyuncunun aynı açılış aksiyonu içerisinde yere indirdiği kombinasyonların toplam puanı en az 101 olmalıdır.

Okey, temsil ettiği taşın sayısal değeri kadar puan sayılır.

Örnek:

```text
Kırmızı 4
Kırmızı 5
OKEY = Kırmızı 6

Puan = 4 + 5 + 6 = 15
```

Masada daha önce açık perlere taş işlemek, oyuncunun ilk 101 puanını tamamlamak için kullanılamaz.

Önce oyuncunun kendi açılışı legal olmalıdır.

---

# 8. Katlamalı oyun

Katlamalı mod `GameConfig` ile açılıp kapanmalıdır.

## Seri/per açılışı

İlk açan oyuncu en az 101 açar.

Bir oyuncu örneğin 116 açtıysa sonraki seri/per açılışı en az:

```text
117
```

olmalıdır.

Bir sonraki oyuncu 124 açarsa yeni eşik:

```text
125
```

olur.

Seri/per açılış eşiği ile çift açılış eşiği ayrı state olarak tutulmalıdır.

## Çift açılışı

İlk çift açılışı en az 5 çifttir.

Katlamalı oyunda:

```text
5 çift açıldı → sonraki en az 6
6 çift açıldı → sonraki en az 7
...
```

---

# 9. Çift

Bir çift normalde aynı tile value'nun iki fiziksel eşidir.

Örnek:

```text
Kırmızı 7 + Kırmızı 7
```

Gerçek Okey çift tamamlamak için kullanılabilir:

```text
Kırmızı 7 + OKEY
```

legal çifttir.

Ayrıca:

```text
OKEY + OKEY
```

legal çift kabul edilir.

Bu stratejik olarak kötü olabilir ancak **illegal yapılmamalıdır**. AI bunun değerini kendi öğrenmelidir.

---

# 10. Çift açan oyuncu

Çift açmış oyuncu daha sonra yeni seri/per indiremez.

Yapabilir:

- mevcut açık seri/perlere legal taş işlemek,
- çift alanına yeni çiftler indirmek,
- gerekli koşulları sağlıyorsa masadaki Okeyi gerçek taşıyla değiştirmek,
- normal legal discard yapmak.

Yapamaz:

- yeni bağımsız seri/per açmak.

---

# 11. Seri/per açmış oyuncu ve çift alanı

Seri/per açmış oyuncu, başka bir oyuncu tarafından çift alanı açılmışsa elindeki legal çiftleri bu alana indirebilir.

Bu, "ikinci kez çift açılışı yapmak" olarak değil, mevcut çift alanına işleme olarak modellenmelidir.

---

# 12. Masaya işleme

Oyuncu kendi açılışını yapmadan açık perlere işleyemez.

Açıldıktan sonra masadaki legal meldlere taş ekleyebilir.

Açılmış perlerin içindeki normal taşlar sökülüp başka perlerde kullanılamaz.

Rummikub benzeri masa yeniden düzenleme yoktur.

---

# 13. Aynı meld'e art arda işleme limiti

Tek turda aynı mevcut meld'in aynı tarafına/uzantısına **yan yana en fazla 2 yeni taş** eklenebilir.

Örnek:

Masada:

```text
Kırmızı 2-3-4
```

Oyuncu:

```text
Kırmızı 5
Kırmızı 6
```

ekleyebilir.

Ama tek turda aynı uzantıya:

```text
Kırmızı 5
Kırmızı 6
Kırmızı 7
```

üçlüsünü birlikte ekleyemez.

Bu, oyuncunun tur boyunca toplam yalnızca 2 taş işleyebileceği anlamına gelmez.

Farklı perlere daha fazla taş işleyebilir.

---

# 14. Yandan / soldan discard alma

Oyuncu önceki oyuncunun attığı son taşı alabilir ancak bu taş o tur **bizzat kullanılmak zorundadır**.

Henüz açmamış oyuncu için:

- alınan taş, legal açılış kombinasyonlarından en az birinin içinde yer almalıdır,
- açılış 101/katlamalı veya çift açılış şartlarını sağlamalıdır.

Daha önce açmış oyuncu için:

- alınan taş o tur legal biçimde masaya işlenebilmelidir.

Yandan alınan taş elde tutulup tur sonunda başka taş atılamaz.

Yandan taş alındığında ortadaki kapalı taş stoğu azalmaz.

Örnek:

```text
stock_count = 37
soldan taş al
stock_count hâlâ 37
```

Ortadan taş çekilirse:

```text
37 → 36
```

---

# 15. Yandan taşla açtıktan sonra tur devam eder

Yandan alınan taş açılış içerisinde bizzat kullanıldıktan sonra oyuncunun turu bitmez.

Oyuncu aynı turda:

- ek legal perler indirebilir,
- açık perlere taş işleyebilir,
- açık çift alanına çift indirebilir,
- Okey geri alabilir,
- diğer legal masa aksiyonlarını yapabilir,
- en sonunda bir taş atar.

---

# 16. Masadaki Okeyi geri alma

Oyuncu daha önce açmış olmalıdır.

Masadaki bir per içerisinde gerçek Okey belirli bir normal taşı temsil ediyorsa ve oyuncunun elinde **tam olarak o gerçek normal taş** bulunuyorsa:

1. gerçek taş Okeyin yerine konur,
2. Okey oyuncunun eline geçer.

Örnek:

```text
Masada:
Kırmızı 2 - OKEY(as Kırmızı 3) - Kırmızı 4

El:
Kırmızı 3
```

Oyuncu:

```text
Kırmızı 3'ü masaya koyar
OKEY'i eline alır
```

Bu işlem turu sonlandırmaz.

Geri alınan Okey aynı turun devamında:

- yeni perlerde,
- mevcut meldlere işlemede,
- çift alanında

legal olduğu sürece kullanılabilir.

---

# 17. Discard

Her normal tur, oyuncu bitmediyse bir taş atarak tamamlanmalıdır.

Oyuncu elindeki bütün taşları per olarak yere indirip sıfır taşla bitmiş sayılamaz.

**Bitiş için son bir discard zorunludur.**

Örnek:

```text
Elde 4 taş:
Kırmızı 5, 6, 7 ve Siyah 11

5-6-7 yere
Siyah 11 discard
→ bitiş
```

---

# 18. İşlek taş atma

Bir discard, masadaki açık bir meld'e o anda legal biçimde eklenebiliyorsa "işlek" kabul edilebilir.

Normal discard olarak işlek taş atmak:

```text
+101 anlık ceza
```

üretir.

Bu hareket legal fakat cezalıdır.

AI action mask tarafından engellenmemelidir.

## Bitiş istisnası

Oyuncunun **son taşı** işlek olsa bile bu taşla bitmesine izin verilir ve:

```text
+101 işlek cezası uygulanmaz.
```

Bitiş kontrolü işlek-ceza kontrolünden önce yapılmalıdır.

---

# 19. Okey atma

Gerçek Okey normal discard olarak atılırsa:

```text
+101 anlık ceza
```

uygulanır.

Bu hareket legal fakat cezalıdır.

## Okey ile bitiş

Gerçek Okey oyuncunun final discard'ı ise:

- normal +101 Okey-atma cezası verilmez,
- bu `OKEY_FINISH` olarak değerlendirilir.

---

# 20. Anlık cezaların birikmesi

Anlık cezalar round boyunca birikir.

Örnek:

```text
İşlek discard       +101
Daha sonra Okey at  +101
El açmadan round bitti +202

Toplam = 404
```

El-sonu cezası, daha önce birikmiş anlık cezalara eklenir.

---

# 21. Hatalı açılış

İnsan arayüzünü taklit eden modda örneğin:

- 101 altı açma denemesi,
- katlamalı eşikten düşük açma,
- geçersiz per,
- geçersiz çift açılışı

+101 gibi hata cezasına bağlanabilir.

Ancak RL self-play ortamında bu aksiyonlar **legal action listesine hiç verilmemelidir**.

Model deterministik kuralları keşfetmeye zorlanmamalıdır.

---

# 22. Destenin bitmesi

Ortadaki kapalı taşlardan son taş çekildiyse oyuncu mevcut turunu tamamlamaya devam eder.

Yani:

```text
stock = 1
oyuncu son taşı çeker
stock = 0
```

Oyuncu hâlâ:

- açabilir,
- işleyebilir,
- bitebilir,
- discard yapabilir.

Tur tamamlandıktan sonra kimse bitmemişse round sona erer.

Round sonunda:

- açmamış oyuncular: 202,
- açmış oyuncular: ellerinde kalan taşların değerine göre,
- çift açmış oyuncular: ilgili çift ceza kuralına göre

puanlanır.

---

# 23. Dört oyuncunun da çift açması

Dört oyuncunun tamamı çift açarsa el iptal edilir.

Bu durum ayrı terminal reason olarak tutulmalıdır:

```text
ALL_PLAYERS_OPENED_PAIRS
```

Bu el için skor davranışı configurable tutulabilir ancak proje varsayımı:

```text
round void / no normal hand score
```

ve yeni el dağıtılmasıdır.

---

# 24. El sonunda açmamış oyuncu

Oyuncu hiç açmamışsa:

```text
202 ceza
```

alır.

Elinde Okey olup olmaması bu taban cezayı değiştirmez.

Örnek:

```text
açmadı + elde Okey var = 202
```

Okey elde kaldığı için ayrıca +101 eklenmez.

Ancak round sırasında daha önce yaptığı anlık cezalar ayrıca korunur.

---

# 25. El sonunda seri/per açmış oyuncu

Seri/per açmış oyuncunun el-sonu temel cezası:

```text
elde kalan normal taşların değer toplamı
```

Okeyin elde kalması için proje kuralı:

```text
+101 ek sabit ceza
```

olarak tutulmalıdır.

Bu +101'in diğer score multiplier'larla nasıl birleşeceğine dair kesin proje kararı `09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md` içinde belirtilmiştir.

---

# 26. El sonunda çift açmış oyuncu

Çift açmış oyuncunun elde kalan normal taş değer toplamı:

```text
×2
```

olarak cezalandırılır.

Örnek:

```text
elde kalan normal değer = 26
round herhangi bir nedenle bitti
26 × 2 = 52
```

Bunun için başka bir oyuncunun "normal bitmiş" olması gerekmez.

Deste bittiğinde de aynı kural uygulanır.

Okeyle bitiş gibi global finish multiplier'ları ayrıca uygulanabilir.

---

# 27. Bitiş tipleri

Engine bitişi boolean ile değil explicit enum ile temsil etmelidir.

Öneri:

```text
NORMAL_FINISH
SAME_TURN_OPEN_FINISH
ELDEN_FINISH
OKEY_FINISH
ELDEN_OKEY_FINISH
PAIR_FINISH
PAIR_OKEY_FINISH
STOCK_EXHAUSTED
ALL_PLAYERS_OPENED_PAIRS
```

Bazı kategorilerin nihai puan farkları proje konfigürasyonunda açıkça tanımlanmalıdır.

---

# 28. Varsayılan bitiş puanları

Şu anda proje için çalışma tabanı:

```text
Normal bitiş             winner = -101
Okeyle bitiş             winner = -202, opponents ×2
Elden bitiş              winner = -202, opponents ×2
Çiftten bitiş            winner = -202, opponents ×2
Elden + Okey             winner = -404, opponents ×4
Çiftten + Okey           winner = -404, opponents ×4
```

Kullanıcının hatırladığı ek davranış:

> Oyuncu daha önce açmış ve sonraki turda bitmişse -101; ilk açılışını yaptığı turun içinde aynı zamanda bitmişse -202.

Bu davranış engine'de `same_turn_open_finish` olarak ayrı ve configurable tutulmalıdır. Nihai Plus uyumluluk testi yapılmadan `ELDEN_FINISH` ile körü körüne birleştirilmemelidir.

---

# 29. Score multiplier mimarisi

Puanlama mümkün olduğunca bileşenlere ayrılmalıdır.

Örneğin:

```text
base remaining tile penalty
pair-open multiplier
finish multiplier
fixed Okey-in-hand surcharge
accumulated immediate penalties
winner bonus
```

Birbirinden bağımsız bileşenler tek büyük if/else zincirine gömülmemelidir.

Bu sayede tartışmalı kurallar config/test ile değiştirilebilir.

---

# 30. Maç / el sayısı

1 el / 2 el / 3 el vb. engine'in temel round kurallarını değiştirmemelidir.

Öneri:

```python
GameConfig(
    rounds=3,
    progressive_opening=False,
)
```

Her round sonunda score match total'a eklenir.

Training sırasında çoğunlukla tek-round episodic environment kullanılabilir.

Daha sonra multi-round match stratejisi ayrı training/evaluation aşamasında eklenebilir.
