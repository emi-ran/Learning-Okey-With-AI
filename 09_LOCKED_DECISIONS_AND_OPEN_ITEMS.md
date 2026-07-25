# Locked Decisions and Open Items

Bu dosya konuşmada kesinleştirilen proje kararlarını ve hâlâ kontrollü biçimde tutulması gereken küçük belirsizlikleri ayırır.

---

# LOCKED — Kesin proje kararları

## Engine önceliği

AI'dan önce eksiksiz engine.

## AI taş dizilimini öğrenmez

Engine/Solver legal seri/per/çift kombinasyonlarını otomatik bulur.

AI stratejik seçim yapar.

## 106 taş

104 normal + 2 sahte Okey.

## 13 gösterge → 1 Okey

Seride 12-13-1 yine geçersiz.

## Çift açılışı

Minimum 5 çift.

Katlamalıda bir sonraki oyuncu önceki çift sayısından en az 1 fazla açmalı.

## Okey çiftte kullanılabilir

```text
normal + Okey
Okey + Okey
```

legal.

## Birden fazla Okey aynı per içinde kullanılabilir

Legal kabul edilir.

## Çift açan yeni seri/per açamaz

Ama masadaki serilere işleyebilir ve çift alanına devam edebilir.

## Seri/per açmış oyuncu mevcut çift alanına çift indirebilir

Legal.

## Masadaki taşlar yeniden düzenlenemez

Yalnızca legal extension / joker replacement.

## Aynı meld uzantısına tek turda en fazla 2 ardışık yeni taş

Toplam işleme limiti değildir.

## Yandan alınan taş bizzat kullanılmalı

Açılış veya legal işleme içerisinde.

## Yandan alınca stock azalmaz

Ortadan çekince azalır.

## Masadaki Okeyi gerçek taşıyla değiştirip almak

Oyuncu daha önce açmışsa legal.

Okey alındıktan sonra aynı tur devam eder.

Geri alınan Okey aynı tur yeni per/çift/işleme için kullanılabilir.

## Final discard zorunlu

Bütün taşları yere koyup zero-hand ile bitmek yok.

## İşlek normal discard

Legal, +101.

## İşlek final discard

Legal, **ceza yok**.

## Normal Okey discard

Legal, +101.

## Final Okey discard

+101 yok; Okey bitişi.

## Açmamış oyuncunun el sonu cezası

202.

Elinde Okey kalması bunu 303 yapmaz.

## Çift açanın hand penalty'si

Remaining normal tile sum ×2.

Birinin ayrıca bitmiş olması gerekmez.

Stock exhaustion'da da uygulanır.

Örnek:

```text
remaining = 26
pair opened
=> 52
```

## Stock exhaustion

Son kapalı taşı çeken oyuncu turunu tamamlar.

Tur sonunda kimse bitmediyse hand score hesaplanır.

## Dört oyuncu da çift açarsa

Round void / yeniden dağıtım yönünde davran.

---

# OPEN / CONFIGURABLE — Bilinçli olarak config/test altında tutulacaklar

## 1. Aynı tur açıp bitme

Kullanıcının hatırladığı 101 Okey Plus davranışı:

```text
önceden açılmış, sonraki tur bitiş → -101
ilk açılışını yaptığı aynı turda bitiş → -202
```

Bu davranış `same_turn_open_finish_bonus=True` benzeri config altında ayrı tutulmalıdır.

Bunu klasik "elden bitme" kavramıyla sessizce aynı kategori yapma.

Test edilebilir ayrı finish reason üret.

---

## 2. Elden bitmenin tam tanımı

Proje için iki kavram ayrı tutulabilir:

```text
SAME_TURN_OPEN_FINISH
ELDEN_FINISH
```

Böylece gerçek Plus davranışı daha sonra gözlemle doğrulandığında scoring config kolayca değişir.

---

## 3. Açmış oyuncunun elinde Okey kalması

Çalışma varsayımı:

```text
normal remaining tile sum
+ fixed 101 Okey surcharge
```

Çift açmış oyuncu için kullanıcının hatırladığı davranış:

```text
normal_remaining ×2 +101
```

Örnek:

```text
20 normal + elde Okey
pair opened
=> 20×2 +101 = 141
```

Yani +101 Okey surcharge çift multiplier'ına girmiyor.

---

## 4. Rakip Okeyle bittiğinde elde kalan Okey surcharge

Henüz kesinleştirilmemiş küçük kombinasyon:

```text
pair opened
normal remaining = 20
Okey in hand
opponent Okey finish
```

Olası proje davranışı:

```text
20 ×2(pair) ×2(opponent Okey finish) +101
=181
```

Burada +101 sabit surcharge global multiplier dışında bırakılıyor.

Bunu config altında tut ve test ekle.

Agent bunun yerine kendiliğinden 484 vb. farklı davranış seçmemeli.

---

# ScoringConfig önerisi

```python
ScoringConfig(
    normal_finish_reward=-101,
    same_turn_open_finish_reward=-202,
    okey_finish_reward=-202,
    pair_finish_reward=-202,
    elden_finish_reward=-202,

    unopened_end_penalty=202,

    pair_remaining_multiplier=2,
    okey_finish_opponent_multiplier=2,
    elden_finish_opponent_multiplier=2,
    pair_finish_opponent_multiplier=2,

    playable_discard_penalty=101,
    normal_okey_discard_penalty=101,
    opened_player_okey_in_hand_surcharge=101,

    multiply_okey_in_hand_surcharge_by_pair=False,
    multiply_okey_in_hand_surcharge_by_finish=False,
)
```

Bu yapı tartışmalı puanlama ayrıntılarını engine akışını bozmadan değiştirmeyi sağlar.
