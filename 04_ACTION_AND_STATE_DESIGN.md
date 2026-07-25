# Action and Observation Design

## 1. Neden action design kritik?

101 Okey'de tek tur içinde çok sayıda alt karar bulunabilir:

```text
draw / take discard
↓
opening
↓
additional melds
↓
attachments
↓
joker replacement
↓
pairs
↓
discard
```

Bütün turun tüm kombinasyonlarını tek dev action olarak enumerate etmek combinatorial explosion yaratabilir.

Bu nedenle hierarchical / phased action design tercih edilmelidir.

---

# 2. Turn phases

Öneri:

```text
START_TURN
DRAW_DECISION
POST_DRAW
OPENING_OR_TABLE_ACTIONS
DISCARD
TURN_END
```

Daha detaylı state machine kullanılabilir.

---

# 3. Örnek action türleri

```text
DRAW_FROM_STOCK
TAKE_PREVIOUS_DISCARD

OPEN_MELDS(candidate_id)
OPEN_PAIRS(candidate_id)

ADD_TO_MELD(meld_id, tile_or_sequence)
ADD_PAIR(pair_candidate_id)

REPLACE_JOKER(meld_id, replacement_tile_id)

END_TABLE_ACTIONS
DISCARD(tile_id)
```

`candidate_id`ler o anda legal candidate generator tarafından oluşturulabilir.

---

# 4. Yandan taş constraint'i

`TAKE_PREVIOUS_DISCARD` action'ı yalnızca gerçekten kullanılabilecekse legal yapılabilir.

Alternatif implementation:

- taş alınmasına izin ver,
- state üzerinde `must_use_taken_discard=True`,
- discard phase'e geçmeden önce constraint sağlanmak zorunda.

İkinci yöntem kuralları daha açık temsil eder fakat action tree büyüyebilir.

Correctness önceliklidir.

---

# 5. Legal ama cezalı action

Action mask yalnızca illegal action'ları kaldırır.

Örnek legal-cezalı:

```text
işlek taş discard
normal Okey discard
```

Bunlar maskelenmemelidir.

Environment event/reward sisteminde ceza uygulanır.

---

# 6. Observation

Model hidden information görmemelidir.

Önerilen observation bileşenleri:

## Kendi eli

Her tile value için adet:

```text
4 colors × 13 numbers
```

Normal iki kopya bilgisi count ile tutulabilir.

Sahte Okey ayrı kanalda encode edilir.

Gerçek Okey round bilgisi ayrıca verilir.

## Public table

- meldler,
- her meldde Okey assignment,
- pair area,
- visible discards,
- son discard,
- oyuncuların opening mode'u.

## Round context

- sıra,
- stock count,
- progressive threshold,
- pair threshold,
- accumulated penalties,
- round/match score.

---

# 7. Canonicalization

Color symmetry gelecekte kullanılabilir.

Örneğin renk isimlerinin stratejik anlamı yoktur.

Modelin aynı yapıyı dört renkte tekrar öğrenmesini azaltmak için color canonicalization araştırılabilir.

Ancak ilk engine correctness aşamasında gereksiz karmaşıklık yaratmamalıdır.

---

# 8. Action equivalence

Aynı sonuç state'ine giden birden fazla action ordering varsa mümkün olduğunca canonical action oluşturulmalıdır.

Örnek:

```text
önce meld A'yı aç sonra B
```

ve

```text
önce B sonra A
```

aynı final state'i yaratıyorsa policy'nin gereksiz yere iki farklı action sequence öğrenmesi engellenebilir.

Bu optimizasyon daha sonraki aşamadır.

---

# 9. Advisor uyumluluğu

Action representation ileride insan tavsiyesi üretmeye uygun olmalıdır.

Örnek action:

```json
{
  "type": "DISCARD",
  "tile": "RED_8"
}
```

UI'da:

```text
Kırmızı 8'i at.
```

olarak açıklanabilir.

Aynı şekilde model seçiminin expected value / policy probability gibi bilgileri de future advisor kullanabilir.
