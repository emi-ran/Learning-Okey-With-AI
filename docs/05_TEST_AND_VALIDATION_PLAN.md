# Test and Validation Plan

## Kural

AI eğitimine başlamadan önce engine testleri güçlü olmalıdır.

---

# 1. Unit test grupları

## Tiles

- 106 taş oluşuyor mu?
- 104 normal + 2 fake Okey mi?
- her normal value'dan tam iki kopya var mı?
- physical IDs unique mi?

## Gösterge / Okey

- 1→2
- 12→13
- 13→1
- fake Okey doğru value mu?

## Seri

Geçerli:

```text
1-2-3
11-12-13
3-4-5-6
```

Geçersiz:

```text
12-13-1
13-1-2
1-3-4
```

## Aynı sayı peri

- 3 farklı renk geçerli
- 4 farklı renk geçerli
- duplicate color geçersiz
- 2 taş yetersiz

## Joker

- tek Okey ile run
- iki Okey ile run
- Okey ile set
- Okey assignment correctness
- fake Okey wildcard değil

## Çift

- exact duplicate pair
- normal + Okey pair
- Okey + Okey pair
- unrelated two normal tiles invalid

---

# 2. Açılış testleri

- tam 101 geçerli
- 100 geçersiz
- >101 geçerli
- Okey represented value doğru puan
- table attachment ilk 101 hesabına dahil edilmiyor
- katlamalı eşik +1
- seri ve çift katlama sayaçları bağımsız

---

# 3. Yandan taş testleri

- kullanılmayacaksa alınamaz
- opening içinde bizzat kullanılıyorsa alınabilir
- zaten açmış oyuncu attach edebiliyorsa alabilir
- stock count değişmez
- alınan tile fiziksel olarak discard'tan hand/table'a doğru taşınır
- constraint tamamlanmadan turn discard ile sonlandırılamaz

---

# 4. İşleme testleri

- açmadan işleme yasak
- açtıktan sonra legal attach
- aynı uzantıya 2 yeni tile legal
- aynı uzantıya 3 yeni tile illegal
- farklı meldlere toplam >2 tile legal
- table meld reorganization yasak

---

# 5. Okey geri alma testleri

- açmamış oyuncu alamaz
- exact represented tile yoksa alamaz
- exact tile varsa alabilir
- normal replacement masada kalır
- Okey hand'e geçer
- tur devam eder
- aynı tur Okey başka meldde kullanılabilir
- çift alanına kullanılabilir

---

# 6. Discard testleri

## İşlek

- normal işlek discard +101
- birden çok işlek ceza birikir
- final discard işlekse +101 yok

## Okey

- normal Okey discard +101
- final Okey discard +101 değil
- final Okey discard OKEY_FINISH üretir

---

# 7. Bitiş testleri

- sıfır taşla meld açarak bitme yasak
- final discard zorunlu
- normal finish
- same-turn-open finish config
- Okey finish
- pair finish
- elden finish
- combined Okey variants
- finish event'ten sonra turn ilerlemez

---

# 8. Stock exhaustion

```text
stock = 1
draw
stock = 0
```

Oyuncu mevcut turunu tamamlayabilmeli.

Eğer bitmezse tur tamamlanınca round sona ermeli.

Score:

- unopened = 202
- opened series = remaining
- opened pairs = remaining ×2

Anlık cezalar korunmalı.

---

# 9. Scoring regression testleri

Örnek:

```text
pair-opened
remaining normal sum = 26
stock exhausted

expected base hand penalty = 52
```

Okey bitiş multiplier'ı varsa ayrıca uygulanmalı.

Okey-in-opened-hand surcharge için config'e göre ayrı test.

---

# 10. Property / invariant tests

Her engine step sonrası debug mode'da:

```text
TOTAL_PHYSICAL_TILES == 106
UNIQUE_PHYSICAL_IDS == 106
NO_TILE_IN_TWO_LOCATIONS
stock_count >= 0
all melds legal
all joker assignments legal
current player valid
terminal state cannot accept action
```

---

# 11. Random simulation

RandomAgent yalnızca legal actionlardan seçer.

Aşamalar:

```text
100 games
1,000 games
10,000 games
100,000 games
1,000,000 games
```

Her crash'te kaydet:

```text
seed
initial state
full action log
failure
```

Reproduction komutu sağlanmalıdır.

---

# 12. Differential tests

Mümkünse aynı küçük state için iki ayrı implementasyon kullanılabilir:

- basit ama yavaş reference solver,
- optimize solver.

İkisinin legal candidate set'leri karşılaştırılır.

Bu, optimizasyon sonrası correctness'i korumak için değerlidir.

---

# 13. Benchmark

Ayrı benchmark script:

```text
python -m benchmarks.engine
```

Rapor:

```text
games/sec
turns/sec
median legal-action generation ms
p95 legal-action generation ms
opening solver ms
memory / environment
```

Optimizasyon ölçüm olmadan yapılmamalıdır.
