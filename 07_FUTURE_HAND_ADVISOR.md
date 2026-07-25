# Future Human Hand Advisor

## Amaç

Self-play model yalnızca bot olarak kalmamalıdır.

İleride kullanıcı kendi elini ve public game state'i vererek:

> "Bu durumda ne yapmalıyım?"

diye sorabilmelidir.

---

# 1. Input

Minimum:

```text
kendi eli
gösterge
masadaki açık perler
çift alanı
son atılan taş
kim açtı
kim çift açtı
katlamalı eşikler
stock count
discard history (varsa)
```

Rakiplerin gizli elleri bilinmeyecektir.

Bu, training observation'ıyla aynı bilgi kısıtına uymalıdır.

---

# 2. Output

Advisor yalnızca tek hareket değil, mümkünse ranked action döndürebilir.

Örnek:

```text
1. Soldaki Kırmızı 8'i al — %43
2. Ortadan çek — %31
3. ...
```

Bir sonraki karar:

```text
Kırmızı 12'yi at.
```

Ek açıklama:

```text
Kırmızı 12 mevcut kombinasyonların en azını bozuyor.
```

Açıklamalar doğrudan neural modelin "düşüncesi" olarak sunulmamalıdır.

Engine features ve action comparison'dan kullanıcıya anlaşılır açıklamalar üretilebilir.

---

# 3. Hand arrangement

UI kullanıcı için taşları otomatik görsel sıralayabilir.

Bu yalnızca presentation katmanıdır.

Örnek sıralamalar:

- renge göre,
- olası serilere göre,
- olası perlere göre,
- çiftlere göre.

Modelin state'i fiziksel raftaki sıra değildir.

---

# 4. Candidate evaluation

İleride advisor modu için modelden:

```text
policy probability
value estimate
counterfactual value
```

alınabilir.

Örneğin aynı state'te:

```text
discard Red 8 → V1
discard Blue 11 → V2
```

karşılaştırması yapılabilir.

---

# 5. Search

Model yeterince iyi hale geldikten sonra advisor modunda daha fazla hesaplama kullanmak mümkün olabilir:

- shallow lookahead,
- Monte Carlo rollout,
- belief sampling for hidden hands,
- MCTS benzeri yöntemler.

Self-play engine bunun için cloneable/simulatable state sağlamalıdır.

---

# 6. Gerçek oyuna entegrasyon

Daha sonraki ayrı proje katmanı:

```text
screen/image recognition
↓
tile detection
↓
game-state reconstruction
↓
advisor model
↓
suggested move
```

Bu ilk engine/RL implementation'ın parçası değildir.
