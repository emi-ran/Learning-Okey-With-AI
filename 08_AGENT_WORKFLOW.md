# AI Coding Agent Workflow

## Bu repo üzerinde çalışırken uyulacak ana talimatlar

### 1. Önce belgeleri oku

Kod yazmadan önce:

```text
00_START_HERE.md
01_GOALS_AND_SCOPE.md
02_RULES_SPEC.md
03_ENGINE_ARCHITECTURE.md
04_ACTION_AND_STATE_DESIGN.md
05_TEST_AND_VALIDATION_PLAN.md
06_SELF_PLAY_AI_PLAN.md
07_FUTURE_HAND_ADVISOR.md
09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md
```

dosyalarını oku.

---

# 2. Kuralları kafandan tamamlama

Belgede açık olmayan kuralı sessizce uydurma.

İki seçenek:

1. davranışı config altında açıkça işaretle,
2. `OPEN_ITEM` olarak kaydet.

Engine'in geri kalanını geliştirmeye devam et.

---

# 3. Milestone 1 — Rules core

Implement:

```text
tiles
deck
indicator/Okey
fake Okey
meld validation
pair validation
joker assignment
```

Unit testleri tamamla.

---

# 4. Milestone 2 — Round engine

Implement:

```text
deal
turn state machine
draw
take discard
opening
attachment
pair area
joker replacement
discard
finish
stock exhaustion
```

Unit/integration testleri tamamla.

---

# 5. Milestone 3 — Scoring

Implement:

```text
immediate penalties
end-of-hand penalties
finish types
multipliers
match totals
```

Scoring'i action engine'den mümkün olduğunca ayrı tut.

---

# 6. Milestone 4 — Legal action generator

Her state'te yalnızca legal action üret.

Debug API:

```python
explain_why_action_is_illegal(...)
```

eklemek yararlı olabilir.

---

# 7. Milestone 5 — Random stress test

Dört RandomAgent.

Minimum hedef:

```text
100,000 complete rounds without invariant failure
```

Tercihen:

```text
1,000,000
```

Run sonunda benchmark raporu üret.

---

# 8. Milestone 6 — Optimize

Profiler kullan.

En pahalı alanları ölçmeden premature optimization yapma.

Özellikle:

```text
meld enumeration
opening combination solver
legal action enumeration
state copying
```

kontrol edilmeli.

---

# 9. Milestone 7 — RL environment

Engine testleri güçlü olmadan başlamamalı.

Implement:

```text
observation encoder
action codec
action masks
vectorized environments
terminal reward
```

Random policy environment smoke test.

---

# 10. Milestone 8 — Neural self-play

Küçük baseline model.

Hedef önce:

```text
beats random
```

sonra:

```text
beats heuristic
```

sonra checkpoint league.

---

# 11. Kod kalitesi

- type hints,
- docstrings where useful,
- deterministic seeds,
- no hidden global state,
- pytest,
- clear config,
- reproducible benchmark,
- JSON serializable debug state where possible.

---

# 12. Her milestone sonunda rapor

Agent şu formatta durum raporu bırakmalı:

```markdown
## Completed
...

## Tests
- X passed
- Y failed

## Benchmark
...

## Known limitations
...

## Next step
...
```

---

# 13. Done criteria

"Çalışıyor" demek için yalnızca birkaç demo el yeterli değildir.

Engine done kriterleri:

- rule tests pass,
- scoring regression tests pass,
- invariants pass,
- random stress test pass,
- reproducible seeds,
- no hidden info leakage,
- benchmark recorded.

Bundan sonra AI aşamasına geç.
