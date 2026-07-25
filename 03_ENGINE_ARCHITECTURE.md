# Engine Architecture

## Ana ilke

Engine:

- deterministik,
- neural network'ten bağımsız,
- kolay test edilebilir,
- serializable,
- hızlı,
- reproducible

olmalıdır.

Aynı seed ve aynı action dizisi her zaman aynı sonucu üretmelidir.

---

# 1. Önerilen paket yapısı

```text
src/
├── okey101/
│   ├── engine/
│   │   ├── config.py
│   │   ├── tiles.py
│   │   ├── deck.py
│   │   ├── melds.py
│   │   ├── pairs.py
│   │   ├── joker.py
│   │   ├── table.py
│   │   ├── player.py
│   │   ├── state.py
│   │   ├── actions.py
│   │   ├── legal_actions.py
│   │   ├── transition.py
│   │   ├── penalties.py
│   │   ├── scoring.py
│   │   ├── round.py
│   │   └── match.py
│   │
│   ├── solver/
│   │   ├── meld_generator.py
│   │   ├── opening_solver.py
│   │   ├── attachment_solver.py
│   │   ├── pair_solver.py
│   │   ├── hand_analyzer.py
│   │   └── canonicalization.py
│   │
│   ├── agents/
│   │   ├── base.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   └── neural_agent.py
│   │
│   ├── rl/
│   │   ├── env.py
│   │   ├── observation.py
│   │   ├── action_codec.py
│   │   ├── masks.py
│   │   ├── rewards.py
│   │   ├── vector_env.py
│   │   ├── self_play.py
│   │   └── trainer.py
│   │
│   └── evaluation/
│       ├── tournament.py
│       ├── exploitability.py
│       └── metrics.py
│
└── tests/
```

---

# 2. Tile representation

Normal value identity:

```python
TileValue(color, number)
```

Physical identity:

```python
PhysicalTile(
    id=...,
    kind=NORMAL | FAKE_OKEY,
    color=...,
    number=...
)
```

Gerçek Okey ayrı tile türü yaratılmamalıdır.

Bir normal tile value, o round'da Okey olarak seçilir.

Örneğin:

```text
Physical Kırmızı 12 #A
Physical Kırmızı 12 #B
```

round Okeyi Kırmızı 12 ise ikisi de gerçek Okey davranışı gösterir.

Bu, taş setinin fiziksel doğruluğunu korur.

---

# 3. GameState

Örnek kavramsal yapı:

```python
GameState(
    round_id,
    turn_number,
    current_player,
    dealer_or_starting_player,
    indicator,
    okey_value,
    stock,
    discard_top,
    discard_history,
    players,
    table_melds,
    pair_area,
    progressive_series_threshold,
    progressive_pair_threshold,
    terminal,
    terminal_reason,
)
```

PlayerState:

```python
PlayerState(
    hand,
    opened_mode,        # NONE | SERIES | PAIRS
    opening_turn,
    immediate_penalty,
    score,
)
```

---

# 4. Immutable veya controlled mutation

Tercihen engine transition katmanı dışından GameState mutate edilmemelidir.

Öneri:

```python
new_state, events = engine.step(state, action)
```

veya yüksek performans gerekiyorsa controlled mutable state kullanılabilir.

Ancak debug build'de invariant validation mümkün olmalıdır.

---

# 5. Event log

Her engine step structured event üretmelidir.

Örnek:

```text
DRAW_STOCK
TAKE_DISCARD
OPEN_SERIES
OPEN_PAIRS
ADD_TO_MELD
ADD_PAIR
REPLACE_JOKER
DISCARD
PENALTY
FINISH
ROUND_END
```

Bu log:

- debugging,
- replay,
- training analysis,
- advisor açıklamaları

için çok değerlidir.

---

# 6. Solver strateji vermemeli

Solver'ın görevi exhaustive/legal candidate generation'dır.

Örneğin:

```python
find_all_legal_openings(hand, threshold, ...)
```

şunları döndürebilir:

```text
Opening candidate A
Opening candidate B
Opening candidate C
...
```

Solver:

> "A en mantıklı"

dememelidir.

Bu policy'nin görevidir.

---

# 7. Combinatorial explosion

101 Okey'de açılış kombinasyonları çok büyüyebilir.

Naive brute force eğitim için yeterince hızlı olmayabilir.

Optimize edilebilecek yöntemler:

- bitmask tile representation,
- value-count arrays,
- precomputed possible runs,
- precomputed same-number sets,
- memoization,
- DP over candidate melds,
- canonicalized joker assignments,
- dominated-action pruning where strategically equivalent,
- incremental recomputation.

Ancak optimizasyon correctness'ten önce yapılmamalıdır.

İlk sürüm doğru çalışmalı; profiler sonrası optimize edilmelidir.

---

# 8. Physical tile conservation invariant

Her durumda:

```text
stock
+ all player hands
+ all table meld physical tiles
+ pair area physical tiles
+ current discard-visible tiles according to representation
= exactly all 106 physical tiles
```

olmalıdır.

Aynı physical ID iki yerde bulunamaz.

Hiçbir physical tile kaybolamaz.

Bu invariant debug/test modunda sık sık kontrol edilmelidir.

---

# 9. Joker assignment

Bir table meld içerisinde kullanılan Okey için explicit assignment saklanmalıdır.

Örnek:

```python
MeldTile(
    physical_tile_id=93,
    represented_value=TileValue(RED, 7)
)
```

Normal tile'da represented_value kendi value'sudur.

Bu sistem Okey geri alma işlemini kolaylaştırır.

---

# 10. Configuration

Kurallar hard-code edilmek yerine mümkün olduğunca `RulesConfig` altında tutulmalıdır.

Örnek:

```python
RulesConfig(
    opening_min_score=101,
    opening_min_pairs=5,
    progressive=False,
    max_contiguous_attach=2,
    unopened_end_penalty=202,
    playable_discard_penalty=101,
    normal_okey_discard_penalty=101,
    okey_in_opened_hand_penalty=101,
    require_final_discard=True,
)
```

Tartışmalı bitiş/çarpan davranışları da config'te tutulmalıdır.

---

# 11. Engine API

Minimum:

```python
reset(seed=None) -> GameState
get_observation(player_id)
get_legal_actions(player_id)
step(action)
is_terminal()
get_scores()
serialize_state()
load_state()
```

Ayrıca:

```python
clone_state()
```

gelecekte search/MCTS için faydalı olabilir.

---

# 12. Debugging utilities

Gerekli:

```text
pretty_print_hand
pretty_print_table
state_to_json
replay_from_seed_and_actions
validate_invariants
explain_action_legality
```

Bir test başarısız olduğunda seed + action history ile aynı el yeniden üretilebilmelidir.
