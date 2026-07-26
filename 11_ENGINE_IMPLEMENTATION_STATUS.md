# Engine Implementation Status

Son güncelleme: 2026-07-26

Bu belge kuralların kaynağı değildir. Kurallar için
`02_RULES_SPEC.md`, kesin/açık kararlar için
`09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md` authoritative kalır.

## Completed

- Python 3.11 `src/` paket yapısı
- 106 fiziksel taş ve canonical physical ID doğrulaması
- Gösterge, gerçek Okey ve sahte Okey semantiği
- Seri, set, çift ve explicit joker assignment
- Deterministik round deal ve phased turn state machine
- Ortadan çekme ve kullanılabilirlik kontrollü yandan taş alma
- Seri/çift açılışı ve bağımsız katlamalı eşikler
- Table attachment, çift alanı ve Okey geri alma
- Final-discard zorunluluğu
- İşlek/Okey discard cezaları ve explicit finish reasons
- Configurable, bileşenlere ayrılmış round scoring
- Canonical legal-action generator
- Solver katmanı:
  - meld generation
  - opening set packing
  - pair opening
  - table attachments
- Physical conservation, table legality ve terminal invariantleri
- JSON state/action serialization, load ve seed/action replay
- Public discard history: atan oyuncu, turn ve alan oyuncu
- Hidden-information-safe, relative-seat observation
- Deterministik `RandomAgent`
- Multi-round `MatchEngine`
- Paralel benchmark/stress ve replayable failure artifacts

## Public entry points

Development install:

```powershell
python -m pip install -e ".[dev]"
```

```python
from okey101.engine.match import MatchEngine
from okey101.engine.round import RoundEngine

round_engine = RoundEngine()
state = round_engine.reset(seed=42)
legal_actions = round_engine.get_legal_actions()
observation = round_engine.get_observation(state.current_player)

match = MatchEngine()
match.reset(seed=42)
```

## Verification commands

```powershell
python -m pytest -q
python -m compileall -q src benchmarks tests
python -m benchmarks.engine --rounds 100 --seed 0 --json
python -m benchmarks.solver --seeds 100 --measure-memory
python -m benchmarks.stress --rounds 1000 --workers 8
python -m benchmarks.replay_failure stress_failures\seed-<seed>.json
```

Current verified gates:

- unit/integration/differential suite: passing
- compileall: passing
- independent read-only engine audit: no unresolved blocker
- 1,000-round post-hardening invariant stress: passing
- 5,000-round post-hardening invariant stress: passing
- 100,000-round final invariant stress: running

## Checkpoints

```text
cb796f6 feat(engine): add deterministic 101 Okey core
a0bcdb9 refactor(solver): extract canonical candidate generation
4e50fbe feat(engine): add multi-round matches and stress runner
72a20cb fix(engine): harden tile identity and optimize legal search
```

## Known limitations

- `ELDEN_FINISH` exact classification remains intentionally configurable/open.
- Match-level serialize/load is not implemented; round-level state and replay are.
- Actions are typed objects; RL-facing stable candidate IDs/action codec are not yet built.
- RL environment, vector environments and neural self-play have not started.
- Color canonicalization is deferred until after the first RL environment baseline.

## Next gate

Do not begin neural training before the running 100,000-round stress gate
completes without invariant, dead-end or scoring failure.

After that:

1. RL observation tensor encoder
2. stable action codec and masks
3. vectorized environment smoke tests
4. Random baseline
5. small self-play baseline
