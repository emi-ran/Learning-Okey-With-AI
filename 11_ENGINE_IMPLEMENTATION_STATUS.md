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
- Match snapshot serialization, strict load ve full-action deterministic replay
- Public discard history: atan oyuncu, turn ve alan oyuncu
- Hidden-information-safe, relative-seat observation
- Versioned, fixed-shape observation encoder without physical IDs
- Deterministic state-local action catalog and padding-only masks
- Zero-sum relative terminal rewards
- Dependency-free single-round and sequential vector RL environments
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
from okey101.rl import (
    SingleRoundEnv,
    catalog_from_actions,
    encode_observation,
)

round_engine = RoundEngine()
state = round_engine.reset(seed=42)
legal_actions = round_engine.get_legal_actions()
observation = round_engine.get_observation(state.current_player)

match = MatchEngine()
match.reset(seed=42)

env = SingleRoundEnv()
decision = env.reset(seed=42)
features = encode_observation(decision.observation, env.config)
catalog = catalog_from_actions(decision.legal_actions)
transition = env.step(catalog.decode(0))
```

## Verification commands

```powershell
python -m pytest -q
python -m compileall -q src benchmarks tests
python -m benchmarks.engine --rounds 100 --seed 0 --json
python -m benchmarks.solver --seeds 100 --measure-memory
python -m benchmarks.rl --episodes 100 --start-seed 0
python -m benchmarks.stress --rounds 1000 --workers 8
python -m benchmarks.replay_failure stress_failures\seed-<seed>.json
```

Current verified gates:

- unit/integration/differential suite: `145 passed`
- compileall: passing
- independent read-only engine audit: no unresolved blocker
- 1,000-round post-hardening invariant stress: passing
- 5,000-round post-hardening invariant stress: passing
- 100,000-round final invariant stress: passing
  - seeds: `17000..116999`
  - actions: `7,228,425`
  - failures: `0`
  - throughput: `52.02 games/s`, `3,760.10 actions/s`
  - terminal coverage: stock, normal, Okey, pair, same-turn, all-pairs

Recorded 100-seed benchmarks:

- sequential engine: `18.03 games/s`, `1,309.97 actions/s`
- legal-action generation: median `0.260 ms`, p95 `1.293 ms`
- meld generation: median `0.209 ms`, p95 `2.098 ms`
- opening solver: median `0.283 ms`, p95 `20.701 ms`
- pair generation: median `0.074 ms`, p95 `0.329 ms`
- pair opening: median `0.086 ms`, p95 `1.591 ms`
- traced solver peak memory: `1,775,212 bytes`
- RL interface: `15.01 episodes/s`, `1,032.41 actions/s`
- observation encoding: median `0.154 ms`, p95 `0.246 ms`
- action catalog: median `0.012 ms`, p95 `0.044 ms`
- legal candidates: median `1`, p95 `22`, maximum `2,889`

## Checkpoints

```text
cb796f6 feat(engine): add deterministic 101 Okey core
a0bcdb9 refactor(solver): extract canonical candidate generation
4e50fbe feat(engine): add multi-round matches and stress runner
72a20cb fix(engine): harden tile identity and optimize legal search
6773bde feat(engine): add verified match snapshots
```

## Known limitations

- `ELDEN_FINISH` exact classification remains intentionally configurable/open.
- Candidate IDs are deliberately state-local; stale catalogs cannot be reused.
- `VectorRoundEnv` is a sequential batching API, not yet a multiprocessing backend.
- Encoder V1 targets single-round training and omits match completed-round context.
- Neural policy, trainer, checkpointing and self-play league have not started.
- Color canonicalization is deferred until after the first RL environment baseline.

## Next gate

The 100,000-round stress gate completed without invariant, dead-end or scoring
failure. The dependency-free RL interface baseline is also complete.

Next:

1. candidate feature encoder for variable-action scoring
2. formal RandomAgent evaluation report through the RL environment
3. framework adapter and batched inference boundary
4. small policy/value self-play baseline

The observed maximum of `2,889` legal candidates makes a fixed-size flat
categorical head unattractive. Do not silently truncate legal actions; use
candidate scoring or a measured hierarchical policy.
