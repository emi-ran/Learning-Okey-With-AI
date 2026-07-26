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
- Fixed-shape, ID-free candidate encoder for all nine action types
- Strict model input containing only encoded observation, candidate rows and mask
- Zero-sum relative terminal rewards
- Dependency-free single-round and sequential vector RL environments
- Deterministic RandomAgent evaluation benchmark through the RL API
- Shared four-seat NumPy actor-critic with variable legal-candidate scoring
- Stochastic self-play, terminal-return policy/value learning and Adam updates
- Atomic, pickle-free `.npz` checkpoints with exact RNG/optimizer resume
- Rotating learned-seat evaluation against three deterministic random agents
- Deterministic learned-policy replay selector with top-k probabilities and value
- Versioned spectator replay JSON with SHA-256 and engine replay verification
- Fixed-seed checkpoint comparison manifests
- Responsive replay viewer with player/spectator visibility and policy analysis
- Pillow frame renderer and FFmpeg H.264 MP4 export
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
    prepare_model_input,
)

round_engine = RoundEngine()
state = round_engine.reset(seed=42)
legal_actions = round_engine.get_legal_actions()
observation = round_engine.get_observation(state.current_player)

match = MatchEngine()
match.reset(seed=42)

env = SingleRoundEnv()
decision = env.reset(seed=42)
catalog = catalog_from_actions(decision.legal_actions)
model_input = prepare_model_input(decision, env.config)
transition = env.step(catalog.decode(0))
```

## Verification commands

```powershell
python -m pytest -q
python -m compileall -q src benchmarks tests
python -m benchmarks.engine --rounds 100 --seed 0 --json
python -m benchmarks.solver --seeds 100 --measure-memory
python -m benchmarks.rl --episodes 100 --start-seed 0
python -m benchmarks.random_baseline --episodes 1000 --start-seed 0 --json
python -m okey101.training.cli --episodes 40 --seed 0 --checkpoint training_runs\checkpoint-40.npz --evaluate 20 --json
python -m benchmarks.replay --model-checkpoint training_runs\checkpoint-40.npz --seeds 42 --output-dir replay_runs\checkpoint-40 --top-candidates 5 --json
python -m benchmarks.render_replay replay_runs\checkpoint-40\checkpoint-40-seed-42.json --output replay_runs\checkpoint-40\checkpoint-40-seed-42.mp4 --fps 2 --json
python -m benchmarks.stress --rounds 1000 --workers 8
python -m benchmarks.replay_failure stress_failures\seed-<seed>.json
```

Current verified gates:

- unit/integration/differential suite: `192 passed`
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
- RL interface with model-input encoding: `9.50 episodes/s`, `653.27 actions/s`
- observation encoding: median `0.130 ms`, p95 `0.211 ms`
- action catalog: median `0.010 ms`, p95 `0.040 ms`
- complete model input: median `0.246 ms`, p95 `1.422 ms`, max `350.477 ms`
- legal candidates: median `1`, p95 `22`, maximum `2,889`

Recorded 1,000-episode RandomAgent baseline, seeds `0..999`:

- actions: `72,099`
- throughput: `16.95 episodes/s`, `1,221.94 actions/s`
- finish rate: `4.50%`
- unopened player rate: `50.45%`
- terminal reasons: 954 stock, 44 normal, 1 same-turn, 1 all-pairs
- penalties/game: deliberately unavailable without post-terminal diagnostic state

Recorded short learning smoke, 20 fixed evaluation seeds `10000..10019`:

- untrained mean relative reward: `-1.1479`
- checkpoint-40 mean relative reward: `0.2762`
- untrained mean score: `275.80`
- checkpoint-40 mean score: `136.95` (lower is better)
- untrained real-Okey discards: `2/109`
- checkpoint-40 real-Okey discards: `0/111`
- untrained playable discards: `17/109`
- checkpoint-40 playable discards: `2/111`

This is a small deterministic smoke comparison, not statistical proof of a
strong player. The policies take different actions after the shared initial
deal, so later states naturally diverge.

## Checkpoints

```text
cb796f6 feat(engine): add deterministic 101 Okey core
a0bcdb9 refactor(solver): extract canonical candidate generation
4e50fbe feat(engine): add multi-round matches and stress runner
72a20cb fix(engine): harden tile identity and optimize legal search
6773bde feat(engine): add verified match snapshots
9461fee feat(rl): add dependency-free environment baseline
4a68f54 feat(rl): add ID-free candidate policy inputs
```

## Known limitations

- `ELDEN_FINISH` exact classification remains intentionally configurable/open.
- Candidate IDs are deliberately state-local; stale catalogs cannot be reused.
- `VectorRoundEnv` is a sequential batching API, not yet a multiprocessing backend.
- Encoder V1 targets single-round training and omits match completed-round context.
- The current learner is a compact NumPy actor-critic, not PPO.
- A long-running self-play league, opponent pool and promotion gate are not yet built.
- The recorded 40-episode result is a pipeline/learning smoke, not a strength claim.
- MP4 export requires a system FFmpeg binary; Pillow is available via the `video` extra.
- Candidate V1 uses 465-float rows and preserves per-group Okey assignments.
- The 2,889-candidate outlier takes about 350 ms and 23.64 MB peak to encode.
- Color canonicalization is deferred until after the first RL environment baseline.

## Next gate

The 100,000-round stress gate completed without invariant, dead-end or scoring
failure. The dependency-free RL interface baseline is also complete.

Next:

1. longer fixed-seed experiments with confidence intervals
2. checkpoint opponent pool and promotion gate
3. hierarchical or streaming scorer for large opening candidate sets
4. batched inference and multiprocessing self-play workers

The observed maximum of `2,889` legal candidates makes a fixed-size flat
categorical head unattractive. Do not silently truncate legal actions; use
candidate scoring or a measured hierarchical policy.
