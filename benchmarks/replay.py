"""Record deterministic random or trained-policy episodes for replay/video pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.engine.config import GameConfig
from okey101.replay import (
    ActionSelector,
    CheckpointMetadata,
    build_comparison_manifest,
    record_episode,
    record_random_episode,
    verify_replay,
    write_comparison_manifest,
    write_replay,
)


@dataclass(frozen=True, slots=True)
class ReplayBenchmarkResult:
    replay_count: int
    seeds: tuple[int, ...]
    actions: int
    elapsed_seconds: float
    actions_per_second: float
    output_paths: tuple[str, ...]
    manifest_path: str | None


def record_random_replays(
    *,
    seeds: tuple[int, ...],
    output_dir: str | Path,
    checkpoint: CheckpointMetadata | None = None,
    verify: bool = True,
    manifest_name: str | None = "comparison.json",
) -> ReplayBenchmarkResult:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    metadata = checkpoint or CheckpointMetadata(
        id="random-0",
        label="Random başlangıç",
        training_step=0,
    )
    output = Path(output_dir)
    replay_items: list[tuple[str, dict[str, object]]] = []
    action_count = 0
    started = perf_counter()
    for seed in seeds:
        document = record_random_episode(
            seed=seed,
            checkpoint=metadata,
            spectator_mode=True,
        )
        if verify:
            verify_replay(document)
        path = output / f"{metadata.id}-seed-{seed}.json"
        write_replay(document, path)
        replay_items.append((path.name, document))
        action_count += int(document["summary"]["action_count"])

    manifest_path: Path | None = None
    if manifest_name:
        manifest = build_comparison_manifest(
            title=f"{metadata.label} sabit-seed karşılaştırması",
            fixed_seeds=seeds,
            checkpoints=((document["checkpoint"], replay_items),),
        )
        manifest_path = output / manifest_name
        write_comparison_manifest(manifest, manifest_path)
    elapsed = perf_counter() - started
    return ReplayBenchmarkResult(
        replay_count=len(seeds),
        seeds=seeds,
        actions=action_count,
        elapsed_seconds=elapsed,
        actions_per_second=action_count / elapsed if elapsed > 0 else 0.0,
        output_paths=tuple(str(output / path) for path, _document in replay_items),
        manifest_path=None if manifest_path is None else str(manifest_path),
    )


def record_policy_replays(
    *,
    seeds: tuple[int, ...],
    output_dir: str | Path,
    selector: ActionSelector,
    checkpoint: CheckpointMetadata,
    policy_name: str,
    policy_version: str | None,
    config: GameConfig,
    verify: bool = True,
    manifest_name: str | None = "comparison.json",
) -> ReplayBenchmarkResult:
    """Record deterministic policy episodes using the normal replay schema."""

    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    output = Path(output_dir)
    replay_items: list[tuple[str, dict[str, object]]] = []
    action_count = 0
    started = perf_counter()
    for seed in seeds:
        document = record_episode(
            seed=seed,
            selector=selector,
            checkpoint=checkpoint,
            policy_name=policy_name,
            policy_version=policy_version,
            config=config,
            spectator_mode=True,
        )
        if verify:
            verify_replay(document)
        path = output / f"{checkpoint.id}-seed-{seed}.json"
        write_replay(document, path)
        replay_items.append((path.name, document))
        action_count += int(document["summary"]["action_count"])

    manifest_path: Path | None = None
    if manifest_name:
        manifest = build_comparison_manifest(
            title=f"{checkpoint.label} sabit-seed karşılaştırması",
            fixed_seeds=seeds,
            checkpoints=((document["checkpoint"], replay_items),),
        )
        manifest_path = output / manifest_name
        write_comparison_manifest(manifest, manifest_path)
    elapsed = perf_counter() - started
    return ReplayBenchmarkResult(
        replay_count=len(seeds),
        seeds=seeds,
        actions=action_count,
        elapsed_seconds=elapsed,
        actions_per_second=action_count / elapsed if elapsed > 0 else 0.0,
        output_paths=tuple(str(output / path) for path, _document in replay_items),
        manifest_path=None if manifest_path is None else str(manifest_path),
    )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/replays/random"),
    )
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--checkpoint-label")
    parser.add_argument("--training-step", type=int)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.model_checkpoint is None:
        result = record_random_replays(
            seeds=args.seeds,
            output_dir=args.output_dir,
            checkpoint=CheckpointMetadata(
                id=args.checkpoint_id or "random-0",
                label=args.checkpoint_label or "Random başlangıç",
                training_step=(
                    0 if args.training_step is None else args.training_step
                ),
            ),
            verify=not args.no_verify,
            manifest_name=None if args.no_manifest else "comparison.json",
        )
    else:
        from okey101.training import (
            DeterministicPolicySelector,
            load_checkpoint,
        )

        trainer = load_checkpoint(args.model_checkpoint)
        checkpoint_id = args.checkpoint_id or args.model_checkpoint.stem
        result = record_policy_replays(
            seeds=args.seeds,
            output_dir=args.output_dir,
            selector=DeterministicPolicySelector(
                trainer.model,
                trainer.game_config,
                top_candidates=args.top_candidates,
            ),
            checkpoint=CheckpointMetadata(
                id=checkpoint_id,
                label=args.checkpoint_label or f"Model {checkpoint_id}",
                training_step=(
                    trainer.optimizer.step
                    if args.training_step is None
                    else args.training_step
                ),
            ),
            policy_name="NumpyActorCritic",
            policy_version="numpy-actor-critic-v1",
            config=trainer.game_config,
            verify=not args.no_verify,
            manifest_name=None if args.no_manifest else "comparison.json",
        )
    payload = {
        "replay_count": result.replay_count,
        "seeds": result.seeds,
        "actions": result.actions,
        "elapsed_seconds": result.elapsed_seconds,
        "actions_per_second": result.actions_per_second,
        "output_paths": result.output_paths,
        "manifest_path": result.manifest_path,
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
