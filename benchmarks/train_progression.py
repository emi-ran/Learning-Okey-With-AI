"""Train episode-by-episode and publish live status plus checkpoint videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.engine.config import GameConfig
from okey101.replay import (
    CheckpointMetadata,
    record_episode,
    verify_replay,
    write_replay,
)
from okey101.training import (
    DeterministicPolicySelector,
    SelfPlayTrainer,
    TrainingConfig,
    evaluate_against_random,
    save_checkpoint,
)
from okey101.visualization import render_contact_sheet, render_mp4

_STATUS_SCHEMA_VERSION = 2
_STATUS_HISTORY_LIMIT = 1_200


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_checkpoints(raw: str, target_episodes: int) -> tuple[int, ...]:
    try:
        values = tuple(
            int(item.strip())
            for item in raw.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "checkpoints must be comma-separated integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one checkpoint is required")
    if len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise argparse.ArgumentTypeError(
            "checkpoints must be unique and sorted"
        )
    if values[0] != 0:
        raise argparse.ArgumentTypeError("checkpoints must start with 0")
    if values[-1] != target_episodes:
        raise argparse.ArgumentTypeError(
            "the final checkpoint must equal --episodes"
        )
    return values


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        for attempt in range(12):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(0.025 * (attempt + 1))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _web_path(path: Path, repository_root: Path) -> str:
    relative = path.resolve().relative_to(repository_root.resolve())
    return "/" + quote(relative.as_posix())


def _sample_history(
    history: list[dict[str, Any]],
    limit: int = _STATUS_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Keep chart coverage bounded while preserving the newest episodes."""

    if len(history) <= limit:
        return history.copy()
    recent_count = min(300, limit // 3)
    older = history[:-recent_count]
    slots = limit - recent_count
    stride = max(1, (len(older) + slots - 1) // slots)
    sampled = older[::stride][-slots:]
    return [*sampled, *history[-recent_count:]]


def _empty_telemetry(player_count: int) -> dict[str, Any]:
    return {
        "performance": {
            "training_seconds": 0.0,
            "last_episode_seconds": None,
            "episodes_per_second": 0.0,
            "rolling_episodes_per_second": 0.0,
            "actions_per_second": 0.0,
            "eta_seconds": None,
        },
        "totals": {
            "episodes": 0,
            "actions": 0,
            "discard_actions": 0,
            "real_okey_discards": 0,
            "playable_discards": 0,
            "penalty_events": 0,
            "immediate_penalty_points": 0,
            "penalized_episodes": 0,
            "finishes": 0,
            "opened_series": 0,
            "opened_pairs": 0,
        },
        "rates": {
            "real_okey_discard": 0.0,
            "playable_discard": 0.0,
            "penalty_per_episode": 0.0,
            "finish": 0.0,
        },
        "terminal_reasons": {},
        "players": [
            {
                "seat": seat,
                "actions": 0,
                "discards": 0,
                "real_okey_discards": 0,
                "playable_discards": 0,
                "penalty_events": 0,
                "penalty_points": 0,
                "finishes": 0,
                "opened_series": 0,
                "opened_pairs": 0,
                "score_total": 0,
                "mean_score": 0.0,
            }
            for seat in range(player_count)
        ],
    }


def _update_telemetry(
    telemetry: dict[str, Any],
    result: Any,
    *,
    episode_seconds: float,
    rolling_seconds: deque[float],
    target_episodes: int,
) -> None:
    totals = telemetry["totals"]
    totals["episodes"] += 1
    totals["actions"] += result.actions
    totals["discard_actions"] += result.discard_actions
    totals["real_okey_discards"] += result.real_okey_discards
    totals["playable_discards"] += result.playable_discards
    penalty_events = result.real_okey_discards + result.playable_discards
    totals["penalty_events"] += penalty_events
    episode_penalty = sum(result.immediate_penalties)
    totals["immediate_penalty_points"] += episode_penalty
    totals["penalized_episodes"] += int(episode_penalty > 0)
    totals["finishes"] += int(result.winner_seat is not None)
    totals["opened_series"] += sum(
        mode == "series" for mode in result.opened_modes
    )
    totals["opened_pairs"] += sum(
        mode == "pairs" for mode in result.opened_modes
    )

    for seat, player in enumerate(telemetry["players"]):
        player["actions"] += result.actions_by_seat[seat]
        player["discards"] += result.discards_by_seat[seat]
        player["real_okey_discards"] += (
            result.real_okey_discards_by_seat[seat]
        )
        player["playable_discards"] += (
            result.playable_discards_by_seat[seat]
        )
        player["penalty_events"] += (
            result.real_okey_discards_by_seat[seat]
            + result.playable_discards_by_seat[seat]
        )
        player["penalty_points"] += result.immediate_penalties[seat]
        player["finishes"] += int(result.winner_seat == seat)
        player["opened_series"] += int(result.opened_modes[seat] == "series")
        player["opened_pairs"] += int(result.opened_modes[seat] == "pairs")
        player["score_total"] += result.final_scores[seat]
        player["mean_score"] = player["score_total"] / totals["episodes"]

    reasons = telemetry["terminal_reasons"]
    reasons[result.terminal_reason] = reasons.get(result.terminal_reason, 0) + 1
    episodes = totals["episodes"]
    discards = totals["discard_actions"]
    telemetry["rates"].update(
        {
            "real_okey_discard": (
                totals["real_okey_discards"] / discards if discards else 0.0
            ),
            "playable_discard": (
                totals["playable_discards"] / discards if discards else 0.0
            ),
            "penalty_per_episode": (
                totals["immediate_penalty_points"] / episodes
            ),
            "finish": totals["finishes"] / episodes,
        }
    )

    rolling_seconds.append(episode_seconds)
    performance = telemetry["performance"]
    performance["training_seconds"] += episode_seconds
    performance["last_episode_seconds"] = episode_seconds
    performance["episodes_per_second"] = (
        episodes / performance["training_seconds"]
    )
    performance["rolling_episodes_per_second"] = (
        len(rolling_seconds) / sum(rolling_seconds)
    )
    performance["actions_per_second"] = (
        totals["actions"] / performance["training_seconds"]
    )
    remaining = max(0, target_episodes - episodes)
    performance["eta_seconds"] = (
        remaining / performance["rolling_episodes_per_second"]
        if remaining
        else 0.0
    )


def _checkpoint_paths(
    output_dir: Path,
    episode: int,
    replay_seed: int,
) -> dict[str, Path]:
    checkpoint_id = f"checkpoint-{episode:04d}"
    return {
        "checkpoint": output_dir / "checkpoints" / f"{checkpoint_id}.npz",
        "replay": (
            output_dir
            / "replays"
            / f"{checkpoint_id}-seed-{replay_seed}.json"
        ),
        "video": output_dir / "videos" / f"{checkpoint_id}.mp4",
        "poster": output_dir / "posters" / f"{checkpoint_id}.png",
    }


def run_progression(
    *,
    episodes: int,
    checkpoints: tuple[int, ...],
    seed: int,
    replay_seed: int,
    evaluation_episodes: int,
    output_dir: Path,
    top_candidates: int = 5,
    fps: float = 2.0,
    video_size: tuple[int, int] = (1280, 720),
    render_videos: bool = True,
    status_interval_seconds: float = 1.0,
    game_config: GameConfig | None = None,
    training_config: TrainingConfig | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run a fresh progression experiment and atomically publish status."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if checkpoints[0] != 0 or checkpoints[-1] != episodes:
        raise ValueError("checkpoints must start at 0 and end at episodes")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be unique and sorted")
    if evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be positive")
    if status_interval_seconds <= 0:
        raise ValueError("status_interval_seconds must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    history_path = output_dir / "history.jsonl"
    trainer = SelfPlayTrainer(
        seed=seed,
        game_config=game_config,
        training_config=training_config,
    )
    state: dict[str, Any] = {
        "schema_version": _STATUS_SCHEMA_VERSION,
        "run": {
            "name": output_dir.name,
            "seed": seed,
            "replay_seed": replay_seed,
            "target_episodes": episodes,
            "checkpoint_episodes": list(checkpoints),
            "evaluation_episodes": evaluation_episodes,
            "started_at": _utc_now(),
            "completed_at": None,
            "history_path": _web_path(history_path, _REPOSITORY_ROOT),
            "status_interval_seconds": status_interval_seconds,
        },
        "state": {
            "phase": "starting",
            "current_episode": 0,
            "target_episodes": episodes,
            "current_checkpoint": None,
            "message": "Eğitim hazırlanıyor.",
        },
        "history": [],
        "history_entries_total": 0,
        "telemetry": _empty_telemetry(trainer.game_config.player_count),
        "checkpoints": [],
    }

    last_publish = 0.0

    def publish(*, force: bool = True) -> None:
        nonlocal last_publish
        now = time.monotonic()
        if not force and now - last_publish < status_interval_seconds:
            return
        payload = {
            **state,
            "history": _sample_history(state["history"]),
            "history_entries_total": len(state["history"]),
        }
        _atomic_write_json(status_path, payload)
        last_publish = now

    def announce(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    def create_checkpoint(episode: int) -> None:
        checkpoint_id = f"checkpoint-{episode:04d}"
        label = f"{episode} bölüm eğitim"
        paths = _checkpoint_paths(output_dir, episode, replay_seed)
        checkpoint_entry: dict[str, Any] = {
            "episode": episode,
            "id": checkpoint_id,
            "label": label,
            "status": "working",
            "checkpoint_path": None,
            "replay_path": None,
            "video_path": None,
            "poster_path": None,
            "evaluation": None,
            "summary": None,
        }
        state["checkpoints"].append(checkpoint_entry)
        state["state"].update(
            {
                "phase": "checkpointing",
                "current_checkpoint": episode,
                "message": f"Checkpoint {episode} kaydediliyor.",
            }
        )
        publish()
        announce(f"[checkpoint {episode:>4}] model kaydediliyor")

        save_checkpoint(trainer, paths["checkpoint"])
        evaluation = evaluate_against_random(
            trainer.model,
            episodes=evaluation_episodes,
            start_seed=10_000,
            game_config=trainer.game_config,
        )
        selector = DeterministicPolicySelector(
            trainer.model,
            trainer.game_config,
            top_candidates=top_candidates,
        )
        document = record_episode(
            seed=replay_seed,
            selector=selector,
            checkpoint=CheckpointMetadata(
                id=checkpoint_id,
                label=label,
                training_step=episode,
            ),
            policy_name="NumpyActorCritic",
            policy_version="numpy-actor-critic-v1",
            config=trainer.game_config,
            spectator_mode=True,
        )
        verify_replay(document)
        write_replay(document, paths["replay"])

        if render_videos:
            state["state"].update(
                {
                    "phase": "rendering",
                    "message": f"Checkpoint {episode} videosu hazırlanıyor.",
                }
            )
            publish()
            announce(f"[checkpoint {episode:>4}] MP4 hazirlaniyor")
            render_contact_sheet(
                document,
                paths["poster"],
                frame_size=video_size,
            )
            render_mp4(
                document,
                paths["video"],
                fps=fps,
                size=video_size,
            )

        checkpoint_entry.update(
            {
                "status": "ready",
                "checkpoint_path": _web_path(
                    paths["checkpoint"],
                    _REPOSITORY_ROOT,
                ),
                "replay_path": _web_path(paths["replay"], _REPOSITORY_ROOT),
                "video_path": (
                    _web_path(paths["video"], _REPOSITORY_ROOT)
                    if render_videos
                    else None
                ),
                "poster_path": (
                    _web_path(paths["poster"], _REPOSITORY_ROOT)
                    if render_videos
                    else None
                ),
                "evaluation": {
                    **asdict(evaluation),
                    "real_okey_discard_rate": (
                        evaluation.real_okey_discard_rate
                    ),
                    "playable_discard_rate": (
                        evaluation.playable_discard_rate
                    ),
                    "mean_immediate_penalty": (
                        evaluation.mean_immediate_penalty
                    ),
                    "penalized_episode_rate": (
                        evaluation.penalized_episode_rate
                    ),
                },
                "summary": document["summary"],
            }
        )
        state["state"].update(
            {
                "phase": (
                    "training" if episode < episodes else "finalizing"
                ),
                "current_checkpoint": None,
                "message": f"Checkpoint {episode} hazır.",
            }
        )
        publish()

    publish()
    try:
        create_checkpoint(0)
        checkpoint_set = set(checkpoints[1:])
        state["state"].update(
            {
                "phase": "training",
                "message": "Self-play eğitimi başladı.",
            }
        )
        publish()
        rolling_seconds: deque[float] = deque(maxlen=100)
        for episode in range(1, episodes + 1):
            episode_started = time.perf_counter()
            result = trainer.train_episode()
            episode_seconds = time.perf_counter() - episode_started
            history_entry = {
                "episode": result.episode,
                "episode_seed": result.episode_seed,
                "actions": result.actions,
                "loss": result.loss,
                "gradient_norm": result.gradient_norm,
                "mean_score": sum(result.final_scores)
                / len(result.final_scores),
                "best_score": min(result.final_scores),
                "mean_absolute_reward": sum(
                    abs(value) for value in result.rewards
                )
                / len(result.rewards),
                "terminal_reason": result.terminal_reason,
                "winner_seat": result.winner_seat,
                "discard_actions": result.discard_actions,
                "real_okey_discards": result.real_okey_discards,
                "playable_discards": result.playable_discards,
                "penalty_events": (
                    result.real_okey_discards + result.playable_discards
                ),
                "immediate_penalties": list(result.immediate_penalties),
                "opened_modes": list(result.opened_modes),
                "episode_seconds": episode_seconds,
            }
            state["history"].append(history_entry)
            with history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(
                    json.dumps(
                        history_entry,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            _update_telemetry(
                state["telemetry"],
                result,
                episode_seconds=episode_seconds,
                rolling_seconds=rolling_seconds,
                target_episodes=episodes,
            )
            state["state"].update(
                {
                    "phase": "training",
                    "current_episode": episode,
                    "message": f"Episode {episode}/{episodes} tamamlandı.",
                }
            )
            publish(force=False)
            announce(
                f"[{episode:>4}/{episodes}] "
                f"loss={result.loss:+.4f} "
                f"skor={history_entry['mean_score']:.1f} "
                f"hamle={result.actions}"
            )
            if episode in checkpoint_set:
                create_checkpoint(episode)

        state["run"]["completed_at"] = _utc_now()
        state["state"].update(
            {
                "phase": "complete",
                "current_episode": episodes,
                "current_checkpoint": None,
                "message": (
                    f"{episodes} episode ve {len(checkpoints)} checkpoint hazır."
                ),
            }
        )
        publish()
        announce(
            f"Tamamlandi: {episodes} episode, "
            f"{len(checkpoints)} checkpoint, status: {status_path}"
        )
        return state
    except Exception as error:
        state["state"].update(
            {
                "phase": "error",
                "message": f"{type(error).__name__}: {error}",
            }
        )
        publish()
        raise


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_head(self):
        request_path = urlsplit(self.path).path
        if not (
            request_path.startswith("/viewer/")
            or request_path.startswith("/artifacts/training/")
        ):
            self.send_error(404)
            return None
        return super().send_head()


def _start_server(
    repository_root: Path,
    port: int,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = partial(_QuietHandler, directory=str(repository_root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="okey101-training-dashboard",
        daemon=True,
    )
    thread.start()
    return server, thread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--checkpoints", default="0,20,80,200")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-seed", type=int, default=42)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training/live-200"),
    )
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="seconds between live status writes during training",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--serve-port", type=int)
    parser.add_argument("--open-dashboard", action="store_true")
    parser.add_argument("--keep-serving", action="store_true")
    args = parser.parse_args()

    checkpoints = _parse_checkpoints(args.checkpoints, args.episodes)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise SystemExit(
            "--output-dir must stay inside the repository"
        ) from error

    server: ThreadingHTTPServer | None = None
    dashboard_url: str | None = None
    if args.serve_port is not None:
        server, _thread = _start_server(_REPOSITORY_ROOT, args.serve_port)
        status_url = _web_path(output_dir / "status.json", _REPOSITORY_ROOT)
        dashboard_url = (
            f"http://127.0.0.1:{args.serve_port}/viewer/training.html"
            f"?status={quote(status_url, safe='/%')}"
        )
        print(f"Canli panel: {dashboard_url}", flush=True)
        if args.open_dashboard:
            webbrowser.open(dashboard_url)
    elif args.open_dashboard or args.keep_serving:
        parser.error("--open-dashboard/--keep-serving requires --serve-port")

    try:
        run_progression(
            episodes=args.episodes,
            checkpoints=checkpoints,
            seed=args.seed,
            replay_seed=args.replay_seed,
            evaluation_episodes=args.evaluation_episodes,
            output_dir=output_dir,
            top_candidates=args.top_candidates,
            fps=args.fps,
            video_size=(args.width, args.height),
            render_videos=not args.no_video,
            status_interval_seconds=args.status_interval,
            quiet=args.quiet,
        )
        if server is not None and args.keep_serving:
            print(
                "Egitim tamamlandi; panel acik. Kapatmak icin Ctrl+C.",
                flush=True,
            )
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
