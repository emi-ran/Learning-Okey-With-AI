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

_STATUS_SCHEMA_VERSION = 1


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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
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
        },
        "state": {
            "phase": "starting",
            "current_episode": 0,
            "target_episodes": episodes,
            "current_checkpoint": None,
            "message": "Eğitim hazırlanıyor.",
        },
        "history": [],
        "checkpoints": [],
    }

    def publish() -> None:
        _atomic_write_json(status_path, state)

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
        for episode in range(1, episodes + 1):
            result = trainer.train_episode()
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
            }
            state["history"].append(history_entry)
            state["state"].update(
                {
                    "phase": "training",
                    "current_episode": episode,
                    "message": f"Episode {episode}/{episodes} tamamlandı.",
                }
            )
            publish()
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
            or request_path.startswith("/training_runs/")
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
        default=Path("training_runs/live-200"),
    )
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-video", action="store_true")
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
