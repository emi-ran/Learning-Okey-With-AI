"""Train or resume the dependency-light NumPy self-play baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .trainer import SelfPlayTrainer, TrainingConfig, evaluate_against_random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate", type=int, default=0, metavar="EPISODES")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    trainer = (
        load_checkpoint(args.resume)
        if args.resume is not None
        else SelfPlayTrainer(
            seed=args.seed,
            training_config=TrainingConfig(
                hidden_size=args.hidden_size,
                learning_rate=args.learning_rate,
            ),
        )
    )
    training_results = trainer.train(args.episodes)
    if args.checkpoint is not None:
        save_checkpoint(trainer, args.checkpoint)
    evaluation = (
        evaluate_against_random(
            trainer.model,
            episodes=args.evaluate,
            game_config=trainer.game_config,
        )
        if args.evaluate
        else None
    )
    payload = {
        "episodes_completed": trainer.episodes_completed,
        "actions_completed": trainer.actions_completed,
        "last_episode": asdict(training_results[-1]),
        "checkpoint": (
            str(args.checkpoint.resolve())
            if args.checkpoint is not None
            else None
        ),
        "evaluation": asdict(evaluation) if evaluation is not None else None,
    }
    if evaluation is not None:
        payload["evaluation"]["real_okey_discard_rate"] = (
            evaluation.real_okey_discard_rate
        )
        payload["evaluation"]["playable_discard_rate"] = (
            evaluation.playable_discard_rate
        )
    if args.as_json:
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
