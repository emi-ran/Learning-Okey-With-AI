from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from okey101.engine.actions import EndTableActions
from okey101.engine.config import GameConfig
from okey101.replay import (
    ActionSelection,
    CheckpointMetadata,
    SelectionContext,
    load_replay,
    verify_replay,
)
from okey101.rl.env import SingleRoundEnv
from okey101.training import (
    DeterministicPolicySelector,
    SelfPlayTrainer,
    TrainingConfig,
    save_checkpoint,
)


def _fast_config() -> GameConfig:
    return GameConfig(initial_hand_size=3, opening_min_score=3)


def test_deterministic_selector_returns_ranked_public_policy_data() -> None:
    trainer = SelfPlayTrainer(
        seed=5,
        game_config=_fast_config(),
        training_config=TrainingConfig(hidden_size=4),
    )
    env = SingleRoundEnv(_fast_config())
    first = env.reset(seed=9)
    table_decision = env.step(
        next(
            action
            for action in first.legal_actions
            if isinstance(action, EndTableActions)
        )
    ).next_decision
    assert table_decision is not None
    selector = DeterministicPolicySelector(
        trainer.model,
        trainer.game_config,
        top_candidates=3,
    )
    context = SelectionContext(
        seed=9,
        action_index=1,
        player_id=table_decision.seat,
        checkpoint=CheckpointMetadata("fixture", "Fixture", 0),
    )

    first_selection = selector(
        table_decision.observation,
        table_decision.legal_actions,
        context,
    )
    second_selection = selector(
        table_decision.observation,
        tuple(reversed(table_decision.legal_actions)),
        context,
    )

    assert isinstance(first_selection, ActionSelection)
    assert first_selection == second_selection
    assert first_selection.action in table_decision.legal_actions
    assert first_selection.selected_probability is not None
    assert first_selection.value is not None
    assert 1 <= len(first_selection.candidates) <= 3
    probabilities = [
        candidate.probability
        for candidate in first_selection.candidates
    ]
    assert probabilities == sorted(probabilities, reverse=True)
    assert first_selection.candidates[0].action == first_selection.action
    assert first_selection.candidates[0].probability == pytest.approx(
        first_selection.selected_probability
    )


def test_replay_cli_records_loaded_checkpoint_policy(
    tmp_path: Path,
) -> None:
    trainer = SelfPlayTrainer(
        seed=17,
        game_config=_fast_config(),
        training_config=TrainingConfig(
            hidden_size=4,
            max_actions_per_episode=500,
        ),
    )
    trainer.train_episode()
    checkpoint = save_checkpoint(trainer, tmp_path / "learned.npz")
    output_dir = tmp_path / "replays"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.replay",
            "--model-checkpoint",
            str(checkpoint),
            "--seeds",
            "71",
            "--output-dir",
            str(output_dir),
            "--top-candidates",
            "3",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    replay = load_replay(payload["output_paths"][0])

    assert replay["checkpoint"] == {
        "id": "learned",
        "label": "Model learned",
        "training_step": 1,
    }
    assert replay["policy"] == {
        "name": "NumpyActorCritic",
        "version": "numpy-actor-critic-v1",
    }
    policy_steps = [
        frame["policy_step"]
        for frame in replay["frames"][1:]
    ]
    assert all(step["selected_probability"] is not None for step in policy_steps)
    assert all(step["value"] is not None for step in policy_steps)
    assert all(1 <= len(step["candidates"]) <= 3 for step in policy_steps)
    verify_replay(replay)
