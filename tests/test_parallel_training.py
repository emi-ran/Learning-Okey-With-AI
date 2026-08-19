from __future__ import annotations

import pytest

from okey101.training import (
    ParallelSelfPlayTrainer,
    SelfPlayTrainer,
    TrainingConfig,
)


def test_parallel_self_play_trainer_completes_batch() -> None:
    trainer = SelfPlayTrainer(
        seed=101,
        training_config=TrainingConfig(hidden_size=16),
    )
    with ParallelSelfPlayTrainer(trainer, max_workers=2) as parallel_trainer:
        results = parallel_trainer.train_batch(4)
        assert len(results) == 4
        assert trainer.episodes_completed == 4
        assert trainer.actions_completed > 0
        for result in results:
            assert result.loss is not None
            assert len(result.final_scores) == 4


def test_parallel_self_play_trainer_rejects_invalid_batch_size() -> None:
    trainer = SelfPlayTrainer(seed=42)
    with ParallelSelfPlayTrainer(trainer, max_workers=1) as parallel_trainer:
        with pytest.raises(ValueError, match="positive"):
            parallel_trainer.train_batch(0)
