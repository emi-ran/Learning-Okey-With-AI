"""A compact NumPy actor-critic for variable legal-action catalogs."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from math import sqrt
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from okey101.rl.candidate_encoder import CANDIDATE_FEATURE_SIZE
from okey101.rl.policy import ModelInput

FloatArray = NDArray[np.float64]


def _iter_numeric(value: object) -> Iterator[float]:
    if is_dataclass(value):
        for field in fields(value):
            yield from _iter_numeric(getattr(value, field.name))
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_numeric(item)
        return
    if isinstance(value, (bool, int, float)):
        yield float(value)
        return
    raise TypeError(f"unsupported model feature value: {type(value).__name__}")


def observation_vector(model_input: ModelInput) -> FloatArray:
    """Flatten the versioned observation without accessing raw engine state."""

    values = np.fromiter(
        _iter_numeric(model_input.observation),
        dtype=np.float64,
    )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("observation features must be finite and non-empty")
    return np.tanh(values)


def candidate_matrix(model_input: ModelInput) -> FloatArray:
    """Return enabled candidate rows only; padding can never be selected."""

    enabled = np.asarray(model_input.action_mask, dtype=np.bool_)
    if enabled.ndim != 1 or not np.any(enabled):
        raise ValueError("model input requires at least one enabled candidate")
    rows = np.asarray(model_input.candidate_features, dtype=np.float64)
    if rows.shape != (len(enabled), CANDIDATE_FEATURE_SIZE):
        raise ValueError("candidate feature matrix has an invalid shape")
    if not np.all(np.isfinite(rows)):
        raise ValueError("candidate features must be finite")
    return np.tanh(rows[enabled])


class NumpyActorCritic:
    """Shared four-seat policy with contextual variable-candidate scoring."""

    PARAMETER_NAMES = (
        "observation_weight",
        "observation_bias",
        "candidate_weight",
        "candidate_bias",
        "candidate_score_weight",
        "value_weight",
        "value_bias",
    )

    def __init__(
        self,
        observation_size: int,
        *,
        hidden_size: int = 16,
        rng: np.random.Generator,
    ) -> None:
        if observation_size < 1:
            raise ValueError("observation_size must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.observation_weight = rng.normal(
            0.0,
            1.0 / sqrt(observation_size),
            (observation_size, hidden_size),
        )
        self.observation_bias = np.zeros(hidden_size, dtype=np.float64)
        self.candidate_weight = rng.normal(
            0.0,
            1.0 / sqrt(CANDIDATE_FEATURE_SIZE),
            (CANDIDATE_FEATURE_SIZE, hidden_size),
        )
        self.candidate_bias = np.zeros(hidden_size, dtype=np.float64)
        self.candidate_score_weight = np.zeros(
            CANDIDATE_FEATURE_SIZE,
            dtype=np.float64,
        )
        self.value_weight = rng.normal(
            0.0,
            1.0 / sqrt(hidden_size),
            hidden_size,
        )
        self.value_bias = np.zeros(1, dtype=np.float64)

    @property
    def parameters(self) -> dict[str, FloatArray]:
        return {
            name: getattr(self, name)
            for name in self.PARAMETER_NAMES
        }

    def _forward_arrays(
        self,
        observation: FloatArray,
        candidates: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        if observation.shape != (self.observation_size,):
            raise ValueError("observation vector has an invalid shape")
        observation_hidden = np.tanh(
            observation @ self.observation_weight + self.observation_bias
        )
        candidate_hidden = np.tanh(
            candidates @ self.candidate_weight + self.candidate_bias
        )
        logits = (
            candidate_hidden @ observation_hidden / sqrt(self.hidden_size)
            + candidates @ self.candidate_score_weight
        )
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        value = float(observation_hidden @ self.value_weight + self.value_bias[0])
        return observation_hidden, candidate_hidden, probabilities, value

    def forward(
        self,
        model_input: ModelInput,
    ) -> tuple[FloatArray, float]:
        observation = observation_vector(model_input)
        candidates = candidate_matrix(model_input)
        _oh, _ch, probabilities, value = self._forward_arrays(
            observation,
            candidates,
        )
        return probabilities, value

    def select(
        self,
        model_input: ModelInput,
        *,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float]:
        probabilities, value = self.forward(model_input)
        if deterministic:
            selected = int(np.argmax(probabilities))
        else:
            if rng is None:
                raise ValueError("stochastic selection requires an RNG")
            selected = int(rng.choice(len(probabilities), p=probabilities))
        return selected, value

    def loss_and_gradients(
        self,
        samples: list[tuple[ModelInput, int, float]],
        *,
        value_coefficient: float,
        entropy_coefficient: float,
    ) -> tuple[float, dict[str, FloatArray]]:
        """Compute full-episode REINFORCE and critic gradients."""

        if not samples:
            raise ValueError("at least one training sample is required")
        gradients = {
            name: np.zeros_like(parameter)
            for name, parameter in self.parameters.items()
        }
        loss = 0.0
        scale = sqrt(self.hidden_size)

        for model_input, selected, terminal_return in samples:
            observation = observation_vector(model_input)
            candidates = candidate_matrix(model_input)
            (
                observation_hidden,
                candidate_hidden,
                probabilities,
                value,
            ) = self._forward_arrays(observation, candidates)
            if not 0 <= selected < len(probabilities):
                raise ValueError("selected candidate index is outside the mask")

            advantage = terminal_return - value
            log_probabilities = np.log(np.maximum(probabilities, 1e-12))
            entropy = -float(probabilities @ log_probabilities)
            loss += (
                -advantage * log_probabilities[selected]
                + 0.5 * value_coefficient * (value - terminal_return) ** 2
                - entropy_coefficient * entropy
            )

            logit_gradient = probabilities.copy()
            logit_gradient[selected] -= 1.0
            logit_gradient *= advantage
            logit_gradient += (
                entropy_coefficient
                * probabilities
                * (log_probabilities + entropy)
            )

            gradients["candidate_score_weight"] += (
                candidates.T @ logit_gradient
            )
            candidate_hidden_gradient = (
                logit_gradient[:, None]
                * observation_hidden[None, :]
                / scale
            )
            observation_hidden_gradient = (
                candidate_hidden.T @ logit_gradient / scale
            )
            candidate_pre_gradient = candidate_hidden_gradient * (
                1.0 - candidate_hidden**2
            )
            gradients["candidate_weight"] += (
                candidates.T @ candidate_pre_gradient
            )
            gradients["candidate_bias"] += np.sum(
                candidate_pre_gradient,
                axis=0,
            )

            value_gradient = value_coefficient * (value - terminal_return)
            gradients["value_weight"] += (
                value_gradient * observation_hidden
            )
            gradients["value_bias"][0] += value_gradient
            observation_hidden_gradient += value_gradient * self.value_weight
            observation_pre_gradient = observation_hidden_gradient * (
                1.0 - observation_hidden**2
            )
            gradients["observation_weight"] += np.outer(
                observation,
                observation_pre_gradient,
            )
            gradients["observation_bias"] += observation_pre_gradient

        denominator = float(len(samples))
        for gradient in gradients.values():
            gradient /= denominator
        return loss / denominator, gradients

    def load_parameters(self, parameters: dict[str, FloatArray]) -> None:
        if set(parameters) != set(self.PARAMETER_NAMES):
            raise ValueError("checkpoint model parameter names do not match")
        for name, current in self.parameters.items():
            incoming = np.asarray(parameters[name], dtype=np.float64)
            if incoming.shape != current.shape:
                raise ValueError(f"checkpoint parameter shape mismatch: {name}")
            current[...] = incoming
