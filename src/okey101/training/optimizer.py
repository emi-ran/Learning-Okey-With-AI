"""Minimal deterministic Adam optimizer used by the NumPy baseline."""

from __future__ import annotations

from math import isfinite, sqrt

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Adam:
    def __init__(
        self,
        parameters: dict[str, FloatArray],
        *,
        learning_rate: float,
        max_gradient_norm: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not isfinite(max_gradient_norm) or max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be finite and positive")
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step = 0
        self.first_moment = {
            name: np.zeros_like(parameter)
            for name, parameter in parameters.items()
        }
        self.second_moment = {
            name: np.zeros_like(parameter)
            for name, parameter in parameters.items()
        }

    def update(
        self,
        parameters: dict[str, FloatArray],
        gradients: dict[str, FloatArray],
    ) -> float:
        if set(parameters) != set(gradients):
            raise ValueError("gradient names do not match model parameters")
        squared_norm = sum(
            float(np.sum(gradient * gradient))
            for gradient in gradients.values()
        )
        gradient_norm = sqrt(squared_norm)
        clip_scale = min(1.0, self.max_gradient_norm / max(gradient_norm, 1e-12))
        self.step += 1

        for name, parameter in parameters.items():
            gradient = gradients[name] * clip_scale
            first = self.first_moment[name]
            second = self.second_moment[name]
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * gradient * gradient
            first_hat = first / (1.0 - self.beta1**self.step)
            second_hat = second / (1.0 - self.beta2**self.step)
            parameter -= (
                self.learning_rate
                * first_hat
                / (np.sqrt(second_hat) + self.epsilon)
            )
        return gradient_norm
