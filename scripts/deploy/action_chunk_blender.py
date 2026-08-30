"""Client-side overlap blending for LingBot-VLA action chunks."""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np


class ActionChunkBlender:
    """Replan every ``execute_steps`` and cosine-blend the chunk boundary.

    The policy server returns the full 50-step model prediction and must return
    at least ``execute_steps + blend_steps``.
    The returned dictionary keeps the action feature keys used by LingBot-VLA.
    """

    def __init__(self, execute_steps: int = 15, blend_steps: int = 5):
        if execute_steps < 1 or blend_steps < 0:
            raise ValueError("execute_steps must be >= 1 and blend_steps >= 0")
        self.execute_steps = execute_steps
        self.blend_steps = blend_steps
        self._old_tail: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        self._old_tail = None

    def process(self, new_chunk: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        required = self.execute_steps + self.blend_steps
        arrays = {key: np.asarray(value).copy() for key, value in new_chunk.items()}
        for key, value in arrays.items():
            if value.shape[0] < required:
                raise ValueError(f"{key} has {value.shape[0]} steps; need at least {required}")

        output = {key: value[: self.execute_steps].copy() for key, value in arrays.items()}
        if self._old_tail is not None and self.blend_steps:
            # Starts close to the old plan and ends close to the new observation-conditioned plan.
            phase = np.linspace(0.0, np.pi, self.blend_steps + 2, dtype=np.float32)[1:-1]
            new_weight = (0.5 - 0.5 * np.cos(phase)).reshape((-1,) + (1,) * (next(iter(arrays.values())).ndim - 1))
            for key in output:
                old = self._old_tail[key]
                new = output[key][: self.blend_steps]
                output[key][: self.blend_steps] = (1.0 - new_weight) * old + new_weight * new

        self._old_tail = {
            key: value[self.execute_steps : required].copy()
            for key, value in arrays.items()
        }
        return output
