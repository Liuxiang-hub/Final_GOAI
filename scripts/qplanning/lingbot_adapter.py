"""Interface contract for connecting LingBot-VLA 2.0 to Q-Planning.

The checkpoint loader is intentionally not duplicated here: import the exact
server-side policy object already validated in the LingBot environment and
wrap it with ``LingBotCandidateAdapter``. This prevents a second preprocessing
path from silently changing camera order, RGB decoding, or normalization.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .planner import validate_lingbot_candidates


class LingBotSampler(Protocol):
    def sample_action_chunks(
        self, observation: dict[str, Any], *, n_samples: int, denoise_steps: int | None
    ) -> Any: ...


class LingBotCandidateAdapter:
    chunk_size = 50
    action_dim = 14

    def __init__(self, policy: LingBotSampler):
        self.policy = policy

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()

    def sample_chunks(
        self,
        observation: dict[str, Any],
        n_samples: int,
        *,
        denoise_steps: int | None = None,
    ) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        chunks = self.policy.sample_action_chunks(
            observation, n_samples=n_samples, denoise_steps=denoise_steps
        )
        if hasattr(chunks, "detach"):
            chunks = chunks.detach().float().cpu().numpy()
        return validate_lingbot_candidates(chunks)

    @staticmethod
    def unnormalize_actions(normalized: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Convert Q-selected normalized actions to raw units exactly once."""
        values = np.asarray(normalized, dtype=np.float32)
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (14,) or std.shape != (14,) or np.any(std <= 0):
            raise ValueError("mean/std must be 14-D and std must be positive")
        return values * std + mean
