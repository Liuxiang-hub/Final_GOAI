"""Numerically stable Q-weighted action-chunk aggregation.

This module is deliberately model-agnostic. LingBot generates normalized
candidate chunks; a separately trained Q-function supplies one score per
chunk. The selected normalized chunk then follows the existing deployment
post-processing and safety pipeline.
"""

from __future__ import annotations

import numpy as np


def q_weighted_chunk(
    candidates: np.ndarray,
    q_values: np.ndarray,
    *,
    temperature: float = 1.0,
    n_elites: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted chunk and full-size candidate weights.

    Args:
        candidates: ``[N, H, A]`` normalized action chunks.
        q_values: ``[N]`` finite scalar values.
        temperature: positive softmax temperature.
        n_elites: average only the top-k candidates; 0 means all.
    """
    chunks = np.asarray(candidates, dtype=np.float32)
    values = np.asarray(q_values, dtype=np.float64)
    if chunks.ndim != 3 or chunks.shape[0] < 1:
        raise ValueError("candidates must have shape [N, H, A] with N >= 1")
    if values.shape != (chunks.shape[0],):
        raise ValueError("q_values must have shape [N]")
    if not np.isfinite(chunks).all() or not np.isfinite(values).all():
        raise ValueError("candidates and q_values must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if n_elites < 0 or n_elites > chunks.shape[0]:
        raise ValueError("n_elites must be 0 or between 1 and N")

    selected = np.arange(chunks.shape[0])
    if n_elites:
        selected = np.argpartition(values, -n_elites)[-n_elites:]
    logits = values[selected] / temperature
    logits -= logits.max()
    elite_weights = np.exp(logits)
    elite_weights /= elite_weights.sum()

    weights = np.zeros(chunks.shape[0], dtype=np.float64)
    weights[selected] = elite_weights
    weighted = np.tensordot(weights.astype(np.float32), chunks, axes=(0, 0))
    return weighted.astype(np.float32, copy=False), weights.astype(np.float32)


def validate_lingbot_candidates(
    candidates: np.ndarray, *, horizon: int = 50, action_dim: int = 14
) -> np.ndarray:
    """Validate the adapter boundary before a candidate reaches Q scoring."""
    chunks = np.asarray(candidates, dtype=np.float32)
    expected_tail = (horizon, action_dim)
    if chunks.ndim != 3 or chunks.shape[1:] != expected_tail:
        raise ValueError(f"expected [N, {horizon}, {action_dim}], got {chunks.shape}")
    if chunks.shape[0] < 1 or not np.isfinite(chunks).all():
        raise ValueError("candidate batch must be non-empty and finite")
    return chunks
