"""Causal temporal ensembling and adaptive smoothing for VLA action chunks."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import numpy as np


class _Plan:
    def __init__(self, start_step: int, values: dict[str, np.ndarray]) -> None:
        self.start_step = start_step
        self.values = values


class TemporalEnsembleFilter:
    """Fuse timestamp-aligned action chunks, then apply continuous adaptive EMA.

    Every call adds one observation-conditioned action chunk starting at
    ``start_step`` and returns the next ``execute_steps`` actions. Predictions
    from older chunks are used only when they target the exact same absolute
    control step. No deadband or hard hold is used.
    """

    def __init__(
        self,
        execute_steps: int = 15,
        history_chunks: int = 4,
        ensemble_decay: float = 1.0,
        slow_alpha: float = 0.2,
        fast_alpha: float = 0.8,
        arm_transition=(0.003, 0.030),
        gripper_transition=(0.010, 0.080),
        robust_arm_scale: float | None = None,
        robust_gripper_scale: float | None = None,
        consensus_arm_scale: float | None = None,
        consensus_gripper_scale: float | None = None,
        oscillation_window_steps: int | None = None,
        oscillation_min_sign_changes: int = 3,
        oscillation_arm_delta: float = 0.006,
        oscillation_gripper_delta: float = 0.020,
        oscillation_alpha: float = 0.10,
    ) -> None:
        if execute_steps < 1 or history_chunks < 1:
            raise ValueError("execute_steps and history_chunks must be positive")
        if ensemble_decay < 0:
            raise ValueError("ensemble_decay must be non-negative")
        if not 0 < slow_alpha <= fast_alpha <= 1:
            raise ValueError("require 0 < slow_alpha <= fast_alpha <= 1")
        self.execute_steps = execute_steps
        self.history_chunks = history_chunks
        self.ensemble_decay = ensemble_decay
        self.slow_alpha = slow_alpha
        self.fast_alpha = fast_alpha
        self.arm_transition = tuple(map(float, arm_transition))
        self.gripper_transition = tuple(map(float, gripper_transition))
        self.robust_arm_scale = robust_arm_scale
        self.robust_gripper_scale = robust_gripper_scale
        self.consensus_arm_scale = consensus_arm_scale
        self.consensus_gripper_scale = consensus_gripper_scale
        if oscillation_window_steps is not None and oscillation_window_steps < 3:
            raise ValueError("oscillation_window_steps must be at least 3")
        if not 0 < oscillation_alpha <= 1:
            raise ValueError("oscillation_alpha must be in (0, 1]")
        self.oscillation_window_steps = oscillation_window_steps
        self.oscillation_min_sign_changes = int(oscillation_min_sign_changes)
        self.oscillation_arm_delta = float(oscillation_arm_delta)
        self.oscillation_gripper_delta = float(oscillation_gripper_delta)
        self.oscillation_alpha = float(oscillation_alpha)
        self._plans: deque[_Plan] = deque(maxlen=history_chunks)
        self._ema_state: dict[str, np.ndarray] = {}
        self._target_history: dict[str, deque[np.ndarray]] = {}

    def reset(self) -> None:
        self._plans.clear()
        self._ema_state.clear()
        self._target_history.clear()

    @staticmethod
    def _feature_kind(key: str, width: int) -> str:
        if key.endswith("effector.position") or width == 2:
            return "gripper"
        return "arm"

    @staticmethod
    def _smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _dimension_scales(self, key: str, width: int, arm_value, gripper_value) -> np.ndarray:
        values = np.full(width, arm_value, dtype=np.float32)
        if key == "actions" and width == 14:
            values[[6, 13]] = gripper_value
        elif self._feature_kind(key, width) == "gripper":
            values.fill(gripper_value)
        return values

    def _adaptive_alpha(self, key: str, delta: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
        if key == "actions" and delta.shape[-1] == 14:
            arm_low, arm_high = self.arm_transition
            grip_low, grip_high = self.gripper_transition
            low = np.full(delta.shape[-1], arm_low, dtype=np.float32)
            high = np.full(delta.shape[-1], arm_high, dtype=np.float32)
            low[[6, 13]] = grip_low
            high[[6, 13]] = grip_high
            amount = self._smoothstep((np.abs(delta) - low) / (high - low))
            alpha = self.slow_alpha + (self.fast_alpha - self.slow_alpha) * amount
            return alpha if confidence is None else self.slow_alpha + (alpha - self.slow_alpha) * confidence
        kind = self._feature_kind(key, delta.shape[-1])
        low, high = self.gripper_transition if kind == "gripper" else self.arm_transition
        amount = self._smoothstep((np.abs(delta) - low) / (high - low))
        alpha = self.slow_alpha + (self.fast_alpha - self.slow_alpha) * amount
        return alpha if confidence is None else self.slow_alpha + (alpha - self.slow_alpha) * confidence

    def _ensemble_at(self, key: str, absolute_step: int) -> tuple[np.ndarray, np.ndarray]:
        predictions: list[np.ndarray] = []
        ages: list[int] = []
        newest_index = len(self._plans) - 1
        for index, plan in enumerate(self._plans):
            offset = absolute_step - plan.start_step
            values = plan.values[key]
            if 0 <= offset < values.shape[0]:
                predictions.append(values[offset])
                ages.append(newest_index - index)
        if not predictions:
            raise RuntimeError(f"no prediction covers absolute step {absolute_step}")
        weights = np.exp(-self.ensemble_decay * np.asarray(ages, dtype=np.float32))
        stacked = np.stack(predictions, axis=0)
        expanded = weights.reshape((-1,) + (1,) * (stacked.ndim - 1))
        median = np.median(stacked, axis=0)
        if self.robust_arm_scale is not None and len(predictions) > 1:
            scales = self._dimension_scales(
                key, stacked.shape[-1], self.robust_arm_scale,
                self.robust_gripper_scale or self.robust_arm_scale,
            )
            residual = np.abs(stacked - median)
            robust = np.minimum(1.0, scales / np.maximum(residual, 1e-8))
            expanded = expanded * robust
        fused = np.sum(stacked * expanded, axis=0) / np.sum(expanded, axis=0)
        confidence = np.ones_like(fused, dtype=np.float32)
        if self.consensus_arm_scale is not None and len(predictions) > 1:
            scales = self._dimension_scales(
                key, stacked.shape[-1], self.consensus_arm_scale,
                self.consensus_gripper_scale or self.consensus_arm_scale,
            )
            # Mean absolute deviation still detects one dissenting plan among
            # four; median absolute deviation would be exactly zero there.
            spread = np.mean(np.abs(stacked - median), axis=0)
            confidence = 1.0 / (1.0 + np.square(spread / scales))
        return fused, confidence

    def _oscillation_mask(self, key: str, target: np.ndarray) -> np.ndarray:
        """Detect short, low-amplitude reversals without blocking coherent drift."""
        if self.oscillation_window_steps is None:
            return np.zeros_like(target, dtype=bool)
        history = self._target_history.setdefault(
            key, deque(maxlen=self.oscillation_window_steps)
        )
        history.append(target.copy())
        if len(history) < self.oscillation_window_steps:
            return np.zeros_like(target, dtype=bool)
        samples = np.stack(history, axis=0)
        deltas = np.diff(samples, axis=0)
        signs = np.sign(deltas)
        sign_changes = np.sum((signs[1:] * signs[:-1]) < 0, axis=0)
        thresholds = self._dimension_scales(
            key, target.shape[-1], self.oscillation_arm_delta,
            self.oscillation_gripper_delta,
        )
        small_steps = np.max(np.abs(deltas), axis=0) <= thresholds
        low_net_motion = np.abs(samples[-1] - samples[0]) <= thresholds
        return (sign_changes >= self.oscillation_min_sign_changes) & small_steps & low_net_motion

    def process(self, new_chunk: Mapping[str, np.ndarray], start_step: int) -> dict[str, np.ndarray]:
        values = {key: np.asarray(value, dtype=np.float32).copy() for key, value in new_chunk.items()}
        if not values:
            raise ValueError("new_chunk must not be empty")
        for key, value in values.items():
            if value.ndim < 2 or value.shape[0] < self.execute_steps:
                raise ValueError(f"{key} must contain at least {self.execute_steps} time steps")
        self._plans.append(_Plan(int(start_step), values))

        output: dict[str, np.ndarray] = {}
        for key in values:
            pairs = [self._ensemble_at(key, start_step + offset) for offset in range(self.execute_steps)]
            fused = np.stack([pair[0] for pair in pairs], axis=0)
            confidence = np.stack([pair[1] for pair in pairs], axis=0)
            previous = self._ema_state.get(key)
            for step in range(fused.shape[0]):
                if previous is None:
                    previous = fused[step].copy()
                else:
                    delta = fused[step] - previous
                    alpha = self._adaptive_alpha(key, delta, confidence[step])
                    oscillating = self._oscillation_mask(key, fused[step])
                    alpha = np.where(oscillating, np.minimum(alpha, self.oscillation_alpha), alpha)
                    previous = alpha * fused[step] + (1.0 - alpha) * previous
                    fused[step] = previous
            self._ema_state[key] = previous.copy()
            output[key] = fused
        return output
