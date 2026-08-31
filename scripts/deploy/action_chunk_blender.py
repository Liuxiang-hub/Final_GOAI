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

    def __init__(
        self,
        execute_steps: int = 15,
        blend_steps: int = 5,
        ema_alpha: float = 0.8,
        enable_deadband: bool = False,
        arm_deadband=(0.004, 0.004, 0.004, 0.006, 0.006, 0.006) * 2,
        arm_max_step=(0.020, 0.020, 0.020, 0.025, 0.025, 0.025) * 2,
        gripper_deadband: float = 0.015,
        gripper_max_step: float = 0.08,
        gripper_confirm_frames: int = 2,
    ):
        if execute_steps < 1 or blend_steps < 0:
            raise ValueError("execute_steps must be >= 1 and blend_steps >= 0")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if gripper_confirm_frames < 1:
            raise ValueError("gripper_confirm_frames must be >= 1")
        self.execute_steps = execute_steps
        self.blend_steps = blend_steps
        self.ema_alpha = ema_alpha
        self.enable_deadband = enable_deadband
        self.arm_deadband = np.asarray(arm_deadband, dtype=np.float32)
        self.arm_max_step = np.asarray(arm_max_step, dtype=np.float32)
        if self.arm_deadband.shape != (12,) or self.arm_max_step.shape != (12,):
            raise ValueError("arm_deadband and arm_max_step must each have 12 values")
        self.gripper_deadband = float(gripper_deadband)
        self.gripper_max_step = float(gripper_max_step)
        self.gripper_confirm_frames = gripper_confirm_frames
        self._old_tail: dict[str, np.ndarray] | None = None
        self._ema_state: dict[str, np.ndarray] | None = None
        self._last_sent: dict[str, np.ndarray] = {}
        self._gripper_count: dict[str, np.ndarray] = {}
        self._gripper_direction: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._old_tail = None
        self._ema_state = None
        self._last_sent = {}
        self._gripper_count = {}
        self._gripper_direction = {}

    @staticmethod
    def _feature_type(key: str, width: int) -> str | None:
        if key.endswith("arm.position") or (key == "actions" and width == 12):
            return "arm"
        if key.endswith("effector.position") or (key == "actions" and width == 2):
            return "gripper"
        return None

    def _apply_accumulated_deadband(self, key: str, values: np.ndarray) -> None:
        if not self.enable_deadband:
            return
        feature_type = self._feature_type(key, values.shape[-1])
        if feature_type is None:
            return

        if key not in self._last_sent:
            self._last_sent[key] = values[0].copy()
            if feature_type == "gripper":
                self._gripper_count[key] = np.zeros(values.shape[-1], dtype=np.int32)
                self._gripper_direction[key] = np.zeros(values.shape[-1], dtype=np.int8)

        for step in range(values.shape[0]):
            target = values[step]
            previous = self._last_sent[key]
            delta = target - previous

            if feature_type == "arm":
                update = np.abs(delta) >= self.arm_deadband
                limited = np.clip(delta, -self.arm_max_step, self.arm_max_step)
            else:
                direction = np.sign(delta).astype(np.int8)
                above = np.abs(delta) >= self.gripper_deadband
                same_direction = direction == self._gripper_direction[key]
                self._gripper_count[key] = np.where(
                    above,
                    np.where(same_direction, self._gripper_count[key] + 1, 1),
                    0,
                )
                self._gripper_direction[key] = np.where(above, direction, 0)
                update = above & (self._gripper_count[key] >= self.gripper_confirm_frames)
                limited = np.clip(delta, -self.gripper_max_step, self.gripper_max_step)

            sent = previous.copy()
            sent[update] = previous[update] + limited[update]
            values[step] = sent
            self._last_sent[key] = sent

            if feature_type == "gripper":
                # A confirmed update starts a fresh confirmation window for any remainder.
                self._gripper_count[key][update] = 0

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

        # Light causal smoothing across both individual steps and chunk boundaries:
        # y[t] = alpha * x[t] + (1 - alpha) * y[t-1].  alpha=1 disables EMA.
        for key, value in output.items():
            previous = None if self._ema_state is None else self._ema_state.get(key)
            for step in range(value.shape[0]):
                if previous is None:
                    previous = value[step].copy()
                else:
                    previous = self.ema_alpha * value[step] + (1.0 - self.ema_alpha) * previous
                    value[step] = previous
            if self._ema_state is None:
                self._ema_state = {}
            self._ema_state[key] = previous.copy()
            self._apply_accumulated_deadband(key, value)
        return output


# Preserve the validated fixed-EMA implementation as an explicit rollback path.
LegacyActionChunkBlender = ActionChunkBlender


class ActionChunkBlender:
    """Formal GOAI entry: timestamp-aligned temporal ensemble + adaptive EMA.

    The public client API remains ``process(full_50_step_chunk)``; absolute
    chunk start indices are maintained internally. No deadband is applied.
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
        enable_deadband: bool = False,
        **legacy_options,
    ) -> None:
        if enable_deadband:
            raise ValueError("deadband is disabled in the formal deployment")
        try:
            from .temporal_ensemble_filter import TemporalEnsembleFilter
        except ImportError:
            from temporal_ensemble_filter import TemporalEnsembleFilter

        self.execute_steps = execute_steps
        self.legacy_options = legacy_options
        self._next_start_step = 0
        self._processor = TemporalEnsembleFilter(
            execute_steps=execute_steps,
            history_chunks=history_chunks,
            ensemble_decay=ensemble_decay,
            slow_alpha=slow_alpha,
            fast_alpha=fast_alpha,
            arm_transition=arm_transition,
            gripper_transition=gripper_transition,
            robust_arm_scale=robust_arm_scale,
            robust_gripper_scale=robust_gripper_scale,
            consensus_arm_scale=consensus_arm_scale,
            consensus_gripper_scale=consensus_gripper_scale,
            oscillation_window_steps=oscillation_window_steps,
            oscillation_min_sign_changes=oscillation_min_sign_changes,
            oscillation_arm_delta=oscillation_arm_delta,
            oscillation_gripper_delta=oscillation_gripper_delta,
            oscillation_alpha=oscillation_alpha,
        )

    def reset(self) -> None:
        self._next_start_step = 0
        self._processor.reset()

    def process(self, new_chunk: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = self._processor.process(new_chunk, start_step=self._next_start_step)
        self._next_start_step += self.execute_steps
        return output
