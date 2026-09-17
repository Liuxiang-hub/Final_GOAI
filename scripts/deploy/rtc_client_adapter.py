"""Small adapter connecting LingBot's websocket client to RTC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from .real_time_chunking import RTCConfig, RealTimeChunkingController
except ImportError:
    from real_time_chunking import RTCConfig, RealTimeChunkingController


class LingBotRTCClient:
    """Robot-loop adapter; it never sends a motor command by itself."""

    def __init__(self, websocket_policy: Any, config: RTCConfig | None = None):
        self.controller = RealTimeChunkingController(
            websocket_policy.infer,
            config or RTCConfig(),
        )

    def start(self, initial_observation: Mapping[str, Any]) -> None:
        self.controller.start(initial_observation)

    def next_action(self) -> dict:
        """Return one command for the robot SDK; does not execute it."""
        return self.controller.next_action()

    def commit(self, observation_after_action: Mapping[str, Any]) -> None:
        """Call only after the SDK confirms the previous command was consumed."""
        self.controller.commit(observation_after_action)

    def stop(self) -> None:
        self.controller.stop()
