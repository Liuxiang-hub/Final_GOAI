from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from scripts.deploy.real_time_chunking import (
    RTCConfig,
    RTCPlanExhausted,
    RealTimeChunkingController,
    build_rtc_guidance,
    latency_to_steps,
    soft_mask_weights,
)


def _response(offset: float, horizon: int = 10) -> dict:
    values = np.arange(horizon, dtype=np.float32)[:, None] + offset
    return {
        "left_arm.position": values.copy(),
        "_normalized_actions": values.copy(),
        "server_timing": {"infer_ms": 300.0},
    }


class TestRTCUtilities(unittest.TestCase):
    def test_latency_rounds_up_with_margin(self):
        self.assertEqual(latency_to_steps(300.0, 25.0, margin_steps=1), 9)

    def test_soft_mask_has_hard_prefix_decay_and_zero_tail(self):
        weights = soft_mask_weights(horizon=50, start_steps=15, delay_steps=8)
        np.testing.assert_array_equal(weights[:8], np.ones(8, dtype=np.float32))
        self.assertTrue(np.all(np.diff(weights[8:35]) < 0))
        np.testing.assert_array_equal(weights[35:], np.zeros(15, dtype=np.float32))

    def test_guidance_right_pads_remaining_plan(self):
        chunk = np.arange(20, dtype=np.float32).reshape(10, 2)
        guidance = build_rtc_guidance(chunk, start_steps=3, delay_steps=2, beta=5.0)
        np.testing.assert_array_equal(guidance["actions"][:7], chunk[3:])
        np.testing.assert_array_equal(guidance["actions"][7:], np.zeros((3, 2)))
        self.assertEqual(guidance["beta"], 5.0)


class TestRTCController(unittest.TestCase):
    def test_execution_continues_while_inference_is_blocked(self):
        inference_started = threading.Event()
        release_inference = threading.Event()
        requests = []

        def infer(request):
            requests.append(request)
            if len(requests) == 1:
                self.assertTrue(request["_rtc_return_normalized"])
                return _response(0)
            inference_started.set()
            if not release_inference.wait(timeout=2.0):
                raise TimeoutError("test inference was not released")
            return _response(100)

        controller = RealTimeChunkingController(
            infer,
            RTCConfig(
                prediction_horizon=10,
                minimum_execution_steps=3,
                initial_delay_steps=2,
                delay_buffer_size=4,
            ),
        )
        controller.start({"state": np.zeros(1, dtype=np.float32)})
        try:
            for expected in range(3):
                action = controller.next_action()
                self.assertEqual(float(action["left_arm.position"][0]), expected)
                controller.commit({"state": np.array([expected], dtype=np.float32)})
            self.assertTrue(inference_started.wait(timeout=1.0))

            # Two more old-plan actions remain available during GPU inference.
            for expected in (3, 4):
                action = controller.next_action()
                self.assertEqual(float(action["left_arm.position"][0]), expected)
                controller.commit({"state": np.array([expected], dtype=np.float32)})

            release_inference.set()
            deadline = time.monotonic() + 1.0
            while controller.cursor != 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(controller.cursor, 2)
            self.assertEqual(controller.delay_history[-1], 2)
            self.assertEqual(requests[1]["_rtc"]["start_steps"], 3)
            self.assertEqual(requests[1]["_rtc"]["delay_steps"], 2)

            action = controller.next_action()
            self.assertEqual(float(action["left_arm.position"][0]), 102.0)
        finally:
            release_inference.set()
            controller.stop()

    def test_exhausted_plan_requires_stop_instead_of_repeating_delta(self):
        inference_started = threading.Event()
        release_inference = threading.Event()
        calls = 0

        def infer(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _response(0)
            inference_started.set()
            release_inference.wait(timeout=2.0)
            return _response(100)

        controller = RealTimeChunkingController(
            infer,
            RTCConfig(
                prediction_horizon=10,
                minimum_execution_steps=3,
                initial_delay_steps=2,
            ),
        )
        controller.start({"state": np.zeros(1, dtype=np.float32)})
        try:
            for index in range(3):
                controller.next_action()
                controller.commit({"state": np.array([index], dtype=np.float32)})
            self.assertTrue(inference_started.wait(timeout=1.0))
            for index in range(3, 10):
                controller.next_action()
                controller.commit({"state": np.array([index], dtype=np.float32)})
            with self.assertRaises(RTCPlanExhausted):
                controller.next_action()
        finally:
            release_inference.set()
            controller.stop()


if __name__ == "__main__":
    unittest.main()
