#!/usr/bin/env python3
"""Exercise the RTC WebSocket server with a real GOAI validation frame."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


def _to_transport(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--final-goai-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--episode", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.final_goai_root))
    sys.path.insert(0, str(args.lingbot_root))

    from deploy.websocket_client_policy import WebsocketClientPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from scripts.deploy.real_time_chunking import build_rtc_guidance

    dataset = LeRobotDataset(args.data_path.name, root=args.data_path)
    frame_index = int(dataset.meta.episodes[args.episode]["dataset_from_index"])
    raw = dict(dataset[frame_index])
    observation = {key: _to_transport(value) for key, value in raw.items()}
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        observation[key] = (
            raw[key].to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        )

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    client.reset("goai_piper_x")

    start = time.perf_counter()
    first = client.infer({**observation, "_rtc_return_normalized": True})
    first_ms = (time.perf_counter() - start) * 1000.0
    previous = np.asarray(first["_normalized_actions"], dtype=np.float32)
    if previous.ndim != 2 or previous.shape[0] != 50 or not np.isfinite(previous).all():
        raise RuntimeError(f"invalid initial normalized chunk: {previous.shape}")

    delays_ms = [0, 100, 200, 300, 400]
    measurements: dict[str, dict[str, float]] = {}
    for artificial_delay_ms in delays_ms:
        latencies: list[float] = []
        constraint_errors: list[float] = []
        for _ in range(args.iterations):
            delay_steps = int(round(artificial_delay_ms / 40.0))
            guidance = build_rtc_guidance(
                previous,
                start_steps=15,
                delay_steps=delay_steps,
                beta=5.0,
            )
            started = time.perf_counter()
            if artificial_delay_ms:
                time.sleep(artificial_delay_ms / 1000.0)
            response = client.infer(
                {
                    **observation,
                    "_rtc": guidance,
                    "_rtc_return_normalized": True,
                }
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            current = np.asarray(response["_normalized_actions"], dtype=np.float32)
            if current.shape != previous.shape or not np.isfinite(current).all():
                raise RuntimeError(f"invalid RTC normalized chunk: {current.shape}")
            active = np.asarray(guidance["weights"]) > 0
            target = np.asarray(guidance["actions"], dtype=np.float32)
            constraint_errors.append(float(np.mean((current[active] - target[active]) ** 2)))
            latencies.append(elapsed_ms)
            previous = current
        summary = _percentiles(latencies)
        summary["mean_constraint_mse"] = float(np.mean(constraint_errors))
        summary["within_1400ms"] = bool(summary["max_ms"] < 1400.0)
        measurements[str(artificial_delay_ms)] = summary

    result = {
        "transport": "WebSocket msgpack",
        "server": f"{args.host}:{args.port}",
        "episode": args.episode,
        "frame_index": frame_index,
        "initial_round_trip_ms": first_ms,
        "prediction_horizon": 50,
        "normalized_action_dimension": int(previous.shape[1]),
        "execution_steps": 15,
        "control_period_ms": 40,
        "available_buffer_ms": 1400,
        "delay_injection_includes_sleep": True,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
