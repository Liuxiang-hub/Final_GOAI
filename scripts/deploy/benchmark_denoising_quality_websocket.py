#!/usr/bin/env python3
"""Measure one action chunk per GOAI validation task through WebSocket."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--denoising-steps", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.lingbot_root))
    from deploy.websocket_client_policy import WebsocketClientPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(args.data_path.name, root=args.data_path)
    dataset = LeRobotDataset(
        args.data_path.name,
        root=args.data_path,
        delta_timestamps={"action": [i / metadata.fps for i in range(50)]},
    )
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    client.reset("goai_piper_x")
    rows = []
    for episode in args.episodes:
        frame_index = int(dataset.meta.episodes[episode]["dataset_from_index"])
        raw = dict(dataset[frame_index])
        observation = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in raw.items()
        }
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            observation[key] = (
                raw[key].to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
            )

        started = time.perf_counter()
        response = client.infer(observation)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if "action" in response:
            predicted = np.asarray(response["action"], dtype=np.float32)
        else:
            predicted = np.concatenate(
                [
                    np.asarray(response["action.arm.position"], dtype=np.float32),
                    np.asarray(response["action.effector.position"], dtype=np.float32),
                ],
                axis=-1,
            )
        raw_action = raw["action"].float().cpu().numpy()
        ground_truth = raw_action if "action" in response else np.concatenate(
            [raw_action[:, :6], raw_action[:, 7:13], raw_action[:, 6:7], raw_action[:, 13:14]],
            axis=-1,
        )
        rows.append(
            {
                "task": str(raw["task"]),
                "episode": episode,
                "frame_index": frame_index,
                "latency_ms": latency_ms,
                "mse": float(np.mean((predicted - ground_truth) ** 2)),
                "mae": float(np.mean(np.abs(predicted - ground_truth))),
                "finite": bool(np.isfinite(predicted).all()),
            }
        )

    result = {
        "denoising_steps": args.denoising_steps,
        "dataset_fps": float(metadata.fps),
        "action_horizon": 50,
        "tasks": rows,
        "mean_mse": float(np.mean([row["mse"] for row in rows])),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_latency_ms": float(np.mean([row["latency_ms"] for row in rows])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
