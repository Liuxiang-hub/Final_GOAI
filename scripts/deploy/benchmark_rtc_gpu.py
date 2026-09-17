#!/usr/bin/env python3
"""Benchmark LingBot-VLA 2.0 plain and RTC-guided inference on one real frame."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


def timed_infer(policy, observation: dict) -> tuple[dict, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = policy.infer(dict(observation))
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--final-goai-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--norm-path", type=Path, required=True)
    parser.add_argument("--robo-name", default="goai_piper_x")
    parser.add_argument("--episode", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.lingbot_root,
        args.final_goai_root,
        args.model_path,
        args.qwen_path,
        args.data_path,
        args.norm_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.denoising_steps < 1:
        raise ValueError("denoising steps must be positive")

    os.environ["QWEN3VL_PATH"] = str(args.qwen_path)
    sys.path.insert(0, str(args.final_goai_root))
    sys.path.insert(0, str(args.lingbot_root))
    os.chdir(args.lingbot_root)

    from scripts.deploy.real_time_chunking import build_rtc_guidance
    from scripts.open_loop_eval import (
        LEROBOT_DATASET_API,
        LeRobotDataset,
        LeRobotDatasetMetadata,
        load_policy_server,
        prepare_eval_observation,
    )

    policy_class = load_policy_server("qwen3vl", str(args.model_path))
    policy = policy_class(
        path_to_pi_model=str(args.model_path),
        robot_norm_path=str(args.norm_path),
        use_length=50,
        use_bf16=True,
        use_fp32=False,
        chunk_ret=True,
        use_compile=False,
    )
    policy.vla.model.config.num_steps = args.denoising_steps
    policy.reset(args.robo_name)

    policy.data_config.num_episode = None
    policy.data_config.chunk_size = policy.config.chunk_size
    policy.data_config.train_path = str(args.data_path)
    policy.data_config.data_name = args.robo_name
    metadata = LeRobotDatasetMetadata(args.data_path.name, root=args.data_path)
    delta_timestamps = {
        name: [index / metadata.fps for index in range(policy.config.chunk_size)]
        for name in policy.vla.feature_transform.org_features["actions"]
    }
    dataset = LeRobotDataset(
        args.data_path.name,
        root=args.data_path,
        delta_timestamps=delta_timestamps,
    )
    if LEROBOT_DATASET_API == "v2":
        frame_index = int(dataset.episode_data_index["from"][args.episode])
    else:
        frame_index = int(
            dataset.meta.episodes[args.episode]["dataset_from_index"]
        )
    observation, _ = prepare_eval_observation(policy, dataset[frame_index])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    reference, cold_plain_ms = timed_infer(
        policy, {**observation, "_rtc_return_normalized": True}
    )
    reference_chunk = np.asarray(reference["_normalized_actions"], dtype=np.float32)
    if reference_chunk.shape[0] != 50 or not np.isfinite(reference_chunk).all():
        raise RuntimeError(f"invalid normalized reference chunk {reference_chunk.shape}")
    guidance = build_rtc_guidance(
        reference_chunk,
        start_steps=15,
        delay_steps=10,
        beta=5.0,
    )

    _, cold_rtc_ms = timed_infer(
        policy,
        {
            **observation,
            "_rtc": guidance,
            "_rtc_return_normalized": True,
        },
    )

    plain_latencies: list[float] = []
    rtc_latencies: list[float] = []
    plain_prefix_mse: list[float] = []
    rtc_prefix_mse: list[float] = []
    active = np.asarray(guidance["weights"]) > 0
    target = np.asarray(guidance["actions"], dtype=np.float32)

    for _ in range(args.iterations):
        plain, elapsed_ms = timed_infer(
            policy, {**observation, "_rtc_return_normalized": True}
        )
        plain_chunk = np.asarray(plain["_normalized_actions"], dtype=np.float32)
        if not np.isfinite(plain_chunk).all():
            raise RuntimeError("plain inference returned NaN or Inf")
        plain_latencies.append(elapsed_ms)
        plain_prefix_mse.append(float(np.mean((plain_chunk[active] - target[active]) ** 2)))

        guided, elapsed_ms = timed_infer(
            policy,
            {
                **observation,
                "_rtc": guidance,
                "_rtc_return_normalized": True,
            },
        )
        guided_chunk = np.asarray(guided["_normalized_actions"], dtype=np.float32)
        if not np.isfinite(guided_chunk).all():
            raise RuntimeError("RTC inference returned NaN or Inf")
        rtc_latencies.append(elapsed_ms)
        rtc_prefix_mse.append(float(np.mean((guided_chunk[active] - target[active]) ** 2)))

    total_memory = torch.cuda.get_device_properties(0).total_memory
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "denoising_steps": args.denoising_steps,
        "episode": args.episode,
        "frame_index": frame_index,
        "dataset_fps": float(metadata.fps),
        "prediction_horizon": 50,
        "minimum_execution_steps": 15,
        "rtc_delay_steps": 10,
        "rtc_beta": 5.0,
        "cold_plain_ms": cold_plain_ms,
        "cold_rtc_ms": cold_rtc_ms,
        "plain": percentile_summary(plain_latencies),
        "rtc": percentile_summary(rtc_latencies),
        "plain_active_prefix_mse_mean": float(np.mean(plain_prefix_mse)),
        "rtc_active_prefix_mse_mean": float(np.mean(rtc_prefix_mse)),
        "rtc_to_plain_prefix_mse_ratio": float(
            np.mean(rtc_prefix_mse) / max(np.mean(plain_prefix_mse), 1e-12)
        ),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "total_vram_gib": total_memory / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
