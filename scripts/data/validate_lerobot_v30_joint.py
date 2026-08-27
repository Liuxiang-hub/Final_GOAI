#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/goai2026_real_piper_joint")
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, video_backend="pyav")
    if args.expected_episodes is not None and dataset.num_episodes != args.expected_episodes:
        raise AssertionError((dataset.num_episodes, args.expected_episodes))
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    checked = []
    for index in indices:
        item = dataset[index]
        assert tuple(item["observation.state"].shape) == (14,)
        assert tuple(item["action"].shape) == (14,)
        for key in dataset.meta.video_keys:
            assert tuple(item[key].shape) == (3, 480, 640), (key, item[key].shape)
            assert np.isfinite(item[key].numpy()).all()
        checked.append({"index": index, "task": item["task"]})

    # Official format semantics: action[t] equals state[t+1], except clamped final frame.
    hf = dataset.hf_dataset
    pair_count = min(10_000, max(1, len(dataset) - 1))
    left_indices = np.unique(np.linspace(0, len(dataset) - 2, pair_count, dtype=np.int64))
    right_indices = left_indices + 1
    left = hf[left_indices.tolist()]
    right = hf[right_indices.tolist()]
    action = np.asarray(left["action"], dtype=np.float32)
    next_state = np.asarray(right["observation.state"], dtype=np.float32)
    left_episode = np.asarray(left["episode_index"])
    right_episode = np.asarray(right["episode_index"])
    valid = left_episode == right_episode
    max_next_state_error = float(np.max(np.abs(action[valid] - next_state[valid])))
    if max_next_state_error > 1e-6:
        raise AssertionError(f"action/next-state mismatch: {max_next_state_error}")

    print(json.dumps({
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "tasks": dataset.meta.total_tasks,
        "fps": dataset.fps,
        "video_keys": dataset.meta.video_keys,
        "checked": checked,
        "max_next_state_error": max_next_state_error,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
