#!/usr/bin/env python3
"""Create deterministic, task-balanced episode splits for a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-per-task", type=int, default=85)
    parser.add_argument("--val-per-task", type=int, default=10)
    parser.add_argument("--test-per-task", type=int, default=5)
    return parser.parse_args()


def normalize_task(value) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one task per episode, got: {value}")
        return str(value[0])
    return str(value)


def main() -> None:
    args = parse_args()
    episode_files = sorted((args.dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError("No LeRobot episode metadata parquet files found")

    table = pa.concat_tables([pq.read_table(path) for path in episode_files])
    rows = table.select(["episode_index", "tasks"]).to_pylist()
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_task[normalize_task(row["tasks"])].append(int(row["episode_index"]))

    required = args.train_per_task + args.val_per_task + args.test_per_task
    rng = random.Random(args.seed)
    per_task = {}
    global_splits = {"train": [], "val": [], "test": []}
    for task in sorted(by_task):
        episode_ids = sorted(by_task[task])
        if len(episode_ids) != required:
            raise ValueError(f"Task {task!r} has {len(episode_ids)} episodes, expected {required}")
        rng.shuffle(episode_ids)
        train_end = args.train_per_task
        val_end = train_end + args.val_per_task
        task_splits = {
            "train": sorted(episode_ids[:train_end]),
            "val": sorted(episode_ids[train_end:val_end]),
            "test": sorted(episode_ids[val_end:]),
        }
        per_task[task] = task_splits
        for split, ids in task_splits.items():
            global_splits[split].extend(ids)

    for split in global_splits:
        global_splits[split].sort()
    all_ids = [episode for ids in global_splits.values() for episode in ids]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Split overlap detected")
    expected_ids = sorted(int(row["episode_index"]) for row in rows)
    if sorted(all_ids) != expected_ids:
        raise AssertionError("Splits do not cover all episodes exactly once")

    payload = {
        "format_version": 1,
        "dataset_root": str(args.dataset_root),
        "strategy": "task_balanced_episode_level_random",
        "seed": args.seed,
        "counts_per_task": {
            "train": args.train_per_task,
            "val": args.val_per_task,
            "test": args.test_per_task,
        },
        "total_counts": {split: len(ids) for split, ids in global_splits.items()},
        "splits": global_splits,
        "per_task": per_task,
    }

    output_dir = args.dataset_root / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"episode_splits_seed{args.seed}.json"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    manifest_path.write_text(serialized, encoding="utf-8")
    for split, ids in global_splits.items():
        (output_dir / f"{split}_episodes.txt").write_text(
            "\n".join(str(episode) for episode in ids) + "\n", encoding="utf-8"
        )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    (output_dir / "SHA256SUMS").write_text(f"{digest}  {manifest_path.name}\n", encoding="utf-8")

    print(json.dumps({
        "manifest": str(manifest_path),
        "sha256": digest,
        "tasks": len(per_task),
        "episodes": len(all_ids),
        "total_counts": payload["total_counts"],
        "per_task_counts": payload["counts_per_task"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
