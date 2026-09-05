"""Create and validate episode-level replay metadata for Q-only training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = {
    "fill_pen_holder",
    "insert_charger",
    "put_objects_into_basket",
    "stack_and_cover_blocks",
    "stack_bowls",
    "stand_up_bottles",
}


def validate_episode(record: dict[str, Any]) -> None:
    required = {"episode_id", "task", "success", "frames", "source", "data_path"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if record["task"] not in TASKS:
        raise ValueError(f"unknown task: {record['task']}")
    if not isinstance(record["success"], bool):
        raise ValueError("success must be a boolean terminal label")
    if not isinstance(record["frames"], int) or record["frames"] <= 0:
        raise ValueError("frames must be a positive integer")
    if record["source"] not in {"demonstration", "baseline_rollout", "qplanning_rollout"}:
        raise ValueError("unsupported source")
    if not str(record["episode_id"]).strip() or not str(record["data_path"]).strip():
        raise ValueError("episode_id and data_path cannot be empty")


def validate_manifest(path: Path) -> int:
    count = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_episode(record)
            episode_id = str(record["episode_id"])
            if episode_id in seen:
                raise ValueError(f"line {line_number}: duplicate episode_id {episode_id}")
            seen.add(episode_id)
            count += 1
    if count == 0:
        raise ValueError("manifest contains no episodes")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(f"valid episodes: {validate_manifest(args.manifest)}")


if __name__ == "__main__":
    main()
