"""CPU-only static preflight for the optional GOAI Q-Planning path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/qplanning/offline_prepare.yaml"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    # JSON is a strict subset of YAML. Keeping the checked-in YAML in JSON
    # syntax makes this safety preflight dependency-free on a fresh robot PC.
    config = json.loads(config_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    bc = config["bc_policy"]
    q = config["q_function"]
    if bc["checkpoint"] != "global_step_8884" or not bc["frozen"]:
        errors.append("BC must remain frozen global_step_8884")
    if bc["prediction_horizon"] != q["horizon"] or bc["action_dim"] != q["action_dim"]:
        errors.append("BC and Q horizon/action_dim differ")
    for key in ("norm_stats",):
        if not (repo / bc[key]).is_file():
            errors.append(f"missing {key}: {bc[key]}")
    for key in ("train_split", "validation_split"):
        if not (repo / q[key]).is_file():
            errors.append(f"missing {key}: {q[key]}")
    stats = json.loads((repo / bc["norm_stats"]).read_text(encoding="utf-8"))
    if not isinstance(stats, dict) or not stats:
        errors.append("normalization statistics are empty")
    if config["deployment"]["execute_steps"] != 15:
        errors.append("validated deployment cadence must remain 15 steps")
    postprocessor = repo / config["deployment"]["postprocessor"]
    if not postprocessor.is_file():
        errors.append(f"missing postprocessor config: {postprocessor}")

    print(f"config: {config_path}")
    print(f"baseline: {bc['checkpoint']} (frozen={bc['frozen']})")
    print(f"candidate contract: [N, {bc['prediction_horizon']}, {bc['action_dim']}]")
    print(f"planner enabled: {config['enabled']}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("PASS: static Q-Planning preparation is internally consistent")


if __name__ == "__main__":
    main()
