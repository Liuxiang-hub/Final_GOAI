#!/usr/bin/env python3
"""Read-only LingBot server preflight. Never opens CAN or sends robot commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import signal
import sys
import time

import numpy as np


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo(config_path: Path, configured: str) -> Path:
    override = os.environ.get("LINGBOT_VLA_REPO")
    value = Path(override or configured).expanduser()
    if not value.is_absolute():
        value = (config_path.parent / value).resolve()
    return value


def tcp_check(host: str, port: int, timeout: float) -> None:
    with socket.create_connection((host, port), timeout=timeout):
        pass


def synthetic_observation(config: dict, prompt: str) -> dict:
    obs_cfg = config["observation"]
    height, width = obs_cfg["image_height"], obs_cfg["image_width"]
    # Mid-gray is deliberate: it avoids all-black normalization edge cases while
    # remaining a stationary, obviously synthetic observation.
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    observation = {
        key: image.copy() for key in obs_cfg["camera_keys"].values()
    }
    observation[obs_cfg["state_key"]] = np.zeros(obs_cfg["state_dimension"], dtype=np.float32)
    observation[obs_cfg["prompt_key"]] = [prompt] if obs_cfg.get("prompt_as_list", True) else prompt
    return observation


def find_action_chunk(value, path="response"):
    if isinstance(value, dict):
        # Some robot configs return separate arm/gripper arrays. Reassemble them
        # only when they share the same 50-step time axis and total 14 columns.
        parts = []
        for key, child in value.items():
            try:
                array = np.asarray(child)
            except Exception:
                continue
            while array.ndim > 2 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 2 and array.shape[0] == 50:
                parts.append((key, array))
        if parts and sum(array.shape[1] for _, array in parts) == 14:
            parts.sort(key=lambda item: item[0])
            return path + ".{joined_actions}", np.concatenate([array for _, array in parts], axis=1)
        preferred = ("action", "actions", "action_chunk")
        for key in preferred:
            if key in value:
                found = find_action_chunk(value[key], f"{path}.{key}")
                if found:
                    return found
        for key, child in value.items():
            found = find_action_chunk(child, f"{path}.{key}")
            if found:
                return found
    else:
        try:
            array = np.asarray(value)
        except Exception:
            return None
        if array.ndim >= 2 and array.shape[-1] == 14:
            array = array.reshape(-1, 14)
            if array.shape[0] >= 1:
                return path, array
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="GOAI LingBot 服务端静止观测预热")
    parser.add_argument("--config", default=str(Path(__file__).with_name("client_config.json")))
    parser.add_argument("--prompt", default="Fill the pen holder")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if config["safety"].get("hardware_output_enabled", False):
        raise RuntimeError("安全拒绝：预检程序要求 hardware_output_enabled=false")

    server = config["server"]
    if args.host:
        server["host"] = args.host
    if args.port:
        server["port"] = args.port
    if hasattr(signal, "SIGALRM"):
        def deadline_handler(_signum, _frame):
            raise TimeoutError("服务端预检超过总时限，已终止；请检查SSH隧道和推理服务")
        signal.signal(signal.SIGALRM, deadline_handler)
        total_timeout = int(server["connect_timeout_seconds"] + server["inference_timeout_seconds"] + 5)
        signal.alarm(total_timeout)
    print(f"[1/5] TCP检查 {server['host']}:{server['port']}", flush=True)
    tcp_check(server["host"], int(server["port"]), float(server["connect_timeout_seconds"]))

    repo = resolve_repo(config_path, config["lingbot_repo"])
    client_file = repo / "deploy" / "websocket_client_policy.py"
    if not client_file.exists():
        raise FileNotFoundError(
            f"找不到 {client_file}；请修改 client_config.json 的 lingbot_repo，"
            "或设置 LINGBOT_VLA_REPO。"
        )
    sys.path.insert(0, str(repo))
    from deploy.websocket_client_policy import WebsocketClientPolicy

    print("[2/5] 建立WebSocket并读取服务端元数据", flush=True)
    policy = WebsocketClientPolicy(host=server["host"], port=int(server["port"]))
    metadata = policy.get_server_metadata()
    print("SERVER_METADATA=" + json.dumps(metadata, ensure_ascii=False, default=str), flush=True)

    print(f"[3/5] reset({server['robot_name']!r})", flush=True)
    policy.reset(server["robot_name"])

    observation = synthetic_observation(config, args.prompt)
    print("[4/5] 发送静止合成观测（不会连接CAN或机械臂）", flush=True)
    started = time.perf_counter()
    response = policy.infer(observation)
    latency_ms = (time.perf_counter() - started) * 1000

    found = find_action_chunk(response)
    if not found:
        keys = list(response) if isinstance(response, dict) else type(response).__name__
        raise RuntimeError(f"服务端已响应，但没有找到末维为14的动作数组；响应摘要={keys}")
    action_path, actions = found
    finite = bool(np.isfinite(actions).all())
    print(
        "[5/5] PREFLIGHT_OK "
        + json.dumps({
            "latency_ms": round(latency_ms, 2),
            "action_path": action_path,
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "finite": finite,
            "min": float(np.nanmin(actions)),
            "max": float(np.nanmax(actions)),
            "hardware_output": False,
        }, ensure_ascii=False),
        flush=True,
    )
    if actions.shape != (50, 14) or not finite:
        raise RuntimeError(f"动作块校验失败：期望(50,14)且全部有限，实际{actions.shape}, finite={finite}")
    if hasattr(signal, "alarm"):
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
