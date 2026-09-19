#!/usr/bin/env python3
"""Verify configured cameras and save one frame from each for mapping review."""

import argparse
import json
from pathlib import Path
import time

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("client_config.json")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("camera_preflight")))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    failures = []
    for name, device in config["cameras"].items():
        cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
        if not cap.isOpened():
            failures.append(f"{name}: /dev/video{device} 无法打开")
            continue
        frame = None
        for _ in range(10):
            ok, candidate = cap.read()
            if ok:
                frame = candidate
            time.sleep(0.03)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if frame is None:
            failures.append(f"{name}: 已打开但没有图像帧")
            continue
        target = output / f"{name}_device_{device}.jpg"
        cv2.imwrite(str(target), frame)
        print(f"CAMERA_OK {name}: /dev/video{device}, {width}x{height}, {fps:.1f} FPS -> {target}")

    if failures:
        for failure in failures:
            print("CAMERA_ERROR", failure)
        return 1
    print("CAMERA_PREFLIGHT_OK：三路相机均已采集；请人工核对顶部/左腕/右腕没有接反。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

