#!/usr/bin/env python3
"""Convert GOAI-2026 real Piper HDF5 demonstrations to LeRobot v3 Joint."""

from __future__ import annotations

import argparse
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import video_utils
from lerobot.datasets.video_utils import StreamingVideoEncoder


TASK_PROMPTS = {
    "fill_pen_holder": "fill the pen holder",
    "insert_charger": "insert the charger",
    "put_objects_into_basket": "put the objects into the basket",
    "stack_and_cover_blocks": "stack and cover the blocks",
    "stack_bowls": "stack the bowls",
    "stand_up_bottles": "stand up the bottles",
}

JOINT_NAMES = [
    *(f"left_joint_{i}" for i in range(6)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(6)),
    "right_gripper",
]


_ORIGINAL_CODEC_OPTIONS = video_utils._get_codec_options


def fast_codec_options(vcodec: str, g: int | None = 2, crf: int | None = 30, preset=None) -> dict:
    options = _ORIGINAL_CODEC_OPTIONS(vcodec, g, crf, preset)
    if vcodec == "h264":
        options["preset"] = "ultrafast"
        options["tune"] = "fastdecode"
    return options


def blocking_feed_frame(self: StreamingVideoEncoder, video_key: str, image: np.ndarray) -> None:
    """Use backpressure for offline conversion; never drop a frame."""
    if not self._episode_active:
        raise RuntimeError("No active streaming episode")
    frame = image.copy()
    while True:
        thread = self._threads[video_key]
        if not thread.is_alive():
            try:
                status, message = self._result_queues[video_key].get_nowait()
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} crashed: {message}")
            except queue.Empty:
                pass
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")
        try:
            self._frame_queues[video_key].put(frame, timeout=0.5)
            return
        except queue.Full:
            continue


video_utils._get_codec_options = fast_codec_options
StreamingVideoEncoder.feed_frame = blocking_feed_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/goai2026_real_piper_joint")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASK_PROMPTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vcodec", default="h264", choices=("h264", "h264_nvenc"))
    parser.add_argument("--decode-workers", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--encoder-threads", type=int, default=1)
    return parser.parse_args()


def decode_jpeg(encoded: np.ndarray, source: Path, camera: str, frame: int) -> np.ndarray:
    payload = encoded.tobytes().rstrip(b"\x00")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {source}: {camera} frame {frame}")
    return np.ascontiguousarray(image)


def episode_files(source: Path, selected_tasks: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for task in selected_tasks:
        files = sorted((source / task / "piper_x" / "data").glob("episode_*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No episodes found for task: {task}")
        result.extend((task, file) for file in files)
    return result


def make_features() -> dict:
    image_feature = {
        "dtype": "video", "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    }
    vector_feature = {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
    return {
        "observation.images.cam_high": image_feature,
        "observation.images.cam_left_wrist": image_feature.copy(),
        "observation.images.cam_right_wrist": image_feature.copy(),
        "observation.state": vector_feature,
        "action": vector_feature.copy(),
    }


def load_states(handle: h5py.File, source: Path) -> np.ndarray:
    left_joint = np.asarray(handle["left_arm/joint"], dtype=np.float32)
    right_joint = np.asarray(handle["right_arm/joint"], dtype=np.float32)
    left_gripper = np.asarray(handle["left_arm/gripper"], dtype=np.float32).reshape(-1, 1)
    right_gripper = np.asarray(handle["right_arm/gripper"], dtype=np.float32).reshape(-1, 1)
    lengths = {len(left_joint), len(right_joint), len(left_gripper), len(right_gripper)}
    if len(lengths) != 1 or left_joint.shape[1:] != (6,) or right_joint.shape[1:] != (6,):
        raise ValueError(f"Invalid joint shapes in {source}")
    states = np.concatenate((left_joint, left_gripper, right_joint, right_gripper), axis=1)
    if states.shape[1] != 14 or not np.isfinite(states).all():
        raise ValueError(f"Invalid state data in {source}: {states.shape}")
    return states


def add_episode(
    dataset: LeRobotDataset,
    task: str,
    source: Path,
    executor: ThreadPoolExecutor,
    decode_batch_size: int,
) -> int:
    with h5py.File(source, "r") as handle:
        states = load_states(handle, source)
        cameras = {
            "observation.images.cam_high": handle["cam_head/color"],
            "observation.images.cam_left_wrist": handle["cam_left_wrist/color"],
            "observation.images.cam_right_wrist": handle["cam_right_wrist/color"],
        }
        lengths = {len(states), *(len(value) for value in cameras.values())}
        if len(lengths) != 1:
            raise ValueError(f"Camera/state length mismatch in {source}: {sorted(lengths)}")
        actions = np.concatenate((states[1:], states[-1:]), axis=0)
        for batch_start in range(0, len(states), decode_batch_size):
            batch_end = min(len(states), batch_start + decode_batch_size)
            futures = {}
            for index in range(batch_start, batch_end):
                for output_key, encoded_frames in cameras.items():
                    encoded = np.asarray(encoded_frames[index])
                    futures[(index, output_key)] = executor.submit(
                        decode_jpeg, encoded, source, output_key, index
                    )
            for index in range(batch_start, batch_end):
                frame = {
                    "task": TASK_PROMPTS[task],
                    "observation.state": states[index],
                    "action": actions[index],
                }
                for output_key in cameras:
                    frame[output_key] = futures[(index, output_key)].result()
                dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
        return len(states)


def main() -> None:
    args = parse_args()
    selected_tasks = args.tasks or list(TASK_PROMPTS)
    files = episode_files(args.source, selected_tasks)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError("No episodes selected")
    if args.decode_workers < 1 or args.decode_batch_size < 1:
        raise ValueError("Decode worker and batch counts must be positive")
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {args.output}")
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.output,
            streaming_encoding=True,
            vcodec=args.vcodec,
            encoder_queue_maxsize=8,
            encoder_threads=args.encoder_threads,
            batch_encoding_size=1,
            video_backend="pyav",
        )
        completed_episodes = dataset.meta.total_episodes
        if completed_episodes > len(files):
            raise ValueError(f"Dataset has {completed_episodes} episodes but only {len(files)} inputs")
        print(
            f"RESUME completed_episodes={completed_episodes} completed_frames={dataset.meta.total_frames} "
            f"vcodec={args.vcodec}",
            flush=True,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=25, features=make_features(), root=args.output,
            robot_type="piper_x", use_videos=True, vcodec=args.vcodec, streaming_encoding=True,
            encoder_queue_maxsize=8, encoder_threads=args.encoder_threads, batch_encoding_size=1,
        )
        completed_episodes = 0

    total_frames = dataset.meta.total_frames
    remaining_files = files[completed_episodes:]
    with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        for episode_index, (task, source) in enumerate(remaining_files, start=completed_episodes + 1):
            frames = add_episode(dataset, task, source, executor, args.decode_batch_size)
            total_frames += frames
            print(f"CONVERTED episode={episode_index}/{len(files)} task={task} frames={frames} total_frames={total_frames} source={source.name}", flush=True)
    dataset.finalize()
    print(f"COMPLETE episodes={len(files)} frames={total_frames} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
