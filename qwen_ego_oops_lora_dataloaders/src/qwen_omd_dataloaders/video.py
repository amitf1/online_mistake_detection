from __future__ import annotations

import math
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]

from .config import VideoSamplingConfig


def video_duration_seconds(video_path: str | Path) -> float:
    if cv2 is None:
        return 0.0
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0:
            return 0.0
        return float(frames / fps)
    finally:
        cap.release()


def build_sample_timestamps(
    *,
    start_time: float,
    end_time: float,
    config: VideoSamplingConfig,
) -> list[float]:
    start = max(0.0, float(start_time))
    end = max(start, float(end_time))
    duration = end - start
    if duration <= 0:
        return [start]

    fps = max(config.sample_fps, 1e-6)
    count = max(1, int(math.ceil(duration * fps)))
    step = duration / count # 1/fps
    raw = [start + index * step for index in range(count)]

    if len(raw) > config.max_frames:
        if config.max_frames <= 1:
            raw = [raw[0]]
        else:
            keep = [
                round(index * (len(raw) - 1) / (config.max_frames - 1))
                for index in range(config.max_frames)
            ]
            raw = [raw[index] for index in keep]

    return [float(timestamp) for timestamp in raw]


def _resize_short_side(frame: np.ndarray, short_side: int | None) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError("opencv-python is required for frame resizing")
    if short_side is None:
        return frame
    height, width = frame.shape[:2]
    current_short = min(height, width)
    if current_short <= 0 or current_short == short_side:
        return frame
    scale = short_side / current_short
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def read_video_frames_at_timestamps(
    video_path: str | Path,
    timestamps: list[float],
    *,
    resize_short_side: int | None = None,
) -> np.ndarray:
    """Return RGB frames with shape [T, H, W, 3]."""

    if cv2 is None:
        raise ModuleNotFoundError("opencv-python is required for video frame loading")
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise ValueError(f"Could not read frame at {timestamp:.3f}s from {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_short_side(frame_rgb, resize_short_side)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames sampled from {video_path}")
    return np.stack(frames, axis=0)


def sample_video_window(
    *,
    video_path: str | Path,
    start_time: float,
    end_time: float,
    config: VideoSamplingConfig,
) -> tuple[np.ndarray, list[float]]:
    timestamps = build_sample_timestamps(
        start_time=start_time,
        end_time=end_time,
        config=config,
    )
    frames = read_video_frames_at_timestamps(
        video_path,
        timestamps,
        resize_short_side=config.resize_short_side,
    )
    return frames, timestamps
