from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoSamplingConfig:
    """Frame sampling settings shared by datasets and collators."""

    sample_fps: float = 2.0
    max_frames: int = 64
    resize_short_side: int | None = None


@dataclass(frozen=True)
class ModuleAConfig:
    stride_seconds: float = 5.0
    completion_margin: float = 0.10
    negative_to_positive_ratio: int = 2
    keep_last_wait_windows: int = 2
    seed: int = 13


@dataclass(frozen=True)
class ModuleBConfig:
    pre_pad_ratio_min: float = 0.10
    pre_pad_ratio_max: float = 0.50
    post_pad_ratio_min: float = 0.10
    post_pad_ratio_max: float = 0.50
    include_incomplete_negatives: bool = False
    negative_ratio: float = 0.15
    seed: int = 17


@dataclass(frozen=True)
class ModuleCConfig:
    jitter_ratio_min: float = 0.05
    jitter_ratio_max: float = 0.10
    seed: int = 19


@dataclass(frozen=True)
class TimeTokenConfig:
    num_bins: int = 101
    token_template: str = "<time_{index:03d}>"
    no_action_token: str = "<no_action>"
