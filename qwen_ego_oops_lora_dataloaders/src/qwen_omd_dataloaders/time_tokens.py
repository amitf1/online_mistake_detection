from __future__ import annotations

from .config import TimeTokenConfig


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def percent_to_token(value: float, config: TimeTokenConfig | None = None) -> str:
    cfg = config or TimeTokenConfig()
    if cfg.num_bins < 2:
        raise ValueError("TimeTokenConfig.num_bins must be >= 2")
    index = round(clamp01(value) * (cfg.num_bins - 1))
    return cfg.token_template.format(index=index)


def relative_span_to_tokens(
    *,
    gt_start: float,
    gt_end: float,
    window_start: float,
    window_end: float,
    config: TimeTokenConfig | None = None,
) -> tuple[str, str]:
    duration = window_end - window_start
    if duration <= 0:
        raise ValueError("window_end must be greater than window_start")
    start_rel = (gt_start - window_start) / duration
    end_rel = (gt_end - window_start) / duration
    return percent_to_token(start_rel, config), percent_to_token(end_rel, config)


def render_time_span_target(
    *,
    gt_start: float,
    gt_end: float,
    window_start: float,
    window_end: float,
    config: TimeTokenConfig | None = None,
) -> str:
    start_token, end_token = relative_span_to_tokens(
        gt_start=gt_start,
        gt_end=gt_end,
        window_start=window_start,
        window_end=window_end,
        config=config,
    )
    return f"{start_token} to {end_token}"
