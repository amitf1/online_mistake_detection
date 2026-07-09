from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from qwen_omd_dataloaders.config import VideoSamplingConfig  # noqa: E402
from qwen_omd_dataloaders.time_tokens import render_seconds_span_target  # noqa: E402
from qwen_omd_dataloaders.video import build_sample_timestamps  # noqa: E402
from train_module_b_unsloth import parse_temporal_prediction, temporal_metrics  # noqa: E402


def test_render_seconds_span_target() -> None:
    target = render_seconds_span_target(gt_start=13.2, gt_end=17.8, window_start=10.0)
    parsed = json.loads(target)
    assert parsed == {"relevant_windows": [["3.20", "7.80"]]}


def test_parse_temporal_prediction() -> None:
    assert parse_temporal_prediction('{"relevant_windows":[["3.20","7.80"]]}') == (3.2, 7.8)
    assert parse_temporal_prediction('{"start_time":"3.20","end_time":"7.80"}') == (3.2, 7.8)
    assert parse_temporal_prediction("the span is 3.20 to 7.80 seconds") == (3.2, 7.8)
    assert parse_temporal_prediction('{"relevant_windows":[]}') is None
    assert parse_temporal_prediction("not completed") is None


def test_temporal_metrics() -> None:
    metrics = temporal_metrics(
        targets=[(13.0, 18.0), (20.0, 25.0)],
        predictions=[(13.0, 18.0), None],
    )
    assert metrics["eval_temporal/num_samples"] == 2.0
    assert metrics["eval_temporal/invalid_rate"] == 0.5
    assert metrics["eval_temporal/recall_at_0.5"] == 0.5
    assert metrics["eval_temporal/f1_at_0.5"] == 2 / 3

    mixed_metrics = temporal_metrics(
        targets=[(13.0, 18.0), (20.0, 25.0)],
        predictions=[(13.0, 18.0), None],
        labels=["LOCALIZE", "NO_ACTION"],
    )
    assert mixed_metrics["eval_temporal/num_positive_samples"] == 1.0
    assert mixed_metrics["eval_temporal/num_no_action_samples"] == 1.0
    assert mixed_metrics["eval_temporal/invalid_rate"] == 0.0
    assert mixed_metrics["eval_temporal/no_action_accuracy"] == 1.0
    assert mixed_metrics["eval_temporal/recall_at_0.5"] == 1.0


def test_downsampled_timestamps_preserve_window_coverage() -> None:
    timestamps = build_sample_timestamps(
        start_time=0.0,
        end_time=10.0,
        config=VideoSamplingConfig(sample_fps=4.0, max_frames=5),
    )
    assert len(timestamps) == 5
    assert timestamps[0] == 0.0
    assert timestamps[-1] > 9.0
    assert timestamps == sorted(timestamps)


def main() -> None:
    test_render_seconds_span_target()
    test_parse_temporal_prediction()
    test_temporal_metrics()
    test_downsampled_timestamps_preserve_window_coverage()
    print("Module B temporal tests passed.")


if __name__ == "__main__":
    main()
