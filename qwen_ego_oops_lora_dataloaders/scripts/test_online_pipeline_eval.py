from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from run_online_pipeline_eval import (  # noqa: E402
    GroundTruthEvent,
    PipelineEvent,
    compute_metrics,
    match_events,
)
from qwen_omd_dataloaders.schema import WindowSpec  # noqa: E402


def gt(
    *,
    step_index: int = 0,
    start: float = 10.0,
    end: float = 20.0,
    mistake: bool = True,
    video_id: str = "video",
) -> GroundTruthEvent:
    return GroundTruthEvent(
        video_id=video_id,
        task_id="task",
        step_index=step_index,
        instruction=f"step {step_index}",
        gt_start=start,
        gt_end=end,
        gt_mistake=mistake,
        gt_reasoning="reference reasoning",
        eligible=True,
    )


def event(
    *,
    step_index: int = 0,
    start: float = 10.0,
    end: float = 20.0,
    trigger_time: float | None = None,
    module_a_seconds: float | None = None,
    module_b_seconds: float | None = None,
    module_c_seconds: float | None = None,
    mistake: bool | None = True,
    video_id: str = "video",
) -> PipelineEvent:
    pipeline_seconds = None
    if module_a_seconds is not None or module_b_seconds is not None or module_c_seconds is not None:
        pipeline_seconds = sum(
            value
            for value in (module_a_seconds, module_b_seconds, module_c_seconds)
            if value is not None
        )
    return PipelineEvent(
        video_id=video_id,
        task_id="task",
        step_index=step_index,
        instruction=f"step {step_index}",
        trigger_time=end if trigger_time is None else trigger_time,
        accumulation_start=0.0,
        accumulation_end=end,
        module_a_prediction="COMPLETE",
        module_b_prediction=(start, end),
        pred_global_start=start,
        pred_global_end=end,
        module_c_mistake=mistake,
        module_c_reasoning="predicted reasoning" if mistake is not None else None,
        module_a_prediction_seconds=module_a_seconds,
        module_b_prediction_seconds=module_b_seconds,
        module_c_prediction_seconds=module_c_seconds,
        pipeline_prediction_seconds=pipeline_seconds,
    )


def window(label: str = "COMPLETE") -> WindowSpec:
    return WindowSpec(
        video_path=Path("/tmp/fake.mp4"),
        video_id="video",
        task_id="task",
        step_index=0,
        current_step="step 0",
        window_start=0.0,
        window_end=20.0,
        gt_start=10.0,
        gt_end=20.0,
        label=label,
    )


def test_matching_keeps_best_and_marks_duplicate() -> None:
    gt_events = [gt(start=10.0, end=20.0)]
    events = [
        event(start=9.0, end=20.0),
        event(start=10.0, end=20.0),
    ]
    match_events(events, gt_events)
    assert events[1].matched_gt_index == 0
    assert events[1].matched_iou == 1.0
    assert events[0].matched_gt_index is None
    assert events[0].duplicate is True


def test_end_to_end_mistake_metrics_count_fp_and_fn() -> None:
    gt_events = [
        gt(step_index=0, mistake=True, start=10.0, end=20.0),
        gt(step_index=1, mistake=False, start=30.0, end=40.0),
        gt(step_index=2, mistake=True, start=50.0, end=60.0),
    ]
    events = [
        event(step_index=0, start=10.0, end=20.0, mistake=True),
        event(step_index=1, start=30.0, end=40.0, mistake=True),
        event(step_index=9, start=70.0, end=80.0, mistake=True),
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["end_to_end_mistake_at_iou_0.5/tp"] == 1.0
    assert metrics["end_to_end_mistake_at_iou_0.5/fp"] == 2.0
    assert metrics["end_to_end_mistake_at_iou_0.5/fn"] == 1.0
    assert metrics["end_to_end_mistake_precision_at_iou_0.5"] == 1 / 3
    assert metrics["end_to_end_mistake_recall_at_iou_0.5"] == 0.5
    matrix = metrics["end_to_end_mistake_at_iou_0.5/confusion_matrix"]
    assert matrix["gt_mistake"]["pred_mistake"] == 1
    assert matrix["gt_mistake"]["missed"] == 1
    assert matrix["gt_correct"]["pred_mistake"] == 1
    assert matrix["unmatched_prediction"]["pred_mistake"] == 1


def test_conditional_module_c_metrics_only_use_matched_windows() -> None:
    gt_events = [
        gt(step_index=0, mistake=True, start=10.0, end=20.0),
        gt(step_index=1, mistake=False, start=30.0, end=40.0),
    ]
    events = [
        event(step_index=0, start=10.0, end=20.0, mistake=True),
        event(step_index=1, start=30.0, end=40.0, mistake=False),
        event(step_index=9, start=70.0, end=80.0, mistake=True),
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["module_c_given_match_iou_0.5/tp"] == 1.0
    assert metrics["module_c_given_match_iou_0.5/tn"] == 1.0
    assert metrics["module_c_given_match_iou_0.5/fp"] == 0.0
    assert metrics["mistake_f1_given_match_iou_0.5"] == 1.0


def test_latency_statistics_include_early_and_late_triggers() -> None:
    gt_events = [
        gt(step_index=0, start=10.0, end=20.0),
        gt(step_index=1, start=30.0, end=40.0),
    ]
    events = [
        event(step_index=0, start=10.0, end=20.0, trigger_time=15.0),
        event(step_index=1, start=30.0, end=40.0, trigger_time=45.0),
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["latency/count"] == 2.0
    assert metrics["latency/mean_seconds"] == 0.0
    assert metrics["latency/mean_absolute_seconds"] == 5.0
    assert metrics["latency/early_count"] == 1.0
    assert metrics["latency/late_count"] == 1.0
    assert metrics["latency_at_iou_0.5/count"] == 2.0


def test_prediction_runtime_statistics_for_completed_actions() -> None:
    gt_events = [gt(step_index=0, start=10.0, end=20.0)]
    events = [
        event(
            step_index=0,
            start=10.0,
            end=20.0,
            module_a_seconds=1.0,
            module_b_seconds=2.0,
            module_c_seconds=3.0,
        )
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["prediction_runtime/module_a/count"] == 1.0
    assert metrics["prediction_runtime/module_a/mean_seconds"] == 1.0
    assert metrics["prediction_runtime/module_b/mean_seconds"] == 2.0
    assert metrics["prediction_runtime/module_c/mean_seconds"] == 3.0
    assert metrics["prediction_runtime/pipeline_completed_action/mean_seconds"] == 6.0


def test_stage_metrics_are_separated_from_full_pipeline_metrics() -> None:
    gt_events = [gt(step_index=0, start=10.0, end=20.0)]
    events = [event(step_index=0, start=10.0, end=20.0, mistake=True)]
    match_events(events, gt_events)
    module_a_predictions = [(window("COMPLETE"), "COMPLETE", "COMPLETE", 0.25)]
    metrics = compute_metrics(events, gt_events, module_a_predictions=module_a_predictions)
    assert metrics["module_a/num_windows"] == 1.0
    assert metrics["module_a/recall"] == 1.0
    assert metrics["module_b_given_module_a_trigger/num_triggers"] == 1.0
    assert metrics["module_b_given_module_a_trigger/recall_at_0.5"] == 1.0
    assert metrics["end_to_end_mistake_recall_at_iou_0.5"] == 1.0


def main() -> None:
    test_matching_keeps_best_and_marks_duplicate()
    test_end_to_end_mistake_metrics_count_fp_and_fn()
    test_conditional_module_c_metrics_only_use_matched_windows()
    test_latency_statistics_include_early_and_late_triggers()
    test_prediction_runtime_statistics_for_completed_actions()
    test_stage_metrics_are_separated_from_full_pipeline_metrics()
    print("Online pipeline eval tests passed.")


if __name__ == "__main__":
    main()
