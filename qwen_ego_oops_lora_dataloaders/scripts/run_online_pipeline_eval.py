from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT, Path(__file__).resolve().parent):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from qwen_omd_dataloaders.build import DEFAULT_METADATA_PATH, DEFAULT_MISTAKE_CLASSES_PATH, DEFAULT_VIDEO_ROOT  # noqa: E402
from qwen_omd_dataloaders.config import ModuleAConfig  # noqa: E402
from qwen_omd_dataloaders.datasets import EgoOopsModuleADataset, EgoOopsProvider  # noqa: E402
from qwen_omd_dataloaders.datasets.ego_oops import (  # noqa: E402
    EXTRA_STEP_INSTRUCTION,
    MODULE_A_STEP_ID_NONE_LETTER,
    _module_b_prompt,
    _module_c_prompt,
    _reasoning_for_segment,
    is_module_a_completion_label,
    module_a_instruction_for_step_id_letter,
)
from qwen_omd_dataloaders.schema import Segment, TrainingExample, VideoRecord, WindowSpec  # noqa: E402
from qwen_omd_dataloaders.video import video_duration_seconds  # noqa: E402

from train_module_a_unsloth import (  # noqa: E402
    ModuleADetectionMetricsCallback,
    Qwen3VLMetadataVisionDataCollator,
    binary_detection_metrics,
    extract_module_a_label,
    json_safe,
    load_generation_eval_model,
    load_or_create_video_split,
    multiclass_letter_metrics,
    release_cuda_memory,
    training_example_to_conversation as module_a_to_conversation,
)
from train_module_b_unsloth import (  # noqa: E402
    ModuleBTemporalMetricsCallback,
    parse_temporal_prediction,
    temporal_metrics,
    temporal_iou,
)
from train_module_c_unsloth import (  # noqa: E402
    ModuleCMistakeMetricsCallback,
    lexical_overlap,
    parse_module_c_prediction,
)

IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class GroundTruthEvent:
    video_id: str
    task_id: str
    step_index: int
    instruction: str
    gt_start: float
    gt_end: float
    gt_mistake: bool
    gt_reasoning: str
    eligible: bool


@dataclass
class PipelineEvent:
    video_id: str
    task_id: str
    step_index: int
    instruction: str
    trigger_time: float
    accumulation_start: float
    accumulation_end: float
    module_a_prediction: str
    module_b_raw: str | None = None
    module_b_prediction: tuple[float, float] | None = None
    pred_global_start: float | None = None
    pred_global_end: float | None = None
    module_c_raw: str | None = None
    module_c_mistake: bool | None = None
    module_c_reasoning: str | None = None
    module_a_prediction_seconds: float | None = None
    module_b_prediction_seconds: float | None = None
    module_c_prediction_seconds: float | None = None
    pipeline_prediction_seconds: float | None = None
    matched_gt_index: int | None = None
    matched_iou: float = 0.0
    matched_gt_start: float | None = None
    matched_gt_end: float | None = None
    gt_mistake: bool | None = None
    gt_reasoning: str | None = None
    duplicate: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end online Module A -> B -> C pipeline evaluation.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--mistake-classes", default=DEFAULT_MISTAKE_CLASSES_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--module-a-checkpoint", default=None)
    parser.add_argument("--module-b-checkpoint", default=None)
    parser.add_argument("--module-c-checkpoint", required=True)
    parser.add_argument(
        "--reuse-events-json",
        default=None,
        help=(
            "Reuse saved pipeline events (A/B windows) and only re-run Module C. "
            "Skips Module A/B inference. Useful after fixing Module C frame/resize settings."
        ),
    )
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--regenerate-split", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-videos-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--completion-margin", type=float, default=0.10)
    parser.add_argument("--negative-to-positive-ratio", type=int, default=2)
    parser.add_argument("--keep-last-wait-windows", type=int, default=2)
    parser.add_argument(
        "--module-a-label-mode",
        choices=("legacy", "step_id"),
        default="step_id",
        help="Module A label/prompt mode. step_id uses A/B/C+ letters; legacy uses WAIT/COMPLETE.",
    )
    parser.add_argument(
        "--online-reset-from-b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In step_id mode, after each Module A trigger run Module B and advance the stream "
            "start to B's predicted end for subsequent Module A windows on that video."
        ),
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames-a", type=int, default=32)
    parser.add_argument("--max-frames-b", type=int, default=24)
    parser.add_argument("--max-frames-c", type=int, default=16)
    parser.add_argument("--vision-resize-a", type=int, default=384)
    parser.add_argument("--vision-resize-b", type=int, default=512)
    parser.add_argument("--vision-resize-c", type=int, default=384)
    parser.add_argument("--max-seq-length-a", type=int, default=6144)
    parser.add_argument("--max-seq-length-b", type=int, default=8192)
    parser.add_argument("--max-seq-length-c", type=int, default=3072)
    parser.add_argument("--module-a-max-new-tokens", type=int, default=8)
    parser.add_argument("--module-b-max-new-tokens", type=int, default=64)
    parser.add_argument("--module-c-max-new-tokens", type=int, default=128)
    parser.add_argument("--label-score-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fbeta-beta", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=-1, help="Cap validation windows/events for smoke tests.")
    parser.add_argument("--output-dir", default="/home/amit/online_mistake_detection/outputs/online_pipeline_eval")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--skip-inference", action="store_true", help="Build GT/split only; useful for debugging setup.")
    args = parser.parse_args()
    if args.max_videos <= 0:
        parser.error("--max-videos must be positive.")
    if args.stride_seconds <= 0:
        parser.error("--stride-seconds must be positive.")
    if args.max_samples < -1:
        parser.error("--max-samples must be -1 or non-negative.")
    if args.reuse_events_json:
        if args.skip_inference:
            parser.error("--reuse-events-json cannot be combined with --skip-inference.")
    elif not args.skip_inference and (not args.module_a_checkpoint or not args.module_b_checkpoint):
        parser.error("--module-a-checkpoint and --module-b-checkpoint are required unless --reuse-events-json is set.")
    return args


def make_provider(args: argparse.Namespace) -> EgoOopsProvider:
    return EgoOopsProvider(
        metadata_path=args.metadata,
        mistake_classes_path=args.mistake_classes,
        video_root=args.video_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        max_videos=args.max_videos,
        require_existing_videos=True,
    )


def module_a_config(args: argparse.Namespace) -> ModuleAConfig:
    return ModuleAConfig(
        stride_seconds=args.stride_seconds,
        completion_margin=args.completion_margin,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        keep_last_wait_windows=args.keep_last_wait_windows,
        seed=args.seed,
        label_mode=getattr(args, "module_a_label_mode", "step_id"),
    )


def gt_events_for_records(
    records: list[VideoRecord],
    val_video_ids: set[str],
    *,
    label_mode: str = "step_id",
) -> list[GroundTruthEvent]:
    events: list[GroundTruthEvent] = []
    for record in records:
        if record.video_id not in val_video_ids:
            continue
        for segment in record.segments:
            if label_mode == "step_id":
                eligible = segment.duration >= 0.5
            else:
                eligible = segment.instruction_index >= 0 and segment.duration >= 0.5
            events.append(
                GroundTruthEvent(
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    instruction=segment.instruction,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    gt_mistake=segment.is_mistake,
                    gt_reasoning=_reasoning_for_segment(segment),
                    eligible=eligible,
                )
            )
    return events


def val_windows(dataset: EgoOopsModuleADataset, val_video_ids: set[str], max_samples: int) -> list[WindowSpec]:
    windows = [window for window in dataset.windows if window.video_id in val_video_ids]
    windows.sort(key=lambda item: (item.video_id, item.window_end, item.step_index, item.window_start))
    if max_samples >= 0:
        return windows[:max_samples]
    return windows


def window_to_example(window: WindowSpec, dataset: EgoOopsModuleADataset) -> TrainingExample:
    return dataset._window_to_example(window)


def resolved_instruction_for_prediction(
    *,
    window: WindowSpec,
    prediction: str,
    label_mode: str,
) -> str:
    if label_mode != "step_id":
        return window.current_step
    mapped = module_a_instruction_for_step_id_letter(prediction, window.all_instructions or ())
    if mapped is None:
        return EXTRA_STEP_INSTRUCTION if prediction.upper().startswith("B") else window.current_step
    return mapped


def make_model_args(args: argparse.Namespace, checkpoint: str, max_seq_length: int, resize: int) -> argparse.Namespace:
    return argparse.Namespace(
        eval_checkpoint=checkpoint,
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        vision_resize=resize,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
    )


def make_collator(model: Any, tokenizer: Any, max_seq_length: int, resize: int) -> Qwen3VLMetadataVisionDataCollator:
    return Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=max_seq_length,
        resize=resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )


def run_module_a(
    args: argparse.Namespace,
    windows: list[WindowSpec],
    dataset: EgoOopsModuleADataset,
) -> list[tuple[WindowSpec, str, str, float]]:
    if not windows:
        return []
    label_mode = args.module_a_label_mode
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_a_checkpoint, args.max_seq_length_a, args.vision_resize_a)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_a, args.vision_resize_a)
    generator = ModuleADetectionMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_a_max_new_tokens,
        beta=args.fbeta_beta,
        label_score_eval=args.label_score_eval,
    )
    predictions: list[tuple[WindowSpec, str, str, float]] = []
    try:
        for window in windows:
            example = module_a_to_conversation(
                window_to_example(window, dataset),
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=args.max_frames_a,
            )
            if args.label_score_eval:
                started = time.perf_counter()
                prediction = generator._score_label(model, example)
                elapsed = time.perf_counter() - started
                raw = prediction
            else:
                started = time.perf_counter()
                raw = generator._generate_label(model, example)
                elapsed = time.perf_counter() - started
                prediction = extract_module_a_label(raw, label_mode=label_mode)
            predictions.append((window, prediction, raw, elapsed))
    finally:
        del model
        release_cuda_memory()
    return predictions


def b_example_from_window(window: WindowSpec, instruction: str) -> TrainingExample:
    return TrainingExample(
        module="B",
        source_dataset="ego_oops",
        video_path=window.video_path,
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        prompt_text=_module_b_prompt(instruction),
        target_text="not completed",
        label="LOCALIZE",
        metadata={"instruction": instruction},
    )


def run_module_b(args: argparse.Namespace, triggers: list[tuple[WindowSpec, str, str, float]]) -> list[PipelineEvent]:
    if not triggers:
        return []
    label_mode = args.module_a_label_mode
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_b_checkpoint, args.max_seq_length_b, args.vision_resize_b)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_b, args.vision_resize_b)
    generator = ModuleBTemporalMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_b_max_new_tokens,
    )
    events: list[PipelineEvent] = []
    try:
        for window, module_a_prediction, _module_a_raw, module_a_elapsed in triggers:
            events.append(
                run_module_b_one(
                    args,
                    model=model,
                    generator=generator,
                    window=window,
                    module_a_prediction=module_a_prediction,
                    module_a_elapsed=module_a_elapsed,
                    label_mode=label_mode,
                )
            )
    finally:
        del model
        release_cuda_memory()
    return events


def run_module_b_one(
    args: argparse.Namespace,
    *,
    model: Any,
    generator: ModuleBTemporalMetricsCallback,
    window: WindowSpec,
    module_a_prediction: str,
    module_a_elapsed: float,
    label_mode: str,
) -> PipelineEvent:
    instruction = resolved_instruction_for_prediction(
        window=window,
        prediction=module_a_prediction,
        label_mode=label_mode,
    )
    example = b_example_from_window(window, instruction)
    conversation = __import__("train_module_b_unsloth").training_example_to_conversation(
        example,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames_b,
    )
    started = time.perf_counter()
    raw = generator._generate_span(model, conversation)
    module_b_elapsed = time.perf_counter() - started
    parsed = parse_temporal_prediction(raw)
    event = PipelineEvent(
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        instruction=instruction,
        trigger_time=window.window_end,
        accumulation_start=window.window_start,
        accumulation_end=window.window_end,
        module_a_prediction=module_a_prediction,
        module_b_raw=raw,
        module_b_prediction=parsed,
        module_a_prediction_seconds=module_a_elapsed,
        module_b_prediction_seconds=module_b_elapsed,
    )
    if parsed is not None:
        start = window.window_start + parsed[0]
        end = window.window_start + parsed[1]
        if end > start:
            event.pred_global_start = max(0.0, start)
            event.pred_global_end = end
    return event


def select_triggers_with_online_b_reset(
    args: argparse.Namespace,
    module_a_predictions: list[tuple[WindowSpec, str, str, float]],
) -> tuple[list[tuple[WindowSpec, str, str, float]], list[PipelineEvent]]:
    """Walk Module A predictions chronologically per video; after each trigger, run B and
    skip later windows whose start is before B's predicted end (fallback: accumulation end).
    """
    label_mode = args.module_a_label_mode
    by_video: dict[str, list[tuple[WindowSpec, str, str, float]]] = {}
    for item in module_a_predictions:
        by_video.setdefault(item[0].video_id, []).append(item)
    for items in by_video.values():
        items.sort(key=lambda row: (row[0].window_start, row[0].window_end, row[0].step_index))

    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_b_checkpoint, args.max_seq_length_b, args.vision_resize_b)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_b, args.vision_resize_b)
    generator = ModuleBTemporalMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_b_max_new_tokens,
    )
    triggers: list[tuple[WindowSpec, str, str, float]] = []
    events: list[PipelineEvent] = []
    try:
        for video_id in sorted(by_video):
            stream_start = 0.0
            for window, prediction, raw, elapsed in by_video[video_id]:
                if window.window_start + 1e-6 < stream_start:
                    continue
                if not is_module_a_completion_label(prediction, label_mode=label_mode):
                    continue
                trigger = (window, prediction, raw, elapsed)
                triggers.append(trigger)
                event = run_module_b_one(
                    args,
                    model=model,
                    generator=generator,
                    window=window,
                    module_a_prediction=prediction,
                    module_a_elapsed=elapsed,
                    label_mode=label_mode,
                )
                events.append(event)
                stream_start = (
                    float(event.pred_global_end)
                    if event.pred_global_end is not None
                    else float(window.window_end)
                )
    finally:
        del model
        release_cuda_memory()
    return triggers, events


def c_example_from_event(event: PipelineEvent) -> TrainingExample | None:
    if event.pred_global_start is None or event.pred_global_end is None:
        return None
    return TrainingExample(
        module="C",
        source_dataset="ego_oops",
        video_path=Path(""),
        video_id=event.video_id,
        task_id=event.task_id,
        step_index=event.step_index,
        window_start=event.pred_global_start,
        window_end=event.pred_global_end,
        prompt_text=_module_c_prompt(event.instruction),
        target_text='{"mistake":false,"reasoning":""}',
        label=None,
        metadata={"instruction": event.instruction},
    )


def attach_video_paths(
    events: list[PipelineEvent],
    windows_by_key: dict[tuple[str, int], WindowSpec],
    records: list[VideoRecord] | None = None,
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    record_paths = {record.video_id: record.video_path for record in (records or [])}
    for index, event in enumerate(events):
        window = windows_by_key.get((event.video_id, event.step_index))
        if window is not None:
            paths[index] = window.video_path
        elif event.video_id in record_paths:
            paths[index] = record_paths[event.video_id]
    return paths


def event_from_row(row: dict[str, Any]) -> PipelineEvent:
    module_b = row.get("module_b_prediction")
    if isinstance(module_b, list) and len(module_b) >= 2:
        module_b_prediction = (float(module_b[0]), float(module_b[1]))
    else:
        module_b_prediction = None
    return PipelineEvent(
        video_id=str(row["video_id"]),
        task_id=str(row["task_id"]),
        step_index=int(row["step_index"]),
        instruction=str(row["instruction"]),
        trigger_time=float(row["trigger_time"]),
        accumulation_start=float(row["accumulation_start"]),
        accumulation_end=float(row["accumulation_end"]),
        module_a_prediction=str(row.get("module_a_prediction") or "COMPLETE"),
        module_b_raw=row.get("module_b_raw"),
        module_b_prediction=module_b_prediction,
        pred_global_start=None if row.get("pred_global_start") is None else float(row["pred_global_start"]),
        pred_global_end=None if row.get("pred_global_end") is None else float(row["pred_global_end"]),
        module_a_prediction_seconds=row.get("module_a_prediction_seconds"),
        module_b_prediction_seconds=row.get("module_b_prediction_seconds"),
        # Module C fields are cleared by the caller before re-inference.
        module_c_raw=row.get("module_c_raw"),
        module_c_mistake=row.get("module_c_mistake"),
        module_c_reasoning=row.get("module_c_reasoning"),
        module_c_prediction_seconds=row.get("module_c_prediction_seconds"),
        pipeline_prediction_seconds=row.get("pipeline_prediction_seconds"),
        matched_gt_index=row.get("matched_gt_index"),
        matched_iou=float(row.get("matched_iou") or 0.0),
        matched_gt_start=None if row.get("matched_gt_start") is None else float(row["matched_gt_start"]),
        matched_gt_end=None if row.get("matched_gt_end") is None else float(row["matched_gt_end"]),
        gt_mistake=row.get("gt_mistake"),
        gt_reasoning=row.get("gt_reasoning"),
        duplicate=bool(row.get("duplicate")),
    )


def load_events_for_module_c_reuse(path: Path) -> list[PipelineEvent]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or "events" not in payload:
        raise ValueError(f"Expected pipeline metrics JSON with an 'events' list: {path}")
    events = [event_from_row(row) for row in payload["events"]]
    for event in events:
        event.module_c_raw = None
        event.module_c_mistake = None
        event.module_c_reasoning = None
        event.module_c_prediction_seconds = None
        event.pipeline_prediction_seconds = None
        event.matched_gt_index = None
        event.matched_iou = 0.0
        event.matched_gt_start = None
        event.matched_gt_end = None
        event.gt_mistake = None
        event.gt_reasoning = None
        event.duplicate = False
    return events


def reconstruct_module_a_predictions(
    *,
    windows: list[WindowSpec],
    events: list[PipelineEvent],
) -> list[tuple[WindowSpec, str, str, float]]:
    trigger_keys = {
        (
            event.video_id,
            event.step_index,
            round(event.accumulation_start, 6),
            round(event.accumulation_end, 6),
        )
        for event in events
    }
    predictions: list[tuple[WindowSpec, str, str, float]] = []
    for window in windows:
        key = (
            window.video_id,
            window.step_index,
            round(window.window_start, 6),
            round(window.window_end, 6),
        )
        prediction = "COMPLETE" if key in trigger_keys else "WAIT"
        predictions.append((window, prediction, prediction, float(0.0)))
    return predictions


def run_module_c(args: argparse.Namespace, events: list[PipelineEvent], video_paths: dict[int, Path]) -> None:
    valid_indices = [
        index for index, event in enumerate(events)
        if event.pred_global_start is not None and event.pred_global_end is not None and index in video_paths
    ]
    if not valid_indices:
        return
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_c_checkpoint, args.max_seq_length_c, args.vision_resize_c)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_c, args.vision_resize_c)
    generator = ModuleCMistakeMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_c_max_new_tokens,
    )
    try:
        for index in valid_indices:
            event = events[index]
            example = c_example_from_event(event)
            if example is None:
                continue
            example = TrainingExample(**{**asdict(example), "video_path": video_paths[index]})
            conversation = __import__("train_module_c_unsloth").training_example_to_conversation(
                example,
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=args.max_frames_c,
            )
            started = time.perf_counter()
            raw = generator._generate_json(model, conversation)
            module_c_elapsed = time.perf_counter() - started
            parsed = parse_module_c_prediction(raw)
            event.module_c_raw = raw
            event.module_c_prediction_seconds = module_c_elapsed
            event.pipeline_prediction_seconds = sum(
                value
                for value in (
                    event.module_a_prediction_seconds,
                    event.module_b_prediction_seconds,
                    event.module_c_prediction_seconds,
                )
                if value is not None
            )
            if parsed is not None:
                event.module_c_mistake = bool(parsed["mistake"])
                event.module_c_reasoning = str(parsed["reasoning"])
    finally:
        del model
        release_cuda_memory()


def match_events(events: list[PipelineEvent], gt_events: list[GroundTruthEvent]) -> None:
    candidates: list[tuple[float, int, int]] = []
    for event_index, event in enumerate(events):
        if event.pred_global_start is None or event.pred_global_end is None:
            continue
        pred = (event.pred_global_start, event.pred_global_end)
        for gt_index, gt in enumerate(gt_events):
            if not gt.eligible:
                continue
            if event.video_id != gt.video_id or event.step_index != gt.step_index:
                continue
            iou = temporal_iou(pred, (gt.gt_start, gt.gt_end))
            candidates.append((iou, event_index, gt_index))
    used_events: set[int] = set()
    used_gt: set[int] = set()
    for iou, event_index, gt_index in sorted(candidates, reverse=True):
        if event_index in used_events:
            continue
        event = events[event_index]
        if gt_index in used_gt:
            event.duplicate = True
            continue
        used_events.add(event_index)
        used_gt.add(gt_index)
        gt = gt_events[gt_index]
        event.matched_gt_index = gt_index
        event.matched_iou = iou
        event.matched_gt_start = gt.gt_start
        event.matched_gt_end = gt.gt_end
        event.gt_mistake = gt.gt_mistake
        event.gt_reasoning = gt.gt_reasoning
    for event in events:
        if event.matched_gt_index is None and event.pred_global_start is not None:
            for gt_index, gt in enumerate(gt_events):
                if event.video_id == gt.video_id and event.step_index == gt.step_index:
                    event.duplicate = gt_index in used_gt
                    break


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(latencies: list[float], *, prefix: str) -> dict[str, float]:
    early_count = sum(1 for value in latencies if value < 0)
    on_time_count = sum(1 for value in latencies if value == 0)
    late_count = sum(1 for value in latencies if value > 0)
    absolute = [abs(value) for value in latencies]
    return {
        f"{prefix}/count": float(len(latencies)),
        f"{prefix}/mean_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        f"{prefix}/median_seconds": percentile(latencies, 0.5),
        f"{prefix}/p90_seconds": percentile(latencies, 0.9),
        f"{prefix}/p95_seconds": percentile(latencies, 0.95),
        f"{prefix}/min_seconds": min(latencies) if latencies else 0.0,
        f"{prefix}/max_seconds": max(latencies) if latencies else 0.0,
        f"{prefix}/mean_absolute_seconds": sum(absolute) / len(absolute) if absolute else 0.0,
        f"{prefix}/median_absolute_seconds": percentile(absolute, 0.5),
        f"{prefix}/early_count": float(early_count),
        f"{prefix}/on_time_count": float(on_time_count),
        f"{prefix}/late_count": float(late_count),
        f"{prefix}/early_rate": safe_div(early_count, len(latencies)),
        f"{prefix}/on_time_rate": safe_div(on_time_count, len(latencies)),
        f"{prefix}/late_rate": safe_div(late_count, len(latencies)),
    }


def duration_summary(durations: list[float], *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/count": float(len(durations)),
        f"{prefix}/mean_seconds": sum(durations) / len(durations) if durations else 0.0,
        f"{prefix}/median_seconds": percentile(durations, 0.5),
        f"{prefix}/p90_seconds": percentile(durations, 0.9),
        f"{prefix}/p95_seconds": percentile(durations, 0.95),
        f"{prefix}/min_seconds": min(durations) if durations else 0.0,
        f"{prefix}/max_seconds": max(durations) if durations else 0.0,
    }


def binary_stats(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, total)
    f1 = safe_div(2 * precision * recall, precision + recall)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "num_samples": float(total),
    }


def confusion_matrix(tp: int, fp: int, tn: int, fn: int) -> dict[str, dict[str, int]]:
    return {
        "gt_mistake": {"pred_mistake": tp, "pred_correct_or_missed": fn},
        "gt_correct_or_unmatched": {"pred_mistake": fp, "pred_correct_or_none": tn},
    }


def end_to_end_confusion_matrix(
    *,
    events: list[PipelineEvent],
    gt_events: list[GroundTruthEvent],
    threshold: float,
) -> dict[str, dict[str, int]]:
    eligible_indices = [index for index, gt in enumerate(gt_events) if gt.eligible]
    detected_gt_indices = {
        event.matched_gt_index
        for event in events
        if event.matched_gt_index is not None and event.matched_iou >= threshold
    }
    return {
        "gt_mistake": {
            "pred_mistake": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is True
                and event.module_c_mistake is True
            ),
            "pred_correct": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is True
                and event.module_c_mistake is not True
            ),
            "missed": sum(
                index not in detected_gt_indices
                for index in eligible_indices
                if gt_events[index].gt_mistake
            ),
        },
        "gt_correct": {
            "pred_mistake": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is False
                and event.module_c_mistake is True
            ),
            "pred_correct": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is False
                and event.module_c_mistake is not True
            ),
            "missed": sum(
                index not in detected_gt_indices
                for index in eligible_indices
                if not gt_events[index].gt_mistake
            ),
        },
        "unmatched_prediction": {
            "pred_mistake": sum(
                1 for event in events
                if event.module_c_mistake is True
                and (event.matched_gt_index is None or event.matched_iou < threshold or event.duplicate)
            ),
            "pred_correct": sum(
                1 for event in events
                if event.module_c_mistake is not True
                and event.pred_global_start is not None
                and (event.matched_gt_index is None or event.matched_iou < threshold or event.duplicate)
            ),
            "missed": 0,
        },
    }


def eligible_gt(gt_events: list[GroundTruthEvent]) -> list[GroundTruthEvent]:
    return [gt for gt in gt_events if gt.eligible]


def prefixed_metrics(metrics: dict[str, float], old_prefix: str, new_prefix: str) -> dict[str, float]:
    return {
        key.replace(old_prefix, new_prefix, 1): value
        for key, value in metrics.items()
        if key.startswith(old_prefix)
    }


def module_a_stage_metrics(
    module_a_predictions: list[tuple[WindowSpec, str, str, float]] | None,
    *,
    label_mode: str = "step_id",
) -> dict[str, float]:
    if not module_a_predictions:
        return {}
    targets = [window.label for window, _prediction, _raw, _elapsed in module_a_predictions]
    predictions = [prediction for _window, prediction, _raw, _elapsed in module_a_predictions]
    if label_mode == "step_id":
        metrics = multiclass_letter_metrics(targets, predictions, prefix="module_a_step_id")
        metrics.update(
            binary_detection_metrics(
                targets,
                predictions,
                prefix="module_a",
                label_mode="step_id",
            )
        )
        metrics["module_a/num_complete_predictions"] = float(
            sum(is_module_a_completion_label(prediction, label_mode="step_id") for prediction in predictions)
        )
    else:
        metrics = binary_detection_metrics(targets, predictions, prefix="module_a", label_mode="legacy")
        metrics["module_a/num_complete_predictions"] = float(
            sum(prediction == "COMPLETE" for prediction in predictions)
        )
    metrics["module_a/num_windows"] = float(len(module_a_predictions))
    return metrics


def module_b_stage_metrics(events: list[PipelineEvent]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "module_b_given_module_a_trigger/num_triggers": float(len(events)),
        "module_b_given_module_a_trigger/num_valid_windows": float(
            sum(1 for event in events if event.pred_global_start is not None and event.pred_global_end is not None)
        ),
        "module_b_given_module_a_trigger/invalid_rate": safe_div(
            sum(1 for event in events if event.pred_global_start is None or event.pred_global_end is None),
            len(events),
        ),
    }
    evaluable = [
        event for event in events
        if event.matched_gt_start is not None and event.matched_gt_end is not None
    ]
    if not evaluable:
        return metrics
    targets = [(float(event.matched_gt_start), float(event.matched_gt_end)) for event in evaluable]
    predictions = [
        (
            None
            if event.pred_global_start is None or event.pred_global_end is None
            else (float(event.pred_global_start), float(event.pred_global_end))
        )
        for event in evaluable
    ]
    temporal = temporal_metrics(targets, predictions)
    metrics.update(prefixed_metrics(
        temporal,
        "eval_temporal",
        "module_b_given_module_a_trigger",
    ))
    return metrics


def compute_metrics(
    events: list[PipelineEvent],
    gt_events: list[GroundTruthEvent],
    module_a_predictions: list[tuple[WindowSpec, str, str, float]] | None = None,
    *,
    label_mode: str = "step_id",
) -> dict[str, Any]:
    eligible = eligible_gt(gt_events)
    matched = [event for event in events if event.matched_gt_index is not None]
    ious = [event.matched_iou for event in matched]
    metrics: dict[str, Any] = {
        "num_gt_events": len(eligible),
        "num_gt_mistakes": sum(1 for gt in eligible if gt.gt_mistake),
        "num_gt_correct": sum(1 for gt in eligible if not gt.gt_mistake),
        "num_ineligible_gt_events": sum(1 for gt in gt_events if not gt.eligible),
        "num_predictions": len(events),
        "num_predictions_with_window": sum(1 for event in events if event.pred_global_start is not None),
        "num_matched_predictions": len(matched),
        "num_duplicates": sum(1 for event in events if event.duplicate),
        "duplicate_detection_rate": safe_div(sum(1 for event in events if event.duplicate), len(events)),
        "module_c_invalid_json_rate": safe_div(
            sum(1 for event in events if event.pred_global_start is not None and event.module_c_mistake is None),
            sum(1 for event in events if event.pred_global_start is not None),
        ),
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
    }
    metrics.update(module_a_stage_metrics(module_a_predictions, label_mode=label_mode))
    metrics.update(module_b_stage_metrics(events))
    boundary_events = [
        event for event in matched
        if event.pred_global_start is not None
        and event.pred_global_end is not None
        and event.matched_gt_start is not None
        and event.matched_gt_end is not None
    ]
    if boundary_events:
        metrics["boundary_mae/start_seconds"] = sum(
            abs(float(event.pred_global_start) - float(event.matched_gt_start)) for event in boundary_events
        ) / len(boundary_events)
        metrics["boundary_mae/end_seconds"] = sum(
            abs(float(event.pred_global_end) - float(event.matched_gt_end)) for event in boundary_events
        ) / len(boundary_events)
        metrics["boundary_mae/center_seconds"] = sum(
            abs(
                ((float(event.pred_global_start) + float(event.pred_global_end)) / 2.0)
                - ((float(event.matched_gt_start) + float(event.matched_gt_end)) / 2.0)
            )
            for event in boundary_events
        ) / len(boundary_events)
        metrics["boundary_mae/duration_seconds"] = sum(
            abs(
                (float(event.pred_global_end) - float(event.pred_global_start))
                - (float(event.matched_gt_end) - float(event.matched_gt_start))
            )
            for event in boundary_events
        ) / len(boundary_events)
    else:
        metrics["boundary_mae/start_seconds"] = 0.0
        metrics["boundary_mae/end_seconds"] = 0.0
        metrics["boundary_mae/center_seconds"] = 0.0
        metrics["boundary_mae/duration_seconds"] = 0.0
    reasoning_pairs = [
        (str(event.gt_reasoning or ""), str(event.module_c_reasoning or ""))
        for event in matched
        if event.module_c_reasoning
    ]
    metrics["reasoning/non_empty_rate"] = safe_div(
        sum(1 for event in matched if event.module_c_reasoning),
        len(matched),
    )
    metrics["reasoning/lexical_overlap"] = (
        sum(lexical_overlap(reference, prediction) for reference, prediction in reasoning_pairs) / len(reasoning_pairs)
        if reasoning_pairs
        else 0.0
    )
    latencies = [
        event.trigger_time - float(event.matched_gt_end)
        for event in matched
        if event.matched_gt_end is not None
    ]
    metrics.update(latency_summary(latencies, prefix="latency"))
    metrics["latency_mean_seconds"] = metrics["latency/mean_seconds"]
    metrics["early_trigger_rate"] = metrics["latency/early_rate"]
    metrics.update(duration_summary(
        [event.module_a_prediction_seconds for event in events if event.module_a_prediction_seconds is not None],
        prefix="prediction_runtime/module_a",
    ))
    metrics.update(duration_summary(
        [event.module_b_prediction_seconds for event in events if event.module_b_prediction_seconds is not None],
        prefix="prediction_runtime/module_b",
    ))
    metrics.update(duration_summary(
        [event.module_c_prediction_seconds for event in events if event.module_c_prediction_seconds is not None],
        prefix="prediction_runtime/module_c",
    ))
    metrics.update(duration_summary(
        [event.pipeline_prediction_seconds for event in events if event.pipeline_prediction_seconds is not None],
        prefix="prediction_runtime/pipeline_completed_action",
    ))

    for threshold in IOU_THRESHOLDS:
        suffix = f"{threshold:.1f}"
        matched_gt_indices = {
            event.matched_gt_index
            for event in matched
            if event.matched_gt_index is not None and event.matched_iou >= threshold
        }
        temporal_tp = len(matched_gt_indices)
        temporal_fp = sum(
            1 for event in events
            if event.pred_global_start is not None
            and (event.matched_gt_index is None or event.matched_iou < threshold or event.duplicate)
        )
        temporal_fn = len(eligible) - temporal_tp
        temporal_precision = safe_div(temporal_tp, temporal_tp + temporal_fp)
        temporal_recall = safe_div(temporal_tp, temporal_tp + temporal_fn)
        metrics[f"temporal/precision_at_iou_{suffix}"] = temporal_precision
        metrics[f"temporal/recall_at_iou_{suffix}"] = temporal_recall
        metrics[f"temporal/f1_at_iou_{suffix}"] = safe_div(
            2 * temporal_precision * temporal_recall,
            temporal_precision + temporal_recall,
        )
        metrics[f"recall_at_iou_{suffix}"] = temporal_recall
        metrics[f"missed_step_rate_at_iou_{suffix}"] = safe_div(temporal_fn, len(eligible))
        threshold_latencies = [
            event.trigger_time - float(event.matched_gt_end)
            for event in matched
            if event.matched_iou >= threshold and event.matched_gt_end is not None
        ]
        metrics.update(latency_summary(threshold_latencies, prefix=f"latency_at_iou_{suffix}"))

        conditional_events = [
            event for event in matched
            if event.matched_iou >= threshold and event.gt_mistake is not None
        ]
        c_tp = sum(1 for event in conditional_events if event.gt_mistake and event.module_c_mistake is True)
        c_fp = sum(1 for event in conditional_events if not event.gt_mistake and event.module_c_mistake is True)
        c_tn = sum(1 for event in conditional_events if not event.gt_mistake and event.module_c_mistake is not True)
        c_fn = sum(1 for event in conditional_events if event.gt_mistake and event.module_c_mistake is not True)
        c_stats = binary_stats(c_tp, c_fp, c_tn, c_fn)
        for key, value in c_stats.items():
            metrics[f"module_c_given_match_iou_{suffix}/{key}"] = value
        metrics[f"mistake_precision_given_match_iou_{suffix}"] = c_stats["precision"]
        metrics[f"mistake_recall_given_match_iou_{suffix}"] = c_stats["recall"]
        metrics[f"mistake_f1_given_match_iou_{suffix}"] = c_stats["f1"]
        metrics[f"module_c_given_match_iou_{suffix}/confusion_matrix"] = confusion_matrix(c_tp, c_fp, c_tn, c_fn)

        e_tp = sum(
            1 for event in events
            if event.gt_mistake is True and event.matched_iou >= threshold and event.module_c_mistake is True
        )
        e_fp = sum(
            1 for event in events
            if event.module_c_mistake is True
            and (
                event.matched_gt_index is None
                or event.matched_iou < threshold
                or event.gt_mistake is not True
                or event.duplicate
            )
        )
        detected_mistake_gt = {
            event.matched_gt_index
            for event in events
            if event.matched_gt_index is not None
            and event.matched_iou >= threshold
            and event.gt_mistake is True
            and event.module_c_mistake is True
        }
        total_gt_mistakes = sum(1 for gt in eligible if gt.gt_mistake)
        e_fn = total_gt_mistakes - len(detected_mistake_gt)
        total_gt_correct = sum(1 for gt in eligible if not gt.gt_mistake)
        e_tn = max(0, total_gt_correct - e_fp)
        e_stats = binary_stats(e_tp, e_fp, e_tn, e_fn)
        for key, value in e_stats.items():
            metrics[f"end_to_end_mistake_at_iou_{suffix}/{key}"] = value
        metrics[f"end_to_end_mistake_precision_at_iou_{suffix}"] = e_stats["precision"]
        metrics[f"end_to_end_mistake_recall_at_iou_{suffix}"] = e_stats["recall"]
        metrics[f"end_to_end_mistake_f1_at_iou_{suffix}"] = e_stats["f1"]
        metrics[f"end_to_end_mistake_accuracy_at_iou_{suffix}"] = e_stats["accuracy"]
        metrics[f"end_to_end_mistake_specificity_at_iou_{suffix}"] = e_stats["specificity"]
        metrics[f"end_to_end_mistake_balanced_accuracy_at_iou_{suffix}"] = e_stats["balanced_accuracy"]
        metrics[f"end_to_end_mistake_mcc_at_iou_{suffix}"] = e_stats["mcc"]
        metrics[f"end_to_end_mistake_false_positive_rate_at_iou_{suffix}"] = e_stats["false_positive_rate"]
        metrics[f"end_to_end_mistake_false_negative_rate_at_iou_{suffix}"] = e_stats["false_negative_rate"]
        metrics[f"end_to_end_mistake_at_iou_{suffix}/confusion_matrix"] = end_to_end_confusion_matrix(
            events=events,
            gt_events=gt_events,
            threshold=threshold,
        )
    return metrics


def event_row(event: PipelineEvent) -> dict[str, Any]:
    row = asdict(event)
    if event.module_b_prediction is not None:
        row["module_b_prediction"] = list(event.module_b_prediction)
    return json_safe(row)


def write_outputs(args: argparse.Namespace, events: list[PipelineEvent], metrics: dict[str, Any], split: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else output_dir / "pipeline_metrics.json"
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else output_dir / "pipeline_events.jsonl"
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "pipeline_events.csv"
    payload = {
        "args": json_safe(vars(args)),
        "split": json_safe(split),
        "metrics": json_safe(metrics),
        "events": [event_row(event) for event in events],
    }
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    with open(output_jsonl, "w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event_row(event), sort_keys=True, ensure_ascii=False) + "\n")
    rows = [event_row(event) for event in events]
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with open(output_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved aggregate metrics to: {output_json}")
    print(f"Saved event JSONL to: {output_jsonl}")
    if rows:
        print(f"Saved event CSV to: {output_csv}")


def main() -> None:
    args = parse_args()
    provider = make_provider(args)
    dataset = EgoOopsModuleADataset(provider, config=module_a_config(args))
    split, split_path = load_or_create_video_split(dataset, args)
    val_ids = set(split["val_video_ids"])
    windows = val_windows(dataset, val_ids, args.max_samples)
    gt_events = gt_events_for_records(dataset.records, val_ids, label_mode=args.module_a_label_mode)
    print(f"Online pipeline split: {split_path}")
    print(f"Module A label mode: {args.module_a_label_mode}")
    print(f"Validation windows: {len(windows)}")
    print(f"Eligible GT events: {len(eligible_gt(gt_events))}")
    print(f"GT mistakes: {sum(1 for gt in eligible_gt(gt_events) if gt.gt_mistake)}")
    module_a_predictions: list[tuple[WindowSpec, str, str, float]] = []
    if args.skip_inference:
        events: list[PipelineEvent] = []
    elif args.reuse_events_json:
        reuse_path = Path(args.reuse_events_json)
        print(
            f"Reusing A/B events from {reuse_path}; re-running Module C only "
            f"(max_frames_c={args.max_frames_c}, vision_resize_c={args.vision_resize_c})."
        )
        events = load_events_for_module_c_reuse(reuse_path)
        if args.max_samples >= 0:
            events = events[: args.max_samples]
        module_a_predictions = reconstruct_module_a_predictions(windows=windows, events=events)
        windows_by_key = {(window.video_id, window.step_index): window for window in windows}
        video_paths = attach_video_paths(events, windows_by_key, records=dataset.records)
        missing = sum(
            1
            for index, event in enumerate(events)
            if event.pred_global_start is not None and index not in video_paths
        )
        if missing:
            print(f"Warning: {missing} events missing video paths; Module C will skip them.")
        run_module_c(args, events, video_paths)
        match_events(events, gt_events)
    else:
        module_a_predictions = run_module_a(args, windows, dataset)
        if args.module_a_label_mode == "step_id" and args.online_reset_from_b:
            triggers, events = select_triggers_with_online_b_reset(args, module_a_predictions)
            print(f"Module A triggers (online B-reset): {len(triggers)}")
        else:
            triggers = [
                item for item in module_a_predictions
                if is_module_a_completion_label(item[1], label_mode=args.module_a_label_mode)
            ]
            print(f"Module A triggers: {len(triggers)}")
            events = run_module_b(args, triggers)
        windows_by_key = {(window.video_id, window.step_index): window for window, _pred, _raw, _elapsed in triggers}
        run_module_c(args, events, attach_video_paths(events, windows_by_key, records=dataset.records))
        match_events(events, gt_events)
    metrics = compute_metrics(
        events,
        gt_events,
        module_a_predictions=module_a_predictions,
        label_mode=args.module_a_label_mode,
    )
    print(json.dumps(json_safe(metrics), indent=2, sort_keys=True))
    write_outputs(args, events, metrics, split)


if __name__ == "__main__":
    main()
