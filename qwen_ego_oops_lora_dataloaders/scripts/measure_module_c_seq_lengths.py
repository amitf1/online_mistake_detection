from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from qwen_omd_dataloaders.build import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MISTAKE_CLASSES_PATH,
    DEFAULT_VIDEO_ROOT,
)

from train_module_c_unsloth import (  # noqa: E402
    build_module_c_dataset,
    conversations_for_video_ids,
    load_or_create_video_split,
    summarize_conversations,
    summarize_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Module C processor token lengths before training.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--mistake-classes", default=DEFAULT_MISTAKE_CLASSES_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-videos-per-task", type=int, default=2)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--regenerate-split", action="store_true")
    parser.add_argument("--module-c-min-duration-seconds", type=float, default=0.5)
    parser.add_argument("--module-c-jitter-ratio-min", type=float, default=0.05)
    parser.add_argument("--module-c-jitter-ratio-max", type=float, default=0.10)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument(
        "--max-frames",
        type=int,
        nargs="+",
        default=[16],
        help="One or more training frame caps to compare.",
    )
    parser.add_argument(
        "--eval-max-frames",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Validation frame caps. Omit to mirror --max-frames, pass one value to use for all "
            "settings, or pass the same number of values as --max-frames."
        ),
    )
    parser.add_argument("--vision-resize", type=int, default=384)
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[3072, 4096, 5120, 6144, 8192],
        help="Candidate MAX_SEQ_LENGTH values to summarize.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=-1,
        help="Examples per split to measure. Use -1 for all.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--histogram-dir",
        default=None,
        help="Directory for PNG histograms. One file is written per frame setting and split.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.max_videos <= 0:
        parser.error("--max-videos must be positive.")
    if not args.max_frames or any(value < args.min_frames for value in args.max_frames):
        parser.error("--max-frames values must be >= --min-frames.")
    if args.eval_max_frames is None or not args.eval_max_frames:
        args.eval_max_frames = list(args.max_frames)
    elif len(args.eval_max_frames) == 1:
        args.eval_max_frames = args.eval_max_frames * len(args.max_frames)
    elif len(args.eval_max_frames) != len(args.max_frames):
        parser.error("--eval-max-frames must have either one value or the same count as --max-frames.")
    if any(value < args.min_frames for value in args.eval_max_frames):
        parser.error("--eval-max-frames values must be >= --min-frames.")
    if args.max_samples_per_split < -1:
        parser.error("--max-samples-per-split must be -1 or positive.")
    if not args.seq_lengths or any(value <= 0 for value in args.seq_lengths):
        parser.error("--seq-lengths must contain positive integers.")
    args.seq_lengths = sorted(set(args.seq_lengths))
    return args


def release_memory() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def select_examples(examples: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples < 0:
        return examples
    return examples[:max_samples]


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((quantile / 100.0) * len(ordered)) - 1))
    return int(ordered[index])


def with_resize(messages: list[dict[str, Any]], resize: int) -> list[dict[str, Any]]:
    copied = json.loads(json.dumps(messages))
    for message in copied:
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "video":
                part.setdefault("resized_height", resize)
                part.setdefault("resized_width", resize)
    return copied


def extract_vision_inputs(
    *,
    processor: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[Any], list[Any], dict[str, Any], list[dict[str, Any]] | None]:
    from qwen_vl_utils import process_vision_info

    signature = inspect.signature(process_vision_info)
    kwargs: dict[str, Any] = {
        "return_video_kwargs": True,
        "return_video_metadata": True,
    }
    if "image_patch_size" in signature.parameters:
        image_processor = getattr(processor, "image_processor", None)
        kwargs["image_patch_size"] = getattr(image_processor, "patch_size", 14)
    elif "size_factor" in signature.parameters:
        kwargs["size_factor"] = 28
    else:
        raise RuntimeError("Unsupported qwen_vl_utils.process_vision_info signature.")

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, **kwargs)
    images = [] if image_inputs is None else list(image_inputs)
    videos: list[Any] = []
    video_metadata: list[dict[str, Any]] = []
    if video_inputs is not None:
        for item in video_inputs:
            if isinstance(item, tuple) and len(item) == 2:
                video, metadata = item
                videos.append(video)
                video_metadata.append(metadata)
            else:
                videos.append(item)
    return images, videos, video_kwargs, video_metadata or None


def processor_sequence_length(
    *,
    processor: Any,
    messages: list[dict[str, Any]],
    resize: int,
    add_generation_prompt: bool,
) -> int:
    messages = with_resize(messages, resize)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    images, videos, video_kwargs, video_metadata = extract_vision_inputs(
        processor=processor,
        messages=messages,
    )
    proc_kwargs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    if images:
        proc_kwargs["images"] = images
    if videos:
        proc_kwargs["videos"] = videos
        proc_kwargs["do_resize"] = False
        proc_kwargs.update(video_kwargs)
        if video_metadata:
            proc_kwargs["video_metadata"] = video_metadata
    batch = processor(**proc_kwargs)
    try:
        return int(batch["input_ids"].shape[-1])
    finally:
        del batch
        release_memory()


def measure_example(
    *,
    processor: Any,
    example: dict[str, Any],
    resize: int,
    frame_setting: str,
    max_frames: int,
) -> dict[str, Any]:
    prompt_messages = [
        message
        for message in example["messages"]
        if message.get("role") != "assistant"
    ]
    input_tokens = processor_sequence_length(
        processor=processor,
        messages=prompt_messages,
        resize=resize,
        add_generation_prompt=True,
    )
    full_tokens = processor_sequence_length(
        processor=processor,
        messages=example["messages"],
        resize=resize,
        add_generation_prompt=False,
    )
    return {
        "frame_setting": frame_setting,
        "max_frames": max_frames,
        "video_id": example["video_id"],
        "task_id": example["task_id"],
        "step_index": example["step_index"],
        "label": example.get("label"),
        "window_seconds": float(example["window_end"]) - float(example["window_start"]),
        "input_tokens": input_tokens,
        "full_tokens": full_tokens,
        "target_tokens": max(0, full_tokens - input_tokens),
    }


def summarize_lengths(
    *,
    split_name: str,
    rows: list[dict[str, Any]],
    seq_lengths: list[int],
    frame_setting: str,
    max_frames: int,
) -> dict[str, Any]:
    input_lengths = [int(row["input_tokens"]) for row in rows]
    full_lengths = [int(row["full_tokens"]) for row in rows]
    target_lengths = [int(row["target_tokens"]) for row in rows]
    by_label = Counter(str(row.get("label")) for row in rows)
    candidate_counts = {}
    for seq_length in seq_lengths:
        input_over = sum(1 for value in input_lengths if value > seq_length)
        full_over = sum(1 for value in full_lengths if value > seq_length)
        target_cut_risk = sum(
            1
            for input_tokens, full_tokens in zip(input_lengths, full_lengths)
            if input_tokens <= seq_length < full_tokens
        )
        candidate_counts[str(seq_length)] = {
            "input_over_limit_count": input_over,
            "full_over_limit_count": full_over,
            "target_cut_risk_count": target_cut_risk,
            "input_over_limit_rate": input_over / len(rows) if rows else 0.0,
            "full_over_limit_rate": full_over / len(rows) if rows else 0.0,
            "target_cut_risk_rate": target_cut_risk / len(rows) if rows else 0.0,
        }
    return {
        "split": split_name,
        "frame_setting": frame_setting,
        "max_frames": max_frames,
        "num_examples": len(rows),
        "label_counts": dict(sorted(by_label.items())),
        "input_tokens": {
            "max": max(input_lengths, default=0),
            "p50": percentile(input_lengths, 50),
            "p90": percentile(input_lengths, 90),
            "p95": percentile(input_lengths, 95),
            "p99": percentile(input_lengths, 99),
        },
        "full_tokens": {
            "max": max(full_lengths, default=0),
            "p50": percentile(full_lengths, 50),
            "p90": percentile(full_lengths, 90),
            "p95": percentile(full_lengths, 95),
            "p99": percentile(full_lengths, 99),
        },
        "target_tokens": {
            "max": max(target_lengths, default=0),
            "p50": percentile(target_lengths, 50),
            "p90": percentile(target_lengths, 90),
            "p95": percentile(target_lengths, 95),
            "p99": percentile(target_lengths, 99),
        },
        "candidate_seq_lengths": candidate_counts,
    }


def measure_split(
    *,
    split_name: str,
    examples: list[dict[str, Any]],
    processor: Any,
    resize: int,
    max_samples: int,
    frame_setting: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    selected = select_examples(examples, max_samples)
    rows = []
    for index, example in enumerate(selected, start=1):
        rows.append(
            measure_example(
                processor=processor,
                example=example,
                resize=resize,
                frame_setting=frame_setting,
                max_frames=max_frames,
            )
        )
        if index % 25 == 0:
            print(f"Measured {split_name}: {index}/{len(selected)}")
    return rows


def write_csv(path: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_setting",
        "split",
        "max_frames",
        "video_id",
        "task_id",
        "step_index",
        "label",
        "window_seconds",
        "input_tokens",
        "full_tokens",
        "target_tokens",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, rows in rows_by_split.items():
            for row in rows:
                writer.writerow({"split": split_name, **row})


def write_histograms(
    *,
    output_dir: Path,
    rows_by_setting_and_split: dict[str, dict[str, list[dict[str, Any]]]],
    seq_lengths: list[int],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for frame_setting, rows_by_split in rows_by_setting_and_split.items():
        for split_name, rows in rows_by_split.items():
            fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
            series = [
                ("input_tokens", "Input tokens: video + prompt"),
                ("full_tokens", "Full tokens: video + prompt + target"),
                ("target_tokens", "Target tokens: assistant JSON/reasoning"),
            ]
            for axis, (field, title) in zip(axes, series):
                values = [int(row[field]) for row in rows]
                bins = min(40, max(5, len(set(values)))) if values else 5
                axis.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
                if field != "target_tokens":
                    for seq_length in seq_lengths:
                        axis.axvline(seq_length, color="#E45756", linestyle="--", linewidth=1)
                axis.set_title(title)
                axis.set_xlabel("tokens")
                axis.set_ylabel("examples")
                axis.grid(alpha=0.25)
            fig.suptitle(f"Module C token lengths: {frame_setting} / {split_name}")
            output_path = output_dir / f"module_c_seq_lengths_{frame_setting}_{split_name}.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            written.append(str(output_path))
    return written


def print_candidate_summary(report: dict[str, Any]) -> None:
    for frame_setting, setting_report in report["frame_settings"].items():
        print(f"\nFrame setting: {frame_setting}")
        for split_name in ("train", "validation"):
            split = setting_report["splits"][split_name]
            print(f"{split_name} ({split['num_examples']} examples)")
            print("seq_length,input_over,full_over,target_cut_risk")
            for seq_length, counts in split["candidate_seq_lengths"].items():
                print(
                    f"{seq_length},"
                    f"{counts['input_over_limit_count']},"
                    f"{counts['full_over_limit_count']},"
                    f"{counts['target_cut_risk_count']}"
                )


def main() -> None:
    args = parse_args()
    from transformers import AutoProcessor

    dataset = build_module_c_dataset(args)
    split, split_path = load_or_create_video_split(dataset, args)
    print("Module C dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module C split file: {split_path}")

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    rows_by_setting_and_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    frame_setting_reports: dict[str, Any] = {}
    base_train_summary: dict[str, Any] | None = None
    base_validation_summary: dict[str, Any] | None = None

    for train_max_frames, eval_max_frames in zip(args.max_frames, args.eval_max_frames):
        frame_setting = f"train{train_max_frames}_eval{eval_max_frames}"
        print(f"\nMeasuring frame setting: {frame_setting}")
        train_conversations = conversations_for_video_ids(
            dataset,
            set(split["train_video_ids"]),
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=train_max_frames,
        )
        val_conversations = conversations_for_video_ids(
            dataset,
            set(split["val_video_ids"]),
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=eval_max_frames,
        )
        if not train_conversations or not val_conversations:
            raise ValueError("Train/validation split produced an empty conversation split.")
        train_summary = summarize_conversations(train_conversations)
        validation_summary = summarize_conversations(val_conversations)
        base_train_summary = base_train_summary or train_summary
        base_validation_summary = base_validation_summary or validation_summary
        print("Module C train summary:")
        print(json.dumps(train_summary, indent=2, ensure_ascii=False))
        print("Module C validation summary:")
        print(json.dumps(validation_summary, indent=2, ensure_ascii=False))

        rows_by_split = {
            "train": measure_split(
                split_name="train",
                examples=train_conversations,
                processor=processor,
                resize=args.vision_resize,
                max_samples=args.max_samples_per_split,
                frame_setting=frame_setting,
                max_frames=train_max_frames,
            ),
            "validation": measure_split(
                split_name="validation",
                examples=val_conversations,
                processor=processor,
                resize=args.vision_resize,
                max_samples=args.max_samples_per_split,
                frame_setting=frame_setting,
                max_frames=eval_max_frames,
            ),
        }
        rows_by_setting_and_split[frame_setting] = rows_by_split
        frame_setting_reports[frame_setting] = {
            "train_max_frames": train_max_frames,
            "eval_max_frames": eval_max_frames,
            "train_summary": train_summary,
            "validation_summary": validation_summary,
            "splits": {
                split_name: summarize_lengths(
                    split_name=split_name,
                    rows=rows,
                    seq_lengths=args.seq_lengths,
                    frame_setting=frame_setting,
                    max_frames=train_max_frames if split_name == "train" else eval_max_frames,
                )
                for split_name, rows in rows_by_split.items()
            },
        }

    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "args": json_safe(vars(args)),
        "split_file": str(split_path),
        "dataset_summary": summarize_dataset(dataset),
        "train_summary": base_train_summary,
        "validation_summary": base_validation_summary,
        "frame_settings": frame_setting_reports,
    }
    if args.histogram_dir:
        report["histogram_files"] = write_histograms(
            output_dir=Path(args.histogram_dir),
            rows_by_setting_and_split=rows_by_setting_and_split,
            seq_lengths=args.seq_lengths,
        )
    print_candidate_summary(report)
    print("\nFull report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")
        print(f"Saved JSON report to: {output_json}")
    if args.output_csv:
        output_csv = Path(args.output_csv)
        flat_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rows_by_split in rows_by_setting_and_split.values():
            for split_name, rows in rows_by_split.items():
                flat_rows_by_split[split_name].extend(rows)
        write_csv(output_csv, dict(flat_rows_by_split))
        print(f"Saved per-example CSV to: {output_csv}")


if __name__ == "__main__":
    main()
