#!/usr/bin/env python3
"""Minimal Module A seq-length probe: longest windows + blacklight, at target frame/resize settings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from measure_module_a_seq_lengths import (  # noqa: E402
    measure_example,
    percentile,
    release_memory,
)
from train_module_a_unsloth import (  # noqa: E402
    build_module_a_dataset,
    load_or_create_video_split,
    training_example_to_conversation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--vision-resize", type=int, default=512)
    parser.add_argument("--max-frames", type=int, nargs="+", default=[24, 32])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[3072, 4096, 5120, 6144, 8192, 12288])
    parser.add_argument("--top-k", type=int, default=80, help="Longest windows per split to measure.")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--module-a-label-mode", default="step_id")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def select_stress_examples(conversations: list[dict], top_k: int) -> list[dict]:
    by_duration = sorted(
        conversations,
        key=lambda item: float(item["window_end"]) - float(item["window_start"]),
        reverse=True,
    )
    selected: list[dict] = []
    seen: set[tuple] = set()

    def add(item: dict) -> None:
        key = (item["video_id"], item["step_index"], item["window_start"], item["window_end"])
        if key in seen:
            return
        seen.add(key)
        selected.append(item)

    for item in by_duration[:top_k]:
        add(item)
    # Longest prompts tend to be blacklight / late already-done lists.
    for item in conversations:
        if str(item.get("task_id")) == "blacklight":
            add(item)
    return selected


def summarize(rows: list[dict], seq_lengths: list[int]) -> dict:
    full = [int(r["full_tokens"]) for r in rows]
    out = {
        "num_examples": len(rows),
        "full_tokens": {
            "max": max(full, default=0),
            "p50": percentile(full, 50),
            "p90": percentile(full, 90),
            "p95": percentile(full, 95),
            "p99": percentile(full, 99),
        },
        "candidate_seq_lengths": {},
    }
    for seq in seq_lengths:
        over = sum(1 for value in full if value > seq)
        out["candidate_seq_lengths"][str(seq)] = {
            "full_over_limit_count": over,
            "full_over_limit_rate": over / len(full) if full else 0.0,
        }
    return out


def main() -> None:
    args = parse_args()
    from transformers import AutoProcessor

    ns = argparse.Namespace(
        metadata="/workspace/ego_oops/EgoOops-annotations/meta/metadata_edited.json"
        if Path("/workspace/ego_oops/EgoOops-annotations/meta/metadata_edited.json").exists()
        else str(PROJECT_ROOT.parent / "ego_oops/EgoOops-annotations/meta/metadata_edited.json"),
        mistake_classes="/workspace/ego_oops/EgoOops-annotations/meta/mistake_classes.json"
        if Path("/workspace/ego_oops/EgoOops-annotations/meta/mistake_classes.json").exists()
        else str(PROJECT_ROOT.parent / "ego_oops/EgoOops-annotations/meta/mistake_classes.json"),
        video_root=args.video_root,
        video_ids=None,
        task_ids=None,
        max_videos=args.max_videos,
        stride_seconds=5.0,
        completion_margin=0.10,
        negative_to_positive_ratio=2,
        keep_last_wait_windows=2,
        seed=args.seed,
        module_a_label_mode=args.module_a_label_mode,
        val_fraction=0.2,
        val_videos_per_task=2,
        split_file=None,
        regenerate_split=False,
    )
    dataset = build_module_a_dataset(ns)
    split, split_path = load_or_create_video_split(dataset, ns)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    report = {
        "mode": "stress_subset",
        "note": (
            "Measures longest windows + all blacklight examples only; "
            "use smallest seq with 0% over here, then prefer +1 bucket for safety."
        ),
        "split_file": str(split_path),
        "vision_resize": args.vision_resize,
        "top_k": args.top_k,
        "frame_settings": {},
    }

    for max_frames in args.max_frames:
        print(f"\nMeasuring stress subset @ frames={max_frames} resize={args.vision_resize}")
        setting_report = {}
        for split_name, video_ids in (
            ("train", set(split["train_video_ids"])),
            ("validation", set(split["val_video_ids"])),
        ):
            conversations = [
                training_example_to_conversation(
                    example,
                    fps=args.fps,
                    min_frames=args.min_frames,
                    max_frames=max_frames,
                )
                for example in dataset
                if example.video_id in video_ids
            ]
            selected = select_stress_examples(conversations, args.top_k)
            print(f"  {split_name}: measuring {len(selected)}/{len(conversations)} examples")
            rows = []
            for index, example in enumerate(selected, start=1):
                rows.append(
                    measure_example(
                        processor=processor,
                        example=example,
                        resize=args.vision_resize,
                        frame_setting=f"f{max_frames}",
                        max_frames=max_frames,
                    )
                )
                if index % 25 == 0:
                    print(f"  Measured {split_name}: {index}/{len(selected)}")
                    release_memory()
            setting_report[split_name] = summarize(rows, sorted(set(args.seq_lengths)))
            print(
                f"  {split_name} full max={setting_report[split_name]['full_tokens']['max']} "
                f"p95={setting_report[split_name]['full_tokens']['p95']}"
            )
        report["frame_settings"][f"frames{max_frames}"] = setting_report

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSaved: {output}")
    print(json.dumps(report["frame_settings"], indent=2))


if __name__ == "__main__":
    main()
