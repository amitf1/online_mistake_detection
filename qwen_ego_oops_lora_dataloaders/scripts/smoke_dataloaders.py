from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_omd_dataloaders.build import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MISTAKE_CLASSES_PATH,
    DEFAULT_VIDEO_ROOT,
    build_dataloaders,
    build_datasets,
    build_processor,
)
from qwen_omd_dataloaders.config import VideoSamplingConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Qwen OMD dataloaders.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--mistake-classes", default=DEFAULT_MISTAKE_CLASSES_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--load-processor", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--resize-short-side", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = build_datasets(
        metadata_path=args.metadata,
        mistake_classes_path=args.mistake_classes,
        video_root=args.video_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        max_videos=args.max_videos,
    )

    for name, dataset in datasets.items():
        print(f"Module {name}: {len(dataset)} examples")
        if len(dataset) == 0:
            raise SystemExit(f"Module {name} produced no examples")
        example = dataset[0]
        print(f"  first video={example.video_id} step={example.step_index}")
        print(f"  window={example.window_start:.2f}-{example.window_end:.2f}s")
        print(f"  prompt={example.prompt_text[:160]!r}")
        print(f"  target={example.target_text!r}")
        validate_example(name, example.target_text)

    if not args.load_processor:
        print("Dataset-only smoke test passed. Use --load-processor to test real Qwen batches.")
        return

    processor = build_processor(args.model_id)
    video_config = VideoSamplingConfig(
        sample_fps=2.0,
        max_frames=args.max_frames,
        resize_short_side=args.resize_short_side,
    )
    loaders = build_dataloaders(
        processor=processor,
        datasets=datasets,
        video_config_a=video_config,
        video_config_b=video_config,
        video_config_c=video_config,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        include_metadata=True,
    )

    for name, loader in loaders.items():
        batch = next(iter(loader))
        print(f"Batch {name}:")
        for key, value in batch.items():
            if key == "debug":
                debug = value[0]
                print(f"  debug_frames={len(debug.frame_timestamps)} target={debug.target_text!r}")
                if len(debug.frame_timestamps) > args.max_frames:
                    raise SystemExit(f"Module {name} exceeded max_frames")
                continue
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            print(f"  {key}: {shape}")
        labels = batch["labels"]
        if not (labels == -100).any():
            raise SystemExit(f"Module {name} labels were not masked")
        if not (labels != -100).any():
            raise SystemExit(f"Module {name} has no supervised target tokens")

    print("Processor smoke test passed.")


def validate_example(module: str, target: str) -> None:
    if module == "A" and target not in {"WAIT", "COMPLETE"}:
        raise SystemExit(f"Invalid Module A target: {target}")
    if module == "B" and not (re.match(r"^<time_\d{3}> to <time_\d{3}>$", target) or target == "<no_action>"):
        raise SystemExit(f"Invalid Module B target: {target}")
    if module == "C":
        parsed = json.loads(target)
        if set(parsed) != {"mistake", "reasoning"}:
            raise SystemExit(f"Invalid Module C JSON target: {target}")
        if not isinstance(parsed["mistake"], bool) or not isinstance(parsed["reasoning"], str):
            raise SystemExit(f"Invalid Module C JSON field types: {target}")


if __name__ == "__main__":
    main()
