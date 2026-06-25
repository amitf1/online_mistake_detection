from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_omd_dataloaders.build import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MISTAKE_CLASSES_PATH,
    DEFAULT_VIDEO_ROOT,
)
from qwen_omd_dataloaders.config import ModuleAConfig  # noqa: E402
from qwen_omd_dataloaders.datasets import EgoOopsModuleADataset, EgoOopsProvider  # noqa: E402
from train_module_a_unsloth import (  # noqa: E402
    Qwen3VLMetadataVisionDataCollator,
    load_unsloth_model,
    training_example_to_conversation,
)


DEFAULT_TASKS = ("blacklight", "cardboard", "electronics", "ion", "tsumiki")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Qwen/Unsloth token counts for one EgoOops video per task."
    )
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--mistake-classes", default=DEFAULT_MISTAKE_CLASSES_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--vision-resize", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def build_collator(args: argparse.Namespace) -> Any:
    model_args = argparse.Namespace(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        seed=args.seed,
    )
    model, processor = load_unsloth_model(model_args)
    collator = Qwen3VLMetadataVisionDataCollator(
        model,
        processor,
        max_seq_length=args.max_seq_length,
        resize=args.vision_resize,
        train_on_responses_only=False,
        completion_only_loss=False,
    )
    return processor, collator


def first_example_for_task(args: argparse.Namespace, task_id: str):
    provider = EgoOopsProvider(
        metadata_path=args.metadata,
        mistake_classes_path=args.mistake_classes,
        video_root=args.video_root,
        task_ids={task_id},
        max_videos=1,
        require_existing_videos=True,
    )
    dataset = EgoOopsModuleADataset(provider, config=ModuleAConfig(seed=args.seed))
    return dataset[0] if len(dataset) else None


def main() -> None:
    args = parse_args()
    processor, collator = build_collator(args)
    tokenizer = getattr(processor, "tokenizer", processor)
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    pad_token_id = tokenizer.pad_token_id

    totals: list[int] = []
    video_totals: list[int] = []
    print("task,video_id,total_tokens,video_pad_tokens,target")
    for task_id in args.tasks:
        example = first_example_for_task(args, task_id)
        if example is None:
            print(f"{task_id},NO_EXAMPLE,0,0,NA")
            continue

        conversation = training_example_to_conversation(
            example,
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        )
        batch = collator([conversation])
        input_ids = batch["input_ids"][0]
        total_tokens = (
            int((input_ids != pad_token_id).sum().item())
            if pad_token_id is not None
            else int(input_ids.numel())
        )
        video_tokens = int((input_ids == video_token_id).sum().item())
        totals.append(total_tokens)
        video_totals.append(video_tokens)
        print(
            f"{task_id},{example.video_id},{total_tokens},{video_tokens},{example.target_text}"
        )

    if totals:
        print(f"average,,{mean(totals):.1f},{mean(video_totals):.1f},")


if __name__ == "__main__":
    main()
