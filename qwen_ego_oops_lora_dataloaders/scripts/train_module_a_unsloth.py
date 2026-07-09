from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
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
from qwen_omd_dataloaders.schema import TrainingExample  # noqa: E402

MODULE_A_SYSTEM_PROMPT = (
    "You review a person following procedural instructions. Given the video, completed steps, "
    "current step, and pending steps, answer WAIT or COMPLETE. Say COMPLETE when the attempt at "
    "the current step has ended, even if it included mistakes; otherwise say WAIT."
)
DEFAULT_OUTPUT_DIR = "/home/amit/online_mistake_detection/outputs/module_a_qwen35_2b_lora_wait_complete_vision"
MODULE_A_SPLIT_VERSION = 1
MODULE_A_LABELS = ("WAIT", "COMPLETE")
MODULE_A_POSITIVE_LABEL = "COMPLETE"


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_unsloth_logits(args: argparse.Namespace) -> None:
    if args.positive_loss_weight != 1.0 or args.label_score_eval:
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", "/tmp/unsloth_compiled_cache_module_a_logits")
        print(
            "Unsloth logits/hidden states enabled for weighted loss / label scoring "
            f"(compile cache: {os.environ['UNSLOTH_COMPILE_LOCATION']})."
        )


def logits_for_loss(model: Any, outputs: Any) -> Any:
    import torch

    logits = outputs.logits
    output_embeddings = model.get_output_embeddings()
    vocab_size = getattr(output_embeddings, "out_features", None)
    if torch.is_tensor(logits) and logits.ndim >= 3 and (vocab_size is None or logits.shape[-1] == vocab_size):
        return logits
    if torch.is_tensor(logits) and logits.ndim >= 3:
        logits = output_embeddings(logits)
        return logits

    hidden_states = getattr(outputs, "hidden_states", None)
    if isinstance(hidden_states, list | tuple) and hidden_states:
        hidden_states = hidden_states[-1]
    if torch.is_tensor(hidden_states):
        weight = getattr(output_embeddings, "weight", None)
        if torch.is_tensor(weight):
            hidden_states = hidden_states.to(dtype=weight.dtype)
        return output_embeddings(hidden_states)
    raise RuntimeError(
        "Could not obtain logits or hidden states from Unsloth output. "
        "Weighted positive loss requires UNSLOTH_RETURN_LOGITS=1 or output_hidden_states=True."
    )


def generation_eval_examples(
    examples: list[dict[str, Any]],
    max_samples: int,
) -> list[dict[str, Any]]:
    if max_samples < 0:
        return list(examples)
    return examples[:max_samples]


def move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def offload_trainer_cuda_state(trainer: Any) -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    trainer.model.to("cpu")
    move_optimizer_state_to_device(getattr(trainer, "optimizer", None), torch.device("cpu"))
    release_cuda_memory()
    return torch.device("cuda")


def restore_trainer_cuda_state(trainer: Any, device: Any | None) -> None:
    if device is None:
        return
    trainer.model.to(device)
    move_optimizer_state_to_device(getattr(trainer, "optimizer", None), device)
    release_cuda_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5 for Module A online step detection with Unsloth."
    )

    # Data
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--mistake-classes", default=DEFAULT_MISTAKE_CLASSES_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="FPS value written into video messages. Keep low for 16GB GPUs.",
    )
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)

    # Module A simulation
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--completion-margin", type=float, default=0.10)
    parser.add_argument("--negative-to-positive-ratio", type=int, default=2)
    parser.add_argument("--keep-last-wait-windows", type=int, default=2)
    parser.add_argument(
        "--positive-oversample-factor",
        type=float,
        default=1.0,
        help="Train-only COMPLETE oversampling factor. 1.0 keeps the natural split.",
    )
    parser.add_argument(
        "--positive-loss-weight",
        type=float,
        default=1.0,
        help="Train-time loss multiplier for COMPLETE assistant response tokens.",
    )

    # Model / LoRA
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vision-resize", type=int, default=512)

    # Training
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--train-mode",
        choices=("steps", "epochs"),
        default="steps",
        help="Cap training by optimizer steps or by epochs.",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=2,
        help="Save rolling ckpt_epN checkpoints every N completed epochs.",
    )
    parser.add_argument(
        "--save-strategy",
        choices=("steps", "epoch"),
        default="steps",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--report-to",
        default="tensorboard",
        help="Comma-separated Trainer loggers, for example: tensorboard,wandb.",
    )
    parser.add_argument("--dataset-num-proc", type=int, default=1)

    # Validation / execution mode
    parser.add_argument(
        "--split-file",
        default=None,
        help="JSON file with train/validation video IDs. Created if missing.",
    )
    parser.add_argument(
        "--regenerate-split",
        action="store_true",
        help="Overwrite an existing split file using the configured split policy.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-videos-per-task",
        type=int,
        default=2,
        help="Validation videos per task for the default 50-video corpus.",
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=2,
        help="Run validation every N completed epochs.",
    )
    parser.add_argument(
        "--eval-strategy",
        choices=("steps", "epoch"),
        default="steps",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-generation-max-samples",
        type=int,
        default=-1,
        help="Generated-label metric sample cap. Use -1 for the full validation split, 0 to skip.",
    )
    parser.add_argument(
        "--generation-eval-mode",
        choices=("subprocess", "inline", "off"),
        default="subprocess",
        help="How to run generated-label validation metrics.",
    )
    parser.add_argument("--generation-eval-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--eval-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generation-eval-output-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--label-score-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate Module A by comparing WAIT/COMPLETE label likelihoods instead of generation.",
    )
    parser.add_argument("--fbeta-beta", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--wandb-log-best-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload every newly best checkpoint as a W&B model artifact.",
    )
    parser.add_argument(
        "--wandb-artifact-prefix",
        default="module-a",
        help="Prefix for W&B checkpoint artifact names.",
    )
    parser.add_argument("--keep-last-checkpoints", type=int, default=4)
    parser.add_argument("--keep-best-checkpoints", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and print examples only.")
    parser.add_argument("--print-examples", type=int, default=1)
    args = parser.parse_args()
    args.report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    if args.train_mode == "steps" and args.max_steps <= 0:
        parser.error("--max-steps must be positive when --train-mode=steps.")
    if args.train_mode == "epochs" and args.num_train_epochs <= 0:
        parser.error("--num-train-epochs must be positive when --train-mode=epochs.")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1.")
    if args.val_videos_per_task < 1:
        parser.error("--val-videos-per-task must be positive.")
    if args.eval_steps < 1:
        parser.error("--eval-steps must be positive.")
    if args.eval_epochs < 1:
        parser.error("--eval-epochs must be positive.")
    if args.eval_generation_max_samples < -1:
        parser.error("--eval-generation-max-samples must be -1, 0, or positive.")
    if args.generation_eval_only and not args.eval_checkpoint:
        parser.error("--generation-eval-only requires --eval-checkpoint.")
    if args.checkpoint_epochs < 1:
        parser.error("--checkpoint-epochs must be positive.")
    if args.keep_last_checkpoints < 0:
        parser.error("--keep-last-checkpoints cannot be negative.")
    if args.keep_best_checkpoints < 0:
        parser.error("--keep-best-checkpoints cannot be negative.")
    if args.positive_oversample_factor < 1.0:
        parser.error("--positive-oversample-factor must be >= 1.0.")
    if args.positive_loss_weight <= 0:
        parser.error("--positive-loss-weight must be positive.")
    if args.fbeta_beta <= 0:
        parser.error("--fbeta-beta must be positive.")
    if args.greater_is_better is None:
        args.greater_is_better = not args.metric_for_best_model.endswith("loss")
    return args


def is_bf16_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def summarize_trainable_parameters(model: Any) -> dict[str, int]:
    summary = {
        "total": 0,
        "visual": 0,
        "language": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        summary["total"] += count
        if ".visual" in name or ".vision" in name:
            summary["visual"] += count
        elif "language_model" in name or "lm_head" in name or "embed_tokens" in name:
            summary["language"] += count
        else:
            summary["other"] += count
    return summary


def build_module_a_dataset(args: argparse.Namespace) -> EgoOopsModuleADataset:
    provider = EgoOopsProvider(
        metadata_path=args.metadata,
        mistake_classes_path=args.mistake_classes,
        video_root=args.video_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        max_videos=args.max_videos,
        require_existing_videos=True,
    )
    config = ModuleAConfig(
        stride_seconds=args.stride_seconds,
        completion_margin=args.completion_margin,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        keep_last_wait_windows=args.keep_last_wait_windows,
        seed=args.seed,
    )
    return EgoOopsModuleADataset(provider, config=config)


def default_split_file(args: argparse.Namespace) -> Path:
    return PROJECT_ROOT / "splits" / f"module_a_video_split_seed{args.seed}.json"


def grouped_video_ids(dataset: EgoOopsModuleADataset) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in dataset.records:
        grouped[record.task_id].add(record.video_id)
    return {
        task_id: sorted(video_ids)
        for task_id, video_ids in sorted(grouped.items())
    }


def split_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "per_task_video_holdout",
        "val_fraction": args.val_fraction,
        "val_videos_per_task": args.val_videos_per_task,
    }


def data_config_for_split(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_videos": args.max_videos,
        "task_ids": sorted(args.task_ids or []),
        "video_ids": sorted(args.video_ids or []),
        "stride_seconds": args.stride_seconds,
        "completion_margin": args.completion_margin,
        "negative_to_positive_ratio": args.negative_to_positive_ratio,
        "keep_last_wait_windows": args.keep_last_wait_windows,
    }


def make_video_split(dataset: EgoOopsModuleADataset, args: argparse.Namespace) -> dict[str, Any]:
    grouped = grouped_video_ids(dataset)
    if not grouped:
        raise ValueError("Cannot create a split from an empty dataset.")

    rng = random.Random(args.seed)
    tasks: dict[str, dict[str, list[str]]] = {}
    train_video_ids: list[str] = []
    val_video_ids: list[str] = []
    for task_id, video_ids in grouped.items():
        shuffled = list(video_ids)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError(f"Task {task_id!r} has too few videos for validation splitting.")
        val_count = min(args.val_videos_per_task, len(shuffled) - 1)
        if len(grouped) != 5 or len(shuffled) != 10:
            val_count = max(1, round(len(shuffled) * args.val_fraction))
            val_count = min(val_count, len(shuffled) - 1)
        val = sorted(shuffled[:val_count])
        train = sorted(shuffled[val_count:])
        tasks[task_id] = {"train": train, "val": val}
        train_video_ids.extend(train)
        val_video_ids.extend(val)

    return {
        "version": MODULE_A_SPLIT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "split_policy": split_policy(args),
        "metadata_path": str(Path(args.metadata).resolve()),
        "video_root": str(Path(args.video_root).resolve()),
        "data_config": data_config_for_split(args),
        "train_video_ids": sorted(train_video_ids),
        "val_video_ids": sorted(val_video_ids),
        "tasks": tasks,
    }


def validate_video_split(split: dict[str, Any], dataset: EgoOopsModuleADataset) -> None:
    train_ids = set(split.get("train_video_ids", []))
    val_ids = set(split.get("val_video_ids", []))
    if not train_ids or not val_ids:
        raise ValueError("Split file must contain non-empty train_video_ids and val_video_ids.")
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Split file has overlapping train/val video IDs: {sorted(overlap)}")

    available = {record.video_id for record in dataset.records}
    missing = (train_ids | val_ids) - available
    if missing:
        raise ValueError(f"Split file references videos not in the selected dataset: {sorted(missing)}")

    grouped = grouped_video_ids(dataset)
    for task_id, task_split in split.get("tasks", {}).items():
        if task_id not in grouped:
            raise ValueError(f"Split file references missing task: {task_id}")
        task_ids = set(task_split.get("train", [])) | set(task_split.get("val", []))
        outside_task = task_ids - set(grouped[task_id])
        if outside_task:
            raise ValueError(
                f"Split file places videos under the wrong task {task_id!r}: {sorted(outside_task)}"
            )


def load_or_create_video_split(dataset: EgoOopsModuleADataset, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    split_path = Path(args.split_file) if args.split_file else default_split_file(args)
    if split_path.exists() and not args.regenerate_split:
        with open(split_path, "r", encoding="utf-8") as file:
            split = json.load(file)
        validate_video_split(split, dataset)
        print(f"Loaded Module A split from: {split_path}")
        return split, split_path

    split = make_video_split(dataset, args)
    validate_video_split(split, dataset)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(split, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Wrote Module A split to: {split_path}")
    return split, split_path


def split_conversations(
    conversations: list[dict[str, Any]],
    split: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids = set(split["train_video_ids"])
    val_ids = set(split["val_video_ids"])
    train = [item for item in conversations if item["video_id"] in train_ids]
    val = [item for item in conversations if item["video_id"] in val_ids]
    if not train or not val:
        raise ValueError("Train/validation split produced an empty conversation split.")
    return train, val


def oversample_positive_conversations(
    conversations: list[dict[str, Any]],
    *,
    factor: float,
    seed: int,
) -> list[dict[str, Any]]:
    if factor <= 1.0:
        return list(conversations)
    positives = [
        item for item in conversations
        if label_to_positive(str(item.get("target", "")))
    ]
    if not positives:
        return list(conversations)

    extra_count = int(math.floor((factor - 1.0) * len(positives)))
    fractional = (factor - 1.0) * len(positives) - extra_count
    rng = random.Random(seed)
    if fractional > 0 and rng.random() < fractional:
        extra_count += 1
    extras = [copy.deepcopy(rng.choice(positives)) for _ in range(extra_count)]
    expanded = list(conversations) + extras
    rng.shuffle(expanded)
    return expanded


def apply_positive_loss_weights(
    conversations: list[dict[str, Any]],
    *,
    positive_weight: float,
) -> None:
    for item in conversations:
        item["loss_weight"] = (
            positive_weight
            if label_to_positive(str(item.get("target", "")))
            else 1.0
        )


def to_unsloth_conversations(
    dataset: EgoOopsModuleADataset,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    return [
        training_example_to_conversation(
            example,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        for example in dataset
    ]


def training_example_to_conversation(
    example: TrainingExample,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": MODULE_A_SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(example.video_path),
                        "video_start": example.window_start,
                        "video_end": example.window_end,
                        "fps": fps,
                        "min_frames": min_frames,
                        "max_frames": max_frames,
                    },
                    {
                        "type": "text",
                        "text": example.prompt_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": example.target_text,
                    }
                ],
            },
        ],
        "video_id": example.video_id,
        "task_id": example.task_id,
        "step_index": example.step_index,
        "window_start": example.window_start,
        "window_end": example.window_end,
        "target": example.target_text,
        "loss_weight": 1.0,
    }


def summarize_module_a(dataset: EgoOopsModuleADataset) -> dict[str, Any]:
    counts = Counter(example.target_text for example in dataset)
    videos = {example.video_id for example in dataset}
    return {
        "num_examples": len(dataset),
        "num_videos": len(videos),
        "label_counts": dict(sorted(counts.items())),
    }


def summarize_conversations(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(item["target"]) for item in conversations)
    videos_by_task: dict[str, set[str]] = defaultdict(set)
    examples_by_task = Counter()
    for item in conversations:
        task_id = str(item["task_id"])
        videos_by_task[task_id].add(str(item["video_id"]))
        examples_by_task[task_id] += 1
    return {
        "num_examples": len(conversations),
        "num_videos": len({item["video_id"] for item in conversations}),
        "label_counts": dict(sorted(label_counts.items())),
        "tasks": {
            task_id: {
                "num_videos": len(video_ids),
                "num_examples": examples_by_task[task_id],
            }
            for task_id, video_ids in sorted(videos_by_task.items())
        },
    }


def print_dry_run(
    dataset: EgoOopsModuleADataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
    limit: int,
) -> None:
    print("Module A full dataset summary:")
    print(json.dumps(summarize_module_a(dataset), indent=2, ensure_ascii=False))
    print(f"Module A split file: {split_path}")
    print("Module A split video summary:")
    print(json.dumps({
        "train_video_ids": split["train_video_ids"],
        "val_video_ids": split["val_video_ids"],
        "tasks": split["tasks"],
    }, indent=2, ensure_ascii=False))
    print("Module A train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module A validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
    for index, conversation in enumerate(train_conversations[: max(0, limit)]):
        print(f"\nExample {index}:")
        print(json.dumps(conversation, indent=2, ensure_ascii=False))


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


def save_run_config(
    *,
    args: argparse.Namespace,
    dataset: EgoOopsModuleADataset,
    conversations: list[dict[str, Any]],
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
) -> None:
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_config = {
        "args": json_safe(vars(args)),
        "dataset_summary": summarize_module_a(dataset),
        "num_conversations": len(conversations),
        "split_file": str(split_path),
        "split": json_safe(split),
        "train_summary": summarize_conversations(train_conversations),
        "validation_summary": summarize_conversations(val_conversations),
    }
    config_path = output_path / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    print(f"Saved run config to: {config_path}")


def label_to_positive(label: str) -> bool:
    return label.strip().strip("[]") == MODULE_A_POSITIVE_LABEL


def binary_detection_metrics(
    targets: list[str],
    predictions: list[str],
    *,
    prefix: str,
    beta: float = 2.0,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for target, prediction in zip(targets, predictions):
        target_positive = label_to_positive(target)
        predicted_positive = label_to_positive(prediction)
        if target_positive and predicted_positive:
            tp += 1
        elif not target_positive and predicted_positive:
            fp += 1
        elif not target_positive and not predicted_positive:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_sq = beta * beta
    fbeta = (
        (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)
        if precision + recall else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0

    return {
        f"{prefix}/accuracy": accuracy,
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/f1": f1,
        f"{prefix}/f_beta": fbeta,
        f"{prefix}/specificity": specificity,
        f"{prefix}/balanced_accuracy": balanced_accuracy,
        f"{prefix}/mcc": mcc,
        f"{prefix}/tp": float(tp),
        f"{prefix}/fp": float(fp),
        f"{prefix}/tn": float(tn),
        f"{prefix}/fn": float(fn),
        f"{prefix}/num_samples": float(total),
    }


def extract_module_a_label(text: str) -> str:
    for label in MODULE_A_LABELS:
        if label in text.strip().strip("[]").upper():
            return label
    normalized = text.strip().upper()
    if "COMPLETE" in normalized:
        return "COMPLETE"
    if "WAIT" in normalized:
        return "WAIT"
    return "WAIT"


def load_unsloth_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        use_gradient_checkpointing="unsloth",
    )
    peft_kwargs = dict(
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        ensure_weight_tying=True,
    )
    try:
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    except TypeError as error:
        if "ensure_weight_tying" not in str(error):
            raise
        peft_kwargs.pop("ensure_weight_tying")
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    trainable = summarize_trainable_parameters(model)
    print("Trainable parameter summary:")
    print(json.dumps(trainable, indent=2, sort_keys=True))
    if args.finetune_vision_layers and trainable["visual"] == 0:
        raise RuntimeError(
            "finetune_vision_layers=True, but no trainable visual parameters were found."
        )
    return model, tokenizer


def load_generation_eval_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.eval_checkpoint or args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        use_gradient_checkpointing=False,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def collapse_fps(fps_values: list[float], tol: float = 1e-4) -> float | list[float] | None:
    if not fps_values:
        return None
    first = float(fps_values[0])
    if all(math.isclose(float(value), first, rel_tol=tol, abs_tol=tol) for value in fps_values[1:]):
        return first
    return [float(value) for value in fps_values]


class Qwen3VLMetadataVisionDataCollator:
    """Unsloth vision collator wrapper that preserves Qwen3-VL video metadata."""

    def __init__(self, *args: Any, include_response_loss_weight: bool = False, **kwargs: Any) -> None:
        from unsloth.trainer import UnslothVisionDataCollator

        self._base = UnslothVisionDataCollator(*args, **kwargs)
        self.include_response_loss_weight = include_response_loss_weight

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _extract_images_videos_metadata(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[Any], list[Any], dict[str, Any], list[dict[str, Any]] | None]:
        from qwen_vl_utils import process_vision_info

        messages = copy.deepcopy(messages)
        if isinstance(self._base.image_size, int):
            for message in messages:
                for part in message.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "video":
                        part.setdefault("resized_height", self._base.image_size)
                        part.setdefault("resized_width", self._base.image_size)

        signature = inspect.signature(process_vision_info)
        kwargs: dict[str, Any] = {
            "return_video_kwargs": True,
            "return_video_metadata": True,
        }
        if "image_patch_size" in signature.parameters:
            image_processor = getattr(self._base.processor, "image_processor", None)
            kwargs["image_patch_size"] = getattr(image_processor, "patch_size", self._base.patch_size)
        elif "size_factor" in signature.parameters:
            kwargs["size_factor"] = self._base.patch_size * 2
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

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "prompt" in examples[0] and "completion" in examples[0]:
            return self._base(examples)

        texts = []
        images = []
        videos = []
        video_metadatas = []
        video_processor_kwargs: dict[str, Any] = {}
        fps_values: list[float] = []

        for example in examples:
            messages = self._base._select_messages_or_raw(example)
            if len(messages) != 0:
                messages = self._base._validate_and_normalize_first_message(messages)
                if self._base.assistant_single_content:
                    messages = self._base._collapse_assistant_content(messages)
                messages = self._base._clean_none_keys(messages)

            texts.append(self._base.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ))
            image, video, video_kwargs, video_metadata = self._extract_images_videos_metadata(messages)
            if image:
                images.append(image)
            if video:
                videos.append(video)
                for key, value in video_kwargs.items():
                    if key == "fps":
                        fps_values.extend(value)
                    else:
                        video_processor_kwargs[key] = value
                if video_metadata:
                    video_metadatas.extend(video_metadata)

        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "truncation": self._base.truncation,
            "max_length": self._base.max_seq_length,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
            proc_kwargs["do_resize"] = False
            proc_kwargs.update(video_processor_kwargs)
            if video_metadatas:
                proc_kwargs["video_metadata"] = video_metadatas
            else:
                collapsed_fps = collapse_fps(fps_values)
                if collapsed_fps is not None:
                    proc_kwargs["fps"] = collapsed_fps
        if self._base.pad_to_multiple_of is not None:
            proc_kwargs["pad_to_multiple_of"] = self._base.pad_to_multiple_of

        batch = self._base.processor(**proc_kwargs)
        if "pixel_values" in batch:
            batch = self._base._cast_pixel_values_dtype_inplace(batch)
        if "pixel_values_videos" in batch:
            batch = self._base._cast_pixel_values_dtype_inplace(batch, "pixel_values_videos")

        import torch

        labels = batch["input_ids"].clone()
        padding_ids = self._base._get_padding_token_ids_on_device(labels.device)
        labels[torch.isin(labels, padding_ids)] = self._base.ignore_index
        batch["labels"] = labels
        if self._base.train_on_responses_only:
            batch["labels"] = self._base.train_on_responses_only(batch)["labels"]
        if self.include_response_loss_weight:
            weights = torch.tensor(
                [float(example.get("loss_weight", 1.0)) for example in examples],
                dtype=torch.float32,
                device=batch["labels"].device,
            )
            response_mask = (batch["labels"] != self._base.ignore_index).to(torch.float32)
            batch["response_loss_weight"] = response_mask * weights[:, None]
        return batch


class ModuleADetectionMetricsCallback:
    def __init__(
        self,
        *,
        eval_examples: list[dict[str, Any]],
        data_collator: Qwen3VLMetadataVisionDataCollator,
        tokenizer: Any,
        max_samples: int,
        max_new_tokens: int,
        beta: float,
        label_score_eval: bool,
    ) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self.eval_examples = generation_eval_examples(eval_examples, max_samples)
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.beta = beta
        self.label_score_eval = label_score_eval

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None or not self.eval_examples:
            return control

        model = kwargs["model"]
        was_training = model.training
        predictions: list[str] = []
        targets: list[str] = []
        task_targets: dict[str, list[str]] = defaultdict(list)
        task_predictions: dict[str, list[str]] = defaultdict(list)

        import torch
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for example in self.eval_examples:
                    if self.label_score_eval:
                        prediction = self._score_label(model, example)
                    else:
                        generated_text = self._generate_label(model, example)
                        prediction = extract_module_a_label(generated_text)
                    target = str(example["target"])
                    task_id = str(example["task_id"])
                    predictions.append(prediction)
                    targets.append(target)
                    task_predictions[task_id].append(prediction)
                    task_targets[task_id].append(target)
        finally:
            if was_training:
                FastVisionModel.for_training(model)
            release_cuda_memory()

        metrics = binary_detection_metrics(targets, predictions, prefix="eval_detection", beta=self.beta)
        for task_id in sorted(task_targets):
            task_metrics = binary_detection_metrics(
                task_targets[task_id],
                task_predictions[task_id],
                prefix=f"eval_detection/task_{task_id}",
                beta=self.beta,
            )
            metrics.update({
                key: value
                for key, value in task_metrics.items()
                if key.endswith(("/accuracy", "/f1", "/recall", "/num_samples"))
            })
        callback_metrics = kwargs.get("metrics")
        if isinstance(callback_metrics, dict):
            callback_metrics.update(metrics)
        self.trainer.log(metrics)
        release_cuda_memory()
        return control

    def _generation_inputs(self, example: dict[str, Any]) -> dict[str, Any]:
        messages = [
            copy.deepcopy(message)
            for message in example["messages"]
            if message.get("role") != "assistant"
        ]
        text = self.data_collator._base.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image, video, video_kwargs, video_metadata = self.data_collator._extract_images_videos_metadata(messages)
        proc_kwargs: dict[str, Any] = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if image:
            proc_kwargs["images"] = image
        if video:
            proc_kwargs["videos"] = video
            proc_kwargs["do_resize"] = False
            proc_kwargs.update(video_kwargs)
            if video_metadata:
                proc_kwargs["video_metadata"] = video_metadata
        return self.data_collator._base.processor(**proc_kwargs)

    def _generate_label(self, model: Any, example: dict[str, Any]) -> str:
        inputs = self._generation_inputs(example)
        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": False,
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        try:
            output_ids = model.generate(**inputs, **generate_kwargs)
            generated_ids = output_ids[:, input_length:]
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        finally:
            del inputs
            if "output_ids" in locals():
                del output_ids
            if "generated_ids" in locals():
                del generated_ids
            release_cuda_memory()

    def _score_label(self, model: Any, example: dict[str, Any]) -> str:
        import torch

        scores = {}
        device = next(model.parameters()).device
        for label in MODULE_A_LABELS:
            scored_example = copy.deepcopy(example)
            for message in scored_example["messages"]:
                if message.get("role") == "assistant":
                    message["content"] = [{"type": "text", "text": label}]
                    break
            batch = self.data_collator([scored_example])
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            inputs.pop("response_loss_weight", None)
            labels = inputs.get("labels")
            try:
                with torch.inference_mode():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                logits = logits_for_loss(model, outputs)[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                token_count = (shift_labels != -100).sum().clamp_min(1)
                scores[label] = float((loss / token_count).detach().cpu())
            finally:
                del inputs
                if "outputs" in locals():
                    del outputs
                if "logits" in locals():
                    del logits
                release_cuda_memory()
        return min(scores, key=scores.get)


def compute_generation_detection_metrics(
    *,
    model: Any,
    tokenizer: Any,
    eval_examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    data_collator = Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
        resize=args.vision_resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )
    generator = ModuleADetectionMetricsCallback(
        eval_examples=eval_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_samples=args.eval_generation_max_samples,
        max_new_tokens=args.eval_generation_max_new_tokens,
        beta=args.fbeta_beta,
        label_score_eval=args.label_score_eval,
    )
    predictions: list[str] = []
    targets: list[str] = []
    task_targets: dict[str, list[str]] = defaultdict(list)
    task_predictions: dict[str, list[str]] = defaultdict(list)

    for example in generator.eval_examples:
        if args.label_score_eval:
            prediction = generator._score_label(model, example)
        else:
            generated_text = generator._generate_label(model, example)
            prediction = extract_module_a_label(generated_text)
        target = str(example["target"])
        task_id = str(example["task_id"])
        predictions.append(prediction)
        targets.append(target)
        task_predictions[task_id].append(prediction)
        task_targets[task_id].append(target)

    metrics = binary_detection_metrics(targets, predictions, prefix="eval_detection", beta=args.fbeta_beta)
    for task_id in sorted(task_targets):
        task_metrics = binary_detection_metrics(
            task_targets[task_id],
            task_predictions[task_id],
            prefix=f"eval_detection/task_{task_id}",
            beta=args.fbeta_beta,
        )
        metrics.update({
            key: value
            for key, value in task_metrics.items()
            if key.endswith(("/accuracy", "/f1", "/recall", "/num_samples"))
        })
    return metrics


def build_generation_eval_command(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--generation-eval-only",
        "--eval-checkpoint",
        str(checkpoint_path),
        "--generation-eval-output-json",
        str(output_json),
        "--metadata",
        str(args.metadata),
        "--mistake-classes",
        str(args.mistake_classes),
        "--video-root",
        str(args.video_root),
        "--max-videos",
        str(args.max_videos),
        "--fps",
        str(args.fps),
        "--min-frames",
        str(args.min_frames),
        "--max-frames",
        str(args.max_frames),
        "--stride-seconds",
        str(args.stride_seconds),
        "--completion-margin",
        str(args.completion_margin),
        "--negative-to-positive-ratio",
        str(args.negative_to_positive_ratio),
        "--keep-last-wait-windows",
        str(args.keep_last_wait_windows),
        "--max-seq-length",
        str(args.max_seq_length),
        "--vision-resize",
        str(args.vision_resize),
        "--eval-generation-max-samples",
        str(args.eval_generation_max_samples),
        "--eval-generation-max-new-tokens",
        str(args.eval_generation_max_new_tokens),
        "--fbeta-beta",
        str(args.fbeta_beta),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
    ]
    if args.split_file:
        command.extend(["--split-file", str(args.split_file)])
    if args.video_ids:
        command.append("--video-ids")
        command.extend(str(item) for item in args.video_ids)
    if args.task_ids:
        command.append("--task-ids")
        command.extend(str(item) for item in args.task_ids)
    command.append("--load-in-4bit" if args.load_in_4bit else "--no-load-in-4bit")
    command.append("--load-in-16bit" if args.load_in_16bit else "--no-load-in-16bit")
    command.append("--label-score-eval" if args.label_score_eval else "--no-label-score-eval")
    return command


def sanitize_wandb_artifact_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    sanitized = "".join(char if char in allowed else "-" for char in name)
    return sanitized.strip(".-") or "checkpoint"


class EpochCheckpointCallback:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        eval_epochs: int,
        checkpoint_epochs: int,
        metric_for_best_model: str,
        greater_is_better: bool,
        keep_last_checkpoints: int,
        keep_best_checkpoints: int,
        early_stopping_patience: int,
        early_stopping_threshold: float,
        wandb_log_best_checkpoints: bool,
        wandb_artifact_prefix: str,
        generation_eval_args: argparse.Namespace,
    ) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self.output_dir = Path(output_dir)
        self.eval_epochs = eval_epochs
        self.checkpoint_epochs = checkpoint_epochs
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.keep_last_checkpoints = keep_last_checkpoints
        self.keep_best_checkpoints = keep_best_checkpoints
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.wandb_log_best_checkpoints = wandb_log_best_checkpoints
        self.wandb_artifact_prefix = wandb_artifact_prefix
        self.generation_eval_args = generation_eval_args
        self.best_metric: float | None = None
        self.no_improvement_count = 0
        self.best_records: list[dict[str, Any]] = []
        self._last_latest_epoch: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        epoch = self._completed_epoch(state)
        if epoch is None:
            return control

        self._save_named_checkpoint("latest")
        self._last_latest_epoch = epoch

        if epoch % self.checkpoint_epochs == 0:
            self._save_named_checkpoint(f"ckpt_ep{epoch}")
            self._prune_epoch_checkpoints("ckpt_ep", self.keep_last_checkpoints)

        if epoch % self.eval_epochs == 0:
            control.should_evaluate = True
        release_cuda_memory()
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if not metrics:
            return control
        epoch = self._epoch_label(state)
        if self.generation_eval_args.generation_eval_mode == "subprocess":
            generation_metrics = self._run_subprocess_generation_eval(epoch=epoch)
            if generation_metrics:
                metrics.update(generation_metrics)
                if self.trainer is not None:
                    self.trainer.log(generation_metrics)
        metric = self._metric_value(metrics)
        if metric is None:
            print(f"Metric {self.metric_for_best_model!r} was not found; skipping best checkpoint update.")
            return control

        if self._is_improvement(metric):
            self.best_metric = metric
            self.no_improvement_count = 0
            path = self._save_named_checkpoint(f"best_ep{epoch}")
            self.best_records.append({
                "epoch": epoch,
                "metric": metric,
                "path": str(path),
                "global_step": state.global_step,
            })
            self._prune_best_checkpoints()
            self._write_best_manifest()
            if self.wandb_log_best_checkpoints and path.exists():
                self._log_wandb_checkpoint(path=path, epoch=epoch, metric=metric, state=state)
        else:
            self.no_improvement_count += 1
            if (
                self.early_stopping_patience > 0
                and self.no_improvement_count >= self.early_stopping_patience
            ):
                print(
                    "Early stopping: "
                    f"{self.no_improvement_count} validation checks without improvement in "
                    f"{self.metric_for_best_model}."
                )
                control.should_training_stop = True
        release_cuda_memory()
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        epoch = self._completed_epoch(state)
        if epoch is None or epoch != self._last_latest_epoch:
            self._save_named_checkpoint("latest")
        return control

    def _completed_epoch(self, state: Any) -> int | None:
        if state.epoch is None:
            return None
        rounded = round(float(state.epoch))
        if rounded < 1 or not math.isclose(float(state.epoch), rounded, abs_tol=1e-3):
            return None
        return int(rounded)

    def _epoch_label(self, state: Any) -> str:
        completed = self._completed_epoch(state)
        if completed is not None:
            return str(completed)
        if state.epoch is None:
            return f"step{state.global_step}"
        return str(float(state.epoch)).replace(".", "_")

    def _metric_value(self, metrics: dict[str, Any]) -> float | None:
        keys = [
            self.metric_for_best_model,
            f"eval_{self.metric_for_best_model}",
        ]
        for key in keys:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return metric > self.best_metric + self.early_stopping_threshold
        return metric < self.best_metric - self.early_stopping_threshold

    def _save_named_checkpoint(self, name: str) -> Path:
        if self.trainer is None:
            raise RuntimeError("EpochCheckpointCallback.trainer must be assigned before training.")
        path = self.output_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(path)
        self.trainer.state.save_to_json(str(path / "trainer_state.json"))
        release_cuda_memory()
        return path

    def _prune_epoch_checkpoints(self, prefix: str, keep: int) -> None:
        if keep == 0:
            candidates = list(self.output_dir.glob(f"{prefix}*"))
        else:
            candidates = sorted(
                self.output_dir.glob(f"{prefix}*"),
                key=lambda path: self._epoch_from_name(path.name, prefix),
            )[:-keep]
        for path in candidates:
            if path.is_dir():
                shutil.rmtree(path)

    def _prune_best_checkpoints(self) -> None:
        if self.keep_best_checkpoints == 0:
            records_to_remove = list(self.best_records)
            self.best_records = []
        else:
            reverse = self.greater_is_better
            self.best_records.sort(key=lambda item: item["metric"], reverse=reverse)
            records_to_remove = self.best_records[self.keep_best_checkpoints :]
            self.best_records = self.best_records[: self.keep_best_checkpoints]
        for record in records_to_remove:
            path = Path(record["path"])
            if path.is_dir():
                shutil.rmtree(path)

    def _write_best_manifest(self) -> None:
        manifest = {
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "best_metric": self.best_metric,
            "best_checkpoints": self.best_records,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "best_checkpoints.json", "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")

    def _log_wandb_checkpoint(self, *, path: Path, epoch: str, metric: float, state: Any) -> None:
        try:
            import wandb
        except ModuleNotFoundError:
            print("W&B is not installed; skipping best checkpoint artifact upload.")
            return
        if wandb.run is None:
            print("W&B run is not active; skipping best checkpoint artifact upload.")
            return

        run_id = wandb.run.id or "run"
        artifact_name = sanitize_wandb_artifact_name(f"{self.wandb_artifact_prefix}-{run_id}-best")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={
                "checkpoint": path.name,
                "epoch": epoch,
                "global_step": state.global_step,
                "metric_for_best_model": self.metric_for_best_model,
                "metric": metric,
                "greater_is_better": self.greater_is_better,
            },
        )
        artifact.add_dir(str(path), name=path.name)
        run_config = self.output_dir / "run_config.json"
        if run_config.exists():
            artifact.add_file(str(run_config), name="run_config.json")
        best_manifest = self.output_dir / "best_checkpoints.json"
        if best_manifest.exists():
            artifact.add_file(str(best_manifest), name="best_checkpoints.json")
        wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
        print(f"Logged W&B best checkpoint artifact: {artifact_name} ({path})")

    def _run_subprocess_generation_eval(self, *, epoch: str) -> dict[str, float]:
        if self.trainer is None:
            return {}
        if self.generation_eval_args.eval_generation_max_samples == 0:
            return {}

        checkpoint_path = self.output_dir / "latest"
        if not checkpoint_path.exists():
            return {}

        output_json = self.output_dir / f"generation_eval_ep{epoch}.json"
        command = build_generation_eval_command(
            args=self.generation_eval_args,
            checkpoint_path=checkpoint_path,
            output_json=output_json,
        )
        restore_device = offload_trainer_cuda_state(self.trainer)
        try:
            subprocess.run(command, check=True)
        finally:
            restore_trainer_cuda_state(self.trainer, restore_device)

        if not output_json.exists():
            return {}
        with open(output_json, "r", encoding="utf-8") as file:
            payload = json.load(file)
        metrics = payload.get("metrics", {})
        return {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }

    @staticmethod
    def _epoch_from_name(name: str, prefix: str) -> int:
        try:
            return int(name.removeprefix(prefix))
        except ValueError:
            return -1


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel

    class WeightedSFTTrainer(SFTTrainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            import torch

            response_loss_weight = inputs.pop("response_loss_weight", None)
            if response_loss_weight is None:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs.get("labels")
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            logits = logits_for_loss(model, outputs)[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            shift_weights = response_loss_weight[:, 1:].to(logits.device)
            token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(shift_labels)
            weighted_loss = token_loss * shift_weights
            denominator = shift_weights.sum().clamp_min(1.0)
            loss = weighted_loss.sum() / denominator
            return (loss, outputs) if return_outputs else loss

    FastVisionModel.for_training(model)
    bf16 = is_bf16_available()
    hf_train_dataset = Dataset.from_list(train_dataset)
    hf_eval_dataset = Dataset.from_list(eval_dataset)
    callbacks: list[TrainerCallback] = []
    trainer_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=args.report_to,
        dataset_num_proc=args.dataset_num_proc,
        max_seq_length=args.max_seq_length,
        torch_empty_cache_steps=1,
        fp16=not bf16,
        bf16=bf16,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    if args.train_mode == "epochs":
        trainer_args["num_train_epochs"] = args.num_train_epochs
    else:
        trainer_args["max_steps"] = args.max_steps

    data_collator = Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
        resize=args.vision_resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
        include_response_loss_weight=args.positive_loss_weight != 1.0,
    )
    detection_callback: ModuleADetectionMetricsCallback | None = None
    if args.generation_eval_mode == "inline" and args.eval_generation_max_samples != 0:
        detection_callback = ModuleADetectionMetricsCallback(
            eval_examples=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            max_samples=args.eval_generation_max_samples,
            max_new_tokens=args.eval_generation_max_new_tokens,
            beta=args.fbeta_beta,
            label_score_eval=args.label_score_eval,
        )
        callbacks.append(detection_callback)
    checkpoint_callback = EpochCheckpointCallback(
        output_dir=args.output_dir,
        eval_epochs=args.eval_epochs,
        checkpoint_epochs=args.checkpoint_epochs,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        keep_last_checkpoints=args.keep_last_checkpoints,
        keep_best_checkpoints=args.keep_best_checkpoints,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        wandb_log_best_checkpoints=args.wandb_log_best_checkpoints,
        wandb_artifact_prefix=args.wandb_artifact_prefix,
        generation_eval_args=args,
    )
    callbacks.append(checkpoint_callback)
    trainer_cls = WeightedSFTTrainer if args.positive_loss_weight != 1.0 else SFTTrainer
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        data_collator=data_collator,
        args=SFTConfig(**trainer_args),
        callbacks=callbacks,
    )
    if detection_callback is not None:
        detection_callback.trainer = trainer
    checkpoint_callback.trainer = trainer
    return trainer


def save_outputs(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved LoRA adapter and tokenizer to: {output_path}")


def run_generation_eval_only(
    *,
    args: argparse.Namespace,
    val_conversations: list[dict[str, Any]],
) -> None:
    if args.eval_generation_max_samples == 0:
        metrics: dict[str, float] = {}
    else:
        model, tokenizer = load_generation_eval_model(args)
        metrics = compute_generation_detection_metrics(
            model=model,
            tokenizer=tokenizer,
            eval_examples=val_conversations,
            args=args,
        )
        del model
        release_cuda_memory()

    payload = {
        "checkpoint": args.eval_checkpoint,
        "num_validation_examples": len(val_conversations),
        "num_generation_examples": (
            len(val_conversations)
            if args.eval_generation_max_samples < 0
            else min(args.eval_generation_max_samples, len(val_conversations))
        ),
        "metrics": metrics,
    }
    if args.generation_eval_output_json:
        output_path = Path(args.generation_eval_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    configure_unsloth_logits(args)
    dataset = build_module_a_dataset(args)
    conversations = to_unsloth_conversations(
        dataset,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )

    if not conversations:
        raise SystemExit("Module A dataset produced no conversations.")
    split, split_path = load_or_create_video_split(dataset, args)
    train_conversations, val_conversations = split_conversations(conversations, split)
    train_conversations = oversample_positive_conversations(
        train_conversations,
        factor=args.positive_oversample_factor,
        seed=args.seed,
    )
    apply_positive_loss_weights(
        train_conversations,
        positive_weight=args.positive_loss_weight,
    )
    apply_positive_loss_weights(val_conversations, positive_weight=1.0)
    if args.generation_eval_only:
        run_generation_eval_only(args=args, val_conversations=val_conversations)
        return
    if args.dry_run:
        print_dry_run(
            dataset,
            train_conversations,
            val_conversations,
            split,
            split_path,
            args.print_examples,
        )
        return
    save_run_config(
        args=args,
        dataset=dataset,
        conversations=conversations,
        train_conversations=train_conversations,
        val_conversations=val_conversations,
        split=split,
        split_path=split_path,
    )

    print("Module A full dataset summary:")
    print(json.dumps(summarize_module_a(dataset), indent=2, ensure_ascii=False))
    print(f"Module A split file: {split_path}")
    print("Module A train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module A validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))

    model, tokenizer = load_unsloth_model(args)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_conversations,
        eval_dataset=val_conversations,
        args=args,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    save_outputs(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
